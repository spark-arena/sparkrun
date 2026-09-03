"""Tests for sparkrun.models.vram module."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.models.kv import ArchInfo, arch_marker_names, is_valid_kv_dtype, resolve_kv_strategy
from sparkrun.models.kv.mla import (
    mla_kv_bytes_per_token,
    mla_latent_dim,
    reconcile_compress_ratios,
)
from sparkrun.models.vram import (
    _resolve_quant_dtype,
    bytes_per_element,
    estimate_vram,
    extract_model_info,
    fetch_safetensors_params,
    fetch_safetensors_size,
    kv_bytes_per_element,
    parse_param_count,
)

_MLA_KEYS = frozenset({"kv_lora_rank", "qk_rope_head_dim", "compress_ratios", "index_head_dim"})


def _arch_from(info: dict) -> dict:
    """The architecture markers in an ``extract_model_info`` result, as ``arch=``.

    Extraction and estimation agree on field names by construction — both are
    keyed off :func:`arch_marker_names` — so handing one to the other needs no
    per-architecture list.
    """
    return {k: v for k, v in info.items() if k in arch_marker_names()}


class TestParseParamCount:
    """Test parameter count parsing from various formats."""

    def test_integer(self):
        assert parse_param_count(7_000_000_000) == 7_000_000_000

    def test_float(self):
        assert parse_param_count(7e9) == 7_000_000_000

    def test_string_7b(self):
        assert parse_param_count("7B") == 7_000_000_000

    def test_string_70b(self):
        assert parse_param_count("70B") == 70_000_000_000

    def test_string_half_b(self):
        assert parse_param_count("0.5B") == 500_000_000

    def test_string_480m(self):
        assert parse_param_count("480M") == 480_000_000

    def test_string_1t(self):
        assert parse_param_count("1T") == 1_000_000_000_000

    def test_string_underscore(self):
        assert parse_param_count("7_000_000_000") == 7_000_000_000

    def test_string_lowercase(self):
        assert parse_param_count("7b") == 7_000_000_000

    def test_string_float_value(self):
        assert parse_param_count("9.4B") == 9_400_000_000

    def test_invalid_string(self):
        assert parse_param_count("not_a_number") is None

    def test_empty_string(self):
        assert parse_param_count("") is None

    def test_none_returns_none(self):
        # Not a valid input type, but should handle gracefully
        assert parse_param_count(None) is None  # type: ignore[arg-type]


class TestBytesPerElement:
    """Test dtype to bytes-per-element mapping."""

    def test_float32(self):
        assert bytes_per_element("float32") == 4.0

    def test_fp32(self):
        assert bytes_per_element("fp32") == 4.0

    def test_float16(self):
        assert bytes_per_element("float16") == 2.0

    def test_fp16(self):
        assert bytes_per_element("fp16") == 2.0

    def test_bfloat16(self):
        assert bytes_per_element("bfloat16") == 2.0

    def test_bf16(self):
        assert bytes_per_element("bf16") == 2.0

    def test_int8(self):
        assert bytes_per_element("int8") == 1.0

    def test_fp8(self):
        assert bytes_per_element("fp8") == 1.0

    def test_fp8_e5m2(self):
        assert bytes_per_element("fp8_e5m2") == 1.0

    def test_fp8_e4m3(self):
        assert bytes_per_element("fp8_e4m3") == 1.0

    def test_int4(self):
        assert bytes_per_element("int4") == 0.5

    def test_nvfp4(self):
        assert bytes_per_element("nvfp4") == 0.5

    def test_awq4(self):
        val = bytes_per_element("awq4")
        assert val is not None
        assert 0.5 <= val <= 0.75

    def test_awq8(self):
        val = bytes_per_element("awq8")
        assert val is not None
        assert 1.0 <= val <= 1.25

    def test_gptq(self):
        assert bytes_per_element("gptq") == 0.5

    def test_unknown(self):
        assert bytes_per_element("unknown_dtype") is None

    def test_case_insensitive(self):
        assert bytes_per_element("FLOAT16") == 2.0
        assert bytes_per_element("BFloat16") == 2.0

    def test_strip_whitespace(self):
        assert bytes_per_element("  float16  ") == 2.0

    def test_bits_per_weight_families(self):
        """exl2/exl3 widths are per-checkpoint and fractional, so they ride in
        the dtype string rather than in the table."""
        assert bytes_per_element("exl3:2.05") == 2.05 / 8
        assert bytes_per_element("exl3:4.05") == 4.05 / 8
        assert bytes_per_element("exl2:4.25") == 4.25 / 8
        assert bytes_per_element("EXL3:2.05") == 2.05 / 8

    def test_bare_family_is_not_sizable(self):
        """The bpw *is* the width — a family name alone carries none, and the
        metadata validator is `bytes_per_element(...) is None`, so this is also
        what makes a bare `exl3` in a recipe get reported."""
        assert bytes_per_element("exl3") is None

    def test_malformed_bits_per_weight_is_rejected(self):
        assert bytes_per_element("exl3:") is None
        assert bytes_per_element("exl3:abc") is None
        assert bytes_per_element("foo:2.05") is None  # undeclared family

    def test_bits_per_weight_is_bounded(self):
        """A typo'd `exl3:205` must not silently size the model 100x too large."""
        assert bytes_per_element("exl3:205") is None
        assert bytes_per_element("exl3:0.5") is None

    def test_bits_per_weight_is_weight_only(self):
        """These are never KV-cache layouts, so the parse is deliberately absent
        from the KV path."""
        from sparkrun.models.kv import is_valid_kv_dtype

        assert kv_bytes_per_element("exl3:2.05") is None
        assert is_valid_kv_dtype("exl3:2.05") is False


class TestEstimateVram:
    """Test VRAM estimation calculations."""

    def test_basic_7b_fp16(self):
        """7B params * 2 bytes = 14 GB total, ~13.04 GiB."""
        est = estimate_vram(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=1,
        )
        # 7B * 2 bytes / 1024^3 ≈ 13.04 GiB
        assert abs(est.model_weights_gb - 13.04) < 0.1
        assert est.kv_cache_total_gb is not None
        assert est.kv_cache_total_gb > 0
        assert est.total_per_gpu_gb > 0
        assert len(est.warnings) == 0

    def test_tp2_halves_per_gpu(self):
        """With tp=2, per-GPU VRAM should be half of tp=1."""
        kwargs = dict(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
        )
        est1 = estimate_vram(**kwargs, tensor_parallel=1)
        est2 = estimate_vram(**kwargs, tensor_parallel=2)

        assert est2.total_per_gpu_gb < est1.total_per_gpu_gb
        assert abs(est2.total_per_gpu_gb - est1.total_per_gpu_gb / 2) < 0.01

    def test_pp2_halves_per_gpu(self):
        """With pp=2, per-GPU VRAM should be half of pp=1."""
        kwargs = dict(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
        )
        est1 = estimate_vram(**kwargs, tensor_parallel=1, pipeline_parallel=1)
        est2 = estimate_vram(**kwargs, tensor_parallel=1, pipeline_parallel=2)

        assert est2.total_per_gpu_gb < est1.total_per_gpu_gb
        assert abs(est2.total_per_gpu_gb - est1.total_per_gpu_gb / 2) < 0.01
        assert est2.pipeline_parallel == 2

    def test_tp2_pp2_quarters_per_gpu(self):
        """With tp=2 and pp=2, per-GPU VRAM should be 1/4 of single GPU."""
        kwargs = dict(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
        )
        est1 = estimate_vram(**kwargs, tensor_parallel=1, pipeline_parallel=1)
        est4 = estimate_vram(**kwargs, tensor_parallel=2, pipeline_parallel=2)

        assert abs(est4.total_per_gpu_gb - est1.total_per_gpu_gb / 4) < 0.01

    def test_pp_default_is_one(self):
        """Default pipeline_parallel should be 1."""
        est = estimate_vram(model_vram=10.0)
        assert est.pipeline_parallel == 1

    def test_model_vram_override_with_pp(self):
        """model_vram is divided by tp * pp."""
        est = estimate_vram(model_vram=12.0, tensor_parallel=2, pipeline_parallel=3)
        # 12 GB / (2 * 3) = 2 GB per GPU
        assert abs(est.total_per_gpu_gb - 2.0) < 0.01

    def test_kv_vram_per_token_with_tp_and_pp(self):
        """kv_vram_per_token is divided by tp * pp."""
        est = estimate_vram(
            model_vram=12.0,
            kv_vram_per_token=0.0001,
            max_model_len=10000,
            tensor_parallel=2,
            pipeline_parallel=3,
        )
        # model: 12/(2*3) = 2, KV: 0.0001*10000/(2*3) = 0.1667, total ≈ 2.1667
        assert abs(est.total_per_gpu_gb - (12.0 / 6 + 0.0001 * 10000 / 6)) < 0.01

    def test_gpu_memory_utilization_with_tp_and_pp(self):
        """Budget analysis should work with tensor and pipeline parallel."""
        from sparkrun.models.vram import DGX_SPARK_VRAM_GB

        est = estimate_vram(
            model_vram=24.0,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=2,
            pipeline_parallel=3,
            gpu_memory_utilization=0.9,
        )
        usable = DGX_SPARK_VRAM_GB * 0.9
        per_gpu_weights = 24.0 / 6
        assert est.available_kv_gb == pytest.approx(usable - per_gpu_weights, abs=0.01)

    def test_missing_params_warns(self):
        est = estimate_vram(model_dtype="float16")
        assert len(est.warnings) > 0
        assert est.model_weights_gb == 0.0
        assert any("model_params" in w for w in est.warnings)

    def test_missing_dtype_warns(self):
        est = estimate_vram(model_params=7_000_000_000)
        assert len(est.warnings) > 0
        assert any("model_dtype" in w for w in est.warnings)

    def test_unknown_dtype_warns(self):
        est = estimate_vram(model_params=7_000_000_000, model_dtype="bogus")
        assert any("Unknown dtype" in w for w in est.warnings)

    def test_fp8_kv_cache_smaller(self):
        """fp8 KV cache should be half the size of bfloat16 KV cache."""
        kwargs = dict(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=1,
        )
        est_bf16 = estimate_vram(**kwargs, kv_dtype="bfloat16")
        est_fp8 = estimate_vram(**kwargs, kv_dtype="fp8")

        assert est_fp8.kv_cache_total_gb is not None
        assert est_bf16.kv_cache_total_gb is not None
        assert est_fp8.kv_cache_total_gb < est_bf16.kv_cache_total_gb
        # fp8 is 1 byte vs bfloat16 2 bytes -> exactly half
        assert abs(est_fp8.kv_cache_total_gb - est_bf16.kv_cache_total_gb / 2) < 0.001

    def test_int4_model_half_of_int8(self):
        """int4 weights should be half the size of int8 weights."""
        kwargs = dict(
            model_params=7_000_000_000,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=1,
        )
        est_int8 = estimate_vram(**kwargs, model_dtype="int8")
        est_int4 = estimate_vram(**kwargs, model_dtype="int4")

        assert abs(est_int4.model_weights_gb - est_int8.model_weights_gb / 2) < 0.01

    def test_missing_architecture_warns(self):
        """Missing num_layers/num_kv_heads/head_dim should warn."""
        est = estimate_vram(
            model_params=7_000_000_000,
            model_dtype="float16",
            max_model_len=4096,
        )
        assert any("Missing architecture info" in w for w in est.warnings)
        assert est.kv_cache_total_gb is None

    def test_no_max_model_len(self):
        """Without max_model_len, kv_cache_total_gb should be None."""
        est = estimate_vram(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
        )
        assert est.kv_cache_per_token_bytes is not None
        assert est.kv_cache_total_gb is None
        assert len(est.warnings) == 0

    def test_model_vram_override(self):
        """model_vram override should bypass param-based calculation."""
        est = estimate_vram(
            model_vram=5.2,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=1,
        )
        assert est.model_weights_gb == 5.2
        assert est.total_per_gpu_gb > 5.2  # weights + kv cache

    def test_model_vram_override_with_tp(self):
        """model_vram is divided by tp."""
        est = estimate_vram(model_vram=10.0, tensor_parallel=2)
        # 10 GB / 2 tp = 5 GB per GPU
        assert abs(est.total_per_gpu_gb - 5.0) < 0.01

    def test_kv_vram_per_token_override(self):
        """kv_vram_per_token override should bypass formula-based KV calc."""
        est = estimate_vram(
            model_vram=5.0,
            kv_vram_per_token=0.0001,  # GB per token
            max_model_len=10000,
            tensor_parallel=1,
        )
        # KV total = 0.0001 * 10000 = 1.0 GB
        assert est.kv_cache_total_gb is not None
        assert abs(est.kv_cache_total_gb - 1.0) < 0.001
        # Total = 5.0 + 1.0 = 6.0
        assert abs(est.total_per_gpu_gb - 6.0) < 0.01

    def test_kv_vram_per_token_with_tp(self):
        """kv_vram_per_token is divided by tp."""
        est = estimate_vram(
            model_vram=10.0,
            kv_vram_per_token=0.0001,
            max_model_len=10000,
            tensor_parallel=2,
        )
        # model: 10/2 = 5, KV: 0.0001*10000/2 = 0.5, total = 5.5
        assert abs(est.total_per_gpu_gb - 5.5) < 0.01

    def test_fits_dgx_spark_property(self):
        est_small = estimate_vram(model_vram=10.0, tensor_parallel=1)
        assert est_small.fits_dgx_spark is True

        est_big = estimate_vram(model_vram=150.0, tensor_parallel=1)
        assert est_big.fits_dgx_spark is False

    def test_default_kv_dtype_is_none(self):
        """When no kv_dtype is passed, the estimate uses bfloat16 for computation but
        leaves est.kv_dtype as None so display code can show "bfloat16 (default)".
        """
        est = estimate_vram(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
        )
        # est.kv_dtype is None (not "bfloat16") so the formatter's (default) branch fires
        assert est.kv_dtype is None
        # but the KV cache was still sized with bfloat16 (2 bytes)
        assert est.kv_cache_per_token_bytes == 2.0 * 32 * 32 * 128 * 2

    def test_explicit_kv_dtype_is_preserved(self):
        """An explicit kv_dtype is preserved on the estimate."""
        est = estimate_vram(
            model_params=7_000_000_000,
            model_dtype="float16",
            kv_dtype="fp8",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
        )
        assert est.kv_dtype == "fp8"

    def test_gpu_memory_utilization_budget(self):
        """gpu_memory_utilization should compute usable memory and available KV."""
        from sparkrun.models.vram import DGX_SPARK_VRAM_GB

        est = estimate_vram(
            model_vram=10.0,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=1,
            gpu_memory_utilization=0.9,
        )
        assert est.gpu_memory_utilization == 0.9
        assert est.usable_gpu_memory_gb == pytest.approx(DGX_SPARK_VRAM_GB * 0.9, abs=0.1)
        assert est.available_kv_gb is not None
        assert est.available_kv_gb == pytest.approx(est.usable_gpu_memory_gb - 10.0, abs=0.01)
        assert est.max_context_tokens is not None
        assert est.max_context_tokens > 0

    def test_gpu_memory_utilization_context_multiplier(self):
        """context_multiplier should reflect how many max_model_lens fit."""
        est = estimate_vram(
            model_vram=10.0,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=1,
            gpu_memory_utilization=0.9,
        )
        assert est.context_multiplier is not None
        assert est.context_multiplier > 0
        # max_context_tokens / max_model_len
        assert est.context_multiplier == pytest.approx(est.max_context_tokens / 4096, abs=0.01)

    def test_gpu_memory_utilization_none_skips_budget(self):
        """Without gpu_memory_utilization, budget fields should be None."""
        est = estimate_vram(
            model_vram=10.0,
            tensor_parallel=1,
        )
        assert est.gpu_memory_utilization is None
        assert est.usable_gpu_memory_gb is None
        assert est.available_kv_gb is None
        assert est.max_context_tokens is None
        assert est.context_multiplier is None

    def test_gpu_memory_utilization_model_exceeds_budget(self):
        """Model larger than usable memory should warn."""
        est = estimate_vram(
            model_vram=200.0,
            tensor_parallel=1,
            gpu_memory_utilization=0.5,
        )
        assert est.available_kv_gb == 0.0
        assert any("exceed" in w.lower() for w in est.warnings)

    def test_gpu_memory_utilization_with_tp(self):
        """Budget analysis should work with tensor parallel."""
        from sparkrun.models.vram import DGX_SPARK_VRAM_GB

        est = estimate_vram(
            model_vram=20.0,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
            tensor_parallel=2,
            gpu_memory_utilization=0.9,
        )
        usable = DGX_SPARK_VRAM_GB * 0.9
        per_gpu_weights = 20.0 / 2
        assert est.available_kv_gb == pytest.approx(usable - per_gpu_weights, abs=0.01)


class TestExtractModelInfo:
    """Test HuggingFace config.json extraction."""

    def test_llama_style_config(self):
        config = {
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 4096,
        }
        info = extract_model_info(config)
        assert info["model_dtype"] == "bfloat16"
        assert info["num_layers"] == 32
        assert info["num_kv_heads"] == 8
        assert info["head_dim"] == 128  # 4096 / 32

    def test_gpt_neox_style(self):
        config = {
            "torch_dtype": "float16",
            "n_layer": 24,
            "n_head": 16,
            "hidden_size": 2048,
        }
        info = extract_model_info(config)
        assert info["num_layers"] == 24
        assert info["num_kv_heads"] == 16  # MHA fallback
        assert info["head_dim"] == 128  # 2048 / 16

    def test_explicit_head_dim(self):
        config = {
            "num_hidden_layers": 32,
            "head_dim": 64,
            "num_attention_heads": 32,
            "hidden_size": 2048,
        }
        info = extract_model_info(config)
        # Explicit head_dim should be used, not derived
        assert info["head_dim"] == 64

    def test_missing_fields(self):
        info = extract_model_info({})
        assert info == {}

    def test_gqa_kv_heads_preferred(self):
        """num_key_value_heads should be preferred over num_attention_heads."""
        config = {
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "hidden_size": 4096,
        }
        info = extract_model_info(config)
        assert info["num_kv_heads"] == 4

    def test_mha_fallback(self):
        """Without GQA fields, fall back to num_attention_heads."""
        config = {
            "num_attention_heads": 32,
            "hidden_size": 4096,
        }
        info = extract_model_info(config)
        assert info["num_kv_heads"] == 32

    def test_nested_text_config(self):
        """Multimodal models nest text architecture under text_config."""
        config = {
            "architectures": ["SomeVLModel"],
            "model_type": "some_vl",
            "text_config": {
                "dtype": "bfloat16",
                "num_hidden_layers": 40,
                "num_key_value_heads": 2,
                "num_attention_heads": 16,
                "head_dim": 256,
                "hidden_size": 2048,
            },
            "vision_config": {
                "hidden_size": 1280,
            },
        }
        info = extract_model_info(config)
        assert info["model_dtype"] == "bfloat16"
        assert info["num_layers"] == 40
        assert info["num_kv_heads"] == 2
        assert info["head_dim"] == 256

    def test_nested_llm_config(self):
        """Some models use llm_config instead of text_config."""
        config = {
            "llm_config": {
                "torch_dtype": "float16",
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "hidden_size": 4096,
            },
        }
        info = extract_model_info(config)
        assert info["model_dtype"] == "float16"
        assert info["num_layers"] == 32
        assert info["num_kv_heads"] == 8
        assert info["head_dim"] == 128

    def test_top_level_takes_precedence_over_nested(self):
        """Top-level fields should win over nested text_config fields."""
        config = {
            "torch_dtype": "float16",
            "num_hidden_layers": 32,
            "text_config": {
                "dtype": "bfloat16",
                "num_hidden_layers": 40,
                "num_key_value_heads": 2,
                "head_dim": 256,
            },
        }
        info = extract_model_info(config)
        assert info["model_dtype"] == "float16"  # top-level wins
        assert info["num_layers"] == 32  # top-level wins
        assert info["num_kv_heads"] == 2  # filled from nested
        assert info["head_dim"] == 256  # filled from nested

    def test_dtype_field_recognized(self):
        """The 'dtype' field should be recognized as model_dtype."""
        config = {
            "dtype": "bfloat16",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
        }
        info = extract_model_info(config)
        assert info["model_dtype"] == "bfloat16"

    def test_torch_dtype_preferred_over_dtype(self):
        """torch_dtype should take precedence over dtype."""
        config = {
            "torch_dtype": "float16",
            "dtype": "bfloat16",
            "num_hidden_layers": 32,
        }
        info = extract_model_info(config)
        assert info["model_dtype"] == "float16"

    def test_fp8_quantization_config(self):
        """FP8 quantization_config should produce quant_dtype."""
        config = {
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 32,
            "quantization_config": {"quant_method": "fp8"},
        }
        info = extract_model_info(config)
        assert info["model_dtype"] == "bfloat16"  # torch_dtype unchanged
        assert info["quant_dtype"] == "fp8"

    def test_awq_quantization_config(self):
        """AWQ quantization_config should produce quant_dtype."""
        config = {
            "torch_dtype": "float16",
            "quantization_config": {"quant_method": "awq", "bits": 4},
        }
        info = extract_model_info(config)
        assert info["quant_dtype"] == "awq4"

    def test_gptq_quantization_config(self):
        """GPTQ quantization_config should produce quant_dtype."""
        config = {
            "torch_dtype": "float16",
            "quantization_config": {"quant_method": "gptq", "bits": 4},
        }
        info = extract_model_info(config)
        assert info["quant_dtype"] == "gptq"

    def test_no_quantization_config(self):
        """Without quantization_config, quant_dtype should be absent."""
        config = {"torch_dtype": "bfloat16", "num_hidden_layers": 32}
        info = extract_model_info(config)
        assert "quant_dtype" not in info

    def test_empty_quantization_config(self):
        """Empty quantization_config should not produce quant_dtype."""
        config = {
            "torch_dtype": "bfloat16",
            "quantization_config": {},
        }
        info = extract_model_info(config)
        assert "quant_dtype" not in info


class TestResolveQuantDtype:
    """Test _resolve_quant_dtype helper."""

    def test_fp8(self):
        assert _resolve_quant_dtype({"quant_method": "fp8"}) == "fp8"

    def test_awq_default_4bit(self):
        assert _resolve_quant_dtype({"quant_method": "awq"}) == "awq4"

    def test_awq_explicit_4bit(self):
        assert _resolve_quant_dtype({"quant_method": "awq", "bits": 4}) == "awq4"

    def test_awq_8bit(self):
        assert _resolve_quant_dtype({"quant_method": "awq", "bits": 8}) == "awq8"

    def test_gptq_default_4bit(self):
        assert _resolve_quant_dtype({"quant_method": "gptq"}) == "gptq"

    def test_gptq_8bit(self):
        assert _resolve_quant_dtype({"quant_method": "gptq", "bits": 8}) == "int8"

    def test_marlin(self):
        assert _resolve_quant_dtype({"quant_method": "marlin", "bits": 4}) == "gptq"

    def test_bitsandbytes_4bit(self):
        assert _resolve_quant_dtype({"quant_method": "bitsandbytes", "load_in_4bit": True}) == "int4"

    def test_bitsandbytes_nf4(self):
        assert _resolve_quant_dtype({"quant_method": "bitsandbytes", "quant_type": "nf4"}) == "int4"

    def test_bitsandbytes_8bit(self):
        assert _resolve_quant_dtype({"quant_method": "bitsandbytes", "load_in_8bit": True}) == "int8"

    def test_unknown_method(self):
        assert _resolve_quant_dtype({"quant_method": "unknown_method"}) is None

    def test_empty_method(self):
        assert _resolve_quant_dtype({"quant_method": ""}) is None

    def test_no_method(self):
        assert _resolve_quant_dtype({}) is None


class _FakeSafeTensorsInfo:
    """Minimal stand-in for huggingface_hub SafeTensorsInfo."""

    def __init__(self, parameters: dict[str, int], total: int):
        self.parameters = parameters
        self.total = total


class _FakeModelInfo:
    """Minimal stand-in for huggingface_hub ModelInfo."""

    def __init__(self, safetensors=None):
        self.safetensors = safetensors


class TestFetchSafetensorsParams:
    """Tests for fetch_safetensors_params (HF API-based param count)."""

    def test_returns_total_from_model_info(self):
        """Should return total param count from model_info API."""
        st_info = _FakeSafeTensorsInfo(
            parameters={"BF16": 5_000_000_000, "F32": 100_000},
            total=5_000_100_000,
        )
        mi = _FakeModelInfo(safetensors=st_info)
        with mock.patch("huggingface_hub.model_info", return_value=mi):
            result = fetch_safetensors_params("org/test-model")
        assert result == 5_000_100_000

    def test_returns_none_when_no_safetensors(self):
        """Should return None for models without safetensors (e.g. GGUF)."""
        mi = _FakeModelInfo(safetensors=None)
        with mock.patch("huggingface_hub.model_info", return_value=mi):
            result = fetch_safetensors_params("org/gguf-model")
        assert result is None

    def test_returns_none_on_api_error(self):
        """Should return None when API call fails."""
        with mock.patch("huggingface_hub.model_info", side_effect=Exception("network error")):
            result = fetch_safetensors_params("org/missing-model")
        assert result is None

    def test_passes_revision(self):
        """Should forward revision kwarg to model_info."""
        st_info = _FakeSafeTensorsInfo(parameters={"BF16": 1000}, total=1000)
        mi = _FakeModelInfo(safetensors=st_info)
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return mi

        with mock.patch("huggingface_hub.model_info", side_effect=_capture):
            fetch_safetensors_params("org/model", revision="v2")
        assert captured.get("revision") == "v2"

    def test_returns_none_when_total_zero(self):
        """Should return None when total is 0."""
        st_info = _FakeSafeTensorsInfo(parameters={}, total=0)
        mi = _FakeModelInfo(safetensors=st_info)
        with mock.patch("huggingface_hub.model_info", return_value=mi):
            result = fetch_safetensors_params("org/empty-model")
        assert result is None


class _FakeSibling:
    """Minimal stand-in for huggingface_hub RepoSibling."""

    def __init__(self, rfilename: str):
        self.rfilename = rfilename


class TestFetchSafetensorsSizeOrder:
    """Tests for try ordering in fetch_safetensors_size."""

    def test_index_total_size_used_for_sharded_models(self, tmp_path):
        """Index total_size should be used for sharded models (most reliable)."""
        import json

        index_file = tmp_path / "model.safetensors.index.json"
        index_file.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 20_000_000_000},
                    "weight_map": {"layer.weight": "model-00001.safetensors"},
                }
            )
        )

        # API would give different (unreliable for packed formats) value
        api_called = []

        def _fake_download(**kwargs):
            if kwargs.get("filename") == "model.safetensors.index.json":
                return str(index_file)
            raise FileNotFoundError("no such file")

        def _fake_model_info(**kwargs):
            api_called.append(1)
            return _FakeModelInfo(safetensors=None)

        with (
            mock.patch("huggingface_hub.model_info", side_effect=_fake_model_info),
            mock.patch("huggingface_hub.hf_hub_download", side_effect=_fake_download),
            mock.patch("huggingface_hub.utils.disable_progress_bars"),
            mock.patch("huggingface_hub.utils.enable_progress_bars"),
        ):
            result = fetch_safetensors_size("org/awq-model")

        assert result == 20_000_000_000  # index total_size
        assert api_called == []  # API not called when index succeeds

    def test_list_repo_tree_preferred_over_api_for_single_file(self):
        """list_repo_tree LFS size preferred over model_info API (more accurate
        for packed quant formats).
        """
        api_called = []

        def _fake_model_info(**kwargs):
            api_called.append(1)
            return _FakeModelInfo(safetensors=None)

        class _Entry:
            def __init__(self, rfilename, size):
                self.rfilename = rfilename
                self.size = size

        tree = [
            _Entry("config.json", 1024),
            _Entry("model.safetensors", 17_500_000_000),
            _Entry("tokenizer.json", 2048),
        ]

        with (
            mock.patch("huggingface_hub.model_info", side_effect=_fake_model_info),
            mock.patch("huggingface_hub.list_repo_tree", return_value=iter(tree)),
            mock.patch("huggingface_hub.hf_hub_download", side_effect=Exception("no index")),
            mock.patch("huggingface_hub.utils.disable_progress_bars"),
            mock.patch("huggingface_hub.utils.enable_progress_bars"),
        ):
            result = fetch_safetensors_size("org/single-file-model")

        assert result == 17_500_000_000
        assert api_called == []  # API not called when tree size succeeds

    def test_api_used_when_no_index_and_no_tree(self):
        """API is the last-resort metadata source when list_repo_tree yields nothing."""
        st_info = _FakeSafeTensorsInfo(
            parameters={"BF16": 7_000_000_000},
            total=7_000_000_000,
        )
        mi = _FakeModelInfo(safetensors=st_info)

        with (
            mock.patch("huggingface_hub.model_info", return_value=mi),
            mock.patch("huggingface_hub.list_repo_tree", return_value=iter([])),
            mock.patch("huggingface_hub.hf_hub_download", side_effect=Exception("not found")),
            mock.patch("huggingface_hub.utils.disable_progress_bars"),
            mock.patch("huggingface_hub.utils.enable_progress_bars"),
        ):
            result = fetch_safetensors_size("org/single-file-model")

        # BF16 = 2 bytes per element → 7B * 2 = 14B bytes
        assert result == 14_000_000_000

    def test_no_weight_file_download_when_index_missing(self):
        """Regression for #186: missing index must not trigger model.safetensors download.

        For quantized single-file models (e.g. NVFP4) without an
        ``index.json``, falling through to ``hf_hub_download(model.safetensors)``
        would download many GB of weights just to read a header, hanging the
        CLI.  Only ``model.safetensors.index.json`` is an acceptable file
        download from this function.
        """
        downloaded: list[str] = []

        def _fake_download(**kwargs):
            downloaded.append(kwargs.get("filename", ""))
            raise FileNotFoundError("no index here")

        class _Entry:
            def __init__(self, rfilename, size):
                self.rfilename = rfilename
                self.size = size

        tree = [
            _Entry("config.json", 1024),
            _Entry("model.safetensors", 17_500_000_000),
        ]

        st_info = _FakeSafeTensorsInfo(
            parameters={"F8_E4M3": 35_000_000_000},
            total=35_000_000_000,
        )
        mi = _FakeModelInfo(safetensors=st_info)

        with (
            mock.patch("huggingface_hub.model_info", return_value=mi),
            mock.patch("huggingface_hub.list_repo_tree", return_value=iter(tree)),
            mock.patch("huggingface_hub.hf_hub_download", side_effect=_fake_download),
            mock.patch("huggingface_hub.utils.disable_progress_bars"),
            mock.patch("huggingface_hub.utils.enable_progress_bars"),
        ):
            result = fetch_safetensors_size("org/nvfp4-single-file")

        assert result == 17_500_000_000  # LFS size from tree, not API params
        assert downloaded == ["model.safetensors.index.json"]


class TestTargetGpuMemory:
    """Accelerator-aware GPU memory budget (fixes DGX-hardcoded estimate)."""

    def test_override_scales_usable(self):
        e = estimate_vram(model_vram=28.75, gpu_memory_utilization=0.5, total_gpu_memory_gb=48.0)
        assert e.total_gpu_memory_gb == 48.0
        assert e.usable_gpu_memory_gb == pytest.approx(24.0)  # 48 * 0.5

    def test_default_is_dgx_spark(self):
        from sparkrun.models.vram import DGX_SPARK_VRAM_GB

        e = estimate_vram(model_vram=28.75, gpu_memory_utilization=0.5)
        assert e.total_gpu_memory_gb == DGX_SPARK_VRAM_GB
        assert e.usable_gpu_memory_gb == pytest.approx(DGX_SPARK_VRAM_GB * 0.5)

    def test_total_set_without_utilization(self):
        # total_gpu_memory_gb is populated even when no gpu_memory_utilization.
        e = estimate_vram(model_vram=28.75, total_gpu_memory_gb=48.0)
        assert e.total_gpu_memory_gb == 48.0

    def test_fit_uses_target_memory(self):
        # 28.75 GB weights fit a 48 GB card but not a 24 GB card.
        e48 = estimate_vram(model_vram=28.75, total_gpu_memory_gb=48.0)
        e24 = estimate_vram(model_vram=28.75, total_gpu_memory_gb=24.0)
        assert e48.total_per_gpu_gb <= e48.total_gpu_memory_gb
        assert e24.total_per_gpu_gb > e24.total_gpu_memory_gb


class TestResolveTargetAccelerator:
    def _cluster(self, model, mem):
        from sparkrun.core.hardware import AcceleratorSpec, HostHardware

        class C:
            hosts = ["h1"]
            hosts_hardware = {"h1": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model=model, memory_gb=mem)])}

        return C()

    def test_resolves_from_cluster_hardware(self):
        from sparkrun.utils.cli_formatters import _resolve_target_accelerator

        mem, model = _resolve_target_accelerator(self._cluster("rtx-a6000", 48.0), None)
        assert mem == 48.0 and model == "rtx-a6000"

    def test_none_when_no_hardware(self):
        from sparkrun.utils.cli_formatters import _resolve_target_accelerator

        class C:
            hosts = ["h1"]
            hosts_hardware = {}

        assert _resolve_target_accelerator(C(), None) == (None, None)
        assert _resolve_target_accelerator(None, None) == (None, None)


class TestKvBytesPerElement:
    """KV cache packing differs from weight packing for NVFP4."""

    def test_nvfp4_kv_includes_block_scales(self):
        # fp4 data (head_size // 2) + fp8 block scales (head_size // 16)
        assert bytes_per_element("nvfp4") == 0.5
        assert kv_bytes_per_element("nvfp4") == 0.5625

    def test_falls_back_to_weight_table(self):
        assert kv_bytes_per_element("bfloat16") == 2.0
        assert kv_bytes_per_element("fp8") == 1.0

    def test_unknown_dtype(self):
        assert kv_bytes_per_element("not-a-dtype") is None


class TestMlaKvLayout:
    """Fixed-width DeepSeek MLA KV slot layouts."""

    def _claims(self, dtype: str) -> bool:
        strategy, _ = resolve_kv_strategy(ArchInfo(kv_dtype=dtype))
        return strategy.name == "mla"

    def test_recognized_layouts(self):
        assert self._claims("nvfp4_ds_mla")
        assert self._claims("fp8_ds_mla")
        assert self._claims("nvfp4-ds-mla")  # hyphen spelling
        assert not self._claims("bfloat16")
        assert not self._claims("nvfp4")

    def test_packed_layouts_are_valid_kv_dtypes(self):
        """They have no per-element width, so validation must ask the registry.

        Without this a recipe naming ``kv_dtype: nvfp4_ds_mla`` fails validation
        as an unrecognized dtype.
        """
        assert kv_bytes_per_element("nvfp4_ds_mla") is None
        assert is_valid_kv_dtype("nvfp4_ds_mla")
        assert is_valid_kv_dtype("bfloat16")
        assert not is_valid_kv_dtype("bogus_dtype")

    def test_v3_style_656_byte_slot(self):
        """Without compress_ratios every layer stores one 656-byte slot per token."""
        assert mla_kv_bytes_per_token(kv_dtype="fp8_ds_mla", num_layers=61) == 61 * 656

    def test_deepseek_v4_uses_584_byte_envelope(self, deepseek_v4_config):
        # 21 layers at ratio 4 + 20 at ratio 128; ratio-0 layers are excluded.
        expected = 21 * (584 / 4) + 20 * (584 / 128)
        got = mla_kv_bytes_per_token(
            kv_dtype="nvfp4_ds_mla",
            num_layers=43,
            compress_ratios=deepseek_v4_config["compress_ratios"],
            model_type="deepseek_v4",
        )
        assert got == pytest.approx(expected)
        assert got == pytest.approx(3157.25)

    def test_fp8_and_nvfp4_share_the_v4_envelope(self, deepseek_v4_config):
        kwargs = dict(num_layers=43, compress_ratios=deepseek_v4_config["compress_ratios"], model_type="deepseek_v4")
        assert mla_kv_bytes_per_token(kv_dtype="fp8_ds_mla", **kwargs) == mla_kv_bytes_per_token(kv_dtype="nvfp4_ds_mla", **kwargs)

    def test_generic_mla_sizes_from_latent(self):
        """Non-slot dtypes: (kv_lora_rank + qk_rope_head_dim) elements per layer."""
        got = mla_kv_bytes_per_token(kv_dtype="bfloat16", num_layers=61, kv_lora_rank=512, qk_rope_head_dim=64)
        assert got == 61 * (512 + 64) * 2.0

    def test_returns_none_without_enough_info(self):
        assert mla_kv_bytes_per_token(kv_dtype="bfloat16", num_layers=61) is None
        assert mla_kv_bytes_per_token(kv_dtype="fp8_ds_mla") is None


class TestMlaEstimateVram:
    """End-to-end MLA sizing through estimate_vram()."""

    def _v4_kwargs(self, config, **overrides):
        info = extract_model_info(config)
        kwargs = dict(
            model_vram=340.0,
            num_layers=info["num_layers"],
            num_kv_heads=info["num_kv_heads"],
            head_dim=info["head_dim"],
            arch=_arch_from(info),
            model_type=info.get("model_type"),
            max_model_len=1_048_576,
            kv_dtype="nvfp4_ds_mla",
        )
        kwargs.update(overrides)
        return kwargs

    def test_nvfp4_ds_mla_replaces_the_bf16_mha_estimate(self, deepseek_v4_config):
        """The generic 2*L*heads*head_dim formula reads 86 GB; MLA is ~3 GB."""
        generic = estimate_vram(
            model_vram=340.0,
            num_layers=43,
            num_kv_heads=1,
            head_dim=512,
            max_model_len=1_048_576,
        )
        assert generic.kv_cache_total_gb == pytest.approx(86.0, abs=0.1)
        assert generic.kv_arch != "mla"

        est = estimate_vram(**self._v4_kwargs(deepseek_v4_config))
        assert est.kv_arch == "mla"
        assert est.kv_cache_per_token_bytes == pytest.approx(3157.25)
        assert est.kv_cache_total_gb == pytest.approx(3.08, abs=0.01)

    def test_kv_cache_is_replicated_across_tensor_parallel_ranks(self, deepseek_v4_config):
        """MLA's latent has no head dim to shard, so TP does not shrink the KV cache."""
        tp1 = estimate_vram(**self._v4_kwargs(deepseek_v4_config, tensor_parallel=1))
        tp2 = estimate_vram(**self._v4_kwargs(deepseek_v4_config, tensor_parallel=2))
        assert tp2.kv_cache_replicated
        # Weights halve, KV does not.
        assert tp2.total_per_gpu_gb == pytest.approx(tp1.model_weights_gb / 2 + tp1.kv_cache_total_gb)

    def test_pipeline_parallel_still_splits_the_kv_cache(self, deepseek_v4_config):
        """Layers — and therefore their latent caches — do split across PP stages."""
        pp1 = estimate_vram(**self._v4_kwargs(deepseek_v4_config, pipeline_parallel=1))
        pp2 = estimate_vram(**self._v4_kwargs(deepseek_v4_config, pipeline_parallel=2))
        assert pp2.total_per_gpu_gb == pytest.approx(pp1.model_weights_gb / 2 + pp1.kv_cache_total_gb / 2)

    def test_warns_that_auxiliary_caches_are_excluded(self, deepseek_v4_config):
        est = estimate_vram(**self._v4_kwargs(deepseek_v4_config))
        assert any("sliding-window" in w for w in est.warnings)

    def test_mla_layout_alone_is_enough(self):
        """A recipe naming nvfp4_ds_mla gets MLA sizing even with no architecture info."""
        est = estimate_vram(model_vram=340.0, kv_dtype="nvfp4_ds_mla", num_layers=43, max_model_len=131_072)
        assert est.kv_arch == "mla"
        assert est.kv_cache_per_token_bytes == 43 * 656

    def test_kv_vram_per_token_override_still_wins(self, deepseek_v4_config):
        est = estimate_vram(**self._v4_kwargs(deepseek_v4_config, kv_vram_per_token=1e-6))
        assert est.kv_cache_total_gb == pytest.approx(1e-6 * 1_048_576)

    def test_unsizable_mla_degrades_with_a_warning(self):
        est = estimate_vram(model_vram=10.0, kv_dtype="bfloat16", max_model_len=4096, arch={"kv_lora_rank": 512})
        assert est.kv_cache_total_gb is None
        assert any("MLA" in w for w in est.warnings)


class TestExtractModelInfoMla:
    """MLA architecture fields pulled out of a HuggingFace config."""

    def test_deepseek_v4(self, deepseek_v4_config):
        info = extract_model_info(deepseek_v4_config)
        assert info["model_type"] == "deepseek_v4"
        assert info["qk_rope_head_dim"] == 64
        # V4 has no kv_lora_rank: head_dim (512) is the whole cached width with
        # the RoPE tail carved out of it, so the NoPE part is 512 - 64 = 448.
        assert info["kv_lora_rank"] == 448
        # The contract that matters: NoPE + RoPE reconstructs head_dim exactly,
        # so the tail is counted once.
        assert info["kv_lora_rank"] + info["qk_rope_head_dim"] == deepseek_v4_config["head_dim"]
        assert info["compress_ratios"][:4] == [0, 0, 4, 128]

    def test_deepseek_v3_uses_kv_lora_rank(self):
        info = extract_model_info(
            {
                "model_type": "deepseek_v3",
                "num_hidden_layers": 61,
                "num_attention_heads": 128,
                "hidden_size": 7168,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
            }
        )
        assert info["kv_lora_rank"] == 512
        assert "compress_ratios" not in info

    def test_non_mla_model_has_no_mla_fields(self):
        info = extract_model_info({"model_type": "llama", "num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096})
        assert "kv_lora_rank" not in info
        assert "qk_rope_head_dim" not in info


class TestMlaDetectionSignals:
    """Each independent signal that should select MLA sizing."""

    def test_qk_rope_head_dim_with_pinned_head_dim(self):
        """A recipe pinning head_dim + qk_rope_head_dim skips HF detection but still gets MLA sizing.

        V4 shape: head_dim is the full cached width, so the per-layer footprint
        is ``head_dim * bytes`` — the RoPE tail must not be added on top.
        """
        est = estimate_vram(
            model_vram=340.0, num_layers=43, num_kv_heads=1, head_dim=512, max_model_len=32768, arch={"qk_rope_head_dim": 64}
        )
        assert est.kv_arch == "mla"
        assert est.kv_cache_per_token_bytes == 43 * 512 * 2.0

    def test_kv_lora_rank_alone(self):
        est = estimate_vram(model_vram=340.0, num_layers=61, max_model_len=32768, arch={"kv_lora_rank": 512, "qk_rope_head_dim": 64})
        assert est.kv_arch == "mla"
        # DeepSeek's published figure for V3: ~70 KB per token.
        assert est.kv_cache_per_token_bytes == 70_272

    def test_non_mla_model_is_untouched(self):
        """Standard GQA sizing must not change for models with no MLA markers."""
        est = estimate_vram(model_vram=60.0, num_layers=64, num_kv_heads=8, head_dim=128, max_model_len=32768)
        assert est.kv_arch != "mla"
        assert not est.kv_cache_replicated
        assert est.kv_cache_per_token_bytes == 2.0 * 64 * 8 * 128 * 2.0


class TestMlaLatentDim:
    """The V2/V3 vs V4 asymmetry that makes the RoPE tail easy to double-count.

    Upstream vLLM sizes V4's MLA cache from ``head_dim`` directly (documented as
    "448B NoPE + 128B RoPE + 8B fp8 scale = 584B"), but sizes V3.2's from
    ``kv_lora_rank + qk_rope_head_dim`` (head_size=576).  Normalizing both to
    the NoPE width lets one formula add the tail exactly once.
    """

    def test_v3_shape_uses_kv_lora_rank_verbatim(self):
        """V2/V3 cache the RoPE tail *in addition* to the named latent."""
        assert mla_latent_dim(kv_lora_rank=512, qk_rope_head_dim=64) == 512

    def test_v4_shape_carves_the_tail_out_of_head_dim(self):
        """V4 has no kv_lora_rank; head_dim is the whole width."""
        assert mla_latent_dim(head_dim=512, qk_rope_head_dim=64) == 448

    def test_explicit_kv_lora_rank_wins_over_head_dim(self):
        """When a config carries both, the named latent is authoritative."""
        assert mla_latent_dim(kv_lora_rank=512, head_dim=56, qk_rope_head_dim=64) == 512

    def test_degenerate_head_dim_is_not_carved(self):
        """A head_dim at or below the tail width would go zero/negative."""
        assert mla_latent_dim(head_dim=64, qk_rope_head_dim=64) == 64
        assert mla_latent_dim(head_dim=32, qk_rope_head_dim=64) == 32

    def test_unresolvable(self):
        assert mla_latent_dim(qk_rope_head_dim=64) is None
        assert mla_latent_dim() is None

    def test_v4_width_matches_upstream_slot_geometry(self, deepseek_v4_config):
        """End-to-end: the generic-dtype V4 estimate must equal head_dim * bytes.

        Upstream's fp8 slot decomposes as 448 NoPE + 64 RoPE = 512 elements, so
        a bf16 KV cache is 512 * 2 = 1024 B per token per layer before
        compression.  Summed over the compressed layers that is 5,536 B/token —
        adding the tail twice gives 6,228 (+12.5%).
        """
        info = extract_model_info(deepseek_v4_config)
        got = mla_kv_bytes_per_token(
            kv_dtype="bfloat16",
            num_layers=info["num_layers"],
            kv_lora_rank=info["kv_lora_rank"],
            qk_rope_head_dim=info["qk_rope_head_dim"],
            compress_ratios=info["compress_ratios"],
            model_type=info["model_type"],
        )
        multiplier = 21 / 4 + 20 / 128  # 5.40625
        assert got == pytest.approx(512 * 2.0 * multiplier)
        assert got == pytest.approx(5536.0)

    def test_fixed_slot_path_is_unaffected_by_the_latent(self, deepseek_v4_config):
        """*_ds_mla short-circuits on the slot table before the latent is read."""
        kwargs = dict(
            num_layers=43,
            compress_ratios=deepseek_v4_config["compress_ratios"],
            model_type="deepseek_v4",
            qk_rope_head_dim=64,
        )
        assert mla_kv_bytes_per_token(kv_dtype="nvfp4_ds_mla", kv_lora_rank=448, **kwargs) == pytest.approx(3157.25)
        assert mla_kv_bytes_per_token(kv_dtype="nvfp4_ds_mla", kv_lora_rank=99999, **kwargs) == pytest.approx(3157.25)


class TestDegenerateCompressRatios:
    """A ratio list with nothing above 1 must not read as "0 GB of KV needed".

    ``sum()`` over an empty generator is ``0``, which is not ``None`` — so the
    unsizable guard would be skipped, ``kv_cache_total_gb`` would be 0.0, and
    the falsy checks downstream would suppress the per-GPU KV and the context
    budget too.  A zero claim passes every fit check, so the scheduler places a
    workload that then OOMs — the one failure mode here that is worse than a
    refused placement.
    """

    @pytest.mark.parametrize(
        "ratios",
        [
            pytest.param([0] * 43, id="all-zero"),
            pytest.param([1] * 43, id="all-one"),
            pytest.param([0, 1, 0, 1, 1, 0], id="mixed-at-or-below-one"),
            pytest.param([-4, 0, 1], id="negative"),
        ],
    )
    def test_unsizable_rather_than_zero(self, ratios):
        assert (
            mla_kv_bytes_per_token(
                kv_dtype="nvfp4_ds_mla",
                num_layers=len(ratios),
                compress_ratios=ratios,
                model_type="deepseek_v4",
            )
            is None
        )

    def test_estimate_reports_it_instead_of_claiming_zero(self):
        est = estimate_vram(
            model_vram=340.0,
            kv_dtype="nvfp4_ds_mla",
            num_layers=43,
            model_type="deepseek_v4",
            max_model_len=1_048_576,
            arch={"compress_ratios": [0] * 43},
        )
        assert est.kv_cache_total_gb is None
        assert est.kv_cache_per_token_bytes is None
        assert any("Cannot size MLA KV cache" in w for w in est.warnings)
        # Detection is independent of sizing: failing to size an MLA model does
        # not make it a dense one.  Relabelling it would misreport the
        # architecture to `to_dict()` (benchmark export) and the CLI, and flip
        # the replication rule that a kv_vram_per_token override still needs.
        assert est.kv_arch == "mla"
        assert est.kv_cache_replicated

    def test_one_compressed_layer_is_enough(self):
        """The guard must not swallow a genuinely small but non-zero cache."""
        got = mla_kv_bytes_per_token(
            kv_dtype="nvfp4_ds_mla",
            num_layers=43,
            compress_ratios=[0] * 42 + [128],
            model_type="deepseek_v4",
        )
        assert got == pytest.approx(584 / 128)


class TestUnsizableMlaWarningNamesTheCause:
    """The warning must name the actual gap, not blame the dtype every time."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param(
                dict(kv_dtype="nvfp4_ds_mla", num_layers=43, arch={"compress_ratios": [0] * 43}, model_type="deepseek_v4"),
                "compress_ratios",
                id="degenerate-ratios",
            ),
            pytest.param(dict(kv_dtype="bfloat16", num_layers=61, arch={"qk_rope_head_dim": 64}), "kv_lora_rank", id="no-latent"),
            pytest.param(dict(kv_dtype="bogus_dtype", num_layers=61, arch={"kv_lora_rank": 512}), "unknown KV cache dtype", id="bad-dtype"),
            pytest.param(dict(kv_dtype="fp8_ds_mla"), "num_layers", id="no-layers"),
        ],
    )
    def test_reason_is_specific(self, kwargs, expected):
        est = estimate_vram(model_vram=10.0, max_model_len=4096, **kwargs)
        assert est.kv_cache_total_gb is None
        assert any(expected in w for w in est.warnings), est.warnings


class TestAuxiliaryCacheWarning:
    """The estimate is a floor for sparse-attention models; say so.

    Keying the warning on ``compress_ratios`` misses DeepSeek V3.2, which has a
    sparse indexer but no per-layer compression — an under-estimate of roughly
    20% delivered with no caveat at all.
    """

    def _estimate(self, info, **overrides):
        kwargs = dict(
            model_vram=340.0,
            kv_dtype="fp8_ds_mla",
            max_model_len=32768,
            num_layers=info.get("num_layers"),
            num_kv_heads=info.get("num_kv_heads"),
            head_dim=info.get("head_dim"),
            arch=_arch_from(info),
            model_type=info.get("model_type"),
        )
        kwargs.update(overrides)
        return estimate_vram(**kwargs)

    def test_v32_warns_about_the_sparse_indexer(self, deepseek_v32_config):
        info = extract_model_info(deepseek_v32_config)
        assert info["index_head_dim"] == 128
        warnings = self._estimate(info).warnings
        assert any("sparse-indexer" in w for w in warnings), warnings
        # V3.2 has no sliding-window cache, so the wording must not claim one.
        assert not any("sliding-window" in w for w in warnings)

    def test_v4_warns_about_both(self, deepseek_v4_config):
        info = extract_model_info(deepseek_v4_config)
        warnings = self._estimate(info, kv_dtype="nvfp4_ds_mla").warnings
        assert any("sliding-window" in w and "sparse-indexer" in w for w in warnings), warnings
        assert any("caches are not included" in w for w in warnings), warnings

    def test_plain_mla_gets_no_spurious_warning(self, deepseek_v3_config):
        """DeepSeek-V3 has no auxiliary cache — the estimate is complete."""
        info = extract_model_info(deepseek_v3_config)
        assert "index_head_dim" not in info
        assert self._estimate(info).warnings == []


class TestReconcileCompressRatios:
    """`compress_ratios` is indexed by layer upstream, not summed wholesale.

    DeepSeek-V4-Flash ships 46 entries for 43 layers (the extras cover MTP /
    non-standard layers), so the list length legitimately exceeds the layer
    count and trimming must be silent in that case — but a short list, or a
    trimmed tail holding real compressed layers, changes the estimate and has
    to be surfaced.
    """

    def test_exact_length_is_untouched(self):
        ratios, note = reconcile_compress_ratios([4] * 43, 43)
        assert len(ratios) == 43
        assert note is None

    def test_trailing_padding_is_trimmed_silently(self, deepseek_v4_config):
        """The real V4 shape: 46 entries, 43 layers, trailing zeros."""
        ratios, note = reconcile_compress_ratios(deepseek_v4_config["compress_ratios"], 43)
        assert len(ratios) == 43
        assert note is None

    def test_real_v4_estimate_is_unchanged_and_quiet(self, deepseek_v4_config):
        info = extract_model_info(deepseek_v4_config)
        est = estimate_vram(
            model_vram=340.0,
            kv_dtype="nvfp4_ds_mla",
            max_model_len=1_048_576,
            num_layers=info["num_layers"],
            model_type=info["model_type"],
            arch={"compress_ratios": info["compress_ratios"], "index_head_dim": info.get("index_head_dim")},
        )
        assert est.kv_cache_per_token_bytes == pytest.approx(3157.25)
        assert not any("compress_ratios lists" in w for w in est.warnings)

    def test_short_list_is_reported(self):
        ratios, note = reconcile_compress_ratios([4] * 10, 61)
        assert len(ratios) == 10  # nothing to trim; we cannot invent the rest
        assert note is not None and "the remainder is unsized" in note

    def test_trimmed_tail_with_real_layers_is_reported(self):
        ratios, note = reconcile_compress_ratios([4] * 43 + [4, 128], 43)
        assert len(ratios) == 43
        assert note is not None and "beyond the layer count were ignored" in note

    def test_note_surfaces_as_an_estimate_warning(self):
        est = estimate_vram(
            model_vram=10.0,
            kv_dtype="nvfp4_ds_mla",
            num_layers=61,
            model_type="deepseek_v4",
            max_model_len=4096,
            arch={"compress_ratios": [4] * 10},
        )
        assert any("compress_ratios lists 10 layers" in w for w in est.warnings), est.warnings

    def test_no_num_layers_means_no_reconciliation(self):
        ratios, note = reconcile_compress_ratios([4, 128], None)
        assert list(ratios) == [4, 128]
        assert note is None


class TestKvVramPerTokenOverrideSharding:
    """The override follows the same replication rule as a computed MLA estimate.

    `is_mla` is resolved before the override branch and drives `kv_shard_factor`
    for it too, so an MLA recipe's hand-calibrated per-token figure is divided
    by PP alone. That is the correct behaviour — the latent is replicated — but
    it is a contract change from the pre-MLA code, so pin it.
    """

    _BASE = dict(model_vram=100.0, kv_vram_per_token=1e-5, max_model_len=100_000)

    def test_non_mla_override_is_divided_by_tp_and_pp(self):
        est = estimate_vram(**self._BASE, tensor_parallel=4, pipeline_parallel=2)
        assert not est.kv_cache_replicated
        assert est.total_per_gpu_gb == pytest.approx(100.0 / 8 + (1e-5 * 100_000) / 8)

    def test_mla_override_is_divided_by_pp_only(self):
        est = estimate_vram(**self._BASE, tensor_parallel=4, pipeline_parallel=2, arch={"kv_lora_rank": 512, "qk_rope_head_dim": 64})
        assert est.kv_arch == "mla" and est.kv_cache_replicated
        # Weights still shard by TP*PP; the KV override does not shard by TP.
        assert est.total_per_gpu_gb == pytest.approx(100.0 / 8 + (1e-5 * 100_000) / 2)

    def test_mla_layout_alone_also_replicates_the_override(self):
        est = estimate_vram(**self._BASE, tensor_parallel=4, kv_dtype="nvfp4_ds_mla")
        assert est.kv_cache_replicated
        assert est.total_per_gpu_gb == pytest.approx(100.0 / 4 + 1e-5 * 100_000)

    def test_override_still_beats_computed_mla_sizing(self, deepseek_v4_config):
        """The override wins outright — no MLA arithmetic is performed."""
        info = extract_model_info(deepseek_v4_config)
        est = estimate_vram(
            model_vram=340.0,
            kv_vram_per_token=1e-6,
            max_model_len=1_048_576,
            kv_dtype="nvfp4_ds_mla",
            num_layers=info["num_layers"],
            model_type=info["model_type"],
            arch={"compress_ratios": info["compress_ratios"]},
        )
        assert est.kv_cache_total_gb == pytest.approx(1e-6 * 1_048_576)


class TestNestedMlaConfigs:
    """Multimodal wrappers hide the text architecture in a nested sub-config.

    Two ways MLA was lost: `model_type` was read only from the top level (so a
    wrapper took the 656 B fallback instead of V4's 584 B envelope), and the
    nested scan was gated on the *core* architecture keys — a wrapper complete
    for those never had its nested MLA markers read at all.
    """

    def _wrapper(self, inner, **top):
        return {"model_type": "deepseek_vl_v2", "text_config": inner, **top}

    def test_nested_mla_found_when_top_level_is_incomplete(self, deepseek_v4_config):
        info = extract_model_info(self._wrapper(deepseek_v4_config))
        assert info["model_type"] == "deepseek_v4"
        assert info["kv_lora_rank"] == 448
        assert info["compress_ratios"] == deepseek_v4_config["compress_ratios"]

    def test_nested_mla_found_when_top_level_is_complete(self, deepseek_v4_config):
        """The regression case: core keys present up top, MLA markers only below."""
        info = extract_model_info(
            self._wrapper(
                deepseek_v4_config,
                torch_dtype="bfloat16",
                num_hidden_layers=43,
                num_attention_heads=64,
                num_key_value_heads=1,
                hidden_size=4096,
                head_dim=512,
            )
        )
        assert info["model_type"] == "deepseek_v4"
        assert not _MLA_KEYS.isdisjoint(info)

    def test_nested_wrapper_sizes_to_the_v4_envelope(self, deepseek_v4_config):
        """End-to-end: the wrapper must not fall back to the 656 B default."""
        info = extract_model_info(self._wrapper(deepseek_v4_config))
        got = mla_kv_bytes_per_token(
            kv_dtype="nvfp4_ds_mla",
            num_layers=info["num_layers"],
            compress_ratios=info["compress_ratios"],
            model_type=info["model_type"],
        )
        assert got == pytest.approx(3157.25)

    def test_wrapper_model_type_kept_when_nested_is_not_mla(self):
        """An ordinary multimodal model keeps the wrapper's model_type."""
        info = extract_model_info(
            {
                "model_type": "qwen2_vl",
                "text_config": {
                    "model_type": "qwen2",
                    "torch_dtype": "bfloat16",
                    "num_hidden_layers": 28,
                    "num_attention_heads": 28,
                    "num_key_value_heads": 4,
                    "hidden_size": 3584,
                },
            }
        )
        assert info["model_type"] == "qwen2_vl"
        assert info["num_layers"] == 28
        assert _MLA_KEYS.isdisjoint(info)

    def test_flat_configs_are_unaffected(self, deepseek_v4_config, deepseek_v3_config):
        assert extract_model_info(deepseek_v4_config)["model_type"] == "deepseek_v4"
        assert extract_model_info(deepseek_v4_config)["kv_lora_rank"] == 448
        assert extract_model_info(deepseek_v3_config)["model_type"] == "deepseek_v3"
        assert extract_model_info(deepseek_v3_config)["kv_lora_rank"] == 512


class TestWeakMlaSignalIsFlagged:
    """`qk_rope_head_dim` alone triggers MLA — say so rather than tighten it.

    Every shipping DeepSeek/Kimi config resolves a latent (naming kv_lora_rank,
    or being V4-shaped), so the weak path is unreachable from auto-detection
    and only arises from hand-pinned metadata. Requiring corroboration would
    cost that ergonomic for a hypothetical model; warning keeps it while making
    the assumption visible — and the assumption matters, because sizing a
    non-MLA model this way drops `2 * num_kv_heads` and under-estimates.
    """

    _MSG = "inferred from qk_rope_head_dim alone"

    def test_pinned_qk_rope_alone_warns(self):
        est = estimate_vram(model_vram=10.0, num_layers=32, num_kv_heads=8, head_dim=128, max_model_len=4096, arch={"qk_rope_head_dim": 64})
        assert est.kv_arch == "mla"
        assert any(self._MSG in w for w in est.warnings), est.warnings

    def test_kv_lora_rank_alone_warns_about_the_tail(self):
        """The mirror image: kv_lora_rank alone silently drops the RoPE tail."""
        est = estimate_vram(model_vram=10.0, num_layers=61, kv_dtype="bfloat16", max_model_len=4096, arch={"kv_lora_rank": 512})
        assert est.kv_arch == "mla"
        assert any("RoPE tail" in w and "kv_lora_rank" in w for w in est.warnings), est.warnings
        # And it genuinely under-estimates: tail omitted.
        assert est.kv_cache_per_token_bytes == 61 * 512 * 2.0

    def test_both_markers_warn_never(self):
        est = estimate_vram(
            model_vram=10.0, num_layers=61, kv_dtype="bfloat16", max_model_len=4096, arch={"kv_lora_rank": 512, "qk_rope_head_dim": 64}
        )
        assert not any(self._MSG in w for w in est.warnings)
        assert not any("RoPE tail" in w for w in est.warnings)

    def test_explicit_latent_does_not_warn(self):
        est = estimate_vram(model_vram=10.0, num_layers=61, max_model_len=4096, arch={"kv_lora_rank": 512, "qk_rope_head_dim": 64})
        assert not any(self._MSG in w for w in est.warnings)

    def test_ds_mla_layout_does_not_warn(self):
        est = estimate_vram(model_vram=10.0, kv_dtype="nvfp4_ds_mla", num_layers=43, max_model_len=4096)
        assert not any(self._MSG in w for w in est.warnings)

    @pytest.mark.parametrize("fixture_name", ["deepseek_v4_config", "deepseek_v3_config", "deepseek_v32_config"])
    def test_auto_detected_configs_never_warn(self, request, fixture_name):
        """Auto-detection always resolves a latent, so this must stay quiet."""
        info = extract_model_info(request.getfixturevalue(fixture_name))
        est = estimate_vram(
            model_vram=340.0,
            kv_dtype="fp8_ds_mla",
            max_model_len=32768,
            num_layers=info["num_layers"],
            num_kv_heads=info.get("num_kv_heads"),
            head_dim=info.get("head_dim"),
            model_type=info.get("model_type"),
            arch={
                "kv_lora_rank": info.get("kv_lora_rank"),
                "qk_rope_head_dim": info.get("qk_rope_head_dim"),
                "compress_ratios": info.get("compress_ratios"),
                "index_head_dim": info.get("index_head_dim"),
            },
        )
        assert not any(self._MSG in w for w in est.warnings), est.warnings

    def test_the_underestimate_it_guards_against(self):
        """Documents the magnitude: a GQA shape sized as MLA reads ~10x low."""
        gqa = estimate_vram(model_vram=10.0, num_layers=32, num_kv_heads=8, head_dim=128, max_model_len=4096)
        as_mla = estimate_vram(
            model_vram=10.0, num_layers=32, num_kv_heads=8, head_dim=128, max_model_len=4096, arch={"qk_rope_head_dim": 64}
        )
        assert gqa.kv_cache_per_token_bytes == 2.0 * 32 * 8 * 128 * 2.0
        assert as_mla.kv_cache_per_token_bytes < gqa.kv_cache_per_token_bytes / 10


class TestMlaAloneLayoutOnNonMlaModel:
    """An `*_ds_mla` layout is authoritative MLA even with no architectural marker.

    That is only safe when the model genuinely is MLA. Forced onto a non-MLA
    model it sizes the latent instead of the real attention heads — an order of
    magnitude under-estimate in the OOM direction.
    """

    _MSG = "forces MLA sizing but the model has no MLA architecture markers"

    def test_non_mla_model_warns(self):
        est = estimate_vram(model_vram=60.0, num_layers=64, num_kv_heads=8, head_dim=128, kv_dtype="nvfp4_ds_mla", max_model_len=32768)
        assert est.kv_arch == "mla"
        assert any(self._MSG in w for w in est.warnings), est.warnings

    def test_mla_model_does_not_warn(self, deepseek_v4_config):
        est = estimate_vram(
            model_vram=340.0,
            kv_dtype="nvfp4_ds_mla",
            num_layers=43,
            max_model_len=32768,
            arch={"kv_lora_rank": 448, "qk_rope_head_dim": 64, "index_head_dim": 128},
        )
        assert not any(self._MSG in w for w in est.warnings)

    def test_v4_config_does_not_warn(self, deepseek_v4_config):
        """Auto-detected V4 carries kv_lora_rank, so no spurious warning."""
        info = extract_model_info(deepseek_v4_config)
        est = estimate_vram(
            model_vram=340.0,
            kv_dtype="nvfp4_ds_mla",
            num_layers=info["num_layers"],
            max_model_len=32768,
            arch={
                "kv_lora_rank": info["kv_lora_rank"],
                "qk_rope_head_dim": info["qk_rope_head_dim"],
                "index_head_dim": info.get("index_head_dim"),
            },
        )
        assert not any(self._MSG in w for w in est.warnings)


class TestShortCompressRatiosSizesAvailableLayersWithWarning:
    """A ratio list shorter than num_layers sizes what it covers and warns.

    Returning None for a short list serialized to 0.0 GB of KV at the placement
    boundary (total_per_gpu_gb is what the scheduler reads), which *also* passes
    the fit check and is strictly worse than sizing the layers the list does
    cover. So a short list sizes those layers and surfaces the gap as a warning,
    matching reconcile_compress_ratios' contract.
    """

    def test_short_list_sizes_the_available_layers(self):
        # 10 layers at ratio 4 -> 10 * 584/4 = 1460
        got = mla_kv_bytes_per_token(kv_dtype="nvfp4_ds_mla", num_layers=61, compress_ratios=[4] * 10, model_type="deepseek_v4")
        assert got == pytest.approx(10 * 584 / 4)

    def test_exact_and_long_lists_size_normally(self):
        assert mla_kv_bytes_per_token(
            kv_dtype="nvfp4_ds_mla", num_layers=43, compress_ratios=[4] * 43, model_type="deepseek_v4"
        ) == pytest.approx(43 * 584 / 4)
        real = [0, 0] + [4, 128] * 20 + [4, 0, 0, 0]
        assert mla_kv_bytes_per_token(
            kv_dtype="nvfp4_ds_mla", num_layers=43, compress_ratios=real, model_type="deepseek_v4"
        ) == pytest.approx(3157.25)

    def test_estimate_sizes_and_warns_for_the_short_list(self):
        est = estimate_vram(
            model_vram=10.0,
            kv_dtype="nvfp4_ds_mla",
            num_layers=61,
            model_type="deepseek_v4",
            max_model_len=4096,
            arch={"compress_ratios": [4] * 10},
        )
        assert est.kv_cache_per_token_bytes == pytest.approx(10 * 584 / 4)
        assert any("compress_ratios lists 10 layers" in w for w in est.warnings), est.warnings
