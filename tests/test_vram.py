"""Tests for sparkrun.models.vram module."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.models.vram import (
    VRAMEstimate,
    _resolve_quant_dtype,
    bytes_per_element,
    estimate_vram,
    extract_model_info,
    fetch_safetensors_params,
    fetch_safetensors_size,
    parse_param_count,
)


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

    def test_default_kv_dtype_is_bfloat16(self):
        est = estimate_vram(
            model_params=7_000_000_000,
            model_dtype="float16",
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            max_model_len=4096,
        )
        assert est.kv_dtype == "bfloat16"

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
        assert est.context_multiplier == pytest.approx(
            est.max_context_tokens / 4096, abs=0.01
        )

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


class TestFetchSafetensorsSizeTry0:
    """Tests for model_info API Try 0 in fetch_safetensors_size."""

    def test_model_info_api_used_first(self, monkeypatch):
        """API try should succeed without downloading any files."""
        st_info = _FakeSafeTensorsInfo(
            parameters={"BF16": 7_000_000_000},
            total=7_000_000_000,
        )
        mi = _FakeModelInfo(safetensors=st_info)
        download_called = []

        with mock.patch("huggingface_hub.model_info", return_value=mi), \
             mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda **kw: download_called.append(1)):
            result = fetch_safetensors_size("org/test-model")

        # BF16 = 2 bytes per element → 7B * 2 = 14B bytes
        assert result == 14_000_000_000
        assert download_called == []  # no file download attempted

    def test_falls_through_when_api_fails(self, monkeypatch, tmp_path):
        """When API fails, should fall through to index file (Try 1)."""
        import json

        index_file = tmp_path / "model.safetensors.index.json"
        index_file.write_text(json.dumps({"metadata": {"total_size": 20_000_000_000}}))

        def _fake_download(**kwargs):
            if kwargs.get("filename") == "model.safetensors.index.json":
                return str(index_file)
            raise FileNotFoundError("no such file")

        with mock.patch("huggingface_hub.model_info", side_effect=Exception("offline")), \
             mock.patch("huggingface_hub.hf_hub_download", side_effect=_fake_download), \
             mock.patch("huggingface_hub.utils.disable_progress_bars"), \
             mock.patch("huggingface_hub.utils.enable_progress_bars"):
            result = fetch_safetensors_size("org/sharded-model")

        assert result == 20_000_000_000
