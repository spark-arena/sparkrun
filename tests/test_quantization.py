"""Tests for sparkrun.models.quantization module."""

from __future__ import annotations

from sparkrun.models.quantization import (
    _resolve_from_hf_quant_config,
    _resolve_from_quantization_config,
    fetch_hf_quant_config,
    resolve_from_gguf,
    resolve_quantization,
)


class TestResolveFromQuantizationConfig:
    """Test _resolve_from_quantization_config for all known quant methods."""

    def test_fp8(self):
        info = _resolve_from_quantization_config({"quant_method": "fp8"})
        assert info is not None
        assert info.method == "fp8"
        assert info.bits == 8
        assert info.weight_dtype == "fp8"

    def test_awq_default_4bit(self):
        info = _resolve_from_quantization_config({"quant_method": "awq"})
        assert info is not None
        assert info.method == "awq"
        assert info.bits == 4
        assert info.weight_dtype == "awq4"

    def test_awq_explicit_4bit(self):
        info = _resolve_from_quantization_config({"quant_method": "awq", "bits": 4})
        assert info.weight_dtype == "awq4"

    def test_awq_8bit(self):
        info = _resolve_from_quantization_config({"quant_method": "awq", "bits": 8})
        assert info.weight_dtype == "awq8"
        assert info.bits == 8

    def test_gptq_default_4bit(self):
        info = _resolve_from_quantization_config({"quant_method": "gptq"})
        assert info.weight_dtype == "gptq"
        assert info.bits == 4

    def test_gptq_8bit(self):
        info = _resolve_from_quantization_config({"quant_method": "gptq", "bits": 8})
        assert info.weight_dtype == "int8"
        assert info.bits == 8

    def test_marlin(self):
        info = _resolve_from_quantization_config({"quant_method": "marlin", "bits": 4})
        assert info.method == "marlin"
        assert info.weight_dtype == "gptq"

    def test_bitsandbytes_4bit(self):
        info = _resolve_from_quantization_config({"quant_method": "bitsandbytes", "load_in_4bit": True})
        assert info.method == "bitsandbytes"
        assert info.bits == 4
        assert info.weight_dtype == "int4"

    def test_bitsandbytes_nf4(self):
        info = _resolve_from_quantization_config({"quant_method": "bitsandbytes", "quant_type": "nf4"})
        assert info.weight_dtype == "int4"

    def test_bitsandbytes_8bit(self):
        info = _resolve_from_quantization_config({"quant_method": "bitsandbytes", "load_in_8bit": True})
        assert info.weight_dtype == "int8"
        assert info.bits == 8

    def test_bitsandbytes_no_bits_returns_none(self):
        info = _resolve_from_quantization_config({"quant_method": "bitsandbytes"})
        assert info is None

    def test_mxfp4(self):
        info = _resolve_from_quantization_config({"quant_method": "mxfp4"})
        assert info is not None
        assert info.method == "mxfp4"
        assert info.bits == 4
        assert info.weight_dtype == "mxfp4"

    def test_nvfp4(self):
        info = _resolve_from_quantization_config({"quant_method": "nvfp4"})
        assert info is not None
        assert info.method == "nvfp4"
        assert info.bits == 4
        assert info.weight_dtype == "nvfp4"

    def test_compressed_tensors_awq_style(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "strategy": "pack-quantized",
                            "type": "int",
                            "num_bits": 4,
                        }
                    }
                },
            }
        )
        assert info is not None
        assert info.method == "awq"
        assert info.bits == 4
        assert info.weight_dtype == "awq4"

    def test_compressed_tensors_channel_quantized(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "strategy": "channel-quantized",
                            "type": "int",
                            "num_bits": 8,
                        }
                    }
                },
            }
        )
        assert info is not None
        assert info.method == "compressed-tensors"
        assert info.bits == 8
        assert info.weight_dtype == "int8"

    def test_compressed_tensors_float_fp8(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "strategy": "channel-quantized",
                            "type": "float",
                            "num_bits": 8,
                        }
                    }
                },
            }
        )
        assert info is not None
        assert info.method == "fp8"
        assert info.bits == 8
        assert info.weight_dtype == "fp8"

    def test_compressed_tensors_fallback(self):
        """Unrecognizable config_groups structure falls back."""
        info = _resolve_from_quantization_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {"group_0": "not_a_dict"},
            }
        )
        assert info is not None
        assert info.method == "compressed-tensors"
        assert info.weight_dtype == "compressed-tensors"

    def test_compressed_tensors_no_config_groups(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "compressed-tensors",
            }
        )
        assert info is not None
        assert info.method == "compressed-tensors"

    def test_compressed_tensors_group_strategy_with_top_format(self):
        """GLM-style: weights.strategy='group' but top-level format='pack-quantized'."""
        info = _resolve_from_quantization_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "format": "pack-quantized",
                        "targets": ["Linear"],
                        "weights": {
                            "strategy": "group",
                            "type": "int",
                            "num_bits": 4,
                            "group_size": 32,
                            "symmetric": True,
                        },
                    }
                },
            }
        )
        assert info is not None
        assert info.method == "awq"
        assert info.bits == 4
        assert info.weight_dtype == "awq4"
        assert info.group_size == 32

    def test_compressed_tensors_group_strategy_with_group_format(self):
        """Group-level format takes precedence over top-level."""
        info = _resolve_from_quantization_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "format": "pack-quantized",
                        "weights": {
                            "strategy": "group",
                            "type": "int",
                            "num_bits": 4,
                        },
                    }
                },
            }
        )
        assert info is not None
        assert info.method == "awq"
        assert info.bits == 4
        assert info.weight_dtype == "awq4"

    def test_auto_round_int4(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "auto-round",
                "bits": 4,
                "data_type": "int",
                "group_size": 128,
                "sym": True,
            }
        )
        assert info is not None
        assert info.method == "auto-round"
        assert info.bits == 4
        assert info.weight_dtype == "int4"
        assert info.group_size == 128

    def test_autoround_alias(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "autoround",
                "bits": 4,
                "data_type": "int",
            }
        )
        assert info is not None
        assert info.method == "auto-round"
        assert info.bits == 4
        assert info.weight_dtype == "int4"

    def test_auto_round_8bit(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "auto-round",
                "bits": 8,
                "data_type": "int",
            }
        )
        assert info.bits == 8
        assert info.weight_dtype == "int8"

    def test_auto_round_fp_dtype(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "auto-round",
                "bits": 4,
                "data_type": "float",
            }
        )
        assert info.weight_dtype == "fp4"

    def test_auto_round_no_bits(self):
        info = _resolve_from_quantization_config(
            {
                "quant_method": "auto-round",
                "data_type": "int",
            }
        )
        assert info.bits == 4  # default
        assert info.weight_dtype == "int4"

    def test_unknown_method(self):
        assert _resolve_from_quantization_config({"quant_method": "unknown_method"}) is None

    def test_empty_method(self):
        assert _resolve_from_quantization_config({"quant_method": ""}) is None

    def test_no_method(self):
        assert _resolve_from_quantization_config({}) is None


class TestResolveFromHfQuantConfig:
    """Test _resolve_from_hf_quant_config for modelopt-style configs."""

    def test_nvfp4_with_kv_cache(self):
        info = _resolve_from_hf_quant_config(
            {
                "quantization": {
                    "quant_algo": "nvfp4",
                    "kv_cache_quant_algo": "fp8",
                    "group_size": 64,
                }
            }
        )
        assert info is not None
        assert info.method == "nvfp4"
        assert info.bits == 4
        assert info.weight_dtype == "nvfp4"
        assert info.kv_cache_quant == "fp8"
        assert info.group_size == 64

    def test_fp8_algo(self):
        info = _resolve_from_hf_quant_config(
            {
                "quantization": {
                    "quant_algo": "FP8",
                }
            }
        )
        assert info is not None
        assert info.method == "fp8"
        assert info.weight_dtype == "fp8"

    def test_no_kv_cache(self):
        info = _resolve_from_hf_quant_config(
            {
                "quantization": {
                    "quant_algo": "nvfp4",
                }
            }
        )
        assert info is not None
        assert info.kv_cache_quant is None
        assert info.group_size is None

    def test_no_quantization_block(self):
        assert _resolve_from_hf_quant_config({}) is None

    def test_empty_algo(self):
        assert _resolve_from_hf_quant_config({"quantization": {"quant_algo": ""}}) is None

    def test_non_dict_quantization(self):
        assert _resolve_from_hf_quant_config({"quantization": "not_a_dict"}) is None

    def test_mixed_precision_resolves_dominant_algo(self):
        """MIXED_PRECISION should resolve to the most common per-layer algo."""
        info = _resolve_from_hf_quant_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "FP8",
                    "quantized_layers": {
                        "layer.0.proj": {"quant_algo": "FP8"},
                        "layer.0.out": {"quant_algo": "FP8"},
                        "layer.1.expert.0.up": {"quant_algo": "NVFP4", "group_size": 16},
                        "layer.1.expert.0.down": {"quant_algo": "NVFP4", "group_size": 16},
                        "layer.1.expert.1.up": {"quant_algo": "NVFP4", "group_size": 16},
                        "layer.1.expert.1.down": {"quant_algo": "NVFP4", "group_size": 16},
                        "layer.1.expert.2.up": {"quant_algo": "NVFP4", "group_size": 16},
                        "layer.1.expert.2.down": {"quant_algo": "NVFP4", "group_size": 16},
                    },
                }
            }
        )
        assert info is not None
        assert info.method == "nvfp4"
        assert info.bits == 4
        assert info.weight_dtype == "nvfp4"
        assert info.kv_cache_quant == "fp8"
        assert info.group_size == 16

    def test_mixed_precision_all_fp8(self):
        """When all layers are FP8, dominant algo is fp8."""
        info = _resolve_from_hf_quant_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "layer.0.proj": {"quant_algo": "FP8"},
                        "layer.1.proj": {"quant_algo": "FP8"},
                    },
                }
            }
        )
        assert info is not None
        assert info.method == "fp8"
        assert info.weight_dtype == "fp8"

    def test_mixed_precision_empty_layers(self):
        """MIXED_PRECISION with no quantized_layers returns None."""
        info = _resolve_from_hf_quant_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {},
                }
            }
        )
        assert info is None

    def test_mixed_precision_no_layers_key(self):
        """MIXED_PRECISION without quantized_layers key returns None."""
        info = _resolve_from_hf_quant_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                }
            }
        )
        assert info is None


class TestFetchHfQuantConfig:
    """Test fetch_hf_quant_config with mocked HF download."""

    def test_successful_fetch(self, tmp_path, monkeypatch):
        import json

        config_path = tmp_path / "hf_quant_config.json"
        data = {"quantization": {"quant_algo": "nvfp4", "kv_cache_quant_algo": "fp8"}}
        config_path.write_text(json.dumps(data))

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda **kwargs: str(config_path),
        )
        result = fetch_hf_quant_config("org/model")
        assert result == data
        assert result["quantization"]["quant_algo"] == "nvfp4"

    def test_file_not_found_returns_none(self):
        """When hf_quant_config.json doesn't exist, return None."""
        result = fetch_hf_quant_config("nonexistent/model-without-quant-config")
        # This will fail gracefully (no HF hub access in tests)
        # Just verify it returns None without raising
        assert result is None


class TestResolveQuantization:
    """Test resolve_quantization priority ordering and source merging."""

    def test_recipe_quant_takes_priority(self):
        """Recipe metadata.quantization overrides HF config."""
        info = resolve_quantization(
            hf_config={
                "quantization_config": {"quant_method": "fp8"},
            },
            recipe_quant="awq",
        )
        assert info is not None
        assert info.method == "awq"
        assert info.weight_dtype == "awq4"

    def test_hf_config_over_hf_quant_config(self):
        """config.json quantization_config takes priority over hf_quant_config.json for method."""
        info = resolve_quantization(
            hf_config={
                "quantization_config": {"quant_method": "fp8"},
            },
            hf_quant_config={
                "quantization": {"quant_algo": "nvfp4", "kv_cache_quant_algo": "fp8"},
            },
        )
        assert info is not None
        assert info.method == "fp8"
        # But kv_cache_quant should be absorbed from hf_quant_config
        assert info.kv_cache_quant == "fp8"

    def test_hf_quant_config_alone(self):
        """hf_quant_config.json alone provides quantization info."""
        info = resolve_quantization(
            hf_config={"torch_dtype": "bfloat16"},  # no quantization_config
            hf_quant_config={
                "quantization": {"quant_algo": "nvfp4", "kv_cache_quant_algo": "fp8", "group_size": 64},
            },
        )
        assert info is not None
        assert info.method == "nvfp4"
        assert info.kv_cache_quant == "fp8"
        assert info.group_size == 64

    def test_recipe_quant_absorbs_kv_cache(self):
        """Recipe quant override still absorbs kv_cache_quant from hf_quant_config."""
        info = resolve_quantization(
            hf_quant_config={
                "quantization": {"quant_algo": "nvfp4", "kv_cache_quant_algo": "fp8"},
            },
            recipe_quant="nvfp4",
        )
        assert info is not None
        assert info.method == "nvfp4"
        assert info.kv_cache_quant == "fp8"

    def test_no_sources_returns_none(self):
        assert resolve_quantization() is None

    def test_empty_configs_returns_none(self):
        info = resolve_quantization(
            hf_config={"torch_dtype": "bfloat16"},
            hf_quant_config={},
        )
        assert info is None

    def test_recipe_quant_none_ignored(self):
        """recipe_quant='none' should not produce QuantizationInfo from recipe."""
        info = resolve_quantization(
            hf_config={"quantization_config": {"quant_method": "fp8"}},
            recipe_quant="none",
        )
        assert info is not None
        assert info.method == "fp8"  # falls through to HF config

    def test_recipe_quant_auto_ignored(self):
        info = resolve_quantization(
            hf_config={"quantization_config": {"quant_method": "awq", "bits": 4}},
            recipe_quant="auto",
        )
        assert info is not None
        assert info.method == "awq"

    def test_recipe_quant_empty_ignored(self):
        info = resolve_quantization(
            hf_config={"quantization_config": {"quant_method": "gptq"}},
            recipe_quant="",
        )
        assert info is not None
        assert info.method == "gptq"

    def test_group_size_merged_from_hf_quant_config(self):
        """group_size from hf_quant_config should be merged into config.json result."""
        info = resolve_quantization(
            hf_config={"quantization_config": {"quant_method": "fp8"}},
            hf_quant_config={
                "quantization": {"quant_algo": "fp8", "group_size": 128},
            },
        )
        assert info.group_size == 128

    def test_compressed_tensors_via_resolve(self):
        info = resolve_quantization(
            hf_config={
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "strategy": "pack-quantized",
                                "type": "int",
                                "num_bits": 4,
                            }
                        }
                    },
                }
            },
        )
        assert info is not None
        assert info.method == "awq"
        assert info.bits == 4

    def test_auto_round_via_resolve(self):
        info = resolve_quantization(
            hf_config={
                "quantization_config": {
                    "quant_method": "auto-round",
                    "bits": 4,
                    "data_type": "int",
                    "group_size": 128,
                }
            },
        )
        assert info is not None
        assert info.method == "auto-round"
        assert info.bits == 4
        assert info.weight_dtype == "int4"
        assert info.group_size == 128

    def test_gguf_via_resolve(self):
        """GGUF model spec should resolve when no other source matches."""
        info = resolve_quantization(model_id="Qwen/Qwen3-1.7B-GGUF:Q4_K_M")
        assert info is not None
        assert info.method == "gguf"
        assert info.weight_dtype == "q4_k_m"

    def test_gguf_not_used_when_hf_config_has_quant(self):
        """HF config quantization takes priority over GGUF model spec."""
        info = resolve_quantization(
            hf_config={"quantization_config": {"quant_method": "fp8"}},
            model_id="some/model:Q4_K_M",
        )
        assert info is not None
        assert info.method == "fp8"  # HF config wins

    def test_gguf_not_used_when_recipe_quant(self):
        """Recipe quant takes priority over GGUF model spec."""
        info = resolve_quantization(
            recipe_quant="awq",
            model_id="some/model:Q4_K_M",
        )
        assert info is not None
        assert info.method == "awq"


class TestResolveFromGguf:
    """Test resolve_from_gguf for GGUF model spec detection."""

    def test_q4_k_m(self):
        info = resolve_from_gguf("Qwen/Qwen3-1.7B-GGUF:Q4_K_M")
        assert info is not None
        assert info.method == "gguf"
        assert info.weight_dtype == "q4_k_m"

    def test_q8_0(self):
        info = resolve_from_gguf("Qwen/Qwen3-1.7B-GGUF:Q8_0")
        assert info is not None
        assert info.weight_dtype == "q8_0"

    def test_q6_k(self):
        info = resolve_from_gguf("some/model-GGUF:Q6_K")
        assert info is not None
        assert info.weight_dtype == "q6_k"

    def test_q3_k_m(self):
        info = resolve_from_gguf("some/model-GGUF:Q3_K_M")
        assert info is not None
        assert info.weight_dtype == "q3_k_m"

    def test_no_colon(self):
        assert resolve_from_gguf("meta-llama/Llama-3-8B") is None

    def test_empty_suffix(self):
        assert resolve_from_gguf("some/model:") is None

    def test_unrecognized_quant(self):
        assert resolve_from_gguf("some/model:BOGUS_QUANT") is None

    def test_non_gguf_with_colon(self):
        """Non-GGUF model with colon (e.g. docker image) should return None."""
        assert resolve_from_gguf("registry.io/image:latest") is None

    def test_q5_k_s_exact(self):
        """Q5_K_S has an exact match in _DTYPE_BYTES."""
        info = resolve_from_gguf("some/model:Q5_K_S")
        assert info is not None
        assert info.weight_dtype == "q5_k_s"

    def test_q5_k_l_strips_suffix(self):
        """Q5_K_L has no exact match, falls back to base q5_k."""
        info = resolve_from_gguf("some/model:Q5_K_L")
        assert info is not None
        assert info.weight_dtype == "q5_k"

    def test_q3_k_l_strips_suffix(self):
        info = resolve_from_gguf("some/model:Q3_K_L")
        assert info is not None
        assert info.weight_dtype == "q3_k"

    def test_iq4_xs(self):
        info = resolve_from_gguf("some/model:IQ4_XS")
        assert info is not None
        assert info.weight_dtype == "iq4_xs"

    def test_iq2_xxs(self):
        info = resolve_from_gguf("some/model:IQ2_XXS")
        assert info is not None
        assert info.weight_dtype == "iq2_xxs"

    def test_q4_0(self):
        info = resolve_from_gguf("some/model:Q4_0")
        assert info is not None
        assert info.weight_dtype == "q4_0"

    def test_q2_k(self):
        info = resolve_from_gguf("some/model:Q2_K")
        assert info is not None
        assert info.weight_dtype == "q2_k"

    def test_tq1_0(self):
        info = resolve_from_gguf("some/model:TQ1_0")
        assert info is not None
        assert info.weight_dtype == "tq1_0"
