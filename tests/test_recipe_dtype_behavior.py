"""Regression tests for dtype-resolution and max_model_len coercion in Recipe.estimate_vram.

Two behaviors pinned:

1.  ``_storage_dtype = hf_info.get("quant_dtype") or hf_info.get("model_dtype")``
    (recipe.py ~line 1099) — quant_dtype wins when both are present; model_dtype
    is used when quant_dtype is absent.

2.  ``max_model_len == "auto"`` is coerced to ``None`` before ``int()`` is called
    (recipe.py ~line 1169-1170) so VRAM estimation doesn't raise ValueError.

All tests use ``auto_detect=True`` with mocked HF helpers so no network access
is required.  The mock shapes match what the real helpers return.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sparkrun.core.recipe import Recipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recipe(model="test-org/test-model", **defaults) -> Recipe:
    """Build a minimal Recipe with optional defaults."""
    data: dict = {"name": "test", "model": model, "runtime": "vllm"}
    if defaults:
        data["defaults"] = defaults
    return Recipe.from_dict(data)


_MOCK_HF_CONFIG_BASE = {
    "torch_dtype": "bfloat16",
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}

_MOCK_HF_CONFIG_WITH_QUANT = {
    **_MOCK_HF_CONFIG_BASE,
    # A quantization_config that extract_model_info will resolve to fp8
    "quantization_config": {
        "quant_type": "fp8",
    },
}

# Patch targets — functions are imported locally inside Recipe.estimate_vram
_PATCH_FETCH_MODEL_CONFIG = "sparkrun.models.vram.fetch_model_config"
_PATCH_FETCH_HF_QUANT_CONFIG = "sparkrun.models.quantization.fetch_hf_quant_config"
_PATCH_FETCH_SAFETENSORS_SIZE = "sparkrun.models.vram.fetch_safetensors_size"
_PATCH_FETCH_SAFETENSORS_PARAMS = "sparkrun.models.vram.fetch_safetensors_params"


def _run_estimate(recipe: Recipe, hf_config: dict) -> object:
    """Run estimate_vram with a fake hf_config, suppressing network calls."""
    with (
        patch(_PATCH_FETCH_MODEL_CONFIG, return_value=hf_config),
        patch(_PATCH_FETCH_HF_QUANT_CONFIG, return_value=None),
        patch(_PATCH_FETCH_SAFETENSORS_SIZE, return_value=None),
        patch(_PATCH_FETCH_SAFETENSORS_PARAMS, return_value=None),
    ):
        return recipe.estimate_vram(auto_detect=True)


# ---------------------------------------------------------------------------
# Storage-dtype preference tests
# ---------------------------------------------------------------------------


def test_quant_dtype_wins_over_model_dtype():
    """When hf_config yields both quant_dtype and model_dtype, quant_dtype is used
    as _storage_dtype (and therefore as the fallback model_dtype for VRAM estimation).

    We inject a quantization_config that extract_model_info resolves to a non-None
    quant_dtype (fp8, 1 byte/param), alongside torch_dtype=bfloat16 (2 bytes/param).
    Because model_dtype is NOT set in recipe metadata, the HF path fills it in from
    _storage_dtype.  With quant_dtype winning, the estimate uses fp8 weights (~half
    the size of bfloat16).
    """
    # quant_method=fp8 is the key _resolve_from_quantization_config recognises.
    hf_config_with_quant = {
        **_MOCK_HF_CONFIG_BASE,
        "quantization_config": {"quant_method": "fp8"},
    }
    hf_config_no_quant = dict(_MOCK_HF_CONFIG_BASE)  # no quantization_config

    # No model_dtype in metadata so the HF path (not the metadata path) fills it in.
    recipe_quant = Recipe.from_dict(
        {
            "name": "quant-test",
            "model": "test-org/fp8-model",
            "runtime": "vllm",
            "metadata": {"model_params": "7B"},
        }
    )
    recipe_bf16 = Recipe.from_dict(
        {
            "name": "bf16-test",
            "model": "test-org/bf16-model",
            "runtime": "vllm",
            "metadata": {"model_params": "7B"},
        }
    )

    est_quant = _run_estimate(recipe_quant, hf_config_with_quant)
    est_bf16 = _run_estimate(recipe_bf16, hf_config_no_quant)

    # quant_dtype (fp8, 1 B/element) → smaller weights than model_dtype (bfloat16, 2 B/element)
    assert est_quant is not None
    assert est_bf16 is not None
    assert est_quant.model_weights_gb < est_bf16.model_weights_gb, (
        f"Expected fp8 ({est_quant.model_weights_gb:.2f} GB) < bfloat16 ({est_bf16.model_weights_gb:.2f} GB) — quant_dtype must win"
    )


def test_model_dtype_used_when_quant_dtype_absent():
    """When hf_config has no quantization_config, model_dtype (torch_dtype) is used."""
    # Recipe with model_params so the dtype actually drives the weight estimate.
    recipe = Recipe.from_dict(
        {
            "name": "bf16-only",
            "model": "test-org/plain-model",
            "runtime": "vllm",
            "metadata": {"model_params": "7B"},
        }
    )
    hf_config = {
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }

    est = _run_estimate(recipe, hf_config)

    assert est is not None
    # bfloat16 is 2 bytes/param; 7B params → ~13 GB weights
    assert est.model_weights_gb is not None
    assert est.model_weights_gb > 0, "model_dtype (bfloat16) must drive a positive weight estimate"


def test_quant_dtype_key_preferred_not_model_dtype_key():
    """Direct unit test: extract_model_info returns quant_dtype when quantization_config present.

    ``_resolve_from_quantization_config`` requires ``quant_method`` (the standard HF key)
    to identify the quantization type — not ``quant_type``.
    """
    from sparkrun.models.vram import extract_model_info

    hf_config = {
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        # quant_method is the key _resolve_from_quantization_config checks first.
        "quantization_config": {"quant_method": "fp8"},
    }

    info = extract_model_info(hf_config)

    # Both keys must be present — quant_dtype from quantization_config, model_dtype from torch_dtype
    assert "model_dtype" in info, "extract_model_info must return model_dtype"
    assert "quant_dtype" in info, "extract_model_info must return quant_dtype when quantization_config present"
    # The _storage_dtype expression ``hf_info.get("quant_dtype") or hf_info.get("model_dtype")``
    # resolves to quant_dtype.
    storage_dtype = info.get("quant_dtype") or info.get("model_dtype")
    assert storage_dtype == info["quant_dtype"], f"_storage_dtype should be quant_dtype={info['quant_dtype']!r}, got {storage_dtype!r}"
    assert storage_dtype != info.get("model_dtype"), "quant_dtype must differ from model_dtype so the preference is meaningful"


# ---------------------------------------------------------------------------
# max_model_len: "auto" coercion tests
# ---------------------------------------------------------------------------


def test_max_model_len_auto_in_defaults_resolves_to_none():
    """A recipe with ``max_model_len: 'auto'`` in defaults must yield None in the
    VRAMEstimate (not raise ValueError from int('auto')).
    """
    recipe = _recipe(max_model_len="auto")
    # auto_detect=False: no HF access needed to exercise the coercion path.
    est = recipe.estimate_vram(auto_detect=False)
    assert est is not None
    assert est.max_model_len is None, f"max_model_len='auto' must be coerced to None, got {est.max_model_len!r}"


def test_max_model_len_auto_via_cli_override_resolves_to_none():
    """``auto`` passed as a CLI override (not a recipe default) also coerces to None."""
    recipe = Recipe.from_dict({"name": "test", "model": "m", "runtime": "vllm"})
    est = recipe.estimate_vram(cli_overrides={"max_model_len": "auto"}, auto_detect=False)
    assert est is not None
    assert est.max_model_len is None, f"CLI override max_model_len='auto' must be coerced to None, got {est.max_model_len!r}"


def test_max_model_len_auto_does_not_raise():
    """Calling estimate_vram with max_model_len='auto' must not raise ValueError."""
    recipe = _recipe(max_model_len="auto")
    # Would raise ValueError: invalid literal for int() with base 10: 'auto'
    # before the fix at recipe.py ~line 1169.
    try:
        recipe.estimate_vram(auto_detect=False)
    except ValueError as exc:
        pytest.fail(f"estimate_vram raised ValueError with max_model_len='auto': {exc}")


def test_max_model_len_integer_still_works():
    """Numeric max_model_len values must still be coerced to int (not None)."""
    recipe = _recipe(max_model_len=4096)
    est = recipe.estimate_vram(auto_detect=False)
    assert est is not None
    assert est.max_model_len == 4096, f"Integer max_model_len must be preserved, got {est.max_model_len!r}"


def test_max_model_len_none_stays_none():
    """When max_model_len is absent from the recipe, the estimate returns None."""
    recipe = Recipe.from_dict({"name": "test", "model": "m", "runtime": "vllm"})
    est = recipe.estimate_vram(auto_detect=False)
    assert est is not None
    assert est.max_model_len is None
