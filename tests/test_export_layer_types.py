"""``metadata.layer_types`` is left out of recipe export unless the KV estimate reads it."""

from __future__ import annotations

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.models.kv.mla import layer_types_affect_sizing

HYBRID = ["linear_attention", "linear_attention", "full_attention"] * 16


def _export(metadata):
    recipe = Recipe.from_dict({"model": "org/m", "runtime": "vllm-distributed", "container": "img:1", "metadata": metadata})
    return recipe, recipe.to_dict()["metadata"]


@pytest.mark.parametrize(
    "metadata,kept",
    [
        ({"model_type": "qwen4_exp_text", "layer_types": HYBRID}, False),  # hybrid but not MLA: dense sizing never reads it
        ({"model_type": "deepseek_v3", "layer_types": ["full_attention"] * 8}, False),  # MLA without linear layers
        ({"model_type": "deepseek_v4", "layer_types": HYBRID}, True),  # hybrid MLA: the refusal depends on it
        ({"kv_lora_rank": 512, "layer_types": HYBRID}, True),
        ({"kv_dtype": "fp8_ds_mla", "layer_types": HYBRID}, True),
    ],
)
def test_layer_types_kept_only_where_the_estimate_reads_it(metadata, kept):
    assert layer_types_affect_sizing(metadata) is kept
    recipe, exported = _export(dict(metadata))
    assert ("layer_types" in exported) is kept
    assert recipe.metadata["layer_types"] == metadata["layer_types"]  # the recipe itself keeps it


def test_other_metadata_is_untouched():
    _recipe, exported = _export({"model_type": "qwen4_exp_text", "layer_types": HYBRID, "num_layers": 48})
    assert exported["num_layers"] == 48 and exported["model_type"] == "qwen4_exp_text"
