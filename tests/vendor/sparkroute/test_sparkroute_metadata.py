# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Tests for the bridge's public model-card metadata.

The contract is `SPARKRUN_MODEL_METADATA_CONTRACT.md` in the SparkRoute
repository. Two properties dominate:

* **advisory** — nothing here may make a valid inference endpoint
  unpublishable, so every failure path degrades to "omit the field";
* **reported, never inferred** — an unknown value is absent rather than
  guessed, because the gateway ranks real deployments against these numbers.
"""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.plugins.sparkroute import metadata as metadata_mod
from sparkrun.plugins.sparkroute.metadata import build_model_metadata


def _recipe(**overrides) -> Recipe:
    data = {
        "model": "Qwen/Qwen3-32B",
        "runtime": "vllm",
        "container": "vllm/vllm-openai:latest",
        "defaults": {"max_model_len": 65536},
        "metadata": {"model_params": 32_000_000_000},
    }
    data.update(overrides)
    return Recipe(data)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_metadata_matches_the_contract_example():
    result = build_model_metadata(_recipe(), ["Qwen/Qwen3-32B"])
    assert result == {
        "Qwen/Qwen3-32B": {
            "size_b": 32.0,
            "context": 65536,
            "input_price": 0.0,
            "output_price": 0.0,
            "tags": ["local", "vllm"],
        }
    }


def test_every_key_also_appears_in_served_models():
    """The contract requires the map key to be an exact served model identity."""
    served = ["Qwen/Qwen3-32B", "qwen-alias"]
    result = build_model_metadata(_recipe(), served)
    assert set(result) <= set(served)


def test_aliases_of_one_workload_share_the_model_card():
    """Several served IDs are one model under --served-model-name, so each
    carries the same values rather than only the canonical name being described."""
    result = build_model_metadata(_recipe(), ["Qwen/Qwen3-32B", "qwen"])
    assert result["Qwen/Qwen3-32B"] == result["qwen"]


def test_entries_are_bounded(monkeypatch):
    monkeypatch.setattr(metadata_mod, "MAX_METADATA_ENTRIES", 2)
    result = build_model_metadata(_recipe(), ["a", "b", "c", "d"])
    assert len(result) == 2


def test_values_are_independent_objects():
    """A shared dict would let the gateway's own normalization of one entry
    silently rewrite every other served name."""
    result = build_model_metadata(_recipe(), ["a", "b"])
    result["a"]["context"] = 1
    assert result["b"]["context"] == 65536


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_size_is_reported_in_billions_of_parameters():
    result = build_model_metadata(_recipe(metadata={"model_params": 1_700_000_000}), ["m"])
    assert result["m"]["size_b"] == 1.7


def test_sub_billion_models_keep_a_positive_size():
    result = build_model_metadata(_recipe(metadata={"model_params": 500_000_000}), ["m"])
    assert result["m"]["size_b"] == 0.5


def test_unknown_parameter_count_omits_size_rather_than_guessing():
    """``model_params`` lands in metadata only when a launch estimated VRAM;
    absent means unknown, and a model-family default would be a number with no
    source that the gateway would rank against real deployments."""
    result = build_model_metadata(_recipe(metadata={}), ["m"])
    assert "size_b" not in result["m"]


@pytest.mark.parametrize("value", [0, -1, "not a number", None, float("inf"), float("nan")])
def test_invalid_parameter_counts_are_dropped(value):
    result = build_model_metadata(_recipe(metadata={"model_params": value}), ["m"])
    assert "size_b" not in result["m"]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def test_context_is_the_effective_runtime_limit_not_the_family_limit():
    """A recipe pinning 65536 on a 262k-context model serves 65536; reporting
    the larger number would have the gateway route requests the runtime rejects."""
    result = build_model_metadata(_recipe(defaults={"max_model_len": 65536}), ["m"])
    assert result["m"]["context"] == 65536


def test_auto_context_is_not_reportable():
    """``auto`` means the runtime decides, which sparkrun cannot report."""
    result = build_model_metadata(_recipe(defaults={"max_model_len": "auto"}), ["m"])
    assert "context" not in result["m"]


def test_absent_context_is_omitted():
    result = build_model_metadata(_recipe(defaults={}), ["m"])
    assert "context" not in result["m"]


@pytest.mark.parametrize("value", [0, -4096, "sixty thousand"])
def test_invalid_context_values_are_dropped(value):
    result = build_model_metadata(_recipe(defaults={"max_model_len": value}), ["m"])
    assert "context" not in result["m"]


def test_unreadable_config_chain_only_costs_the_context_field():
    recipe = _recipe()
    with mock.patch.object(Recipe, "build_config_chain", side_effect=RuntimeError("boom")):
        result = build_model_metadata(recipe, ["m"])
    assert "context" not in result["m"]
    assert result["m"]["size_b"] == 32.0


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------


def test_local_inference_is_a_known_zero_price_not_an_unknown_one():
    """The contract distinguishes numeric zero (known free) from omission
    (unknown).  Sparkrun workloads run on the operator's own hardware, and
    reporting them as unpriced would stop a lowest_cost selector preferring
    them over a paid API."""
    result = build_model_metadata(_recipe(), ["m"])
    assert result["m"]["input_price"] == 0.0
    assert result["m"]["output_price"] == 0.0


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_tags_describe_locality_runtime_and_quantization():
    recipe = _recipe(runtime="sglang", metadata={"model_params": 1, "quantization": "FP8"})
    assert build_model_metadata(recipe, ["m"])["m"]["tags"] == ["fp8", "local", "sglang"]


def test_tags_are_sorted_and_deduplicated():
    tags = build_model_metadata(_recipe(metadata={"quantization": "vllm"}), ["m"])["m"]["tags"]
    assert tags == sorted(set(tags))


def test_unprintable_tag_values_are_dropped():
    recipe = _recipe(metadata={"model_params": 1, "quantization": "bad\x00value"})
    assert build_model_metadata(recipe, ["m"])["m"]["tags"] == ["local", "vllm"]


def test_oversized_tags_are_dropped(monkeypatch):
    monkeypatch.setattr(metadata_mod, "MAX_TAG_BYTES", 4)
    recipe = _recipe(metadata={"model_params": 1, "quantization": "a" * 32})
    assert "a" * 32 not in build_model_metadata(recipe, ["m"])["m"]["tags"]


# ---------------------------------------------------------------------------
# Degradation — advisory data must never break discovery
# ---------------------------------------------------------------------------


def test_absent_recipe_yields_no_metadata():
    """A job predating recipe-state persistence has no model card, which the
    Go struct treats as an ordinary omission."""
    assert build_model_metadata(None, ["m"]) == {}


def test_no_served_models_yields_no_metadata():
    assert build_model_metadata(_recipe(), []) == {}


def test_a_broken_recipe_degrades_to_omission_rather_than_raising():
    """The endpoint stays publishable: metadata cannot make a valid inference
    endpoint ineligible."""
    broken = mock.Mock()
    type(broken).metadata = mock.PropertyMock(side_effect=RuntimeError("boom"))
    assert build_model_metadata(broken, ["m"]) == {}


def test_deployment_catalog_metadata_uses_overridden_context_and_preserves_unknown_size():
    from sparkrun.plugins.sparkroute.recipe_config import catalog_sparkroute

    details = {"model": "coding", "runtime": "vllm", "defaults": {"max_model_len": 65536}, "metadata": {"parameters_b": 8}}
    fields = catalog_sparkroute(details, {"max_model_len": "8192"})["model_metadata"]
    assert fields == {"size_b": 8, "context": 8192, "input_price": 0, "output_price": 0, "tags": ["local", "vllm"]}
    assert "context" not in catalog_sparkroute(details, {"max_model_len": "auto"})["model_metadata"]
    assert "size_b" not in catalog_sparkroute({"model": "unknown"})["model_metadata"]


def test_binding_metadata_uses_effective_overrides():
    assert build_model_metadata(_recipe(), ["coding"], overrides={"max_model_len": "4096"})["coding"]["context"] == 4096


def test_generated_deployment_metadata_reduces_multiple_jobs():
    from sparkrun.plugins.sparkroute.metadata import reduce_model_metadata
    from sparkrun.plugins.sparkroute.projection import build_sparkrun_set

    fields = reduce_model_metadata(
        {"size_b": 8, "context": 65536, "input_price": 0, "tags": ["vllm"]},
        {"size_b": 70, "context": 8192, "input_price": 0, "tags": ["local"]},
    )
    result = build_sparkrun_set([], discovered_models=["coding"], discovered_apis={"coding": {"model_metadata": fields}})
    assert result["deployments"][0]["model_metadata"] == {"size_b": 70, "context": 8192, "input_price": 0, "tags": ["local", "vllm"]}
