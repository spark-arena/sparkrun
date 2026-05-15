"""Tests for sparkrun.core.layout + Recipe layout round-trip (Phase 1)."""

from __future__ import annotations

import pytest

from sparkrun.core.layout import CapabilityRequirement, Placement, RecipeLayout
from sparkrun.core.recipe import Recipe


def test_capability_requirement_round_trip():
    req = CapabilityRequirement(capability="cuda")
    assert CapabilityRequirement.from_dict(req.to_dict()) == req


def test_placement_round_trip_minimal():
    p = Placement(host="spark-01", ranks=(0,))
    assert p.to_dict() == {"host": "spark-01", "ranks": [0]}
    assert Placement.from_dict(p.to_dict()) == p


def test_placement_round_trip_with_local_gpus():
    p = Placement(host="rtx-box", ranks=(2, 3), local_gpus=(0, 1))
    d = p.to_dict()
    assert d == {"host": "rtx-box", "ranks": [2, 3], "local_gpus": [0, 1]}
    assert Placement.from_dict(d) == p


def test_recipe_layout_empty_round_trip():
    """An empty layout serializes to an empty dict."""
    layout = RecipeLayout()
    assert layout.to_dict() == {}
    assert RecipeLayout.from_dict({}) == RecipeLayout()


def test_recipe_layout_full_round_trip():
    layout = RecipeLayout(
        requires=[CapabilityRequirement(capability="cuda")],
        placements=[
            Placement(host="spark-01", ranks=(0,)),
            Placement(host="rtx-box", ranks=(1, 2), local_gpus=(0, 1)),
        ],
    )
    restored = RecipeLayout.from_dict(layout.to_dict())
    assert restored == layout


# --------------------------------------------------------------------------
# Recipe integration
# --------------------------------------------------------------------------


@pytest.fixture
def recipe_with_layout_data() -> dict:
    return {
        "recipe_version": "2",
        "model": "test-model",
        "runtime": "vllm",
        "container": "img:latest",
        "layout": {
            "requires": [{"capability": "cuda"}],
            "placements": [
                {"host": "spark-01", "ranks": [0]},
                {"host": "spark-02", "ranks": [1]},
            ],
        },
    }


def test_recipe_parses_layout(recipe_with_layout_data: dict):
    """Recipe surfaces the parsed layout as a RecipeLayout object."""
    recipe = Recipe.from_dict(recipe_with_layout_data)
    assert recipe.layout is not None
    assert len(recipe.layout.requires) == 1
    assert recipe.layout.requires[0].capability == "cuda"
    assert len(recipe.layout.placements) == 2
    assert recipe.layout.placements[0].host == "spark-01"
    assert recipe.layout.placements[1].ranks == (1,)


def test_recipe_without_layout_has_none():
    """Layout defaults to None when absent."""
    recipe = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img:latest"})
    assert recipe.layout is None


def test_recipe_layout_invalid_input_is_ignored():
    """Non-dict layout values are dropped silently (permissive Phase 1 parsing)."""
    recipe = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img:latest", "layout": "not-a-dict"})
    assert recipe.layout is None


def test_recipe_layout_survives_serialize_round_trip(recipe_with_layout_data: dict):
    """layout round-trips through __getstate__/__setstate__."""
    original = Recipe.from_dict(recipe_with_layout_data)
    restored = Recipe._deserialize(original.__getstate__())
    assert restored.layout == original.layout


def test_recipe_layout_appears_in_export_dict(recipe_with_layout_data: dict):
    """to_dict() includes the layout block."""
    recipe = Recipe.from_dict(recipe_with_layout_data)
    exported = recipe.to_dict()
    assert "layout" in exported
    assert exported["layout"]["placements"][0]["host"] == "spark-01"


def test_recipe_export_omits_layout_when_none():
    """Recipes without layout don't get a stray empty layout key in export."""
    recipe = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img:latest"})
    assert "layout" not in recipe.to_dict()


def test_recipe_layout_does_not_pollute_runtime_config(recipe_with_layout_data: dict):
    """`layout` is a known top-level key — it must not leak into runtime_config."""
    recipe = Recipe.from_dict(recipe_with_layout_data)
    assert "layout" not in recipe.runtime_config
