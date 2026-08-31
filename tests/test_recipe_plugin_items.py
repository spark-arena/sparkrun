from __future__ import annotations

from dataclasses import dataclass

import pytest

from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.core.recipe_items import unregister_recipe_item
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint
from sparkrun.plugins import register_recipe_item


@dataclass(frozen=True)
class _DemoConfig:
    enabled: bool


class _DemoHandler:
    def parse(self, value, recipe):
        if not isinstance(value, dict):
            raise ValueError("must be a mapping")
        return _DemoConfig(enabled=bool(value.get("enabled", False)))

    def validate(self, value, recipe):
        return [] if value.enabled else ["enabled must be true"]

    def export(self, value, recipe):
        return {"enabled": value.enabled}


@pytest.fixture
def demo_handler():
    handler = _DemoHandler()
    register_recipe_item("demo_plugin", handler, owner="tests.demo")
    yield handler
    unregister_recipe_item("demo_plugin", owner="tests.demo")


def _recipe(item):
    return Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "org/model",
            "model_revision": "commit",
            "runtime": "vllm-distributed",
            "demo_plugin": item,
        }
    )


def test_plugin_claimed_item_is_not_swept_into_runtime_config(demo_handler):
    recipe = _recipe({"enabled": True})

    assert recipe.plugin_item("demo_plugin") == _DemoConfig(enabled=True)
    assert "demo_plugin" not in recipe.runtime_config
    assert recipe.to_dict()["demo_plugin"] == {"enabled": True}


def test_plugin_validation_is_namespaced(demo_handler):
    recipe = _recipe({"enabled": False})
    assert "demo_plugin.enabled must be true" in recipe.validate()


def test_plugin_item_participates_in_recipe_fingerprint(demo_handler):
    enabled = _recipe({"enabled": True})
    disabled = _recipe({"enabled": False})

    assert derive_recipe_fingerprint(enabled) != derive_recipe_fingerprint(disabled)


def test_plugin_parse_failure_names_owner(demo_handler):
    with pytest.raises(RecipeError, match="tests.demo"):
        _recipe("wrong")


def test_recipe_item_ownership_conflict_is_rejected(demo_handler):
    with pytest.raises(ValueError, match="already owned"):
        register_recipe_item("demo_plugin", _DemoHandler(), owner="someone.else")


def test_recipe_item_cannot_claim_a_core_key():
    with pytest.raises(ValueError, match="conflicts with a core recipe key"):
        register_recipe_item("model", _DemoHandler(), owner="tests.demo")


def test_plugin_item_survives_recipe_state_round_trip(demo_handler):
    recipe = _recipe({"enabled": True})
    restored = Recipe._deserialize(recipe.__getstate__())
    assert restored.plugin_item("demo_plugin") == _DemoConfig(enabled=True)
    assert restored.to_dict()["demo_plugin"] == {"enabled": True}


def test_raw_plugin_item_survives_reserialization_while_plugin_is_unavailable(
    demo_handler,
):
    recipe = _recipe({"enabled": True})
    state = recipe.__getstate__()
    unregister_recipe_item("demo_plugin", owner="tests.demo")

    unavailable = Recipe._deserialize(state)
    assert unavailable.plugin_item("demo_plugin") is None
    assert unavailable.to_dict()["demo_plugin"] == {"enabled": True}
    second_state = unavailable.__getstate__()

    register_recipe_item("demo_plugin", demo_handler, owner="tests.demo")
    restored = Recipe._deserialize(second_state)
    assert restored.plugin_item("demo_plugin") == _DemoConfig(enabled=True)
