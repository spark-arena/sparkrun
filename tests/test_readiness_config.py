"""Readiness is a recipe/global/default chain, not an engine flag."""

import copy
import math
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.launcher import ReadinessWatcher, wait_for_serve_ready
from sparkrun.core.readiness import ReadinessSettings, resolve_readiness_settings
from sparkrun.core.recipe import Recipe, RecipeError


def settings(global_layer=None):
    config = SparkrunConfig.__new__(SparkrunConfig)
    config._data = {} if global_layer is None else {"readiness": global_layer}
    return config


def recipe(layer=None):
    data = {"model": "test/model", "runtime": "vllm", "container": "test-image"}
    if layer is not None:
        data["readiness"] = layer
    return Recipe(data)


def test_three_level_resolution_partial_override_and_no_mutation():
    global_layer = {"inference": True, "port_timeout_s": 2400, "inference_prompt": "global"}
    recipe_layer = {"inference": False, "inference_timeout_s": 35}
    config, model = settings(global_layer), recipe(recipe_layer)
    resolved = resolve_readiness_settings(config=config, recipe=model)
    assert resolved == ReadinessSettings(inference=False, port_timeout_s=2400, inference_prompt="global", inference_timeout_s=35)
    assert config.get("readiness") == global_layer
    assert model.readiness == recipe_layer
    assert resolve_readiness_settings(config=config, recipe=recipe()).inference is True
    assert resolve_readiness_settings() == ReadinessSettings()


def test_recipe_can_enable_inference_over_a_global_false_and_override_all_fields():
    layer = {"inference": True, "port_timeout_s": 10, "health_timeout_s": 20, "inference_timeout_s": 30, "inference_prompt": "local"}
    resolved = resolve_readiness_settings(config=settings({"inference": False}), recipe=recipe(layer))
    assert resolved == ReadinessSettings(**layer)


def test_unbounded_port_health_and_invalid_global_fallbacks():
    resolved = resolve_readiness_settings(
        config=settings({"port_timeout_s": "soon", "inference_timeout_s": 0, "inference_prompt": ""}),
        recipe=recipe({"port_timeout_s": 0, "health_timeout_s": -1}),
    )
    assert math.isinf(resolved.port_timeout_s) and math.isinf(resolved.health_timeout_s)
    assert resolved.inference_timeout_s == 1200 and resolved.inference_prompt == ReadinessSettings().inference_prompt
    assert settings({"port_timeout_s": "soon"}).readiness_port_timeout_s == 1800


@pytest.mark.parametrize(
    "layer",
    [
        None,
        False,
        [],
        "off",
        {"inference": "false"},
        {"inference": None},
        {"inference": 0},
        {"inference_prompt": ""},
        {"inference_timeout_s": 0},
        {"inference_timeout_s": float("inf")},
        {"port_timeout_s": float("nan")},
        {"port_timeout_s": True},
        {"health_timeout_s": "soon"},
        {"typo": 1},
    ],
)
def test_invalid_recipe_policy_is_rejected_before_launch(layer):
    with pytest.raises(RecipeError, match="readiness"):
        Recipe({"model": "test/model", "runtime": "vllm", "readiness": layer})


def test_recipe_export_pickle_deepcopy_and_old_state_preserve_only_explicit_layer():
    model = recipe({"inference": False, "health_timeout_s": 0})
    assert "readiness" not in model.runtime_config
    exported = model.to_dict()
    assert exported["readiness"] == {"inference": False, "health_timeout_s": 0}
    assert "inference_timeout_s" not in exported["readiness"]
    for restored in (
        Recipe(exported),
        pickle.loads(pickle.dumps(model)),
        copy.deepcopy(model),
        Recipe._deserialize_yaml(model._serialize_yaml()),
    ):
        assert restored.readiness == model.readiness
        assert "readiness" not in restored.runtime_config
    assert "readiness" not in recipe().to_dict()
    state = recipe().__getstate__()
    state.pop("readiness")
    assert Recipe._deserialize(state).readiness == {}


def test_environment_and_runtime_defaults_do_not_enter_readiness_chain(monkeypatch):
    monkeypatch.setenv("inference", "false")
    model = recipe()
    model.defaults["inference"] = False
    assert resolve_readiness_settings(config=settings(), recipe=model).inference is True


def test_watcher_does_not_replace_effective_budgets_with_infinite_waits():
    with patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=SimpleNamespace(ready=False)) as wait:
        watcher = ReadinessWatcher(SimpleNamespace())
        watcher._run()
    assert "port_timeout_s" not in wait.call_args.kwargs
    assert "health_timeout_s" not in wait.call_args.kwargs


def test_legacy_endpoint_uses_same_effective_timeouts_and_explicit_api_budget():
    result = SimpleNamespace(
        config=settings({"port_timeout_s": 321}),
        recipe=recipe({"health_timeout_s": 123}),
        runtime=SimpleNamespace(),
        runtime_info={},
        cluster_id="id",
        host_list=["localhost"],
        is_solo=True,
        serve_port=8000,
        timeline=None,
    )
    with patch("sparkrun.core.launcher.wait_for_endpoint_ready") as wait:
        wait_for_serve_ready(result)
        assert wait.call_args.kwargs["port_timeout_s"] == 321
        assert wait.call_args.kwargs["health_timeout_s"] == 123
        wait_for_serve_ready(result, health_timeout_s=12)
        assert wait.call_args.kwargs["health_timeout_s"] == 12
