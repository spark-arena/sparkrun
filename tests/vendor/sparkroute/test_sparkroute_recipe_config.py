# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Recipe defaults project to API-specific profiles without new workloads."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.plugins.sparkroute.recipe_config import parse_sparkroute, SparkrouteRecipeError
from sparkrun.plugins.sparkroute.projection import ProjectedBinding, ProjectionError, resolve_bindings, dedupe_bindings, build_sparkrun_set
from sparkrun.plugins.sparkroute.engine import SparkrouteEngine
from sparkrun.plugins.sparkroute.operations import _catalog
from sparkrun.plugins.sparkroute.protocol import Request, ProtocolError
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

PROFILES = {
    "low": {
        "chat_completions": {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.2},
        "responses": {"reasoning": {"effort": "low"}},
    }
}
SETTINGS = {"capabilities": ["vision"], "request_profiles": PROFILES}


@pytest.fixture(autouse=True)
def recipe_item_registered(monkeypatch):
    from sparkrun.core import recipe_items
    from sparkrun.plugins.sparkroute import register

    monkeypatch.setattr(recipe_items, "_RECIPE_ITEMS", dict(recipe_items._RECIPE_ITEMS))
    register(None)


def binding(revision="one", profiles=None, model="coding"):
    return ProjectedBinding(
        recipe="@test/" + revision,
        recipe_revision=revision,
        model=model,
        virtual_model=model,
        cluster_candidates=["lab"],
        request_profiles=profiles or {},
    )


def test_recipe_defaults_project_and_do_not_change_deployment_or_binding_identity():
    recipe = Recipe({"model": "test/model", "runtime": "vllm", "container": "nightly:latest", "defaults": {"served_model_name": "coding"}})
    with (
        patch("sparkrun.api.resolve_catalog_recipe", return_value=(recipe, {})) as resolve,
        patch("sparkrun.api._resolve.resolve_runtime", return_value=VllmDistributedRuntime()),
    ):
        before = build_sparkrun_set(resolve_bindings([{"recipe": "@test/coder"}]))
        recipe = Recipe({**recipe.to_dict(), "sparkroute": deepcopy(SETTINGS)})
        resolve.return_value = (recipe, {})
        after = build_sparkrun_set(resolve_bindings([{"recipe": "@test/coder"}]), {"code": "coding"})
    deployment = after["deployments"][0]
    assert len(after["deployments"]) == 1
    assert deployment["capabilities"] == ["responses", "vision"]
    assert deployment["endpoint_source"] == before["deployments"][0]["endpoint_source"]
    models = {model["name"]: model for model in after["virtual_models"]}
    assert models["coding:low"]["profile"] == {"parent": "coding", "selector": "low"}
    assert models["coding"]["aliases"] == ["code"]
    assert "pools" not in models["coding:low"]
    assert deployment["request_profiles"] == PROFILES
    assert "request_overrides" not in models["coding"]
    assert "required_capabilities" not in models["coding"]  # declaring vision does not require it on every request
    deployment["request_profiles"]["low"]["responses"]["reasoning"]["effort"] = "high"
    assert recipe.plugin_item("sparkroute") == SETTINGS


def test_profile_routes_only_to_recipes_that_declare_it_and_matching_definitions_merge():
    one, two, three = binding("one", PROFILES), binding("two"), binding("three", PROFILES)
    document = build_sparkrun_set([three, two, one])
    models = {model["name"]: model for model in document["virtual_models"]}
    assert len(models["coding"]["pools"][0]["targets"]) == 3
    assert [d["name"] for d in document["deployments"] if "low" in d.get("request_profiles", {})] == ["sparkrun:one", "sparkrun:three"]
    assert document == build_sparkrun_set([one, two, three])


@pytest.mark.parametrize("same_identity", [False, True])
def test_conflicting_recipe_profiles_fail_instead_of_selecting_arbitrary_parameters(same_identity):
    one = binding("one", PROFILES)
    other = binding("one" if same_identity else "two", {"low": {"chat_completions": {"temperature": 1}}})
    if same_identity:
        with pytest.raises(ProjectionError, match="conflicting request profile"):
            build_sparkrun_set(dedupe_bindings([one, other]))
    else:
        document = build_sparkrun_set(dedupe_bindings([one, other]))
        assert len(document["deployments"]) == 2
        assert document["deployments"][0]["request_profiles"] != document["deployments"][1]["request_profiles"]


@pytest.mark.parametrize("alias", [False, True])
def test_profiles_cannot_shadow_an_existing_model_or_alias(alias):
    entries = [binding(profiles=PROFILES)]
    if not alias:
        entries.append(binding("two", model="coding:low"))
    with pytest.raises(ProjectionError, match="conflicts with existing"):
        build_sparkrun_set(entries, {"coding:low": "coding"} if alias else {})


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"unknown": True},
        {"capabilities": "vision"},
        {"capabilities": ["vision", "vision"]},
        {"capabilities": ["Vision"]},
        {"capabilities": ["unknown"]},
        {"request_profiles": []},
        {"request_profiles": {":low": PROFILES["low"]}},
        {"request_profiles": {"low": {}}},
        {"request_profiles": {"low": {"unsupported": {"temperature": 0}}}},
        {"request_profiles": {"low": {"chat_completions": "{}"}}},
    ],
)
def test_malformed_recipe_configuration_is_rejected(value):
    with pytest.raises(SparkrouteRecipeError, match="sparkroute"):
        parse_sparkroute(value)


@pytest.mark.parametrize("parameter", ["model", "messages", "stream", "input", "system", "previous_response_id", "tools", "extra_body"])
def test_profiles_cannot_replace_request_structure(parameter):
    with pytest.raises(SparkrouteRecipeError, match="protected"):
        parse_sparkroute({"request_profiles": {"low": {"responses": {parameter: "bad"}}}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "numeric key"}, {"too_large": "x" * (256 * 1024)}, {"set"}, b"bytes"])
def test_yaml_values_must_be_bounded_json(value):
    with pytest.raises(SparkrouteRecipeError):
        parse_sparkroute({"request_profiles": {"low": {"messages": {"thinking": value}}}})


def test_bridge_preview_enriches_capabilities_and_reports_field_errors():
    details = {
        "name": "@test/coder",
        "capabilities": ["responses"],
        "plugin_items": {"sparkroute": SETTINGS, "other_plugin": {"internal": True}},
    }
    with patch("sparkrun.api.get_recipe_details", return_value=details):
        preview = _catalog(Request("test", "catalog_resolve", arguments={"reference": "@test/coder"}), None)
    assert preview["capabilities"] == ["responses", "vision"]
    assert preview["sparkroute"] == SETTINGS
    assert "plugin_items" not in preview
    assert details["plugin_items"]["other_plugin"] == {"internal": True}
    details["plugin_items"]["sparkroute"] = {"request_profiles": {"low": {"responses": {"model": "bad"}}}}
    with patch("sparkrun.api.get_recipe_details", return_value=details), pytest.raises(ProtocolError, match="protected"):
        _catalog(Request("test", "catalog_resolve", arguments={"reference": "@test/coder"}), None)


def test_discovery_preserves_per_job_profiles_for_a_shared_upstream(tmp_path):
    engine = SparkrouteEngine(host="127.0.0.1", port=8000, state_dir=tmp_path / "proxy")
    endpoint = SimpleNamespace(
        cluster_id="job-a",
        healthy=True,
        actual_models=["coding"],
        native_protocols=["openai"],
        capabilities=[],
        plugin_items={"sparkroute": SETTINGS},
    )
    plain = SimpleNamespace(**{**vars(endpoint), "cluster_id": "job-b", "plugin_items": {}})
    different = SimpleNamespace(
        **{
            **vars(endpoint),
            "cluster_id": "job-c",
            "plugin_items": {"sparkroute": {"request_profiles": {"low": {"responses": {"temperature": 1}}}}},
        }
    )
    engine._persist_discovered_apis([endpoint, plain, different])
    doc = build_sparkrun_set([], discovered_models=["coding"], discovered_apis=engine._read_discovered_apis())
    assert len(doc["deployments"]) == 4  # Includes the legacy aggregate for saved references.
    by_job = {d["discovery_job_ids"][0]: d for d in doc["deployments"] if d.get("discovery_job_ids")}
    assert len(doc["virtual_models"][0]["pools"][0]["targets"]) == 3
    assert by_job["job-a"]["request_profiles"] == PROFILES
    assert "request_profiles" not in by_job["job-b"]
    assert by_job["job-c"]["request_profiles"] != PROFILES
    assert doc["virtual_models"][1]["profile"] == {"parent": "coding", "selector": "low"}
    before = {job: d["name"] for job, d in by_job.items()}
    endpoint.plugin_items = {"sparkroute": {"request_profiles": {"high": PROFILES["low"]}}}
    engine._persist_discovered_apis([different, plain, endpoint])
    updated = build_sparkrun_set([], discovered_models=["coding"], discovered_apis=engine._read_discovered_apis())
    assert {d["discovery_job_ids"][0]: d["name"] for d in updated["deployments"] if d.get("discovery_job_ids")} == before


def test_discovery_profiles_without_job_identity_are_not_exposed(tmp_path):
    engine = SparkrouteEngine(host="127.0.0.1", port=8000, state_dir=tmp_path / "proxy")
    endpoint = SimpleNamespace(healthy=True, actual_models=["coding"], plugin_items={"sparkroute": SETTINGS})
    engine._persist_discovered_apis([endpoint])
    doc = build_sparkrun_set([], discovered_models=["coding"], discovered_apis=engine._read_discovered_apis())
    assert len(doc["virtual_models"]) == 1
    assert "request_profiles" not in doc["deployments"][0]


def test_expanded_profile_parameters_keep_ingress_and_upstream_separate():
    profile = {
        "supported_operations": ["responses"],
        "upstream_overrides": {"chat_completions": {"chat_template_kwargs": {"enable_thinking": False}}},
    }
    assert parse_sparkroute({"request_profiles": {"low": profile}})["request_profiles"]["low"] == profile
    with pytest.raises(SparkrouteRecipeError):
        parse_sparkroute({"request_profiles": {"low": {"upstream_overrides": {"chat_completions": {"temperature": 0}}}}})


def test_plugin_owns_recipe_schema_and_saved_state_without_core_attributes():
    from sparkrun.core.recipe import _KNOWN_KEYS
    from sparkrun.core.recipe_items import unregister_recipe_item
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    assert "sparkroute" not in _KNOWN_KEYS
    base = {"model": "test/model", "runtime": "vllm", "container": "test:latest"}
    recipe = Recipe({**base, "sparkroute": deepcopy(SETTINGS)})
    assert not hasattr(recipe, "sparkroute")
    assert "sparkroute" not in recipe.runtime_config
    assert recipe.plugin_item("sparkroute") == SETTINGS
    assert recipe.to_dict()["sparkroute"] == SETTINGS
    assert not any(issue.startswith("sparkroute.") for issue in recipe.validate())
    fingerprint = derive_recipe_fingerprint(Recipe(base))
    assert derive_recipe_fingerprint(recipe) == fingerprint

    state = recipe.__getstate__()
    assert "sparkroute" not in state
    assert state["plugin_items"]["sparkroute"] == SETTINGS
    unregister_recipe_item("sparkroute", owner="sparkrun.plugins.sparkroute")
    restored = Recipe._deserialize_yaml(Recipe._deserialize(state)._serialize_yaml())
    assert restored.to_dict()["sparkroute"] == SETTINGS
    assert derive_recipe_fingerprint(restored) == fingerprint


@pytest.mark.parametrize("value", [None, [], "vision", True, {"capabilities": ["typo"]}])
def test_plugin_parse_rejects_invalid_recipe_settings(value):
    with pytest.raises(RecipeError, match="sparkrun.plugins.sparkroute"):
        Recipe({"model": "test/model", "runtime": "vllm", "sparkroute": value})


def test_recipe_validation_uses_plugin_handler():
    recipe = Recipe({"model": "test/model", "runtime": "vllm", "sparkroute": deepcopy(SETTINGS)})
    recipe.plugin_item("sparkroute")["capabilities"] = ["typo"]
    assert any(issue.startswith("sparkroute.capabilities contains unknown") for issue in recipe.validate())


def test_absent_annotations_do_not_pollute_export():
    assert "sparkroute" not in Recipe({"model": "test/model", "runtime": "vllm"}).to_dict()


def test_catalog_plugin_parse_failures_retain_field_diagnostics(tmp_path):
    from sparkrun import api

    sctx = api.default_sctx()
    path = sctx.config.config_path.parent / "recipes" / "invalid-profile.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "model: test/model\nruntime: vllm\nsparkroute:\n  request_profiles:\n    low:\n      responses:\n        model: forbidden\n"
    )
    with pytest.raises(ProtocolError, match="sparkroute.request_profiles.low.responses.model") as error:
        _catalog(Request("test", "catalog_resolve", arguments={"reference": str(path)}), sctx)
    assert error.value.code == "catalog_invalid"


def test_shared_recipe_profile_contract():
    import json
    from pathlib import Path

    fixtures = json.loads((Path(__file__).parent / "fixtures/request_profiles.json").read_text())
    for fixture in fixtures:
        if fixture["valid"]:
            parse_sparkroute({"request_profiles": fixture["profiles"]})
        else:
            with pytest.raises(SparkrouteRecipeError):
                parse_sparkroute({"request_profiles": fixture["profiles"]})


def test_discovery_keeps_aggregate_identity_and_deduplicates_job_observations(tmp_path):
    from sparkrun.plugins.sparkroute.projection import discovered_deployment_name

    engine = SparkrouteEngine(host="127.0.0.1", port=8000, state_dir=tmp_path / "proxy")
    endpoint = SimpleNamespace(
        cluster_id="job-a", healthy=True, actual_models=["coding"], native_protocols=["openai"], capabilities=[], plugin_items={}
    )
    engine._persist_discovered_apis([endpoint, endpoint])
    plain = build_sparkrun_set([], discovered_models=["coding"], discovered_apis=engine._read_discovered_apis())
    assert [d["name"] for d in plain["deployments"]] == [discovered_deployment_name("coding")]
    endpoint.plugin_items = {"sparkroute": SETTINGS}
    engine._persist_discovered_apis([endpoint, endpoint])
    split = build_sparkrun_set([], discovered_models=["coding"], discovered_apis=engine._read_discovered_apis())
    assert len(split["deployments"]) == 2
    assert discovered_deployment_name("coding") in {d["name"] for d in split["deployments"]}
    endpoint.plugin_items = {}
    engine._persist_discovered_apis([endpoint])
    removed = build_sparkrun_set([], discovered_models=["coding"], discovered_apis=engine._read_discovered_apis())
    assert [d["name"] for d in removed["deployments"]] == [d["name"] for d in split["deployments"]]
    assert len(removed["virtual_models"]) == 1
