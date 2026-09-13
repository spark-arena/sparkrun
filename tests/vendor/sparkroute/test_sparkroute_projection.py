# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Tests for the recipe-catalog → managed-set projection.

The contract is documented in ``docs/SPARKROUTE_BRIDGE.md`` and SparkRoute's
public ``docs/SPARKRUN_ON_DEMAND.md`` guide. Four properties carry weight, each
with a concrete failure behind it:

* **catalog, not discovery** — a deployment must exist for the gateway to be
  able to activate its workload, so an offline workload must not delete it;
* **deterministic** — array order participates in the gateway's content
  revision, so unstable ordering means a new runtime generation on every
  reconcile;
* **fingerprint dedupe** — deployment identity is the recipe fingerprint, which
  excludes registry, so the same recipe in two registries would otherwise emit
  a duplicate deployment name and be rejected;
* **binding revision excludes client-facing names** — a rename must not change
  workload identity.
"""

from __future__ import annotations

import json
from unittest import mock
from types import SimpleNamespace

import pytest

from sparkrun.plugins.sparkroute.projection import (
    EMBEDDING_CAPABILITY,
    SPARKRUN_PROVIDER,
    ProjectedBinding,
    ProjectionError,
    build_sparkrun_set,
    dedupe_bindings,
    derive_binding_revision,
    deployment_name,
    discovered_deployment_name,
)


def _binding(**changes) -> ProjectedBinding:
    kwargs = {
        "recipe": "@local/qwen",
        "recipe_revision": "abc123abc123",
        "model": "qwen3-8b",
        "virtual_model": "qwen3-8b",
        "cluster_candidates": ["spark-a"],
    }
    kwargs.update(changes)
    return ProjectedBinding(**kwargs)


# ---------------------------------------------------------------------------
# Binding revision
# ---------------------------------------------------------------------------


def test_binding_revision_is_stable_and_bounded():
    args = {
        "deployment": "sparkrun:abc123abc123",
        "recipe": "@local/qwen",
        "recipe_revision": "abc123abc123",
        "cluster_candidates": ["a", "b"],
        "overrides": {"tensor_parallel": "2"},
    }
    first = derive_binding_revision(**args)
    assert len(first) == 12
    assert all(c in "0123456789abcdef" for c in first)
    assert derive_binding_revision(**args) == first


def test_binding_revision_ignores_override_insertion_order():
    a = derive_binding_revision(deployment="d", recipe="r", recipe_revision="v", cluster_candidates=[], overrides={"x": "1", "y": "2"})
    b = derive_binding_revision(deployment="d", recipe="r", recipe_revision="v", cluster_candidates=[], overrides={"y": "2", "x": "1"})
    assert a == b


def test_binding_revision_honours_cluster_candidate_order():
    """Order encodes placement preference, so it is content, not noise."""
    a = derive_binding_revision(deployment="d", recipe="r", recipe_revision="v", cluster_candidates=["a", "b"], overrides={})
    b = derive_binding_revision(deployment="d", recipe="r", recipe_revision="v", cluster_candidates=["b", "a"], overrides={})
    assert a != b


def test_renaming_a_virtual_model_does_not_change_the_binding():
    """Several virtual models may route to one deployment, so a client-facing
    rename must not change workload identity and force re-admission."""
    document_a = build_sparkrun_set([_binding(virtual_model="friendly-name")])
    document_b = build_sparkrun_set([_binding(virtual_model="other-name")])
    assert document_a["deployments"][0]["endpoint_source"]["revision"] == document_b["deployments"][0]["endpoint_source"]["revision"]


# ---------------------------------------------------------------------------
# Document shape
# ---------------------------------------------------------------------------


def test_singleton_provider_regardless_of_cluster_count():
    """A provider is protocol config, not cluster identity: one deployment has
    one provider but a binding may list several cluster candidates."""
    document = build_sparkrun_set(
        [
            _binding(recipe_revision="aaa", virtual_model="a", cluster_candidates=["spark-a", "spark-b"]),
            _binding(recipe_revision="bbb", virtual_model="b", cluster_candidates=["spark-c"]),
        ]
    )
    assert document["providers"] == [{"name": "sparkrun", "type": "sparkrun"}]
    assert SPARKRUN_PROVIDER == "sparkrun"
    assert {d["provider"] for d in document["deployments"]} == {SPARKRUN_PROVIDER}


def test_activatable_endpoint_source_shape():
    document = build_sparkrun_set([_binding(overrides={"tensor_parallel": "2"})])
    source = document["deployments"][0]["endpoint_source"]
    assert source["type"] == "activatable"
    assert source["controller"] == "sparkrun"
    assert source["recipe"] == "@local/qwen"
    assert source["recipe_revision"] == "abc123abc123"
    assert source["cluster_candidates"] == ["spark-a"]
    assert source["cold_start"] == "wait"
    assert source["overrides"] == {"tensor_parallel": "2"}
    assert document["deployments"][0]["name"] == deployment_name("abc123abc123")


def test_two_bindings_for_one_model_share_a_virtual_model_pool():
    document = build_sparkrun_set([_binding(recipe_revision="aaa"), _binding(recipe_revision="bbb", recipe="@local/qwen-alt")])
    assert len(document["virtual_models"]) == 1
    targets = document["virtual_models"][0]["pools"][0]["targets"]
    assert [t["deployment"] for t in targets] == ["sparkrun:aaa", "sparkrun:bbb"]


def test_discovered_model_projects_a_warm_only_route():
    document = build_sparkrun_set([], discovered_models=["deepseek-ai/DeepSeek-V4-Flash-0731"])
    deployment = document["deployments"][0]
    assert deployment == {
        "name": discovered_deployment_name("deepseek-ai/DeepSeek-V4-Flash-0731"),
        "title": "sparkrun:discovered:deepseek-ai/DeepSeek-V4-Flash-0731",
        "provider": "sparkrun",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "native_protocols": ["openai"],
        "capabilities": [],
        "endpoint_source": {"type": "discovered", "controller": "sparkrun"},
        "capability_policy": {"unknown": "try"},
    }
    assert document["virtual_models"] == [
        {
            "name": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "pools": [{"priority": 0, "targets": [{"deployment": deployment["name"], "weight": 1}]}],
        }
    ]


def test_discovery_does_not_duplicate_an_activatable_model():
    document = build_sparkrun_set([_binding()], discovered_models=["qwen3-8b"])
    assert len(document["deployments"]) == 1
    assert document["deployments"][0]["endpoint_source"]["type"] == "activatable"


def test_projection_is_byte_stable_across_input_order():
    """Array order feeds the gateway's content revision — an unstable
    projection would build a new runtime generation on every reconcile."""
    a = _binding(recipe_revision="aaa", virtual_model="a")
    b = _binding(recipe_revision="bbb", virtual_model="b")
    assert json.dumps(build_sparkrun_set([a, b], {"x": "a"})) == json.dumps(build_sparkrun_set([b, a], {"x": "a"}))


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_permissive_policy_is_emitted_by_default():
    """Sparkrun cannot determine model capabilities, so undeclared optional
    features are attempted rather than refused up front."""
    deployment = build_sparkrun_set([_binding()])["deployments"][0]
    assert deployment["capability_policy"] == {"unknown": "try"}
    assert "capabilities" not in deployment


def test_strict_policy_emits_no_unknown_try():
    deployment = build_sparkrun_set([_binding()], permissive=False)["deployments"][0]
    assert "capability_policy" not in deployment


def test_declared_capabilities_and_unsupported_are_both_emitted():
    deployment = build_sparkrun_set([_binding(capabilities=["tools", "vision"], unsupported_capabilities=["audio_output"])])["deployments"][
        0
    ]
    assert deployment["capabilities"] == ["tools", "vision"]
    assert deployment["capability_policy"] == {"unknown": "try", "unsupported": ["audio_output"]}


def test_contradictory_capability_is_rejected_with_the_recipe_named():
    """The gateway rejects this too, but its error cannot name the recipe."""
    with pytest.raises(ProjectionError, match="@local/qwen"):
        build_sparkrun_set([_binding(capabilities=["tools"], unsupported_capabilities=["tools"])])


def test_embedding_capability_propagates_to_required_capabilities():
    """An embedding route must require the capability, so a binding that loses
    the declaration fails validation instead of failing requests."""
    document = build_sparkrun_set([_binding(capabilities=[EMBEDDING_CAPABILITY])])
    assert document["virtual_models"][0]["required_capabilities"] == [EMBEDDING_CAPABILITY]


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_aliases_attach_to_their_target_and_sort():
    document = build_sparkrun_set([_binding()], {"zzz": "qwen3-8b", "fast": "qwen3-8b"})
    assert document["virtual_models"][0]["aliases"] == ["fast", "zzz"]


def test_alias_without_a_bound_target_is_skipped():
    document = build_sparkrun_set([_binding()], {"fast": "not-bound"})
    assert "aliases" not in document["virtual_models"][0]


# ---------------------------------------------------------------------------
# Fingerprint dedupe
# ---------------------------------------------------------------------------


def test_same_recipe_in_two_registries_projects_one_deployment():
    """derive_recipe_fingerprint excludes registry, so a local copy of a
    community recipe shares a deployment name. Emitting both would be rejected
    as a duplicate deployment on an ordinary catalog."""
    entries = dedupe_bindings([_binding(recipe="@community/qwen"), _binding(recipe="@local/qwen")])
    assert len(entries) == 1
    # Lexically first, so the choice is stable rather than iteration-dependent.
    assert entries[0].recipe == "@community/qwen"


def test_dedupe_keeps_the_union_of_declared_capabilities():
    """Dropping a capability because a duplicate lacked it would silently
    disable a route."""
    entries = dedupe_bindings(
        [
            _binding(recipe="@local/qwen", capabilities=["tools"]),
            _binding(recipe="@community/qwen", capabilities=[EMBEDDING_CAPABILITY]),
        ]
    )
    assert entries[0].capabilities == sorted(["tools", EMBEDDING_CAPABILITY])


def test_distinct_fingerprints_are_not_collapsed():
    entries = dedupe_bindings([_binding(recipe_revision="aaa"), _binding(recipe_revision="bbb")])
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# Binding resolution from proxy.yaml
# ---------------------------------------------------------------------------


def test_recipe_capabilities_reach_the_generated_deployment():
    """Sparkrun cannot infer capabilities, so a recipe declares them and they
    must survive the projection — embeddings above all, which the gateway
    treats as fail-closed."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.plugins.sparkroute import projection

    recipe = Recipe(
        {
            "recipe_version": "2",
            "model": "BAAI/bge-m3",
            "runtime": "vllm",
            "container": "vllm/vllm-openai:latest",
            "defaults": {"port": 8000},
            "capabilities": ["single_vector_embedding"],
            "unsupported_capabilities": ["audio_output"],
        }
    )
    assert recipe.capabilities == ["single_vector_embedding"]

    with mock.patch("sparkrun.api.resolve_catalog_recipe", return_value=(recipe, {})):
        entries = projection.resolve_bindings([{"recipe": "@local/bge", "cluster": "spark-a"}])

    document = build_sparkrun_set(entries)
    deployment = document["deployments"][0]
    assert deployment["capabilities"] == ["single_vector_embedding"]
    assert deployment["capability_policy"]["unsupported"] == ["audio_output"]
    assert document["virtual_models"][0]["required_capabilities"] == [EMBEDDING_CAPABILITY]


def test_a_binding_without_a_recipe_is_rejected():
    from sparkrun.plugins.sparkroute.projection import resolve_bindings

    with pytest.raises(ProjectionError, match="no 'recipe'"):
        resolve_bindings([{"cluster": "spark-a"}])


def test_an_invalid_cold_start_is_rejected():
    from sparkrun.core.recipe import Recipe
    from sparkrun.plugins.sparkroute.projection import resolve_bindings

    recipe = Recipe({"recipe_version": "2", "model": "m", "runtime": "vllm", "container": "c", "defaults": {"port": 8000}})
    with mock.patch("sparkrun.api.resolve_catalog_recipe", return_value=(recipe, {})):
        with pytest.raises(ProjectionError, match="cold_start"):
            resolve_bindings([{"recipe": "@local/x", "cold_start": "maybe"}])


# ---------------------------------------------------------------------------
# Native protocols
# ---------------------------------------------------------------------------
#
# Protocol selects the upstream URL, headers, parser, streaming framing, error
# vocabulary and retry classification, so the gateway models it as a routing
# dimension (`Deployment.native_protocols`) rather than a capability string —
# which is what makes it structurally immune to `capability_policy.unknown:
# try` instead of immune by convention.


def test_openai_is_the_default_native_protocol():
    deployment = build_sparkrun_set([_binding()])["deployments"][0]
    assert deployment["native_protocols"] == ["openai"]


def test_multi_protocol_deployment_keeps_declared_order():
    """Order is observable when several translated fallbacks are possible and
    none is native, so it is preference rather than a set."""
    deployment = build_sparkrun_set([_binding(native_protocols=["openai", "anthropic"])])["deployments"][0]
    assert deployment["native_protocols"] == ["openai", "anthropic"]


def test_protocols_are_never_emitted_as_capability_strings():
    """The gateway rejected that spelling: `unknown: try` must never be able to
    infer protocol support."""
    document = build_sparkrun_set([_binding(native_protocols=["openai", "anthropic"])])
    rendered = json.dumps(document)
    for proposed in ("openai-chat-compatible", "anthropic-messages"):
        assert proposed not in rendered
    assert "capabilities" not in document["deployments"][0]


def test_protocol_knowledge_does_not_move_any_identity():
    """Learning that a deployment also speaks Anthropic describes the same
    workload more precisely; it must not re-admit it."""
    single = build_sparkrun_set([_binding()])["deployments"][0]
    multi = build_sparkrun_set([_binding(native_protocols=["openai", "anthropic"])])["deployments"][0]
    assert single["name"] == multi["name"]
    assert single["endpoint_source"]["revision"] == multi["endpoint_source"]["revision"]
    assert single["endpoint_source"]["recipe_revision"] == multi["endpoint_source"]["recipe_revision"]


def test_capability_policy_survives_alongside_protocols():
    deployment = build_sparkrun_set(
        [_binding(native_protocols=["openai", "anthropic"], capabilities=["tools"], unsupported_capabilities=["audio_output"])]
    )["deployments"][0]
    assert deployment["native_protocols"] == ["openai", "anthropic"]
    assert deployment["capabilities"] == ["tools"]
    assert deployment["capability_policy"] == {"unknown": "try", "unsupported": ["audio_output"]}


def test_runtime_hook_supplies_protocols_and_a_binding_may_override():
    from sparkrun.core.recipe import Recipe
    from sparkrun.plugins.sparkroute.projection import resolve_bindings

    recipe = Recipe({"recipe_version": "2", "model": "m", "runtime": "vllm", "container": "c", "defaults": {"port": 8000}})
    runtime = mock.Mock()
    runtime.native_protocols.return_value = ["openai", "anthropic"]

    with (
        mock.patch("sparkrun.api.resolve_catalog_recipe", return_value=(recipe, {})),
        mock.patch("sparkrun.api._resolve.resolve_runtime", return_value=runtime),
    ):
        from_runtime = resolve_bindings([{"recipe": "@local/x"}])
        overridden = resolve_bindings([{"recipe": "@local/x", "native_protocols": ["openai"]}])

    assert from_runtime[0].native_protocols == ["openai", "anthropic"]
    assert overridden[0].native_protocols == ["openai"]


@pytest.mark.parametrize("apis", [None, ["chat_completions"], ["messages"]])
def test_nightly_vllm_runtime_declarations_reach_existing_loaded_bindings(apis):
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
    from sparkrun.plugins.sparkroute.projection import resolve_bindings

    recipe = Recipe(
        {
            "recipe_version": "2",
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "runtime": "vllm",
            "container": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest",
            "metadata": {} if apis is None else {"native_apis": apis},
        }
    )
    # proxy load persists recipe/cluster, not a snapshot of runtime API defaults.
    with (
        mock.patch("sparkrun.api.resolve_catalog_recipe", return_value=(recipe, {})),
        mock.patch("sparkrun.api._resolve.resolve_runtime", return_value=VllmDistributedRuntime()),
    ):
        bindings = resolve_bindings([{"recipe": "@official/deepseek", "cluster": "sparks25"}])
    deployment = build_sparkrun_set(bindings)["deployments"][0]
    assert deployment["native_protocols"] == (
        ["openai", "anthropic"] if apis is None else ["openai"] if apis == ["chat_completions"] else ["anthropic"]
    )
    assert deployment.get("capabilities", []) == (["responses"] if apis is None else [])
    assert deployment["title"] == "sparkrun:sparks25:deepseek-ai/DeepSeek-V4-Flash-0731"


def test_unresolvable_runtime_degrades_to_openai_rather_than_failing():
    """Omitting a dialect costs a translation; failing the reconcile would take
    down configuration for every other binding."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.plugins.sparkroute.projection import resolve_bindings

    recipe = Recipe({"recipe_version": "2", "model": "m", "runtime": "vllm", "container": "c", "defaults": {"port": 8000}})
    with (
        mock.patch("sparkrun.api.resolve_catalog_recipe", return_value=(recipe, {})),
        mock.patch("sparkrun.api._resolve.resolve_runtime", side_effect=RuntimeError("no plugin")),
    ):
        assert resolve_bindings([{"recipe": "@local/x"}])[0].native_protocols == ["openai"]


def test_base_runtime_declares_only_openai():
    """Fail-closed: a runtime opts into a dialect, never inherits one."""
    from sparkrun.runtimes.base import RuntimePlugin

    class DefaultRuntime(RuntimePlugin):
        def generate_command(self, *args, **kwargs):
            return ""

    assert DefaultRuntime().native_protocols(SimpleNamespace(metadata={})) == ["openai"]


def test_friendly_titles_preserve_deployment_ids_and_binding_revisions():
    binding = _binding(cluster_candidates=["spark-a", "spark-b"])
    deployment = build_sparkrun_set([binding])["deployments"][0]
    assert deployment["title"] == "sparkrun:spark-a,spark-b:qwen3-8b"
    assert deployment["name"] == "sparkrun:abc123abc123"
    binding.model = "friendly/new-model"
    renamed = build_sparkrun_set([binding])["deployments"][0]
    assert renamed["title"] == "sparkrun:spark-a,spark-b:friendly/new-model"
    assert renamed["name"] == deployment["name"]
    assert renamed["endpoint_source"]["revision"] == deployment["endpoint_source"]["revision"]
    warm_a = build_sparkrun_set([], discovered_models=["model"], discovered_clusters={"model": ["spark-a"]})["deployments"][0]
    warm_b = build_sparkrun_set([], discovered_models=["model"], discovered_clusters={"model": ["spark-b"]})["deployments"][0]
    assert warm_a["name"] == warm_b["name"] == discovered_deployment_name("model")
    assert warm_a["title"] == "sparkrun:spark-a:model"
    assert warm_b["title"] == "sparkrun:spark-b:model"


def test_observed_cluster_overrides_placement_label_without_changing_binding():
    binding = _binding(cluster_candidates=["configured-candidate"])
    configured = build_sparkrun_set([binding])["deployments"][0]
    live = build_sparkrun_set([binding], binding_clusters={binding.recipe_revision: ["actual-cluster"]})["deployments"][0]
    assert live["title"] == "sparkrun:actual-cluster:qwen3-8b"
    assert live["name"] == configured["name"]
    assert live["endpoint_source"] == configured["endpoint_source"]
    unknown = build_sparkrun_set([_binding(cluster_candidates=[])])["deployments"][0]
    assert unknown["title"] == "sparkrun:unassigned:qwen3-8b"
