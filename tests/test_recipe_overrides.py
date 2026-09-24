"""Recipe ``overrides:`` — parsing, predicates, application, identity, and the plan hook."""

from __future__ import annotations

import pytest
import yaml

from sparkrun.core.hardware import AcceleratorSpec, HostHardware, default_dgx_spark_hardware
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.core.recipe_overrides import (
    OverrideConflictError,
    OverrideContext,
    apply_recipe_override_layers,
    build_override_context,
    evaluate_override,
    parse_overrides,
)
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint, generate_intent_id

_BASE = {
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "vllm-distributed",
    "container": "vllm/vllm-openai:latest",
    "defaults": {"port": 8000, "tensor_parallel": 1, "max_num_seqs": 8, "gpu_memory_utilization": 0.9},
    "env": {"BASE": "1"},
}


def _recipe(overrides, **extra) -> Recipe:
    return Recipe.from_dict({**_BASE, **extra, "overrides": overrides})


def _gb10() -> HostHardware:
    return default_dgx_spark_hardware()


def _rtx() -> HostHardware:
    return HostHardware(
        accelerators=[AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000-blackwell-workstation-edition", memory_gb=96.0)],
        source="detected",
    )


def _ctx(*, hosts=None, config=None, **shape) -> OverrideContext:
    base_shape = {"tp": 1, "pp": 1, "dp": 1, "ep": 1, "cp": 1, "nodes": 1, "gpus_per_node": 1}
    base_shape.update(shape)
    return OverrideContext(shape=base_shape, runtime="vllm-distributed", runtime_family="vllm", config=config or {}, hosts=hosts or {})


# --- parsing ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,match",
    [
        ({"when": {"tp": 1}}, "must be a list"),
        ([{"defaults": {"x": 1}}], "non-empty `when:`"),
        ([{"when": {}, "defaults": {"x": 1}}], "non-empty `when:`"),
        ([{"when": {"tp": 1}}], "sets nothing"),
        ([{"when": {"tp": 1}, "mods": ["x"]}], "cannot be overridden yet"),
        ([{"when": {"tp": 1}, "command": "x"}], "cannot be overridden yet"),
        ([{"when": {"tp": 1}, "defaultz": {"x": 1}}], "unknown key"),
        ([{"when": {"tp": 1}, "defaults": {"port": 9000}}], "identify the workload"),
        ([{"when": {"tp": 1}, "defaults": {"tensor_parallel": 2}}], "identify the workload"),
        ([{"when": {"tp": 1}, "defaults": {"served_model_name": "x"}}], "identify the workload"),
        ([{"when": {"tp": 1}, "container": ""}], "non-empty image"),
        ([{"when": {"config": "x"}, "defaults": {"a": 1}}], "when.config"),
    ],
)
def test_structural_errors_fail_at_load(raw, match):
    with pytest.raises(RecipeError, match=match):
        parse_overrides(raw)


def test_parse_keeps_unknown_selectors_for_evaluation():
    [override] = parse_overrides([{"when": {"future_thing": 1}, "defaults": {"a": 1}}])
    assert override.when == {"future_thing": 1}


def test_recipe_rejects_a_broken_block_at_load():
    with pytest.raises(RecipeError):
        _recipe([{"when": {"tp": 1}, "defaults": {"port": 1}}])


# --- predicates -----------------------------------------------------------------------------


def _one(when):
    return parse_overrides([{"when": when, "defaults": {"a": 1}}])[0]


@pytest.mark.parametrize(
    "when,shape,matched",
    [
        ({"tp": 2}, {"tp": 2}, True),
        ({"tp": 2}, {"tp": 1}, False),
        ({"tp": [1, 2]}, {"tp": 2}, True),  # list = any-of
        ({"nodes": {"gt": 1}}, {"nodes": 2}, True),
        ({"nodes": {"gt": 1}}, {"nodes": 1}, False),
        ({"nodes": {"gte": 2, "lte": 4}}, {"nodes": 3}, True),  # operators AND
        ({"nodes": {"not_in": [1]}}, {"nodes": 1}, False),
        ({"tp": 2, "nodes": 1}, {"tp": 2, "nodes": 2}, False),  # keys AND
        ({"gpus_per_node": 2, "nodes": 1}, {"gpus_per_node": 2, "nodes": 1}, True),
    ],
)
def test_shape_predicates(when, shape, matched):
    assert evaluate_override(_one(when), _ctx(**shape)).matched is matched


def test_runtime_predicates():
    assert evaluate_override(_one({"runtime_family": "vllm"}), _ctx()).matched
    assert not evaluate_override(_one({"runtime": "sglang"}), _ctx()).matched


def test_config_predicates_read_the_given_config():
    ctx = _ctx(config={"speculator": "none", "max_num_seqs": 8})
    assert evaluate_override(_one({"config": {"speculator": "none"}}), ctx).matched
    assert not evaluate_override(_one({"config": {"speculator": "dflash"}}), ctx).matched
    assert evaluate_override(_one({"config": {"max_num_seqs": {"gte": 8}}}), ctx).matched
    assert not evaluate_override(_one({"config": {"absent": "x"}}), ctx).matched


@pytest.mark.parametrize(
    "when",
    [{"future_selector": 1}, {"tp": {"between": [1, 2]}}, {"arch": {"like": "sm_*"}}, {"config": {"k": {"regex": "."}}}],
)
def test_unknown_selectors_and_operators_never_match(when):
    decision = evaluate_override(_one(when), _ctx(hosts={"h1": _gb10()}, config={"k": "v"}))
    assert not decision.matched
    assert decision.unknown


@pytest.mark.parametrize(
    "when,matched",
    [
        ({"arch": "sm_121"}, True),
        ({"arch": "sm_121a"}, True),  # lil / CuTe spelling
        ({"arch": "12.1"}, True),
        ({"arch": "sm_120"}, False),
        ({"platform": "dgx-spark"}, True),
        ({"accelerator": "gb10"}, True),
        ({"capability": "unified-memory"}, True),
        ({"capability": {"ne": "unified-memory"}}, False),
        ({"memory_gb": {"gte": 100}}, True),
        ({"arch": "sm_121", "platform": "nvidia-generic"}, False),
    ],
)
def test_hardware_predicates_on_gb10(when, matched):
    assert evaluate_override(_one(when), _ctx(hosts={"h1": _gb10(), "h2": _gb10()})).matched is matched


def test_hardware_predicates_on_rtx_pro():
    ctx = _ctx(hosts={"w1": _rtx()})
    assert evaluate_override(_one({"arch": "sm_120"}), ctx).matched
    assert evaluate_override(_one({"platform": "nvidia-generic"}), ctx).matched
    assert not evaluate_override(_one({"capability": "unified-memory"}), ctx).matched


def test_hardware_split_across_hosts_is_refused():
    with pytest.raises(OverrideConflictError, match="h2") as excinfo:
        evaluate_override(_one({"arch": "sm_121"}), _ctx(hosts={"h1": _gb10(), "h2": _rtx()}))
    assert "--hosts" in str(excinfo.value)


def test_hardware_predicate_without_hosts_does_not_match():
    decision = evaluate_override(_one({"arch": "sm_121"}), _ctx())
    assert not decision.matched and "no host hardware" in decision.reason


def test_hardware_selectors_match_jointly_per_accelerator():
    """A host with a GB10 and an unrelated card must not match `gb10 AND sm_120`."""
    mixed = HostHardware(
        accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10"), AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000")],
        source="detected",
    )
    assert not evaluate_override(_one({"accelerator": "gb10", "arch": "sm_120"}), _ctx(hosts={"h": mixed})).matched
    assert evaluate_override(_one({"accelerator": "gb10", "arch": "sm_121"}), _ctx(hosts={"h": mixed})).matched


# --- application ------------------------------------------------------------------------------


def test_matched_layers_apply_in_order_later_wins():
    recipe = _recipe(
        [
            {"when": {"nodes": 1}, "defaults": {"max_num_seqs": 1, "max_model_len": 4096}, "env": {"X": "a"}},
            {"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}},
            {"when": {"tp": 2}, "defaults": {"max_num_seqs": 99}},
        ]
    )
    resolution = apply_recipe_override_layers(recipe, _ctx())
    assert resolution.matched == (0, 1)
    assert recipe.defaults["max_num_seqs"] == 2
    assert recipe.defaults["max_model_len"] == 4096
    assert recipe.env == {"BASE": "1", "X": "a"}
    assert recipe.declared_defaults["max_num_seqs"] == 8


def test_cli_option_still_beats_an_override():
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}])
    apply_recipe_override_layers(recipe, _ctx())
    assert recipe.build_config_chain({"max_num_seqs": 64}).get("max_num_seqs") == 64
    assert recipe.build_config_chain().get("max_num_seqs") == 2


def test_cli_env_and_image_are_never_clobbered():
    from sparkrun.core.resolve import apply_env_overrides, apply_recipe_overrides

    recipe = _recipe([{"when": {"tp": 1}, "env": {"X": "override", "Y": "override"}, "container": "img:override"}])
    apply_env_overrides(recipe, ["X=cli"])
    apply_recipe_overrides(("env.Y=cli2",), image="img:cli", recipe=recipe)
    resolution = apply_recipe_override_layers(recipe, _ctx())
    assert recipe.env["X"] == "cli" and recipe.env["Y"] == "cli2"
    assert recipe.container == "img:cli"
    assert resolution.skipped_env == ("X", "Y") and resolution.skipped_container


def test_container_layer_applies():
    recipe = _recipe([{"when": {"tp": 1}, "container": "img:tuned"}])
    apply_recipe_override_layers(recipe, _ctx())
    assert recipe.container == "img:tuned"
    assert recipe.declared_container == "vllm/vllm-openai:latest"


def test_reapplying_is_idempotent_and_follows_the_new_context():
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}, {"when": {"tp": 2}, "env": {"TP2": "1"}}])
    apply_recipe_override_layers(recipe, _ctx(tp=1))
    apply_recipe_override_layers(recipe, _ctx(tp=1))
    assert recipe.defaults["max_num_seqs"] == 2
    apply_recipe_override_layers(recipe, _ctx(tp=2))
    assert recipe.defaults["max_num_seqs"] == 8  # tp1 layer no longer applies
    assert recipe.env == {"BASE": "1", "TP2": "1"}


def test_describe_explains_each_entry():
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"a": 1}}, {"when": {"tp": 4}, "defaults": {"b": 1}}])
    lines = apply_recipe_override_layers(recipe, _ctx()).describe()
    assert lines[0].startswith("overrides[0]: matched")
    assert lines[1].startswith("overrides[1]: skipped") and "tp=1" in lines[1]


# --- identity ------------------------------------------------------------------------------------


def test_applying_overrides_moves_neither_intent_nor_fingerprint():
    """stop / logs / --ensure recompute both without applying overrides."""
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}, "env": {"X": "1"}, "container": "img:t"}])
    fresh = _recipe(recipe.to_dict()["overrides"])
    apply_recipe_override_layers(recipe, _ctx())
    assert generate_intent_id(recipe) == generate_intent_id(fresh)
    assert derive_recipe_fingerprint(recipe) == derive_recipe_fingerprint(fresh)


def test_recipes_without_overrides_hash_as_before():
    plain = Recipe.from_dict(dict(_BASE))
    empty = Recipe.from_dict({**_BASE, "overrides": []})
    assert derive_recipe_fingerprint(plain) == derive_recipe_fingerprint(empty)
    assert generate_intent_id(plain) == generate_intent_id(empty)


def test_declared_block_is_part_of_the_fingerprint():
    a = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}])
    b = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 3}}])
    assert derive_recipe_fingerprint(a) != derive_recipe_fingerprint(b)
    assert generate_intent_id(a) == generate_intent_id(b)  # serve args stay out of the intent


def test_declared_override_images_are_part_of_the_intent():
    plain = _recipe([{"when": {"tp": 1}, "defaults": {"a": 1}}])
    imaged = _recipe([{"when": {"arch": "sm_120"}, "container": "img:rtx"}])
    assert generate_intent_id(plain) != generate_intent_id(imaged)


def test_export_keeps_overrides_conditional():
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}])
    apply_recipe_override_layers(recipe, _ctx())
    exported = yaml.safe_load(recipe.export())
    assert exported["defaults"]["max_num_seqs"] == 8
    assert exported["overrides"] == [{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}]


def test_state_round_trip_keeps_declared_snapshot():
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}])
    apply_recipe_override_layers(recipe, _ctx())
    restored = Recipe._deserialize_yaml(recipe._serialize_yaml())
    assert restored.defaults["max_num_seqs"] == 2
    assert restored.declared_defaults["max_num_seqs"] == 8
    assert derive_recipe_fingerprint(restored) == derive_recipe_fingerprint(recipe)


# --- context -----------------------------------------------------------------------------------


def test_context_derives_nodes_from_parallelism_and_gpus_per_host():
    from sparkrun.core.cluster_manager import ClusterDefinition

    recipe = Recipe.from_dict({**_BASE, "defaults": {**_BASE["defaults"], "tensor_parallel": 4}})
    two_gpu = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000", count=2)], source="detected")
    cluster = ClusterDefinition(name="c", hosts=["w1", "w2"])
    ctx = build_override_context(recipe, cluster=cluster, hosts=["w1", "w2"], host_hardware={"w1": two_gpu, "w2": two_gpu})
    assert ctx.shape["tp"] == 4 and ctx.shape["gpus_per_node"] == 2 and ctx.shape["nodes"] == 2

    solo = build_override_context(recipe, cluster=cluster, hosts=["w1"], host_hardware={"w1": two_gpu}, solo=True)
    assert solo.shape["nodes"] == 1


def test_context_reads_cli_overrides_and_declared_defaults():
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}])
    apply_recipe_override_layers(recipe, _ctx())
    ctx = build_override_context(recipe, {"tensor_parallel": 2})
    assert ctx.shape["tp"] == 2
    assert ctx.config.get("max_num_seqs") == 8  # declared, not the applied override


def test_context_falls_back_to_cluster_inventory():
    from sparkrun.core.cluster_manager import ClusterDefinition

    recipe = Recipe.from_dict(dict(_BASE))
    ctx = build_override_context(recipe, cluster=ClusterDefinition(name="c", hosts=["s1"]), hosts=["s1"], host_hardware={})
    assert set(ctx.hosts) == {"s1"}


# --- adjacent seams ------------------------------------------------------------------------------


def test_selector_only_config_key_is_not_reported_unmapped():
    from sparkrun.core.launcher import report_unmapped_config_keys
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    recipe = _recipe(
        [{"when": {"config": {"speculator": "none"}}, "defaults": {"max_num_seqs": 32}}],
        defaults={**_BASE["defaults"], "speculator": "mtp"},
    )
    messages = report_unmapped_config_keys(recipe, VllmDistributedRuntime(), None, log=False)
    assert not any("speculator" in m for m in messages)


def test_unknown_selector_validation_warning():
    from sparkrun.core.validation import WARNING, check_override_selectors

    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"a": 1}}, {"when": {"gpu_family": "x", "nodes": {"approx": 2}}, "env": {"Z": "1"}}])
    [issue] = check_override_selectors(recipe)
    assert (issue.severity, issue.code) == (WARNING, "override-unknown-selector")
    assert "overrides[1]" in issue.summary and "'gpu_family'" in issue.summary and "'approx'" in issue.summary


# --- api.plan -------------------------------------------------------------------------------------


def _plan(tmp_path, overrides, *, hosts=("s1", "s2"), recipe_extra=None, **option_kw):
    import sparkrun.api as api
    from sparkrun.core.cluster_manager import ClusterDefinition

    data = {**_BASE, "overrides": overrides, **(recipe_extra or {})}
    path = tmp_path / "ov.yaml"
    path.write_text(yaml.safe_dump(data))
    recipe = Recipe.load(str(path), resolve=False)
    options = api.RunOptions(
        recipe=recipe, cluster=ClusterDefinition(name="c", hosts=list(hosts)), hosts=tuple(hosts), dry_run=True, **option_kw
    )
    return api.plan(options)


def test_plan_applies_matched_overrides_and_reports_them(tmp_path, v):
    run_plan = _plan(tmp_path, [{"when": {"platform": "dgx-spark", "nodes": 1}, "defaults": {"max_num_seqs": 3}}])
    assert run_plan.override_resolution.matched == (0,)
    assert run_plan.recipe.defaults["max_num_seqs"] == 3
    fresh = Recipe.load(str(tmp_path / "ov.yaml"))
    assert run_plan.intent_id == generate_intent_id(fresh)
    assert run_plan.recipe_fingerprint == derive_recipe_fingerprint(fresh)


def test_plan_without_overrides_has_no_resolution(tmp_path, v):
    run_plan = _plan(tmp_path, [])
    assert run_plan.override_resolution is None


def test_plan_refuses_a_hardware_split(tmp_path, v):
    from sparkrun.api import SparkrunError
    from sparkrun.core.cluster_manager import ClusterDefinition

    import sparkrun.api as api

    path = tmp_path / "ov.yaml"
    path.write_text(yaml.safe_dump({**_BASE, "overrides": [{"when": {"arch": "sm_121"}, "defaults": {"a": 1}}]}))
    recipe = Recipe.load(str(path), resolve=False)
    cluster = ClusterDefinition(name="c", hosts=["s1", "w1"], hosts_hardware={"w1": _rtx()})
    with pytest.raises(SparkrunError, match="every host to agree"):
        api.plan(api.RunOptions(recipe=recipe, cluster=cluster, hosts=("s1", "w1"), dry_run=True))


# --- review hardening ---------------------------------------------------------------------


def test_ne_matches_an_unset_config_key_but_not_unknown_hardware():
    assert evaluate_override(_one({"config": {"speculator": {"ne": "eagle"}}}), _ctx(config={})).matched
    assert evaluate_override(_one({"config": {"speculator": {"not_in": ["eagle"]}}}), _ctx(config={})).matched
    assert not evaluate_override(_one({"config": {"speculator": "eagle"}}), _ctx(config={})).matched
    unknown_arch = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="mystery")], source="detected")
    assert not evaluate_override(_one({"arch": {"ne": "sm_121"}}), _ctx(hosts={"h": unknown_arch})).matched


def test_booleans_do_not_equal_numbers():
    assert not evaluate_override(_one({"config": {"flag": 1}}), _ctx(config={"flag": True})).matched
    assert evaluate_override(_one({"config": {"flag": True}}), _ctx(config={"flag": True})).matched
    assert not evaluate_override(_one({"tp": True}), _ctx(tp=1)).matched


@pytest.mark.parametrize("when", [{"arch": {"gt": "sm_100"}}, {"capability": {"gte": 1}}, {"runtime": {"lt": "z"}}])
def test_ordering_on_names_is_reported_not_silently_false(when):
    decision = evaluate_override(_one(when), _ctx(hosts={"h": _gb10()}))
    assert not decision.matched and decision.unknown and "numeric selector" in decision.reason


def test_cli_facts_survive_serialization():
    """Resumed / discovered recipes recompute the intent; --image must still suppress override images."""
    from sparkrun.core.resolve import apply_env_overrides, apply_recipe_overrides

    recipe = _recipe([{"when": {"arch": "sm_120"}, "container": "img:rtx"}])
    apply_env_overrides(recipe, ["X=cli"])
    apply_recipe_overrides((), image="img:cli", recipe=recipe)
    restored = Recipe._deserialize_yaml(recipe._serialize_yaml())
    assert restored._cli_image and restored._cli_env_keys == {"X"}
    assert generate_intent_id(restored) == generate_intent_id(recipe)


def test_effective_export_bakes_matched_layers_in():
    recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}, "container": "img:t"}])
    apply_recipe_override_layers(recipe, _ctx())
    effective = recipe.to_dict(effective=True)
    assert effective["defaults"]["max_num_seqs"] == 2 and effective["container"] == "img:t"
    assert "overrides" not in effective


def test_benchmark_fingerprints_normalize_the_declared_snapshot():
    """Image equivalence and credential redaction must hold for recipes with overrides applied."""
    from sparkrun.benchmarking._specification import _configuration_fingerprint
    from sparkrun.benchmarking.metadata import benchmark_recipe_fingerprint

    def _applied(image, key):
        recipe = _recipe([{"when": {"tp": 1}, "defaults": {"max_num_seqs": 2}}], container=image)
        recipe.defaults["api_key"] = key
        apply_recipe_override_layers(recipe, _ctx())
        return recipe

    a, b = _applied("img:a", "secret-one"), _applied("img:b", "secret-two")
    assert _configuration_fingerprint(a, {}) == _configuration_fingerprint(b, {})
    assert benchmark_recipe_fingerprint(_applied("img:a", "one")) == benchmark_recipe_fingerprint(_applied("img:a", "two"))
    assert a.declared_container == "img:a"  # the caller's snapshot is untouched


def test_plan_refuses_an_override_that_changes_the_runtime(tmp_path, v):
    from sparkrun.api import SparkrunError

    with pytest.raises(SparkrunError, match="change the resolved runtime"):
        _plan(
            tmp_path,
            [{"when": {"nodes": 1}, "defaults": {"distributed_executor_backend": "ray"}}],
            recipe_extra={"runtime": "vllm"},
        )
