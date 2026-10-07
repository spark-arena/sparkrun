# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core.env_templates import PreparedModelPath
from sparkrun.core.execution import ExecutionContext, resolve_recipe_execution
from sparkrun.core.resolve import apply_env_overrides
from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.plugins.coldsnap.environment import ENV_INPUTS
from sparkrun.plugins.coldsnap.request import build_request
from sparkrun.plugins.coldsnap.runtime_cache import CANONICAL_RUNTIME_CACHE_ROOT, CaptureRuntimeCacheStage
from sparkrun.plugins.coldsnap.service import ColdSnapService, PreparedRestore, RestoreActivationReceipt
from test_coldsnap_plugin import _setup, _sglang_setup


def _artifact(tmp_path, request):
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "snapshot_driver": request["snapshot_driver"],
                "launch": request["launch"],
            }
        )
    )
    return str(path)


def _probe(monkeypatch):
    probe = Mock(
        side_effect=lambda host, model, revision, cache, *_a, **_kw: PreparedModelPath(
            cache + "/hub/models--Qwen--Qwen3.5-0.8B/snapshots/" + revision, revision
        )
    )
    monkeypatch.setattr("sparkrun.plugins.coldsnap.environment.probe_model_path", probe)
    return probe


@pytest.mark.parametrize("setup", [_setup, _sglang_setup])
@pytest.mark.parametrize("driver", ["n580", "n610"])
def test_capture_and_restore_templates_use_artifact_without_host_cache(monkeypatch, tmp_path, setup, driver):
    recipe, options, plan, sctx = setup()
    recipe.env.update(
        {
            "MODEL_FILES": "{launch.model_path}",
            "COMMIT": "{launch.model_revision}",
            "CACHE": "{launch.runtime_cache_dir}/adapter",
            "RANK": "{launch.node_rank}/{launch.num_nodes}",
            "CONFIG": "{config.tensor_parallel}",
            "JSON": '{"literal": true}',
            "ESCAPED": "{{launch.node_rank}}",
        }
    )
    probe = _probe(monkeypatch)
    capture = build_request("capture", options, plan=plan, sctx=sctx, snapshot_driver=driver)
    assert probe.call_count == 2
    envs = [u["environment"] for u in capture["launch"]["units"]]
    assert [e["RANK"] for e in envs] == ["0/2", "1/2"]
    assert all(e["MODEL_FILES"].startswith("/cache/huggingface/hub/") for e in envs)
    assert all(e["COMMIT"] == "model-commit" and e["CONFIG"] == "2" for e in envs)
    assert all(e["CACHE"] == CANONICAL_RUNTIME_CACHE_ROOT + "/adapter" for e in envs)
    assert all(e["ESCAPED"] == "{launch.node_rank}" and e["JSON"] == '{"literal": true}' for e in envs)
    path = _artifact(tmp_path, capture)
    probe.side_effect = AssertionError("restore must not probe the model cache")
    plan.cluster.cache_dir = "/unavailable/new-hf-cache"
    plan = replace(
        plan,
        host_list=("new1", "new2"),
        cluster_id="new-cluster",
        placement=RankAssignment(by_rank=(RankSlot("new1", 0), RankSlot("new2", 0)), hosts_used=("new1", "new2")),
    )
    restored = build_request("restore", options, plan=plan, sctx=sctx, artifact=path, snapshot_driver=driver, weight_mode="native")
    for captured, requested in zip(capture["launch"]["units"], restored["launch"]["units"], strict=True):
        for key in (*recipe.env, ENV_INPUTS):
            assert requested["environment"][key] == captured["environment"][key]
    assert probe.call_count == 2


@pytest.mark.parametrize("change", ["config", "template", "remove", "cli", "host", "cluster"])
def test_restore_rejects_changed_declared_inputs_or_bound_placement(tmp_path, change):
    recipe, options, plan, sctx = _setup()
    recipe.env["VALUE"] = {"host": "{launch.node_host}", "cluster": "{launch.cluster_id}"}.get(change, "{config.tensor_parallel}")
    path = _artifact(tmp_path, build_request("capture", options, plan=plan, sctx=sctx))
    if change == "config":
        recipe.defaults["tensor_parallel"] = 4
    elif change == "template":
        recipe.env["VALUE"] += "changed"
    elif change == "remove":
        recipe.env.clear()
    elif change == "cli":
        apply_env_overrides(recipe, ["VALUE=2"])
    elif change == "host":
        plan = replace(
            plan,
            host_list=("new1", "new2"),
            placement=RankAssignment(by_rank=(RankSlot("new1", 0), RankSlot("new2", 0)), hosts_used=("new1", "new2")),
        )
    else:
        plan = replace(plan, cluster_id="different")
    with pytest.raises(ValueError, match="recapture"):
        build_request("restore", options, plan=plan, sctx=sctx, artifact=path)


def test_template_restore_requires_captured_provenance(tmp_path):
    recipe, options, plan, sctx = _setup()
    request = build_request("capture", options, plan=plan, sctx=sctx)
    path = _artifact(tmp_path, request)
    recipe.env["RANK"] = "{launch.node_rank}"
    with pytest.raises(ValueError, match="provenance.*recapture"):
        build_request("restore", options, plan=plan, sctx=sctx, artifact=path)


def test_explicit_literal_override_is_not_interpreted():
    recipe, options, plan, sctx = _setup()
    recipe.env.update({"RANK": "{launch.node_rank}", "VALUE": "{launch.model_path}"})
    apply_env_overrides(recipe, ["VALUE={config.not_a_field}"])
    request = build_request("capture", options, plan=plan, sctx=sctx)
    assert all(u["environment"]["VALUE"] == "{config.not_a_field}" for u in request["launch"]["units"])


@pytest.mark.parametrize("policy", [{"seed": False}, {"paths": ["/root/.cache/torch"]}])
def test_cache_template_requires_capsule_cache_policy(policy):
    recipe, options, plan, sctx = _setup()
    recipe.plugin_items["coldsnap"] = replace(
        recipe.plugin_item("coldsnap"), cache_seed=policy.get("seed", True), cache_paths=tuple(policy.get("paths", ()))
    )
    recipe.env["CACHE"] = "{launch.runtime_cache_dir}"
    with pytest.raises(ValueError, match="requires ColdSnap cache seeding"):
        build_request("capture", options, plan=plan, sctx=sctx)


@pytest.mark.parametrize("field,target", [("runtime_cache_dir", CANONICAL_RUNTIME_CACHE_ROOT), ("model_path", "/cache/huggingface")])
def test_capture_refuses_shadowed_asset_mounts(monkeypatch, field, target):
    recipe, options, plan, sctx = _setup()
    recipe.env["PATH_VALUE"] = "{launch." + field + "}"
    recipe.executor_config["volumes"] = ["/other:" + target]
    _probe(monkeypatch)
    with pytest.raises(ValueError, match="shadowed|not visible"):
        build_request("capture", options, plan=plan, sctx=sctx)


@pytest.mark.parametrize("hook", ["pre_exec", "post_exec", "post_commands", "mods"])
def test_unsupported_hooks_fail_before_hardware_or_replacement(monkeypatch, hook):
    recipe, options, plan, sctx = _setup()
    setattr(recipe, hook, ["required-prepare"])
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", lambda *_a, **_kw: pytest.fail("hardware probe"))
    with pytest.raises(ValueError, match="does not support recipe hooks or mods"):
        ColdSnapService().execute_explicit("capture", options, plan=plan, sctx=sctx)
    with pytest.raises(ValueError, match="does not support recipe hooks or mods"):
        resolve_recipe_execution(ExecutionContext(options=options, plan=plan, sctx=sctx))


def test_capture_preview_leaves_asset_paths_explicitly_unresolved(monkeypatch):
    recipe, options, plan, sctx = _setup()
    recipe.env.update({"MODEL": "{launch.model_path}", "CACHE": "{launch.runtime_cache_dir}"})
    monkeypatch.setattr("sparkrun.plugins.coldsnap.environment.probe_model_path", lambda *_a, **_kw: pytest.fail("preview probe"))
    request, _, _ = ColdSnapService().execute_explicit("capture", options, plan=plan, sctx=sctx, render_only=True)
    assert request["launch"]["units"][0]["environment"]["MODEL"] == "<unresolved:launch.model_path>"


def test_capture_resolves_after_preparation_and_before_replacement(monkeypatch):
    recipe, options, plan, sctx = _setup()
    recipe.env["MODEL"] = "{launch.model_path}"
    events = []
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.prepare_capture_images",
        lambda *_a, **_kw: events.append("images-models") or SimpleNamespace(content_images_by_node=None, comm_env=None),
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.environment.probe_model_path",
        lambda *_a, **_kw: events.append("resolve") or PreparedModelPath("/cache/hf/model", "model-commit"),
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.stage_capture_runtime_cache",
        lambda request, **_kw: events.append("cache") or CaptureRuntimeCacheStage(request),
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.stage_native_packs", lambda request, **_kw: SimpleNamespace(request=request, failures=())
    )
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.replace_capture_workload", lambda **_kw: events.append("replace"))

    def invoke(self, request, **kwargs):
        events.append("invoke")
        assert all(u["environment"]["MODEL"] == "/cache/huggingface/model" for u in request["launch"]["units"])

    monkeypatch.setattr(ColdSnapService, "_invoke", invoke)
    ColdSnapService().execute_explicit("capture", options, plan=plan, sctx=sctx, output="/tmp/test-capture.json")
    assert events == ["images-models", "resolve", "resolve", "cache", "replace", "invoke"]


def test_explicit_restore_resolves_imported_artifact_before_environment(monkeypatch, tmp_path):
    recipe, options, plan, sctx = _setup()
    recipe.env["RANK"] = "{launch.node_rank}"
    artifact = _artifact(tmp_path, build_request("capture", options, plan=plan, sctx=sctx))
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", lambda *_a, **_kw: None)
    monkeypatch.setattr(ColdSnapService, "_restore_artifact_path", lambda *_a, **_kw: artifact)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.stage_native_packs", lambda request, **_kw: SimpleNamespace(request=request, failures=())
    )
    monkeypatch.setattr(ColdSnapService, "_invoke", lambda *_a, **_kw: None)
    request, _, _ = ColdSnapService().execute_explicit("restore", options, plan=plan, sctx=sctx, artifact="oci://remote/capture")
    assert request["artifact"] == artifact
    assert [u["environment"]["RANK"] for u in request["launch"]["units"]] == ["0", "1"]


def test_restore_descriptor_and_activation_reuse_same_environment(monkeypatch, tmp_path):
    recipe, options, plan, sctx = _setup()
    recipe.env["RANK"] = "{launch.node_rank}"
    artifact = _artifact(tmp_path, build_request("capture", options, plan=plan, sctx=sctx))
    monkeypatch.setattr(ColdSnapService, "_restore_artifact_path", lambda *_a, **_kw: artifact)
    execution = ExecutionContext(options=options, plan=plan, sctx=sctx)
    service = ColdSnapService()
    descriptor = service.describe_restore(execution)
    state = PreparedRestore(request=descriptor.request, artifact=descriptor.artifact, selected_mode="native")
    prepared = SimpleNamespace(state=state, receipts={"coldsnap.capsules": RestoreActivationReceipt(state.request, {})})
    context = SimpleNamespace(execution=execution, prepared=prepared, comm_env=None)
    receipt = service.prepare_activation(context)
    assert receipt.request["launch"]["units"] == descriptor.request["launch"]["units"]


def test_input_marker_is_reserved():
    recipe, options, plan, sctx = _setup()
    recipe.env[ENV_INPUTS] = "spoofed"
    with pytest.raises(ValueError, match="reserved"):
        build_request("capture", options, plan=plan, sctx=sctx)


def test_multiple_units_on_one_host_share_host_rank_and_probe(monkeypatch, tmp_path):
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    recipe, options, plan, sctx = _setup()
    recipe.defaults.update({"tensor_parallel": 1, "data_parallel": 2})
    recipe.env.update({"MODEL": "{launch.model_path}", "RANK": "{launch.node_rank}"})
    plan.cluster.hosts_hardware = {"h1": HostHardware([AcceleratorSpec("nvidia", "h100", count=2, capabilities=frozenset({"cuda"}))])}
    plan = replace(plan, host_list=("h1",), placement=RankAssignment(by_rank=(RankSlot("h1", 0), RankSlot("h1", 1)), hosts_used=("h1",)))
    probe = _probe(monkeypatch)
    capture = build_request("capture", options, plan=plan, sctx=sctx)
    assert len(capture["launch"]["units"]) == 2
    assert probe.call_count == 1
    assert [u["environment"]["RANK"] for u in capture["launch"]["units"]] == ["0", "0"]
    probe.side_effect = AssertionError("restore probe")
    restored = build_request("restore", options, plan=plan, sctx=sctx, artifact=_artifact(tmp_path, capture))
    assert [u["environment"] for u in restored["launch"]["units"]] == [u["environment"] for u in capture["launch"]["units"]]


def test_bad_capture_context_never_reaches_replacement(monkeypatch):
    recipe, options, plan, sctx = _setup()
    recipe.env["MODEL"] = "{launch.model_path}"
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.prepare_capture_images",
        lambda *_a, **_kw: SimpleNamespace(content_images_by_node=None, comm_env=None),
    )
    monkeypatch.setattr("sparkrun.plugins.coldsnap.environment.probe_model_path", Mock(side_effect=ValueError("missing prepared model")))
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.replace_capture_workload", lambda **_kw: pytest.fail("replaced workload"))
    with pytest.raises(ValueError, match="missing prepared model"):
        ColdSnapService().execute_explicit("capture", options, plan=plan, sctx=sctx, output="/tmp/test-capture.json")


def test_native_restore_environment_keeps_model_preparation_disabled(monkeypatch, tmp_path):
    recipe, options, plan, sctx = _setup()
    recipe.env["MODEL"] = "{launch.model_path}"
    probe = _probe(monkeypatch)
    path = _artifact(tmp_path, build_request("capture", options, plan=plan, sctx=sctx))
    probe.side_effect = AssertionError("restore probe")
    request = build_request("restore", options, plan=plan, sctx=sctx, artifact=path, weight_mode="native")
    state = PreparedRestore(request=request, artifact={}, selected_mode="native")
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service._capsule_images", lambda *_a: ("capsule1", "capsule2"))
    prepared = ColdSnapService().finalize_restore(ExecutionContext(options=options, plan=plan, sctx=sctx), {"coldsnap.weights": state})
    assert prepared.assets.prepare_model is False


@pytest.mark.parametrize("scope", ["config", "cluster", "model", "missing"])
def test_capture_model_context_must_be_valid(monkeypatch, tmp_path, scope):
    recipe, options, plan, sctx = _setup()
    recipe.env["MODEL"] = "{launch.model_path}"
    probe = _probe(monkeypatch)
    if scope == "config":
        options = replace(options, overrides={"model_revision": "different"})
        with pytest.raises(ValueError, match="prepared recipe model"):
            build_request("capture", options, plan=plan, sctx=sctx)
        probe.assert_not_called()
    elif scope == "cluster":
        sctx.config.for_cluster = lambda cluster: SimpleNamespace(ssh_user="cluster-user", ssh_key=None, ssh_options=None)
        build_request("capture", options, plan=plan, sctx=sctx)
        assert probe.call_args.args[4]["ssh_user"] == "cluster-user"
    elif scope == "model":
        options = replace(options, overrides={"model": "other/model"})
        with pytest.raises(ValueError, match="prepared recipe model"):
            build_request("capture", options, plan=plan, sctx=sctx)
        probe.assert_not_called()
    else:
        request = build_request("capture", options, plan=plan, sctx=sctx)
        del request["launch"]["units"][0]["environment"]["MODEL"]
        with pytest.raises(ValueError, match="missing captured template"):
            build_request("restore", options, plan=plan, sctx=sctx, artifact=_artifact(tmp_path, request))
