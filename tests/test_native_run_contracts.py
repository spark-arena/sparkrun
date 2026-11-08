"""Native handlers preserve portable identity while returning usable substrate results."""

import re
from unittest.mock import Mock

import pytest
import yaml

from sparkrun import api
from sparkrun.api._run import plan, run
from sparkrun.core.run_handlers import RunHandler
from sparkrun.plugins.k8s import api as k8s
from sparkrun.plugins.k8s.orchestration.client import KubectlClient
from sparkrun.plugins.k8s.orchestration.names import native_jobset_name
from sparkrun.plugins.k8s.run import run_k8s
from test_benchmark_startup_collection import bench_env as bench_env
from test_run_option_contracts import run_env as run_env
from test_k8s_setup import _nodes_for, _SPARK_LABELS


@pytest.fixture
def native_env(run_env, monkeypatch):
    env = run_env
    monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda _: {"k8s": RunHandler("k8s", run_k8s)})
    env.client = KubectlClient("/unused/kubectl")
    env.make_client = Mock(return_value=env.client)
    monkeypatch.setattr(k8s._ops, "make_client", env.make_client)
    monkeypatch.setattr(
        "sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda *a, **kw: _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])
    )
    env.submitted = []
    monkeypatch.setattr("sparkrun.plugins.k8s.executor.K8sExecutor._client", lambda self: env.client)
    original = k8s._ops._launch_jobset

    def launch(*args, **kwargs):
        result = original(*args, **kwargs)
        env.submitted.append(result)
        return result

    monkeypatch.setattr(k8s._ops, "_launch_jobset", launch)
    return env


@pytest.mark.parametrize("default,override", [(8000, None), (9022, None), (8000, 9001), (8000, "9001"), (None, None)])
def test_native_plan_run_manifest_and_lifecycle_reference(native_env, monkeypatch, default, override):
    env = native_env
    if default is None:
        env.recipe.defaults.pop("port")
    else:
        env.recipe.defaults["port"] = default
    options = api.RunOptions(
        recipe=env.recipe,
        hosts=("localhost",),
        solo=True,
        executor="k8s",
        overrides={} if override is None else {"port": override},
        dry_run=True,
    )
    planned = plan(options, sctx=env.sctx)
    result = run(options, sctx=env.sctx, plan=planned)
    manifest = yaml.safe_load(env.submitted[0].manifests_yaml)
    name = manifest["metadata"]["name"]
    assert result.workload_id == planned.workload_id != name
    assert name == result.metadata["k8s_jobset"] == native_jobset_name(planned.workload_id, "gb10")
    assert manifest["metadata"]["annotations"]["sparkrun.cluster_id"] == planned.workload_id
    assert manifest["metadata"]["annotations"]["sparkrun.recipe_fingerprint"] == planned.recipe_fingerprint
    pod = manifest["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    master = next(e["value"] for e in pod["containers"][0]["env"] if e["name"] == "MASTER_ADDR")
    assert master == name + "-gb10-0-0." + name
    assert all(len(part) <= 63 and re.fullmatch(r"[a-z][-a-z0-9]*[a-z0-9]", part) for part in master.split("."))
    effective_port = int(override if override is not None else default or 8000)
    assert "--port %d" % effective_port in result.serve_command
    assert result.serve_port == effective_port
    assert result.recipe_fingerprint == planned.recipe_fingerprint
    assert result.intent_id == planned.intent_id and result.placement_token == planned.placement_token
    assert result.timeline is not None and result.launch_result is None
    assert result.dry_run and result.started_at > 0

    # Returned resource references work directly across lifecycle APIs.
    lookup = Mock(return_value=manifest)
    monkeypatch.setattr(env.client, "run_json", lookup)
    monkeypatch.setattr(env.client, "run", Mock(return_value=Mock(success=True)))
    manifest["metadata"]["uid"] = "submitted-jobset"
    monkeypatch.setattr(
        "sparkrun.plugins.k8s.orchestration.logs._query_json",
        lambda client, args, **kw: manifest if args[1] == "jobset" else {"items": []},
    )
    assert k8s.jobset_status(env.sctx, name=name) == manifest
    with pytest.raises(api.SparkrunError, match="pending, queued"):
        k8s.logs(env.sctx, name=name)
    assert k8s.stop_jobset(env.sctx, name=name)
    assert all(call.args[0][2] == name for call in lookup.call_args_list)


@pytest.mark.parametrize("port", [0, -1, 65536, True, 3.5, "invalid"])
def test_invalid_native_port_fails_before_cluster_preparation(native_env, port):
    env = native_env
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, executor="k8s", overrides={"port": port}, dry_run=True)
    with pytest.raises(api.SparkrunError, match="port"):
        run(options, sctx=env.sctx)
    env.make_client.assert_not_called()
    assert env.submitted == []


@pytest.mark.parametrize(
    "name,model,replicas",
    [
        ("bad_name", "gb10", 1),
        ("9bad", "gb10", 1),
        ("a.b", "gb10", 1),
        ("a" * 55, "gb10", 1),
        ("a" * 54, "gb10", 11),
        ("good", "bad_model", 1),
    ],
)
def test_invalid_jobset_names_fail_before_replacement(native_env, name, model, replicas):
    env = native_env
    replacement = Mock()
    env.client.apply = Mock(side_effect=AssertionError("invalid manifests must not be submitted"))
    with pytest.raises(k8s.JobSetLaunchError, match="resource name"):
        k8s.launch_jobset(env.sctx, name=name, rank_models=[model] * replicas, image="img", serve_command="serve", before_start=replacement)
    replacement.assert_not_called()
    env.client.apply.assert_not_called()


def test_portable_id_projection_is_bounded_and_distinguishes_normalization_collisions():
    ids = ["sparkrun_a_b", "sparkrun-a-b", "sparkrun_" + "a" * 200, "sparkrun_" + "a" * 199 + "b", "123", "雪"]
    names = [native_jobset_name(value, "gb10") for value in ids]
    assert len(set(names)) == len(ids)
    assert names == [native_jobset_name(value, "gb10") for value in ids]
    assert all(len(name + "-gb10-0-0") <= 63 and re.fullmatch(r"[a-z][-a-z0-9]*[a-z0-9]", name) for name in names)


@pytest.mark.parametrize("fingerprint", ["known-original-fingerprint", ""])
def test_handler_reuse_preserves_existing_metadata(run_env, monkeypatch, fingerprint):
    from dataclasses import replace
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    env = run_env
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, ensure=True)
    planned = plan(options, sctx=env.sctx)
    old_id = generate_cluster_id(planned.intent_id, "c" * 12)
    old = api.RunResult(
        workload_id=old_id,
        host_list=("localhost",),
        placement=None,
        scheduler="greedy",
        runtime=planned.runtime.runtime_name,
        executor="docker",
        started_at=123.0,
        dry_run=False,
        is_solo=True,
        already_running=True,
        recipe_fingerprint=fingerprint,
    )
    monkeypatch.setattr("sparkrun.api._intent.find_running_intent", lambda *a, **kw: None)
    monkeypatch.setattr(
        "sparkrun.core.run_handlers.registered_run_handlers", lambda _: {"docker": RunHandler("docker", lambda *a, **kw: old)}
    )
    replacement = Mock(side_effect=AssertionError("reuse must not replace anything"))
    monkeypatch.setattr("sparkrun.api._run._evict_superseded_deployments", replacement)
    result = run(options, sctx=env.sctx, plan=planned)
    assert result == replace(old, started_at=result.started_at, intent_id=planned.intent_id, placement_token="c" * 12)
    assert result.timeline is None and result.launch_result is None
    replacement.assert_not_called()


@pytest.mark.parametrize(
    "kind,value",
    [("pre_exec", ["echo patch"]), ("mods", ["fix-something"]), ("post_exec", ["curl localhost"]), ("post_commands", ["echo done"])],
)
@pytest.mark.parametrize("dry_run", [True, False])
def test_recipe_hooks_and_mods_are_refused_not_silently_skipped(native_env, kind, value, dry_run):
    """The JobSet path runs no hooks; a modded recipe must not serve an unpatched engine."""
    env = native_env
    env.client.context = "test-context"  # planning pins the context; skip that read-only kubectl call
    setattr(env.recipe, kind, value)
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, executor="k8s", trust=True, dry_run=dry_run)

    reason = "not applied on this path" if kind in ("pre_exec", "mods") else "no reachable endpoint to wait on"
    with pytest.raises(api.SparkrunError, match=r"cannot launch this recipe: %s (is|runs) .*%s" % (kind, reason)):
        run(options, sctx=env.sctx)

    env.make_client.assert_not_called()
    assert env.submitted == []


def test_the_refusal_names_each_reason_once(native_env):
    env = native_env
    env.client.context = "test-context"
    env.recipe.pre_exec, env.recipe.mods, env.recipe.post_commands = ["echo a"], ["fix-x"], ["echo b"]
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, executor="k8s", trust=True, dry_run=True)

    with pytest.raises(api.SparkrunError) as error:
        run(options, sctx=env.sctx)

    message = str(error.value)
    assert "pre_exec and mods are not applied on this path" in message
    assert "post_commands runs once the server is ready" in message
    assert "post_exec" not in message
