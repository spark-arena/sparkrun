"""Destination identity, preparation preservation, and native endpoint contracts."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from sparkrun import api
from sparkrun.api._run import run, plan
from sparkrun.api._stop import stop
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.orchestration.job_metadata import load_job_metadata
from test_benchmark_startup_collection import bench_env as bench_env
from test_run_option_contracts import run_env as run_env
from test_native_lifecycle_contracts import lifecycle_env as lifecycle_env


def _options(env):
    return replace(env.options, cluster=ClusterDefinition(name="native-target", hosts=["localhost"], executor="k8s"))


def _other_target(env, options, dimension):
    cfg = dict(options.executor_config)
    target = list(env.target)
    if dimension == "namespace":
        cfg["k8s_namespace"] = target[2] = "other-namespace"
    elif dimension == "context":
        cfg["k8s_context"] = target[1] = "other-context"
    else:
        cfg["kubeconfig"] = target[0] = env.kubeconfig + "-other"
    return replace(options, executor_config=cfg), tuple(target)


@pytest.mark.parametrize("dimension", ["namespace", "context", "kubeconfig"])
def test_native_ensure_uses_requested_target(lifecycle_env, dimension):
    env = lifecycle_env
    options = _options(env)
    first = run(options, sctx=env.sctx)
    second_options, second_target = _other_target(env, options, dimension)
    env.calls.clear()
    result = run(replace(second_options, ensure=True), sctx=env.sctx)
    assert not result.already_running and result.workload_id != first.workload_id
    assert env.objects[env.target] and env.objects[second_target]
    assert env.calls and all(target == second_target for target, args in env.calls)
    reused = run(replace(second_options, ensure=True), sctx=env.sctx)
    assert reused.already_running and reused.workload_id == result.workload_id


@pytest.mark.parametrize("dimension", ["namespace", "context", "kubeconfig"])
def test_native_targets_have_independent_ids_and_lifecycle(lifecycle_env, dimension):
    env = lifecycle_env
    options = _options(env)
    first = run(options, sctx=env.sctx)
    second_options, second_target = _other_target(env, options, dimension)
    second = run(second_options, sctx=env.sctx)
    assert first.workload_id != second.workload_id
    assert env.objects[env.target] and env.objects[second_target]
    saved = load_job_metadata(first.workload_id, cache_dir=str(env.sctx.config.cache_dir))
    assert (
        saved["executor_config"]["kubeconfig"],
        saved["executor_config"]["k8s_context"],
        saved["executor_config"]["k8s_namespace"],
    ) == env.target
    env.calls.clear()
    stopped = stop(workload_id=first.workload_id, sctx=replace(env.sctx, timing=None))
    assert stopped.success and stopped.containers_removed == 1
    assert not env.objects[env.target] and env.objects[second_target]
    assert load_job_metadata(second.workload_id, cache_dir=str(env.sctx.config.cache_dir))
    assert stop(workload_id=second.workload_id, sctx=replace(env.sctx, timing=None)).success
    assert not env.objects[second_target]
    assert load_job_metadata(first.workload_id, cache_dir=str(env.sctx.config.cache_dir)) is None


@pytest.mark.parametrize("failure", ["policy", "runtime_prepare", "distribution", "replacement"])
def test_failed_preparation_preserves_live_metadata(lifecycle_env, monkeypatch, tmp_path, failure):
    env = lifecycle_env
    first = run(_options(env), sctx=env.sctx)
    before = load_job_metadata(first.workload_id, cache_dir=str(env.sctx.config.cache_dir))
    assert before["native_resource"] and before["executor"] == "k8s"
    monkeypatch.setattr(env.launch.runtime, "prepare", lambda *a, **kw: None)
    launched = Mock(return_value=0)
    monkeypatch.setattr(env.launch.runtime, "run", launched)
    replaced = Mock()
    monkeypatch.setattr("sparkrun.api._run._evict_superseded_deployments", replaced)
    monkeypatch.setattr("sparkrun.core.launcher.resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda *a, **kw: SimpleNamespace(mode="local"))
    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", lambda *a, **kw: (None, {}, {}, {}))
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    options = api.RunOptions(
        recipe=env.recipe,
        hosts=("localhost",),
        solo=True,
        executor="docker",
        trust=True,
        workload_id_override=first.workload_id,
        executor_config={"security_opt": ["seccomp=" + str(tmp_path / "missing-policy.json")]} if failure == "policy" else None,
        sync_tuning=False,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("preparation interrupted")

    if failure == "runtime_prepare":
        monkeypatch.setattr(env.launch.runtime, "prepare", fail)
    elif failure == "distribution":
        monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", fail)
    elif failure == "replacement":
        replaced.side_effect = fail
    with pytest.raises(api.SparkrunError, match="Cannot load Docker seccomp profile|preparation interrupted"):
        run(options, sctx=env.sctx)
    launched.assert_not_called()
    if failure == "replacement":
        replaced.assert_called_once()
    else:
        replaced.assert_not_called()
    after = load_job_metadata(first.workload_id, cache_dir=str(env.sctx.config.cache_dir))
    assert after == before
    assert env.objects[env.target]


def test_plan_pins_default_context_before_configuration_changes(lifecycle_env):
    env = lifecycle_env
    env.sctx.config.set("k8s", {"kubeconfig": env.kubeconfig, "namespace": "launch-ns"})
    planned = plan(env.options, sctx=env.sctx)
    env.current_context = "changed-context"
    env.sctx.config.set("k8s", {"kubeconfig": "/different", "context": "changed-context", "namespace": "different"})
    result = run(env.options, sctx=env.sctx, plan=planned)
    assert result.workload_id == planned.workload_id
    assert env.objects[env.target]
    assert all(target == env.target for target, args in env.calls if args[0] in {"get", "apply"})


@pytest.mark.parametrize("skip_run", [False, True])
def test_native_benchmark_rejects_missing_endpoint_before_submission(lifecycle_env, monkeypatch, skip_run):
    env = lifecycle_env
    env.recipe.executor = "k8s"
    env.recipe.executor_config = {"kubectl_path": "/fake/kubectl"}
    monkeypatch.setattr(api, "plan", plan)
    monkeypatch.setattr(api, "run", run)
    options = api.BenchmarkOptions(
        recipe=env.recipe,
        hosts=("localhost",),
        framework="test-bench",
        solo=True,
        export_files=False,
        resume=api.ResumeMode.FRESH,
        skip_run=skip_run,
    )
    with pytest.raises(api.BenchmarkFailed, match="reachable inference endpoint"):
        api.benchmark(options, sctx=env.sctx)
    assert not env.calls
    env.endpoint.assert_not_called()
    env.fw.build_benchmark_command.assert_not_called()


@pytest.mark.parametrize("source", ["cluster", "recipe", "caller"])
def test_effective_target_precedence_reaches_occupancy_and_launch(lifecycle_env, monkeypatch, source):
    env = lifecycle_env
    options = _options(env)
    first = run(options, sctx=env.sctx)
    settings = {"k8s_namespace": "other-namespace"}
    if source == "cluster":
        options = replace(options, cluster=replace(options.cluster, executor_config=settings))
    elif source == "recipe":
        env.recipe.executor_config = settings
    else:
        env.recipe.executor_config = {"k8s_namespace": "recipe-namespace"}
        options = replace(options, executor_config={**options.executor_config, **settings})
    seen = []

    def placement(*args, cluster_def, **kwargs):
        seen.append(cluster_def.executor_config["k8s_namespace"])
        return ["localhost"], True, [], None

    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", placement)
    second = run(replace(options, ensure=True), sctx=env.sctx)
    assert not second.already_running and first.workload_id != second.workload_id
    assert seen == ["other-namespace"]
    assert env.objects[(env.kubeconfig, "launch-context", "other-namespace")]


def test_connection_binary_does_not_change_destination_identity(lifecycle_env):
    env = lifecycle_env
    first = plan(env.options, sctx=env.sctx)
    second = plan(replace(env.options, executor_config={"kubectl_path": "/another/kubectl"}), sctx=env.sctx)
    assert first.workload_id == second.workload_id
    assert first.cluster.executor_config["kubectl_path"] == "/fake/kubectl"
    assert second.cluster.executor_config["kubectl_path"] == "/another/kubectl"
    with pytest.raises(TypeError):
        first.executor_target.config["k8s_namespace"] = "changed"
    overrides = first.executor_target.overrides
    overrides["k8s_namespace"] = "changed"
    assert first.executor_target.config["k8s_namespace"] == "launch-ns"


def test_cli_early_ensure_observes_recipe_target(lifecycle_env, monkeypatch):
    from click.testing import CliRunner
    from sparkrun.cli import main

    env = lifecycle_env
    env.recipe.executor = "k8s"
    first = run(env.options, sctx=env.sctx)
    env.recipe.executor_config = {"k8s_namespace": "other-namespace"}
    second = run(env.options, sctx=env.sctx)
    assert second.workload_id != first.workload_id
    monkeypatch.setattr("sparkrun.cli._run._get_context", lambda ctx: env.sctx)
    monkeypatch.setattr("sparkrun.cli._run._load_recipe", lambda *a, **kw: (env.recipe, None, None))
    monkeypatch.setattr(api, "plan", Mock(side_effect=AssertionError("ensure should return before placement")))
    result = CliRunner().invoke(main, ["run", "test", "--hosts", "localhost", "--ensure", "--no-ready-wait"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Job already running (cluster_id: %s)" % second.workload_id in result.output
    api.plan.assert_not_called()


def test_recipe_lifecycle_discovery_uses_recipe_destination(lifecycle_env, monkeypatch):
    env = lifecycle_env
    first = run(env.options, sctx=env.sctx)
    env.recipe.executor = "k8s"
    env.recipe.executor_config = {"k8s_namespace": "other-namespace"}
    second = run(env.options, sctx=env.sctx)
    sources = Mock(return_value=[])
    monkeypatch.setattr("sparkrun.plugins.k8s.executor.K8sExecutor.native_log_sources", sources)
    assert list(api.logs(recipe=env.recipe, hosts=("localhost",), sctx=env.sctx)) == []
    assert sources.call_args.args[0] == second.workload_id
    stopped = stop(recipe=env.recipe, hosts=("localhost",), sctx=env.sctx)
    assert stopped.success and stopped.workload_id == second.workload_id
    assert env.objects[env.target]
    assert load_job_metadata(first.workload_id, cache_dir=str(env.sctx.config.cache_dir))


@pytest.mark.parametrize("interrupt", [False, True])
def test_normal_submission_records_effective_executor_and_preserves_ip_maps(lifecycle_env, monkeypatch, tmp_path, interrupt):
    env = lifecycle_env
    monkeypatch.setattr(env.launch.runtime, "prepare", lambda *a, **kw: None)
    monkeypatch.setattr(env.launch.runtime, "_collect_runtime_info", lambda *a, **kw: {"engine": "test-version"})
    monkeypatch.setattr("sparkrun.api._run._evict_superseded_deployments", lambda **kw: ([], set()))
    monkeypatch.setattr("sparkrun.core.launcher.resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda *a, **kw: SimpleNamespace(mode="local"))
    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.distribute_from_config",
        lambda *a, **kw: (None, {"localhost": "ib-address"}, {"localhost": "mgmt-address"}, {}),
    )
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    options = api.RunOptions(
        recipe=env.recipe,
        hosts=("localhost",),
        solo=True,
        executor="docker",
        trust=True,
        sync_tuning=False,
        executor_config={"shm_size": "7g"},
        cluster=ClusterDefinition(name="example", hosts=["localhost"], executor="local"),
    )
    planned = plan(options, sctx=env.sctx)

    # The canned probe observed its own IB address for this host; the one
    # distribution returned must still be what is recorded.
    assert planned.host_hardware["localhost"].ib_info["DETECTED_IB_IPS"] == "192.168.99.13"

    def submit(**kwargs):
        saved = load_job_metadata(planned.workload_id, cache_dir=str(env.sctx.config.cache_dir))
        assert saved["executor"] == "docker" and saved["executor_config"]["shm_size"] == "7g"
        assert saved["ib_ip_map"] == {"localhost": "ib-address"}
        assert saved["mgmt_ip_map"] == {"localhost": "mgmt-address"}
        if interrupt:
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(env.launch.runtime, "run", submit)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            run(options, sctx=env.sctx, plan=planned)
    else:
        assert run(options, sctx=env.sctx, plan=planned).rc == 0
    saved = load_job_metadata(planned.workload_id, cache_dir=str(env.sctx.config.cache_dir))
    assert saved["executor"] == "docker" and saved["executor_config"]["shm_size"] == "7g"
    assert saved["ib_ip_map"] == {"localhost": "ib-address"} and saved["mgmt_ip_map"] == {"localhost": "mgmt-address"}
    if not interrupt:
        assert saved["runtime_info"] == {"engine": "test-version"}


def test_executor_target_detaches_nested_connection_settings():
    from sparkrun.orchestration.executor import ExecutorTarget

    source = {"connection": {"servers": ["first"]}}
    target = ExecutorTarget("example", source, destination_key="derived")
    source["connection"]["servers"].append("second")
    assert target.config["connection"]["servers"] == ("first",)
    with pytest.raises(TypeError):
        target.config["connection"]["servers"] = []
    copy = target.overrides
    copy["connection"]["servers"].append("third")
    assert target.overrides == {"executor": "example", "connection": {"servers": ["first"]}}


def test_cli_ensure_leaves_executor_config_the_callers_own(tmp_path, monkeypatch):
    """``--ensure`` scopes the cluster only: target settings (selector, pid/log dirs,
    kubeconfig) joining ``executor_config`` would be recorded as the caller's own
    executor layer and baked into later exports (#323)."""
    import yaml
    from click.testing import CliRunner
    from sparkrun.cli import main

    recipe = tmp_path / "ensure-test.yaml"
    recipe.write_text(yaml.safe_dump({"model": "test/model", "runtime": "vllm", "container": "test/image"}), encoding="utf-8")
    seen = []

    def stop_at_plan(options, **kwargs):
        seen.append(options)
        raise api.SparkrunError("stop here")

    monkeypatch.setattr(api, "find_running_intent", lambda *a, **kw: None)
    monkeypatch.setattr(api, "plan", stop_at_plan)
    result = CliRunner().invoke(main, ["run", str(recipe), "--hosts", "localhost", "--ensure", "-o", "entrypoint="])
    assert seen, (result.output, result.exception)
    assert seen[0].executor_config == {"entrypoint": ""}
    assert seen[0].executor_overrides() == {"entrypoint": ""}
