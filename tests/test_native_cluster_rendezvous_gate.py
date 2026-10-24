"""The native cluster path's head gate is a rendezvous gate, not a liveness check.

Steps 6/7 start the head, wait for its distributed store to bind, then start the
workers — so a worker cannot race the store.  A launch with no shared store
(SGLang pure data parallelism: N standalone replicas) has nothing to wait for,
and waiting anyway burns the whole budget and then reports a healthy head as
dead (issue #284).
"""

from types import SimpleNamespace

from sparkrun.core.recipe import Recipe
from unittest import mock

import pytest

from sparkrun.runtimes import _cluster_ops
from sparkrun.runtimes._cluster_ops import ClusterContext


@pytest.fixture
def native_cluster(monkeypatch):
    """A run_native_cluster harness with every remote call stubbed out."""
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    monkeypatch.setattr(_cluster_ops, "detect_ib_with_ips", lambda *a, **k: (ClusterCommEnv.empty(), {}, {}))
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda ctx: "10.0.0.1")
    monkeypatch.setattr(_cluster_ops, "resolve_hosts_for_init", lambda ctx, head_ip: ctx.hosts)
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *a, **k: 0)
    monkeypatch.setattr(_cluster_ops, "run_pre_serve_hooks", lambda *a, **k: None)
    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *a, **k: None)

    calls = {"find_port": [], "find_port_range": [], "wait_for_port": [], "remote_hosts": []}

    def fake_remote(host, script, **kwargs):
        calls["remote_hosts"].append(host)
        return mock.MagicMock(success=True, returncode=0, stderr="", stdout="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", fake_remote)
    monkeypatch.setattr("sparkrun.orchestration.ssh.start_log_capture", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.ssh.stop_log_capture", lambda *a, **k: [])

    monkeypatch.setattr(_cluster_ops, "find_port", lambda ctx, host, port: calls["find_port"].append(port) or port)
    monkeypatch.setattr(
        _cluster_ops,
        "find_port_range",
        lambda ctx, host, port, count: calls["find_port_range"].append((host, port, count)) or port,
    )

    def fake_wait(host, port, **kwargs):
        calls["wait_for_port"].append((host, port))
        return True

    monkeypatch.setattr("sparkrun.orchestration.primitives.wait_for_port", fake_wait)

    def build_runtime(rendezvous_port, *, start_order="head_first", port_span=1):
        runtime = mock.MagicMock()
        runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
        runtime._resolve_executor.return_value.workload_labels_for_cluster = lambda **kw: {}
        runtime._resolve_executor.return_value.generate_exec_serve_script = lambda **kw: "#!/bin/bash\necho noop\n"
        runtime.generate_node_command = mock.MagicMock(return_value="serve")
        runtime.get_extra_docker_opts = lambda: []
        runtime._print_cluster_banner = mock.MagicMock()
        runtime._cluster_log_mode = lambda: "docker"
        runtime._cluster_init_port = lambda recipe, overrides, head_ip, num_nodes, **kw: 8000
        runtime._cluster_skip_keys = lambda recipe, overrides, head_ip, num_nodes, **kw: frozenset()
        runtime._cluster_extra_volumes = lambda recipe, overrides: {}
        runtime.native_rendezvous_port = mock.MagicMock(return_value=rendezvous_port)
        runtime.native_start_order = lambda: start_order
        runtime.native_rendezvous_port_span = lambda: port_span
        return runtime

    ctx = ClusterContext(
        hosts=["h1", "h2"],
        head_host="h1",
        worker_hosts=["h2"],
        num_nodes=2,
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="cid",
        image="img",
        dry_run=False,
        config=None,
    )
    return build_runtime, ctx, calls


@pytest.mark.usefixtures("reachable_init_workers")
def test_rendezvous_gate_waits_when_a_store_exists(native_cluster):
    """The default (dp == 1 / tp across nodes) behaviour is unchanged."""
    build_runtime, ctx, calls = native_cluster

    rc = _cluster_ops.run_native_cluster(
        runtime=build_runtime(25000),
        ctx=ctx,
        follow=False,
        recipe=Recipe.from_dict({"runtime": "vllm-distributed", "model": "fixture/model"}),
    )

    assert rc == 0
    assert calls["wait_for_port"] == [("h1", 25000)]
    assert calls["find_port"] == [25000]


@pytest.mark.usefixtures("reachable_init_workers")
def test_no_gate_when_runtime_reports_no_rendezvous(native_cluster):
    """Independent replicas: workers start immediately, nothing is probed.

    Nothing binds the init port under pure DP, so both the port hunt and the
    wait are not merely useless but actively wrong — the wait would time out on
    a launch that is coming up fine.
    """
    build_runtime, ctx, calls = native_cluster
    runtime = build_runtime(None)

    rc = _cluster_ops.run_native_cluster(
        runtime=runtime, ctx=ctx, follow=False, recipe=Recipe.from_dict({"runtime": "vllm-distributed", "model": "fixture/model"})
    )

    assert rc == 0
    assert calls["wait_for_port"] == []
    assert calls["find_port"] == []
    # Every replica is still launched — one container per host.
    assert {c.kwargs["node_rank"] for c in runtime.generate_node_command.call_args_list} == {0, 1}


@pytest.mark.usefixtures("reachable_init_workers")
def test_workers_first_starts_followers_before_head_and_skips_gate(native_cluster):
    """TensorFold-style follower ranks must be listening before rank 0 starts.

    The head-first path would deadlock here: rank 0 blocks on the followers it
    is supposed to wait for, while the orchestrator has not started them yet.
    """
    build_runtime, ctx, calls = native_cluster
    runtime = build_runtime(25000, start_order="workers_first")

    rc = _cluster_ops.run_native_cluster(
        runtime=runtime,
        ctx=ctx,
        follow=False,
        recipe=Recipe.from_dict({"runtime": "tensorfold", "model": "fixture/model"}),
    )

    assert rc == 0
    # Worker rank 1 is executed first; rank 0 is executed last.
    assert calls["remote_hosts"] == ["h2", "h1"]
    assert calls["wait_for_port"] == []
    assert calls["find_port"] == [25000]
    # The rendezvous port is still reserved and passed to both commands.
    assert {c.kwargs["node_rank"] for c in runtime.generate_node_command.call_args_list} == {0, 1}


@pytest.mark.usefixtures("reachable_init_workers")
def test_workers_first_reserves_contiguous_port_span(native_cluster):
    """An engine opening ``port + 1`` too must get a reserved range, not a lucky port."""
    build_runtime, ctx, calls = native_cluster
    runtime = build_runtime(25000, start_order="workers_first", port_span=2)

    rc = _cluster_ops.run_native_cluster(
        runtime=runtime,
        ctx=ctx,
        follow=False,
        recipe=Recipe.from_dict({"runtime": "tensorfold", "model": "fixture/model"}),
    )

    assert rc == 0
    assert calls["find_port_range"] == [("h1", 25000, 2)]
    assert calls["find_port"] == []
    assert calls["wait_for_port"] == []


@pytest.mark.usefixtures("reachable_init_workers")
def test_head_first_can_reserve_a_port_span_too(native_cluster):
    """The span hook is orthogonal to the start order; head-first keeps its gate."""
    build_runtime, ctx, calls = native_cluster
    runtime = build_runtime(25000, port_span=2)

    rc = _cluster_ops.run_native_cluster(
        runtime=runtime,
        ctx=ctx,
        follow=False,
        recipe=Recipe.from_dict({"runtime": "tensorfold", "model": "fixture/model"}),
    )

    assert rc == 0
    assert calls["find_port_range"] == [("h1", 25000, 2)]
    assert calls["wait_for_port"] == [("h1", 25000)]
    assert calls["remote_hosts"] == ["h1", "h2"]


def test_find_port_range_skips_an_occupied_tail(monkeypatch):
    """The span is a reservation, not two independent single-port probes."""
    from sparkrun.orchestration import primitives

    probes = []

    def fake_find(host, preferred, **kwargs):
        return preferred

    def fake_run(host, cmd, **kwargs):
        port = int(cmd.rsplit(" ", 1)[-1])
        probes.append(port)
        return SimpleNamespace(success=port == 25001, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(primitives, "find_available_port", fake_find)
    monkeypatch.setattr(primitives, "run_command_on_host", fake_run)

    ctx = SimpleNamespace(ssh_kwargs={}, dry_run=False)
    assert _cluster_ops.find_port_range(ctx, "h1", 25000, 2) == 25001
    assert probes == [25001, 25002]


def test_find_port_range_is_a_single_port_hunt_for_span_one(monkeypatch):
    from sparkrun.orchestration import primitives

    monkeypatch.setattr(primitives, "find_available_port", lambda host, preferred, **kw: preferred + 3)
    monkeypatch.setattr(primitives, "run_command_on_host", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe")))
    ctx = SimpleNamespace(ssh_kwargs={}, dry_run=False)
    assert _cluster_ops.find_port_range(ctx, "h1", 25000, 1) == 25003
