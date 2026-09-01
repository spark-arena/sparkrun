"""The native cluster path's head gate is a rendezvous gate, not a liveness check.

Steps 6/7 start the head, wait for its distributed store to bind, then start the
workers — so a worker cannot race the store.  A launch with no shared store
(SGLang pure data parallelism: N standalone replicas) has nothing to wait for,
and waiting anyway burns the whole budget and then reports a healthy head as
dead (issue #284).
"""

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
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_script",
        lambda *a, **k: mock.MagicMock(success=True, returncode=0, stderr="", stdout=""),
    )
    monkeypatch.setattr("sparkrun.orchestration.ssh.start_log_capture", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.ssh.stop_log_capture", lambda *a, **k: [])

    calls = {"find_port": [], "wait_for_port": []}
    monkeypatch.setattr(_cluster_ops, "find_port", lambda ctx, host, port: calls["find_port"].append(port) or port)

    def fake_wait(host, port, **kwargs):
        calls["wait_for_port"].append((host, port))
        return True

    monkeypatch.setattr("sparkrun.orchestration.primitives.wait_for_port", fake_wait)

    def build_runtime(rendezvous_port):
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


def test_rendezvous_gate_waits_when_a_store_exists(native_cluster):
    """The default (dp == 1 / tp across nodes) behaviour is unchanged."""
    build_runtime, ctx, calls = native_cluster

    rc = _cluster_ops.run_native_cluster(runtime=build_runtime(25000), ctx=ctx, follow=False)

    assert rc == 0
    assert calls["wait_for_port"] == [("h1", 25000)]
    assert calls["find_port"] == [25000]


def test_no_gate_when_runtime_reports_no_rendezvous(native_cluster):
    """Independent replicas: workers start immediately, nothing is probed.

    Nothing binds the init port under pure DP, so both the port hunt and the
    wait are not merely useless but actively wrong — the wait would time out on
    a launch that is coming up fine.
    """
    build_runtime, ctx, calls = native_cluster
    runtime = build_runtime(None)

    rc = _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx, follow=False)

    assert rc == 0
    assert calls["wait_for_port"] == []
    assert calls["find_port"] == []
    # Every replica is still launched — one container per host.
    assert {c.kwargs["node_rank"] for c in runtime.generate_node_command.call_args_list} == {0, 1}
