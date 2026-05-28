"""Tests for the launch-failure cleanup helpers in ``runtimes/_cluster_ops``.

Covers ``cleanup_after_failure``, ``cleanup_solo_after_failure``, and
``dump_serve_log`` -- the helpers added for 0.3.x to stop leaking containers
on launch failure and to surface the real error from the serve log file.
"""

from __future__ import annotations

import logging
from unittest import mock

from sparkrun.runtimes import _cluster_ops
from sparkrun.runtimes._cluster_ops import (
    ClusterContext,
    cleanup_after_failure,
    cleanup_solo_after_failure,
    dump_serve_log,
)


def _make_ctx(hosts: list[str] | None = None, *, dry_run: bool = False) -> ClusterContext:
    hosts = hosts or ["h1", "h2"]
    return ClusterContext(
        hosts=hosts,
        head_host=hosts[0],
        worker_hosts=hosts[1:],
        num_nodes=len(hosts),
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="cid",
        image="img",
        dry_run=dry_run,
        config=None,
    )


def _make_executor(*, auto_remove: bool = True) -> mock.MagicMock:
    """Build a minimal Executor mock with the attributes the helpers touch."""
    executor = mock.MagicMock()
    executor.config = mock.MagicMock(auto_remove=auto_remove)
    executor.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    executor.stop_cmd = lambda name: "docker rm -f %s" % name
    return executor


# ---------------------------------------------------------------------------
# cleanup_after_failure
# ---------------------------------------------------------------------------


def test_cleanup_after_failure_ranked_runs_stop_on_each_host(monkeypatch):
    """auto_remove=True → stops cid_node_<rank> on each host (one call per host)."""
    ctx = _make_ctx()
    executor = _make_executor(auto_remove=True)

    calls: list[tuple[str, str]] = []

    def fake_run_remote_command(host, cmd, *, timeout, dry_run, **ssh_kwargs):
        calls.append((host, cmd))
        return mock.MagicMock(success=True)

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", fake_run_remote_command)

    cleanup_after_failure(ctx, executor, reason="boom")

    assert calls == [
        ("h1", "docker rm -f cid_node_0"),
        ("h2", "docker rm -f cid_node_1"),
    ]


def test_cleanup_after_failure_respects_no_rm(monkeypatch, caplog):
    """auto_remove=False (--no-rm) → no stop calls; warning names cluster_id."""
    ctx = _make_ctx()
    executor = _make_executor(auto_remove=False)

    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_command",
        lambda *a, **k: called.append(a) or mock.MagicMock(success=True),
    )

    with caplog.at_level(logging.WARNING, logger="sparkrun.runtimes._cluster_ops"):
        cleanup_after_failure(ctx, executor, reason="serve crashed")

    assert called == []
    assert any(
        "Containers left running because --no-rm was set" in rec.message
        and "sparkrun stop cid" in rec.message
        and "serve crashed" in rec.message
        for rec in caplog.records
    )


def test_cleanup_after_failure_explicit_container_names_uses_named_primitive(monkeypatch):
    """container_names=... routes to cleanup_containers (named primitive)."""
    ctx = _make_ctx()
    executor = _make_executor(auto_remove=True)

    captured: dict = {}

    def fake_cleanup_containers(hosts, names, *, ssh_kwargs, dry_run):
        captured["hosts"] = list(hosts)
        captured["names"] = list(names)
        captured["dry_run"] = dry_run

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers", fake_cleanup_containers)

    cleanup_after_failure(
        ctx,
        executor,
        container_names=["cid_head", "cid_worker"],
        reason="ray serve failed",
    )

    assert captured["hosts"] == ["h1", "h2"]
    assert captured["names"] == ["cid_head", "cid_worker"]


def test_cleanup_after_failure_hosts_subset_restricts_cleanup(monkeypatch):
    """hosts=[ctx.head_host] only stops on the supplied subset."""
    ctx = _make_ctx(hosts=["h1", "h2", "h3"])
    executor = _make_executor(auto_remove=True)

    calls: list[str] = []
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_command",
        lambda host, cmd, **kw: calls.append(host) or mock.MagicMock(success=True),
    )

    cleanup_after_failure(ctx, executor, hosts=["h1"], reason="head only")

    assert calls == ["h1"]


def test_cleanup_after_failure_swallows_cleanup_errors(monkeypatch, caplog):
    """Errors during cleanup are logged at warning and don't propagate."""
    ctx = _make_ctx()
    executor = _make_executor(auto_remove=True)

    def boom(*a, **k):
        raise RuntimeError("ssh down")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", boom)

    with caplog.at_level(logging.WARNING, logger="sparkrun.runtimes._cluster_ops"):
        cleanup_after_failure(ctx, executor, reason="x")

    assert any("Cleanup encountered errors (continuing): ssh down" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# cleanup_solo_after_failure
# ---------------------------------------------------------------------------


def test_cleanup_solo_after_failure_runs_stop_on_single_host(monkeypatch):
    """auto_remove=True solo → single stop call for the named container."""
    executor = _make_executor(auto_remove=True)

    calls: list[tuple[str, str]] = []

    def fake_run_remote_command(host, cmd, *, timeout, dry_run, **ssh_kwargs):
        calls.append((host, cmd))
        return mock.MagicMock(success=True)

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", fake_run_remote_command)

    cleanup_solo_after_failure(
        executor,
        host="h1",
        container_name="cid_solo",
        ssh_kwargs={},
        dry_run=False,
        cluster_id="cid",
        reason="serve crashed",
    )

    assert calls == [("h1", "docker rm -f cid_solo")]


def test_cleanup_solo_after_failure_respects_no_rm(monkeypatch, caplog):
    """auto_remove=False solo → no stop call; warning names cluster_id."""
    executor = _make_executor(auto_remove=False)

    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_command",
        lambda *a, **k: called.append(a) or mock.MagicMock(success=True),
    )

    with caplog.at_level(logging.WARNING, logger="sparkrun.runtimes._cluster_ops"):
        cleanup_solo_after_failure(
            executor,
            host="h1",
            container_name="cid_solo",
            ssh_kwargs={},
            dry_run=False,
            cluster_id="cid",
            reason="solo serve failed",
        )

    assert called == []
    assert any(
        "Container left running because --no-rm was set" in rec.message and "sparkrun stop cid" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# dump_serve_log
# ---------------------------------------------------------------------------


def test_dump_serve_log_emits_content_at_error_level(monkeypatch, caplog):
    """Happy path: each non-empty log line is logged at ERROR with the host/container header."""
    log_text = "Traceback (most recent call last):\n  ImportError: libtorch_cuda.so\n"

    def fake_run_remote_command(host, cmd, *, timeout, dry_run, **ssh_kwargs):
        assert "docker exec mycontainer cat /tmp/sparkrun_serve.log" in cmd
        return mock.MagicMock(stdout=log_text, stderr="", returncode=0, success=True)

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", fake_run_remote_command)

    with caplog.at_level(logging.ERROR, logger="sparkrun.runtimes._cluster_ops"):
        dump_serve_log("h1", "mycontainer", ssh_kwargs={})

    messages = [r.message for r in caplog.records]
    assert any("Serve log /tmp/sparkrun_serve.log on h1 (mycontainer)" in m for m in messages)
    assert any("ImportError: libtorch_cuda.so" in m for m in messages)


def test_dump_serve_log_empty_content_reports_no_content(monkeypatch, caplog):
    """Empty stdout → "no content" diagnostic, not a misleading blank section."""

    def fake_run_remote_command(host, cmd, *, timeout, dry_run, **ssh_kwargs):
        return mock.MagicMock(stdout="", stderr="", returncode=0, success=True)

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", fake_run_remote_command)

    with caplog.at_level(logging.ERROR, logger="sparkrun.runtimes._cluster_ops"):
        dump_serve_log("h1", "mycontainer", ssh_kwargs={})

    assert any("No content in /tmp/sparkrun_serve.log for mycontainer on h1" in r.message for r in caplog.records)


def test_dump_serve_log_dry_run_skips_logging(monkeypatch, caplog):
    """dry_run=True returns without emitting log records."""

    def fake_run_remote_command(host, cmd, *, timeout, dry_run, **ssh_kwargs):
        return mock.MagicMock(stdout="should-not-be-logged", stderr="", returncode=0, success=True)

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", fake_run_remote_command)

    with caplog.at_level(logging.ERROR, logger="sparkrun.runtimes._cluster_ops"):
        dump_serve_log("h1", "mycontainer", ssh_kwargs={}, dry_run=True)

    assert not any("should-not-be-logged" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration with run_native_cluster Step 6 failure
# ---------------------------------------------------------------------------


def test_native_cluster_step6_head_serve_failure_triggers_cleanup(monkeypatch):
    """When the head serve exec returns non-zero, cleanup_after_failure runs."""
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    monkeypatch.setattr(_cluster_ops, "detect_ib_with_ips", lambda *a, **k: (ClusterCommEnv.empty(), {}))
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda ctx: "10.0.0.1")
    monkeypatch.setattr(_cluster_ops, "resolve_hosts_for_init", lambda ctx, head_ip: ctx.hosts)
    monkeypatch.setattr(_cluster_ops, "find_port", lambda ctx, host, port: port)
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *a, **k: 0)
    monkeypatch.setattr(_cluster_ops, "run_pre_serve_hooks", lambda *a, **k: None)
    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *a, **k: None)

    # Head serve exec fails.
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_script",
        lambda *a, **k: mock.MagicMock(success=False, returncode=1, stderr="ImportError: libtorch_cuda.so", stdout=""),
    )

    cleanup_calls: list[str] = []

    def fake_cleanup(ctx, executor, **kw):
        cleanup_calls.append(kw.get("reason", ""))

    monkeypatch.setattr(_cluster_ops, "cleanup_after_failure", fake_cleanup)

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

    rc = _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx)

    assert rc == 1
    assert cleanup_calls, "cleanup_after_failure should be invoked on head serve exec failure"
    assert any("head serve exec failed" in reason for reason in cleanup_calls)
