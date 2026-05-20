"""Tests that runtimes thread per-host backends through to the collective path (A2)."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.core.backend_select import BackendBundle
from sparkrun.orchestration.collectives import NcclBackend
from sparkrun.orchestration.infiniband import detect_ib_for_hosts


# Canonical NVIDIA + IB detection output (mirrors test_collectives.py).
_IB_DETECTED_RAW = (
    "IB_DETECTED=1\n"
    "DETECTED_HCA_LIST=mlx5_0,mlx5_1\n"
    "DETECTED_NET_LIST=ibp1s0f0,ibp2s0f0\n"
    "DETECTED_SOCKET_IFNAME=enp1s0f0\n"
    "DETECTED_GID_INDEX=3\n"
    "DETECTED_UCX_LIST=mlx5_0:1,mlx5_1:1\n"
    "DETECTED_MGMT_IP=10.0.0.5\n"
    "DETECTED_IB_IPS=10.1.0.5\n"
)


def _fake_result(host: str):
    r = mock.MagicMock()
    r.host = host
    r.success = True
    r.stdout = _IB_DETECTED_RAW
    return r


# ---------------------------------------------------------------------------
# detect_ib_for_hosts: backends path vs legacy path are byte-identical for NVIDIA
# ---------------------------------------------------------------------------


def test_detect_ib_for_hosts_backends_path_matches_legacy_for_nvidia():
    """When backends maps a host to NcclBackend, the emitted env is identical
    to the legacy ``generate_nccl_env`` path used when backends=None."""
    hosts = ["h1"]

    with mock.patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[_fake_result("h1")],
    ):
        legacy = detect_ib_for_hosts(hosts, dry_run=False)
        backends = {"h1": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend())}
        via_backend = detect_ib_for_hosts(hosts, dry_run=False, backends=backends)

    assert via_backend.comm_env.shared == legacy.comm_env.shared
    # per_host should be empty for single-host (everything is shared)
    assert via_backend.comm_env.per_host == legacy.comm_env.per_host


def test_detect_ib_for_hosts_missing_host_in_backends_uses_legacy():
    """Hosts absent from the backends map still get the legacy NCCL env."""
    hosts = ["h1", "h2"]

    with mock.patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[_fake_result("h1"), _fake_result("h2")],
    ):
        # Only h1 has a backend; h2 falls through to generate_nccl_env.
        backends = {"h1": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend())}
        result = detect_ib_for_hosts(hosts, dry_run=False, backends=backends)

    # Both hosts produce non-empty env; they should be identical and thus shared.
    assert result.comm_env.shared  # populated
    # No host-specific divergence (both NCCL paths produce same dict for same input)
    assert result.comm_env.per_host == {}


# ---------------------------------------------------------------------------
# Runtime kwargs forwarding: backends reaches _cluster_ops
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_resolve_comm_env(monkeypatch):
    """Patch resolve_comm_env to capture the backends kwarg."""
    captured = {}

    def _fake_resolve_comm_env(ctx, comm_env, backends=None):
        captured["backends"] = backends
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        return ClusterCommEnv.empty()

    monkeypatch.setattr(
        "sparkrun.runtimes._cluster_ops.resolve_comm_env",
        _fake_resolve_comm_env,
    )
    return captured


def test_vllm_ray_routes_backends_through_resolve_comm_env(captured_resolve_comm_env, monkeypatch):
    """vllm_ray._run_cluster uses resolve_comm_env when backends is provided."""
    from sparkrun.runtimes.vllm_ray import VllmRayRuntime

    # Stub out everything except the early IB-resolution step.
    rt = VllmRayRuntime()
    backends = {"h1": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend())}

    # Patch the rest of the cluster path so we only exercise step 2.
    monkeypatch.setattr(
        "sparkrun.runtimes._cluster_ops.cleanup_named_containers",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "sparkrun.runtimes._cluster_ops.find_port",
        lambda ctx, host, port: port,
    )
    monkeypatch.setattr(
        "sparkrun.runtimes._cluster_ops.ClusterContext.build",
        classmethod(
            lambda cls, runtime, hosts, image, cluster_id, env, cache_dir, config, dry_run, **kw: cls(
                hosts=hosts,
                head_host=hosts[0],
                worker_hosts=hosts[1:],
                num_nodes=len(hosts),
                ssh_kwargs={},
                volumes={},
                all_env={},
                cluster_id=cluster_id,
                image=image,
                dry_run=dry_run,
                config=config,
            )
        ),
    )
    # Force early exit at the dry-run head-launch step
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_script",
        lambda *a, **kw: mock.MagicMock(success=False, stdout="", stderr="abort"),
    )

    rt._run_cluster(
        hosts=["h1", "h2"],
        image="img",
        serve_command="serve",
        cluster_id="cid",
        env={},
        dry_run=True,
        comm_env=None,
        backends=backends,
    )

    assert captured_resolve_comm_env["backends"] is backends


def test_trtllm_routes_backends_through_resolve_comm_env(captured_resolve_comm_env, monkeypatch):
    """trtllm._run_cluster uses resolve_comm_env when backends is provided."""
    from sparkrun.runtimes.trtllm import TrtllmRuntime

    rt = TrtllmRuntime()
    backends = {"h1": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend())}

    # Same cluster-context shim used above.
    monkeypatch.setattr(
        "sparkrun.runtimes._cluster_ops.cleanup_ranked_containers",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "sparkrun.runtimes._cluster_ops.ClusterContext.build",
        classmethod(
            lambda cls, runtime, hosts, image, cluster_id, env, cache_dir, config, dry_run, **kw: cls(
                hosts=hosts,
                head_host=hosts[0],
                worker_hosts=hosts[1:],
                num_nodes=len(hosts),
                ssh_kwargs={},
                volumes={},
                all_env={},
                cluster_id=cluster_id,
                image=image,
                dry_run=dry_run,
                config=config,
            )
        ),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.detect_host_ip",
        lambda *a, **kw: "10.0.0.1",
    )
    # Abort early after the IB step by failing container launch
    monkeypatch.setattr(
        "sparkrun.runtimes._cluster_ops.launch_containers_parallel",
        lambda *a, **kw: 1,
    )

    rt._run_cluster(
        hosts=["h1", "h2"],
        image="img",
        serve_command="serve",
        cluster_id="cid",
        env={},
        dry_run=True,
        comm_env=None,
        backends=backends,
    )

    assert captured_resolve_comm_env["backends"] is backends


# ---------------------------------------------------------------------------
# Native-cluster runtimes: backends flow through **kwargs to detect_ib_with_ips
# ---------------------------------------------------------------------------


def test_native_cluster_threads_backends_to_detect_ib_with_ips(monkeypatch):
    """run_native_cluster forwards backends to detect_ib_with_ips."""
    from sparkrun.runtimes import _cluster_ops

    captured = {}

    def _fake_detect(ctx, comm_env, ib_ip_map, backends=None):
        captured["backends"] = backends
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        return ClusterCommEnv.empty(), {}

    monkeypatch.setattr(_cluster_ops, "detect_ib_with_ips", _fake_detect)
    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *a, **k: None)
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda ctx: "10.0.0.1")
    monkeypatch.setattr(_cluster_ops, "resolve_hosts_for_init", lambda ctx, head_ip: ctx.hosts)
    monkeypatch.setattr(_cluster_ops, "find_port", lambda ctx, host, port: port)
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *a, **k: 1)

    runtime = mock.MagicMock()
    runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    runtime.generate_node_command = mock.MagicMock(return_value="serve")
    runtime.get_extra_docker_opts = lambda: []
    runtime._print_cluster_banner = mock.MagicMock()

    ctx = _cluster_ops.ClusterContext(
        hosts=["h1", "h2"],
        head_host="h1",
        worker_hosts=["h2"],
        num_nodes=2,
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="cid",
        image="img",
        dry_run=True,
        config=None,
    )
    backends = {"h1": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend())}
    _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx, backends=backends)

    assert captured["backends"] is backends


# ---------------------------------------------------------------------------
# Deprecation warning on resolve_ib_env
# ---------------------------------------------------------------------------


def test_resolve_ib_env_emits_deprecation_warning(monkeypatch):
    """resolve_ib_env raises a DeprecationWarning pointing at the replacement."""
    from sparkrun.runtimes import _cluster_ops
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    # Patch to avoid actually probing hosts.
    monkeypatch.setattr(
        "sparkrun.orchestration.infiniband.detect_ib_for_hosts",
        lambda *a, **kw: mock.MagicMock(comm_env=ClusterCommEnv.empty()),
    )

    ctx = _cluster_ops.ClusterContext(
        hosts=["h1"],
        head_host="h1",
        worker_hosts=[],
        num_nodes=1,
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="cid",
        image="img",
        dry_run=True,
        config=None,
    )

    with pytest.warns(DeprecationWarning, match="resolve_ib_env is deprecated"):
        _cluster_ops.resolve_ib_env(ctx, None)
