"""Endpoint removal grace in the auto-discover daemon.

A health probe that times out once is not evidence a workload is gone, and
evicting it costs a gateway restart plus a window where clients get a 404 for a
model that is serving fine.
"""

from __future__ import annotations

from dataclasses import dataclass

from sparkrun.proxy.autodiscover import _EndpointRemovalGrace, _endpoint_identity


@dataclass
class _Endpoint:
    host: str = "10.0.0.1"
    port: int = 8000
    cluster_id: str = ""


def test_a_single_miss_does_not_evict_at_the_default_grace():
    grace = _EndpointRemovalGrace(2)
    ep = _Endpoint(cluster_id="sparkrun_abc")

    effective, deferred = grace.reconcile([ep])
    assert effective == [ep]
    assert deferred == 0

    # Probe blipped: still served, and reported as a deferral.
    effective, deferred = grace.reconcile([])
    assert effective == [ep]
    assert deferred == 1

    # Second consecutive miss: now it goes.
    effective, deferred = grace.reconcile([])
    assert effective == []
    assert deferred == 0


def test_a_grace_of_one_restores_remove_on_first_miss():
    """The documented escape hatch for anyone who wants the old behaviour."""
    grace = _EndpointRemovalGrace(1)
    ep = _Endpoint(cluster_id="sparkrun_abc")

    grace.reconcile([ep])
    effective, deferred = grace.reconcile([])
    assert effective == []
    assert deferred == 0


def test_a_reappearance_resets_the_miss_count():
    grace = _EndpointRemovalGrace(3)
    ep = _Endpoint(cluster_id="sparkrun_abc")

    grace.reconcile([ep])
    grace.reconcile([])  # miss 1
    grace.reconcile([ep])  # recovered
    effective, _ = grace.reconcile([])  # miss 1 again, not 2
    assert effective == [ep]
    effective, _ = grace.reconcile([])
    assert effective == [ep]
    effective, _ = grace.reconcile([])
    assert effective == []


def test_identity_prefers_cluster_id_over_address():
    """An address alone is not stable across a relaunch."""
    assert _endpoint_identity(_Endpoint(cluster_id="sparkrun_abc")) == ("cluster", "sparkrun_abc")
    assert _endpoint_identity(_Endpoint(host="10.0.0.2", port=8001)) == ("address", "10.0.0.2", 8001)

    # Same workload, re-homed: still one identity, so no spurious deferral.
    grace = _EndpointRemovalGrace(2)
    grace.reconcile([_Endpoint(host="10.0.0.1", cluster_id="sparkrun_abc")])
    effective, deferred = grace.reconcile([_Endpoint(host="10.0.0.9", cluster_id="sparkrun_abc")])
    assert len(effective) == 1
    assert deferred == 0


def test_duplicate_endpoints_in_one_sweep_are_emitted_once():
    grace = _EndpointRemovalGrace(2)
    ep = _Endpoint(cluster_id="sparkrun_abc")
    effective, _ = grace.reconcile([ep, _Endpoint(cluster_id="sparkrun_abc")])
    assert len(effective) == 1


def test_grace_is_clamped_to_at_least_one():
    assert _EndpointRemovalGrace(0).required_misses == 1
    assert _EndpointRemovalGrace(-5).required_misses == 1


def test_the_daemon_config_names_a_gateway_and_state_dir_not_a_port(tmp_path):
    """The sidecar resolves the engine from the state file, so it needs neither
    the proxy port nor the master key — one fewer secret on disk."""
    from unittest.mock import MagicMock, patch

    import yaml

    from sparkrun.proxy.engine import ProxyEngine

    state_dir = tmp_path / "proxy"
    engine = ProxyEngine(master_key="sk-secret", state_dir=state_dir)

    with patch("subprocess.Popen") as popen:
        popen.return_value = MagicMock(pid=999)
        engine.start_autodiscover(proxy_pid=1, interval=15, removal_grace_sweeps=4)

    cfg = yaml.safe_load((state_dir / "autodiscover.yaml").read_text())
    assert cfg["gateway"] == "litellm"
    assert cfg["state_dir"] == str(state_dir)
    assert cfg["removal_grace_sweeps"] == 4
    assert cfg["interval"] == 15
    assert "master_key" not in cfg
    assert "proxy_port" not in cfg
