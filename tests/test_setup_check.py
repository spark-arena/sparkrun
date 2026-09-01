"""Tests for the 'sparkrun setup check' readiness command."""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.cli._setup._check import (
    FAIL,
    OK,
    SKIP,
    WARN,
    CheckContext,
    HostState,
    evaluate_host,
)
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface, CX7Persistence
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cluster_mgr(tmp_path):
    root = tmp_path / "check_config"
    root.mkdir(parents=True, exist_ok=True)
    return ClusterManager(root)


@pytest.fixture
def patched_cluster_mgr(cluster_mgr):
    with (
        mock.patch("sparkrun.cli._common._get_cluster_manager", return_value=cluster_mgr),
        mock.patch("sparkrun.cli._setup._get_cluster_manager", return_value=cluster_mgr),
    ):
        yield cluster_mgr


# A fully-configured single host: every check passes.
_FACTS_ALL_GOOD = {
    "CHECK_USER": "drew",
    "CHECK_DOCKER_INSTALLED": "1",
    "CHECK_DOCKER_USABLE": "1",
    "CHECK_DOCKER_GROUP": "1",
    "CHECK_GPU_PRESENT": "1",
    "CHECK_NVIDIA_CTK": "1",
    "CHECK_CDI_SPEC": "1",
    "CHECK_EARLYOOM_INSTALLED": "1",
    "CHECK_EARLYOOM_ACTIVE": "1",
    "CHECK_SUDOERS_CHOWN": "1",
    "CHECK_SUDOERS_DROPCACHES": "1",
    "CHECK_MESH_TOTAL": "0",
    "CHECK_MESH_OK": "0",
    "CHECK_COMPLETE": "1",
}


def _facts_kv(facts: dict[str, str]) -> str:
    return "\n".join("%s=%s" % (k, v) for k, v in facts.items()) + "\n"


def _state(facts: dict[str, str], cx7=None, host: str = "10.0.0.1") -> HostState:
    return HostState(host=host, facts=facts, cx7=cx7)


def _cx7_detection(
    *,
    states: list[str],
    ips: list[str],
    netplan: bool = True,
    persistence: CX7Persistence = CX7Persistence.PERSISTENT,
    source: str = "netplan",
    detail: str = "",
    dhcp: bool = False,
) -> CX7HostDetection:
    """Build a CX7HostDetection with one interface per (state, ip) pair."""
    ifaces = [
        CX7Interface(
            name="enp%d" % idx,
            ip=ip,
            prefix=24,
            subnet="192.168.1%d.0/24" % idx if ip else "",
            mtu=9000,
            state=st,
            hca="mlx5_%d" % idx,
            persistence=persistence,
            persistence_source=source if persistence is CX7Persistence.PERSISTENT else "",
            persistence_detail=detail,
            dhcp=dhcp,
        )
        for idx, (st, ip) in enumerate(zip(states, ips, strict=True))
    ]
    return CX7HostDetection(host="h", interfaces=ifaces, netplan_exists=netplan, detected=bool(ifaces))


# ---------------------------------------------------------------------------
# Pure evaluation unit tests
# ---------------------------------------------------------------------------


def _status(items, key):
    return next(i.status for i in items if i.key == key)


def test_evaluate_all_good_single_host():
    ctx = CheckContext(cluster_name="mylab", multi_host=False)
    items = evaluate_host(_state(_FACTS_ALL_GOOD), ctx)
    assert _status(items, "docker_installed") == OK
    assert _status(items, "docker_group") == OK
    assert _status(items, "docker_usable") == OK
    assert _status(items, "nvidia_ctk") == OK
    assert _status(items, "cdi_spec") == OK
    assert _status(items, "earlyoom") == OK
    assert _status(items, "sudoers") == OK
    # SSH mesh / CX7 are multi-host only — omitted for a single host.
    assert all(i.key not in ("ssh_mesh", "cx7") for i in items)


def test_evaluate_missing_cdi_is_critical():
    facts = dict(_FACTS_ALL_GOOD, CHECK_CDI_SPEC="0")
    items = evaluate_host(_state(facts), CheckContext("mylab", False))
    cdi = next(i for i in items if i.key == "cdi_spec")
    assert cdi.status == FAIL
    assert "nvidia-ctk cdi generate" in cdi.guidance
    assert "--cluster mylab" in cdi.guidance


def test_evaluate_nvidia_checks_skipped_without_gpu():
    facts = dict(_FACTS_ALL_GOOD, CHECK_GPU_PRESENT="0", CHECK_NVIDIA_CTK="0", CHECK_CDI_SPEC="0")
    items = evaluate_host(_state(facts), CheckContext(None, False))
    # No GPU → toolkit + CDI are SKIP, not FAIL.
    assert _status(items, "nvidia_ctk") == SKIP
    assert _status(items, "cdi_spec") == SKIP


def test_evaluate_docker_group_and_earlyoom_advisories():
    facts = dict(_FACTS_ALL_GOOD, CHECK_DOCKER_GROUP="0", CHECK_EARLYOOM_ACTIVE="0", CHECK_EARLYOOM_INSTALLED="0")
    items = evaluate_host(_state(facts), CheckContext("mylab", False))
    assert _status(items, "docker_group") == WARN
    assert _status(items, "earlyoom") == WARN
    # docker still usable (group not required if daemon works)
    assert _status(items, "docker_usable") == OK


def test_evaluate_sudoers_unknown_is_skip():
    facts = dict(_FACTS_ALL_GOOD, CHECK_SUDOERS_CHOWN="unknown", CHECK_SUDOERS_DROPCACHES="unknown")
    items = evaluate_host(_state(facts), CheckContext(None, False))
    assert _status(items, "sudoers") == SKIP


def test_evaluate_multi_host_mesh_warns_on_unreachable_peer():
    facts = dict(_FACTS_ALL_GOOD, CHECK_MESH_TOTAL="2", CHECK_MESH_OK="1")
    items = evaluate_host(_state(facts), CheckContext("mylab", True))
    mesh = next(i for i in items if i.key == "ssh_mesh")
    assert mesh.status == WARN
    assert "1/2" in mesh.detail


def test_evaluate_cx7_ok_when_interfaces_up_and_persistent():
    # Two CX7 interfaces up with IPs + a persistent netplan → OK.
    cx7 = _cx7_detection(states=["up", "up"], ips=["192.168.10.1", "192.168.11.1"], netplan=True)
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=cx7), CheckContext("mylab", True))
    assert _status(items, "cx7") == OK


def test_evaluate_cx7_warns_when_interface_down():
    # One interface up, one down → effective-state gap (not a mere file check).
    cx7 = _cx7_detection(states=["up", "down"], ips=["192.168.10.1", ""], netplan=True)
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=cx7), CheckContext("mylab", True))
    cx7_item = next(i for i in items if i.key == "cx7")
    assert cx7_item.status == WARN
    assert "not ready" in cx7_item.detail


def test_evaluate_cx7_warns_when_nothing_persists_the_address():
    # Up with IPs but no config source declares them → won't survive reboot.
    cx7 = _cx7_detection(states=["up"], ips=["192.168.10.1"], netplan=False, persistence=CX7Persistence.EPHEMERAL)
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=cx7), CheckContext("mylab", True))
    cx7_item = next(i for i in items if i.key == "cx7")
    assert cx7_item.status == WARN
    assert "won't survive reboot" in cx7_item.detail


def test_evaluate_cx7_ok_when_persisted_outside_sparkrun_netplan():
    # The regression this check existed to cause: a cluster configured by hand
    # (nmcli, a differently-numbered netplan file, a .network unit) has no
    # 40-cx7.yaml, and was reported as "won't survive reboot" every time.
    cx7 = _cx7_detection(
        states=["up", "up"],
        ips=["192.168.10.1", "192.168.11.1"],
        netplan=False,
        source="networkmanager",
        detail="cx7-a",
    )
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=cx7), CheckContext("mylab", True))
    cx7_item = next(i for i in items if i.key == "cx7")
    assert cx7_item.status == OK
    assert "networkmanager (cx7-a)" in cx7_item.detail


def test_evaluate_cx7_unverifiable_persistence_is_visible_but_not_a_warning():
    # No probe available on the host: "couldn't tell" must not read as "gone".
    cx7 = _cx7_detection(states=["up"], ips=["192.168.10.1"], netplan=False, persistence=CX7Persistence.UNKNOWN)
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=cx7), CheckContext("mylab", True))
    cx7_item = next(i for i in items if i.key == "cx7")
    assert cx7_item.status == OK
    assert "could not verify persistence" in cx7_item.detail


def test_evaluate_cx7_warns_on_dhcp_leased_fabric_address():
    # Persistent, but not pinned — NCCL peer addressing needs stable IPs.
    cx7 = _cx7_detection(states=["up"], ips=["192.168.10.1"], dhcp=True)
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=cx7), CheckContext("mylab", True))
    cx7_item = next(i for i in items if i.key == "cx7")
    assert cx7_item.status == WARN
    assert "DHCP" in cx7_item.detail


def test_evaluate_cx7_skips_without_hardware():
    # No CX7 interfaces detected → SKIP (not every cluster has CX7).
    cx7 = CX7HostDetection(host="h", interfaces=[], detected=False)
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=cx7), CheckContext("mylab", True))
    assert _status(items, "cx7") == SKIP


def test_evaluate_cx7_skips_when_detection_unavailable():
    # Detection couldn't run (None) → SKIP, never a false gap.
    items = evaluate_host(_state(_FACTS_ALL_GOOD, cx7=None), CheckContext("mylab", True))
    assert _status(items, "cx7") == SKIP


def test_evaluate_docker_not_installed_fails():
    facts = dict(_FACTS_ALL_GOOD, CHECK_DOCKER_INSTALLED="0", CHECK_DOCKER_USABLE="0")
    items = evaluate_host(_state(facts), CheckContext(None, False))
    assert _status(items, "docker_installed") == FAIL
    # docker_usable omitted when docker isn't installed.
    assert all(i.key != "docker_usable" for i in items)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_check_help(runner):
    result = runner.invoke(main, ["setup", "check", "--help"])
    assert result.exit_code == 0
    assert "--cluster" in result.output
    assert "--json" in result.output


def test_check_all_good_exits_zero(runner, v, patched_cluster_mgr):
    patched_cluster_mgr.create("mylab", ["10.0.0.1"])
    patched_cluster_mgr.set_default("mylab")

    with mock.patch("sparkrun.orchestration.ssh.run_remote_script") as mock_run:
        mock_run.return_value = RemoteResult("10.0.0.1", 0, _facts_kv(_FACTS_ALL_GOOD), "")
        result = runner.invoke(main, ["setup", "check", "--cluster", "mylab"])

    assert result.exit_code == 0
    assert "No setup gaps found" in result.output


def test_check_reports_critical_gap_exits_one(runner, v, patched_cluster_mgr):
    patched_cluster_mgr.create("mylab", ["10.0.0.1"])

    facts = dict(_FACTS_ALL_GOOD, CHECK_DOCKER_USABLE="0", CHECK_DOCKER_GROUP="0")
    with mock.patch("sparkrun.orchestration.ssh.run_remote_script") as mock_run:
        mock_run.return_value = RemoteResult("10.0.0.1", 0, _facts_kv(facts), "")
        result = runner.invoke(main, ["setup", "check", "--cluster", "mylab"])

    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "critical gap" in result.output
    assert "sparkrun setup docker-group" in result.output


def test_check_missing_cdi_is_not_a_gap_when_cluster_uses_gpus_mode(runner, v, patched_cluster_mgr):
    """A DGX Spark cluster requests GPUs with --gpus, so no CDI spec is needed.

    Hardware defaults to DGX Spark when the cluster carries none (same fallback
    the launcher uses), whose platform tier pins ``gpu_access_mode: gpus``.
    """
    patched_cluster_mgr.create("mylab", ["10.0.0.1"])

    facts = dict(_FACTS_ALL_GOOD, CHECK_CDI_SPEC="0")
    with mock.patch("sparkrun.orchestration.ssh.run_remote_script") as mock_run:
        mock_run.return_value = RemoteResult("10.0.0.1", 0, _facts_kv(facts), "")
        result = runner.invoke(main, ["setup", "check", "--cluster", "mylab"])

    assert result.exit_code == 0
    assert "No setup gaps found" in result.output


def test_check_missing_cdi_is_critical_when_cluster_pins_cdi_mode(runner, v, patched_cluster_mgr):
    """The same host fails the check once the cluster opts back into CDI."""
    patched_cluster_mgr.create("mylab", ["10.0.0.1"], executor_config={"gpu_access_mode": "cdi"})

    facts = dict(_FACTS_ALL_GOOD, CHECK_CDI_SPEC="0")
    with mock.patch("sparkrun.orchestration.ssh.run_remote_script") as mock_run:
        mock_run.return_value = RemoteResult("10.0.0.1", 0, _facts_kv(facts), "")
        result = runner.invoke(main, ["setup", "check", "--cluster", "mylab"])

    assert result.exit_code == 1
    assert "critical gap" in result.output
    assert "nvidia-ctk cdi generate" in result.output


def test_check_unreachable_host_exits_one(runner, v, patched_cluster_mgr):
    patched_cluster_mgr.create("mylab", ["10.0.0.1"])

    with mock.patch("sparkrun.orchestration.ssh.run_remote_script") as mock_run:
        mock_run.return_value = RemoteResult("10.0.0.1", 255, "", "Connection refused")
        result = runner.invoke(main, ["setup", "check", "--cluster", "mylab"])

    assert result.exit_code == 1
    assert "SSH connectivity" in result.output
    assert "unreachable" in result.output


def test_check_multi_host_uses_cx7_detection(runner, v, patched_cluster_mgr):
    patched_cluster_mgr.create("mylab", ["10.0.0.1", "10.0.0.2"])

    facts = dict(_FACTS_ALL_GOOD, CHECK_MESH_TOTAL="1", CHECK_MESH_OK="1")
    good_cx7 = _cx7_detection(states=["up", "up"], ips=["192.168.10.1", "192.168.11.1"], netplan=True)

    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_script") as mock_run,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
    ):
        mock_run.side_effect = lambda host, *a, **k: RemoteResult(host, 0, _facts_kv(facts), "")
        mock_cx7.return_value = {"10.0.0.1": good_cx7, "10.0.0.2": good_cx7}
        result = runner.invoke(main, ["setup", "check", "--cluster", "mylab"])

    assert result.exit_code == 0
    assert mock_cx7.called  # effective CX7 detection was consulted
    assert "CX7 high-speed networking" in result.output
    assert "interface(s) up" in result.output
    assert "No setup gaps found" in result.output


def test_check_json_output(runner, v, patched_cluster_mgr):
    patched_cluster_mgr.create("mylab", ["10.0.0.1"])

    with mock.patch("sparkrun.orchestration.ssh.run_remote_script") as mock_run:
        mock_run.return_value = RemoteResult("10.0.0.1", 0, _facts_kv(_FACTS_ALL_GOOD), "")
        result = runner.invoke(main, ["setup", "check", "--cluster", "mylab", "--json"])

    assert result.exit_code == 0
    import json

    # The human summary contains no braces, so the first '{' starts the JSON.
    start = result.output.index("{")
    payload = json.loads(result.output[start:])
    assert payload["cluster"] == "mylab"
    assert payload["results"]["10.0.0.1"]["reachable"] is True
    assert payload["critical_gaps"] == 0
