"""Tests for sparkrun.core.hardware_probe — combined accelerator + IB probe."""

from __future__ import annotations

from sparkrun.core.hardware import HostHardware
from sparkrun.core.hardware_probe import (
    _ACCEL_END,
    _ACCEL_START,
    _IB_END,
    _IB_START,
    generate_combined_probe_script,
    probe_host,
    probe_hosts,
    split_probe_output,
)
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# Helpers: build fake combined stdout
# ---------------------------------------------------------------------------


def _accel_section(content: str) -> str:
    return "%s\n%s\n%s" % (_ACCEL_START, content.strip(), _ACCEL_END)


def _ib_section(content: str) -> str:
    return "%s\n%s\n%s" % (_IB_START, content.strip(), _IB_END)


def _combined(accel: str, ib: str) -> str:
    return "%s\n%s" % (_accel_section(accel), _ib_section(ib))


_ACCEL_DGX = (
    "NVIDIA_PRESENT=1\n"
    "NVIDIA_GPU_COUNT=1\n"
    "NVIDIA_GPU_0_NAME=NVIDIA GB10\n"
    "NVIDIA_GPU_0_MEMORY_MIB=131072\n"
    "AMD_PRESENT=0\n"
    "AMD_GPU_COUNT=0\n"
    "INTEL_PRESENT=0\n"
    "INTEL_GAUDI_COUNT=0\n"
    "APPLE_PRESENT=0\n"
    "IB_PRESENT=1\n"
    "OS=Linux\n"
    "ARCH=aarch64\n"
)

_IB_DETECTED = (
    "IB_DETECTED=1\n"
    "DETECTED_GID_INDEX=3\n"
    "DETECTED_HCA_LIST=mlx5_0\n"
    "DETECTED_SOCKET_IFNAME=eth0\n"
    "DETECTED_NET_LIST=ib0\n"
    "DETECTED_UCX_LIST=mlx5_0:1\n"
    "DETECTED_IB_IPS=192.168.1.10\n"
    "DETECTED_MGMT_IP=10.0.0.1\n"
)

_IB_NOT_DETECTED = "IB_DETECTED=0\n"


# ---------------------------------------------------------------------------
# generate_combined_probe_script
# ---------------------------------------------------------------------------


def test_combined_script_is_bash():
    script = generate_combined_probe_script()
    assert script.startswith("#!/bin/bash")


def test_combined_script_contains_both_sentinels():
    script = generate_combined_probe_script()
    assert _ACCEL_START in script
    assert _ACCEL_END in script
    assert _IB_START in script
    assert _IB_END in script


def test_combined_script_has_nvidia_and_ib_detection():
    script = generate_combined_probe_script()
    assert "nvidia-smi" in script
    assert "NVIDIA_GPU_COUNT" in script
    assert "show_gids" in script
    assert "/sys/class/infiniband" in script
    assert "IB_DETECTED" in script


# ---------------------------------------------------------------------------
# split_probe_output
# ---------------------------------------------------------------------------


def test_split_probe_output_both_present():
    stdout = _combined(_ACCEL_DGX, _IB_DETECTED)
    accel, ib = split_probe_output(stdout)
    assert "NVIDIA_GPU_COUNT=1" in accel
    assert "IB_DETECTED=1" in ib


def test_split_probe_output_missing_ib_section():
    stdout = _accel_section(_ACCEL_DGX)  # no IB section
    accel, ib = split_probe_output(stdout)
    assert "NVIDIA_GPU_COUNT=1" in accel
    assert ib == ""


def test_split_probe_output_missing_accel_section():
    stdout = _ib_section(_IB_DETECTED)  # no accel section
    accel, ib = split_probe_output(stdout)
    assert accel == ""
    assert "IB_DETECTED=1" in ib


def test_split_probe_output_empty_string():
    accel, ib = split_probe_output("")
    assert accel == ""
    assert ib == ""


# ---------------------------------------------------------------------------
# probe_host — mocked SSH
# ---------------------------------------------------------------------------


def test_probe_host_parses_combined_output(monkeypatch):
    """probe_host returns HostHardware with both accelerators and ib_info."""
    stdout = _combined(_ACCEL_DGX, _IB_DETECTED)

    def _fake_run(host, script, timeout=None, **kwargs):
        return RemoteResult(host=host, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", _fake_run)

    hw = probe_host("host-a")

    # Accelerator section parsed
    assert len(hw.accelerators) == 1
    assert hw.accelerators[0].vendor == "nvidia"
    assert hw.accelerators[0].model == "gb10"
    assert hw.fingerprint and len(hw.fingerprint) == 16

    # IB section parsed and attached
    assert hw.ib_info is not None
    assert hw.ib_info["IB_DETECTED"] == "1"
    assert hw.ib_info["DETECTED_HCA_LIST"] == "mlx5_0"
    assert hw.ib_info["DETECTED_IB_IPS"] == "192.168.1.10"


def test_probe_host_handles_missing_ib_section(monkeypatch):
    """Only accelerator section present — ib_info is None."""
    stdout = _accel_section(_ACCEL_DGX)  # no IB markers

    def _fake_run(host, script, timeout=None, **kwargs):
        return RemoteResult(host=host, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", _fake_run)

    hw = probe_host("host-b")
    assert hw.accelerators[0].model == "gb10"
    assert hw.ib_info is None


def test_probe_host_handles_missing_accel_section(monkeypatch):
    """Only IB section present — empty HostHardware but ib_info populated."""
    stdout = _ib_section(_IB_DETECTED)  # no accel markers

    def _fake_run(host, script, timeout=None, **kwargs):
        return RemoteResult(host=host, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", _fake_run)

    hw = probe_host("host-c")
    # Accelerator section was missing — HostHardware is the "missing" sentinel
    assert hw.accelerators == []
    assert "accelerator probe section missing" in hw.notes
    # IB still populated
    assert hw.ib_info is not None
    assert hw.ib_info["IB_DETECTED"] == "1"


def test_probe_host_ib_not_detected(monkeypatch):
    """IB section present but IB_DETECTED=0 — ib_info dict present, value is '0'."""
    stdout = _combined(_ACCEL_DGX, _IB_NOT_DETECTED)

    def _fake_run(host, script, timeout=None, **kwargs):
        return RemoteResult(host=host, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", _fake_run)

    hw = probe_host("host-d")
    assert hw.ib_info is not None
    assert hw.ib_info.get("IB_DETECTED") == "0"


def test_probe_host_dry_run():
    """dry_run returns empty HostHardware without SSH."""
    hw = probe_host("unreachable-host", dry_run=True)
    assert isinstance(hw, HostHardware)
    assert hw.accelerators == []
    assert hw.ib_info is None
    assert "dry-run" in hw.notes


def test_probe_host_ssh_failure(monkeypatch):
    """SSH failure returns HostHardware with failure note."""

    def _fake_run(host, script, timeout=None, **kwargs):
        return RemoteResult(host=host, returncode=255, stdout="", stderr="Connection refused")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", _fake_run)

    hw = probe_host("dead-host")
    assert hw.accelerators == []
    assert "hardware probe failed" in hw.notes


# ---------------------------------------------------------------------------
# probe_hosts — mocked parallel SSH
# ---------------------------------------------------------------------------


def test_probe_hosts_parallel(monkeypatch):
    """probe_hosts returns one HostHardware per host."""
    hosts = ["host-1", "host-2", "host-3"]
    stdout = _combined(_ACCEL_DGX, _IB_DETECTED)

    fake_results = [RemoteResult(host=h, returncode=0, stdout=stdout, stderr="") for h in hosts]

    def _fake_parallel(host_list, script, timeout=None, **kwargs):
        return [r for r in fake_results if r.host in host_list]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", _fake_parallel)

    result = probe_hosts(hosts)
    assert set(result.keys()) == set(hosts)
    for host, hw in result.items():
        assert hw.accelerators[0].model == "gb10"
        assert hw.ib_info is not None
        assert hw.ib_info["IB_DETECTED"] == "1"


def test_probe_hosts_dry_run():
    """dry_run returns a placeholder HostHardware per host, no SSH."""
    hosts = ["node-1", "node-2"]
    result = probe_hosts(hosts, dry_run=True)
    assert set(result.keys()) == set(hosts)
    for hw in result.values():
        assert hw.accelerators == []
        assert "dry-run" in hw.notes


def test_probe_hosts_empty_list():
    result = probe_hosts([])
    assert result == {}


def test_probe_hosts_partial_failure(monkeypatch):
    """A failed host gets a failure HostHardware; successful hosts parse normally."""
    hosts = ["good-host", "bad-host"]
    stdout = _combined(_ACCEL_DGX, _IB_DETECTED)

    fake_results = [
        RemoteResult(host="good-host", returncode=0, stdout=stdout, stderr=""),
        RemoteResult(host="bad-host", returncode=255, stdout="", stderr="ssh: no route to host"),
    ]

    def _fake_parallel(host_list, script, timeout=None, **kwargs):
        return fake_results

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", _fake_parallel)

    result = probe_hosts(hosts)
    assert result["good-host"].accelerators[0].model == "gb10"
    assert "hardware probe failed" in result["bad-host"].notes
