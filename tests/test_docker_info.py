"""Tests for sparkrun.orchestration.docker_info (issue #152)."""

from __future__ import annotations

from unittest.mock import patch

from sparkrun.orchestration.docker_info import (
    DockerDriverInfo,
    check_driver_consistency,
    detect_docker_drivers,
    format_driver_warning,
    parse_docker_info_output,
)
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# parse_docker_info_output
# ---------------------------------------------------------------------------


def test_parse_overlay2_classic():
    """overlay2 classic produces no snapshotter flag."""
    out = "overlay2|Backing Filesystem=extfs;Supports d_type=true;Native Overlay Diff=true;"
    info = parse_docker_info_output(out)
    assert info.driver == "overlay2"
    assert info.snapshotter is False
    assert info.signature == "overlay2"


def test_parse_containerd_snapshotter():
    """overlayfs with containerd snapshotter is detected via driver-type."""
    out = "overlayfs|driver-type=io.containerd.snapshotter.v1;"
    info = parse_docker_info_output(out)
    assert info.driver == "overlayfs"
    assert info.snapshotter is True
    assert info.signature == "overlayfs+snapshotter"


def test_parse_empty_returns_unavailable():
    """Empty stdout is reported as UNAVAILABLE rather than crashing."""
    info = parse_docker_info_output("")
    assert info.driver == "UNAVAILABLE"
    assert info.snapshotter is False
    assert info.is_available is False


def test_parse_unavailable_marker():
    """Probe failure path produces UNAVAILABLE."""
    info = parse_docker_info_output("UNAVAILABLE|")
    assert info.driver == "UNAVAILABLE"
    assert info.is_available is False


def test_parse_ignores_trailing_garbage_lines():
    """Only the last non-empty line is parsed (defends against shell noise)."""
    out = "warning: something\noverlay2|key=value;"
    info = parse_docker_info_output(out)
    assert info.driver == "overlay2"


# ---------------------------------------------------------------------------
# check_driver_consistency
# ---------------------------------------------------------------------------


def test_consistency_all_same_driver():
    drivers = {
        "h1": DockerDriverInfo(driver="overlay2", snapshotter=False),
        "h2": DockerDriverInfo(driver="overlay2", snapshotter=False),
    }
    consistent, groups = check_driver_consistency(drivers)
    assert consistent is True
    assert groups == {"overlay2": ["h1", "h2"]}


def test_consistency_overlay2_vs_snapshotter():
    """Issue #152 case: same driver name, different snapshotter setting."""
    drivers = {
        "lenovo": DockerDriverInfo(driver="overlay2", snapshotter=False),
        "msi": DockerDriverInfo(driver="overlayfs", snapshotter=True),
    }
    consistent, groups = check_driver_consistency(drivers)
    assert consistent is False
    assert set(groups) == {"overlay2", "overlayfs+snapshotter"}


def test_consistency_excludes_unavailable():
    """Hosts where docker info failed shouldn't make us flag drift."""
    drivers = {
        "h1": DockerDriverInfo(driver="overlay2", snapshotter=False),
        "h2": DockerDriverInfo(driver="UNAVAILABLE", snapshotter=False),
    }
    consistent, groups = check_driver_consistency(drivers)
    assert consistent is True
    assert set(groups) == {"overlay2", "UNAVAILABLE"}


def test_consistency_empty_map():
    consistent, groups = check_driver_consistency({})
    assert consistent is True
    assert groups == {}


# ---------------------------------------------------------------------------
# format_driver_warning
# ---------------------------------------------------------------------------


def test_format_driver_warning_lists_hosts_and_remediation():
    groups = {
        "overlay2": ["lenovo"],
        "overlayfs+snapshotter": ["msi"],
    }
    text = format_driver_warning(groups)
    assert "inconsistent" in text.lower()
    assert "lenovo: overlay2" in text
    assert "msi: overlayfs+snapshotter" in text
    # remediation instructions present
    assert "/etc/docker/daemon.json" in text
    assert "overlay2" in text
    assert "issue #152" in text


# ---------------------------------------------------------------------------
# detect_docker_drivers
# ---------------------------------------------------------------------------


def _result(host: str, stdout: str = "", returncode: int = 0) -> RemoteResult:
    return RemoteResult(host=host, returncode=returncode, stdout=stdout, stderr="")


def test_detect_docker_drivers_parses_each_host():
    """Each host's stdout is parsed into a DockerDriverInfo entry."""
    results = [
        _result("h1", "overlay2|Backing Filesystem=extfs;"),
        _result("h2", "overlayfs|driver-type=io.containerd.snapshotter.v1;"),
    ]
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=results,
    ):
        info = detect_docker_drivers(["h1", "h2"])

    assert info["h1"].driver == "overlay2"
    assert info["h1"].snapshotter is False
    assert info["h2"].driver == "overlayfs"
    assert info["h2"].snapshotter is True


def test_detect_docker_drivers_failed_host_marked_unavailable():
    """Hosts where the probe returns non-zero get UNAVAILABLE."""
    results = [
        _result("h1", "overlay2|", returncode=0),
        _result("h2", "", returncode=255),
    ]
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=results,
    ):
        info = detect_docker_drivers(["h1", "h2"])

    assert info["h1"].driver == "overlay2"
    assert info["h2"].driver == "UNAVAILABLE"
    assert info["h2"].is_available is False


def test_detect_docker_drivers_empty_hosts():
    """Empty host list short-circuits without any SSH calls."""
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
    ) as mock_run:
        info = detect_docker_drivers([])
    assert info == {}
    mock_run.assert_not_called()


def test_detect_docker_drivers_dry_run_returns_placeholder():
    """Dry-run skips real probes and returns plausible placeholders."""
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[_result("h1", "[dry-run]")],
    ):
        info = detect_docker_drivers(["h1"], dry_run=True)
    assert info["h1"].driver == "overlay2"
    assert info["h1"].snapshotter is False
