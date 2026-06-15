"""CLI-level tests for `sparkrun setup cx7` interface selection (issue #203).

These exercise the `--interfaces` / `--port` flags, mutual exclusion, the
cross-host `--port` validation, and the saved-cluster `fabric_interfaces`
fallback.  SSH/detection are mocked — no hosts are required.
"""

from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from sparkrun.cli._setup._commands import setup_cx7
from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface

_FOUR_PORT = ("enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0", "enP2p1s0f1np1")


def _det(host: str, names=_FOUR_PORT) -> CX7HostDetection:
    ifaces = [CX7Interface(name=n, ip="", prefix=0, subnet="", mtu=9000, state="up", hca="roce") for n in names]
    return CX7HostDetection(
        host=host,
        interfaces=ifaces,
        mgmt_ip=host,
        mgmt_iface="eth0",
        used_subnets={"10.0.0.0/24"},
        netplan_exists=False,
        sudo_ok=True,
        detected=True,
    )


def _run(args, detections, cluster_mgr=None):
    # When no explicit cluster manager is supplied, stub one that reports no
    # default cluster so resolution can't read real on-disk cluster state.
    if cluster_mgr is None:
        cluster_mgr = mock.Mock()
        cluster_mgr.get_default.return_value = None
    with (
        mock.patch("sparkrun.cli._setup._commands._resolve_setup_context", return_value=(list(detections), "me", {})),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=detections),
        mock.patch("sparkrun.cli._setup._commands._get_cluster_manager", return_value=cluster_mgr),
    ):
        return CliRunner().invoke(setup_cx7, args)


def _assigned(output: str) -> list[str]:
    return [line.split()[0] for line in output.splitlines() if "->" in line]


def test_interfaces_glob_pins_pair():
    dets = {"h1": _det("h1"), "h2": _det("h2")}
    r = _run(["--hosts", "h1,h2", "--interfaces", "*np1", "--dry-run"], dets)
    assert r.exit_code == 0, r.output
    assert "Interface filter: *np1" in r.output
    assert _assigned(r.output) == ["enp1s0f1np1", "enP2p1s0f1np1", "enp1s0f1np1", "enP2p1s0f1np1"]


def test_port_index_equivalent_to_glob():
    dets = {"h1": _det("h1")}
    r = _run(["--hosts", "h1", "--port", "1", "--dry-run"], dets)
    assert r.exit_code == 0, r.output
    assert _assigned(r.output) == ["enp1s0f1np1", "enP2p1s0f1np1"]


def test_default_no_flag_never_mixes_ports():
    dets = {"h1": _det("h1")}
    r = _run(["--hosts", "h1", "--dry-run"], dets)
    assert r.exit_code == 0, r.output
    # Same physical port across both NICs (np0), not a mixed np0+np1 pair.
    assert _assigned(r.output) == ["enp1s0f0np0", "enP2p1s0f0np0"]


def test_interfaces_and_port_mutually_exclusive():
    dets = {"h1": _det("h1")}
    r = _run(["--hosts", "h1", "--interfaces", "*np1", "--port", "1", "--dry-run"], dets)
    assert r.exit_code == 1
    assert "cannot be used together" in r.output


def test_port_validates_across_heterogeneous_hosts():
    """--port derives a glob from one host; a host lacking that port fails fast."""
    # h2 only has np0 interfaces, so the np1 glob from --port 1 selects <2 there.
    dets = {
        "h1": _det("h1"),
        "h2": _det("h2", names=("enp1s0f0np0", "enP2p1s0f0np0")),
    }
    r = _run(["--hosts", "h1,h2", "--port", "1", "--dry-run"], dets)
    assert r.exit_code == 1, r.output
    assert "does not select 2 interfaces on: h2" in r.output


def test_saved_fabric_interfaces_used_when_no_flag():
    """With no flag, a default cluster's saved fabric_interfaces is applied."""
    dets = {"h1": _det("h1")}

    saved = mock.Mock()
    saved.fabric_interfaces = ["*np1"]
    mgr = mock.Mock()
    mgr.get_default.return_value = "two"
    mgr.get.return_value = saved

    # No --cluster given; resolution falls back to the default cluster's filter.
    r = _run(["--hosts", "h1", "--dry-run"], dets, cluster_mgr=mgr)
    assert r.exit_code == 0, r.output
    assert "Interface filter: *np1" in r.output
    assert _assigned(r.output) == ["enp1s0f1np1", "enP2p1s0f1np1"]
