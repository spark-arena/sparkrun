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
        # Hardware selection is covered in test_setup_plans; these cases
        # exercise the already-approved topology adapter.
        mock.patch("sparkrun.cli._setup._step_runner.require_setup_step_targets"),
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


def _apply_run(apply_results, cluster_mgr):
    dets = {h: _det(h) for h in ("h1", "h2")}
    with (
        mock.patch("sparkrun.orchestration.networking.apply_cx7_plan", return_value=apply_results),
        mock.patch("sparkrun.cli._setup._sudo.ensure_sudo_password", return_value=(None, None)),
        mock.patch("sparkrun.cli._setup._commands._record_setup_phase"),
    ):
        return _run(["--hosts", "h1,h2", "--cluster", "lab", "--force"], dets, cluster_mgr=cluster_mgr)


def test_partial_apply_does_not_save_topology():
    """Issue #311: 1 configured, 1 failed is not a configured fabric."""
    from sparkrun.orchestration.ssh import RemoteResult

    mgr = mock.Mock()
    mgr.get.return_value.fabric_interfaces = None
    r = _apply_run([RemoteResult("h1", 0, "", ""), RemoteResult("h2", 255, "", "Network is unreachable")], mgr)

    assert r.exit_code == 1, r.output
    assert "Topology not saved to cluster 'lab': 1 host(s) failed." in r.output
    assert not any("topology" in c.kwargs for c in mgr.update.call_args_list)
    # Printed once by the CLI (apply_cx7_plan no longer logs it as well).
    assert r.output.count("[FAIL] h2") == 1


def test_full_apply_saves_topology():
    from sparkrun.orchestration.ssh import RemoteResult

    mgr = mock.Mock()
    mgr.get.return_value.fabric_interfaces = None
    r = _apply_run([RemoteResult("h1", 0, "", ""), RemoteResult("h2", 0, "", "")], mgr)

    assert r.exit_code == 0, r.output
    mgr.update.assert_any_call("lab", topology="switch")
