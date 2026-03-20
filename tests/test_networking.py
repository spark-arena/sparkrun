"""Tests for CX7 networking detection, planning, and configuration."""

from __future__ import annotations

import ipaddress
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.orchestration.networking import (
    CX7HostDetection,
    CX7HostPlan,
    CX7Interface,
    CX7InterfaceAssignment,
    build_host_detection,
    detect_cx7_for_hosts,
    generate_cx7_configure_script,
    parse_cx7_detect_output,
    plan_cluster_cx7,
    select_subnets,
)


# ---------------------------------------------------------------------------
# Sample detection outputs
# ---------------------------------------------------------------------------

SAMPLE_CONFIGURED_OUTPUT = """\
CX7_DETECTED=1
CX7_MGMT_IP=10.24.11.13
CX7_MGMT_IFACE=enP7s7
CX7_NETPLAN_EXISTS=1
CX7_SUDO_OK=1
CX7_IFACE_COUNT=2
CX7_IFACE_0_NAME=enp1s0f0np0
CX7_IFACE_0_IP=192.168.11.13
CX7_IFACE_0_PREFIX=24
CX7_IFACE_0_SUBNET=192.168.11.0/24
CX7_IFACE_0_MTU=9000
CX7_IFACE_0_STATE=up
CX7_IFACE_0_HCA=rocep1s0f0
CX7_IFACE_1_NAME=enP2p1s0f0np0
CX7_IFACE_1_IP=192.168.12.13
CX7_IFACE_1_PREFIX=24
CX7_IFACE_1_SUBNET=192.168.12.0/24
CX7_IFACE_1_MTU=9000
CX7_IFACE_1_STATE=up
CX7_IFACE_1_HCA=roceP2p1s0f0
CX7_USED_SUBNETS=10.24.11.0/24,172.17.0.0/16
"""

SAMPLE_UNCONFIGURED_OUTPUT = """\
CX7_DETECTED=1
CX7_MGMT_IP=10.24.11.14
CX7_MGMT_IFACE=enP7s7
CX7_NETPLAN_EXISTS=0
CX7_SUDO_OK=1
CX7_IFACE_COUNT=2
CX7_IFACE_0_NAME=enp1s0f0np0
CX7_IFACE_0_IP=
CX7_IFACE_0_PREFIX=
CX7_IFACE_0_SUBNET=
CX7_IFACE_0_MTU=1500
CX7_IFACE_0_STATE=up
CX7_IFACE_0_HCA=rocep1s0f0
CX7_IFACE_1_NAME=enP2p1s0f0np0
CX7_IFACE_1_IP=
CX7_IFACE_1_PREFIX=
CX7_IFACE_1_SUBNET=
CX7_IFACE_1_MTU=1500
CX7_IFACE_1_STATE=up
CX7_IFACE_1_HCA=roceP2p1s0f0
CX7_USED_SUBNETS=10.24.11.0/24,172.17.0.0/16
"""

SAMPLE_NO_CX7_OUTPUT = """\
CX7_DETECTED=0
"""


# ---------------------------------------------------------------------------
# parse_cx7_detect_output
# ---------------------------------------------------------------------------


class TestParseCX7DetectOutput:
    def test_configured_host(self):
        raw = parse_cx7_detect_output(SAMPLE_CONFIGURED_OUTPUT)
        assert raw["CX7_DETECTED"] == "1"
        assert raw["CX7_MGMT_IP"] == "10.24.11.13"
        assert raw["CX7_IFACE_COUNT"] == "2"
        assert raw["CX7_IFACE_0_NAME"] == "enp1s0f0np0"
        assert raw["CX7_IFACE_0_IP"] == "192.168.11.13"
        assert raw["CX7_IFACE_0_MTU"] == "9000"
        assert raw["CX7_IFACE_1_NAME"] == "enP2p1s0f0np0"
        assert raw["CX7_USED_SUBNETS"] == "10.24.11.0/24,172.17.0.0/16"

    def test_no_cx7(self):
        raw = parse_cx7_detect_output(SAMPLE_NO_CX7_OUTPUT)
        assert raw["CX7_DETECTED"] == "0"

    def test_empty_output(self):
        raw = parse_cx7_detect_output("")
        assert raw == {}

    def test_skips_comments(self):
        raw = parse_cx7_detect_output("# comment\nKEY=VALUE\n")
        assert raw == {"KEY": "VALUE"}


# ---------------------------------------------------------------------------
# build_host_detection
# ---------------------------------------------------------------------------


class TestBuildHostDetection:
    def test_configured_host(self):
        raw = parse_cx7_detect_output(SAMPLE_CONFIGURED_OUTPUT)
        det = build_host_detection("10.24.11.13", raw)
        assert det.detected is True
        assert det.mgmt_ip == "10.24.11.13"
        assert det.mgmt_iface == "enP7s7"
        assert det.netplan_exists is True
        assert det.sudo_ok is True
        assert len(det.interfaces) == 2
        assert det.interfaces[0].name == "enp1s0f0np0"
        assert det.interfaces[0].ip == "192.168.11.13"
        assert det.interfaces[0].mtu == 9000
        assert det.interfaces[1].name == "enP2p1s0f0np0"
        assert "10.24.11.0/24" in det.used_subnets
        assert "172.17.0.0/16" in det.used_subnets

    def test_unconfigured_host(self):
        raw = parse_cx7_detect_output(SAMPLE_UNCONFIGURED_OUTPUT)
        det = build_host_detection("10.24.11.14", raw)
        assert det.detected is True
        assert det.mgmt_ip == "10.24.11.14"
        assert len(det.interfaces) == 2
        assert det.interfaces[0].ip == ""
        assert det.interfaces[0].mtu == 1500
        assert det.netplan_exists is False

    def test_no_cx7(self):
        raw = parse_cx7_detect_output(SAMPLE_NO_CX7_OUTPUT)
        det = build_host_detection("10.0.0.1", raw)
        assert det.detected is False
        assert len(det.interfaces) == 0


# ---------------------------------------------------------------------------
# select_subnets
# ---------------------------------------------------------------------------


class TestSelectSubnets:
    def test_override(self):
        s1, s2 = select_subnets({}, override1="192.168.11.0/24", override2="192.168.12.0/24")
        assert s1 == ipaddress.IPv4Network("192.168.11.0/24")
        assert s2 == ipaddress.IPv4Network("192.168.12.0/24")

    def test_preserves_existing_common_subnets(self):
        """When all hosts share the same CX7 subnets, those are selected."""
        det1 = _make_detection(
            "h1",
            "10.0.0.1",
            [
                ("enp1", "192.168.11.1", "192.168.11.0/24", 9000),
                ("enp2", "192.168.12.1", "192.168.12.0/24", 9000),
            ],
        )
        det2 = _make_detection(
            "h2",
            "10.0.0.2",
            [
                ("enp1", "192.168.11.2", "192.168.11.0/24", 9000),
                ("enp2", "192.168.12.2", "192.168.12.0/24", 9000),
            ],
        )
        s1, s2 = select_subnets({"h1": det1, "h2": det2})
        assert s1 == ipaddress.IPv4Network("192.168.11.0/24")
        assert s2 == ipaddress.IPv4Network("192.168.12.0/24")

    def test_fresh_selection_avoids_conflicts(self):
        """When no existing CX7 subnets, pick from RFC 1918 avoiding used subnets."""
        det = _make_detection(
            "h1",
            "10.0.0.1",
            [
                ("enp1", "", "", 1500),
                ("enp2", "", "", 1500),
            ],
            used_subnets={"10.0.0.0/24", "172.17.0.0/16", "192.168.0.0/24"},
        )
        s1, s2 = select_subnets({"h1": det})
        # Should get /24 subnets that don't overlap used
        assert s1.prefixlen == 24
        assert s2.prefixlen == 24
        assert s1 != s2
        used = {
            ipaddress.IPv4Network("10.0.0.0/24"),
            ipaddress.IPv4Network("172.17.0.0/16"),
            ipaddress.IPv4Network("192.168.0.0/24"),
        }
        for u in used:
            assert not s1.overlaps(u)
            assert not s2.overlaps(u)

    def test_no_detections_picks_default(self):
        s1, s2 = select_subnets({})
        assert s1.prefixlen == 24
        assert s2.prefixlen == 24


# ---------------------------------------------------------------------------
# plan_cluster_cx7
# ---------------------------------------------------------------------------


class TestPlanClusterCX7:
    def test_all_valid_no_changes(self):
        det1 = _make_detection(
            "h1",
            "10.0.0.1",
            [
                ("enp1", "192.168.11.1", "192.168.11.0/24", 9000),
                ("enp2", "192.168.12.1", "192.168.12.0/24", 9000),
            ],
        )
        det2 = _make_detection(
            "h2",
            "10.0.0.2",
            [
                ("enp1", "192.168.11.2", "192.168.11.0/24", 9000),
                ("enp2", "192.168.12.2", "192.168.12.0/24", 9000),
            ],
        )
        s1 = ipaddress.IPv4Network("192.168.11.0/24")
        s2 = ipaddress.IPv4Network("192.168.12.0/24")
        plan = plan_cluster_cx7({"h1": det1, "h2": det2}, s1, s2)
        assert plan.all_valid is True
        assert all(not hp.needs_change for hp in plan.host_plans)

    def test_needs_config(self):
        det = _make_detection(
            "h1",
            "10.0.0.13",
            [
                ("enp1", "", "", 1500),
                ("enp2", "", "", 1500),
            ],
        )
        s1 = ipaddress.IPv4Network("192.168.11.0/24")
        s2 = ipaddress.IPv4Network("192.168.12.0/24")
        plan = plan_cluster_cx7({"h1": det}, s1, s2)
        assert plan.all_valid is False
        hp = plan.host_plans[0]
        assert hp.needs_change is True
        assert len(hp.assignments) == 2
        # Last octet should be 13 (from mgmt 10.0.0.13)
        assert hp.assignments[0].ip == "192.168.11.13"
        assert hp.assignments[1].ip == "192.168.12.13"

    def test_force_reconfigures_valid(self):
        det = _make_detection(
            "h1",
            "10.0.0.1",
            [
                ("enp1", "192.168.11.1", "192.168.11.0/24", 9000),
                ("enp2", "192.168.12.1", "192.168.12.0/24", 9000),
            ],
        )
        s1 = ipaddress.IPv4Network("192.168.11.0/24")
        s2 = ipaddress.IPv4Network("192.168.12.0/24")
        plan = plan_cluster_cx7({"h1": det}, s1, s2, force=True)
        assert plan.all_valid is False
        assert plan.host_plans[0].needs_change is True

    def test_octet_conflict_uses_next(self):
        """When two hosts have the same mgmt last octet, second gets incremented."""
        det1 = _make_detection(
            "h1",
            "10.0.0.5",
            [
                ("enp1", "", "", 1500),
                ("enp2", "", "", 1500),
            ],
        )
        det2 = _make_detection(
            "h2",
            "10.0.0.5",
            [
                ("enp1", "", "", 1500),
                ("enp2", "", "", 1500),
            ],
        )
        s1 = ipaddress.IPv4Network("192.168.11.0/24")
        s2 = ipaddress.IPv4Network("192.168.12.0/24")
        plan = plan_cluster_cx7({"h1": det1, "h2": det2}, s1, s2)
        ips_s1 = {hp.assignments[0].ip for hp in plan.host_plans}
        ips_s2 = {hp.assignments[1].ip for hp in plan.host_plans}
        # Both should have different IPs
        assert len(ips_s1) == 2
        assert len(ips_s2) == 2
        assert "192.168.11.5" in ips_s1

    def test_partial_config_mixed(self):
        """One host valid, one needs config."""
        det1 = _make_detection(
            "h1",
            "10.0.0.1",
            [
                ("enp1", "192.168.11.1", "192.168.11.0/24", 9000),
                ("enp2", "192.168.12.1", "192.168.12.0/24", 9000),
            ],
        )
        det2 = _make_detection(
            "h2",
            "10.0.0.2",
            [
                ("enp1", "", "", 1500),
                ("enp2", "", "", 1500),
            ],
        )
        s1 = ipaddress.IPv4Network("192.168.11.0/24")
        s2 = ipaddress.IPv4Network("192.168.12.0/24")
        plan = plan_cluster_cx7({"h1": det1, "h2": det2}, s1, s2)
        assert plan.all_valid is False
        # h1 should be preserved, h2 needs config
        h1_plan = next(hp for hp in plan.host_plans if hp.host == "h1")
        h2_plan = next(hp for hp in plan.host_plans if hp.host == "h2")
        assert h1_plan.needs_change is False
        assert h2_plan.needs_change is True
        assert h2_plan.assignments[0].ip == "192.168.11.2"


# ---------------------------------------------------------------------------
# generate_cx7_configure_script
# ---------------------------------------------------------------------------


class TestGenerateCX7ConfigureScript:
    def test_script_has_correct_values(self):
        hp = CX7HostPlan(
            host="h1",
            needs_change=True,
            assignments=[
                CX7InterfaceAssignment("enp1s0f0np0", "192.168.11.13", "192.168.11.0/24"),
                CX7InterfaceAssignment("enP2p1s0f0np0", "192.168.12.13", "192.168.12.0/24"),
            ],
        )
        script = generate_cx7_configure_script(hp, mtu=9000, prefix_len=24)
        assert 'ADAPTER1="enp1s0f0np0"' in script
        assert 'ADAPTER2="enP2p1s0f0np0"' in script
        assert 'IP1="192.168.11.13"' in script
        assert 'IP2="192.168.12.13"' in script
        assert 'MTU="9000"' in script
        assert 'PREFIX="24"' in script

    def test_raises_on_wrong_assignment_count(self):
        hp = CX7HostPlan(host="h1", needs_change=True, assignments=[])
        with pytest.raises(ValueError, match="Expected 2"):
            generate_cx7_configure_script(hp, mtu=9000, prefix_len=24)


# ---------------------------------------------------------------------------
# detect_cx7_for_hosts (mocked)
# ---------------------------------------------------------------------------


class TestDetectCX7ForHosts:
    def test_parallel_detection(self):
        from sparkrun.orchestration.ssh import RemoteResult

        mock_results = [
            RemoteResult(host="h1", returncode=0, stdout=SAMPLE_CONFIGURED_OUTPUT, stderr=""),
            RemoteResult(host="h2", returncode=0, stdout=SAMPLE_UNCONFIGURED_OUTPUT, stderr=""),
        ]

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=mock_results):
            detections = detect_cx7_for_hosts(["h1", "h2"])

        assert "h1" in detections
        assert "h2" in detections
        assert detections["h1"].detected is True
        assert detections["h1"].mgmt_ip == "10.24.11.13"
        assert detections["h2"].detected is True
        assert detections["h2"].mgmt_ip == "10.24.11.14"

    def test_failed_host(self):
        from sparkrun.orchestration.ssh import RemoteResult

        mock_results = [
            RemoteResult(host="h1", returncode=1, stdout="", stderr="connection refused"),
        ]

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=mock_results):
            detections = detect_cx7_for_hosts(["h1"])

        assert detections["h1"].detected is False

    def test_empty_hosts(self):
        detections = detect_cx7_for_hosts([])
        assert detections == {}


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestSetupCX7CLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help(self, runner):
        from sparkrun.cli import main

        result = runner.invoke(main, ["setup", "cx7", "--help"])
        assert result.exit_code == 0
        assert "--hosts" in result.output
        assert "--dry-run" in result.output
        assert "--force" in result.output
        assert "--mtu" in result.output
        assert "--subnet1" in result.output
        assert "--subnet2" in result.output
        assert "CX7" in result.output

    def test_no_hosts_error(self, runner, v):
        from sparkrun.cli import main

        # Mock resolve_hosts to return empty list (simulates no config/no args)
        with mock.patch("sparkrun.core.hosts.resolve_hosts", return_value=[]):
            result = runner.invoke(main, ["setup", "cx7"])
        assert result.exit_code != 0
        assert "No hosts" in result.output or "no hosts" in result.output.lower()

    def test_subnet_pair_validation(self, runner, v):
        from sparkrun.cli import main

        # --subnet1 without --subnet2 should fail before any SSH calls
        result = runner.invoke(main, ["setup", "cx7", "--hosts", "h1", "--subnet1", "192.168.11.0/24"])
        assert result.exit_code != 0
        assert "together" in result.output.lower() or "subnet" in result.output.lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# discover_host_network_ips
# ---------------------------------------------------------------------------


class TestDiscoverHostNetworkIps:
    def test_finds_ib_ips(self):
        """Mock IB detection, verify IPs collected."""
        from sparkrun.orchestration.ssh import RemoteResult

        ib_output = (
            "IB_DETECTED=1\n"
            "DETECTED_IB_IPS=192.168.11.1,192.168.12.1\n"
        )
        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=ib_output, stderr=""),
        ]

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=mock_results):
            from sparkrun.orchestration.networking import discover_host_network_ips
            result = discover_host_network_ips(["10.0.0.1"])

        assert "10.0.0.1" in result
        assert "192.168.11.1" in result["10.0.0.1"]
        assert "192.168.12.1" in result["10.0.0.1"]

    def test_filters_loopback(self):
        """127.0.0.1 is excluded from discovered IPs."""
        from sparkrun.orchestration.ssh import RemoteResult

        ib_output = (
            "IB_DETECTED=1\n"
            "DETECTED_IB_IPS=127.0.0.1,192.168.11.1\n"
        )
        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=ib_output, stderr=""),
        ]

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=mock_results):
            from sparkrun.orchestration.networking import discover_host_network_ips
            result = discover_host_network_ips(["10.0.0.1"])

        assert "10.0.0.1" in result
        assert "127.0.0.1" not in result["10.0.0.1"]
        assert "192.168.11.1" in result["10.0.0.1"]

    def test_filters_mgmt_ips(self):
        """Management IPs already in host list are excluded."""
        from sparkrun.orchestration.ssh import RemoteResult

        ib_output = (
            "IB_DETECTED=1\n"
            "DETECTED_IB_IPS=10.0.0.1,192.168.11.1\n"
        )
        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=ib_output, stderr=""),
        ]

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=mock_results):
            from sparkrun.orchestration.networking import discover_host_network_ips
            result = discover_host_network_ips(["10.0.0.1"])

        assert "10.0.0.1" in result
        assert result["10.0.0.1"] == ["192.168.11.1"]

    def test_empty_on_failure(self):
        """Detection failure returns empty."""
        from sparkrun.orchestration.ssh import RemoteResult

        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=1, stdout="", stderr="connection refused"),
        ]

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=mock_results):
            from sparkrun.orchestration.networking import discover_host_network_ips
            result = discover_host_network_ips(["10.0.0.1"])

        assert result == {}

    def test_empty_hosts(self):
        from sparkrun.orchestration.networking import discover_host_network_ips
        assert discover_host_network_ips([]) == {}


class TestDistributeHostKeysAlias:
    def test_alias(self):
        from sparkrun.orchestration.networking import distribute_cx7_host_keys, distribute_host_keys
        assert distribute_cx7_host_keys is distribute_host_keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    host: str,
    mgmt_ip: str,
    interfaces: list[tuple[str, str, str, int]],
    used_subnets: set[str] | None = None,
    sudo_ok: bool = True,
) -> CX7HostDetection:
    """Helper to build a CX7HostDetection for tests.

    Each interface tuple: (name, ip, subnet, mtu).
    """
    ifaces = []
    for name, ip, subnet, mtu in interfaces:
        ifaces.append(
            CX7Interface(
                name=name,
                ip=ip,
                prefix=24 if subnet else 0,
                subnet=subnet,
                mtu=mtu,
                state="up",
                hca="roce" + name,
            )
        )
    return CX7HostDetection(
        host=host,
        interfaces=ifaces,
        mgmt_ip=mgmt_ip,
        mgmt_iface="eth0",
        used_subnets=used_subnets or {"10.0.0.0/24"},
        netplan_exists=False,
        sudo_ok=sudo_ok,
        detected=True,
    )
