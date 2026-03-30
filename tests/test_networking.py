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
    CX7Topology,
    CX7TopologyResult,
    _group_interfaces_by_port,
    _parse_arping_output,
    build_host_detection,
    classify_topology,
    detect_cx7_for_hosts,
    detect_switch,
    generate_cx7_configure_script,
    parse_cx7_detect_output,
    plan_cluster_cx7,
    plan_ring_cx7,
    select_subnets,
    select_subnets_for_topology,
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
CX7_IFACE_0_MAC=aa:bb:cc:dd:ee:01
CX7_IFACE_1_NAME=enP2p1s0f0np0
CX7_IFACE_1_IP=192.168.12.13
CX7_IFACE_1_PREFIX=24
CX7_IFACE_1_SUBNET=192.168.12.0/24
CX7_IFACE_1_MTU=9000
CX7_IFACE_1_STATE=up
CX7_IFACE_1_HCA=roceP2p1s0f0
CX7_IFACE_1_MAC=aa:bb:cc:dd:ee:02
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
        assert det.interfaces[0].mac == "aa:bb:cc:dd:ee:01"
        assert det.interfaces[1].name == "enP2p1s0f0np0"
        assert det.interfaces[1].mac == "aa:bb:cc:dd:ee:02"
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
        with pytest.raises(ValueError, match="at least 2"):
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
    interfaces: list[tuple],
    used_subnets: set[str] | None = None,
    sudo_ok: bool = True,
) -> CX7HostDetection:
    """Helper to build a CX7HostDetection for tests.

    Each interface tuple: (name, ip, subnet, mtu) or (name, ip, subnet, mtu, mac).
    """
    ifaces = []
    for iface_data in interfaces:
        if len(iface_data) == 5:
            name, ip, subnet, mtu, mac = iface_data
        else:
            name, ip, subnet, mtu = iface_data
            mac = ""
        ifaces.append(
            CX7Interface(
                name=name,
                ip=ip,
                prefix=24 if subnet else 0,
                subnet=subnet,
                mtu=mtu,
                state="up",
                hca="roce" + name,
                mac=mac,
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


# ---------------------------------------------------------------------------
# CX7Topology enum
# ---------------------------------------------------------------------------


class TestCX7Topology:
    def test_enum_values(self):
        assert CX7Topology.DIRECT == "direct"
        assert CX7Topology.SWITCH == "switch"
        assert CX7Topology.RING == "ring"
        assert CX7Topology.UNKNOWN == "unknown"

    def test_string_enum(self):
        assert isinstance(CX7Topology.RING, str)


# ---------------------------------------------------------------------------
# MAC parsing in build_host_detection
# ---------------------------------------------------------------------------


class TestMACParsing:
    def test_mac_default_empty(self):
        """When no MAC in detection output, mac defaults to empty string."""
        raw = parse_cx7_detect_output(SAMPLE_UNCONFIGURED_OUTPUT)
        det = build_host_detection("10.24.11.14", raw)
        for iface in det.interfaces:
            assert iface.mac == ""

    def test_mac_parsed(self):
        """MAC addresses are parsed from detection output."""
        raw = parse_cx7_detect_output(SAMPLE_CONFIGURED_OUTPUT)
        det = build_host_detection("10.24.11.13", raw)
        assert det.interfaces[0].mac == "aa:bb:cc:dd:ee:01"
        assert det.interfaces[1].mac == "aa:bb:cc:dd:ee:02"


# ---------------------------------------------------------------------------
# classify_topology
# ---------------------------------------------------------------------------


class TestClassifyTopology:
    def test_single_host_unknown(self):
        assert classify_topology([], ["h1"]) == CX7Topology.UNKNOWN

    def test_two_hosts_always_switch(self):
        """2 hosts always classified as SWITCH (can't distinguish direct vs switch)."""
        links = [("h1", "enp1", "h2", "enp2")]
        assert classify_topology(links, ["h1", "h2"]) == CX7Topology.SWITCH

    def test_two_hosts_no_links_switch(self):
        """With 2 hosts and no link data -> SWITCH."""
        assert classify_topology([], ["h1", "h2"]) == CX7Topology.SWITCH

    def test_three_hosts_ring(self):
        """3 hosts each connected to 2 others -> RING."""
        links = [
            ("h1", "enp1", "h2", "enp2"),
            ("h2", "enp3", "h3", "enp4"),
            ("h3", "enp5", "h1", "enp6"),
        ]
        assert classify_topology(links, ["h1", "h2", "h3"]) == CX7Topology.RING

    def test_three_hosts_incomplete_switch(self):
        """3 hosts but not all pairs connected -> SWITCH."""
        links = [
            ("h1", "enp1", "h2", "enp2"),
            # h3 not connected to both others
        ]
        assert classify_topology(links, ["h1", "h2", "h3"]) == CX7Topology.SWITCH

    def test_four_hosts_switch(self):
        """4+ hosts -> SWITCH."""
        links = [
            ("h1", "e1", "h2", "e2"),
            ("h2", "e3", "h3", "e4"),
            ("h3", "e5", "h4", "e6"),
            ("h4", "e7", "h1", "e8"),
        ]
        assert classify_topology(links, ["h1", "h2", "h3", "h4"]) == CX7Topology.SWITCH


# ---------------------------------------------------------------------------
# detect_switch
# ---------------------------------------------------------------------------


class TestDetectSwitch:
    def test_switch_detected(self):
        from sparkrun.orchestration.ssh import RemoteResult

        det = _make_detection("h1", "10.0.0.1", [
            ("enp1", "", "", 1500, "aa:00:00:00:01:01"),
            ("enp2", "", "", 1500, "aa:00:00:00:01:02"),
        ])
        mock_result = RemoteResult(host="h1", returncode=0, stdout="CX7_SWITCH_DETECTED=1\n", stderr="")

        with mock.patch("sparkrun.orchestration.ssh.run_remote_script", return_value=mock_result):
            result = detect_switch({"h1": det}, ["h1"])

        assert result is True

    def test_no_switch(self):
        from sparkrun.orchestration.ssh import RemoteResult

        det = _make_detection("h1", "10.0.0.1", [
            ("enp1", "", "", 1500, "aa:00:00:00:01:01"),
            ("enp2", "", "", 1500, "aa:00:00:00:01:02"),
        ])
        mock_result = RemoteResult(host="h1", returncode=0, stdout="CX7_SWITCH_DETECTED=0\n", stderr="")

        with mock.patch("sparkrun.orchestration.ssh.run_remote_script", return_value=mock_result):
            result = detect_switch({"h1": det}, ["h1"])

        assert result is False

    def test_tcpdump_unavailable(self):
        from sparkrun.orchestration.ssh import RemoteResult

        det = _make_detection("h1", "10.0.0.1", [
            ("enp1", "", "", 1500, "aa:00:00:00:01:01"),
        ])
        mock_result = RemoteResult(host="h1", returncode=0, stdout="CX7_SWITCH_DETECTED=-1\n", stderr="")

        with mock.patch("sparkrun.orchestration.ssh.run_remote_script", return_value=mock_result):
            result = detect_switch({"h1": det}, ["h1"])

        assert result is None

    def test_dry_run(self):
        det = _make_detection("h1", "10.0.0.1", [
            ("enp1", "", "", 1500, "aa:00:00:00:01:01"),
        ])
        result = detect_switch({"h1": det}, ["h1"], dry_run=True)
        assert result is None

    def test_no_hosts(self):
        result = detect_switch({}, [])
        assert result is None

    def test_no_sudo_skips(self):
        """Without passwordless sudo, switch detection is skipped."""
        det = _make_detection("h1", "10.0.0.1", [
            ("enp1", "", "", 1500, "aa:00:00:00:01:01"),
        ], sudo_ok=False)
        result = detect_switch({"h1": det}, ["h1"])
        assert result is None

    def test_ssh_failure(self):
        from sparkrun.orchestration.ssh import RemoteResult

        det = _make_detection("h1", "10.0.0.1", [
            ("enp1", "", "", 1500, "aa:00:00:00:01:01"),
        ])
        mock_result = RemoteResult(host="h1", returncode=1, stdout="", stderr="connection refused")

        with mock.patch("sparkrun.orchestration.ssh.run_remote_script", return_value=mock_result):
            result = detect_switch({"h1": det}, ["h1"])

        assert result is None


# ---------------------------------------------------------------------------
# _parse_arping_output
# ---------------------------------------------------------------------------


class TestParseArpingOutput:
    def test_empty_output(self):
        assert _parse_arping_output("") == []

    def test_no_neighbors(self):
        result = _parse_arping_output("CX7_NEIGHBOR_COUNT=0\n")
        assert result == []

    def test_two_neighbors(self):
        output = (
            "CX7_NEIGHBOR_0_LOCAL_IFACE=enp1s0f0np0\n"
            "CX7_NEIGHBOR_0_REMOTE_MAC=aa:bb:cc:dd:ee:01\n"
            "CX7_NEIGHBOR_1_LOCAL_IFACE=enP2p1s0f0np0\n"
            "CX7_NEIGHBOR_1_REMOTE_MAC=aa:bb:cc:dd:ee:02\n"
            "CX7_NEIGHBOR_COUNT=2\n"
        )
        result = _parse_arping_output(output)
        assert len(result) == 2
        assert result[0] == ("enp1s0f0np0", "aa:bb:cc:dd:ee:01")
        assert result[1] == ("enP2p1s0f0np0", "aa:bb:cc:dd:ee:02")


# ---------------------------------------------------------------------------
# select_subnets_for_topology
# ---------------------------------------------------------------------------


class TestSelectSubnetsForTopology:
    def test_direct_returns_two(self):
        result = select_subnets_for_topology({}, CX7Topology.DIRECT)
        assert len(result) == 2

    def test_switch_returns_two(self):
        result = select_subnets_for_topology({}, CX7Topology.SWITCH)
        assert len(result) == 2

    def test_ring_returns_six(self):
        result = select_subnets_for_topology({}, CX7Topology.RING)
        assert len(result) == 6
        # All should be unique /24 subnets
        assert len(set(result)) == 6
        for s in result:
            assert s.prefixlen == 24


# ---------------------------------------------------------------------------
# plan_ring_cx7
# ---------------------------------------------------------------------------


class TestPlanRingCX7:
    def _make_ring_detections(self):
        """Build 3 hosts with 4 CX7 interfaces each + MAC addresses."""
        return {
            "h1": _make_detection("h1", "10.0.0.1", [
                ("enp1a", "", "", 1500, "aa:00:00:00:01:01"),
                ("enp1b", "", "", 1500, "aa:00:00:00:01:02"),
                ("enp1c", "", "", 1500, "aa:00:00:00:01:03"),
                ("enp1d", "", "", 1500, "aa:00:00:00:01:04"),
            ]),
            "h2": _make_detection("h2", "10.0.0.2", [
                ("enp2a", "", "", 1500, "aa:00:00:00:02:01"),
                ("enp2b", "", "", 1500, "aa:00:00:00:02:02"),
                ("enp2c", "", "", 1500, "aa:00:00:00:02:03"),
                ("enp2d", "", "", 1500, "aa:00:00:00:02:04"),
            ]),
            "h3": _make_detection("h3", "10.0.0.3", [
                ("enp3a", "", "", 1500, "aa:00:00:00:03:01"),
                ("enp3b", "", "", 1500, "aa:00:00:00:03:02"),
                ("enp3c", "", "", 1500, "aa:00:00:00:03:03"),
                ("enp3d", "", "", 1500, "aa:00:00:00:03:04"),
            ]),
        }

    def _make_ring_topology(self):
        """Build a ring topology result: h1<->h2, h2<->h3, h3<->h1."""
        return CX7TopologyResult(
            topology=CX7Topology.RING,
            links=[
                ("h1", "enp1a", "h2", "enp2b"),
                ("h2", "enp2c", "h3", "enp3b"),
                ("h3", "enp3c", "h1", "enp1d"),
            ],
        )

    def test_ring_plan_basic(self):
        detections = self._make_ring_detections()
        topo = self._make_ring_topology()
        subnets = [ipaddress.IPv4Network("192.168.%d.0/24" % i) for i in range(10, 16)]

        plan = plan_ring_cx7(detections, topo, subnets)

        assert plan.topology == CX7Topology.RING
        assert len(plan.subnets) == 6
        assert len(plan.host_plans) == 3
        # Each host should have assignments
        for hp in plan.host_plans:
            assert len(hp.assignments) >= 2, "%s has %d assignments" % (hp.host, len(hp.assignments))
            assert hp.needs_change is True

    def test_ring_plan_too_few_subnets(self):
        detections = self._make_ring_detections()
        topo = self._make_ring_topology()
        subnets = [ipaddress.IPv4Network("192.168.10.0/24")]

        plan = plan_ring_cx7(detections, topo, subnets)
        assert len(plan.errors) > 0

    def test_ring_plan_idempotent(self):
        """Already-configured ring hosts should have needs_change=False."""
        # IPs match what planner assigns: link0 h1<->h2 on 10/11, link1 h2<->h3 on 12/13, link2 h3<->h1 on 14/15
        # Partners via sorted-pair grouping: (enp1a,enp1b), (enp1c,enp1d), etc.
        detections = {
            "h1": _make_detection("h1", "10.0.0.1", [
                ("enp1a", "192.168.10.1", "192.168.10.0/24", 9000, "aa:00:00:00:01:01"),
                ("enp1b", "192.168.11.1", "192.168.11.0/24", 9000, "aa:00:00:00:01:02"),
                ("enp1c", "192.168.15.2", "192.168.15.0/24", 9000, "aa:00:00:00:01:03"),
                ("enp1d", "192.168.14.2", "192.168.14.0/24", 9000, "aa:00:00:00:01:04"),
            ]),
            "h2": _make_detection("h2", "10.0.0.2", [
                ("enp2a", "192.168.11.2", "192.168.11.0/24", 9000, "aa:00:00:00:02:01"),
                ("enp2b", "192.168.10.2", "192.168.10.0/24", 9000, "aa:00:00:00:02:02"),
                ("enp2c", "192.168.12.1", "192.168.12.0/24", 9000, "aa:00:00:00:02:03"),
                ("enp2d", "192.168.13.1", "192.168.13.0/24", 9000, "aa:00:00:00:02:04"),
            ]),
            "h3": _make_detection("h3", "10.0.0.3", [
                ("enp3a", "192.168.13.2", "192.168.13.0/24", 9000, "aa:00:00:00:03:01"),
                ("enp3b", "192.168.12.2", "192.168.12.0/24", 9000, "aa:00:00:00:03:02"),
                ("enp3c", "192.168.14.1", "192.168.14.0/24", 9000, "aa:00:00:00:03:03"),
                ("enp3d", "192.168.15.1", "192.168.15.0/24", 9000, "aa:00:00:00:03:04"),
            ]),
        }
        topo = CX7TopologyResult(
            topology=CX7Topology.RING,
            links=[
                ("h1", "enp1a", "h2", "enp2b"),
                ("h2", "enp2c", "h3", "enp3b"),
                ("h3", "enp3c", "h1", "enp1d"),
            ],
        )
        subnets = [ipaddress.IPv4Network("192.168.%d.0/24" % i) for i in range(10, 16)]
        plan = plan_ring_cx7(detections, topo, subnets, mtu=9000)

        # All hosts should be valid — no changes needed
        assert plan.all_valid is True
        assert all(not hp.needs_change for hp in plan.host_plans)

    def test_ring_plan_force_overrides_valid(self):
        """--force should reconfigure even when ring is already valid."""
        detections = {
            "h1": _make_detection("h1", "10.0.0.1", [
                ("enp1a", "192.168.10.1", "192.168.10.0/24", 9000, "aa:00:00:00:01:01"),
                ("enp1b", "192.168.11.1", "192.168.11.0/24", 9000, "aa:00:00:00:01:02"),
                ("enp1c", "192.168.15.2", "192.168.15.0/24", 9000, "aa:00:00:00:01:03"),
                ("enp1d", "192.168.14.2", "192.168.14.0/24", 9000, "aa:00:00:00:01:04"),
            ]),
            "h2": _make_detection("h2", "10.0.0.2", [
                ("enp2a", "192.168.11.2", "192.168.11.0/24", 9000, "aa:00:00:00:02:01"),
                ("enp2b", "192.168.10.2", "192.168.10.0/24", 9000, "aa:00:00:00:02:02"),
                ("enp2c", "192.168.12.1", "192.168.12.0/24", 9000, "aa:00:00:00:02:03"),
                ("enp2d", "192.168.13.1", "192.168.13.0/24", 9000, "aa:00:00:00:02:04"),
            ]),
            "h3": _make_detection("h3", "10.0.0.3", [
                ("enp3a", "192.168.13.2", "192.168.13.0/24", 9000, "aa:00:00:00:03:01"),
                ("enp3b", "192.168.12.2", "192.168.12.0/24", 9000, "aa:00:00:00:03:02"),
                ("enp3c", "192.168.14.1", "192.168.14.0/24", 9000, "aa:00:00:00:03:03"),
                ("enp3d", "192.168.15.1", "192.168.15.0/24", 9000, "aa:00:00:00:03:04"),
            ]),
        }
        topo = CX7TopologyResult(
            topology=CX7Topology.RING,
            links=[
                ("h1", "enp1a", "h2", "enp2b"),
                ("h2", "enp2c", "h3", "enp3b"),
                ("h3", "enp3c", "h1", "enp1d"),
            ],
        )
        subnets = [ipaddress.IPv4Network("192.168.%d.0/24" % i) for i in range(10, 16)]
        plan = plan_ring_cx7(detections, topo, subnets, mtu=9000, force=True)

        assert plan.all_valid is False
        assert all(hp.needs_change for hp in plan.host_plans)

    def test_ring_plan_insufficient_ports(self):
        """Hosts with only 1 port (2 interfaces) can't do ring."""
        detections = {
            "h1": _make_detection("h1", "10.0.0.1", [
                ("enp1s0f0np0", "", "", 1500, "aa:00:00:00:01:01"),
                ("enP2p1s0f0np0", "", "", 1500, "aa:00:00:00:01:02"),
            ]),
            "h2": _make_detection("h2", "10.0.0.2", [
                ("enp1s0f0np0", "", "", 1500, "aa:00:00:00:02:01"),
                ("enP2p1s0f0np0", "", "", 1500, "aa:00:00:00:02:02"),
            ]),
            "h3": _make_detection("h3", "10.0.0.3", [
                ("enp1s0f0np0", "", "", 1500, "aa:00:00:00:03:01"),
                ("enP2p1s0f0np0", "", "", 1500, "aa:00:00:00:03:02"),
            ]),
        }
        topo = CX7TopologyResult(topology=CX7Topology.RING, links=[])
        subnets = [ipaddress.IPv4Network("192.168.%d.0/24" % i) for i in range(10, 16)]

        plan = plan_ring_cx7(detections, topo, subnets)
        assert len(plan.errors) > 0
        assert any("2 physical ports" in e for e in plan.errors)

    def test_ring_plan_wrong_host_count(self):
        detections = {
            "h1": _make_detection("h1", "10.0.0.1", [
                ("enp1a", "", "", 1500, "aa:00:00:00:01:01"),
                ("enp1b", "", "", 1500, "aa:00:00:00:01:02"),
            ]),
            "h2": _make_detection("h2", "10.0.0.2", [
                ("enp2a", "", "", 1500, "aa:00:00:00:02:01"),
                ("enp2b", "", "", 1500, "aa:00:00:00:02:02"),
            ]),
        }
        topo = CX7TopologyResult(topology=CX7Topology.RING, links=[])
        subnets = [ipaddress.IPv4Network("192.168.%d.0/24" % i) for i in range(10, 16)]

        plan = plan_ring_cx7(detections, topo, subnets)
        assert len(plan.errors) > 0


# ---------------------------------------------------------------------------
# generate_cx7_configure_script (extended for ring)
# ---------------------------------------------------------------------------


class TestGenerateCX7ConfigureScriptRing:
    def test_four_assignments_generates_dynamic_script(self):
        hp = CX7HostPlan(
            host="h1",
            needs_change=True,
            assignments=[
                CX7InterfaceAssignment("enp1a", "192.168.10.1", "192.168.10.0/24"),
                CX7InterfaceAssignment("enp1b", "192.168.11.1", "192.168.11.0/24"),
                CX7InterfaceAssignment("enp1c", "192.168.14.1", "192.168.14.0/24"),
                CX7InterfaceAssignment("enp1d", "192.168.15.1", "192.168.15.0/24"),
            ],
        )
        script = generate_cx7_configure_script(hp, mtu=9000, prefix_len=24)
        # Dynamic script should have all 4 interfaces in netplan
        assert "enp1a:" in script
        assert "enp1b:" in script
        assert "enp1c:" in script
        assert "enp1d:" in script
        assert "192.168.10.1/24" in script
        assert "192.168.11.1/24" in script
        assert "192.168.14.1/24" in script
        assert "192.168.15.1/24" in script
        assert "mtu: 9000" in script
        assert "CX7_CONFIGURED=1" in script

    def test_two_assignments_uses_template(self):
        """2-assignment plans still use the original template."""
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

    def test_one_assignment_raises(self):
        hp = CX7HostPlan(host="h1", needs_change=True, assignments=[
            CX7InterfaceAssignment("enp1", "192.168.10.1", "192.168.10.0/24"),
        ])
        with pytest.raises(ValueError, match="at least 2"):
            generate_cx7_configure_script(hp, mtu=9000, prefix_len=24)


# ---------------------------------------------------------------------------
# CX7ClusterPlan topology field
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _group_interfaces_by_port
# ---------------------------------------------------------------------------


class TestGroupInterfacesByPort:
    def _iface(self, name):
        return CX7Interface(name=name, ip="", prefix=0, subnet="", mtu=1500, state="up", hca="")

    def test_dgx_spark_names(self):
        """DGX Spark names: npN suffix determines port grouping."""
        ifaces = [
            self._iface("enp1s0f0np0"),
            self._iface("enP2p1s0f0np0"),
            self._iface("enp1s0f1np1"),
            self._iface("enP2p1s0f1np1"),
        ]
        groups = _group_interfaces_by_port(ifaces)
        assert len(groups) == 2
        # Port 0: both np0 interfaces
        port0_names = {i.name for i in groups[0]}
        assert port0_names == {"enp1s0f0np0", "enP2p1s0f0np0"}
        # Port 1: both np1 interfaces
        port1_names = {i.name for i in groups[1]}
        assert port1_names == {"enp1s0f1np1", "enP2p1s0f1np1"}

    def test_two_interfaces_single_group(self):
        """2 or fewer interfaces → single group."""
        ifaces = [self._iface("enp1"), self._iface("enp2")]
        groups = _group_interfaces_by_port(ifaces)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_fallback_no_np_suffix(self):
        """Without npN suffix, falls back to sorted pairs."""
        ifaces = [
            self._iface("enp1a"),
            self._iface("enp1b"),
            self._iface("enp1c"),
            self._iface("enp1d"),
        ]
        groups = _group_interfaces_by_port(ifaces)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2

    def test_mixed_np_and_non_np(self):
        """Interfaces with npN grouped by port; others appended individually."""
        ifaces = [
            self._iface("enp1s0f0np0"),
            self._iface("enP2p1s0f0np0"),
            self._iface("eth0"),
        ]
        groups = _group_interfaces_by_port(ifaces)
        # 1 port group (np0) + 1 ungrouped
        assert len(groups) == 2
        port0_names = {i.name for i in groups[0]}
        assert port0_names == {"enp1s0f0np0", "enP2p1s0f0np0"}
        assert groups[1][0].name == "eth0"


# ---------------------------------------------------------------------------
# Netplan link-local
# ---------------------------------------------------------------------------


class TestNetplanLinkLocal:
    def test_static_script_has_empty_link_local(self):
        """The 2-interface template script uses link-local: []."""
        hp = CX7HostPlan(
            host="h1",
            needs_change=True,
            assignments=[
                CX7InterfaceAssignment("enp1s0f0np0", "192.168.11.13", "192.168.11.0/24"),
                CX7InterfaceAssignment("enP2p1s0f0np0", "192.168.12.13", "192.168.12.0/24"),
            ],
        )
        script = generate_cx7_configure_script(hp, mtu=9000, prefix_len=24)
        assert "link-local: []" in script
        assert "link-local: [ ipv4 ]" not in script

    def test_dynamic_script_has_empty_link_local(self):
        """The dynamic N-interface script uses link-local: []."""
        hp = CX7HostPlan(
            host="h1",
            needs_change=True,
            assignments=[
                CX7InterfaceAssignment("enp1a", "192.168.10.1", "192.168.10.0/24"),
                CX7InterfaceAssignment("enp1b", "192.168.11.1", "192.168.11.0/24"),
                CX7InterfaceAssignment("enp1c", "192.168.14.1", "192.168.14.0/24"),
                CX7InterfaceAssignment("enp1d", "192.168.15.1", "192.168.15.0/24"),
            ],
        )
        script = generate_cx7_configure_script(hp, mtu=9000, prefix_len=24)
        assert "link-local: []" in script
        assert "link-local: [ ipv4 ]" not in script


class TestCX7ClusterPlanTopology:
    def test_default_topology(self):
        from sparkrun.orchestration.networking import CX7ClusterPlan
        plan = CX7ClusterPlan()
        assert plan.topology == CX7Topology.SWITCH
        assert plan.subnets == []

    def test_ring_topology(self):
        from sparkrun.orchestration.networking import CX7ClusterPlan
        subnets = [ipaddress.IPv4Network("192.168.%d.0/24" % i) for i in range(10, 16)]
        plan = CX7ClusterPlan(topology=CX7Topology.RING, subnets=subnets)
        assert plan.topology == CX7Topology.RING
        assert len(plan.subnets) == 6
