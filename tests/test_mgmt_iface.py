"""Management-interface detection (``scripts/_mgmt_iface.sh``) — issue #275.

Air-gapped DGX Sparks have no default route, and every probe used to fall back
to a hardcoded ``eth0`` that does not exist on the hardware.  The bogus name
reached ``GLOO_SOCKET_IFNAME`` and aborted the launch with "Unable to find
address for: eth0".

The shell half is exercised for real: the helper is run under ``bash`` against
a fixture sysfs tree (``SPARKRUN_NET_SYSFS``) with a stub ``ip`` on ``PATH``,
because the whole defect lived in shell fallback semantics that no amount of
Python mocking would have caught.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess

import pytest

from sparkrun.core.hardware_probe import generate_combined_probe_script
from sparkrun.orchestration.infiniband import generate_ib_detect_script, generate_nccl_env
from sparkrun.orchestration.networking import generate_cx7_detect_script
from sparkrun.orchestration.scripts import generate_ip_detect_script
from sparkrun.scripts import inject_mgmt_vars, inject_shell_vars, mgmt_peer_list, read_script

# Scripts that must carry the shared helper rather than their own copy of the
# old one-liner.  ray_head/ray_worker are templates, so they are checked from
# the raw include expansion rather than through a generator.
_INCLUDING_SCRIPTS = (
    "ib_detect.sh",
    "cx7_detect.sh",
    "ip_detect.sh",
    "ray_head.sh",
    "ray_worker.sh",
    "spark_diagnose.sh",
)

# The reporter's cluster: management on enP7s7, two CX7 ports on their own
# subnets, no default route.  docker0 and lo are present because both carry a
# non-loopback global IPv4 on a real Spark and are exactly what a naive
# "first interface with an address" scan would pick.
_FIXTURE_IFACES = {
    # name: (has_device, is_rdma, operstate, ipv4 or None)
    "enP7s7": (True, False, "up", "10.10.10.126"),
    "enp1s0f0np0": (True, True, "up", "10.20.3.3"),
    "enP2p1s0f0np0": (True, True, "up", "10.20.4.3"),
    "enp1s0f1np1": (True, True, "down", None),
    "wlP9s9": (True, False, "down", None),
    "docker0": (False, False, "up", "172.17.0.1"),
    "br-06e9b3b31f13": (False, False, "up", "172.22.0.1"),
    "veth03f5832": (False, False, "up", None),
    "lo": (False, False, "unknown", "127.0.0.1"),
    "tailscale0": (False, False, "unknown", "100.124.41.47"),
}

_STUB_IP = """#!/bin/bash
# Stub `ip` for the fixture host.  `route get` fails the way an air-gapped
# host's does unless SPARKRUN_TEST_DEFAULT_ROUTE names an interface, or the
# routing fixture (SPARKRUN_TEST_ROUTES) answers for that address.
# `route show ... to match <addr>` prints the fixture's routes for <addr>.
case " $* " in
    *" route show "*)
        for target in "$@"; do :; done
        cat "${SPARKRUN_TEST_ROUTES:-/nonexistent}/match-$target" 2>/dev/null
        exit 0
        ;;
esac
if [ "$1 $2" = "route get" ] && [ -f "${SPARKRUN_TEST_ROUTES:-/nonexistent}/get-$3" ]; then
    cat "$SPARKRUN_TEST_ROUTES/get-$3"
    exit 0
fi
if [ "$1 $2" = "route get" ]; then
    if [ -n "${SPARKRUN_TEST_DEFAULT_ROUTE:-}" ]; then
        echo "8.8.8.8 via 10.10.10.1 dev ${SPARKRUN_TEST_DEFAULT_ROUTE} src 10.10.10.126"
        exit 0
    fi
    echo "RTNETLINK answers: Network is unreachable" >&2
    exit 2
fi
dev=""
prev=""
for a in "$@"; do
    [ "$prev" = "dev" ] && dev="$a"
    prev="$a"
done
# An address file may list several addresses, one per line.
if [ -z "$dev" ]; then
    # `ip -4 -o addr show` with no device: every address on the host.
    for f in "$SPARKRUN_TEST_ADDRS"/*; do
        [ -f "$f" ] || continue
        while read -r addr || [ -n "$addr" ]; do
            [ -n "$addr" ] && echo "9: $(basename "$f")    inet $addr/24 scope global $(basename "$f")"
        done < "$f"
    done
    exit 0
fi
[ -f "$SPARKRUN_TEST_ADDRS/$dev" ] || exit 0
while read -r addr || [ -n "$addr" ]; do
    [ -n "$addr" ] && echo "9: $dev    inet $addr/24 scope global $dev"
done < "$SPARKRUN_TEST_ADDRS/$dev"
"""


@pytest.fixture
def net_fixture(tmp_path):
    """Build a fixture sysfs tree + stub ``ip``; return (env, runner)."""
    netdir = tmp_path / "net"
    addrs = tmp_path / "addrs"
    bindir = tmp_path / "bin"
    for d in (netdir, addrs, bindir):
        d.mkdir(parents=True)

    for name, (has_device, is_rdma, operstate, ipv4) in _FIXTURE_IFACES.items():
        iface = netdir / name
        iface.mkdir()
        (iface / "operstate").write_text(operstate + "\n")
        if has_device:
            (iface / "device").mkdir()
            if is_rdma:
                (iface / "device" / "infiniband").mkdir()
        if ipv4:
            (addrs / name).write_text(ipv4)

    ip_stub = bindir / "ip"
    ip_stub.write_text(_STUB_IP)
    ip_stub.chmod(ip_stub.stat().st_mode | stat.S_IXUSR)

    script = tmp_path / "probe.sh"

    def run(*, excludes: str = "", pinned: str | None = None, **env_extra) -> subprocess.CompletedProcess:
        body = "set -uo pipefail\n" + read_script("_mgmt_iface.sh")
        if pinned is not None:
            body = inject_shell_vars(body, SPARKRUN_MGMT_IFACE=pinned)
        script.write_text(body + '\nsparkrun_mgmt_iface "$1"\n')
        env = {
            "PATH": "%s:%s" % (bindir, os.environ.get("PATH", "/usr/bin:/bin")),
            "SPARKRUN_NET_SYSFS": str(netdir),
            "SPARKRUN_TEST_ADDRS": str(addrs),
            **env_extra,
        }
        return subprocess.run(["bash", str(script), excludes], capture_output=True, text=True, env=env, timeout=30)

    return run


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------


@needs_bash
def test_air_gapped_host_resolves_management_interface(net_fixture):
    """The bug: no default route must not yield a phantom interface."""
    result = net_fixture()
    assert result.stdout.strip() == "enP7s7"
    assert "eth0" not in result.stdout


@needs_bash
def test_default_route_is_the_fallback_without_peers_or_ssh(net_fixture):
    """With nothing to measure against, the default route still answers."""
    result = net_fixture(SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9")
    # wlP9s9 is down and has no address — the default route still wins here,
    # because no peer list or SSH session contradicts it.
    assert result.stdout.strip() == "wlP9s9"


@needs_bash
def test_ssh_arrival_outranks_the_default_route(net_fixture):
    """The interface sparkrun reached the host on beats where internet goes."""
    result = net_fixture(SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9", SSH_CONNECTION="10.10.10.9 52344 10.10.10.126 22")
    assert result.stdout.strip() == "enP7s7"


@needs_bash
def test_virtual_ssh_arrival_does_not_outrank_the_default_route(net_fixture):
    """A control machine on a VPN must not move cluster traffic onto it."""
    result = net_fixture(SPARKRUN_TEST_DEFAULT_ROUTE="enP7s7", SSH_CONNECTION="100.124.41.9 52344 100.124.41.47 22")
    assert result.stdout.strip() == "enP7s7"


@needs_bash
def test_virtual_ssh_arrival_beats_the_scan(net_fixture):
    """With no default route, a VPN the control machine uses is real evidence."""
    result = net_fixture(SSH_CONNECTION="100.124.41.9 52344 100.124.41.47 22")
    assert result.stdout.strip() == "tailscale0"


@needs_bash
def test_default_route_naming_absent_interface_is_rejected(net_fixture):
    """A route naming an interface we can't see is not passed through."""
    result = net_fixture(SPARKRUN_TEST_DEFAULT_ROUTE="eth0")
    assert result.stdout.strip() == "enP7s7"


@needs_bash
def test_ssh_connection_identifies_the_management_interface(net_fixture):
    """Field 3 of SSH_CONNECTION is this host's address on the mgmt network."""
    result = net_fixture(SSH_CONNECTION="10.10.10.9 52344 10.10.10.126 22")
    assert result.stdout.strip() == "enP7s7"


@needs_bash
def test_ssh_connection_over_fabric_selects_that_interface(net_fixture):
    """If we genuinely arrived over CX7, that *is* the control path."""
    result = net_fixture(SSH_CONNECTION="10.20.3.9 52344 10.20.3.3 22")
    assert result.stdout.strip() == "enp1s0f0np0"


@needs_bash
def test_ipv6_ssh_connection_falls_through(net_fixture):
    """An IPv6 session finds no IPv4 match and degrades to the scan."""
    result = net_fixture(SSH_CONNECTION="fd00::9 52344 fd00::126 22")
    assert result.stdout.strip() == "enP7s7"


@needs_bash
def test_virtual_interfaces_are_never_selected(net_fixture):
    """docker0/br-*/veth*/tailscale0/lo hold addresses but are not NICs.

    Several sort ahead of enP7s7, so this is a real hazard, not a hypothetical.
    """
    result = net_fixture()
    assert result.stdout.strip() not in {"docker0", "br-06e9b3b31f13", "veth03f5832", "lo", "tailscale0"}


@needs_bash
def test_rdma_backed_nic_is_never_selected_by_the_scan(net_fixture):
    """The fabric is not the management network.

    enp1s0f0np0 sorts before enP7s7 under C collation, so without the
    device/infiniband test the scan would pick a CX7 port.
    """
    result = net_fixture(excludes="")
    assert result.stdout.strip() == "enP7s7"


@needs_bash
def test_nothing_resolvable_prints_nothing(net_fixture):
    """Empty is the honest answer; callers degrade, a guess would crash."""
    result = net_fixture(excludes="enP7s7")
    assert result.stdout.strip() == ""
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Peer coverage: the interface that reaches the cluster's other hosts
# ---------------------------------------------------------------------------

# name: (has_device, is_rdma, is_wireless, operstate, ipv4 or None)
_WIRED = (True, False, False, "up")
_WIFI = (True, False, True, "up")
_FABRIC = (True, True, False, "up")
_VPN = (False, False, False, "unknown")


@pytest.fixture
def scenario(tmp_path):
    """Build a fixture host from an interface spec and a routing table.

    ``routes`` maps a peer address to the lines ``ip route show table all to
    match <peer>`` prints; ``kernel`` maps a peer to the device ``ip route get``
    names for it.
    """

    def run(ifaces, *, routes=None, kernel=None, peers="", pinned=None, call='sparkrun_mgmt_iface ""', **env_extra):
        root = tmp_path / ("s%d" % len(list(tmp_path.iterdir())))
        netdir, addrs, rdir, bindir = root / "net", root / "addrs", root / "routes", root / "bin"
        for d in (netdir, addrs, rdir, bindir):
            d.mkdir(parents=True)
        for name, (has_device, is_rdma, is_wireless, operstate, ipv4) in ifaces.items():
            iface = netdir / name
            iface.mkdir()
            (iface / "operstate").write_text(operstate + "\n")
            if has_device:
                (iface / "device").mkdir()
                if is_rdma:
                    (iface / "device" / "infiniband").mkdir()
            if is_wireless:
                (iface / "wireless").mkdir()
            if ipv4:
                (addrs / name).write_text(ipv4)
        for peer, lines in (routes or {}).items():
            (rdir / ("match-" + peer)).write_text("\n".join(lines) + "\n")
        for peer, dev in (kernel or {}).items():
            (rdir / ("get-" + peer)).write_text("%s dev %s src 0.0.0.0\n" % (peer, dev))
        stub = bindir / "ip"
        stub.write_text(_STUB_IP)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

        body = inject_mgmt_vars("set -uo pipefail\n" + read_script("_mgmt_iface.sh"), mgmt_interface=pinned, peers=peers.split(","))
        script = root / "probe.sh"
        script.write_text(body + "\n" + call + "\n")
        env = {
            "PATH": "%s:%s" % (bindir, os.environ.get("PATH", "/usr/bin:/bin")),
            "SPARKRUN_NET_SYSFS": str(netdir),
            "SPARKRUN_TEST_ADDRS": str(addrs),
            "SPARKRUN_TEST_ROUTES": str(rdir),
            **env_extra,
        }
        result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=30)
        return result.stdout.strip()

    return run


def _link(subnet, dev, src):
    return "%s dev %s proto kernel scope link src %s" % (subnet, dev, src)


def _default(gw, dev, metric=100):
    return "default via %s dev %s proto dhcp metric %d" % (gw, dev, metric)


# The reported case: wired reaches the other Sparks but has no gateway; WiFi
# holds the only default route.
_REPORTED = {
    "enP7s7": (*_WIRED, "10.24.68.102"),
    "wlP9s9": (*_WIFI, "192.168.1.50"),
}
_REPORTED_ROUTES = {
    peer: [_default("192.168.1.1", "wlP9s9", 600), _link("10.24.68.0/24", "enP7s7", "10.24.68.102")]
    for peer in ("10.24.68.103", "10.24.68.104")
}


@needs_bash
def test_wired_link_to_peers_beats_wifi_default_route(scenario):
    """The reported bug: WiFi used to win just by holding the default route."""
    picked = scenario(
        _REPORTED,
        routes=_REPORTED_ROUTES,
        peers="10.24.68.102,10.24.68.103,10.24.68.104",
        SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9",
    )
    assert picked == "enP7s7"


@needs_bash
def test_peer_coverage_needs_no_ssh_session(scenario):
    """Local dispatch (no SSH_CONNECTION) gets the same answer from the peers."""
    picked = scenario(_REPORTED, routes=_REPORTED_ROUTES, peers="10.24.68.103,10.24.68.104", SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9")
    assert picked == "enP7s7"


@needs_bash
def test_wired_beats_wireless_on_the_same_lan(scenario):
    """Both reach every peer directly: wired wins, whatever the kernel prefers."""
    picked = scenario(
        {"enP7s7": (*_WIRED, "192.168.1.20"), "awl0": (*_WIFI, "192.168.1.50")},
        routes={"192.168.1.21": [_link("192.168.1.0/24", "awl0", "192.168.1.50"), _link("192.168.1.0/24", "enP7s7", "192.168.1.20")]},
        kernel={"192.168.1.21": "awl0"},
        peers="192.168.1.21",
    )
    assert picked == "enP7s7"


@needs_bash
def test_specific_route_beats_a_default_route(scenario):
    """A route naming the peers' network is evidence; a default route is not."""
    picked = scenario(
        {"enP7s7": (*_WIRED, "10.24.68.102"), "wlP9s9": (*_WIFI, "192.168.1.50")},
        routes={"10.30.0.5": [_default("192.168.1.1", "wlP9s9"), "10.30.0.0/16 via 10.24.68.1 dev enP7s7"]},
        peers="10.30.0.5",
    )
    assert picked == "enP7s7"


@needs_bash
def test_vpn_only_management_is_found(scenario):
    """Peers reachable only over a VPN: the VPN, not the internet uplink."""
    picked = scenario(
        {"enP7s7": (*_WIRED, "10.0.0.5"), "tailscale0": (*_VPN, "100.64.0.2")},
        routes={"100.64.0.3": [_default("10.0.0.1", "enP7s7"), "100.64.0.3 dev tailscale0 table 52"]},
        peers="100.64.0.3",
    )
    assert picked == "tailscale0"


@needs_bash
def test_fabric_wins_only_when_it_is_what_reaches_the_peers(scenario):
    """A cluster listed by CX7 addresses has its management on the fabric."""
    picked = scenario(
        {"enP7s7": (*_WIRED, "10.24.68.102"), "enp1s0f0np0": (*_FABRIC, "192.168.11.13")},
        routes={"192.168.11.14": [_default("10.24.68.1", "enP7s7"), _link("192.168.11.0/24", "enp1s0f0np0", "192.168.11.13")]},
        peers="192.168.11.14",
    )
    assert picked == "enp1s0f0np0"


@needs_bash
def test_ssh_arrival_breaks_a_coverage_tie(scenario):
    """Two wired NICs on the peers' subnet: the one sparkrun arrived on."""
    picked = scenario(
        {"enA": (*_WIRED, "10.1.0.2"), "enB": (*_WIRED, "10.1.0.3")},
        routes={"10.1.0.9": [_link("10.1.0.0/24", "enA", "10.1.0.2"), _link("10.1.0.0/24", "enB", "10.1.0.3")]},
        peers="10.1.0.9",
        SSH_CONNECTION="10.1.0.100 52344 10.1.0.3 22",
    )
    assert picked == "enB"


@needs_bash
def test_operator_login_over_wifi_does_not_pull_the_cluster_onto_it(scenario):
    """Local dispatch inherits the operator's SSH_CONNECTION; wired still wins."""
    picked = scenario(
        {"enP7s7": (*_WIRED, "192.168.1.20"), "wlP9s9": (*_WIFI, "192.168.1.50")},
        routes={"192.168.1.21": [_link("192.168.1.0/24", "enP7s7", "192.168.1.20"), _link("192.168.1.0/24", "wlP9s9", "192.168.1.50")]},
        peers="192.168.1.21",
        SSH_CONNECTION="192.168.1.99 52344 192.168.1.50 22",
    )
    assert picked == "enP7s7"


@needs_bash
def test_multipath_default_route_is_still_a_default_route(scenario):
    """ECMP nexthops inherit their header: a multipath default is grade 1,
    so it must not outrank a NIC that has a specific route to the peers."""
    picked = scenario(
        {"enA": (*_WIRED, "10.0.0.5"), "enB": (*_WIRED, "10.0.1.5"), "enC": (*_WIRED, "10.24.68.5")},
        routes={
            "10.30.0.5": [
                "default proto static metric 100",
                "\tnexthop via 10.0.0.1 dev enA weight 1",
                "\tnexthop via 10.0.1.1 dev enB weight 1",
                "10.30.0.0/16 via 10.24.68.1 dev enC",
            ]
        },
        peers="10.30.0.5",
    )
    assert picked == "enC"


@needs_bash
def test_own_address_is_not_a_peer(scenario):
    """A single-host cluster has nobody to measure against: fallback chain."""
    picked = scenario(_REPORTED, routes=_REPORTED_ROUTES, peers="10.24.68.102", SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9")
    assert picked == "wlP9s9"


@needs_bash
def test_down_interface_is_not_a_candidate(scenario):
    ifaces = {"enP7s7": (True, False, False, "down", "10.24.68.102"), "wlP9s9": (*_WIFI, "192.168.1.50")}
    picked = scenario(ifaces, routes=_REPORTED_ROUTES, peers="10.24.68.103")
    assert picked == "wlP9s9"


@needs_bash
def test_pin_outranks_peer_coverage(scenario):
    picked = scenario(_REPORTED, routes=_REPORTED_ROUTES, peers="10.24.68.103", pinned="wlP9s9")
    assert picked == "wlP9s9"


@needs_bash
def test_scan_prefers_wired_over_wireless(scenario):
    """Last resort, wireless sorting first: the wired NIC is still chosen."""
    picked = scenario({"awl0": (*_WIFI, "192.168.1.50"), "enP7s7": (*_WIRED, "10.24.68.102")})
    assert picked == "enP7s7"


@needs_bash
def test_physical_nic_beats_vpn_when_both_reach_the_peers(scenario):
    """Equally reachable over a VPN and a physical NIC: the physical NIC, even
    when the control machine came in over the VPN."""
    picked = scenario(
        {"enP7s7": (*_WIRED, "10.1.0.2"), "tailscale0": (*_VPN, "100.64.0.2")},
        routes={"10.1.0.9": [_link("10.1.0.0/24", "enP7s7", "10.1.0.2"), "10.1.0.9 dev tailscale0 table 52"]},
        peers="10.1.0.9",
        SSH_CONNECTION="100.64.0.9 52344 100.64.0.2 22",
    )
    assert picked == "enP7s7"


# --- the address the cluster uses (NODE_IP, the head address) ---------------

_TWO_ADDRESSES = {"enP7s7": (*_WIRED, "192.168.1.5\n10.0.0.5")}


@needs_bash
def test_node_ip_is_the_address_facing_the_peers(scenario):
    """An interface's first address is not necessarily on the peers' network:
    DHCP primary plus a static cluster address on one NIC."""
    ip = scenario(
        _TWO_ADDRESSES,
        routes={"10.0.0.9": [_link("10.0.0.0/24", "enP7s7", "10.0.0.5")]},
        peers="10.0.0.9",
        call="sparkrun_mgmt_ip enP7s7",
    )
    assert ip == "10.0.0.5"


@needs_bash
def test_node_ip_falls_back_to_the_ssh_arrival_address(scenario):
    ip = scenario(_TWO_ADDRESSES, call="sparkrun_mgmt_ip enP7s7", SSH_CONNECTION="10.0.0.100 52344 10.0.0.5 22")
    assert ip == "10.0.0.5"


@needs_bash
def test_node_ip_falls_back_to_the_first_address(scenario):
    assert scenario(_TWO_ADDRESSES, call="sparkrun_mgmt_ip enP7s7") == "192.168.1.5"


@needs_bash
def test_node_ip_ignores_a_route_source_not_on_the_interface(scenario):
    """A route's src must be one of this interface's own addresses."""
    ip = scenario(
        _TWO_ADDRESSES,
        routes={"10.0.0.9": ["10.0.0.0/24 dev enP7s7 proto kernel scope link src 172.16.0.1"]},
        peers="10.0.0.9",
        call="sparkrun_mgmt_ip enP7s7",
    )
    assert ip == "192.168.1.5"


# --- the report: which rule decided, and what kind of interface -------------


@needs_bash
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(peers="10.24.68.103", SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9"), ("enP7s7", "peers", "wired")),
        (dict(SSH_CONNECTION="10.0.0.1 1 10.24.68.102 22", SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9"), ("enP7s7", "ssh", "wired")),
        (dict(SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9"), ("wlP9s9", "default-route", "wireless")),
        (dict(pinned="wlP9s9"), ("wlP9s9", "pinned", "wireless")),
        (dict(), ("enP7s7", "scan", "wired")),
    ],
)
def test_report_names_the_deciding_rule_and_kind(scenario, kwargs, expected):
    out = scenario(_REPORTED, routes=_REPORTED_ROUTES, call='sparkrun_mgmt_report ""', **kwargs)
    facts = dict(line.split("=", 1) for line in out.splitlines())
    assert (facts["DETECTED_MGMT_IFACE"], facts["DETECTED_MGMT_RULE"], facts["DETECTED_MGMT_KIND"]) == expected
    assert facts["DETECTED_MGMT_IP"] == ("10.24.68.102" if expected[0] == "enP7s7" else "192.168.1.50")


@needs_bash
def test_report_with_nothing_resolved_has_empty_values(scenario):
    out = scenario({"awl0": (True, False, True, "down", None)}, call='sparkrun_mgmt_report ""')
    assert out.splitlines() == ["DETECTED_MGMT_IFACE=", "DETECTED_MGMT_IP=", "DETECTED_MGMT_RULE=", "DETECTED_MGMT_KIND="]


@needs_bash
def test_virtual_kind(scenario):
    assert scenario({"tailscale0": (*_VPN, "100.64.0.2")}, call="sparkrun_mgmt_kind tailscale0") == "virtual"
    assert scenario({"enp1s0f0np0": (*_FABRIC, "192.168.11.13")}, call="sparkrun_mgmt_kind enp1s0f0np0") == "fabric"


def test_peer_list_is_normalized():
    assert mgmt_peer_list(["drew@10.0.0.1", "10.0.0.2", "10.0.0.1", "127.0.0.1", "localhost", "spark-02"]) == ("10.0.0.1,10.0.0.2,spark-02")
    assert mgmt_peer_list(["a,b", "x;rm -rf /", ""]) is None
    assert mgmt_peer_list(None) is None


def test_inject_mgmt_vars_passes_pin_and_peers():
    out = inject_mgmt_vars("set -u\n", mgmt_interface="enP7s7", peers=["10.0.0.1", "10.0.0.2"])
    assert "SPARKRUN_MGMT_IFACE=enP7s7" in out
    assert "SPARKRUN_MGMT_PEERS=10.0.0.1,10.0.0.2" in out
    assert inject_mgmt_vars("set -u\n") == "set -u\n"


# ---------------------------------------------------------------------------
# Operator override
# ---------------------------------------------------------------------------


@needs_bash
def test_pinned_interface_outranks_detection(net_fixture):
    result = net_fixture(pinned="wlP9s9", SPARKRUN_TEST_DEFAULT_ROUTE="enP7s7")
    assert result.stdout.strip() == "wlP9s9"


@needs_bash
def test_pinned_interface_that_does_not_exist_falls_back_and_warns(net_fixture):
    """A pin is not licence to emit a phantom name — that is the whole bug."""
    result = net_fixture(pinned="eth0")
    assert result.stdout.strip() == "enP7s7"
    assert "eth0" in result.stderr
    assert "not present" in result.stderr


@needs_bash
def test_pin_survives_the_helpers_own_include_position(net_fixture):
    """Regression: the helper must not default SPARKRUN_MGMT_IFACE itself.

    It is included partway down the script, so its own assignment would run
    after the injected one and silently erase the operator's pin.
    """
    body = inject_shell_vars("set -uo pipefail\n" + read_script("_mgmt_iface.sh"), SPARKRUN_MGMT_IFACE="wlP9s9")
    assert "SPARKRUN_MGMT_IFACE=wlP9s9" in body
    assert 'SPARKRUN_MGMT_IFACE=""' not in body


# ---------------------------------------------------------------------------
# Wiring: includes, formatting, generators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _INCLUDING_SCRIPTS)
def test_script_includes_the_shared_helper(name):
    from sparkrun.scripts import INCLUDE_DIRECTIVE

    script = read_script(name)
    assert "sparkrun_mgmt_iface() (" in script, "include directive not expanded in %s" % name
    # Line-level, not substring: the helper's own comments quote the directive
    # while describing it, and that must not read as an unexpanded include.
    unexpanded = [ln for ln in script.splitlines() if ln.strip().startswith(INCLUDE_DIRECTIVE)]
    assert not unexpanded, unexpanded


@pytest.mark.parametrize("name", _INCLUDING_SCRIPTS)
def test_script_has_no_hardcoded_eth0_fallback(name):
    """No probe may substitute a literal interface name for a failed lookup."""
    for line in read_script(name).splitlines():
        code = line.split("#", 1)[0]
        assert 'echo "eth0"' not in code, "%s still falls back to a hardcoded eth0" % name


def test_helper_is_brace_free():
    """Load-bearing: ray_head/ray_worker/the combined probe run str.format().

    A single brace here — including inside a comment — raises KeyError at a
    distance, in a Ray launch rather than in this file.
    """
    helper = read_script("_mgmt_iface.sh")
    assert "{" not in helper and "}" not in helper


@pytest.mark.parametrize("name", ("ray_head.sh", "ray_worker.sh"))
def test_templated_scripts_still_format(name):
    rendered = read_script(name).format(cleanup_cmd="echo cleanup", run_cmd="echo run", head_ip="10.0.0.1", ray_port=6379)
    assert "sparkrun_mgmt_iface" in rendered
    assert "echo run" in rendered


def test_include_directive_rejects_cycles(tmp_path, monkeypatch):
    import sparkrun.scripts as scripts_pkg

    def fake_load(_package, name, **_kw):
        return "# sparkrun:include %s\n" % name

    monkeypatch.setattr(scripts_pkg, "load_resource", fake_load)
    with pytest.raises(ValueError, match="Circular"):
        scripts_pkg.read_script("loop.sh")


def test_inject_shell_vars_places_assignments_after_the_shebang():
    out = inject_shell_vars("#!/bin/bash\nset -u\n", SPARKRUN_MGMT_IFACE="enP7s7")
    assert out.splitlines()[:2] == ["#!/bin/bash", "SPARKRUN_MGMT_IFACE=enP7s7"]


def test_inject_shell_vars_quotes_and_skips_empty():
    assert inject_shell_vars("set -u\n", A=None) == "set -u\n"
    assert inject_shell_vars("set -u\n", A="") == "set -u\n"
    assert "A='a b; rm -rf /'" in inject_shell_vars("set -u\n", A="a b; rm -rf /")


@pytest.mark.parametrize(
    "generate",
    (generate_ib_detect_script, generate_cx7_detect_script, generate_combined_probe_script, generate_ip_detect_script),
)
def test_generators_thread_the_pin_and_peers(generate):
    assert "SPARKRUN_MGMT_IFACE=" not in generate()
    assert "SPARKRUN_MGMT_PEERS=" not in generate()
    script = generate("enP7s7", peers=["10.0.0.1", "10.0.0.2"])
    assert "SPARKRUN_MGMT_IFACE=enP7s7" in script
    assert "SPARKRUN_MGMT_PEERS=10.0.0.1,10.0.0.2" in script


def test_cx7_probe_gets_no_peers(monkeypatch):
    """cx7_detect.sh drops its management NIC from the CX7 list ("the link not
    to touch"); peer coverage may choose a CX7 port and would hide a real one."""
    from sparkrun.orchestration import networking

    scripts = []
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", lambda hosts, script, **kw: scripts.append(script) or [])
    monkeypatch.setattr("sparkrun.utils.is_local_host", lambda host: False)
    networking.detect_cx7_for_hosts(["10.0.0.1", "10.0.0.2"])
    assert scripts
    assert not [ln for script in scripts for ln in script.splitlines() if ln.startswith("SPARKRUN_MGMT_PEERS=")]


@pytest.mark.parametrize("method", ("generate_ray_head_script", "generate_ray_worker_script"))
def test_ray_scripts_thread_the_pin_and_peers(method):
    """ray_head/ray_worker detect NODE_IP themselves; they used to ignore the pin."""
    from sparkrun.orchestration.executors.docker import DockerExecutor

    kwargs = {"head_ip": "10.0.0.1"} if method == "generate_ray_worker_script" else {}
    script = getattr(DockerExecutor(), method)("img", "c", mgmt_interface="enP7s7", peers=["10.0.0.1", "10.0.0.2"], **kwargs)
    assert "SPARKRUN_MGMT_IFACE=enP7s7" in script
    assert "SPARKRUN_MGMT_PEERS=10.0.0.1,10.0.0.2" in script


def test_detect_host_ip_threads_the_pin_and_peers():
    """The head address must come from the same rule as the IB probe's NODE_IP."""
    from unittest import mock

    from sparkrun.orchestration.primitives import detect_host_ip
    from sparkrun.orchestration.ssh import RemoteResult

    with mock.patch(
        "sparkrun.orchestration.primitives.run_remote_script",
        return_value=RemoteResult(host="h", returncode=0, stdout="10.0.0.1\n", stderr=""),
    ) as run:
        assert detect_host_ip("h", mgmt_interface="enP7s7", peers=["h", "h2"]) == "10.0.0.1"
    script = run.call_args.args[1]
    assert "SPARKRUN_MGMT_IFACE=enP7s7" in script
    assert "SPARKRUN_MGMT_PEERS=h,h2" in script


# ---------------------------------------------------------------------------
# Downstream: an unresolved interface must not reach the comm env
# ---------------------------------------------------------------------------


def test_empty_socket_ifname_falls_back_to_the_fabric_interfaces():
    """The Python half of the fix: empty means "use the adapters that exist"."""
    env = generate_nccl_env(
        {
            "IB_DETECTED": "1",
            "DETECTED_HCA_LIST": "rocep1s0f0",
            "DETECTED_SOCKET_IFNAME": "",
            "DETECTED_NET_LIST": "enp1s0f0np0,enP2p1s0f0np0",
            "DETECTED_MGMT_IP": "",
        }
    )
    assert env["GLOO_SOCKET_IFNAME"] == "enp1s0f0np0,enP2p1s0f0np0"
    assert env["TP_SOCKET_IFNAME"] == "enp1s0f0np0,enP2p1s0f0np0"
    assert env["MN_IF_NAME"] == "enp1s0f0np0,enP2p1s0f0np0"
    assert env["NCCL_SOCKET_IFNAME"] == "enp1s0f0np0,enP2p1s0f0np0"
    assert env["NODE_IP"] == ""


def test_detected_mgmt_interface_still_leads():
    """Unchanged behaviour when detection succeeds — mgmt first, fabric after."""
    env = generate_nccl_env(
        {
            "IB_DETECTED": "1",
            "DETECTED_SOCKET_IFNAME": "enP7s7",
            "DETECTED_NET_LIST": "enp1s0f0np0,enP2p1s0f0np0",
            "DETECTED_MGMT_IP": "10.10.10.126",
        }
    )
    assert env["GLOO_SOCKET_IFNAME"] == "enP7s7"
    assert env["NCCL_SOCKET_IFNAME"] == "enP7s7,enp1s0f0np0,enP2p1s0f0np0"
    assert env["NODE_IP"] == "10.10.10.126"
