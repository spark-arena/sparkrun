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
from sparkrun.scripts import inject_shell_vars, read_script

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
# host's does unless SPARKRUN_TEST_DEFAULT_ROUTE names an interface.
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
[ -n "$dev" ] || exit 0
addr=$(cat "$SPARKRUN_TEST_ADDRS/$dev" 2>/dev/null)
[ -n "$addr" ] || exit 0
echo "9: $dev    inet $addr/24 scope global $dev"
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
def test_default_route_wins_when_present(net_fixture):
    """Connected clusters keep resolving exactly as they always have."""
    result = net_fixture(SPARKRUN_TEST_DEFAULT_ROUTE="wlP9s9")
    # wlP9s9 is down and has no address — the default route still wins,
    # because that is the historical behaviour and it is authoritative.
    assert result.stdout.strip() == "wlP9s9"


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
    (generate_ib_detect_script, generate_cx7_detect_script, generate_combined_probe_script),
)
def test_generators_thread_the_pin(generate):
    assert "SPARKRUN_MGMT_IFACE=" not in generate()
    assert "SPARKRUN_MGMT_IFACE=enP7s7" in generate("enP7s7")


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
