"""Address-persistence attribution (``scripts/_net_persist.sh``).

``setup check`` used to answer "will this CX7 address survive a reboot?" by
testing for ``/etc/netplan/40-cx7.yaml``, which really answers "did sparkrun
write it?".  Every host configured another way — a differently-numbered netplan
file, ``nmcli`` (which on Ubuntu 24.04 writes its *own* ``90-NM-<uuid>.yaml``),
a ``.network`` unit — was reported as "won't survive reboot" forever, pointing
at a ``sparkrun setup cx7`` that would then correctly do nothing.

The helper is exercised under real ``bash`` with stub ``netplan`` / ``nmcli`` /
``networkctl`` on a closed ``PATH``, because the whole point is which external
probe answers first and what happens when none of them is there — semantics no
Python mock of the Python layer would reach.

Not covered here: the ifupdown branch reads ``/etc/network/interfaces`` by
absolute path, so it cannot be faked in a fixture tree.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess

import pytest

from sparkrun.scripts import read_script

# Coreutils the helper needs.  PATH is closed to exactly these plus the stubs,
# so "probe absent" is a property of the fixture and not of the dev machine
# (which, being a DGX Spark, has a real netplan).
_SYS_BINS = ("awk", "sed", "grep", "cut", "head", "tr", "cat", "basename", "python3", "ip")

_STUB_NETPLAN = """#!/bin/bash
# `netplan status --format=json` is the only form the helper uses, and it is
# the only one that works unprivileged (the config files are mode 600).
if [ "$1" = "status" ]; then
    [ -s "$SPARKRUN_TEST_NETPLAN_JSON" ] || exit 1
    cat "$SPARKRUN_TEST_NETPLAN_JSON"
    exit 0
fi
exit 1
"""

_STUB_NMCLI = """#!/bin/bash
# Two queries: the active device->profile map, and one profile's properties.
[ -s "$SPARKRUN_TEST_NM_ACTIVE" ] || exit 1
case "$*" in
    *"connection show --active"*)
        cat "$SPARKRUN_TEST_NM_ACTIVE" ;;
    *"connection show "*)
        prof="${@: -1}"
        cat "$SPARKRUN_TEST_NM_PROPS/$prof" 2>/dev/null || exit 1 ;;
    *) exit 1 ;;
esac
"""

_STUB_NETWORKCTL = """#!/bin/bash
echo "* 3: $3"
echo "         Network File: ${SPARKRUN_TEST_NETWORK_FILE:-n/a}"
echo "                State: routable"
"""

_STUB_IP = """#!/bin/bash
echo "3: dev    inet ${SPARKRUN_TEST_IP:-192.168.11.1}/24 scope global ${SPARKRUN_TEST_IP_FLAGS:-} dev"
"""


@pytest.fixture
def persist_fixture(tmp_path):
    """Return a runner for ``sparkrun_net_persistence`` with stubbed probes."""
    bindir = tmp_path / "bin"
    sysbin = tmp_path / "sysbin"
    props = tmp_path / "nm-props"
    for d in (bindir, sysbin, props):
        d.mkdir(parents=True)

    for name in _SYS_BINS:
        found = shutil.which(name)
        if found:
            (sysbin / name).symlink_to(found)

    def _stub(name: str, body: str) -> None:
        path = bindir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    script = tmp_path / "probe.sh"

    def run(
        *,
        iface: str = "enp1s0f0np0",
        addr: str = "192.168.11.1",
        netplan: dict | None = None,
        nm_active: dict[str, str] | None = None,
        nm_props: dict[str, dict[str, str]] | None = None,
        network_file: str | None = None,
        probes: tuple[str, ...] = ("netplan", "nmcli", "networkctl"),
        func: str = "sparkrun_net_persistence",
        ip_flags: str = "",
    ) -> str:
        for name, body in (("netplan", _STUB_NETPLAN), ("nmcli", _STUB_NMCLI), ("networkctl", _STUB_NETWORKCTL)):
            if name in probes:
                _stub(name, body)
            elif (bindir / name).exists():
                (bindir / name).unlink()
        _stub("ip", _STUB_IP)

        np_json = tmp_path / "netplan.json"
        np_json.write_text(json.dumps(netplan) if netplan else "")
        active = tmp_path / "nm-active"
        active.write_text("".join("%s:%s\n" % (dev, prof) for dev, prof in (nm_active or {}).items()))
        for prof, kv in (nm_props or {}).items():
            (props / prof).write_text("".join("%s:%s\n" % (k, val) for k, val in kv.items()))

        script.write_text("set -uo pipefail\n" + read_script("_net_persist.sh") + '\n%s "$1" "$2"\n' % func)
        env = {
            "PATH": "%s:%s" % (bindir, sysbin),
            "SPARKRUN_TEST_NETPLAN_JSON": str(np_json),
            "SPARKRUN_TEST_NM_ACTIVE": str(active),
            "SPARKRUN_TEST_NM_PROPS": str(props),
            "SPARKRUN_TEST_IP": addr,
            "SPARKRUN_TEST_IP_FLAGS": ip_flags,
        }
        if network_file is not None:
            env["SPARKRUN_TEST_NETWORK_FILE"] = network_file
        # bash by absolute path: PATH is deliberately closed to the stubs.
        result = subprocess.run([shutil.which("bash"), str(script), iface, addr], capture_output=True, text=True, env=env, timeout=30)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    return run


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")

# `netplan status` reports the merged view of every config directory and never
# names a file, which is exactly why it survives an arbitrary filename.
_NETPLAN_OWNED = {
    "enp1s0f0np0": {"backend": "NetworkManager", "id": "enp1s0f0np0", "addresses": [{"192.168.11.1": {"prefix": 24}}]},
    "enP7s7": {"addresses": [{"10.24.11.1": {"prefix": 24}}]},
}


@needs_bash
def test_netplan_owned_interface_is_persistent_whatever_the_filename(persist_fixture):
    assert persist_fixture(netplan=_NETPLAN_OWNED) == "persistent|netplan|enp1s0f0np0"


@needs_bash
def test_interface_netplan_does_not_own_falls_through(persist_fixture):
    """An interface absent from netplan's view carries no ``id``."""
    out = persist_fixture(iface="enP7s7", addr="10.24.11.1", netplan=_NETPLAN_OWNED)
    assert out == "ephemeral||"


@needs_bash
def test_networkmanager_manual_profile_is_persistent(persist_fixture):
    out = persist_fixture(
        nm_active={"enp1s0f0np0": "cx7-a"},
        nm_props={"cx7-a": {"connection.autoconnect": "yes", "ipv4.method": "manual", "ipv4.addresses": "192.168.11.1/24"}},
    )
    assert out == "persistent|networkmanager|cx7-a"


@needs_bash
def test_networkmanager_profile_name_containing_a_colon(persist_fixture):
    """Device names cannot contain ``:``; profile names can, so split once."""
    out = persist_fixture(
        nm_active={"enp1s0f0np0": "lab:cx7:a"},
        nm_props={"lab:cx7:a": {"connection.autoconnect": "yes", "ipv4.method": "manual", "ipv4.addresses": "192.168.11.1/24"}},
    )
    assert out == "persistent|networkmanager|lab:cx7:a"


@needs_bash
def test_networkmanager_profile_that_will_not_autoconnect_is_ephemeral(persist_fixture):
    """A saved profile that never comes up at boot does not persist anything."""
    out = persist_fixture(
        nm_active={"enp1s0f0np0": "cx7-a"},
        nm_props={"cx7-a": {"connection.autoconnect": "no", "ipv4.method": "manual", "ipv4.addresses": "192.168.11.1/24"}},
    )
    assert out.startswith("ephemeral|networkmanager|cx7-a (autoconnect disabled)")


@needs_bash
def test_networkmanager_profile_pinning_a_different_address_is_ephemeral(persist_fixture):
    """The live address is what has to come back, not merely *an* address."""
    out = persist_fixture(
        nm_active={"enp1s0f0np0": "cx7-a"},
        nm_props={"cx7-a": {"connection.autoconnect": "yes", "ipv4.method": "manual", "ipv4.addresses": "192.168.99.9/24"}},
    )
    assert out.startswith("ephemeral|networkmanager|")
    assert "192.168.99.9/24" in out


@needs_bash
def test_networkmanager_dhcp_profile_is_persistent_but_labelled(persist_fixture):
    out = persist_fixture(
        nm_active={"enp1s0f0np0": "cx7-a"},
        nm_props={"cx7-a": {"connection.autoconnect": "yes", "ipv4.method": "auto"}},
    )
    assert out == "persistent|networkmanager|cx7-a (ipv4.method=auto)"


@needs_bash
def test_systemd_networkd_unit_is_persistent(persist_fixture):
    out = persist_fixture(probes=("networkctl",), network_file="/etc/systemd/network/40-cx7.network")
    assert out == "persistent|systemd-networkd|/etc/systemd/network/40-cx7.network"


@needs_bash
def test_unmanaged_networkd_interface_does_not_claim_a_file(persist_fixture):
    assert persist_fixture(probes=("networkctl",), network_file="n/a") == "ephemeral||"


@needs_bash
def test_hand_added_address_with_probes_available_is_ephemeral(persist_fixture):
    """`ip addr add` with every probe answering: genuinely won't survive."""
    assert persist_fixture(netplan={}, nm_active={}) == "ephemeral||"


@needs_bash
def test_no_probe_available_is_unknown_not_ephemeral(persist_fixture):
    """The load-bearing rule: "couldn't tell" must never mean "not persistent"."""
    assert persist_fixture(probes=()) == "unknown||"


@needs_bash
def test_failing_nmcli_does_not_count_as_a_probe(persist_fixture):
    """nmcli present but NetworkManager down is unprobed, not "NM says no"."""
    # The stub exits 1 when its active-connection file is empty, which is what
    # a stopped NetworkManager looks like.
    out = persist_fixture(probes=("nmcli",), nm_active=None)
    assert out == "unknown||"


@needs_bash
def test_dhcp_lease_is_detected(persist_fixture):
    assert persist_fixture(func="sparkrun_net_is_dhcp", ip_flags="dynamic") == "1"


@needs_bash
def test_static_address_is_not_a_dhcp_lease(persist_fixture):
    assert persist_fixture(func="sparkrun_net_is_dhcp") == "0"
