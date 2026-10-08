"""Built-in setup checks, shared by the wizard and readiness command."""

from __future__ import annotations
from sparkrun.core.setup_models import CheckContext, CheckItem, HostState, OK, WARN, FAIL, SKIP
from sparkrun.orchestration.executors.docker import DOCKER_IPC_DEFAULT


def _truthy(facts: dict[str, str], key: str) -> bool:
    return facts.get(key, "").strip() == "1"


def _int_fact(facts: dict[str, str], key: str) -> int:
    """Read a numeric probe fact, treating anything unparseable as 0.

    A missing or malformed value means "the probe could not tell us", which
    must read as *no finding* rather than as a finding of zero severity.
    """
    try:
        return int(facts.get(key, "").strip())
    except (TypeError, ValueError):
        return 0


#: Shared remedy for both CDI failure modes (absent spec, stale spec). One
#: string so the manual command can never drift from the wizard step that
#: runs it.
_CDI_REGENERATE = "sparkrun setup wizard%s (NVIDIA CDI step) — or: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"


def _as_int(value: str | None) -> int:
    try:
        return int((value or "0").strip())
    except (TypeError, ValueError):
        return 0


# --- Individual checks ------------------------------------------------------
#
# Each takes the host's HostState + the cluster CheckContext and returns a
# CheckItem, or None when the check does not apply to this host (omitted).


def _check_docker_installed(state: HostState, ctx: CheckContext) -> CheckItem:
    if _truthy(state.facts, "CHECK_DOCKER_INSTALLED"):
        return CheckItem("docker_installed", "Docker installed", OK)
    return CheckItem(
        "docker_installed",
        "Docker installed",
        FAIL,
        "docker not found on PATH",
        "Install Docker Engine on this host (https://docs.docker.com/engine/install/)",
    )


def _check_docker_group(state: HostState, ctx: CheckContext) -> CheckItem:
    if _truthy(state.facts, "CHECK_DOCKER_GROUP"):
        return CheckItem("docker_group", "Docker group membership", OK)
    if _truthy(state.facts, "CHECK_DOCKER_USABLE"):
        return CheckItem("docker_group", "Docker group membership", SKIP, "Docker is already usable without group membership")
    user = state.facts.get("CHECK_USER", "the ssh user")
    return CheckItem(
        "docker_group",
        "Docker group membership",
        WARN,
        "user '%s' is not in the docker group" % user,
        "sparkrun setup docker-group%s" % ctx.cluster_flag,
    )


def _check_docker_usable(state: HostState, ctx: CheckContext) -> CheckItem | None:
    facts = state.facts
    if not _truthy(facts, "CHECK_DOCKER_INSTALLED"):
        return None  # already reported as a failure by docker_installed
    if _truthy(facts, "CHECK_DOCKER_USABLE"):
        return CheckItem("docker_usable", "Docker usable without sudo", OK)
    if _truthy(facts, "CHECK_DOCKER_GROUP"):
        detail = "in the docker group but the daemon is not usable yet"
        guidance = "Log out and back in (or run 'newgrp docker'); ensure dockerd is running"
    else:
        detail = "cannot run docker without sudo"
        guidance = "sparkrun setup docker-group%s, then re-login" % ctx.cluster_flag
    return CheckItem("docker_usable", "Docker usable without sudo", FAIL, detail, guidance)


def _check_nvidia_ctk(state: HostState, ctx: CheckContext) -> CheckItem:
    facts = state.facts
    if not state.has_nvidia:
        return CheckItem("nvidia_ctk", "NVIDIA Container Toolkit", SKIP, "no NVIDIA GPU detected")
    if _truthy(facts, "CHECK_NVIDIA_CTK"):
        return CheckItem("nvidia_ctk", "NVIDIA Container Toolkit", OK)
    return CheckItem(
        "nvidia_ctk",
        "NVIDIA Container Toolkit",
        FAIL,
        "nvidia-ctk not found",
        "Install the NVIDIA Container Toolkit (provides nvidia-ctk)",
    )


def _check_cdi_spec(state: HostState, ctx: CheckContext) -> CheckItem | None:
    if not ctx.cdi_required(state.host):
        return None
    facts = state.facts
    if not state.has_nvidia:
        return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", SKIP, "no NVIDIA GPU detected")
    if not _truthy(facts, "CHECK_NVIDIA_CTK"):
        return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", SKIP, "requires nvidia-ctk (see above)")

    missing = _int_fact(facts, "CHECK_CDI_PATHS_MISSING")
    checked = _int_fact(facts, "CHECK_CDI_PATHS_CHECKED")
    stale = bool(checked and missing)

    if _truthy(facts, "CHECK_CDI_SPEC"):
        # Present, but possibly stale: the spec pins absolute driver-library
        # and device-node paths, and a driver upgrade moves them. Reported as
        # WARN rather than FAIL because a spec may legitimately reference an
        # optional path, and a false hard-failure on a working host is worse
        # than a prompt to regenerate.
        if not stale:
            return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", OK)
        detail = "%d of %d referenced paths missing — spec looks stale (driver upgraded?)" % (missing, checked)
        return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", WARN, detail, _CDI_REGENERATE % ctx.cluster_flag)

    return CheckItem(
        "cdi_spec",
        "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)",
        FAIL,
        "/etc/cdi/nvidia.yaml missing or empty",
        _CDI_REGENERATE % ctx.cluster_flag,
    )


#: systemd never reaps IPC for UIDs at or below this — ``SYSTEM_UID_MAX``, whose
#: compiled-in default is 999. A root or system-user workload is not at risk,
#: which is exactly why ``-o user=root`` "fixes" the failure (at the cost of the
#: rootless hardening and a cache full of root-owned files).
_LOGIND_SYSTEM_UID_MAX = 999

_IPC_LABEL = "Host IPC vs systemd RemoveIPC"


def _check_host_ipc(state: HostState, ctx: CheckContext) -> CheckItem | None:
    """Would systemd-logind reap this host's workload IPC after launch?

    Only applies when a launch here shares the *host's* IPC namespace — an
    explicit Docker ``ipc: host`` (sparkrun defaults to ``shareable``, which is
    immune) or the ``local`` executor, which runs natively and so has no
    namespace to hide in. Everything else is omitted rather than reported OK:
    there is no gap to describe.

    The hazard needs all three of host IPC, a regular UID, and a user manager
    that stops — so the check reports only when it can see all three, and any
    one of them being absent or unknowable is a reason to stay quiet or say
    "couldn't tell" rather than to warn.
    """
    facts = state.facts
    if not ctx.shares_host_ipc(state.host):
        return None

    raw_uid = facts.get("CHECK_UID", "").strip()
    try:
        uid = int(raw_uid)
    except (TypeError, ValueError):
        return CheckItem("host_ipc", _IPC_LABEL, SKIP, "could not read the ssh user's UID")
    if uid <= _LOGIND_SYSTEM_UID_MAX:
        return None  # logind does not reap system users; nothing at risk

    remove_ipc = facts.get("CHECK_LOGIND_REMOVE_IPC", "unknown").strip()
    linger = facts.get("CHECK_LOGIND_LINGER", "unknown").strip()
    user = facts.get("CHECK_USER", "the ssh user")

    if linger == "1":
        return CheckItem("host_ipc", _IPC_LABEL, OK, "lingering enabled for '%s', so its user manager never stops" % user)
    if remove_ipc == "no":
        return CheckItem("host_ipc", _IPC_LABEL, OK, "logind RemoveIPC=no")
    if remove_ipc != "yes":
        return CheckItem("host_ipc", _IPC_LABEL, SKIP, "could not read logind's RemoveIPC setting")

    detail = "RemoveIPC=yes and no lingering for '%s' (uid %d)" % (user, uid)
    if linger != "0":
        detail += "; lingering could not be confirmed"
    detail += " — logind deletes this workload's /dev/shm semaphores shortly after the launching SSH session closes"
    return CheckItem(
        "host_ipc",
        _IPC_LABEL,
        WARN,
        detail,
        "sudo loginctl enable-linger %s — or drop the 'ipc: host' override (sparkrun defaults to '%s')" % (user, DOCKER_IPC_DEFAULT),
    )


def _check_earlyoom(state: HostState, ctx: CheckContext) -> CheckItem:
    facts = state.facts
    if _truthy(facts, "CHECK_EARLYOOM_ACTIVE"):
        return CheckItem("earlyoom", "earlyoom OOM protection", OK)
    if _truthy(facts, "CHECK_EARLYOOM_INSTALLED"):
        detail = "earlyoom installed but not active"
    else:
        detail = "earlyoom not installed"
    return CheckItem(
        "earlyoom",
        "earlyoom OOM protection",
        WARN,
        detail,
        "sparkrun setup earlyoom%s" % ctx.cluster_flag,
    )


def _check_sudoers(state: HostState, ctx: CheckContext) -> CheckItem:
    chown = state.facts.get("CHECK_SUDOERS_CHOWN", "unknown")
    dropcaches = state.facts.get("CHECK_SUDOERS_DROPCACHES", "unknown")
    if chown == "unknown" or dropcaches == "unknown":
        return CheckItem(
            "sudoers",
            "Sudoers entries (fix-permissions, clear-cache)",
            SKIP,
            "cannot verify without passwordless sudo",
        )
    if chown == "1" and dropcaches == "1":
        return CheckItem("sudoers", "Sudoers entries (fix-permissions, clear-cache)", OK)
    missing = []
    if chown != "1":
        missing.append("fix-permissions")
    if dropcaches != "1":
        missing.append("clear-cache")
    return CheckItem(
        "sudoers",
        "Sudoers entries (fix-permissions, clear-cache)",
        WARN,
        "missing: %s" % ", ".join(missing),
        "sparkrun setup wizard%s — or setup fix-permissions/clear-cache with --save-sudo" % ctx.cluster_flag,
    )


def _check_ssh_mesh(state: HostState, ctx: CheckContext) -> CheckItem | None:
    """Every mesh leg from this host: peers' management names, the control
    machine when it joins the mesh, and peers' CX7 addresses on shared subnets.
    """
    if not ctx.multi_host:
        return None  # SSH mesh is only meaningful with peers
    facts = state.facts
    unknown = _as_int(facts.get("CHECK_MESH_FABRIC_UNKNOWN"))
    fabric = _as_int(facts.get("CHECK_MESH_FABRIC_TOTAL")) + unknown
    total = _as_int(facts.get("CHECK_MESH_TOTAL")) + fabric
    reachable = _as_int(facts.get("CHECK_MESH_OK")) + _as_int(facts.get("CHECK_MESH_FABRIC_OK"))
    if total == 0:
        return None
    scope = " (incl. %d CX7 address(es))" % fabric if fabric else ""
    if reachable >= total:
        return CheckItem("ssh_mesh", "SSH mesh to peers", OK, "all %d peer address(es) reachable%s" % (total, scope))
    detail = "%d/%d peer address(es) reachable over SSH%s" % (reachable, total, scope)
    hostkey = _as_int(facts.get("CHECK_MESH_HOSTKEY")) + _as_int(facts.get("CHECK_MESH_FABRIC_HOSTKEY"))
    if hostkey:
        detail += "; %d without a recorded host key" % hostkey
    if unknown:
        # Unverified is not "reachable": a probe that timed out must not let
        # the wizard skip a mesh it never confirmed.
        detail += "; %d CX7 address(es) could not be checked" % unknown
    return CheckItem("ssh_mesh", "SSH mesh to peers", WARN, detail, "sparkrun setup ssh%s" % ctx.cluster_flag)


def _check_cx7(state: HostState, ctx: CheckContext) -> CheckItem | None:
    """Vet *effective* CX7 state via :func:`detect_cx7_for_hosts` output.

    Reuses the same detection the real CX7 flow uses — per-interface link
    state, assigned IP, subnet, and *who persists the address* — so this
    reflects working networking rather than the presence of one config file.

    Persistence is attributed to whatever owns the interface (any netplan
    file, a NetworkManager profile, a ``.network`` unit, ifupdown), because
    the question is "will this address come back after a reboot", not "did
    sparkrun write it". Checking for sparkrun's own ``40-cx7.yaml`` flagged
    every hand-configured cluster — including one set up with ``nmcli``,
    which on Ubuntu 24.04 writes its own ``90-NM-<uuid>.yaml``.

    :attr:`CX7Persistence.UNKNOWN` (no probe available on the host) is
    reported but is **not** a warning: "couldn't tell" is not "won't persist".

    CX7 is inter-node, so it is only evaluated for multi-host clusters.
    """
    from sparkrun.orchestration.networking import CX7Persistence

    if not ctx.multi_host:
        return None
    det = state.cx7
    if det is None:
        return CheckItem("cx7", "CX7 high-speed networking", SKIP, "could not probe CX7 on this host")
    if not det.detected or not det.interfaces:
        return CheckItem("cx7", "CX7 high-speed networking", SKIP, "no CX7/InfiniBand interfaces detected")

    up = [i for i in det.interfaces if i.state.lower() == "up" and i.ip]
    not_ready = [i for i in det.interfaces if i.state.lower() != "up" or not i.ip]

    if not up:
        names = ", ".join(i.name for i in det.interfaces)
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "CX7 interface(s) present but none up with an IP (%s)" % names,
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )

    subnets = ", ".join(sorted({i.subnet for i in up if i.subnet}))
    detail = "%d interface(s) up%s" % (len(up), " on %s" % subnets if subnets else "")

    if not_ready:
        bad = ", ".join("%s(%s)" % (i.name, i.state or "no-ip") for i in not_ready)
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "%s; not ready: %s" % (detail, bad),
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )
    ephemeral = [i for i in up if i.persistence is CX7Persistence.EPHEMERAL]
    if ephemeral:
        # Nothing on the host declares these addresses — they were added by
        # hand (or by a profile that won't auto-connect) and are gone on boot.
        named = ", ".join("%s (%s)" % (i.name, i.describe_persistence()) if i.persistence_source else i.name for i in ephemeral)
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "%s but no network config declares %s (won't survive reboot)" % (detail, named),
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )

    leased = [i for i in up if i.dhcp]
    if leased:
        # Persistent, but not pinned: NCCL peer addressing needs these stable.
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "%s but %s hold DHCP leases, so the addresses may change" % (detail, ", ".join(i.name for i in leased)),
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )

    unknown = [i for i in up if i.persistence is CX7Persistence.UNKNOWN]
    if unknown:
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            OK,
            "%s; could not verify persistence of %s (no netplan/nmcli/networkctl on host)" % (detail, ", ".join(i.name for i in unknown)),
        )

    owners = sorted({i.describe_persistence() for i in up})
    return CheckItem("cx7", "CX7 high-speed networking", OK, "%s; persisted by %s" % (detail, ", ".join(owners)))


def _check_rdma(state: "HostState", ctx: CheckContext) -> "CheckItem | None":
    """RDMA devices are present and up, and the tooling to exercise them exists.

    Deliberately cheap: it reads what the host reports and never sends a byte
    over the fabric. "The link is configured" and "the link performs" are
    different questions, and answering the second takes minutes of real
    transfers — that is ``sparkrun setup rdma-test``, which this check points
    at rather than becoming.

    Absent perftest is **not** a warning on its own. It is one command away,
    and ``setup rdma-test`` falls back to its container image, so reporting it
    as a defect would fire on a host where nothing is wrong.

    Inactive additional ports are informational when active devices exist:
    switch and single-cable topologies need not cable every physical port.
    An inactive device backing an UP, addressed CX7 interface remains a warning.
    This probe does not carry all-to-all connectivity/performance evidence.

    Inter-node, so single-host clusters are skipped.
    """
    if not ctx.multi_host:
        return None
    facts = state.rdma
    if facts is None or not facts.complete:
        return CheckItem("rdma", "RDMA fabric", SKIP, "could not probe RDMA devices on this host")
    if not facts.devices:
        return CheckItem("rdma", "RDMA fabric", SKIP, "no RDMA devices present")

    active = [d for d in facts.devices if d.active]
    inactive = [d for d in facts.devices if not d.active]

    if not active:
        return CheckItem(
            "rdma",
            "RDMA fabric",
            WARN,
            "%d device(s) present but none ACTIVE: %s" % (len(inactive), ", ".join(d.name for d in inactive)),
            guidance="check cabling, then: sparkrun setup cx7%s" % ctx.cluster_flag,
        )

    detail = "%d device(s) ACTIVE: %s" % (len(active), ", ".join(d.name for d in active))
    rate = next((d.rate_gbps for d in active if d.rate_gbps), None)
    if rate:
        detail += " @ %.0f Gb/s" % rate
    if not facts.has_perftest:
        detail += "; perftest not installed (rdma-test will use its container image)"

    if inactive:
        configured_up = (
            {interface.name for interface in state.cx7.interfaces if interface.state.lower() == "up" and interface.ip}
            if state.cx7 is not None
            else set()
        )
        required_inactive = [device for device in inactive if device.netdev and device.netdev in configured_up]
        if required_inactive:
            return CheckItem(
                "rdma",
                "RDMA fabric",
                WARN,
                "%s; configured interface(s) have inactive RDMA devices: %s"
                % (
                    detail,
                    ", ".join("%s (%s)" % (device.name, device.netdev) for device in required_inactive),
                ),
                guidance="check RDMA link state on those interfaces, then: sparkrun setup rdma-test%s" % ctx.cluster_flag,
            )
        detail += "; %d additional device(s) inactive: %s (unused ports may remain down)" % (
            len(inactive),
            ", ".join(device.name for device in inactive),
        )
    return CheckItem(
        "rdma",
        "RDMA fabric",
        OK,
        detail + "; link state only",
        guidance="if connectivity/performance are not yet verified: sparkrun setup rdma-test%s" % ctx.cluster_flag,
    )


#: Ordered registry of readiness checks. Order is the display/evaluation
#: order and is the seed of the future ordered/dependency-driven step system.
