"""CX7 network interface detection, planning, and configuration.

Detects ConnectX-7 interfaces on DGX Spark hosts via SSH, plans static
IP assignments across the cluster, and applies netplan configuration.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass, field
from enum import Enum

from sparkrun.orchestration.infiniband import parse_kv_output
from sparkrun.scripts import read_script

logger = logging.getLogger(__name__)

# RFC 1918 private ranges, in preference order for subnet selection
_RFC1918_RANGES = [
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
]

DEFAULT_MTU = 9000
DEFAULT_PREFIX_LEN = 24


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------


class CX7Topology(str, Enum):
    """Topology classification for CX7 cluster networking."""

    DIRECT = "direct"  # 2 nodes, all interfaces to same peer
    SWITCH = "switch"  # switch-connected or undetermined
    RING = "ring"  # 3 nodes, each connects to 2 others
    UNKNOWN = "unknown"


@dataclass
class CX7Interface:
    """A single CX7/RoCE network interface on a host."""

    name: str
    ip: str
    prefix: int
    subnet: str
    mtu: int
    state: str
    hca: str
    mac: str = ""


@dataclass
class CX7HostDetection:
    """All CX7 state detected on one host."""

    host: str
    interfaces: list[CX7Interface] = field(default_factory=list)
    mgmt_ip: str = ""
    mgmt_iface: str = ""
    used_subnets: set[str] = field(default_factory=set)
    netplan_exists: bool = False
    sudo_ok: bool = False
    detected: bool = False


@dataclass
class CX7InterfaceAssignment:
    """Planned IP assignment for one CX7 interface on one host."""

    iface_name: str
    ip: str
    subnet: str


@dataclass
class CX7HostPlan:
    """Full CX7 plan for one host."""

    host: str
    assignments: list[CX7InterfaceAssignment] = field(default_factory=list)
    needs_change: bool = False
    reason: str = ""


@dataclass
class CX7TopologyResult:
    """Result of topology detection for a CX7 cluster."""

    topology: CX7Topology = CX7Topology.UNKNOWN
    links: list[tuple[str, str, str, str]] = field(default_factory=list)  # [(hostA, ifaceA, hostB, ifaceB), ...]


@dataclass
class CX7ClusterPlan:
    """Complete CX7 plan for the entire cluster."""

    subnet1: ipaddress.IPv4Network | None = None
    subnet2: ipaddress.IPv4Network | None = None
    topology: CX7Topology = CX7Topology.SWITCH
    subnets: list[ipaddress.IPv4Network] = field(default_factory=list)
    mtu: int = DEFAULT_MTU
    prefix_len: int = DEFAULT_PREFIX_LEN
    host_plans: list[CX7HostPlan] = field(default_factory=list)
    all_valid: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def generate_cx7_detect_script() -> str:
    """Load the CX7 detection bash script."""
    return read_script("cx7_detect.sh")


def parse_cx7_detect_output(output: str) -> dict[str, str]:
    """Parse key=value stdout from the CX7 detection script."""
    return parse_kv_output(output)


def build_host_detection(host: str, raw: dict[str, str]) -> CX7HostDetection:
    """Convert parsed detection dict into a ``CX7HostDetection``."""
    detection = CX7HostDetection(host=host)

    if raw.get("CX7_DETECTED") != "1":
        return detection

    detection.detected = True
    detection.mgmt_ip = raw.get("CX7_MGMT_IP", "")
    detection.mgmt_iface = raw.get("CX7_MGMT_IFACE", "")
    detection.netplan_exists = raw.get("CX7_NETPLAN_EXISTS") == "1"
    detection.sudo_ok = raw.get("CX7_SUDO_OK") == "1"

    # Parse used subnets
    used_raw = raw.get("CX7_USED_SUBNETS", "")
    if used_raw:
        detection.used_subnets = {s.strip() for s in used_raw.split(",") if s.strip()}

    # Parse per-interface data
    count = int(raw.get("CX7_IFACE_COUNT", "0"))
    for i in range(count):
        prefix_str = raw.get("CX7_IFACE_%d_PREFIX" % i, "")
        mtu_str = raw.get("CX7_IFACE_%d_MTU" % i, "0")
        iface = CX7Interface(
            name=raw.get("CX7_IFACE_%d_NAME" % i, ""),
            ip=raw.get("CX7_IFACE_%d_IP" % i, ""),
            prefix=int(prefix_str) if prefix_str else 0,
            subnet=raw.get("CX7_IFACE_%d_SUBNET" % i, ""),
            mtu=int(mtu_str) if mtu_str else 0,
            state=raw.get("CX7_IFACE_%d_STATE" % i, ""),
            hca=raw.get("CX7_IFACE_%d_HCA" % i, ""),
            mac=raw.get("CX7_IFACE_%d_MAC" % i, ""),
        )
        detection.interfaces.append(iface)

    return detection


def detect_cx7_for_hosts(
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> dict[str, CX7HostDetection]:
    """Run CX7 detection on all hosts in parallel.

    Returns:
        Dict mapping host -> CX7HostDetection.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not hosts:
        return {}

    kw = ssh_kwargs or {}
    logger.info("Detecting CX7 interfaces on %d host(s)...", len(hosts))
    script = generate_cx7_detect_script()
    results = run_remote_scripts_parallel(
        hosts,
        script,
        timeout=30,
        dry_run=dry_run,
        **kw,
    )

    detections: dict[str, CX7HostDetection] = {}
    for result in results:
        if not result.success:
            logger.warning("CX7 detection failed on %s: %s", result.host, result.stderr[:200])
            detections[result.host] = CX7HostDetection(host=result.host)
            continue
        raw = parse_cx7_detect_output(result.stdout)
        detection = build_host_detection(result.host, raw)
        detections[result.host] = detection
        if detection.detected:
            logger.info("  %s: %d CX7 interface(s), mgmt=%s", result.host, len(detection.interfaces), detection.mgmt_ip)
        else:
            logger.info("  %s: no CX7 interfaces detected", result.host)

    return detections


# ---------------------------------------------------------------------------
# Subnet selection
# ---------------------------------------------------------------------------


def _generate_candidate_subnets(
        used: set[ipaddress.IPv4Network],
) -> list[ipaddress.IPv4Network]:
    """Generate candidate /24 subnets from RFC 1918 ranges, skipping conflicts."""
    candidates: list[ipaddress.IPv4Network] = []
    for private_range in _RFC1918_RANGES:
        for subnet in private_range.subnets(new_prefix=24):
            if any(subnet.overlaps(u) for u in used):
                continue
            candidates.append(subnet)
            # Don't generate the entire 10.0.0.0/8 space -- stop once we have plenty
            if len(candidates) >= 100:
                return candidates
    return candidates


def select_subnets(
        detections: dict[str, CX7HostDetection],
        override1: str | None = None,
        override2: str | None = None,
) -> tuple[ipaddress.IPv4Network, ipaddress.IPv4Network]:
    """Select two /24 subnets for CX7 interfaces.

    Prefers existing common CX7 subnets. Falls back to RFC 1918 selection
    with conflict avoidance.

    Args:
        detections: Per-host detection results.
        override1: User-specified subnet for partition 1.
        override2: User-specified subnet for partition 2.

    Returns:
        Tuple of two IPv4Network objects.

    Raises:
        RuntimeError: If unable to find two suitable subnets.
    """
    # User overrides
    if override1 and override2:
        return (
            ipaddress.IPv4Network(override1, strict=False),
            ipaddress.IPv4Network(override2, strict=False),
        )

    # Collect all non-CX7 used subnets
    all_used: set[ipaddress.IPv4Network] = set()
    for det in detections.values():
        for s in det.used_subnets:
            try:
                all_used.add(ipaddress.IPv4Network(s, strict=False))
            except ValueError:
                continue

    # Collect existing CX7 /24 subnets per host
    host_cx7_subnets: list[set[ipaddress.IPv4Network]] = []
    for det in detections.values():
        if not det.detected:
            continue
        subnets_for_host: set[ipaddress.IPv4Network] = set()
        for iface in det.interfaces:
            if iface.subnet and iface.ip:
                try:
                    net = ipaddress.IPv4Network(iface.subnet, strict=False)
                    if net.prefixlen == 24:
                        subnets_for_host.add(net)
                except ValueError:
                    continue
        if subnets_for_host:
            host_cx7_subnets.append(subnets_for_host)

    # Find common CX7 subnets across ALL hosts
    selected: list[ipaddress.IPv4Network] = []
    if host_cx7_subnets:
        common = host_cx7_subnets[0]
        for other in host_cx7_subnets[1:]:
            common = common.intersection(other)
        # Sort for determinism
        sorted_common = sorted(common, key=lambda n: int(n.network_address))
        for net in sorted_common:
            if net not in selected:
                selected.append(net)
            if len(selected) >= 2:
                break

    # Fill remaining from RFC 1918
    if len(selected) < 2:
        # Include already-selected subnets in the used set for conflict check
        exclude = all_used | set(selected)
        candidates = _generate_candidate_subnets(exclude)
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= 2:
                break

    if len(selected) < 2:
        raise RuntimeError("Could not find 2 available /24 subnets for CX7 configuration")

    # Sort for stable ordering
    selected.sort(key=lambda n: int(n.network_address))
    return selected[0], selected[1]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _mgmt_last_octet(mgmt_ip: str) -> int:
    """Extract the last octet from a management IP address."""
    try:
        return int(mgmt_ip.strip().split(".")[-1])
    except (ValueError, IndexError):
        return 0


def _find_available_octet(taken: set[int], preferred: int) -> int:
    """Find an available last octet, starting from preferred."""
    if preferred not in taken and 1 <= preferred <= 254:
        return preferred
    # Search upward from preferred, then wrap around
    for offset in range(1, 254):
        candidate = (preferred + offset - 1) % 254 + 1  # 1..254
        if candidate not in taken:
            return candidate
    raise RuntimeError("No available octets in /24 subnet (all 254 taken)")


def _is_host_valid(
        det: CX7HostDetection,
        subnet1: ipaddress.IPv4Network,
        subnet2: ipaddress.IPv4Network,
        target_mtu: int,
) -> tuple[bool, str]:
    """Check if a host's existing CX7 config is valid for the cluster.

    Returns:
        Tuple of (is_valid, reason_if_invalid).
    """
    if not det.detected:
        return False, "no CX7 interfaces detected"
    if len(det.interfaces) < 2:
        return False, "need 2 CX7 interfaces, found %d" % len(det.interfaces)

    # Find interfaces on each subnet
    on_subnet1 = [i for i in det.interfaces if i.subnet == str(subnet1)]
    on_subnet2 = [i for i in det.interfaces if i.subnet == str(subnet2)]

    if not on_subnet1:
        return False, "no interface on subnet %s" % subnet1
    if not on_subnet2:
        return False, "no interface on subnet %s" % subnet2

    # Check MTU
    for iface in on_subnet1 + on_subnet2:
        if iface.mtu != target_mtu:
            return False, "interface %s has MTU %d, need %d" % (iface.name, iface.mtu, target_mtu)

    return True, ""


def plan_cluster_cx7(
        detections: dict[str, CX7HostDetection],
        subnet1: ipaddress.IPv4Network,
        subnet2: ipaddress.IPv4Network,
        mtu: int = DEFAULT_MTU,
        force: bool = False,
) -> CX7ClusterPlan:
    """Build a complete CX7 configuration plan for all hosts.

    Uses a two-pass approach:
    1. Collect existing valid IPs (preserves working configurations)
    2. Assign new IPs from management IP last octet for hosts needing config

    Args:
        detections: Per-host detection results.
        subnet1: First /24 subnet.
        subnet2: Second /24 subnet.
        mtu: Target MTU (default 9000).
        force: Reconfigure even if existing config is valid.

    Returns:
        CX7ClusterPlan with per-host plans.
    """
    plan = CX7ClusterPlan(
        subnet1=subnet1,
        subnet2=subnet2,
        mtu=mtu,
        prefix_len=DEFAULT_PREFIX_LEN,
    )

    # Track taken octets per subnet
    taken_s1: set[int] = set()
    taken_s2: set[int] = set()

    # --- Pass 1: Collect existing valid IPs ---
    for host in sorted(detections.keys()):
        det = detections[host]
        if not det.detected:
            continue
        if force:
            continue  # Skip preservation when forcing

        valid, reason = _is_host_valid(det, subnet1, subnet2, mtu)
        if not valid:
            continue

        # Record existing IPs as taken
        for iface in det.interfaces:
            if iface.subnet == str(subnet1) and iface.ip:
                octet = int(iface.ip.split(".")[-1])
                taken_s1.add(octet)
            elif iface.subnet == str(subnet2) and iface.ip:
                octet = int(iface.ip.split(".")[-1])
                taken_s2.add(octet)

    # --- Pass 2: Build per-host plans ---
    all_valid = True
    for host in sorted(detections.keys()):
        det = detections[host]
        host_plan = CX7HostPlan(host=host)

        if not det.detected:
            host_plan.needs_change = True
            host_plan.reason = "no CX7 interfaces detected"
            plan.errors.append("%s: no CX7 interfaces detected" % host)
            all_valid = False
            plan.host_plans.append(host_plan)
            continue

        if len(det.interfaces) < 2:
            host_plan.needs_change = True
            host_plan.reason = "need 2 CX7 interfaces, found %d" % len(det.interfaces)
            plan.errors.append("%s: need 2 CX7 interfaces, found %d" % (host, len(det.interfaces)))
            all_valid = False
            plan.host_plans.append(host_plan)
            continue

        valid, reason = _is_host_valid(det, subnet1, subnet2, mtu)

        if valid and not force:
            # Existing config is valid -- preserve it
            host_plan.needs_change = False
            host_plan.reason = "already configured"
            # Record assignments from existing config
            for iface in det.interfaces:
                if iface.subnet == str(subnet1):
                    host_plan.assignments.append(
                        CX7InterfaceAssignment(
                            iface_name=iface.name,
                            ip=iface.ip,
                            subnet=str(subnet1),
                        )
                    )
                elif iface.subnet == str(subnet2):
                    host_plan.assignments.append(
                        CX7InterfaceAssignment(
                            iface_name=iface.name,
                            ip=iface.ip,
                            subnet=str(subnet2),
                        )
                    )
        else:
            # Need to assign IPs
            host_plan.needs_change = True
            host_plan.reason = reason if not force else "forced reconfiguration"
            all_valid = False

            preferred_octet = _mgmt_last_octet(det.mgmt_ip)
            if preferred_octet == 0:
                plan.warnings.append("%s: could not derive last octet from mgmt IP '%s'" % (host, det.mgmt_ip))
                preferred_octet = 1

            # Assign IP on subnet1
            # Use the first two interfaces (sorted by name for determinism).
            # On DGX Spark, this picks a consistent pair regardless of which
            # partition/port combination is active (1x200G vs 2x100G).
            sorted_ifaces = sorted(det.interfaces, key=lambda i: i.name.lower())
            if len(sorted_ifaces) > 2:
                plan.warnings.append(
                    "%s: %d CX7 interfaces found, using first 2: %s, %s"
                    % (host, len(sorted_ifaces), sorted_ifaces[0].name, sorted_ifaces[1].name)
                )
            octet1 = _find_available_octet(taken_s1, preferred_octet)
            taken_s1.add(octet1)
            ip1 = str(subnet1.network_address + octet1)
            if octet1 != preferred_octet:
                plan.warnings.append("%s: octet %d taken on %s, using %d instead" % (host, preferred_octet, subnet1, octet1))

            # Assign IP on subnet2
            octet2 = _find_available_octet(taken_s2, preferred_octet)
            taken_s2.add(octet2)
            ip2 = str(subnet2.network_address + octet2)
            if octet2 != preferred_octet:
                plan.warnings.append("%s: octet %d taken on %s, using %d instead" % (host, preferred_octet, subnet2, octet2))

            host_plan.assignments = [
                CX7InterfaceAssignment(
                    iface_name=sorted_ifaces[0].name,
                    ip=ip1,
                    subnet=str(subnet1),
                ),
                CX7InterfaceAssignment(
                    iface_name=sorted_ifaces[1].name,
                    ip=ip2,
                    subnet=str(subnet2),
                ),
            ]

        # Check sudo
        if host_plan.needs_change and not det.sudo_ok:
            plan.warnings.append("%s: passwordless sudo not available" % host)

        plan.host_plans.append(host_plan)

    plan.all_valid = all_valid
    return plan


# ---------------------------------------------------------------------------
# Topology detection
# ---------------------------------------------------------------------------


def _parse_arping_output(output: str) -> list[tuple[str, str]]:
    """Parse CX7_NEIGHBOR_N_LOCAL_IFACE / CX7_NEIGHBOR_N_REMOTE_MAC from arping output.

    Returns:
        List of (local_iface, remote_mac) tuples.
    """
    raw = parse_kv_output(output)
    count = int(raw.get("CX7_NEIGHBOR_COUNT", "0"))
    neighbors: list[tuple[str, str]] = []
    for i in range(count):
        local_iface = raw.get("CX7_NEIGHBOR_%d_LOCAL_IFACE" % i, "")
        remote_mac = raw.get("CX7_NEIGHBOR_%d_REMOTE_MAC" % i, "")
        if local_iface and remote_mac:
            neighbors.append((local_iface, remote_mac))
    return neighbors


def detect_switch(
        detections: dict[str, CX7HostDetection],
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        sudo_password: str | None = None,
) -> bool | None:
    """Detect whether CX7 interfaces are connected through a switch.

    Primary method: cross-subnet L2 arping.  In a direct connection each
    cable is its own L2 segment — an interface on subnet1 cannot reach
    interfaces on subnet2 at L2.  Through a switch, all interfaces share
    the same L2 fabric and CAN reach across subnets via ARP.

    Fallback: listens for STP BPDUs / LLDP frames (less reliable — some
    switches don't send these).

    Args:
        detections: Per-host CX7 detection results.
        hosts: List of host identifiers.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
        sudo_password: Password for hosts requiring interactive sudo.

    Returns:
        True if switch detected, False if no switch frames seen,
        None if detection was not possible (dry-run, no sudo, etc.).
    """
    from sparkrun.orchestration.ssh import run_remote_script, run_remote_sudo_script

    kw = ssh_kwargs or {}

    # Pick first host with CX7 interfaces that we can sudo on
    target_host = None
    target_det: CX7HostDetection | None = None
    needs_sudo_pw = False
    for host in hosts:
        det = detections.get(host)
        if det and det.detected and det.interfaces:
            if det.sudo_ok:
                target_host = host
                target_det = det
                break
            elif sudo_password:
                target_host = host
                target_det = det
                needs_sudo_pw = True
                break

    if not target_host or not target_det:
        logger.debug("No host available for switch detection (no sudo access)")
        return None

    if dry_run:
        logger.info("[dry-run] Would detect switch topology on %s", target_host)
        return None

    # Build cross-subnet test parameters:
    # Pick a local interface on subnet1, and a remote IP on subnet2.
    # If the arping succeeds, the L2 segments are shared (= switch).
    local_iface = ""
    remote_ip = ""

    # Find a remote host with configured CX7 interfaces on different subnets
    local_subnets: dict[str, str] = {}  # subnet -> iface_name
    for iface in target_det.interfaces:
        if iface.ip and iface.subnet:
            local_subnets[iface.subnet] = iface.name

    if len(local_subnets) >= 2:
        # Pick the first subnet's interface as local
        subnets = sorted(local_subnets.keys())
        local_iface = local_subnets[subnets[0]]
        target_subnet = subnets[1]  # We want a remote IP on this subnet

        # Find a remote host's IP on target_subnet
        for other_host in hosts:
            if other_host == target_host:
                continue
            other_det = detections.get(other_host)
            if not other_det or not other_det.detected:
                continue
            for iface in other_det.interfaces:
                if iface.ip and iface.subnet == target_subnet:
                    remote_ip = iface.ip
                    break
            if remote_ip:
                break

    # Build script with env vars
    script = read_script("cx7_switch_detect.sh")
    env_lines = []
    if local_iface:
        env_lines.append("export CX7_LOCAL_IFACE='%s'" % local_iface)
    if remote_ip:
        env_lines.append("export CX7_REMOTE_IP='%s'" % remote_ip)
    # Fallback interfaces for BPDU detection
    ifaces_str = " ".join(iface.name for iface in target_det.interfaces[:2])
    env_lines.append("export CX7_IFACES='%s'" % ifaces_str)

    full_script = "\n".join(env_lines) + "\n" + script

    logger.info("Checking for switch on %s (local=%s, remote_target=%s)...", target_host, local_iface, remote_ip)
    if needs_sudo_pw:
        r = run_remote_sudo_script(target_host, full_script, sudo_password, timeout=30, dry_run=dry_run, **kw)
    else:
        r = run_remote_script(target_host, full_script, timeout=30, dry_run=dry_run, **kw)

    if not r.success:
        logger.warning("Switch detection failed on %s: %s", target_host, r.stderr[:200])
        return None

    raw = parse_kv_output(r.stdout)
    val = raw.get("CX7_SWITCH_DETECTED", "")
    if val == "1":
        return True
    elif val == "0":
        return False
    else:
        return None  # tools not available or other issue


def classify_topology(
        links: list[tuple[str, str, str, str]],
        hosts: list[str],
) -> CX7Topology:
    """Classify topology from discovered links.

    Detection logic:
    - 1 host: UNKNOWN
    - 2 hosts: SWITCH (cannot reliably distinguish direct vs switch
      with only 2 hosts — neighbor discovery shows the same pattern).
    - 3 hosts where each host connects to exactly 2 others: RING.
    - Otherwise: SWITCH.

    Args:
        links: List of (hostA, ifaceA, hostB, ifaceB) link tuples.
        hosts: List of host identifiers.

    Returns:
        CX7Topology enum value.
    """
    n_hosts = len(hosts)
    if n_hosts < 2:
        return CX7Topology.UNKNOWN

    if n_hosts == 2:
        return CX7Topology.SWITCH

    if n_hosts == 3:
        # Check if each host connects to exactly 2 other hosts
        peer_map: dict[str, set[str]] = {h: set() for h in hosts}
        for hostA, _ifA, hostB, _ifB in links:
            if hostA in peer_map and hostB in peer_map:
                peer_map[hostA].add(hostB)
                peer_map[hostB].add(hostA)

        if all(len(peers) == 2 for peers in peer_map.values()):
            return CX7Topology.RING

    return CX7Topology.SWITCH


def detect_topology(
        detections: dict[str, CX7HostDetection],
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> CX7TopologyResult:
    """Detect CX7 topology via MAC/ARP neighbor discovery.

    Phase 1: Bring up all CX7 interfaces (link-local).
    Phase 2: Build MAC -> (host, iface) lookup from detection data.
    Phase 3: Run arping-based neighbor discovery on all hosts.
    Phase 4: Match neighbor MACs to build link graph.
    Phase 5: Classify topology.

    Args:
        detections: Per-host CX7 detection results (must include MAC addresses).
        hosts: List of host identifiers.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.

    Returns:
        CX7TopologyResult with topology classification and link list.
    """
    kw = ssh_kwargs or {}
    result = CX7TopologyResult()

    # Phase 1 — Bringup: ensure all CX7 interfaces are link-up
    all_iface_names: dict[str, list[str]] = {}
    for host in hosts:
        det = detections.get(host)
        if det and det.detected:
            all_iface_names[host] = [iface.name for iface in det.interfaces]

    if not all_iface_names:
        return result

    bringup_script = read_script("cx7_bringup.sh")
    bringup_hosts = list(all_iface_names.keys())
    bringup_scripts = []
    for host in bringup_hosts:
        ifaces_str = " ".join(all_iface_names[host])
        bringup_scripts.append("export CX7_IFACES='%s'\n%s" % (ifaces_str, bringup_script))

    if not dry_run:
        logger.info("Bringing up CX7 interfaces on %d host(s)...", len(bringup_hosts))
        # Run bringup per-host with custom scripts
        for i, host in enumerate(bringup_hosts):
            from sparkrun.orchestration.ssh import run_remote_script

            run_remote_script(host, bringup_scripts[i], timeout=30, dry_run=dry_run, **kw)

    # Phase 2 — Build MAC lookup: mac -> (host, iface_name)
    mac_lookup: dict[str, tuple[str, str]] = {}
    for host in hosts:
        det = detections.get(host)
        if not det or not det.detected:
            continue
        for iface in det.interfaces:
            if iface.mac:
                mac_lookup[iface.mac.lower()] = (host, iface.name)

    if not mac_lookup:
        logger.warning("No MAC addresses available for topology detection")
        return result

    # Phase 3 — Neighbor discovery via arping
    arping_script = read_script("cx7_arping.sh")
    arping_hosts = list(all_iface_names.keys())
    arping_scripts = []
    for host in arping_hosts:
        ifaces_str = " ".join(all_iface_names[host])
        arping_scripts.append("export CX7_IFACES='%s'\n%s" % (ifaces_str, arping_script))

    host_neighbors: dict[str, list[tuple[str, str]]] = {}
    if not dry_run:
        logger.info("Running neighbor discovery on %d host(s)...", len(arping_hosts))
        for i, host in enumerate(arping_hosts):
            from sparkrun.orchestration.ssh import run_remote_script

            r = run_remote_script(host, arping_scripts[i], timeout=30, dry_run=dry_run, **kw)
            if r.success:
                host_neighbors[host] = _parse_arping_output(r.stdout)
            else:
                logger.warning("  %s: arping failed: %s", host, r.stderr[:200])
    else:
        logger.info("[dry-run] Would run neighbor discovery on %d host(s)", len(arping_hosts))

    # Phase 4 — Build link graph
    links: list[tuple[str, str, str, str]] = []

    for hostA, neighbors in host_neighbors.items():
        for local_iface, remote_mac in neighbors:
            remote_mac_lower = remote_mac.lower()
            if remote_mac_lower not in mac_lookup:
                continue
            hostB, ifaceB = mac_lookup[remote_mac_lower]
            if hostA == hostB:
                continue  # Skip self-links
            links.append((hostA, local_iface, hostB, ifaceB))

    # Phase 5 — Classify
    result.links = links
    result.topology = classify_topology(links, hosts)
    logger.info("Detected topology: %s (%d links)", result.topology.value, len(links))
    return result


def select_subnets_for_topology(
        detections: dict[str, CX7HostDetection],
        topology: CX7Topology,
        count: int = 2,
        override1: str | None = None,
        override2: str | None = None,
) -> list[ipaddress.IPv4Network]:
    """Select subnets based on topology requirements.

    For RING topology: 6 subnets (2 per link, 3 links).
    For DIRECT/SWITCH: 2 subnets (backward compatible).

    Args:
        detections: Per-host detection results.
        topology: Detected or specified topology.
        count: Number of subnets needed (auto-determined from topology if not overridden).
        override1: User-specified subnet 1 (only for 2-subnet topologies).
        override2: User-specified subnet 2 (only for 2-subnet topologies).

    Returns:
        List of IPv4Network subnets.
    """
    if topology == CX7Topology.RING:
        count = 6
    else:
        count = 2

    # For 2-subnet case, delegate to existing function
    if count == 2:
        s1, s2 = select_subnets(detections, override1=override1, override2=override2)
        return [s1, s2]

    # For ring (6 subnets), select from RFC 1918 avoiding conflicts
    all_used: set[ipaddress.IPv4Network] = set()
    for det in detections.values():
        for s in det.used_subnets:
            try:
                all_used.add(ipaddress.IPv4Network(s, strict=False))
            except ValueError:
                continue

    candidates = _generate_candidate_subnets(all_used)
    selected: list[ipaddress.IPv4Network] = []
    for candidate in candidates:
        selected.append(candidate)
        if len(selected) >= count:
            break

    if len(selected) < count:
        raise RuntimeError("Could not find %d available /24 subnets for ring CX7 configuration" % count)

    return selected


def _group_interfaces_by_port(interfaces: list[CX7Interface]) -> list[list[CX7Interface]]:
    """Group CX7 interfaces by physical port.

    DGX Spark interface names end with ``npN`` where *N* identifies
    the physical network port.  For example ``enp1s0f0np0`` and
    ``enP2p1s0f0np0`` both end with ``np0`` and share physical port 0.

    Falls back to grouping in sorted pairs if the ``npN`` suffix is
    not present on any interface.
    """
    import re

    if len(interfaces) <= 2:
        return [interfaces]

    np_pattern = re.compile(r"np(\d+)$")

    groups_by_port: dict[str, list[CX7Interface]] = {}
    ungrouped: list[CX7Interface] = []
    for iface in interfaces:
        match = np_pattern.search(iface.name)
        if match:
            port = match.group(1)
            groups_by_port.setdefault(port, []).append(iface)
        else:
            ungrouped.append(iface)

    if groups_by_port:
        result: list[list[CX7Interface]] = []
        for port in sorted(groups_by_port.keys()):
            result.append(sorted(groups_by_port[port], key=lambda i: i.name.lower()))
        for iface in ungrouped:
            result.append([iface])
        return result

    # Fallback: sorted pairs
    sorted_ifaces = sorted(interfaces, key=lambda i: i.name.lower())
    return [sorted_ifaces[i: i + 2] for i in range(0, len(sorted_ifaces), 2)]


def plan_ring_cx7(
        detections: dict[str, CX7HostDetection],
        topology_result: CX7TopologyResult,
        subnets: list[ipaddress.IPv4Network],
        mtu: int = DEFAULT_MTU,
        force: bool = False,
) -> CX7ClusterPlan:
    """Build CX7 configuration plan for a 3-node ring topology.

    The ring has 3 links. Each link uses 2 subnets (one per partition on that cable):
    - Link 0 (NodeA.Port0 <-> NodeB.Port1): subnets[0], subnets[1]
    - Link 1 (NodeB.Port0 <-> NodeC.Port1): subnets[2], subnets[3]
    - Link 2 (NodeC.Port0 <-> NodeA.Port1): subnets[4], subnets[5]

    Each interface gets IP .1 or .2 based on which end of the link it is.

    Args:
        detections: Per-host CX7 detection results.
        topology_result: Topology detection result with link information.
        subnets: 6 /24 subnets for ring configuration.
        mtu: Target MTU.
        force: Reconfigure even if existing config is valid.

    Returns:
        CX7ClusterPlan with per-host plans for 4 interfaces each.
    """
    plan = CX7ClusterPlan(
        subnet1=subnets[0] if subnets else None,
        subnet2=subnets[1] if len(subnets) > 1 else None,
        topology=CX7Topology.RING,
        subnets=list(subnets),
        mtu=mtu,
        prefix_len=DEFAULT_PREFIX_LEN,
    )

    if len(subnets) < 6:
        plan.errors.append("Ring topology requires 6 subnets, got %d" % len(subnets))
        return plan

    # Build ordered link list from topology result.
    # We need to identify which port on each host connects to which peer.
    # From the topology_result.links, build adjacency: host -> [(peer, local_iface, remote_iface)]
    host_links: dict[str, list[tuple[str, str, str]]] = {}
    for hostA, ifaceA, hostB, ifaceB in topology_result.links:
        host_links.setdefault(hostA, []).append((hostB, ifaceA, ifaceB))
        host_links.setdefault(hostB, []).append((hostA, ifaceB, ifaceA))

    hosts = sorted(detections.keys())
    hosts_with_cx7 = [h for h in hosts if detections[h].detected]

    if len(hosts_with_cx7) != 3:
        plan.errors.append("Ring topology requires exactly 3 hosts with CX7, found %d" % len(hosts_with_cx7))
        return plan

    # Ring requires 2 physical ports per host (4 interfaces) — one port per peer
    for host in hosts_with_cx7:
        det = detections[host]
        port_groups = _group_interfaces_by_port(det.interfaces)
        if len(port_groups) < 2:
            plan.errors.append(
                "%s: ring topology requires 2 physical ports (4 interfaces), "
                "but only %d port group(s) found (%d interfaces)"
                % (host, len(port_groups), len(det.interfaces))
            )
    if plan.errors:
        return plan

    # Order hosts into a ring: A -> B -> C -> A
    # Start with first host, find its two peers, order them consistently
    ring_hosts = [hosts_with_cx7[0]]
    remaining = set(hosts_with_cx7[1:])

    for _ in range(2):
        current = ring_hosts[-1]
        peers = host_links.get(current, [])
        for peer_host, _, _ in peers:
            if peer_host in remaining:
                ring_hosts.append(peer_host)
                remaining.discard(peer_host)
                break
        else:
            # If no link found, just pick from remaining
            if remaining:
                ring_hosts.append(remaining.pop())

    # Now we have ring_hosts = [A, B, C] where A-B, B-C, C-A are the links
    # Assign subnets to links:
    # Link 0: A <-> B uses subnets[0], subnets[1]
    # Link 1: B <-> C uses subnets[2], subnets[3]
    # Link 2: C <-> A uses subnets[4], subnets[5]

    link_pairs = [
        (ring_hosts[0], ring_hosts[1]),
        (ring_hosts[1], ring_hosts[2]),
        (ring_hosts[2], ring_hosts[0]),
    ]

    # For each link, determine which interfaces connect
    # Build assignment map: host -> list of (iface_name, ip, subnet)
    assignments: dict[str, list[CX7InterfaceAssignment]] = {h: [] for h in ring_hosts}

    for link_idx, (hostA, hostB) in enumerate(link_pairs):
        subnet_a = subnets[link_idx * 2]
        subnet_b = subnets[link_idx * 2 + 1]

        # Find which interfaces connect these two hosts
        ifaceA = None
        ifaceB = None
        for peer, local_if, remote_if in host_links.get(hostA, []):
            if peer == hostB:
                ifaceA = local_if
                ifaceB = remote_if
                break

        if not ifaceA or not ifaceB:
            # Fallback: use interface ordering
            detA = detections[hostA]
            detB = detections[hostB]
            sorted_a = sorted(detA.interfaces, key=lambda i: i.name.lower())
            sorted_b = sorted(detB.interfaces, key=lambda i: i.name.lower())
            # Use port based on link index within this host's connections
            a_idx = min(link_idx, len(sorted_a) - 1)
            b_idx = min(link_idx, len(sorted_b) - 1)
            ifaceA = sorted_a[a_idx].name if sorted_a else "unknown"
            ifaceB = sorted_b[b_idx].name if sorted_b else "unknown"
            plan.warnings.append(
                "Link %d (%s <-> %s): no link data, using interface ordering" % (link_idx, hostA, hostB)
            )

        # Host A gets .1 on both subnets for this link
        # Host B gets .2 on both subnets for this link
        ip_a1 = str(subnet_a.network_address + 1)
        ip_b1 = str(subnet_a.network_address + 2)

        # The second subnet in the pair is for the second partition on the same cable
        # Find the partner partition interfaces
        # For now, assign the primary interfaces to the first subnet
        assignments[hostA].append(CX7InterfaceAssignment(iface_name=ifaceA, ip=ip_a1, subnet=str(subnet_a)))
        assignments[hostB].append(CX7InterfaceAssignment(iface_name=ifaceB, ip=ip_b1, subnet=str(subnet_a)))

        # Find partner partition interfaces (second interface on same port)
        detA = detections[hostA]
        detB = detections[hostB]
        port_groups_a = _group_interfaces_by_port(detA.interfaces)
        port_groups_b = _group_interfaces_by_port(detB.interfaces)

        partnerA = None
        for group in port_groups_a:
            names = [i.name for i in group]
            if ifaceA in names and len(group) > 1:
                partnerA = [i.name for i in group if i.name != ifaceA][0]
                break

        partnerB = None
        for group in port_groups_b:
            names = [i.name for i in group]
            if ifaceB in names and len(group) > 1:
                partnerB = [i.name for i in group if i.name != ifaceB][0]
                break

        if partnerA and partnerB:
            ip_a2 = str(subnet_b.network_address + 1)
            ip_b2 = str(subnet_b.network_address + 2)
            assignments[hostA].append(CX7InterfaceAssignment(iface_name=partnerA, ip=ip_a2, subnet=str(subnet_b)))
            assignments[hostB].append(CX7InterfaceAssignment(iface_name=partnerB, ip=ip_b2, subnet=str(subnet_b)))
        else:
            plan.warnings.append(
                "Link %d: could not find partner partitions for %s/%s" % (link_idx, ifaceA, ifaceB)
            )

    # Build host plans
    all_valid = True
    for host in sorted(detections.keys()):
        det = detections[host]
        host_plan = CX7HostPlan(host=host)

        if not det.detected:
            host_plan.needs_change = True
            host_plan.reason = "no CX7 interfaces detected"
            plan.errors.append("%s: no CX7 interfaces detected" % host)
            all_valid = False
        elif host in assignments and assignments[host]:
            host_plan.assignments = assignments[host]
            host_plan.needs_change = True
            host_plan.reason = "ring topology configuration" if force else "ring topology configuration needed"
            all_valid = False
        else:
            host_plan.needs_change = True
            host_plan.reason = "no ring assignments computed"
            plan.errors.append("%s: no ring assignments computed" % host)
            all_valid = False

        if host_plan.needs_change and not det.sudo_ok:
            plan.warnings.append("%s: passwordless sudo not available" % host)

        plan.host_plans.append(host_plan)

    plan.all_valid = all_valid
    return plan


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def generate_cx7_configure_script(host_plan: CX7HostPlan, mtu: int, prefix_len: int) -> str:
    """Generate the netplan configuration script for one host.

    Supports both 2-interface (direct/switch) and N-interface (ring) plans.

    Args:
        host_plan: The plan for this host (must have >= 2 assignments).
        mtu: Target MTU.
        prefix_len: Subnet prefix length.

    Returns:
        Bash script content ready to pipe via SSH.
    """
    if len(host_plan.assignments) < 2:
        raise ValueError("Expected at least 2 interface assignments for %s, got %d" % (host_plan.host, len(host_plan.assignments)))

    if len(host_plan.assignments) == 2:
        # Use the original template for 2-interface case
        template = read_script("cx7_configure.sh")
        a1, a2 = host_plan.assignments[0], host_plan.assignments[1]
        return template.format(
            adapter1=a1.iface_name,
            adapter2=a2.iface_name,
            ip1=a1.ip,
            ip2=a2.ip,
            mtu=mtu,
            prefix_len=prefix_len,
        )

    # Dynamic netplan generation for N interfaces (ring topology)
    return _generate_dynamic_configure_script(host_plan, mtu, prefix_len)


def _generate_dynamic_configure_script(host_plan: CX7HostPlan, mtu: int, prefix_len: int) -> str:
    """Generate a netplan configuration script for N CX7 interfaces.

    Builds netplan YAML in Python and embeds it in a bash script that
    writes and applies the configuration.
    """
    # Build netplan ethernets dict
    ethernets_lines = []
    for a in host_plan.assignments:
        ethernets_lines.append(
            "    %s:\n"
            "      dhcp4: no\n"
            "      dhcp6: no\n"
            "      link-local: []\n"
            "      mtu: %d\n"
            "      addresses: [%s/%d]" % (a.iface_name, mtu, a.ip, prefix_len)
        )

    netplan_content = "network:\n  version: 2\n  ethernets:\n" + "\n".join(ethernets_lines) + "\n"

    # Build summary for logging
    summary_lines = []
    for a in host_plan.assignments:
        summary_lines.append('echo "  %s -> %s/%d (MTU %d)" >&2' % (a.iface_name, a.ip, prefix_len, mtu))

    script = (
                 "#!/bin/bash\n"
                 "set -euo pipefail\n"
                 "\n"
                 'echo "Configuring %d CX7 interfaces:" >&2\n'
                 "%s\n"
                 "\n"
                 "sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<'NETPLAN_EOF'\n"
                 "%s"
                 "NETPLAN_EOF\n"
                 "\n"
                 "sudo chmod 600 /etc/netplan/40-cx7.yaml\n"
                 'echo "Applying netplan configuration..." >&2\n'
                 "sudo netplan apply\n"
                 "\n"
                 'echo "Verifying configuration..." >&2\n'
             ) % (len(host_plan.assignments), "\n".join(summary_lines), netplan_content)

    # Add verification for each interface
    for a in host_plan.assignments:
        script += 'ip -4 addr show "%s" 2>/dev/null | head -3 >&2\n' % a.iface_name

    script += 'echo "CX7_CONFIGURED=1"\n'
    return script


def configure_cx7_host(
        host_plan: CX7HostPlan,
        mtu: int,
        prefix_len: int,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        sudo_password: str | None = None,
):
    """Apply CX7 netplan configuration to a single host.

    Args:
        host_plan: Plan for this host (must have at least 2 assignments).
        mtu: Target MTU.
        prefix_len: Subnet prefix length.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
        sudo_password: If set, run the script via ``sudo -S bash -s``
            with the password prepended to stdin.  Only use for hosts
            that do **not** have passwordless sudo.

    Returns:
        RemoteResult with the outcome.
    """
    from sparkrun.orchestration.ssh import run_remote_script, run_remote_sudo_script

    script = generate_cx7_configure_script(host_plan, mtu, prefix_len)
    kw = ssh_kwargs or {}

    if sudo_password:
        return run_remote_sudo_script(
            host_plan.host,
            script,
            sudo_password,
            timeout=60,
            dry_run=dry_run,
            **kw,
        )
    else:
        return run_remote_script(host_plan.host, script, timeout=60, dry_run=dry_run, **kw)


def apply_cx7_plan(
        plan: CX7ClusterPlan,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        sudo_password: str | None = None,
        sudo_hosts: set[str] | None = None,
) -> list:
    """Apply CX7 configuration to all hosts that need changes.

    Args:
        plan: The cluster plan to apply.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
        sudo_password: Password for hosts requiring interactive sudo.
        sudo_hosts: Set of hostnames that need ``sudo -S`` (no NOPASSWD).
            Hosts not in this set use the normal ``run_remote_script``
            path where the script's internal ``sudo`` calls rely on NOPASSWD.

    Returns:
        List of RemoteResult for hosts that were configured.
    """
    kw = ssh_kwargs or {}
    results = []
    sudo_hosts = sudo_hosts or set()

    hosts_to_configure = [hp for hp in plan.host_plans if hp.needs_change and len(hp.assignments) >= 2]
    if not hosts_to_configure:
        logger.info("No hosts need CX7 configuration changes")
        return results

    logger.info("Applying CX7 configuration to %d host(s)...", len(hosts_to_configure))
    for hp in hosts_to_configure:
        host_pw = sudo_password if hp.host in sudo_hosts else None
        result = configure_cx7_host(
            hp,
            plan.mtu,
            plan.prefix_len,
            ssh_kwargs=kw,
            dry_run=dry_run,
            sudo_password=host_pw,
        )
        results.append(result)
        if result.success:
            logger.info("  [OK] %s: configured", hp.host)
        else:
            logger.error("  [FAIL] %s: %s", hp.host, result.stderr[:200])

    return results


def verify_cx7_config(
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> dict[str, CX7HostDetection]:
    """Re-run CX7 detection to verify configuration was applied."""
    return detect_cx7_for_hosts(hosts, ssh_kwargs=ssh_kwargs, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Host key distribution
# ---------------------------------------------------------------------------


def discover_host_network_ips(
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> dict[str, list[str]]:
    """Discover additional network IPs (IB/CX7) for a set of hosts.

    Runs IB detection on each host and collects all IB IPs that differ
    from the management host identifiers.  Filters out 127.0.0.1.

    Args:
        hosts: Management hostnames/IPs.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.

    Returns:
        Dict mapping management host -> list of additional IPs.
    """
    from sparkrun.orchestration.infiniband import (
        extract_ib_ips,
        generate_ib_detect_script,
        parse_ib_detect_output,
    )
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not hosts:
        return {}

    host_set = set(hosts)
    kw = ssh_kwargs or {}

    logger.info("Discovering additional network IPs on %d host(s)...", len(hosts))
    ib_script = generate_ib_detect_script()
    ib_results = run_remote_scripts_parallel(
        hosts,
        ib_script,
        timeout=30,
        dry_run=dry_run,
        **kw,
    )

    discovered: dict[str, list[str]] = {}
    for result in ib_results:
        if not result.success:
            continue
        ib_info = parse_ib_detect_output(result.stdout)
        ib_ips = extract_ib_ips(ib_info)
        # Filter out management IPs already in host list and loopback
        extra = [ip for ip in ib_ips if ip not in host_set and ip != "127.0.0.1"]
        if extra:
            discovered[result.host] = extra

    return discovered


def distribute_host_keys(
        ips: list[str],
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> list:
    """Scan IPs and add to ``known_hosts`` on control machine and all hosts.

    After network configuration (CX7, IB) the new IPs are unknown SSH
    endpoints.  This runs ``ssh-keyscan`` to register their host keys so
    that transfers and inter-node SSH over additional networks succeed
    without host-key-verification prompts.

    Args:
        ips: All additional IPs across the cluster.
        hosts: Management-IP host list (used to reach each host via SSH).
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.

    Returns:
        List of RemoteResult from the remote keyscan step.
    """
    import subprocess

    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not ips:
        return []

    ip_list = " ".join(ips)
    script = (
                 "#!/bin/bash\n"
                 "set -uo pipefail\n"
                 "mkdir -p ~/.ssh\n"
                 "touch ~/.ssh/known_hosts\n"
                 "ADDED=0\n"
                 "for ip in %s; do\n"
                 '    keys=$(ssh-keyscan -H "$ip" 2>/dev/null)\n'
                 '    if [ -n "$keys" ]; then\n'
                 '        echo "$keys" >> ~/.ssh/known_hosts\n'
                 "        ADDED=$((ADDED + 1))\n"
                 "    fi\n"
                 "done\n"
                 "sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts\n"
                 'echo "KEYSCAN_ADDED=$ADDED"\n'
             ) % ip_list

    # Local keyscan (control machine)
    if not dry_run:
        try:
            subprocess.run(
                ["bash", "-c", script],
                timeout=30,
                capture_output=True,
                text=True,
            )
            logger.info("  local: host keys added to known_hosts")
        except Exception as e:
            logger.warning("  local: keyscan failed: %s", e)
    else:
        logger.info("[dry-run] Would scan %d IPs locally", len(ips))

    # Remote keyscan on all hosts (via management IPs)
    kw = ssh_kwargs or {}
    results = run_remote_scripts_parallel(hosts, script, dry_run=dry_run, **kw)

    for r in results:
        if r.success:
            logger.info("  %s: host keys added to known_hosts", r.host)
        else:
            logger.warning("  %s: keyscan failed: %s", r.host, r.stderr.strip()[:100])

    return results


# Backward-compatible alias (used by setup_cx7 command)
distribute_cx7_host_keys = distribute_host_keys


# ---------------------------------------------------------------------------
# CX7 peer discovery
# ---------------------------------------------------------------------------


def discover_cx7_peers(
        subnets: list[str],
        exclude_ips: list[str] | None = None,
        timeout: int = 15,
) -> list[str]:
    """Discover peer hosts on CX7 subnets via ARP table + ping sweep.

    Runs locally (no SSH). Pings the /24 broadcast to populate the ARP
    table, then reads ARP entries for the given subnets. Returns a list
    of discovered peer IPs (excluding local IPs).

    Args:
        subnets: List of CIDR subnet strings (e.g. ``["192.168.11.0/24"]``).
        exclude_ips: Additional IPs to exclude from results.
        timeout: Subprocess timeout in seconds.

    Returns:
        List of discovered peer IP addresses.
    """
    import subprocess

    if not subnets:
        return []

    script = read_script("cx7_discover_peers.sh")
    try:
        result = subprocess.run(
            ["bash", "-s"] + subnets,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("CX7 peer discovery timed out after %ds", timeout)
        return []

    if result.returncode != 0:
        logger.warning("CX7 peer discovery failed: %s", result.stderr[:200])
        return []

    parsed = parse_kv_output(result.stdout)
    count = int(parsed.get("PEER_COUNT", "0"))
    exclude = set(exclude_ips or [])
    peers = []
    for i in range(count):
        ip = parsed.get("PEER_%d_IP" % i, "")
        if ip and ip not in exclude:
            peers.append(ip)

    return peers
