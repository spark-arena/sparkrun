"""CX7 network interface detection, planning, and configuration.

Detects ConnectX-7 interfaces on DGX Spark hosts via SSH, plans static
IP assignments across the cluster, and applies netplan configuration.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass, field

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
# Dataclasses
# ---------------------------------------------------------------------------


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
class CX7ClusterPlan:
    """Complete CX7 plan for the entire cluster."""

    subnet1: ipaddress.IPv4Network | None = None
    subnet2: ipaddress.IPv4Network | None = None
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
                plan.warnings.append(
                    "%s: octet %d taken on %s, using %d instead" % (host, preferred_octet, subnet1, octet1)
                )

            # Assign IP on subnet2
            octet2 = _find_available_octet(taken_s2, preferred_octet)
            taken_s2.add(octet2)
            ip2 = str(subnet2.network_address + octet2)
            if octet2 != preferred_octet:
                plan.warnings.append(
                    "%s: octet %d taken on %s, using %d instead" % (host, preferred_octet, subnet2, octet2)
                )

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
# Configuration
# ---------------------------------------------------------------------------


def generate_cx7_configure_script(host_plan: CX7HostPlan, mtu: int, prefix_len: int) -> str:
    """Generate the netplan configuration script for one host.

    Args:
        host_plan: The plan for this host (must have exactly 2 assignments).
        mtu: Target MTU.
        prefix_len: Subnet prefix length.

    Returns:
        Bash script content ready to pipe via SSH.
    """
    if len(host_plan.assignments) != 2:
        raise ValueError(
            "Expected 2 interface assignments for %s, got %d" % (host_plan.host, len(host_plan.assignments))
        )

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
        host_plan: Plan for this host (must have exactly 2 assignments).
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
            host_plan.host, script, sudo_password, timeout=60, dry_run=dry_run, **kw,
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

    hosts_to_configure = [hp for hp in plan.host_plans if hp.needs_change and len(hp.assignments) == 2]
    if not hosts_to_configure:
        logger.info("No hosts need CX7 configuration changes")
        return results

    logger.info("Applying CX7 configuration to %d host(s)...", len(hosts_to_configure))
    for hp in hosts_to_configure:
        host_pw = sudo_password if hp.host in sudo_hosts else None
        result = configure_cx7_host(
            hp, plan.mtu, plan.prefix_len,
            ssh_kwargs=kw, dry_run=dry_run, sudo_password=host_pw,
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
        hosts, ib_script, timeout=30, dry_run=dry_run, **kw,
    )

    discovered: dict[str, list[str]] = {}
    for result in ib_results:
        if not result.success:
            continue
        ib_info = parse_ib_detect_output(result.stdout)
        ib_ips = extract_ib_ips(ib_info)
        # Filter out management IPs already in host list and loopback
        extra = [
            ip for ip in ib_ips
            if ip not in host_set and ip != "127.0.0.1"
        ]
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
                timeout=30, capture_output=True, text=True,
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
