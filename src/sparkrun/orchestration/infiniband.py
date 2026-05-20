"""InfiniBand/RDMA detection script generator.

Generates a bash script that detects InfiniBand interfaces and outputs
NCCL/RDMA environment variables. The script is piped to remote hosts
via SSH bash -s.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.scripts import read_script
from sparkrun.utils import parse_kv_output

if TYPE_CHECKING:
    from sparkrun.core.backend_select import BackendBundle

logger = logging.getLogger(__name__)


@dataclass
class IBDetectionResult:
    """Aggregated IB detection results across multiple hosts.

    Contains NCCL env vars (from head) and per-host IB IP mappings
    for fast internal transfers.
    """

    comm_env: ClusterCommEnv = field(default_factory=ClusterCommEnv.empty)
    """Inter-node comm env derived from IB detection.

    ``comm_env.shared`` holds keys whose values are identical across
    all hosts (``NCCL_NET``, ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``,
    ``UCX_NET_DEVICES``, …).  ``comm_env.per_host`` holds the keys
    that differ between hosts — typically the socket-interface names
    (``GLOO_SOCKET_IFNAME``, ``MN_IF_NAME``, ``TP_SOCKET_IFNAME``,
    ``OMPI_MCA_btl_tcp_if_include``) — so heterogeneous mgmt
    interfaces (e.g. wired on the head, wifi on a worker) don't crash
    gloo at init time.
    """
    ib_ip_map: dict[str, str] = field(default_factory=dict)
    """Mapping of queried host → first IB interface IP.

    Empty for hosts where no IB was detected or no IB IP was found.
    For switched fabrics this single IP is reachable from anywhere on
    the fabric; for mesh/ring topologies it's only the *first* detected
    candidate and may not be reachable from a given peer — see
    :attr:`ib_candidates` for the full per-host list used by
    :func:`validate_ib_connectivity` to find a working IP.
    """
    ib_candidates: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of queried host → all detected IB interface IPs, in detection order.

    For switched topologies all entries are reachable from anywhere on
    the fabric; for mesh/ring topologies only specific IPs are reachable
    from specific peers, so :func:`validate_ib_connectivity` probes each
    candidate rather than assuming the first one works.
    """
    mgmt_ip_map: dict[str, str] = field(default_factory=dict)
    """Mapping of queried host → management interface IP.

    Useful when clusters are defined by IB IPs: lets callers
    display the management IP alongside the IB address.
    """


def generate_ib_detect_script() -> str:
    """Generate a bash script that detects InfiniBand interfaces.

    The script outputs key=value pairs on stdout that can be parsed
    to configure NCCL and RDMA settings for multi-node inference.

    Output variables (if IB is found)::

        DETECTED_GID_INDEX=<n>
        DETECTED_HCA_LIST=<comma-separated HCA names>
        DETECTED_SOCKET_IFNAME=<interface>
        DETECTED_NET_LIST=<comma-separated net interfaces>
        DETECTED_UCX_LIST=<comma-separated UCX devices>
        IB_DETECTED=1

    If no IB is found, outputs::

        IB_DETECTED=0

    Returns:
        Bash script content as a string.
    """
    return read_script("ib_detect.sh")


def parse_ib_detect_output(output: str) -> dict[str, str]:
    """Parse the output of the IB detection script into a dict.

    Args:
        output: Raw stdout from the IB detection script.

    Returns:
        Dictionary of detected key=value pairs.
    """
    return parse_kv_output(output)


def generate_ring_nccl_overrides(ib_info: dict[str, str]) -> dict[str, str]:
    """NCCL overrides required for 3-node ring/mesh topology.

    Ring topologies use direct CX7 links without a switch, so the
    standard IB transport plugin cannot be used.  These variables
    force socket-based NCCL transport with subnet-aware routing.
    """
    return {
        "NCCL_NET_PLUGIN": "none",
        "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
        "NCCL_IB_MERGE_NICS": "0",
    }


def generate_nccl_env(ib_info: dict[str, str], topology: str | None = None) -> dict[str, str]:
    """Generate NCCL environment variables from IB detection results.

    Args:
        ib_info: Parsed output from :func:`parse_ib_detect_output`.
        topology: CX7 topology (e.g. ``"ring"``).  When ``"ring"``,
            additional overrides are applied for mesh networking.

    Returns:
        Dictionary of NCCL/network environment variables.
        Empty dict if no InfiniBand was detected.
    """
    if ib_info.get("IB_DETECTED") != "1":
        return {}

    env: dict[str, str] = {
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_CROSS_NIC": "1",
    }

    def _set_eth_interfaces(target):
        net_list = ib_info[target]
        env["MN_IF_NAME"] = net_list
        env["OMPI_MCA_btl_tcp_if_include"] = net_list
        env["GLOO_SOCKET_IFNAME"] = net_list
        env["TP_SOCKET_IFNAME"] = net_list

    def _nccl_socket_ifname_list(mgmt_if: str | None, ib_nets: str) -> str:
        """Build an ordered, deduped NCCL_SOCKET_IFNAME list.

        NCCL accepts a comma-separated list and tries them in order.
        Put the mgmt/default-route interface first (that's the one
        reachable from the control machine and between hosts for
        bootstrap / rendezvous TCP), then fall through to the
        detected IB adapters so NCCL has options if mgmt is missing.
        Duplicates are preserved in order of first appearance.
        """
        parts: list[str] = []
        seen: set[str] = set()
        if mgmt_if:
            mgmt_if = mgmt_if.strip()
            if mgmt_if and mgmt_if not in seen:
                parts.append(mgmt_if)
                seen.add(mgmt_if)
        for ifname in (ib_nets or "").split(","):
            ifname = ifname.strip()
            if ifname and ifname not in seen:
                parts.append(ifname)
                seen.add(ifname)
        return ",".join(parts)

    if ib_info.get("DETECTED_HCA_LIST"):
        env["NCCL_IB_HCA"] = ib_info["DETECTED_HCA_LIST"]
    if ib_info.get("DETECTED_SOCKET_IFNAME"):  # prefer MGMT/default interface for non-IB HCA adapters since it works for mesh or non-mesh
        _set_eth_interfaces("DETECTED_SOCKET_IFNAME")
        env["NCCL_SOCKET_IFNAME"] = _nccl_socket_ifname_list(
            ib_info["DETECTED_SOCKET_IFNAME"],
            ib_info.get("DETECTED_NET_LIST", ""),
        )
    elif ib_info.get("DETECTED_NET_LIST"):  # fallback to specifying CX7 interfaces
        _set_eth_interfaces("DETECTED_NET_LIST")
        env["NCCL_SOCKET_IFNAME"] = _nccl_socket_ifname_list(
            None,
            ib_info["DETECTED_NET_LIST"],
        )
    if ib_info.get("DETECTED_UCX_LIST"):
        env["UCX_NET_DEVICES"] = ib_info["DETECTED_UCX_LIST"]

    # add NODE_IP for management interface for ray script compatibility
    env["NODE_IP"] = ib_info.get("DETECTED_MGMT_IP", "")

    if topology == "ring":
        env.update(generate_ring_nccl_overrides(ib_info))
    elif ib_info.get("DETECTED_GID_INDEX"):  # only do group ID for non-mesh topologies
        env["NCCL_IB_GID_INDEX"] = ib_info["DETECTED_GID_INDEX"]

    return env


def extract_ib_ips(ib_info: dict[str, str]) -> list[str]:
    """Extract InfiniBand interface IPv4 addresses from detection results.

    Args:
        ib_info: Parsed output from :func:`parse_ib_detect_output`.

    Returns:
        List of IB interface IPs (may be empty if no IB or no IPs found).
    """
    raw = ib_info.get("DETECTED_IB_IPS", "")
    if not raw:
        return []
    return [ip.strip() for ip in raw.split(",") if ip.strip()]


def validate_ib_connectivity(
    ib_candidates: dict[str, str] | dict[str, list[str]],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Validate that the control machine can reach detected IB IPs.

    Tests SSH connectivity from the control machine to candidate IB
    interface IPs.  In switched fabrics every IB IP is reachable from
    anywhere on the fabric, so a single working probe suffices.  In
    mesh/ring topologies each host has multiple IB IPs (one/two per
    point-to-point link) and only some are reachable from a given
    peer; this function iterates the per-host candidates and records
    the first one that responds.

    Accepts either the legacy ``dict[host, str]`` (single best-guess
    IP per host) or the new ``dict[host, list[str]]`` (full per-host
    candidate list from :class:`IBDetectionResult.ib_candidates`).
    Returns an empty dict — signalling management-network fallback —
    only when *every* host has zero reachable candidates.

    Args:
        ib_candidates: Mapping of management host → candidate IB IP(s).
        ssh_kwargs: SSH connection parameters (user, key, options).
        dry_run: Skip the check and return a single-IP-per-host map
            without probing.

    Returns:
        Mapping of host → verified-reachable IB IP, or an empty dict
        when no host has any reachable IB IP.
    """
    if not ib_candidates:
        return {}

    # Normalize legacy single-IP shape to per-host list.
    normalized: dict[str, list[str]] = {}
    for host, val in ib_candidates.items():
        if isinstance(val, str):
            normalized[host] = [val] if val else []
        else:
            normalized[host] = [ip for ip in val if ip]

    if dry_run:
        # Preserve legacy behavior: return a single-IP-per-host map
        # without performing probes.
        return {host: ips[0] for host, ips in normalized.items() if ips}

    from concurrent.futures import ThreadPoolExecutor

    from sparkrun.orchestration.ssh import run_remote_command

    kw = ssh_kwargs or {}
    logger.info("Verifying IB network reachability from control machine...")

    # Flatten to (host, ip) work items so every candidate is probed
    # concurrently — important for mesh/ring where the "wrong" candidate
    # takes the full SSH timeout to fail and serial probing would stack
    # 10s of latency per extra candidate per host.
    work: list[tuple[str, str]] = []
    for host, cands in normalized.items():
        for ip in cands:
            work.append((host, ip))

    probe_results: dict[tuple[str, str], bool] = {}
    if work:

        def _probe(host_ip: tuple[str, str]) -> tuple[tuple[str, str], bool]:
            _host, _ip = host_ip
            result = run_remote_command(_ip, "true", connect_timeout=5, timeout=10, **kw)
            return host_ip, result.success

        # TODO: review parallelism limits!
        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            for host_ip, ok in pool.map(_probe, work):
                probe_results[host_ip] = ok

    # For each host, pick the first candidate (in detection order) that
    # responded; record the rest as unreachable for diagnostic logging.
    verified: dict[str, str] = {}
    unreachable: dict[str, list[str]] = {}
    for host, cands in normalized.items():
        if not cands:
            unreachable[host] = []
            continue
        working = next((ip for ip in cands if probe_results.get((host, ip))), None)
        if working is None:
            unreachable[host] = list(cands)
            continue
        verified[host] = working
        primary = cands[0]
        if working == primary:
            logger.info("  %s reachable via %s", host, working)
        else:
            logger.info("  %s reachable via %s", host, working)
            failed_before = [ip for ip in cands if ip != working and not probe_results.get((host, ip))]
            logger.debug(
                "  %s reachable via %s (primary candidate %s unreachable)",
                host,
                working,
                ", ".join(failed_before) if failed_before else primary,
            )

    if unreachable:
        for host, tried in unreachable.items():
            if tried:
                logger.warning(
                    "  Control machine cannot reach any IB IP for host %s (tried %s)",
                    host,
                    ", ".join(tried),
                )
            else:
                logger.warning("  No IB candidates available for host %s", host)
        logger.warning("  Falling back to management network for transfers")
        return {}

    logger.info("  IB network reachable — will use IB IPs for transfers")
    return verified


def detect_ib_for_hosts(
    hosts: list[str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    topology: str | None = None,
    backends: "dict[str, BackendBundle] | None" = None,
) -> IBDetectionResult:
    """Run IB detection on all hosts and return aggregated results.

    Detects InfiniBand on all hosts in parallel, computes NCCL env
    from the head (``hosts[0]``), and builds a mapping of management
    host → first IB IP for use as transfer targets.

    Args:
        hosts: Management hostnames/IPs.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
        topology: Optional cluster topology hint (e.g. ``"ring"``).
        backends: Optional per-host :class:`BackendBundle`.  When
            provided, each host's env block is produced by
            ``backends[host].collective.env_for_host(ib_info, topology=...)``.
            Hosts missing from *backends* fall back to the legacy
            :func:`generate_nccl_env` (NCCL) path.  When *backends* is
            ``None`` every host uses the legacy path.

    Returns:
        :class:`IBDetectionResult` with NCCL env and IB IP mapping.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not hosts:
        return IBDetectionResult()

    kw = ssh_kwargs or {}
    head_host = hosts[0]

    logger.info("Detecting InfiniBand on %d host(s)...", len(hosts))
    ib_script = generate_ib_detect_script()
    ib_results = run_remote_scripts_parallel(
        hosts,
        ib_script,
        timeout=30,
        dry_run=dry_run,
        **kw,
    )

    per_host_env: dict[str, dict[str, str]] = {}
    ib_ip_map: dict[str, str] = {}
    ib_candidates: dict[str, list[str]] = {}
    mgmt_ip_map: dict[str, str] = {}

    for result in ib_results:
        if not result.success:
            continue
        ib_info = parse_ib_detect_output(result.stdout)

        # Per-host comm env: route through backend when provided so
        # NCCL/RCCL/HCCL hosts emit the right env block.  Hosts missing
        # from *backends* (or all hosts when *backends* is None) use the
        # legacy NCCL generator — byte-identical to NcclBackend on NVIDIA.
        if backends is not None and result.host in backends:
            host_env = backends[result.host].collective.env_for_host(ib_info, topology=topology)
        else:
            host_env = generate_nccl_env(ib_info, topology=topology)
        if host_env:
            per_host_env[result.host] = host_env
            if result.host == head_host:
                logger.info("  InfiniBand detected on %s, comm env configured", head_host)

        # IB IP for transfer routing.  Preserve all detected IPs as
        # candidates so validate_ib_connectivity() can try alternates
        # in mesh/ring topologies where only specific point-to-point
        # links are reachable from any given peer.
        ib_ips = extract_ib_ips(ib_info)
        if ib_ips:
            ib_ip_map[result.host] = ib_ips[0]
            ib_candidates[result.host] = list(ib_ips)
            if len(ib_ips) > 1:
                logger.debug("  %s IB transfer IP candidates: %s", result.host, ib_ips)
            else:
                logger.debug("  %s IB transfer IP: %s", result.host, ib_ips[0])

        # Management IP (from default route interface)
        mgmt_ip = ib_info.get("DETECTED_MGMT_IP", "").strip()
        if mgmt_ip:
            mgmt_ip_map[result.host] = mgmt_ip
            logger.debug("  %s mgmt IP: %s", result.host, mgmt_ip)

    comm_env = ClusterCommEnv.from_per_host(per_host_env)
    if comm_env.is_empty():
        logger.info("  No InfiniBand detected, using default networking")

    if ib_ip_map:
        logger.info("  IB transfer IPs resolved for %d/%d host(s)", len(ib_ip_map), len(hosts))
    else:
        logger.info("  No IB IPs found, transfers will use management network")

    return IBDetectionResult(
        comm_env=comm_env,
        ib_ip_map=ib_ip_map,
        ib_candidates=ib_candidates,
        mgmt_ip_map=mgmt_ip_map,
    )
