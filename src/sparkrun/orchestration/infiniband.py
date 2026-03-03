"""InfiniBand/RDMA detection script generator.

Generates a bash script that detects InfiniBand interfaces and outputs
NCCL/RDMA environment variables. The script is piped to remote hosts
via SSH bash -s.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sparkrun.scripts import read_script
from sparkrun.utils import parse_kv_output  # noqa: F401 — re-exported for callers

logger = logging.getLogger(__name__)


@dataclass
class IBDetectionResult:
    """Aggregated IB detection results across multiple hosts.

    Contains NCCL env vars (from head) and per-host IB IP mappings
    for fast internal transfers.
    """

    nccl_env: dict[str, str] = field(default_factory=dict)
    ib_ip_map: dict[str, str] = field(default_factory=dict)
    """Mapping of queried host → first IB interface IP.

    Empty for hosts where no IB was detected or no IB IP was found.
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


def generate_nccl_env(ib_info: dict[str, str]) -> dict[str, str]:
    """Generate NCCL environment variables from IB detection results.

    Args:
        ib_info: Parsed output from :func:`parse_ib_detect_output`.

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

    if ib_info.get("DETECTED_GID_INDEX"):
        env["NCCL_IB_GID_INDEX"] = ib_info["DETECTED_GID_INDEX"]
    if ib_info.get("DETECTED_HCA_LIST"):
        env["NCCL_IB_HCA"] = ib_info["DETECTED_HCA_LIST"]
    if ib_info.get("DETECTED_NET_LIST"):
        net_list = ib_info["DETECTED_NET_LIST"]
        # NCCL_SOCKET_IFNAME uses '=' prefix per interface to pin exact devices (refer to NCCL docs for details)
        nccl_socket = ",".join("=" + if_ for if_ in net_list.split(","))
        env["NCCL_SOCKET_IFNAME"] = nccl_socket
        env["MN_IF_NAME"] = net_list
        env["OMPI_MCA_btl_tcp_if_include"] = net_list
        env["GLOO_SOCKET_IFNAME"] = net_list
        env["TP_SOCKET_IFNAME"] = net_list
    if ib_info.get("DETECTED_UCX_LIST"):
        env["UCX_NET_DEVICES"] = ib_info["DETECTED_UCX_LIST"]

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
        ib_ip_map: dict[str, str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> dict[str, str]:
    """Validate that the control machine can reach detected IB IPs.

    Tests SSH connectivity from the control machine to a sample IB IP.
    If the control machine is not on the InfiniBand network, the IB IPs
    are unreachable and transfers must fall back to management IPs.

    Only one IB IP is tested since all are on the same subnet — if one
    is reachable, the rest should be too.

    Args:
        ib_ip_map: Mapping of management host → IB IP (from detection).
        ssh_kwargs: SSH connection parameters (user, key, options).
        dry_run: Skip the check and return the map unchanged.

    Returns:
        The original *ib_ip_map* if connectivity is confirmed, or an
        empty dict if the IB network is unreachable from the control
        machine (signalling fallback to management network).
    """
    if not ib_ip_map or dry_run:
        return ib_ip_map

    from sparkrun.orchestration.ssh import run_remote_command

    # Test connectivity to the first IB IP
    test_host, test_ip = next(iter(ib_ip_map.items()))
    kw = ssh_kwargs or {}

    logger.info("Verifying IB network reachability from control machine (testing %s)...", test_ip)
    result = run_remote_command(
        test_ip, "true",
        connect_timeout=5, timeout=10,
        **kw,
    )

    if result.success:
        logger.info("  IB network reachable — will use IB IPs for transfers")
        return ib_ip_map

    logger.warning(
        "  Control machine cannot reach IB network (tested %s for host %s) "
        "— falling back to management network for transfers",
        test_ip, test_host,
    )
    return {}


def detect_ib_for_hosts(
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> IBDetectionResult:
    """Run IB detection on all hosts and return aggregated results.

    Detects InfiniBand on all hosts in parallel, computes NCCL env
    from the head (``hosts[0]``), and builds a mapping of management
    host → first IB IP for use as transfer targets.

    Args:
        hosts: Management hostnames/IPs.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.

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
        hosts, ib_script, timeout=30, dry_run=dry_run, **kw,
    )

    nccl_env: dict[str, str] = {}
    ib_ip_map: dict[str, str] = {}
    mgmt_ip_map: dict[str, str] = {}

    for result in ib_results:
        if not result.success:
            continue
        ib_info = parse_ib_detect_output(result.stdout)

        # NCCL env from head host
        if result.host == head_host and not nccl_env:
            nccl_env = generate_nccl_env(ib_info)
            if nccl_env:
                logger.info("  InfiniBand detected on %s, NCCL configured", head_host)

        # IB IP for transfer routing
        ib_ips = extract_ib_ips(ib_info)
        if ib_ips:
            ib_ip_map[result.host] = ib_ips[0]
            logger.debug("  %s IB transfer IP: %s", result.host, ib_ips[0])

        # Management IP (from default route interface)
        mgmt_ip = ib_info.get("DETECTED_MGMT_IP", "").strip()
        if mgmt_ip:
            mgmt_ip_map[result.host] = mgmt_ip
            logger.debug("  %s mgmt IP: %s", result.host, mgmt_ip)

    if not nccl_env:
        logger.info("  No InfiniBand detected, using default networking")

    if ib_ip_map:
        logger.info("  IB transfer IPs resolved for %d/%d host(s)",
                    len(ib_ip_map), len(hosts))
    else:
        logger.info("  No IB IPs found, transfers will use management network")

    return IBDetectionResult(nccl_env=nccl_env, ib_ip_map=ib_ip_map, mgmt_ip_map=mgmt_ip_map)
