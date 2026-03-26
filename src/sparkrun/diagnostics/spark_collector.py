"""Spark host diagnostics collector.

Runs ``spark_diagnose.sh`` on remote hosts, parses the key=value output,
and transforms flat results into structured diagnostic records.
"""

from __future__ import annotations

import logging
import time

from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.orchestration.ssh import run_remote_scripts_parallel
from sparkrun.scripts import read_script
from sparkrun.utils import parse_kv_output

logger = logging.getLogger(__name__)

# Keys grouped by record type for structured output.
_HARDWARE_KEYS = (
    "DIAG_CPU_MODEL",
    "DIAG_CPU_CORES",
    "DIAG_CPU_THREADS",
    "DIAG_RAM_TOTAL_KB",
    "DIAG_RAM_AVAILABLE_KB",
    "DIAG_DISK_ROOT_TOTAL_KB",
    "DIAG_DISK_ROOT_AVAIL_KB",
    "DIAG_DISK_HOME_TOTAL_KB",
    "DIAG_DISK_HOME_AVAIL_KB",
    "DIAG_GPU_NAME",
    "DIAG_GPU_MEMORY_MB",
    "DIAG_GPU_DRIVER",
    "DIAG_GPU_PSTATE",
    "DIAG_GPU_TEMP_C",
    "DIAG_GPU_POWER_W",
    "DIAG_GPU_SERIAL",
    "DIAG_GPU_UUID",
)

_FIRMWARE_KEYS = (
    "DIAG_HOSTNAME",
    "DIAG_HOSTNAME_FQDN",
    "DIAG_OS_NAME",
    "DIAG_OS_VERSION",
    "DIAG_OS_PRETTY",
    "DIAG_KERNEL",
    "DIAG_ARCH",
    "DIAG_BIOS_VERSION",
    "DIAG_BOARD_NAME",
    "DIAG_PRODUCT_NAME",
    "DIAG_JETPACK_VERSION",
    "DIAG_CUDA_VERSION",
)

_DMI_KEYS = (
    "DIAG_DMI_BIOS_VENDOR",
    "DIAG_DMI_BIOS_VERSION",
    "DIAG_DMI_BIOS_DATE",
    "DIAG_DMI_SYS_MANUFACTURER",
    "DIAG_DMI_SYS_PRODUCT",
    "DIAG_DMI_SYS_VERSION",
    "DIAG_DMI_SYS_SERIAL",
    "DIAG_DMI_SYS_UUID",
    "DIAG_DMI_BOARD_MANUFACTURER",
    "DIAG_DMI_BOARD_PRODUCT",
    "DIAG_DMI_BOARD_VERSION",
    "DIAG_DMI_BOARD_SERIAL",
    "DIAG_DMI_MEM_SLOTS",
    "DIAG_DMI_MEM_POPULATED",
    "DIAG_DMI_MEM_MAX",
)

_DOCKER_KEYS = (
    "DIAG_DOCKER_VERSION",
    "DIAG_DOCKER_STORAGE",
    "DIAG_DOCKER_ROOT",
    "DIAG_DOCKER_NVIDIA_RUNTIME",
    "DIAG_DOCKER_RUNNING",
    "DIAG_DOCKER_SPARKRUN",
)


def _extract_keys(kv: dict[str, str], keys: tuple[str, ...], strip_prefix: str = "DIAG_") -> dict[str, str]:
    """Extract and rename keys from a flat kv dict."""
    result: dict[str, str] = {}
    for k in keys:
        if k in kv:
            short = k[len(strip_prefix) :].lower() if k.startswith(strip_prefix) else k.lower()
            result[short] = kv[k]
    return result


def _extract_network(kv: dict[str, str]) -> dict:
    """Extract indexed network interface records from flat kv output."""
    count = int(kv.get("DIAG_NET_COUNT", "0"))
    interfaces = []
    for i in range(count):
        prefix = "DIAG_NET_%d_" % i
        iface = {}
        for k, v in kv.items():
            if k.startswith(prefix):
                short = k[len(prefix) :].lower()
                iface[short] = v
        if iface:
            interfaces.append(iface)

    return {
        "interfaces": interfaces,
        "default_iface": kv.get("DIAG_DEFAULT_IFACE", ""),
        "mgmt_ip": kv.get("DIAG_MGMT_IP", ""),
    }


def _extract_indexed(kv: dict[str, str], prefix: str, count_key: str) -> list[str]:
    """Extract indexed values like DIAG_FWUPD_DEV_0, DIAG_FWUPD_DEV_1, ..."""
    count = int(kv.get(count_key, "0"))
    items = []
    for i in range(count):
        key = "%s%d" % (prefix, i)
        if key in kv:
            items.append(kv[key])
    return items


def _extract_firmware_devices(kv: dict[str, str]) -> list[dict[str, str]]:
    """Parse fwupdmgr device entries from indexed kv output."""
    raw = _extract_indexed(kv, "DIAG_FWUPD_DEV_", "DIAG_FWUPD_DEV_COUNT")
    devices = []
    for entry in raw:
        parts = entry.split("|")
        devices.append(
            {
                "name": parts[0].strip() if len(parts) > 0 else "",
                "version": parts[1].strip() if len(parts) > 1 else "",
                "guid": parts[2].strip() if len(parts) > 2 else "",
            }
        )
    return devices


def _extract_firmware_history(kv: dict[str, str]) -> list[dict[str, str]]:
    """Parse fwupdmgr history entries from indexed kv output."""
    raw = _extract_indexed(kv, "DIAG_FWUPD_HIST_", "DIAG_FWUPD_HIST_COUNT")
    history = []
    for entry in raw:
        parts = entry.split("|")
        history.append(
            {
                "name": parts[0].strip() if len(parts) > 0 else "",
                "version": parts[1].strip() if len(parts) > 1 else "",
                "date": parts[2].strip() if len(parts) > 2 else "",
            }
        )
    return history


def collect_sudo_diagnostics(
    hosts: list[str],
    ssh_kwargs: dict,
    sudo_password: str,
    writer: NDJSONWriter | None = None,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Collect sudo-only diagnostics (dmidecode) from hosts.

    Uses password-based sudo via ``run_sudo_script_on_host`` for each host.

    Args:
        hosts: Target hostnames or IPs.
        ssh_kwargs: SSH connection parameters.
        sudo_password: Sudo password for privilege escalation.
        writer: Optional NDJSONWriter for NDJSON output.
        dry_run: If True, don't actually execute.

    Returns:
        ``{host: parsed_kv_dict}`` for programmatic use.
    """
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    script = read_script("spark_diagnose_sudo.sh")
    host_data: dict[str, dict] = {}

    for host in hosts:
        result = run_sudo_script_on_host(
            host,
            script,
            sudo_password,
            ssh_kwargs=ssh_kwargs,
            timeout=60,
            dry_run=dry_run,
        )
        if not result.success:
            host_data[host] = {}
            if writer:
                writer.emit(
                    "host_error",
                    {
                        "host": host,
                        "error": "Sudo diagnostics failed with rc=%d" % result.returncode,
                        "stderr": result.stderr.strip()[:500],
                    },
                )
            logger.warning("Sudo diagnostics failed on %s: rc=%d", host, result.returncode)
            continue

        kv = parse_kv_output(result.stdout)
        host_data[host] = kv

        if writer and kv.get("DIAG_SUDO_COMPLETE") == "1":
            dmi = _extract_keys(kv, _DMI_KEYS)
            dmi["host"] = host
            writer.emit("host_dmi", dmi)

    return host_data


def collect_spark_diagnostics(
    hosts: list[str],
    ssh_kwargs: dict,
    writer: NDJSONWriter | None = None,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Collect hardware/firmware/network/Docker diagnostics from hosts.

    Runs ``spark_diagnose.sh`` on all hosts in parallel, parses stdout,
    and emits structured records via *writer* (if provided).

    Args:
        hosts: Target hostnames or IPs.
        ssh_kwargs: SSH connection parameters (ssh_user, ssh_key, etc.).
        writer: Optional NDJSONWriter for NDJSON output.
        dry_run: If True, don't actually execute on remote hosts.

    Returns:
        ``{host: parsed_kv_dict}`` for programmatic use.  Failed hosts
        have an empty dict.
    """
    script = read_script("spark_diagnose.sh")
    t0 = time.monotonic()

    results = run_remote_scripts_parallel(
        hosts=hosts,
        script=script,
        timeout=60,
        dry_run=dry_run,
        **ssh_kwargs,
    )

    host_data: dict[str, dict] = {}
    successful = 0
    failed = 0

    for result in results:
        if not result.success:
            failed += 1
            host_data[result.host] = {}
            if writer:
                writer.emit(
                    "host_error",
                    {
                        "host": result.host,
                        "error": "Script failed with rc=%d" % result.returncode,
                        "stderr": result.stderr.strip()[:500],
                    },
                )
            logger.warning("Diagnostics failed on %s: rc=%d", result.host, result.returncode)
            continue

        kv = parse_kv_output(result.stdout)
        if kv.get("DIAG_COMPLETE") != "1":
            failed += 1
            host_data[result.host] = kv
            if writer:
                writer.emit(
                    "host_error",
                    {
                        "host": result.host,
                        "error": "Incomplete diagnostics (missing DIAG_COMPLETE sentinel)",
                        "stderr": result.stderr.strip()[:500],
                    },
                )
            continue

        successful += 1
        host_data[result.host] = kv

        if writer:
            hw = _extract_keys(kv, _HARDWARE_KEYS)
            hw["host"] = result.host
            writer.emit("host_hardware", hw)

            fw = _extract_keys(kv, _FIRMWARE_KEYS)
            fw["host"] = result.host
            writer.emit("host_firmware", fw)

            net = _extract_network(kv)
            net["host"] = result.host
            writer.emit("host_network", net)

            docker = _extract_keys(kv, _DOCKER_KEYS)
            docker["host"] = result.host
            writer.emit("host_docker", docker)

            fw_devices = _extract_firmware_devices(kv)
            fw_history = _extract_firmware_history(kv)
            if fw_devices or fw_history:
                writer.emit(
                    "host_firmware_updates",
                    {
                        "host": result.host,
                        "devices": fw_devices,
                        "history": fw_history,
                    },
                )

    duration = time.monotonic() - t0

    if writer:
        writer.emit(
            "diag_summary",
            {
                "total_hosts": len(hosts),
                "successful": successful,
                "failed": failed,
                "duration_seconds": round(duration, 2),
            },
        )

    logger.info(
        "Diagnostics: %d/%d hosts OK in %.1fs",
        successful,
        len(hosts),
        duration,
    )

    return host_data


def collect_config_diagnostics(
    writer: NDJSONWriter,
    config=None,
    cluster_mgr=None,
    registry_mgr=None,
) -> None:
    """Emit sparkrun configuration state: clusters, registries, config.

    Reads local configuration only — no SSH required.

    Args:
        writer: NDJSONWriter for NDJSON output.
        config: Optional SparkrunConfig instance.
        cluster_mgr: Optional ClusterManager instance.
        registry_mgr: Optional RegistryManager instance.
    """
    # Sparkrun config
    if config is not None:
        writer.emit(
            "config_sparkrun",
            {
                "config_path": str(config.config_path),
                "cache_dir": str(config.cache_dir),
                "hf_cache_dir": str(config.hf_cache_dir),
                "ssh_user": config.ssh_user,
                "default_hosts": config.default_hosts,
            },
        )

    # Cluster definitions
    if cluster_mgr is not None:
        default_cluster = cluster_mgr.get_default()
        clusters = []
        try:
            for c in cluster_mgr.list_clusters():
                clusters.append(
                    {
                        "name": c.name,
                        "hosts": c.hosts,
                        "description": c.description,
                        "user": c.user,
                        "cache_dir": c.cache_dir,
                        "transfer_mode": c.transfer_mode,
                        "transfer_interface": c.transfer_interface,
                        "is_default": c.name == default_cluster,
                    }
                )
        except Exception as e:
            logger.warning("Failed to list clusters: %s", e)

        writer.emit(
            "config_clusters",
            {
                "default": default_cluster,
                "count": len(clusters),
                "clusters": clusters,
            },
        )

    # Registry configuration
    if registry_mgr is not None:
        registries = []
        try:
            for r in registry_mgr.list_registries():
                registries.append(
                    {
                        "name": r.name,
                        "url": r.url,
                        "subpath": r.subpath,
                        "description": r.description,
                        "enabled": r.enabled,
                        "visible": r.visible,
                        "tuning_subpath": r.tuning_subpath,
                        "benchmark_subpath": r.benchmark_subpath,
                    }
                )
        except Exception as e:
            logger.warning("Failed to list registries: %s", e)

        writer.emit(
            "config_registries",
            {
                "count": len(registries),
                "registries": registries,
            },
        )
