"""Combined per-host hardware probe: accelerators + InfiniBand in one SSH round-trip.

:func:`probe_host` runs a single bash script over SSH that emits both
accelerator fingerprint data (as in :mod:`sparkrun.core.fingerprint`) and
InfiniBand detection data (as in :mod:`sparkrun.orchestration.infiniband`).
The two sections are delimited by sentinel markers so a single ``stdout``
stream can be parsed into both halves.

:func:`probe_hosts` parallelises :func:`probe_host` across a list of hosts
using :func:`~sparkrun.orchestration.ssh.run_remote_scripts_parallel`.

The bash script body is the *concatenation* of the two existing scripts,
each wrapped in a section delimiter — no logic is merged, just sequential
execution.  This keeps the two probes independently testable while saving
one SSH round-trip per host.
"""

from __future__ import annotations

import logging
from typing import Any

from sparkrun.core.hardware import HostHardware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel markers used to split combined output into sections
# ---------------------------------------------------------------------------

_ACCEL_START = "SPARKRUN_PROBE_ACCEL_START"
_ACCEL_END = "SPARKRUN_PROBE_ACCEL_END"
_IB_START = "SPARKRUN_PROBE_IB_START"
_IB_END = "SPARKRUN_PROBE_IB_END"

# ---------------------------------------------------------------------------
# Combined probe script
# ---------------------------------------------------------------------------

# The fingerprint section emits normalised KEY=VALUE lines (same as
# _FINGERPRINT_SCRIPT in fingerprint.py).  The IB section emits the same
# KEY=VALUE lines as ib_detect.sh.  Both sections write diagnostic output
# to stderr so stdout stays machine-parseable.

_COMBINED_PROBE_SCRIPT = r"""#!/bin/bash
set -uo pipefail

# ===========================================================================
# SECTION 1: Accelerator fingerprint
# ===========================================================================
echo "{accel_start}"

emit() {{ printf '%s=%s\n' "$1" "$2"; }}

# --- NVIDIA ---
NVIDIA_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        name=$(printf '%s' "$line" | awk -F', *' '{{print $1}}')
        mem=$(printf '%s' "$line" | awk -F', *' '{{print $2}}' | awk '{{print $1}}')
        emit "NVIDIA_GPU_${{NVIDIA_COUNT}}_NAME" "$name"
        emit "NVIDIA_GPU_${{NVIDIA_COUNT}}_MEMORY_MIB" "$mem"
        NVIDIA_COUNT=$((NVIDIA_COUNT + 1))
    done < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
fi
emit NVIDIA_GPU_COUNT "$NVIDIA_COUNT"
emit NVIDIA_PRESENT "$([[ $NVIDIA_COUNT -gt 0 ]] && echo 1 || echo 0)"

# --- AMD ROCm ---
AMD_COUNT=0
if command -v rocm-smi >/dev/null 2>&1; then
    while IFS= read -r line; do
        if [[ "$line" =~ GPU\[([0-9]+)\][[:space:]]*:[[:space:]]*Card[[:space:]]series:[[:space:]]*(.*) ]]; then
            idx="${{BASH_REMATCH[1]}}"
            name="${{BASH_REMATCH[2]}}"
            emit "AMD_GPU_${{idx}}_NAME" "$name"
            [[ $idx -ge $AMD_COUNT ]] && AMD_COUNT=$((idx + 1))
        elif [[ "$line" =~ GPU\[([0-9]+)\][[:space:]]*:[[:space:]]*VRAM[[:space:]]Total[[:space:]]Memory[[:space:]]\(B\):[[:space:]]*([0-9]+) ]]; then
            idx="${{BASH_REMATCH[1]}}"
            bytes="${{BASH_REMATCH[2]}}"
            mib=$((bytes / 1024 / 1024))
            emit "AMD_GPU_${{idx}}_MEMORY_MIB" "$mib"
        fi
    done < <(rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true)
fi
emit AMD_GPU_COUNT "$AMD_COUNT"
emit AMD_PRESENT "$([[ $AMD_COUNT -gt 0 ]] && echo 1 || echo 0)"

# --- Intel Gaudi ---
INTEL_COUNT=0
if command -v hl-smi >/dev/null 2>&1; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        name=$(printf '%s' "$line" | awk -F', *' '{{print $1}}')
        mem=$(printf '%s' "$line" | awk -F', *' '{{print $2}}' | awk '{{print $1}}')
        emit "INTEL_GAUDI_${{INTEL_COUNT}}_NAME" "$name"
        emit "INTEL_GAUDI_${{INTEL_COUNT}}_MEMORY_MIB" "$mem"
        INTEL_COUNT=$((INTEL_COUNT + 1))
    done < <(hl-smi --query-aip=name,memory.total --format=csv,noheader 2>/dev/null || true)
fi
emit INTEL_GAUDI_COUNT "$INTEL_COUNT"
emit INTEL_PRESENT "$([[ $INTEL_COUNT -gt 0 ]] && echo 1 || echo 0)"

# --- Apple Silicon ---
APPLE_PRESENT=0
APPLE_MODEL=""
if [[ "$(uname -s)" == "Darwin" ]] && command -v system_profiler >/dev/null 2>&1; then
    chip=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip:/ {{print $2}}' | head -1)
    if [[ -n "$chip" ]]; then
        APPLE_PRESENT=1
        APPLE_MODEL="$chip"
    fi
fi
emit APPLE_PRESENT "$APPLE_PRESENT"
emit APPLE_MODEL "$APPLE_MODEL"

# --- InfiniBand presence (capability tag only) ---
IB_PRESENT=0
if [[ -d /sys/class/infiniband ]] && compgen -G "/sys/class/infiniband/*" >/dev/null; then
    IB_PRESENT=1
fi
emit IB_PRESENT "$IB_PRESENT"

# --- OS / arch ---
emit OS "$(uname -s 2>/dev/null || echo unknown)"
emit ARCH "$(uname -m 2>/dev/null || echo unknown)"

echo "{accel_end}"

# ===========================================================================
# SECTION 2: InfiniBand detection
# ===========================================================================
echo "{ib_start}"

find_rocev2_ipv4_index() {{
    local hca=$1
    if ! command -v show_gids &>/dev/null; then
        return 1
    fi
    show_gids \
    | awk -v dev="$hca" \
      '$1 == dev && $6 == "v2" && $5 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/ {{print $3; exit}}'
}}

echo "Running InfiniBand detection..." >&2

if ! [ -d /sys/class/infiniband ]; then
    echo "IB_DETECTED=0"
    echo "{ib_end}"
    exit 0
fi

ACTIVE_HCAS=()
ACTIVE_NETIFS=()
UCX_DEVS=()
GID_INDEX=""

for ib_path in /sys/class/infiniband/*; do
    [ -e "$ib_path" ] || continue
    hca_name=$(basename "$ib_path")

    state_file="$ib_path/ports/1/state"
    if [ ! -f "$state_file" ]; then continue; fi

    state_val=$(cat "$state_file" 2>/dev/null)

    if [[ "$state_val" == *"ACTIVE"* ]]; then
        ACTIVE_HCAS+=("$hca_name")
        UCX_DEVS+=("${{hca_name}}:1")

        if [ -z "$GID_INDEX" ]; then
            idx=$(find_rocev2_ipv4_index "$hca_name")
            if [ -n "$idx" ]; then
                GID_INDEX=$idx
                echo "Device $hca_name: Active (RoCEv2 GID Index: $idx)" >&2
            else
                port_gid_dir="$ib_path/ports/1/gid_attrs/types"
                if [ -d "$port_gid_dir" ]; then
                    for type_file in "$port_gid_dir"/*; do
                        gid_type=$(cat "$type_file" 2>/dev/null || true)
                        if [[ "$gid_type" == *"RoCE v2"* ]] || [[ "$gid_type" == *"RoCEv2"* ]]; then
                            GID_INDEX=$(basename "$type_file")
                            echo "Device $hca_name: Active (RoCEv2 GID Index: $GID_INDEX via sysfs)" >&2
                            break
                        fi
                    done
                fi
                if [ -z "$GID_INDEX" ]; then
                    echo "Device $hca_name: Active (GID detect failed, defaulting to 3)" >&2
                    GID_INDEX=3
                fi
            fi
        else
            echo "Device $hca_name: Active" >&2
        fi

        net_dir="$ib_path/device/net"
        if [ -d "$net_dir" ]; then
            net_if=$(ls "$net_dir" | head -n 1)
            [ -n "$net_if" ] && ACTIVE_NETIFS+=("$net_if")
        fi
    fi
done

if [ ${{#ACTIVE_HCAS[@]}} -eq 0 ]; then
    echo "No active RDMA devices found." >&2
    echo "IB_DETECTED=0"
    echo "{ib_end}"
    exit 0
fi

DEFAULT_IF=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")
MGMT_IP=$(ip -4 addr show "$DEFAULT_IF" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)

HCA_LIST=$(IFS=,; echo "${{ACTIVE_HCAS[*]}}")
NET_LIST=$(IFS=,; echo "${{ACTIVE_NETIFS[*]}}")
UCX_LIST=$(IFS=,; echo "${{UCX_DEVS[*]}}")

IB_IPS=()
for net_if in "${{ACTIVE_NETIFS[@]}}"; do
    ip_addr=$(ip -4 addr show "$net_if" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)
    if [ -n "$ip_addr" ]; then
        IB_IPS+=("$ip_addr")
    fi
done
IB_IP_LIST=$(IFS=,; echo "${{IB_IPS[*]}}")

echo "IB_DETECTED=1"
echo "DETECTED_GID_INDEX=$GID_INDEX"
echo "DETECTED_HCA_LIST=$HCA_LIST"
echo "DETECTED_SOCKET_IFNAME=$DEFAULT_IF"
echo "DETECTED_NET_LIST=$NET_LIST"
echo "DETECTED_UCX_LIST=$UCX_LIST"
echo "DETECTED_IB_IPS=$IB_IP_LIST"
echo "DETECTED_MGMT_IP=$MGMT_IP"

echo "{ib_end}"
""".format(
    accel_start=_ACCEL_START,
    accel_end=_ACCEL_END,
    ib_start=_IB_START,
    ib_end=_IB_END,
)


def generate_combined_probe_script() -> str:
    """Return the combined accelerator + IB probe script.

    Run over SSH and parse stdout with :func:`split_probe_output`.
    """
    return _COMBINED_PROBE_SCRIPT


# ---------------------------------------------------------------------------
# Output splitting
# ---------------------------------------------------------------------------


def split_probe_output(stdout: str) -> tuple[str, str]:
    """Split combined probe stdout into accelerator and IB sections.

    Args:
        stdout: Raw stdout from :func:`generate_combined_probe_script`.

    Returns:
        ``(accel_section, ib_section)`` — the raw text between the
        respective sentinel markers.  Either may be empty if the
        sentinel was not present (e.g. probe aborted early).
    """
    accel_section = _extract_section(stdout, _ACCEL_START, _ACCEL_END)
    ib_section = _extract_section(stdout, _IB_START, _IB_END)
    return accel_section, ib_section


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Return text between *start_marker* and *end_marker* lines, or ``""``."""
    lines = text.splitlines()
    collecting = False
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == start_marker:
            collecting = True
            continue
        if stripped == end_marker:
            collecting = False
            continue
        if collecting:
            result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# High-level probe functions
# ---------------------------------------------------------------------------


def probe_host(
    host: str,
    ssh_kwargs: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> HostHardware:
    """Run a single combined SSH probe and return :class:`HostHardware`.

    Executes one bash script over SSH that emits both the accelerator
    fingerprint data and InfiniBand detection data.  Parsing the combined
    stdout populates both :attr:`HostHardware.accelerators` (with
    fingerprint hash) and :attr:`HostHardware.ib_info`.

    This replaces calling :func:`~sparkrun.core.fingerprint.fingerprint_host`
    and :func:`~sparkrun.orchestration.infiniband.detect_ib_for_hosts`
    separately — one SSH round-trip instead of two.

    Args:
        host: Hostname or IP.
        ssh_kwargs: Passed through to
            :func:`~sparkrun.orchestration.ssh.run_remote_script`
            (``ssh_user``, ``ssh_key``, ``ssh_options``, …).
        dry_run: Return an empty :class:`HostHardware` without SSH.

    Returns:
        :class:`HostHardware` with ``accelerators``, ``fingerprint``,
        ``notes``, and ``ib_info`` populated.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    if dry_run:
        logger.info("[dry-run] Would probe host %s", host)
        return HostHardware(notes="dry-run probe")

    result = run_remote_script(
        host,
        generate_combined_probe_script(),
        timeout=30,
        **(ssh_kwargs or {}),
    )
    if not result.success:
        logger.warning("Hardware probe failed on %s: %s", host, result.stderr[:200])
        return HostHardware(notes="hardware probe failed: %s" % result.stderr[:120])

    return _parse_probe_result(result.stdout)


def probe_hosts(
    hosts: list[str],
    ssh_kwargs: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, HostHardware]:
    """Run :func:`probe_host` on multiple hosts in parallel.

    Uses :func:`~sparkrun.orchestration.ssh.run_remote_scripts_parallel`
    for the SSH layer (one SSH connection per host, all concurrent).

    Args:
        hosts: Hostnames or IPs to probe.
        ssh_kwargs: SSH connection parameters (forwarded to parallel helper).
        dry_run: Return empty :class:`HostHardware` per host without SSH.

    Returns:
        Mapping of host → :class:`HostHardware`.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if not hosts:
        return {}

    if dry_run:
        logger.info("[dry-run] Would probe %d host(s)", len(hosts))
        return {host: HostHardware(notes="dry-run probe") for host in hosts}

    kw = ssh_kwargs or {}
    logger.info("Probing hardware on %d host(s): %s", len(hosts), ", ".join(hosts))
    results = run_remote_scripts_parallel(
        hosts,
        generate_combined_probe_script(),
        timeout=30,
        **kw,
    )

    hardware: dict[str, HostHardware] = {}
    for result in results:
        if not result.success:
            logger.warning("Hardware probe failed on %s: %s", result.host, result.stderr[:200])
            hardware[result.host] = HostHardware(notes="hardware probe failed: %s" % result.stderr[:120])
        else:
            hardware[result.host] = _parse_probe_result(result.stdout)

    return hardware


def _parse_probe_result(stdout: str) -> HostHardware:
    """Parse combined probe stdout into a :class:`HostHardware`."""
    from sparkrun.core.fingerprint import build_host_hardware, parse_fingerprint_output
    from sparkrun.utils import parse_kv_output

    accel_section, ib_section = split_probe_output(stdout)

    if accel_section:
        parsed_accel = parse_fingerprint_output(accel_section)
        hw = build_host_hardware(parsed_accel)
    else:
        logger.warning("Accelerator section missing from combined probe output")
        hw = HostHardware(notes="accelerator probe section missing")

    if ib_section:
        ib_info = parse_kv_output(ib_section)
    else:
        logger.debug("IB section missing from combined probe output")
        ib_info = None

    # Attach ib_info to a new HostHardware (dataclass is not frozen)
    hw.ib_info = ib_info
    return hw
