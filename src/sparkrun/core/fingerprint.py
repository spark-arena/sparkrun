"""Per-host hardware fingerprinting over SSH.

Phase 6 of the hardware-agnostic refactor: introspects a host's
accelerators (NVIDIA / AMD / Intel Gaudi / Apple Silicon) plus IB
adapters, returning a :class:`~sparkrun.core.hardware.HostHardware`
plus a stable fingerprint hash.

The probe runs a single normalised bash script over SSH (mirroring the
:mod:`sparkrun.orchestration.infiniband` ``ib_detect.sh`` pattern):
vendor-specific CLIs do their parsing in bash and emit ``KEY=VALUE``
lines, so Python only needs to consume a structured key/value blob.
This keeps parser tests deterministic and avoids depending on volatile
vendor CLI output formats.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from sparkrun.core.hardware import AcceleratorSpec, HostHardware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probe script
# ---------------------------------------------------------------------------

_FINGERPRINT_SCRIPT = r"""#!/bin/bash
set -uo pipefail

# Normalised KEY=VALUE output consumed by parse_fingerprint_output().
# Bash does the vendor-specific parsing so Python parsers stay decoupled
# from `nvidia-smi --format=csv` / `rocm-smi --json` output drift.

emit() { printf '%s=%s\n' "$1" "$2"; }

# --- NVIDIA ---
NVIDIA_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
    # Each line: "name, memory_total_mib"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        name=$(printf '%s' "$line" | awk -F', *' '{print $1}')
        mem=$(printf '%s' "$line" | awk -F', *' '{print $2}' | awk '{print $1}')
        emit "NVIDIA_GPU_${NVIDIA_COUNT}_NAME" "$name"
        emit "NVIDIA_GPU_${NVIDIA_COUNT}_MEMORY_MIB" "$mem"
        NVIDIA_COUNT=$((NVIDIA_COUNT + 1))
    done < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
fi
emit NVIDIA_GPU_COUNT "$NVIDIA_COUNT"
emit NVIDIA_PRESENT "$([[ $NVIDIA_COUNT -gt 0 ]] && echo 1 || echo 0)"

# --- AMD ROCm ---
AMD_COUNT=0
if command -v rocm-smi >/dev/null 2>&1; then
    # rocm-smi --showproductname --showmeminfo vram outputs lines like:
    #   GPU[0]    : Card series:  Instinct MI300X
    #   GPU[0]    : VRAM Total Memory (B): 205520896000
    while IFS= read -r line; do
        if [[ "$line" =~ GPU\[([0-9]+)\][[:space:]]*:[[:space:]]*Card[[:space:]]series:[[:space:]]*(.*) ]]; then
            idx="${BASH_REMATCH[1]}"
            name="${BASH_REMATCH[2]}"
            emit "AMD_GPU_${idx}_NAME" "$name"
            [[ $idx -ge $AMD_COUNT ]] && AMD_COUNT=$((idx + 1))
        elif [[ "$line" =~ GPU\[([0-9]+)\][[:space:]]*:[[:space:]]*VRAM[[:space:]]Total[[:space:]]Memory[[:space:]]\(B\):[[:space:]]*([0-9]+) ]]; then
            idx="${BASH_REMATCH[1]}"
            bytes="${BASH_REMATCH[2]}"
            mib=$((bytes / 1024 / 1024))
            emit "AMD_GPU_${idx}_MEMORY_MIB" "$mib"
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
        name=$(printf '%s' "$line" | awk -F', *' '{print $1}')
        mem=$(printf '%s' "$line" | awk -F', *' '{print $2}' | awk '{print $1}')
        emit "INTEL_GAUDI_${INTEL_COUNT}_NAME" "$name"
        emit "INTEL_GAUDI_${INTEL_COUNT}_MEMORY_MIB" "$mem"
        INTEL_COUNT=$((INTEL_COUNT + 1))
    done < <(hl-smi --query-aip=name,memory.total --format=csv,noheader 2>/dev/null || true)
fi
emit INTEL_GAUDI_COUNT "$INTEL_COUNT"
emit INTEL_PRESENT "$([[ $INTEL_COUNT -gt 0 ]] && echo 1 || echo 0)"

# --- Apple Silicon ---
APPLE_PRESENT=0
APPLE_MODEL=""
if [[ "$(uname -s)" == "Darwin" ]] && command -v system_profiler >/dev/null 2>&1; then
    chip=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip:/ {print $2}' | head -1)
    if [[ -n "$chip" ]]; then
        APPLE_PRESENT=1
        APPLE_MODEL="$chip"
    fi
fi
emit APPLE_PRESENT "$APPLE_PRESENT"
emit APPLE_MODEL "$APPLE_MODEL"

# --- InfiniBand / RoCE (capability tag only — not a separate accelerator) ---
IB_PRESENT=0
if [[ -d /sys/class/infiniband ]] && compgen -G "/sys/class/infiniband/*" >/dev/null; then
    IB_PRESENT=1
fi
emit IB_PRESENT "$IB_PRESENT"

# --- OS / arch (debugging aid) ---
emit OS "$(uname -s 2>/dev/null || echo unknown)"
emit ARCH "$(uname -m 2>/dev/null || echo unknown)"
"""


def generate_fingerprint_script() -> str:
    """Return the bash probe script.  Run over SSH and parse stdout with :func:`parse_fingerprint_output`."""
    return _FINGERPRINT_SCRIPT


# ---------------------------------------------------------------------------
# Pure parsers (deterministic — used by tests and the SSH wrapper alike)
# ---------------------------------------------------------------------------


def parse_fingerprint_output(stdout: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines emitted by :func:`generate_fingerprint_script`."""
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def _mib_to_gb(mib: str | None) -> float | None:
    if not mib:
        return None
    try:
        # Convert MiB → GiB (binary), then to GB-ish for HostHardware which expects GB.
        # We keep "GB" colloquial — same convention as DEFAULT_VRAM_GB (121 ≈ 128 GiB).
        return round(int(mib) / 1024.0, 1)
    except ValueError:
        return None


def _normalize_nvidia_model(raw: str) -> str:
    """Lower-snake-case a vendor product string.  ``"NVIDIA GB10"`` → ``"gb10"``."""
    name = raw.strip()
    for prefix in ("NVIDIA ", "Tesla ", "RTX "):
        if name.upper().startswith(prefix.upper()):
            name = name[len(prefix) :]
            break
    return name.lower().replace(" ", "-")


def _normalize_amd_model(raw: str) -> str:
    name = raw.strip()
    for prefix in ("Instinct ",):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.lower().replace(" ", "-")


def _normalize_intel_model(raw: str) -> str:
    return raw.strip().lower().replace(" ", "-")


def _normalize_apple_model(raw: str) -> str:
    # "Apple M5", "Apple M5 Max" -> "m5", "m5-max"
    name = raw.strip()
    if name.lower().startswith("apple "):
        name = name[6:]
    return name.lower().replace(" ", "-")


def _group_accelerators(
    entries: list[tuple[str, str, float | None]],
    vendor: str,
    capabilities: frozenset[str],
) -> list[AcceleratorSpec]:
    """Compact a list of per-index (idx, model, memory_gb) into AcceleratorSpec groups."""
    if not entries:
        return []
    grouped: list[AcceleratorSpec] = []
    current_model: str | None = None
    current_mem: float | None = None
    current_count = 0
    for _idx, model, mem in entries:
        if model == current_model and mem == current_mem:
            current_count += 1
            continue
        if current_model is not None:
            grouped.append(
                AcceleratorSpec(
                    vendor=vendor,
                    model=current_model,
                    count=current_count,
                    memory_gb=current_mem,
                    capabilities=capabilities,
                )
            )
        current_model = model
        current_mem = mem
        current_count = 1
    if current_model is not None:
        grouped.append(
            AcceleratorSpec(
                vendor=vendor,
                model=current_model,
                count=current_count,
                memory_gb=current_mem,
                capabilities=capabilities,
            )
        )
    return grouped


def build_host_hardware(parsed: dict[str, str]) -> HostHardware:
    """Construct :class:`HostHardware` from a parsed fingerprint dict."""
    has_ib = parsed.get("IB_PRESENT") == "1"
    rdma_cap = frozenset({"rdma:roce-v2"}) if has_ib else frozenset()

    accelerators: list[AcceleratorSpec] = []

    # NVIDIA
    nvidia_count = int(parsed.get("NVIDIA_GPU_COUNT", "0") or 0)
    if nvidia_count > 0:
        entries = []
        for i in range(nvidia_count):
            raw_name = parsed.get("NVIDIA_GPU_%d_NAME" % i, "")
            mem = _mib_to_gb(parsed.get("NVIDIA_GPU_%d_MEMORY_MIB" % i))
            entries.append((i, _normalize_nvidia_model(raw_name), mem))
        caps = frozenset({"cuda"}) | rdma_cap
        accelerators.extend(_group_accelerators(entries, vendor="nvidia", capabilities=caps))

    # AMD
    amd_count = int(parsed.get("AMD_GPU_COUNT", "0") or 0)
    if amd_count > 0:
        entries = []
        for i in range(amd_count):
            raw_name = parsed.get("AMD_GPU_%d_NAME" % i, "")
            mem = _mib_to_gb(parsed.get("AMD_GPU_%d_MEMORY_MIB" % i))
            entries.append((i, _normalize_amd_model(raw_name), mem))
        caps = frozenset({"rocm"}) | rdma_cap
        accelerators.extend(_group_accelerators(entries, vendor="amd", capabilities=caps))

    # Intel Gaudi
    intel_count = int(parsed.get("INTEL_GAUDI_COUNT", "0") or 0)
    if intel_count > 0:
        entries = []
        for i in range(intel_count):
            raw_name = parsed.get("INTEL_GAUDI_%d_NAME" % i, "")
            mem = _mib_to_gb(parsed.get("INTEL_GAUDI_%d_MEMORY_MIB" % i))
            entries.append((i, _normalize_intel_model(raw_name), mem))
        caps = frozenset({"habana"}) | rdma_cap
        accelerators.extend(_group_accelerators(entries, vendor="intel", capabilities=caps))

    # Apple Silicon
    if parsed.get("APPLE_PRESENT") == "1":
        model = parsed.get("APPLE_MODEL", "")
        if model:
            accelerators.append(
                AcceleratorSpec(
                    vendor="apple",
                    model=_normalize_apple_model(model),
                    count=1,
                    memory_gb=None,
                    capabilities=frozenset({"mlx"}),
                )
            )

    fingerprint = compute_fingerprint_hash(accelerators)
    return HostHardware(
        accelerators=accelerators,
        fingerprint=fingerprint,
        notes=_detection_note(parsed),
    )


def compute_fingerprint_hash(accelerators: list[AcceleratorSpec]) -> str:
    """Stable, content-addressed hash of the accelerator list.

    SHA-256 of a canonical JSON serialization (sorted keys, accelerator
    list order preserved — order is meaningful for heterogeneous-on-host
    setups).  Truncated to 16 hex chars.
    """
    canonical = [
        {
            "vendor": a.vendor,
            "model": a.model,
            "count": a.count,
            "memory_gb": a.memory_gb,
            "capabilities": sorted(a.capabilities),
        }
        for a in accelerators
    ]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _detection_note(parsed: dict[str, str]) -> str:
    os_name = parsed.get("OS", "?")
    arch = parsed.get("ARCH", "?")
    return "fingerprinted from %s/%s" % (os_name, arch)


# ---------------------------------------------------------------------------
# SSH wrapper
# ---------------------------------------------------------------------------


def fingerprint_host(host: str, ssh_kwargs: dict[str, Any] | None = None) -> HostHardware:
    """Run the probe script over SSH and parse the result into ``HostHardware``.

    Args:
        host: Hostname or IP.
        ssh_kwargs: Passed straight through to
            :func:`sparkrun.orchestration.ssh.run_remote_script`.

    Returns:
        Parsed :class:`HostHardware` with a populated ``fingerprint``.
        An empty hardware (no accelerators) is returned when the probe
        runs but detects nothing — callers should treat this as
        "manual configuration required".
    """
    from sparkrun.orchestration.ssh import run_remote_script

    result = run_remote_script(
        host,
        generate_fingerprint_script(),
        timeout=30,
        **(ssh_kwargs or {}),
    )
    if not result.success:
        logger.warning("Fingerprint probe failed on %s: %s", host, result.stderr[:200])
        return HostHardware(notes="fingerprint probe failed: %s" % result.stderr[:120])

    parsed = parse_fingerprint_output(result.stdout)
    return build_host_hardware(parsed)
