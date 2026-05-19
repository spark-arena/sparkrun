"""Per-host cache directory status + free-space probe and table formatting.

Shared between ``sparkrun cluster inspect`` (full directory status display)
and the model-distribution error path (show disk-space table when rsync fails
with "no space left").  The remote probe gathers sparkrun cache existence/size,
HF cache existence/size, and root-FS free space in a single SSH round-trip;
the table formatter renders them in the same layout used by ``cluster inspect``
for consistency.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheStatus:
    """Per-host cache directory status and free-space readout."""

    host: str
    sparkrun_dir: str = "?"
    sparkrun_exists: bool = False
    sparkrun_size: str = "-"
    hf_dir: str = "?"
    hf_exists: bool = False
    hf_size: str = "-"
    free_space: str = "-"
    error: str | None = None
    """Populated when the probe itself failed (SSH error, command not found)."""


# Single shell command that captures everything we care about in one round trip.
# `du -sh` is portable; `df -h /` + awk handles busybox + GNU coreutils equally.
_CACHE_STATUS_PROBE = (
    'sr_dir="{sr_dir}"; hf_dir="{hf_dir}"; '
    'sr_exists="no"; hf_exists="no"; sr_du="-"; hf_du="-"; '
    'if [ -d "$sr_dir" ]; then sr_exists="yes"; sr_du=$(du -sh "$sr_dir" 2>/dev/null | cut -f1); fi; '
    'if [ -d "$hf_dir" ]; then hf_exists="yes"; hf_du=$(du -sh "$hf_dir" 2>/dev/null | cut -f1); fi; '
    'free_space=$(df -h / 2>/dev/null | awk "NR==2{{print \\$4}}"); '
    'echo "sr_exists=$sr_exists|sr_du=$sr_du|hf_exists=$hf_exists|hf_du=$hf_du'
    '|sr_dir=$sr_dir|hf_dir=$hf_dir|free_space=${{free_space:--}}"'
)


def _parse_probe_output(host: str, stdout: str) -> CacheStatus:
    """Parse the single-line ``key=value|key=value`` output from the probe."""
    parts: dict[str, str] = {}
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    for entry in line.split("|"):
        if "=" in entry:
            key, _, value = entry.partition("=")
            parts[key.strip()] = value.strip()
    return CacheStatus(
        host=host,
        sparkrun_dir=parts.get("sr_dir", "?"),
        sparkrun_exists=parts.get("sr_exists") == "yes",
        sparkrun_size=parts.get("sr_du", "-"),
        hf_dir=parts.get("hf_dir", "?"),
        hf_exists=parts.get("hf_exists") == "yes",
        hf_size=parts.get("hf_du", "-"),
        free_space=parts.get("free_space", "-"),
    )


def probe_cache_status(
    hosts: list[str],
    hf_cache_dir: str,
    sparkrun_cache_dir: str | None = None,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> dict[str, CacheStatus]:
    """Probe each host for cache directory status and root-FS free space.

    Args:
        hosts: Remote hostnames (management IPs).
        hf_cache_dir: HuggingFace cache path on the remote hosts.
        sparkrun_cache_dir: Optional explicit sparkrun cache path; when
            ``None``, the probe uses ``$HOME/.cache/sparkrun`` resolved on
            each remote host.
        ssh_kwargs: Standard SSH parameters.
        dry_run: When ``True``, returns synthetic placeholders without
            making any SSH calls.

    Returns:
        Map of ``host → CacheStatus``.  Hosts where the probe itself
        failed have :attr:`CacheStatus.error` populated and remaining
        fields left as default placeholders.
    """
    if not hosts:
        return {}

    sr_dir = sparkrun_cache_dir or "$HOME/.cache/sparkrun"
    cmd = _CACHE_STATUS_PROBE.format(sr_dir=sr_dir, hf_dir=hf_cache_dir)

    if dry_run:
        return {h: CacheStatus(host=h, sparkrun_dir=sr_dir, hf_dir=hf_cache_dir) for h in hosts}

    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    kw = ssh_kwargs or {}
    results = run_remote_scripts_parallel(
        hosts,
        cmd,
        timeout=15,
        quiet=True,
        **kw,
    )

    out: dict[str, CacheStatus] = {}
    for r in results:
        if r.success and r.stdout.strip():
            out[r.host] = _parse_probe_output(r.host, r.stdout)
        else:
            out[r.host] = CacheStatus(
                host=r.host,
                sparkrun_dir=sr_dir,
                hf_dir=hf_cache_dir,
                error=(r.stderr or "").strip() or "SSH failed",
            )
    return out


def probe_local_cache_status(
    hf_cache_dir: str,
    sparkrun_cache_dir: str,
) -> CacheStatus:
    """Inspect the control machine's caches and free space.

    Mirrors :func:`probe_cache_status` but uses local ``os.path.isdir`` +
    ``du`` + ``df`` so callers (e.g. ``cluster inspect``) can include a
    ``(local)`` row alongside the remote rows.
    """

    def _dir_info(path: str) -> tuple[bool, str]:
        if not os.path.isdir(path):
            return False, "-"
        du = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
        size = du.stdout.split()[0] if du.returncode == 0 and du.stdout.strip() else "-"
        return True, size

    sr_exists, sr_du = _dir_info(sparkrun_cache_dir)
    hf_exists, hf_du = _dir_info(hf_cache_dir)

    free_space = "-"
    df = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
    if df.returncode == 0 and df.stdout.strip():
        lines = df.stdout.strip().splitlines()
        if len(lines) >= 2:
            free_space = lines[1].split()[3]

    return CacheStatus(
        host="(local)",
        sparkrun_dir=sparkrun_cache_dir,
        sparkrun_exists=sr_exists,
        sparkrun_size=sr_du,
        hf_dir=hf_cache_dir,
        hf_exists=hf_exists,
        hf_size=hf_du,
        free_space=free_space,
    )
