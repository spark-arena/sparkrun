"""Shared utility functions for sparkrun.

Small, self-contained helpers that are used across multiple modules.
Keeping them here avoids circular imports and reduces duplication.
"""

from __future__ import annotations
import logging

# Loggers that produce excessive output at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore.http11",
    "httpcore.connection",
    "urllib3.connectionpool",
)


def suppress_noisy_loggers() -> None:
    """Suppress verbose HTTP/transport loggers."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def coerce_value(value: str):
    """Coerce a string value to int, float, or bool where possible."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def resolve_ssh_user(cluster_user: str | None, config, fallback: str = "root") -> str:
    """Resolve SSH user from cluster definition, config, or OS environment."""
    import os

    return cluster_user or config.ssh_user or os.environ.get("USER", fallback)


def is_valid_ip(ip: str) -> bool:
    """Basic check if a string looks like an IPv4 address."""
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def parse_kv_output(output: str) -> dict[str, str]:
    """Parse key=value lines from script output.

    Lines starting with ``#`` are ignored. Leading/trailing whitespace
    on keys and values is stripped.

    Args:
        output: Raw stdout containing key=value lines.

    Returns:
        Dictionary of parsed key=value pairs.
    """
    result: dict[str, str] = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def load_yaml(path) -> dict:
    """Load a YAML file, returning an empty dict on parse failure."""
    from pathlib import Path as _Path
    import yaml

    with _Path(path).open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def parse_scoped_name(name: str) -> tuple[str | None, str]:
    """Parse ``@registry/lookup_name`` into ``(registry, lookup_name)``.

    Returns ``(None, name)`` when the input has no ``@`` prefix or
    no ``/`` separator.
    """
    if name.startswith("@") and "/" in name:
        prefix, lookup_name = name.split("/", 1)
        return prefix[1:], lookup_name  # strip leading @
    return None, name


def merge_env(*env_dicts: dict[str, str] | None) -> dict[str, str]:
    """Merge multiple environment dicts (later values win)."""
    merged: dict[str, str] = {}
    for d in env_dicts:
        if d:
            merged.update(d)
    return merged


def is_local_host(host: str) -> bool:
    """Check if a host string refers to the local machine."""
    return host in ("localhost", "127.0.0.1", "")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Returns ``"Xs"`` for durations under 60s, ``"Xm Ys"`` for durations
    under an hour, and ``"Xh Ym Zs"`` for longer durations.
    """
    s = int(seconds)
    if s < 60:
        return "%.1fs" % seconds
    m, s = divmod(s, 60)
    if m < 60:
        return "%dm %ds" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %dm %ds" % (h, m, s)
