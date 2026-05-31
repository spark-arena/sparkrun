"""Process/runtime environment helpers (SSH user, env merge, logging config)."""

from __future__ import annotations

import logging
import os

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


def resolve_ssh_user(cluster_user: str | None, config, fallback: str = "root") -> str:
    """Resolve SSH user from cluster definition, config, or OS environment."""
    return cluster_user or config.ssh_user or os.environ.get("USER", fallback)


def merge_env(*env_dicts: dict[str, str] | None) -> dict[str, str]:
    """Merge multiple environment dicts (later values win)."""
    merged: dict[str, str] = {}
    for d in env_dicts:
        if d:
            merged.update(d)
    return merged
