"""Lightweight pending-operation tracking for sparkrun.

When ``sparkrun run`` is downloading HuggingFace models or distributing
container images, no Docker containers are running yet.  This means
``sparkrun cluster status`` would show idle hosts even though a launch
is actively in progress (and will soon consume VRAM).

This module writes small "lock" files into the sparkrun cache directory
(``~/.cache/sparkrun/pending/``) so that ``cluster status`` can report
operations that are underway.

Lock files are best-effort: they are cleaned up on normal exit via a
context manager.  Stale locks (where the owning PID has exited) are
automatically pruned when the pending directory is read.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from sparkrun.core.config import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

PENDING_DIR_NAME = "pending"


def _pending_dir(cache_dir: str | None = None) -> Path:
    base = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return base / PENDING_DIR_NAME


def _lock_path(pending_dir: Path, cluster_id: str, operation: str) -> Path:
    safe_cid = cluster_id.replace("/", "_")
    return pending_dir / f"{safe_cid}_{operation}.json"


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process is still running (Unix only)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def create_pending_op(
    cluster_id: str,
    operation: str,
    *,
    recipe: str = "",
    model: str = "",
    image: str = "",
    hosts: list[str] | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Write a pending-operation lock file.

    Args:
        cluster_id: The sparkrun cluster id (e.g. ``sparkrun_abc123``).
        operation: Short tag — ``"model_download"``, ``"image_distribute"``,
            ``"image_pull"``, etc.
        recipe: Recipe name for display.
        model: Model identifier for display.
        image: Container image reference for display.
        hosts: Target hosts.
        cache_dir: Override for the sparkrun cache directory.

    Returns:
        Path to the created lock file.
    """
    d = _pending_dir(cache_dir)
    d.mkdir(parents=True, exist_ok=True)

    info = {
        "cluster_id": cluster_id,
        "operation": operation,
        "pid": os.getpid(),
        "started_at": time.time(),
        "recipe": recipe,
        "model": model,
        "image": image,
        "hosts": hosts or [],
    }

    path = _lock_path(d, cluster_id, operation)
    try:
        path.write_text(json.dumps(info))
    except OSError:
        logger.debug("Could not write pending-op lock: %s", path)
    return path


def remove_pending_op(
    cluster_id: str,
    operation: str,
    cache_dir: str | None = None,
) -> None:
    """Remove a pending-operation lock file."""
    path = _lock_path(_pending_dir(cache_dir), cluster_id, operation)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not remove pending-op lock: %s", path)


@contextmanager
def pending_op(
    cluster_id: str,
    operation: str,
    **kwargs,
):
    """Context manager that creates a lock on entry and removes it on exit.

    Any keyword arguments are forwarded to :func:`create_pending_op`.
    """
    create_pending_op(cluster_id, operation, **kwargs)
    try:
        yield
    finally:
        remove_pending_op(cluster_id, operation, cache_dir=kwargs.get("cache_dir"))


def list_pending_ops(cache_dir: str | None = None) -> list[dict]:
    """Return all live pending operations, pruning stale ones.

    A lock is considered stale when the PID that created it is no longer
    running.  Stale locks are silently deleted.
    """
    d = _pending_dir(cache_dir)
    if not d.is_dir():
        return []

    ops: list[dict] = []
    for path in d.glob("*.json"):
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # Corrupt lock — remove it
            path.unlink(missing_ok=True)
            continue

        pid = info.get("pid", -1)
        if not _is_pid_alive(pid):
            # Owning process is gone — stale lock
            logger.debug("Pruning stale pending-op lock: %s (pid %d)", path.name, pid)
            path.unlink(missing_ok=True)
            continue

        # Add human-readable elapsed time
        elapsed = time.time() - info.get("started_at", time.time())
        info["elapsed_seconds"] = round(elapsed, 1)
        ops.append(info)

    return ops
