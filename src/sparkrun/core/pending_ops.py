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
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from sparkrun.core.config import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

PENDING_DIR_NAME = "pending"

# A lock older than this is treated as stale regardless of whether its PID
# still appears alive.  Guards against PID reuse (a dead launcher's PID being
# recycled by an unrelated process) and abandoned locks on an NFS-shared cache
# where ``os.kill(pid, 0)`` is meaningless for a remote host's PID.  Generous
# enough to cover a slow multi-GB model download + image distribution to a
# large cluster.
LOCK_MAX_AGE_SECONDS = 12 * 60 * 60  # 12 hours


def _pending_dir(cache_dir: str | None = None) -> Path:
    base = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return base / PENDING_DIR_NAME


def _lock_path(pending_dir: Path, cluster_id: str, operation: str) -> Path:
    safe_cid = cluster_id.replace("/", "_")
    return pending_dir / f"{safe_cid}_{operation}.json"


def _hostname() -> str:
    """Best-effort short hostname for lock identity."""
    try:
        return socket.gethostname() or ""
    except OSError:
        return ""


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process is still running (Unix only)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _is_stale(info: dict) -> bool:
    """Return True if *info* describes a lock that is no longer live.

    A lock is stale when ANY of the following holds:

    - It exceeds :data:`LOCK_MAX_AGE_SECONDS` (secondary signal that also
      guards against PID reuse, where a recycled PID would otherwise read as
      "alive" forever).
    - It was created on *this* host and the owning PID is no longer running.
      PID liveness is only meaningful for locks created on the same host —
      a PID from another host (NFS-shared cache) tells us nothing, so such
      locks rely solely on the max-age signal.
    """
    started_at = info.get("started_at")
    if isinstance(started_at, (int, float)) and (time.time() - started_at) > LOCK_MAX_AGE_SECONDS:
        return True

    lock_host = info.get("host")
    # Only trust PID liveness for same-host locks (or legacy locks that
    # predate host recording, where host is absent → assume local).
    if lock_host in (None, "", _hostname()):
        pid = info.get("pid", -1)
        if not _is_pid_alive(pid):
            return True

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
    token: str | None = None,
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
        token: Ownership token to record.  When ``None``, a fresh one is
            generated.  Pass an explicit token (see :func:`pending_op`) when
            the caller needs to compare against the on-disk token later.

    Returns:
        Path to the created lock file.  When a live lock for the same key
        already exists, it is left intact (a concurrent run must not clobber
        it) and the existing path is returned unchanged.

    A unique ``token`` (and the creating ``host``) is recorded so the lock
    can be removed only by its owner — see :func:`remove_pending_op`.
    """
    d = _pending_dir(cache_dir)
    d.mkdir(parents=True, exist_ok=True)

    info = {
        "cluster_id": cluster_id,
        "operation": operation,
        "pid": os.getpid(),
        "host": _hostname(),
        "token": token or uuid.uuid4().hex,
        "started_at": time.time(),
        "recipe": recipe,
        "model": model,
        "image": image,
        "hosts": hosts or [],
    }

    path = _lock_path(d, cluster_id, operation)
    try:
        # A concurrent same-key run must NOT overwrite an existing live lock —
        # write only when the lock is absent or stale (ours to reclaim).
        if path.exists():
            existing = _read_lock(path)
            if existing is not None and not _is_stale(existing):
                logger.debug(
                    "Pending-op lock already held by pid %s on %s: %s — not clobbering",
                    existing.get("pid"),
                    existing.get("host") or "?",
                    path.name,
                )
                # Leave the held lock intact and return its path.  The caller's
                # token (if any) won't match the on-disk token, so its
                # context-exit removal is correctly suppressed.
                return path
        path.write_text(json.dumps(info))
    except OSError:
        logger.debug("Could not write pending-op lock: %s", path)
    return path


def _read_lock(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def remove_pending_op(
    cluster_id: str,
    operation: str,
    cache_dir: str | None = None,
    *,
    token: str | None = None,
) -> None:
    """Remove a pending-operation lock file.

    When *token* is provided, the lock is unlinked only if its recorded
    ``token`` matches — so a second run that found the key already held (and
    therefore did not write its own lock) cannot delete the first run's lock
    on its way out.  When *token* is ``None``, the lock is removed
    unconditionally (legacy/no-ownership behavior).
    """
    path = _lock_path(_pending_dir(cache_dir), cluster_id, operation)
    try:
        if token is not None:
            info = _read_lock(path)
            if info is not None and info.get("token") not in (None, token):
                logger.debug(
                    "Not removing pending-op lock owned by another run (token mismatch): %s",
                    path.name,
                )
                return
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

    Any keyword arguments are forwarded to :func:`create_pending_op`.  This
    invocation mints its own ownership *token* up front; on exit the lock is
    removed only when the on-disk token still matches that token.  If another
    live run already held the key (so ``create_pending_op`` left their lock
    intact), our token won't match and we leave their lock alone — concurrent
    runs sharing a key never delete each other's lock.
    """
    own_token = uuid.uuid4().hex
    create_pending_op(cluster_id, operation, token=own_token, **kwargs)
    try:
        yield
    finally:
        remove_pending_op(
            cluster_id,
            operation,
            cache_dir=kwargs.get("cache_dir"),
            token=own_token,
        )


def list_pending_ops(cache_dir: str | None = None) -> list[dict]:
    """Return all live pending operations, pruning stale ones.

    A lock is considered stale when it exceeds :data:`LOCK_MAX_AGE_SECONDS`
    or (for same-host locks) when the owning PID is no longer running — see
    :func:`_is_stale`.  Stale and corrupt locks are silently deleted.
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
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            continue

        if _is_stale(info):
            logger.debug(
                "Pruning stale pending-op lock: %s (pid=%s host=%s)",
                path.name,
                info.get("pid"),
                info.get("host") or "?",
            )
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            continue

        # Add human-readable elapsed time
        elapsed = time.time() - info.get("started_at", time.time())
        info["elapsed_seconds"] = round(elapsed, 1)
        ops.append(info)

    return ops
