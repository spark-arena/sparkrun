# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Controller-local durable operations, shared by one-shot bridge processes.

The short SQLite transaction serializes admission only. Each workload runs in
its own detached process; losing the requesting gateway never kills a launch.
The pre-launch placement is recorded before touching the remote workload.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, replace
from contextlib import contextmanager

from .protocol import ProtocolError, Request, parse_request

_ID = re.compile(r"^[0-9a-f]{32}$")
_current: tuple[Path, str] | None = None


@contextmanager
def _connect(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Create with restricted permissions before SQLite opens it.
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY, key TEXT, request TEXT, state TEXT, phase TEXT, pid INTEGER, updated REAL, result TEXT, error TEXT, placement TEXT)"
    )
    try:
        with db:
            yield db
    finally:
        db.close()


def _path(sctx) -> Path:
    return Path(sctx.config.config_path).parent / "sparkroute" / "operations.sqlite3"


def _public(row) -> dict:
    value = {"operation_id": row["id"], "state": row["state"], "phase": row["phase"], "updated_at": row["updated"]}
    if row["placement"]:
        placement = json.loads(row["placement"])
        value["cluster_id"] = placement.get("cluster_id", "")
        value["cluster_name"] = placement.get("cluster", "")
    for key in ("result", "error"):
        if row[key]:
            value[key] = json.loads(row[key])
    return value


def _alive(pid) -> bool:
    from sparkrun.utils.process import process_exists

    return process_exists(pid, inaccessible=True)


def start_operation(request: Request, *, sctx) -> dict:
    from uuid import uuid4

    binding = asdict(request.binding) if request.binding else {}
    # File aliases of the same resolved recipe share a worker on this cluster.
    identity = {
        "operation": "activation" if request.operation in {"ensure_ready", "sleep", "wake"} else request.operation,
        "revision": binding.get("recipe_revision"),
        "clusters": binding.get("cluster_candidates"),
        "arguments": request.arguments if request.operation.startswith("catalog_") else {},
    }
    if request.binding and not identity["revision"]:
        from .operations import _resolve_binding

        _, fingerprint = _resolve_binding(request.binding, sctx)
        request = replace(request, binding=replace(request.binding, recipe_revision=fingerprint))
        identity["revision"] = fingerprint
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = _path(sctx)
    with _connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if request.operation in {"ensure_ready", "sleep", "wake"}:
            for active in db.execute("SELECT * FROM operations WHERE state='running' AND key!=?", (key,)).fetchall():
                other = json.loads(active["request"])
                other_binding = other.get("binding") or {}
                if other_binding.get("recipe_revision") != binding.get("recipe_revision") or not _alive(active["pid"]):
                    continue
                a, b = set(binding.get("cluster_candidates") or ()), set(other_binding.get("cluster_candidates") or ())
                if not a or not b or a.intersection(b):
                    raise ProtocolError(
                        "workload_busy",
                        "A lifecycle operation is still running on an overlapping cluster; retry after it finishes",
                        retryable=True,
                    )
        row = db.execute(
            "SELECT * FROM operations WHERE key=? AND state IN ('running', 'failed') ORDER BY updated DESC LIMIT 1", (key,)
        ).fetchone()
        if row and row["state"] == "running" and _alive(row["pid"]):
            original = json.loads(row["request"])
            if original["operation"] != request.operation or original.get("cluster_id", "") != request.cluster_id:
                raise ProtocolError(
                    "workload_busy", "A lifecycle operation is already running for this recipe; retry after it finishes", retryable=True
                )
            return _public(row)
        if row:
            operation_id = row["id"]  # resume, preserving the last placement
        else:
            operation_id = uuid4().hex
            db.execute(
                "INSERT INTO operations VALUES (?, ?, ?, 'running', 'queued', NULL, ?, NULL, NULL, NULL)",
                (operation_id, key, json.dumps(asdict(request)), time.time()),
            )
        # The child blocks on this transaction until its PID is committed.
        kwargs = (
            {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        try:
            child = subprocess.Popen(
                [sys.executable, "-m", "sparkrun.plugins.sparkroute.jobs", str(sctx.config.config_path), operation_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                **kwargs,
            )
        except OSError as exc:
            raise ProtocolError("worker_unavailable", "Could not start the sparkrun operation worker", retryable=True) from exc
        db.execute(
            "UPDATE operations SET pid=?, updated=?, request=?, state='running', result=NULL, error=NULL WHERE id=?",
            (child.pid, time.time(), json.dumps(asdict(request)), operation_id),
        )
        row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
        # Retain terminal records for a week; running operations never expire.
        cutoff = time.time() - 7 * 86400
        expired = db.execute("SELECT id FROM operations WHERE state!='running' AND updated<?", (cutoff,)).fetchall()
        db.execute("DELETE FROM operations WHERE state!='running' AND updated<?", (cutoff,))
        for item in expired:
            if _ID.fullmatch(item["id"]):
                for suffix in (".log", ".log.1"):
                    try:
                        (path.parent / (item["id"] + suffix)).unlink(missing_ok=True)
                    except OSError:
                        pass  # Log cleanup must not roll back an admitted worker.
        return _public(row)


def operation_status(operation_id: str, *, sctx) -> dict:
    if not _ID.fullmatch(operation_id):
        raise ProtocolError("operation_not_found", "Operation was not found")
    with _connect(_path(sctx)) as db:
        row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
    if not row:
        raise ProtocolError("operation_not_found", "Operation was not found")
    if row["state"] == "running" and not _alive(row["pid"]):
        value = _public(row)
        value.update(
            state="failed",
            phase="interrupted",
            error={
                "code": "worker_interrupted",
                "message": "The sparkrun worker stopped. Retry the model request to reconcile and resume its launch.",
                "retryable": True,
            },
        )
        return value
    return _public(row)


def wait_operation(operation: dict, timeout: float, *, sctx) -> dict:
    """Synchronous callers wait on the same durable operation as async callers."""
    deadline = time.monotonic() + timeout
    while operation["state"] == "running":
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProtocolError("activation_timeout", "Activation continues in the background; retry to check its progress", retryable=True)
        time.sleep(min(0.5, remaining))
        operation = operation_status(operation["operation_id"], sctx=sctx)
    if operation["state"] == "failed":
        error = operation["error"]
        raise ProtocolError(error["code"], error["message"], retryable=error["retryable"])
    return operation["result"]


def progress(phase: str, placement: dict | None = None) -> None:
    if _current is None:
        return
    path, operation_id = _current
    with _connect(path) as db:
        db.execute(
            "UPDATE operations SET phase=?, updated=?, placement=COALESCE(?, placement) WHERE id=?",
            (phase, time.time(), json.dumps(placement) if placement else None, operation_id),
        )


def previous_placement() -> dict | None:
    if _current is None:
        return None
    path, operation_id = _current
    with _connect(path) as db:
        row = db.execute("SELECT placement FROM operations WHERE id=?", (operation_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def run_worker(config_path: Path, operation_id: str) -> None:
    import sparkrun.api as api
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.cluster_manager import ClusterManager
    import logging
    from logging.handlers import RotatingFileHandler
    from .operations import _ensure_ready, _require_feature_enabled, _resolve_binding

    global _current
    sctx = api.default_sctx()
    sctx.config = SparkrunConfig(config_path)
    sctx.cluster_manager = ClusterManager(config_path.parent)
    path = _path(sctx)
    with _connect(path) as db:
        row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
    if not row or row["state"] != "running" or row["pid"] != os.getpid():
        return
    _current = (path, operation_id)
    log_path = path.parent / (operation_id + ".log")
    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    handler = RotatingFileHandler(log_path, maxBytes=256 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        _require_feature_enabled()
        request = parse_request(json.loads(row["request"]))
        if request.operation in {"sleep", "wake"}:
            from .workload_plugins import control_workload

            progress(request.operation + " workload")
            result = control_workload(request, sctx=sctx)
        elif request.operation == "catalog_capacity":
            progress("checking cluster occupancy")
            result = api.catalog_cluster_capacity(request.arguments["cluster"], sctx=sctx)
        elif request.operation == "catalog_refresh":
            progress("refreshing registries")
            result = api.refresh_registries(
                sctx=sctx, progress=lambda name, ok: progress("refreshed " + name if ok else "could not refresh " + name)
            )
        else:
            progress("resolving recipe")
            recipe, fingerprint = _resolve_binding(request.binding, sctx)
            result = _ensure_ready(replace(request, wait=True), request.binding, recipe, fingerprint, sctx)
        with _connect(path) as db:
            db.execute(
                "UPDATE operations SET state='succeeded', phase='complete', result=?, updated=? WHERE id=?",
                (json.dumps(result), time.time(), operation_id),
            )
    except Exception as exc:
        logging.getLogger(__name__).exception("sparkrun operation failed")
        error = (
            exc
            if isinstance(exc, ProtocolError)
            else ProtocolError("operation_failed", "sparkrun operation failed; inspect the controller logs", retryable=True)
        )
        with _connect(path) as db:
            db.execute(
                "UPDATE operations SET state='failed', phase='failed', error=?, updated=? WHERE id=?",
                (json.dumps({"code": error.code, "message": error.message, "retryable": error.retryable}), time.time(), operation_id),
            )
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
        _current = None


if __name__ == "__main__":
    sys.modules[__package__ + ".jobs"] = sys.modules[__name__]
    run_worker(Path(sys.argv[1]), sys.argv[2])
