# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.
"""Plugin availability and receipt-backed workload controls."""

from __future__ import annotations

import importlib.util
import time

from .protocol import ProtocolError


def plugin_availability() -> list[dict]:
    from sparkrun.core.recipe_items import registered_recipe_items

    registrations = list(registered_recipe_items())
    enabled = any(item.owner == "sparkrun.plugins.coldsnap" for item in registrations)
    try:
        installed = importlib.util.find_spec("sparkrun.plugins.coldsnap") is not None
        from sparkrun.plugins.coldsnap.api import LIFECYCLE_API_VERSION

        supported = LIFECYCLE_API_VERSION == 1
    except (ImportError, ModuleNotFoundError):
        installed, supported = False, False
        try:
            installed = importlib.util.find_spec("sparkrun.plugins.coldsnap") is not None
        except (ImportError, ModuleNotFoundError):
            pass
    return [{"name": "coldsnap", "installed": installed, "enabled": enabled, "lifecycle": enabled and supported}]


def _state_path(sctx):
    from .jobs import _connect, _path

    return _connect(_path(sctx))


def _stored_state(job, sctx) -> str:
    with _state_path(sctx) as db:
        db.execute("CREATE TABLE IF NOT EXISTS workload_states (job_id TEXT PRIMARY KEY, state TEXT, updated REAL)")
        row = db.execute("SELECT state FROM workload_states WHERE job_id=?", (job.cluster_id,)).fetchone()
    return row["state"] if row else str((job.metadata or {}).get("runtime_info", {}).get("lifecycle_state") or "running")


def _set_state(job, state, sctx):
    with _state_path(sctx) as db:
        db.execute("CREATE TABLE IF NOT EXISTS workload_states (job_id TEXT PRIMARY KEY, state TEXT, updated REAL)")
        db.execute("INSERT OR REPLACE INTO workload_states VALUES (?, ?, ?)", (job.cluster_id, state, time.time()))


def describe_job(job, *, sctx, availability=None) -> dict:
    metadata = job.metadata or {}
    runtime = metadata.get("runtime_info") or {}
    used = ["coldsnap"] if runtime.get("execution_strategy") == "coldsnap" else []
    available = availability if availability is not None else plugin_availability()
    supported = any(p["name"] == "coldsnap" and p["lifecycle"] for p in available)
    owned = metadata.get("owner") == "sparkroute"
    actions = ["status"] if used and supported and runtime.get("capture_id") else []
    if actions and owned:
        actions += ["sleep", "wake"]
    return {
        "job_id": job.cluster_id,
        "cluster_name": str(metadata.get("cluster") or ""),
        "recipe_revision": str(metadata.get("recipe_fingerprint") or ""),
        "owned": owned,
        "plugins_in_use": used,
        "lifecycle_actions": actions,
        "lifecycle_state": _stored_state(job, sctx) if used else "running",
    }


def inspect_workloads(*, sctx) -> dict:
    import sparkrun.api as api

    availability = plugin_availability()
    return {
        "plugins": availability,
        "workloads": [describe_job(j, sctx=sctx, availability=availability) for j in api.list_jobs(sctx=sctx)[:256]],
    }


def control_workload(request, *, sctx) -> dict:
    import sparkrun.api as api

    job = next((j for j in api.list_jobs(sctx=sctx) if j.cluster_id == request.cluster_id), None)
    if job is None or request.binding is None:
        raise ProtocolError("workload_not_found", "The configured workload is no longer available")
    info = describe_job(job, sctx=sctx)
    binding = request.binding
    if (
        not binding.recipe_revision
        or binding.recipe_revision != info["recipe_revision"]
        or (binding.cluster_candidates and info["cluster_name"] not in binding.cluster_candidates)
    ):
        raise ProtocolError("workload_mismatch", "The job does not belong to this recipe binding and cluster")
    action = "status" if request.operation == "workload_status" else request.operation
    if action not in info["lifecycle_actions"]:
        raise ProtocolError("lifecycle_unavailable", "This workload does not have an enabled ColdSnap lifecycle API or is externally owned")
    from sparkrun.plugins.coldsnap.api import control_job

    if action != "status":
        _set_state(job, "sleeping_pending" if action == "sleep" else "waking", sctx)
    try:
        result = control_job(action, job, sctx=sctx)
    except Exception as exc:
        # An interrupted operation may have changed the remote state. Never
        # advertise it as ready until a later explicit status/wake reconciles it.
        raise ProtocolError(
            "lifecycle_failed", "ColdSnap could not verify or control this exact job; check its status before retrying", retryable=True
        ) from exc
    _set_state(job, result["state"], sctx)
    return {**describe_job(job, sctx=sctx), "state": result["state"]}


def wake_existing(binding, *, sctx):
    """Wake an existing receipt-backed job before the normal readiness path."""
    import sparkrun.api as api
    from .protocol import Request

    for job in api.list_jobs(sctx=sctx):
        metadata = job.metadata or {}
        if metadata.get("recipe_fingerprint") != binding.recipe_revision or (
            binding.cluster_candidates and metadata.get("cluster") not in binding.cluster_candidates
        ):
            continue
        if (metadata.get("runtime_info") or {}).get("execution_strategy") != "coldsnap":
            continue
        request = Request(request_id="wake-existing", operation="workload_status", binding=binding, cluster_id=job.cluster_id)
        result = control_workload(request, sctx=sctx)
        if result["state"] in {"sleeping", "warm"}:
            from dataclasses import replace

            control_workload(replace(request, operation="wake"), sctx=sctx)
