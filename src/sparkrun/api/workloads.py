"""Reusable workload activation, recovery and exact-identity lifecycle control.

.. warning::
   **Experimental.** The functions, receipts and result dictionaries in this
   module may change in a minor release while the gateway integrations that
   drive them settle. Pin a sparkrun version if you depend on them.

Domain receipts survive daemon job retention. There is no queue or detached
worker here: every action runs in the supplied operation context. Integrations
supply a fixed owner policy; remote adapters must never accept an arbitrary
owner label as authorization.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import sqlite3
import time
from typing import Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import SparkrunError
from sparkrun.core.coordination import reserve_resources
from sparkrun.core.operation import operation_scope
from sparkrun.core.workload_lifecycle import lifecycle_provider, registered_workload_lifecycles


class WorkloadError(SparkrunError):
    def __init__(self, code, message, *, retryable=False):
        super().__init__(message)
        self.code, self.retryable = code, retryable


@dataclass(frozen=True)
class WorkloadBinding:
    recipe: str
    recipe_revision: str = ""
    cluster_candidates: tuple[str, ...] = ()
    overrides: dict[str, Any] = field(default_factory=dict)


class Receipts:
    """Placement/effect journal; contains no worker ownership or submission queue."""

    def __init__(self, sctx):
        self.path = sctx.config.config_path.parent / "workload-receipts.sqlite3"
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS activations (id TEXT PRIMARY KEY, revision TEXT, owner TEXT, placement TEXT, phase TEXT, updated REAL)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS lifecycle (id TEXT PRIMARY KEY, state TEXT, updated REAL)")
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def placements(self, revision, owner):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM activations WHERE revision=? AND owner=? ORDER BY updated DESC", (revision, owner)).fetchall()
        return [{**dict(row), "placement": _placement(row["placement"])} for row in rows]

    def save(self, identifier, revision, owner, placement, phase):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO activations VALUES(?,?,?,?,?,?)",
                (identifier, revision, owner, json.dumps(placement), phase, time.time()),
            )

    def blocked_endpoints(self):
        with self.connect() as db:
            rows = db.execute("SELECT placement,phase FROM activations").fetchall()
        return {
            p["workload_id"]
            for row in rows
            if (p := _placement(row["placement"]))
            and row["phase"] not in {"ready", "stopped"}
            and (row["phase"] == "hooks_started" or p.get("post_launch_required"))
        }

    def state(self, identifier, value=None):
        with self.connect() as db:
            if value is not None:
                db.execute("INSERT OR REPLACE INTO lifecycle VALUES(?,?,?)", (identifier, value, time.time()))
            row = db.execute("SELECT state FROM lifecycle WHERE id=?", (identifier,)).fetchone()
            return row[0] if row else None


def _placement(raw):
    """Decode a stored placement; receipts written before the rename say ``cluster_id``."""
    placement = json.loads(raw)
    if "workload_id" not in placement and "cluster_id" in placement:
        placement["workload_id"] = placement.pop("cluster_id")
    return placement


def resolve_binding(binding, *, sctx=None):
    from sparkrun import api
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    sctx = resolve_sctx(sctx)
    recipe, overrides = api.resolve_catalog_recipe(binding.recipe, binding.overrides, sctx=sctx)
    revision = derive_recipe_fingerprint(recipe, overrides)
    if binding.recipe_revision and revision != binding.recipe_revision:
        raise WorkloadError("recipe_revision_mismatch", "Recipe revision changed; refresh the binding before executing it")
    return recipe, revision, overrides


def plugins(*, sctx=None):
    """Installed and loaded inventory with registered lifecycle capabilities."""
    from sparkrun.core.plugin_inventory import list_plugins

    sctx = resolve_sctx(sctx)
    providers = {p.name for p in registered_workload_lifecycles()}
    return [
        {
            "name": row.name,
            "plugin_id": row.plugin_id,
            "enabled": row.enabled,
            "loaded": row.loaded,
            "version": row.version,
            "lifecycle": row.enabled and row.loaded and row.plugin_id in providers,
        }
        for row in list_plugins(sctx.config, sctx.variables)
    ]


def _matches(job, revision, clusters=()):
    metadata = job.metadata or {}
    return metadata.get("recipe_fingerprint") == revision and (not clusters or metadata.get("cluster") in clusters)


def endpoints(*, recipe_revision="", cluster_candidates=(), workload_id="", sctx=None):
    """Return live, credential-free endpoint observations filtered before probing."""
    from sparkrun import api
    from sparkrun.proxy.discovery import discover_snapshot

    sctx = resolve_sctx(sctx)
    jobs = api.list_jobs(sctx=sctx)
    allowed = {
        j.workload_id
        for j in jobs
        if (not recipe_revision or _matches(j, recipe_revision, cluster_candidates))
        and (not cluster_candidates or j.metadata.get("cluster") in cluster_candidates)
        and (not workload_id or j.workload_id == workload_id)
    }
    journal = Receipts(sctx)
    allowed = {identifier for identifier in allowed if journal.state(identifier) in (None, "running")} - journal.blocked_endpoints()
    snapshot = discover_snapshot(cluster_ids=allowed, sctx=sctx)
    # A successful live HTTP probe is required even when host occupancy scope is
    # absent. Metadata model names alone must not authorize endpoint admission.
    value = snapshot.to_dict()
    return {**value, "endpoints": [ep for ep in value["endpoints"] if ep["healthy"] and ep["actual_models"]]}


def inventory(*, owner="", sctx=None):
    """Recorded workloads, including unfinished placements after caller loss.

    ``complete`` permits absence decisions only when no execution is in flight
    and the bounded response includes every recorded workload.
    """
    from sparkrun.core.coordination import ResourceBusy

    sctx = resolve_sctx(sctx)
    try:
        # No wait: a busy controller already has an answer here (incomplete).
        with reserve_resources(sctx.config.config_path.parent, {"*"}, wait=0):
            return _inventory(owner, sctx, complete=True)
    except ResourceBusy:
        return _inventory(owner, sctx, complete=False)


def _inventory(owner, sctx, *, complete):
    from sparkrun import api

    receipts = Receipts(sctx)
    result = [
        _workload_row(job.workload_id, job.metadata or {}, lifecycle_provider(job), owner, receipts) for job in api.list_jobs(sctx=sctx)
    ]
    known = {row["workload_id"] for row in result}
    with receipts.connect() as db:
        pending = db.execute("SELECT * FROM activations WHERE phase NOT IN ('ready', 'stopped') ORDER BY updated DESC").fetchall()
    for row in pending:
        placement = _placement(row["placement"])
        identifier = placement["workload_id"]
        if identifier in known or receipts.state(identifier) == "stopped":
            continue
        known.add(identifier)
        metadata = {
            "recipe_fingerprint": row["revision"],
            "owner": row["owner"],
            "cluster": placement.get("cluster"),
            "runtime_info": {"lifecycle_state": row["phase"]},
        }
        result.append(_workload_row(identifier, metadata, None, owner, receipts))
    return {
        "workloads": result[:256],
        "complete": complete and len(result) <= 256,
        "providers": [{"name": p.name, "version": p.version} for p in registered_workload_lifecycles()],
    }


def _workload_row(workload_id, metadata, provider, owner, receipts):
    owned = bool(owner and metadata.get("owner") == owner)
    return {
        "workload_id": workload_id,
        "cluster_name": metadata.get("cluster"),
        "recipe_revision": str(metadata.get("recipe_fingerprint") or ""),
        "owner": str(metadata.get("owner") or ""),
        "owned": owned,
        "provider": provider.name if provider else None,
        "actions": (["status", "sleep", "wake"] if owned else ["status"]) if provider else [],
        "state": receipts.state(workload_id) or str(metadata.get("runtime_info", {}).get("lifecycle_state") or "unknown"),
        "observation": "recorded",
    }


def _ready_result(endpoint, *, adopted, owner, sctx):
    from sparkrun import api

    identifier = endpoint["workload_id"]
    job = next((job for job in api.list_jobs(sctx=sctx) if job.workload_id == identifier), None)
    metadata = (
        (job.metadata or {})
        if job
        else {
            "owner": endpoint.get("owner", ""),
            "cluster": endpoint.get("cluster_name"),
            "recipe_fingerprint": endpoint.get("recipe_revision", ""),
        }
    )
    return {
        "state": "ready",
        "workload_id": identifier,
        "endpoint": {key: value for key, value in endpoint.items() if key != "observed_at"},
        "observed_at": endpoint["observed_at"],
        "adopted": adopted,
        "workload": _workload_row(identifier, metadata, lifecycle_provider(job) if job else None, owner, Receipts(sctx)),
    }


def _exact_job(binding, workload_id, owner, sctx, *, mutation):
    from sparkrun import api

    if not workload_id or not binding.recipe_revision:
        raise WorkloadError("workload_identity_required", "Exact workload ID and recipe revision are required")
    job = next((job for job in api.list_jobs(sctx=sctx) if job.workload_id == workload_id), None)
    if job is None or not _matches(job, binding.recipe_revision, binding.cluster_candidates):
        raise WorkloadError("workload_mismatch", "Workload does not match this recipe revision and cluster selection")
    if mutation and (not owner or job.metadata.get("owner") != owner):
        raise WorkloadError("job_not_owned", "Workload is not owned by this integration")
    return job


def control(binding: WorkloadBinding, workload_id: str, action: str, *, owner: str, sctx=None):
    """Guard and execute a provider operation for an exact persisted workload."""
    if action not in {"status", "sleep", "wake"}:
        raise ValueError("Unsupported workload action")
    sctx = resolve_sctx(sctx)
    mutation = action != "status"
    with operation_scope(sctx), reserve_resources(sctx.config.config_path.parent, {"*"}):
        job = _exact_job(binding, workload_id, owner, sctx, mutation=mutation)
        provider = lifecycle_provider(job)
        if provider is None:
            raise WorkloadError("lifecycle_unavailable", "No enabled lifecycle provider supports this workload")
        receipts = Receipts(sctx)
        if mutation:
            receipts.state(workload_id, "sleeping_pending" if action == "sleep" else "waking")
        result = provider.control(action, job, sctx=sctx)
        if result.get("workload_id") != workload_id or not isinstance(result.get("state"), str):
            raise WorkloadError("lifecycle_identity_mismatch", "Provider returned a different or invalid workload identity")
        receipts.state(workload_id, result["state"])
        return _workload_row(workload_id, job.metadata or {}, provider, owner, receipts)


def stop(binding: WorkloadBinding, workload_id: str, *, owner: str, sctx=None):
    from sparkrun import api

    sctx = resolve_sctx(sctx)
    with operation_scope(sctx), reserve_resources(sctx.config.config_path.parent, {"*"}):
        _exact_job(binding, workload_id, owner, sctx, mutation=True)
        result = api.stop(workload_id=workload_id, sctx=sctx)
        if not result.success:
            raise WorkloadError("stop_failed", "Could not confirm workload teardown", retryable=True)
        receipts = Receipts(sctx)
        receipts.state(workload_id, "stopped")
        for row in receipts.placements(binding.recipe_revision, owner):
            if row["placement"]["workload_id"] == workload_id:
                receipts.save(row["id"], binding.recipe_revision, owner, row["placement"], "stopped")
        return {"state": "offline", "workload_ids": [workload_id]}


def _ready(binding, revision, sctx, *, workload_id="", owner=""):
    observation = endpoints(recipe_revision=revision, cluster_candidates=binding.cluster_candidates, workload_id=workload_id, sctx=sctx)
    found = [{**ep, "observed_at": observation["observed_at"]} for ep in observation["endpoints"]]
    found.sort(key=lambda ep: ep["owner"] != owner)
    return found[0] if found else None


def _await_existing(binding, recipe, revision, placement, deadline, sctx, operation):
    from sparkrun import api

    snapshot = api.status(placement["hosts"], cluster=placement.get("cluster"), sctx=sctx)
    if snapshot.observation_errors or set(placement["hosts"]) - {h.host for h in snapshot.hosts}:
        raise WorkloadError("recovery_unavailable", "Previous placement could not be completely observed", retryable=True)
    if placement["workload_id"] not in snapshot.running_cluster_ids():
        return None
    if (recipe.post_exec or recipe.post_commands) and not placement.get("post_launch_completed"):
        raise WorkloadError(
            "recovery_requires_operator", "Previous workload may have incomplete post-launch hooks; inspect it before retrying"
        )
    while time.monotonic() < deadline:
        operation.check()
        if endpoint := _ready(binding, revision, sctx, workload_id=placement["workload_id"]):
            return endpoint
        operation.cancel.wait(min(1, max(0, deadline - time.monotonic())))
    raise WorkloadError("readiness_timeout", "Previous workload is still running but not serving the expected model", retryable=True)


def ensure_ready(binding: WorkloadBinding, *, owner: str, timeout: float = 900, sctx=None):
    """Adopt or recover before launch; one operation owns placement through readiness.

    Launches always run with ``trust=False``: a recipe whose hooks need explicit
    authorization fails with ``WorkloadError("trust_required")`` before any
    placement receipt is written.
    """
    from sparkrun import api
    from dataclasses import replace

    if not owner or not 0 < timeout <= 3600:
        raise ValueError("An owner policy and a timeout between 0 and 3600 seconds are required")
    sctx = resolve_sctx(sctx)
    with operation_scope(sctx) as operation, reserve_resources(sctx.config.config_path.parent, {"*"}):
        deadline = min(time.monotonic() + timeout, operation.deadline or float("inf"))
        # An exact existing workload can outlive an edit to its source recipe.
        # Resolve current launch contents only when recovery/launch needs them.
        # Receipt checks still precede adoption, including uncertain hook effects.
        resolved = resolve_binding(binding, sctx=sctx) if not binding.recipe_revision else None
        revision = resolved[1] if resolved is not None else binding.recipe_revision
        binding = replace(binding, recipe_revision=revision)
        receipts = Receipts(sctx)
        previous = [
            r
            for r in receipts.placements(revision, owner)
            if r["phase"] != "stopped" and (not binding.cluster_candidates or r["placement"].get("cluster") in binding.cluster_candidates)
        ]
        uncertain_hooks = [r for r in previous if r["phase"] == "hooks_started"]
        if uncertain_hooks:
            raise WorkloadError(
                "recovery_requires_operator", "A previous post-launch action has uncertain effects; reconcile it before retrying"
            )
        if endpoint := _ready(binding, revision, sctx, owner=owner):
            return _ready_result(endpoint, adopted=True, owner=owner, sctx=sctx)
        operation.emit("progress", stage="recovery", message="Checking previous workload placements")
        jobs = [j for j in api.list_jobs(sctx=sctx) if _matches(j, revision, binding.cluster_candidates)]
        for job in jobs:
            if lifecycle_provider(job) and job.metadata.get("owner") == owner:
                result = control(binding, job.workload_id, "status", owner=owner, sctx=sctx)
                if result["state"] in {"sleeping", "warm"}:
                    control(binding, job.workload_id, "wake", owner=owner, sctx=sctx)
                    if endpoint := _ready(binding, revision, sctx, owner=owner):
                        return _ready_result(endpoint, adopted=True, owner=owner, sctx=sctx)
        recipe, revision, overrides = resolved or resolve_binding(binding, sctx=sctx)
        placements = [{**r["placement"], "post_launch_completed": r["phase"] == "ready"} for r in previous]
        known = {p["workload_id"] for p in placements}
        for job in jobs:
            if job.workload_id not in known and job.hosts:
                placements.append({"workload_id": job.workload_id, "cluster": job.metadata.get("cluster"), "hosts": list(job.hosts)})
        for placement in placements:
            if endpoint := _await_existing(binding, recipe, revision, placement, deadline, sctx, operation):
                return _ready_result(endpoint, adopted=True, owner=owner, sctx=sctx)
        result = None
        for candidate in binding.cluster_candidates or (None,):
            operation.check()
            if time.monotonic() >= deadline:
                raise WorkloadError("readiness_timeout", "Activation budget expired before launch", retryable=True)
            options = api.RunOptions(
                recipe=recipe, cluster=candidate, overrides=overrides, auto_port=True, detached=True, follow=False, trust=False, owner=owner
            )
            try:
                plan = api.plan(options, sctx=sctx)
            except api.InsufficientCapacity:
                continue
            # ``run`` below is called with trust=False, so untrusted hooks would
            # raise TrustRejected only after the receipt was saved, leaving a
            # ``launch_started`` placement every later call has to re-observe.
            if api.untrusted_hooks(plan.recipe, trust=False, sctx=sctx):
                raise WorkloadError(
                    "trust_required",
                    "Recipe hooks come from an untrusted source; trust its registry or launch it with explicit trust first",
                )
            placement = {
                "workload_id": plan.workload_id,
                "hosts": list(plan.host_list),
                "cluster": plan.cluster.name if plan.cluster else None,
                "port": plan.recipe.defaults.get("port", 8000),
                "post_launch_required": bool(recipe.post_exec or recipe.post_commands),
            }
            receipts.save(operation.id, revision, owner, placement, "launch_started")
            operation.emit("progress", stage="launch", message="Launching workload", workload_id=plan.workload_id)
            result = api.run(options, plan=plan, sctx=sctx)
            break
        if result is None:
            raise WorkloadError("insufficient_capacity", "No selected cluster has capacity", retryable=True)
        if result.rc or (result.recipe_fingerprint and result.recipe_fingerprint != revision):
            raise WorkloadError("activation_failed", "Launch did not complete with the requested recipe revision", retryable=True)
        if getattr(result, "already_running", False) or getattr(result, "launch_result", None) is None:
            endpoint = _await_existing(binding, recipe, revision, placement, deadline, sctx, operation)
            if endpoint is None:
                raise WorkloadError("recovery_unavailable", "Launch did not record a recoverable running workload")
            receipts.save(operation.id, revision, owner, placement, "ready")
            return _ready_result(endpoint, adopted=True, owner=owner, sctx=sctx)
        receipts.save(operation.id, revision, owner, placement, "readiness")
        # A separate deadline event bounds readiness without cancelling the
        # caller's durable operation or losing its placement receipt.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkloadError("readiness_timeout", "Activation exceeded its readiness budget", retryable=True)
        readiness = api.wait_ready(result, timeout=remaining, sctx=sctx)
        if not readiness.ready:
            raise WorkloadError("readiness_failed", "Workload readiness could not be confirmed", retryable=True)
        if recipe.post_exec or recipe.post_commands:
            receipts.save(operation.id, revision, owner, placement, "hooks_started")
            api.post_launch(result, trust=False, readiness=readiness, sctx=sctx)
        receipts.save(operation.id, revision, owner, placement, "ready")
        endpoint = _ready(binding, revision, sctx, workload_id=result.workload_id, owner=owner)
        if endpoint is None:
            raise WorkloadError("endpoint_unavailable", "Workload did not publish the expected live model identity", retryable=True)
        return _ready_result(endpoint, adopted=False, owner=owner, sctx=sctx)


def reconcile(binding: WorkloadBinding, workload_id: str, *, owner: str, acknowledge_effects=False, sctx=None):
    """Clear uncertain hooks only after operator acknowledgement and live evidence.

    This does not re-run hooks or mutate a remote workload. A complete absent
    placement releases its old effect journal; a healthy matching workload is
    marked complete. Unknown coverage remains blocked.
    """
    from sparkrun import api
    from sparkrun.proxy.discovery import discover_snapshot

    if not acknowledge_effects:
        raise WorkloadError("acknowledgement_required", "Inspect post-launch effects, then explicitly acknowledge their reconciliation")
    sctx = resolve_sctx(sctx)
    with operation_scope(sctx), reserve_resources(sctx.config.config_path.parent, {"*"}):
        if not binding.recipe_revision or not workload_id:
            raise WorkloadError("workload_identity_required", "Exact workload ID and recipe revision are required")
        journal = Receipts(sctx)
        rows = [
            r
            for r in journal.placements(binding.recipe_revision, owner)
            if r["placement"]["workload_id"] == workload_id
            and (not binding.cluster_candidates or r["placement"].get("cluster") in binding.cluster_candidates)
        ]
        if not rows:
            raise WorkloadError("receipt_not_found", "No matching activation receipts")
        placement = rows[0]["placement"]
        status = api.status(placement["hosts"], cluster=placement.get("cluster"), sctx=sctx)
        if status.observation_errors or set(placement["hosts"]) - {h.host for h in status.hosts}:
            raise WorkloadError("recovery_unavailable", "Placement could not be completely observed", retryable=True)
        phase = "stopped"
        if workload_id in status.running_cluster_ids():
            _exact_job(binding, workload_id, owner, sctx, mutation=True)
            observed = discover_snapshot(cluster_ids={workload_id}, sctx=sctx)
            if not any(ep.healthy and ep.actual_models and ep.recipe_revision == binding.recipe_revision for ep in observed.endpoints):
                raise WorkloadError("recovery_unavailable", "Workload is running but its live model identity is unverified", retryable=True)
            phase = "ready"
        for row in rows:
            journal.save(row["id"], binding.recipe_revision, owner, row["placement"], phase)
        journal.state(workload_id, "running" if phase == "ready" else "stopped")
        return {"workload_id": workload_id, "state": phase, "receipts": len(rows)}


__all__ = [
    "WorkloadBinding",
    "WorkloadError",
    "resolve_binding",
    "endpoints",
    "inventory",
    "ensure_ready",
    "stop",
    "control",
    "plugins",
    "reconcile",
]
