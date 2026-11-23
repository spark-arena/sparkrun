"""Cross-process coordination shared by library, CLI, and daemon callers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import logging
import inspect
from pathlib import Path
import sqlite3
import threading
import time

from sparkrun.core.process_identity import process_identity
from sparkrun.utils.fs import file_lock

logger = logging.getLogger(__name__)

_HELD = ContextVar("sparkrun_resources", default=frozenset())
_OWNERS = ContextVar("sparkrun_reservation_owners", default=())
_DOCUMENTS = threading.local()


class ResourceBusy(RuntimeError):
    """Another owner holds a resource this operation needs.

    ``holders`` lists the conflicting claims (``resource``, ``owner``,
    ``pid``, ``alive``, ``command``) so callers can say who to wait for or
    which dead claim to reconcile.
    """

    def __init__(self, message: str, holders: list[dict] | None = None):
        super().__init__(message)
        self.holders = holders or []


#: How long a reservation waits out a live ``*`` hold before refusing.  Those
#: are brief controller-wide sweeps (``workloads.inventory`` polled by a
#: client), so a launch colliding with one should not fail outright.
CONTROLLER_HOLD_WAIT_S = 3.0
_RETRY_INTERVAL_S = 0.25


class Conflict(ValueError):
    """A request conflicts with recorded state (an idempotency key, a job's lifecycle).

    Lives here rather than in ``sparkrun.daemon.jobs`` (which re-exports it) so
    the console-free wire layer can classify it without importing the daemon.
    The class name is part of the wire contract: error bodies carry it as
    ``error.type``.
    """


@contextmanager
def document_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = str((root / ".documents.lock").resolve())
    held = getattr(_DOCUMENTS, "held", set())
    if path in held:
        yield
        return
    with file_lock(path):
        _DOCUMENTS.held = held | {path}
        try:
            yield
        finally:
            _DOCUMENTS.held = held


def locked_document(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        with document_lock(self.config_root):
            return function(self, *args, **kwargs)

    return invoke


@contextmanager
def reserve_resources(root, keys, *, wait: float = CONTROLLER_HOLD_WAIT_S):
    """Atomically reserve destinations. Uncertain dead-owner claims stay visible.

    A conflict made only of live ``*`` holds is retried for up to *wait*
    seconds; any other conflict raises :class:`ResourceBusy` at once.
    """
    import json
    import uuid

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    keys = frozenset(keys or {"*"})
    held = _HELD.get()
    if keys <= held or "*" in held:
        yield
        return
    owner = uuid.uuid4().hex
    path = root / "operations.sqlite3"
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS reservations (owner TEXT, resource TEXT UNIQUE, process TEXT)")
        path.chmod(0o600)
        deadline = time.monotonic() + wait
        while True:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT resource, owner, process FROM reservations").fetchall()
            conflicts = [row for row in rows if row[0] == "*" or "*" in keys or row[0] in keys - held]
            if not conflicts:
                break
            db.execute("ROLLBACK")
            holders = _describe_holders(conflicts)
            brief = all(h["resource"] == "*" and h["alive"] for h in holders)
            if not brief or time.monotonic() >= deadline:
                raise ResourceBusy(_busy_message(holders), holders)
            time.sleep(_RETRY_INTERVAL_S)
        for key in sorted(keys - held):
            db.execute("INSERT INTO reservations VALUES(?,?,?)", (owner, key, json.dumps(process_identity())))
        db.commit()
        token = _HELD.set(held | keys)
        owners = _OWNERS.set(_OWNERS.get() + ((path, owner),))
        try:
            yield
        finally:
            _OWNERS.reset(owners)
            _HELD.reset(token)
            db.execute("DELETE FROM reservations WHERE owner=?", (owner,))
    finally:
        db.close()


def _describe_holders(rows):
    import json
    from sparkrun.core.process_identity import alive, process_command

    holders = []
    for resource, owner, process in sorted(rows):
        try:
            identity = json.loads(process)
        except (TypeError, ValueError):
            identity = None
        try:
            is_alive = alive(identity)
        except PermissionError:
            is_alive = None  # Cannot inspect: not proof of death.
        pid = (identity or {}).get("pid")
        holders.append(
            {
                "resource": resource,
                "owner": owner,
                "pid": pid,
                "alive": is_alive,
                "command": process_command(pid) if is_alive and pid else None,
            }
        )
    return holders


def _busy_message(holders):
    """Name each conflicting owner once, with what to do about it."""
    lines = ["Execution resources are busy:"]
    by_owner: dict[str, list[dict]] = {}
    for holder in holders:
        by_owner.setdefault(holder["owner"], []).append(holder)
    for owner, claims in by_owner.items():
        first = claims[0]
        resources = ", ".join(h["resource"] for h in claims)
        if first["alive"] is False:
            lines.append(
                "  %s: held by pid %s, which has exited without releasing it. Check those hosts for leftover workloads, then run "
                "`sparkrun daemon resources reconcile %s --acknowledge-effects`" % (resources, first["pid"], owner)
            )
        elif first["alive"] is None:
            lines.append("  %s: held by pid %s, whose status could not be inspected" % (resources, first["pid"]))
        else:
            running = " (%s)" % first["command"] if first["command"] else ""
            lines.append("  %s: held by running pid %s%s; retry when it finishes" % (resources, first["pid"], running))
    lines.append("Inspect all claims with `sparkrun daemon resources list`.")
    return "\n".join(lines)


def release_reservations():
    """Drop every reservation this context holds, ahead of a long non-mutating wait.

    For an operation whose mutating part is over but whose call keeps running
    (a ``proxy start --foreground`` supervising its gateway for days). Holding
    the claim through the wait refused every launch on those hosts for as long
    as the proxy ran. The enclosing ``reserve_resources`` exits normally later;
    its delete is then a no-op.

    Releases *every* claim in this context, an outer caller's included, so call
    it only where nothing further up still needs exclusivity.
    """
    for path, owner in _OWNERS.get():
        try:
            db = sqlite3.connect(path, timeout=10, isolation_level=None)
            try:
                db.execute("DELETE FROM reservations WHERE owner=?", (owner,))
            finally:
                db.close()
        except sqlite3.Error as exc:
            # Never fail the caller (a gateway already running): the enclosing
            # reserve_resources still deletes the claim when it exits.
            logger.warning("Could not release reservation %s early: %s", owner, exc)
    _OWNERS.set(())
    _HELD.set(frozenset())


def reservation_keys(hosts=None, *, executor_config=None, controller=False):
    """Canonical destination keys shared by daemon admission and API calls."""
    keys = {"host:" + str(host).rsplit("@", 1)[-1].lower() for host in hosts} if hosts and not controller else {"*"}
    if executor_config:
        import json

        keys.add("destination:" + hashlib.sha256(json.dumps(executor_config, sort_keys=True).encode()).hexdigest())
    return keys


def infer_reservation_hosts(config, *, cluster=None, workload_id=None, benchmark_id=None):
    """Hosts a call naming no hosts will act on, and the cluster they came from.

    Shared by API calls (``operation_boundary``) and daemon jobs, which must
    lock the same machines for the same request. Mirrors
    ``api._resolve.resolve_cluster``: named cluster → the job's recorded hosts
    → the default cluster (``cluster set-default``) → ``config.default_hosts``.
    A ``benchmark_id`` (resume) is answered by the benchmark's recorded hosts,
    which need not be the default cluster's; unknown state reserves ``*``.
    Skipping the default-cluster step locked ``default_hosts`` while the call
    tore down the default cluster's hosts: two concurrent mutations holding
    disjoint locks over the same machines. ``(None, cluster)`` means unknown,
    which callers reserve as ``*``.
    """
    try:
        from sparkrun.core.cluster_manager import ClusterManager

        cluster_mgr = ClusterManager(config.config_path.parent)
        if isinstance(cluster, str):
            cluster = cluster_mgr.get(cluster)
        if cluster is not None:
            return cluster.hosts, cluster
        if benchmark_id:
            from sparkrun.benchmarking.run_state import recorded_benchmark_hosts

            return recorded_benchmark_hosts(benchmark_id, str(config.cache_dir)), cluster
        if workload_id:
            from sparkrun.orchestration.job_metadata import load_job_metadata

            meta = load_job_metadata(workload_id, cache_dir=str(config.cache_dir))
            return (meta or {}).get("hosts"), cluster
        default_name = cluster_mgr.get_default()
        default_cluster = cluster_mgr.get(default_name) if default_name else None
        if default_cluster is not None:
            return default_cluster.hosts, default_cluster
        return config.default_hosts, cluster
    except Exception:  # Resolution errors belong to the API body, not lock inference.
        return None, cluster


def operation_boundary(*, mutation=False):
    """Bind operation state and destination locks around a public API call."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def invoke(*args, **kwargs):
            from sparkrun.core.config import SparkrunConfig
            from sparkrun.core.operation import operation_scope

            bound = signature.bind(*args, **kwargs)
            sctx = bound.arguments.get("sctx")
            try:
                config = sctx.config if sctx is not None else SparkrunConfig()
            except Exception:
                # Preserve the operation's typed initialization/recovery errors.
                return function(*bound.args, **bound.kwargs)
            with operation_scope(sctx):
                if not mutation or config is None:
                    return function(*bound.args, **bound.kwargs)
                options = bound.arguments.get("options")
                if getattr(options, "dry_run", False) or bound.arguments.get("dry_run", False):
                    return function(*bound.args, **bound.kwargs)
                hosts = getattr(options, "hosts", None) or bound.arguments.get("hosts")
                cluster = getattr(options, "cluster", None) or bound.arguments.get("cluster")
                plan = bound.arguments.get("plan")
                if plan is not None:
                    # candidate_hosts, not host_list: eviction of a superseded
                    # deployment sweeps every host placement could have chosen,
                    # not just the ones this launch landed on.
                    hosts = getattr(plan, "candidate_hosts", None) or getattr(plan, "host_list", hosts)
                    cluster = getattr(plan, "cluster", cluster)
                if not hosts:
                    hosts, cluster = infer_reservation_hosts(
                        config,
                        cluster=cluster,
                        workload_id=bound.arguments.get("workload_id") or bound.arguments.get("cluster_id"),
                        benchmark_id=bound.arguments.get("benchmark_id"),
                    )
                # Host conflicts are deliberately independent of SSH usernames:
                # different login users still mutate the same machine/devices.
                keys = reservation_keys(hosts, executor_config=getattr(cluster, "executor_config", None))
                with reserve_resources(config.config_path.parent, keys):
                    return function(*bound.args, **bound.kwargs)

        return invoke

    return decorate


def reservations(root):
    """Inspect claims, including dead owners whose remote effects are unknown."""
    import json
    from sparkrun.core.process_identity import alive

    path = Path(root) / "operations.sqlite3"
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=10) as db:
        rows = db.execute("SELECT owner,resource,process FROM reservations ORDER BY owner,resource").fetchall()
    return [
        {"owner": owner, "resource": resource, "process": json.loads(process), "alive": alive(json.loads(process))}
        for owner, resource, process in rows
    ]


def reconcile_reservation(root, owner, *, acknowledge_effects=False):
    """Release only a verified dead owner's claims after explicit reconciliation."""
    import json
    from sparkrun.core.process_identity import alive

    if not acknowledge_effects:
        raise ValueError("Inspect the remote workloads, then acknowledge that effects have been reconciled")
    path = Path(root) / "operations.sqlite3"
    if not path.exists():
        raise LookupError("No reservations for this controller")
    with sqlite3.connect(path, timeout=10) as db:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT process FROM reservations WHERE owner=?", (owner,)).fetchall()
        if not rows:
            raise LookupError("Reservation not found")
        if any(alive(json.loads(row[0])) for row in rows):
            raise ResourceBusy("The reservation owner is still alive")
        db.execute("DELETE FROM reservations WHERE owner=?", (owner,))
