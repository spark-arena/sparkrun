"""Endpoint discovery via the ``sparkrun.api`` surface.

Single discovery pass:

1. ``api.list_jobs(sctx=sctx)`` enumerates persisted job metadata —
   every cluster_id, recipe, runtime, hosts, and the original metadata
   dict (port, served_model_name, api_key, ...).
2. ``api.status(host_list, sctx=sctx)`` produces the live
   :class:`ClusterStatus` snapshot.  Its
   :meth:`ClusterStatus.running_cluster_ids` is the authoritative
   liveness filter.

A persisted job surfaces as an :class:`DiscoveredEndpoint` iff its
``cluster_id`` is reported running by the snapshot.  When *host_list*
is omitted the liveness step is skipped (metadata-only mode) — useful
for callers that have no host context and want to enumerate everything
they've ever launched.

Health checks (TCP/HTTP probe of ``/v1/models``) run as a final pass,
identical to the previous implementation.
"""

from __future__ import annotations

from sparkrun.core.operation import submit as _operation_submit

import json
import logging
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sparkrun.api as api

if TYPE_CHECKING:
    from sparkrun.api import JobInfo
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT = 3  # seconds


@dataclass
class DiscoveredEndpoint:
    """A discovered inference endpoint from job metadata."""

    cluster_id: str
    model: str
    served_model_name: str | None
    runtime: str
    host: str
    port: int
    healthy: bool
    actual_models: list[str] = field(default_factory=list)
    recipe_name: str = ""
    tensor_parallel: int = 1
    owner: str = ""
    api_key: str | None = None
    #: Max context length (tokens) of the served model, captured from the
    #: backend's ``/v1/models`` ``max_model_len`` during the health probe.
    #: ``None`` when unknown. Surfaces the true model window to clients so
    #: the gateway does not advertise an arbitrary per-key cap.
    max_model_len: int | None = None
    #: Task vocabulary advertised by the backend's ``/v1/models`` entry — the
    #: single default ``task_type`` and the ``supported_task_types`` pool
    #: (a diffusion backend reports e.g. ``T2I`` / ``I2I`` / ``T2V``).  Both
    #: are empty when the backend reports none, which is every chat backend
    #: today.  Pure metadata, like ``max_model_len``: it decides only how the
    #: gateway *describes* the model, never whether it is discovered.
    task_type: str | None = None
    task_types: list[str] = field(default_factory=list)
    # Named placement and launch fingerprint are metadata, not the opaque job ID.
    cluster_name: str | None = None
    recipe_revision: str = ""
    native_protocols: list[str] = field(default_factory=lambda: ["openai"])
    capabilities: list[str] = field(default_factory=list)
    plugin_items: dict = field(default_factory=dict)


@dataclass
class DiscoverySnapshot:
    """A bounded observation, with coverage separate from endpoint health.

    Endpoints retain credentials for local application only. ``to_dict`` is the
    public projection; credentials and arbitrary recipe/plugin payloads are not
    part of the discovery wire contract.
    """

    endpoints: list[DiscoveredEndpoint] = field(default_factory=list)
    observed_at: float = field(default_factory=time.time)
    sweep_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    hosts: list[str] = field(default_factory=list)
    covered_hosts: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    mode: str = "metadata"

    @property
    def complete(self) -> bool:
        return self.mode == "live" and not self.errors and set(self.hosts) <= set(self.covered_hosts)

    def to_dict(self) -> dict:
        return {
            "sweep_id": self.sweep_id,
            "observed_at": self.observed_at,
            "hosts": self.hosts,
            "covered_hosts": self.covered_hosts,
            "complete": self.complete,
            "mode": self.mode,
            "errors": self.errors,
            "endpoints": [
                {
                    "workload_id": ep.cluster_id,
                    "cluster_name": ep.cluster_name,
                    "recipe_revision": ep.recipe_revision,
                    "owner": ep.owner,
                    "model": ep.model,
                    "served_model_name": ep.served_model_name,
                    "actual_models": ep.actual_models,
                    "runtime": ep.runtime,
                    "host": ep.host,
                    "port": ep.port,
                    "healthy": ep.healthy,
                    "native_protocols": ep.native_protocols,
                    "capabilities": ep.capabilities,
                    "max_model_len": ep.max_model_len,
                }
                for ep in self.endpoints
            ],
        }


def discover_snapshot(**kwargs) -> DiscoverySnapshot:
    """Observe endpoints without confusing failed coverage with absence."""
    snapshot = DiscoverySnapshot(hosts=list(kwargs.get("host_list") or ()))
    snapshot.endpoints = discover_endpoints(**kwargs, _observation=snapshot)
    return snapshot


def discover_endpoints(
    host_filter: list[str] | None = None,
    cache_dir: str | None = None,
    check_health: bool = True,
    host_list: list[str] | None = None,
    ssh_kwargs: dict | None = None,
    *,
    cluster_ids: "set[str] | frozenset[str] | None" = None,
    cluster_def: "ClusterDefinition | None" = None,
    sctx: "SparkrunContext | None" = None,
    _observation: DiscoverySnapshot | None = None,
) -> list[DiscoveredEndpoint]:
    """Discover running inference endpoints via :mod:`sparkrun.api`.

    The metadata source is :func:`sparkrun.api.list_jobs`; the liveness
    filter is :func:`sparkrun.api.status` when *host_list* is provided.

    Args:
        host_filter: Optional management-IP allowlist applied after
            normalization.  Only endpoints whose resolved host matches
            are emitted.
        cache_dir: Override the sparkrun cache root (forwarded to
            :func:`sparkrun.api.list_jobs`).
        check_health: When ``True``, TCP/HTTP-probe each endpoint and
            drop the unreachable ones.
        host_list: Hosts to inspect via :func:`sparkrun.api.status`.
            When ``None``, liveness is skipped (metadata-only mode).
        ssh_kwargs: Optional SSH kwargs forwarded to ``api.status``.
        cluster_ids: Optional cluster-id allowlist applied *before* the health
            probe.  A caller watching one known job would otherwise pay an HTTP
            probe against every endpoint on the machine on every poll.
        cluster_def: Optional pre-resolved cluster definition forwarded
            to ``api.status``.
        sctx: Optional shared :class:`SparkrunContext`.

    Returns:
        List of discovered endpoints (healthy only when *check_health*
        is ``True``).
    """
    jobs = api.list_jobs(cache_dir=cache_dir, sctx=sctx)

    running_ids: set[str] | None = None
    if host_list:
        try:
            snapshot = api.status(
                list(host_list),
                cluster=cluster_def,
                ssh_kwargs=ssh_kwargs,
                sctx=sctx,
            )
            running_ids = set(snapshot.running_cluster_ids())
            if _observation is not None:
                _observation.mode = "live"
                # Never expose executor errors verbatim: SSH diagnostics can
                # include connection arguments and credentials.
                failed = snapshot.observation_errors
                _observation.covered_hosts = [entry.host for entry in snapshot.hosts if entry.host not in failed]
                _observation.errors = {
                    host: "Host status was not completely observed" for host in host_list if host not in _observation.covered_hosts
                }
        except Exception:
            logger.warning(
                "api.status query failed, falling back to metadata-only discovery",
                exc_info=True,
            )
            running_ids = None
            if _observation is not None:
                _observation.errors = {host: "Host status query failed" for host in host_list}

    # Build IB→mgmt reverse map from every metadata entry, used to
    # normalize IB IPs to management IPs for consistent display/dedup.
    ib_to_mgmt: dict[str, str] = {}
    for job in jobs:
        meta = job.metadata
        ib_map = meta.get("ib_ip_map", {}) or {}
        mgmt_map = meta.get("mgmt_ip_map", {}) or {}
        for raw_host, ib_ip in ib_map.items():
            mgmt_ip = mgmt_map.get(raw_host, raw_host)
            ib_to_mgmt[ib_ip] = mgmt_ip

    # Deduplicate: one entry per host:port — last write wins.
    # ``api.list_jobs`` returns most-recent-first; iterate in reverse so
    # the most recent job overwrites older entries on the same key.
    candidates: dict[str, DiscoveredEndpoint] = {}
    for job in reversed(jobs):
        if running_ids is not None and job.workload_id not in running_ids:
            continue
        if cluster_ids is not None and job.workload_id not in cluster_ids:
            continue

        ep = _endpoint_from_job(job, ib_to_mgmt=ib_to_mgmt)
        if ep is None:
            continue

        if host_filter and ep.host not in host_filter:
            continue

        key = "%s:%d" % (ep.host, ep.port)
        candidates[key] = ep

    endpoints = list(candidates.values())

    if check_health and endpoints:
        _check_health_parallel(endpoints)
        if _observation is not None:
            for ep in endpoints:
                if not ep.healthy:
                    _observation.errors[ep.host] = "Endpoint health was not verified"
                    if ep.host in _observation.covered_hosts:
                        _observation.covered_hosts.remove(ep.host)
        endpoints = [ep for ep in endpoints if ep.healthy]
        # Collapse entries that turn out to serve the same models on the
        # same port via different network interfaces.
        endpoints = _deduplicate_by_identity(endpoints)

    return endpoints


def _endpoint_from_job(
    job: "JobInfo",
    *,
    ib_to_mgmt: dict[str, str],
) -> DiscoveredEndpoint | None:
    """Build a :class:`DiscoveredEndpoint` from a :class:`JobInfo`.

    Returns ``None`` when essential metadata (hosts, port) is missing.
    """
    meta = job.metadata or {}
    hosts = job.hosts or tuple(meta.get("hosts") or ())
    if not hosts:
        return None

    head_host = hosts[0]

    # Prefer management IP for user-facing display.
    mgmt_map = meta.get("mgmt_ip_map") or {}
    if head_host in mgmt_map:
        head_host = mgmt_map[head_host]
    elif head_host in ib_to_mgmt:
        head_host = ib_to_mgmt[head_host]

    port_raw = meta.get("port", 8000)
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        return None

    served_name = meta.get("served_model_name")

    protocols = ["openai"]
    capabilities = []
    plugin_items = {}
    if meta.get("recipe_state"):
        try:
            from sparkrun.core.recipe import Recipe

            recipe = Recipe.from_state(meta["recipe_state"])
            plugin_items = recipe.export_plugin_items()
            serving = api.recipe_protocols(recipe)
            protocols = list(serving.protocols)
            capabilities = sorted(set(recipe.capabilities or []) | set(serving.capabilities))
        except Exception:
            logger.debug("Cannot resolve serving APIs from saved recipe for %s", job.workload_id, exc_info=True)

    return DiscoveredEndpoint(
        cluster_id=job.workload_id,
        cluster_name=meta.get("cluster") or None,
        recipe_revision=str(meta.get("recipe_fingerprint") or ""),
        native_protocols=protocols,
        capabilities=capabilities,
        plugin_items=plugin_items,
        model=meta.get("model", "") or "",
        served_model_name=served_name,
        runtime=(job.runtime or meta.get("runtime") or "") or "",
        host=head_host,
        port=port,
        healthy=False,
        recipe_name=(job.recipe or meta.get("recipe") or meta.get("recipe_ref") or "") or "",
        tensor_parallel=int(meta.get("tensor_parallel", 1) or 1),
        owner=str(meta.get("owner") or ""),
        api_key=meta.get("api_key") or None,
    )


def _check_health_parallel(endpoints: list[DiscoveredEndpoint]) -> None:
    """Run health checks in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=min(len(endpoints), 8)) as pool:
        futures = {_operation_submit(pool, _check_single_health, ep): ep for ep in endpoints}
        for future in as_completed(futures):
            ep = futures[future]
            try:
                healthy, models, max_model_len, task_fields = future.result()
                ep.healthy = healthy
                ep.actual_models = models
                ep.max_model_len = max_model_len
                ep.task_type, ep.task_types = task_fields
            except Exception:
                logger.debug(
                    "Health check failed for %s:%d",
                    ep.host,
                    ep.port,
                    exc_info=True,
                )


def _deduplicate_by_identity(endpoints: list[DiscoveredEndpoint]) -> list[DiscoveredEndpoint]:
    """Collapse endpoints that serve the same models on the same port.

    When a DGX Spark is reachable via both management and ConnectX-7
    IPs, two metadata entries may point to the same running server.
    After health checks confirm they serve the same model set, keep
    the last occurrence (most recent metadata by mtime).
    """
    seen: dict[tuple, DiscoveredEndpoint] = {}
    for ep in endpoints:
        identity = (frozenset(ep.actual_models), ep.port) if ep.actual_models else (ep.model, ep.port)
        if identity in seen:
            logger.debug(
                "Dedup: %s:%d is same server as %s:%d (same models on port %d), keeping newer",
                seen[identity].host,
                seen[identity].port,
                ep.host,
                ep.port,
                ep.port,
            )
        seen[identity] = ep
    return list(seen.values())


def _check_single_health(ep: DiscoveredEndpoint) -> tuple[bool, list[str], int | None, tuple[str | None, list[str]]]:
    """Check a single endpoint's health via GET /v1/models.

    Sends an ``Authorization: Bearer`` header when the endpoint has an
    ``api_key`` set, so backends that require auth still report healthy.

    Returns:
        Tuple of (healthy, list_of_model_ids, max_model_len, task_fields).
        ``max_model_len`` is the model's true context window (``max_model_len``
        from the backend's ``/v1/models``), or ``None`` when the backend does
        not report it.  ``task_fields`` is ``(task_type, supported_task_types)``
        exactly as the backend advertises them — ``None``/``[]`` for a chat
        backend that reports no task vocabulary, the first entry that carries
        any for a diffusion pipeline (one pipeline per endpoint, so the entry
        that advertises is the one this endpoint serves).
    """
    url = "http://%s:%d/v1/models" % (ep.host, ep.port)
    headers: dict[str, str] = {}
    if ep.api_key:
        headers["Authorization"] = "Bearer %s" % ep.api_key
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=_HEALTH_TIMEOUT) as resp:
            if resp.status == 200:
                body = json.loads(resp.read())
                data = body.get("data", [])
                models = [m.get("id", "") for m in data]
                max_model_len = None
                for m in data:
                    mml = m.get("max_model_len")
                    if isinstance(mml, int) and mml > 0:
                        max_model_len = mml
                        break
                task_type, task_types = _task_fields(data)
                return True, models, max_model_len, (task_type, task_types)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        pass
    return False, [], None, (None, [])


def _task_fields(data: object) -> tuple[str | None, list[str]]:
    """The ``(task_type, supported_task_types)`` a backend's /v1/models body
    advertises, taken from the first entry that carries either.

    Deliberately the *raw* strings: this probe records what the backend said,
    and only the gateway's ``media_mode`` interprets it.  Anything
    non-string — null task_type, a list of numbers — is dropped here so a
    malformed body cannot propagate a non-string into the config builder.
    """
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict):
            continue
        task_types = [t for t in (entry.get("supported_task_types") or []) if isinstance(t, str) and t.strip()]
        raw_type = entry.get("task_type")
        primary = raw_type.strip() if isinstance(raw_type, str) and raw_type.strip() else None
        if primary or task_types:
            return primary, task_types
    return None, []
