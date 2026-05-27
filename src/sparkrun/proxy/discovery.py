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

import json
import logging
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
    api_key: str | None = None


def discover_endpoints(
    host_filter: list[str] | None = None,
    cache_dir: str | None = None,
    check_health: bool = True,
    host_list: list[str] | None = None,
    ssh_kwargs: dict | None = None,
    *,
    cluster_def: "ClusterDefinition | None" = None,
    sctx: "SparkrunContext | None" = None,
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
        except Exception:
            logger.warning(
                "api.status query failed, falling back to metadata-only discovery",
                exc_info=True,
            )
            running_ids = None

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
        if running_ids is not None and job.cluster_id not in running_ids:
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

    return DiscoveredEndpoint(
        cluster_id=job.cluster_id,
        model=meta.get("model", "") or "",
        served_model_name=served_name,
        runtime=(job.runtime or meta.get("runtime") or "") or "",
        host=head_host,
        port=port,
        healthy=False,
        recipe_name=(job.recipe or meta.get("recipe") or meta.get("recipe_ref") or "") or "",
        tensor_parallel=int(meta.get("tensor_parallel", 1) or 1),
        api_key=meta.get("api_key") or None,
    )


def _check_health_parallel(endpoints: list[DiscoveredEndpoint]) -> None:
    """Run health checks in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=min(len(endpoints), 8)) as pool:
        futures = {pool.submit(_check_single_health, ep): ep for ep in endpoints}
        for future in as_completed(futures):
            ep = futures[future]
            try:
                healthy, models = future.result()
                ep.healthy = healthy
                ep.actual_models = models
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


def _check_single_health(ep: DiscoveredEndpoint) -> tuple[bool, list[str]]:
    """Check a single endpoint's health via GET /v1/models.

    Sends an ``Authorization: Bearer`` header when the endpoint has an
    ``api_key`` set, so backends that require auth still report healthy.

    Returns:
        Tuple of (healthy, list_of_model_ids).
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
                models = [m.get("id", "") for m in body.get("data", [])]
                return True, models
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        pass
    return False, []
