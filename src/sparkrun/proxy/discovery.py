"""Endpoint discovery via live container queries and job metadata.

Primary path (live): SSHes into hosts, runs ``docker ps`` to find
actually-running sparkrun containers, then enriches with cached job
metadata.  This is authoritative — no stale metadata can appear.

Fallback path (metadata-only): Scans ``~/.cache/sparkrun/jobs/*.yaml``
metadata files, deduplicates, and health-checks.  Used when SSH info
is not available (e.g. internal proxy callers with no host context).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

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


def discover_endpoints(
    host_filter: list[str] | None = None,
    cache_dir: str | None = None,
    check_health: bool = True,
    host_list: list[str] | None = None,
    ssh_kwargs: dict | None = None,
) -> list[DiscoveredEndpoint]:
    """Discover running inference endpoints.

    When *host_list* and *ssh_kwargs* are provided, uses the **live**
    path: queries actual running containers via ``docker ps`` over SSH,
    then enriches with cached job metadata.  This is the most reliable
    method and should be preferred by CLI commands.

    When SSH info is not available, falls back to scanning job metadata
    files with health checks to filter stale entries.

    Args:
        host_filter: Only include endpoints on these hosts (metadata path).
        cache_dir: Override cache directory (default: ~/.cache/sparkrun).
        check_health: Whether to perform HTTP health checks.
        host_list: Hosts to query via SSH (enables live path).
        ssh_kwargs: SSH connection kwargs (enables live path).

    Returns:
        List of discovered endpoints (healthy only when check_health=True).
    """
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR

        cache_dir = str(DEFAULT_CACHE_DIR)

    # Live path: query actual running containers
    if host_list and ssh_kwargs is not None:
        try:
            return _discover_live(host_list, ssh_kwargs, cache_dir, check_health)
        except Exception:
            logger.warning(
                "Live discovery failed, falling back to metadata scan",
                exc_info=True,
            )

    return _discover_from_metadata(host_filter, cache_dir, check_health)


# ---------------------------------------------------------------------------
# Live discovery (primary)
# ---------------------------------------------------------------------------


def _discover_live(
    host_list: list[str],
    ssh_kwargs: dict,
    cache_dir: str,
    check_health: bool,
) -> list[DiscoveredEndpoint]:
    """Discover endpoints by querying running containers on hosts.

    Uses :func:`query_cluster_status` to get authoritative container
    state, then enriches each running cluster with cached job metadata.
    """
    from sparkrun.core.cluster_manager import query_cluster_status
    from sparkrun.orchestration.job_metadata import load_job_metadata

    result = query_cluster_status(host_list, ssh_kwargs=ssh_kwargs, cache_dir=cache_dir)

    endpoints: list[DiscoveredEndpoint] = []

    for cid, group in result.groups.items():
        ep = _endpoint_from_meta(cid, group.meta, fallback_host=group.members[0][0])
        if ep:
            endpoints.append(ep)

    for entry in result.solo_entries:
        cid = entry.name.removesuffix("_solo")
        meta = load_job_metadata(cid, cache_dir=cache_dir) or {}
        ep = _endpoint_from_meta(cid, meta, fallback_host=entry.host)
        if ep:
            endpoints.append(ep)

    if check_health and endpoints:
        _check_health_parallel(endpoints)
        endpoints = [ep for ep in endpoints if ep.healthy]

    return endpoints


def _endpoint_from_meta(
    cluster_id: str,
    meta: dict,
    fallback_host: str,
) -> DiscoveredEndpoint | None:
    """Build a DiscoveredEndpoint from job metadata.

    Uses mgmt_ip_map to normalise the head host to a management IP.
    Falls back to *fallback_host* when metadata has no host info.
    """
    hosts = meta.get("hosts", [fallback_host])
    if not hosts:
        return None

    head_host = hosts[0]

    # Prefer management IP for user-facing display
    mgmt_map = meta.get("mgmt_ip_map", {})
    if head_host in mgmt_map:
        head_host = mgmt_map[head_host]

    return DiscoveredEndpoint(
        cluster_id=cluster_id,
        model=meta.get("model", ""),
        served_model_name=meta.get("served_model_name"),
        runtime=meta.get("runtime", ""),
        host=head_host,
        port=int(meta.get("port", 8000)),
        healthy=False,
        recipe_name=meta.get("recipe", meta.get("recipe_ref", "")),
        tensor_parallel=int(meta.get("tensor_parallel", 1)),
    )


# ---------------------------------------------------------------------------
# Metadata-only discovery (fallback)
# ---------------------------------------------------------------------------


def _discover_from_metadata(
    host_filter: list[str] | None,
    cache_dir: str,
    check_health: bool,
) -> list[DiscoveredEndpoint]:
    """Discover endpoints by scanning job metadata files.

    Fallback path when SSH is not available.  Deduplicates by host:port
    (most recent file wins) and optionally health-checks each endpoint.
    """
    jobs_dir = Path(cache_dir) / "jobs"
    if not jobs_dir.is_dir():
        return []

    from sparkrun.utils import load_yaml

    # Two-pass approach:
    # 1. Load all metadata and build an IB→mgmt IP reverse map.
    # 2. Build candidates keyed by normalised host:port (mgmt IP).
    all_meta: list[dict] = []
    ib_to_mgmt: dict[str, str] = {}

    for meta_path in sorted(jobs_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime):
        try:
            meta = load_yaml(meta_path)
        except Exception:
            logger.debug("Failed to load job metadata: %s", meta_path, exc_info=True)
            continue
        if not meta or not meta.get("hosts"):
            continue
        all_meta.append(meta)
        # Collect IB→mgmt mappings from every metadata file
        ib_map = meta.get("ib_ip_map", {})
        mgmt_map = meta.get("mgmt_ip_map", {})
        for raw_host, ib_ip in ib_map.items():
            mgmt_ip = mgmt_map.get(raw_host, raw_host)
            ib_to_mgmt[ib_ip] = mgmt_ip

    # Later entries (sorted by mtime) overwrite earlier ones,
    # so the most recent metadata wins for each host:port.
    candidates: dict[str, DiscoveredEndpoint] = {}

    for meta in all_meta:
        hosts = meta["hosts"]
        head_host = hosts[0]

        # Normalise IB IPs to management IPs for consistent
        # dedup and user-facing display.
        mgmt_map = meta.get("mgmt_ip_map", {})
        if head_host in mgmt_map:
            head_host = mgmt_map[head_host]
        elif head_host in ib_to_mgmt:
            head_host = ib_to_mgmt[head_host]

        # Apply host filter
        if host_filter and head_host not in host_filter:
            continue

        port = int(meta.get("port", 8000))
        model = meta.get("model", "")
        runtime = meta.get("runtime", "")
        cluster_id = meta.get("cluster_id", "")
        recipe_name = meta.get("recipe", meta.get("recipe_ref", ""))
        served_name = meta.get("served_model_name")
        tp = int(meta.get("tensor_parallel", 1))

        # Deduplicate: one entry per host:port (last write wins)
        key = "%s:%d" % (head_host, port)
        candidates[key] = DiscoveredEndpoint(
            cluster_id=cluster_id,
            model=model,
            served_model_name=served_name,
            runtime=runtime,
            host=head_host,
            port=port,
            healthy=False,
            recipe_name=recipe_name,
            tensor_parallel=tp,
        )

    endpoints = list(candidates.values())

    if check_health and endpoints:
        _check_health_parallel(endpoints)
        # Only return healthy endpoints — stale metadata for dead
        # servers is not useful to callers.
        endpoints = [ep for ep in endpoints if ep.healthy]
        # Deduplicate endpoints serving identical models on the same
        # port but reachable via different network interfaces (e.g.
        # management IP vs ConnectX-7 IP).  Keep the newest entry.
        endpoints = _deduplicate_by_identity(endpoints)

    return endpoints


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

    Returns:
        Tuple of (healthy, list_of_model_ids).
    """
    url = "http://%s:%d/v1/models" % (ep.host, ep.port)
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=_HEALTH_TIMEOUT) as resp:
            if resp.status == 200:
                body = json.loads(resp.read())
                models = [m.get("id", "") for m in body.get("data", [])]
                return True, models
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        pass
    return False, []
