"""Job and cluster metadata storage.

Persists cluster_id → recipe mapping in ``~/.cache/sparkrun/jobs/`` so
``cluster status`` and other commands can display recipe info for
running clusters.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    """Result of checking whether a sparkrun job is running."""

    running: bool
    cluster_id: str
    healthy: bool | None = None  # None = not checked
    metadata: dict | None = None
    container_statuses: dict[str, bool] = field(default_factory=dict)
    hosts: list[str] = field(default_factory=list)


def check_job_running(
        *,
        cluster_id: str | None = None,
        recipe: "Recipe | None" = None,
        hosts: list[str] | None = None,
        overrides: dict | None = None,
        ssh_kwargs: dict | None = None,
        cache_dir: str | None = None,
        check_http_models: bool = False,
        port: int | None = None,
) -> JobStatus:
    """Check whether a sparkrun job is currently running.

    Resolves the cluster_id (from params or recipe+hosts), checks head-node
    containers for liveness, and optionally performs an HTTP health check.

    Args:
        cluster_id: Explicit cluster ID.  If not given, generated from
            *recipe*, *hosts*, and *overrides*.
        recipe: Recipe object (used to generate cluster_id if not given).
        hosts: Host list.  Falls back to job metadata if not provided.
        overrides: Recipe overrides (port, served_model_name, etc.).
        ssh_kwargs: SSH connection parameters.
        cache_dir: Cache directory for job metadata lookup.
        check_http_models: When True and container is running, probe the
            ``/v1/models`` endpoint.
        port: Explicit port for health checks.  Falls back to metadata
            then default 8000.

    Returns:
        :class:`JobStatus` with liveness and optional health info.
    """
    from sparkrun.orchestration.primitives import is_container_running

    # Resolve cluster_id
    if cluster_id is None:
        if recipe is None or hosts is None:
            raise ValueError("Either cluster_id or both recipe and hosts must be provided")
        cluster_id = generate_cluster_id(recipe, hosts, overrides=overrides)

    # Load metadata
    meta = load_job_metadata(cluster_id, cache_dir=cache_dir)

    # Resolve hosts
    if hosts is None:
        if meta and meta.get("hosts"):
            hosts = meta["hosts"]
        else:
            return JobStatus(running=False, cluster_id=cluster_id, metadata=meta, hosts=[])

    head_host = hosts[0]
    is_solo = len(hosts) == 1

    # Determine candidate container names on the head host
    candidates: list[str] = []
    if is_solo:
        candidates.append("%s_solo" % cluster_id)
    else:
        # Native distributed: node_0; Ray: head
        candidates.append("%s_node_0" % cluster_id)
        candidates.append("%s_head" % cluster_id)

    # Check each candidate
    container_statuses: dict[str, bool] = {}
    for name in candidates:
        container_statuses[name] = is_container_running(head_host, name, ssh_kwargs=ssh_kwargs)

    running = any(container_statuses.values())

    # Optional health check
    healthy: bool | None = None
    if check_http_models and running:
        from sparkrun.orchestration.primitives import wait_for_healthy
        effective_port = port or (meta.get("port") if meta else None) or 8000
        url = "http://%s:%d/v1/models" % (head_host, effective_port)
        healthy = wait_for_healthy(url, max_retries=1, retry_interval=0, max_consecutive_refused=2)

    return JobStatus(
        running=running,
        cluster_id=cluster_id,
        healthy=healthy,
        metadata=meta,
        container_statuses=container_statuses,
        hosts=hosts,
    )


def generate_cluster_id(recipe: "Recipe", hosts: list[str], overrides: dict | None = None) -> str:
    """Deterministic cluster identifier from recipe, host set, and overrides.

    Hashes: runtime + model + sorted hosts + port + served_model_name.
    Port and served_model_name are resolved from overrides -> recipe defaults
    so that two instances of the same model on different ports get distinct IDs.
    """
    # Resolve effective port
    port = None
    if overrides:
        port = overrides.get("port")
    if port is None and recipe.defaults:
        port = recipe.defaults.get("port")

    # Resolve effective served_model_name
    served_name = None
    if overrides:
        served_name = overrides.get("served_model_name")
    if served_name is None and recipe.defaults:
        served_name = recipe.defaults.get("served_model_name")

    parts = [recipe.runtime, recipe.model] + sorted(hosts)
    if port is not None:
        parts.append("port=%s" % port)
    if served_name is not None:
        parts.append("name=%s" % served_name)
    key = "\0".join(parts)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return "sparkrun_%s" % digest


def save_job_metadata(
        cluster_id: str,
        recipe: "Recipe",
        hosts: list[str],
        overrides: dict | None = None,
        cache_dir: str | None = None,
        ib_ip_map: dict[str, str] | None = None,
        mgmt_ip_map: dict[str, str] | None = None,
        recipe_ref: str | None = None,
        runtime_info: dict[str, str] | None = None,
        container_image: Optional[str] = None,
) -> None:
    """Persist job metadata so ``cluster status`` can display recipe info.

    Writes a small YAML file to ``{cache_dir}/jobs/{hash}.yaml`` where
    *hash* is the 12-char hex portion of *cluster_id*.
    """
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    jobs_dir = Path(cache_dir) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    tp = None
    if overrides:
        tp = overrides.get("tensor_parallel")
    if tp is None and recipe.defaults:
        tp = recipe.defaults.get("tensor_parallel")

    meta: dict = {
        "cluster_id": cluster_id,
        "recipe": recipe.qualified_name,
        "model": recipe.model,
        "runtime": recipe.runtime,
        "hosts": hosts,
    }
    if recipe_ref:
        meta["recipe_ref"] = recipe_ref
    if tp is not None:
        meta["tensor_parallel"] = int(tp)
    # Persist port for proxy discovery
    port = None
    if overrides:
        port = overrides.get("port")
    if port is None and recipe.defaults:
        port = recipe.defaults.get("port")
    if port is not None:
        meta["port"] = int(port)

    # Persist served_model_name for proxy discovery
    served_name = None
    if overrides:
        served_name = overrides.get("served_model_name")
    if served_name is None and recipe.defaults:
        served_name = recipe.defaults.get("served_model_name")
    if served_name is not None:
        meta["served_model_name"] = str(served_name)

    if ib_ip_map:
        meta["ib_ip_map"] = ib_ip_map
    if mgmt_ip_map:
        meta["mgmt_ip_map"] = mgmt_ip_map
    if runtime_info:
        meta["runtime_info"] = runtime_info
    if container_image:
        meta["effective_container_image"] = container_image

    # Full overrides dict for export reconstruction
    if overrides:
        meta["overrides"] = dict(overrides)

    # Serialize full recipe state for faithful export reconstruction.
    try:
        meta["recipe_state"] = recipe.__getstate__()
    except Exception:
        logger.debug("Failed to serialize recipe state for %s", cluster_id, exc_info=True)

    meta_path = jobs_dir / f"{digest}.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
    logger.debug("Saved job metadata to %s", meta_path)


def remove_job_metadata(cluster_id: str, cache_dir: str | None = None) -> None:
    """Delete the cached job metadata file for a cluster_id.

    No-op if the file does not exist.
    """
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    meta_path.unlink(missing_ok=True)
    logger.debug("Removed job metadata %s", meta_path)


def load_job_metadata(cluster_id: str, cache_dir: str | None = None) -> dict | None:
    """Load job metadata for a cluster_id.  Returns ``None`` if not found."""
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    if not meta_path.exists():
        return None
    try:
        from sparkrun.utils import load_yaml
        data = load_yaml(meta_path)
        return data or None
    except Exception:
        logger.debug("Failed to load job metadata for %s", cluster_id, exc_info=True)
        return None
