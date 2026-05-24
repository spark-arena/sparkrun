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
from typing import Any, TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    from sparkrun.core.backend_select import BackendBundle
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.base import RuntimePlugin

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

    def to_dict(self) -> dict[str, Any]:
        """Convert the job status to a JSON-serializable dictionary."""
        from dataclasses import asdict

        result = asdict(self)
        if self.metadata:
            result["recipe"] = self.metadata.get("recipe")
        return result


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
    from sparkrun.orchestration.executor import resolve_executor

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

    # Determine candidate container names on the head host (preserved for
    # backward-compatible ``container_statuses`` shape).
    candidates: list[str] = []
    if is_solo:
        candidates.append("%s_solo" % cluster_id)
    else:
        # Native distributed: node_0; Ray: head
        candidates.append("%s_node_0" % cluster_id)
        candidates.append("%s_head" % cluster_id)

    # Source liveness from the executor's canonical introspection path
    # (``executor.query_status``) rather than per-container ``docker
    # inspect`` probes.  Use metadata-derived overrides so we query via
    # the same executor that launched the workload — mirrors what
    # ``api.stop`` / ``api.logs`` do.
    cli_overrides: dict | None = None
    if meta:
        meta_exec = meta.get("executor")
        meta_exec_cfg = meta.get("executor_config")
        cli_overrides = {}
        if meta_exec:
            cli_overrides["executor"] = meta_exec
        if isinstance(meta_exec_cfg, dict):
            cli_overrides.update(meta_exec_cfg)
        if not cli_overrides:
            cli_overrides = None

    executor = resolve_executor(
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
    )
    status_snapshot = executor.query_status(hosts, ssh_kwargs=ssh_kwargs)

    running = cluster_id in status_snapshot.running_cluster_ids()

    # Reconstruct the legacy ``container_statuses`` dict shape.  We can't
    # recover exact container *names* from a ``RunningWorkload`` (which
    # carries docker IDs, not names), so we mark every candidate name
    # uniformly based on whether the cluster has any workload on the
    # head host.
    head_occupancy = status_snapshot.for_host(head_host)
    cluster_on_head = False
    if head_occupancy is not None:
        cluster_on_head = any(w.cluster_id == cluster_id for w in head_occupancy.workloads)

    container_statuses: dict[str, bool] = {name: cluster_on_head for name in candidates}

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


def _resolve_override(key: str, overrides: dict | None, defaults: dict | None):
    """Resolve a value from overrides -> recipe defaults."""
    val = overrides.get(key) if overrides else None
    if val is None and defaults:
        val = defaults.get(key)
    return val


# TODO: this might need to be based on host+ranks? we ditched early host trimming --
#       what implications does that have for deterministic cluster IDs here
def generate_cluster_id(recipe: "Recipe", hosts: list[str], overrides: dict | None = None) -> str:
    """Deterministic cluster identifier from recipe, host set, and overrides.

    Hashes: runtime + model + sorted hosts + port + served_model_name +
    non-default parallelism (every dimension in
    :data:`sparkrun.core.parallelism.PARALLELISM_KEYS` — tp, pp, dp, ep, cp).
    Port, served_model_name, and parallelism are resolved from
    overrides -> recipe defaults so that two instances of the same model
    on different ports or parallelism configs get distinct IDs.
    """
    from sparkrun.core.parallelism import PARALLELISM_KEYS

    port = _resolve_override("port", overrides, recipe.defaults)
    served_name = _resolve_override("served_model_name", overrides, recipe.defaults)

    parts = [recipe.runtime, recipe.model] + sorted(hosts)
    if port is not None:
        parts.append("port=%s" % port)
    if served_name is not None:
        parts.append("name=%s" % served_name)

    # Include every non-default parallelism dimension in the hash so
    # configs that differ only in dp/ep/cp also get distinct cluster IDs.
    # Iterating PARALLELISM_KEYS keeps this in lockstep with
    # save_job_metadata (single source of truth for parallelism dims).
    for long_key, short_key in PARALLELISM_KEYS:
        val = _resolve_override(long_key, overrides, recipe.defaults)
        if val is not None and int(val) != 1:
            parts.append("%s=%s" % (short_key, int(val)))

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
    runtime: "RuntimePlugin | None" = None,
    backends: "dict[str, BackendBundle] | None" = None,
    *,
    sctx: "SparkrunContext | None" = None,
) -> None:
    """Persist job metadata so ``cluster status`` can display recipe info.

    Writes a small YAML file to ``{cache_dir}/jobs/{hash}.yaml`` where
    *hash* is the 12-char hex portion of *cluster_id*.

    Args:
        backends: Per-host backend bundles resolved by the launcher.
            Persisted as ``{host: {vendor, backend}}`` so ``stop``/``logs``
            can recover the collective backend without re-probing.
        sctx: Optional shared :class:`SparkrunContext`.  When provided
            (and *cache_dir* is unset) ``sctx.config.cache_dir`` is the
            cache root.
    """
    cache_dir = _resolve_cache_dir(cache_dir, sctx)

    digest = cluster_id.removeprefix("sparkrun_")
    jobs_dir = Path(cache_dir) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    from sparkrun.core.parallelism import PARALLELISM_KEYS

    # Stamp the metadata with the producing sparkrun version so future
    # readers can detect schema/cluster-id format drift and migrate or
    # warn appropriately.  See ``load_job_metadata`` for the read side.
    try:
        from sparkrun import __version__ as _sparkrun_version
    except Exception:
        _sparkrun_version = "unknown"

    meta: dict = {
        "sparkrun_version": _sparkrun_version,
        "cluster_id": cluster_id,
        "recipe": recipe.qualified_name,
        "model": recipe.model,
        "runtime": recipe.runtime,
        "hosts": hosts,
    }
    if recipe_ref:
        meta["recipe_ref"] = recipe_ref

    # Store all parallelism values (not just tensor_parallel)
    for long_key, _ in PARALLELISM_KEYS:
        val = _resolve_override(long_key, overrides, recipe.defaults)
        if val is not None:
            meta[long_key] = int(val)
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

    # Resolve upstream API key via the runtime plugin so proxy discovery
    # can authenticate to the inference endpoint.  Runtimes that don't
    # support api-keys return None from resolve_api_key().
    if runtime is not None:
        try:
            api_key = runtime.resolve_api_key(recipe, overrides)
        except Exception:
            logger.debug("resolve_api_key failed for %s", cluster_id, exc_info=True)
            api_key = None
        if api_key:
            meta["api_key"] = str(api_key)

    if ib_ip_map:
        meta["ib_ip_map"] = ib_ip_map
    if mgmt_ip_map:
        meta["mgmt_ip_map"] = mgmt_ip_map
    if runtime_info:
        meta["runtime_info"] = runtime_info
    if container_image:
        meta["effective_container_image"] = container_image

    # Persist per-host backend bundle so stop/logs can recover collective
    # backend selection without re-probing hardware.  Schema:
    #   backends: { host: { vendor, backend } }
    if backends:
        meta["backends"] = {
            host: {"vendor": bundle.accelerator_vendor, "backend": bundle.collective.name} for host, bundle in backends.items()
        }

    # Persist executor selection so stop/logs can reproduce the same
    # executor (Docker vs experimental local) without re-running the
    # launcher's resolution logic.
    executor_selector = getattr(recipe, "executor", "") or ""
    if executor_selector:
        meta["executor"] = executor_selector
    recipe_exec_cfg = getattr(recipe, "executor_config", None)
    if isinstance(recipe_exec_cfg, dict) and recipe_exec_cfg:
        meta["executor_config"] = dict(recipe_exec_cfg)

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


def remove_job_metadata(
    cluster_id: str,
    cache_dir: str | None = None,
    *,
    sctx: "SparkrunContext | None" = None,
) -> None:
    """Delete the cached job metadata file for a cluster_id.

    No-op if the file does not exist.  When *cache_dir* is unset, the
    cache root is resolved from ``sctx.config.cache_dir`` (when *sctx*
    is provided) and falls back to :data:`DEFAULT_CACHE_DIR`.
    """
    cache_dir = _resolve_cache_dir(cache_dir, sctx)
    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    meta_path.unlink(missing_ok=True)
    logger.debug("Removed job metadata %s", meta_path)


def load_job_metadata(
    cluster_id: str,
    cache_dir: str | None = None,
    *,
    sctx: "SparkrunContext | None" = None,
) -> dict | None:
    """Load job metadata for a cluster_id.  Returns ``None`` if not found.

    When *cache_dir* is unset, the cache root is resolved from
    ``sctx.config.cache_dir`` (when *sctx* is provided) and falls back
    to :data:`DEFAULT_CACHE_DIR`.

    Metadata schema may evolve across sparkrun versions; readers can
    inspect ``data["sparkrun_version"]`` to detect potential drift and
    handle migration.  Today this function returns the data verbatim;
    a version-mismatch policy can land here later.
    """
    cache_dir = _resolve_cache_dir(cache_dir, sctx)
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


def _resolve_cache_dir(cache_dir: str | None, sctx: "SparkrunContext | None") -> str:
    """Resolve the effective cache root for job-metadata I/O.

    Priority: explicit *cache_dir* > ``sctx.config.cache_dir`` > module
    default :data:`DEFAULT_CACHE_DIR`.  Used by every public function in
    this module so the resolution chain stays consistent.
    """
    if cache_dir is not None:
        return cache_dir
    if sctx is not None:
        try:
            return str(sctx.config.cache_dir)
        except Exception:
            logger.debug("sctx.config.cache_dir unavailable; using default", exc_info=True)
    from sparkrun.core.config import DEFAULT_CACHE_DIR

    return str(DEFAULT_CACHE_DIR)
