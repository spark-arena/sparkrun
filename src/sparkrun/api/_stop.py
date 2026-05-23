"""``sparkrun.api.stop`` — stop a running sparkrun workload.

Two modes:

- **By cluster_id**: provide the literal ``cluster_id`` (as returned
  by :func:`sparkrun.api.run`); the API loads the job metadata, picks
  the executor that originally launched it, and runs ``stop_cmd``
  against each candidate container name on every host.
- **By recipe+hosts+overrides**: derive the same ``cluster_id`` the
  launcher would have produced and dispatch identically.  Useful for
  ``sparkrun stop <recipe>`` semantics.

Returns a :class:`StopResult` summarizing how many containers were
removed and any per-host errors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import JobNotFound, SparkrunError
from sparkrun.api._models import StopResult

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def stop(
    *,
    cluster_id: str | None = None,
    recipe: "str | Recipe | None" = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> StopResult:
    """Stop a running sparkrun workload.

    Either ``cluster_id`` *or* (``recipe`` + a host source) is required.
    When both are provided, ``cluster_id`` wins.

    Args:
        sctx: Optional shared :class:`SparkrunContext` for chained
            api calls (registry/cluster manager + config sharing).
    """
    from sparkrun.api._resolve import (
        resolve_cluster,
        resolve_recipe,
    )
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.job_metadata import (
        generate_cluster_id,
        load_job_metadata,
        remove_job_metadata,
    )

    # Derive cluster_id from recipe+hosts when not given explicitly.
    if not cluster_id:
        if recipe is None:
            raise SparkrunError("api.stop requires cluster_id or recipe+hosts")
        cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
        resolved_recipe = resolve_recipe(recipe, sctx=sctx)
        cluster_id = generate_cluster_id(resolved_recipe, list(cluster_def.hosts), overrides=overrides)
        # Default cache_dir from sctx.config when not explicitly passed.
        if cache_dir is None and sctx is not None:
            try:
                cache_dir = str(sctx.config.cache_dir)
            except Exception:
                cache_dir = None
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
        target_hosts = list(cluster_def.hosts)
    else:
        # cluster_id given — load metadata to recover hosts/executor.
        if cache_dir is None and sctx is not None:
            try:
                cache_dir = str(sctx.config.cache_dir)
            except Exception:
                cache_dir = None
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
        if meta is None and not hosts and cluster is None:
            raise JobNotFound("No job metadata found for cluster_id %r and no hosts provided" % cluster_id)

        # Determine target hosts: explicit > metadata > error.
        if hosts:
            target_hosts = list(hosts)
        elif meta and meta.get("hosts"):
            target_hosts = list(meta["hosts"])
        else:
            target_hosts = []

        if not target_hosts:
            raise JobNotFound("No hosts known for cluster_id %r" % cluster_id)

        # Now that we know the hosts, build a cluster definition for
        # downstream consumers.  `resolve_cluster` synthesizes an
        # anonymous one when no explicit cluster is given.
        cluster_def = resolve_cluster(cluster, target_hosts, sctx=sctx)

    # Resolve the executor — prefer recipe-encoded selection from metadata
    # so we use the same executor that launched the workload.
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
        cluster=cluster_def,
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
        v=sctx.variables if sctx is not None else None,
    )

    container_names = executor.enumerate_containers(cluster_id, len(target_hosts))

    # Use cleanup_containers — the established teardown path used by the
    # CLI.  Tests mock ``sparkrun.orchestration.primitives.cleanup_containers``;
    # this keeps the api stop dispatch on the conventional path.
    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers

    config = sctx.config if sctx is not None else _maybe_load_config()
    if config is not None and cluster_def.user:
        # Apply cluster SSH user so downstream ssh_kwargs picks it up.
        try:
            config.ssh_user = cluster_def.user
        except Exception:
            logger.debug("Failed to apply cluster SSH user", exc_info=True)
    ssh_kwargs = build_ssh_kwargs(config) if config else {}

    errors: list[str] = []
    try:
        cleanup_containers(target_hosts, container_names, ssh_kwargs=ssh_kwargs)
        removed_count = len(target_hosts)
    except Exception as e:
        errors.append(str(e))
        removed_count = 0

    # Cleanup persistent metadata after best-effort stop (matches CLI behaviour).
    try:
        remove_job_metadata(cluster_id, cache_dir=cache_dir)
    except Exception:
        logger.debug("Failed to remove job metadata for %s", cluster_id, exc_info=True)

    return StopResult(
        cluster_id=cluster_id,
        hosts_targeted=tuple(target_hosts),
        containers_removed=removed_count,
        errors=tuple(errors),
    )


def _maybe_load_config():
    """Load SparkrunConfig once for the SSH kwargs, returning ``None`` on failure."""
    try:
        from sparkrun.core.config import SparkrunConfig

        return SparkrunConfig()
    except Exception:  # pragma: no cover - defensive
        return None


__all__ = ["stop"]
