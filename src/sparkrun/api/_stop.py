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
) -> StopResult:
    """Stop a running sparkrun workload.

    Either ``cluster_id`` *or* (``recipe`` + a host source) is required.
    When both are provided, ``cluster_id`` wins.
    """
    from sparkrun.api._resolve import (
        resolve_cluster_def,
        resolve_hosts,
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
        cluster_def = resolve_cluster_def(cluster)
        resolved_recipe = resolve_recipe(recipe)
        resolved_hosts = resolve_hosts(hosts, cluster=cluster_def)
        cluster_id = generate_cluster_id(resolved_recipe, resolved_hosts, overrides=overrides)
    else:
        cluster_def = resolve_cluster_def(cluster)

    # Load metadata to recover the host list and executor selection.
    meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
    if meta is None and not hosts:
        raise JobNotFound("No job metadata found for cluster_id %r and no hosts provided" % cluster_id)

    if hosts:
        target_hosts = list(hosts)
    elif meta and meta.get("hosts"):
        target_hosts = list(meta["hosts"])
    else:
        target_hosts = []

    if not target_hosts:
        raise JobNotFound("No hosts known for cluster_id %r" % cluster_id)

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
    )

    container_names = executor.enumerate_containers(cluster_id, len(target_hosts))

    # Use cleanup_containers — the established teardown path used by the
    # CLI.  Tests mock ``sparkrun.orchestration.primitives.cleanup_containers``;
    # this keeps the api stop dispatch on the conventional path.
    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers

    config = _maybe_load_config()
    if config is not None and cluster_def is not None and cluster_def.user:
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
