"""``sparkrun.api.run`` — launch an inference workload from the library API.

Orchestrates the full launch path:

1. Resolve recipe / cluster / hosts / runtime.
2. Apply overrides; resolve recipe (runtime selection finalized).
3. Run the scheduler via :func:`sparkrun.api.schedule` to compute placement.
4. Apply orthogonal constraints (solo, ``max_nodes``).
5. Delegate to :func:`sparkrun.core.launcher.launch_inference`.
6. Translate the launcher's :class:`LaunchResult` into :class:`RunResult`.

The function raises :class:`~sparkrun.api.SparkrunError` (or a
subclass) for any failure; on success it returns a populated
:class:`RunResult`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import (
    InsufficientCapacity,
    LayoutRequired,
    SparkrunError,
)
from sparkrun.api._models import RunOptions, RunResult

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.scheduler import RankAssignment

logger = logging.getLogger(__name__)


def run(options: RunOptions, *, sctx: "SparkrunContext | None" = None) -> RunResult:
    """Launch the workload described by *options* and return a :class:`RunResult`.

    Args:
        options: Inputs for the launch.
        sctx: Optional shared :class:`SparkrunContext`.  When omitted a
            fresh session is built; callers chaining multiple ``api.*``
            calls can construct one ``sctx`` and pass it to share
            config / registry-manager / cluster-manager state.

    Raises:
        :class:`InsufficientCapacity`: Scheduler can't fit the workload.
        :class:`LayoutRequired`: Cluster needs an explicit ``recipe.layout``.
        :class:`~sparkrun.api.RecipeNotFound`: Recipe lookup failed.
        :class:`~sparkrun.api.HostsUnreachable`: No usable host source.
        :class:`~sparkrun.api.TrustRejected`: Recipe hooks rejected.
        :class:`SparkrunError`: For other launch failures.
    """
    from sparkrun.api._resolve import (
        resolve_cluster,
        resolve_recipe,
        resolve_runtime,
    )
    from sparkrun.api._schedule import schedule
    from sparkrun.core.launcher import launch_inference
    from sparkrun.core.parallelism import extract_parallelism
    from sparkrun.core.scheduler import SchedulingRequest
    from sparkrun.orchestration.job_metadata import (
        generate_cluster_id,
        generate_intent_id,
        generate_placement_token,
        parse_cluster_id,
    )

    sctx = resolve_sctx(sctx)
    started_at = time.time()
    config = sctx.config

    # 1. Resolve inputs.  `resolve_cluster` always returns a populated
    # ClusterDefinition (anonymous when only --hosts was given) so
    # downstream code never has to branch on ``cluster is None``.
    cluster_def = resolve_cluster(options.cluster, options.hosts, sctx=sctx, config=config)
    recipe = resolve_recipe(options.recipe, sctx=sctx, overrides=options.overrides)
    hosts = list(cluster_def.hosts)
    runtime = resolve_runtime(recipe, sctx=sctx)

    # Scheduler selection chain: caller > recipe > cluster > greedy default.
    effective_scheduler = (
        options.scheduler or (getattr(recipe, "scheduler", "") or None) or (getattr(cluster_def, "scheduler", None) or None)
    )

    # Apply the cluster's SSH user (if any) to the config so downstream
    # SSH operations (executor.run / distribution / build_ssh_kwargs)
    # log in as the right user.  Matches the CLI's resolution chain
    # where ``_resolve_hosts_or_exit`` applies ``cluster.user`` to
    # ``config.ssh_user`` before launch.
    if getattr(cluster_def, "user", None):
        try:
            config.ssh_user = cluster_def.user
        except Exception:
            logger.debug("Failed to apply cluster SSH user to config", exc_info=True)

    # 2. Compute placement (single source of truth for the effective host list).
    placement: "RankAssignment | None" = None
    host_list = list(hosts)
    is_solo_request = bool(options.solo) or recipe.mode == "solo"

    if not is_solo_request and len(host_list) > 1:
        import dataclasses

        parallelism = extract_parallelism(recipe.build_config_chain(options.overrides))
        if any(getattr(parallelism, k) > 1 for k in ("tensor_parallel", "pipeline_parallel", "data_parallel")):
            # Bake the runtime-derived rank count into the parallelism config
            # so the scheduler asks ``parallelism.world_size()`` and gets the
            # right answer regardless of which runtime owns the workload.
            total_ranks = runtime.world_size(parallelism, recipe=recipe, cluster=cluster_def)
            parallelism = dataclasses.replace(parallelism, total_ranks=total_ranks)
            # Acquire a ClusterStatus snapshot so occupancy-sparse / occupancy-dense schedulers can
            # subtract already-committed workloads from each host's capacity.
            # Best-effort — a partially-unreachable cluster shouldn't crash the
            # launch path; the scheduler falls through to its no-status
            # behaviour (greedy-equivalent for OccupancyAware) when this is
            # None.
            cluster_status = _safe_acquire_status(host_list, cluster_def, sctx)
            sched_request = SchedulingRequest(
                parallelism=parallelism,
                hosts=tuple(host_list),
                host_hardware=cluster_def.hosts_hardware or None,
                layout=getattr(recipe, "layout", None),
                status=cluster_status,
                resources=None,
            )
            sched_result = schedule(sched_request, scheduler=effective_scheduler, sctx=sctx)
            placement = sched_result.assignment
            host_list = list(placement.hosts_used)
            logger.debug("placement consumed %d of %d hosts", len(host_list), len(hosts))

    # 3. Apply orthogonal constraints (max_nodes is a recipe hard cap).
    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        host_list = host_list[: recipe.max_nodes]
        # Placement is now potentially stale; clear so the launcher recomputes.
        placement = None

    is_solo = is_solo_request or len(host_list) <= 1
    if is_solo and len(host_list) > 1:
        host_list = host_list[:1]
        placement = None

    # 3a. Compute intent_id + placement_token; compose cluster_id.
    # The launcher honours ``cluster_id_override`` so we hand it the
    # composed cluster_id rather than letting it derive one from
    # (recipe, hosts).  Per-launch uniqueness via placement_token
    # ensures load-aware schedulers can place the same intent on
    # different host sets without identifier collisions.
    intent_id = generate_intent_id(recipe, options.overrides)
    placement_token = generate_placement_token()
    cluster_id_for_launch = options.cluster_id_override or generate_cluster_id(intent_id, placement_token)
    # Recover intent + token from the override when one was supplied so
    # the result still carries accurate metadata.
    if options.cluster_id_override:
        try:
            parsed_intent, parsed_token = parse_cluster_id(options.cluster_id_override)
            intent_id = parsed_intent
            placement_token = parsed_token
        except ValueError:
            # Non-canonical override (e.g. a user-supplied label) — keep
            # the freshly-computed intent_id but blank the token so
            # downstream consumers don't surface a fake one.
            placement_token = ""

    # 4. Translate options → launch_inference kwargs.
    launch_kwargs: dict[str, Any] = {
        "recipe": recipe,
        "runtime": runtime,
        "host_list": host_list,
        "overrides": dict(options.overrides),
        "config": config,
        "v": sctx.variables,
        "sctx": sctx,
        "is_solo": is_solo,
        "transfer_mode": options.transfer_mode,
        "transfer_interface": options.transfer_interface,
        "cache_dir": options.cache_dir,
        "local_cache_dir": options.local_cache_dir,
        "dry_run": options.dry_run,
        "detached": options.detached,
        "follow": options.follow,
        "ray_port": options.ray_port,
        "dashboard_port": options.dashboard_port,
        "dashboard": options.dashboard,
        "init_port": options.init_port,
        "executor_config": _build_executor_overrides(options),
        "extra_docker_opts": list(options.extra_docker_opts) if options.extra_docker_opts else None,
        "rootless": not options.rootful,
        "auto_user": not options.rootful,
        "cluster": cluster_def,
        "placement": placement,
        "trust": bool(options.trust),
        "sync_tuning": options.sync_tuning,
        "topology": options.topology,
        "cluster_id_override": cluster_id_for_launch,
        "recipe_ref": options.recipe_ref,
    }

    # 5. Launch.
    try:
        result = launch_inference(**launch_kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except (InsufficientCapacity, LayoutRequired, SparkrunError):
        # Typed API errors flow through unchanged.
        raise
    except Exception as e:
        raise SparkrunError("launch_inference failed: %s" % e) from e

    # 6. Build RunResult.
    metadata: dict[str, Any] = {
        "recipe": getattr(recipe, "qualified_name", None) or getattr(recipe, "name", None),
        "model": getattr(recipe, "model", None),
        "container_image": result.container_image,
        "serve_port": result.serve_port,
        "effective_cache_dir": result.effective_cache_dir,
    }
    if result.recipe_ref:
        metadata["recipe_ref"] = result.recipe_ref
    if result.runtime_info:
        metadata["runtime_info"] = dict(result.runtime_info)

    # Recover identifier components from the launcher's final cluster_id
    # in case it differs from the one we composed (e.g. an external
    # caller passed a non-canonical cluster_id_override through).
    final_cluster_id = result.cluster_id
    final_intent_id = intent_id
    final_placement_token = placement_token
    try:
        parsed_intent, parsed_token = parse_cluster_id(final_cluster_id)
        final_intent_id = parsed_intent
        final_placement_token = parsed_token
    except ValueError:
        # Non-canonical cluster_id (manual override) — keep the values
        # we computed pre-launch so RunResult still carries something
        # meaningful.
        pass

    return RunResult(
        cluster_id=final_cluster_id,
        intent_id=final_intent_id,
        placement_token=final_placement_token,
        host_list=tuple(result.host_list),
        placement=placement,
        scheduler=_resolve_scheduler_name(effective_scheduler, sctx),
        runtime=runtime.runtime_name,
        executor=_executor_name_from_result(result),
        started_at=started_at,
        dry_run=options.dry_run,
        is_solo=result.is_solo,
        rc=int(result.rc),
        serve_command=result.serve_command or "",
        container_image=result.container_image or "",
        serve_port=int(result.serve_port or 0),
        effective_cache_dir=result.effective_cache_dir or "",
        runtime_info=dict(result.runtime_info or {}),
        metadata=metadata,
        launch_result=result,
    )


def _build_executor_overrides(options: RunOptions) -> dict[str, Any]:
    """Flatten ``options.executor`` + ``options.executor_config`` into the
    ``cli_overrides`` dict that ``launch_inference`` forwards to
    :func:`sparkrun.orchestration.executor.resolve_executor`."""
    overrides: dict[str, Any] = {}
    if options.executor:
        overrides["executor"] = options.executor
    if options.executor_config:
        for key, value in options.executor_config.items():
            overrides[key] = value
    return overrides


def _safe_acquire_status(host_list, cluster_def, sctx):
    """Best-effort :class:`ClusterStatus` query for the active host list.

    Failures (partial reachability, missing executor, transient SSH
    errors) are swallowed and ``None`` is returned so the scheduling
    path stays resilient.  Schedulers that don't consume occupancy
    info (e.g. greedy) simply ignore the field; ``occupancy-sparse``
    and ``occupancy-dense`` degrade to their no-status path.
    """
    try:
        from sparkrun.api._status import status as get_status

        return get_status(
            list(host_list),
            cluster=cluster_def,
            sctx=sctx,
        )
    except Exception as e:
        logger.debug("Cluster status query failed; scheduling without occupancy info: %s", e)
        return None


def _resolve_scheduler_name(effective_scheduler, sctx):
    """Return the registered ``scheduler_name`` for *effective_scheduler*.

    Looking up the scheduler plugin guarantees ``RunResult.scheduler``
    carries the *actually-used* name (e.g. ``"occupancy-sparse"`` when
    the caller relied on the project default) rather than echoing the
    possibly-``None`` selector that was passed in.
    """
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER, get_scheduler

    try:
        plugin = get_scheduler(effective_scheduler, v=sctx.variables if sctx is not None else None)
        return plugin.scheduler_name
    except Exception:
        return effective_scheduler or FALLBACK_DEFAULT_SCHEDULER


def _executor_name_from_result(result) -> str:
    """Recover the executor's name from the launcher's runtime, if it was set.

    The launcher stamps ``runtime.executor`` during launch; we read its
    ``executor_name`` attribute.  Falls back to ``"docker"`` (the
    library default) when the launcher didn't populate it (e.g. dry-run
    paths that short-circuit before executor resolution).
    """
    executor = getattr(result.runtime, "executor", None)
    if executor is None:
        return "docker"
    return getattr(executor, "executor_name", "docker")


__all__ = ["run"]
