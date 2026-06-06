"""``sparkrun.api`` host + placement resolution (console-free).

This module is the single authority for turning a raw host list plus a
recipe into the *effective* host list and solo flag, honouring:

* the scheduler's placement (``hosts_used`` IS the effective list),
* ``recipe.max_nodes`` (hard cap / hard error),
* the single-host short-circuit and ``solo`` / ``recipe.mode == 'solo'``.

It bakes ``runtime.world_size(...)`` into the scheduling request when a
runtime is supplied, so ``run`` and ``benchmark`` place identically
regardless of which runtime owns the workload.  All failures raise typed
:class:`~sparkrun.api.SparkrunError` subclasses; no console I/O.

The CLI wrapper (``cli/_common.py:resolve_effective_hosts_for_recipe``)
calls :func:`resolve_effective_hosts`, echoes the returned *notes*, and
translates exceptions into ``click.echo`` + ``sys.exit``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import HostsUnreachable, InsufficientCapacity, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def resolve_host_list(
    hosts: str | None,
    hosts_file: str | None,
    cluster_name: str | None,
    config,
    *,
    sctx: "SparkrunContext | None" = None,
) -> list[str]:
    """Resolve a host list from CLI-shaped inputs without console I/O.

    Mirrors the resolution chain of
    ``cli/_common.py:_resolve_hosts_or_exit`` (CLI → file → cluster →
    default), and applies the resolved cluster's SSH user to *config*
    the same way, but raises :class:`HostsUnreachable` instead of
    echoing + exiting when nothing resolves.

    Returns the resolved host list.
    """
    from sparkrun.core.hosts import resolve_hosts
    from sparkrun.core.cluster_manager import resolve_cluster_config

    if sctx is not None:
        cluster_mgr = sctx.cluster_manager
    else:
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import get_config_root

        cluster_mgr = ClusterManager(get_config_root())

    host_list = resolve_hosts(
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        cluster_manager=cluster_mgr,
        config_default_hosts=config.default_hosts,
    )
    if not host_list:
        raise HostsUnreachable("No hosts specified. Use --hosts or configure defaults.")

    cluster_user = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr).user
    if cluster_user:
        config.ssh_user = cluster_user
    return host_list


def resolve_effective_hosts(
    host_list: list[str],
    recipe,
    overrides: dict | None = None,
    *,
    cluster_def=None,
    runtime=None,
    sctx: "SparkrunContext | None" = None,
    solo: bool = False,
    scheduler: str | None = None,
) -> tuple[list[str], bool, list[str], "object | None"]:
    """Compute the effective host list + solo flag via the scheduler.

    This is the single placement authority shared by ``api.run``, the
    benchmark flow, and the CLI ``run`` command.  Placement is a
    *structural* property: the scheduler's ``hosts_used`` IS the
    effective host list.  The three orthogonal constraints that sit
    outside the scheduler are applied here too:

    * ``solo`` (or ``recipe.mode == 'solo'``): force a one-host run.
    * ``recipe.max_nodes``: hard upper bound (hard error when the
      runtime's implied node count exceeds it).
    * Single-host short-circuit: a one-host input bypasses the scheduler.

    When *runtime* is supplied, ``runtime.world_size(parallelism, ...)``
    is baked into the scheduling request's ``total_ranks`` so
    runtime-specific rank math (e.g. Atlas's MoE mesh) is honoured.

    Args:
        host_list: Resolved hosts (CLI / cluster / file).
        recipe: Loaded recipe.
        overrides: CLI overrides (``-o key=value`` flattened).
        cluster_def: Optional :class:`ClusterDefinition` carrying
            per-host hardware (used by the scheduler for multi-GPU
            placement).
        runtime: Optional runtime plugin; when given, its
            ``world_size`` override is baked into the request.
        sctx: Optional shared :class:`SparkrunContext`.
        solo: ``--solo`` flag value.

    Returns:
        ``(effective_host_list, is_solo, notes, placement)`` where
        *notes* is a list of human-readable strings the CLI echoes
        verbatim (e.g. ``"Note: 2 nodes required, using 2 of 4 hosts"``)
        and *placement* is the scheduler's :class:`RankAssignment` (or
        ``None`` when scheduling was bypassed / later invalidated by a
        ``max_nodes`` trim or solo short-circuit).

    Raises:
        InsufficientCapacity: Scheduler can't fit the workload (carries
            ``status`` + ``host_list`` + ``required`` for diagnostics).
        LayoutRequired: Heterogeneous cluster without an explicit layout.
        SparkrunError: Other scheduling failures.
    """
    import dataclasses

    import sparkrun.api as api
    from sparkrun.core.limits import resolved_hardware_for_scheduling
    from sparkrun.core.parallelism import extract_parallelism
    from sparkrun.core.scheduler import ResourceRequest, SchedulingRequest

    overrides = overrides or {}
    notes: list[str] = []
    placement = None

    config_chain = recipe.build_config_chain(overrides)
    parallelism_configured = any(config_chain.get(k) is not None for k in ("tensor_parallel", "pipeline_parallel", "data_parallel"))

    if not solo and len(host_list) > 1 and parallelism_configured:
        parallelism = extract_parallelism(config_chain)
        if runtime is not None:
            # Bake the runtime-derived rank count so the scheduler asks
            # ``parallelism.world_size()`` and gets the right answer
            # regardless of which runtime owns the workload.
            total_ranks = runtime.world_size(parallelism, recipe=recipe, cluster=cluster_def)
            parallelism = dataclasses.replace(parallelism, total_ranks=total_ranks)

        # Best-effort cluster status snapshot so occupancy-aware schedulers
        # can subtract already-running workloads.  Failures are swallowed.
        cluster_status = None
        try:
            cluster_status = api.status(list(host_list), cluster=cluster_def, sctx=sctx)
        except Exception as e:
            logger.debug("Cluster status query failed; scheduling without occupancy info: %s", e)

        # Per-host hardware with the usable-memory cap (max_gpu_memory_utilization)
        # resolved + folded into each AcceleratorSpec, so the scheduler packs
        # against usable memory rather than nominal memory_gb.
        effective_hw = resolved_hardware_for_scheduling(cluster_def, list(host_list))

        # Best-effort per-rank VRAM claim so the scheduler can reject GPUs that
        # can't hold the model within the capped usable memory.  Any estimation
        # failure degrades to resources=None (memory-blind, today's behavior)
        # rather than blocking the launch.
        resources = None
        try:
            est = recipe.estimate_vram(cli_overrides=overrides)
            per_rank = float(getattr(est, "total_per_gpu_gb", 0.0) or 0.0)
            if per_rank > 0:
                resources = ResourceRequest(memory_gb=per_rank, util_fraction=1.0)
        except Exception as e:
            logger.debug("VRAM estimate unavailable; scheduling without memory claim: %s", e)

        request = SchedulingRequest(
            parallelism=parallelism,
            hosts=tuple(host_list),
            host_hardware=effective_hw,
            layout=getattr(recipe, "layout", None),
            status=cluster_status,
            resources=resources,
        )
        try:
            result = api.schedule(request, scheduler=scheduler, sctx=sctx)
        except InsufficientCapacity as e:
            required = parallelism.total_nodes
            if len(host_list) < required:
                msg = "runtime requires %d nodes, but only %d hosts provided" % (required, len(host_list))
            else:
                detail = str(e) or "no free accelerator slots across %d host(s)" % len(host_list)
                msg = "cluster has insufficient free capacity for %d node(s): %s" % (required, detail)
            raise InsufficientCapacity(msg, status=cluster_status, host_list=list(host_list), required=required) from e

        placement = result.assignment
        scheduled_hosts = list(result.assignment.hosts_used)

        # max_nodes is a hard recipe constraint; fail rather than truncate.
        if recipe.max_nodes is not None and len(scheduled_hosts) > recipe.max_nodes:
            raise SparkrunError(
                "runtime requires %d nodes (from parallelism settings), "
                "but recipe '%s' specifies max_nodes=%d" % (len(scheduled_hosts), recipe.qualified_name, recipe.max_nodes)
            )

        if len(scheduled_hosts) < len(host_list):
            notes.append("Note: %d nodes required, using %d of %d hosts" % (len(scheduled_hosts), len(scheduled_hosts), len(host_list)))
        host_list = scheduled_hosts

    # Enforce recipe.max_nodes as an orthogonal cap when scheduling did not run.
    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        if parallelism_configured:
            parallelism = extract_parallelism(config_chain)
            required = parallelism.total_nodes
            if required > recipe.max_nodes:
                raise SparkrunError(
                    "runtime requires %d nodes (from parallelism settings), "
                    "but recipe '%s' specifies max_nodes=%d" % (required, recipe.qualified_name, recipe.max_nodes)
                )
        notes.append("Note: recipe max_nodes=%d, using %d of %d hosts" % (recipe.max_nodes, recipe.max_nodes, len(host_list)))
        host_list = host_list[: recipe.max_nodes]
        # Placement is now stale; clear so the launcher recomputes.
        placement = None

    # Determine final solo mode after scheduling / max_nodes.
    is_solo = bool(solo) or recipe.mode == "solo" or len(host_list) <= 1
    if is_solo and len(host_list) > 1:
        notes.append("Note: solo mode enabled, using 1 of %d hosts" % len(host_list))
        host_list = host_list[:1]
        placement = None

    return host_list, is_solo, notes, placement


__all__ = ["resolve_host_list", "resolve_effective_hosts"]
