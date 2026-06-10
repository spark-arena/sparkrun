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
    exclude_intent_id: str | None = None,
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
        scheduler: Registered scheduler name (caller-resolved selection
            chain).  ``None`` lets :func:`sparkrun.api.schedule` apply the
            default.
        exclude_intent_id: When set, workloads belonging to this intent
            are subtracted from the live occupancy snapshot before
            scheduling.  This makes relaunching / benchmark-resuming a
            workload reuse the hosts its own still-running containers
            occupy instead of treating them as foreign load (which would
            falsely report "no capacity" or relocate the workload off its
            own hosts).

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

    from sparkrun.core.parallelism import extract_parallelism

    overrides = overrides or {}
    notes: list[str] = []
    placement = None

    config_chain = recipe.build_config_chain(overrides)
    # All five parallelism dimensions gate the multi-node scheduling block:
    # a runtime whose world_size derives from expert/context parallelism (e.g.
    # Atlas's tp*ep MoE mesh) must still be scheduled across hosts even when
    # tp/pp/dp are left at 1.
    parallelism_configured = any(
        config_chain.get(k) is not None
        for k in ("tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel", "context_parallel")
    )

    # ``--solo`` / ``recipe.mode == 'solo'`` force a one-host run.  Both skip the
    # multi-node scheduling block and route to the single-host occupancy pick
    # below, so a solo workload still lands on a host that has room rather than
    # blindly on ``host_list[0]``.
    requested_solo = bool(solo) or recipe.mode == "solo"

    if not requested_solo and len(host_list) > 1 and parallelism_configured:
        parallelism = extract_parallelism(config_chain)
        if runtime is not None:
            # Bake the runtime-derived rank count so the scheduler asks
            # ``parallelism.world_size()`` and gets the right answer
            # regardless of which runtime owns the workload.
            total_ranks = runtime.world_size(parallelism, recipe=recipe, cluster=cluster_def)
            parallelism = dataclasses.replace(parallelism, total_ranks=total_ranks)

        required = parallelism.world_size()
        try:
            result, cluster_status = _schedule_hosts(
                host_list,
                recipe,
                overrides,
                parallelism,
                cluster_def=cluster_def,
                sctx=sctx,
                scheduler=scheduler,
                exclude_intent_id=exclude_intent_id,
                layout=recipe.layout,
            )
        except InsufficientCapacity as e:
            cluster_status = getattr(e, "status", None)
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

    # Enforce recipe.max_nodes as an orthogonal cap on the paths where the
    # multi-node scheduling block above did NOT run (solo, single host, or no
    # parallelism configured).
    elif recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        # A contradiction between explicit parallelism and max_nodes is a hard
        # error regardless of solo (a tp=4 recipe cannot honour max_nodes=2).
        if parallelism_configured:
            parallelism = extract_parallelism(config_chain)
            if runtime is not None:
                parallelism = dataclasses.replace(
                    parallelism, total_ranks=runtime.world_size(parallelism, recipe=recipe, cluster=cluster_def)
                )
            required = parallelism.world_size()
            if required > recipe.max_nodes:
                raise SparkrunError(
                    "runtime requires %d nodes (from parallelism settings), "
                    "but recipe '%s' specifies max_nodes=%d" % (required, recipe.qualified_name, recipe.max_nodes)
                )

        orig_count = len(host_list)
        # Occupancy-aware trim when a cluster is known (live load can be
        # consulted): run a one-rank-per-host pass capped at max_nodes so the
        # retained hosts are the least-loaded, and thread the resulting placement
        # rather than clearing it to None for a downstream recompute.  With no
        # cluster (and so no occupancy to consult) the blind prefix slice is
        # equivalent and avoids a pointless scheduler round-trip.
        if cluster_def is not None and not requested_solo:
            implied = dataclasses.replace(extract_parallelism(config_chain), total_ranks=recipe.max_nodes)
            try:
                result, _status = _schedule_hosts(
                    host_list,
                    recipe,
                    overrides,
                    implied,
                    cluster_def=cluster_def,
                    sctx=sctx,
                    scheduler=scheduler,
                    exclude_intent_id=exclude_intent_id,
                    layout=None,
                )
                placement = result.assignment
                host_list = list(result.assignment.hosts_used)
            except (InsufficientCapacity, SparkrunError):
                placement = None
                host_list = host_list[: recipe.max_nodes]
        else:
            host_list = host_list[: recipe.max_nodes]
            placement = None
        notes.append("Note: recipe max_nodes=%d, using %d of %d hosts" % (recipe.max_nodes, len(host_list), orig_count))

    # Determine final solo mode after scheduling / max_nodes.
    is_solo = requested_solo or len(host_list) <= 1
    if is_solo and len(host_list) > 1:
        # Occupancy-aware single-host pick: even a one-host (solo / single-rank)
        # workload should land on a host that has room.  A 1-rank scheduling
        # request lets the configured scheduler choose the least-loaded host
        # with capacity; when no occupancy snapshot is available it greedy-packs
        # and returns the first host with capacity (today's behavior).  When
        # *every* host is full it raises InsufficientCapacity rather than
        # stacking onto an occupied host.
        chosen = _pick_single_host(
            host_list, recipe, overrides, cluster_def=cluster_def, sctx=sctx, scheduler=scheduler, exclude_intent_id=exclude_intent_id
        )
        notes.append("Note: solo mode enabled, using 1 of %d hosts" % len(host_list))
        host_list = [chosen]
        placement = None

    return host_list, is_solo, notes, placement


def _schedule_hosts(
    host_list,
    recipe,
    overrides,
    parallelism,
    *,
    cluster_def,
    sctx,
    scheduler,
    exclude_intent_id,
    layout,
):
    """Build a :class:`SchedulingRequest` and run it through ``api.schedule``.

    The single scheduling call site shared by the multi-node block, the
    ``max_nodes`` trim, and :func:`_pick_single_host`, so occupancy gathering,
    self-intent exclusion, and request construction live in exactly one place.

    Returns ``(SchedulingResult, cluster_status)``.  May raise
    :class:`InsufficientCapacity` / :class:`LayoutRequired` / :class:`SparkrunError`
    from :func:`sparkrun.api.schedule`.
    """
    import sparkrun.api as api
    from sparkrun.core.scheduler import SchedulingRequest

    cluster_status, effective_hw, resources = _gather_scheduling_inputs(
        host_list, recipe, overrides, cluster_def=cluster_def, sctx=sctx, exclude_intent_id=exclude_intent_id
    )
    request = SchedulingRequest(
        parallelism=parallelism,
        hosts=tuple(host_list),
        host_hardware=effective_hw,
        layout=layout,
        status=cluster_status,
        resources=resources,
    )
    try:
        result = api.schedule(request, scheduler=scheduler, sctx=sctx)
    except InsufficientCapacity as e:
        # Attach the snapshot so callers can render capacity diagnostics.
        if getattr(e, "status", None) is None:
            e.status = cluster_status
        raise
    return result, cluster_status


def _gather_scheduling_inputs(host_list, recipe, overrides, *, cluster_def, sctx, exclude_intent_id=None):
    """Best-effort ``(status, host_hardware, resources)`` for a scheduling request.

    Shared by the multi-node scheduling block and the single-host occupancy
    pick so the occupancy snapshot, usable-memory cap baking, and per-rank VRAM
    estimation live in exactly one place.

    * ``status`` — live :class:`ClusterStatus` snapshot (``None`` when the query
      fails; occupancy-aware schedulers then degrade to greedy packing).  When
      *exclude_intent_id* is set, workloads of that intent are subtracted so a
      relaunch / resume reuses the hosts its own containers occupy.
    * ``host_hardware`` — per-host hardware with the ``max_gpu_memory_utilization``
      cap resolved + folded into each ``AcceleratorSpec`` so the scheduler packs
      against usable rather than nominal memory.
    * ``resources`` — per-rank whole-GPU VRAM claim (``None`` when estimation is
      unavailable, preserving the memory-blind path).
    """
    import sparkrun.api as api
    from sparkrun.core.limits import resolved_hardware_for_scheduling
    from sparkrun.core.scheduler import ResourceRequest

    cluster_status = None
    try:
        cluster_status = api.status(list(host_list), cluster=cluster_def, sctx=sctx)
    except Exception as e:
        logger.debug("Cluster status query failed; scheduling without occupancy info: %s", e)

    if cluster_status is not None and exclude_intent_id:
        cluster_status = _status_excluding_intent(cluster_status, exclude_intent_id)

    effective_hw = resolved_hardware_for_scheduling(cluster_def, list(host_list))

    # Defensive: every scheduled host must carry baked hardware.  A host missing
    # from the map would fall back to default_dgx_spark_hardware() inside the
    # scheduler (cap=None → full nominal memory), silently over-committing capped
    # memory.  resolved_hardware_for_scheduling bakes every host today, so a gap
    # means a future regression — log it loudly.
    missing_hw = [h for h in host_list if h not in effective_hw]
    if missing_hw:
        logger.warning(
            "Baked scheduling hardware is missing %d host(s) %s; the scheduler "
            "will fall back to uncapped defaults for them (potential memory over-commit)",
            len(missing_hw),
            missing_hw,
        )

    resources = None
    try:
        est = recipe.estimate_vram(cli_overrides=overrides)
        per_rank = float(getattr(est, "total_per_gpu_gb", 0.0) or 0.0)
        if per_rank > 0:
            resources = ResourceRequest(memory_gb=per_rank, util_fraction=1.0)
    except Exception as e:
        logger.debug("VRAM estimate unavailable; scheduling without memory claim: %s", e)

    return cluster_status, effective_hw, resources


def _pick_single_host(host_list, recipe, overrides, *, cluster_def, sctx, scheduler, exclude_intent_id=None):
    """Choose the single least-loaded host with room for a solo (1-rank) run.

    Runs a ``world_size == 1`` scheduling request over *host_list* so the
    configured scheduler picks an occupancy-appropriate host.  ``layout`` is
    intentionally dropped — a solo run ignores recipe parallelism/layout and
    only needs one host.

    Returns the chosen host name.  Raises
    :class:`~sparkrun.api.InsufficientCapacity` when no host can fit the
    workload (every accelerator occupied / too little usable VRAM); when no
    occupancy snapshot is available the scheduler greedy-packs and returns the
    first host with capacity, so the no-status path is byte-identical to the
    pre-occupancy ``host_list[0]`` behavior.
    """
    import dataclasses

    from sparkrun.core.parallelism import extract_parallelism

    config_chain = recipe.build_config_chain(overrides)
    parallelism = dataclasses.replace(extract_parallelism(config_chain), total_ranks=1)

    try:
        result, cluster_status = _schedule_hosts(
            host_list,
            recipe,
            overrides,
            parallelism,
            cluster_def=cluster_def,
            sctx=sctx,
            scheduler=scheduler,
            exclude_intent_id=exclude_intent_id,
            layout=None,
        )
    except InsufficientCapacity as e:
        detail = str(e) or "all %d host(s) occupied" % len(host_list)
        raise InsufficientCapacity(
            "cluster has no free capacity for a solo (1-node) run: %s" % detail,
            status=getattr(e, "status", None),
            host_list=list(host_list),
            required=1,
        ) from e

    hosts_used = list(result.assignment.hosts_used)
    return hosts_used[0] if hosts_used else host_list[0]


def _status_excluding_intent(status, intent_id: str):
    """Return a copy of *status* with workloads of *intent_id* subtracted.

    Used so a relaunch / benchmark-resume does not count its own still-running
    containers as foreign occupancy.  A workload matches when its ``intent_id``
    equals *intent_id* or its ``cluster_id`` parses to that intent prefix.  Host
    ``used_slots`` / ``free_slots`` are rebalanced by the removed workloads'
    ``ranks_on_host``; per-GPU entries that become workload-free are reset to
    zero util/memory (we cannot partially attribute residual usage).
    """
    import dataclasses

    def _intent_of(cluster_id: str) -> str | None:
        try:
            from sparkrun.orchestration.job_metadata import parse_cluster_id

            return parse_cluster_id(cluster_id)[0]
        except Exception:
            return None

    def _matches(w) -> bool:
        return (w.intent_id == intent_id) or (_intent_of(w.cluster_id) == intent_id)

    new_hosts = []
    changed = False
    for occ in status.hosts:
        removed = [w for w in occ.workloads if _matches(w)]
        if not removed:
            new_hosts.append(occ)
            continue
        changed = True
        kept = tuple(w for w in occ.workloads if not _matches(w))
        freed = sum(max(1, int(w.ranks_on_host)) for w in removed)
        new_used = max(0, occ.used_slots - freed)
        new_free = occ.free_slots + (occ.used_slots - new_used)
        new_gpus = []
        for g in occ.gpus:
            g_kept = tuple(w for w in g.workloads if not _matches(w))
            if len(g_kept) == len(g.workloads):
                new_gpus.append(g)
            elif g_kept:
                # Other workloads remain on this GPU; can't attribute residual
                # util/mem, so keep the observed values (conservative).
                new_gpus.append(dataclasses.replace(g, workloads=g_kept))
            else:
                new_gpus.append(dataclasses.replace(g, workloads=(), used_memory_gb=0.0, used_util_fraction=0.0))
        new_hosts.append(dataclasses.replace(occ, workloads=kept, used_slots=new_used, free_slots=new_free, gpus=tuple(new_gpus)))

    if not changed:
        return status
    return dataclasses.replace(status, hosts=tuple(new_hosts))


__all__ = ["resolve_host_list", "resolve_effective_hosts"]
