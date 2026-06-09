"""GreedyScheduler — first-fit by host capacity, ignores cluster status.

This module owns the greedy placement algorithm: sequentially packs
ranks onto hosts up to each host's accelerator capacity, honoring an
explicit :class:`~sparkrun.core.layout.RecipeLayout` when provided.

The algorithm is exposed in two forms:

- :func:`pack` — raw entry point.  Returns a
  :class:`~sparkrun.core.placement.RankAssignment` and raises
  placement-level exceptions
  (:class:`~sparkrun.core.placement.InsufficientCapacityError`,
  :class:`~sparkrun.core.placement.LayoutRequiredError`,
  :class:`~sparkrun.core.placement.PlacementError`).
- :class:`GreedyScheduler` — the SAF-registered scheduler that wraps
  :func:`pack` and translates exceptions into the scheduler-level
  vocabulary (:class:`InfeasibleScheduleError` /
  :class:`LayoutConflictError`).

Other schedulers (occupancy-sparse, occupancy-dense, best-fit) are
expected to write their own packing loops rather than reuse this one.
The greedy algorithm and the greedy scheduler are intentionally
co-located.
"""

from __future__ import annotations

from typing import Mapping

from sparkrun.core.hardware import HostHardware, default_dgx_spark_hardware
from sparkrun.core.layout import RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import (
    InfeasibleScheduleError,
    InsufficientCapacityError,
    LayoutConflictError,
    LayoutRequiredError,
    PlacementError,
    RankAssignment,
    RankSlot,
    Scheduler,
    SchedulingError,
    SchedulingRequest,
    SchedulingResult,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _hw_for(host: str, host_hardware: Mapping[str, HostHardware] | None) -> HostHardware:
    """Return the per-host hardware spec or the DGX Spark default fallback."""
    if host_hardware and host in host_hardware:
        return host_hardware[host]
    return default_dgx_spark_hardware()


def _host_capacity(hw: HostHardware) -> int:
    """Total accelerator count across all entries on this host."""
    return hw.total_gpus


def _host_vendor(hw: HostHardware) -> str | None:
    """Single vendor name iff this host advertises exactly one vendor, else ``None``."""
    vendors = hw.vendors
    return next(iter(vendors)) if len(vendors) == 1 else None


def _host_gpu_memory(hw: HostHardware) -> list[float | None]:
    """Per-local-GPU *usable* memory budget on a host, in local-index order.

    Mirrors ``schedulers._occupancy_base._host_gpu_memory``: each entry is
    ``memory_gb × max_gpu_memory_utilization`` (the usable-memory cap baked into
    ``AcceleratorSpec.max_gpu_memory_utilization`` upstream).  An entry is
    ``None`` when the spec did not declare ``memory_gb`` — callers treat that
    slot as "memory ignored".  Kept local to greedy so the scheduler fallback
    stays free of ``platforms`` / ``cluster_manager`` imports.
    """
    mem: list[float | None] = []
    for spec in hw.accelerators:
        if spec.memory_gb is None:
            usable: float | None = None
        else:
            cap = spec.max_gpu_memory_utilization
            usable = spec.memory_gb * (cap if cap is not None else 1.0)
        for _ in range(spec.count):
            mem.append(usable)
    return mem


# --------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------


def pack(
    parallelism: ParallelismConfig,
    hosts: list[str],
    *,
    host_hardware: Mapping[str, HostHardware] | None = None,
    layout: RecipeLayout | None = None,
    per_rank_memory_gb: float | None = None,
) -> RankAssignment:
    """Greedy placement entry point.

    Maps global ranks onto a cluster, honoring an optional explicit
    layout.  Raises placement-level exceptions on infeasibility.

    The number of ranks packed is :meth:`ParallelismConfig.world_size`,
    which honours :attr:`ParallelismConfig.total_ranks` overrides
    (baked in by :meth:`Runtime.world_size` upstream) and falls back to
    the ``tp * pp * dp`` formula when no override is set.

    Args:
        parallelism: ``tp/pp/dp/ep/cp`` dimensions plus an optional
            runtime-derived ``total_ranks`` override (consulted via
            :meth:`ParallelismConfig.world_size`).
        hosts: Ordered host list.
        host_hardware: Optional per-host accelerator metadata.  Missing
            entries default to DGX Spark (1× GB10, 121 GB).
        layout: Optional explicit recipe layout.  When provided with
            non-empty ``placements`` it is honored verbatim.
        per_rank_memory_gb: Optional per-rank (whole-GPU) VRAM requirement.
            When set, a GPU slot is rejected from the auto-fit pack when its
            declared usable memory (``memory_gb × max_gpu_memory_utilization``)
            is below this value.  Slots with undeclared memory (``None``) are
            always accepted (memory ignored), preserving today's behavior.
            ``None`` disables the memory check entirely (byte-identical to the
            pre-memory-fit behavior).

    Raises:
        :class:`LayoutRequiredError`: Heterogeneous cluster (>1 vendor across
            placed hosts) without an explicit layout.
        :class:`InsufficientCapacityError`: Cluster doesn't have enough slots
            for the requested parallelism (auto-fit path).
        :class:`PlacementError`: Explicit layout doesn't cover the requested
            rank count or references unknown hosts.
    """
    effective_total = parallelism.world_size()

    if layout is not None and layout.placements:
        return _placement_from_layout(layout, hosts, effective_total)

    return _auto_pack(hosts, host_hardware, effective_total, per_rank_memory_gb=per_rank_memory_gb)


def _placement_from_layout(
    layout: RecipeLayout,
    hosts: list[str],
    total_gpus: int,
) -> RankAssignment:
    """Honor an explicit ``RecipeLayout.placements`` verbatim."""
    host_set = set(hosts)
    by_rank_dict: dict[int, RankSlot] = {}
    hosts_used: list[str] = []
    seen_hosts: set[str] = set()

    for placement in layout.placements:
        if placement.host not in host_set:
            raise PlacementError("Layout placement references host '%s' not present in cluster hosts %s" % (placement.host, hosts))
        if placement.host not in seen_hosts:
            hosts_used.append(placement.host)
            seen_hosts.add(placement.host)

        local_gpus = placement.local_gpus or tuple(range(len(placement.ranks)))
        if len(local_gpus) != len(placement.ranks):
            raise PlacementError(
                "Layout placement for host '%s' has %d ranks but %d local_gpus" % (placement.host, len(placement.ranks), len(local_gpus))
            )
        for rank, local_gpu in zip(placement.ranks, local_gpus):
            if rank in by_rank_dict:
                raise PlacementError("Layout assigns rank %d to multiple hosts" % rank)
            by_rank_dict[rank] = RankSlot(host=placement.host, local_gpu=int(local_gpu))

    if total_gpus > 0:
        missing = sorted(set(range(total_gpus)) - by_rank_dict.keys())
        if missing:
            raise PlacementError("Layout does not cover ranks %s (parallelism requires %d ranks)" % (missing, total_gpus))

    ordered = tuple(by_rank_dict[i] for i in sorted(by_rank_dict)) if not total_gpus else tuple(by_rank_dict[i] for i in range(total_gpus))
    return RankAssignment(by_rank=ordered, hosts_used=tuple(hosts_used))


def _auto_pack(
    hosts: list[str],
    host_hardware: Mapping[str, HostHardware] | None,
    total_gpus: int,
    *,
    per_rank_memory_gb: float | None = None,
) -> RankAssignment:
    """Sequentially pack ranks onto hosts up to each host's capacity.

    Requires that all placed hosts share a single accelerator vendor;
    raises :class:`LayoutRequiredError` for multi-vendor clusters.

    A multi-vendor cluster is one where the *placed* hosts (those
    receiving at least one rank) advertise more than one distinct
    vendor across their accelerators.  Hosts beyond the parallelism
    budget are not consulted, so a cluster with mixed vendors can
    still be auto-fit as long as the leading slice it consumes is
    single-vendor.

    When *per_rank_memory_gb* is set, a GPU slot whose declared usable
    memory is below the requirement is rejected (it does not count toward
    capacity and receives no rank); slots with undeclared memory (``None``)
    are always accepted.  When *per_rank_memory_gb* is ``None`` every slot is
    accepted — byte-identical to the pre-memory-fit greedy pack.
    """
    if total_gpus <= 0:
        return RankAssignment(by_rank=(), hosts_used=())

    by_rank: list[RankSlot] = []
    hosts_used: list[str] = []
    placed_vendors: set[str] = set()
    remaining = total_gpus

    for host in hosts:
        if remaining <= 0:
            break
        hw = _hw_for(host, host_hardware)
        capacity = _host_capacity(hw)
        if capacity <= 0:
            continue
        vendor = _host_vendor(hw)
        if vendor is None:
            # Multi-vendor *single host* (e.g. Apple M5 + RTX): refuse to auto-fit.
            raise LayoutRequiredError(
                "Host '%s' advertises multiple accelerator vendors (%s); "
                "recipe.layout.placements must specify which ranks land on which accelerator" % (host, sorted(hw.vendors))
            )

        # Per-slot memory eligibility.  A slot is eligible when memory is not
        # requested, the slot declares no memory (None → ignored), or its
        # capped usable memory meets the per-rank requirement.
        if per_rank_memory_gb is None:
            eligible_indices = list(range(capacity))
        else:
            gpu_mem = _host_gpu_memory(hw)
            eligible_indices = [idx for idx, usable in enumerate(gpu_mem) if usable is None or usable >= per_rank_memory_gb]
        if not eligible_indices:
            # No GPU on this host can hold the model under the cap — skip it
            # (mirrors a fully-occupied host in the occupancy path).  The host
            # places no ranks, so it does not count toward the vendor mix.
            continue

        # This host will place at least one rank → it counts as a placed host
        # for the single-vendor invariant.
        placed_vendors.add(vendor)
        if len(placed_vendors) > 1:
            raise LayoutRequiredError(
                "Cluster spans multiple accelerator vendors %s; "
                "recipe.layout.placements is required for heterogeneous-vendor clusters" % sorted(placed_vendors)
            )
        take = min(len(eligible_indices), remaining)
        for local_idx in eligible_indices[:take]:
            by_rank.append(RankSlot(host=host, local_gpu=local_idx))
        hosts_used.append(host)
        remaining -= take

    if remaining > 0:
        raise InsufficientCapacityError(
            "Cluster cannot satisfy %d ranks: only %d accelerator slot(s) available across %d host(s)"
            % (total_gpus, total_gpus - remaining, len(hosts))
        )

    return RankAssignment(by_rank=tuple(by_rank), hosts_used=tuple(hosts_used))


# --------------------------------------------------------------------------
# Scheduler plugin
# --------------------------------------------------------------------------


class GreedyScheduler(Scheduler):
    """Greedy first-fit scheduler — the default.

    Wraps :func:`pack` and translates placement-level exceptions into
    the scheduler-level vocabulary so ``sparkrun.api`` can surface a
    single error hierarchy regardless of which scheduler ran.

    **Whole-GPU only.**  Rejects requests carrying a fractional
    :class:`~sparkrun.core.scheduler.ResourceRequest`
    (``util_fraction < 1.0``) rather than silently treating them as
    whole-GPU.  Fractional sharing is the job of a separate scheduler
    (e.g. ``SparsePackScheduler`` / ``DensePackScheduler``).

    Also ignores :attr:`SchedulingRequest.status` — load-aware behavior
    is out of scope for this strategy (see ``occupancy-sparse`` /
    ``occupancy-dense``).
    """

    scheduler_name = "greedy"

    def schedule(self, request: SchedulingRequest) -> SchedulingResult:
        if request.resources is not None and request.resources.is_fractional():
            raise SchedulingError(
                "GreedyScheduler does not support fractional GPU sharing "
                "(util_fraction=%s); select a fractional-capable scheduler "
                "or omit ResourceRequest.util_fraction" % request.resources.util_fraction
            )

        try:
            assignment = pack(
                request.parallelism,
                list(request.hosts),
                host_hardware=request.host_hardware,
                layout=request.layout,
            )
        except InsufficientCapacityError as e:
            raise InfeasibleScheduleError(str(e)) from e
        except LayoutRequiredError as e:
            raise LayoutConflictError(str(e)) from e
        except PlacementError as e:
            raise SchedulingError(str(e)) from e

        diagnostics = self._diagnostics(request, assignment)
        return SchedulingResult(
            assignment=assignment,
            scheduler_name=self.scheduler_name,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _diagnostics(request: SchedulingRequest, assignment: RankAssignment) -> tuple[str, ...]:
        hosts_used = assignment.hosts_used
        if not hosts_used:
            return ()
        if len(hosts_used) < len(request.hosts):
            return ("packed %d ranks across %d of %d hosts" % (assignment.total_ranks, len(hosts_used), len(request.hosts)),)
        return ("packed %d ranks across %d hosts" % (assignment.total_ranks, len(hosts_used)),)
