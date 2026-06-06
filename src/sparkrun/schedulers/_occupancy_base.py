"""Shared base for the occupancy-sparse / occupancy-dense schedulers.

This module is **private** to the schedulers package — it carries the
shared layout/fallback/packing logic for the public
:class:`~sparkrun.schedulers.sparse_pack.SparsePackScheduler` and
:class:`~sparkrun.schedulers.dense_pack.DensePackScheduler` plugins.

Both public schedulers inherit from :class:`_OccupancyAwareBase` and
override two strategy hooks:

- :meth:`_OccupancyAwareBase._sort_hosts` — orders the candidate host list
  by effective load before packing begins.  Sparse variants sort ascending
  (least-loaded first); dense variants sort descending (most-loaded
  first).
- :meth:`_OccupancyAwareBase._select_gpu_index` — picks a local GPU on the
  current host.  Sparse variants pick the GPU with the *most* remaining
  util budget that still fits; dense variants pick the one with the
  *least* remaining util budget that still fits (classical best-fit).

The base class never registers as a scheduler (its
:attr:`scheduler_name` is ``""``) and ``bootstrap.init_sparkrun`` skips
plugins with empty :attr:`scheduler_name` so only the concrete subclasses
appear in :func:`list_schedulers`.

Head-node overhead heuristic
============================

The head node of any existing workload (rank 0's host) typically does
extra coordination work compared to its rank-only peers.  We model that
with :attr:`HEAD_NODE_OVERHEAD` — a small constant (5%) added to a
host's effective *load score* when the heuristic identifies it as the
head of a previously-launched workload.

The overhead is used **only** for sparse/dense decision-making
(ordering and tie-breaking).  It is **never** subtracted from a host's
real capacity — a host that physically still has room is still
considered eligible regardless of head status.

The heuristic does not have rank-position information in
:class:`~sparkrun.core.cluster_status.ClusterStatus`, so it groups every
:class:`~sparkrun.core.cluster_status.RunningWorkload` by
``cluster_id`` and calls the host with the most ranks for a given
cluster_id the "head" (lexicographic host name breaks ties).  This is a
best-effort signal; it degrades gracefully when no status is provided.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Mapping

from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
from sparkrun.core.hardware import HostHardware, default_dgx_spark_hardware
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
from sparkrun.schedulers.greedy import pack


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _hw_for(host: str, host_hardware: Mapping[str, HostHardware] | None) -> HostHardware:
    if host_hardware and host in host_hardware:
        return host_hardware[host]
    return default_dgx_spark_hardware()


def _host_vendor(hw: HostHardware) -> str | None:
    vendors = hw.vendors
    return next(iter(vendors)) if len(vendors) == 1 else None


def _host_gpu_memory(hw: HostHardware) -> list[float | None]:
    """Per-local-GPU *usable* memory budget on a host, expanded from ``accelerators``.

    Returns one entry per accelerator slot (in local-index order).  Each entry
    is ``memory_gb × max_gpu_memory_utilization`` — the usable-memory cap
    resolved upstream (in ``resolve_effective_hosts`` via
    :func:`sparkrun.core.limits.resolved_hardware_for_scheduling`) and baked
    into ``AcceleratorSpec.max_gpu_memory_utilization``.  A missing cap means
    ``1.0`` (no cap).  An entry is ``None`` when the spec did not declare
    ``memory_gb`` — callers fall back to "memory ignored" for that slot.
    """
    mem: list[float | None] = []
    for spec in hw.accelerators:
        if spec.memory_gb is None:
            usable: float | None = None
        else:
            usable = spec.memory_gb * (spec.max_gpu_memory_utilization or 1.0)
        for _ in range(spec.count):
            mem.append(usable)
    return mem


# --------------------------------------------------------------------------
# Base scheduler
# --------------------------------------------------------------------------


class _OccupancyAwareBase(Scheduler):
    """Private base implementing the shared occupancy-conscious packing loop.

    Subclasses define the *ordering* policy (host preference + GPU
    selection) — the data flow, layout shortcircuit, fallback path, and
    multi-vendor detection are inherited.

    See the module docstring for the head-node overhead heuristic.
    """

    # Subclasses provide a concrete name; this base intentionally has
    # ``""`` so bootstrap skips registering it as a plugin.
    scheduler_name: ClassVar[str] = ""

    #: Extra effective load attributed to the host that hosts rank 0 of an
    #: existing workload.  Affects host-ordering / tie-breaking only;
    #: never subtracted from real capacity.
    HEAD_NODE_OVERHEAD: ClassVar[float] = 0.05

    # ----------------------------------------------------------------------
    # Scheduler entry point
    # ----------------------------------------------------------------------

    def schedule(self, request: SchedulingRequest) -> SchedulingResult:
        # 1. Honor an explicit layout verbatim (same as greedy).
        if request.layout is not None and request.layout.placements:
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
            return SchedulingResult(
                assignment=assignment,
                scheduler_name=self.scheduler_name,
                diagnostics=("layout honored verbatim",),
            )

        resources = request.resources
        is_fractional = resources is not None and resources.is_fractional()
        has_status = request.status is not None and len(request.status.hosts) > 0

        # 2. Fallback path: no occupancy + no fractional claim → greedy.
        # Whole-GPU memory claims (util_fraction == 1.0, memory_gb set)
        # also flow through the fallback because greedy ignores memory
        # but accepts the request.
        if not is_fractional and not has_status:
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
            return SchedulingResult(
                assignment=assignment,
                scheduler_name=self.scheduler_name,
                diagnostics=self._diagnostics(request, assignment, mode="fallback"),
            )

        # 3. Occupancy-aware pack.
        assignment = self._pack_with_occupancy(request)
        return SchedulingResult(
            assignment=assignment,
            scheduler_name=self.scheduler_name,
            diagnostics=self._diagnostics(
                request,
                assignment,
                mode="fractional" if is_fractional else "occupancy",
            ),
        )

    # ----------------------------------------------------------------------
    # Strategy hooks (subclasses must override)
    # ----------------------------------------------------------------------

    @abstractmethod
    def _sort_hosts(self, hosts: tuple[str, ...], scores: dict[str, float]) -> list[str]:
        """Reorder *hosts* by load preference.

        Subclasses pick ascending (sparse) or descending (dense) order.
        Stability matters — Python's :func:`sorted` is stable, so ties
        preserve the input order.
        """
        ...

    @abstractmethod
    def _select_gpu_index(
        self,
        *,
        num_gpus: int,
        util_remaining: list[float],
        mem_remaining: list[float | None],
        gpu_eligible: list[bool],
        per_rank_util: float,
        per_rank_mem: float | None,
    ) -> int | None:
        """Pick the local-GPU index for the next rank on the current host.

        Returns ``None`` when no GPU on the host can host the rank.
        """
        ...

    # ----------------------------------------------------------------------
    # Core packing logic
    # ----------------------------------------------------------------------

    def _pack_with_occupancy(self, request: SchedulingRequest) -> RankAssignment:
        """Pack ranks honoring per-host occupancy + per-rank resource claims.

        Each accelerator is modelled as a (util_budget, memory_budget) pair:

        - ``util_budget = 1.0 - used_util`` (or ``1.0`` when status is absent).
        - ``memory_budget = memory_gb - used_memory`` (or ``+inf`` when
          memory accounting is not requested).

        A rank consumes ``(util_fraction, memory_gb)``; an accelerator
        accepts the rank when both budgets remain non-negative.

        Host ordering is delegated to :meth:`_sort_hosts`; GPU selection
        is delegated to :meth:`_select_gpu_index`.  After the first rank
        of the *new* workload lands somewhere, that host's score gains
        :attr:`HEAD_NODE_OVERHEAD` and the remaining (unvisited) hosts
        are re-sorted.
        """
        total_ranks = request.parallelism.world_size()
        if total_ranks <= 0:
            return RankAssignment(by_rank=(), hosts_used=())

        resources = request.resources
        per_rank_util = resources.util_fraction if resources is not None else 1.0
        per_rank_mem = resources.memory_gb if resources is not None else None
        is_fractional = resources is not None and resources.is_fractional()

        status = request.status

        # Compute per-host load scores once, then ask the subclass to order them.
        scores = self._compute_host_load_scores(request)
        sorted_hosts: list[str] = self._sort_hosts(request.hosts, scores)

        by_rank: list[RankSlot] = []
        hosts_used: list[str] = []
        placed_vendors: set[str] = set()
        remaining = total_ranks
        first_rank_placed = False

        # Walk the sorted list by index so we can re-sort the *remaining*
        # slice in place once the new workload's head host is known.
        i = 0
        while i < len(sorted_hosts) and remaining > 0:
            host = sorted_hosts[i]

            hw = _hw_for(host, request.host_hardware)
            if hw.total_gpus <= 0:
                i += 1
                continue

            vendor = _host_vendor(hw)
            if vendor is None:
                raise LayoutConflictError(
                    "Host '%s' advertises multiple accelerator vendors (%s); "
                    "recipe.layout.placements must specify which ranks land on which accelerator" % (host, sorted(hw.vendors))
                )

            # Build per-GPU (util_remaining, memory_remaining) budgets.
            gpu_mem_specs = _host_gpu_memory(hw)
            num_gpus = len(gpu_mem_specs)
            host_occ = status.for_host(host) if status is not None else None

            util_remaining, mem_remaining, gpu_eligible = self._build_budgets(
                num_gpus=num_gpus,
                gpu_mem_specs=gpu_mem_specs,
                host_occ=host_occ,
                is_fractional=is_fractional,
            )

            # Skip hosts that ended up with no eligible GPUs (whole-GPU
            # placement on a fully-occupied host).
            if not any(gpu_eligible):
                i += 1
                continue

            host_placed_any = False
            for _ in range(remaining):
                gpu_idx = self._select_gpu_index(
                    num_gpus=num_gpus,
                    util_remaining=util_remaining,
                    mem_remaining=mem_remaining,
                    gpu_eligible=gpu_eligible,
                    per_rank_util=per_rank_util,
                    per_rank_mem=per_rank_mem,
                )
                if gpu_idx is None:
                    break

                util_remaining[gpu_idx] -= per_rank_util
                if per_rank_mem is not None and mem_remaining[gpu_idx] is not None:
                    mem_remaining[gpu_idx] -= per_rank_mem
                if not is_fractional:
                    # Whole-GPU placement: mark slot occupied so the next
                    # rank picks a different GPU on this host.
                    gpu_eligible[gpu_idx] = False

                by_rank.append(
                    RankSlot(
                        host=host,
                        local_gpu=gpu_idx,
                        util_fraction=per_rank_util,
                        memory_gb=per_rank_mem,
                    )
                )
                remaining -= 1
                host_placed_any = True

                if remaining <= 0:
                    break

            if host_placed_any:
                placed_vendors.add(vendor)
                if len(placed_vendors) > 1:
                    raise LayoutConflictError(
                        "Cluster spans multiple accelerator vendors %s; "
                        "recipe.layout.placements is required for heterogeneous-vendor clusters" % sorted(placed_vendors)
                    )
                hosts_used.append(host)

                # After placing the first rank of the new workload, mark
                # this host as the new head and re-sort the remaining
                # (unvisited) hosts.  This matters when the head's
                # overhead would flip ordering against neighbouring
                # candidates.
                if not first_rank_placed:
                    first_rank_placed = True
                    scores = dict(scores)
                    scores[host] = scores.get(host, 0.0) + self.HEAD_NODE_OVERHEAD
                    if remaining > 0 and i + 1 < len(sorted_hosts):
                        tail = tuple(sorted_hosts[i + 1 :])
                        resorted_tail = self._sort_hosts(tail, scores)
                        sorted_hosts = sorted_hosts[: i + 1] + resorted_tail

            i += 1

        if remaining > 0:
            placed = total_ranks - remaining
            raise InfeasibleScheduleError(
                "Cluster cannot satisfy %d ranks under current occupancy: "
                "only placed %d of %d ranks across %d host(s)" % (total_ranks, placed, total_ranks, len(request.hosts))
            )

        return RankAssignment(by_rank=tuple(by_rank), hosts_used=tuple(hosts_used))

    # ----------------------------------------------------------------------
    # Host load scoring + head identification
    # ----------------------------------------------------------------------

    def _compute_host_load_scores(self, request: SchedulingRequest) -> dict[str, float]:
        """Return per-host effective load scores in ``[0, ~N]``.

        For each host in :attr:`SchedulingRequest.hosts`:

        - When per-GPU detail is available, ``score = sum(used_util_fraction)``
          across the host's GPUs.
        - When only host-level :attr:`HostOccupancy.used_slots` is
          available, ``score = used_slots / total_gpus`` (a 0-1
          normalized fraction representing average per-GPU load).
        - When the host is absent from status, ``score = 0.0``.

        If the host is identified as the head of an existing workload
        (see :meth:`_identify_existing_heads`), :attr:`HEAD_NODE_OVERHEAD`
        is added on top.
        """
        status = request.status
        heads = self._identify_existing_heads(status)
        scores: dict[str, float] = {}

        for host in request.hosts:
            base = 0.0
            occ = status.for_host(host) if status is not None else None
            if occ is not None:
                if occ.gpus:
                    base = sum(g.used_util_fraction for g in occ.gpus)
                else:
                    total = occ.total_slots
                    if total > 0:
                        base = occ.used_slots / total
                    elif occ.used_slots > 0:
                        # No total info but something is used — treat as
                        # whole-GPU-busy.
                        base = float(occ.used_slots)
            if host in heads:
                base += self.HEAD_NODE_OVERHEAD
            scores[host] = base

        return scores

    @staticmethod
    def _identify_existing_heads(status: ClusterStatus | None) -> set[str]:
        """Identify each existing workload's head host.

        Without rank-position information we treat the host with the
        most ranks for a given ``cluster_id`` as the head; ties are
        broken by lexicographic host name (so the heuristic is
        deterministic).

        Returns the set of host names so identified.  An empty set is
        returned when *status* is ``None`` or carries no workloads.
        """
        if status is None:
            return set()

        # cluster_id -> {host: rank-count}
        ranks_per_cluster: dict[str, dict[str, int]] = {}
        for entry in status.hosts:
            for w in entry.workloads:
                cluster = ranks_per_cluster.setdefault(w.cluster_id, {})
                cluster[entry.host] = cluster.get(entry.host, 0) + max(1, int(w.ranks_on_host))

        heads: set[str] = set()
        for _cluster_id, host_counts in ranks_per_cluster.items():
            if not host_counts:
                continue
            # Sort by (-count, host) so the densest host wins; lex name breaks ties.
            best_host = sorted(host_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            heads.add(best_host)

        return heads

    # ----------------------------------------------------------------------
    # Budget construction (shared between subclasses)
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_budgets(
        *,
        num_gpus: int,
        gpu_mem_specs: list[float | None],
        host_occ: HostOccupancy | None,
        is_fractional: bool,
    ) -> tuple[list[float], list[float | None], list[bool]]:
        """Compute per-GPU (util_remaining, mem_remaining, eligible) lists.

        When ``host_occ`` carries per-GPU detail (:attr:`gpus`) we use it
        directly.  Otherwise we fall back to host-level
        ``used_slots`` / ``free_slots``: for whole-GPU placements that
        translates to "the first ``used_slots`` local-GPU indices are
        already occupied"; for fractional placements we conservatively
        assume even distribution of the host-level utilization across
        the host's GPUs.
        """
        util_remaining: list[float] = [1.0] * num_gpus
        mem_remaining: list[float | None] = list(gpu_mem_specs)
        eligible: list[bool] = [True] * num_gpus

        if host_occ is None:
            return util_remaining, mem_remaining, eligible

        # Per-GPU detail path.
        if host_occ.gpus:
            for gpu in host_occ.gpus:
                if 0 <= gpu.gpu_index < num_gpus:
                    util_remaining[gpu.gpu_index] = max(0.0, 1.0 - gpu.used_util_fraction)
                    if mem_remaining[gpu.gpu_index] is not None:
                        mem_remaining[gpu.gpu_index] = max(
                            0.0,
                            (mem_remaining[gpu.gpu_index] or 0.0) - gpu.used_memory_gb,
                        )
                    if not is_fractional:
                        # Whole-GPU mode: any non-trivial occupancy excludes the GPU.
                        if gpu.workloads or gpu.used_util_fraction > 0 or gpu.used_memory_gb > 0:
                            eligible[gpu.gpu_index] = False
            return util_remaining, mem_remaining, eligible

        # Host-level fallback path.
        used_slots = host_occ.used_slots
        if not is_fractional:
            # Conservatively mark the leading ``used_slots`` GPUs as occupied.
            for i in range(min(used_slots, num_gpus)):
                eligible[i] = False
        elif used_slots > 0 and num_gpus > 0:
            # Fractional mode: spread the assumed utilization evenly.
            # Each "used slot" implies one whole GPU consumed; we treat
            # that as ``used_slots / num_gpus`` per-GPU utilization to
            # avoid double-subtracting on hosts where exact per-GPU info
            # is unavailable.
            per_gpu_used = min(1.0, used_slots / num_gpus)
            for i in range(num_gpus):
                util_remaining[i] = max(0.0, 1.0 - per_gpu_used)

        return util_remaining, mem_remaining, eligible

    # ----------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------

    def _diagnostics(self, request: SchedulingRequest, assignment: RankAssignment, *, mode: str) -> tuple[str, ...]:
        if not assignment.hosts_used:
            return ()
        return (
            "%s (%s): packed %d ranks across %d of %d hosts"
            % (self.scheduler_name, mode, assignment.total_ranks, len(assignment.hosts_used), len(request.hosts)),
        )


__all__ = ["_OccupancyAwareBase"]
