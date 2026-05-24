"""OccupancyAwareScheduler — fractional GPU + cluster occupancy awareness.

This scheduler extends the greedy strategy with two capabilities the
default :class:`~sparkrun.schedulers.greedy.GreedyScheduler` deliberately
refuses to model:

- **Cluster occupancy**: when
  :attr:`~sparkrun.core.scheduler.SchedulingRequest.status` carries a
  :class:`~sparkrun.core.cluster_status.ClusterStatus` snapshot, used
  slots are subtracted from each host's nominal capacity before packing.
  When per-accelerator detail
  (:attr:`~sparkrun.core.cluster_status.HostOccupancy.gpus`) is available
  the scheduler reads per-GPU used util/memory; otherwise it falls back
  to host-level :attr:`used_slots` / :attr:`free_slots`.
- **Fractional GPU sharing**: when
  :attr:`~sparkrun.core.scheduler.SchedulingRequest.resources` carries a
  :class:`~sparkrun.core.scheduler.ResourceRequest` with
  ``util_fraction < 1.0`` (or a non-``None`` ``memory_gb`` budget),
  multiple ranks may share one accelerator as long as their cumulative
  util fits within ``1.0`` and their cumulative memory fits within the
  GPU's physical memory.

When neither :attr:`status` nor a fractional :attr:`resources` is set,
this scheduler delegates to the greedy whole-GPU algorithm so callers
that opt in to ``"occupancy-aware"`` get sensible behavior on idle
homogeneous clusters.

The scheduler is a pure function over its inputs: it does **not** call
any executor.  Callers (``sparkrun.api.run``, ``sparkrun.api.schedule``)
are responsible for populating
:attr:`~sparkrun.core.scheduler.SchedulingRequest.status` before
invocation when occupancy-awareness is desired.
"""

from __future__ import annotations

from typing import Mapping

from sparkrun.core.cluster_status import HostOccupancy
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
    """Per-local-GPU memory budget on a host, expanded from ``accelerators``.

    Returns one entry per accelerator slot (in local-index order).  An
    entry is ``None`` when the underlying :class:`AcceleratorSpec` did
    not declare ``memory_gb`` — callers fall back to "memory ignored"
    for that slot.
    """
    mem: list[float | None] = []
    for spec in hw.accelerators:
        for _ in range(spec.count):
            mem.append(spec.memory_gb)
    return mem


# --------------------------------------------------------------------------
# Scheduler plugin
# --------------------------------------------------------------------------


class OccupancyAwareScheduler(Scheduler):
    """Schedule respecting existing cluster occupancy + fractional resource claims.

    Reads ``request.status`` (a :class:`ClusterStatus` snapshot) to subtract
    already-committed slots/VRAM from each host's capacity before packing.
    Honors ``request.resources`` (:class:`ResourceRequest`) for fractional
    GPU sharing: a rank with ``util_fraction=0.3`` consumes 30% of one
    GPU, enabling multiple ranks to share a GPU when their combined
    claims fit.

    Falls back to greedy whole-GPU behavior when ``status`` is ``None``
    and ``resources`` is ``None`` (or non-fractional).

    Honors an explicit :class:`RecipeLayout` verbatim (same semantics as
    the greedy scheduler).

    Raises:
        :class:`InfeasibleScheduleError`: When packing fails (no GPU has
            enough remaining util/memory budget).
        :class:`LayoutConflictError`: When the cluster spans multiple
            accelerator vendors without an explicit layout.
    """

    scheduler_name = "occupancy-aware"

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
        # but accepts the request (see test_greedy_accepts_whole_gpu_resource_request).
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
        """
        total_ranks = request.parallelism.world_size()
        if total_ranks <= 0:
            return RankAssignment(by_rank=(), hosts_used=())

        resources = request.resources
        per_rank_util = resources.util_fraction if resources is not None else 1.0
        per_rank_mem = resources.memory_gb if resources is not None else None
        is_fractional = resources is not None and resources.is_fractional()

        status = request.status

        by_rank: list[RankSlot] = []
        hosts_used: list[str] = []
        placed_vendors: set[str] = set()
        remaining = total_ranks

        for host in request.hosts:
            if remaining <= 0:
                break

            hw = _hw_for(host, request.host_hardware)
            if hw.total_gpus <= 0:
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
                continue

            host_placed_any = False
            for _ in range(remaining):
                gpu_idx = self._select_gpu(
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

        if remaining > 0:
            placed = total_ranks - remaining
            raise InfeasibleScheduleError(
                "Cluster cannot satisfy %d ranks under current occupancy: "
                "only placed %d of %d ranks across %d host(s)" % (total_ranks, placed, total_ranks, len(request.hosts))
            )

        return RankAssignment(by_rank=tuple(by_rank), hosts_used=tuple(hosts_used))

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

    @staticmethod
    def _select_gpu(
        *,
        num_gpus: int,
        util_remaining: list[float],
        mem_remaining: list[float | None],
        gpu_eligible: list[bool],
        per_rank_util: float,
        per_rank_mem: float | None,
    ) -> int | None:
        """Pick the local-GPU index that should host the next rank.

        Strategy:

        - Whole-GPU mode (``per_rank_util == 1.0`` and memory unset):
          return the first eligible GPU, mirroring greedy.
        - Fractional mode: prefer the GPU with the **most** remaining
          util budget that still fits the claim (best-fit-decreasing
          flavour).  Ties broken by remaining memory, then by index.

        Returns ``None`` when no GPU can host the rank.
        """
        # A small epsilon avoids rejecting exact-fit placements due to
        # floating-point rounding (e.g. 0.5 + 0.5 sometimes summing to
        # 0.9999...).
        eps = 1e-9

        best: int | None = None
        best_util_remaining = -1.0
        best_mem_remaining = -1.0

        for i in range(num_gpus):
            if not gpu_eligible[i]:
                continue
            if util_remaining[i] + eps < per_rank_util:
                continue
            if per_rank_mem is not None and mem_remaining[i] is not None:
                if mem_remaining[i] + eps < per_rank_mem:
                    continue

            mem_left = mem_remaining[i] if mem_remaining[i] is not None else float("inf")
            if (
                best is None
                or util_remaining[i] > best_util_remaining
                or (util_remaining[i] == best_util_remaining and mem_left > best_mem_remaining)
            ):
                best = i
                best_util_remaining = util_remaining[i]
                best_mem_remaining = mem_left

        return best

    @staticmethod
    def _diagnostics(request: SchedulingRequest, assignment: RankAssignment, *, mode: str) -> tuple[str, ...]:
        if not assignment.hosts_used:
            return ()
        return (
            "occupancy-aware (%s): packed %d ranks across %d of %d hosts"
            % (mode, assignment.total_ranks, len(assignment.hosts_used), len(request.hosts)),
        )


__all__ = ["OccupancyAwareScheduler"]
