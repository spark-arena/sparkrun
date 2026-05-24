"""SparsePackScheduler — spread workloads across hosts/GPUs.

Sparse packing prefers idle (least-loaded) hosts and GPUs so independent
workloads avoid each other when possible.  This is the default sparkrun
scheduler.

See :mod:`sparkrun.schedulers._occupancy_base` for the shared packing
loop; this module only specifies the ordering policy.
"""

from __future__ import annotations

from sparkrun.schedulers._occupancy_base import _OccupancyAwareBase


class SparsePackScheduler(_OccupancyAwareBase):
    """Spread workloads — prefer the least-loaded eligible target.

    Host level: visit hosts in **ascending** order of effective load
    (idle hosts first).  Ties preserve input order thanks to Python's
    stable sort.

    GPU level: pick the GPU with the **most** remaining util budget that
    still fits the rank's claim.  Ties broken by remaining memory
    (more first), then by GPU index (lower first).

    Honors :attr:`SchedulingRequest.layout` verbatim and falls back to
    the greedy whole-GPU path when neither :attr:`status` nor a
    fractional :attr:`resources` is set, mirroring the base class.
    """

    scheduler_name = "occupancy-sparse"

    def _sort_hosts(self, hosts, scores):
        # Stable ascending sort; ties keep input order (Python sort is stable).
        return sorted(hosts, key=lambda h: scores.get(h, 0.0))

    def _select_gpu_index(
        self,
        *,
        num_gpus,
        util_remaining,
        mem_remaining,
        gpu_eligible,
        per_rank_util,
        per_rank_mem,
    ):
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


__all__ = ["SparsePackScheduler"]
