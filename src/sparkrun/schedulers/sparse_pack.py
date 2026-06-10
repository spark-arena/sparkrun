"""SparsePackScheduler — spread *distinct workloads* across hosts/GPUs.

Sparse packing prefers idle (least-loaded) hosts and GPUs so independent
workloads avoid each other when possible.  This is the default sparkrun
scheduler.

**Granularity of "sparse".**  The spread guarantee is *between* workloads,
not *within* one.  A single workload's ranks are still packed onto as few
hosts as possible (each chosen host is filled to capacity before the loop
advances) — this is deliberate: ranks of one job want to be co-located so
tensor-parallel traffic rides the high-bandwidth intra-node link
(NVLink-class) rather than the slower inter-node fabric.  What "sparse"
controls is *which* hosts a new workload lands on: the least-loaded ones,
so it sits as far as possible from already-running jobs.  (On DGX Spark —
1 GPU per host — intra- vs inter-host is the only axis, so a multi-rank
job simply takes the N least-loaded hosts.)

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
            # Tie-break on memory only when util budgets are equal *within eps*:
            # after repeated subtraction two "equal" budgets can differ by a ULP,
            # which would otherwise skip the intended memory tie-break.
            util_tie = abs(util_remaining[i] - best_util_remaining) <= eps
            if best is None or util_remaining[i] > best_util_remaining or (util_tie and mem_left > best_mem_remaining):
                best = i
                best_util_remaining = util_remaining[i]
                best_mem_remaining = mem_left

        return best


__all__ = ["SparsePackScheduler"]
