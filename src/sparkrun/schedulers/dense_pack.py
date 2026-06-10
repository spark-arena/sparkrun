"""DensePackScheduler — bin-pack workloads onto already-loaded targets.

Dense packing prefers the most-loaded eligible host/GPU so workloads
share resources tightly, minimizing host fanout for capacity utilisation.

See :mod:`sparkrun.schedulers._occupancy_base` for the shared packing
loop; this module only specifies the ordering policy.
"""

from __future__ import annotations

from sparkrun.schedulers._occupancy_base import _OccupancyAwareBase


class DensePackScheduler(_OccupancyAwareBase):
    """Bin-pack workloads — prefer the most-loaded eligible target.

    Host level: visit hosts in **descending** order of effective load
    (busiest hosts first).  Ties preserve input order via Python's
    stable sort.

    GPU level: pick the GPU with the **least** remaining util budget
    that still fits the rank's claim (classical best-fit).  Ties broken
    by smaller remaining memory (tighter pack), then by GPU index.

    Honors :attr:`SchedulingRequest.layout` verbatim and falls back to
    the greedy whole-GPU path when neither :attr:`status` nor a
    fractional :attr:`resources` is set, mirroring the base class.
    """

    scheduler_name = "occupancy-dense"

    def _sort_hosts(self, hosts, scores):
        # Stable descending sort; ties keep input order (Python sort is stable).
        return sorted(hosts, key=lambda h: -scores.get(h, 0.0))

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
        # floating-point rounding.
        eps = 1e-9

        best: int | None = None
        best_util_remaining = float("inf")
        best_mem_remaining = float("inf")

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
            if best is None or util_remaining[i] < best_util_remaining or (util_tie and mem_left < best_mem_remaining):
                best = i
                best_util_remaining = util_remaining[i]
                best_mem_remaining = mem_left

        return best


__all__ = ["DensePackScheduler"]
