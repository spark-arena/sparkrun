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
from typing import ClassVar

from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, with_gpu_allocations
from sparkrun.core.allocations import EXCLUSIVE_GPU_THRESHOLD
from sparkrun.core.scheduler import (
    InfeasibleScheduleError,
    CapacityRejection,
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
from sparkrun.schedulers.greedy import _host_vendor, _hw_for, pack


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

    supports_fractional = True

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
        # 1. Honor explicit indices after checking occupancy and shared budgets.
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
            if request.single_host and len(assignment.hosts_used) > 1:
                raise InfeasibleScheduleError("Solo allocation requires every layout rank on one host")
            assignment = self._validate_layout(request, assignment)
            return SchedulingResult(
                assignment=assignment, scheduler_name=self.scheduler_name, diagnostics=("layout validated against GPU allocations",)
            )

        if request.single_host and len(request.hosts) > 1:
            from dataclasses import replace

            reasons = []
            for host in self._sort_hosts(request.hosts, self._compute_host_load_scores(request)):
                try:
                    return self.schedule(replace(request, hosts=(host,), single_host=False))
                except InfeasibleScheduleError as error:
                    reasons.extend(error.rejections)
                except SchedulingError:
                    if request.layout is not None:
                        raise
            raise InfeasibleScheduleError("No single host can satisfy the requested GPU allocation", rejections=tuple(reasons))

        resources = request.resources
        is_fractional = resources is not None and resources.is_fractional()
        has_status = request.status is not None and bool(request.status.hosts or request.status.observation_errors)

        # Without a requested occupancy snapshot, exclusive allocation uses
        # physical capacity. Model/KV estimates remain a runtime-fit report.
        if not is_fractional and not has_status:
            per_rank_memory_gb = None  # Exclusive ownership reserves the GPU, not an estimated footprint.
            try:
                assignment = pack(
                    request.parallelism,
                    list(request.hosts),
                    host_hardware=request.host_hardware,
                    layout=request.layout,
                    per_rank_memory_gb=per_rank_memory_gb,
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

    @staticmethod
    def _memory_claims(hw, resources):
        if resources is None or not resources.is_fractional():
            return [None] * hw.total_gpus
        return [
            resources.memory_gb
            if resources.memory_gb is not None
            else spec.memory_gb * resources.util_fraction
            if spec.memory_gb is not None
            else None
            for spec in hw.accelerators
            for _ in range(spec.count)
        ]

    @staticmethod
    def _rejections(host, hw, eligible, util, memory, fraction, claims, *, own=frozenset()):
        """Why each GPU on *host* refused the next rank. *own* GPUs hold a rank of
        this request: they are in use, not blocked, so they are not reported."""
        specs = [a for a in hw.accelerators for _ in range(a.count)]
        reasons = []
        for i, free in enumerate(eligible):
            if i in own:
                continue
            if not free:
                reason = "exclusive_owner" if fraction < EXCLUSIVE_GPU_THRESHOLD else "occupied"
                detail = "GPU has an existing allocation; exclusive ownership prevents sharing"
            elif util[i] + 1e-9 < fraction:
                reason, detail = "compute_budget", "Requested fraction %.2f; unreserved fraction %.2f" % (fraction, util[i])
            elif fraction < EXCLUSIVE_GPU_THRESHOLD and (memory[i] is None or claims[i] is None):
                reason, detail = "memory_unknown", "Shared allocation requires known memory capacity and reservation"
            elif claims[i] is not None and memory[i] is not None and claims[i] > memory[i] + 1e-9:
                spec = specs[i]
                cap = spec.max_gpu_memory_utilization if spec.max_gpu_memory_utilization is not None else 1.0
                reason = "memory_budget"
                detail = "Requires %.2f GiB; unreserved %.2f GiB (%.2f GiB x %.0f%%, %s)" % (
                    claims[i],
                    memory[i],
                    spec.memory_gb,
                    cap * 100,
                    getattr(spec, "memory_limit_source", None) or "hardware capacity",
                )
            else:
                continue
            reasons.append(CapacityRejection(host, i, reason, detail, claims[i], memory[i]))
        return reasons

    def _validate_layout(self, request, assignment):
        """Explicit rank placement still must respect physical GPU ownership."""
        from dataclasses import replace

        resources = request.resources
        fractional = resources is not None and resources.is_fractional()
        fraction = resources.util_fraction if resources is not None and fractional else 1.0
        budgets = {}
        held: dict[str, set[int]] = {}  # GPUs already holding a rank of this layout
        slots = []
        errors = request.status.observation_errors if request.status is not None else {}
        for slot in assignment.by_rank:
            host, index = slot.host, slot.local_gpu
            hw = _hw_for(host, request.host_hardware)
            if host in errors or not 0 <= index < hw.total_gpus:
                reason = "observation_failed" if host in errors else "invalid_gpu"
                detail = errors.get(host, "Layout references a GPU absent from hardware inventory")
                raise InfeasibleScheduleError(detail, rejections=(CapacityRejection(host, index, reason, detail),))
            if host not in budgets:
                occ = request.status.for_host(host) if request.status is not None else None
                if occ is not None:
                    occ = with_gpu_allocations(occ, hw.total_gpus)
                if self._is_ambiguous_host_level(occ, hw.total_gpus, fractional):
                    raise InfeasibleScheduleError(
                        "Layout cannot establish which GPUs are free",
                        rejections=(CapacityRejection(host, index, "unknown_gpu_indices", "Per-GPU occupancy is required"),),
                    )
                budgets[host] = self._build_budgets(
                    num_gpus=hw.total_gpus, gpu_mem_specs=hw.usable_gpu_memory_slots(), host_occ=occ, is_fractional=fractional
                )
            if index in held.setdefault(host, set()):
                # One NCCL world refuses two ranks on a device, so no layout may ask for it.
                detail = "Layout places two ranks of this launch on one GPU; each rank needs its own"
                raise InfeasibleScheduleError(
                    "Explicit layout conflicts with GPU allocation", rejections=(CapacityRejection(host, index, "same_request", detail),)
                )
            held[host].add(index)
            util, memory, eligible = budgets[host]
            claims = self._memory_claims(hw, resources)
            rejection = tuple(r for r in self._rejections(host, hw, eligible, util, memory, fraction, claims) if r.gpu_index == index)
            if rejection:
                raise InfeasibleScheduleError("Explicit layout conflicts with GPU allocation", rejections=rejection)
            util[index] -= fraction
            if memory[index] is not None and claims[index] is not None:
                memory[index] -= claims[index]
            if not fractional:
                eligible[index] = False
            slots.append(replace(slot, util_fraction=fraction, memory_gb=claims[index]))
        return replace(assignment, by_rank=tuple(slots))

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
        is delegated to :meth:`_select_gpu_index`.  Each host is filled to
        capacity (every rank that fits) before the loop advances, so host
        ordering is decided once up front from
        :meth:`_compute_host_load_scores` (which already folds
        :attr:`HEAD_NODE_OVERHEAD` for the heads of *existing* workloads).

        Note the fill-to-capacity inner loop means the sparse/dense policy
        is applied at *host-selection* granularity: a single workload's
        ranks are packed onto as few hosts as possible (co-located for
        intra-node tensor-parallel bandwidth), while the *choice* of which
        hosts — least-loaded first (sparse) or most-loaded first (dense) —
        is what spreads or stacks distinct workloads relative to each
        other.  See the subclass docstrings.
        """
        total_ranks = request.parallelism.world_size()
        if total_ranks <= 0:
            return RankAssignment(by_rank=(), hosts_used=())

        resources = request.resources
        is_fractional = resources is not None and resources.is_fractional()
        per_rank_util = resources.util_fraction if resources is not None and is_fractional else 1.0
        per_rank_mem = resources.memory_gb if resources is not None and is_fractional else None

        status = request.status
        unavailable = status.observation_errors if status is not None else {}

        # Compute per-host load scores once, then ask the subclass to order them.
        scores = self._compute_host_load_scores(request)
        sorted_hosts: list[str] = self._sort_hosts(request.hosts, scores)

        by_rank: list[RankSlot] = []
        hosts_used: list[str] = []
        placed_vendors: set[str] = set()
        ambiguous_hosts: list[str] = []
        rejections: list[CapacityRejection] = []
        remaining = total_ranks

        i = 0
        while i < len(sorted_hosts) and remaining > 0:
            host = sorted_hosts[i]
            if host in unavailable:
                rejections.append(CapacityRejection(host, None, "observation_failed", unavailable[host]))
                i += 1
                continue

            hw = _hw_for(host, request.host_hardware)
            if hw.total_gpus <= 0:
                rejections.append(CapacityRejection(host, None, "no_accelerators", "No accelerator slots"))
                i += 1
                continue

            vendor = _host_vendor(hw)
            if vendor is None:
                raise LayoutConflictError(
                    "Host '%s' advertises multiple accelerator vendors (%s); "
                    "recipe.layout.placements must specify which ranks land on which accelerator" % (host, sorted(hw.vendors))
                )

            # Build per-GPU (util_remaining, memory_remaining) budgets.
            gpu_mem_specs = hw.usable_gpu_memory_slots()
            num_gpus = len(gpu_mem_specs)
            host_occ = status.for_host(host) if status is not None else None
            if host_occ is not None:
                host_occ = with_gpu_allocations(host_occ, num_gpus)
            claims = self._memory_claims(hw, resources)

            # Whole-GPU placement on a multi-GPU host that reports only a
            # host-level slot count (no per-GPU detail) and is *partially* busy is
            # ambiguous: we know how many GPUs are busy but not which indices, so
            # any local_gpu we emit could collide with an occupied GPU.  Skip the
            # host (other hosts may serve) and record it so a total failure can
            # explain why — rather than guessing or aborting a feasible schedule.
            if self._is_ambiguous_host_level(host_occ, num_gpus, is_fractional):
                ambiguous_hosts.append(host)
                rejections.append(CapacityRejection(host, None, "unknown_gpu_indices", "Occupied GPU indices are unknown"))
                i += 1
                continue

            util_remaining, mem_remaining, gpu_eligible = self._build_budgets(
                num_gpus=num_gpus,
                gpu_mem_specs=gpu_mem_specs,
                host_occ=host_occ,
                is_fractional=is_fractional,
            )

            # Skip hosts that ended up with no eligible GPUs (whole-GPU
            # placement on a fully-occupied host).
            if not any(gpu_eligible):
                rejections.extend(self._rejections(host, hw, gpu_eligible, util_remaining, mem_remaining, per_rank_util, claims))
                i += 1
                continue

            host_placed_any = False
            own: set[int] = set()  # GPUs on this host already holding a rank of this request
            for _ in range(remaining):
                gpu_idx = self._select_gpu_index(
                    num_gpus=num_gpus,
                    util_remaining=util_remaining,
                    mem_remaining=mem_remaining,
                    gpu_eligible=[
                        eligible
                        and (not is_fractional or (claim is not None and remaining_memory is not None and claim <= remaining_memory + 1e-9))
                        for eligible, claim, remaining_memory in zip(gpu_eligible, claims, mem_remaining, strict=True)
                    ],
                    per_rank_util=per_rank_util,
                    per_rank_mem=per_rank_mem,
                )
                if gpu_idx is None:
                    rejections.extend(
                        self._rejections(host, hw, gpu_eligible, util_remaining, mem_remaining, per_rank_util, claims, own=own)
                    )
                    break

                util_remaining[gpu_idx] -= per_rank_util
                memory = mem_remaining[gpu_idx]
                reserved_memory = claims[gpu_idx]
                if reserved_memory is not None and memory is not None:
                    mem_remaining[gpu_idx] = memory - reserved_memory
                # One rank per device, shared or not: ranks of one request form
                # one NCCL world, which refuses two ranks on a GPU. A shared GPU
                # still takes other requests' ranks.
                gpu_eligible[gpu_idx] = False
                own.add(gpu_idx)

                by_rank.append(
                    RankSlot(
                        host=host,
                        local_gpu=gpu_idx,
                        util_fraction=per_rank_util,
                        memory_gb=reserved_memory,
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

            i += 1

        if remaining > 0:
            # If the only reason we couldn't place was ambiguous host-level
            # occupancy, surface that as an actionable layout error rather than a
            # generic "no capacity" — the operator can fix it with per-GPU status
            # from the executor's durable allocation records.
            if ambiguous_hosts:
                raise LayoutConflictError(
                    "Cluster cannot place %d ranks: host(s) %s report only a host-level "
                    "busy count on multi-GPU hardware, so the scheduler cannot tell which "
                    "GPU indices are free. Refresh per-GPU occupancy or stop/relaunch "
                    "legacy workloads to record their GPU assignments." % (total_ranks, sorted(ambiguous_hosts))
                )
            placed = total_ranks - remaining
            raise InfeasibleScheduleError(
                "Cluster cannot satisfy %d ranks: only placed %d of %d ranks across %d host(s)"
                % (total_ranks, placed, total_ranks, len(request.hosts)),
                rejections=tuple(rejections),
            )

        return RankAssignment(by_rank=tuple(by_rank), hosts_used=tuple(hosts_used))

    @staticmethod
    def _is_ambiguous_host_level(host_occ: HostOccupancy | None, num_gpus: int, is_fractional: bool) -> bool:
        """``True`` when placement cannot identify safe GPU indices.

        Holds for a multi-GPU host that reports only a host-level slot count
        (no per-GPU :attr:`HostOccupancy.gpus` detail) and is *partially* busy
        (``0 < used_slots < num_gpus``).  Single-GPU hosts, fully-free / fully-busy
        hosts are unambiguous; shared placement also needs known ownership.
        """
        if host_occ is None or host_occ.gpus or num_gpus <= 1:
            return False
        return 0 < host_occ.used_slots < num_gpus

    # ----------------------------------------------------------------------
    # Host load scoring + head identification
    # ----------------------------------------------------------------------

    def _compute_host_load_scores(self, request: SchedulingRequest) -> dict[str, float]:
        """Return per-host effective load scores in ``[0, ~N]``.

        All branches return an *average per-GPU* load in ``[0, ~1]`` so hosts
        with and without per-GPU detail are comparable on the same scale (the
        score feeds :meth:`_sort_hosts`).  For each host in
        :attr:`SchedulingRequest.hosts`:

        - When per-GPU detail is available,
          ``score = mean(used_util_fraction)`` across the host's GPUs.
        - When only host-level :attr:`HostOccupancy.used_slots` is
          available, ``score = used_slots / total_slots`` (the same 0-1
          average-per-GPU load).
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
                    # Average per-GPU utilization (not the sum) so a host with
                    # per-GPU detail is on the same 0-1 scale as the host-level
                    # branch below — otherwise a 2-GPU host scoring sum=2.0 would
                    # be ordered against a host-level host scoring 0.5 and the
                    # sparse/dense host ordering would be corrupted on multi-GPU
                    # hosts.  Divide by the host's *total* GPU count (not the
                    # number of GPUs reported — occupancy may list only the busy
                    # ones) so the result is true average load.
                    total_gpus = _hw_for(host, request.host_hardware).total_gpus or len(occ.gpus)
                    raw = sum(g.used_util_fraction for g in occ.gpus) / total_gpus if total_gpus else 0.0
                    # Clamp to [0, 1] so a stale hardware map (total_gpus smaller
                    # than the reported busy GPUs) or an out-of-range util reading
                    # can't push the score off the 0-1 scale the host-level branch
                    # uses, which would re-corrupt cross-host ordering.
                    base = min(1.0, max(0.0, raw))
                else:
                    total = occ.total_slots
                    if total > 0:
                        base = occ.used_slots / total
                    elif occ.used_slots > 0:
                        # No total info but something is used — treat as
                        # whole-GPU-busy (already a per-GPU-equivalent fraction).
                        base = 1.0
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
        already occupied" only when the indices are unambiguous. Legacy
        host-only occupancy never authorizes a shared allocation.
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
                    legacy_owner = any(
                        not any(c.allocations for c in w.containers)
                        and (w.util_fraction is None or w.util_fraction >= EXCLUSIVE_GPU_THRESHOLD)
                        for w in gpu.workloads
                    )
                    exclusive = (
                        gpu.exclusive
                        or legacy_owner
                        or (
                            len(gpu.workloads) <= 1
                            and gpu.used_util_fraction >= EXCLUSIVE_GPU_THRESHOLD
                            and not any(c.allocations for w in gpu.workloads for c in w.containers)
                        )
                    )
                    if exclusive:
                        util_remaining[gpu.gpu_index] = 0.0
                        eligible[gpu.gpu_index] = False
                    if not is_fractional:
                        # Whole-GPU mode: any non-trivial occupancy excludes the GPU.
                        if gpu.workloads or gpu.used_util_fraction > 0 or gpu.used_memory_gb > 0:
                            eligible[gpu.gpu_index] = False
            return util_remaining, mem_remaining, eligible

        # Host-level fallback path.
        used_slots = host_occ.used_slots
        if not is_fractional:
            # Whole-GPU placement on a host with only a host-level slot count:
            # the leading ``used_slots`` indices are marked busy.  This index
            # assumption is exact only when unambiguous (single GPU, all-free, or
            # all-busy); the ambiguous partial-multi-GPU case is screened out by
            # :meth:`_is_ambiguous_host_level` *before* this method is called, so
            # we never emit a guessed local_gpu for it.
            for i in range(min(used_slots, num_gpus)):
                eligible[i] = False
        elif used_slots > 0:
            # A legacy workload has no durable permission to share. Its indices
            # are known only for one GPU or a fully occupied host.
            for i in range(num_gpus):
                eligible[i] = False
                util_remaining[i] = 0.0

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
