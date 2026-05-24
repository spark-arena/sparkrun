"""Tests for :class:`sparkrun.schedulers.sparse_pack.SparsePackScheduler`.

Covers:

- Scheduler metadata + SAF registration.
- Fallback path: behaves like :class:`GreedyScheduler` when neither
  ``status`` nor a fractional ``resources`` is set.
- Whole-GPU placement with cluster occupancy.
- Fractional placement honoring ``util_fraction``.
- Memory accounting via ``memory_gb``.
- Combined occupancy + fractional packing.
- Explicit :class:`~sparkrun.core.layout.RecipeLayout` honored verbatim.
- Heterogeneous-vendor :class:`LayoutConflictError`.
- Empty hosts / zero parallelism edge cases.
- End-to-end via :func:`sparkrun.api.schedule`.
- ``feasibility()`` predicate matches schedule.
- **Sparse spread**: free hosts preferred over loaded ones.
- **Head-node overhead**: head of existing workload gets +5% effective load,
  so the sparse scheduler avoids it when equally-loaded alternatives exist.
"""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_status import (
    ClusterStatus,
    GpuOccupancy,
    HostOccupancy,
    RunningWorkload,
)
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.layout import Placement, RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import (
    InfeasibleScheduleError,
    LayoutConflictError,
    ResourceRequest,
    SchedulingRequest,
    SchedulingResult,
    get_scheduler,
    list_schedulers,
)
from sparkrun.schedulers.greedy import GreedyScheduler
from sparkrun.schedulers.sparse_pack import SparsePackScheduler


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _multi_gpu_host_hw(count: int, memory_gb: float = 80.0) -> HostHardware:
    """A single-host hardware spec with ``count`` identical GPUs."""
    return HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", count=count, memory_gb=memory_gb)])


# --------------------------------------------------------------------------
# Scheduler metadata + SAF registration
# --------------------------------------------------------------------------


def test_scheduler_name_is_occupancy_sparse():
    assert SparsePackScheduler.scheduler_name == "occupancy-sparse"
    assert SparsePackScheduler().scheduler_name == "occupancy-sparse"


def test_sparse_registered_via_bootstrap():
    """init_sparkrun discovers and registers SparsePackScheduler."""
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    plugin = get_scheduler("occupancy-sparse", v=v)
    assert plugin.scheduler_name == "occupancy-sparse"
    assert isinstance(plugin, SparsePackScheduler)


def test_list_schedulers_includes_occupancy_sparse():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    names = list_schedulers(v=v)
    assert "occupancy-sparse" in names
    assert "greedy" in names


def test_list_schedulers_includes_both_new_names():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    names = set(list_schedulers(v=v))
    # Subset (not equality) — Python's Scheduler.__subclasses__() leaks
    # test-defined subclasses (e.g. _NoopScheduler) across modules.
    assert {"greedy", "occupancy-dense", "occupancy-sparse"} <= names


# --------------------------------------------------------------------------
# Fallback path — no status + no resources → matches greedy
# --------------------------------------------------------------------------


def test_fallback_matches_greedy_on_idle_cluster():
    """No status + no resources → identical assignment to GreedyScheduler."""
    parallelism = ParallelismConfig(tensor_parallel=3)
    hosts = ("spark-01", "spark-02", "spark-03")

    greedy_req = SchedulingRequest(parallelism=parallelism, hosts=hosts)
    sparse_req = SchedulingRequest(parallelism=parallelism, hosts=hosts)

    greedy_result = GreedyScheduler().schedule(greedy_req)
    sparse_result = SparsePackScheduler().schedule(sparse_req)

    assert sparse_result.assignment == greedy_result.assignment
    assert sparse_result.scheduler_name == "occupancy-sparse"


def test_fallback_matches_greedy_multi_gpu():
    """Multi-GPU single-host fallback matches greedy behaviour."""
    hw = {"big": _multi_gpu_host_hw(count=4, memory_gb=141.0)}
    parallelism = ParallelismConfig(tensor_parallel=4)

    greedy = GreedyScheduler().schedule(SchedulingRequest(parallelism=parallelism, hosts=("big", "h2"), host_hardware=hw))
    sparse = SparsePackScheduler().schedule(SchedulingRequest(parallelism=parallelism, hosts=("big", "h2"), host_hardware=hw))
    assert sparse.assignment == greedy.assignment


def test_fallback_whole_gpu_resource_request_accepted():
    """ResourceRequest with util_fraction=1.0 is whole-GPU → flows through fallback."""
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        resources=ResourceRequest(memory_gb=80.0, util_fraction=1.0),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1", "h2")


def test_fallback_diagnostics_indicate_mode():
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
    )
    result = sched.schedule(req)
    assert isinstance(result, SchedulingResult)
    assert result.diagnostics
    assert "fallback" in result.diagnostics[0]


# --------------------------------------------------------------------------
# Whole-GPU placement with occupancy
# --------------------------------------------------------------------------


def test_whole_gpu_skips_fully_occupied_single_gpu_host():
    """Host with used_slots=1, free_slots=0 (1-GPU host) is skipped."""
    sched = SparsePackScheduler()
    status = ClusterStatus(
        hosts=(
            HostOccupancy(host="h1", used_slots=1, free_slots=0),
            HostOccupancy(host="h2", used_slots=0, free_slots=1),
            HostOccupancy(host="h3", used_slots=0, free_slots=1),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2", "h3"),
        status=status,
    )
    result = sched.schedule(req)
    # h1 is fully occupied; sparse picks h2 and h3 (both idle).
    assert set(result.assignment.hosts_used) == {"h2", "h3"}


def test_whole_gpu_infeasible_when_all_hosts_full():
    sched = SparsePackScheduler()
    status = ClusterStatus(
        hosts=(
            HostOccupancy(host="h1", used_slots=1, free_slots=0),
            HostOccupancy(host="h2", used_slots=1, free_slots=0),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        status=status,
    )
    with pytest.raises(InfeasibleScheduleError) as exc_info:
        sched.schedule(req)
    assert "2" in str(exc_info.value)


def test_whole_gpu_idle_status_behaves_like_greedy():
    """Status snapshot reporting zero usage everywhere → greedy-like placement."""
    sched = SparsePackScheduler()
    status = ClusterStatus(
        hosts=(
            HostOccupancy(host="h1", used_slots=0, free_slots=1),
            HostOccupancy(host="h2", used_slots=0, free_slots=1),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        status=status,
    )
    result = sched.schedule(req)
    assert set(result.assignment.hosts_used) == {"h1", "h2"}


def test_whole_gpu_per_gpu_occupancy_excludes_busy_gpu():
    """Per-GPU detail with non-trivial usage excludes that GPU from whole-GPU placement."""
    sched = SparsePackScheduler()
    hw = {"big": _multi_gpu_host_hw(count=4, memory_gb=80.0)}
    # GPU 0 is busy on the host; whole-GPU ranks should skip it.
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="big",
                used_slots=1,
                free_slots=3,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.7, used_memory_gb=40.0),),
            ),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=3),
        hosts=("big",),
        host_hardware=hw,
        status=status,
    )
    result = sched.schedule(req)
    placed_gpus = {slot.local_gpu for slot in result.assignment.by_rank}
    assert 0 not in placed_gpus
    assert placed_gpus == {1, 2, 3}


# --------------------------------------------------------------------------
# Fractional placement
# --------------------------------------------------------------------------


def test_fractional_two_ranks_share_single_gpu():
    """tp=2, util_fraction=0.5 → both ranks on local_gpu=0."""
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.5),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1",)
    assert len(result.assignment.by_rank) == 2
    for slot in result.assignment.by_rank:
        assert slot.host == "h1"
        assert slot.local_gpu == 0
        assert slot.util_fraction == 0.5


def test_fractional_infeasible_when_combined_util_exceeds_one():
    """tp=2, util_fraction=0.6 → 0.6+0.6=1.2 > 1.0 on single-GPU host."""
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.6),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)


def test_fractional_packs_two_per_gpu_across_multi_gpu_host():
    """4-GPU host, tp=8, util_fraction=0.5 → 2 ranks per GPU × 4 GPUs."""
    sched = SparsePackScheduler()
    hw = {"big": _multi_gpu_host_hw(count=4, memory_gb=80.0)}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=8),
        hosts=("big",),
        host_hardware=hw,
        resources=ResourceRequest(util_fraction=0.5),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("big",)
    assert len(result.assignment.by_rank) == 8

    counts: dict[int, int] = {}
    for slot in result.assignment.by_rank:
        assert slot.util_fraction == 0.5
        counts[slot.local_gpu] = counts.get(slot.local_gpu, 0) + 1
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2}


def test_fractional_diagnostics_mode_label():
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.5),
    )
    result = sched.schedule(req)
    assert result.diagnostics
    assert "fractional" in result.diagnostics[0]


# --------------------------------------------------------------------------
# Memory accounting
# --------------------------------------------------------------------------


def test_memory_two_ranks_fit_in_one_gpu():
    """80GB GPU, tp=2 on one host with memory_gb=40 each → must fit on one GPU."""
    sched = SparsePackScheduler()
    hw = {"big": _multi_gpu_host_hw(count=1, memory_gb=80.0)}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("big",),
        host_hardware=hw,
        resources=ResourceRequest(memory_gb=40.0, util_fraction=0.5),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("big",)
    for slot in result.assignment.by_rank:
        assert slot.local_gpu == 0
        assert slot.memory_gb == 40.0


def test_memory_infeasible_when_combined_memory_exceeds_capacity():
    """80GB GPU, tp=2, memory_gb=50 each → 50+50=100 > 80."""
    sched = SparsePackScheduler()
    hw = {"big": _multi_gpu_host_hw(count=1, memory_gb=80.0)}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("big",),
        host_hardware=hw,
        resources=ResourceRequest(memory_gb=50.0, util_fraction=0.5),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)


# --------------------------------------------------------------------------
# Combined occupancy + fractional
# --------------------------------------------------------------------------


def test_combined_occupancy_and_fractional_fits():
    """GPU 0 has used_util_fraction=0.3; new rank with util_fraction=0.5 still fits."""
    sched = SparsePackScheduler()
    hw = {"h1": _multi_gpu_host_hw(count=1, memory_gb=80.0)}
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=1,
                free_slots=0,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.3, used_memory_gb=20.0),),
            ),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=status,
        resources=ResourceRequest(util_fraction=0.5),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1",)
    assert result.assignment.by_rank[0].local_gpu == 0
    assert result.assignment.by_rank[0].util_fraction == 0.5


def test_combined_occupancy_and_fractional_infeasible():
    """GPU 0 has used_util_fraction=0.7; new rank with util_fraction=0.5 → 0.7+0.5=1.2 > 1.0."""
    sched = SparsePackScheduler()
    hw = {"h1": _multi_gpu_host_hw(count=1, memory_gb=80.0)}
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=1,
                free_slots=0,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.7),),
            ),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=status,
        resources=ResourceRequest(util_fraction=0.5),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)


def test_occupancy_diagnostics_mode_label():
    """Status set but resources unset → diagnostics mention occupancy mode."""
    sched = SparsePackScheduler()
    status = ClusterStatus(
        hosts=(HostOccupancy(host="h1", used_slots=0, free_slots=1),),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        status=status,
    )
    result = sched.schedule(req)
    assert result.diagnostics
    assert "occupancy" in result.diagnostics[0]


# --------------------------------------------------------------------------
# Explicit layout honored
# --------------------------------------------------------------------------


def test_explicit_layout_honored_verbatim():
    """An explicit RecipeLayout is passed through unchanged."""
    layout = RecipeLayout(
        placements=[
            Placement(host="h2", ranks=(0,)),
            Placement(host="h1", ranks=(1,)),
        ],
    )
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        layout=layout,
    )
    result = sched.schedule(req)
    assert result.assignment.host_for_rank(0) == "h2"
    assert result.assignment.host_for_rank(1) == "h1"
    assert "layout" in result.diagnostics[0]


def test_explicit_layout_overrides_status_and_resources():
    """When layout is set, status + resources are ignored — layout wins."""
    layout = RecipeLayout(
        placements=[
            Placement(host="h1", ranks=(0,)),
            Placement(host="h2", ranks=(1,)),
        ],
    )
    sched = SparsePackScheduler()
    # Even though h1 looks busy in status, the explicit layout is honored.
    status = ClusterStatus(
        hosts=(
            HostOccupancy(host="h1", used_slots=1, free_slots=0),
            HostOccupancy(host="h2", used_slots=0, free_slots=1),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        layout=layout,
        status=status,
        resources=ResourceRequest(util_fraction=0.5),
    )
    result = sched.schedule(req)
    assert result.assignment.host_for_rank(0) == "h1"
    assert result.assignment.host_for_rank(1) == "h2"


# --------------------------------------------------------------------------
# Heterogeneous vendor → LayoutConflictError
# --------------------------------------------------------------------------


def test_multi_vendor_cluster_raises_layout_conflict():
    """Mixed NVIDIA + AMD without explicit layout → LayoutConflictError."""
    hw = {
        "h1": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", count=1, memory_gb=121.0)]),
        "h2": HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300", count=1, memory_gb=192.0)]),
    }
    sched = SparsePackScheduler()
    # tp=3 with util_fraction=0.5 means 2 ranks fit on h1 (nvidia), then 1
    # spills to h2 (amd) — which triggers the multi-vendor guard.
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=3),
        hosts=("h1", "h2"),
        host_hardware=hw,
        resources=ResourceRequest(util_fraction=0.5),
    )
    with pytest.raises(LayoutConflictError):
        sched.schedule(req)


def test_multi_vendor_single_host_raises_layout_conflict():
    """One host with two different vendor accelerators → LayoutConflictError."""
    hw = {
        "h1": HostHardware(
            accelerators=[
                AcceleratorSpec(vendor="nvidia", model="h100", count=1, memory_gb=80.0),
                AcceleratorSpec(vendor="amd", model="mi300", count=1, memory_gb=192.0),
            ]
        ),
    }
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        resources=ResourceRequest(util_fraction=0.5),
    )
    with pytest.raises(LayoutConflictError):
        sched.schedule(req)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_empty_hosts_with_nonzero_parallelism_raises_infeasible():
    """Empty host list + tp>=1 → cannot place any ranks → InfeasibleScheduleError."""
    sched = SparsePackScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=(),
        resources=ResourceRequest(util_fraction=0.5),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)


def test_zero_ranks_returns_empty_assignment():
    """Parallelism with zero total ranks → empty assignment (no error)."""
    import dataclasses

    sched = SparsePackScheduler()
    parallelism = dataclasses.replace(ParallelismConfig(tensor_parallel=1), total_ranks=0)
    req = SchedulingRequest(
        parallelism=parallelism,
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.5),
    )
    result = sched.schedule(req)
    assert result.assignment.by_rank == ()
    assert result.assignment.hosts_used == ()


def test_status_with_running_workload_per_gpu():
    """GPU with workloads list + used_util>0 is excluded from whole-GPU placement."""
    sched = SparsePackScheduler()
    hw = {"big": _multi_gpu_host_hw(count=2, memory_gb=80.0)}
    workload = RunningWorkload(cluster_id="job-x", util_fraction=0.4, memory_used_gb=30.0)
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="big",
                used_slots=1,
                free_slots=1,
                gpus=(
                    GpuOccupancy(
                        gpu_index=0,
                        used_util_fraction=0.4,
                        used_memory_gb=30.0,
                        workloads=(workload,),
                    ),
                ),
            ),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("big",),
        host_hardware=hw,
        status=status,
    )
    result = sched.schedule(req)
    # GPU 0 has a running workload; whole-GPU placement avoids it.
    assert result.assignment.by_rank[0].local_gpu == 1


# --------------------------------------------------------------------------
# End-to-end via sparkrun.api.schedule
# --------------------------------------------------------------------------


def test_api_schedule_fractional_via_occupancy_sparse():
    """api.schedule(...) with scheduler='occupancy-sparse' succeeds where greedy rejects."""
    from sparkrun.api import schedule as api_schedule

    parallelism = ParallelismConfig(tensor_parallel=2)
    req = SchedulingRequest(
        parallelism=parallelism,
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.5),
    )
    result = api_schedule(req, scheduler="occupancy-sparse")
    assert result.scheduler_name == "occupancy-sparse"
    assert result.assignment.hosts_used == ("h1",)
    assert len(result.assignment.by_rank) == 2
    for slot in result.assignment.by_rank:
        assert slot.local_gpu == 0
        assert slot.util_fraction == 0.5


def test_api_schedule_translates_infeasible_to_insufficient_capacity():
    """api.schedule wraps InfeasibleScheduleError as InsufficientCapacity."""
    from sparkrun.api import InsufficientCapacity
    from sparkrun.api import schedule as api_schedule

    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.6),
    )
    with pytest.raises(InsufficientCapacity):
        api_schedule(req, scheduler="occupancy-sparse")


def test_feasibility_predicate_matches_schedule():
    """Scheduler.feasibility returns True iff schedule succeeds."""
    sched = SparsePackScheduler()

    feasible = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.5),
    )
    assert sched.feasibility(feasible) is True

    infeasible = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1",),
        resources=ResourceRequest(util_fraction=0.6),
    )
    assert sched.feasibility(infeasible) is False


# --------------------------------------------------------------------------
# Sparse-specific: host ordering (spread behaviour)
# --------------------------------------------------------------------------


def test_sparse_prefers_free_hosts_over_loaded():
    """4 hosts; h1+h2 already at util=0.5 (GPU half-full), h3+h4 idle.

    Each new rank needs util_fraction=0.6, which does NOT fit on h1/h2's
    remaining 0.5 budget (0.5 < 0.6), so those hosts are ineligible.
    Sparse must place tp=2 on h3 and h4 (the idle hosts).

    This separates "host ordering preference" from "capacity infeasibility":
    h1/h2 are physically too full to accept the new rank, not merely
    avoided due to load preference.  The ordering test
    (test_sparse_host_ordering_with_room_on_all) covers the preference
    case where all hosts still have room.
    """
    sched = SparsePackScheduler()
    hw = {h: _multi_gpu_host_hw(count=1, memory_gb=80.0) for h in ("h1", "h2", "h3", "h4")}
    workload = RunningWorkload(cluster_id="existing", util_fraction=0.5)
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=0,
                free_slots=1,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.5, workloads=(workload,)),),
            ),
            HostOccupancy(
                host="h2",
                used_slots=0,
                free_slots=1,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.5, workloads=(workload,)),),
            ),
            HostOccupancy(host="h3", used_slots=0, free_slots=1),
            HostOccupancy(host="h4", used_slots=0, free_slots=1),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2", "h3", "h4"),
        host_hardware=hw,
        status=status,
        resources=ResourceRequest(util_fraction=0.6),
    )
    result = sched.schedule(req)
    # h1/h2 have only 0.5 util remaining — not enough for util_fraction=0.6.
    # Sparse lands on h3 and h4 (idle, full 1.0 budget).
    assert set(result.assignment.hosts_used) == {"h3", "h4"}


def test_sparse_host_ordering_with_room_on_all():
    """4 hosts all with room; h1+h2 at util=0.3, h3+h4 idle.

    Each new rank needs util_fraction=0.6.  h1/h2 have 0.7 util remaining
    (fits one rank: 0.7 ≥ 0.6, but 0.7-0.6=0.1 < 0.6 so a second won't fit).
    h3/h4 have 1.0 remaining (fits one rank: 1.0-0.6=0.4 < 0.6 for a second).
    So exactly one rank per GPU everywhere.  Sparse visits ascending load:
    h3 (0.0) → h4 (0.0) → done after tp=2.  h1/h2 never visited.
    """
    sched = SparsePackScheduler()
    hw = {h: _multi_gpu_host_hw(count=1, memory_gb=80.0) for h in ("h1", "h2", "h3", "h4")}
    workload = RunningWorkload(cluster_id="existing", util_fraction=0.3)
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=0,
                free_slots=1,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.3, workloads=(workload,)),),
            ),
            HostOccupancy(
                host="h2",
                used_slots=0,
                free_slots=1,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.3, workloads=(workload,)),),
            ),
            HostOccupancy(host="h3", used_slots=0, free_slots=1),
            HostOccupancy(host="h4", used_slots=0, free_slots=1),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2", "h3", "h4"),
        host_hardware=hw,
        status=status,
        resources=ResourceRequest(util_fraction=0.6),
    )
    result = sched.schedule(req)
    # Sparse visits least-loaded first: h3 (score 0.0) → h4 (0.0) → done.
    # h1/h2 (score 0.3) are never visited because tp=2 is already filled.
    assert set(result.assignment.hosts_used) == {"h3", "h4"}


def test_sparse_head_overhead_avoids_head_host():
    """2 hosts both at load 0.3; h1 is head of an existing workload.

    The head overhead (+0.05) pushes h1's effective score to 0.35, so
    sparse places the single new rank on h2 (score 0.3).
    """
    sched = SparsePackScheduler()
    hw = {h: _multi_gpu_host_hw(count=1, memory_gb=80.0) for h in ("h1", "h2")}
    # h1 is the head: it hosts 1 rank of cluster "alpha".
    workload = RunningWorkload(cluster_id="alpha", util_fraction=0.3, ranks_on_host=1)
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=0,
                free_slots=1,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.3, workloads=(workload,)),),
                workloads=(workload,),
            ),
            HostOccupancy(
                host="h2",
                used_slots=0,
                free_slots=1,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=0.3),),
            ),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1", "h2"),
        host_hardware=hw,
        status=status,
        resources=ResourceRequest(util_fraction=0.3),
    )
    result = sched.schedule(req)
    # h1 effective score = 0.3 + 0.05 = 0.35 > h2's 0.3 → sparse picks h2.
    assert result.assignment.hosts_used == ("h2",)


def test_sparse_full_host_rejected_regardless_of_head_status():
    """Head-node overhead never affects capacity rejection.

    A host that is physically full (util_remaining < per_rank_util) must
    be rejected even if it carries zero head overhead.
    """
    sched = SparsePackScheduler()
    hw = {"h1": _multi_gpu_host_hw(count=1, memory_gb=80.0)}
    # GPU fully occupied.
    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=1,
                free_slots=0,
                gpus=(GpuOccupancy(gpu_index=0, used_util_fraction=1.0),),
            ),
        ),
        executor="docker",
    )
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=status,
        resources=ResourceRequest(util_fraction=0.3),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)
