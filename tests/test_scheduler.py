"""Tests for sparkrun.core.scheduler and sparkrun.schedulers.greedy.

Covers the foundation tier:

- :class:`ClusterStatus` dataclasses and helpers.
- :class:`SchedulingRequest` / :class:`SchedulingResult` shapes.
- :class:`GreedyScheduler` parity with the underlying
  :func:`sparkrun.core.placement.compute_placement` primitive.
- SAF registration via bootstrap (:data:`EXT_SCHEDULER`).
- Error translation: ``InsufficientCapacityError`` →
  ``InfeasibleScheduleError``, ``LayoutRequiredError`` →
  ``LayoutConflictError``.
"""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_status import (
    ClusterStatus,
    GpuOccupancy,
    HostOccupancy,
    RunningWorkload,
    empty_status,
)
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.layout import Placement, RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import (
    EXT_SCHEDULER,
    InfeasibleScheduleError,
    LayoutConflictError,
    RankAssignment,
    RankSlot,
    ResourceRequest,
    Scheduler,
    SchedulingError,
    SchedulingRequest,
    SchedulingResult,
    get_scheduler,
    list_schedulers,
)
from sparkrun.schedulers.greedy import GreedyScheduler


# --------------------------------------------------------------------------
# ClusterStatus dataclasses
# --------------------------------------------------------------------------


def test_cluster_status_empty():
    cs = empty_status(["a", "b", "c"], executor="docker")
    assert cs.executor == "docker"
    assert len(cs.hosts) == 3
    assert all(h.free_slots == 0 for h in cs.hosts)
    assert all(h.used_slots == 0 for h in cs.hosts)
    assert cs.running_cluster_ids() == ()


def test_cluster_status_for_host_lookup():
    occ = HostOccupancy(host="spark-01", used_slots=1, free_slots=0)
    cs = ClusterStatus(hosts=(occ,), executor="docker")
    assert cs.for_host("spark-01") is occ
    assert cs.for_host("missing") is None
    assert cs.free_slots("spark-01") == 0
    assert cs.free_slots("missing") == 0


def test_cluster_status_running_cluster_ids_dedup():
    w1 = RunningWorkload(cluster_id="job-a")
    w2 = RunningWorkload(cluster_id="job-b")
    occ1 = HostOccupancy(host="h1", workloads=(w1, w2), used_slots=2)
    occ2 = HostOccupancy(host="h2", workloads=(w1,), used_slots=1)
    cs = ClusterStatus(hosts=(occ1, occ2))
    assert cs.running_cluster_ids() == ("job-a", "job-b")


def test_host_occupancy_total_slots():
    assert HostOccupancy(host="h", used_slots=3, free_slots=5).total_slots == 8


def test_gpu_occupancy_defaults_empty():
    g = GpuOccupancy(gpu_index=0)
    assert g.gpu_index == 0
    assert g.used_memory_gb == 0.0
    assert g.used_util_fraction == 0.0
    assert g.workloads == ()


def test_host_occupancy_gpus_defaults_empty():
    """Per-GPU detail is opt-in; default HostOccupancy has empty gpus tuple."""
    occ = HostOccupancy(host="h", used_slots=1, free_slots=1)
    assert occ.gpus == ()


def test_host_occupancy_with_per_gpu_detail():
    occ = HostOccupancy(
        host="h",
        used_slots=1,
        free_slots=1,
        gpus=(
            GpuOccupancy(gpu_index=0, used_memory_gb=70.0, used_util_fraction=0.6),
            GpuOccupancy(gpu_index=1),
        ),
    )
    assert len(occ.gpus) == 2
    assert occ.gpus[0].used_memory_gb == 70.0
    assert occ.gpus[1].used_util_fraction == 0.0


def test_running_workload_resource_accounting_optional():
    w = RunningWorkload(cluster_id="job-a")
    assert w.memory_used_gb is None
    assert w.util_fraction is None

    w2 = RunningWorkload(cluster_id="job-b", memory_used_gb=24.0, util_fraction=0.3)
    assert w2.memory_used_gb == 24.0
    assert w2.util_fraction == 0.3


# --------------------------------------------------------------------------
# ResourceRequest + fractional vocabulary
# --------------------------------------------------------------------------


def test_resource_request_defaults_whole_gpu():
    r = ResourceRequest()
    assert r.util_fraction == 1.0
    assert r.memory_gb is None
    assert not r.is_fractional()


def test_resource_request_fractional_predicate():
    assert ResourceRequest(util_fraction=0.3).is_fractional()
    assert ResourceRequest(util_fraction=0.99).is_fractional()
    assert not ResourceRequest(util_fraction=1.0).is_fractional()


def test_rank_slot_defaults_preserve_whole_gpu_equality():
    """Default RankSlot equals a positional RankSlot — the existing assignment
    equality checks across the codebase continue to work after we added the
    new util_fraction / memory_gb fields with defaults."""
    a = RankSlot(host="h", local_gpu=0)
    b = RankSlot(host="h", local_gpu=0, util_fraction=1.0, memory_gb=None)
    assert a == b


def test_rank_slot_can_express_fractional():
    s = RankSlot(host="h", local_gpu=0, util_fraction=0.5, memory_gb=24.0)
    assert s.util_fraction == 0.5
    assert s.memory_gb == 24.0


def test_greedy_rejects_fractional_resource_request():
    """GreedyScheduler fails loudly on fractional claims rather than treating
    them as whole-GPU.  Fractional sharing is the job of a different scheduler."""
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        resources=ResourceRequest(util_fraction=0.5),
    )
    with pytest.raises(SchedulingError, match="fractional"):
        sched.schedule(req)


def test_greedy_accepts_whole_gpu_resource_request():
    """ResourceRequest with util_fraction=1.0 is fine — equivalent to no request."""
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        resources=ResourceRequest(memory_gb=80.0, util_fraction=1.0),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1", "h2")


# --------------------------------------------------------------------------
# GreedyScheduler — parity with compute_placement
# --------------------------------------------------------------------------


def test_greedy_dgx_homogeneous_tp3():
    """Three DGX Spark hosts, tp=3 → first three hosts used in order."""
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=3),
        hosts=("spark-01", "spark-02", "spark-03"),
    )
    result = sched.schedule(req)

    assert isinstance(result, SchedulingResult)
    assert result.scheduler_name == "greedy"
    assert result.assignment.hosts_used == ("spark-01", "spark-02", "spark-03")
    assert result.assignment.by_rank == (
        RankSlot(host="spark-01", local_gpu=0),
        RankSlot(host="spark-02", local_gpu=0),
        RankSlot(host="spark-03", local_gpu=0),
    )


def test_greedy_oversized_host_list_packs_prefix():
    """Greedy uses only as many hosts as needed; leftovers are ignored."""
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2", "h3", "h4"),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1", "h2")


def test_greedy_multi_gpu_host_consumes_single_host():
    """A 4-GPU host with tp=4 fits on one host — the multi-GPU regression case."""
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200", count=4, memory_gb=141.0)])
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=4),
        hosts=("big", "small-1", "small-2", "small-3"),
        host_hardware={"big": hw},
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("big",)
    assert {slot.local_gpu for slot in result.assignment.by_rank} == {0, 1, 2, 3}


def test_greedy_explicit_layout_honored():
    """An explicit RecipeLayout is passed through and honored verbatim."""
    layout = RecipeLayout(
        placements=[
            Placement(host="h2", ranks=(0,)),
            Placement(host="h1", ranks=(1,)),
        ],
    )
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        layout=layout,
    )
    result = sched.schedule(req)
    # First rank lands on h2, not h1 — layout overrides natural greedy order.
    assert result.assignment.host_for_rank(0) == "h2"
    assert result.assignment.host_for_rank(1) == "h1"


def test_greedy_diagnostics_present():
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2", "h3"),
    )
    result = sched.schedule(req)
    assert result.diagnostics  # non-empty
    assert "2" in result.diagnostics[0]  # mentions rank count


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------


def test_greedy_insufficient_capacity_raises_infeasible():
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=5),
        hosts=("h1", "h2"),
    )
    with pytest.raises(InfeasibleScheduleError) as exc_info:
        sched.schedule(req)
    assert "5" in str(exc_info.value)


def test_greedy_multi_vendor_raises_layout_conflict():
    hw = {
        "h1": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", count=1)]),
        "h2": HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300", count=1)]),
    }
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        host_hardware=hw,
    )
    with pytest.raises(LayoutConflictError):
        sched.schedule(req)


def test_infeasible_is_subclass_of_scheduling_error():
    assert issubclass(InfeasibleScheduleError, SchedulingError)
    assert issubclass(LayoutConflictError, SchedulingError)


# --------------------------------------------------------------------------
# Feasibility predicate
# --------------------------------------------------------------------------


def test_feasibility_true_when_schedulable():
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
    )
    assert sched.feasibility(req) is True


def test_feasibility_false_when_infeasible():
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=5),
        hosts=("h1", "h2"),
    )
    assert sched.feasibility(req) is False


# --------------------------------------------------------------------------
# SAF registration via bootstrap
# --------------------------------------------------------------------------


def test_greedy_registered_via_bootstrap():
    """init_sparkrun discovers and registers GreedyScheduler."""
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    plugin = get_scheduler("greedy", v=v)
    assert plugin.scheduler_name == "greedy"


def test_get_scheduler_default_matches_fallback():
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER

    v = init_sparkrun()
    assert get_scheduler(None, v=v).scheduler_name == FALLBACK_DEFAULT_SCHEDULER
    assert get_scheduler("default", v=v).scheduler_name == FALLBACK_DEFAULT_SCHEDULER


def test_list_schedulers_includes_greedy():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    names = list_schedulers(v=v)
    assert "greedy" in names


def test_get_scheduler_unknown_raises():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    with pytest.raises(ValueError) as exc_info:
        get_scheduler("does-not-exist", v=v)
    assert "does-not-exist" in str(exc_info.value)


# --------------------------------------------------------------------------
# Extension-point smoke: a custom scheduler registers and is selectable
# --------------------------------------------------------------------------


class _NoopScheduler(Scheduler):
    """Test-only scheduler that always returns an empty assignment."""

    scheduler_name = "noop-test"

    def schedule(self, request: SchedulingRequest) -> SchedulingResult:
        return SchedulingResult(
            assignment=RankAssignment(by_rank=(), hosts_used=()),
            scheduler_name=self.scheduler_name,
            diagnostics=("noop",),
        )


def test_custom_scheduler_can_be_registered():
    from scitrera_app_framework import register_plugin

    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    register_plugin(_NoopScheduler, v=v)

    plugin = get_scheduler("noop-test", v=v)
    assert plugin.scheduler_name == "noop-test"
    req = SchedulingRequest(parallelism=ParallelismConfig(), hosts=())
    result = plugin.schedule(req)
    assert result.scheduler_name == "noop-test"
    assert result.diagnostics == ("noop",)


def test_extension_point_name_is_canonical():
    sched = GreedyScheduler()
    # The Plugin's extension point matches EXT_SCHEDULER
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    assert sched.extension_point_name(v) == EXT_SCHEDULER


# --- ParallelismConfig.world_size + total_ranks override ---


def test_parallelism_world_size_falls_back_to_total_gpus_when_no_override():
    """``ParallelismConfig.world_size()`` returns ``total_gpus`` when override unset."""
    p = ParallelismConfig(tensor_parallel=2, pipeline_parallel=2)
    assert p.total_gpus == 4
    assert p.world_size() == 4
    assert p.total_ranks is None


def test_parallelism_world_size_honors_total_ranks_override():
    """When ``total_ranks`` is set, ``world_size()`` returns the override."""
    p = ParallelismConfig(tensor_parallel=2, expert_parallel=4, total_ranks=8)
    # The formula would say tp*pp*dp = 2, but the override wins.
    assert p.total_gpus == 2
    assert p.world_size() == 8


def test_total_ranks_overrides_parallelism_total_gpus():
    """``ParallelismConfig.total_ranks`` (set via ``dataclasses.replace``)
    overrides the ``tp*pp*dp`` formula in scheduling."""
    import dataclasses

    sched = GreedyScheduler()
    # Parallelism implies tp*pp*dp = 1 rank, but total_ranks asks for 4.
    parallelism = ParallelismConfig(tensor_parallel=1)
    parallelism = dataclasses.replace(parallelism, total_ranks=4)
    req = SchedulingRequest(
        parallelism=parallelism,
        hosts=("h1", "h2", "h3", "h4"),
    )
    result = sched.schedule(req)
    assert result.assignment.total_ranks == 4
    assert len(result.assignment.hosts_used) == 4


def test_total_ranks_none_falls_back_to_total_gpus():
    """When ``total_ranks`` is unset the scheduler uses ``parallelism.total_gpus``."""
    sched = GreedyScheduler()
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
    )
    result = sched.schedule(req)
    # total_gpus = tp * pp * dp = 2
    assert result.assignment.total_ranks == 2


def test_total_ranks_atlas_mesh_math():
    """Mimic Atlas's tp*ep mesh math via the total_ranks override.

    Atlas's :meth:`Runtime.world_size` returns ``tp * ep``; api.run bakes
    that into the parallelism via ``dataclasses.replace``.  The scheduler
    sees the override and packs 8 ranks, even though the base formula
    would say 2.
    """
    import dataclasses

    sched = GreedyScheduler()
    parallelism = ParallelismConfig(tensor_parallel=2, expert_parallel=4)
    # Atlas's world_size override would return tp * ep = 8.
    parallelism = dataclasses.replace(parallelism, total_ranks=8)
    req = SchedulingRequest(
        parallelism=parallelism,
        hosts=("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"),
    )
    result = sched.schedule(req)
    assert result.assignment.total_ranks == 8
    assert len(result.assignment.hosts_used) == 8
