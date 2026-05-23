"""Tests for placement data types and the greedy scheduler.

Exercises the scheduler interface (:class:`GreedyScheduler`) rather than
calling the legacy :func:`compute_placement` shim directly.  Error
assertions use the scheduler-level vocabulary
(:class:`InfeasibleScheduleError`, :class:`LayoutConflictError`).
"""

from __future__ import annotations

import pytest

from sparkrun.core.hardware import AcceleratorSpec, HostHardware, default_dgx_spark_hardware
from sparkrun.core.layout import Placement, RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import (
    InfeasibleScheduleError,
    LayoutConflictError,
    RankSlot,
    SchedulingError,
    SchedulingRequest,
)
from sparkrun.schedulers.greedy import GreedyScheduler


def _place(
    parallelism: ParallelismConfig,
    hosts,
    *,
    host_hardware=None,
    layout=None,
):
    """Run the greedy scheduler and return the assignment."""
    return (
        GreedyScheduler()
        .schedule(
            SchedulingRequest(
                parallelism=parallelism,
                hosts=tuple(hosts),
                host_hardware=host_hardware,
                layout=layout,
            )
        )
        .assignment
    )


# --------------------------------------------------------------------------
# Homogeneous DGX Spark (1 GPU/host) — current behavior preserved byte-for-byte
# --------------------------------------------------------------------------


def test_dgx_homogeneous_three_hosts_tp3():
    p = ParallelismConfig(tensor_parallel=3)
    hosts = ["spark-01", "spark-02", "spark-03"]

    placement = _place(p, hosts)

    assert placement.hosts_used == ("spark-01", "spark-02", "spark-03")
    assert placement.by_rank == (
        RankSlot(host="spark-01", local_gpu=0),
        RankSlot(host="spark-02", local_gpu=0),
        RankSlot(host="spark-03", local_gpu=0),
    )
    # Old hosts[i] indexing produces the same answer.
    for i, host in enumerate(hosts):
        assert placement.host_for_rank(i) == host
        assert placement.local_gpu_for_rank(i) == 0


def test_dgx_homogeneous_tp_pp_product():
    """tp=2, pp=2 -> 4 ranks across 4 single-GPU hosts."""
    p = ParallelismConfig(tensor_parallel=2, pipeline_parallel=2)
    hosts = ["h1", "h2", "h3", "h4"]

    placement = _place(p, hosts)

    assert len(placement.by_rank) == 4
    assert placement.hosts_used == ("h1", "h2", "h3", "h4")


def test_dgx_data_parallel_three_replicas():
    """dp=3 with tp=1 occupies 3 single-GPU hosts."""
    p = ParallelismConfig(data_parallel=3)
    placement = _place(p, ["a", "b", "c", "d"])
    assert placement.hosts_used == ("a", "b", "c")
    assert placement.max_ranks_per_host == 1


def test_default_dgx_when_hardware_missing():
    """Hosts without explicit metadata fall back to DGX Spark (1 GPU each)."""
    p = ParallelismConfig(tensor_parallel=2)
    placement = _place(p, ["x", "y"], host_hardware={})
    assert placement.hosts_used == ("x", "y")
    assert placement.max_ranks_per_host == 1


# --------------------------------------------------------------------------
# Multi-GPU hosts (homogeneous vendor, large per-host capacity)
# --------------------------------------------------------------------------


def _h200_8gpu(memory_gb: float = 141.0) -> HostHardware:
    return HostHardware(
        accelerators=[
            AcceleratorSpec(
                vendor="nvidia",
                model="h200",
                count=8,
                memory_gb=memory_gb,
                capabilities=frozenset({"cuda", "nvlink"}),
            )
        ]
    )


def test_single_host_8gpu_tp8():
    """tp=8 on one 8-GPU host = 1 host, 8 ranks."""
    p = ParallelismConfig(tensor_parallel=8)
    placement = _place(p, ["dgx-h200"], host_hardware={"dgx-h200": _h200_8gpu()})

    assert placement.hosts_used == ("dgx-h200",)
    assert placement.total_ranks == 8
    assert placement.max_ranks_per_host == 8
    # Each rank gets a distinct local GPU index.
    assert tuple(slot.local_gpu for slot in placement.by_rank) == (0, 1, 2, 3, 4, 5, 6, 7)


def test_two_hosts_8gpu_each_tp16():
    """tp=16 across two 8-GPU hosts."""
    p = ParallelismConfig(tensor_parallel=16)
    hw = {"a": _h200_8gpu(), "b": _h200_8gpu()}
    placement = _place(p, ["a", "b"], host_hardware=hw)

    assert placement.hosts_used == ("a", "b")
    assert placement.ranks_on_host("a") == tuple(range(0, 8))
    assert placement.ranks_on_host("b") == tuple(range(8, 16))


def test_shape_heterogeneous_same_vendor_packs():
    """Different gpus_per_host but single vendor still auto-packs."""
    p = ParallelismConfig(tensor_parallel=10)
    hw = {
        "big": _h200_8gpu(),
        "small": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000", count=2)]),
    }
    placement = _place(p, ["big", "small"], host_hardware=hw)
    assert placement.hosts_used == ("big", "small")
    assert placement.ranks_on_host("big") == tuple(range(0, 8))
    assert placement.ranks_on_host("small") == (8, 9)


# --------------------------------------------------------------------------
# Multi-vendor heterogeneous: explicit layout required
# --------------------------------------------------------------------------


def test_multi_vendor_without_layout_raises():
    p = ParallelismConfig(tensor_parallel=2)
    hw = {
        "spark-01": default_dgx_spark_hardware(),
        "amd-box": HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x", memory_gb=192.0)]),
    }
    with pytest.raises(LayoutConflictError, match="multiple accelerator vendors"):
        _place(p, ["spark-01", "amd-box"], host_hardware=hw)


def test_multi_vendor_single_host_without_layout_raises():
    """One host with multiple vendors (e.g. Apple M5 + RTX) requires explicit layout."""
    p = ParallelismConfig(tensor_parallel=2)
    hw = {
        "laptop": HostHardware(
            accelerators=[
                AcceleratorSpec(vendor="apple", model="m5"),
                AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000"),
            ]
        )
    }
    with pytest.raises(LayoutConflictError, match="multiple accelerator vendors"):
        _place(p, ["laptop"], host_hardware=hw)


def test_multi_vendor_with_explicit_layout_honored():
    p = ParallelismConfig(tensor_parallel=3)
    hw = {
        "spark-01": default_dgx_spark_hardware(),
        "amd-box": HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x", count=2, memory_gb=192.0)]),
    }
    layout = RecipeLayout(
        placements=[
            Placement(host="spark-01", ranks=(0,)),
            Placement(host="amd-box", ranks=(1, 2), local_gpus=(0, 1)),
        ]
    )
    placement = _place(p, ["spark-01", "amd-box"], host_hardware=hw, layout=layout)

    assert placement.by_rank == (
        RankSlot(host="spark-01", local_gpu=0),
        RankSlot(host="amd-box", local_gpu=0),
        RankSlot(host="amd-box", local_gpu=1),
    )
    assert placement.hosts_used == ("spark-01", "amd-box")


# --------------------------------------------------------------------------
# Layout validation errors — surface as SchedulingError (base class)
# --------------------------------------------------------------------------


def test_layout_unknown_host_raises():
    p = ParallelismConfig(tensor_parallel=1)
    layout = RecipeLayout(placements=[Placement(host="ghost", ranks=(0,))])
    with pytest.raises(SchedulingError, match="not present in cluster"):
        _place(p, ["real-host"], layout=layout)


def test_layout_missing_ranks_raises():
    p = ParallelismConfig(tensor_parallel=4)
    layout = RecipeLayout(placements=[Placement(host="h", ranks=(0, 1))])
    with pytest.raises(SchedulingError, match="does not cover"):
        _place(p, ["h"], layout=layout)


def test_layout_duplicate_rank_raises():
    p = ParallelismConfig(tensor_parallel=2)
    layout = RecipeLayout(
        placements=[
            Placement(host="a", ranks=(0,)),
            Placement(host="b", ranks=(0,)),
        ]
    )
    with pytest.raises(SchedulingError, match="multiple hosts"):
        _place(p, ["a", "b"], layout=layout)


def test_layout_ranks_local_gpus_length_mismatch_raises():
    p = ParallelismConfig(tensor_parallel=2)
    layout = RecipeLayout(placements=[Placement(host="h", ranks=(0, 1), local_gpus=(0,))])
    with pytest.raises(SchedulingError, match="local_gpus"):
        _place(p, ["h"], layout=layout)


# --------------------------------------------------------------------------
# Capacity edge cases
# --------------------------------------------------------------------------


def test_insufficient_capacity_raises():
    p = ParallelismConfig(tensor_parallel=4)
    # 2 DGX hosts = 2 slots, requesting 4.
    with pytest.raises(InfeasibleScheduleError, match="cannot satisfy 4 ranks"):
        _place(p, ["a", "b"])


def test_zero_parallelism_returns_empty_assignment():
    """tp=pp=dp=1 -> 1 rank, but a no-op parallelism still produces 1-rank assignment."""
    p = ParallelismConfig()
    placement = _place(p, ["h"])
    assert placement.total_ranks == 1
    assert placement.hosts_used == ("h",)
