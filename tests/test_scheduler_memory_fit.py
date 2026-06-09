"""Regression tests for whole-GPU memory-fit in the scheduler fallback path.

These tests pin three correctness fixes around the usable-memory cap:

BUG 1 (HIGH) — the whole-GPU memory fit guard must NOT vanish when the
cluster status query is unavailable (``status=None``).  Before the fix, a
whole-GPU claim exceeding the host's capped usable memory was routed to the
memory-blind greedy fallback and silently ACCEPTED whenever status probing
failed.  The greedy fallback now enforces the per-rank whole-GPU memory
budget so the same launch is rejected regardless of status availability.

BUG 2 (LOW) — a baked cap of ``0.0`` must not be coalesced to ``1.0``.

The memory-aware fallback must also remain byte-identical to today's
behavior when memory IS satisfiable or when the spec declares no memory.
"""

from __future__ import annotations

import pytest

from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import (
    InfeasibleScheduleError,
    ResourceRequest,
    SchedulingRequest,
)
from sparkrun.schedulers.dense_pack import DensePackScheduler
from sparkrun.schedulers.greedy import GreedyScheduler, _host_gpu_memory, pack
from sparkrun.schedulers.sparse_pack import SparsePackScheduler


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _capped_hw(memory_gb: float, cap: float | None, *, count: int = 1, model: str = "h100") -> HostHardware:
    return HostHardware(
        accelerators=[AcceleratorSpec(vendor="nvidia", model=model, count=count, memory_gb=memory_gb, max_gpu_memory_utilization=cap)]
    )


def _uncapped_hw(memory_gb: float, *, count: int = 1) -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", count=count, memory_gb=memory_gb)])


def _no_memory_hw(*, count: int = 1) -> HostHardware:
    """A spec with no declared memory_gb → memory ignored."""
    return HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", count=count)])


_SCHEDULER_FACTORIES = (SparsePackScheduler, DensePackScheduler)


# --------------------------------------------------------------------------
# BUG 1 — whole-GPU memory rejected even when status is None
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _SCHEDULER_FACTORIES)
def test_whole_gpu_over_capped_memory_rejected_without_status(factory):
    """A whole-GPU claim exceeding the host's CAPPED usable memory is rejected
    even though cluster status is unavailable (status=None → fallback path).

    Usable = 80 × 0.85 = 68 GB; the model needs 75 GB → no GPU fits → infeasible.
    """
    sched = factory()
    hw = {"h1": _capped_hw(80.0, 0.85)}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=None,  # status query failed / unavailable
        resources=ResourceRequest(memory_gb=75.0, util_fraction=1.0),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)


@pytest.mark.parametrize("factory", _SCHEDULER_FACTORIES)
def test_whole_gpu_over_capped_memory_rejected_multi_host_without_status(factory):
    """Multi-host, multi-rank: no host's capped GPU can hold the model →
    infeasible even without status."""
    sched = factory()
    hw = {h: _capped_hw(80.0, 0.85) for h in ("h1", "h2")}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        host_hardware=hw,
        status=None,
        resources=ResourceRequest(memory_gb=70.0, util_fraction=1.0),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)


def test_greedy_pack_rejects_over_capped_memory_directly():
    """The raw greedy pack() rejects an over-budget whole-GPU claim."""
    from sparkrun.core.scheduler import InsufficientCapacityError

    hw = {"h1": _capped_hw(80.0, 0.85)}
    with pytest.raises(InsufficientCapacityError):
        pack(
            ParallelismConfig(tensor_parallel=1),
            ["h1"],
            host_hardware=hw,
            per_rank_memory_gb=75.0,
        )


# --------------------------------------------------------------------------
# Satisfiable claims still pass (byte-identical to today)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _SCHEDULER_FACTORIES)
def test_whole_gpu_within_capped_memory_passes_without_status(factory):
    """A whole-GPU claim that fits within the CAPPED usable memory passes.

    Usable = 80 × 0.85 = 68 GB; the model needs 60 GB → fits.
    """
    sched = factory()
    hw = {"h1": _capped_hw(80.0, 0.85)}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=None,
        resources=ResourceRequest(memory_gb=60.0, util_fraction=1.0),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1",)
    assert result.assignment.by_rank[0].local_gpu == 0


@pytest.mark.parametrize("factory", _SCHEDULER_FACTORIES)
def test_whole_gpu_uncapped_memory_fits_full_nominal(factory):
    """Without a cap (cap=None → 1.0), the full nominal 80 GB is usable."""
    sched = factory()
    hw = {"h1": _uncapped_hw(80.0)}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=None,
        resources=ResourceRequest(memory_gb=75.0, util_fraction=1.0),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1",)


def test_fallback_matches_greedy_when_memory_fits():
    """When memory fits, the occupancy fallback assignment equals greedy's."""
    hw = {h: _capped_hw(80.0, 0.85) for h in ("h1", "h2", "h3")}
    parallelism = ParallelismConfig(tensor_parallel=2)
    # GreedyScheduler ignores memory; the fallback now applies it but a fitting
    # claim must produce the identical assignment.
    greedy = GreedyScheduler().schedule(SchedulingRequest(parallelism=parallelism, hosts=("h1", "h2", "h3"), host_hardware=hw))
    sparse = SparsePackScheduler().schedule(
        SchedulingRequest(
            parallelism=parallelism,
            hosts=("h1", "h2", "h3"),
            host_hardware=hw,
            resources=ResourceRequest(memory_gb=60.0, util_fraction=1.0),
        )
    )
    assert sparse.assignment == greedy.assignment


# --------------------------------------------------------------------------
# None-memory specs are still "ignored"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _SCHEDULER_FACTORIES)
def test_none_memory_spec_ignores_memory_claim(factory):
    """A spec without declared memory_gb accepts any whole-GPU claim
    (memory ignored), even a huge one, without status."""
    sched = factory()
    hw = {"h1": _no_memory_hw()}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=None,
        resources=ResourceRequest(memory_gb=9999.0, util_fraction=1.0),
    )
    result = sched.schedule(req)
    assert result.assignment.hosts_used == ("h1",)


def test_greedy_pack_none_memory_ignores_claim():
    """Raw pack() with a None-memory spec ignores the per-rank requirement."""
    hw = {"h1": _no_memory_hw()}
    result = pack(
        ParallelismConfig(tensor_parallel=1),
        ["h1"],
        host_hardware=hw,
        per_rank_memory_gb=9999.0,
    )
    assert result.hosts_used == ("h1",)


def test_greedy_pack_none_per_rank_memory_is_byte_identical():
    """per_rank_memory_gb=None disables the memory check (today's behavior)."""
    hw = {"h1": _capped_hw(80.0, 0.85)}
    with_none = pack(ParallelismConfig(tensor_parallel=1), ["h1"], host_hardware=hw, per_rank_memory_gb=None)
    without = pack(ParallelismConfig(tensor_parallel=1), ["h1"], host_hardware=hw)
    assert with_none == without
    assert with_none.hosts_used == ("h1",)


# --------------------------------------------------------------------------
# BUG 2 — a baked cap of 0.0 must NOT coalesce to 1.0
# --------------------------------------------------------------------------


def test_zero_cap_yields_zero_usable_memory():
    """A baked cap of 0.0 → usable = 0.0, not the full nominal memory."""
    hw = _capped_hw(80.0, 0.0)
    mem = _host_gpu_memory(hw)
    assert mem == [0.0]


def test_zero_cap_rejects_any_positive_memory_claim_without_status():
    """With cap=0.0, usable=0 → any positive whole-GPU memory claim is infeasible.

    Before the fix the falsy-coalesce turned 0.0 into 1.0 (full memory),
    silently accepting the claim — the opposite of intent.
    """
    sched = SparsePackScheduler()
    hw = {"h1": _capped_hw(80.0, 0.0)}
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=1),
        hosts=("h1",),
        host_hardware=hw,
        status=None,
        resources=ResourceRequest(memory_gb=1.0, util_fraction=1.0),
    )
    with pytest.raises(InfeasibleScheduleError):
        sched.schedule(req)


def test_one_cap_and_none_cap_give_same_usable():
    """cap=1.0 and cap=None both yield the full nominal memory."""
    assert _host_gpu_memory(_capped_hw(80.0, 1.0)) == [80.0]
    assert _host_gpu_memory(_uncapped_hw(80.0)) == [80.0]


# --------------------------------------------------------------------------
# Mixed multi-GPU host: only fitting slots are used
# --------------------------------------------------------------------------


def test_multi_gpu_host_partial_memory_fit_selects_only_fitting_slots():
    """A host with mixed-memory accelerators only exposes the slots that fit.

    GPU 0 (40 GB) cannot hold a 60 GB model; GPU 1 (80 GB) can.  A tp=1
    whole-GPU claim must land on GPU 1.
    """
    hw = {
        "h1": HostHardware(
            accelerators=[
                AcceleratorSpec(vendor="nvidia", model="a", count=1, memory_gb=40.0),
                AcceleratorSpec(vendor="nvidia", model="b", count=1, memory_gb=80.0),
            ]
        )
    }
    result = pack(
        ParallelismConfig(tensor_parallel=1),
        ["h1"],
        host_hardware=hw,
        per_rank_memory_gb=60.0,
    )
    assert result.hosts_used == ("h1",)
    assert result.by_rank[0].local_gpu == 1
