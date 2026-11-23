"""A launch's GPU claim comes from its runtime's memory fraction (#335).

Two ``--gpu-mem 0.3`` vLLM launches on one DGX Spark used to be refused by the
occupancy schedulers: every launch asked for, and recorded, an exclusive GPU.
"""

from __future__ import annotations

import pytest

from sparkrun.api._errors import InsufficientCapacity
from sparkrun.api._hosts import resolve_effective_hosts
from sparkrun.core.allocations import GpuAllocation, allocations_for_host
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload, with_gpu_allocations
from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.llama_cpp import LlamaCppRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

CID = "sparkrun_" + "a" * 16 + "_" + "b" * 12


def recipe(command=None, runtime="vllm-distributed", **defaults):
    return Recipe(
        {
            "model": "Qwen/Qwen3-VL-Reranker-8B",
            "runtime": runtime,
            "container": "image",
            "metadata": {"model_vram": 16.3},
            "defaults": defaults,
            "command": command,
        }
    )


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("defaults", "overrides", "command", "expected"),
    [
        ({}, {}, None, 1.0),  # unset: exclusive
        ({"gpu_memory_utilization": 0.3}, {}, None, 0.3),
        ({"gpu_memory_utilization": 0.9}, {"gpu_memory_utilization": 0.25}, None, 0.25),  # CLI wins
        ({}, {}, "vllm serve {model} --gpu-memory-utilization 0.4", 0.4),  # template literal
        ({}, {}, "vllm serve {model} --gpu-memory-utilization {gpu_memory_utilization}", 1.0),  # placeholder, unset
        ({}, {"gpu_memory_utilization": 0.3}, "vllm serve {model} --gpu-memory-utilization {gpu_memory_utilization}", 0.3),
        # A template is rendered verbatim: its literal is what the engine gets, whatever -o says...
        ({}, {"gpu_memory_utilization": 0.3}, "vllm serve {model} --gpu-memory-utilization 0.9", 0.9),
        # ...and without the flag the engine uses its own default, so no share is declared.
        ({"gpu_memory_utilization": 0.3}, {}, "vllm serve {model} --port 8000", 1.0),
        ({"gpu_memory_utilization": 1.5}, {}, None, 1.0),  # out of range: never under-claims
        ({"gpu_memory_utilization": "lots"}, {}, None, 1.0),  # malformed: never under-claims
    ],
)
def test_vllm_claim(defaults, overrides, command, expected):
    assert VllmDistributedRuntime().scheduler_gpu_fraction(recipe(command, **defaults), overrides) == expected


def test_sglang_reads_its_own_flag_spelling():
    r = recipe("python3 -m sglang.launch_server --mem-fraction-static 0.35", runtime="sglang")
    assert SglangRuntime().scheduler_gpu_fraction(r) == 0.35
    # vLLM's spelling means nothing to SGLang.
    assert SglangRuntime().scheduler_gpu_fraction(recipe("sglang serve --gpu-memory-utilization 0.35", runtime="sglang")) == 1.0


def test_runtime_without_a_fraction_stays_exclusive():
    r = recipe(runtime="llama-cpp", gpu_memory_utilization=0.3)
    assert LlamaCppRuntime().declared_gpu_utilization(r) is None
    assert LlamaCppRuntime().scheduler_gpu_fraction(r) == 1.0


def test_estimate_uses_the_runtimes_reading():
    """The fit report reads the same fraction the scheduler claims, template literal included."""
    est = recipe("vllm serve {model} --gpu-memory-utilization 0.4").estimate_vram(auto_detect=False, total_gpu_memory_gb=121)
    assert est.gpu_memory_utilization == 0.4
    assert est.usable_gpu_memory_gb == pytest.approx(121 * 0.4)
    assert recipe(runtime="llama-cpp", gpu_memory_utilization=0.3).estimate_vram(auto_detect=False).gpu_memory_utilization is None


# ---------------------------------------------------------------------------
# Scheduling: the reporter's flow
# ---------------------------------------------------------------------------


def existing(fraction, memory_gb):
    """spark0 / GPU 0 running one workload whose label records *fraction*."""
    allocation = GpuAllocation(0, 0, fraction, memory_gb)
    detail = ContainerDetail(CID + "_solo", "solo", "Up", "image", allocations=(allocation,))
    occupancy = with_gpu_allocations(HostOccupancy("spark0", (RunningWorkload(CID, containers=(detail,)),), used_slots=1), 1)
    return ClusterStatus((occupancy,))


def place(r, overrides, *, scheduler="occupancy-dense", status=None):
    return resolve_effective_hosts(
        ["spark0"],
        r,
        overrides,
        runtime=VllmDistributedRuntime(),
        scheduler=scheduler,
        status_snapshot=status if status is not None else ClusterStatus((HostOccupancy("spark0", ()),)),
    )


@pytest.mark.parametrize("scheduler", ["occupancy-dense", "occupancy-sparse"])
def test_two_shared_launches_share_one_gpu(scheduler):
    # First launch records a 0.3 share, not an exclusive GPU.
    _hosts, _solo, _notes, first = place(recipe(), {"gpu_memory_utilization": 0.3}, scheduler=scheduler)
    (allocation,) = allocations_for_host(first, "spark0")
    assert allocation.util_fraction == 0.3 and not allocation.exclusive
    assert allocation.memory_gb == pytest.approx(0.3 * first_capacity(first))

    # Second launch is admitted beside it.
    _hosts, _solo, _notes, second = place(
        recipe(), {"gpu_memory_utilization": 0.3}, scheduler=scheduler, status=existing(0.3, allocation.memory_gb)
    )
    assert second.by_rank[0].local_gpu == 0
    assert second.by_rank[0].util_fraction == 0.3


def first_capacity(placement):
    # memory_gb / util_fraction recovers the GPU capacity the claim was taken against.
    slot = placement.by_rank[0]
    return slot.memory_gb / slot.util_fraction


@pytest.mark.parametrize("fraction", [None, 0.85])
def test_unset_or_high_fraction_stays_exclusive(fraction):
    overrides = {} if fraction is None else {"gpu_memory_utilization": fraction}
    _hosts, _solo, _notes, placement = place(recipe(), overrides)
    assert placement.by_rank[0].util_fraction == 1.0
    with pytest.raises(InsufficientCapacity):
        place(recipe(), {"gpu_memory_utilization": 0.3}, status=existing(1.0, None))


def test_shares_beyond_the_gpu_are_refused():
    with pytest.raises(InsufficientCapacity) as error:
        place(recipe(), {"gpu_memory_utilization": 0.5}, status=existing(0.5, 60.0))
    assert {r.reason for r in error.value.rejections} & {"memory_budget", "compute_budget"}


def test_greedy_keeps_whole_gpu_requests():
    """greedy refuses fractional requests, so it never receives one."""
    _hosts, _solo, _notes, placement = place(recipe(), {"gpu_memory_utilization": 0.3}, scheduler="greedy")
    assert placement.by_rank[0].util_fraction == 1.0


def place_tp2(hosts, status):
    return resolve_effective_hosts(
        list(hosts),
        recipe(tensor_parallel=2),
        {"gpu_memory_utilization": 0.3},
        runtime=VllmDistributedRuntime(),
        scheduler="occupancy-sparse",
        status_snapshot=status,
    )


def test_multi_rank_launch_shares_with_one_rank_per_gpu():
    """tp=2 at 0.3 across two Sparks: each rank takes a 0.3 share of its own GPU,
    beside another workload's share."""
    status = ClusterStatus((existing(0.3, 36.0).hosts[0], HostOccupancy("spark1", ())))
    _hosts, _solo, _notes, placement = place_tp2(["spark0", "spark1"], status)
    assert [slot.util_fraction for slot in placement.by_rank] == [0.3, 0.3]
    assert sorted(slot.host for slot in placement.by_rank) == ["spark0", "spark1"]


def test_multi_rank_launch_never_doubles_up_on_one_gpu():
    """Both ranks would fit spark0's GPU budget, but one NCCL world needs a GPU per rank."""
    with pytest.raises(InsufficientCapacity, match="placed 1 of 2") as error:
        place_tp2(["spark0"], ClusterStatus((HostOccupancy("spark0", ()),)))
    assert not error.value.rejections  # spark0 is in use by rank 0, not blocked


def test_too_few_gpus_blames_the_busy_host_not_the_used_ones():
    """tp=4 on three idle Sparks plus one exclusively owned one: only the owned host is
    reported, so a picker does not disable the hosts this launch already used."""
    from dataclasses import replace

    owned = replace(existing(1.0, None).hosts[0], host="spark3")
    status = ClusterStatus((HostOccupancy("spark0", ()), HostOccupancy("spark1", ()), HostOccupancy("spark2", ()), owned))
    with pytest.raises(InsufficientCapacity) as error:
        resolve_effective_hosts(
            ["spark0", "spark1", "spark2", "spark3"],
            recipe(tensor_parallel=4),
            {"gpu_memory_utilization": 0.3},
            runtime=VllmDistributedRuntime(),
            scheduler="occupancy-sparse",
            status_snapshot=status,
        )
    assert {r.host for r in error.value.rejections} == {"spark3"}


def test_explicit_layout_cannot_repeat_a_gpu():
    """A layout is placed verbatim, except where no engine could run it."""
    from sparkrun.core.layout import Placement, RecipeLayout

    r = recipe(tensor_parallel=2)
    r.layout = RecipeLayout(placements=[Placement(host="spark0", ranks=(0, 1), local_gpus=(0, 0))])
    with pytest.raises(InsufficientCapacity) as error:
        place(r, {"gpu_memory_utilization": 0.3})
    assert [x.reason for x in error.value.rejections] == ["same_request"]


def test_max_nodes_trim_spreads_shared_ranks():
    """The max_nodes trim schedules one rank per retained host through the same rule."""
    r = Recipe(
        {
            "model": "Qwen/Qwen3-VL-Reranker-8B",
            "runtime": "vllm-distributed",
            "container": "image",
            "max_nodes": 2,
            "metadata": {"model_vram": 16.3},
        }
    )
    hosts = ["spark0", "spark1", "spark2"]
    _hosts, _solo, _notes, placement = resolve_effective_hosts(
        hosts,
        r,
        {"gpu_memory_utilization": 0.3},
        cluster_def=ClusterDefinition(name="c", hosts=hosts),
        runtime=VllmDistributedRuntime(),
        scheduler="occupancy-sparse",
        status_snapshot=ClusterStatus(tuple(HostOccupancy(h, ()) for h in hosts)),
    )
    assert len({(slot.host, slot.local_gpu) for slot in placement.by_rank}) == len(placement.by_rank) == 2
    assert {slot.util_fraction for slot in placement.by_rank} == {0.3}


def test_atlas_fraction_is_its_budget_and_its_claim():
    from sparkrun.runtimes.atlas import AtlasRuntime

    r = recipe(runtime="atlas", gpu_memory_utilization=0.3)
    assert AtlasRuntime().declared_gpu_utilization(r) == 0.3
    assert AtlasRuntime().scheduler_gpu_fraction(r) == 0.3
