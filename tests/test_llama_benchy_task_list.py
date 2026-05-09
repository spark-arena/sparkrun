"""Tests for LlamaBenchyFramework.build_task_list."""

from __future__ import annotations

import pytest

from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
from sparkrun.core.benchmark_profiles import BenchmarkError


@pytest.fixture
def fw() -> LlamaBenchyFramework:
    """Return an uninitialized LlamaBenchyFramework instance."""
    return LlamaBenchyFramework()


# ---------------------------------------------------------------------------
# schedule=None — cartesian product
# ---------------------------------------------------------------------------


def test_build_task_list_no_schedule_count(fw: LlamaBenchyFramework):
    """schedule=None builds depth × concurrency cartesian product tasks."""
    base_args = {"depth": [0, 4096], "concurrency": [1, 2, 5]}
    tasks = fw.build_task_list(base_args, schedule=None)
    # 2 depths × 3 concurrencies = 6 tasks
    assert len(tasks) == 6


def test_build_task_list_no_schedule_depth_major_order(fw: LlamaBenchyFramework):
    """schedule=None iterates depth-major (outer loop = depth, inner = concurrency)."""
    base_args = {"depth": [0, 4096], "concurrency": [1, 2]}
    tasks = fw.build_task_list(base_args, schedule=None)

    # Expected order: (d=0,c=1), (d=0,c=2), (d=4096,c=1), (d=4096,c=2)
    assert tasks[0].run_args["depth"] == [0]
    assert tasks[0].run_args["concurrency"] == [1]
    assert tasks[1].run_args["depth"] == [0]
    assert tasks[1].run_args["concurrency"] == [2]
    assert tasks[2].run_args["depth"] == [4096]
    assert tasks[2].run_args["concurrency"] == [1]
    assert tasks[3].run_args["depth"] == [4096]
    assert tasks[3].run_args["concurrency"] == [2]


def test_build_task_list_no_schedule_single_element_run_args(fw: LlamaBenchyFramework):
    """Each generated task has depth and concurrency as single-element lists."""
    base_args = {"depth": [0, 4096], "concurrency": [1, 5]}
    tasks = fw.build_task_list(base_args, schedule=None)

    for task in tasks:
        assert isinstance(task.run_args["depth"], list)
        assert len(task.run_args["depth"]) == 1
        assert isinstance(task.run_args["concurrency"], list)
        assert len(task.run_args["concurrency"]) == 1


def test_build_task_list_no_schedule_indices(fw: LlamaBenchyFramework):
    """Task indices are 0-based and contiguous."""
    base_args = {"depth": [0, 4096], "concurrency": [1, 2, 5]}
    tasks = fw.build_task_list(base_args, schedule=None)
    assert [t.index for t in tasks] == list(range(6))


# ---------------------------------------------------------------------------
# schedule provided
# ---------------------------------------------------------------------------


def test_build_task_list_with_schedule_count(fw: LlamaBenchyFramework):
    """Explicit schedule builds exactly as many tasks as schedule entries."""
    schedule = [
        {"depth": 0, "concurrency": 1},
        {"depth": 4096, "concurrency": 5},
    ]
    tasks = fw.build_task_list({}, schedule=schedule)
    assert len(tasks) == 2


def test_build_task_list_with_schedule_labels_and_run_args(fw: LlamaBenchyFramework):
    """Each task from an explicit schedule has the correct label and run_args."""
    schedule = [
        {"depth": 0, "concurrency": 1},
        {"depth": 4096, "concurrency": 5},
    ]
    tasks = fw.build_task_list({}, schedule=schedule)

    assert tasks[0].label == "d=0 c=1"
    assert tasks[0].run_args["depth"] == [0]
    assert tasks[0].run_args["concurrency"] == [1]

    assert tasks[1].label == "d=4096 c=5"
    assert tasks[1].run_args["depth"] == [4096]
    assert tasks[1].run_args["concurrency"] == [5]


def test_build_task_list_with_schedule_schedule_entry_preserved(fw: LlamaBenchyFramework):
    """schedule_entry on each task echoes the input dict."""
    schedule = [
        {"depth": 0, "concurrency": 1},
        {"depth": 4096, "concurrency": 5},
    ]
    tasks = fw.build_task_list({}, schedule=schedule)

    assert tasks[0].schedule_entry == {"depth": 0, "concurrency": 1}
    assert tasks[1].schedule_entry == {"depth": 4096, "concurrency": 5}


def test_build_task_list_per_task_override_flows_through(fw: LlamaBenchyFramework):
    """Per-task override keys (e.g. runs) flow into run_args; depth/concurrency become lists."""
    schedule = [{"depth": 0, "concurrency": 1, "runs": 5}]
    tasks = fw.build_task_list({"runs": 3}, schedule=schedule)

    assert len(tasks) == 1
    task = tasks[0]
    # depth and concurrency pinned to single-element lists
    assert task.run_args["depth"] == [0]
    assert task.run_args["concurrency"] == [1]
    # runs override applied
    assert task.run_args["runs"] == 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_build_task_list_missing_depth_raises(fw: LlamaBenchyFramework):
    """schedule entry missing 'depth' raises BenchmarkError."""
    schedule = [{"concurrency": 1}]
    with pytest.raises(BenchmarkError, match="depth"):
        fw.build_task_list({}, schedule=schedule)


def test_build_task_list_missing_concurrency_raises(fw: LlamaBenchyFramework):
    """schedule entry missing 'concurrency' raises BenchmarkError."""
    schedule = [{"depth": 0}]
    with pytest.raises(BenchmarkError, match="concurrency"):
        fw.build_task_list({}, schedule=schedule)


def test_build_task_list_non_int_depth_raises(fw: LlamaBenchyFramework):
    """schedule entry with non-int 'depth' raises BenchmarkError."""
    schedule = [{"depth": "zero", "concurrency": 1}]
    with pytest.raises(BenchmarkError, match="depth"):
        fw.build_task_list({}, schedule=schedule)


def test_build_task_list_non_int_concurrency_raises(fw: LlamaBenchyFramework):
    """schedule entry with non-int 'concurrency' raises BenchmarkError."""
    schedule = [{"depth": 0, "concurrency": "one"}]
    with pytest.raises(BenchmarkError, match="concurrency"):
        fw.build_task_list({}, schedule=schedule)
