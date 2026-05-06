"""Tests for run_schedule() in scheduler.py.

All subprocess calls are mocked — no real subprocesses are spawned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.benchmarking.scheduler import BenchTask, run_schedule


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal fake subprocess.Popen result."""

    def __init__(self, returncode: int = 0, write_result_to: Path | None = None) -> None:
        self.returncode = returncode
        self._target = write_result_to

    def wait(self, timeout: int | None = None) -> None:
        """Write a fake llama-benchy JSON file when a result path was captured."""
        if self._target is not None:
            data = {
                "model": "org/model",
                "max_concurrency": self._concurrency,
                "benchmarks": [
                    {
                        "concurrency": self._concurrency,
                        "context_size": self._depth,
                        "prompt_size": 2048,
                        "response_size": 32,
                        "is_context_prefill_phase": False,
                    }
                ],
            }
            self._target.write_text(json.dumps(data))

    def kill(self) -> None:
        pass

    # depth/concurrency set by factory after creation
    _depth: int = 0
    _concurrency: int = 1


def _make_popen_factory(returncodes: list[int]):
    """Return a Popen side-effect factory that consumes return codes from a list."""
    rc_iter = iter(returncodes)

    def _factory(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        rc = next(rc_iter)
        result_file: Path | None = None
        for i, arg in enumerate(cmd):
            if arg == "--save-result" and i + 1 < len(cmd):
                result_file = Path(cmd[i + 1])
                break

        proc = _FakeProc(returncode=rc, write_result_to=result_file)

        # Extract depth/concurrency from the result filename (e.g. "000_d0_c1.json")
        if result_file is not None:
            name = result_file.stem  # e.g. "000_d0_c1"
            parts = name.split("_")
            for p in parts:
                if p.startswith("d"):
                    try:
                        proc._depth = int(p[1:])
                    except ValueError:
                        pass
                if p.startswith("c"):
                    try:
                        proc._concurrency = int(p[1:])
                    except ValueError:
                        pass

        return proc

    return _factory


class _FakeFW:
    """Minimal fake BenchmarkingPlugin that builds a deterministic command."""

    framework_name = "fake"

    def build_benchmark_command(
        self,
        target_url: str,
        model: str,
        args: dict[str, Any],
        result_file: str | None = None,
    ) -> list[str]:
        cmd = ["/usr/bin/echo", "task"]
        if result_file:
            cmd.extend(["--save-result", result_file])
        return cmd


def _make_state(tmp_path: Path, n_tasks: int = 3) -> BenchmarkRunState:
    """Return a fresh BenchmarkRunState saved to tmp_path."""
    state = BenchmarkRunState(
        benchmark_id="bench_aabbccddeeff",
        cluster_id="cluster-abc",
        recipe_qualified_name="@registry/my-recipe",
        framework="fake",
        profile="default",
        base_args={"pp": [2048]},
        schedule=[{"depth": i, "concurrency": 1} for i in range(n_tasks)],
    )
    state.save(str(tmp_path))
    return state


def _make_tasks(n: int = 3) -> list[BenchTask]:
    """Return n BenchTask instances with distinct (depth, concurrency) pairs."""
    return [
        BenchTask(
            index=i,
            label="d=%d c=1" % i,
            run_args={"depth": [i], "concurrency": [1]},
            schedule_entry={"depth": i, "concurrency": 1},
        )
        for i in range(n)
    ]


def _run(
    tasks: list[BenchTask],
    state: BenchmarkRunState,
    tmp_path: Path,
    returncodes: list[int],
    *,
    exit_on_first_fail: bool = False,
):
    """Run run_schedule with a mocked Popen and a real BenchmarkProgressUI."""
    fw = _FakeFW()
    ui = BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=state.benchmark_id)

    with ui:
        with patch("subprocess.Popen", side_effect=_make_popen_factory(returncodes)):
            result = run_schedule(
                fw=fw,  # type: ignore[arg-type]
                tasks=tasks,
                state=state,
                target_url="http://localhost:8000/v1",
                model="org/model",
                timeout=None,
                progress_ui=ui,
                cache_dir=str(tmp_path),
                exit_on_first_fail=exit_on_first_fail,
            )
    return result


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_schedule_happy_path(tmp_path: Path):
    """3 tasks all return rc=0 → completed_indices=[0,1,2], is_complete=True, session_count=1."""
    tasks = _make_tasks(3)
    state = _make_state(tmp_path, n_tasks=3)

    result = _run(tasks, state, tmp_path, [0, 0, 0])

    assert result.success is True
    assert sorted(result.state.completed_indices) == [0, 1, 2]
    assert result.state.is_complete(3) is True
    assert result.state.session_count == 1
    assert result.state.crash_count == 0
    assert result.failed_count == 0


# ---------------------------------------------------------------------------
# Warmup rule
# ---------------------------------------------------------------------------


def test_run_schedule_warmup_rule(tmp_path: Path):
    """Task 0 must NOT have no_warmup; task 1+ must have no_warmup=True and skip_coherence=True."""
    tasks = _make_tasks(2)
    state = _make_state(tmp_path, n_tasks=2)
    fw = _FakeFW()

    # Capture actual args passed to build_benchmark_command
    call_args_list: list[dict[str, Any]] = []
    original_build = fw.build_benchmark_command

    def _capturing_build(target_url, model, args, result_file=None):
        call_args_list.append(dict(args))
        return original_build(target_url, model, args, result_file=result_file)

    fw.build_benchmark_command = _capturing_build  # type: ignore[method-assign]

    ui = BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=state.benchmark_id)
    with ui:
        with patch("subprocess.Popen", side_effect=_make_popen_factory([0, 0])):
            run_schedule(
                fw=fw,  # type: ignore[arg-type]
                tasks=tasks,
                state=state,
                target_url="http://localhost:8000/v1",
                model="org/model",
                timeout=None,
                progress_ui=ui,
                cache_dir=str(tmp_path),
            )

    assert len(call_args_list) == 2, "Expected exactly 2 build_benchmark_command calls"

    # First task — warmup must NOT be suppressed
    first_args = call_args_list[0]
    assert first_args.get("no_warmup") is not True, "Task 0 should not have no_warmup=True"
    assert first_args.get("skip_coherence") is not True, "Task 0 should not have skip_coherence=True"

    # Second task — warmup must be suppressed
    second_args = call_args_list[1]
    assert second_args.get("no_warmup") is True, "Task 1 should have no_warmup=True"
    assert second_args.get("skip_coherence") is True, "Task 1 should have skip_coherence=True"


# ---------------------------------------------------------------------------
# Resume warmup rule
# ---------------------------------------------------------------------------


def test_run_schedule_resume_warmup_rule(tmp_path: Path):
    """On resume (tasks 0,1 pre-completed), task 2 is session-first so no warmup suppression.

    Tasks 3+ should have no_warmup=True (not first of the new session).
    consolidate_results is patched to return all (depth, concurrency) pairs as present so
    the post-loop gap pass finds nothing to re-queue and exactly 2 Popen calls are made.
    """
    tasks = _make_tasks(4)
    state = _make_state(tmp_path, n_tasks=4)

    # Pre-populate tasks 0 and 1 as already completed
    state.completed_indices = [0, 1]
    state.save(str(tmp_path))

    fw = _FakeFW()
    call_args_list: list[dict[str, Any]] = []
    original_build = fw.build_benchmark_command

    def _capturing_build(target_url, model, args, result_file=None):
        call_args_list.append(dict(args))
        return original_build(target_url, model, args, result_file=result_file)

    fw.build_benchmark_command = _capturing_build  # type: ignore[method-assign]

    # Return a fully-populated consolidated result so gap_analysis returns [] every time.
    # Without this patch, the gap pass detects tasks 0/1 as missing from runs/ (their result
    # files were never written by the pre-completed session) and re-queues them, causing
    # more than 2 Popen calls and making warmup assertions ambiguous.
    full_consolidated = {
        "model": "org/model",
        "max_concurrency": 1,
        "benchmarks": [{"context_size": i, "concurrency": 1} for i in range(4)],
    }

    ui = BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=state.benchmark_id)
    with ui:
        with patch("subprocess.Popen", side_effect=_make_popen_factory([0, 0])):
            with patch("sparkrun.benchmarking.scheduler.consolidate_results", return_value=full_consolidated):
                run_schedule(
                    fw=fw,  # type: ignore[arg-type]
                    tasks=tasks,
                    state=state,
                    target_url="http://localhost:8000/v1",
                    model="org/model",
                    timeout=None,
                    progress_ui=ui,
                    cache_dir=str(tmp_path),
                )

    # Only tasks 2 and 3 ran in this session (0 and 1 were pre-completed)
    assert len(call_args_list) == 2, "Expected exactly 2 build_benchmark_command calls, got %d" % len(call_args_list)

    # Task index 2 is first of new session — warmup should NOT be suppressed
    first_resumed_args = call_args_list[0]
    assert first_resumed_args.get("no_warmup") is not True, "First resumed task should not have no_warmup=True"
    assert first_resumed_args.get("skip_coherence") is not True, "First resumed task should not have skip_coherence=True"

    # Task index 3 is subsequent — warmup and coherence should be suppressed
    second_resumed_args = call_args_list[1]
    assert second_resumed_args.get("no_warmup") is True, "Second resumed task should have no_warmup=True"
    assert second_resumed_args.get("skip_coherence") is True, "Second resumed task should have skip_coherence=True"


# ---------------------------------------------------------------------------
# Failure with exit_on_first_fail
# ---------------------------------------------------------------------------


def test_run_schedule_exit_on_first_fail(tmp_path: Path):
    """Task 1 fails (rc=2) with exit_on_first_fail=True → stops, task 2 never runs."""
    tasks = _make_tasks(3)
    state = _make_state(tmp_path, n_tasks=3)

    # Task 0: success, task 1: fail — task 2 should never be called
    popen_factory = _make_popen_factory([0, 2])

    popen_call_count = 0
    original_factory = popen_factory

    def _counting_factory(cmd, *a, **kw):
        nonlocal popen_call_count
        popen_call_count += 1
        return original_factory(cmd, *a, **kw)

    fw = _FakeFW()
    ui = BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=state.benchmark_id)
    with ui:
        with patch("subprocess.Popen", side_effect=_counting_factory):
            result = run_schedule(
                fw=fw,  # type: ignore[arg-type]
                tasks=tasks,
                state=state,
                target_url="http://localhost:8000/v1",
                model="org/model",
                timeout=None,
                progress_ui=ui,
                cache_dir=str(tmp_path),
                exit_on_first_fail=True,
            )

    assert result.success is False
    assert 1 in result.state.failed_indices
    assert popen_call_count == 2, "Only tasks 0 and 1 should have been launched"
    assert 2 not in result.state.completed_indices


# ---------------------------------------------------------------------------
# Gap re-queue
# ---------------------------------------------------------------------------


def test_run_schedule_gap_requeue(tmp_path: Path):
    """When consolidate_results shows a gap, the missing task is re-dispatched once."""
    tasks = _make_tasks(2)
    state = _make_state(tmp_path, n_tasks=2)

    popen_call_count = 0

    # First pass: both tasks return rc=0 and write result files.
    # But we'll mock consolidate_results to report a gap on the first call
    # (simulating a missing benchmark row), then return complete on the second.
    fake_full_consolidated = {
        "model": "org/model",
        "max_concurrency": 1,
        "benchmarks": [
            {"context_size": 0, "concurrency": 1},
            {"context_size": 1, "concurrency": 1},
        ],
    }
    fake_gap_consolidated = {
        "model": "org/model",
        "max_concurrency": 1,
        "benchmarks": [
            # Only task 0 present — task 1 (depth=1, concurrency=1) is a gap
            {"context_size": 0, "concurrency": 1},
        ],
    }

    consolidate_call_count = 0

    def _fake_consolidate(state_dir):
        nonlocal consolidate_call_count
        consolidate_call_count += 1
        # First 3 calls (initial + after each task success) return the gap result
        # After that, return full so gap pass completes
        if consolidate_call_count <= 3:
            return fake_gap_consolidated
        return fake_full_consolidated

    def _counting_popen(cmd, *a, **kw):
        nonlocal popen_call_count
        popen_call_count += 1
        return _make_popen_factory([0])(cmd, *a, **kw)

    fw = _FakeFW()
    ui = BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=state.benchmark_id)
    with ui:
        with patch("subprocess.Popen", side_effect=_counting_popen):
            with patch("sparkrun.benchmarking.scheduler.consolidate_results", side_effect=_fake_consolidate):
                result = run_schedule(
                    fw=fw,  # type: ignore[arg-type]
                    tasks=tasks,
                    state=state,
                    target_url="http://localhost:8000/v1",
                    model="org/model",
                    timeout=None,
                    progress_ui=ui,
                    cache_dir=str(tmp_path),
                )

    # Task 1 (the gap) should have been re-dispatched: 2 initial + 1 gap re-run = 3 total
    assert popen_call_count >= 3, "Expected at least 3 Popen calls (2 initial + 1 gap re-run), got %d" % popen_call_count
    assert result.success is True
