"""Benchmark scheduling: per-task abstraction and the executor loop.

Frameworks that opt into batched/scheduled execution build a list of
:class:`BenchTask` instances via
:meth:`sparkrun.benchmarking.base.BenchmarkingPlugin.build_task_list`. The
scheduler then dispatches each task as a single benchmark subprocess,
persists per-task JSON, and updates :class:`BenchmarkRunState` so the run
can be resumed after a crash.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sparkrun.benchmarking.aggregator import consolidate_results, gap_analysis
from sparkrun.benchmarking.run_state import BenchmarkRunState

if TYPE_CHECKING:
    from sparkrun.benchmarking.base import BenchmarkingPlugin
    from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchTask:
    """One scheduled benchmark invocation.

    Attributes:
        index: 0-based position in the schedule. Stable across resumes.
        label: Short human description for the progress UI (e.g. ``"d=4096 c=2"``).
        run_args: Final merged arg dict passed to ``build_benchmark_command``.
            For llama-benchy with a (depth, concurrency) entry this contains
            single-element ``depth`` and ``concurrency`` lists.
        schedule_entry: Raw per-task override dict from the YAML schedule
            (or the auto-generated default). Persisted in run state so the
            schedule can be reconstructed on resume.
    """

    index: int
    label: str
    run_args: dict[str, Any] = field(default_factory=dict)
    schedule_entry: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleRunResult:
    """Result returned by :func:`run_schedule`."""

    success: bool
    completed_count: int
    failed_count: int
    state: BenchmarkRunState
    consolidated: dict[str, Any]


def _extract_depth_concurrency(run_args: dict[str, Any]) -> tuple[int, int]:
    """Extract (depth, concurrency) scalars from run_args lists, falling back to (0, 1)."""
    depth_list = run_args.get("depth", [])
    concurrency_list = run_args.get("concurrency", [])
    depth = depth_list[0] if depth_list else 0
    concurrency = concurrency_list[0] if concurrency_list else 1
    return int(depth), int(concurrency)


def run_schedule(
    fw: "BenchmarkingPlugin",
    tasks: list[BenchTask],
    state: BenchmarkRunState,
    *,
    target_url: str,
    model: str,
    timeout: int | None,
    progress_ui: "BenchmarkProgressUI",
    cache_dir: str | None = None,
    exit_on_first_fail: bool = False,
    skip_run: bool = False,
) -> ScheduleRunResult:
    """Iterate pending tasks. Returns when the schedule is complete or aborts.

    Args:
        fw: The benchmarking plugin that can construct subprocess commands.
        tasks: Ordered list of :class:`BenchTask` instances to execute.
        state: Resumable run state — mutated and saved throughout execution.
        target_url: Inference endpoint URL forwarded to the benchmark command.
        model: Model name forwarded to the benchmark command.
        timeout: Per-task subprocess timeout in seconds, or ``None`` for no limit.
        progress_ui: Live progress display context manager (must already be entered).
        cache_dir: Override for the sparkrun cache directory root.
        exit_on_first_fail: Stop immediately after the first task failure.
        skip_run: When ``True``, the warmup/coherence steps are suppressed even
            for the first task of the session.

    Returns:
        :class:`ScheduleRunResult` describing the outcome.
    """
    total = len(tasks)
    consolidated: dict[str, Any] = consolidate_results(state.state_dir(cache_dir))

    # Session bookkeeping — mark this execution session.
    state.mark_session_started()
    state.save(cache_dir)

    session_first_task = True
    _gap_pass_done = False

    def _do_loop() -> tuple[bool, bool]:
        """Inner loop over pending tasks.

        Returns:
            (aborted, exit_requested) — aborted=True means we should stop immediately.
        """
        nonlocal session_first_task, consolidated

        while True:
            idx = state.next_pending(total)
            if idx is None:
                break

            task = tasks[idx]
            progress_ui.start_task(idx, task.label)

            # Build per-task args, applying warmup/coherence rule.
            run_args: dict[str, Any] = dict(task.run_args)
            if not session_first_task:
                run_args.setdefault("no_warmup", True)
                run_args.setdefault("skip_coherence", True)

            depth, concurrency = _extract_depth_concurrency(run_args)
            result_file = state.runs_dir(cache_dir) / ("%03d_d%d_c%d.json" % (idx, depth, concurrency))
            log_file = state.runs_dir(cache_dir) / ("%03d_d%d_c%d.log" % (idx, depth, concurrency))

            cmd = fw.build_benchmark_command(target_url, model, run_args, result_file=str(result_file))

            state.mark_started(idx)
            state.save(cache_dir)

            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc: subprocess.Popen | None = None
            t_start = time.monotonic()

            try:
                log_fh = open(log_file, "w")  # closed in finally block below
                try:
                    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        logger.warning("Task %d (%s) timed out after %s seconds; killing process", idx, task.label, timeout)
                        proc.kill()
                        proc.wait()
                        duration_s = time.monotonic() - t_start
                        state.mark_failed(idx, "timeout after %ds" % timeout)
                        state.save(cache_dir)
                        progress_ui.end_task(idx, success=False, duration_s=duration_s)
                        if exit_on_first_fail:
                            state.mark_session_ended("partial")
                            state.save(cache_dir)
                            return True, False
                        continue
                finally:
                    log_fh.close()

                duration_s = time.monotonic() - t_start
                rc = proc.returncode

                if rc == 0:
                    state.mark_completed(idx)
                    state.save(cache_dir)
                    progress_ui.end_task(idx, success=True, duration_s=duration_s)
                    consolidated = consolidate_results(state.state_dir(cache_dir))
                    progress_ui.update_results_table(consolidated)
                    session_first_task = False
                else:
                    state.mark_failed(idx, "exit code %d" % rc)
                    state.save(cache_dir)
                    progress_ui.end_task(idx, success=False, duration_s=duration_s)
                    if exit_on_first_fail:
                        state.mark_session_ended("partial")
                        state.save(cache_dir)
                        return True, False

            except KeyboardInterrupt:
                if proc is not None:
                    proc.kill()
                    proc.wait()
                return False, True  # signal KeyboardInterrupt to caller

        return False, False

    try:
        aborted, interrupted = _do_loop()

        if interrupted:
            state.mark_session_ended("interrupted")
            state.save(cache_dir)
            raise KeyboardInterrupt

        if aborted:
            # exit_on_first_fail already called mark_session_ended("partial") inside loop.
            return ScheduleRunResult(
                success=False,
                completed_count=len(state.completed_indices),
                failed_count=len(state.failed_indices),
                state=state,
                consolidated=consolidated,
            )

        # Post-loop gap analysis — done at most once.
        if not _gap_pass_done:
            _gap_pass_done = True
            gaps = gap_analysis(tasks, consolidated)
            if gaps:
                progress_ui.log("Found %d gap(s); re-queueing" % len(gaps))
                for gap_task in gaps:
                    if gap_task.index in state.completed_indices:
                        state.completed_indices.remove(gap_task.index)
                state.save(cache_dir)
                # Re-enter the loop for gap tasks.
                aborted, interrupted = _do_loop()
                if interrupted:
                    state.mark_session_ended("interrupted")
                    state.save(cache_dir)
                    raise KeyboardInterrupt
                if aborted:
                    return ScheduleRunResult(
                        success=False,
                        completed_count=len(state.completed_indices),
                        failed_count=len(state.failed_indices),
                        state=state,
                        consolidated=consolidated,
                    )

        # Final consolidation and session close.
        consolidated = consolidate_results(state.state_dir(cache_dir))
        progress_ui.update_results_table(consolidated)

        if state.is_complete(total):
            state.mark_session_ended("completed")
        else:
            state.mark_session_ended("partial")
        state.save(cache_dir)

        return ScheduleRunResult(
            success=state.is_complete(total),
            completed_count=len(state.completed_indices),
            failed_count=len(state.failed_indices),
            state=state,
            consolidated=consolidated,
        )

    except KeyboardInterrupt:
        # Already handled mark_session_ended("interrupted") above; just re-raise.
        raise

    except Exception:
        logger.exception("Unexpected error in run_schedule for benchmark %s", state.benchmark_id)
        state.mark_crash()
        state.mark_session_ended("crashed")
        state.save(cache_dir)
        return ScheduleRunResult(
            success=False,
            completed_count=len(state.completed_indices),
            failed_count=len(state.failed_indices),
            state=state,
            consolidated=consolidated,
        )
