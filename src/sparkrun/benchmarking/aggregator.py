"""Benchmark result aggregation: merge per-task JSON files into a framework-shaped dict."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.benchmarking.base import BenchmarkingPlugin
    from sparkrun.benchmarking.scheduler import BenchTask

logger = logging.getLogger(__name__)

_RUNS_FILENAME_RE = re.compile(r"^(\d+)_?")


def consolidate_results(state_dir: Path, fw: "BenchmarkingPlugin") -> dict[str, Any]:
    """Read every JSON file under ``state_dir/runs/`` in schedule-index order and
    delegate consolidation to ``fw.consolidate_per_task_results()``.

    Per-file errors (missing/invalid JSON, non-dict top level) are logged and
    skipped so that one bad file does not abort the whole consolidation.

    Returns the framework-shaped dict produced by the plugin.  When no files
    exist yet, the empty list is passed to the plugin so it can decide on a
    safe default.
    """
    runs_dir = state_dir / "runs"
    if not runs_dir.is_dir():
        return fw.consolidate_per_task_results([])

    json_files = sorted(
        runs_dir.glob("*.json"),
        key=lambda p: int(m.group(1)) if (m := _RUNS_FILENAME_RE.match(p.name)) else 0,
    )

    per_task_jsons: list[dict[str, Any]] = []
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("aggregator: skipping %s — %s", json_path.name, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("aggregator: skipping %s — top-level value is not a dict", json_path.name)
            continue
        per_task_jsons.append(data)

    return fw.consolidate_per_task_results(per_task_jsons)


def gap_analysis(
    task_list: list["BenchTask"],
    consolidated: dict[str, Any],
    fw: "BenchmarkingPlugin",
    expected_per_task: int = 1,
) -> list["BenchTask"]:
    """Return tasks whose coverage key is absent from the consolidated dict.

    Coverage is defined by the framework via ``fw.task_coverage_key`` and
    ``fw.consolidated_coverage_keys``.  When a task's coverage key is ``None``
    or missing, it is also flagged as a gap (with a warning) — this preserves
    the "tasks with malformed run_args are surfaced" behavior.
    """
    observed = fw.consolidated_coverage_keys(consolidated)

    gaps: list[BenchTask] = []
    for task in task_list:
        try:
            key = fw.task_coverage_key(task)
        except Exception:  # pragma: no cover — defensive: don't abort the run on plugin bug
            logger.warning("gap_analysis: task %d (%s) coverage_key raised; treating as gap", task.index, task.label)
            gaps.append(task)
            continue

        if key is None or (isinstance(key, tuple) and any(v is None for v in key)):
            logger.warning("gap_analysis: task %d (%s) missing depth or concurrency in run_args", task.index, task.label)
            gaps.append(task)
            continue

        if key not in observed:
            gaps.append(task)

    return gaps
