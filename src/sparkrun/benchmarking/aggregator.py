"""Benchmark result aggregation: merge per-task JSON files into a single llama-benchy-shaped dict."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.benchmarking.scheduler import BenchTask

logger = logging.getLogger(__name__)

_RUNS_FILENAME_RE = re.compile(r"^(\d+)_")


def consolidate_results(state_dir: Path) -> dict[str, Any]:
    """Read every JSON file under state_dir/runs/ in schedule-index order
    and merge into one llama-benchy-shaped dict. Uses model name and
    max_concurrency from the first JSON found; concatenates all
    `benchmarks` arrays.

    Returns a minimal empty dict if no per-task files exist yet.
    """
    runs_dir = state_dir / "runs"
    if not runs_dir.is_dir():
        return {"model": "", "max_concurrency": 0, "benchmarks": []}

    json_files = sorted(
        runs_dir.glob("*.json"),
        key=lambda p: int(m.group(1)) if (m := _RUNS_FILENAME_RE.match(p.name)) else 0,
    )

    if not json_files:
        return {"model": "", "max_concurrency": 0, "benchmarks": []}

    model: str = ""
    max_concurrency: int = 0
    all_benchmarks: list[dict[str, Any]] = []
    first_file_loaded = False

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("aggregator: skipping %s — %s", json_path.name, exc)
            continue

        if not isinstance(data, dict):
            logger.warning("aggregator: skipping %s — top-level value is not a dict", json_path.name)
            continue

        if not first_file_loaded:
            model = data.get("model", "")
            first_file_loaded = True

        benchmarks = data.get("benchmarks", [])
        if not isinstance(benchmarks, list):
            logger.warning("aggregator: skipping benchmarks in %s — 'benchmarks' is not a list", json_path.name)
            continue

        for entry in benchmarks:
            if not isinstance(entry, dict):
                continue
            all_benchmarks.append(entry)
            concurrency = entry.get("concurrency", 0)
            if isinstance(concurrency, (int, float)) and concurrency > max_concurrency:
                max_concurrency = int(concurrency)

    return {
        "model": model,
        "max_concurrency": max_concurrency,
        "benchmarks": all_benchmarks,
    }


def gap_analysis(
    task_list: list[BenchTask],
    consolidated: dict[str, Any],
    expected_per_task: int = 1,
) -> list[BenchTask]:
    """Return tasks whose (depth, concurrency) combination is missing from
    the consolidated `benchmarks` array.

    A task is considered a gap if no entry in consolidated["benchmarks"]
    matches both its depth (context_size) and concurrency values.

    Uses run_args["depth"][0] and run_args["concurrency"][0] to determine
    expected (depth, concurrency) per task.
    """
    benchmarks: list[dict[str, Any]] = consolidated.get("benchmarks", [])

    # Build a set of (context_size, concurrency) pairs present in the results.
    observed: set[tuple[Any, Any]] = set()
    for entry in benchmarks:
        if not isinstance(entry, dict):
            continue
        depth = entry.get("context_size")
        concurrency = entry.get("concurrency")
        if depth is not None and concurrency is not None:
            observed.add((depth, concurrency))

    gaps: list[BenchTask] = []
    for task in task_list:
        run_args = task.run_args
        depth_list = run_args.get("depth", [])
        concurrency_list = run_args.get("concurrency", [])
        if not depth_list or not concurrency_list:
            logger.warning("gap_analysis: task %d (%s) missing depth or concurrency in run_args", task.index, task.label)
            gaps.append(task)
            continue
        key = (depth_list[0], concurrency_list[0])
        if key not in observed:
            gaps.append(task)

    return gaps
