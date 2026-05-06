"""Tests for consolidate_results and gap_analysis in aggregator.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sparkrun.benchmarking.aggregator import consolidate_results, gap_analysis
from sparkrun.benchmarking.scheduler import BenchTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_result(runs_dir: Path, filename: str, data: dict) -> Path:
    """Write a JSON result file into runs_dir and return its path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    p = runs_dir / filename
    p.write_text(json.dumps(data))
    return p


def _make_result(model: str, concurrency: int, context_size: int) -> dict:
    """Return a minimal llama-benchy-shaped JSON dict for one benchmark entry."""
    return {
        "model": model,
        "max_concurrency": concurrency,
        "benchmarks": [
            {
                "concurrency": concurrency,
                "context_size": context_size,
                "prompt_size": 2048,
                "response_size": 32,
                "is_context_prefill_phase": False,
            }
        ],
    }


def _make_task(index: int, depth: int, concurrency: int) -> BenchTask:
    """Return a minimal BenchTask with run_args set for gap_analysis."""
    return BenchTask(
        index=index,
        label="d=%d c=%d" % (depth, concurrency),
        run_args={"depth": [depth], "concurrency": [concurrency]},
        schedule_entry={"depth": depth, "concurrency": concurrency},
    )


# ---------------------------------------------------------------------------
# consolidate_results — empty / missing
# ---------------------------------------------------------------------------


def test_consolidate_empty_dir_no_runs_subdir(tmp_path: Path):
    """consolidate_results returns empty dict when runs/ does not exist."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    state_dir.mkdir()
    result = consolidate_results(state_dir)
    assert result == {"model": "", "max_concurrency": 0, "benchmarks": []}


def test_consolidate_empty_runs_dir(tmp_path: Path):
    """consolidate_results returns empty dict when runs/ exists but has no JSON files."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    (state_dir / "runs").mkdir(parents=True)
    result = consolidate_results(state_dir)
    assert result == {"model": "", "max_concurrency": 0, "benchmarks": []}


# ---------------------------------------------------------------------------
# consolidate_results — merging
# ---------------------------------------------------------------------------


def test_consolidate_multiple_files_correct_order(tmp_path: Path):
    """Multiple JSON files are merged in numeric-prefix order; model comes from first."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    runs_dir = state_dir / "runs"

    _write_result(runs_dir, "000_d0_c1.json", _make_result("org/model-a", concurrency=1, context_size=0))
    _write_result(runs_dir, "001_d4096_c2.json", _make_result("org/model-b", concurrency=2, context_size=4096))
    _write_result(runs_dir, "002_d0_c5.json", _make_result("org/model-c", concurrency=5, context_size=0))

    result = consolidate_results(state_dir)

    # model taken from the first file
    assert result["model"] == "org/model-a"
    # All benchmark entries concatenated in order
    assert len(result["benchmarks"]) == 3
    assert result["benchmarks"][0]["context_size"] == 0
    assert result["benchmarks"][1]["context_size"] == 4096
    assert result["benchmarks"][2]["context_size"] == 0


def test_consolidate_max_concurrency_from_benchmark_rows(tmp_path: Path):
    """max_concurrency is the max of all observed concurrency values across all merged benchmarks."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    runs_dir = state_dir / "runs"

    # First file has concurrency=1 in its benchmarks row; second has concurrency=8
    _write_result(runs_dir, "000_d0_c1.json", _make_result("org/model", concurrency=1, context_size=0))
    _write_result(runs_dir, "001_d0_c8.json", _make_result("org/model", concurrency=8, context_size=0))

    result = consolidate_results(state_dir)
    assert result["max_concurrency"] == 8


# ---------------------------------------------------------------------------
# consolidate_results — malformed file is skipped
# ---------------------------------------------------------------------------


def test_consolidate_malformed_json_skipped(tmp_path: Path, caplog):
    """A malformed JSON file is skipped with a warning; other files still merge."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    runs_dir = state_dir / "runs"
    runs_dir.mkdir(parents=True)

    _write_result(runs_dir, "000_d0_c1.json", _make_result("org/model", concurrency=1, context_size=0))
    (runs_dir / "001_d4096_c2.json").write_text("THIS IS NOT VALID JSON {{{{")
    _write_result(runs_dir, "002_d0_c5.json", _make_result("org/model", concurrency=5, context_size=0))

    with caplog.at_level(logging.WARNING, logger="sparkrun.benchmarking.aggregator"):
        result = consolidate_results(state_dir)

    # Bad file skipped; 2 valid files merged
    assert len(result["benchmarks"]) == 2
    assert result["model"] == "org/model"

    # Warning was logged for the bad file
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("001_d4096_c2.json" in m for m in warning_messages), "Expected warning mentioning the bad file, got: %s" % warning_messages


# ---------------------------------------------------------------------------
# gap_analysis
# ---------------------------------------------------------------------------


def test_gap_analysis_no_gaps(tmp_path: Path):
    """gap_analysis returns empty list when all tasks are present in consolidated."""
    tasks = [
        _make_task(0, depth=0, concurrency=1),
        _make_task(1, depth=4096, concurrency=2),
    ]
    consolidated = {
        "model": "org/model",
        "max_concurrency": 2,
        "benchmarks": [
            {"context_size": 0, "concurrency": 1},
            {"context_size": 4096, "concurrency": 2},
        ],
    }
    gaps = gap_analysis(tasks, consolidated)
    assert gaps == []


def test_gap_analysis_detects_missing_task(tmp_path: Path):
    """gap_analysis returns tasks whose (depth, concurrency) pair is missing."""
    tasks = [
        _make_task(0, depth=0, concurrency=1),
        _make_task(1, depth=4096, concurrency=2),
    ]
    # Only task 0 is present in results
    consolidated = {
        "model": "org/model",
        "max_concurrency": 1,
        "benchmarks": [
            {"context_size": 0, "concurrency": 1},
        ],
    }
    gaps = gap_analysis(tasks, consolidated)
    assert len(gaps) == 1
    assert gaps[0].index == 1


def test_gap_analysis_tasks_missing_run_args_flagged(tmp_path: Path, caplog):
    """Tasks with missing depth or concurrency in run_args are flagged as gaps with a warning."""
    tasks = [
        BenchTask(index=0, label="bad-task", run_args={}, schedule_entry={}),
    ]
    consolidated = {"model": "org/model", "max_concurrency": 0, "benchmarks": []}

    with caplog.at_level(logging.WARNING, logger="sparkrun.benchmarking.aggregator"):
        gaps = gap_analysis(tasks, consolidated)

    assert len(gaps) == 1
    assert gaps[0].index == 0

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("missing depth or concurrency" in m for m in warning_messages), (
        "Expected warning about missing run_args, got: %s" % warning_messages
    )
