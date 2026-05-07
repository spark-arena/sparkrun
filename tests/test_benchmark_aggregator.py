"""Tests for consolidate_results and gap_analysis in aggregator.py.

The aggregator delegates schema knowledge to the framework plugin.  Most tests
here drive the delegation contract via a small ``_FakeFW`` stub; one test uses
the real :class:`LlamaBenchyFramework` to lock in the end-to-end shape that
llama-benchy emits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sparkrun.benchmarking.aggregator import consolidate_results, gap_analysis
from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
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


class _FakeFW:
    """Minimal stub plugin exercising the aggregator's delegation contract."""

    def __init__(self) -> None:
        self.consolidate_calls: list[list[dict[str, Any]]] = []

    def consolidate_per_task_results(self, per_task_jsons: list[dict[str, Any]]) -> dict[str, Any]:
        self.consolidate_calls.append(per_task_jsons)
        return {"runs": list(per_task_jsons)}

    def task_coverage_key(self, task: BenchTask) -> Any:
        return task.index

    def consolidated_coverage_keys(self, consolidated: dict[str, Any]) -> set[Any]:
        return set(range(len(consolidated.get("runs") or [])))


# ---------------------------------------------------------------------------
# consolidate_results — delegation
# ---------------------------------------------------------------------------


def test_consolidate_empty_dir_no_runs_subdir(tmp_path: Path):
    """consolidate_results delegates an empty list when runs/ does not exist."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    state_dir.mkdir()
    fw = _FakeFW()
    result = consolidate_results(state_dir, fw)
    assert fw.consolidate_calls == [[]]
    assert result == {"runs": []}


def test_consolidate_empty_runs_dir(tmp_path: Path):
    """consolidate_results delegates an empty list when runs/ has no JSON files."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    (state_dir / "runs").mkdir(parents=True)
    fw = _FakeFW()
    result = consolidate_results(state_dir, fw)
    assert fw.consolidate_calls == [[]]
    assert result == {"runs": []}


def test_consolidate_preserves_filename_index_ordering(tmp_path: Path):
    """Files are read in numeric-prefix order regardless of suffix."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    runs_dir = state_dir / "runs"
    _write_result(runs_dir, "002_d0_c5.json", {"tag": "third"})
    _write_result(runs_dir, "000.json", {"tag": "first"})
    _write_result(runs_dir, "001_d4096_c2.json", {"tag": "second"})

    fw = _FakeFW()
    consolidate_results(state_dir, fw)

    # The plugin should see exactly one call with the dicts in numeric order.
    assert len(fw.consolidate_calls) == 1
    tags = [d.get("tag") for d in fw.consolidate_calls[0]]
    assert tags == ["first", "second", "third"]


def test_consolidate_malformed_json_skipped(tmp_path: Path, caplog):
    """A malformed JSON file is skipped with a warning; remaining files are forwarded."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    runs_dir = state_dir / "runs"
    runs_dir.mkdir(parents=True)

    _write_result(runs_dir, "000_d0_c1.json", {"ok": 1})
    (runs_dir / "001_d4096_c2.json").write_text("THIS IS NOT VALID JSON {{{{")
    _write_result(runs_dir, "002_d0_c5.json", {"ok": 2})

    fw = _FakeFW()
    with caplog.at_level(logging.WARNING, logger="sparkrun.benchmarking.aggregator"):
        consolidate_results(state_dir, fw)

    # Only the 2 valid dicts forwarded
    assert len(fw.consolidate_calls) == 1
    assert len(fw.consolidate_calls[0]) == 2

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("001_d4096_c2.json" in m for m in warning_messages), "Expected warning mentioning the bad file, got: %s" % warning_messages


def test_consolidate_non_dict_top_level_skipped(tmp_path: Path, caplog):
    """A JSON file whose top level is not a dict is skipped with a warning."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    runs_dir = state_dir / "runs"
    runs_dir.mkdir(parents=True)

    _write_result(runs_dir, "000_d0_c1.json", {"ok": 1})
    (runs_dir / "001_d4096_c2.json").write_text(json.dumps([1, 2, 3]))  # list, not dict

    fw = _FakeFW()
    with caplog.at_level(logging.WARNING, logger="sparkrun.benchmarking.aggregator"):
        consolidate_results(state_dir, fw)

    assert len(fw.consolidate_calls[0]) == 1
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not a dict" in m for m in warning_messages)


# ---------------------------------------------------------------------------
# consolidate_results — end-to-end with LlamaBenchyFramework
# ---------------------------------------------------------------------------


def test_consolidate_llama_benchy_end_to_end(tmp_path: Path):
    """Real LlamaBenchyFramework: model from first file, max_concurrency from rows, benchmarks concatenated."""
    state_dir = tmp_path / "bench_aabbccddeeff"
    runs_dir = state_dir / "runs"

    _write_result(runs_dir, "000_d0_c1.json", _make_result("org/model-a", concurrency=1, context_size=0))
    _write_result(runs_dir, "001_d4096_c8.json", _make_result("org/model-b", concurrency=8, context_size=4096))
    _write_result(runs_dir, "002_d0_c5.json", _make_result("org/model-c", concurrency=5, context_size=0))

    fw = LlamaBenchyFramework()
    result = consolidate_results(state_dir, fw)

    assert result["model"] == "org/model-a"
    assert result["max_concurrency"] == 8
    assert len(result["benchmarks"]) == 3
    assert result["benchmarks"][0]["context_size"] == 0
    assert result["benchmarks"][1]["context_size"] == 4096
    assert result["benchmarks"][2]["context_size"] == 0


# ---------------------------------------------------------------------------
# gap_analysis
# ---------------------------------------------------------------------------


def test_gap_analysis_no_gaps_with_stub_fw():
    """Stub FW: when every task index appears in consolidated_coverage_keys, no gaps."""
    tasks = [_make_task(0, depth=0, concurrency=1), _make_task(1, depth=4096, concurrency=2)]
    consolidated = {"runs": [{"a": 1}, {"a": 2}]}
    fw = _FakeFW()
    gaps = gap_analysis(tasks, consolidated, fw)
    assert gaps == []


def test_gap_analysis_detects_missing_with_stub_fw():
    """Stub FW: a task whose index is not in consolidated_coverage_keys is a gap."""
    tasks = [_make_task(0, depth=0, concurrency=1), _make_task(1, depth=4096, concurrency=2)]
    consolidated = {"runs": [{"a": 1}]}  # only index 0 covered
    fw = _FakeFW()
    gaps = gap_analysis(tasks, consolidated, fw)
    assert len(gaps) == 1
    assert gaps[0].index == 1


def test_gap_analysis_no_gaps_with_llama_benchy():
    """Real llama-benchy plugin: (depth, concurrency) keys derived from run_args / benchmarks."""
    fw = LlamaBenchyFramework()
    tasks = [_make_task(0, depth=0, concurrency=1), _make_task(1, depth=4096, concurrency=2)]
    consolidated = {
        "model": "org/model",
        "max_concurrency": 2,
        "benchmarks": [
            {"context_size": 0, "concurrency": 1},
            {"context_size": 4096, "concurrency": 2},
        ],
    }
    gaps = gap_analysis(tasks, consolidated, fw)
    assert gaps == []


def test_gap_analysis_detects_missing_with_llama_benchy():
    """Real llama-benchy plugin: tasks missing from benchmarks[] are surfaced as gaps."""
    fw = LlamaBenchyFramework()
    tasks = [_make_task(0, depth=0, concurrency=1), _make_task(1, depth=4096, concurrency=2)]
    consolidated = {
        "model": "org/model",
        "max_concurrency": 1,
        "benchmarks": [{"context_size": 0, "concurrency": 1}],
    }
    gaps = gap_analysis(tasks, consolidated, fw)
    assert len(gaps) == 1
    assert gaps[0].index == 1


def test_gap_analysis_tasks_missing_run_args_flagged(caplog):
    """A llama-benchy task with empty run_args yields a (None, None) coverage key → gap + warning."""
    fw = LlamaBenchyFramework()
    tasks = [BenchTask(index=0, label="bad-task", run_args={}, schedule_entry={})]
    consolidated = {"model": "org/model", "max_concurrency": 0, "benchmarks": []}

    with caplog.at_level(logging.WARNING, logger="sparkrun.benchmarking.aggregator"):
        gaps = gap_analysis(tasks, consolidated, fw)

    assert len(gaps) == 1
    assert gaps[0].index == 0

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("missing depth or concurrency" in m for m in warning_messages), (
        "Expected warning about missing run_args, got: %s" % warning_messages
    )
