"""Tests for the new BenchmarkingPlugin methods on LlamaBenchyFramework.

Locks in the framework-specific schema knowledge that was moved out of the
generic aggregator / progress_ui modules.
"""

from __future__ import annotations

from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
from sparkrun.benchmarking.scheduler import BenchTask


def _task(depth: int | None, conc: int | None) -> BenchTask:
    run_args: dict = {}
    if depth is not None:
        run_args["depth"] = [depth]
    if conc is not None:
        run_args["concurrency"] = [conc]
    return BenchTask(index=0, label="t", run_args=run_args, schedule_entry={})


# ---------------------------------------------------------------------------
# result_filename_suffix
# ---------------------------------------------------------------------------


def test_result_filename_suffix_with_depth_and_conc():
    fw = LlamaBenchyFramework()
    assert fw.result_filename_suffix(_task(0, 1)) == "_d0_c1"
    assert fw.result_filename_suffix(_task(4096, 8)) == "_d4096_c8"


def test_result_filename_suffix_empty_when_args_missing():
    fw = LlamaBenchyFramework()
    assert fw.result_filename_suffix(_task(None, 1)) == ""
    assert fw.result_filename_suffix(_task(0, None)) == ""
    assert fw.result_filename_suffix(_task(None, None)) == ""


# ---------------------------------------------------------------------------
# consolidate_per_task_results
# ---------------------------------------------------------------------------


def test_consolidate_per_task_results_merges_three_files():
    fw = LlamaBenchyFramework()
    per_task = [
        {
            "model": "org/model-a",
            "max_concurrency": 1,
            "benchmarks": [{"context_size": 0, "concurrency": 1}],
        },
        {
            "model": "org/model-b",
            "max_concurrency": 8,
            "benchmarks": [{"context_size": 4096, "concurrency": 8}],
        },
        {
            "model": "org/model-c",
            "max_concurrency": 2,
            "benchmarks": [{"context_size": 0, "concurrency": 2}],
        },
    ]
    out = fw.consolidate_per_task_results(per_task)
    assert out["model"] == "org/model-a", "model should be from first non-empty entry"
    assert out["max_concurrency"] == 8
    assert len(out["benchmarks"]) == 3
    assert out["benchmarks"][0]["context_size"] == 0
    assert out["benchmarks"][1]["context_size"] == 4096
    assert out["benchmarks"][2]["context_size"] == 0


def test_consolidate_per_task_results_empty_input():
    fw = LlamaBenchyFramework()
    out = fw.consolidate_per_task_results([])
    assert out == {"model": "", "max_concurrency": 0, "benchmarks": []}


def test_consolidate_per_task_results_skips_non_dict_entries():
    fw = LlamaBenchyFramework()
    per_task: list = [
        {"model": "org/m", "max_concurrency": 1, "benchmarks": [{"context_size": 0, "concurrency": 1}]},
        "garbage",  # type: ignore[list-item]
        {"model": "org/m2", "max_concurrency": 2, "benchmarks": [{"context_size": 1, "concurrency": 2}]},
    ]
    out = fw.consolidate_per_task_results(per_task)
    assert len(out["benchmarks"]) == 2


# ---------------------------------------------------------------------------
# task_coverage_key / consolidated_coverage_keys
# ---------------------------------------------------------------------------


def test_task_coverage_key():
    fw = LlamaBenchyFramework()
    assert fw.task_coverage_key(_task(0, 1)) == (0, 1)
    assert fw.task_coverage_key(_task(4096, 8)) == (4096, 8)
    assert fw.task_coverage_key(_task(None, 1)) == (None, 1)


def test_consolidated_coverage_keys():
    fw = LlamaBenchyFramework()
    consolidated = {
        "model": "org/m",
        "max_concurrency": 8,
        "benchmarks": [
            {"context_size": 0, "concurrency": 1},
            {"context_size": 4096, "concurrency": 8},
            {"context_size": 0, "concurrency": 1},  # duplicates dedupe via set
        ],
    }
    keys = fw.consolidated_coverage_keys(consolidated)
    assert keys == {(0, 1), (4096, 8)}


def test_consolidated_coverage_keys_skips_entries_with_missing_fields():
    fw = LlamaBenchyFramework()
    consolidated = {
        "benchmarks": [
            {"context_size": 0},  # no concurrency
            {"concurrency": 1},  # no context_size
            {"context_size": 0, "concurrency": 1},
        ]
    }
    assert fw.consolidated_coverage_keys(consolidated) == {(0, 1)}


# ---------------------------------------------------------------------------
# progress_table_spec.rows_from_consolidated
# ---------------------------------------------------------------------------


def test_progress_table_spec_columns():
    """The 6-column shape is preserved: depth / conc / pp t/s / tg t/s / ttfr ms / runs."""
    fw = LlamaBenchyFramework()
    spec = fw.progress_table_spec()
    names = [c.name for c in spec.columns]
    assert names == ["depth", "conc", "pp t/s", "tg t/s", "ttfr ms", "runs"]


def test_progress_table_spec_rows_from_consolidated_simple():
    """Single non-prefill row produces (depth, conc, pp_mean, tg_mean, ttfr_mean, 1)."""
    fw = LlamaBenchyFramework()
    spec = fw.progress_table_spec()
    consolidated = {
        "model": "org/m",
        "max_concurrency": 1,
        "benchmarks": [
            {
                "context_size": 0,
                "concurrency": 1,
                "is_context_prefill_phase": False,
                "pp_throughput": {"mean": 1500.0, "values": [1500.0]},
                "tg_throughput": {"mean": 75.0, "values": [75.0]},
                "ttfr": {"mean": 200.0, "values": [200.0]},
            }
        ],
    }
    rows = spec.rows_from_consolidated(consolidated)
    assert rows == [(0, 1, 1500.0, 75.0, 200.0, 1)]


def test_progress_table_spec_rows_skip_prefill():
    """Context-prefill entries register their key but contribute no metric values."""
    fw = LlamaBenchyFramework()
    spec = fw.progress_table_spec()
    consolidated = {
        "benchmarks": [
            {
                "context_size": 4096,
                "concurrency": 1,
                "is_context_prefill_phase": True,
                "pp_throughput": {"mean": 999.0},
            },
            {
                "context_size": 4096,
                "concurrency": 1,
                "is_context_prefill_phase": False,
                "pp_throughput": {"mean": 1500.0, "values": [1500.0, 1501.0]},
                "tg_throughput": {"mean": 75.0, "values": [75.0, 75.5]},
                "ttfr": {"mean": 200.0, "values": [200.0, 201.0]},
            },
        ]
    }
    rows = spec.rows_from_consolidated(consolidated)
    # Single (depth, conc) key with metrics from the non-prefill entry only,
    # and runs_count == 2 (length of values[]).
    assert rows == [(4096, 1, 1500.0, 75.0, 200.0, 2)]


def test_progress_table_spec_row_key_is_depth_conc():
    fw = LlamaBenchyFramework()
    spec = fw.progress_table_spec()
    assert spec.row_key((4096, 8, 1.0, 2.0, 3.0, 1)) == (4096, 8)


def test_progress_table_spec_format_cell_floats_and_ints():
    fw = LlamaBenchyFramework()
    spec = fw.progress_table_spec()
    columns = {c.name: c for c in spec.columns}

    assert spec.format_cell is not None
    assert spec.format_cell(1500.0, columns["pp t/s"]) == "1500.0"
    assert spec.format_cell(None, columns["tg t/s"]) == "…"
    assert spec.format_cell(4096, columns["depth"]) == "4096"
    assert spec.format_cell(1, columns["runs"]) == "1"
