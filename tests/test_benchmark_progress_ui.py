"""Tests for BenchmarkProgressUI's plugin-driven column spec, scrollback snapshots,
live-area windowing, and __exit__ snapshot.

These tests use small stub plugins to drive the progress UI without depending
on the llama-benchy schema.  Rich-backed tests force a known terminal size by
patching ``rich.console.Console`` so windowing behavior is deterministic.
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import patch

from rich.console import Console

from sparkrun.benchmarking.base import BenchmarkingPlugin, ProgressColumn, ProgressTableSpec
from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI, _PlainUI, _RichUI


# ---------------------------------------------------------------------------
# Stub plugins
# ---------------------------------------------------------------------------


class _StubFW(BenchmarkingPlugin):
    """A tiny stub plugin with a 2-column spec sourced from ``consolidated["rows"]``."""

    framework_name = "stub"

    def check_prerequisites(self) -> list[str]:  # pragma: no cover
        return []

    def build_benchmark_command(self, target_url, model, args, result_file=None):  # pragma: no cover
        return []

    def parse_results(self, stdout, stderr, result_file=None):  # pragma: no cover
        return {}

    def progress_table_spec(self) -> ProgressTableSpec:
        return ProgressTableSpec(
            columns=[
                ProgressColumn(name="key", justify="right"),
                ProgressColumn(name="value", justify="right"),
            ],
            rows_from_consolidated=lambda c: [(r["key"], r["value"]) for r in c.get("rows", [])],
            row_key=lambda r: r[0],
        )


class _BigStubFW(BenchmarkingPlugin):
    """Stub plugin that emits N rows on demand."""

    framework_name = "big-stub"

    def check_prerequisites(self) -> list[str]:  # pragma: no cover
        return []

    def build_benchmark_command(self, target_url, model, args, result_file=None):  # pragma: no cover
        return []

    def parse_results(self, stdout, stderr, result_file=None):  # pragma: no cover
        return {}

    def progress_table_spec(self) -> ProgressTableSpec:
        return ProgressTableSpec(
            columns=[ProgressColumn(name="i", justify="right")],
            rows_from_consolidated=lambda c: [(i,) for i in range(c.get("n", 0))],
            row_key=lambda r: r[0],
        )


# ---------------------------------------------------------------------------
# Test 1 — table renders with the spec's columns; rows accumulate
# ---------------------------------------------------------------------------


def test_rich_backend_columns_match_spec():
    """The Rich table built by ``_make_empty_table`` matches the spec column count and names."""
    fw = _StubFW()
    backend = _RichUI(total_tasks=5, benchmark_id="bench_test01", fw=fw)
    table = backend._make_empty_table()
    assert len(table.columns) == 2
    column_headers = [c.header for c in table.columns]
    assert column_headers == ["key", "value"]


def test_rich_backend_update_results_table_replaces_rows():
    """Each ``update_results_table`` call rebuilds the live table with the spec's rows."""
    fw = _StubFW()
    backend = _RichUI(total_tasks=5, benchmark_id="bench_test01", fw=fw)
    backend.update_results_table({"rows": [{"key": "a", "value": "1"}]})
    assert backend._table.row_count == 1
    backend.update_results_table({"rows": [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]})
    assert backend._table.row_count == 2


# ---------------------------------------------------------------------------
# Test 2 — scrollback snapshots emitted on each new completion
# ---------------------------------------------------------------------------


def test_rich_backend_scrollback_snapshot_per_new_row():
    """Each new (row_key) seen by ``update_results_table`` triggers a console.print()."""
    fw = _StubFW()
    backend = _RichUI(total_tasks=5, benchmark_id="bench_test01", fw=fw)

    print_calls: list[Any] = []
    original_print = backend._console.print

    def _capturing_print(*args, **kwargs):
        print_calls.append(args)
        return original_print(*args, **kwargs)

    backend._console.print = _capturing_print  # type: ignore[method-assign]

    rows: list[dict[str, Any]] = []
    for i in range(5):
        rows.append({"key": "k%d" % i, "value": str(i)})
        backend.update_results_table({"rows": list(rows)})

    # Exactly 5 distinct (row_key) values were introduced over 5 calls,
    # so we expect 5 console.print() snapshot calls.
    assert len(print_calls) == 5, "Expected one scrollback snapshot per new row, got %d" % len(print_calls)


def test_rich_backend_scrollback_does_not_reprint_existing_rows():
    """Re-emitting the same rows must not produce duplicate scrollback snapshots."""
    fw = _StubFW()
    backend = _RichUI(total_tasks=5, benchmark_id="bench_test01", fw=fw)

    print_calls: list[Any] = []
    original_print = backend._console.print

    def _capturing_print(*args, **kwargs):
        print_calls.append(args)
        return original_print(*args, **kwargs)

    backend._console.print = _capturing_print  # type: ignore[method-assign]

    backend.update_results_table({"rows": [{"key": "a", "value": "1"}]})
    backend.update_results_table({"rows": [{"key": "a", "value": "1"}]})  # same key
    backend.update_results_table({"rows": [{"key": "a", "value": "1"}]})  # same key

    assert len(print_calls) == 1, "Expected one snapshot for the single distinct row, got %d" % len(print_calls)


def test_rich_backend_scrollback_reprints_when_row_value_changes():
    """When the row tuple for an existing key changes (e.g. gap re-run added repetitions),
    the updated row must re-emit to scrollback so users see the latest data."""
    fw = _StubFW()
    backend = _RichUI(total_tasks=5, benchmark_id="bench_test01", fw=fw)

    print_calls: list[Any] = []
    original_print = backend._console.print

    def _capturing_print(*args, **kwargs):
        print_calls.append(args)
        return original_print(*args, **kwargs)

    backend._console.print = _capturing_print  # type: ignore[method-assign]

    backend.update_results_table({"rows": [{"key": "a", "value": "1"}]})
    backend.update_results_table({"rows": [{"key": "a", "value": "2"}]})  # same key, new value
    backend.update_results_table({"rows": [{"key": "a", "value": "3"}]})  # same key, new value

    assert len(print_calls) == 3, "Expected three snapshots (one per row-tuple change), got %d" % len(print_calls)


# ---------------------------------------------------------------------------
# Test 3 — live-area windowing caps displayed rows on small terminals
# ---------------------------------------------------------------------------


def test_rich_backend_live_area_windowing_caps_rows():
    """When the spec produces more rows than the live cap, only the trailing slice is displayed."""
    fw = _BigStubFW()

    # Force a small terminal: height=20 → cap = max(10, 20-8) = 12.
    fake_console = Console(file=io.StringIO(), force_terminal=True, width=80, height=20)

    with patch("sparkrun.benchmarking.progress_ui.Console", return_value=fake_console):
        backend = _RichUI(total_tasks=50, benchmark_id="bench_test01", fw=fw)

    # Avoid spamming the captured stdout with 50 row prints
    backend._console.print = lambda *a, **kw: None  # type: ignore[method-assign]

    backend.update_results_table({"n": 50})
    cap = backend._window_cap()
    assert cap == 12, "Expected window cap of 12 for height=20 terminal, got %d" % cap
    assert backend._table.row_count <= cap, "Live table row count %d exceeded window cap %d" % (backend._table.row_count, cap)


# ---------------------------------------------------------------------------
# Test 4 — __exit__ emits a final full-table snapshot
# ---------------------------------------------------------------------------


def test_rich_backend_exit_emits_full_snapshot():
    """On ``__exit__``, the full (un-windowed) table is printed to scrollback."""
    fw = _BigStubFW()

    fake_console = Console(file=io.StringIO(), force_terminal=True, width=80, height=20)
    with patch("sparkrun.benchmarking.progress_ui.Console", return_value=fake_console):
        backend = _RichUI(total_tasks=50, benchmark_id="bench_test01", fw=fw)

    print_calls: list[Any] = []
    original_print = backend._console.print

    def _capturing_print(*args, **kwargs):
        print_calls.append(args)
        return original_print(*args, **kwargs)

    backend._console.print = _capturing_print  # type: ignore[method-assign]

    with backend:
        backend.update_results_table({"n": 50})

    # The final snapshot is the last print() call and must be a Rich Table
    # carrying all 50 rows.
    assert print_calls, "Expected at least one print call for final snapshot"
    final_args = print_calls[-1]
    assert final_args, "Final print call had no positional args"
    final = final_args[0]
    # The full snapshot is a rich.table.Table; assert row_count is the full N.
    assert hasattr(final, "row_count"), "Final snapshot is not a Rich Table"
    assert final.row_count == 50, "Final snapshot should contain all 50 rows, got %d" % final.row_count


# ---------------------------------------------------------------------------
# Test 5 — plain backend appends only newly-seen rows on each call
# ---------------------------------------------------------------------------


def test_plain_backend_append_on_change(capsys):
    """Plain backend prints only NEW rows on subsequent ``update_results_table`` calls."""
    fw = _StubFW()
    backend = _PlainUI(total_tasks=3, benchmark_id="bench_test01", fw=fw)

    backend.update_results_table({"rows": [{"key": "a", "value": "1"}]})
    out1 = capsys.readouterr().out
    # Header + first row
    assert "key" in out1 and "value" in out1
    assert "a" in out1 and "1" in out1

    backend.update_results_table({"rows": [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]})
    out2 = capsys.readouterr().out
    # Only the new row should print this time (no header re-emit, no row "a" re-emit)
    assert "b" in out2 and "2" in out2
    assert "a" not in out2, "Plain backend should not re-print already-seen rows, got: %r" % out2

    backend.update_results_table({"rows": [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}, {"key": "c", "value": "3"}]})
    out3 = capsys.readouterr().out
    assert "c" in out3 and "3" in out3
    assert "a" not in out3 and "b" not in out3, "Plain backend should not re-print existing rows, got: %r" % out3


def test_plain_backend_reprints_when_row_value_changes(capsys):
    """Plain backend re-emits a row when its tuple changed for an already-seen key."""
    fw = _StubFW()
    backend = _PlainUI(total_tasks=2, benchmark_id="bench_test01", fw=fw)

    backend.update_results_table({"rows": [{"key": "a", "value": "1"}]})
    capsys.readouterr()  # discard initial output

    backend.update_results_table({"rows": [{"key": "a", "value": "2"}]})
    out = capsys.readouterr().out
    assert "2" in out, "Updated row value should be re-emitted, got: %r" % out


def test_plain_backend_exit_dumps_final_snapshot(capsys):
    """On ``__exit__``, the plain backend prints the full row set as a final snapshot."""
    fw = _StubFW()
    backend = _PlainUI(total_tasks=2, benchmark_id="bench_test01", fw=fw)
    with backend:
        backend.update_results_table({"rows": [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]})

    out = capsys.readouterr().out
    # The "===" final marker must appear and both rows must be present in the output.
    assert "Benchmark complete" in out
    assert "a" in out and "b" in out


# ---------------------------------------------------------------------------
# Public facade smoke test — selects backend, accepts fw=
# ---------------------------------------------------------------------------


def test_facade_accepts_fw_keyword():
    """``BenchmarkProgressUI(...)`` accepts the new ``fw=`` keyword and entering doesn't crash."""
    fw = _StubFW()
    ui = BenchmarkProgressUI(total_tasks=1, benchmark_id="bench_test01", fw=fw, title="t")
    with ui:
        ui.update_results_table({"rows": [{"key": "a", "value": "1"}]})
