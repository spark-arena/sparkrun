"""User-facing progress display for scheduled benchmark runs.

Provides :class:`BenchmarkProgressUI`, a context manager that wraps either a
rich-based live layout (progress bar + results table) or a plain-stdout fallback
when ``rich`` is not importable.

Auto-selection happens at module import time via :data:`_HAS_RICH`.

The displayed table is fully framework-driven via the plugin's
:class:`~sparkrun.benchmarking.base.ProgressTableSpec` — column definitions
and per-row data come from the plugin, so this module has no
framework-specific schema knowledge.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sparkrun.benchmarking.base import BenchmarkingPlugin, ProgressTableSpec

# ---------------------------------------------------------------------------
# Rich availability probe
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_row(spec: "ProgressTableSpec", row: tuple[Any, ...]) -> list[str]:
    """Apply the spec's optional per-cell formatter to each value in *row*."""
    fmt = spec.format_cell
    if fmt is None:
        return [str(v) for v in row]
    return [fmt(v, col) for v, col in zip(row, spec.columns)]


def _safe_console_height(console: "Console") -> int:
    """Return console height with a defensive fallback for non-tty environments."""
    try:
        h = int(console.size.height)
    except Exception:  # pragma: no cover — Rich Console.size raises in some envs
        return 24
    return h if h > 0 else 24


# ---------------------------------------------------------------------------
# Rich backend
# ---------------------------------------------------------------------------


class _RichUI:
    """Live terminal UI built on ``rich``.

    Renders a progress bar (task count + ETA) and a live results table below it.
    """

    def __init__(self, *, total_tasks: int, benchmark_id: str, fw: "BenchmarkingPlugin", title: str = "") -> None:
        self._total = total_tasks
        self._benchmark_id = benchmark_id
        self._title = title or benchmark_id
        self._fw = fw
        self._spec = fw.progress_table_spec()
        self._seen_rows: dict[Any, tuple[Any, ...]] = {}

        self._console = Console(stderr=False)

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            description=f"Benchmark {self._title}",
            total=total_tasks,
        )

        self._last_consolidated: dict[str, Any] = {}
        self._table: Table = self._make_empty_table()
        self._live = Live(
            self._render_group(),
            console=self._console,
            refresh_per_second=4,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_empty_table(self) -> "Table":
        t = Table(show_header=True, header_style="bold magenta", show_lines=False, expand=False)
        for col in self._spec.columns:
            kwargs: dict[str, Any] = {"justify": col.justify, "no_wrap": True}
            if col.style:
                kwargs["style"] = col.style
            t.add_column(col.name, **kwargs)
        return t

    def _render_group(self) -> "Table":
        """Return a single renderable that stacks progress + table."""
        outer = Table.grid(padding=0)
        outer.add_row(self._progress)
        outer.add_row(self._table)
        return outer

    def _refresh(self) -> None:
        self._live.update(self._render_group())

    def _window_cap(self) -> int:
        """Max rows displayable in the live area without forcing scroll."""
        return max(10, _safe_console_height(self._console) - 8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __enter__(self) -> "_RichUI":
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        # Final scrollback dump: emit a complete copy of the table so the
        # post-run terminal contains every row regardless of live windowing.
        try:
            full_table = self._build_table_from_rows(self._spec.rows_from_consolidated(self._last_consolidated))
        except Exception:  # pragma: no cover — defensive: don't break exit on plugin bug
            full_table = None
        self._live.__exit__(*exc)
        if full_table is not None:
            self._console.print(full_table)

    def start_task(self, idx: int, label: str) -> None:
        self._progress.update(self._task_id, description=f"[{idx + 1}/{self._total}] {label}")
        self._refresh()

    def end_task(self, idx: int, success: bool, duration_s: float | None = None) -> None:
        dur_str = f" ({duration_s:.1f}s)" if duration_s is not None else ""
        status = "[green]done[/green]" if success else "[red]failed[/red]"
        self._console.log(f"Task {idx + 1}/{self._total} {status}{dur_str}")
        self._progress.advance(self._task_id)
        self._refresh()

    def _build_table_from_rows(self, rows: list[tuple[Any, ...]]) -> "Table":
        """Build a Rich Table that contains every row (no windowing)."""
        t = self._make_empty_table()
        for row in rows:
            t.add_row(*_format_row(self._spec, row))
        return t

    def update_results_table(self, consolidated: dict[str, Any]) -> None:
        self._last_consolidated = consolidated
        rows = self._spec.rows_from_consolidated(consolidated)

        # Emit a per-completion scrollback snapshot for new or updated rows.
        # Survives the Live region because console.print() is logged above it.
        # Re-emit when the row tuple changes (e.g., a gap re-run added new
        # measurement repetitions for the same coverage key).
        changed_rows: list[tuple[Any, ...]] = []
        for row in rows:
            try:
                key = self._spec.row_key(row)
            except Exception:  # pragma: no cover — defensive: bad row_key shouldn't crash live UI
                key = row
            if self._seen_rows.get(key) != row:
                self._seen_rows[key] = row
                changed_rows.append(row)
        for row in changed_rows:
            cells = _format_row(self._spec, row)
            self._console.print("  ".join(cells))

        # Render the live table with a window cap so the live region does not
        # exceed terminal height and start clipping silently.
        cap = self._window_cap()
        if len(rows) > cap:
            display_rows = rows[-cap:]
        else:
            display_rows = rows

        new_table = self._make_empty_table()
        for row in display_rows:
            new_table.add_row(*_format_row(self._spec, row))
        self._table = new_table
        self._refresh()

    def log(self, message: str) -> None:
        self._console.log(message)


# ---------------------------------------------------------------------------
# Plain-stdout backend
# ---------------------------------------------------------------------------


class _PlainUI:
    """Minimal fallback when ``rich`` is unavailable."""

    def __init__(self, *, total_tasks: int, benchmark_id: str, fw: "BenchmarkingPlugin", title: str = "") -> None:
        self._total = total_tasks
        self._benchmark_id = benchmark_id
        self._title = title or benchmark_id
        self._fw = fw
        self._spec = fw.progress_table_spec()
        self._seen_rows: dict[Any, tuple[Any, ...]] = {}
        self._header_emitted = False
        self._last_consolidated: dict[str, Any] = {}

    def __enter__(self) -> "_PlainUI":
        print(f"=== Benchmark {self._title} — {self._total} tasks ===")
        return self

    def __exit__(self, *exc: Any) -> None:
        # Final full snapshot so scrollback contains everything.
        try:
            rows = self._spec.rows_from_consolidated(self._last_consolidated)
        except Exception:  # pragma: no cover — defensive
            rows = []
        if rows:
            print("---")
            self._print_header()
            for row in rows:
                self._print_row(row)
        print("=== Benchmark complete ===")

    def start_task(self, idx: int, label: str) -> None:
        print(f"[{idx + 1}/{self._total}] running {label}")

    def end_task(self, idx: int, success: bool, duration_s: float | None = None) -> None:
        status = "ok" if success else "FAILED"
        dur_str = f" ({duration_s:.1f}s)" if duration_s is not None else ""
        print(f"[{idx + 1}/{self._total}] {status}{dur_str}")

    def _print_header(self) -> None:
        cols = self._spec.columns
        print("  ".join(c.name for c in cols))

    def _print_row(self, row: tuple[Any, ...]) -> None:
        print("  ".join(_format_row(self._spec, row)))

    def update_results_table(self, consolidated: dict[str, Any]) -> None:
        self._last_consolidated = consolidated
        rows = self._spec.rows_from_consolidated(consolidated)

        changed_rows: list[tuple[Any, ...]] = []
        for row in rows:
            try:
                key = self._spec.row_key(row)
            except Exception:  # pragma: no cover — defensive
                key = row
            if self._seen_rows.get(key) != row:
                self._seen_rows[key] = row
                changed_rows.append(row)

        if not changed_rows:
            return
        if not self._header_emitted:
            self._print_header()
            self._header_emitted = True
        for row in changed_rows:
            self._print_row(row)

    def log(self, message: str) -> None:
        print(message)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class BenchmarkProgressUI:
    """Context-manager facade that delegates to the appropriate backend.

    Selects :class:`_RichUI` when ``rich`` is available; falls back to
    :class:`_PlainUI` otherwise.

    Args:
        total_tasks: Total number of benchmark tasks in the schedule.
        benchmark_id: Stable identifier for this run (e.g. ``"bench_8a4f2c0d1e9b"``).
        fw: The benchmarking plugin — supplies the column / row spec for the
            progress table.
        title: Optional display title shown next to the benchmark id.  Defaults
            to *benchmark_id* when omitted.
    """

    def __init__(self, *, total_tasks: int, benchmark_id: str, fw: "BenchmarkingPlugin", title: str = "") -> None:
        self._backend: _RichUI | _PlainUI
        if _HAS_RICH:
            self._backend = _RichUI(total_tasks=total_tasks, benchmark_id=benchmark_id, fw=fw, title=title)
        else:
            self._backend = _PlainUI(total_tasks=total_tasks, benchmark_id=benchmark_id, fw=fw, title=title)

    def __enter__(self) -> "BenchmarkProgressUI":
        self._backend.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._backend.__exit__(*exc)

    def start_task(self, idx: int, label: str) -> None:
        """Signal that task *idx* (0-based) with human label *label* has started."""
        self._backend.start_task(idx, label)

    def end_task(self, idx: int, success: bool, duration_s: float | None = None) -> None:
        """Signal that task *idx* has finished.

        Args:
            idx: 0-based task index.
            success: ``True`` if the task completed without error.
            duration_s: Wall-clock seconds the task took, if known.
        """
        self._backend.end_task(idx, success, duration_s)

    def update_results_table(self, consolidated: dict[str, Any]) -> None:
        """Rebuild the live results table from a framework-shaped consolidated dict.

        Rows and columns come from the plugin's
        :meth:`~sparkrun.benchmarking.base.BenchmarkingPlugin.progress_table_spec`.
        Calling this method is idempotent — the live region simply replaces the
        rendered table while previously-seen rows accumulate in scrollback.
        """
        self._backend.update_results_table(consolidated)

    def log(self, message: str) -> None:
        """Emit *message* above the live area (or to stdout on the plain backend)."""
        self._backend.log(message)


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    from sparkrun.benchmarking.base import (
        BenchmarkingPlugin,
        ProgressColumn,
        ProgressTableSpec,
    )

    class _StubPlugin(BenchmarkingPlugin):
        framework_name = "smoke-test"

        def check_prerequisites(self) -> list[str]:
            return []

        def build_benchmark_command(self, target_url, model, args, result_file=None):  # pragma: no cover
            return []

        def parse_results(self, stdout, stderr, result_file=None):  # pragma: no cover
            return {}

        def progress_table_spec(self) -> ProgressTableSpec:
            return ProgressTableSpec(
                columns=[
                    ProgressColumn(name="depth", justify="right", style="cyan"),
                    ProgressColumn(name="conc", justify="right", style="cyan"),
                    ProgressColumn(name="metric", justify="right"),
                ],
                rows_from_consolidated=lambda c: [(e["depth"], e["conc"], e["metric"]) for e in c.get("rows", [])],
                row_key=lambda r: (r[0], r[1]),
            )

    fw = _StubPlugin()

    # ~30 distinct (depth, concurrency) pairs to exceed a typical 24-row term.
    _DEMO_TASKS = []
    for d in (0, 1024, 2048, 4096, 8192, 16384):
        for c in (1, 2, 5, 8, 16):
            _DEMO_TASKS.append((d, c, "d=%d c=%d" % (d, c)))

    state = {"rows": []}

    print(f"Rich available: {_HAS_RICH}")

    with BenchmarkProgressUI(total_tasks=len(_DEMO_TASKS), benchmark_id="bench_demo0000", fw=fw, title="demo/smoke-test") as ui:
        ui.log("Starting smoke test with %d simulated tasks" % len(_DEMO_TASKS))

        for i, (depth, conc, label) in enumerate(_DEMO_TASKS):
            ui.start_task(i, label)
            time.sleep(0.05)
            state["rows"].append({"depth": depth, "conc": conc, "metric": round(random.uniform(10, 99), 2)})
            ui.end_task(i, success=True, duration_s=0.05)
            ui.update_results_table(state)

        ui.log("All tasks complete.")
