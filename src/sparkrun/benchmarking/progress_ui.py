"""User-facing progress display for scheduled benchmark runs.

Provides :class:`BenchmarkProgressUI`, a context manager that wraps either a
rich-based live layout (progress bar + results table) or a plain-stdout fallback
when ``rich`` is not importable.

Auto-selection happens at module import time via :data:`_HAS_RICH`.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

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

_MISSING = object()  # sentinel for "key not present in dict"


def _safe_mean(values: list[float | None]) -> float | None:
    """Return arithmetic mean of non-None values, or None if list is empty."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _fmt_float(value: float | None, decimals: int = 1, missing: str = "…") -> str:
    """Format a float to *decimals* decimal places, or return *missing* if None."""
    if value is None:
        return missing
    return f"{value:.{decimals}f}"


def _build_table_rows(consolidated: dict[str, Any]) -> list[tuple[int, int, float | None, float | None, float | None, int]]:
    """Aggregate consolidated llama-benchy JSON into (depth, conc, pp_ts, tg_ts, ttfr_ms, runs) rows.

    Groups benchmark entries by ``(context_size, concurrency)``.  Only
    non-context-prefill rows contribute to pp/tg/ttfr means (context-prefill
    entries measure cache warm-up and are intentionally excluded from the
    primary table columns).

    Returns rows sorted by (context_size, concurrency).
    """
    benchmarks: list[dict[str, Any]] = consolidated.get("benchmarks") or []

    # group: key -> lists of individual metric values
    pp_vals: dict[tuple[int, int], list[float | None]] = defaultdict(list)
    tg_vals: dict[tuple[int, int], list[float | None]] = defaultdict(list)
    ttfr_vals: dict[tuple[int, int], list[float | None]] = defaultdict(list)
    # `runs` reflects the number of measurement repetitions actually executed.
    # Each non-prefill benchmark entry's throughput dict carries a `values`
    # array of length == --runs; we sum those lengths across entries (a single
    # llama-benchy call yields one non-prefill entry, but the same (d, c)
    # could be re-run later, e.g. via gap re-queue, in which case lengths add).
    runs_count: dict[tuple[int, int], int] = defaultdict(int)
    seen_keys: set[tuple[int, int]] = set()

    for b in benchmarks:
        depth = int(b.get("context_size") or 0)
        conc = int(b.get("concurrency") or 1)
        key = (depth, conc)
        seen_keys.add(key)

        is_prefill = bool(b.get("is_context_prefill_phase"))
        if is_prefill:
            # Skip context-prefill rows for the primary metrics
            continue

        def _mean_val(field_name: str) -> float | None:
            raw = b.get(field_name)
            if raw is None:
                return None
            if isinstance(raw, dict):
                return raw.get("mean")
            return float(raw)

        pp_vals[key].append(_mean_val("pp_throughput"))
        tg_vals[key].append(_mean_val("tg_throughput"))
        ttfr_vals[key].append(_mean_val("ttfr"))

        # Number of measurement repetitions == len(values) on whichever
        # throughput dict carries it; prefer tg_throughput, then pp_throughput.
        run_repetitions = 0
        for field in ("tg_throughput", "pp_throughput"):
            metric = b.get(field)
            if isinstance(metric, dict):
                values = metric.get("values")
                if isinstance(values, list) and values:
                    run_repetitions = len(values)
                    break
        if run_repetitions == 0:
            run_repetitions = 1  # fall back: at least one row was emitted
        runs_count[key] += run_repetitions

    rows = []
    for key in sorted(seen_keys):
        depth, conc = key
        rows.append(
            (
                depth,
                conc,
                _safe_mean(pp_vals.get(key, [])),
                _safe_mean(tg_vals.get(key, [])),
                _safe_mean(ttfr_vals.get(key, [])),
                runs_count.get(key, 0),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Rich backend
# ---------------------------------------------------------------------------


class _RichUI:
    """Live terminal UI built on ``rich``.

    Renders a progress bar (task count + ETA) and a live results table below it.
    """

    def __init__(self, total_tasks: int, benchmark_id: str, title: str = "") -> None:
        self._total = total_tasks
        self._benchmark_id = benchmark_id
        self._title = title or benchmark_id

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

        self._table: Table = self._make_empty_table()
        self._live = Live(
            self._render_group(),
            console=self._console,
            refresh_per_second=4,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_empty_table() -> "Table":
        t = Table(show_header=True, header_style="bold magenta", show_lines=False, expand=False)
        t.add_column("depth", justify="right", style="cyan", no_wrap=True)
        t.add_column("conc", justify="right", style="cyan", no_wrap=True)
        t.add_column("pp t/s", justify="right")
        t.add_column("tg t/s", justify="right")
        t.add_column("ttfr ms", justify="right")
        t.add_column("runs", justify="right")
        return t

    def _render_group(self) -> "Table":
        """Return a single renderable that stacks progress + table."""
        # Rich Group not available in all versions; use a wrapper table instead.
        outer = Table.grid(padding=0)
        outer.add_row(self._progress)
        outer.add_row(self._table)
        return outer

    def _refresh(self) -> None:
        self._live.update(self._render_group())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __enter__(self) -> "_RichUI":
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._live.__exit__(*exc)

    def start_task(self, idx: int, label: str) -> None:
        self._progress.update(self._task_id, description=f"[{idx + 1}/{self._total}] {label}")
        self._refresh()

    def end_task(self, idx: int, success: bool, duration_s: float | None = None) -> None:
        dur_str = f" ({duration_s:.1f}s)" if duration_s is not None else ""
        status = "[green]done[/green]" if success else "[red]failed[/red]"
        self._console.log(f"Task {idx + 1}/{self._total} {status}{dur_str}")
        self._progress.advance(self._task_id)
        self._refresh()

    def update_results_table(self, consolidated: dict[str, Any]) -> None:
        rows = _build_table_rows(consolidated)
        new_table = self._make_empty_table()
        for depth, conc, pp_ts, tg_ts, ttfr_ms, run_count in rows:
            new_table.add_row(
                str(depth),
                str(conc),
                _fmt_float(pp_ts, missing="…"),
                _fmt_float(tg_ts, missing="…"),
                _fmt_float(ttfr_ms, missing="…"),
                str(run_count),
            )
        self._table = new_table
        self._refresh()

    def log(self, message: str) -> None:
        self._console.log(message)


# ---------------------------------------------------------------------------
# Plain-stdout backend
# ---------------------------------------------------------------------------


class _PlainUI:
    """Minimal fallback when ``rich`` is unavailable."""

    def __init__(self, total_tasks: int, benchmark_id: str, title: str = "") -> None:
        self._total = total_tasks
        self._benchmark_id = benchmark_id
        self._title = title or benchmark_id

    def __enter__(self) -> "_PlainUI":
        print(f"=== Benchmark {self._title} — {self._total} tasks ===")
        return self

    def __exit__(self, *exc: Any) -> None:
        print("=== Benchmark complete ===")

    def start_task(self, idx: int, label: str) -> None:
        print(f"[{idx + 1}/{self._total}] running {label}")

    def end_task(self, idx: int, success: bool, duration_s: float | None = None) -> None:
        status = "ok" if success else "FAILED"
        dur_str = f" ({duration_s:.1f}s)" if duration_s is not None else ""
        print(f"[{idx + 1}/{self._total}] {status}{dur_str}")

    def update_results_table(self, consolidated: dict[str, Any]) -> None:
        rows = _build_table_rows(consolidated)
        if not rows:
            return
        header = f"{'depth':>8}  {'conc':>4}  {'pp t/s':>8}  {'tg t/s':>8}  {'ttfr ms':>8}  {'runs':>4}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for depth, conc, pp_ts, tg_ts, ttfr_ms, run_count in rows:
            print(
                f"{depth:>8}  {conc:>4}  {_fmt_float(pp_ts, missing='-'):>8}"
                f"  {_fmt_float(tg_ts, missing='-'):>8}  {_fmt_float(ttfr_ms, missing='-'):>8}  {run_count:>4}"
            )
        print(sep)

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
        title: Optional display title shown next to the benchmark id.  Defaults
            to *benchmark_id* when omitted.
    """

    def __init__(self, total_tasks: int, benchmark_id: str, title: str = "") -> None:
        self._backend: _RichUI | _PlainUI
        if _HAS_RICH:
            self._backend = _RichUI(total_tasks, benchmark_id, title)
        else:
            self._backend = _PlainUI(total_tasks, benchmark_id, title)

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
        """Rebuild the live results table from a llama-benchy consolidated JSON dict.

        The dict shape is ``{model, max_concurrency, benchmarks: [...]}``.  Rows
        in *benchmarks* are grouped by ``(context_size, concurrency)``; pp/tg/ttfr
        values are averaged across runs in each group.  Calling this method is
        idempotent — successive calls simply replace the rendered table.
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

    _DEMO_TASKS = [
        (0, 1, "d=0 c=1"),
        (0, 2, "d=0 c=2"),
        (4096, 1, "d=4096 c=1"),
        (4096, 5, "d=4096 c=5"),
    ]

    _CANNED_JSON: dict[str, Any] = {
        "model": "org/demo-model",
        "max_concurrency": 5,
        "benchmarks": [],
    }

    print(f"Rich available: {_HAS_RICH}")

    with BenchmarkProgressUI(total_tasks=len(_DEMO_TASKS), benchmark_id="bench_demo0000", title="demo-model/smoke-test") as ui:
        ui.log("Starting smoke test with 4 simulated tasks")

        for i, (depth, conc, label) in enumerate(_DEMO_TASKS):
            ui.start_task(i, label)
            time.sleep(1.2)

            # Simulate a completed benchmark entry
            _CANNED_JSON["benchmarks"].append(
                {
                    "concurrency": conc,
                    "context_size": depth,
                    "prompt_size": 2048,
                    "response_size": 128,
                    "is_context_prefill_phase": False,
                    "pp_throughput": {"mean": round(random.uniform(800, 1600), 1), "std": 30.0, "values": []},
                    "tg_throughput": {"mean": round(random.uniform(30, 90), 1), "std": 2.0, "values": []},
                    "ttfr": {"mean": round(random.uniform(100, 300), 1), "std": 10.0, "values": []},
                }
            )

            ui.end_task(i, success=True, duration_s=1.2)
            ui.update_results_table(_CANNED_JSON)

        ui.log("All tasks complete.")
