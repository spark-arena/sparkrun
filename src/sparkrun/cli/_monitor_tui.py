"""Textual TUI for ``sparkrun cluster monitor``.

Driven by a :class:`~sparkrun.api.LiveMonitorSession` (``api.open_live_monitor``),
so each host row combines live resource telemetry with the workloads occupying
it from ``api.status`` — i.e. docker **and** ``local`` (and provider) workloads,
not just the docker containers the old telemetry-embedded ``docker ps`` saw.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Sequence

import click
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Static

from sparkrun.api import LogTarget, RunResult, StopResult, open_log_target, post_launch, run, stop
from sparkrun.core.application_profile import get_application_profile
from sparkrun.cli._monitor_metrics import RuntimeMetricsScraper, _at, _spark_palette, _sparkline
from sparkrun.cli._monitor_logs import (
    DumpLogScreen,
    LogPane,
    LogScope,
    StopWorkloadScreen,
    WorkloadList,
    entry_plain_text,
    workload_rows,
)
from sparkrun.cli._monitor_recipes import BackgroundWork, RecipeLaunch, RecipePickerScreen, call_ui, local_post_commands_refusal
from sparkrun.core.log_source import SCOPE_ALL, LogLine
from sparkrun.core.monitoring import HostActivity, MonitorFrame, MonitorSample

if TYPE_CHECKING:
    from sparkrun.api import LiveMonitorSession

logger = logging.getLogger(__name__)


class _TuiLogSink(logging.Handler):
    """Root-logger sink for as long as the app is mounted.

    The CLI installs a ``StreamHandler`` bound to the real stderr *before* the
    TUI starts, and that handler keeps the terminal stream object; Textual's
    print capture only swaps ``sys.stderr`` for later lookups, so a launch-time
    ``logger.info`` would paint over the screen. Progress lines are dropped (the
    log pane carries the workload's own output) and warnings/errors surface as
    notifications.
    """

    def __init__(self, app: "ClusterMonitorApp") -> None:
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING:
            return
        severity = "error" if record.levelno >= logging.ERROR else "warning"
        try:
            app = self._app
            if threading.current_thread() is threading.main_thread():
                app.notify(record.getMessage(), severity=severity, timeout=8)
            else:
                app.call_from_thread(app.notify, record.getMessage(), severity=severity)
        except Exception:  # noqa: BLE001 - logging must never break the app
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sparkline glyphs, palette, and the metric scraper live in the leaf module
# _monitor_metrics so the log pane's footer can share them (import direction is
# one-way: _monitor_tui -> _monitor_logs / _monitor_metrics).
_CELL_LABEL_WIDTH = 9  # longest label ("GPU Power") keeps the grid columns lined up
_CELL_VALUE_WIDTH = 10  # "999.9 kb/s"-class values; longer ones just overflow
_CELL_SEP = "   "  # between grid cells; the row keeps a two-space indent


def _stats_layout(width: int | None) -> tuple[int, int]:
    """(columns, sparkline width) for the available *width*; (3, 20) when unknown.

    Picks the widest grid that fits: a cell is label + space + sparkline + space
    + value, prefixed by the two-space indent with ``_CELL_SEP`` between
    columns.  Below the narrowest sparkline the spark column is dropped
    entirely: ``(1, 0)`` renders 22 columns but the ladder never proves a fit
    for it, so narrower panes may still overflow.
    """
    if width is None or width <= 0:
        return 3, 20
    for columns in (3, 2, 1):
        for spark_width in (20, 12, 8, 4):
            cell = _CELL_LABEL_WIDTH + 1 + spark_width + 1 + _CELL_VALUE_WIDTH
            if 2 + columns * cell + len(_CELL_SEP) * (columns - 1) <= width:
                return columns, spark_width
    return 1, 0


def _series(history: list[tuple[float, MonitorSample]] | None, field: str) -> list[float]:
    """Parsable finite float values of *field* across *history* points (skips empty/garbage)."""
    out: list[float] = []
    for _ts, sample in history or ():
        try:
            value = float(getattr(sample, field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):  # a remote "nan"/"inf" must not break the tick
            out.append(value)
    return out


_RATE_UNITS = ("b/s", "kb/s", "Mb/s", "Gb/s", "Tb/s")


def _fmt_bitrate(bytes_per_s: float) -> str:
    """Network bytes/s → bits/s with an adaptive magnitude prefix (decimal steps).

    Rises and falls with the value: 60 B/s reads '480.0 b/s', 1 MB/s '8.0 Mb/s'.
    Clamps at Tb/s and reads '0.0 b/s' on an idle link.
    """
    bits = bytes_per_s * 8
    idx = 0
    while idx + 1 < len(_RATE_UNITS) and bits >= 1000.0:
        bits /= 1000.0
        idx += 1
    return f"{bits:.1f} {_RATE_UNITS[idx]}"


def _cell(
    label: str,
    sample: MonitorSample | None,
    history: list[tuple[float, MonitorSample]] | None,
    field: str,
    fmt: Callable[[float], str],
    *,
    lo: float | None = None,
    hi: float | None = None,
    color: str | None = None,
    series_field: str | None = None,
    spark_width: int = 20,
) -> str:
    """``'{label} {sparkline} {value}'`` for one grid cell.

    The value parses from the fresh frame sample (``-`` when empty/unparsable)
    because ``history()`` lags the frame by up to one record interval; the
    sparkline comes from the history series.  ``series_field`` overrides which
    column feeds the sparkline when value and trend come from different ones.
    ``spark_width`` sets the sparkline length; 0 drops the spark column (and its
    surrounding spaces) so a narrow pane keeps the value.
    """
    try:
        value = fmt(float(getattr(sample, field, "")))
    except (TypeError, ValueError):
        value = "-"
    if spark_width <= 0:
        return f"{label:<{_CELL_LABEL_WIDTH}} {value:<{_CELL_VALUE_WIDTH}}"
    spark = _sparkline(_series(history, series_field or field), width=spark_width, lo=lo, hi=hi)
    if color:
        spark = f"[{color}]{spark}[/{color}]"
    return f"{label:<{_CELL_LABEL_WIDTH}} {spark} {value:<{_CELL_VALUE_WIDTH}}"


def _container_count(activity: HostActivity | None) -> int:
    """Total containers/processes occupying the host (across its workloads)."""
    if activity is None:
        return 0
    return sum(len(w.containers) or 1 for w in activity.workloads)


def _render_workloads(activity: Any, cache_dir: str | None = None) -> list[str]:
    return [r.text for r in workload_rows(activity, cache_dir=cache_dir)]


def _stat_rows(cells: Sequence[str], columns: int) -> list[str]:
    """Indented lines holding *cells* chunked into rows of *columns*."""
    return ["  " + _CELL_SEP.join(cells[i : i + columns]) for i in range(0, len(cells), columns)]


def _mem_row_fits(values: Sequence[str], columns: int, spark_width: int, width: int) -> bool:
    """Whether the memory rows fit *width* with these (possibly over-budget) values."""
    cells = [_CELL_LABEL_WIDTH + 1 + spark_width + 1 + max(_CELL_VALUE_WIDTH, len(v)) for v in values]
    for i in range(0, len(cells), columns):
        row = cells[i : i + columns]
        if 2 + sum(row) + len(_CELL_SEP) * (len(row) - 1) > width:
            return False
    return True


def _render_node_stats(
    host: str,
    activity: HostActivity | None,
    history: list[tuple[float, MonitorSample]] | None = None,
    palette: Sequence[str] = (),
    width: int | None = None,
) -> str:
    """Build the Rich-markup string for the node stats in the detail panel.

    One header line (host │ hostname │ RAM used/total │ CPU │ GPU (count)) then
    the 3x3 sparkline grid; an nv-monitor-only fourth row adds the App/Cache
    memory breakdown.  ``history`` is the host's ``LiveMonitorSession.history()``
    points (oldest first) — absent, every sparkline renders blank.  ``palette``
    holds the per-cell sparkline colors (see :func:`_spark_palette`); empty
    renders them uncolored.  ``width`` is the available content width: the grid
    collapses to fewer columns and shorter sparklines until a row fits (see
    :func:`_stats_layout`); ``None`` keeps the 3x3, 20-wide layout.
    """
    if activity is None:
        return f"[dim]{host}: connecting…[/dim]"

    s: MonitorSample | None = activity.telemetry

    errors: list[str] = []
    if activity.telemetry_error:
        errors.append(f"  [yellow]telemetry: {activity.telemetry_error}[/yellow]")
    if activity.status_error:
        errors.append(f"  [yellow]status: {activity.status_error}[/yellow]")

    if s is None:
        return "\n".join([f"[bold]{host}[/bold]", *errors, "  [dim]telemetry unavailable[/dim]"])

    # Header line: only non-empty segments, in a fixed order.
    header: list[str] = [f"[bold]{host}[/bold]"]
    if s.hostname:
        header.append(s.hostname)
    if s.mem_used_mb and s.mem_total_mb:
        header.append(f"RAM {s.mem_used_mb}/{s.mem_total_mb} MB")
    if s.cpu_name:
        header.append(s.cpu_name)
    if s.gpu_name:
        header.append(f"{s.gpu_name} ({s.gpu_count})" if s.gpu_count else s.gpu_name)

    lines: list[str] = ["  " + " │ ".join(header), *errors, ""]

    columns, spark_width = _stats_layout(width)

    def _c(label: str, field: str, fmt: Callable[[float], str], **kw: Any) -> str:
        return _cell(label, s, history, field, fmt, spark_width=spark_width, **kw)

    grid = (
        (
            _c("GPU %", "gpu_util_pct", lambda v: f"{v:.1f} %", lo=0.0, hi=100.0, color=_at(palette, 0)),
            _c("GPU Temp", "gpu_temp_c", lambda v: f"{v:.1f} °C", color=_at(palette, 1)),
            _c("GPU Power", "gpu_power_w", lambda v: f"{v:.2f} W", color=_at(palette, 2)),
        ),
        (
            _c("CPU %", "cpu_usage_pct", lambda v: f"{v:.1f} %", lo=0.0, hi=100.0, color=_at(palette, 3)),
            _c("CPU Temp", "cpu_temp_c", lambda v: f"{v:.1f} °C", color=_at(palette, 4)),
            _c("GPU Clock", "gpu_clock_mhz", lambda v: f"{v:.0f} MHz", color=_at(palette, 5)),
        ),
        (
            _c("RAM %", "mem_used_pct", lambda v: f"{v:.1f} %", lo=0.0, hi=100.0, color=_at(palette, 6)),
            _c("Net Send", "net_send_bytes_s", _fmt_bitrate, color=_at(palette, 7)),
            _c("Net Recv", "net_recv_bytes_s", _fmt_bitrate, color=_at(palette, 8)),
        ),
    )
    lines.extend(_stat_rows([cell for row in grid for cell in row], columns))

    # Memory breakdown row (nv-monitor backend only — the bash CSV has no bufcache).
    if s.mem_bufcache_mb and s.mem_used_mb and s.mem_total_mb:
        try:
            total = float(s.mem_total_mb)
        except (ValueError, TypeError):
            total = 0.0
        if total > 0:

            def _mem_fmt(v: float, pct: bool = True) -> str:
                base = f"{v:.0f} MB"
                return f"{base} ({v / total * 100:.1f}%)" if pct else base

            def _mem_text(field: str, pct: bool) -> str:
                try:
                    return _mem_fmt(float(getattr(s, field, "")), pct)
                except (TypeError, ValueError):
                    return "-"

            # The percentage suffix pushes the value past the grid's 10-column
            # budget; keep it only when this row actually fits the pane. The
            # plain "N MB" form always does (it is at most 10 digits + " MB").
            with_pct = (
                width is None
                or width <= 0
                or _mem_row_fits([_mem_text("mem_used_mb", True), _mem_text("mem_bufcache_mb", True)], columns, spark_width, width)
            )

            def _fmt(v: float) -> str:
                return _mem_fmt(v, with_pct)

            lines.extend(
                _stat_rows(
                    [
                        _c(
                            "App",
                            "mem_used_mb",
                            _fmt,
                            series_field="mem_used_pct",
                            lo=0.0,
                            hi=100.0,
                            color=_at(palette, 9),
                        ),
                        _c("Cache", "mem_bufcache_mb", _fmt, color=_at(palette, 10)),
                    ],
                    columns,
                )
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table cell formatters (telemetry columns; the Jobs column comes from occupancy)
# ---------------------------------------------------------------------------


def _cell_cpu(s: MonitorSample) -> str:
    return s.cpu_usage_pct or "-"


def _cell_ram(s: MonitorSample) -> str:
    return "%s%%" % s.mem_used_pct if s.mem_used_pct else "-"


def _cell_gpu(s: MonitorSample) -> str:
    return s.gpu_util_pct or "-"


def _cell_cpu_temp(s: MonitorSample) -> str:
    return "%s C" % s.cpu_temp_c if s.cpu_temp_c else "-"


def _cell_gpu_temp(s: MonitorSample) -> str:
    return "%s C" % s.gpu_temp_c if s.gpu_temp_c else "-"


def _cell_gpu_power(s: MonitorSample) -> str:
    return "%s W" % s.gpu_power_w if s.gpu_power_w else "-"


def _cell_swap(s: MonitorSample) -> str:
    if not s.swap_total_mb or not s.swap_used_mb:
        return "-"
    try:
        total = float(s.swap_total_mb)
        used = float(s.swap_used_mb)
        if total <= 0:
            return "0%"
        return "%.0f%%" % (used / total * 100)
    except (ValueError, TypeError):
        return "-"


def _cell_gpu_enc(s: MonitorSample) -> str:
    return s.gpu_encoder_pct or "-"


def _cell_gpu_dec(s: MonitorSample) -> str:
    return s.gpu_decoder_pct or "-"


# Ordered telemetry column definitions: (key, label, cell_fn). The leading
# Host + Jobs columns are handled separately (Jobs is occupancy-derived).
_TABLE_COLS: list[tuple[str, str, Callable[[MonitorSample], str]]] = [
    ("cpu", "CPU%", _cell_cpu),
    ("ram", "RAM%", _cell_ram),
    ("swap", "Swap%", _cell_swap),
    ("gpu", "GPU%", _cell_gpu),
    ("cpu_temp", "CPU Temp", _cell_cpu_temp),
    ("gpu_temp", "GPU Temp", _cell_gpu_temp),
    ("gpu_power", "GPU Power", _cell_gpu_power),
    ("gpu_enc", "Enc%", _cell_gpu_enc),
    ("gpu_dec", "Dec%", _cell_gpu_dec),
]


class QuitWhileBusyScreen(ModalScreen[bool]):
    """Confirms quitting while a launch or stop is still running; dismisses True to quit.

    The work cannot be cancelled (see :class:`BackgroundWork`), so quitting
    restores the shell and then waits for it. Saying so up front is the
    difference between a deliberate wait and a shell that looks hung, where a
    Ctrl-C would leave the launch half-done.
    """

    DEFAULT_CSS = """
    QuitWhileBusyScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #quit-dialog {
        width: 64;
        height: auto;
        border: thick $warning;
        background: $background-lighten-2;
        padding: 1 2;
    }
    #quit-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #quit-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    #quit-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("left", "app.focus_previous", "Previous", show=False, priority=True),
        Binding("right", "app.focus_next", "Next", show=False, priority=True),
        Binding("escape", "cancel", "Keep monitoring", priority=True),
        Binding("q", "cancel", "Keep monitoring", show=False, priority=True),
    ]

    def __init__(self, busy: Sequence[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.busy = tuple(busy)

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-dialog"):
            yield Static("Work in progress", id="quit-title")
            # Descriptions carry recipe names (registry content): plain text.
            yield Static("Still running: %s." % "; ".join(self.busy), id="quit-busy", markup=False)
            yield Static("Quitting returns to the shell, then waits for it to finish; interrupting it there can leave it half-done.")
            with Horizontal(id="quit-buttons"):
                yield Button("Keep monitoring", id="quit-cancel")
                yield Button("Quit and wait", id="quit-confirm", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#quit-cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-confirm":
            self.dismiss(True)
        elif event.button.id == "quit-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class ClusterMonitorApp(App):
    """Textual TUI for live cluster monitoring (telemetry + occupancy)."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #host-table {
        height: auto;
        max-height: 60%;
        margin: 0 1;
    }
    Screen.-logs-open #host-table,
    .-logs-open #host-table {
        max-height: 25%;
    }
    #detail-panel {
        height: 1fr;
        margin: 0 1;
        border: round $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    #node-stats {
        height: auto;
    }
    #node-body {
        height: 1fr;
    }
    LogPane {
        min-height: 8;
    }
    """

    BINDINGS = [
        Binding("enter", "select_node", "Logs"),
        Binding("r", "open_recipe_picker", "Recipes"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
    ]

    # Log metrics are collected in the background on a fixed cadence, for every
    # workload in the monitor's scope — not just the one a pane happens to show.
    METRICS_TRACK_INTERVAL_S: float = 30.0
    # Tail lines read per source per poll: enough to hold the engine's slowest
    # metric line, cheap enough for one bounded read over SSH.
    METRICS_TAIL_LINES: int = 500
    # The first read of a workload scans further back: an engine that has been
    # idle for a while emits nothing but probes, so a plain tail page yields
    # zeros. A deeper page still contains its last real stat lines; the scraper
    # seeds the window from the page's last sample, never its mean, so the first
    # value is a current reading rather than a multi-minute average. Costs one
    # larger tail read per workload, once; later ticks read the cheap tail and
    # ingest only the lines appended since the previous page.
    METRICS_PRIME_LINES: int = 4000
    # Occupancy sweeps only run every few seconds and can lag a launch further, so
    # the fallback poll is generous before it degrades to a notification.
    PENDING_LOG_OPEN_TIMEOUT_S: float = 60.0

    def notify(self, message: str, *, markup: bool = False, **kwargs: Any) -> None:
        """Toasts default to plain text.

        Every toast carries remote or recipe text (errors, log records, ids),
        and Textual parses markup by default: a ``[/]`` in it raises at render
        time and exits the app, and ``[red]…`` restyles what the user reads.
        """
        super().notify(message, markup=markup, **kwargs)

    def __init__(
        self,
        session: "LiveMonitorSession",
        hosts: list[str],
        interval: int = 2,
        cache_dir: str | None = None,
        cluster: str | None = None,
        sctx: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._hosts = list(hosts)
        self._interval = interval
        self._cache_dir = cache_dir
        self._cluster = cluster
        self._sctx = sctx
        self._selected_host: str | None = self._hosts[0] if self._hosts else None
        self._pinned_host: str | None = None
        self._frame: MonitorFrame | None = None
        # Resolved log targets, written only by the metrics worker thread.
        self._metrics_targets: dict[str, LogTarget] = {}
        # Committed scrapers, created and fed only on the UI thread.
        self._metrics_scrapers: dict[str, RuntimeMetricsScraper] = {}
        self._track_cancel = threading.Event()
        self._metrics_lock = threading.Lock()
        # Whether the first sample has been taken; it must not wait a full interval.
        self._metrics_primed = False
        # (workload_id, deadline, resolved (host, container)) for the log pane that
        # opens once a fresh launch shows up in the occupancy sweep; None when
        # nothing is pending. The source is carried so a mount delayed by an
        # existing pane's prune never has to wait for the sweep.
        self._pending_log_open: tuple[str, float, tuple[str, str] | None] | None = None
        # Root handlers from before the TUI (terminal-bound) parked while mounted.
        self._saved_root_handlers: list[logging.Handler] | None = None
        # Root level from before the TUI; restored alongside the handlers.
        self._saved_root_level: int | None = None
        # Terminal-bound handlers owned by library/plugin loggers, likewise.
        self._saved_logger_handlers: list[tuple[logging.Logger, logging.Handler]] = []
        # Worker-thread calls that quitting would wait on; see BackgroundWork.
        self.background_work = BackgroundWork()

    async def run_async(self, **kwargs: Any) -> Any:
        """Run the app, then say what the shell is still waiting for.

        Runs after the screen is restored and before ``asyncio.run`` joins the
        worker threads, which is the silent wait this explains.
        """
        try:
            return await super().run_async(**kwargs)
        finally:
            self._announce_unfinished_work()

    def _announce_unfinished_work(self) -> None:
        for description in self.background_work.descriptions():
            click.echo("Waiting for %s to finish..." % description, err=True)

    def _report_from_thread(self, message: str, *, severity: str = "information") -> None:
        """A toast while the app runs; a stderr line once it has gone (a launch can outlive it)."""
        if not call_ui(self, self.notify, message, severity=severity):
            click.echo(message, err=True)

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield DataTable(id="host-table", cursor_type="row")
            with Vertical(id="detail-panel"):
                yield Static(id="node-stats", markup=True)
                with Vertical(id="node-body"):
                    yield WorkloadList(id="workload-list", host=self._selected_host or "")
        yield Footer()

    # -- lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        self._install_log_sink()
        self.title = "%s cluster monitor" % get_application_profile().command
        self.sub_title = "%d host(s) — every %ds" % (len(self._hosts), self._interval)

        table = self.query_one("#host-table", DataTable)
        table.add_column("Host", key="host")
        table.add_column("Jobs", key="jobs")
        for key, label, _ in _TABLE_COLS:
            table.add_column(label, key=key)

        for host in self._hosts:
            table.add_row(host, "-", *(["-"] * len(_TABLE_COLS)), key=host)

        # Immediate initial refresh, then tick periodically.
        self._refresh()
        self.set_interval(1.0, self._refresh)
        # No immediate metrics tick: the first sample lands one interval in, so a
        # short-lived pilot run never pays for (or depends on) a log read.
        self.set_interval(self.METRICS_TRACK_INTERVAL_S, self._metrics_tick)

    def on_unmount(self) -> None:
        self._restore_log_handlers()
        self._track_cancel.set()
        self.workers.cancel_group(self, "metrics-track")
        self._session.close()

    def _install_log_sink(self) -> None:
        """Park terminal-bound handlers; see :class:`_TuiLogSink`.

        Root is not enough: hf_hub, SAF and plugin loggers can carry their own
        ``StreamHandler`` created before Textual swapped the streams, and those
        hold the real stderr, so their records paint straight over the screen.
        Every terminal-bound stream handler is parked for the app lifetime and
        put back on unmount.
        """
        root = logging.getLogger()
        self._saved_root_handlers = root.handlers[:]
        self._saved_root_level = root.level
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        # The sink drops everything below WARNING anyway, and line-oriented
        # streaming consumers gate on
        # ``logging.getLogger().isEnabledFor(logging.INFO)``:
        # orchestration/distribution.py picks ``run_remote_script_streaming(
        # quiet=False)`` when root is at INFO, and that subprocess inherits the
        # terminal fds and paints straight over the TUI.
        root.setLevel(logging.WARNING)
        root.addHandler(_TuiLogSink(self))
        self._park_library_handlers()

    def _park_library_handlers(self) -> None:
        """Park terminal-bound stream handlers on named loggers; sweep every tick.

        A library imported during a launch (an HF helper, a plugin) attaches its
        handler *after* mount, and that handler holds the pre-redirect terminal
        stream. Re-sweeping on the cheap 1 s refresh closes that window.

        Only handlers still bound to the real terminal are parked: a
        ``FileHandler`` or a handler writing to some other stream logs somewhere
        harmless and must keep working while the app is mounted.
        """
        for logger in list(logging.Logger.manager.loggerDict.values()):
            if not isinstance(logger, logging.Logger):
                continue
            for handler in logger.handlers[:]:
                if not isinstance(handler, logging.StreamHandler):
                    continue
                if handler.stream is not sys.__stdout__ and handler.stream is not sys.__stderr__:
                    continue
                logger.removeHandler(handler)
                self._saved_logger_handlers.append((logger, handler))

    def _restore_log_handlers(self) -> None:
        saved = self._saved_root_handlers
        if saved is None:
            return
        self._saved_root_handlers = None
        root = logging.getLogger()
        for handler in root.handlers[:]:
            if isinstance(handler, _TuiLogSink) and handler._app is self:
                root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
        for logger, handler in self._saved_logger_handlers:
            logger.addHandler(handler)
        self._saved_logger_handlers = []
        if self._saved_root_level is not None:
            root.setLevel(self._saved_root_level)
            self._saved_root_level = None

    # -- events -------------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None:
            self._selected_host = str(event.row_key.value)
            if self._pinned_host is None:
                self._refresh_detail()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_select_node()

    def action_select_node(self) -> None:
        try:
            table = self.query_one("#host-table", DataTable)
            if not table.has_focus:
                return
        except Exception:
            return
        host = self._selected_host
        if not host:
            return
        frame = getattr(self, "_frame", None)
        activity = frame.for_host(host) if frame is not None else None
        if not activity or not activity.workloads:
            self.notify(f"No sparkrun workloads on {host}")
            return

        self._pinned_host = host
        self.query_one("#workload-list", WorkloadList).focus()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if getattr(event.control, "id", None) == "host-table":
            if self._is_log_open():
                self._close_log_pane(refocus_list=False)
            self._pinned_host = None
            self._refresh_detail()

    def on_workload_list_back_to_hosts(self, event: WorkloadList.BackToHosts) -> None:
        self._pinned_host = None
        self.query_one("#host-table", DataTable).focus()

    def on_workload_list_open_log(self, event: WorkloadList.OpenLog) -> None:
        self._mount_log_pane(event.host, event.cluster_id, event.container)

    def _mount_log_pane(self, host: str, cluster_id: str, container: str) -> bool:
        """Show one container's log pane in the node body (manual and post-launch opens share it).

        Returns False when an existing pane was closed and the caller must retry
        once Textual has pruned it (see ``_try_pending_log_open``).
        """
        if self._is_log_open():
            # A launch's pending auto-open can fire while the user is already
            # watching another workload. Mounting a second #log-pane would raise
            # DuplicateIds, and Textual prunes the old node asynchronously (a
            # widget mounted before the prune lands is marked _pruning and never
            # composes its children), so closing and remounting happen apart.
            self._close_log_pane(refocus_list=False)
            return False
        self._mount_pane_now(host, cluster_id, container)
        return True

    def _mount_pane_now(self, host: str, cluster_id: str, container: str) -> None:
        wl = self.query_one("#workload-list", WorkloadList)
        wl.display = False

        pane = LogPane(
            host=host,
            cluster_id=cluster_id,
            container=container,
            scope=LogScope.CONTAINER,
            frame=self._frame,
            cluster=self._cluster,
            cache_dir=self._cache_dir,
            sctx=self._sctx,
            id="log-pane",
        )
        self.query_one("#node-body", Vertical).mount(pane)
        self.add_class("-logs-open")
        self.screen.add_class("-logs-open")
        # Pin the node whose container is being watched, so the detail panel
        # matches the pane and a background refresh cannot swap it underneath.
        self._pinned_host = host
        self._refresh_detail()
        # The pane container is not focusable; focus its log view so keys land
        # in the logtail panel (the pane's own bindings still fire by bubbling).
        pane.log_view.focus()

    def on_log_pane_closed(self, event: LogPane.Closed) -> None:
        self._close_log_pane(refocus_list=True)

    def _close_log_pane(self, refocus_list: bool = True) -> None:
        try:
            pane = self.query_one("#log-pane", LogPane)
            pane.close()
            pane.remove()
        except Exception:
            pass
        self.remove_class("-logs-open")
        self.screen.remove_class("-logs-open")
        try:
            wl = self.query_one("#workload-list", WorkloadList)
            wl.display = True
            if refocus_list:
                wl.focus()
        except Exception:
            pass

    def _is_log_open(self) -> bool:
        try:
            return bool(self.query("#log-pane"))
        except Exception:
            return False

    def on_workload_list_request_dump(self, event: WorkloadList.RequestDump) -> None:
        self._open_dump_screen(event.host, event.cluster_id, event.container, event.scope)

    def on_log_pane_request_dump(self, event: LogPane.RequestDump) -> None:
        self._open_dump_screen(event.host, event.cluster_id, event.container, event.scope)

    def _open_dump_screen(self, host: str, cluster_id: str, container: str, scope: LogScope) -> None:
        screen = DumpLogScreen(
            host=host,
            cluster_id=cluster_id,
            container=container,
            default_scope=scope,
            frame=self._frame,
            cluster=self._cluster,
            cache_dir=self._cache_dir,
            sctx=self._sctx,
        )
        self.push_screen(screen)

    def on_workload_list_request_stop(self, event: WorkloadList.RequestStop) -> None:
        self._open_stop_screen(event.host, event.cluster_id, event.container)

    def _open_stop_screen(self, host: str, cluster_id: str, container: str) -> None:
        def _confirmed(confirmed: bool | None) -> None:
            if confirmed:
                self._stop_workload_worker(cluster_id)

        self.push_screen(
            StopWorkloadScreen(host=host, cluster_id=cluster_id, container=container),
            callback=_confirmed,
        )

    @work(thread=True, exclusive=False, group="stop")
    def _stop_workload_worker(self, cluster_id: str) -> None:
        """Tear the workload down off the UI thread; api.stop does blocking SSH/docker work."""
        # No hosts=: api.stop must resolve the job's recorded placement (job
        # metadata, then cluster) — the monitored host list is a view, never a
        # placement authority, and passing it would override the real TP>1 hosts.
        with self.background_work.track(f"the stop of {cluster_id}", blocks_quit=True):
            try:
                result = stop(workload_id=cluster_id, cluster=self._cluster, cache_dir=self._cache_dir, sctx=self._sctx)
            except Exception as error:  # noqa: BLE001 - a stop failure must never break the app
                logger.debug("Stop of %s failed", cluster_id, exc_info=True)
                self._report_from_thread(f"Stop failed for {cluster_id}: {error}", severity="error")
                return
        if not call_ui(self, self._on_stop_finished, cluster_id, result):
            click.echo("Stop of %s %s" % (cluster_id, "finished" if result.success else "failed"), err=True)

    def _on_stop_finished(self, cluster_id: str, result: StopResult) -> None:
        if result.success:
            self.notify(f"Stopped {cluster_id} on {len(result.hosts_targeted)} host(s)")
        else:
            details = "; ".join(result.errors) or ", ".join(result.hosts_failed) or "unknown error"
            self.notify(f"Stop failed for {cluster_id}: {details}", severity="error")

    # -- recipe picker ------------------------------------------------------

    def action_open_recipe_picker(self) -> None:
        """Open the recipe selection panel from the node (host) or workload list.

        Only the base screen routes this binding: a pushed modal or the picker
        itself must not stack a second picker, and the log pane is a viewing
        mode of its own. The focus guard keeps ``r`` inert everywhere else
        (e.g. while the log view holds the keyboard).
        """
        if self.screen is not self.screen_stack[0]:
            return
        try:
            table = self.query_one("#host-table", DataTable)
            workloads = self.query_one("#workload-list", WorkloadList)
        except Exception:
            return
        if not (table.has_focus or workloads.has_focus):
            return
        self.push_screen(
            RecipePickerScreen(cluster=self._cluster, hosts=tuple(self._hosts), sctx=self._sctx),
            callback=self._on_recipe_picked,
        )

    def _on_recipe_picked(self, launch: RecipeLaunch | None) -> None:
        if launch is None:
            return
        placed = tuple(getattr(launch.plan, "host_list", None) or ())
        self.notify(f"Launching {launch.label} on {', '.join(placed)}…" if placed else f"Launching {launch.label}…")
        self._recipe_run_worker(launch)

    @work(thread=True, exclusive=False, group="recipe-run")
    def _recipe_run_worker(self, launch: RecipeLaunch) -> None:
        """Launch off the UI thread; api.run does blocking SSH/docker work.

        exclusive=False (like the stop worker): a second launch must never
        cancel one that is already in flight. The preflight's plan is passed
        through, so the launch is the placement the parameter modal checked.
        """
        with self.background_work.track(f"the launch of {launch.label}", blocks_quit=True) as describe:
            try:
                result = run(launch.options, sctx=self._sctx, plan=launch.plan)
            except Exception as error:  # noqa: BLE001 - a launch failure must never break the app
                logger.debug("Recipe launch of %s failed", launch.label, exc_info=True)
                self._report_from_thread(f"Launch failed: {error}", severity="error")
                return
            launched = result.rc == 0 and not result.dry_run
            source = self._launch_log_source(result) if launched else None
            if not call_ui(self, self._on_recipe_launched, result, source):
                click.echo(
                    "Launched %s as %s" % (launch.label, result.workload_id)
                    if launched
                    else "Launch of %s finished (rc=%s)" % (launch.label, result.rc),
                    err=True,
                )
            if launched:
                describe(f"the post-launch hooks of {launch.label}")
                self._run_post_launch(result, launch)

    def _launch_log_source(self, result: RunResult) -> tuple[str, str] | None:
        """The first (host, container) of a fresh launch's logs, or None to wait for the sweep."""
        # Resolved here: the workstation-side metadata is written by run(), and
        # the occupancy sweep can be many seconds behind, so the frame is the
        # slow path, not the source of truth.
        try:
            target = open_log_target(result.workload_id, cluster=self._cluster, cache_dir=self._cache_dir, sctx=self._sctx)
        except Exception:  # noqa: BLE001 - the occupancy poll is the fallback
            logger.debug("Log source resolve for %s failed", result.workload_id, exc_info=True)
            return None
        for candidate in target.sources:
            if candidate.container:
                return candidate.host, candidate.container
        return None

    def _run_post_launch(self, result: RunResult, launch: RecipeLaunch) -> None:
        """Readiness, ``post_exec`` and ``stop_after_post``, as ``sparkrun run`` does after launching.

        ``api.post_launch`` reuses the trust verdict ``run`` recorded on the
        launch. Its output is logged (the sink keeps it off the screen) and its
        failures become toasts. Local ``post_commands`` never run here: the
        parameter modal refuses those recipes (``local_post_commands_refusal``),
        and this guard covers a launch that reached the app another way.
        """
        handle = result.launch_result
        recipe = getattr(handle, "recipe", None) or launch.options.recipe
        if isinstance(recipe, str) or not (getattr(recipe, "post_exec", None) or getattr(recipe, "post_commands", None)):
            # The picker always launches a loaded recipe; a bare name has no hooks to inspect.
            return
        refusal = local_post_commands_refusal(recipe, launch.label)
        if refusal is not None:
            self._report_from_thread(f"Post-launch hooks of {launch.label} not run. {refusal}", severity="error")
            return
        if handle is None:
            self._report_from_thread(
                f"Post-launch hooks of {launch.label} not run: the launch returned no lifecycle handle", severity="error"
            )
            return
        try:
            outcome = post_launch(result, trust=launch.options.trust, sctx=self._sctx)
        except Exception as error:  # noqa: BLE001 - a hook failure must never break the app
            logger.debug("Post-launch lifecycle of %s failed", launch.label, exc_info=True)
            self._report_from_thread(f"Post-launch hooks of {launch.label} failed: {error}", severity="error")
            return
        if outcome.stopped:
            self._report_from_thread(f"{launch.label}: post-launch hooks finished; workload stopped (stop_after_post)")
        else:
            self._report_from_thread(f"{launch.label}: post-launch hooks finished")

    def _on_recipe_launched(self, result: RunResult, source: tuple[str, str] | None = None) -> None:
        if result.rc != 0:
            self.notify(f"Launch failed (rc={result.rc}) for {result.workload_id}", severity="error")
            return
        if result.dry_run:
            self.notify(f"Dry run: nothing to monitor ({result.workload_id})")
            return
        if source is not None and self._mount_log_pane(source[0], result.workload_id, source[1]):
            return
        # No resolved source (or an existing pane had to close first, so the
        # mount must wait for its prune). The resolved source is kept: the
        # occupancy sweep can lag container start by a poll, and the sweep is
        # only needed when nothing resolved the target.
        self._pending_log_open = (
            result.workload_id,
            time.monotonic() + self.PENDING_LOG_OPEN_TIMEOUT_S,
            source,
        )
        self._try_pending_log_open()

    def _try_pending_log_open(self) -> None:
        """Open the log pane for a just-launched workload once the frame shows it."""
        if self._pending_log_open is None:
            return
        workload_id, deadline, source = self._pending_log_open
        if source is not None:
            if self._mount_log_pane(source[0], workload_id, source[1]):
                self._pending_log_open = None
            elif time.monotonic() > deadline:
                self._pending_log_open = None
                self.notify(f"Launched {workload_id}; press enter on it to open logs")
            return
        frame = getattr(self, "_frame", None)
        if frame is not None:
            for host in self._hosts:
                rows = workload_rows(frame.for_host(host), cache_dir=self._cache_dir)
                row = next((r for r in rows if r.cluster_id == workload_id and r.container), None)
                if row is not None:
                    # Keep the pending entry armed when an existing pane had to
                    # be closed first: the next tick mounts once the prune lands.
                    if self._mount_log_pane(host, workload_id, row.container):
                        self._pending_log_open = None
                    return
        if time.monotonic() > deadline:
            self._pending_log_open = None
            self.notify(f"Launched {workload_id}; press enter on it to open logs")

    # -- background log metrics ---------------------------------------------

    def metrics_scraper_for(self, cluster_id: str) -> RuntimeMetricsScraper | None:
        """Committed log metrics for *cluster_id*, or None before the first sample."""
        return self._metrics_scrapers.get(cluster_id)

    @work(thread=True, exclusive=False, group="metrics-track")
    def _metrics_tick(self) -> None:
        # Reads have their own timeout. A slow poll must finish instead of
        # being cancelled and restarted at the same source every interval.
        # Only one worker owns the target cache; overlapping ticks are skipped.
        if not self._metrics_lock.acquire(blocking=False):
            return
        try:
            self._collect_metrics()
        finally:
            self._metrics_lock.release()

    def _collect_metrics(self) -> None:
        """Read every tracked workload's log tail and hand the lines to the UI thread.

        Runs off the UI thread: each source costs one bounded SSH read. Targets
        are cached across ticks and re-resolved only when a workload appears or
        a read fails, so steady state is one tail page per source per tick.
        """
        cancel = self._track_cancel
        if cancel.is_set():
            return
        frame = self._frame
        if frame is None:
            return
        wanted: dict[str, list[str]] = {}
        for activity in frame.hosts:
            for workload in activity.workloads:
                wanted.setdefault(workload.cluster_id, []).append(activity.host)
        for gone in [cid for cid in self._metrics_targets if cid not in wanted]:
            self._metrics_targets.pop(gone, None)
        if not call_ui(self, self._on_metrics_collected, set(wanted), {}, {}):
            return
        # The workload's authoritative source set right now: a re-resolved
        # target names a restarted rank's new container, so the scraper can
        # forget the old one instead of summing both forever.
        live: dict[str, set[tuple[str, str]]] = {}
        for cluster_id in sorted(wanted):
            if cancel.is_set():
                return
            target = self._metrics_targets.get(cluster_id)
            primed = target is not None
            if target is None:
                try:
                    # No hosts=: container naming comes from the job's recorded
                    # topology, exactly as the log pane resolves it. Passing the
                    # monitored hosts would collapse a multi-host workload onto
                    # the solo container name (runtimes/base.py log_sources) and
                    # read a container that does not exist; the frame's placement
                    # is a monitoring view, never a naming authority.
                    target = open_log_target(
                        cluster_id,
                        cluster=self._cluster,
                        cache_dir=self._cache_dir,
                        sctx=self._sctx,
                        scope=SCOPE_ALL,
                        cancel=cancel,
                    )
                except Exception:  # noqa: BLE001 - tracking is best-effort; never break the app
                    logger.debug("Metrics tracking cannot resolve %s", cluster_id, exc_info=True)
                    continue
                self._metrics_targets[cluster_id] = target
            # First sample of a workload reads deep enough to find real stats;
            # steady state is the cheap tail. A target re-resolved after a read
            # failure counts as unprimed, so it re-seeds the same way.
            count = self.METRICS_TAIL_LINES if primed else self.METRICS_PRIME_LINES
            live[cluster_id] = {(source.host, source.container) for source in target.sources}
            lines: list[LogLine] = []
            for source in target.sources:
                if cancel.is_set():
                    return
                try:
                    lines.extend(target.page(source, skip=0, count=count, cancel=cancel))
                except Exception:  # noqa: BLE001 - stale target re-resolves on the next tick
                    logger.debug("Metrics tracking read failed for %s", cluster_id, exc_info=True)
                    self._metrics_targets.pop(cluster_id, None)
                    break
            if lines and not cancel.is_set():
                # Publish completed workloads before reading the next one, so
                # a stalled source cannot hold back already collected samples.
                if not call_ui(self, self._on_metrics_collected, set(wanted), {cluster_id: lines}, live):
                    return

    def _on_metrics_collected(self, wanted: set[str], collected: dict[str, list[LogLine]], live: dict[str, set[tuple[str, str]]]) -> None:
        """UI thread: commit this poll's samples and repaint open pane footers."""
        for gone in [cid for cid in self._metrics_scrapers if cid not in wanted]:
            self._metrics_scrapers.pop(gone, None)
        for cluster_id, lines in collected.items():
            scraper = self._metrics_scrapers.setdefault(cluster_id, RuntimeMetricsScraper())
            sources = live.get(cluster_id)
            if sources is not None:
                scraper.retain(sources)
            grouped: dict[tuple[str, str], list[str]] = {}
            for line in lines:
                grouped.setdefault((line.host, line.container), []).append(entry_plain_text(line))
            for source, texts in grouped.items():
                scraper.observe_page(source, texts)
            scraper.commit()
        for pane in self.query(LogPane):
            try:
                pane._refresh_metrics_footer()
            except NoMatches:
                # A pane mid-prune has lost its children but is still listed.
                continue

    # -- refresh ------------------------------------------------------------

    def _refresh(self) -> None:
        # Textual stops the app (is_running False) before pruning its screens,
        # but closes the interval timers only afterwards, so a tick can land on
        # widgets that are already gone and raise NoMatches out of shutdown.
        if not self.is_running:
            return
        self._frame = self._session.frame()
        self._refresh_table()
        self._refresh_detail()
        # A monitor opened on a running workload should show its metrics now,
        # not one full interval from now: take the first sample as soon as the
        # frame names a workload, then stay on the fixed cadence.
        if not self._metrics_primed and self._frame is not None and any(activity.workloads for activity in self._frame.hosts):
            self._metrics_primed = True
            self._metrics_tick()
        self._try_pending_log_open()
        self._park_library_handlers()

    def _refresh_table(self) -> None:
        table = self.query_one("#host-table", DataTable)
        frame = getattr(self, "_frame", None)
        for host in self._hosts:
            activity = frame.for_host(host) if frame is not None else None
            if activity is None:
                continue

            # Connection status alongside the hostname.
            if activity.telemetry is None and activity.telemetry_error:
                table.update_cell(host, "host", "%s (error)" % host)
            elif activity.telemetry_error or activity.status_error:
                table.update_cell(host, "host", "%s (!)" % host)
            else:
                table.update_cell(host, "host", host)

            # Jobs from occupancy (all executors), not the telemetry stream.
            table.update_cell(host, "jobs", str(_container_count(activity)) if activity.workloads else "-")

            s = activity.telemetry
            if s is None:
                continue
            for key, _label, cell_fn in _TABLE_COLS:
                table.update_cell(host, key, cell_fn(s))

    def _refresh_detail(self) -> None:
        stats = self.query_one("#node-stats", Static)
        host = self._pinned_host or self._selected_host
        if host is None:
            stats.update("[dim]No host selected[/dim]")
            return
        frame = getattr(self, "_frame", None)
        activity = frame.for_host(host) if frame is not None else None

        # history() is optional so duck-typed session stand-ins keep working; a
        # host outside the session's set (cannot happen in production — the same
        # host list feeds session and app) degrades to blank sparklines.
        history: list[tuple[float, MonitorSample]] = []
        history_fn = getattr(self._session, "history", None)
        if history_fn is not None:
            try:
                history = history_fn(host)
            except KeyError:
                history = []

        # Sparkline colors follow the active theme; a theme that lacks the
        # variables yields an empty palette (sparks uncolored), not a crash.
        palette: tuple[str, ...] = ()
        try:
            palette = _spark_palette(self.get_css_variables(), dark=self.current_theme.dark)
        except Exception:  # noqa: BLE001 - cosmetic only; never break the tick
            palette = ()

        stats.update(_render_node_stats(host, activity, history, palette, width=stats.size.width))

        try:
            wl = self.query_one("#workload-list", WorkloadList)
            rows = workload_rows(activity, cache_dir=self._cache_dir)
            wl.set_rows(rows, host=host)
        except Exception:
            pass

    def on_resize(self, event: events.Resize) -> None:
        # Repaint now so a narrowing window collapses the grid immediately
        # instead of waiting for the next 1 s poll.
        try:
            self._refresh_detail()
        except Exception:  # noqa: BLE001 - a resize can arrive before the widgets exist
            pass

    def action_quit(self) -> None:
        if self._is_log_open():
            return
        busy = self.background_work.descriptions(blocking_only=True)
        if busy:
            # ctrl+q is an app-level priority binding, so it reaches here with
            # the confirmation already up; one dialog is enough.
            if not isinstance(self.screen, QuitWhileBusyScreen):
                self.push_screen(QuitWhileBusyScreen(busy), callback=self._on_quit_confirmed)
            return
        self.exit()

    def _on_quit_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()
