"""Textual TUI for ``sparkrun cluster monitor``."""

from __future__ import annotations

import logging

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from sparkrun.core.monitoring import ClusterMonitor, HostMonitorState, MonitorSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BAR_WIDTH = 30


def _bar(value: float, width: int = _BAR_WIDTH) -> str:
    """Render a Unicode bar: ████████░░░░░░░░░░░░."""
    pct = max(0.0, min(value / 100.0, 1.0))
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def _pct(raw: str) -> float:
    """Parse a percentage string to float, defaulting to 0."""
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _parse_container_jobs(container_names: list[str], cache_dir: str | None) -> list[dict]:
    """Resolve container names to job metadata entries.

    Uses the same cluster-ID extraction logic as
    :func:`sparkrun.core.cluster_manager.query_cluster_status`:
    solo containers end with ``_solo``; clustered containers encode the
    cluster_id as ``sparkrun_{12-char hash}`` followed by a role suffix.

    Returns a list of dicts, one per container, with keys:
        name, role, cluster_id, recipe, model, runtime, tp
    """
    from sparkrun.orchestration.job_metadata import load_job_metadata

    # Group containers by cluster_id first so we only load metadata once
    clusters: dict[str, list[tuple[str, str]]] = {}  # cluster_id -> [(name, role)]
    for name in container_names:
        if name.endswith("_solo"):
            cid = name.removesuffix("_solo")
            clusters.setdefault(cid, []).append((name, "solo"))
        else:
            prefix_end = name.find("_", len("sparkrun_"))
            if 0 < prefix_end < len(name) - 1:
                cid = name[:prefix_end]
                role = name[prefix_end + 1:]
            else:
                cid = name
                role = "?"
            clusters.setdefault(cid, []).append((name, role))

    result: list[dict] = []
    for cid, members in clusters.items():
        meta = load_job_metadata(cid, cache_dir=cache_dir) or {}
        for name, role in members:
            result.append({
                "name": name,
                "role": role,
                "cluster_id": cid,
                "recipe": meta.get("recipe", ""),
                "model": meta.get("model", ""),
                "runtime": meta.get("runtime", ""),
                "tp": meta.get("tensor_parallel", ""),
            })
    return result


def _render_detail(host: str, state: HostMonitorState | None, cache_dir: str | None = None) -> str:
    """Build the Rich-markup string for the detail panel."""
    if state is None or (state.latest is None and state.error is None):
        return f"[dim]{host}: connecting…[/dim]"

    if state.error and state.latest is None:
        return f"[red]{host}: {state.error}[/red]"

    s: MonitorSample = state.latest

    cpu = _pct(s.cpu_usage_pct)
    ram = _pct(s.mem_used_pct)
    gpu = _pct(s.gpu_util_pct)

    lines: list[str] = [
        f"[bold]{host}[/bold]",
    ]

    if state.error:
        lines.append(f"  [yellow]{state.error}[/yellow]")

    lines.extend([
        "",
        f"  CPU  [cyan]{_bar(cpu)}[/cyan] {cpu:5.1f}%",
        f"  RAM  [green]{_bar(ram)}[/green] {ram:5.1f}%",
        f"  GPU  [yellow]{_bar(gpu)}[/yellow] {gpu:5.1f}%",
        "",
    ])

    # Hardware details
    extras: list[str] = []
    if s.gpu_name:
        extras.append(f"GPU: {s.gpu_name}")
    if s.cpu_temp_c:
        extras.append(f"CPU temp: {s.cpu_temp_c} °C")
    if s.gpu_temp_c:
        extras.append(f"GPU temp: {s.gpu_temp_c} °C")
    if s.gpu_power_w:
        power_str = f"{s.gpu_power_w} W"
        if s.gpu_power_limit_w:
            power_str += f" / {s.gpu_power_limit_w} W"
        extras.append(f"GPU power: {power_str}")
    if s.mem_used_mb and s.mem_total_mb:
        extras.append(f"RAM: {s.mem_used_mb} / {s.mem_total_mb} MB")
    if s.gpu_mem_used_mb and s.gpu_mem_total_mb:
        extras.append(f"GPU mem: {s.gpu_mem_used_mb} / {s.gpu_mem_total_mb} MB")
    if extras:
        lines.append("  " + "  │  ".join(extras))
        lines.append("")

    # Container list with job metadata
    names = [n for n in s.sparkrun_job_names.split("|") if n] if s.sparkrun_job_names else []
    if names:
        jobs = _parse_container_jobs(names, cache_dir)
        lines.append(f"  [bold]Containers ({len(names)}):[/bold]")
        for job in jobs:
            # Primary line: container name and role
            role_tag = f" [dim]({job['role']})[/dim]" if job["role"] and job["role"] != "?" else ""
            lines.append(f"    {job['name']}{role_tag}")

            # Metadata line: recipe, model, runtime, tp
            meta_parts: list[str] = []
            if job["recipe"]:
                meta_parts.append(f"recipe=[bold]{job['recipe']}[/bold]")
            if job["model"]:
                meta_parts.append(f"model={job['model']}")
            if job["runtime"]:
                meta_parts.append(f"runtime={job['runtime']}")
            if job["tp"]:
                meta_parts.append(f"tp={job['tp']}")
            if meta_parts:
                lines.append(f"      [dim]{', '.join(meta_parts)}[/dim]")
    else:
        lines.append("  [dim]No sparkrun containers running[/dim]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table cell formatters
# ---------------------------------------------------------------------------

def _cell_jobs(s: MonitorSample) -> str:
    return s.sparkrun_jobs or "-"


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


# Ordered column definitions: (key, label, cell_fn)
_TABLE_COLS: list[tuple[str, str, object]] = [
    ("jobs", "Jobs", _cell_jobs),
    ("cpu", "CPU%", _cell_cpu),
    ("ram", "RAM%", _cell_ram),
    ("gpu", "GPU%", _cell_gpu),
    ("cpu_temp", "CPU Temp", _cell_cpu_temp),
    ("gpu_temp", "GPU Temp", _cell_gpu_temp),
    ("gpu_power", "GPU Power", _cell_gpu_power),
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class ClusterMonitorApp(App):
    """Textual TUI for live cluster monitoring."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #host-table {
        height: auto;
        max-height: 60%;
        margin: 0 1;
    }
    #detail-panel {
        height: 1fr;
        margin: 0 1;
        border: round $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        monitor: ClusterMonitor,
        cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._monitor = monitor
        self._cache_dir = cache_dir
        self._selected_host: str | None = monitor.hosts[0] if monitor.hosts else None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield DataTable(id="host-table", cursor_type="row")
            yield Static(id="detail-panel", markup=True)
        yield Footer()

    # -- lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        self.title = "sparkrun cluster monitor"
        self.sub_title = "%d host(s) — every %ds" % (
            len(self._monitor.hosts),
            self._monitor.interval,
        )

        table = self.query_one("#host-table", DataTable)
        table.add_column("Host", key="host")
        for key, label, _ in _TABLE_COLS:
            table.add_column(label, key=key)

        for host in self._monitor.hosts:
            table.add_row(host, *(["-"] * len(_TABLE_COLS)), key=host)

        self._monitor.start()
        self.set_interval(1.0, self._refresh)

    def on_unmount(self) -> None:
        self._monitor.stop()

    # -- events -------------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None:
            self._selected_host = str(event.row_key.value)
            self._refresh_detail()

    # -- refresh ------------------------------------------------------------

    def _refresh(self) -> None:
        self._refresh_table()
        self._refresh_detail()

    def _refresh_table(self) -> None:
        table = self.query_one("#host-table", DataTable)
        for host in self._monitor.hosts:
            state = self._monitor.states.get(host)
            if state is None:
                continue

            # Show connection status alongside the hostname.
            if state.error and state.latest is None:
                table.update_cell(host, "host", "%s (error)" % host)
            elif state.error:
                table.update_cell(host, "host", "%s (!)" % host)
            else:
                table.update_cell(host, "host", host)

            if state.latest is None:
                continue
            s = state.latest
            for key, _label, cell_fn in _TABLE_COLS:
                table.update_cell(host, key, cell_fn(s))

    def _refresh_detail(self) -> None:
        panel = self.query_one("#detail-panel", Static)
        host = self._selected_host
        if host is None:
            panel.update("[dim]No host selected[/dim]")
            return
        state = self._monitor.states.get(host)
        panel.update(_render_detail(host, state, cache_dir=self._cache_dir))
