"""Tests for the resource-telemetry seam + the api live-monitoring surface.

(Distinct from ``test_telemetry.py``, which covers anonymous usage analytics.)

Covers:
- TelemetryProvider discovery / scope selection (SAF).
- HostTelemetryProvider backend selection + session snapshot (SSH mocked).
- api.open_telemetry (None when no provider for the scope).
- LiveMonitorSession / api.open_live_monitor frame composition (telemetry + occupancy).
"""

from __future__ import annotations

import itertools
import time
from unittest import mock

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload
from sparkrun.core.monitoring import HostActivity, HostTelemetry, MonitorFrame, MonitorSample
from sparkrun.orchestration.logs import LogLine, LogSource
from sparkrun.orchestration.telemetry import get_telemetry_provider, list_telemetry_scopes
from sparkrun.orchestration.telemetry._base import TelemetryProvider, TelemetrySession


# --------------------------------------------------------------------------
# Provider discovery
# --------------------------------------------------------------------------


def test_host_provider_registered_and_resolvable():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    assert "host" in list_telemetry_scopes(v)
    provider = get_telemetry_provider("host", v)
    assert provider is not None
    assert provider.scope == "host"


def test_unknown_scope_returns_none():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    # No k8s/modal telemetry provider in-tree — a substrate with no provider
    # yields None (occupancy-only monitoring), never an error.
    assert get_telemetry_provider("k8s", v) is None


def test_base_provider_open_not_implemented():
    with mock.patch.object(TelemetryProvider, "scope", "x"):
        with pytest.raises(NotImplementedError):
            TelemetryProvider().open(["h1"])


# --------------------------------------------------------------------------
# HostTelemetryProvider — backend selection + session (SSH mocked)
# --------------------------------------------------------------------------


class _FakeMonitor:
    """Stand-in for ClusterMonitor: records start/stop, exposes .states."""

    def __init__(self, hosts, ssh_kwargs, interval=2, **kw):
        from sparkrun.core.monitoring import HostMonitorState

        self.hosts = hosts
        self.interval = interval
        self.started = False
        self.stopped = False
        self.states = {h: HostMonitorState(latest=MonitorSample(hostname=h, gpu_util_pct="10")) for h in hosts}

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_host_provider_defaults_to_bash_backend(monkeypatch):
    from sparkrun.orchestration.telemetry.host import HostTelemetryProvider

    created = {}

    def _fake_cm(hosts, ssh_kwargs, interval=2, **kw):
        created["bash"] = _FakeMonitor(hosts, ssh_kwargs, interval)
        return created["bash"]

    monkeypatch.setattr("sparkrun.core.monitoring.ClusterMonitor", _fake_cm)
    session = HostTelemetryProvider().open(["h1"], ssh_kwargs={}, interval=3)
    assert created["bash"].started
    assert session.snapshot()["h1"].sample.gpu_util_pct == "10"
    session.close()
    assert created["bash"].stopped


def test_host_provider_nv_monitor_backend(monkeypatch):
    from sparkrun.orchestration.telemetry.host import HostTelemetryProvider

    created = {}

    def _fake_nv(hosts, ssh_kwargs, interval=2, **kw):
        created["nv"] = _FakeMonitor(hosts, ssh_kwargs, interval)
        return created["nv"]

    monkeypatch.setattr("sparkrun.core.monitoring.NvMonitorClusterMonitor", _fake_nv)
    HostTelemetryProvider().open(["h1"], ssh_kwargs={}, backend="nv-monitor")
    assert "nv" in created and created["nv"].started


# --------------------------------------------------------------------------
# api.open_telemetry / live-monitor composition
# --------------------------------------------------------------------------


class _FakeSession(TelemetrySession):
    def __init__(self, snap):
        self._snap = snap
        self.closed = False

    def snapshot(self):
        return self._snap

    def close(self):
        self.closed = True


class _FakeProvider:
    scope = "host"

    def __init__(self, session):
        self._session = session

    def open(self, hosts, **kw):
        return self._session


def test_open_telemetry_none_when_no_provider(monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    assert api.open_telemetry(["h1"]) is None


def _drive_until(session, pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    frame = session.frame()
    while not pred(frame) and time.monotonic() < deadline:
        time.sleep(0.02)
        frame = session.frame()
    return frame


def test_live_monitor_frame_combines_telemetry_and_occupancy(monkeypatch):
    tel = {"h1": HostTelemetry(host="h1", sample=MonitorSample(hostname="h1", gpu_util_pct="42"))}
    monkeypatch.setattr(
        "sparkrun.orchestration.telemetry.get_telemetry_provider",
        lambda scope, v=None: _FakeProvider(_FakeSession(tel)),
    )
    snap = ClusterStatus(
        hosts=(HostOccupancy(host="h1", workloads=(RunningWorkload(cluster_id="sparkrun_x"),), used_slots=1, free_slots=0),),
        executor="docker",
    )
    # The background poller reads sparkrun.api.status (package attr).
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: snap)

    session = api.open_live_monitor(["h1"], interval=1, status_interval=1)
    try:
        frame = _drive_until(session, lambda f: f.for_host("h1").used_slots == 1)
    finally:
        session.close()

    assert isinstance(frame, MonitorFrame)
    act = frame.for_host("h1")
    assert isinstance(act, HostActivity)
    assert act.telemetry.gpu_util_pct == "42"  # telemetry axis
    assert [w.cluster_id for w in act.workloads] == ["sparkrun_x"]  # occupancy axis
    assert act.used_slots == 1


def test_live_monitor_occupancy_only_when_no_telemetry(monkeypatch):
    """A substrate with no telemetry provider still yields occupancy frames."""
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    snap = ClusterStatus(hosts=(HostOccupancy(host="h1", used_slots=0, free_slots=4),), executor="docker")
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: snap)

    session = api.open_live_monitor(["h1"], interval=1, status_interval=1)
    try:
        frame = _drive_until(session, lambda f: f.for_host("h1").free_slots == 4)
    finally:
        session.close()

    act = frame.for_host("h1")
    assert act.telemetry is None  # no telemetry provider
    assert act.free_slots == 4  # occupancy still present


def test_live_monitor_stops_promptly_when_its_operation_is_cancelled(monkeypatch):
    import threading

    from sparkrun.core.operation import OperationState

    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: ClusterStatus(hosts=(HostOccupancy(host="h1"),), executor="docker"))
    sctx = api.default_sctx()
    sctx.operation = OperationState()
    threading.Timer(0.2, sctx.operation.cancel.set).start()
    started = time.monotonic()
    frames = list(api.live_monitor(["h1"], interval=60, sctx=sctx))
    assert frames == []  # cancelled during the first wait
    assert time.monotonic() - started < 10  # not a 60s uninterruptible sleep


# --------------------------------------------------------------------------
# LiveMonitorSession telemetry history (per-host ring buffer)
# --------------------------------------------------------------------------


class _CountingSession(TelemetrySession):
    """Telemetry session whose every snapshot is a distinct, ordered sample."""

    def __init__(self):
        self._n = itertools.count(1)

    def snapshot(self):
        return {"h1": HostTelemetry(host="h1", sample=MonitorSample(hostname="h1", gpu_util_pct=str(next(self._n))))}

    def close(self):
        pass


def _open_history_session(monkeypatch, session=None, **kwargs):
    """Open a history-recording session; the recorder thread is stubbed out so
    the test drives ``_record_once()`` synchronously (no timing)."""
    from sparkrun.api._telemetry import LiveMonitorSession

    monkeypatch.setattr(
        "sparkrun.orchestration.telemetry.get_telemetry_provider",
        lambda scope, v=None: _FakeProvider(session if session is not None else _CountingSession()),
    )
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: ClusterStatus(hosts=(HostOccupancy(host="h1"),), executor="docker"))
    monkeypatch.setattr(LiveMonitorSession, "_record_loop", lambda self: None)
    return api.open_live_monitor(["h1"], interval=1, status_interval=1, **kwargs)


def test_live_monitor_history_records_per_host(monkeypatch):
    session = _open_history_session(monkeypatch)
    try:
        for _ in range(3):
            session._record_once()
        points = session.history("h1")
    finally:
        session.close()

    assert len(points) == 3
    assert [sample.gpu_util_pct for _, sample in points] == ["1", "2", "3"]  # oldest first
    times = [recorded_at for recorded_at, _ in points]
    assert len(set(times)) == 3 and times == sorted(times)  # strictly increasing record times
    with pytest.raises(KeyError):
        session.history("nope")  # unknown host, not an empty list


def test_live_monitor_history_evicts_beyond_window(monkeypatch):
    session = _open_history_session(monkeypatch, history_points=3)
    for _ in range(5):
        session._record_once()
    points = session.history("h1")
    session.close()
    assert len(points) == 3
    assert [sample.gpu_util_pct for _, sample in points] == ["3", "4", "5"]  # ring kept the newest

    floored = _open_history_session(monkeypatch, history_points=0)
    for _ in range(3):
        floored._record_once()
    kept = floored.history("h1")
    floored.close()
    assert len(kept) == 1 and kept[0][1].gpu_util_pct == "3"  # floor to 1 slot, not disable


def test_live_monitor_history_skips_missing_samples(monkeypatch):
    tel = _FakeSession({"h1": HostTelemetry(host="h1", sample=None, error="ssh down")})
    session = _open_history_session(monkeypatch, session=tel)
    try:
        session._record_once()
        assert session.history("h1") == []  # a gap in the series, not a zero reading
        tel._snap = {"h1": HostTelemetry(host="h1", sample=MonitorSample(hostname="h1", gpu_util_pct="9"))}
        session._record_once()
        assert [sample.gpu_util_pct for _, sample in session.history("h1")] == ["9"]
    finally:
        session.close()


def test_live_monitor_history_frozen_after_close(monkeypatch):
    session = _open_history_session(monkeypatch)
    session._record_once()
    session._record_once()
    session.close()
    assert len(session.history("h1")) == 2  # recorded data stays readable after close
    session._record_once()  # close() set the stop flag; the recorder appends nothing more
    assert len(session.history("h1")) == 2


def test_live_monitor_history_empty_without_telemetry(monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: ClusterStatus(hosts=(HostOccupancy(host="h1"),), executor="docker"))
    session = api.open_live_monitor(["h1"], interval=1, status_interval=1)
    try:
        assert session.history("h1") == []  # no telemetry session: recorder never started
        assert session.frame().for_host("h1") is not None  # occupancy-only frames unaffected
    finally:
        session.close()


# --------------------------------------------------------------------------
# TUI detail rendering — workloads sourced from occupancy (all executors)
# --------------------------------------------------------------------------


def _activity_with_local_job(telemetry=None):
    from sparkrun.core.cluster_status import ContainerDetail, RunningWorkload

    wl = RunningWorkload(
        cluster_id="sparkrun_deadbeefdeadbeef_aabbccddeeff",
        recipe_name="qwen3-1.7b-vllm",
        runtime_name="vllm",
        containers=(
            ContainerDetail(name="sparkrun_deadbeefdeadbeef_aabbccddeeff_solo", role="solo", status="Up (pid 42)", image="(local process)"),
        ),
    )
    return HostActivity(host="h1", telemetry=telemetry, workloads=(wl,), used_slots=1, free_slots=0)


def test_render_node_stats_and_workload_rows(tmp_path):
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _render_node_stats
    from sparkrun.cli._monitor_logs import workload_rows

    act = _activity_with_local_job(MonitorSample(hostname="h1", gpu_util_pct="7"))
    stats_out = _render_node_stats("h1", act)
    assert "GPU" in stats_out
    assert "Workloads" not in stats_out

    rows = workload_rows(act, cache_dir=str(tmp_path))
    texts = [r.text for r in rows]
    joined = "\n".join(texts)
    assert "sparkrun_deadbeefdeadbeef_aabbccddeeff" in joined
    assert "qwen3-1.7b-vllm" in joined
    assert "(local process)" in joined or "solo" in joined


def test_render_node_stats_shows_without_telemetry():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _render_node_stats

    # No telemetry sample (e.g. a provider-less substrate) still renders node stats.
    out = _render_node_stats("h1", _activity_with_local_job(None))
    assert "telemetry unavailable" in out
    assert "h1" in out


def test_container_count_counts_across_workloads():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _container_count

    assert _container_count(_activity_with_local_job()) == 1
    assert _container_count(HostActivity(host="h1")) == 0


def _gb10_sample() -> MonitorSample:
    """Shaped like the live GB10 probe: unified memory, so no GPU-mem fields."""
    return MonitorSample(
        hostname="spark-41e1",
        cpu_name="X925 (10) + A725 (10)",
        gpu_name="NVIDIA GB10",
        gpu_count="1",
        mem_used_mb="113431",
        mem_total_mb="124610",
        mem_used_pct="91.0",
        gpu_util_pct="96",
        gpu_temp_c="71",
        gpu_power_w="45.68",
        gpu_clock_mhz="2515",
        cpu_usage_pct="18.1",
        cpu_temp_c="82.4",
        net_send_bytes_s="1537000000",
        net_recv_bytes_s="1234500000",
    )


def test_sparkline_empty_flat_ramp_and_pinned_scale():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS, _sparkline
    from sparkrun.cli._monitor_tui import _series

    # No history yet → blank, so a cell's spark region is visibly empty.
    assert _sparkline([]) == " " * 20

    # A flat window has no span to scale against: one mid glyph per point.
    assert _sparkline([5.0, 5.0, 5.0]) == " " * 17 + "▄▄▄"

    # A full ramp covers every glyph, oldest left.
    assert _sparkline([i / 7 for i in range(8)], width=8) == _SPARK_GLYPHS

    # Percent metrics pin 0-100 so the glyph stays absolute (a 50% window still
    # reads mid-scale), and short series are right-aligned.
    assert _sparkline([0.0, 100.0], width=4, lo=0.0, hi=100.0) == "  ▁█"

    # An over-long series keeps the NEWEST points (a rising trend must not read
    # as flat history), and a non-finite remote value is dropped, not raised on.
    assert _sparkline([0.0] * 25 + [7.0]) == "▁" * 19 + "█"
    assert _series([(1.0, MonitorSample(gpu_temp_c="nan")), (2.0, MonitorSample(gpu_temp_c="inf"))], "gpu_temp_c") == []
    assert _sparkline(_series([(1.0, MonitorSample(gpu_temp_c="nan"))], "gpu_temp_c")) == " " * 20


def test_spark_palette_degrades_without_theme_variables():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_metrics import _at, _spark_palette

    # A theme missing the expected variables renders uncolored sparks, not a crash.
    assert _spark_palette({}, dark=True) == ()
    assert _spark_palette({}, dark=False) == ()
    # A short/absent palette yields no color for any cell.
    assert _at((), 0) is None
    assert _at(("#123456",), 1) is None
    assert _at(("#123456",), 0) == "#123456"


def test_fmt_bitrate_units_adapt_to_magnitude():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _fmt_bitrate

    assert _fmt_bitrate(0.0) == "0.0 b/s"
    assert _fmt_bitrate(60.0) == "480.0 b/s"  # below 1 kb/s stays in bits
    assert _fmt_bitrate(1000.0) == "8.0 kb/s"
    assert _fmt_bitrate(1_000_000.0) == "8.0 Mb/s"
    assert _fmt_bitrate(1_537_000_000.0) == "12.3 Gb/s"
    assert _fmt_bitrate(125_000_000_000.0) == "1.0 Tb/s"
    assert _fmt_bitrate(10**15) == "8000.0 Tb/s"  # clamps at the largest unit


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
def test_render_node_stats_every_sparkline_resolves_to_a_unique_color(tmp_path, theme):
    """Each sparkline gets a distinct theme color — checked at the resolved-style
    level, because a color name Rich accepts but Textual cannot resolve (e.g.
    ``bright_red``) silently renders as the default foreground.

    Both theme families are pinned: a light theme steps the palette the other way
    (darken instead of lighten), where an aliased pair would otherwise go unseen.
    """
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import Static

    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    # nv-shaped sample: all nine grid series plus the App/Cache row.
    s = _gb10_sample()
    s.mem_bufcache_mb = "45678"
    act = HostActivity(host="h1", telemetry=s)
    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(act,)))
    session.set_history("h1", [(1.0, s), (2.0, s)])
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    app.theme = theme

    async def _run():
        async with app.run_test(size=(180, 40)) as pilot:
            await pilot.pause(0.05)
            assert app.current_theme.dark == theme.endswith("dark")  # the switch took effect
            widget = app.query_one("#node-stats", Static)
            runs = []
            for y in range(widget.size.height):
                for seg in widget.render_line(y):
                    glyphs = seg.text.strip()
                    if glyphs and set(glyphs) <= set(_SPARK_GLYPHS):
                        runs.append(seg)
            return [seg.style.color for seg in runs]

    colors = asyncio.run(_run())
    assert len(colors) == 11  # one colored sparkline per cell
    assert len(set(colors)) == 11
    assert None not in colors  # a fallback-to-default means the color never applied


def test_render_node_stats_gb10_header_and_grid():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS
    from sparkrun.cli._monitor_tui import _render_node_stats

    s = _gb10_sample()
    history = [(float(i), _gb10_sample()) for i in range(3)]
    out = _render_node_stats("10.0.4.30", HostActivity(host="10.0.4.30", telemetry=s), history)

    # Header: host │ hostname │ RAM used/total │ CPU │ GPU (count)
    assert "10.0.4.30" in out
    assert "spark-41e1" in out
    assert "RAM 113431/124610 MB" in out
    assert "X925 (10) + A725 (10)" in out
    assert "NVIDIA GB10 (1)" in out

    # 3x3 grid labels and formatted values.
    for label in ("GPU %", "GPU Temp", "GPU Power", "CPU %", "CPU Temp", "GPU Clock", "RAM %", "Net Send", "Net Recv"):
        assert label in out
    assert "45.68 W" in out
    assert "82.4 °C" in out
    assert "12.3 Gb/s" in out  # 1537000000 B/s * 8 / 1e9
    assert "9.9 Gb/s" in out  # 1234500000 B/s * 8 / 1e9

    # History feeds the sparklines.
    assert any(g in out for g in _SPARK_GLYPHS)

    # The bash backend carries no bufcache, so no App/Cache row appears.
    assert "App" not in out
    assert "Cache" not in out


def test_render_node_stats_sparse_sample_without_history():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS
    from sparkrun.cli._monitor_tui import _render_node_stats

    out = _render_node_stats("h1", HostActivity(host="h1", telemetry=MonitorSample(cpu_usage_pct="5")))

    # Values render without history; empty fields degrade to '-', sparks stay blank.
    assert "5.0 %" in out
    assert " - " in out
    assert not any(g in out for g in _SPARK_GLYPHS)

    # Header omits every empty segment (no hostname/RAM/CPU/GPU in this sample).
    assert out.splitlines()[0] == "  [bold]h1[/bold]"


def test_render_node_stats_nv_monitor_app_cache_row():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS
    from sparkrun.cli._monitor_tui import _render_node_stats

    nv = MonitorSample(
        hostname="nv1",
        mem_bufcache_mb="45678",
        mem_used_mb="113431",
        mem_total_mb="124610",
        mem_used_pct="91.0",
    )
    history = [
        (1.0, MonitorSample(mem_used_pct="90.0", mem_bufcache_mb="40000")),
        (2.0, MonitorSample(mem_used_pct="91.0", mem_bufcache_mb="45678")),
    ]
    out = _render_node_stats("nv1", HostActivity(host="nv1", telemetry=nv), history)

    # Fourth row replaces the old "Memory breakdown" bars, keeping the math.
    assert "App" in out
    assert "Cache" in out
    assert "113431 MB (91.0%)" in out
    assert "45678 MB (36.7%)" in out
    assert any(g in out for g in _SPARK_GLYPHS)

    # Same sample without bufcache (bash backend): row is absent.
    bash = MonitorSample(mem_used_mb="113431", mem_total_mb="124610", mem_used_pct="91.0")
    assert "App" not in _render_node_stats("h", HostActivity(host="h", telemetry=bash), history)


# --------------------------------------------------------------------------
# Detail-panel responsive layout (columns / sparkline width ladder)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (131, (3, 20)),
        (200, (3, 20)),
        (120, (3, 12)),
        (100, (3, 8)),
        (90, (3, 4)),
        (80, (2, 12)),
        (60, (2, 4)),
        (40, (1, 12)),
        (30, (1, 4)),
        (22, (1, 0)),
        (0, (3, 20)),
        (None, (3, 20)),
    ],
)
def test_stats_layout_ladder(width, expected):
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _stats_layout

    assert _stats_layout(width) == expected


def _narrow_sample() -> MonitorSample:
    """Short header segments so the grid rows, not the header, drive the width."""
    return MonitorSample(
        gpu_util_pct="50",
        gpu_temp_c="61",
        gpu_power_w="23.5",
        cpu_usage_pct="12",
        cpu_temp_c="48",
        gpu_clock_mhz="1500",
        mem_used_pct="33",
        net_send_bytes_s="125000",
        net_recv_bytes_s="250000",
    )


def _narrow_activity() -> HostActivity:
    return HostActivity(host="h1", telemetry=_narrow_sample())


def _narrow_history() -> list[tuple[float, MonitorSample]]:
    points: list[tuple[float, MonitorSample]] = []
    for i in range(1, 21):  # 20 points fill a 20-wide sparkline (shorter ones rjust)
        points.append(
            (
                float(i),
                MonitorSample(
                    gpu_util_pct=str(40 + i),
                    gpu_temp_c=str(58 + i),
                    gpu_power_w=str(20 + i),
                    cpu_usage_pct=str(10 + i),
                    cpu_temp_c=str(45 + i),
                    gpu_clock_mhz=str(1400 + i * 100),
                    mem_used_pct=str(30 + i),
                    net_send_bytes_s=str(100000 * i),
                    net_recv_bytes_s=str(200000 * i),
                ),
            )
        )
    return points


@pytest.mark.parametrize("width", [131, 80, 60, 40, 30])
def test_render_node_stats_fits_the_given_width(width):
    pytest.importorskip("textual")
    from rich.text import Text

    from sparkrun.cli._monitor_tui import _render_node_stats

    out = _render_node_stats("h1", _narrow_activity(), _narrow_history(), width=width)
    plain_lines = [Text.from_markup(line).plain for line in out.splitlines()]

    for line in plain_lines:
        assert len(line) <= width, (width, line)
    # No metric value is lost at any width.
    assert "50.0 %" in out
    assert "23.50 W" in out
    assert "1500 MHz" in out
    assert "Net Send" in out


def test_render_node_stats_wide_layout_keeps_three_columns_and_long_sparks():
    pytest.importorskip("textual")
    from rich.text import Text

    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS
    from sparkrun.cli._monitor_tui import _render_node_stats

    out = _render_node_stats("h1", _narrow_activity(), _narrow_history(), width=131)
    lines = [Text.from_markup(line).plain for line in out.splitlines()]
    grid_lines = [line for line in lines if "GPU %" in line or "CPU %" in line or "RAM %" in line]
    assert len(grid_lines) == 3
    for line in grid_lines:
        assert line.count("   ") >= 2  # three cells side by side
        assert sum(1 for ch in line if ch in _SPARK_GLYPHS) == 60  # 3 cells x 20 glyphs


def test_render_node_stats_single_column_puts_each_metric_on_its_own_line():
    pytest.importorskip("textual")
    from rich.text import Text

    from sparkrun.cli._monitor_tui import _render_node_stats

    out = _render_node_stats("h1", _narrow_activity(), _narrow_history(), width=40)
    lines = [Text.from_markup(line).plain for line in out.splitlines()]
    gpu_pct = [line for line in lines if "GPU %" in line]
    assert len(gpu_pct) == 1
    assert "GPU Temp" not in gpu_pct[0]
    assert "GPU Power" not in gpu_pct[0]
    assert any("GPU Power" in line for line in lines)
    assert any("Net Recv" in line for line in lines)


def test_render_node_stats_drops_the_sparkline_below_the_floor():
    pytest.importorskip("textual")
    from rich.text import Text

    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS
    from sparkrun.cli._monitor_tui import _render_node_stats

    # width 22 falls past (1, 4): the spark column goes, the values stay.
    out = _render_node_stats("h1", _narrow_activity(), _narrow_history(), width=22)
    lines = [Text.from_markup(line).plain for line in out.splitlines()]
    for line in lines:
        assert not any(g in line for g in _SPARK_GLYPHS), line
    assert "50.0 %" in out
    assert "23.50 W" in out
    assert "Net Recv" in out


def test_render_node_stats_default_width_is_the_three_by_three_layout():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _render_node_stats

    host, act, history, palette = "h1", _narrow_activity(), _narrow_history(), ()
    assert _render_node_stats(host, act, history, palette, width=None) == _render_node_stats(host, act, history, palette)
    assert _render_node_stats(host, act, history, palette) == _render_node_stats(host, act, history, palette, width=131)


@pytest.mark.parametrize("width", [131, 100, 90, 80, 60, 40, 30, 22])
def test_render_node_stats_memory_row_fits_the_given_width(width):
    """The App/Cache percentage suffix is over the value budget: keep it only when it fits."""
    pytest.importorskip("textual")
    import dataclasses

    from rich.text import Text

    from sparkrun.cli._monitor_tui import _render_node_stats

    sample = dataclasses.replace(_narrow_sample(), mem_used_mb="16384", mem_total_mb="131072", mem_bufcache_mb="4096")
    out = _render_node_stats("h1", HostActivity(host="h1", telemetry=sample), _narrow_history(), width=width)
    memory = [Text.from_markup(line).plain for line in out.splitlines() if "App" in line or "Cache" in line]

    assert memory
    for line in memory:
        assert len(line) <= width, (width, line)
    if width >= 90:  # the wide layouts have room for "(12.5%)"
        assert any("(12.5%)" in line for line in memory), memory
    else:
        assert not any("%)" in line for line in memory), memory


def test_cluster_monitor_app_narrow_window_detail_stays_within_pane_width(tmp_path):
    pytest.importorskip("textual")
    import asyncio
    from rich.text import Text
    from textual.widgets import Static

    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    host = HostActivity(host="h1", telemetry=_narrow_sample())
    app = ClusterMonitorApp(_FakeLiveMonitorSession(MonitorFrame(hosts=(host,))), ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(60, 24)) as pilot:
            await pilot.pause(0.1)
            app._refresh_detail()
            await pilot.pause(0.05)
            stats = app.query_one("#node-stats", Static)
            raw = stats.content.plain if hasattr(stats.content, "plain") else str(stats.content)
            text = Text.from_markup(raw).plain  # the stored content still carries color tags
            assert "GPU %" in text
            lines = text.splitlines()
            assert len(lines) > 3
            for line in lines:
                assert len(line) <= stats.size.width, (stats.size.width, line)

    asyncio.run(_run())


# --------------------------------------------------------------------------
# Canonical serialization (shared by CLI --json and desktop SSE)
# --------------------------------------------------------------------------


def test_serialize_activity_shape():
    from sparkrun.core.monitoring import serialize_activity

    row = serialize_activity(_activity_with_local_job(MonitorSample(hostname="h1", gpu_util_pct="5")))
    assert row["host"] == "h1"
    assert row["sample"]["gpu_util_pct"] == "5"
    assert row["used_slots"] == 1
    wl = row["workloads"][0]
    assert wl["workload_id"] == "sparkrun_deadbeefdeadbeef_aabbccddeeff"
    assert wl["recipe"] == "qwen3-1.7b-vllm"
    assert wl["containers"][0]["image"] == "(local process)"


def test_serialize_activity_no_telemetry():
    from sparkrun.core.monitoring import serialize_activity

    row = serialize_activity(HostActivity(host="h1", status_error="unreachable"))
    assert row["sample"] is None
    assert row["error"] == "unreachable"
    assert row["workloads"] == []


def test_serialize_frame_is_list_of_rows():
    from sparkrun.core.monitoring import MonitorFrame, serialize_frame

    frame = MonitorFrame(hosts=(_activity_with_local_job(), HostActivity(host="h2")), queried_at=1.0)
    rows = serialize_frame(frame)
    assert [r["host"] for r in rows] == ["h1", "h2"]


def test_format_activity_table_jobs_from_occupancy():
    from sparkrun.core.monitoring import MonitorFrame
    from sparkrun.utils.cli_formatters import format_activity_table

    frame = MonitorFrame(hosts=(_activity_with_local_job(MonitorSample(hostname="h1", cpu_usage_pct="9")),))
    out = format_activity_table(frame, ["h1"])
    assert "Jobs" in out  # header
    # The local workload's container is counted even though telemetry is docker-agnostic.
    assert "\nh1" in out and " 1" in out  # jobs=1 column
    # A None frame renders a connecting placeholder without crashing.
    assert "connecting" in format_activity_table(None, ["h1"])


# --------------------------------------------------------------------------
# ClusterMonitorApp Detail Panel Pilot Tests
# --------------------------------------------------------------------------


class _FakeLiveMonitorSession:
    def __init__(self, frame: MonitorFrame) -> None:
        self._frame = frame
        self._history: dict[str, list[tuple[float, MonitorSample]]] = {}
        self.closed = False

    def frame(self) -> MonitorFrame:
        return self._frame

    def set_frame(self, frame: MonitorFrame) -> None:
        self._frame = frame

    def history(self, host: str) -> list[tuple[float, MonitorSample]]:
        return list(self._history.get(host, []))

    def set_history(self, host: str, points: list[tuple[float, MonitorSample]]) -> None:
        self._history[host] = list(points)

    def close(self) -> None:
        self.closed = True


class _DummyLogTarget:
    def __init__(self, **kwargs):
        self.sources = [LogSource(host="h1", container="c1", role="head", rank=0)]
        self.runtime = "vllm-ray"
        self.follow_cancel = None

    def follow(self, tail=200, cancel=None):
        self.follow_cancel = cancel
        yield LogLine(host="h1", container="c1", text="line from c1")

    def page(self, *args, **kwargs):
        return []

    def dump(self, *args, **kwargs):
        return 1


def test_cluster_monitor_app_detail_panel_and_node_selection(tmp_path):
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import DataTable, Static
    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import WorkloadList

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(
            ContainerDetail(name="c1", role="head", status="Up", image="img"),
            ContainerDetail(name="c2", role="worker", status="Up", image="img"),
        ),
    )
    act1 = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="10"), workloads=(w1,))
    act2 = HostActivity(host="h2", telemetry=MonitorSample(hostname="h2", cpu_usage_pct="20"), workloads=())
    frame = MonitorFrame(hosts=(act1, act2))
    session = _FakeLiveMonitorSession(frame)

    app = ClusterMonitorApp(session, ["h1", "h2"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)

            # 1. Detail panel split: #node-stats and #workload-list
            stats = app.query_one("#node-stats", Static)
            wl = app.query_one("#workload-list", WorkloadList)
            assert stats is not None
            assert wl is not None

            # Initially h1 selected; list unfocused but its selection bar is already visible
            assert "h1" in str(stats.content)
            assert wl.highlighted == 1
            styles1 = [seg.style for seg in wl.render_line(1) if seg.text.strip()]
            styles2 = [seg.style for seg in wl.render_line(2) if seg.text.strip()]
            assert styles1[0].bgcolor != styles2[0].bgcolor
            assert app._pinned_host is None

            # 2. Moving table cursor to h2 updates detail panel
            table = app.query_one("#host-table", DataTable)
            table.move_cursor(row=1)
            await pilot.pause(0.05)
            assert "h2" in str(stats.content)
            assert "No sparkrun workloads running" in wl._rows[0].text

            # 3. Enter on empty host h2: notifies and keeps focus on table
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app._pinned_host is None
            assert table.has_focus

            # 4. Move cursor back to h1 and enter -> pins host, focuses list, reveals highlight
            table.move_cursor(row=0)
            await pilot.pause(0.05)
            assert "h1" in str(stats.content)

            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app._pinned_host == "h1"
            assert wl.has_focus
            assert wl.highlighted == 1  # first container c1

            # 5. Frame update refreshes meters while pinned and preserves highlight
            act1_updated = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="95"), workloads=(w1,))
            session.set_frame(MonitorFrame(hosts=(act1_updated, act2)))
            app._refresh()
            await pilot.pause(0.05)
            assert "95.0 %" in str(stats.content)
            assert wl.highlighted == 1

            # 5b. No history yet: values render, sparklines stay blank.
            assert not any(g in str(stats.content) for g in _SPARK_GLYPHS)

            # 5c. History lands: the sparkline seam (session.history -> render) shows glyphs.
            s95 = act1_updated.telemetry
            assert s95 is not None
            session.set_history("h1", [(1.0, s95), (2.0, s95), (3.0, s95)])
            app._refresh()
            await pilot.pause(0.05)
            assert any(g in str(stats.content) for g in _SPARK_GLYPHS)
            assert "95.0 %" in str(stats.content)

            # 6. Container disappears: highlight moves to nearest selectable row
            w1_disappeared = RunningWorkload(
                cluster_id="cid1",
                containers=(ContainerDetail(name="c2", role="worker", status="Up", image="img"),),
            )
            act1_disappeared = HostActivity(
                host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="95"), workloads=(w1_disappeared,)
            )
            session.set_frame(MonitorFrame(hosts=(act1_disappeared, act2)))
            app._refresh()
            await pilot.pause(0.05)
            assert wl.highlighted == 1
            assert wl._rows[wl.highlighted].container == "c2"

    asyncio.run(_run())


def test_cluster_monitor_app_open_and_close_log_pane(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import DataTable, Static
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import WorkloadList, LogPane

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="10"), workloads=(w1,))
    frame = MonitorFrame(hosts=(act1,))
    session = _FakeLiveMonitorSession(frame)

    dummy_target = _DummyLogTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: dummy_target)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            table = app.query_one("#host-table", DataTable)
            wl = app.query_one("#workload-list", WorkloadList)
            stats = app.query_one("#node-stats", Static)

            # Select h1
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert wl.has_focus

            # Enter on c1 opens log pane
            await pilot.press("enter")
            await pilot.pause(0.1)

            # WorkloadList is hidden, LogPane is mounted
            assert wl.display is False
            log_pane = app.query_one("#log-pane", LogPane)
            assert log_pane is not None
            assert stats is not None and stats.display is True  # node stats stay visible
            assert app.has_class("-logs-open") or app.screen.has_class("-logs-open")

            # Enter inside log pane does not trigger action_select_node or break UI
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert len(app.query("#log-pane")) == 1
            assert wl.display is False
            assert not wl.has_focus

            # Node stats keep updating while log pane is open
            act1_updated = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="88"), workloads=(w1,))
            session.set_frame(MonitorFrame(hosts=(act1_updated,)))
            app._refresh()
            await pilot.pause(0.05)
            assert "88.0 %" in str(stats.content)

            # Escape in log pane closes pane and sets its cancel event
            await pilot.press("escape")
            await pilot.pause(0.1)

            assert dummy_target.follow_cancel is not None
            assert dummy_target.follow_cancel.is_set()
            assert len(app.query("#log-pane")) == 0
            assert wl.display is True
            assert wl.has_focus
            assert wl.highlighted == 1  # c1 remains highlighted
            assert not app.has_class("-logs-open")
            assert not app.screen.has_class("-logs-open")

            # Escape in workload list unpins and returns to host table
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert app._pinned_host is None
            assert table.has_focus

    asyncio.run(_run())


def test_cluster_monitor_app_table_refocus_closes_log_pane(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import DataTable
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import LogPane

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    act2 = HostActivity(host="h2", workloads=())
    frame = MonitorFrame(hosts=(act1, act2))
    session = _FakeLiveMonitorSession(frame)

    dummy_target = _DummyLogTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: dummy_target)

    app = ClusterMonitorApp(session, ["h1", "h2"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.05)
            table = app.query_one("#host-table", DataTable)

            # Pin h1 and open log pane
            await pilot.press("enter")  # select h1
            await pilot.pause(0.05)
            await pilot.press("enter")  # open c1 log pane
            await pilot.pause(0.1)

            log_pane = app.query_one("#log-pane", LogPane)
            assert log_pane is not None

            # User clicks table on h2 row
            await pilot.click("#host-table", offset=(3, 2))
            await pilot.pause(0.1)

            # Log pane is closed and cancelled
            assert dummy_target.follow_cancel is not None
            assert dummy_target.follow_cancel.is_set()
            assert len(app.query("#log-pane")) == 0
            assert app._pinned_host is None
            assert table.has_focus

    asyncio.run(_run())


def test_cluster_monitor_app_dump_modal(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import DumpLogScreen, LogScope

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    frame = MonitorFrame(hosts=(act1,))
    session = _FakeLiveMonitorSession(frame)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)

            # Select h1
            await pilot.press("enter")
            await pilot.pause(0.05)

            # 'd' in workload list pushes DumpLogScreen
            await pilot.press("d")
            await pilot.pause(0.1)

            assert isinstance(app.screen, DumpLogScreen)
            assert app.screen.host == "h1"
            assert app.screen.container == "c1"
            assert app.screen.chosen_scope == LogScope.CONTAINER

            # Cancel modal
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert not isinstance(app.screen, DumpLogScreen)

    asyncio.run(_run())


def test_cluster_monitor_app_stop_modal_cancel(tmp_path):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import StopWorkloadScreen, WorkloadList

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    frame = MonitorFrame(hosts=(act1,))
    session = _FakeLiveMonitorSession(frame)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            await pilot.press("enter")  # pin h1, focus the workload list
            await pilot.pause(0.05)
            wl = app.query_one("#workload-list", WorkloadList)
            assert wl.highlighted == 1

            await pilot.press("k")
            await pilot.pause(0.1)
            assert isinstance(app.screen, StopWorkloadScreen)
            assert app.screen.cluster_id == "cid1"
            assert app.screen.container == "c1"

            await pilot.press("escape")
            await pilot.pause(0.05)
            assert not isinstance(app.screen, StopWorkloadScreen)
            assert app.focused is wl
            assert wl.highlighted == 1

    asyncio.run(_run())


def test_cluster_monitor_app_stop_modal_confirmed_stops_workload(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.api import StopResult
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import StopWorkloadScreen

    calls: list[dict] = []

    def _fake_stop(**kwargs):
        calls.append(kwargs)
        return StopResult(workload_id=kwargs.get("workload_id", ""), hosts_targeted=("h1",), containers_removed=1)

    monkeypatch.setattr("sparkrun.cli._monitor_tui.stop", _fake_stop)

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    frame = MonitorFrame(hosts=(act1,))
    session = _FakeLiveMonitorSession(frame)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            await pilot.press("enter")
            await pilot.pause(0.05)
            await pilot.press("k")
            await pilot.pause(0.1)
            await pilot.press("right")  # Cancel -> Stop
            await pilot.pause(0.05)
            await pilot.press("enter")
            assert not isinstance(app.screen, StopWorkloadScreen)

            for _ in range(40):
                if calls:
                    break
                await pilot.pause(0.05)
            assert calls == [{"workload_id": "cid1", "cluster": None, "cache_dir": str(tmp_path), "sctx": None}]

    asyncio.run(_run())


def test_cluster_monitor_app_stop_modal_button_confirms(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.api import StopResult
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import StopWorkloadScreen

    calls: list[dict] = []

    def _fake_stop(**kwargs):
        calls.append(kwargs)
        return StopResult(workload_id=kwargs.get("workload_id", ""), hosts_targeted=("h1",), containers_removed=1)

    monkeypatch.setattr("sparkrun.cli._monitor_tui.stop", _fake_stop)

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w1,)),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            await pilot.press("enter")
            await pilot.pause(0.05)
            await pilot.press("k")
            await pilot.pause(0.1)
            await pilot.click("#stop-confirm")
            await pilot.pause(0.05)
            assert not isinstance(app.screen, StopWorkloadScreen)

            for _ in range(40):
                if calls:
                    break
                await pilot.pause(0.05)
            assert calls and calls[0]["workload_id"] == "cid1"

    asyncio.run(_run())


def test_cluster_monitor_app_stop_modal_failure_does_not_crash(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.api import JobNotFound
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import StopWorkloadScreen

    calls: list[dict] = []

    def _fake_stop(**kwargs):
        calls.append(kwargs)
        raise JobNotFound("no job metadata")

    monkeypatch.setattr("sparkrun.cli._monitor_tui.stop", _fake_stop)

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    frame = MonitorFrame(hosts=(act1,))
    session = _FakeLiveMonitorSession(frame)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            await pilot.press("enter")
            await pilot.pause(0.05)
            await pilot.press("k")
            await pilot.pause(0.1)
            await pilot.press("enter")  # confirm
            for _ in range(40):
                if calls:
                    break
                await pilot.pause(0.05)
            await pilot.pause(0.05)  # let the error notification land on the UI thread
            assert not isinstance(app.screen, StopWorkloadScreen)

    asyncio.run(_run())


def test_cluster_monitor_app_r_opens_recipe_picker(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import DataTable
    from sparkrun.cli._monitor_recipes import RecipePickerScreen
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.search_recipes", lambda **_kwargs: [])

    act1 = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="10"), workloads=())
    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(act1,)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            table = app.query_one("#host-table", DataTable)
            assert table.has_focus  # the initial state is the node selection table
            await pilot.press("r")
            await pilot.pause(0.1)
            assert isinstance(app.screen, RecipePickerScreen)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, RecipePickerScreen)
            assert table.has_focus

    asyncio.run(_run())


def test_cluster_monitor_app_r_opens_picker_from_workload_list_and_stays_inert_in_modes(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_recipes import RecipePickerScreen
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import StopWorkloadScreen, WorkloadList

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.search_recipes", lambda **_kwargs: [])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _DummyLogTarget())

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(act1,)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            await pilot.press("enter")  # pin h1, focus the workload list
            await pilot.pause(0.05)
            assert app.query_one("#workload-list", WorkloadList).has_focus

            # r is not host-table-only: the workload list routes it too.
            await pilot.press("r")
            await pilot.pause(0.1)
            assert isinstance(app.screen, RecipePickerScreen)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, RecipePickerScreen)
            assert app.query_one("#workload-list", WorkloadList).has_focus

            await pilot.press("escape")  # back to the host table
            await pilot.pause(0.05)
            assert app.query_one("#host-table").has_focus

            # A pushed modal owns focus: r must not stack the picker over it.
            await pilot.press("enter")
            await pilot.pause(0.05)
            await pilot.press("k")
            await pilot.pause(0.1)
            assert isinstance(app.screen, StopWorkloadScreen)
            await pilot.press("r")
            await pilot.pause(0.1)
            assert isinstance(app.screen, StopWorkloadScreen)
            await pilot.press("escape")
            await pilot.pause(0.05)

            # The log pane is a viewing mode: r stays inert there.
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert len(app.query("#log-pane")) == 1
            await pilot.press("r")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, RecipePickerScreen)

    asyncio.run(_run())


def test_cluster_monitor_app_recipe_launch_opens_log_pane(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import time as _time
    from sparkrun.api import RunOptions, RunResult
    from sparkrun.cli._monitor_recipes import RecipeLaunch
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import LogPane

    calls: list[RunOptions] = []
    launched = RunResult(
        workload_id="sparkrun_aa_bb",
        host_list=("h1",),
        placement=None,
        scheduler="greedy",
        runtime="vllm-ray",
        executor="docker",
        started_at=_time.time(),
        dry_run=False,
        is_solo=True,
        rc=0,
    )

    def _fake_run(options, **_kwargs):
        calls.append(options)
        return launched

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", _fake_run)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _DummyLogTarget())

    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    launch = RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe")

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(launch)
            for _ in range(40):
                if calls:
                    break
                await pilot.pause(0.05)
            assert calls and calls[0].recipe == "demo-recipe"

            # api.run returns at containers-up; the occupancy sweep names the workload next tick.
            w1 = RunningWorkload(
                cluster_id="sparkrun_aa_bb",
                containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
            )
            session.set_frame(MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w1,)),)))
            for _ in range(40):
                if app.query("#log-pane") and app.query_one("#log-pane", LogPane).has_focus_within:
                    break
                await pilot.pause(0.05)

            pane = app.query_one("#log-pane", LogPane)
            assert pane.cluster_id == "sparkrun_aa_bb"
            assert pane.container == "c1"
            assert pane.has_focus_within  # focus lands in the log view inside the pane
            assert app._pending_log_open is None

    asyncio.run(_run())


def test_cluster_monitor_app_recipe_launch_mounts_from_resolved_source(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import time as _time
    from sparkrun.api import RunOptions, RunResult
    from sparkrun.cli._monitor_recipes import RecipeLaunch
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import LogPane

    launched = RunResult(
        workload_id="sparkrun_aa_bb",
        host_list=("h1",),
        placement=None,
        scheduler="greedy",
        runtime="vllm-ray",
        executor="docker",
        started_at=_time.time(),
        dry_run=False,
        is_solo=True,
        rc=0,
    )
    resolved: list[str] = []

    def _resolve(workload_id, **_kwargs):
        resolved.append(workload_id)
        return _DummyLogTarget()

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: launched)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", _resolve)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _DummyLogTarget())

    # The frame never gains the workload: the pane must open from the resolved source alone.
    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    launch = RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe")

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(launch)
            for _ in range(40):
                if app.query("#log-pane"):
                    break
                await pilot.pause(0.05)

            pane = app.query_one("#log-pane", LogPane)
            assert pane.cluster_id == "sparkrun_aa_bb"
            assert pane.container == "c1"
            assert pane.has_focus_within
            assert app._pending_log_open is None
            assert resolved == ["sparkrun_aa_bb"]
            assert not session.frame().for_host("h1").workloads

    asyncio.run(_run())


def test_cluster_monitor_app_recipe_launch_falls_back_when_resolve_fails(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import time as _time
    from sparkrun.api import RunOptions, RunResult
    from sparkrun.cli._monitor_recipes import RecipeLaunch
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import LogPane

    launched = RunResult(
        workload_id="sparkrun_aa_bb",
        host_list=("h1",),
        placement=None,
        scheduler="greedy",
        runtime="vllm-ray",
        executor="docker",
        started_at=_time.time(),
        dry_run=False,
        is_solo=True,
        rc=0,
    )

    def _boom(workload_id, **_kwargs):
        raise RuntimeError("job metadata not written yet")

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: launched)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", _boom)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _DummyLogTarget())

    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    notifications: list[str] = []
    app.notify = lambda message, **_kwargs: notifications.append(message)  # type: ignore[method-assign]
    launch = RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe")

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(launch)
            for _ in range(40):
                if app._pending_log_open is not None:
                    break
                await pilot.pause(0.05)
            assert app._pending_log_open is not None

            # The occupancy sweep names the workload on a later tick.
            w1 = RunningWorkload(
                cluster_id="sparkrun_aa_bb",
                containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
            )
            session.set_frame(MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w1,)),)))
            for _ in range(40):
                if app.query("#log-pane"):
                    break
                await pilot.pause(0.05)

            pane = app.query_one("#log-pane", LogPane)
            assert pane.cluster_id == "sparkrun_aa_bb"
            assert pane.container == "c1"
            assert app._pending_log_open is None
            assert not any("press enter on it" in message for message in notifications)

    asyncio.run(_run())


def test_cluster_monitor_app_second_log_pane_replaces_the_first(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import time as _time
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _DummyLogTarget())

    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._mount_log_pane("h1", "webscrape_aa", "c1")
            await pilot.pause(0.1)
            assert len(app.query("#log-pane")) == 1

            # A launch's pending auto-open can fire while another pane is up.
            # It closes the old pane and mounts on the next tick, once Textual
            # has pruned it — never stacking a second #log-pane.
            w2 = RunningWorkload(
                cluster_id="webscrape_bb",
                containers=(ContainerDetail(name="c9", role="head", status="Up", image="img"),),
            )
            session.set_frame(MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w2,)),)))
            app._pending_log_open = ("webscrape_bb", _time.monotonic() + 15.0, None)
            app._refresh()
            await pilot.pause(0.1)
            app._refresh()
            await pilot.pause(0.1)

            panes = list(app.query("#log-pane"))
            assert len(panes) == 1
            assert panes[0].cluster_id == "webscrape_bb"
            assert panes[0].has_focus_within
            assert app._pinned_host == "h1"
            assert app._pending_log_open is None

    asyncio.run(_run())


def test_cluster_monitor_app_launch_replaces_an_open_pane_without_the_sweep(tmp_path, monkeypatch):
    """A launch while another pane is up must not wait for the occupancy sweep.

    The replacement mount cannot happen until Textual prunes the old pane, so
    the pending entry carries the resolved ``(host, container)`` and retries
    from it; the frame never lists the new workload here.
    """
    pytest.importorskip("textual")
    import asyncio
    import time as _time
    from sparkrun.api import RunResult
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _DummyLogTarget())

    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    launched = RunResult(
        workload_id="sparkrun_aa_bb",
        host_list=("h1",),
        placement=None,
        scheduler="greedy",
        runtime="vllm-ray",
        executor="docker",
        started_at=_time.time(),
        dry_run=False,
        is_solo=True,
        rc=0,
    )

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._mount_log_pane("h1", "old_aa", "c1")
            await pilot.pause(0.1)
            app._on_recipe_launched(launched, ("h1", "c7"))
            for _ in range(20):
                panes = list(app.query("#log-pane"))
                if panes and panes[0].cluster_id == "sparkrun_aa_bb":
                    break
                app._refresh()
                await pilot.pause(0.05)
            panes = list(app.query("#log-pane"))
            assert len(panes) == 1
            assert panes[0].cluster_id == "sparkrun_aa_bb"
            assert panes[0].container == "c7"
            assert app._pending_log_open is None
            # The frame never listed the workload: no sweep was involved.
            assert not session.frame().for_host("h1").workloads

    asyncio.run(_run())


def test_cluster_monitor_app_keeps_launch_logs_off_the_terminal(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import io
    import logging
    import sys
    import threading
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    stream = io.StringIO()
    terminal = logging.StreamHandler(stream)  # same shape _setup_logging installs
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers[:] = [terminal]
    root.setLevel(logging.INFO)

    # Library-shaped leak: a handler attached to a named logger before Textual
    # swapped the streams keeps the real terminal, so root parking alone is not
    # enough (hf_hub, SAF and plugins do exactly this).
    library_handler = logging.StreamHandler(sys.__stderr__)
    library = logging.getLogger("tests.library_shaped")
    library.addHandler(library_handler)
    library.setLevel(logging.INFO)

    # A FileHandler is a StreamHandler too, but it does not paint on the
    # terminal, so it must keep logging while the app is mounted.
    log_file = tmp_path / "library.log"
    file_handler = logging.FileHandler(log_file)
    file_logger = logging.getLogger("tests.file_logged")
    file_logger.addHandler(file_handler)
    file_logger.setLevel(logging.INFO)

    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    notifications: list[str] = []
    app.notify = lambda message, **_kwargs: notifications.append(message)  # type: ignore[method-assign]

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            # Root sits at WARNING: line-oriented streaming consumers gate on
            # isEnabledFor(INFO) and inherit the terminal otherwise.
            assert root.level == logging.WARNING
            assert root.isEnabledFor(logging.INFO) is False
            # The launcher logs its progress through the root logger; while the
            # TUI owns the screen none of it may reach the terminal handler.
            logging.getLogger("sparkrun.launcher").info("Pulling image (leak check)")
            library.info("library leak check")
            file_logger.info("file leak check")
            await pilot.pause(0.05)
            assert "leak check" not in stream.getvalue()
            assert library_handler not in library.handlers
            assert file_handler in file_logger.handlers
            assert "file leak check" in log_file.read_text()

            # Warnings from the launch worker (another thread) become toasts.
            worker = threading.Thread(target=lambda: logging.getLogger("sparkrun.launcher").warning("disk at 95%"))
            worker.start()
            for _ in range(20):
                if notifications:
                    break
                await pilot.pause(0.05)
            worker.join(timeout=1)
            assert notifications == ["disk at 95%"]

        # Leaving the app restores the CLI's level and handlers, root and named alike.
        assert app._saved_root_handlers is None
        assert root.level == logging.INFO
        assert library.handlers == [library_handler]
        logging.getLogger("sparkrun.launcher").info("after the app")
        library.info("library after the app")
        assert "after the app" in stream.getvalue()

    try:
        asyncio.run(_run())
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        library.removeHandler(library_handler)
        file_logger.removeHandler(file_handler)
        file_handler.close()


def test_cluster_monitor_app_recipe_launch_failures(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import time as _time
    from sparkrun.api import RunOptions, RunResult
    from sparkrun.cli._monitor_recipes import RecipeLaunch
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    def _boom(options, **_kwargs):
        raise RuntimeError("no capacity")

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", _boom)
    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    launch = RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe")

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(launch)
            await pilot.pause(0.3)
            assert not app.query("#log-pane")
            assert app._pending_log_open is None

            # A nonzero rc is a failed launch too.
            bad = RunResult(
                workload_id="sparkrun_cc_dd",
                host_list=("h1",),
                placement=None,
                scheduler="greedy",
                runtime="vllm-ray",
                executor="docker",
                started_at=_time.time(),
                dry_run=False,
                is_solo=True,
                rc=1,
            )
            monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: bad)
            app._on_recipe_picked(launch)
            await pilot.pause(0.3)
            assert not app.query("#log-pane")
            assert app._pending_log_open is None

            # A launch whose workload never shows up degrades to a notify at the
            # deadline instead of waiting forever.
            app._pending_log_open = ("sparkrun_missing", _time.monotonic() - 1.0, None)
            app._refresh()
            await pilot.pause(0.05)
            assert app._pending_log_open is None
            assert not app.query("#log-pane")

    asyncio.run(_run())


def test_cluster_monitor_app_dry_run_arms_no_log_open(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import time as _time
    from sparkrun.api import RunOptions, RunResult
    from sparkrun.cli._monitor_recipes import RecipeLaunch
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    dry = RunResult(
        workload_id="sparkrun_dry_run",
        host_list=("h1",),
        placement=None,
        scheduler="greedy",
        runtime="vllm-ray",
        executor="docker",
        started_at=_time.time(),
        dry_run=True,
        is_solo=True,
        rc=0,
    )
    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: dry)
    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    notifications: list[str] = []
    app.notify = lambda message, **_kwargs: notifications.append(message)  # type: ignore[method-assign]
    launch = RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe")

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(launch)
            await pilot.pause(0.3)
            assert app._pending_log_open is None
            assert any("dry run" in message.lower() for message in notifications)
            assert not any("Launched" in message for message in notifications)
            app._refresh()
            await pilot.pause(0.05)
            assert not app.query("#log-pane")

    asyncio.run(_run())


# --------------------------------------------------------------------------
# ClusterMonitorApp Background Log-Metrics Tracking
# --------------------------------------------------------------------------

# One vLLM v1 engine stat line as the container emits it (vllm/v1/metrics/loggers.py).
_VLLM_STAT_LINE = (
    "INFO 10-17 08:12:34 [loggers.py:331] Engine 000: Avg prompt throughput: 123.4 tokens/s, "
    "Avg generation throughput: 45.6 tokens/s, Running: 3 reqs, Waiting: 1 reqs, "
    "GPU KV cache usage: 12.3%, Prefix cache hit rate: 45.6%"
)


class _MetricsLogTarget:
    """Log target double for the app-level tracker: one page per source, no processes."""

    def __init__(self, lines: list[LogLine], error_on_page: Exception | None = None, sources: list[LogSource] | None = None) -> None:
        self.sources = tuple(sources or [LogSource(host="h1", container="c1", role="head", rank=0)])
        self.runtime = "vllm-ray"
        self._lines = list(lines)
        self.error_on_page = error_on_page
        self.page_calls: list[dict] = []

    def page(self, source, *, skip, count=500, cancel=None, timeout=30.0):
        self.page_calls.append({"source": source, "skip": skip, "count": count})
        if self.error_on_page:
            raise self.error_on_page
        return list(self._lines[-count:])

    def follow(self, tail=200, cancel=None):
        return iter(())


def _tracker_frame(workload: RunningWorkload | None) -> MonitorFrame:
    workloads = (workload,) if workload is not None else ()
    return MonitorFrame(hosts=(HostActivity(host="h1", workloads=workloads),))


def _tracker_workload() -> RunningWorkload:
    return RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )


async def _run_metrics_tick(app) -> None:
    """Drive one tracker tick to completion (never wait on the 30 s timer)."""
    worker = app._metrics_tick()
    await worker.wait()


def test_monitor_metrics_interval_is_30_seconds():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    assert ClusterMonitorApp.METRICS_TRACK_INTERVAL_S == 30.0


def test_metrics_tail_read_fits_the_scraper_alignment_window():
    """A tail page longer than the alignment window would re-ingest seen lines.

    ``observe_page`` remembers only ``_PAGE_TAIL`` line hashes of the previous
    page, so a tracker read wider than that caps the overlap and folds already
    seen samples into every window mean — a silent bias, not a crash.
    """
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_metrics import _PAGE_TAIL
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    assert ClusterMonitorApp.METRICS_TAIL_LINES <= _PAGE_TAIL


@pytest.fixture(autouse=True)
def _no_live_metrics_tracking(monkeypatch):
    """The app primes a metrics tick at mount; keep every pilot test off real hosts.

    Tracker tests patch this same seam with their own target, which wins because a
    test's monkeypatch runs after this fixture.
    """
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", lambda *a, **kw: _MetricsLogTarget([]))


async def _wait_for_scraper(app, cluster_id: str, timeout: float = 3.0):
    """Wait for the app to hold a scraper for *cluster_id* (the mount-time prime)."""
    import asyncio

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        scraper = app.metrics_scraper_for(cluster_id)
        if scraper is not None:
            return scraper
        await asyncio.sleep(0.02)
    return None


def test_monitor_primes_metrics_without_waiting_an_interval(tmp_path, monkeypatch):
    """A monitor opened on a running workload shows its metrics at once, not 30 s later."""
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    target = _MetricsLogTarget([LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE)])
    calls: list[dict] = []

    def _open(*a, **kw):
        calls.append({"workload_id": a[0] if a else None, **kw})
        return target

    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", _open)
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            scraper = await _wait_for_scraper(app, "cid1")
            assert scraper is not None  # primed at mount, not one interval in
            assert scraper.latest("gen") == 45.6
            assert scraper.history("gen") == [45.6]
            # Naming comes from the job's recorded topology: passing the monitored
            # hosts would resolve the solo container name and read nothing.
            assert calls[0]["workload_id"] == "cid1"
            assert "hosts" not in calls[0]
            assert target.page_calls[0]["skip"] == 0
            assert target.page_calls[0]["count"] == ClusterMonitorApp.METRICS_PRIME_LINES

    asyncio.run(_run())


def test_monitor_initial_read_scans_back_for_measured_values(tmp_path, monkeypatch):
    """The first read reaches past an idle streak of probes, so the footer opens measured."""
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    tail = ClusterMonitorApp.METRICS_TAIL_LINES
    assert ClusterMonitorApp.METRICS_PRIME_LINES > tail
    probes = [
        LogLine(host="h1", container="c1", text='INFO:     127.0.0.1:60082 - "GET /health HTTP/1.1" 200 OK') for _ in range(tail + 100)
    ]
    # The last stat line sits well outside a plain tail page: exactly the idle
    # engine case where a shallow read reports nothing but zeros.
    target = _MetricsLogTarget([LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE), *probes])
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", lambda *a, **kw: target)

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            scraper = await _wait_for_scraper(app, "cid1")
            assert scraper is not None
            assert scraper.latest("gen") == 45.6  # measured from the deeper first read
            assert target.page_calls[0]["count"] == ClusterMonitorApp.METRICS_PRIME_LINES

            # Steady state reads the shallow tail; the last measured value holds.
            await _run_metrics_tick(app)
            assert target.page_calls[1]["count"] == ClusterMonitorApp.METRICS_TAIL_LINES
            assert scraper.latest("gen") == 45.6

    asyncio.run(_run())


def test_monitor_metrics_reach_an_open_pane_footer_without_a_tick(tmp_path, monkeypatch):
    """Opening a pane renders the already-tracked sample immediately."""
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import Static
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import LogPane

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _DummyLogTarget())
    monkeypatch.setattr(
        "sparkrun.cli._monitor_tui.open_log_target",
        lambda *a, **kw: _MetricsLogTarget([LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE)]),
    )

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            assert await _wait_for_scraper(app, "cid1") is not None

            await pilot.press("enter")  # pin h1
            await pilot.pause(0.05)
            await pilot.press("enter")  # open c1 log pane
            await pilot.pause(0.15)

            footer = app.query_one("#log-pane", LogPane).query_one("#log-metrics", Static)
            assert footer.display is True
            assert "tg" in str(footer.content)  # 5-column label field
            assert "45.6" in str(footer.content)

    asyncio.run(_run())


def test_monitor_tracker_drops_vanished_workloads(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    monkeypatch.setattr(
        "sparkrun.cli._monitor_tui.open_log_target",
        lambda *a, **kw: _MetricsLogTarget([LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE)]),
    )

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            assert await _wait_for_scraper(app, "cid1") is not None

            session.set_frame(_tracker_frame(None))
            app._refresh()
            await _run_metrics_tick(app)
            assert app.metrics_scraper_for("cid1") is None

    asyncio.run(_run())


def test_monitor_tracker_reopens_target_after_read_failure(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    target = _MetricsLogTarget([], error_on_page=RuntimeError("ssh dropped"))
    opens: list[str] = []

    def _open(*a, **kw):
        opens.append(a[0] if a else "")
        return target

    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", _open)
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            await asyncio.sleep(0.1)  # let the mount-time prime settle
            baseline = len(opens)

            await _run_metrics_tick(app)  # fails again: the cached target is dropped
            assert len(opens) == baseline + 1
            assert app.metrics_scraper_for("cid1") is None  # no sample from a failed read

            target.error_on_page = None
            target._lines = [LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE)]
            await _run_metrics_tick(app)
            scraper = await _wait_for_scraper(app, "cid1")
            assert scraper is not None and scraper.latest("gen") == 45.6

    asyncio.run(_run())


def test_monitor_tracker_forgets_a_restarted_ranks_old_source(tmp_path, monkeypatch):
    """A re-resolved target names the restarted container: the dead rank stops counting."""
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    before = _MetricsLogTarget(
        [LogLine(host="h1", container="c1", text="Avg generation throughput: 100.0 tokens/s")],
        sources=[LogSource(host="h1", container="c1", role="head", rank=0)],
    )
    after = _MetricsLogTarget(
        [LogLine(host="h1", container="c2", text="Avg generation throughput: 40.0 tokens/s")],
        sources=[LogSource(host="h1", container="c2", role="head", rank=0)],
    )
    targets = iter([before, after])
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", lambda *a, **kw: next(targets))

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            scraper = await _wait_for_scraper(app, "cid1")
            assert scraper is not None and scraper.latest("gen") == 100.0

            # The container instance changes: the next read fails against the old
            # target, which drops it so the following tick resolves the new one.
            before.error_on_page = RuntimeError("container instance changed")
            await _run_metrics_tick(app)
            await _run_metrics_tick(app)

            assert scraper.latest("gen") == 40.0  # 140.0 would mean the dead rank still counts

    asyncio.run(_run())


def test_monitor_metrics_window_mean_from_one_poll(tmp_path, monkeypatch):
    """A steady poll's several engine prints land as one point: the window mean."""
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    target = _MetricsLogTarget(
        [
            LogLine(host="h1", container="c1", text="Avg generation throughput: 10.0 tokens/s, GPU KV cache usage: 10.0%"),
            LogLine(host="h1", container="c1", text="Avg generation throughput: 40.0 tokens/s, GPU KV cache usage: 30.0%"),
        ]
    )
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", lambda *a, **kw: target)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            scraper = await _wait_for_scraper(app, "cid1")
            assert scraper is not None
            # The prime read is a deep page: its last sample seeds, never its mean.
            assert scraper.history("gen") == [40.0]

            target._lines.extend(
                [
                    LogLine(host="h1", container="c1", text="Avg generation throughput: 50.0 tokens/s, GPU KV cache usage: 60.0%"),
                    LogLine(host="h1", container="c1", text="Avg generation throughput: 60.0 tokens/s, GPU KV cache usage: 80.0%"),
                ]
            )
            await _run_metrics_tick(app)

            # Only the two appended prints are new to this poll: one point, their mean.
            assert scraper.history("gen") == [40.0, 55.0]
            assert scraper.latest("kv") == 70.0  # window mean; max applies across sources, not within one

    asyncio.run(_run())


def test_monitor_metrics_second_tick_only_samples_new_lines(tmp_path, monkeypatch):
    """A later poll re-reads the overlapping tail; only the appended line counts."""
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    target = _MetricsLogTarget([LogLine(host="h1", container="c1", text="Avg generation throughput: 40.0 tokens/s")])
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", lambda *a, **kw: target)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            scraper = await _wait_for_scraper(app, "cid1")
            assert scraper is not None
            assert scraper.history("gen") == [40.0]

            target._lines.append(LogLine(host="h1", container="c1", text="Avg generation throughput: 50.0 tokens/s"))
            await _run_metrics_tick(app)

            assert scraper.history("gen") == [40.0, 50.0]  # the re-read line adds nothing

    asyncio.run(_run())


def test_monitor_metrics_idle_page_adds_no_point(tmp_path, monkeypatch):
    """An unchanged tail page has no new samples, so it adds no flat point."""
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(_tracker_frame(_tracker_workload()))
    target = _MetricsLogTarget([LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE)])
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", lambda *a, **kw: target)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            scraper = await _wait_for_scraper(app, "cid1")
            assert scraper is not None
            assert len(scraper.history("gen")) == 1

            await _run_metrics_tick(app)
            await _run_metrics_tick(app)
            assert scraper.history("gen") == [45.6]

    asyncio.run(_run())


def test_live_monitor_history_records_each_reading_once(monkeypatch):
    """A session returns the same sample object until a new reading parses —
    on a stalled stream for up to interval*5 — so re-recording it would draw an
    outage as a flat live line."""
    stale = MonitorSample(hostname="h1", gpu_util_pct="5")
    tel = _FakeSession({"h1": HostTelemetry(host="h1", sample=stale)})
    session = _open_history_session(monkeypatch, session=tel)
    try:
        for _ in range(4):
            session._record_once()
        assert [sample.gpu_util_pct for _, sample in session.history("h1")] == ["5"]
        tel._snap = {"h1": HostTelemetry(host="h1", sample=MonitorSample(hostname="h1", gpu_util_pct="5"))}
        session._record_once()  # an equal but new reading is a new point
        assert len(session.history("h1")) == 2
    finally:
        session.close()


def test_live_monitor_history_points_from_config(monkeypatch):
    from sparkrun.core.config import SparkrunConfig

    monkeypatch.setattr(SparkrunConfig, "monitor_history_points", property(lambda self: 7))
    session = _open_history_session(monkeypatch)
    explicit = _open_history_session(monkeypatch, history_points=3)
    try:
        assert session.history_points == 7  # unset → monitor.history_points
        assert explicit.history_points == 3  # an explicit argument wins
    finally:
        session.close()
        explicit.close()


def test_live_monitor_history_snapshot_is_wire_encodable(monkeypatch):
    from sparkrun.api._wire import to_wire

    session = _open_history_session(monkeypatch)
    try:
        session._record_once()
        (series,) = to_wire(session.history_snapshot())
    finally:
        session.close()
    assert series["host"] == "h1"
    assert series["points"][0]["sample"]["gpu_util_pct"] == "1"
    assert isinstance(series["points"][0]["recorded_at"], float)


def test_monitor_sample_wire_allowlist_covers_every_field():
    """The wire codec is an allowlist, so a field added to MonitorSample but not
    to DTO_FIELDS is silently dropped from the daemon monitor stream."""
    import dataclasses

    from sparkrun.api._wire_fields import DTO_FIELDS

    assert DTO_FIELDS["sparkrun.core.monitoring.MonitorSample"] == [f.name for f in dataclasses.fields(MonitorSample)]


@pytest.mark.parametrize("slow_first", [False, True])
def test_metrics_poll_finishes_and_publishes_healthy_workloads_despite_slow_source(tmp_path, monkeypatch, slow_first):
    import asyncio
    import threading
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    entered = threading.Event()
    release = threading.Event()
    reads: list[threading.Event] = []

    class _SlowTarget(_MetricsLogTarget):
        def page(self, source, *, cancel, **kwargs):
            reads.append(cancel)
            entered.set()
            assert release.wait(5), "test did not release slow log read"
            return []

    fast_id, slow_id = ("z-fast", "a-slow") if slow_first else ("a-fast", "z-slow")
    fast = _MetricsLogTarget([LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE)])
    slow = _SlowTarget([])
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", lambda cid, **kw: fast if cid == fast_id else slow)
    frame = MonitorFrame(hosts=(HostActivity(host="h1", workloads=(RunningWorkload(fast_id), RunningWorkload(slow_id))),))
    app = ClusterMonitorApp(_FakeLiveMonitorSession(frame), ["h1"], cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)):
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                if not slow_first:
                    scraper = app.metrics_scraper_for(fast_id)
                    assert scraper is not None and scraper.latest("gen") == 45.6
                for _ in range(3):
                    await _run_metrics_tick(app)
                assert len(reads) == 1  # ticks neither cancel nor overlap the in-flight read
                assert not reads[0].is_set()
            finally:
                release.set()
            scraper = await _wait_for_scraper(app, fast_id)
            assert scraper is not None and scraper.latest("gen") == 45.6

    asyncio.run(_run())
    assert reads[0].is_set()  # app shutdown still cancels its readers


# -- recipe launch: plan reuse, post-launch lifecycle, quitting mid-launch ------


def _launched_result(workload_id: str = "sparkrun_aa_bb", *, launch_result=None, dry_run: bool = False):
    from sparkrun.api import RunResult

    return RunResult(
        workload_id=workload_id,
        host_list=("h1",),
        placement=None,
        scheduler="greedy",
        runtime="vllm-ray",
        executor="docker",
        started_at=time.time(),
        dry_run=dry_run,
        is_solo=True,
        rc=0,
        launch_result=launch_result,
    )


def _monitor_app(tmp_path):
    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    session = _FakeLiveMonitorSession(MonitorFrame(hosts=(HostActivity(host="h1"),)))
    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))
    notes: list[tuple[str, str]] = []
    app.notify = lambda message, severity="information", **_kw: notes.append((severity, message))  # type: ignore[method-assign]
    return app, notes


async def _until(pilot, predicate, tries: int = 100) -> None:
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition never met")


def test_recipe_launch_hands_the_preflight_plan_to_run(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from types import SimpleNamespace
    from sparkrun.api import RunOptions
    from sparkrun.cli._monitor_recipes import RecipeLaunch

    calls: list[tuple] = []

    def _fake_run(options, **kwargs):
        calls.append((options, kwargs.get("plan")))
        return _launched_result()

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", _fake_run)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", mock.Mock(side_effect=RuntimeError("no metadata")))
    app, notes = _monitor_app(tmp_path)
    preflight = SimpleNamespace(host_list=("h1",))
    options = RunOptions(recipe="demo-recipe")
    launch = RecipeLaunch(options=options, label="demo-recipe", plan=preflight)

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(launch)
            await _until(pilot, lambda: bool(calls))

    asyncio.run(_run())
    assert calls == [(options, preflight)]
    assert ("information", "Launching demo-recipe on h1…") in notes


def _hooked_launch_result(*, post_exec=None, post_commands=None):
    from types import SimpleNamespace

    recipe = SimpleNamespace(post_exec=post_exec or [], post_commands=post_commands or [])
    return _launched_result(launch_result=SimpleNamespace(recipe=recipe, recipe_trusted=True))


def test_recipe_launch_runs_the_post_launch_lifecycle(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.api import PostLaunchResult, RunOptions
    from sparkrun.cli._monitor_recipes import RecipeLaunch

    result = _hooked_launch_result(post_exec=["curl localhost"])
    lifecycle: list[tuple] = []

    def _fake_post_launch(run_result, **kwargs):
        lifecycle.append((run_result, kwargs.get("trust")))
        return PostLaunchResult(stopped=True)

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: result)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.post_launch", _fake_post_launch)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", mock.Mock(side_effect=RuntimeError("no metadata")))
    app, notes = _monitor_app(tmp_path)
    launch = RecipeLaunch(options=RunOptions(recipe="demo-recipe", trust=True), label="demo-recipe")

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(launch)
            await _until(pilot, lambda: any("post-launch hooks finished" in message for _s, message in notes))
            assert app.background_work.descriptions() == []

    asyncio.run(_run())
    # The same result and the same authorization the launch carried.
    assert lifecycle == [(result, True)]
    assert any("stop_after_post" in message for _s, message in notes)


def test_recipe_launch_reports_post_launch_failures(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.api import RunOptions, SparkrunError
    from sparkrun.cli._monitor_recipes import RecipeLaunch

    def _failing_post_launch(_result, **_kwargs):
        raise SparkrunError("post_exec[1] failed (exit 3)")

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: _hooked_launch_result(post_exec=["false"]))
    monkeypatch.setattr("sparkrun.cli._monitor_tui.post_launch", _failing_post_launch)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", mock.Mock(side_effect=RuntimeError("no metadata")))
    app, notes = _monitor_app(tmp_path)

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe"))
            await _until(pilot, lambda: any(severity == "error" for severity, _m in notes))

    asyncio.run(_run())
    assert ("error", "Post-launch hooks of demo-recipe failed: post_exec[1] failed (exit 3)") in notes


def test_recipe_launch_never_runs_local_post_commands(tmp_path, monkeypatch):
    """post_commands take the terminal; the monitor refuses them rather than run them under the TUI."""
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.api import RunOptions
    from sparkrun.cli._monitor_recipes import RecipeLaunch

    post_launch = mock.Mock()
    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: _hooked_launch_result(post_commands=["sudo x"]))
    monkeypatch.setattr("sparkrun.cli._monitor_tui.post_launch", post_launch)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", mock.Mock(side_effect=RuntimeError("no metadata")))
    app, notes = _monitor_app(tmp_path)

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe"))
            await _until(pilot, lambda: any(severity == "error" for severity, _m in notes))

    asyncio.run(_run())
    post_launch.assert_not_called()
    assert any("post_commands" in message and "run demo-recipe" in message for _s, message in notes)


def test_recipe_launch_without_hooks_skips_the_lifecycle(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.api import RunOptions
    from sparkrun.cli._monitor_recipes import RecipeLaunch

    post_launch = mock.Mock()
    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", lambda options, **kwargs: _hooked_launch_result())
    monkeypatch.setattr("sparkrun.cli._monitor_tui.post_launch", post_launch)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", mock.Mock(side_effect=RuntimeError("no metadata")))
    app, _notes = _monitor_app(tmp_path)

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            app._on_recipe_picked(RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe"))
            await _until(pilot, lambda: app._pending_log_open is not None)
            await _until(pilot, lambda: app.background_work.descriptions() == [])

    asyncio.run(_run())
    post_launch.assert_not_called()


def test_quit_during_a_launch_asks_first(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    import threading
    from sparkrun.api import RunOptions
    from sparkrun.cli._monitor_recipes import RecipeLaunch
    from textual.widgets import Static
    from sparkrun.cli._monitor_tui import QuitWhileBusyScreen

    entered = threading.Event()
    release = threading.Event()

    def _slow_run(options, **_kwargs):
        entered.set()
        assert release.wait(5), "test did not release the launch"
        return _launched_result(dry_run=True)

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", _slow_run)
    app, _notes = _monitor_app(tmp_path)

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            try:
                app._on_recipe_picked(RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe"))
                assert await asyncio.to_thread(entered.wait, 3)
                await pilot.press("q")
                await pilot.pause(0.1)
                assert isinstance(app.screen, QuitWhileBusyScreen)
                assert "the launch of demo-recipe" in str(app.screen.query_one("#quit-busy", Static).content)
                # ctrl+q is a priority binding that reaches action_quit too: one dialog.
                await pilot.press("ctrl+q")
                await pilot.pause(0.1)
                assert len([s for s in app.screen_stack if isinstance(s, QuitWhileBusyScreen)]) == 1
                # Keep monitoring is the focused default.
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert not isinstance(app.screen, QuitWhileBusyScreen)
                assert app.is_running

                await pilot.press("q")
                await pilot.pause(0.1)
                await pilot.click("#quit-confirm")
                await pilot.pause(0.1)
                assert app._exit
            finally:
                release.set()

    asyncio.run(_run())


def test_quit_without_work_in_flight_exits_at_once(tmp_path):
    pytest.importorskip("textual")
    import asyncio

    app, _notes = _monitor_app(tmp_path)

    async def _run():
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause(0.05)
            await pilot.press("q")
            await pilot.pause(0.1)
            assert app._exit

    asyncio.run(_run())


def test_exit_names_the_launch_the_shell_waits_for_and_late_results_go_to_stderr(tmp_path, monkeypatch, capsys):
    """After the screen is restored the shell says what it is waiting on; the late outcome must not raise."""
    pytest.importorskip("textual")
    import asyncio
    import threading
    from sparkrun.api import RunOptions
    from sparkrun.cli._monitor_recipes import RecipeLaunch

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    worker_errors: list[BaseException] = []

    def _slow_run(options, **_kwargs):
        entered.set()
        assert release.wait(10), "test did not release the launch"
        return _launched_result()

    monkeypatch.setattr("sparkrun.cli._monitor_tui.run", _slow_run)
    monkeypatch.setattr("sparkrun.cli._monitor_tui.open_log_target", mock.Mock(side_effect=RuntimeError("no metadata")))
    app, _notes = _monitor_app(tmp_path)
    original_post = app._run_post_launch

    def _post(result, launch):
        try:
            original_post(result, launch)
        except BaseException as error:  # noqa: BLE001 - recorded for the assertion below, then re-raised
            worker_errors.append(error)
            raise
        finally:
            finished.set()

    app._run_post_launch = _post  # type: ignore[method-assign]

    async def _auto(pilot):
        app._on_recipe_picked(RecipeLaunch(options=RunOptions(recipe="demo-recipe"), label="demo-recipe"))
        assert await asyncio.to_thread(entered.wait, 3)
        app.exit()

    async def _run():
        await app.run_async(headless=True, size=(110, 30), auto_pilot=_auto)
        # run_async has returned: the screen is gone and the launch still runs.
        assert "Waiting for the launch of demo-recipe to finish..." in capsys.readouterr().err
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)

    try:
        asyncio.run(_run())
    finally:
        release.set()
    assert worker_errors == []
    assert "Launched demo-recipe as sparkrun_aa_bb" in capsys.readouterr().err


def test_report_from_thread_falls_back_to_stderr_when_the_app_is_gone(tmp_path, capsys):
    pytest.importorskip("textual")
    app, notes = _monitor_app(tmp_path)
    app._report_from_thread("Stop failed for x: boom", severity="error")
    assert notes == []
    assert "Stop failed for x: boom" in capsys.readouterr().err


def test_cluster_monitor_refresh_tick_after_shutdown_is_a_no_op(tmp_path):
    """Textual prunes the screens before it closes the interval timers.

    A refresh tick landing in that window queried a pruned ``#host-table`` and
    raised NoMatches out of shutdown (seen on CI as a flaky monitor test).
    """
    pytest.importorskip("textual")
    import asyncio

    from sparkrun.cli._monitor_tui import ClusterMonitorApp

    act = HostActivity(host="h1", telemetry=_gb10_sample())
    app = ClusterMonitorApp(_FakeLiveMonitorSession(MonitorFrame(hosts=(act,))), ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.05)

    asyncio.run(_run())
    assert not app.is_running
    app._refresh()
