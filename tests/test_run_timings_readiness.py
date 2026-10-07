"""``sparkrun run`` timings + the background readiness watch.

``--timings`` originally printed its tree immediately after the launch
returned and immediately *before* attaching to the container logs.  Two
things were wrong with that, and only the first is cosmetic:

1. The tree was buried under ``docker logs --tail 100`` plus a live stream
   within milliseconds of being printed, and Ctrl-C exited without ever
   showing it again.
2. Worse, the figure people want — containers-running → serving, i.e. time
   to first inference — was **not measured at all** on the default path.
   ``wait_for_serve_ready`` was only ever called from
   ``post_launch_lifecycle``, which runs only when the recipe defines
   ``post_exec`` / ``post_commands``.  For every other recipe the tree
   reported distribution and container start, while the minutes of weight
   load and graph capture scrolled past unrecorded in the log stream.

The fix runs the readiness poll on a background thread *alongside* the log
stream: a one-line announcement is injected when the endpoint answers, and
the multi-line tree is printed once the stream has stopped.  Timings are
now on by default (``--no-timings`` suppresses the table only).

These tests pin the cancellation contract the watcher depends on, the
"cancelled is not failed" distinction in the timeline, and the CLI wiring.
"""

from __future__ import annotations

import math
import threading
import time
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
from sparkrun.core.launcher import ReadinessWatcher, ServeReadiness, wait_for_endpoint_ready
from sparkrun.core.timing import Timeline
from sparkrun.orchestration.health import wait_for_healthy, wait_for_port


@pytest.mark.parametrize(
    ("verbosity", "expected"),
    [(0, 2), (1, 3), (2, 4), (3, None), (4, None), (-1, 2)],
)
def test_run_timing_tree_depth_tracks_global_verbosity(verbosity, expected):
    from sparkrun.cli._run import _timing_tree_depth

    class Context:
        obj = {"verbose": verbosity}

        def find_root(self):
            return self

    assert _timing_tree_depth(Context()) == expected


# ---------------------------------------------------------------------------
# Cancellation in the underlying waiters
# ---------------------------------------------------------------------------


def test_wait_for_port_does_not_probe_once_cancelled():
    """An already-cancelled wait must not reach the network at all."""
    cancel = threading.Event()
    cancel.set()

    with mock.patch("sparkrun.orchestration.primitives.run_command_on_host") as probe:
        assert wait_for_port("h1", 8000, cancel=cancel) is False

    probe.assert_not_called()


def test_wait_for_port_wakes_from_its_retry_sleep():
    """Cancellation is noticed within one probe, not one retry interval.

    A plain ``time.sleep`` between polls is what let a cancelled wait
    outlive its caller: at the readiness defaults the health stage sleeps
    5s at a time, 120 times over.
    """
    cancel = threading.Event()
    probes = []

    def _never_ready(*args, **kwargs):
        probes.append(1)
        cancel.set()  # fires *during* the first probe, so the sleep must break
        return mock.Mock(success=False, stdout="", stderr="")

    t0 = time.monotonic()
    with mock.patch("sparkrun.orchestration.primitives.run_command_on_host", _never_ready):
        ready = wait_for_port("h1", 8000, max_retries=100, retry_interval=30, cancel=cancel)
    elapsed = time.monotonic() - t0

    assert ready is False
    assert len(probes) == 1
    assert elapsed < 5.0, "cancelled wait slept through its retry interval"


def test_wait_for_healthy_wakes_from_its_retry_sleep():
    cancel = threading.Event()

    def _refused(*args, **kwargs):
        cancel.set()
        raise OSError("connection refused")

    t0 = time.monotonic()
    with mock.patch("urllib.request.urlopen", _refused):
        healthy = wait_for_healthy("http://h1:8000/v1/models", max_retries=100, retry_interval=30, cancel=cancel)
    elapsed = time.monotonic() - t0

    assert healthy is False
    assert elapsed < 5.0


def test_uncancelled_waits_are_unchanged():
    """``cancel=None`` is the pre-existing path, byte for byte."""
    with mock.patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=mock.Mock(success=True)):
        assert wait_for_port("h1", 8000) is True


# ---------------------------------------------------------------------------
# Readiness budgets
# ---------------------------------------------------------------------------


def test_port_budget_is_wall_clock_not_a_retry_count():
    """A retry count is a poor proxy for time, and the gap is not small.

    Every attempt also pays a probe — an SSH round trip on a remote host.
    The old ``120 x 2s`` default bought 240s of *sleeping* but ran for 321s
    of wall clock, then reported the shortfall as an endpoint that never
    came up (measured against a 30B sglang launch that bound its port at
    775s).  A wall-clock budget means what it says.
    """
    probes = []

    def _probe(*args, **kwargs):
        probes.append(1)
        return mock.Mock(success=False, stdout="", stderr="")

    # A retry count that would have stopped the loop immediately must not,
    # because the wall-clock budget is what governs.
    with mock.patch("sparkrun.orchestration.primitives.run_command_on_host", _probe):
        assert wait_for_port("h1", 8000, max_retries=1, retry_interval=0, timeout_s=0.25) is False

    assert len(probes) > 1, "max_retries still bounded a wall-clock budget"

    # ...and the budget does stop it, rather than running forever.
    elapsed_probes = len(probes)
    assert elapsed_probes < 100_000


def test_an_infinite_budget_polls_until_cancelled():
    """What the background watcher uses: no budget, only the cancel."""
    cancel = threading.Event()
    probes = []

    def _probe(*args, **kwargs):
        probes.append(1)
        if len(probes) == 25:
            cancel.set()
        return mock.Mock(success=False, stdout="", stderr="")

    with mock.patch("sparkrun.orchestration.primitives.run_command_on_host", _probe):
        ready = wait_for_port("h1", 8000, retry_interval=0, cancel=cancel, timeout_s=math.inf)

    assert ready is False
    assert len(probes) == 25, "an unbounded wait stopped on something other than its cancel"


def test_retry_count_still_bounds_a_wait_with_no_timeout():
    """``timeout_s=None`` is the pre-existing path and keeps its semantics."""
    probes = []

    def _probe(*args, **kwargs):
        probes.append(1)
        return mock.Mock(success=False, stdout="", stderr="")

    with mock.patch("sparkrun.orchestration.primitives.run_command_on_host", _probe):
        assert wait_for_port("h1", 8000, max_retries=4, retry_interval=0) is False

    assert len(probes) == 4


def test_a_dead_container_still_aborts_regardless_of_the_budget():
    """Liveness — not the budget — is what detects a genuine failure.

    This is what makes a generous (or infinite) budget safe: a workload
    that actually died is caught within one interval either way.
    """
    with (
        mock.patch("sparkrun.orchestration.primitives.run_command_on_host", return_value=mock.Mock(success=False)),
        mock.patch("sparkrun.orchestration.health.is_container_running", return_value=False),
    ):
        ready = wait_for_port("h1", 8000, retry_interval=0, container_name="c0", timeout_s=math.inf)

    assert ready is False


def test_readiness_timeouts_are_configurable(tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import DEFAULT_HEALTH_READY_TIMEOUT_S, DEFAULT_PORT_READY_TIMEOUT_S

    def _cfg(data) -> SparkrunConfig:
        path = tmp_path / ("cfg-%d.yaml" % len(list(tmp_path.iterdir())))
        path.write_text(yaml.safe_dump(data))
        return SparkrunConfig(config_path=path)

    default = _cfg({})
    assert default.readiness_port_timeout_s == DEFAULT_PORT_READY_TIMEOUT_S
    assert default.readiness_health_timeout_s == DEFAULT_HEALTH_READY_TIMEOUT_S

    tuned = _cfg({"readiness": {"port_timeout_s": 4200, "health_timeout_s": 60}})
    assert tuned.readiness_port_timeout_s == 4200.0
    assert tuned.readiness_health_timeout_s == 60.0

    # A budget can only ever expire early, so "no budget" is a legitimate ask.
    assert _cfg({"readiness": {"port_timeout_s": 0}}).readiness_port_timeout_s == math.inf
    # Garbage falls back rather than disabling the wait.
    assert _cfg({"readiness": {"port_timeout_s": "soon"}}).readiness_port_timeout_s == DEFAULT_PORT_READY_TIMEOUT_S


# ---------------------------------------------------------------------------
# "Cancelled" is not "failed"
# ---------------------------------------------------------------------------


def _endpoint_kwargs(**over):
    runtime = mock.Mock()
    runtime.get_head_container_name.return_value = "sparkrun0_node_0"
    kwargs = dict(
        runtime=runtime,
        cluster_id="sparkrun0",
        host_list=["localhost"],
        is_solo=True,
        port=8000,
    )
    kwargs.update(over)
    return kwargs


def test_cancelled_wait_leaves_its_span_open_rather_than_failed():
    """Ctrl-C must not write "the port never opened" into the artifact.

    The waiters report cancellation and genuine failure with the same
    ``False``.  Closing the span ``error`` on that would claim the workload
    is broken when all that happened is we stopped watching.
    """
    from sparkrun.utils.cli_formatters import format_launch_timings

    cancel = threading.Event()
    cancel.set()
    tl = Timeline()

    readiness = wait_for_endpoint_ready(**_endpoint_kwargs(), timeline=tl, cancel=cancel)

    assert readiness.ready is False
    assert readiness.reason == "cancelled"

    spans = {s["name"]: s for s in tl.export()["spans"]}
    assert spans["serve.port_open"]["status"] == "open"
    assert "did not finish" in format_launch_timings(tl.export())


def test_genuine_port_failure_is_still_recorded_as_an_error():
    """The cancel branch must not swallow real failures."""
    tl = Timeline()

    with mock.patch("sparkrun.orchestration.health.wait_for_port", return_value=False):
        readiness = wait_for_endpoint_ready(**_endpoint_kwargs(), timeline=tl)

    assert readiness.reason == "port"
    spans = {s["name"]: s for s in tl.export()["spans"]}
    assert spans["serve.port_open"]["status"] == "error"


def test_cancelled_health_stage_is_distinguished_from_an_unhealthy_server():
    cancel = threading.Event()
    tl = Timeline()

    def _cancel_then_fail(*args, **kwargs):
        cancel.set()
        return False

    with (
        mock.patch("sparkrun.orchestration.health.wait_for_port", return_value=True),
        mock.patch("sparkrun.orchestration.health.wait_for_healthy", _cancel_then_fail),
    ):
        readiness = wait_for_endpoint_ready(**_endpoint_kwargs(), timeline=tl, cancel=cancel)

    assert readiness.reason == "cancelled"
    spans = {s["name"]: s for s in tl.export()["spans"]}
    assert spans["serve.health_ok"]["status"] == "open"
    # The stage that *did* complete keeps its real verdict.
    assert spans["serve.port_open"]["status"] == "ok"


# ---------------------------------------------------------------------------
# ReadinessWatcher
# ---------------------------------------------------------------------------


def _ready(**over) -> ServeReadiness:
    fields = dict(
        ready=True,
        head_host="h1",
        head_ip="10.0.0.1",
        port=8000,
        container="sparkrun0_node_0",
        port_wait_s=12.0,
        health_wait_s=30.0,
    )
    fields.update(over)
    return ServeReadiness(**fields)


def test_watcher_announces_through_its_callback():
    seen: list[ServeReadiness] = []

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=_ready()):
        watcher = ReadinessWatcher(mock.Mock(), on_ready=seen.append).start()
        readiness = watcher.stop(timeout=5.0)

    assert readiness is not None and readiness.ready
    assert [r.total_wait_s for r in seen] == [42.0]


def test_watcher_does_not_announce_a_wait_that_never_succeeded():
    seen: list[ServeReadiness] = []

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=_ready(ready=False, reason="port")):
        watcher = ReadinessWatcher(mock.Mock(), on_ready=seen.append).start()
        readiness = watcher.stop(timeout=5.0)

    assert seen == []
    assert readiness is not None and readiness.reason == "port"


def test_watcher_stop_cancels_a_wait_still_in_flight():
    """``stop()`` must return promptly and set the event the waiters poll."""
    started = threading.Event()
    observed: dict[str, bool] = {}

    def _long_wait(result, *, cancel=None, **kwargs):
        started.set()
        # Stands in for the real waiters' cancel-aware sleep.
        observed["cancelled"] = cancel.wait(10.0)
        return _ready(ready=False, reason="cancelled")

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", _long_wait):
        watcher = ReadinessWatcher(mock.Mock()).start()
        assert started.wait(5.0)
        t0 = time.monotonic()
        watcher.stop(timeout=5.0)
        elapsed = time.monotonic() - t0

    assert observed["cancelled"] is True
    assert elapsed < 5.0


def test_watcher_spans_are_rooted_explicitly_not_taken_from_the_stack():
    """The shared open-span stack is not safe to inherit from off-thread.

    ``Timeline.end`` closes everything open *above* its target, so a span
    the watcher took from the stack would be closed by the next main-thread
    ``end()`` — stamped with that thread's status, and turning the
    watcher's own ``end()`` into a silent no-op.  That would defeat both
    the "leave it open on cancel" contract and the ok/error verdict.
    """
    tl = Timeline()
    main_span = tl.begin("main.phase")  # deliberately left open across the watch

    def _wait(result, *, timeline=None, parent=None, **kwargs):
        timeline.end(timeline.begin("serve.port_open", parent=parent), status="ok")
        return _ready()

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", _wait):
        ReadinessWatcher(mock.Mock(), timeline=tl).start().stop(timeout=5.0)

    tl.end(main_span)
    spans = {s["name"]: s for s in tl.export()["spans"]}
    assert spans["serve.port_open"]["parent"] is None, "watcher span inherited the main thread's open span"


def test_serving_span_accounts_for_the_time_after_readiness():
    """Measured, not derived from the gap between the rows and the total.

    Without it the tree stopped accounting the moment the endpoint
    answered while the total kept running — a launch watched for two hours
    showed ~775s of rows under a 7695s total.
    """
    tl = Timeline()

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=_ready()):
        watcher = ReadinessWatcher(mock.Mock(), timeline=tl).start()
        watcher.stop(timeout=5.0)

    spans = {s["name"]: s for s in tl.export()["spans"]}
    serving = spans["serve.serving"]
    assert serving["status"] == "ok"
    assert serving["parent"] is None, "serving must sit beside run, not inside it"
    assert serving["attrs"]["label"] == "serving"


def test_no_serving_span_when_the_endpoint_never_served():
    """A row claiming "serving" for a launch that never served is a lie."""
    tl = Timeline()

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=_ready(ready=False, reason="port")):
        ReadinessWatcher(mock.Mock(), timeline=tl).start().stop(timeout=5.0)

    assert "serve.serving" not in {s["name"] for s in tl.export()["spans"]}


def test_serving_span_is_left_open_if_never_stopped():
    """An unclosed span renders "did not finish" rather than a duration."""
    tl = Timeline()

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=_ready()):
        watcher = ReadinessWatcher(mock.Mock(), timeline=tl).start()
        watcher._thread.join(timeout=5.0)  # let it open the span, but never stop()

    spans = {s["name"]: s for s in tl.export()["spans"]}
    assert spans["serve.serving"]["status"] == "open"


def test_watcher_exposes_a_failing_wait():
    """Exceptions complete the watch and cannot be mistaken for readiness."""
    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", side_effect=RuntimeError("ssh exploded")):
        watcher = ReadinessWatcher(mock.Mock()).start()
        assert watcher.stop(timeout=5.0) is None
        assert watcher.done.is_set()
        assert str(watcher.error) == "ssh exploded"


def test_watcher_swallows_a_failing_callback():
    def _boom(readiness):
        raise ValueError("bad render")

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=_ready()):
        watcher = ReadinessWatcher(mock.Mock(), on_ready=_boom).start()
        assert watcher.stop(timeout=5.0) is not None


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

_RECIPE_NAME = "test-timings-recipe"

_RECIPE_DATA = {
    "sparkrun_version": "2",
    "name": "Timings regression recipe",
    "description": "solo recipe used to exercise the run timings/readiness wiring",
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "sglang",
    "mode": "solo",
    "min_nodes": 1,
    "max_nodes": 1,
    "container": "scitrera/dgx-spark-sglang:latest",
    "defaults": {"port": 30000, "host": "0.0.0.0"},
    "metadata": {"model_params": 1700000000, "model_dtype": "float16"},
}

_HOST = "10.0.4.30"


@pytest.fixture
def run_env(tmp_path, monkeypatch, v):
    """A single-host cluster plus a discoverable solo recipe."""
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
    ClusterManager(config_root).create("wopr", [_HOST])

    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    recipe_file = recipe_dir / ("%s.yaml" % _RECIPE_NAME)
    recipe_file.write_text(yaml.safe_dump(_RECIPE_DATA))

    import sparkrun.core.recipe

    original = sparkrun.core.recipe.discover_cwd_recipes
    monkeypatch.setattr(
        sparkrun.core.recipe,
        "discover_cwd_recipes",
        lambda directory=None: [recipe_file] + original(directory),
    )

    idle = lambda hosts, **kwargs: ClusterStatus(  # noqa: E731
        hosts=tuple(HostOccupancy(host=h, workloads=(), used_slots=0, free_slots=1) for h in hosts),
        executor="docker",
    )
    monkeypatch.setattr("sparkrun.api._status.status", idle)
    monkeypatch.setattr("sparkrun.api.status", idle)
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    monkeypatch.setattr(
        "sparkrun.core.hardware_probe.probe_hosts",
        lambda hosts, **kwargs: {
            host: HostHardware(accelerators=[AcceleratorSpec("nvidia", "gb10", capabilities=frozenset({"cuda"}))], source="detected")
            for host in hosts
        },
    )
    # Keep the actual diagnostics writer/timeline code, but supply the remote
    # probe result instead of paying SSH's connect timeout on the fake host.
    from sparkrun.orchestration.ssh import RemoteResult

    monkeypatch.setattr(
        "sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel",
        lambda hosts, **kw: [RemoteResult(h, 0, "HOSTNAME=test-host\n", "") for h in hosts],
    )
    return config_root


def _fake_run_result(rc: int = 0):
    """A RunResult whose launch actually populated a timeline."""
    timeline = Timeline()
    with timeline.span("launch"):
        with timeline.span("launch.distribute"):
            pass

    launch = mock.MagicMock()
    launch.rc = rc
    launch.cluster_id = "sparkrun_0123456789abcdef_aabbccddeeff"
    launch.host_list = [_HOST]
    launch.is_solo = True
    launch.serve_port = 30000
    launch.serve_command = "sglang serve"
    launch.container_image = "scitrera/dgx-spark-sglang:latest"
    launch.runtime_info = {}
    launch.hook_launch_context = None
    launch.runtime_cache = None
    launch.effective_cache_dir = "/cache"
    launch.timeline = timeline

    rr = mock.MagicMock()
    rr.rc = rc
    rr.cluster_id = launch.cluster_id
    rr.launch_result = launch
    rr.timeline = timeline
    rr.already_running = False
    return rr


@pytest.fixture
def launched(monkeypatch):
    """Stub the launch itself; these tests are about what happens after it."""
    result = _fake_run_result()

    def launch(options, sctx=None, plan=None):
        from sparkrun.core.bootstrap import get_runtime

        result.launch_result.runtime = get_runtime(options.recipe.runtime, sctx.variables)
        result.launch_result.recipe = options.recipe
        result.launch_result.config = sctx.config
        result.launch_result.timeline = sctx.timing
        result.launch_result.startup_observation = {}
        return result

    monkeypatch.setattr("sparkrun.api.run", launch)
    return result


@pytest.fixture
def follow_marker(monkeypatch):
    """A quiet log stream that exits only when its owner cancels it."""
    from sparkrun.core.log_source import LogLine

    def read(*args, cancel, **kwargs):
        yield LogLine(host=_HOST, container="test_solo", text="<<<LOG STREAM>>>")
        assert cancel.wait(5), "log reader was not cancelled"

    monkeypatch.setattr("sparkrun.orchestration.logs.read_log_sources", read)


def _invoke(runner, *extra):
    return runner.invoke(main, ["run", _RECIPE_NAME, "--cluster", "wopr", "--solo", *extra])


def _table_len(after_header: list[str]) -> int:
    """Number of indented rows following the timings header."""
    n = 0
    for line in after_header:
        if not line.startswith("  "):
            break
        n += 1
    return n


def _all_output(result) -> str:
    """Click's ``output`` already interleaves stderr, which is where the
    readiness line goes."""
    return result.output


def test_timings_print_by_default_and_after_the_log_stream(run_env, launched, follow_marker, monkeypatch):
    """On by default, and rendered only once nothing else owns the terminal.

    A multi-line table written while ``docker logs -f`` is streaming would
    be interleaved mid-render, which is why the live half of the report is
    a single injected line and the tree waits for the stream to stop.
    """
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **k: _ready())

    result = _invoke(CliRunner())
    out = _all_output(result)

    assert "Launch timings" in out
    assert out.index("<<<LOG STREAM>>>") < out.index("Launch timings"), "the table was printed before the stream ended"


def test_readiness_spans_reach_the_printed_table(run_env, launched, follow_marker, monkeypatch):
    """The whole point of the change, end to end.

    Before this, ``serve.port_open`` / ``serve.health_ok`` were recorded
    only for recipes with post hooks — every other run's table stopped at
    "containers started" and the minutes that actually matter went
    unmeasured.
    """

    def _wait(result, *, timeline=None, parent=None, **kwargs):
        # Stands in for the real waiter, which records onto whatever
        # timeline it is handed, under whatever parent it is given.
        timeline.end(timeline.begin("serve.port_open", parent=parent), status="ok")
        timeline.end(timeline.begin("serve.health_ok", parent=parent), status="ok")
        return _ready()

    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", _wait)

    out = _all_output(_invoke(CliRunner()))

    assert "serve.port_open" in out
    assert "serve.health_ok" in out


def test_the_table_accounts_for_the_whole_total(run_env, launched, follow_marker, monkeypatch):
    """Every root-level row summed reaches the reported total.

    The serving row is what closes the gap; before it, the rows stopped at
    readiness while the total ran to whenever the user pressed Ctrl-C.
    """
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **k: _ready())

    out = _all_output(_invoke(CliRunner()))

    assert "serving" in out

    # Scoped to the table: the pre-launch banner is also two-space indented.
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Launch timings"))
    table = lines[start : start + 1 + _table_len(lines[start + 1 :])]
    total = float(table[0].split("total ")[1].rstrip("s):"))

    # Root-level rows are indented exactly two spaces.
    rows = [ln for ln in table[1:] if ln.startswith("  ") and not ln.startswith("    ")]
    assert rows, "no root-level rows parsed"
    summed = sum(float(ln.rsplit(" ", 1)[-1].rstrip("s")) for ln in rows)
    assert summed == pytest.approx(total, abs=0.5), "root rows do not account for the total"


def test_post_hook_recipes_also_account_for_their_serving_time(run_env, launched, follow_marker, monkeypatch):
    """Hook launches use the same watcher and serving span."""
    import sparkrun.core.recipe

    hooked = dict(_RECIPE_DATA, post_commands=["echo hi"])
    recipe_file = run_env.parent / ("%s.yaml" % _RECIPE_NAME)
    recipe_file.write_text(yaml.safe_dump(hooked))
    original = sparkrun.core.recipe.discover_cwd_recipes
    monkeypatch.setattr(
        sparkrun.core.recipe,
        "discover_cwd_recipes",
        lambda directory=None: [recipe_file] + original(directory),
    )
    monkeypatch.setattr("sparkrun.core.launcher.post_launch_lifecycle", lambda *a, **k: None)

    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **k: _ready())
    out = _all_output(_invoke(CliRunner()))

    assert "serving" in out


def test_no_timings_suppresses_the_table_but_keeps_the_readiness_line(run_env, launched, follow_marker, monkeypatch):
    """The two are separable on purpose.

    "The endpoint is up now" is worth having while logs scroll whether or
    not you want a breakdown afterwards.
    """
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **k: _ready())

    result = _invoke(CliRunner(), "--no-timings")
    out = _all_output(result)

    assert "Launch timings" not in out
    assert "Endpoint ready at" in out


def test_ready_line_is_injected_while_the_stream_is_live(run_env, launched, monkeypatch):
    from sparkrun.core.log_source import LogLine
    import sparkrun.cli._run as run_mod

    announced = threading.Event()
    streaming = threading.Event()

    def wait(*args, **kwargs):
        assert streaming.wait(5)
        return _ready()

    def read(*args, cancel, **kwargs):
        streaming.set()
        assert announced.wait(5), "ready was not announced while following"
        yield LogLine(host=_HOST, container="test_solo", text="<<<LOG STREAM>>>")
        cancel.wait(5)

    original_echo = run_mod._echo_endpoint_ready

    def echo(readiness):
        original_echo(readiness)
        announced.set()

    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", wait)
    monkeypatch.setattr("sparkrun.orchestration.logs.read_log_sources", read)
    monkeypatch.setattr(run_mod, "_echo_endpoint_ready", echo)
    result = _invoke(CliRunner())
    assert result.exit_code == 0, result.output
    assert "Endpoint ready at http://10.0.0.1:8000/v1" in result.output
    assert "<<<LOG STREAM>>>" in result.output


def test_post_hook_recipe_shares_one_readiness_result(run_env, launched, follow_marker, monkeypatch):
    recipe_file = run_env.parent / "recipes" / (_RECIPE_NAME + ".yaml")
    recipe_file.write_text(yaml.safe_dump(dict(_RECIPE_DATA, post_commands=["echo hi"])))
    ready = _ready()
    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=ready) as wait:
        with mock.patch("sparkrun.core.launcher.post_launch_lifecycle") as hooks:
            result = _invoke(CliRunner())
    assert result.exit_code == 0, result.output
    wait.assert_called_once()
    hooks.assert_called_once()
    assert hooks.call_args.kwargs["readiness"] is ready
    assert "<<<LOG STREAM>>>" in result.output


@pytest.mark.parametrize("reason", ["port", "health", "inference"])
def test_readiness_failure_fails_the_run(run_env, launched, follow_marker, monkeypatch, reason):
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **k: _ready(ready=False, reason=reason))
    result = _invoke(CliRunner())
    assert result.exit_code == 1, result.output
    assert "did not become ready" in result.output


def test_no_timings_still_records_the_timeline_for_diagnostics(run_env, launched, follow_marker, monkeypatch, tmp_path):
    """``--collect-diagnostics`` owns the timeline independently of the table.

    One ``or`` in the CLI keeps this true; flipping it would silently empty
    the diagnostics record for anyone who also passes ``--no-timings``.
    """
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **k: _ready())
    diag_path = tmp_path / "diag.ndjson"

    result = _invoke(CliRunner(), "--no-timings", "--collect-diagnostics", str(diag_path))
    out = _all_output(result)

    assert "Launch timings" not in out
    records = [line for line in diag_path.read_text().splitlines() if '"run_timeline"' in line]
    assert len(records) == 1, "the diagnostics timeline went missing with --no-timings"


def test_failed_log_stream_does_not_cancel_readiness(run_env, launched, monkeypatch):
    log_ended = threading.Event()

    def wait(*args, **kwargs):
        assert log_ended.wait(5)
        return _ready()

    def read(*args, **kwargs):
        log_ended.set()
        raise RuntimeError("stream died")
        yield  # make a generator

    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", wait)
    monkeypatch.setattr("sparkrun.orchestration.logs.read_log_sources", read)
    result = _invoke(CliRunner())
    assert result.exit_code == 0, result.output
    assert "Endpoint ready" in result.output


def test_no_follow_waits_without_log_readers(run_env, launched, monkeypatch):
    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=_ready()) as wait:
        with mock.patch("sparkrun.cli._run_lifecycle.LogFollower") as reader:
            result = _invoke(CliRunner(), "--no-follow")
    assert result.exit_code == 0, result.output
    wait.assert_called_once()
    reader.assert_not_called()
    assert "Endpoint ready" in result.output


def test_interrupted_watch_is_not_reported_as_a_broken_endpoint(run_env, launched, follow_marker, monkeypatch):
    """Stopping the stream says nothing about the workload."""
    monkeypatch.setattr(
        "sparkrun.core.launcher.wait_for_serve_ready",
        lambda *a, **k: _ready(ready=False, reason="cancelled"),
    )

    result = _invoke(CliRunner())
    out = _all_output(result)

    assert result.exit_code == 130, out
    assert "did not become ready" not in out


@pytest.mark.parametrize("extra", [[], ["--no-follow"]])
def test_no_ready_wait_does_not_probe_or_attach(run_env, launched, extra):
    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready") as wait:
        with mock.patch("sparkrun.cli._run_lifecycle.LogFollower") as logs:
            result = _invoke(CliRunner(), "--no-ready-wait", *extra)
    assert result.exit_code == 0, result.output
    assert "readiness not checked" in result.output
    wait.assert_not_called()
    logs.assert_not_called()


@pytest.mark.parametrize(
    "flags",
    [
        ["--follow", "--no-follow"],
        ["--follow", "--no-ready-wait"],
        ["--foreground", "--no-ready-wait"],
    ],
)
def test_incompatible_wait_options_fail_before_launch(run_env, flags):
    with mock.patch("sparkrun.api.run") as launch:
        result = _invoke(CliRunner(), *flags)
    assert result.exit_code == 2, result.output
    launch.assert_not_called()


def test_no_ready_wait_rejects_hooks_before_launch(run_env):
    recipe = run_env.parent / "recipes" / (_RECIPE_NAME + ".yaml")
    recipe.write_text(yaml.safe_dump(dict(_RECIPE_DATA, post_commands=["echo hi"])))
    with mock.patch("sparkrun.api.run") as launch:
        result = _invoke(CliRunner(), "--no-ready-wait")
    assert result.exit_code == 2, result.output
    assert "post-launch hooks" in result.output
    launch.assert_not_called()


def test_readiness_exception_is_a_failure(run_env, launched):
    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", side_effect=RuntimeError("probe failed")):
        result = _invoke(CliRunner(), "--no-follow")
    assert result.exit_code == 1, result.output
    assert "readiness could not be confirmed: probe failed" in result.output


def test_explicit_follow_survives_readiness_then_ctrl_c_detaches(run_env, launched, monkeypatch):
    from sparkrun.orchestration.logs import LogFollower

    ready = threading.Event()
    cancelled = threading.Event()
    from sparkrun.core.log_source import LogLine

    def read(*a, cancel, **kw):
        yield LogLine(host=_HOST, container="test_solo", text="startup")
        cancel.wait(5)
        cancelled.set()

    # Raise on the main thread only once the ready announcement has happened.
    class InterruptedEvent:
        def is_set(self):
            return False

        def set(self):
            pass

        def wait(self, timeout):
            assert ready.is_set(), "explicit following must reach readiness first"
            raise KeyboardInterrupt

    original_start = LogFollower.start

    def start(self):
        self.done = InterruptedEvent()
        return original_start(self)

    monkeypatch.setattr("sparkrun.orchestration.logs.read_log_sources", read)
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **kw: _ready())
    monkeypatch.setattr("sparkrun.cli._run._echo_endpoint_ready", lambda outcome: ready.set())
    monkeypatch.setattr(LogFollower, "start", start)
    result = _invoke(CliRunner(), "--follow")
    assert result.exit_code == 130, result.output
    assert ready.is_set() and cancelled.is_set()
    assert "Continuing log following" in result.output
    assert "left running" in result.output


def test_ctrl_c_during_readiness_cancels_both_observers(run_env, launched, monkeypatch):
    from sparkrun.cli import _run_lifecycle
    from sparkrun.core.log_source import LogLine

    stopped = threading.Event()
    log_stopped = threading.Event()

    def wait(*a, cancel, **kw):
        cancel.wait(5)
        stopped.set()
        return _ready(ready=False, reason="cancelled")

    def read(*a, cancel, **kw):
        yield LogLine(host=_HOST, container="test_solo", text="startup")
        cancel.wait(5)
        log_stopped.set()

    original_start = _run_lifecycle.ReadinessWatcher.start

    def start(self):
        original_start(self)
        raise KeyboardInterrupt

    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", wait)
    monkeypatch.setattr("sparkrun.orchestration.logs.read_log_sources", read)
    monkeypatch.setattr(_run_lifecycle.ReadinessWatcher, "start", start)
    result = _invoke(CliRunner())
    assert result.exit_code == 130, result.output
    assert stopped.is_set() and log_stopped.is_set()
    assert "did not become ready" not in result.output


@pytest.mark.parametrize("stop_after_post", [False, True])
def test_post_hook_outcome_controls_exit(run_env, launched, monkeypatch, stop_after_post):
    recipe = run_env.parent / "recipes" / (_RECIPE_NAME + ".yaml")
    recipe.write_text(yaml.safe_dump(dict(_RECIPE_DATA, post_commands=["echo hi"], stop_after_post=stop_after_post)))
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **kw: _ready())
    with mock.patch("sparkrun.orchestration.hooks.run_post_commands", side_effect=None if stop_after_post else RuntimeError("hook failed")):
        from sparkrun.runtimes.base import RuntimePlugin

        with mock.patch.object(RuntimePlugin, "stop") as stop:
            result = _invoke(CliRunner(), "--no-follow")
    assert result.exit_code == (0 if stop_after_post else 1), result.output
    assert stop.call_count == int(stop_after_post)
    assert ("Stopping workload" if stop_after_post else "hook failed") in result.output


def test_unsupported_endpoint_rejected_before_launch(run_env, monkeypatch):
    from sparkrun.orchestration.executors.docker import DockerExecutor

    monkeypatch.setattr(DockerExecutor, "supports_host_endpoint", False)
    with mock.patch("sparkrun.api.run") as launch:
        result = _invoke(CliRunner())
    assert result.exit_code == 2, result.output
    assert "--no-ready-wait" in result.output
    launch.assert_not_called()


def test_unsupported_endpoint_allows_submission_only(run_env, launched, monkeypatch):
    from sparkrun.orchestration.executors.docker import DockerExecutor

    monkeypatch.setattr(DockerExecutor, "supports_host_endpoint", False)
    result = _invoke(CliRunner(), "--no-ready-wait")
    assert result.exit_code == 0, result.output
    assert "readiness not checked" in result.output


@pytest.mark.parametrize("partial", [False, True])
def test_ensure_uses_recorded_head_and_port(run_env, monkeypatch, partial):
    from sparkrun.api._intent import IntentMatch

    hosts = [_HOST, "10.0.4.31"]
    match = IntentMatch(
        intent_id="test", cluster_id="sparkrun_0123456789abcdef_aabbccddeeff", hosts=tuple(hosts[1:] if partial else reversed(hosts))
    )
    monkeypatch.setattr("sparkrun.api.find_running_intent", lambda *a, **kw: match)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: {"hosts": hosts, "port": 32100})

    def wait(result, **kw):
        assert result.host_list == hosts
        assert result.serve_port == 32100
        assert result.is_solo is False
        return _ready()

    with mock.patch("sparkrun.core.launcher.wait_for_serve_ready", side_effect=wait) as probe:
        with mock.patch("sparkrun.api.run") as launch:
            result = _invoke(CliRunner(), "--ensure", "--no-follow")
    assert result.exit_code == (1 if partial else 0), result.output
    assert probe.call_count == int(not partial)
    launch.assert_not_called()


def test_local_port_wait_uses_native_liveness(monkeypatch):
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.ssh import RemoteResult

    executor = LocalExecutor()
    commands = []

    def run(host, command, **kwargs):
        commands.append(command)
        return RemoteResult(host=host, returncode=1 if len(commands) == 2 else 0, stdout="", stderr="")

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", run)
    assert wait_for_port("localhost", 8000, container_name="test_solo", executor=executor, retry_interval=0)
    assert len(commands) == 4
    assert commands[0] == commands[2] == executor.status_cmd("test_solo")
    assert all("docker" not in command for command in commands)


def test_readiness_failure_marks_diagnostics_failed(run_env, launched, monkeypatch, tmp_path):
    import json

    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda *a, **kw: _ready(ready=False, reason="health"))
    monkeypatch.setattr("sparkrun.diagnostics.RunDiagnosticsCollector.capture_workload_logs", lambda *a: None)
    output = tmp_path / "failed.ndjson"
    result = _invoke(CliRunner(), "--no-follow", "--collect-diagnostics", str(output))
    assert result.exit_code == 1, result.output
    records = [json.loads(line) for line in output.read_text().splitlines()]
    summary = next(item for item in records if item["_type"] == "run_summary")
    assert summary["success"] is False
    assert any(item["_type"] == "run_error" for item in records)


def test_cancel_during_liveness_does_not_start_another_probe(monkeypatch):
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.ssh import RemoteResult

    cancel = threading.Event()

    def status(host, command, **kw):
        cancel.set()
        return RemoteResult(host=host, returncode=0, stdout="", stderr="")

    with mock.patch("sparkrun.orchestration.primitives.run_command_on_host", side_effect=status) as commands:
        assert not wait_for_port("localhost", 8000, executor=LocalExecutor(), container_name="test_solo", cancel=cancel)
    commands.assert_called_once()
