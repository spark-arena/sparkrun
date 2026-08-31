"""Tests for the launch-stage span timeline (:mod:`sparkrun.core.timing`).

Four properties carry the design and are the ones that break silently:

- the tree nests by **parent id**, not by name (fan-out spans repeat names);
- a phase skipped while another is open must not nest inside it;
- an unclosed span is reported as ``open``, not dropped — a launch that
  raised is when the breakdown is worth most;
- durations are never combined across clocks, and the control clock cannot
  be written to by assertion.
"""

from __future__ import annotations

import time

import pytest

from sparkrun.core.progress import LaunchProgress, phase_span_name
from sparkrun.core.timing import (
    CLOCK_CONTROL,
    STATUS_ERROR,
    STATUS_OPEN,
    STATUS_SKIPPED,
    Timeline,
    remote_clock,
    timed,
)


def _by_name(export: dict) -> dict[str, dict]:
    return {s["name"]: s for s in export["spans"]}


# ---------------------------------------------------------------------------
# Timeline primitives
# ---------------------------------------------------------------------------


def test_span_records_duration_and_nesting():
    tl = Timeline()
    with tl.span("outer") as outer:
        with tl.span("inner"):
            pass
    spans = _by_name(tl.export())
    assert spans["inner"]["parent"] == outer
    assert spans["outer"]["parent"] is None
    assert spans["outer"]["duration_s"] >= spans["inner"]["duration_s"]


def test_span_records_error_status_and_reraises():
    tl = Timeline()
    with pytest.raises(ValueError):
        with tl.span("boom"):
            raise ValueError("nope")
    span = _by_name(tl.export())["boom"]
    assert span["status"] == STATUS_ERROR
    assert span["attrs"]["error"] == "ValueError"


def test_ending_a_parent_closes_open_children_with_its_status():
    """Runtimes open steps and never close them; the phase boundary does.

    The status has to propagate: if the phase failed, the step that was
    running is where it failed, and calling it ``ok`` points the reader at
    the wrong stage.
    """
    tl = Timeline()
    parent = tl.begin("phase")
    tl.begin("step")
    tl.end(parent, status=STATUS_ERROR)

    spans = _by_name(tl.export())
    assert spans["step"]["status"] == STATUS_ERROR
    assert spans["step"]["parent"] == parent
    assert not tl._stack  # fully unwound


def test_export_reports_unclosed_spans_as_open_without_closing_them():
    tl = Timeline()
    tl.begin("still-running")

    first = _by_name(tl.export())["still-running"]
    assert first["status"] == STATUS_OPEN
    assert first["duration_s"] >= 0.0

    # Non-destructive: the span is still open and can still be closed.
    tl.end(status="ok")
    assert _by_name(tl.export())["still-running"]["status"] == "ok"


def test_skipped_is_distinct_from_a_zero_duration_success():
    """ "Ran and found nothing to do" and "did not run" are different facts."""
    tl = Timeline()
    tl.skipped("launch.build", reason="no builder")
    with tl.span("launch.tuning"):
        pass

    spans = _by_name(tl.export())
    assert spans["launch.build"]["status"] == STATUS_SKIPPED
    assert spans["launch.build"]["attrs"]["reason"] == "no builder"
    assert spans["launch.tuning"]["status"] == "ok"


def test_repeated_names_stay_distinct_spans():
    """Fan-out spans share a name, so the export must not be name-keyed."""
    tl = Timeline()
    for host in ("h1", "h2", "h3"):
        with tl.span("launch.distribute.model", host=host):
            pass

    spans = [s for s in tl.export()["spans"] if s["name"] == "launch.distribute.model"]
    assert len(spans) == 3
    assert {s["attrs"]["host"] for s in spans} == {"h1", "h2", "h3"}
    assert len({s["id"] for s in spans}) == 3


def test_timed_helper_tolerates_no_timeline():
    with timed(None, "whatever") as span_id:
        assert span_id is None


def test_total_and_find():
    tl = Timeline()
    for _ in range(3):
        with tl.span("rep"):
            time.sleep(0.001)
    assert tl.total("rep") >= 0.003
    assert tl.find("rep") is not None
    assert tl.find("absent") is None


# ---------------------------------------------------------------------------
# LaunchProgress -> Timeline
# ---------------------------------------------------------------------------


def test_progress_without_timeline_is_inert():
    """``timeline=None`` must stay byte-identical to the pre-timing behaviour."""
    p = LaunchProgress()
    p.phase(1)
    p.step("Doing a thing")
    p.phase_end()
    p.phase_skip(2, "no builder")
    assert p.timeline is None


def test_progress_phases_and_steps_become_spans():
    tl = Timeline()
    p = LaunchProgress(timeline=tl)
    root = tl.begin("launch")
    p.set_root_span(root)

    p.phase(1)
    p.phase_end()
    p.phase(5, "Launching vLLM runtime")
    p.begin_runtime_steps(2)
    p.step("Cleaning up existing containers")
    p.step("Executing serve command on head")
    p.phase_end()
    tl.end(root)

    spans = _by_name(tl.export())
    assert spans["launch.prepare"]["parent"] == root
    assert spans["launch.runtime"]["parent"] == root
    # The phase label is display text; the span name is the stable slug.
    assert spans["launch.runtime"]["attrs"]["label"] == "Launching vLLM runtime"

    step = spans["launch.runtime.executing_serve_command_on_head"]
    assert step["parent"] == spans["launch.runtime"]["id"]
    assert step["attrs"]["step"] == 2
    # Runtimes never call step_done(); the phase boundary closes the last step.
    assert step["status"] == "ok"


def test_skipped_phase_does_not_nest_under_a_still_open_phase():
    """Phases parent to the root explicitly, not via the open-span stack.

    A caller that skips a phase without closing the previous one would
    otherwise produce a phase nested inside its predecessor — invisible at
    the call site and rendered wrong by every consumer.
    """
    tl = Timeline()
    p = LaunchProgress(timeline=tl)
    root = tl.begin("launch")
    p.set_root_span(root)

    p.phase(1)  # deliberately left open
    p.phase_skip(3, "delegating runtime")
    p.phase_end()
    tl.end(root)

    spans = _by_name(tl.export())
    assert spans[phase_span_name(3)]["parent"] == root
    assert spans[phase_span_name(3)]["status"] == STATUS_SKIPPED


def test_step_done_closes_the_step_early():
    tl = Timeline()
    p = LaunchProgress(timeline=tl)
    p.phase(5)
    t0 = p.step("Launching containers")
    p.step_done(t0)
    assert p._step_span is None
    p.phase_end()

    spans = _by_name(tl.export())
    assert spans["launch.runtime.launching_containers"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------
#
# Control-clock spans are measured here and may be combined arithmetically.
# Spans recovered from a remote engine's own log timestamps are not: with NTP
# skew, "container start -> weights loaded" across the two can come out
# negative.  The discriminator exists so that stays impossible by construction
# rather than by callers' discipline.


def test_measured_spans_are_on_the_control_clock():
    tl = Timeline()
    with tl.span("measured"):
        pass
    span = tl.find("measured")
    assert span.clock == CLOCK_CONTROL
    assert span.is_control is True
    # Omitted from the export: it is the documented default, and emitting it
    # on every span would bloat the benchmark artifact to say nothing.
    assert "clock" not in _by_name(tl.export())["measured"]


def test_add_span_refuses_the_control_clock():
    """A control-clock span must be *measured*, never asserted.

    The value of the discriminator is that ``total()`` and friends can trust
    the control clock; letting a parsed duration in through this door would
    put an unverifiable number in the set that gets summed.
    """
    tl = Timeline()
    with pytest.raises(ValueError, match="cannot record on the control clock"):
        tl.add_span("loading_weights", clock=CLOCK_CONTROL, duration_s=12.0)


def test_foreign_span_is_recorded_with_its_clock_and_unconverted_wall_start():
    tl = Timeline()
    wall = tl.wall_origin + 30.0
    tl.add_span(
        "engine.loading_weights",
        clock=remote_clock("h1"),
        duration_s=42.0,
        wall_start=wall,
        stage="weights",
    )

    span = tl.find("engine.loading_weights")
    assert span.clock == "remote:h1"
    assert span.is_control is False
    # Placement is derived through our origin — an estimate carrying the skew,
    # which is why the original is kept alongside it.
    assert span.t_start == pytest.approx(30.0)
    assert span.wall_start == pytest.approx(wall)

    exported = _by_name(tl.export())["engine.loading_weights"]
    assert exported["clock"] == "remote:h1"
    assert exported["wall_start"] == pytest.approx(wall, abs=0.01)
    assert exported["attrs"]["stage"] == "weights"


def test_total_excludes_foreign_clocks_by_default():
    """``total`` is a derived figure, so it may not mix clocks."""
    tl = Timeline()
    with tl.span("stage"):
        time.sleep(0.005)
    tl.add_span("stage", clock=remote_clock("h1"), duration_s=1000.0)

    assert tl.total("stage") < 1.0  # the remote 1000s is not summed in
    assert tl.total("stage", clock=remote_clock("h1")) == 1000.0


def test_export_announces_clocks_only_when_mixed():
    tl = Timeline()
    with tl.span("measured"):
        pass
    assert "clocks" not in tl.export()  # single clock: a consumer may sum freely

    tl.add_span("remote-thing", clock=remote_clock("h2"), duration_s=1.0)
    assert tl.export()["clocks"] == [CLOCK_CONTROL, "remote:h2"]
    assert tl.clocks == {CLOCK_CONTROL, "remote:h2"}


def test_foreign_span_nests_under_the_open_control_span():
    """A probe's spans hang off whatever stage was running when it ran."""
    tl = Timeline()
    parent = tl.begin("serve.health_ok")
    child = tl.add_span("engine.capturing_cudagraphs", clock=remote_clock("h1"), duration_s=8.0)
    tl.end(parent)

    spans = _by_name(tl.export())
    assert spans["engine.capturing_cudagraphs"]["parent"] == parent
    # Closing the parent must not touch it — it was never on the open stack.
    assert spans["engine.capturing_cudagraphs"]["id"] == child
    assert spans["engine.capturing_cudagraphs"]["status"] == "ok"


def test_formatter_names_the_clock_so_the_tree_does_not_read_as_one_total():
    from sparkrun.utils.cli_formatters import format_launch_timings

    tl = Timeline()
    root = tl.begin("launch")
    tl.add_span("engine.loading_weights", clock=remote_clock("h1"), duration_s=42.0)
    tl.end(root)

    rendered = format_launch_timings(tl.export())
    assert "[remote:h1]" in rendered
    # The measured span carries no annotation.
    launch_line = next(ln for ln in rendered.splitlines() if "launch" in ln and "engine" not in ln)
    assert "[" not in launch_line


def test_formatter_accepts_operation_specific_title():
    from sparkrun.utils.cli_formatters import format_launch_timings

    tl = Timeline()
    with tl.span("plugin.capture"):
        pass

    rendered = format_launch_timings(tl.export(), title="Capture timings")
    assert rendered.startswith("Capture timings")
