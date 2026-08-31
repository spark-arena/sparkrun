"""Unified logging & progress system for sparkrun launch pipeline.

Provides :class:`LaunchProgress` for structured phase/step output
using the standard ``logging`` module.  No dependency on Click or
any CLI framework — the CLI layer configures the root logger's level
and formatter; this module only emits log records.

Custom log levels:

- ``PROGRESS`` (25): Phase boundaries and runtime steps.  Always
  visible at the default verbosity.
- ``VERBOSE`` (15): Between INFO and DEBUG.  Adds timestamps and
  logger names at ``-vv``.

Typical verbosity mapping (configured by the CLI layer):

===========  =====  ================================================
Flag         Level  What's visible
===========  =====  ================================================
(default)    25     PROGRESS, WARNING, ERROR
``-v``       20     + INFO detail lines
``-vv``      15     + timestamps / logger names in format
``-vvv``     10     + DEBUG (SSH internals, script content)
===========  =====  ================================================
"""

from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import contextmanager
from enum import IntEnum
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from sparkrun.core.timing import Timeline

# ---------------------------------------------------------------------------
# Custom log levels
# ---------------------------------------------------------------------------

PROGRESS = 25
VERBOSE = 15

logging.addLevelName(PROGRESS, "PROGRESS")
logging.addLevelName(VERBOSE, "VERBOSE")

#: How often :func:`progress_heartbeat` reports that work is still running.
DEFAULT_HEARTBEAT_SECONDS = 30.0


@contextmanager
def progress_heartbeat(logger: logging.Logger, label: str, interval: float = DEFAULT_HEARTBEAT_SECONDS) -> Iterator[None]:
    """Keep a long operation visibly alive at the standard progress level.

    For work that emits nothing while it runs — a multi-minute image build, a
    remote capture — silence is indistinguishable from a hang.  This is the
    display-only counterpart to the session guard: it never affects the work,
    it only reports that the work is still there.
    """
    stopped = threading.Event()
    started = time.monotonic()

    def report() -> None:
        while not stopped.wait(interval):
            logger.log(PROGRESS, "%s — still running (%.0fs)", label, time.monotonic() - started)

    thread = threading.Thread(target=report, name="sparkrun-progress-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, interval))


# ---------------------------------------------------------------------------
# Verbosity enum
# ---------------------------------------------------------------------------

PHASE_LABELS: dict[int, str] = {
    1: "Preparing",
    2: "Building",
    3: "Distributing resources",
    4: "Syncing tuning configs",
    5: "Launching runtime",
    6: "Post-launch hooks",
}

TOTAL_PHASES = len(PHASE_LABELS)

#: Stable span names for the phases.  Deliberately *not* derived from
#: ``PHASE_LABELS`` — those are display strings and rewording one must not
#: rename a metric that benchmark artifacts have already recorded.
PHASE_SLUGS: dict[int, str] = {
    1: "prepare",
    2: "build",
    3: "distribute",
    4: "tuning",
    5: "runtime",
    6: "post_launch",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_") or "step"


def phase_span_name(num: int) -> str:
    """Span name for phase ``num`` (``launch.distribute``, …)."""
    return "launch.%s" % PHASE_SLUGS.get(num, "phase_%d" % num)


class Verbosity(IntEnum):
    """CLI verbosity tiers."""

    DEFAULT = 0  # PROGRESS level (25)
    DETAIL = 1  # INFO level (20)  — ``-v``
    VERBOSE = 2  # VERBOSE level (15) — ``-vv``
    DEBUG = 3  # DEBUG level (10)  — ``-vvv``


# ---------------------------------------------------------------------------
# LaunchProgress
# ---------------------------------------------------------------------------


class LaunchProgress:
    """Structured progress tracker for the launch pipeline.

    All output goes through the ``sparkrun.progress`` logger so the
    CLI layer controls visibility by setting the root logger's level.

    When a :class:`~sparkrun.core.timing.Timeline` is attached, the same
    phase/step brackets that produce the log lines also record spans — the
    elapsed times were already being computed here and discarded.  This is
    why instrumenting the launch pipeline needed almost no new call sites.

    Parameters
    ----------
    verbosity:
        Current verbosity tier (from CLI ``-v`` count).
    timeline:
        Optional span collector.  ``None`` disables timing collection.
    """

    def __init__(self, verbosity: Verbosity = Verbosity.DEFAULT, timeline: "Timeline | None" = None) -> None:
        self.verbosity = verbosity
        self.timeline = timeline
        self._log = logging.getLogger("sparkrun.progress")
        self._current_phase: int | None = None
        self._phase_t0: float | None = None
        self._step_total: int = 0
        self._step_current: int = 0
        self._phase_span: int | None = None
        self._step_span: int | None = None
        self._root_span: int | None = None

    # -- Timeline root -------------------------------------------------------

    def set_root_span(self, span_id: int | None) -> None:
        """Parent every phase span under ``span_id``.

        Phases are parented *explicitly* rather than by the timeline's
        open-span stack: a phase skipped while the previous phase is still
        open would otherwise nest inside it, which no caller could see and
        every consumer would render wrong.
        """
        self._root_span = span_id

    # -- Phase API ----------------------------------------------------------

    def phase(self, num: int, label: str | None = None) -> None:
        """Start a numbered phase.

        Emits ``[N/6] Label`` at PROGRESS level (always visible).
        """
        if self._current_phase is not None:
            self._auto_close_phase()
        self._current_phase = num
        self._phase_t0 = time.monotonic()
        self._step_total = 0
        self._step_current = 0
        effective_label = label or PHASE_LABELS.get(num, "Phase %d" % num)
        if self.timeline is not None:
            self._phase_span = self.timeline.begin(phase_span_name(num), parent=self._root_span, phase=num, label=effective_label)
            self._step_span = None
        self._log.log(PROGRESS, "[%d/%d] %s", num, TOTAL_PHASES, effective_label)

    def phase_end(self, elapsed: float | None = None, *, status: str = "ok") -> None:
        """Close the current phase with a done line.

        Closing the phase span also closes whatever step was still open —
        runtimes call :meth:`step` without ever calling :meth:`step_done`,
        so the phase boundary is what ends the last step.
        """
        if self._phase_t0 is not None:
            dt = elapsed if elapsed is not None else (time.monotonic() - self._phase_t0)
            self._log.log(PROGRESS, "  done (%.1fs)", dt)
        if self.timeline is not None and self._phase_span is not None:
            self.timeline.end(self._phase_span, status=status)
        self._phase_span = None
        self._step_span = None
        self._current_phase = None
        self._phase_t0 = None

    def phase_skip(self, num: int, reason: str = "") -> None:
        """Mark a phase as skipped.

        Always emits a single line so phase numbering stays continuous.
        """
        label = PHASE_LABELS.get(num, "Phase %d" % num)
        suffix = " (%s)" % reason if reason else ""
        if self.timeline is not None:
            self.timeline.skipped(phase_span_name(num), parent=self._root_span, reason=reason, phase=num, label=label)
        self._log.log(PROGRESS, "[%d/%d] %s — skipped%s", num, TOTAL_PHASES, label, suffix)

    # -- Step API (runtime sub-steps within phase 5) ------------------------

    def begin_runtime_steps(self, total: int) -> None:
        """Declare how many sub-steps the runtime will report."""
        self._step_total = total
        self._step_current = 0

    def step(self, label: str) -> float:
        """Emit a sub-step line, returning the start timestamp.

        Returns ``time.monotonic()`` so callers can pass it to
        :meth:`step_done` for elapsed-time reporting.
        """
        self._step_current += 1
        if self.timeline is not None:
            # A step ends when the next one begins; runtimes never call
            # step_done().  Close the previous one before opening this.
            if self._step_span is not None:
                self.timeline.end(self._step_span)
            self._step_span = self.timeline.begin(
                "%s.%s" % (phase_span_name(self._current_phase or 0), _slug(label)),
                parent=self._phase_span,
                label=label,
                step=self._step_current,
            )
        if self._step_total > 0:
            self._log.log(
                PROGRESS,
                "  Step %d/%d: %s",
                self._step_current,
                self._step_total,
                label,
            )
        else:
            self._log.log(PROGRESS, "  Step %d: %s", self._step_current, label)
        return time.monotonic()

    def step_done(self, t0: float) -> None:
        """Optionally log elapsed time for the most recent step (detail level)."""
        dt = time.monotonic() - t0
        if self.timeline is not None and self._step_span is not None:
            self.timeline.end(self._step_span)
            self._step_span = None
        self.detail("  step done (%.1fs)", dt)

    # -- Tiered output helpers -----------------------------------------------

    def detail(self, msg: str, *args: object) -> None:
        """Log at INFO — visible at ``-v`` and above."""
        self._log.info(msg, *args)

    def verbose(self, msg: str, *args: object) -> None:
        """Log at VERBOSE (15) — visible at ``-vv`` and above."""
        self._log.log(VERBOSE, msg, *args)

    def debug(self, msg: str, *args: object) -> None:
        """Log at DEBUG — visible at ``-vvv``."""
        self._log.debug(msg, *args)

    def warn(self, msg: str, *args: object) -> None:
        """Log at WARNING — always visible."""
        self._log.warning(msg, *args)

    def error(self, msg: str, *args: object) -> None:
        """Log at ERROR — always visible."""
        self._log.error(msg, *args)

    # -- Internal -----------------------------------------------------------

    def _auto_close_phase(self) -> None:
        """Close a phase that wasn't explicitly ended."""
        self.phase_end()
