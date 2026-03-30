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
import time
from enum import IntEnum

# ---------------------------------------------------------------------------
# Custom log levels
# ---------------------------------------------------------------------------

PROGRESS = 25
VERBOSE = 15

logging.addLevelName(PROGRESS, "PROGRESS")
logging.addLevelName(VERBOSE, "VERBOSE")

# ---------------------------------------------------------------------------
# Verbosity enum
# ---------------------------------------------------------------------------

PHASE_LABELS: dict[int, str] = {
    1: "Preparing",
    2: "Building image",
    3: "Distributing resources",
    4: "Syncing tuning configs",
    5: "Launching runtime",
    6: "Post-launch hooks",
}

TOTAL_PHASES = len(PHASE_LABELS)


class Verbosity(IntEnum):
    """CLI verbosity tiers."""

    DEFAULT = 0  # PROGRESS level (25)
    DETAIL = 1   # INFO level (20)  — ``-v``
    VERBOSE = 2  # VERBOSE level (15) — ``-vv``
    DEBUG = 3    # DEBUG level (10)  — ``-vvv``


# ---------------------------------------------------------------------------
# LaunchProgress
# ---------------------------------------------------------------------------


class LaunchProgress:
    """Structured progress tracker for the launch pipeline.

    All output goes through the ``sparkrun.progress`` logger so the
    CLI layer controls visibility by setting the root logger's level.

    Parameters
    ----------
    verbosity:
        Current verbosity tier (from CLI ``-v`` count).
    """

    def __init__(self, verbosity: Verbosity = Verbosity.DEFAULT) -> None:
        self.verbosity = verbosity
        self._log = logging.getLogger("sparkrun.progress")
        self._current_phase: int | None = None
        self._phase_t0: float | None = None
        self._step_total: int = 0
        self._step_current: int = 0

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
        self._log.log(PROGRESS, "[%d/%d] %s", num, TOTAL_PHASES, effective_label)

    def phase_end(self, elapsed: float | None = None) -> None:
        """Close the current phase with a done line."""
        if self._phase_t0 is not None:
            dt = elapsed if elapsed is not None else (time.monotonic() - self._phase_t0)
            self._log.log(PROGRESS, "  done (%.1fs)", dt)
        self._current_phase = None
        self._phase_t0 = None

    def phase_skip(self, num: int, reason: str = "") -> None:
        """Mark a phase as skipped.

        Always emits a single line so phase numbering stays continuous.
        """
        label = PHASE_LABELS.get(num, "Phase %d" % num)
        suffix = " (%s)" % reason if reason else ""
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
