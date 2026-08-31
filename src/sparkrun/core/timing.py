"""Launch-pipeline timing — a span timeline for the launch path.

The launch pipeline already brackets its own work in three places
(:class:`sparkrun.core.progress.LaunchProgress` phases and steps, and the
readiness wait) and every one of them computed an elapsed time only to log
it and throw it away.  This module keeps the number.

The recorded shape is a **span timeline**, not a ``{stage: seconds}``
mapping, for three reasons:

- Phases nest: phase 5 contains runtime steps, which contain per-host work.
- Fan-out spans repeat by host, so names are not unique keys.
- The consumers (benchmark provenance, ``run --timings``, the diagnostics
  NDJSON) want the tree, and a flattened summary is derivable from it while
  the reverse is not.

A ``Timeline`` is optional everywhere it is threaded.  ``None`` means "not
collecting" and is byte-identical to the behaviour before this module
existed — the same convention ``progress`` and ``backends`` already use on
:func:`sparkrun.core.launcher.launch_inference`.

Clocks
------

Spans measured here are on the **control node's** ``time.monotonic()``
(:data:`CLOCK_CONTROL`).  That is what makes their durations subtractable.

Spans recovered from a remote engine's own log timestamps — vLLM's "loading
weights" / "capturing cudagraphs", say — are on a *different* clock, and are
not merely another source of spans.  With skew between the control node and
a host, "container start → weights loaded" computed across the two can come
out negative; a few seconds of NTP drift is enough.  So :class:`Span` carries
a ``clock`` discriminator and the rule is:

    **Never use a foreign-clock span as a term in a derived total.**

Two things enforce it rather than leaving it to callers' discipline:

- :meth:`Timeline.add_span` — the only way to introduce a foreign-clock
  span — *refuses* ``clock=CLOCK_CONTROL``.  A control-clock span must be
  measured (:meth:`~Timeline.begin` / :meth:`~Timeline.span`), never
  asserted, so a parsed remote duration cannot be laundered into the clock
  whose spans get summed.
- :meth:`Timeline.total` filters by clock and defaults to the control clock.

A foreign-clock span's ``t_start`` is a *placement estimate* (its wall clock
minus this timeline's origin) — good enough to order it in a rendered tree,
not good enough to subtract.  Its ``wall_start`` carries the unconverted
original.

Threading
---------

Span *creation* is lock-protected, so a fan-out worker may record spans.
The open-span stack, however, is a convenience for sequential control flow
on one thread: a span created off the main thread must pass ``parent=``
explicitly rather than relying on the stack — a span id to nest under one,
or :data:`ROOT` to sit at the root.  Inheriting from the stack across
threads is not merely imprecise: :meth:`Timeline.end` closes everything
open *above* its target, so a main-thread ``end()`` would close the
worker's span too, stamping it with the main thread's status and turning
the worker's own ``end()`` into a silent no-op.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = [
    "Span",
    "Timeline",
    "timed",
    "remote_clock",
    "CLOCK_CONTROL",
    "ROOT",
    "STATUS_OK",
    "STATUS_ERROR",
    "STATUS_SKIPPED",
    "STATUS_OPEN",
]

#: The control node's ``time.monotonic()``.  Spans on this clock are the
#: only ones whose durations may be combined arithmetically.
CLOCK_CONTROL = "control"

#: ``parent=ROOT`` — parent this span to the timeline root, explicitly.
#:
#: ``parent=None`` means "inherit from the open-span stack", which is a
#: *sequential-control-flow* convenience and the wrong default for a span
#: opened off the main thread: whatever that thread happens to find on the
#: stack becomes the parent, and a main-thread ``end()`` above it will slice
#: the worker's span out and stamp it with the main thread's status.  A
#: worker that wants the root must be able to say so rather than depend on
#: the stack being empty at that instant.
ROOT = -1


def remote_clock(host: str) -> str:
    """Clock identifier for timestamps originating on *host*.

    Two spans sharing a ``remote:<host>`` clock are comparable with each
    other but not with :data:`CLOCK_CONTROL` spans, and — since nothing
    disciplines the clocks of two different machines — not with spans from
    another host either.
    """
    return "remote:%s" % host


STATUS_OK = "ok"
STATUS_ERROR = "error"
#: A phase that was deliberately not run.  Distinct from a zero-duration
#: ``ok`` span, which means the work ran and found nothing to do — for a
#: benchmark artifact those are different data points.
STATUS_SKIPPED = "skipped"
#: Never closed — the launch raised or was interrupted while it was open.
STATUS_OPEN = "open"


@dataclass(frozen=True)
class Span:
    """One timed interval on a :class:`Timeline`.

    ``t_start`` is seconds since the timeline's origin, so a span is
    meaningful without reference to the process's monotonic base.  On a
    foreign clock it is a placement estimate only — see the module docstring.
    """

    id: int
    name: str
    parent: int | None
    t_start: float
    duration_s: float
    status: str = STATUS_OK
    clock: str = CLOCK_CONTROL
    wall_start: float | None = None
    """Unconverted epoch start, recorded only for foreign-clock spans.

    Kept because ``t_start`` for those is derived through this timeline's
    origin and so carries the skew; a consumer that later learns the offset
    can re-place the span from this."""
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def is_control(self) -> bool:
        """Whether this span's duration may be combined with others'."""
        return self.clock == CLOCK_CONTROL

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "parent": self.parent,
            "t_start": round(self.t_start, 4),
            "duration_s": round(self.duration_s, 4),
            "status": self.status,
        }
        # Omitted on the control clock — that is the documented default and
        # the overwhelmingly common case, so emitting it on every span would
        # bloat the benchmark artifact to say nothing.  A reader seeing no
        # ``clock`` key gets the value it is safe to assume.
        if not self.is_control:
            d["clock"] = self.clock
            if self.wall_start is not None:
                d["wall_start"] = round(self.wall_start, 3)
        if self.attrs:
            d["attrs"] = dict(self.attrs)
        return d


@dataclass
class _Open:
    id: int
    name: str
    parent: int | None
    t_start: float
    attrs: dict[str, Any]


def _resolve_parent(parent: int | None, stack: list[_Open]) -> int | None:
    """Resolve a ``parent=`` argument against the open-span stack.

    Three inputs, three meanings: a span id parents explicitly, :data:`ROOT`
    parents to the root, and ``None`` inherits from the stack.
    """
    if parent == ROOT:
        return None
    if parent is not None:
        return parent
    return stack[-1].id if stack else None


class Timeline:
    """Collects :class:`Span` records for one launch.

    Usage is either bracketed::

        with timeline.span("launch.distribute", host=h):
            ...

    or explicit, for callers whose begin/end are not lexically paired
    (:class:`~sparkrun.core.progress.LaunchProgress` phases)::

        sid = timeline.begin("launch.prepare")
        ...
        timeline.end(sid)
    """

    def __init__(self, *, wall_origin: float | None = None) -> None:
        self._lock = threading.Lock()
        self._spans: list[Span] = []
        self._stack: list[_Open] = []
        self._next_id = 1
        self._t0 = time.monotonic()
        #: Epoch seconds at timeline creation.  The only absolute
        #: timestamp recorded; every span time is relative to it, so a
        #: consumer can reconstruct wall-clock times without us stamping
        #: (and rounding) one per span.
        self.wall_origin = time.time() if wall_origin is None else wall_origin

    # -- properties ---------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        """Seconds since the timeline was created."""
        return time.monotonic() - self._t0

    @property
    def spans(self) -> list[Span]:
        """Closed spans, in start order."""
        with self._lock:
            return sorted(self._spans, key=lambda s: (s.t_start, s.id))

    # -- explicit begin/end -------------------------------------------------

    def begin(self, name: str, *, parent: int | None = None, **attrs: Any) -> int:
        """Open a span and return its id.

        ``parent`` defaults to the innermost span open *on this thread's
        behalf* (the stack).  From a worker thread pass it explicitly — a
        span id, or :data:`ROOT` for the root.
        """
        with self._lock:
            span_id = self._next_id
            self._next_id += 1
            resolved_parent = _resolve_parent(parent, self._stack)
            self._stack.append(
                _Open(
                    id=span_id,
                    name=name,
                    parent=resolved_parent,
                    t_start=time.monotonic(),
                    attrs=dict(attrs),
                )
            )
        return span_id

    def end(self, span_id: int | None = None, *, status: str = STATUS_OK, **attrs: Any) -> None:
        """Close ``span_id`` (default: the innermost open span).

        Any spans still open *inside* it are closed too, carrying the same
        status: if a phase ends in error, the step that was running is
        where it ended, and reporting that step as ``ok`` would point the
        reader at the wrong place.  This is what lets
        :class:`LaunchProgress` open a step without ever closing it — the
        next phase boundary closes it.
        """
        now = time.monotonic()
        with self._lock:
            if not self._stack:
                return
            if span_id is None:
                idx = len(self._stack) - 1
            else:
                idx = next((i for i, o in enumerate(self._stack) if o.id == span_id), -1)
                if idx < 0:
                    return  # already closed, or never ours
            closing = self._stack[idx:]
            del self._stack[idx:]
            for opened in reversed(closing):
                extra = attrs if opened.id == closing[0].id else {}
                self._spans.append(self._finish(opened, now, status, extra))

    def _finish(self, opened: _Open, now: float, status: str, extra: dict[str, Any]) -> Span:
        merged = dict(opened.attrs)
        merged.update(extra)
        return Span(
            id=opened.id,
            name=opened.name,
            parent=opened.parent,
            t_start=opened.t_start - self._t0,
            duration_s=now - opened.t_start,
            status=status,
            attrs=merged,
        )

    # -- bracketed ----------------------------------------------------------

    @contextmanager
    def span(self, name: str, *, parent: int | None = None, **attrs: Any) -> Iterator[int]:
        """Time a block, recording ``error`` status if it raises."""
        span_id = self.begin(name, parent=parent, **attrs)
        try:
            yield span_id
        except BaseException as exc:
            self.end(span_id, status=STATUS_ERROR, error=type(exc).__name__)
            raise
        else:
            self.end(span_id)

    # -- foreign clocks -----------------------------------------------------

    def add_span(
        self,
        name: str,
        *,
        clock: str,
        duration_s: float,
        wall_start: float | None = None,
        t_start: float | None = None,
        parent: int | None = None,
        status: str = STATUS_OK,
        **attrs: Any,
    ) -> int:
        """Record a span measured somewhere else, on another clock.

        This is how engine-internal stages recovered from a container's log
        stream reach the timeline: their start and duration come from the
        engine's own timestamps, not from anything we could bracket.

        ``clock=CLOCK_CONTROL`` is **rejected**.  A control-clock span is
        one we measured, and the whole value of the discriminator is that
        :meth:`total` and any other derived arithmetic can trust it; letting
        a parsed duration in through this door would put an unverifiable
        number in the set that gets summed.  Use :meth:`begin` /
        :meth:`span` to measure, :func:`remote_clock` to name a host's clock.

        Placement: ``t_start`` wins if given, else it is derived from
        ``wall_start`` against this timeline's origin — a subtraction across
        clocks, and therefore an estimate good only for ordering the span in
        a rendered tree.  The unconverted value is kept on the span.
        """
        if clock == CLOCK_CONTROL:
            raise ValueError("add_span() cannot record on the control clock; measure it with begin()/span() instead")
        if t_start is None:
            t_start = (wall_start - self.wall_origin) if wall_start is not None else 0.0
        with self._lock:
            span_id = self._next_id
            self._next_id += 1
            resolved_parent = _resolve_parent(parent, self._stack)
            self._spans.append(
                Span(
                    id=span_id,
                    name=name,
                    parent=resolved_parent,
                    t_start=t_start,
                    duration_s=duration_s,
                    status=status,
                    clock=clock,
                    wall_start=wall_start,
                    attrs=dict(attrs),
                )
            )
        return span_id

    @property
    def clocks(self) -> set[str]:
        """Distinct clocks present, so a consumer can tell at a glance
        whether this timeline mixes them."""
        with self._lock:
            found = {s.clock for s in self._spans}
            if self._stack:
                found.add(CLOCK_CONTROL)
        return found

    def skipped(self, name: str, *, parent: int | None = None, reason: str = "", **attrs: Any) -> None:
        """Record a zero-duration span for work that was deliberately not run."""
        if reason:
            attrs["reason"] = reason
        span_id = self.begin(name, parent=parent, **attrs)
        self.end(span_id, status=STATUS_SKIPPED)

    # -- export -------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Serialize to a JSON/YAML-able envelope.

        Spans still open are included with their elapsed-so-far and
        ``status="open"`` — a launch that raised is exactly when the
        timeline is most worth reading, so dropping them would lose the
        phase that failed.  Non-destructive: the timeline stays usable.
        """
        now = time.monotonic()
        with self._lock:
            spans = list(self._spans)
            spans.extend(self._finish(o, now, STATUS_OPEN, {}) for o in self._stack)
        spans.sort(key=lambda s: (s.t_start, s.id))
        envelope: dict[str, Any] = {
            "wall_origin": round(self.wall_origin, 3),
            "duration_s": round(now - self._t0, 4),
            "spans": [s.to_dict() for s in spans],
        }
        # Only when mixed: a consumer that never sees this key is reading a
        # single-clock timeline and can sum freely.  Announcing it up front
        # saves scanning every span to find out.
        foreign = sorted(c for c in {s.clock for s in spans} if c != CLOCK_CONTROL)
        if foreign:
            envelope["clocks"] = [CLOCK_CONTROL, *foreign]
        return envelope

    def total(self, name: str, *, clock: str = CLOCK_CONTROL) -> float:
        """Summed duration of every closed span named ``name`` on ``clock``.

        For fan-out spans this is machine-seconds, not wall time.

        Filtered by clock, and defaulting to the control one, because this
        is a derived total: summing across clocks mixes measurements that
        skew independently.  Pass ``clock=`` to total a single foreign
        clock's spans, which *are* comparable with each other.
        """
        # Locked like every other reader: spans are appended off the main
        # thread now that the readiness watch records onto a live timeline.
        with self._lock:
            return sum(s.duration_s for s in self._spans if s.name == name and s.clock == clock)

    def find(self, name: str) -> Span | None:
        """First closed span named ``name``, in start order."""
        return next((s for s in self.spans if s.name == name), None)


@contextmanager
def timed(timeline: Timeline | None, name: str, *, parent: int | None = None, **attrs: Any) -> Iterator[int | None]:
    """Time a block, tolerating ``timeline=None``.

    Call sites deep in the orchestration layer receive an optional timeline;
    this keeps them from wrapping every span in an ``if``.
    """
    if timeline is None:
        yield None
        return
    with timeline.span(name, parent=parent, **attrs) as span_id:
        yield span_id
