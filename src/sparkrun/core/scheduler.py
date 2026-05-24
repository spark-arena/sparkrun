"""Scheduler interface + placement vocabulary.

This module owns the full placement-and-scheduling vocabulary:

- **Data shapes** describing a placement decision:
  :class:`RankSlot`, :class:`RankAssignment`.
- **Low-level errors** raised by scheduler *implementations*
  (e.g. :mod:`sparkrun.schedulers.greedy`):
  :class:`PlacementError`, :class:`LayoutRequiredError`,
  :class:`InsufficientCapacityError`.
- **High-level errors** raised through the
  :class:`Scheduler` *interface* (catchable by ``sparkrun.api``):
  :class:`SchedulingError`, :class:`InfeasibleScheduleError`,
  :class:`LayoutConflictError`.
- **Strategy interface**: :class:`Scheduler` ABC plus
  :class:`SchedulingRequest` / :class:`SchedulingResult`.
- **Registry helpers**: :func:`get_scheduler`,
  :func:`list_schedulers`, :data:`EXT_SCHEDULER`.

A scheduler is an SAF :class:`Plugin` registered at
:data:`EXT_SCHEDULER` and discovered via
``find_types_in_modules('sparkrun.schedulers', Scheduler)``.

Higher layers (``sparkrun.api.schedule``, the launcher, the CLI) only
talk to schedulers through this interface — they never call placement
algorithms directly.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Mapping

from scitrera_app_framework import Plugin, Variables

from sparkrun.core.cluster_status import ClusterStatus
from sparkrun.core.hardware import HostHardware
from sparkrun.core.layout import RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig

logger = logging.getLogger(__name__)

EXT_SCHEDULER = "sparkrun.scheduler"

FALLBACK_DEFAULT_SCHEDULER = "occupancy-sparse"


# --------------------------------------------------------------------------
# Errors — placement-level (raised by scheduler implementations)
# --------------------------------------------------------------------------


class PlacementError(Exception):
    """Base class for placement-level errors raised by scheduler implementations.

    Implementations may raise this (or its subclasses) directly; the
    :class:`Scheduler` interface translates these into the
    scheduler-level vocabulary so callers of the interface see a
    single error hierarchy.
    """


class LayoutRequiredError(PlacementError):
    """Raised when a heterogeneous cluster is missing an explicit ``recipe.layout``.

    Heterogeneous clusters can't be auto-fit (vendor-mixed clusters in
    particular may require splitting work along boundaries the engine
    can't safely infer).  The recipe must declare placements explicitly.
    """


class InsufficientCapacityError(PlacementError):
    """Raised when the cluster doesn't have enough accelerator slots for the requested parallelism."""


# --------------------------------------------------------------------------
# Errors — scheduler-level (raised through the Scheduler interface)
# --------------------------------------------------------------------------


class SchedulingError(Exception):
    """Base class for scheduler-level errors.

    Subclasses are caught by ``sparkrun.api`` and translated into the
    user-facing :class:`~sparkrun.api._errors.InsufficientCapacity` and
    :class:`~sparkrun.api._errors.LayoutRequired` exceptions.
    """


class InfeasibleScheduleError(SchedulingError):
    """The scheduler cannot satisfy the request with the available capacity."""


class LayoutConflictError(SchedulingError):
    """The scheduler requires an explicit layout but none was provided."""


# --------------------------------------------------------------------------
# Data shapes — what a placement decision looks like
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceRequest:
    """Per-rank resource claim against an accelerator's capacity.

    Default values request a whole accelerator — today's behavior.  A
    fractional-capable scheduler (e.g. ``SparsePackScheduler`` /
    ``DensePackScheduler``) honors ``util_fraction < 1.0`` and packs
    multiple ranks onto one accelerator if their combined claims fit
    within ``memory_gb`` and ``util_fraction`` budgets.

    The default :class:`~sparkrun.schedulers.greedy.GreedyScheduler`
    rejects requests with ``util_fraction < 1.0`` rather than silently
    treating them as whole-GPU — schedulers are expected to fail fast
    when asked for behavior they can't deliver.
    """

    memory_gb: float | None = None
    """VRAM budget for this rank.  ``None`` means "as much as available"."""

    util_fraction: float = 1.0
    """Fraction of one accelerator this rank uses.  ``1.0`` = exclusive ownership."""

    def is_fractional(self) -> bool:
        """``True`` if this request asks for a fraction of an accelerator."""
        return self.util_fraction < 1.0


@dataclass(frozen=True)
class RankSlot:
    """One global rank's home: the host, local accelerator index, and resource claim.

    For whole-GPU placements (today's default), :attr:`util_fraction`
    is ``1.0`` and :attr:`memory_gb` is ``None`` — the rank owns the
    accelerator at ``(host, local_gpu)`` exclusively.

    Fractional schedulers may emit multiple :class:`RankSlot` entries
    sharing the same ``(host, local_gpu)`` coordinate with
    ``util_fraction < 1.0``, carrying the per-rank VRAM commitment in
    :attr:`memory_gb`.
    """

    host: str
    local_gpu: int
    util_fraction: float = 1.0
    """Fraction of the named accelerator this rank uses (``1.0`` = exclusive)."""
    memory_gb: float | None = None
    """VRAM committed to this rank, or ``None`` for whole-GPU placements."""


@dataclass(frozen=True)
class RankAssignment:
    """Concrete mapping from global rank to (host, local-GPU index)."""

    by_rank: tuple[RankSlot, ...]
    """``by_rank[i]`` is the slot for global rank *i*."""

    hosts_used: tuple[str, ...]
    """Distinct hosts that participate, in rank-major order (first appearance)."""

    def host_for_rank(self, rank: int) -> str:
        """Host that runs *rank*."""
        return self.by_rank[rank].host

    def local_gpu_for_rank(self, rank: int) -> int:
        """Local accelerator index that runs *rank* on its host."""
        return self.by_rank[rank].local_gpu

    def ranks_on_host(self, host: str) -> tuple[int, ...]:
        """Global ranks scheduled on *host*, in ascending order."""
        return tuple(i for i, slot in enumerate(self.by_rank) if slot.host == host)

    @property
    def total_ranks(self) -> int:
        return len(self.by_rank)

    @property
    def max_ranks_per_host(self) -> int:
        """Largest number of ranks assigned to any single host (slot count for MPI)."""
        if not self.by_rank:
            return 0
        counts: dict[str, int] = {}
        for slot in self.by_rank:
            counts[slot.host] = counts.get(slot.host, 0) + 1
        return max(counts.values())


# --------------------------------------------------------------------------
# Strategy interface — what schedulers take and return
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulingRequest:
    """Inputs to a scheduling decision.

    Future knobs (priority, soft constraints, deadlines) extend this
    dataclass without breaking the :class:`Scheduler` interface.
    """

    parallelism: ParallelismConfig
    hosts: tuple[str, ...]
    host_hardware: Mapping[str, HostHardware] | None = None
    layout: RecipeLayout | None = None
    status: ClusterStatus | None = None
    resources: ResourceRequest | None = None
    """Per-rank resource claim.  ``None`` means a whole accelerator per rank
    (today's default).  Fractional-capable schedulers honor
    :attr:`ResourceRequest.util_fraction` < 1.0 to pack multiple ranks
    onto one accelerator; the default :class:`GreedyScheduler` rejects
    fractional claims rather than silently treating them as whole-GPU."""


@dataclass(frozen=True)
class SchedulingResult:
    """Outputs of a scheduling decision.

    Future fields (reservations, alternatives offered) extend this
    dataclass without breaking callers that only read ``assignment``.
    """

    assignment: RankAssignment
    scheduler_name: str
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


class Scheduler(Plugin):
    """Abstract base for placement schedulers.

    Subclasses must set :attr:`scheduler_name` and implement
    :meth:`schedule`.  :meth:`feasibility` has a default that tries
    :meth:`schedule` and returns ``False`` on :class:`SchedulingError`.

    Concrete schedulers live in :mod:`sparkrun.schedulers` so bootstrap
    can discover them via ``find_types_in_modules``.
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    scheduler_name: ClassVar[str] = ""

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.scheduler.%s" % self.scheduler_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_SCHEDULER

    def is_enabled(self, v: Variables) -> bool:
        # Multi-extension plugins must return False to avoid SAF's
        # single-extension short-circuit (see RuntimePlugin / Executor).
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger=None) -> "Scheduler":
        return self

    # --- Strategy interface ---

    @abstractmethod
    def schedule(self, request: SchedulingRequest) -> SchedulingResult:
        """Produce a placement for *request* or raise :class:`SchedulingError`."""
        ...

    def feasibility(self, request: SchedulingRequest) -> bool:
        """Return ``True`` iff :meth:`schedule` would succeed.

        Default implementation simply runs :meth:`schedule` and catches
        :class:`SchedulingError`.  Subclasses may override with a
        cheaper pre-check that doesn't construct a full assignment.
        """
        try:
            self.schedule(request)
            return True
        except SchedulingError:
            return False


# --------------------------------------------------------------------------
# Registry helpers
# --------------------------------------------------------------------------


def get_scheduler(name: str | None = None, v: Variables | None = None) -> Scheduler:
    """Return the scheduler registered as *name*, or the default greedy one.

    Args:
        name: Registered :attr:`Scheduler.scheduler_name`.  ``None`` or
            ``"default"`` selects ``"greedy"``.
        v: Optional SAF Variables.

    Raises:
        ValueError: When *name* doesn't match any registered scheduler.
    """
    from scitrera_app_framework import get_extensions

    from sparkrun.core.bootstrap import get_variables

    if v is None:
        v = get_variables()

    target = (name or FALLBACK_DEFAULT_SCHEDULER).strip().lower()
    if target == "default":
        target = FALLBACK_DEFAULT_SCHEDULER

    plugins = get_extensions(EXT_SCHEDULER, v=v)
    for plugin in plugins.values():
        if getattr(plugin, "scheduler_name", "") == target:
            return plugin

    available = sorted(getattr(p, "scheduler_name", "") for p in plugins.values())
    raise ValueError("Unknown scheduler: %r. Available: %s" % (name, available))


def list_schedulers(v: Variables | None = None) -> list[str]:
    """Return registered scheduler names, sorted."""
    from scitrera_app_framework import get_extensions

    from sparkrun.core.bootstrap import get_variables

    if v is None:
        v = get_variables()

    plugins = get_extensions(EXT_SCHEDULER, v=v)
    return sorted(p.scheduler_name for p in plugins.values() if getattr(p, "scheduler_name", ""))
