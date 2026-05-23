"""Deprecated re-export module — placement vocabulary lives in :mod:`sparkrun.core.scheduler`.

This module exists solely to keep in-tree callers working during the
v0.3.x multiplatform refactor.  It re-exports the placement data types,
placement-level errors, and a deprecated :func:`compute_placement`
shim that delegates to :func:`sparkrun.schedulers.greedy.pack`.

It will be removed once all in-tree callers route through
:func:`sparkrun.api.schedule` (Tasks 8–11 of the refactor).  New code
should import from :mod:`sparkrun.core.scheduler` and call schedulers
via :class:`~sparkrun.core.scheduler.Scheduler` or ``sparkrun.api``.
"""

from __future__ import annotations

from typing import Mapping

# Re-export the canonical names so existing
# `from sparkrun.core.placement import …` callers keep compiling.
from sparkrun.core.scheduler import (  # noqa: F401
    InsufficientCapacityError,
    LayoutRequiredError,
    PlacementError,
    RankAssignment,
    RankSlot,
)


def compute_placement(
    parallelism,
    hosts: list[str],
    *,
    host_hardware: "Mapping[str, object] | None" = None,
    layout=None,
) -> RankAssignment:
    """Deprecated shim — use ``sparkrun.api.schedule`` or :class:`GreedyScheduler`.

    Retained so in-tree callers that haven't migrated keep working;
    will be removed once Tasks 8–11 of the multiplatform refactor
    route everything through the API surface.

    Raises the same placement-level exceptions as
    :func:`sparkrun.schedulers.greedy.pack`.
    """
    from sparkrun.schedulers.greedy import pack

    return pack(parallelism, hosts, host_hardware=host_hardware, layout=layout)
