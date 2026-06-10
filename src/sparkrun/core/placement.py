"""Deprecated re-export module — placement vocabulary lives in :mod:`sparkrun.core.scheduler`.

This module exists solely to keep in-tree callers working during the
v0.3.x multiplatform refactor.  It re-exports the placement data types,
placement-level errors, and a deprecated :func:`compute_placement`
shim that delegates to :func:`sparkrun.schedulers.greedy.pack`.

All in-tree callers now route through :func:`sparkrun.api.schedule`; only
tests still exercise the shim directly.  It will be removed once those are
migrated.  New code should import from :mod:`sparkrun.core.scheduler` and
call schedulers via :class:`~sparkrun.core.scheduler.Scheduler` or
``sparkrun.api``.
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

    Retained only for tests that exercise the greedy pack directly;
    production code routes through ``sparkrun.api.schedule``.

    Raises the same placement-level exceptions as
    :func:`sparkrun.schedulers.greedy.pack`.
    """
    from sparkrun.schedulers.greedy import pack

    return pack(parallelism, hosts, host_hardware=host_hardware, layout=layout)
