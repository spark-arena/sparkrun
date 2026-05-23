"""``sparkrun.api.schedule`` — invoke a scheduler from the library API.

Translates scheduler-level errors
(:class:`~sparkrun.core.scheduler.InfeasibleScheduleError`,
:class:`~sparkrun.core.scheduler.LayoutConflictError`) into the
user-facing :class:`~sparkrun.api.InsufficientCapacity` and
:class:`~sparkrun.api.LayoutRequired` exceptions so callers see a
single error hierarchy regardless of which scheduler ran.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import InsufficientCapacity, LayoutRequired, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.scheduler import SchedulingRequest, SchedulingResult

logger = logging.getLogger(__name__)


def schedule(
    request: "SchedulingRequest",
    *,
    scheduler: str | None = None,
) -> "SchedulingResult":
    """Run *request* through the named scheduler (or the default greedy one).

    Args:
        request: The :class:`SchedulingRequest` describing the workload.
        scheduler: Registered scheduler name.  ``None`` or ``"default"``
            selects ``"greedy"``.

    Returns:
        A :class:`SchedulingResult` with the resolved
        :class:`RankAssignment` and scheduler-emitted diagnostics.

    Raises:
        InsufficientCapacity: When the scheduler can't fit the workload.
        LayoutRequired: Heterogeneous cluster without an explicit layout.
        SparkrunError: For other scheduling failures.
    """
    from sparkrun.core.scheduler import (
        InfeasibleScheduleError,
        LayoutConflictError,
        SchedulingError,
        get_scheduler,
    )

    try:
        plugin = get_scheduler(scheduler)
    except ValueError as e:
        raise SparkrunError(str(e)) from e

    try:
        return plugin.schedule(request)
    except InfeasibleScheduleError as e:
        raise InsufficientCapacity(str(e)) from e
    except LayoutConflictError as e:
        raise LayoutRequired(str(e)) from e
    except SchedulingError as e:
        raise SparkrunError(str(e)) from e


__all__ = ["schedule"]
