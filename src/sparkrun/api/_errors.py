"""Typed exception hierarchy for the sparkrun library API.

Callers of :mod:`sparkrun.api` see exactly this hierarchy.  Internal
errors from :mod:`sparkrun.core.scheduler` (``InfeasibleScheduleError``,
``LayoutConflictError``, etc.) are translated into these user-facing
exceptions by the API implementation layer.

All errors derive from :class:`SparkrunError` so callers can ``except
SparkrunError`` for a generic failure handler without having to know
the specific subclass.
"""

from __future__ import annotations


class SparkrunError(Exception):
    """Base class for all errors raised by the sparkrun library API.

    Callers can catch this for a generic failure path; in most cases
    callers will want to discriminate on a more specific subclass.
    """


class InsufficientCapacity(SparkrunError):
    """Cluster lacks accelerator slots for the requested parallelism.

    Surfaced when a scheduler raises
    :class:`~sparkrun.core.scheduler.InfeasibleScheduleError`.  The
    message carries the slot count seen vs requested.
    """


class LayoutRequired(SparkrunError):
    """Heterogeneous cluster needs an explicit ``recipe.layout``.

    Surfaced when a scheduler raises
    :class:`~sparkrun.core.scheduler.LayoutConflictError` — i.e. the
    cluster spans multiple accelerator vendors and the auto-pack
    algorithm can't choose splits safely.
    """


class RecipeNotFound(SparkrunError):
    """Named recipe could not be resolved across configured registries."""


class HostsUnreachable(SparkrunError):
    """One or more hosts could not be reached over SSH.

    The exception carries the list of unreachable hosts in
    :attr:`hosts`.  Callers may inspect this to display per-host
    diagnostics; absent listing the message still names them.
    """

    def __init__(self, message: str, hosts: list[str] | None = None) -> None:
        super().__init__(message)
        self.hosts: tuple[str, ...] = tuple(hosts or ())


class JobNotFound(SparkrunError):
    """No running job matches the given identification (cluster_id / recipe+hosts)."""


class AmbiguousWorkload(SparkrunError):
    """Multiple running workloads match the supplied recipe+hosts intent.

    Raised by :func:`sparkrun.api.stop` when the recipe path matches
    more than one running cluster (e.g. two parallel deployments of the
    same recipe on disjoint host sets).  The exception's
    :attr:`cluster_ids` attribute carries the candidates so callers can
    re-invoke with an explicit ``cluster_id``.
    """

    def __init__(self, message: str, cluster_ids: list[str] | tuple[str, ...] | None = None) -> None:
        super().__init__(message)
        self.cluster_ids: tuple[str, ...] = tuple(cluster_ids or ())


class TrustRejected(SparkrunError):
    """User declined trust prompt for a third-party recipe.

    The launcher abandons the run rather than executing untrusted
    pre/post hooks.  Raised by ``api.run`` when ``options.trust=False``
    is the user's final answer.
    """


# --------------------------------------------------------------------------
# Benchmark errors
# --------------------------------------------------------------------------


class BenchmarkFailed(SparkrunError):
    """A benchmark run failed (non-zero rc, task failures, or aborted launch).

    Carries the original exit code in :attr:`exit_code` when known.
    """

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class NoResumableState(SparkrunError):
    """``ResumeMode.REQUIRED`` but no benchmark state exists for the derived id."""


class CategoryNotFound(SparkrunError):
    """Requested benchmark category has no registered frameworks."""


class AmbiguousCategoryError(SparkrunError):
    """Category has multiple frameworks; pin one via config or ``--framework``."""


class FrameworkCategoryMismatch(SparkrunError):
    """Explicit framework does not belong to the explicit category."""


__all__ = [
    "SparkrunError",
    "InsufficientCapacity",
    "LayoutRequired",
    "RecipeNotFound",
    "HostsUnreachable",
    "JobNotFound",
    "AmbiguousWorkload",
    "TrustRejected",
    "BenchmarkFailed",
    "NoResumableState",
    "CategoryNotFound",
    "AmbiguousCategoryError",
    "FrameworkCategoryMismatch",
]
