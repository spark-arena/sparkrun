"""Public library API for sparkrun.

This package is the contract that non-CLI Python callers (tests, third-
party automation, the CLI itself) depend on.  Surfaces:

- **Data models** — :class:`RunOptions`, :class:`RunResult`,
  :class:`StopResult`, :class:`LogLine`, :class:`JobInfo`,
  :class:`BenchmarkOptions`, :class:`BenchmarkResult`,
  :class:`ProgressEvent`, :class:`ResumeMode`.
- **Errors** — :class:`SparkrunError` and typed subclasses
  (:class:`InsufficientCapacity`, :class:`LayoutRequired`,
  :class:`RecipeNotFound`, :class:`HostsUnreachable`,
  :class:`JobNotFound`, :class:`TrustRejected`,
  :class:`BenchmarkFailed`, :class:`NoResumableState`,
  :class:`CategoryNotFound`, :class:`AmbiguousCategoryError`,
  :class:`FrameworkCategoryMismatch`).
- **Functions** — ``run``, ``stop``, ``logs``, ``status``,
  ``schedule``, ``list_jobs``, ``benchmark`` (added incrementally in
  subsequent tasks; this module re-exports them as they land).

The API never writes to ``stdout`` / ``stderr`` and never calls
``sys.exit``.  Errors are raised as :class:`SparkrunError`
subclasses; streaming surfaces (``logs``) return iterators of
structured records that the CLI renders to the TTY.

Stability: the dataclass shapes and exception hierarchy are stable;
field additions are non-breaking, field removals are breaking.
"""

from __future__ import annotations

from sparkrun.api._benchmark import benchmark, resume_benchmark
from sparkrun.api._benchmark_models import (
    BenchmarkOptions,
    BenchmarkResult,
    ProgressEvent,
    ResumeMode,
)
from sparkrun.api._context import default_sctx
from sparkrun.api._errors import (
    AmbiguousCategoryError,
    AmbiguousWorkload,
    BenchmarkFailed,
    CategoryNotFound,
    FrameworkCategoryMismatch,
    HostsUnreachable,
    InsufficientCapacity,
    JobNotFound,
    LayoutRequired,
    NoResumableState,
    RecipeNotFound,
    SparkrunError,
    TrustRejected,
)
from sparkrun.core.context import SparkrunContext
from sparkrun.api._jobs import list_jobs
from sparkrun.api._logs import logs
from sparkrun.api._models import (
    JobInfo,
    LogLine,
    RunOptions,
    RunResult,
    StopResult,
)
from sparkrun.api._run import run
from sparkrun.api._schedule import schedule
from sparkrun.api._status import status
from sparkrun.api._stop import stop

__all__ = [
    # Session context
    "SparkrunContext",
    "default_sctx",
    # Data models
    "RunOptions",
    "RunResult",
    "StopResult",
    "LogLine",
    "JobInfo",
    # Benchmark data models
    "BenchmarkOptions",
    "BenchmarkResult",
    "ProgressEvent",
    "ResumeMode",
    # Errors
    "SparkrunError",
    "InsufficientCapacity",
    "LayoutRequired",
    "RecipeNotFound",
    "HostsUnreachable",
    "JobNotFound",
    "AmbiguousWorkload",
    "TrustRejected",
    # Benchmark errors
    "BenchmarkFailed",
    "NoResumableState",
    "CategoryNotFound",
    "AmbiguousCategoryError",
    "FrameworkCategoryMismatch",
    # Functions
    "run",
    "stop",
    "logs",
    "schedule",
    "status",
    "list_jobs",
    "benchmark",
    "resume_benchmark",
]
