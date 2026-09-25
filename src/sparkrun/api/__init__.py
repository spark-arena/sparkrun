"""Public library API for sparkrun.

This package is the contract that non-CLI Python callers (tests, third-
party automation, the CLI itself) depend on.  Surfaces:

- **Preparation** — :class:`BuildOptions`, :class:`BuildPlan`,
  :class:`BuildResult`, ``plan_build``, and ``build`` prepare assets without
  starting inference. ``materialize`` continues to resolve launch specs only.
- **Data models** — :class:`RunOptions`, :class:`RunPlan`, :class:`RunResult`,
  :class:`StopResult`, :class:`LogLine`, :class:`JobInfo`,
  :class:`RecipeSummary`, :class:`BenchmarkOptions`,
  :class:`BenchmarkResult`, :class:`ProgressEvent`, :class:`ResumeMode`.
- **Errors** — :class:`SparkrunError` and typed subclasses
  (:class:`InsufficientCapacity`, :class:`LayoutRequired`,
  :class:`RecipeNotFound`, :class:`InvalidRegistryFilter`,
  :class:`HostsUnreachable`, :class:`JobNotFound`,
  :class:`TrustRejected`, :class:`BenchmarkFailed`,
  :class:`NoResumableState`, :class:`CategoryNotFound`,
  :class:`AmbiguousCategoryError`,
  :class:`FrameworkCategoryMismatch`).
- **Functions** — ``plan``, ``run``, ``stop``, ``logs``, ``status``,
  ``schedule``, ``list_jobs``, ``search_recipes``, ``benchmark``, and
  ``resume_benchmark``.

The API never writes to ``stdout`` / ``stderr`` and never calls
``sys.exit``. Operational failures use :class:`SparkrunError` subclasses;
pure model/plan validation may raise ValueError or TypeError. Implicit bootstrap
failures are wrapped with their original cause; interrupts propagate. Streaming
surfaces (``logs``) return iterators of structured records that the CLI renders
to the TTY.

Stability: the dataclass shapes, catalog dictionary shapes, and exception hierarchy are stable;
field additions are non-breaking, field removals are breaking.
"""

from __future__ import annotations

from sparkrun.api import proxy
from sparkrun.api import setup
from sparkrun.api import tailscale
from sparkrun.api._build import build, plan_build
from sparkrun.api._build_models import BuildOptions, BuildPlan, BuildResult
from sparkrun.api._benchmark import benchmark, resume_benchmark
from sparkrun.api._benchmark_models import (
    BenchmarkOptions,
    BenchmarkResult,
    ProgressEvent,
    ProgressEventKind,
    ProgressEventData,
    BenchmarkDecision,
    ResumeMode,
)
from sparkrun.api._catalog_models import (
    CatalogBenchmarkContext,
    CatalogCapacity,
    CatalogCluster,
    CatalogFacet,
    CatalogHostCapacity,
    CatalogIssue,
    CatalogPage,
    CatalogRecipe,
    CatalogRecipeDetails,
    CatalogRecipeMetadata,
    CatalogRefreshResult,
    CatalogRegistry,
    CatalogRegistryResult,
    ResolvedCatalogRecipe,
)
from sparkrun.api._context import default_sctx
from sparkrun.api._errors import (
    AmbiguousCategoryError,
    AmbiguousWorkload,
    BenchmarkFailed,
    BenchmarkIntegrationFailed,
    BenchmarkFinalizationFailed,
    CategoryNotFound,
    FrameworkCategoryMismatch,
    HostsUnreachable,
    InsufficientCapacity,
    IntegrationUnavailable,
    InvalidRegistryFilter,
    JobNotFound,
    LayoutRequired,
    NoResumableState,
    RecipeNotFound,
    SparkrunError,
    TrustRejected,
)
from sparkrun.core.context import SparkrunContext
from sparkrun.api._intent import IntentMatch, find_running_intent
from sparkrun.api._jobs import list_jobs
from sparkrun.api._logs import logs
from sparkrun.api._materialize import materialize
from sparkrun.api._models import (
    JobInfo,
    LogLine,
    LogSource,
    RecipeSummary,
    ResolvedAdapterTopology,
    ResolvedExecutionGraph,
    ResolvedLaunchSpec,
    ResolvedLaunchUnit,
    ResolvedMount,
    ResolvedProcessGroup,
    ResolvedServiceDomain,
    ResolvedWorker,
    RunOptions,
    RunPlan,
    RunResult,
    StopAllResult,
    StopResult,
)
from sparkrun.api._realize import RealizedRecipe, realize_recipe
from sparkrun.api._recipes import resolve_recipe_filter, search_recipes
from sparkrun.api._run import plan, run
from sparkrun.api._schedule import schedule
from sparkrun.api._status import status, status_report
from sparkrun.api._stop import stop
from sparkrun.api._stop_all import stop_all
from sparkrun.api._telemetry import LiveMonitorSession, live_monitor, open_live_monitor, open_telemetry
from sparkrun.core.validation import RecipeIssue, validate_recipe

from sparkrun.api._catalog import (
    cleanup_catalog_imports,
    retain_catalog_recipe,
    catalog_recipes,
    configure_registry,
    catalog_cluster_capacity,
    get_recipe_details,
    import_recipe,
    list_clusters,
    list_registries,
    refresh_registries,
    resolve_catalog_recipe,
)

__all__ = [
    # Catalog payload types
    "CatalogBenchmarkContext",
    "CatalogCapacity",
    "CatalogCluster",
    "CatalogFacet",
    "CatalogHostCapacity",
    "CatalogIssue",
    "CatalogPage",
    "CatalogRecipe",
    "CatalogRecipeDetails",
    "CatalogRecipeMetadata",
    "CatalogRefreshResult",
    "CatalogRegistry",
    "CatalogRegistryResult",
    "ResolvedCatalogRecipe",
    "catalog_recipes",
    "configure_registry",
    "catalog_cluster_capacity",
    "cleanup_catalog_imports",
    "retain_catalog_recipe",
    "get_recipe_details",
    "import_recipe",
    "list_clusters",
    "list_registries",
    "refresh_registries",
    "resolve_catalog_recipe",
    # Subpackages
    "proxy",
    "setup",
    "tailscale",
    # Session context
    "SparkrunContext",
    "default_sctx",
    "build",
    "plan_build",
    # Data models
    "BuildOptions",
    "BuildPlan",
    "BuildResult",
    "RunOptions",
    "RunPlan",
    "RunResult",
    "StopResult",
    "StopAllResult",
    "LogLine",
    "LogSource",
    "JobInfo",
    "IntentMatch",
    "RecipeSummary",
    "ResolvedLaunchSpec",
    "ResolvedLaunchUnit",
    "ResolvedMount",
    "ResolvedWorker",
    "ResolvedProcessGroup",
    "ResolvedServiceDomain",
    "ResolvedAdapterTopology",
    "ResolvedExecutionGraph",
    # Benchmark data models
    "BenchmarkOptions",
    "BenchmarkResult",
    "ProgressEvent",
    "ProgressEventKind",
    "ProgressEventData",
    "BenchmarkDecision",
    "ResumeMode",
    # Errors
    "SparkrunError",
    "InsufficientCapacity",
    "IntegrationUnavailable",
    "LayoutRequired",
    "RecipeNotFound",
    "InvalidRegistryFilter",
    "HostsUnreachable",
    "JobNotFound",
    "AmbiguousWorkload",
    "TrustRejected",
    # Benchmark errors
    "BenchmarkFailed",
    "BenchmarkIntegrationFailed",
    "BenchmarkFinalizationFailed",
    "NoResumableState",
    "CategoryNotFound",
    "AmbiguousCategoryError",
    "FrameworkCategoryMismatch",
    # Functions
    "plan",
    "materialize",
    "realize_recipe",
    "RealizedRecipe",
    "run",
    "stop",
    "stop_all",
    "logs",
    "schedule",
    "status",
    "status_report",
    "open_telemetry",
    "open_live_monitor",
    "live_monitor",
    "LiveMonitorSession",
    "list_jobs",
    "find_running_intent",
    "search_recipes",
    "resolve_recipe_filter",
    "validate_recipe",
    "RecipeIssue",
    "benchmark",
    "resume_benchmark",
]
