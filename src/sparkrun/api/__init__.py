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
- **Functions** — ``plan``, ``run``, ``stop``, ``logs``, ``open_log_target``, ``open_exec_target``, ``status``,
  ``schedule``, ``list_jobs``, ``search_recipes``, ``benchmark``, and
  ``resume_benchmark``.
- **Diagnostics** — ``collect_diagnostics`` for standalone bundles and
  ``capture_diagnostics`` for opt-in capture around shared API operations.
- **Returned types** — :class:`ClusterStatus` (with :class:`HostOccupancy`,
  :class:`RunningWorkload`, :class:`GpuOccupancy`) from ``status``,
  :class:`ClusterDefinition` (``RunPlan.cluster``, ``cluster=`` arguments),
  and :class:`SchedulingRequest` / :class:`SchedulingResult` for ``schedule``.
  Imported lazily, so ``import sparkrun.api`` stays cheap.
- **Experimental** (may change in a minor release) — :mod:`sparkrun.api.workloads`,
  the lifecycle helpers ``wait_ready`` / ``post_launch`` / ``discovery_snapshot``
  (:class:`PostLaunchResult`), the out-of-launch helpers
  ``replace_intent_deployments`` / ``fresh_workload_id``, :mod:`sparkrun.api.model_cache`,
  :mod:`sparkrun.api.systemd_export` (imported on demand), and the
  ``sparkrun.api.setup`` planning internals. See
  ``docs/DISTRIBUTION_API_MIGRATION.md``.

The API never writes to ``stdout`` / ``stderr`` and never calls
``sys.exit``. Operational failures generally use :class:`SparkrunError` subclasses;
log readers also expose standard ``OSError`` / ``subprocess`` read failures
(see ``docs/LOGGING_API.md``).
Diagnostic bundle output-open failures raise ``OSError``; section failures are
reported in the bundle/result (see ``docs/DIAGNOSTICS_API.md``).
Pure model/plan validation may raise ValueError or TypeError. Implicit bootstrap
failures are wrapped with their original cause; interrupts propagate. Streaming
surfaces (``logs``) return iterators of structured records that the CLI renders
to the TTY.

Stability: the dataclass shapes, catalog dictionary shapes, and exception hierarchy are stable;
field additions are non-breaking, field removals are breaking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from sparkrun.api._context import default_sctx, resolve_sctx
from sparkrun.api._diagnostics import DiagnosticsCapture, DiagnosticsResult, capture_diagnostics, collect_diagnostics
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
from sparkrun.api._exec import ExecTarget, open_exec_target
from sparkrun.api._logs import LogTarget, logs, open_log_target
from sparkrun.core.log_source import LogResource, LogSourceChanged
from sparkrun.api._materialize import materialize
from sparkrun.api._models import (
    JobInfo,
    LogLine,
    LogSource,
    RecipeProtocols,
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
from sparkrun.api._protocols import recipe_protocols
from sparkrun.api._realize import RealizedRecipe, realize_recipe
from sparkrun.api._recipes import resolve_recipe_filter, search_recipes
from sparkrun.api._run import fresh_workload_id, plan, replace_intent_deployments, run, untrusted_hooks
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

from sparkrun.api import workloads

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
    "resolve_sctx",
    "collect_diagnostics",
    "capture_diagnostics",
    "DiagnosticsCapture",
    "DiagnosticsResult",
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
    "LogResource",
    "LogSourceChanged",
    "LogTarget",
    "ExecTarget",
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
    "open_log_target",
    "open_exec_target",
    "schedule",
    "status",
    "status_report",
    "open_telemetry",
    "open_live_monitor",
    "live_monitor",
    "LiveMonitorSession",
    "list_jobs",
    "find_running_intent",
    "untrusted_hooks",
    "search_recipes",
    "resolve_recipe_filter",
    "validate_recipe",
    "RecipeIssue",
    "benchmark",
    "resume_benchmark",
]

from sparkrun.api import model_cache
from sparkrun.api._lifecycle import PostLaunchResult, wait_ready, post_launch, discover_endpoints, discovery_snapshot  # noqa: F401
from sparkrun.api._clusters import create_cluster, update_cluster, delete_cluster, set_default_cluster

__all__ += [
    "model_cache",
    "wait_ready",
    "post_launch",
    "PostLaunchResult",
    "discovery_snapshot",
    "recipe_protocols",
    "RecipeProtocols",
    "replace_intent_deployments",
    "fresh_workload_id",
    "create_cluster",
    "update_cluster",
    "delete_cluster",
    "set_default_cluster",
]

__all__ += ["workloads"]

from . import plugins, features  # noqa: E402

__all__ += ["plugins", "features"]

# Types public functions return (or take), re-exported lazily: importing their
# core modules eagerly would cost every ``import sparkrun.api`` (and so every
# shell completion) for callers that never touch them.
_LAZY_TYPES = {
    "ClusterStatus": "sparkrun.core.cluster_status",
    "HostOccupancy": "sparkrun.core.cluster_status",
    "RunningWorkload": "sparkrun.core.cluster_status",
    "GpuOccupancy": "sparkrun.core.cluster_status",
    "ClusterDefinition": "sparkrun.core.cluster_manager",
    "SchedulingRequest": "sparkrun.core.scheduler",
    "SchedulingResult": "sparkrun.core.scheduler",
}

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus, GpuOccupancy, HostOccupancy, RunningWorkload
    from sparkrun.core.scheduler import SchedulingRequest, SchedulingResult


def __getattr__(name: str):
    module = _LAZY_TYPES.get(name)
    if module is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ += [
    "ClusterStatus",
    "HostOccupancy",
    "RunningWorkload",
    "GpuOccupancy",
    "ClusterDefinition",
    "SchedulingRequest",
    "SchedulingResult",
]
