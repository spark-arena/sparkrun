"""Data shapes for the sparkrun benchmark API surface.

All public dataclasses and enums for ``sparkrun.api.benchmark`` live here.
These are stable contracts; field additions are non-breaking, field removals
are breaking.

``BenchmarkOptions`` mirrors the CLI ``benchmark`` command's flag set as a
typed struct.  ``BenchmarkResult`` carries the structured outcome of a
completed run.  ``ProgressEvent`` carries presentation-independent notifications.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe
    from sparkrun.api._models import RunResult


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class ResumeMode(str, Enum):
    """Controls how an API benchmark run handles pre-existing state.

    ``AUTO``
        Prompt the user (or use CLI default) when prior state exists.
        Maps to the CLI's behaviour when neither ``--resume`` nor
        ``--fresh`` is passed.
    ``IF_EXISTS``
        Resume if compatible state exists, start fresh otherwise.
        This is the API default — matches the CLI ``--resume`` flag.
    ``FRESH``
        Delete prior state and start a new run unconditionally.
        Matches the CLI ``--fresh`` flag.
    ``REQUIRED``
        API-only: raise :class:`~sparkrun.api._errors.NoResumableState`
        when no resumable state exists for the derived benchmark id.
        Not exposed as a CLI flag.
    """

    AUTO = "auto"
    IF_EXISTS = "if_exists"
    FRESH = "fresh"
    REQUIRED = "required"


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


ProgressEventKind = Literal[
    "banner",
    "info",
    "warning",
    "error",
    "progress_step",
    "schedule_started",
    "task_start",
    "task_end",
    "results_update",
    "schedule_log",
    "schedule_finished",
    "run_complete",
]


class ProgressEventData(TypedDict, total=False):
    """Builtin payload fields; required keys depend on the event kind.

    See BENCHMARK_API.md for the per-kind schema. Consumers should ignore
    additional fields so new observations do not break existing frontends.
    """

    line: str
    msg: str
    step: int
    total: int
    label: str
    benchmark_id: str
    total_tasks: int
    title: str
    index: int
    success: bool
    duration_s: float | None
    results: dict[str, Any]
    message: str
    resumed: bool


@dataclass(frozen=True)
class ProgressEvent:
    """Event dispatched to :attr:`BenchmarkOptions.progress_callback` during a run.

    Carries a stable ``kind`` discriminator and a free-form ``data``
    payload. Scheduled runs and resumes emit ``schedule_started``, ``task_start``,
    ``task_end``, ``results_update``, ``schedule_log`` and ``schedule_finished``.
    Successful API calls also emit ``run_complete``. Text notifications use
    ``banner``, ``info``, ``warning``, ``error`` and ``progress_step``.
    """

    kind: ProgressEventKind
    """Builtin event discriminator; new kinds may be added in future releases."""
    data: ProgressEventData = field(default_factory=ProgressEventData)
    """Per-kind payload documented in BENCHMARK_API.md; tolerate added fields."""


@dataclass(frozen=True)
class BenchmarkDecision:
    """A synchronous choice, separate from progress rendering.

    Kinds: ``resume_incomplete`` (accept resumes; decline starts fresh),
    ``remeasure_complete`` (accept starts fresh; decline reuses measurements),
    and ``integration_confirmation`` (accept continues the integration).
    No callback means use ``default``. Recipe trust uses ``trust`` and is
    separate from these decisions.
    """

    kind: str
    message: str
    default: bool
    benchmark_id: str | None = None


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkOptions:
    """Inputs to :func:`sparkrun.api.benchmark`.

    Mirrors the CLI ``benchmark run`` flag set.  ``recipe`` is required;
    everything else defaults to sensible CLI-equivalent values. Data fields
    (overrides, bench_args, integrations, state_extras) accept string-keyed
    mappings of serializable values and are recursively copied at API entry.
    Invalid shapes raise SparkrunError before integration hooks or launch.
    """

    recipe: "str | Recipe"
    """Recipe name (resolved via registries) or pre-loaded ``Recipe`` object."""

    # --- Category / framework dimensions ---
    category: str | None = None
    """Benchmark category (e.g. ``"performance"``, ``"evals"``).  When
    ``None`` the framework's ``primary_category`` is used for the result."""
    framework: str | None = None
    """Override the resolved benchmarking framework (e.g. ``"llama-benchy"``)."""
    profile: str | None = None
    """Named benchmark profile to run."""
    bench_args: Mapping[str, Any] = field(default_factory=dict)
    """Extra benchmark arguments forwarded as ``-b key=value`` options."""

    # --- Targeting ---
    hosts: tuple[str, ...] | None = None
    """Explicit host list.  When set, overrides any cluster's hosts."""
    cluster: "str | ClusterDefinition | None" = None
    """Named cluster (resolved via ClusterManager) or pre-loaded definition."""
    overrides: Mapping[str, Any] = field(default_factory=dict)
    """Recipe / runtime overrides threaded into the launch.

    Fed wholesale to :func:`sparkrun.core.resolve.apply_recipe_overrides`, so
    this carries every knob ``RunOptions.overrides`` does — notably the
    parallelism keys (``tensor_parallel`` / ``pipeline_parallel`` /
    ``data_parallel``), which decide how many nodes the launch is placed on.
    ``image`` is the one entry that is not an override: it rewrites
    ``recipe.container``.  Flag-shaped spellings (``gpu_mem``) and canonical
    ones (``gpu_memory_utilization``) both resolve.
    """

    # --- Lifecycle ---
    resume: ResumeMode = ResumeMode.IF_EXISTS
    """How to handle pre-existing benchmark state."""
    skip_run: bool = False
    """Skip the inference launch; run only the benchmark phase."""
    no_stop: bool = False
    """Leave the inference workload running after benchmarking."""
    exit_on_first_fail: bool = True
    """Abort the run when the first benchmark task fails."""
    timeout: int | None = None
    """Whole-task deadline in seconds, including output draining.

    ``None`` uses the benchmark specification's timeout, then the shared
    14,400-second default. Framework request timeouts are separate bench_args.
    """
    api_key_env: str | None = None
    """Credential variable name, resolved only for execution; its value is never persisted.

    An explicit missing/empty variable is an error. Resume stores this reference
    and reacquires its current value, independently of measurement identity.
    """

    # --- Mode ---
    integrations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Selected benchmark integrations and their settings, keyed by registered name."""

    # --- Output ---
    output_file: str | None = None
    """Exact primary YAML filename; ``None`` uses the auto-generated path.

    JSON/CSV sidecars replace a .yaml/.yml suffix, or append their suffix to
    any other filename. All returned artifact paths are absolute.
    """
    export_files: bool = True
    """Export benchmark result files (CSV, JSON, YAML) alongside the run."""

    # --- Shared with RunOptions ---
    solo: bool = False
    """Force single-host mode regardless of host count."""
    dry_run: bool = False
    """Compute everything but do not execute scripts on remote hosts."""
    scheduler: str | None = None
    """Registered scheduler name.  ``None`` selects the project default."""
    rootful: bool = False
    """Run containers privileged + as root."""
    trust: bool = False
    offline: bool | None = None
    """Launch offline (:mod:`sparkrun.core.offline`); ``None`` defers to the cluster."""
    """Pre-acknowledge trust for third-party recipe hooks (``pre_exec`` /
    ``post_exec`` / ``post_commands``), mirroring ``run``'s ``--trust``.

    ``False`` does **not** mean "untrusted" — it means "no override", leaving
    :func:`sparkrun.core.launcher.resolve_recipe_trust` to auto-trust local and
    trusted-registry recipes as usual.  It matters only for a recipe that would
    otherwise be prompted for: without it a benchmark on a non-TTY (CI, or an
    agent driving the CLI) can only fail, since there is no prompt to answer."""
    sync_tuning: bool = True
    """Sync tuning configs from registries to local cache before launch."""
    extra_docker_opts: tuple[str, ...] | None = None
    """Extra arguments forwarded to the container executor (``docker run``)."""

    # --- Extension hooks ---
    progress_callback: "Callable[[ProgressEvent], None] | None" = None
    """Callback invoked with :class:`ProgressEvent` instances during the run.
    The API does not render terminal progress when no callback is supplied."""
    state_extras: Mapping[str, Any] = field(default_factory=dict)
    """Caller metadata copied into new scheduled state. Use application-owned
    names (e.g. ``{"my_app.experiment": "run-abc"}``); core-reserved keys
    are documented in ``docs/BENCHMARK_API.md``."""
    decision_callback: "Callable[[BenchmarkDecision], bool] | None" = None
    """Synchronous decision handler for AUTO resume choices and integration
    confirmations. Receives a frozen decision value, never live saved state."""


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Outputs of a completed :func:`sparkrun.api.benchmark` call.

    Shared by initial runs and resumes. ``success`` describes measurement
    completion; integration publication failures carry this result in
    ``BenchmarkIntegrationFailed.result``.
    """

    success: bool
    """``True`` when all tasks satisfy the framework's measurement policy."""
    benchmark_id: str
    """Opaque identifier for this benchmark run (framework + profile + cluster)."""
    category: str
    """Benchmark category (e.g. ``"performance"``, ``"evals"``)."""
    framework: str
    """Benchmarking framework used (e.g. ``"llama-benchy"``)."""
    profile: str | None
    """Named profile that was run; ``None`` when the default profile was used."""

    results: dict[str, Any] = field(default_factory=dict)
    """Consolidated benchmark results dict (throughput, latency, etc.)."""
    outputs: dict[str, str] = field(default_factory=dict)
    """Mapping of output file format keys to absolute paths (``"csv"``, ``"json"``, ``"yaml"``)."""

    run_result: "RunResult | None" = None
    """Structured launch result when this invocation launched inference;
    ``None`` for skip-run, resume-by-ID, and calls that did not launch."""

    cluster_id: str = ""
    """sparkrun cluster id for the inference workload used in this benchmark."""
    host_list: tuple[str, ...] = ()
    """Hosts that participated in the benchmark."""
    container_image: str = ""
    """Container image used for inference; empty for native workloads or unknown provenance."""
    container_image_sha: str | None = None
    """Digest of the pulled container image, when known."""
    container_image_sha_pinned: bool = False
    """``True`` when the container image was pinned to an explicit digest."""
    container_image_longterm_ref: str | None = None
    """Long-term archival reference for the container image (e.g. a registry URL with digest)."""
    container_image_longterm_pinned: bool = False
    """``True`` when the long-term reference was pinned to an explicit digest."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Recipe-derived and benchmark-derived metadata (framework, profile, bench_args, …)."""

    state_dir: str | None = None
    """Directory where benchmark state was persisted, when applicable."""
    measured_at: str | None = None
    """Start of the recorded measurement interval, retained through recovery."""
    completed_at: str | None = None
    """Completion of the recorded measurement interval, independent of publication."""
    resumed: bool = False
    """``True`` when this run resumed from a prior checkpoint."""
    already_complete: bool = False
    """Resume found validated completed measurements and no available integration work."""
    integration_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Integration-provided outcome summaries, keyed by integration name."""
    integration_errors: dict[str, str] = field(default_factory=dict)
    """Completion-hook failures, separate from measurement success."""


__all__ = [
    "ResumeMode",
    "ProgressEvent",
    "ProgressEventKind",
    "ProgressEventData",
    "BenchmarkDecision",
    "BenchmarkOptions",
    "BenchmarkResult",
]
