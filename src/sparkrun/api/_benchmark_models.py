"""Data shapes for the sparkrun benchmark API surface.

All public dataclasses and enums for ``sparkrun.api.benchmark`` live here.
These are stable contracts; field additions are non-breaking, field removals
are breaking.

``BenchmarkOptions`` mirrors the CLI ``benchmark`` command's flag set as a
typed struct.  ``BenchmarkResult`` carries the structured outcome of a
completed run.  ``ProgressEvent`` is a future-facing hook for step 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

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


@dataclass(frozen=True)
class ProgressEvent:
    """Event dispatched to :attr:`BenchmarkOptions.progress_callback` during a run.

    Carries a stable ``kind`` discriminator and a free-form ``data``
    payload.  Currently informational only — step 7 wires actual events
    through; until then the callback is accepted but not invoked.
    """

    kind: str
    """Discriminator string (e.g. ``"launch_started"``, ``"run_complete"``)."""
    data: dict[str, Any] = field(default_factory=dict)
    """Free-form payload; schema varies by ``kind``."""


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkOptions:
    """Inputs to :func:`sparkrun.api.benchmark`.

    Mirrors the CLI ``benchmark run`` flag set.  ``recipe`` is required;
    everything else defaults to sensible CLI-equivalent values.
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
    bench_args: dict[str, Any] = field(default_factory=dict)
    """Extra benchmark arguments forwarded as ``-b key=value`` options."""

    # --- Targeting ---
    hosts: tuple[str, ...] | None = None
    """Explicit host list.  When set, overrides any cluster's hosts."""
    cluster: "str | ClusterDefinition | None" = None
    """Named cluster (resolved via ClusterManager) or pre-loaded definition."""
    overrides: dict[str, Any] = field(default_factory=dict)
    """Recipe / runtime overrides threaded into the launch (``image``, ``port``, …)."""

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
    """Per-task benchmark timeout in seconds.  ``None`` uses the framework default."""
    api_key_env: str | None = None
    """Environment variable name whose value is used as the inference API key."""

    # --- Mode ---
    arena: bool = False
    """Submit results to the Spark Arena leaderboard."""

    # --- Output ---
    output_file: str | None = None
    """Explicit output file base path.  ``None`` uses the auto-generated path."""
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
    sync_tuning: bool = True
    """Sync tuning configs from registries to local cache before launch."""
    extra_docker_opts: tuple[str, ...] | None = None
    """Extra arguments forwarded to the container executor (``docker run``)."""

    # --- Extension hooks ---
    progress_callback: "Callable[[ProgressEvent], None] | None" = None
    """Callback invoked with :class:`ProgressEvent` instances during the run.
    Accepted but not yet wired — step 7 activates this."""
    state_extras: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extras forwarded into the benchmark state (e.g.
    ``{"submission_id": "sub-abc"}`` for arena runs)."""
    on_prompt_required: "Callable[[Any], bool] | None" = None
    """Callback invoked when the CLI would show a confirmation prompt.
    Return ``True`` to accept, ``False`` to cancel.  Step 4 wires this."""


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Outputs of a completed :func:`sparkrun.api.benchmark` call.

    Populated best-effort from the internal CLI ``BenchmarkResult``.
    Step 7 will plumb additional fields once orchestration moves into
    this module.
    """

    success: bool
    """``True`` when all benchmark tasks completed without error."""
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
    """Structured launch result.  Populated by step 7; ``None`` until then."""

    cluster_id: str = ""
    """sparkrun cluster id for the inference workload used in this benchmark."""
    host_list: tuple[str, ...] = ()
    """Hosts that participated in the benchmark."""
    container_image: str = ""
    """Container image used for the inference workload."""
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
    resumed: bool = False
    """``True`` when this run resumed from a prior checkpoint."""
    submission_id: str | None = None
    """Arena submission id, when the run was submitted to the leaderboard."""


__all__ = [
    "ResumeMode",
    "ProgressEvent",
    "BenchmarkOptions",
    "BenchmarkResult",
]
