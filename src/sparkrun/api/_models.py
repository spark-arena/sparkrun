"""Data shapes for the sparkrun library API.

All public dataclasses returned by ``sparkrun.api.*`` functions live
here.  These are stable contracts that third-party Python callers may
depend on; field additions are non-breaking, field removals are
breaking.

The ``RunOptions`` dataclass mirrors the CLI ``run`` command's flag
set as a typed struct so callers can construct it programmatically
without parsing CLI strings.  Other ``Options`` dataclasses follow
the same pattern for ``stop``, ``logs``, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.scheduler import RankAssignment


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    """Inputs to :func:`sparkrun.api.run`.

    Mirrors the CLI ``run`` flag set.  ``recipe`` and one of
    ``hosts`` / ``cluster`` are required; everything else has sensible
    defaults that match the CLI defaults.
    """

    recipe: "str | Recipe"
    """Recipe name (resolved via registries) or pre-loaded ``Recipe`` object."""

    hosts: tuple[str, ...] | None = None
    """Explicit host list.  When set, overrides any cluster's hosts."""

    cluster: "str | ClusterDefinition | None" = None
    """Named cluster (resolved via ClusterManager) or pre-loaded definition."""

    overrides: dict[str, Any] = field(default_factory=dict)
    """Recipe / runtime overrides (tensor_parallel, port, gpu_memory_utilization, …)."""

    # Mode / lifecycle knobs.
    solo: bool = False
    """Force single-host mode regardless of host count."""
    dry_run: bool = False
    """Compute everything but don't execute scripts on remote hosts."""
    follow: bool = True
    """Stream container logs after launch (CLI default)."""
    detached: bool = True
    """Launch container detached (CLI default; inverse of --foreground)."""
    trust: bool | None = None
    """Pre-acknowledge trust for third-party recipe hooks.  ``None`` =
    prompt interactively (CLI default), ``True`` = auto-trust,
    ``False`` = refuse to run untrusted hooks."""
    ensure: bool = False
    """If True, return existing RunResult when an identical job is already
    running rather than launching a duplicate."""

    # Scheduler selection.
    scheduler: str | None = None
    """Registered scheduler name (e.g. ``"greedy"``).  ``None`` selects
    the default (greedy)."""

    # Distribution / networking.
    transfer_mode: str | None = None
    """Override the cluster's transfer mode (``auto`` / ``local`` /
    ``push`` / ``delegated``)."""
    cache_dir: str | None = None
    """Override the remote HuggingFace cache dir on target hosts."""

    # Networking / runtime ports.
    port: int | None = None
    """Override the inference serve port."""
    ray_port: int = 46379
    """Ray GCS port (vllm-ray runtime)."""
    dashboard_port: int = 8265
    """Ray dashboard port."""
    dashboard: bool = False
    """Enable Ray dashboard on head node."""
    init_port: int = 25000
    """vLLM/SGLang distributed init port."""

    # Executor knobs (forwarded to ``resolve_executor`` as cli_overrides).
    executor: str | None = None
    """Override the resolved executor selector (``docker`` / ``local`` / ``k8s``)."""
    executor_config: dict[str, Any] | None = None
    """Executor option overrides (``shm_size``, ``memory_limit``, ``privileged``, …)."""
    rootful: bool = False
    """Run docker containers privileged + as root (disables rootless adjustments)."""

    # Diagnostics / introspection.
    diagnostics_path: str | None = None
    """Path to write run-time diagnostics NDJSON.  ``None`` disables."""

    # Additional launcher passthroughs (CLI-shaped knobs threaded into
    # ``launch_inference`` for parity with the existing CLI command).
    cluster_id_override: str | None = None
    """Override the deterministic cluster ID (static container name)."""
    transfer_interface: str | None = None
    """Network interface used for resource transfers (e.g. ``cx7`` / ``mgmt``)."""
    local_cache_dir: str | None = None
    """Control-machine cache dir for downloads (defaults to the same as ``cache_dir``)."""
    sync_tuning: bool = True
    """Sync tuning configs from registries to local cache before launch."""
    extra_docker_opts: tuple[str, ...] | None = None
    """Extra arguments threaded through to the container executor (``docker run``)."""
    topology: str | None = None
    """Cluster topology hint (carried through to the runtime)."""
    recipe_ref: str | None = None
    """Simplified recipe reference for display (e.g. ``@spark-arena/UUID``)."""


@dataclass(frozen=True)
class RunResult:
    """Outputs of a successful :func:`sparkrun.api.run`."""

    cluster_id: str
    host_list: tuple[str, ...]
    """Hosts actually used (after scheduling / solo / max_nodes constraints)."""
    placement: "RankAssignment | None"
    """Concrete rank → (host, gpu) assignment.  ``None`` in solo mode
    or when parallelism is unset (single-rank job)."""
    scheduler: str
    """Name of the scheduler that produced :attr:`placement`."""
    runtime: str
    """Runtime family name (``vllm-ray`` / ``sglang`` / …)."""
    executor: str
    """Resolved executor name (``docker`` / ``local`` / ``k8s``)."""
    started_at: float
    """Epoch seconds when the launch began."""
    dry_run: bool
    """``True`` when the launch was a dry-run — no remote state changed."""
    is_solo: bool
    """``True`` when the launch ran in solo (single-host) mode."""
    rc: int = 0
    """Process return code reported by the runtime (``0`` on success)."""
    serve_command: str = ""
    """Effective serve command rendered by the runtime."""
    container_image: str = ""
    """Container image actually used for the launch."""
    serve_port: int = 0
    """Inference HTTP port the workload listens on."""
    effective_cache_dir: str = ""
    """Resolved HuggingFace cache directory on the launch target."""
    runtime_info: dict[str, str] = field(default_factory=dict)
    """Runtime-reported version strings (engine, framework, model server)."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Recipe-derived metadata (recipe qualified_name, model, image, …)."""
    launch_result: Any = None
    """Opaque handle to the underlying :class:`LaunchResult` for callers
    that need the raw orchestration object (CLI ``post_launch_lifecycle``,
    crash diagnostics).  External callers should treat this as private."""
    intent_id: str = ""
    """Deterministic hex prefix of :attr:`cluster_id`.  Same value
    across every run of the same recipe + parallelism + port — useful
    for status / stop / logs discovery without re-running the
    scheduler.  Empty string only when the caller supplied a
    non-canonical ``cluster_id_override``."""
    placement_token: str = ""
    """Random hex token disambiguating this specific launch from other
    instances of the same intent.  Empty string only when the caller
    supplied a non-canonical ``cluster_id_override``."""


# --------------------------------------------------------------------------
# Stop
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StopResult:
    """Outputs of :func:`sparkrun.api.stop`."""

    cluster_id: str
    hosts_targeted: tuple[str, ...]
    """Hosts the stop command was issued against."""
    containers_removed: int
    """Total count of containers/processes successfully stopped."""
    errors: tuple[str, ...] = ()
    """Human-readable error messages for any hosts that failed."""


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LogLine:
    """A single line yielded by :func:`sparkrun.api.logs`."""

    host: str
    container: str
    text: str
    stream: str = "stdout"
    """``"stdout"`` or ``"stderr"`` — best-effort, may be ``"stdout"`` if
    the executor doesn't preserve stream identity."""
    timestamp: float | None = None
    """Epoch seconds parsed from the log line, when available."""


# --------------------------------------------------------------------------
# Job listing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JobInfo:
    """A single entry returned by :func:`sparkrun.api.list_jobs`.

    Reflects the on-disk job metadata schema in
    ``~/.cache/sparkrun/jobs/``.  Fields beyond the canonical
    cluster_id / recipe / hosts are exposed verbatim under
    :attr:`metadata` for callers that need them.
    """

    cluster_id: str
    recipe: str | None = None
    runtime: str | None = None
    hosts: tuple[str, ...] = ()
    started_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    intent_id: str | None = None
    """Deterministic hex prefix of the cluster_id.  ``None`` indicates
    a job metadata file whose contents do not parse as a canonical
    sparkrun cluster_id (corrupted YAML, hand-edited, or written by an
    incompatible tool)."""
    placement_token: str | None = None
    """Random hex suffix unique to this launch.  ``None`` indicates a
    job metadata file whose contents do not parse as a canonical
    sparkrun cluster_id (data-quality issue)."""


__all__ = [
    "RunOptions",
    "RunResult",
    "StopResult",
    "LogLine",
    "JobInfo",
]
