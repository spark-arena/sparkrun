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
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterStatusResult
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

    strategy_options: dict[str, Any] = field(default_factory=dict)
    """Run-scoped inputs for the recipe-selected execution strategy.

    These control *execution*, not workload identity, so they are deliberately
    absent from recipe fingerprints and intent IDs — the same rule that keeps
    serve flags out of `generate_intent_id`. A plugin CLI uses this for
    imperative, per-invocation choices such as a restore provider or an
    artifact path.
    """

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
    """If True, skip the launch when this workload is already serving.

    "Already serving" is matched on the launch **intent** (recipe +
    parallelism + port), not on a cluster_id — so it holds regardless of which
    hosts the running deployment landed on or which scheduler placed it (see
    :func:`sparkrun.api.find_running_intent`).  On a hit, ``run`` returns a
    :class:`RunResult` with ``already_running=True`` describing the
    *pre-existing* deployment.  A cluster that can't be queried counts as "not
    running" — refusing to launch because a status probe failed is the worse
    outcome."""
    owner: str | None = None
    """Opaque tag naming the component launching this workload, persisted into
    job metadata.  Lets an automated supervisor distinguish jobs it created
    from identically-configured ones a human started — and so refuse to tear
    the latter down.  ``None`` (the CLI default) records no owner."""

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
    runtime_cache: bool | None = None
    """Toggle the persistent compilation/autotune cache for this launch.

    ``None`` (default) defers to the recipe / cluster / config / runtime
    chain — see
    :func:`sparkrun.core.runtime_cache.resolve_runtime_cache_settings`.
    ``True`` / ``False`` force it on or off, outranking every layer except
    the recipe's own ``runtime_cache:`` block and the
    ``SPARKRUN_NO_RUNTIME_CACHE`` kill switch."""

    # Networking / runtime ports.
    port: int | None = None
    """Override the inference serve port."""
    ray_port: int = 46379
    """Ray GCS port (vllm-ray runtime)."""
    dashboard_port: int = 8265
    """Ray dashboard port."""
    dashboard: bool | None = None
    """Enable the Ray dashboard on the head node (Ray runtimes only).

    Tri-state: ``True``/``False`` force the toggle; ``None`` (default) defers to
    the recipe's ``runtime_config.dashboard``, falling back to on. When off, the
    runtime emits ``--include-dashboard=False`` to override Ray's on-by-default."""
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
    preserve_model_perms: bool | None = None
    """Model-rsync permission preservation override.  ``None`` → derive from
    the resolved cluster's ``distribution.model.preserve_perms``; an explicit
    bool wins (used by the benchmark flow, which launches with explicit hosts
    and loses the named-cluster identity)."""
    skip_model_fan_out: bool | None = None
    """Skip the per-host model rsync fan-out (shared cache).  ``None`` → derive
    from the cluster's ``distribution.model.skip_fan_out``; explicit bool wins."""
    recipe_ref: str | None = None
    """Simplified recipe reference for display (e.g. ``@spark-arena/UUID``)."""


@dataclass(frozen=True)
class RunPlan:
    """Everything :func:`sparkrun.api.run` decides *before* it touches the cluster.

    Produced by :func:`sparkrun.api.plan` and consumed by
    :func:`sparkrun.api.run` via its ``plan=`` argument.  The split exists
    because placement is not free and must not be computed twice: a caller
    that needs to *show* the target hosts before launching (the CLI banner,
    the desktop app's pre-launch preview) would otherwise have to run the
    scheduler itself, narrow the host list, and hand the survivors to
    ``run`` — which then re-schedules over that narrowed set and can no
    longer reach the hosts the first pass discarded.  Planning once and
    threading the result removes that whole failure mode: what is displayed
    and what is launched are the same object.

    Building a plan performs the cluster-facing work that placement needs —
    transport preparation and one occupancy query — but changes no cluster
    state.  It is safe to build a plan and never launch it.
    """

    recipe: "Recipe"
    """Resolved recipe (runtime selection finalized, overrides applied)."""
    runtime: Any
    """Resolved :class:`~sparkrun.runtimes.base.RuntimePlugin`."""
    cluster: "ClusterDefinition"
    """Resolved cluster, after transport preparation.  For provider-backed
    transports this carries the refreshed connection details, so ``run``
    must reuse it rather than preparing again."""

    candidate_hosts: tuple[str, ...]
    """Every host placement was allowed to choose from.

    Distinct from :attr:`host_list`, and both are load-bearing: the
    deterministic (greedy) placement token and the superseded-deployment
    sweep are derived from the *candidates*, so that ``stop`` / ``status``
    — which only know the cluster's full host list — compute the same
    cluster_id the launch used."""
    host_list: tuple[str, ...]
    """Hosts the workload will actually run on (the scheduler's
    ``hosts_used``, after solo / ``max_nodes`` constraints)."""
    is_solo: bool
    """``True`` when the launch resolved to single-host mode."""
    placement: "RankAssignment | None"
    """Concrete rank → (host, GPU) assignment.  ``None`` in solo mode or
    when the scheduler was bypassed (single host / no parallelism)."""
    notes: tuple[str, ...] = ()
    """Human-readable placement notes (e.g. ``"Note: 2 nodes required,
    using 2 of 4 hosts"``).  The library never prints; renderers echo these
    verbatim."""

    scheduler_selector: str | None = None
    """Scheduler name as selected (CLI → recipe → cluster).  ``None`` means
    nothing in the chain named one and the default applies."""
    scheduler: str = ""
    """Resolved scheduler name — what actually produced :attr:`placement`."""
    scheduler_defaulted: bool = False
    """``True`` when :attr:`scheduler_selector` was ``None``, so callers can
    surface the "consider occupancy-aware placement" hint."""

    intent_id: str = ""
    """Deterministic hex identifier for (recipe + parallelism + port)."""
    placement_token: str = ""
    """Token disambiguating this launch from other instances of the intent.
    Derived from :attr:`candidate_hosts` under a deterministic scheduler,
    random under a status-aware one."""
    cluster_id: str = ""
    """``sparkrun_<intent_id>_<placement_token>`` — the id the launch will
    use, so a renderer can show it (and ``--ensure`` can look it up) before
    anything starts."""
    recipe_fingerprint: str = ""
    """Serve-configuration digest of the *declared* recipe.

    Decided here for the same reason :attr:`intent_id` is: ``launch_inference``
    folds host-dependent platform runtime-flag defaults into ``recipe.defaults``
    before it persists job metadata, so a digest taken after that point varies
    with the hardware the job landed on and no caller could reproduce it."""


@dataclass(frozen=True)
class ResolvedMount:
    """One host-to-container mount in a materialized launch."""

    source: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class ResolvedLaunchUnit:
    """One container/process-tree boundary in a materialized launch."""

    id: str
    index: int
    host: str
    devices: tuple[str, ...]
    image: str
    image_digest: str
    command: tuple[str, ...]
    environment: dict[str, str]
    mounts: tuple[ResolvedMount, ...]


@dataclass(frozen=True)
class ResolvedWorker:
    """One accelerator-owning engine worker inside a launch unit."""

    id: str
    unit: str
    service: str
    process_slot: int
    device_slots: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedProcessGroup:
    """An ordered, runtime-owned rank namespace."""

    id: str
    kind: str
    service: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedServiceDomain:
    """An independently addressable engine or cooperating service role."""

    id: str
    role: str
    workers: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedAdapterTopology:
    """Opaque runtime topology identity guarded by a canonical digest."""

    schema: str
    digest: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ResolvedExecutionGraph:
    """Portable worker/group topology independent of physical placement."""

    workers: tuple[ResolvedWorker, ...]
    groups: tuple[ResolvedProcessGroup, ...]
    services: tuple[ResolvedServiceDomain, ...]
    adapter: ResolvedAdapterTopology


@dataclass(frozen=True)
class ResolvedLaunchSpec:
    """Serializable pre-launch profile for an execution-strategy integration.

    Materialization is read-only and does not launch, distribute, or mutate
    cluster state. Container images must be digest-pinned in the recipe for
    ``image_digest`` to be populated without a remote probe.
    """

    format: int
    kind: str
    recipe: str
    cluster_id: str
    runtime: str
    engine: str
    model: str
    model_revision: str
    world_size: int
    tensor_parallel: int
    node_count: int
    cache_dir: str
    units: tuple[ResolvedLaunchUnit, ...]
    execution: ResolvedExecutionGraph


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
    timeline: Any = None
    """Live :class:`~sparkrun.core.timing.Timeline` of launch-stage spans.

    Deliberately the collector rather than an exported snapshot: the
    readiness wait runs *after* ``run`` returns (in ``post_launch_lifecycle``
    or the caller's own wait) and records onto this same object, so a
    snapshot taken here would always be missing the containers-running →
    serving figure.  Call ``.export()`` once you are done waiting.

    ``None`` only when the intent was already running (nothing launched)."""
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
    already_running: bool = False
    """``True`` when :attr:`RunOptions.ensure` found this intent already
    serving, so nothing was launched.  :attr:`cluster_id` / :attr:`host_list`
    then describe the **pre-existing** deployment, and
    :attr:`launch_result` is ``None`` (there was no launch)."""
    recipe_fingerprint: str = ""
    """Serve-configuration digest persisted into this job's metadata.

    Derived from the *declared* recipe before the launcher folds in
    host-dependent platform runtime-flag defaults, so a caller can reproduce it
    without probing hardware — and so matching a job by fingerprint after the
    fact actually works."""


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
    """Count of containers/processes actually removed, as reported by the
    teardown itself — not an assumption of one per host."""
    errors: tuple[str, ...] = ()
    """Human-readable error messages for any hosts that failed."""
    hosts_failed: tuple[str, ...] = ()
    """Hosts whose teardown did not confirm.  Non-empty means containers
    may still be running (and holding VRAM); job metadata is deliberately
    retained so the workload can still be found and stopped."""

    @property
    def success(self) -> bool:
        """True when every targeted host confirmed teardown."""
        return not self.hosts_failed and not self.errors


@dataclass(frozen=True)
class StopAllResult:
    """Outputs of :func:`sparkrun.api.stop_all`."""

    discovered: "ClusterStatusResult"
    """The discovery snapshot the teardown acted on (also what the CLI
    renders), so callers don't have to re-query to describe what was
    found."""
    jobs_stopped: int
    """Discovered jobs (cluster groups + solo entries) whose containers
    all confirmed teardown."""
    containers_removed: int
    """Containers actually removed, summed across hosts."""
    hosts_stopped: tuple[str, ...] = ()
    """Hosts whose teardown confirmed."""
    hosts_failed: dict[str, str] = field(default_factory=dict)
    """Host → error for teardowns that did not confirm."""
    discovery_errors: dict[str, str] = field(default_factory=dict)
    """Host → error for hosts that could not be queried at all.  These are
    *not* "nothing to stop" — an unqueryable host may be running
    containers we never saw."""

    @property
    def success(self) -> bool:
        """True when every host was queried and every teardown confirmed."""
        return not self.hosts_failed and not self.discovery_errors


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------

# ``LogLine`` is defined in ``core.log_source`` alongside ``LogSource`` so
# that ``orchestration.logs`` can produce it without importing ``api``
# (layering: cli → api → {core, orchestration}).  Re-exported here because
# ``sparkrun.api.LogLine`` is the stable public path.
from sparkrun.core.log_source import LogLine, LogSource  # noqa: E402


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


@dataclass(frozen=True)
class RecipeSummary:
    """A single entry returned by :func:`sparkrun.api.search_recipes`.

    A cheap projection of a recipe YAML — enough to render a catalog row
    without paying for version migration, resolver chains, or env
    expansion (see ``sparkrun.core.recipe.recipe_summary``).  Load the
    real thing with :func:`sparkrun.api.resolve_recipe` once the user has
    picked one.

    Fields beyond the ones modelled here are exposed verbatim under
    :attr:`metadata`, and :meth:`to_dict` returns that mapping — the shape
    the CLI formatters and JSON output consume.
    """

    name: str
    """Qualified name — ``@registry/stem`` for a registry recipe, the bare
    file stem for one discovered in the working directory."""
    file: str
    """File stem, i.e. the name that can be typed unqualified."""
    path: str
    model: str = ""
    runtime: str = ""
    description: str = ""
    min_nodes: int = 1
    registry: str | None = None
    """Owning registry, or ``None`` for a working-directory recipe."""
    builder: str | None = None
    tensor_parallel: int | None = None
    """``defaults.tensor_parallel``, or ``None`` when the recipe leaves it
    to the runtime."""
    gpu_memory_utilization: float | None = None
    """``defaults.gpu_memory_utilization``, or ``None`` when unset."""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_summary(cls, entry: dict[str, Any]) -> "RecipeSummary":
        """Build from a ``core.recipe.recipe_summary`` mapping."""
        return cls(
            name=str(entry.get("name", "")),
            file=str(entry.get("file", "")),
            path=str(entry.get("path", "")),
            model=str(entry.get("model", "")),
            runtime=str(entry.get("runtime", "")),
            description=str(entry.get("description", "")),
            min_nodes=_as_int(entry.get("min_nodes"), default=1) or 1,
            registry=entry.get("registry") or None,
            builder=entry.get("builder") or None,
            tensor_parallel=_as_int(entry.get("tp")),
            gpu_memory_utilization=_as_float(entry.get("gpu_mem")),
            metadata=dict(entry),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying recipe-summary mapping."""
        return dict(self.metadata)


def _as_int(value: Any, *, default: int | None = None) -> int | None:
    """Coerce a recipe default to int, or *default* when absent/unparseable."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    """Coerce a recipe default to float, or *default* when absent/unparseable."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "RunOptions",
    "RunResult",
    "StopResult",
    "LogLine",
    "LogSource",
    "JobInfo",
    "RecipeSummary",
]
