"""``sparkrun.api.benchmark`` — public Python entry point for benchmark runs.

Library callers receive structured results, progress, and decisions.
Terminal presentation is supplied by the CLI adapter.
"""

from __future__ import annotations

from sparkrun.benchmarking._measurement import (
    capture_launch_context,
    persist_measurement_context,
    restore_measurement_context as _restore_measurement_context,
)

from sparkrun.core.application_profile import remote_cache_path
from sparkrun.benchmarking._credentials import BenchmarkCredentials, resolve_credentials
from sparkrun.benchmarking.base import resolve_request_model
from sparkrun.core.recipe import resolve_served_model_name
from sparkrun.benchmarking.metadata import public_benchmark_data, benchmark_recipe_fingerprint
from sparkrun.benchmarking.run_state import DEFAULT_BENCHMARK_TIMEOUT

from collections.abc import Mapping
import contextlib
import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Unpack

from sparkrun.api._benchmark_models import (
    BenchmarkDecision,
    BenchmarkOptions,
    BenchmarkResult,
    ProgressEvent,
    ProgressEventData,
    ProgressEventKind,
    ResumeMode,
)
from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import BenchmarkFailed, BenchmarkFinalizationFailed, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.benchmarking.base import BenchmarkExecution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress emitter abstraction
# ---------------------------------------------------------------------------


class _ProgressEmitter:
    """Side-channel for orchestration to emit text/structured events without
    coupling to click/CLI rendering. CLI provides a private adapter; public
    callers supply progress_callback and decision_callback.
    """

    def __init__(self, decision_callback: Callable[[BenchmarkDecision], bool] | None = None) -> None:
        self._decision_callback = decision_callback

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Request an integration decision, using its explicit default if absent."""
        request = BenchmarkDecision("integration_confirmation", message, default)
        return bool(self._decision_callback(request)) if self._decision_callback else default

    @contextlib.contextmanager
    def schedule_progress(self, *, total_tasks, benchmark_id, fw, title):
        progress = _EventScheduleProgress(self, benchmark_id, total_tasks)
        progress.emit("schedule_started", title=title)
        try:
            yield progress
        finally:
            progress.emit("schedule_finished")

    def banner(self, line: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass

    def progress_step(self, step_idx: int, total: int, label: str) -> None:
        pass

    def event(self, ev: ProgressEvent) -> None:
        pass

    def on_recipe_resolved(self, recipe, overrides: dict, *, local_cache_dir: str | None = None) -> None:
        """Hook fired once after the recipe is loaded and overrides applied.

        CLI overrides this to render a VRAM estimate; library callers no-op.
        Lives on the emitter so the orchestration loads the recipe exactly
        once.
        """
        pass


class _NullProgressEmitter(_ProgressEmitter):
    """Silent adapter inheriting the base event and decision behavior."""


class _CallbackProgressEmitter(_ProgressEmitter):
    """For API callers that pass progress_callback."""

    def __init__(self, callback: Callable[[ProgressEvent], None], decision_callback=None) -> None:
        super().__init__(decision_callback)
        self._cb = callback

    def banner(self, line: str) -> None:
        self._cb(ProgressEvent(kind="banner", data={"line": line}))

    def info(self, msg: str) -> None:
        self._cb(ProgressEvent(kind="info", data={"msg": msg}))

    def warning(self, msg: str) -> None:
        self._cb(ProgressEvent(kind="warning", data={"msg": msg}))

    def error(self, msg: str) -> None:
        self._cb(ProgressEvent(kind="error", data={"msg": msg}))

    def progress_step(self, step_idx: int, total: int, label: str) -> None:
        self._cb(ProgressEvent(kind="progress_step", data={"step": step_idx, "total": total, "label": label}))

    def event(self, ev: ProgressEvent) -> None:
        self._cb(ev)


class _EventScheduleProgress:
    """Translate scheduler notifications into the public callback vocabulary."""

    def __init__(self, emitter, benchmark_id: str, total_tasks: int):
        self.emitter, self.benchmark_id, self.total_tasks = emitter, benchmark_id, total_tasks

    def emit(self, kind: ProgressEventKind, **data: Unpack[ProgressEventData]) -> None:
        payload: ProgressEventData = {"benchmark_id": self.benchmark_id, "total_tasks": self.total_tasks}
        payload.update(data)
        self.emitter.event(ProgressEvent(kind, payload))

    def start_task(self, idx: int, label: str) -> None:
        self.emit("task_start", index=idx, label=label)

    def end_task(self, idx: int, success: bool, duration_s: float | None = None) -> None:
        self.emit("task_end", index=idx, success=success, duration_s=duration_s)

    def update_results_table(self, consolidated: dict[str, Any]) -> None:
        from copy import deepcopy

        self.emit("results_update", results=deepcopy(consolidated))

    def log(self, message: str) -> None:
        self.emit("schedule_log", message=message)


# ---------------------------------------------------------------------------
# Internal helpers shared with cli/_benchmark.py
# ---------------------------------------------------------------------------


def _benchmark_title(recipe_name: str, profile: str | None) -> str:
    """Return the recipe/profile title used by the progress UI."""
    return "%s/%s" % (recipe_name, profile) if profile else recipe_name


def _write_consolidated(state_dir: Path, consolidated: dict[str, Any]) -> Path:
    """Write the consolidated dict to ``<state_dir>/consolidated.json`` and return the path."""
    state_dir.mkdir(parents=True, exist_ok=True)
    p = state_dir / "consolidated.json"
    p.write_text(json.dumps(consolidated, indent=2))
    return p


def _should_remeasure_complete_state(
    resume_mode: "ResumeMode",
    decision_callback: "Callable[[BenchmarkDecision], bool] | None",
    existing_state: Any,
) -> bool:
    """Whether COMPLETE prior state should be discarded and re-measured.

    "Resuming" COMPLETE state runs zero tasks and re-emits the previous run's
    results into the new output — indistinguishable from a real measurement, so
    it must never happen silently (the caller warns on the reuse path).

    ``ResumeMode.AUTO`` delegates the choice to *decision_callback* (the CLI
    wires an interactive confirm); with no callback the library default is
    reuse, matching prior behaviour.  ``IF_EXISTS`` / ``REQUIRED`` asked for a
    resume explicitly, so they always reuse.  (``FRESH`` never reaches here —
    it deletes the state before this decision.)
    """
    if resume_mode != ResumeMode.AUTO or decision_callback is None:
        return False
    return bool(
        decision_callback(
            BenchmarkDecision(
                "remeasure_complete",
                "Found COMPLETE benchmark state. Delete and re-measure?",
                False,
                existing_state.benchmark_id,
            )
        )
    )


def _resolve_running_deployment(
    recipe,
    overrides: dict,
    candidate_hosts: list[str],
    *,
    solo: bool,
    cluster: ClusterDefinition | None,
    sctx: "SparkrunContext | None",
    emitter: _ProgressEmitter,
) -> tuple[list[str], bool, str | None]:
    """Locate the deployment ``--skip-run`` is meant to benchmark.

    ``--skip-run`` is the one benchmark path that does not launch, so it is
    also the one that must not *place*: the workload is already serving
    somewhere, and the question is where — not where it would go.  This is the
    same question ``--ensure`` asks, so it uses the same key
    (:func:`~sparkrun.api.find_running_intent`, keyed on the launch intent
    rather than a cluster_id, which also encodes placement and so cannot match
    a job scheduled under an ``occupancy-*`` scheduler).

    Without this the branch simply kept the whole resolved cluster, which made
    ``benchmark --skip-run`` report ``cluster (4 nodes)`` for a solo workload
    and — worse than cosmetically — pointed ``head_host`` at
    ``candidate_hosts[0]`` and recorded every candidate in the exported
    results.  A ``tp: 1`` recipe benchmarked on one node was published as a
    four-node measurement, and the run only reached the right server when the
    workload happened to land on the first host in the list.

    The intent's own hosts also carry the running deployment's **cluster_id**,
    which is returned so the benchmark binds to the job that exists instead of
    to one derived from a host set that was never launched.  Benchmark
    *identity* is unaffected: :func:`derive_benchmark_id` hashes only the
    intent half of a cluster_id, so prior state stays resumable.

    Falls back to the previous behaviour (candidates, narrowed by ``solo``)
    with a warning when nothing matches — an unreachable cluster or a
    hand-started server is "couldn't tell", not "not running", and refusing to
    benchmark on that basis is the worse failure.  Pass the **full** candidate
    list to the lookup: a deployment that landed on a host this benchmark
    would not have chosen still counts.
    """
    import sparkrun.api as api
    from sparkrun.orchestration.job_metadata import generate_intent_id

    fallback = list(candidate_hosts)
    fallback_solo = bool(solo) or recipe.mode == "solo" or len(fallback) <= 1
    if fallback_solo and len(fallback) > 1:
        fallback = fallback[:1]

    try:
        intent_id = generate_intent_id(recipe, overrides)
        match = api.find_running_intent(intent_id, list(candidate_hosts), cluster=cluster, sctx=sctx)
    except Exception:
        logger.debug("--skip-run: running-intent lookup failed", exc_info=True)
        match = None

    if match is None or not match.hosts:
        emitter.warning(
            "--skip-run: no running workload matched this recipe on %s; assuming %s" % (", ".join(candidate_hosts), ", ".join(fallback))
        )
        return fallback, fallback_solo, None

    hosts = list(match.hosts)
    if match.other_cluster_ids:
        # find_running_intent already picked the widest deployment; say so
        # rather than silently benchmarking one of several.
        emitter.warning(
            "--skip-run: %d deployments of this workload are running; benchmarking %s"
            % (len(match.other_cluster_ids) + 1, match.cluster_id)
        )
    return hosts, len(hosts) <= 1, match.cluster_id


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def _validated_benchmark_options(options: BenchmarkOptions) -> BenchmarkOptions:
    """Copy data inputs before hooks without coercing arbitrary containers."""
    from sparkrun.utils.data import normalize_data

    values = {}
    for name in ("overrides", "bench_args", "integrations", "state_extras"):
        value = getattr(options, name)
        if not isinstance(value, Mapping):
            raise TypeError("BenchmarkOptions.%s must be a mapping" % name)
        if name == "integrations":
            for integration, settings in value.items():
                if not isinstance(settings, Mapping):
                    raise TypeError("BenchmarkOptions.integrations.%s must be a mapping" % integration)
        values[name] = normalize_data(value, path="BenchmarkOptions." + name)
    return dataclasses.replace(options, **values)


def _require_host_endpoint(*, recipe, cluster, sctx, executor_overrides=None):
    """Reject control-plane-only executors before launch or endpoint probing."""
    from sparkrun.orchestration.executor import get_executor, resolve_executor_name
    from sparkrun.core.bootstrap import get_runtime

    name = resolve_executor_name(
        recipe=recipe,
        runtime=get_runtime(recipe.runtime, sctx.variables),
        cluster=cluster,
        cli_overrides=executor_overrides,
        config=sctx.config,
        v=sctx.variables,
    )
    if not get_executor(name, sctx.variables).supports_host_endpoint:
        raise BenchmarkFailed(
            "Executor %r does not provide a reachable inference endpoint for benchmarking; "
            "control-plane hosts cannot be used as serving endpoints." % name
        )


def _check_measurement_prerequisites(fw, emitter: _ProgressEmitter) -> None:
    missing = fw.check_prerequisites()
    if missing:
        for message in missing:
            emitter.error("Error: %s" % message)
        raise BenchmarkFailed("Benchmark prerequisites not met: " + "; ".join(missing), exit_code=1)


def _execute_benchmark(
    options: BenchmarkOptions,
    *,
    sctx: "SparkrunContext",
    emitter: _ProgressEmitter,
) -> BenchmarkExecution:
    """Execute the full benchmark flow: launch inference -> benchmark -> stop.

    Returns a ``sparkrun.benchmarking.base.BenchmarkExecution`` (internal type)
    on success.  All sys.exit() paths have been converted to typed exceptions;
    KeyboardInterrupt is re-raised after state is preserved.

    Args:
        options:  Fully-resolved ``BenchmarkOptions`` from the API surface.
        sctx:     Shared ``SparkrunContext`` (variables + config).
        emitter:  Side-channel for progress/banner output.  Pass
                  ``_NullProgressEmitter()`` for headless execution.

    Raises:
        BenchmarkFailed: Any non-zero exit path in the benchmark flow.
        NoResumableState: ``ResumeMode.REQUIRED`` with no existing state.
        FrameworkCategoryMismatch: Pinned framework not in the pinned category.
        CategoryNotFound / AmbiguousCategoryError: Category resolution failure.
        KeyboardInterrupt: Re-raised after state is preserved (Ctrl+C).
    """
    import sparkrun.api as api
    from sparkrun.benchmarking.base import startup_timing_metadata, BenchmarkExecution
    from sparkrun.core.benchmark_profiles import BenchmarkSpec
    from sparkrun.core.bootstrap import get_runtime, get_benchmarking_framework
    from sparkrun.utils import is_local_host
    from sparkrun.core.launcher import wait_for_endpoint_ready, wait_for_serve_ready
    from sparkrun.core.readiness import resolve_readiness_settings
    from sparkrun.orchestration.primitives import (
        build_ssh_kwargs,
        detect_host_ip,
    )
    from sparkrun.core.recipe import (
        expand_recipe_shortcut as _expand_recipe_shortcut,
        is_recipe_url as _is_recipe_url,
        simplify_recipe_ref as _simplify_recipe_ref,
    )
    from sparkrun.api._resolve import resolve_cluster, prepare_transport
    from sparkrun.core.cluster_manager import ResolvedClusterConfig
    from sparkrun.core.resolve import apply_recipe_overrides as _apply_recipe_overrides, load_recipe as _load_recipe
    from sparkrun.api._errors import (
        NoResumableState,
        FrameworkCategoryMismatch,
        AmbiguousCategoryError as _AmbiguousApi,
        CategoryNotFound as _CatNotFoundApi,
    )

    from sparkrun.core.benchmark_integrations import BenchmarkIntegrationSession

    options = _validated_benchmark_options(options)
    integrations = BenchmarkIntegrationSession(options, sctx=sctx, emitter=emitter)
    options = integrations.prepare()

    # --- Unpack options ---
    recipe_name: str
    if isinstance(options.recipe, str):
        recipe_name = options.recipe
    else:
        recipe_name = getattr(options.recipe, "qualified_name", None) or str(options.recipe)

    # ``options.overrides`` is the benchmark peer of ``RunOptions.overrides``.
    # ``image`` is the one entry that is *not* an override — it is a direct
    # write to ``recipe.container`` — so it is pulled out here; ``port`` is
    # named separately only because ``skip_run`` needs it below.  Everything
    # else is forwarded verbatim (see the ``_apply_recipe_overrides`` call).
    cli_overrides = dict(options.overrides)
    image = cli_overrides.pop("image", None)
    port = cli_overrides.pop("port", None)

    solo = options.solo
    profile = options.profile
    framework = options.framework
    output_file = options.output_file
    api_key_env = options.api_key_env
    exit_on_first_fail = options.exit_on_first_fail
    no_stop = options.no_stop
    skip_run = options.skip_run
    sync_tuning = options.sync_tuning
    rootful = options.rootful
    bench_timeout = options.timeout
    dry_run = options.dry_run
    executor_args = options.extra_docker_opts or ()
    export_results_files = options.export_files
    resume_mode = options.resume
    scheduler_name = options.scheduler
    category = options.category

    # bench_args come from options.bench_args (already a dict) — no key=value parsing needed at this layer
    user_bench_args: dict = dict(options.bench_args)

    if resume_mode is None:
        resume_mode = ResumeMode.AUTO

    v = sctx.variables
    config = sctx.config

    # -----------------------------------------------------------------------
    # Category pinning
    # -----------------------------------------------------------------------
    if category:
        from sparkrun.core.bootstrap import (
            get_benchmarking_frameworks_for_category,
            get_default_framework_for_category,
            AmbiguousCategoryError as _AmbiguousBoot,
            CategoryNotFoundError as _CatNotFoundBoot,
        )

        if framework:
            candidates = get_benchmarking_frameworks_for_category(category)
            if not any(fw_obj.framework_name == framework for fw_obj in candidates):
                raise FrameworkCategoryMismatch("Framework %r is not registered for category %r" % (framework, category))
        else:
            try:
                default_fw = get_default_framework_for_category(category, config=config)
            except _CatNotFoundBoot as exc:
                raise _CatNotFoundApi(str(exc)) from exc
            except _AmbiguousBoot as exc:
                raise _AmbiguousApi(str(exc)) from exc
            framework = default_fw.framework_name

    # -----------------------------------------------------------------------
    # 1. Load recipe
    # -----------------------------------------------------------------------
    from sparkrun.core.recipe import RecipeError

    try:
        if isinstance(options.recipe, str):
            recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name, resolve=False)
        else:
            from sparkrun.api._resolve import resolve_recipe

            recipe = resolve_recipe(options.recipe, sctx=sctx)
            registry_mgr = config.get_registry_manager() if profile else None
    except RecipeError as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e

    _resolved_name = _expand_recipe_shortcut(recipe_name)
    recipe_ref = _simplify_recipe_ref(_resolved_name) if _is_recipe_url(_resolved_name) else None

    # -----------------------------------------------------------------------
    # 2. Resolve benchmark configuration
    # -----------------------------------------------------------------------
    bench_spec = None
    bench_args: dict = {}

    if profile:
        from sparkrun.core.benchmark_profiles import find_benchmark_profile
        from sparkrun.core.benchmark_profiles import ProfileAmbiguousError
        from sparkrun.core.benchmark_profiles import ProfileError

        try:
            profile_path = find_benchmark_profile(profile, config, registry_mgr)
        except (ProfileError, ProfileAmbiguousError) as e:
            raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e
        bench_spec = BenchmarkSpec.load(profile_path)
        bench_args = dict(bench_spec.args)
        if not framework and bench_spec.framework:
            framework = bench_spec.framework
    else:
        bench_spec = BenchmarkSpec.from_recipe(recipe)
        if bench_spec:
            bench_args = dict(bench_spec.args)
            if not framework and bench_spec.framework:
                framework = bench_spec.framework

    if not framework:
        framework = config.default_benchmark_framework if config else "llama-benchy"

    try:
        fw = get_benchmarking_framework(framework)
    except ValueError as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e

    # Build layered bench args
    passthrough_layer: dict = {}
    if fw.passthrough_args:
        recipe_bench_block = recipe._raw.get("benchmark", {}) if hasattr(recipe, "_raw") else {}
        if isinstance(recipe_bench_block, dict):
            for key in fw.passthrough_args:
                if key in recipe_bench_block:
                    passthrough_layer[key] = recipe_bench_block[key]

    bench_args = {**fw.get_default_args(), **passthrough_layer, **bench_args}

    # Apply user bench_args overrides (from API options.bench_args dict, already parsed)
    for k, bv in user_bench_args.items():
        stripped_key = k.strip()
        if "api_key" in stripped_key.lower():
            raise BenchmarkFailed(
                "Passing '%s' via bench_args is insecure. Use api_key_env instead." % stripped_key,
                exit_code=1,
            )
        bench_args[stripped_key] = fw.interpret_arg(stripped_key, bv) if isinstance(bv, str) else bv

    effective_timeout = bench_timeout if bench_timeout is not None else (bench_spec.timeout if bench_spec else None)
    if effective_timeout is None:
        effective_timeout = DEFAULT_BENCHMARK_TIMEOUT
    if type(effective_timeout) is not int or effective_timeout <= 0 or type(exit_on_first_fail) is not bool:
        raise BenchmarkFailed("Benchmark requires a positive integer timeout and a boolean exit_on_first_fail")

    # -----------------------------------------------------------------------
    # 4. Build overrides and resolve runtime/hosts
    # -----------------------------------------------------------------------
    # Every remaining caller override is forwarded, so the benchmark builds the
    # *same* overrides dict ``sparkrun run`` does.  Dropping them here is what
    # made ``benchmark --tp 4`` fall back to solo while ``run --tp 4`` took four
    # nodes: placement reads ``tensor_parallel`` off the config chain, and an
    # empty overrides dict left it at the recipe's own value.
    #
    # Forwarding as ``**kwargs`` is deliberate — ``apply_recipe_overrides``
    # binds the flag-shaped names (``gpu_mem`` → ``gpu_memory_utilization``)
    # to its own parameters and passes anything else through untouched, so a
    # caller may use either spelling.  ``options``/``recipe`` are its own
    # parameter names and can never be recipe knobs.
    reserved = {"options", "recipe"}
    for key in sorted(reserved & cli_overrides.keys()):
        emitter.warning("ignoring unsupported override %r" % key)
        cli_overrides.pop(key)
    # The resolver mutates the supplied recipe; retain its known, non-null identity.
    _, overrides = _apply_recipe_overrides(
        (),  # options tuple (CLI only; already flattened into options.overrides)
        image=image,
        recipe=recipe,
        port=port,
        **cli_overrides,
    )

    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e

    # Same contract as ``sparkrun run`` — shared helper so the two cannot
    # drift on what they print or what they refuse.  A benchmark that launches
    # its own workload has the same stake in the recipe being honorable.
    from sparkrun.core.validation import validate_for_launch

    issues, validation_failed = validate_for_launch(recipe, runtime=runtime, config=config, v=v, include_unmapped_keys=False)
    for issue in issues:
        emitter.warning(issue.message)
    if validation_failed:
        blocking = next((i for i in issues if i.is_error), None)
        detail = blocking.message if blocking else "validation threshold not met"
        raise BenchmarkFailed("Recipe '%s' cannot be launched: %s" % (recipe.name, detail), exit_code=1)

    # Preserve preloaded cluster policy, and resolve host overrides once.
    try:
        cluster_cfg = resolve_cluster(options.cluster, options.hosts or None, sctx=sctx, config=config)
    except api.HostsUnreachable as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e
    _require_host_endpoint(recipe=recipe, cluster=cluster_cfg, sctx=sctx)
    if skip_run:
        # Reuse must observe the same recipe/cluster target precedence as run.
        from sparkrun.api._resolve import resolve_operation_target

        cluster_cfg, _ = resolve_operation_target(
            api.RunOptions(recipe=recipe, dry_run=dry_run), recipe=recipe, runtime=runtime, cluster=cluster_cfg, sctx=sctx
        )
        prepare_transport(cluster_cfg, dry_run=dry_run)
    host_list = list(cluster_cfg.hosts)
    if not host_list:
        raise BenchmarkFailed("No hosts configured for the benchmark", exit_code=1)
    sctx = sctx.for_cluster(cluster_cfg)
    config = sctx.config
    integrations.use_context(sctx)
    transfer = ResolvedClusterConfig.from_definition(cluster_cfg)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = transfer.resolve_transfer_config(config)

    run_options: "api.RunOptions | None" = None
    run_plan: "api.RunPlan | None" = None
    # ``--skip-run`` does not launch, so it discovers its hosts rather than
    # planning them — deferred to ``_resolve_running_deployment`` below, once
    # ``overrides`` are final (the intent id hashes the resolved port).  The
    # candidate list stays intact until then, because the lookup needs the
    # cluster's *full* host set.
    skip_run_cluster_id: str | None = None
    is_solo = bool(solo) or recipe.mode == "solo" or len(host_list) <= 1
    if not skip_run:
        # Plan the launch now so the banner below can name the target hosts,
        # then hand the same plan to ``api.run``.  The alternative — placing
        # here and passing the winners as ``hosts`` — would narrow the
        # candidate set, leaving ``api.run`` unable to reach any host this
        # pass discarded, and would sweep the cluster's occupancy twice.
        run_options = api.RunOptions(
            recipe=recipe,
            hosts=tuple(host_list),
            cluster=cluster_cfg,
            overrides=dict(overrides),
            solo=solo,
            dry_run=dry_run,
            follow=False,
            detached=True,
            trust=options.trust,
            scheduler=scheduler_name,
            transfer_mode=effective_transfer_mode,
            transfer_interface=effective_transfer_interface,
            cache_dir=remote_cache_dir,
            local_cache_dir=local_cache_dir,
            # Retain the resolved transfer preferences when explicit hosts
            # override the cluster's host list.
            preserve_model_perms=transfer.preserve_model_perms,
            skip_model_fan_out=transfer.skip_model_fan_out,
            rootful=rootful,
            sync_tuning=sync_tuning,
            extra_docker_opts=tuple(executor_args) if executor_args else None,
            recipe_ref=recipe_ref,
            offline=options.offline,
        )
        try:
            run_plan = api.plan(run_options, sctx=sctx)
        except api.SparkrunError as e:
            raise BenchmarkFailed("Error: inference launch failed: %s" % e, exit_code=1) from e
        host_list = list(run_plan.host_list)
        is_solo = run_plan.is_solo

    # Notify the emitter that the recipe is fully resolved so it can render
    # presentation-only artifacts (e.g. the CLI's VRAM estimate) without
    # forcing the CLI shell to reload the recipe ahead of orchestration.
    try:
        emitter.on_recipe_resolved(recipe, overrides, local_cache_dir=local_cache_dir)
    except Exception:
        logger.debug("emitter.on_recipe_resolved failed", exc_info=True)

    if skip_run:
        config_chain = recipe.build_config_chain(overrides)
        serve_port = _config_integer(config_chain.get("port") or 8000, key="port")
        overrides["port"] = serve_port
        # ``overrides`` are final now, so the intent id is stable — find the
        # deployment that is actually serving rather than assuming the whole
        # cluster is.
        host_list, is_solo, skip_run_cluster_id = _resolve_running_deployment(
            recipe,
            overrides,
            host_list,
            solo=solo,
            cluster=cluster_cfg,
            sctx=sctx,
            emitter=emitter,
        )

    # Launch validates its own asset requirements. Summaries must not resolve
    # unused defaults for native workloads or completed-measurement reuse.
    # Actual image provenance arrives through capture_launch_context below.
    container_image: str | None = None

    config_chain = recipe.build_config_chain(overrides)
    effective_tp = _config_integer(config_chain.get("tensor_parallel") or 1, key="tensor_parallel")
    request_model = resolve_request_model(fw, recipe, config_chain)
    served_model_name = resolve_served_model_name(recipe, config_chain.get("served_model_name"))

    # Only measurement arguments enter task definitions, identity and state.
    for k, bv in fw.prepare_benchmark_args(recipe, config_chain, overrides).items():
        bench_args.setdefault(k, bv)
    benchmark_api_key = bench_args.get("api_key")
    bench_args = public_benchmark_data(bench_args)

    # -----------------------------------------------------------------------
    # 5. Display summary
    # -----------------------------------------------------------------------
    from sparkrun import __version__

    emitter.banner("=" * 60)
    emitter.banner("sparkrun v%s — benchmark" % __version__)
    emitter.banner("=" * 60)
    emitter.banner("Recipe:                %s" % recipe.qualified_name)
    emitter.banner("Model:                 %s" % recipe.model)
    # The banner is what a reader checks the target against, and under an alias
    # the model id is *not* what the requests carry — naming only the id is how
    # a whole suite of 404s reads as a model-quality result (issue #298).
    if served_model_name != recipe.model:
        emitter.banner("Served as:             %s" % served_model_name)
    emitter.banner("Runtime:               %s" % runtime.runtime_name)
    if recipe.container:
        emitter.banner("Declared image:        %s" % recipe.container)
    emitter.banner("Benchmark Framework:   %s" % fw.framework_name)
    if profile:
        emitter.banner("Benchmark Profile:     %s" % profile)
    emitter.banner("Hosts:                 %s" % ", ".join(host_list))
    emitter.banner("Mode:                  %s" % ("solo" if is_solo else "cluster (%d nodes)" % len(host_list)))
    emitter.banner("")
    emitter.banner("Benchmark args:")
    for k, bv in bench_args.items():
        display_val = "***REDACTED***" if "api_key" in k.lower() else bv
        emitter.banner("  %-35s %s" % (k + ":", display_val))
    emitter.banner("=" * 60)
    emitter.banner("")

    # -----------------------------------------------------------------------
    # 6–10: Launch, benchmark, stop
    # -----------------------------------------------------------------------
    from sparkrun.core.progress import PROGRESS as _PROGRESS_LEVEL

    bench_result = BenchmarkExecution(recipe_name=recipe_name)
    bench_result.framework = fw
    bench_result.framework_name = fw.framework_name
    bench_result.category = category or fw.primary_category

    launched = False
    launch_result = None
    ssh_kwargs = build_ssh_kwargs(config)
    head_host = host_list[0]

    from sparkrun.orchestration.job_metadata import derive_cluster_id as _derive_cid

    # Under ``--skip-run`` the running deployment's own id wins: deriving one
    # from a host set that was never launched yields a cluster_id no job
    # metadata, ``stop`` or ``logs`` lookup can match.
    cluster_id = skip_run_cluster_id or _derive_cid(recipe, host_list, overrides=overrides)

    bench_result.recipe = recipe
    bench_result.overrides = overrides
    bench_result.cluster_id = cluster_id
    bench_result.host_list = host_list
    bench_result.container_image = container_image

    # -----------------------------------------------------------------------
    # Scheduled execution setup
    # -----------------------------------------------------------------------
    cache_dir = str(config.cache_dir) if config else None
    from sparkrun.benchmarking.scheduler import build_benchmark_tasks

    tasks = build_benchmark_tasks(fw, bench_args, bench_spec.schedule if bench_spec else None)

    # Released in the ``finally`` below, and eagerly on any early raise between
    # acquisition and that block. Every framework owns resumable task state.
    lock_stack = contextlib.ExitStack()
    state = None
    bench_result.profile = profile
    bench_result.benchmark_args = bench_args

    from sparkrun.benchmarking.run_state import (
        BenchmarkRunState,
        StateDirLocked,
        clear_state_dir,
        derive_benchmark_id,
        hold_state_dir,
    )

    # The cluster_id's intent half — previously all derive_benchmark_id
    # hashed — covers model, port and parallelism, so two recipes differing
    # only in a serve argument (e.g. --speculative-config) collided and
    # resumed into each other's results.  The fingerprint digests the
    # declared serve configuration to separate them; it excludes resolved
    # artifacts and placement, so the ID stays stable across relaunches.
    recipe_fingerprint = benchmark_recipe_fingerprint(recipe, overrides)

    # ``host_list`` is the *resolved* placement (what will actually run),
    # not the candidate set — see the module note on RunPlan.  Two runs of
    # one recipe against different nodes are different measurements and
    # must not share a state directory (issue #267).
    benchmark_id = derive_benchmark_id(
        cluster_id,
        fw.framework_name,
        profile,
        bench_args,
        [t.schedule_entry for t in tasks],
        recipe_fingerprint=recipe_fingerprint,
        hosts=host_list,
    )

    state_dir = (config.cache_dir / "benchmarks" / benchmark_id) if config else None
    state_dir_str = str(state_dir) if state_dir else remote_cache_path("benchmarks/%s" % benchmark_id, home="~")

    emitter.info("Benchmark ID:          %s" % benchmark_id)
    emitter.info("State directory:       %s" % state_dir_str)
    emitter.info("")

    # Hold the state directory for the whole run.  The read/decide/create
    # sequence below and the per-task artefacts it guards are keyed on
    # task index alone, so two runs sharing this directory overwrite each
    # other's measurements silently (issue #267).  Acquire *before* the
    # first read: two concurrent runs that both observe "no state" would
    # both create one.
    try:
        lock_stack.enter_context(hold_state_dir(benchmark_id, cache_dir))
    except StateDirLocked as e:
        raise BenchmarkFailed(
            "another benchmark run (pid %s on %s) is using state directory %s.\n"
            "Runs of the same recipe against the same hosts cannot proceed concurrently — "
            "their per-task results would overwrite each other. Wait for it to finish, or "
            "target different hosts." % (e.info.get("pid", "?"), e.info.get("host") or "?", state_dir_str),
            exit_code=1,
        ) from e

    try:
        if resume_mode == ResumeMode.FRESH:
            if not dry_run:
                clear_state_dir(benchmark_id, cache_dir)
            existing_state = None
        else:
            existing_state = BenchmarkRunState.load(benchmark_id, cache_dir, strict=True)
        if existing_state is not None and not existing_state.matches_hosts(host_list):
            # Only reachable for state written before hosts joined the ID, so
            # this is exactly the state that may hold a *different* node's
            # numbers.  Discard rather than warn: merging two node sets into
            # one result is the failure being fixed, not a lesser one.
            emitter.warning(
                "Discarding prior benchmark state %s: it was measured on %s but this run targets %s. "
                "Measurements from different nodes are not merged."
                % (
                    benchmark_id,
                    ", ".join(existing_state.host_list),
                    ", ".join(host_list),
                )
            )
            if not dry_run and state_dir and state_dir.exists():
                clear_state_dir(benchmark_id, cache_dir)
            existing_state = None

        if existing_state is None:
            if resume_mode == ResumeMode.REQUIRED:
                raise NoResumableState("ResumeMode.REQUIRED but no benchmark state exists for id %s" % benchmark_id)
        elif existing_state.is_complete(len(tasks)) and existing_state.extras.get("measurement_complete"):
            if _should_remeasure_complete_state(resume_mode, options.decision_callback, existing_state):
                if not dry_run and state_dir and state_dir.exists():
                    clear_state_dir(benchmark_id, cache_dir)
                    logger.debug("Deleted complete benchmark state at %s (user chose re-measure)", state_dir)
                existing_state = None
            else:
                emitter.warning(
                    "Prior benchmark state for %s is COMPLETE — re-emitting its recorded results; no requests will be sent. Use --fresh to re-measure."
                    % benchmark_id
                )
                try:
                    return _complete_saved_benchmark(existing_state, integrations, cache_dir)
                finally:
                    lock_stack.close()
        else:
            if resume_mode in (ResumeMode.IF_EXISTS, ResumeMode.REQUIRED):
                pass
            else:  # AUTO
                # Public callbacks receive a decision, never mutable saved state.
                decision = BenchmarkDecision(
                    "resume_incomplete",
                    "Found existing incomplete benchmark state (%d/%d tasks done). Resume?"
                    % (len(existing_state.completed_indices), len(existing_state.schedule)),
                    True,
                    existing_state.benchmark_id,
                )
                prompt_ok = bool(options.decision_callback(decision)) if options.decision_callback else decision.default
                if not prompt_ok:
                    if not dry_run and state_dir and state_dir.exists():
                        clear_state_dir(benchmark_id, cache_dir)
                        logger.debug("Deleted prior benchmark state at %s (user chose fresh start)", state_dir)
                    existing_state = None

        if existing_state is not None and existing_state.is_complete(len(tasks)):
            # Processing was interrupted after commands completed. Reuse the
            # locked resume path without launching inference or taking a second lock.
            try:
                return _resume_locked(
                    benchmark_id,
                    dry_run=dry_run,
                    emitter=emitter,
                    config=config,
                    cache_dir=cache_dir,
                    sctx=sctx,
                    integration_settings=options.integrations,
                    export_files=export_results_files,
                    output_file=output_file,
                    timeout=effective_timeout,
                    exit_on_first_fail=exit_on_first_fail,
                    api_key_env=options.api_key_env,
                )
            finally:
                lock_stack.close()

        if existing_state is not None:
            state = existing_state
            # Any reuse of prior state means some of the numbers below were
            # measured in an earlier session — a fully COMPLETE state emits
            # *only* recorded results.  Record both facts so the exported
            # artifact is self-describing: ``timing`` covers this invocation,
            # ``measured_at`` covers the data (issue #267).
            bench_result.resumed = True
            context = _restore_measurement_context(existing_state, category=bench_result.category)
            for attr, value in context.items():
                setattr(bench_result, attr, value)
            if saved_recipe := existing_state.extras.get("measurement_recipe_state"):
                from sparkrun.core.recipe import Recipe

                bench_result.recipe = Recipe._deserialize(saved_recipe)
            # Backfill on legacy state that predates the field, so the next
            # session can answer the host question this one had to assume.
            if not state.host_list:
                state.host_list = list(host_list)
            if state.cluster_id != cluster_id:
                logger.debug(
                    "Refreshing state.cluster_id %s -> %s on resume (same intent, new placement)",
                    state.cluster_id,
                    cluster_id,
                )
                state.cluster_id = cluster_id
        else:
            state = BenchmarkRunState(
                benchmark_id=benchmark_id,
                cluster_id=cluster_id,
                recipe_qualified_name=recipe.qualified_name,
                framework=fw.framework_name,
                profile=profile,
                base_args=bench_args,
                schedule=[t.schedule_entry for t in tasks],
                host_list=list(host_list),
                completed_indices=[],
                failed_indices=[],
            )
            from copy import deepcopy

            state.extras.update(deepcopy(options.state_extras))
            state.extras["benchmark_category"] = bench_result.category

        from sparkrun.benchmarking._specification import measurement_specification, restore_measurement_specification

        # Capture the same declared inputs used for the ID, before pinning
        # a resolved image into launch overrides (including legacy backfill).
        if state.measurement_spec is None:
            state.measurement_spec = measurement_specification(recipe, overrides)
        else:
            restore_measurement_specification(state, None, config=config)

        if "framework_version" not in state.extras:
            detected_version = fw.detect_version()
            if detected_version:
                state.extras["framework_version"] = detected_version
                emitter.info("Pinned %s version: %s" % (fw.framework_name, detected_version))
            else:
                logger.debug("No framework version detected for %s; version will float", fw.framework_name)
        else:
            emitter.info("Using pinned %s version: %s" % (fw.framework_name, state.extras["framework_version"]))

        pinned_image_sha = state.extras.get("container_image_sha")
        if pinned_image_sha:
            if container_image != pinned_image_sha:
                emitter.info("Using pinned image SHA: %s" % pinned_image_sha)
                emitter.info("  (was: %s)" % container_image)
            container_image = pinned_image_sha
            overrides["image"] = pinned_image_sha
            bench_result.container_image_sha = pinned_image_sha
            bench_result.container_image_sha_pinned = True

        if "container_image_longterm_ref" in state.extras:
            bench_result.longterm_image_ref = state.extras["container_image_longterm_ref"]
            bench_result.longterm_image_pinned = bool(state.extras.get("container_image_longterm_pinned", True))
    except BaseException:
        # The outer ``finally`` that normally releases the lock is not yet
        # in scope on this path (e.g. ResumeMode.REQUIRED with no state).
        lock_stack.close()
        raise

    cleanup_attempted = False

    def cleanup():
        nonlocal cleanup_attempted
        if not launched or no_stop or dry_run or cleanup_attempted:
            return
        cleanup_attempted = True
        _stop_inference(host_list, cluster_id, dry_run, sctx=sctx, strict=True)

    try:
        bench_result.benchmark_id = state.benchmark_id if state else ""
        bench_result.state_dir = str(state.state_dir(cache_dir)) if state else None
        if not dry_run:
            _check_measurement_prerequisites(fw, emitter)
        credentials = (
            BenchmarkCredentials()
            if dry_run
            else resolve_credentials(
                v,
                api_key_env=api_key_env,
                fallback=benchmark_api_key or runtime.resolve_api_key(recipe, overrides),
            )
        )
        if state is not None:
            state.api_key_env = api_key_env
            state.api_key_required = bool(credentials.api_key)
            state.timeout, state.exit_on_first_fail = effective_timeout, exit_on_first_fail
        integrations.bind(bench_result, state, resumed=bench_result.resumed)

        # -----------------------------------------------------------------------
        # 6. Launch inference (unless --skip-run)
        # -----------------------------------------------------------------------
        if not skip_run:
            logger.log(_PROGRESS_LEVEL, "Step 1/3: Launching inference...")

            # ``run_options`` / ``run_plan`` were built together above.  A
            # resumed benchmark may have pinned a container image SHA into
            # ``overrides`` since then; refresh the options so the launch uses
            # it.  The plan stays valid — the image is an input to neither
            # placement (which reads parallelism + VRAM) nor the intent id
            # (runtime + model + port + parallelism).
            assert run_options is not None and run_plan is not None  # not skip_run
            run_options = dataclasses.replace(run_options, overrides=dict(overrides))
            try:
                run_result = api.run(run_options, sctx=sctx, plan=run_plan)
            except api.SparkrunError as e:
                raise BenchmarkFailed("Error: inference launch failed: %s" % e, exit_code=1) from e

            bench_result.run_result = run_result
            launch_result = run_result.launch_result
            if run_result.rc != 0 and not dry_run:
                raise BenchmarkFailed(
                    "inference launch failed (exit code %d)" % run_result.rc,
                    exit_code=run_result.rc,
                )

            cluster_id = run_result.cluster_id
            bench_result.cluster_id = cluster_id
            if state is not None:
                state.cluster_id = cluster_id
            serve_port = run_result.serve_port
            # Establish ownership before invoking any frontend/plugin callback.
            launched = not getattr(run_result, "already_running", False)
            bench_result.launch_result = launch_result
            from sparkrun.orchestration.job_metadata import load_job_metadata

            candidate = capture_launch_context(
                bench_result,
                launch=launch_result,
                metadata=load_job_metadata(cluster_id, cache_dir=cache_dir),
                container_image=getattr(run_result, "container_image", None),
                state=state,
            )
            container_image = bench_result.container_image
            if state is not None and not dry_run:
                persist_measurement_context(bench_result, state)
                state.save(cache_dir)

            if run_result.serve_command:
                logger.info("Serve command:")
                for line in run_result.serve_command.strip().splitlines():
                    logger.info("  %s", credentials.redact(line))
                emitter.info("")

            if not state.extras.get("container_image_sha") and container_image:
                from sparkrun.orchestration.primitives import resolve_image_sha as _resolve_image_sha

                sha = _resolve_image_sha(container_image, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
                if sha:
                    state.extras["container_image_sha"] = sha
                    bench_result.container_image_sha = sha
                    bench_result.container_image_sha_pinned = True
                    emitter.info("Pinned image SHA: %s" % sha)
                    if not dry_run:
                        state.save(cache_dir)
                else:
                    logger.debug(
                        "resolve_image_sha returned None for %s; pin will not be enforced on resume",
                        container_image,
                    )

            if (
                container_image
                and not state.extras.get("container_image_longterm_ref")
                and launch_result is not None
                and launch_result.builder is not None
            ):
                try:
                    lt_ref, lt_pinned = launch_result.builder.resolve_long_term_image(
                        container_image=launch_result.container_image,
                        runtime_info=launch_result.runtime_info,
                        recipe=recipe,
                    )
                    if lt_pinned and lt_ref:
                        state.extras["container_image_longterm_ref"] = lt_ref
                        state.extras["container_image_longterm_pinned"] = True
                        bench_result.longterm_image_ref = lt_ref
                        bench_result.longterm_image_pinned = True
                        if not dry_run:
                            state.save(cache_dir)
                except Exception:
                    logger.debug("Long-term image resolution failed during pin", exc_info=True)
        else:
            logger.log(_PROGRESS_LEVEL, "Step 1/3: Skipping inference launch (--skip-run)")
            from sparkrun.orchestration.job_metadata import load_job_metadata

            candidate = capture_launch_context(bench_result, metadata=load_job_metadata(cluster_id, cache_dir=cache_dir), state=state)

        # -----------------------------------------------------------------------
        # 7. Wait for readiness and build target URL
        # -----------------------------------------------------------------------
        if state is not None and not dry_run:
            from sparkrun.benchmarking._specification import record_job_specification

            record_job_specification(state, candidate)
        integrations.checkpoint()

        if is_local_host(head_host):
            target_ip = "127.0.0.1"
        else:
            if dry_run:
                target_ip = "<HEAD_IP>"
            else:
                try:
                    target_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
                except RuntimeError as e:
                    raise BenchmarkFailed("Error detecting head IP: %s" % e, exit_code=1) from e

        if not dry_run and not skip_run:
            logger.log(_PROGRESS_LEVEL, "Waiting for inference server on %s:%d...", head_host, serve_port)
            logger.log(_PROGRESS_LEVEL, "Note that this could take ~5 minutes!")
            # Use the launch's effective readiness policy and reuse any
            # strategy/post-launch observation. Framework requests remain
            # separate from this one startup readiness measurement.
            if launch_result is not None and not bench_result.resumed:
                readiness = wait_for_serve_ready(
                    launch_result,
                    ssh_kwargs=ssh_kwargs,
                    port_retry_interval=5,
                    health_retry_interval=5,
                )
            else:
                # Alternate API implementations may not return LaunchResult.
                # Without container provenance, only endpoint waits are known.
                # Reused results also keep endpoint-only waits: do not issue a
                # startup inference just to re-emit recorded measurements.
                readiness_settings = resolve_readiness_settings(config=config, recipe=recipe)
                readiness = wait_for_endpoint_ready(
                    runtime=runtime,
                    cluster_id=cluster_id,
                    host_list=host_list,
                    is_solo=is_solo,
                    port=serve_port,
                    ssh_kwargs=ssh_kwargs,
                    dry_run=dry_run,
                    port_timeout_s=readiness_settings.port_timeout_s,
                    port_retry_interval=5,
                    health_timeout_s=readiness_settings.health_timeout_s,
                    health_retry_interval=5,
                )
            bench_result.readiness = readiness
            if not readiness.ready:
                if readiness.reason == "port":
                    raise BenchmarkFailed("Error: inference server did not become ready", exit_code=1)
                if readiness.reason in {"inference", "cancelled"}:
                    raise BenchmarkFailed("Error: inference server startup readiness failed (%s)" % readiness.reason, exit_code=1)
                raise BenchmarkFailed("Error: inference server health check timed out", exit_code=1)
            logger.log(
                _PROGRESS_LEVEL,
                "Inference server ready (endpoint waits: %.1fs port, %.1fs health).",
                readiness.port_wait_s,
                readiness.health_wait_s,
            )
            if startup := startup_timing_metadata(readiness):

                def _seconds(key):
                    return "%.3fs" % startup[key] if key in startup else "not observed"

                logger.log(
                    _PROGRESS_LEVEL,
                    "Docker-start readiness (rank 0, %s): TTR port %s; TTR HTTP %s; TTFT %s.",
                    startup["measurement"],
                    _seconds("ttr_port_open_s"),
                    _seconds("ttr_http_ready_s"),
                    _seconds("ttft_s") if startup["ttft_status"] == "measured" else "not applicable (inference disabled)",
                )
        elif dry_run:
            emitter.info("[dry-run] Would wait for inference server on %s:%d" % (head_host, serve_port))

        base_url = "http://%s:%d/v1" % (target_ip, serve_port)

        # -----------------------------------------------------------------------
        # 8. Run benchmark
        # -----------------------------------------------------------------------
        emitter.info("")
        logger.log(_PROGRESS_LEVEL, "Step 2/3: Running benchmark (%s)...", fw.framework_name)

        bench_result.start_time = datetime.now(tz=timezone.utc)
        if state is not None and not dry_run:
            state.begin_measurement(bench_result.category, bench_result.start_time.isoformat())
            bench_result.measured_at = state.extras["measurement_started_at"]
            persist_measurement_context(bench_result, state)
            state.save(cache_dir)
        integrations.checkpoint()

        est_tests = fw.estimate_test_count(bench_args)
        if est_tests is not None:
            logger.info("Estimated test iterations: %d", est_tests)

        stdout_text = ""
        stderr_text = ""

        # Scheduled execution path
        bench_result.profile = profile
        bench_result.benchmark_args = bench_args

        if dry_run:
            emitter.info("[dry-run] Would execute %d scheduled benchmark tasks via scheduler" % len(tasks))
            for i, t in enumerate(tasks):
                emitter.info("[dry-run]   task %d: %s" % (i, t.label))
        else:
            from sparkrun.benchmarking.scheduler import run_schedule

            title = _benchmark_title(recipe.name, profile)

            with emitter.schedule_progress(total_tasks=len(tasks), benchmark_id=benchmark_id, fw=fw, title=title) as pui:
                sched_result = run_schedule(
                    fw=fw,
                    tasks=tasks,
                    state=state,
                    target_url=base_url,
                    model=request_model,
                    timeout=effective_timeout,
                    task_events=pui,
                    cache_dir=cache_dir,
                    exit_on_first_fail=exit_on_first_fail,
                    credentials=credentials,
                )

            consolidated = sched_result.consolidated

            consolidated_path = _write_consolidated(state.state_dir(cache_dir), consolidated)
            result_file_for_parse = str(consolidated_path)

            if not sched_result.success:
                emitter.info("")
                emitter.info("Benchmark incomplete; you can resume later")
                raise BenchmarkFailed("Benchmark incomplete; schedule did not complete", exit_code=1)

            stdout_text = json.dumps(consolidated)
            bench_result.end_time = datetime.now(tz=timezone.utc)
            bench_result.start_time = bench_result.start_time or datetime.now(tz=timezone.utc)

        # -----------------------------------------------------------------------
        # 9. Parse and export results
        # -----------------------------------------------------------------------
        if not dry_run:
            bench_result.results = _parse_measurement(fw, stdout_text, stderr_text, result_file_for_parse)

        else:
            emitter.info("[dry-run] Would parse and export results to: %s" % (output_file or "benchmark_<recipe>_<framework>.yaml"))

        _finalize_measurement(
            bench_result,
            integrations,
            state,
            cache_dir,
            dry_run=dry_run,
            cleanup=cleanup,
            export=(
                lambda: _export_measurement(
                    bench_result,
                    config=config,
                    tp=effective_tp,
                    pp=_config_integer(config_chain.get("pipeline_parallel") or 1, key="pipeline_parallel"),
                    output_file=output_file,
                    emitter=emitter,
                )
            )
            if export_results_files and not dry_run
            else None,
        )
        logger.log(_PROGRESS_LEVEL, "Benchmark complete.")

    except BaseException as error:
        # Finalization also uses cleanup(); its once-only guard covers both
        # paths, including interrupts and secondary cleanup failures.
        try:
            cleanup()
        except Exception as cleanup_error:
            _record_cleanup_failure(error, cleanup_error)
        if isinstance(error, KeyboardInterrupt):
            _notify_interrupted(emitter, state_preserved=True)
        raise
    finally:
        lock_stack.close()

    return bench_result


def _config_integer(value: object, *, key: str) -> int:
    """Convert numeric configuration scalars, with an actionable API error."""
    from sparkrun.utils.data import integer_setting

    try:
        return integer_setting(value, key=key)
    except ValueError as error:
        raise BenchmarkFailed(str(error)) from error


def _notify_interrupted(emitter: _ProgressEmitter, *, state_preserved: bool) -> None:
    """Best-effort notification; rendering cannot replace cancellation."""
    try:
        emitter.info("Interrupted. State preserved so that you can resume later." if state_preserved else "Interrupted.")
    except Exception:
        logger.debug("Interrupted benchmark notification failed", exc_info=True)


def _parse_measurement(fw, stdout, stderr, result_file):
    """One decoding/validation boundary for initial execution and recovery."""
    results = _framework_results(fw.parse_results(stdout, stderr, result_file=result_file), fw.framework_name)
    if fw.measured_nothing(results):
        raise BenchmarkFailed(
            "benchmark produced no measurements — every request appears to have failed. "
            "%s exited successfully, so the cause is in its output: %s" % (fw.framework_name, result_file or "the benchmark output"),
            exit_code=1,
        )
    return results


def _finish_resumed_measurement(
    result,
    consolidated,
    *,
    state,
    integrations,
    cache_dir,
    config,
    export_files,
    output_file,
    emitter,
    consolidated_path=None,
):
    """Finish resumed tasks or interrupted processing through the same pipeline."""
    if consolidated_path is None:
        consolidated_path = _write_consolidated(state.state_dir(cache_dir), consolidated)
    result.results = _parse_measurement(result.framework, json.dumps(consolidated), "", str(consolidated_path))
    result.end_time = datetime.now(timezone.utc)
    config_chain = result.recipe.build_config_chain(result.overrides)
    _finalize_measurement(
        result,
        integrations,
        state,
        cache_dir,
        export=(
            lambda: _export_measurement(
                result,
                config=config,
                tp=_config_integer(config_chain.get("tensor_parallel") or 1, key="tensor_parallel"),
                pp=_config_integer(config_chain.get("pipeline_parallel") or 1, key="pipeline_parallel"),
                output_file=output_file,
                emitter=emitter,
            )
        )
        if export_files
        else None,
    )
    return result


def _framework_results(values, framework):
    from collections.abc import Mapping
    from sparkrun.utils.data import normalize_data

    try:
        if not isinstance(values, Mapping):
            raise TypeError("results must be a mapping")
        return normalize_data(public_benchmark_data(values), path="results")
    except (TypeError, ValueError) as exc:
        raise BenchmarkFailed("Invalid results from framework %r: %s" % (framework, exc)) from exc


@contextlib.contextmanager
def _integration_completion_errors(integrations):
    from sparkrun.api._errors import BenchmarkIntegrationFailed
    from sparkrun.core.benchmark_integrations import IntegrationCompletionError

    try:
        yield
    except Exception as exc:
        if integrations.result is None or not integrations.result.success:
            raise
        if isinstance(exc, BenchmarkFinalizationFailed):
            raise
        name = exc.integration if isinstance(exc, IntegrationCompletionError) else "<state>"
        integrations.result.integration_errors.setdefault(name, str(exc))
        result = _build_result(integrations.result)
        raise BenchmarkIntegrationFailed(str(exc), integration=name, result=result) from exc


def _complete_integrations(integrations) -> None:
    with _integration_completion_errors(integrations):
        integrations.complete()


@contextlib.contextmanager
def _finalization_errors(execution, stage):
    try:
        yield
    except BenchmarkFinalizationFailed:
        raise
    except Exception as exc:
        raise BenchmarkFinalizationFailed(str(exc), stage=stage, result=_build_result(execution)) from exc


def _record_cleanup_failure(primary: BaseException, cleanup_error: Exception) -> None:
    """Keep cleanup diagnostics without replacing the original failure."""
    if isinstance(primary, BenchmarkFinalizationFailed):
        primary.errors["cleanup"] = str(cleanup_error)
    else:
        primary.add_note("Inference cleanup also failed: %s" % cleanup_error)
        logger.warning("Inference cleanup also failed: %s", cleanup_error)


def _finalize_measurement(execution, integrations, state, cache_dir, *, dry_run=False, export=None, cleanup=None):
    """Commit validated measurements before optional export; always attempt owned cleanup."""
    execution.success = True
    if not dry_run:
        execution.measured_at = execution.measured_at or (execution.start_time or datetime.now(timezone.utc)).isoformat()
        execution.measurement_completed_at = (state.extras.get("measurement_completed_at") if state else None) or (
            execution.end_time or datetime.now(timezone.utc)
        ).isoformat()
    primary = None
    try:
        with _integration_completion_errors(integrations):
            if state is not None and not dry_run:
                persist_measurement_context(execution, state)
                _save_completed_results(state, execution.results, cache_dir)
        if export is not None:
            with _finalization_errors(execution, "export"):
                export()
    except BaseException as exc:
        primary = exc
    finally:
        if cleanup is not None:
            try:
                with _finalization_errors(execution, "cleanup"):
                    cleanup()
            except Exception as exc:
                if primary is None:
                    primary = exc
                else:
                    _record_cleanup_failure(primary, exc)
    if primary is not None:
        raise primary
    with _integration_completion_errors(integrations):
        if state is not None and not dry_run:
            state.extras["benchmark_outputs"] = {key: str(value) for key, value in (execution.outputs or {}).items()}
            state.save(cache_dir)
        integrations.complete()


def _export_measurement(execution, *, config, tp, pp, output_file, emitter):
    """One export implementation for initial and resumed measurements."""
    from sparkrun.benchmarking.base import _write_measurement

    if not output_file:
        profile_slug = execution.profile.replace("/", "_").replace("@", "") if execution.profile else "default"
        pp_suffix = "_pp%d" % pp if pp > 1 else ""
        output_file = str(
            config.default_benchmark_output_dir
            / ("benchmark_%s_%s_tp%d%s.yaml" % (execution.recipe.name.replace("/", "_"), profile_slug, tp, pp_suffix))
        )
    # Bind all artifact references before export, callbacks, and persistence.
    output_paths = _measurement_output_paths(output_file, execution.results)
    output_file = output_paths["yaml"]
    _write_measurement(execution, tp=tp, output_path=output_file)
    if execution.outputs is None:
        execution.outputs = {}
    execution.outputs["yaml"] = str(output_file)
    emitter.info("Results saved to: %s" % output_file)
    _emit_results_outputs(execution.results, output_paths, emitter, outputs=execution.outputs)


def _notify_complete(result, emitter):
    try:
        emitter.event(
            ProgressEvent("run_complete", {"benchmark_id": result.benchmark_id, "success": result.success, "resumed": result.resumed})
        )
    except Exception as exc:
        raise BenchmarkFinalizationFailed(str(exc), stage="notification", result=result) from exc
    return result


def _save_completed_results(state, results, cache_dir):
    """Keep validated results for integration retries without rerunning inference."""
    import yaml

    path = state.state_dir(cache_dir) / "result.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    from sparkrun.utils.fs import atomic_private_write

    atomic_private_write(path, yaml.safe_dump(results, sort_keys=False))
    state.extras["measurement_complete"] = True
    state.save(cache_dir)


def _measurement_output_paths(output_file, results) -> dict[str, Path]:
    """Bind distinct primary/sidecar destinations before writing any export."""
    primary = Path(output_file).expanduser().absolute()
    paths = {"yaml": primary}
    for fmt in ("json", "csv"):
        if results.get(fmt):
            paths[fmt] = primary.with_suffix("." + fmt) if primary.suffix.lower() in {".yaml", ".yml"} else Path(str(primary) + "." + fmt)
    # Existing symlinks or hard links must not make otherwise distinct names
    # overwrite one another. This is a preflight, not a concurrent-writer lock.
    resolved = set()
    inodes = set()
    for path in paths.values():
        canonical = path.resolve()
        stat = path.stat() if path.exists() else None
        inode = (stat.st_dev, stat.st_ino) if stat is not None else None
        if canonical in resolved or (inode is not None and inode in inodes):
            raise ValueError("Benchmark export destinations refer to the same file: %s" % path)
        resolved.add(canonical)
        if inode is not None:
            inodes.add(inode)
    return paths


def _emit_results_outputs(results: dict[str, Any], paths: dict[str, Path], emitter: _ProgressEmitter, *, outputs=None) -> dict[str, Path]:
    """Write planned JSON/CSV exports and record only successful artifacts."""
    writers = {
        "json": lambda data, path: path.write_text(json.dumps(data, indent=2)),
        "csv": lambda data, path: path.write_text(data),
    }
    written: dict[str, Path] = {}
    for fmt, writer in writers.items():
        if fmt not in paths:
            continue
        out = paths[fmt]
        writer(results[fmt], out)
        written[fmt] = out
        if outputs is not None:
            outputs[fmt] = str(out)
        emitter.info("%s output: %s" % (fmt.upper(), out))
    return written


def _stop_inference(host_list, cluster_id, dry_run, sctx=None, emitter: _ProgressEmitter | None = None, *, strict=False):
    """Stop the inference workload via the library API.

    ``emitter`` (optional) lets the orchestration surface the dry-run notice
    and any warning to the CLI without coupling this helper to ``click``.
    """
    import sparkrun.api as api

    if dry_run:
        if emitter is not None:
            emitter.info("[dry-run] Would stop cluster %s on %s" % (cluster_id, ", ".join(host_list)))
        return

    try:
        stopped = api.stop(
            cluster_id=cluster_id,
            hosts=tuple(host_list) if host_list else None,
            sctx=sctx,
        )
        if strict and isinstance(stopped, api.StopResult) and not stopped.success:
            raise BenchmarkFailed("Inference cleanup incomplete: " + "; ".join(stopped.errors or stopped.hosts_failed))
    except Exception as e:
        if strict:
            raise
        logger.warning("Failed to stop inference: %s", e)
        if emitter is not None:
            emitter.warning("failed to stop inference: %s" % e)


# ---------------------------------------------------------------------------
# Resume orchestration
# ---------------------------------------------------------------------------


def _resume_benchmark(
    benchmark_id: str,
    *,
    dry_run: bool = False,
    sctx: "SparkrunContext | None" = None,
    emitter: _ProgressEmitter | None = None,
    integrations: Mapping[str, Mapping[str, Any]] | None = None,
    export_files: bool = True,
    output_file: str | None = None,
    timeout: int | None = None,
    exit_on_first_fail: bool | None = None,
    api_key_env: str | None = None,
) -> BenchmarkResult:
    """Shared resume implementation; the emitter is a private CLI adapter."""
    from sparkrun.benchmarking.run_state import StateDirLocked, hold_state_dir

    if emitter is None:
        emitter = _NullProgressEmitter()

    sctx = resolve_sctx(sctx)
    config = sctx.config
    cache_dir = str(config.cache_dir) if config else None

    # Take the state directory before reading it: a resume racing a `benchmark
    # run` for the same id would otherwise interleave into the same per-task
    # artefacts (issue #267).  Held for the whole resume.
    lock_stack = contextlib.ExitStack()
    try:
        lock_stack.enter_context(hold_state_dir(benchmark_id, cache_dir))
    except StateDirLocked as e:
        raise BenchmarkFailed(
            "benchmark %s is already being run by pid %s on %s. Wait for it to finish before resuming."
            % (benchmark_id, e.info.get("pid", "?"), e.info.get("host") or "?"),
            exit_code=1,
        ) from e

    with lock_stack:
        execution = _resume_locked(
            benchmark_id,
            dry_run=dry_run,
            emitter=emitter,
            config=config,
            cache_dir=cache_dir,
            sctx=sctx,
            integration_settings=integrations,
            export_files=export_files,
            output_file=output_file,
            timeout=timeout,
            exit_on_first_fail=exit_on_first_fail,
            api_key_env=api_key_env,
        )
        result = _build_result(execution)
        return _notify_complete(result, emitter)


def resume_benchmark(
    benchmark_id: str,
    *,
    dry_run: bool = False,
    sctx: SparkrunContext | None = None,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
    decision_callback: Callable[[BenchmarkDecision], bool] | None = None,
    integrations: Mapping[str, Mapping[str, Any]] | None = None,
    export_files: bool = True,
    output_file: str | None = None,
    timeout: int | None = None,
    exit_on_first_fail: bool | None = None,
    api_key_env: str | None = None,
) -> BenchmarkResult:
    """Resume measurement or retry publication, returning the same result as benchmark().

    Progress is silent unless a callback is supplied. Integration decisions use
    decision_callback; absent a handler, each decision uses its stated default.
    Completed commands can finish interrupted result processing without live
    inference. Committed measurements can retry saved integrations directly.
    No pending tasks/integrations returns the saved result with already_complete=True.
    export_files/output_file control exports after resumed measurement; completed
    result loading and publication retries never regenerate optional exports.
    Remaining commands use saved timeout/failure policy and a fresh inference
    credential lookup; result processing and publication do not need that lookup.
    timeout, exit_on_first_fail and api_key_env override unfinished execution only.
    Invalid/unreadable checkpoints raise SparkrunError without replacement.
    Missing state/inference raises NoResumableState. Publication failures raise
    BenchmarkIntegrationFailed with the completed result attached.
    """
    emitter = (
        _CallbackProgressEmitter(progress_callback, decision_callback)
        if progress_callback is not None
        else _NullProgressEmitter(decision_callback)
    )
    try:
        return _resume_benchmark(
            benchmark_id,
            dry_run=dry_run,
            sctx=sctx,
            emitter=emitter,
            integrations=integrations,
            export_files=export_files,
            output_file=output_file,
            timeout=timeout,
            exit_on_first_fail=exit_on_first_fail,
            api_key_env=api_key_env,
        )
    except SparkrunError:
        raise
    except Exception as exc:
        raise SparkrunError("benchmark resume failed: %s" % exc) from exc


def _saved_execution(state, cache_dir):
    """One detached result reconstruction for complete and partial resumes."""
    from sparkrun.benchmarking.base import BenchmarkExecution

    outputs = {}
    for key, value in state.extras.get("benchmark_outputs", {}).items():
        if Path(value).is_absolute():
            outputs[key] = value
        else:
            logger.warning("Omitting legacy relative benchmark output %r: its original directory is unknown", value)
    return BenchmarkExecution(
        benchmark_id=state.benchmark_id,
        framework_name=state.framework,
        host_list=state.host_list,
        outputs=outputs,
        state_dir=str(state.state_dir(cache_dir)),
        recipe_name=state.recipe_qualified_name,
        cluster_id=state.cluster_id,
        profile=state.profile,
        benchmark_args=state.base_args,
        resumed=True,
        **_restore_measurement_context(state),
    )


def _complete_saved_benchmark(state, integrations, cache_dir):
    """Load validated measurements and retry publication without execution."""
    import yaml
    from sparkrun.core.benchmark_integrations import STATE_KEY

    result = _saved_execution(state, cache_dir)
    saved_names = set(state.extras.get(STATE_KEY, {})) & integrations.specs.keys()
    result_path = state.state_dir(cache_dir) / "result.yaml"
    if not state.extras.get("measurement_complete") or not result_path.is_file():
        raise BenchmarkFailed("Completed tasks have no validated benchmark results; rerun with --fresh.", exit_code=1)
    result.results = _framework_results(yaml.safe_load(result_path.read_text()), state.framework)
    result.success = True
    if not integrations.contexts and not saved_names:
        result.already_complete = True
        return result
    with _integration_completion_errors(integrations):
        integrations.bind(result, state, resumed=True)
        integrations.complete()
    return result


def _resume_locked(
    benchmark_id: str,
    *,
    dry_run: bool,
    emitter: _ProgressEmitter,
    config,
    cache_dir: str | None,
    sctx,
    integration_settings=None,
    export_files: bool = True,
    output_file: str | None = None,
    timeout: int | None = None,
    exit_on_first_fail: bool | None = None,
    api_key_env: str | None = None,
) -> BenchmarkExecution:
    """Body of :func:`resume_benchmark`, run while holding the state-dir lock."""
    from sparkrun.api._errors import NoResumableState
    from sparkrun.core.benchmark_integrations import BenchmarkIntegrationSession
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import run_schedule
    from sparkrun.core.bootstrap import get_benchmarking_framework
    from sparkrun.orchestration.job_metadata import check_job_running, load_job_metadata
    from sparkrun.orchestration.primitives import build_ssh_kwargs, detect_host_ip
    from sparkrun.utils import is_local_host

    # Load existing state
    state = BenchmarkRunState.load(benchmark_id, cache_dir, strict=True)
    if state is None:
        raise NoResumableState("no benchmark state found for id: %s" % benchmark_id)

    options = BenchmarkOptions(
        recipe=state.recipe_qualified_name,
        framework=state.framework,
        profile=state.profile,
        dry_run=dry_run,
        integrations={} if integration_settings is None else integration_settings,
    )
    options = _validated_benchmark_options(options)
    integrations = BenchmarkIntegrationSession(options, sctx=sctx, emitter=emitter)
    if state.is_complete(len(state.schedule)) and state.extras.get("measurement_complete"):
        return _complete_saved_benchmark(state, integrations, cache_dir)
    result = _saved_execution(state, cache_dir)

    # Snapshot before ``run_schedule`` starts saving: this is when the tasks
    # already recorded in the state were measured.  Read afterwards it would
    # be ~now for every resume, which is exactly the conflation ``measured_at``
    # exists to prevent.
    # result.measured_at was pinned before any publication/state write.

    recipe_name = state.recipe_qualified_name
    from sparkrun.benchmarking._specification import restore_measurement_specification

    processing_only = state.is_complete(len(state.schedule)) and state.measurement_spec is not None
    meta = None if processing_only else load_job_metadata(state.cluster_id, cache_dir=cache_dir)
    recipe, saved_overrides = restore_measurement_specification(state, meta, config=config)
    from sparkrun.core.recipe import Recipe

    saved_recipe = state.extras.get("measurement_recipe_state")
    result.recipe = Recipe._deserialize(saved_recipe) if saved_recipe else recipe
    result.host_list = state.host_list
    if result.overrides is None:
        result.overrides = saved_overrides
    if meta is not None:
        capture_launch_context(result, metadata=meta, state=state)
    if dry_run:
        integrations.bind(result, state, resumed=True)
        emitter.info(
            "[dry-run] Would resume benchmark %s (%d task(s) remaining)"
            % (
                benchmark_id,
                len(state.schedule) - len(state.completed_indices),
            )
        )
        result.success = True
        _complete_integrations(integrations)
        return result

    # Reconstruct framework
    try:
        fw = get_benchmarking_framework(state.framework)
    except ValueError as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e

    # Rebuild tasks from saved state
    from sparkrun.benchmarking.scheduler import build_benchmark_tasks

    tasks = build_benchmark_tasks(fw, state.base_args, state.schedule)

    if state.measurement_spec is None:
        from sparkrun.benchmarking._specification import measurement_specification, record_job_specification

        state.measurement_spec = measurement_specification(recipe, saved_overrides)
        record_job_specification(state, meta)
    result.framework = fw
    result.category = result.category or fw.primary_category

    if state.is_complete(len(tasks)):
        from sparkrun.benchmarking.scheduler import _collect_completed_results
        from sparkrun.benchmarking.aggregator import gap_analysis

        consolidated = _collect_completed_results(fw, tasks, state, cache_dir)
        gaps = gap_analysis(tasks, consolidated, fw, completed_indices=state.completed_indices)
        if gaps and processing_only:
            meta = load_job_metadata(state.cluster_id, cache_dir=cache_dir)
            restore_measurement_specification(state, meta, config=config)
            capture_launch_context(result, metadata=meta, state=state)
        for task in gaps:
            if task.index in state.completed_indices:
                state.mark_failed(task.index, "missing measurement coverage")
        state.save(cache_dir)
        if state.is_complete(len(tasks)):
            # Commands already succeeded. Finish decoding/commit without live
            # inference, command prerequisites, credentials, or another session.
            integrations.bind(result, state, resumed=True)
            return _finish_resumed_measurement(
                result,
                consolidated,
                state=state,
                integrations=integrations,
                cache_dir=cache_dir,
                config=config,
                export_files=export_files,
                output_file=output_file,
                emitter=emitter,
            )

    # An artifact gap can turn result recovery into measurement execution.
    # Revalidate the running job before using it for the remaining tasks.
    if processing_only:
        meta = load_job_metadata(state.cluster_id, cache_dir=cache_dir)
        restore_measurement_specification(state, meta, config=config)
    meta = meta or {}
    _require_host_endpoint(
        recipe=recipe,
        cluster=None,
        sctx=sctx,
        executor_overrides={
            **(meta.get("executor_config") or {}),
            **({"executor": meta["executor"]} if meta.get("executor") else {}),
        },
    )
    _check_measurement_prerequisites(fw, emitter)

    effective_timeout = timeout if timeout is not None else (state.timeout or DEFAULT_BENCHMARK_TIMEOUT)
    effective_fail_fast = exit_on_first_fail if exit_on_first_fail is not None else state.exit_on_first_fail
    if type(effective_timeout) is not int or effective_timeout <= 0 or type(effective_fail_fast) is not bool:
        raise BenchmarkFailed("Resume requires a positive integer timeout and a boolean exit_on_first_fail")
    # Reconstruct hosts from job metadata
    if not meta or not meta.get("hosts"):
        raise NoResumableState(
            "no job metadata found for cluster_id %r.\n"
            "Please relaunch inference with `sparkrun run` and then retry resume." % state.cluster_id
        )
    from sparkrun.core.bootstrap import get_runtime

    from sparkrun.core.recipe import Recipe

    auth_recipe = Recipe._deserialize(meta["recipe_state"]) if meta.get("recipe_state") else recipe
    credentials = resolve_credentials(
        sctx.variables,
        api_key_env=api_key_env or state.api_key_env,
        fallback=get_runtime(recipe.runtime, sctx.variables).resolve_api_key(auth_recipe, meta.get("overrides") or {}),
    )
    if state.api_key_required and not credentials.api_key:
        raise BenchmarkFailed("This benchmark requires authentication; supply api_key_env when resuming")
    state.api_key_env = api_key_env or state.api_key_env
    state.api_key_required = bool(credentials.api_key)
    state.timeout, state.exit_on_first_fail = effective_timeout, effective_fail_fast

    hosts = meta["hosts"]
    from sparkrun.api._resolve import resolve_cluster_for_job

    cluster = resolve_cluster_for_job(None, hosts, meta=meta, sctx=sctx)
    sctx = sctx.for_cluster(cluster)
    config = sctx.config
    integrations.use_context(sctx)
    result.host_list = hosts
    capture_launch_context(result, metadata=meta, state=state)

    # Check if inference is currently running
    ssh_kwargs = build_ssh_kwargs(config)
    job_status = check_job_running(cluster_id=state.cluster_id, hosts=hosts, ssh_kwargs=ssh_kwargs, cache_dir=cache_dir)
    if not job_status.running:
        raise NoResumableState(
            "inference cluster %r is not currently running.\n"
            "Please relaunch with `sparkrun run %s` first, then retry resume." % (state.cluster_id, recipe_name)
        )

    # Determine the serving URL
    head_host = hosts[0]
    serve_port = meta.get("port") or 8000

    if is_local_host(head_host):
        target_ip = "127.0.0.1"
    elif dry_run:
        target_ip = "<HEAD_IP>"
    else:
        try:
            target_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
        except RuntimeError as e:
            raise BenchmarkFailed("Error detecting head IP: %s" % e, exit_code=1) from e

    base_url = "http://%s:%d/v1" % (target_ip, serve_port)

    integrations.bind(result, state, resumed=True)
    emitter.banner("=" * 60)
    emitter.banner("sparkrun — benchmark resume")
    emitter.banner("=" * 60)
    emitter.banner("Benchmark ID:          %s" % benchmark_id)
    emitter.banner("Recipe:                %s" % recipe_name)
    emitter.banner("Framework:             %s" % state.framework)
    emitter.banner("Profile:               %s" % (state.profile or "(none)"))
    emitter.banner("Hosts:                 %s" % ", ".join(hosts))
    emitter.banner("Completed tasks:       %d / %d" % (len(state.completed_indices), len(tasks)))
    emitter.banner("State directory:       %s" % state.state_dir(cache_dir))
    emitter.banner("=" * 60)
    emitter.banner("")

    title = _benchmark_title(recipe.name, state.profile)

    state.begin_measurement(result.category, datetime.now(timezone.utc).isoformat())
    result.measured_at = state.extras["measurement_started_at"]
    persist_measurement_context(result, state)
    state.save(cache_dir)
    try:
        with emitter.schedule_progress(total_tasks=len(tasks), benchmark_id=benchmark_id, fw=fw, title=title) as pui:
            sched_result = run_schedule(
                fw=fw,
                tasks=tasks,
                state=state,
                target_url=base_url,
                # Resolved off the *recorded* recipe and overrides, like the
                # model id it replaces: a resumed schedule must keep asking for
                # what the measured deployment serves, not what an edited
                # recipe now declares.
                model=resolve_request_model(fw, recipe, recipe.build_config_chain(saved_overrides)),
                timeout=effective_timeout,
                task_events=pui,
                cache_dir=cache_dir,
                exit_on_first_fail=effective_fail_fast,
                credentials=credentials,
            )

        consolidated = sched_result.consolidated

        # Write consolidated.json to state dir
        consolidated_path = _write_consolidated(state.state_dir(cache_dir), consolidated)

        if not sched_result.success:
            emitter.info("")
            emitter.info("Benchmark incomplete; you can resume later.")
            raise BenchmarkFailed("Benchmark incomplete; schedule did not complete", exit_code=1)

        emitter.info("")
        emitter.info("Benchmark resumed and completed successfully.")

        return _finish_resumed_measurement(
            result,
            consolidated,
            state=state,
            integrations=integrations,
            cache_dir=cache_dir,
            config=config,
            export_files=export_files,
            output_file=output_file,
            emitter=emitter,
            consolidated_path=consolidated_path,
        )

    except KeyboardInterrupt:
        _notify_interrupted(emitter, state_preserved=True)
        raise


# ---------------------------------------------------------------------------
# Public API entry point
# ---------------------------------------------------------------------------


def benchmark(
    options: BenchmarkOptions,
    *,
    sctx: "SparkrunContext | None" = None,
) -> BenchmarkResult:
    """Run a benchmark and return a structured :class:`BenchmarkResult`.

    Args:
        options: Inputs for the benchmark run.
        sctx: Optional shared :class:`SparkrunContext`.  When omitted a
            fresh session is built; callers chaining multiple ``api.*``
            calls can construct one ``sctx`` and pass it to share state.

    Raises:
        :class:`BenchmarkFailed`: The run terminated unsuccessfully
            (non-zero exit, task failures, or aborted launch).
        :class:`SparkrunError` (subclass): Other typed failures.
        :class:`KeyboardInterrupt`: Re-raised after the underlying flow
            persists its state.
    """
    sctx = resolve_sctx(sctx)
    effective_options = options

    if effective_options.progress_callback is None:
        emitter: _ProgressEmitter = _NullProgressEmitter(effective_options.decision_callback)
    else:
        emitter = _CallbackProgressEmitter(effective_options.progress_callback, effective_options.decision_callback)

    try:
        bench_result = _execute_benchmark(effective_options, sctx=sctx, emitter=emitter)
    except KeyboardInterrupt:
        raise
    except SparkrunError:
        raise
    except Exception as exc:
        raise SparkrunError("benchmark failed: %s" % exc) from exc

    result = _build_result(bench_result)
    from sparkrun.telemetry import emit_benchmark_telemetry

    emit_benchmark_telemetry(
        sctx.config,
        result=result,
        options=effective_options,
        recipe=getattr(bench_result, "recipe", None),
    )
    return _notify_complete(result, emitter)


def _build_result(execution: BenchmarkExecution) -> BenchmarkResult:
    """Pure, detached snapshot of authoritative measurement/execution values."""
    from copy import deepcopy
    from sparkrun.benchmarking.base import startup_timing_metadata

    startup = startup_timing_metadata(execution.readiness, resumed=execution.resumed)
    return BenchmarkResult(
        success=execution.success,
        benchmark_id=execution.benchmark_id,
        category=execution.category,
        framework=execution.framework_name,
        profile=execution.profile,
        results=deepcopy(execution.results or {}),
        outputs={key: str(value) for key, value in (execution.outputs or {}).items() if value is not None},
        run_result=execution.run_result,
        cluster_id=execution.cluster_id or "",
        host_list=tuple(execution.host_list or ()),
        container_image=execution.container_image or "",
        container_image_sha=execution.container_image_sha,
        container_image_sha_pinned=execution.container_image_sha_pinned,
        container_image_longterm_ref=execution.longterm_image_ref,
        container_image_longterm_pinned=execution.longterm_image_pinned,
        metadata={
            "framework": execution.framework_name,
            "profile": execution.profile,
            "bench_args": public_benchmark_data(execution.benchmark_args or {}),
            **({"timing": {"startup": startup}} if startup else {}),
        },
        state_dir=execution.state_dir,
        resumed=execution.resumed,
        measured_at=execution.measured_at,
        completed_at=execution.measurement_completed_at,
        already_complete=execution.already_complete,
        integration_results=deepcopy(execution.integration_results or {}),
        integration_errors=dict(execution.integration_errors),
    )


__all__ = [
    "benchmark",
    "resume_benchmark",
    "_ProgressEmitter",
    "_NullProgressEmitter",
    "_CallbackProgressEmitter",
    "_execute_benchmark",
    "_build_result",
]
