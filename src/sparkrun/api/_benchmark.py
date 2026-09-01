"""``sparkrun.api.benchmark`` — public Python entry point for benchmark runs.

Step 7: Orchestration body lifted from ``cli._benchmark._run_benchmark`` into
``_execute_benchmark``.  The CLI becomes a thin presentation shell; library
callers get the full flow with no Click / sys.exit coupling.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from sparkrun.api._benchmark_models import (
    BenchmarkOptions,
    BenchmarkResult,
    ProgressEvent,
    ResumeMode,
)
from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import BenchmarkFailed, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress emitter abstraction
# ---------------------------------------------------------------------------


class _ProgressEmitter:
    """Side-channel for orchestration to emit text/structured events without
    coupling to click/CLI rendering.  CLI provides a subclass; library callers
    can supply a no-op emitter or a custom callback-driven one.
    """

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
    """No-op emitter for headless / API callers."""

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


class _CallbackProgressEmitter(_ProgressEmitter):
    """For API callers that pass progress_callback."""

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
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


# ---------------------------------------------------------------------------
# Default benchmark timeout (mirrors cli/_benchmark.py)
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK_TIMEOUT: int = 14400  # 4 hours


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
    on_complete_state: "Callable[[Any], bool] | None",
    existing_state: Any,
) -> bool:
    """Whether COMPLETE prior state should be discarded and re-measured.

    "Resuming" COMPLETE state runs zero tasks and re-emits the previous run's
    results into the new output — indistinguishable from a real measurement, so
    it must never happen silently (the caller warns on the reuse path).

    ``ResumeMode.AUTO`` delegates the choice to *on_complete_state* (the CLI
    wires an interactive confirm); with no callback the library default is
    reuse, matching prior behaviour.  ``IF_EXISTS`` / ``REQUIRED`` asked for a
    resume explicitly, so they always reuse.  (``FRESH`` never reaches here —
    it deletes the state before this decision.)
    """
    if resume_mode != ResumeMode.AUTO or on_complete_state is None:
        return False
    return bool(on_complete_state(existing_state))


def _resolve_running_deployment(
    recipe,
    overrides: dict,
    candidate_hosts: list[str],
    *,
    solo: bool,
    cluster: "str | None",
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


def _execute_benchmark(
    options: BenchmarkOptions,
    *,
    sctx: "SparkrunContext",
    emitter: _ProgressEmitter,
) -> Any:
    """Execute the full benchmark flow: launch inference -> benchmark -> stop.

    Returns a ``sparkrun.benchmarking.base.BenchmarkResult`` (internal type)
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
    from sparkrun.benchmarking.base import export_results, BenchmarkResult as _InternalBenchmarkResult
    from sparkrun.core.benchmark_profiles import BenchmarkSpec
    from sparkrun.core.bootstrap import get_runtime, get_benchmarking_framework
    from sparkrun.utils import is_local_host
    from sparkrun.core.launcher import wait_for_endpoint_ready
    from sparkrun.orchestration.primitives import (
        build_ssh_kwargs,
        detect_host_ip,
    )
    from sparkrun.core.recipe import (
        expand_recipe_shortcut as _expand_recipe_shortcut,
        is_recipe_url as _is_recipe_url,
        simplify_recipe_ref as _simplify_recipe_ref,
    )
    from sparkrun.core.cluster_manager import resolve_cluster_config
    from sparkrun.core.resolve import apply_recipe_overrides as _apply_recipe_overrides, load_recipe as _load_recipe
    from sparkrun.api._hosts import resolve_host_list
    from sparkrun.api._errors import (
        NoResumableState,
        FrameworkCategoryMismatch,
        AmbiguousCategoryError as _AmbiguousApi,
        CategoryNotFound as _CatNotFoundApi,
    )

    # --- Unpack options ---
    recipe_name: str
    if isinstance(options.recipe, str):
        recipe_name = options.recipe
    else:
        recipe_name = getattr(options.recipe, "qualified_name", None) or str(options.recipe)

    cluster_name: str | None = None
    if isinstance(options.cluster, str):
        cluster_name = options.cluster
    elif options.cluster is not None:
        cluster_name = getattr(options.cluster, "name", None)

    hosts = list(options.hosts) if options.hosts else []
    # ``options.overrides`` is the benchmark peer of ``RunOptions.overrides``.
    # ``image`` is the one entry that is *not* an override — it is a direct
    # write to ``recipe.container`` — so it is pulled out here; ``port`` is
    # named separately only because ``skip_run`` needs it below.  Everything
    # else is forwarded verbatim (see the ``_apply_recipe_overrides`` call).
    cli_overrides = dict(options.overrides) if isinstance(options.overrides, dict) else {}
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
    on_prompt_required = options.on_prompt_required
    on_complete_state = options.on_complete_state
    submission_id_for_extras = options.state_extras.get("submission_id") if options.state_extras else None
    scheduler_name = options.scheduler
    category = options.category

    # bench_args come from options.bench_args (already a dict) — no key=value parsing needed at this layer
    user_bench_args: dict = dict(options.bench_args) if options.bench_args else {}

    # Translate legacy fresh bool to the new ResumeMode axis when caller provided a FRESH mode
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
        recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name, resolve=False)
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

    if "api_key" not in bench_args and api_key_env:
        api_key = v.get(api_key_env)
        if not api_key:
            from scitrera_app_framework import add_env_file_source

            try:
                add_env_file_source(".env", v)
                api_key = v.get(api_key_env)
            except ImportError:
                pass
        if api_key:
            bench_args["api_key"] = api_key
        else:
            emitter.warning("--api-key-env '%s' specified, but not found in environment." % api_key_env)

    effective_timeout = bench_timeout or (bench_spec.timeout if bench_spec else None) or DEFAULT_BENCHMARK_TIMEOUT

    # -----------------------------------------------------------------------
    # 3. Check prerequisites
    # -----------------------------------------------------------------------
    # Skipped for a dry run: --dry-run exists to show what *would* happen
    # without executing anything, so requiring the execution toolchain to be
    # installed defeats it — you could not preview a benchmark from a machine
    # that isn't set up to run one.  A real run still fails closed below.
    if not dry_run:
        missing = fw.check_prerequisites()
        if missing:
            for msg in missing:
                emitter.error("Error: %s" % msg)
            raise BenchmarkFailed("Benchmark prerequisites not met", exit_code=1)

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
    recipe, overrides = _apply_recipe_overrides(
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

    # Resolve hosts — resolve_host_list expects a comma-separated string
    # (the raw CLI token), not a list; join any pre-resolved hosts back to string.
    hosts_str = ",".join(hosts) if hosts else ""
    try:
        host_list = resolve_host_list(hosts_str, None, cluster_name, config, sctx=sctx)
    except api.HostsUnreachable as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e
    cluster_mgr = sctx.cluster_manager

    cluster_cfg = resolve_cluster_config(cluster_name, hosts, None, cluster_mgr)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = cluster_cfg.resolve_transfer_config(config)

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
            # Pass the cluster's shared-cache prefs explicitly: this launch
            # uses explicit hosts and so loses the named-cluster identity
            # that launch_inference would otherwise read them from.
            preserve_model_perms=cluster_cfg.preserve_model_perms,
            skip_model_fan_out=cluster_cfg.skip_model_fan_out,
            rootful=rootful,
            sync_tuning=sync_tuning,
            extra_docker_opts=tuple(executor_args) if executor_args else None,
            recipe_ref=recipe_ref,
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
        serve_port = int(config_chain.get("port") or 8000)
        overrides["port"] = serve_port
        # ``overrides`` are final now, so the intent id is stable — find the
        # deployment that is actually serving rather than assuming the whole
        # cluster is.
        host_list, is_solo, skip_run_cluster_id = _resolve_running_deployment(
            recipe,
            overrides,
            host_list,
            solo=solo,
            cluster=cluster_name,
            sctx=sctx,
            emitter=emitter,
        )

    container_image = runtime.resolve_container(recipe, overrides)

    config_chain = recipe.build_config_chain(overrides)
    effective_tp = int(config_chain.get("tensor_parallel") or 1)

    # ``bench_args`` must be *final* here, before anything reads it.  Three
    # consumers below snapshot it — ``build_task_list`` copies it into each
    # task's ``run_args`` (which is what the scheduler actually renders into a
    # command), ``derive_benchmark_id`` hashes it, and ``BenchmarkRunState``
    # persists it as ``base_args`` for resumes — so a contribution merged after
    # them reaches none of the three.  Merging these two *after* the task list
    # was built is exactly that bug: on the scheduled path (the default) the
    # framework's recipe-derived args were silently dropped, taking
    # ``served_model_name`` (issue #257) and the runtime-resolved ``api_key``
    # with them.  Both use ``setdefault``, so a value the user passed with
    # ``-b`` still wins.
    for k, bv in fw.prepare_benchmark_args(recipe, config_chain, overrides).items():
        bench_args.setdefault(k, bv)

    if (api_key := runtime.resolve_api_key(recipe, overrides)) and "api_key" not in bench_args:
        bench_args["api_key"] = api_key

    # -----------------------------------------------------------------------
    # 5. Display summary
    # -----------------------------------------------------------------------
    from sparkrun import __version__

    emitter.banner("=" * 60)
    emitter.banner("sparkrun v%s — benchmark" % __version__)
    emitter.banner("=" * 60)
    emitter.banner("Recipe:                %s" % recipe.qualified_name)
    emitter.banner("Model:                 %s" % recipe.model)
    emitter.banner("Runtime:               %s" % runtime.runtime_name)
    emitter.banner("Image:                 %s" % container_image)
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

    # VRAM estimate — only if emitter is wired to something (CLI will do it separately)
    # We skip it here to avoid importing the CLI helper; the CLI shell calls it before delegating.

    # -----------------------------------------------------------------------
    # 6–10: Launch, benchmark, stop
    # -----------------------------------------------------------------------
    from sparkrun.core.progress import PROGRESS as _PROGRESS_LEVEL

    bench_result = _InternalBenchmarkResult(recipe_name=recipe_name)
    bench_result.framework = fw

    launched = False
    launch_result = None
    ssh_kwargs = build_ssh_kwargs(config)
    head_host = host_list[0]

    result_file = tempfile.mktemp(suffix=".json", prefix="sparkrun_bench_")

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
    tasks = fw.build_task_list(bench_args, bench_spec.schedule if bench_spec else None)

    # Released in the ``finally`` below, and eagerly on any early raise between
    # acquisition and that block.  Empty (and closing is a no-op) on the
    # unscheduled path, which owns no state directory.
    lock_stack = contextlib.ExitStack()

    if tasks is not None:
        from sparkrun.benchmarking.run_state import (
            BenchmarkRunState,
            StateDirLocked,
            clear_state_dir,
            derive_benchmark_id,
            hold_state_dir,
        )
        from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

        # The cluster_id's intent half — previously all derive_benchmark_id
        # hashed — covers model, port and parallelism, so two recipes differing
        # only in a serve argument (e.g. --speculative-config) collided and
        # resumed into each other's results.  The fingerprint digests the
        # declared serve configuration to separate them; it excludes resolved
        # artifacts and placement, so the ID stays stable across relaunches.
        recipe_fingerprint = derive_recipe_fingerprint(recipe, overrides)

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
        state_dir_str = str(state_dir) if state_dir else "~/.cache/sparkrun/benchmarks/%s" % benchmark_id

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
            existing_state = BenchmarkRunState.load(benchmark_id, cache_dir)
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
                if state_dir and state_dir.exists():
                    clear_state_dir(benchmark_id, cache_dir)
                existing_state = None

            if existing_state is None:
                if resume_mode == ResumeMode.REQUIRED:
                    raise NoResumableState("ResumeMode.REQUIRED but no benchmark state exists for id %s" % benchmark_id)
            elif existing_state.is_complete(len(tasks)):
                if resume_mode == ResumeMode.FRESH:
                    if state_dir and state_dir.exists():
                        clear_state_dir(benchmark_id, cache_dir)
                        logger.debug("Deleted complete benchmark state at %s (--fresh)", state_dir)
                    existing_state = None
                elif _should_remeasure_complete_state(resume_mode, on_complete_state, existing_state):
                    if state_dir and state_dir.exists():
                        clear_state_dir(benchmark_id, cache_dir)
                        logger.debug("Deleted complete benchmark state at %s (user chose re-measure)", state_dir)
                    existing_state = None
                else:
                    emitter.warning(
                        "Prior benchmark state for %s is COMPLETE — re-emitting its recorded results; no requests will be sent. Use --fresh to re-measure."
                        % benchmark_id
                    )
            else:
                if resume_mode == ResumeMode.FRESH:
                    if state_dir and state_dir.exists():
                        clear_state_dir(benchmark_id, cache_dir)
                        logger.debug("Deleted prior benchmark state at %s (--fresh)", state_dir)
                    existing_state = None
                elif resume_mode in (ResumeMode.IF_EXISTS, ResumeMode.REQUIRED):
                    pass
                else:  # AUTO
                    # Library policy: consult the caller-supplied
                    # ``on_prompt_required`` callback to decide whether to resume
                    # incomplete state.  When no callback is given, default to
                    # resume (True) — the console-free default that matches the
                    # prior non-TTY behaviour.  The CLI shell supplies a callback
                    # that renders the interactive ``click.confirm`` prompt, so
                    # the API never imports CLI/console code.
                    if on_prompt_required is not None:
                        prompt_ok = bool(on_prompt_required(existing_state))
                    else:
                        prompt_ok = True
                    if not prompt_ok:
                        if state_dir and state_dir.exists():
                            clear_state_dir(benchmark_id, cache_dir)
                            logger.debug("Deleted prior benchmark state at %s (user chose fresh start)", state_dir)
                        existing_state = None

            if existing_state is not None:
                state = existing_state
                # Any reuse of prior state means some of the numbers below were
                # measured in an earlier session — a fully COMPLETE state emits
                # *only* recorded results.  Record both facts so the exported
                # artifact is self-describing: ``timing`` covers this invocation,
                # ``measured_at`` covers the data (issue #267).
                bench_result.resumed = True
                bench_result.measured_at = existing_state.updated_at or None
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
                if submission_id_for_extras:
                    state.extras["submission_id"] = submission_id_for_extras

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
                bench_result.container_image = container_image

            if "container_image_longterm_ref" in state.extras:
                bench_result.longterm_image_ref = state.extras["container_image_longterm_ref"]
                bench_result.longterm_image_pinned = bool(state.extras.get("container_image_longterm_pinned", True))
        except BaseException:
            # The outer ``finally`` that normally releases the lock is not yet
            # in scope on this path (e.g. ResumeMode.REQUIRED with no state).
            lock_stack.close()
            raise

    try:
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

            launch_result = run_result.launch_result
            if launch_result is not None and launch_result.rc != 0 and not dry_run:
                raise BenchmarkFailed(
                    "inference launch failed (exit code %d)" % launch_result.rc,
                    exit_code=launch_result.rc,
                )

            cluster_id = run_result.cluster_id
            serve_port = run_result.serve_port

            if run_result.serve_command:
                logger.info("Serve command:")
                for line in run_result.serve_command.strip().splitlines():
                    logger.info("  %s", line)
                emitter.info("")

            launched = True
            bench_result.launch_result = launch_result

            if tasks is not None:
                if "container_image_sha" not in state.extras:
                    from sparkrun.orchestration.primitives import resolve_image_sha as _resolve_image_sha

                    sha = _resolve_image_sha(container_image, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
                    if sha:
                        state.extras["container_image_sha"] = sha
                        emitter.info("Pinned image SHA: %s" % sha)
                        state.save(cache_dir)
                    else:
                        logger.debug(
                            "resolve_image_sha returned None for %s; pin will not be enforced on resume",
                            container_image,
                        )

                if "container_image_longterm_ref" not in state.extras and launch_result is not None and launch_result.builder is not None:
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
                            state.save(cache_dir)
                    except Exception:
                        logger.debug("Long-term image resolution failed during pin", exc_info=True)
        else:
            logger.log(_PROGRESS_LEVEL, "Step 1/3: Skipping inference launch (--skip-run)")

        # -----------------------------------------------------------------------
        # 7. Wait for readiness and build target URL
        # -----------------------------------------------------------------------
        if is_local_host(head_host):
            target_ip = "127.0.0.1"
        else:
            if dry_run:
                target_ip = "<HEAD_IP>"
            else:
                try:
                    target_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
                except RuntimeError as e:
                    if launched and not no_stop:
                        _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                    raise BenchmarkFailed("Error detecting head IP: %s" % e, exit_code=1) from e

        if not dry_run and not skip_run:
            logger.log(_PROGRESS_LEVEL, "Waiting for inference server on %s:%d...", head_host, serve_port)
            logger.log(_PROGRESS_LEVEL, "Note that this could take ~5 minutes!")
            # Shared with ``sparkrun run`` / ``proxy load`` rather than
            # reimplemented: the two-stage wait is what produces the
            # container-start → serving figure, and a second copy of it with
            # its own retry budgets would make that number incomparable
            # between `run` and `benchmark`.  The budgets stay this path's
            # own (a benchmark is unattended, so it can afford to wait past
            # the interactive default before calling a launch dead).
            readiness = wait_for_endpoint_ready(
                runtime=runtime,
                cluster_id=cluster_id,
                host_list=host_list,
                is_solo=is_solo,
                port=serve_port,
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
                port_timeout_s=3600.0,
                port_retry_interval=5,
                health_timeout_s=1800.0,
                health_retry_interval=5,
                timeline=launch_result.timeline if launch_result is not None else None,
            )
            bench_result.readiness = readiness
            if not readiness.ready:
                if launched and not no_stop:
                    _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                if readiness.reason == "port":
                    raise BenchmarkFailed("Error: inference server did not become ready", exit_code=1)
                raise BenchmarkFailed("Error: inference server health check timed out", exit_code=1)
            logger.log(
                _PROGRESS_LEVEL,
                "Inference server ready (%.1fs to port, %.1fs to healthy).",
                readiness.port_wait_s,
                readiness.health_wait_s,
            )
        elif dry_run:
            emitter.info("[dry-run] Would wait for inference server on %s:%d" % (head_host, serve_port))

        base_url = "http://%s:%d/v1" % (target_ip, serve_port)

        # -----------------------------------------------------------------------
        # 8. Run benchmark
        # -----------------------------------------------------------------------
        emitter.info("")
        logger.log(_PROGRESS_LEVEL, "Step 2/3: Running benchmark (%s)...", fw.framework_name)

        est_tests = fw.estimate_test_count(bench_args)
        if est_tests is not None:
            logger.info("Estimated test iterations: %d", est_tests)

        stdout_text = ""
        stderr_text = ""

        if tasks is not None:
            # Scheduled execution path
            bench_result.profile = profile
            bench_result.benchmark_args = bench_args

            if dry_run:
                emitter.info("[dry-run] Would execute %d scheduled benchmark tasks via scheduler" % len(tasks))
                for i, t in enumerate(tasks):
                    emitter.info("[dry-run]   task %d: %s" % (i, t.label))
            else:
                from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI
                from sparkrun.benchmarking.scheduler import run_schedule

                title = _benchmark_title(recipe.name, profile)

                with BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=benchmark_id, fw=fw, title=title) as pui:
                    sched_result = run_schedule(
                        fw=fw,
                        tasks=tasks,
                        state=state,
                        target_url=base_url,
                        model=recipe.model,
                        timeout=effective_timeout,
                        progress_ui=pui,
                        cache_dir=cache_dir,
                        exit_on_first_fail=exit_on_first_fail,
                        skip_run=skip_run,
                    )

                consolidated = sched_result.consolidated

                if state_dir:
                    consolidated_path = _write_consolidated(state_dir, consolidated)
                    result_file_for_parse = str(consolidated_path)
                else:
                    result_file_for_parse = result_file

                if not sched_result.success:
                    emitter.info("")
                    emitter.info("Benchmark incomplete; you can resume later")
                    if launched and not no_stop:
                        emitter.info("")
                        emitter.info("Stopping inference...")
                        _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                        emitter.info("Inference stopped.")
                    raise BenchmarkFailed("Benchmark incomplete; schedule did not complete", exit_code=1)

                stdout_text = json.dumps(consolidated)
                bench_result.end_time = datetime.now(tz=timezone.utc)
                bench_result.start_time = bench_result.start_time or datetime.now(tz=timezone.utc)
        else:
            # Legacy single-call subprocess path
            bench_cmd = fw.build_benchmark_command(
                target_url=base_url,
                model=recipe.model,
                args=bench_args,
                result_file=result_file,
            )
            bench_result.profile = profile
            bench_result.benchmark_args = bench_args

            logger.info("Benchmark command:")
            logger.info("  %s", " ".join(bench_cmd))
            emitter.info("")

            if dry_run:
                emitter.info("[dry-run] Would execute benchmark command")
            else:
                emitter.info("--- benchmark output ---")
                bench_start = time.monotonic()
                bench_result.start_time = datetime.now(tz=timezone.utc)
                bench_env = os.environ.copy()
                bench_env["PYTHONUNBUFFERED"] = "1"
                try:
                    with subprocess.Popen(
                        bench_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        env=bench_env,
                    ) as proc:
                        stdout_lines: list[str] = []
                        for line in proc.stdout:
                            emitter.info(line.rstrip("\n"))
                            stdout_lines.append(line)

                        try:
                            proc.wait(timeout=effective_timeout)
                        except subprocess.TimeoutExpired as exc:
                            proc.kill()
                            proc.wait()
                            raise BenchmarkFailed(
                                "Error: benchmark timed out after %d seconds" % effective_timeout,
                                exit_code=1,
                            ) from exc

                        stdout_text = "".join(stdout_lines)
                        stderr_text = proc.stderr.read()

                        elapsed = time.monotonic() - bench_start
                        bench_result.end_time = datetime.now(tz=timezone.utc)
                        emitter.info("--- end benchmark output ---")
                        emitter.info("")

                        if proc.returncode != 0:
                            emitter.warning("benchmark exited with code %d (%.0fs elapsed)" % (proc.returncode, elapsed))
                            if stderr_text:
                                emitter.warning("stderr: %s" % stderr_text[:500])
                            if exit_on_first_fail:
                                emitter.warning("Skipping result export (--exit-on-first-fail set and benchmark failed).")
                                if launched and not no_stop:
                                    emitter.info("")
                                    emitter.info("Stopping inference...")
                                    _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                                    emitter.info("Inference stopped.")
                                raise BenchmarkFailed(
                                    "benchmark exited with code %d" % proc.returncode,
                                    exit_code=proc.returncode,
                                )
                        else:
                            emitter.info("Benchmark completed successfully (%.0fs elapsed)." % elapsed)
                except FileNotFoundError as exc:
                    if launched and not no_stop:
                        _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                    raise BenchmarkFailed(
                        "Error: benchmark command not found: %s" % bench_cmd[0],
                        exit_code=1,
                    ) from exc

            result_file_for_parse = result_file

        # -----------------------------------------------------------------------
        # 9. Parse and export results
        # -----------------------------------------------------------------------
        if not dry_run:
            _parse_result_file = result_file_for_parse if tasks is not None else result_file
            results = fw.parse_results(stdout_text, stderr_text, result_file=_parse_result_file)
            bench_result.results = results

            # A framework that failed every request but exited 0 must not be
            # reported as a completed benchmark.  The framework's own output is
            # already captured per task; name it, because that is where the
            # cause is (an HTTP status and body, in llama-benchy's case) and
            # nothing else surfaces it.
            if fw.measured_nothing(results):
                where = "%s/runs/" % state_dir_str if tasks is not None else (_parse_result_file or "the benchmark output")
                raise BenchmarkFailed(
                    "benchmark produced no measurements — every request appears to have failed. "
                    "%s exited successfully, so the cause is in its output: %s" % (fw.framework_name, where),
                    exit_code=1,
                )

            rows = results.get("rows", [])
            if rows:
                emitter.info("")
                emitter.info("Results: %d test row(s) collected" % len(rows))

            if export_results_files:
                if not output_file:
                    profile_slug = profile.replace("/", "_").replace("@", "") if profile else "default"
                    effective_pp = int(config_chain.get("pipeline_parallel") or 1)
                    pp_suffix = "_pp%d" % effective_pp if effective_pp > 1 else ""

                    out_dir = config.default_benchmark_output_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_file = str(
                        out_dir
                        / (
                            "benchmark_%s_%s_tp%d%s.yaml"
                            % (
                                recipe.name.replace("/", "_"),
                                profile_slug,
                                effective_tp,
                                pp_suffix,
                            )
                        )
                    )

                export_results(
                    recipe=recipe,
                    hosts=host_list,
                    tp=effective_tp,
                    cluster_id=cluster_id,
                    framework_name=fw.framework_name,
                    profile_name=profile,
                    args=bench_args,
                    results=results,
                    output_path=output_file,
                    runtime_info=launch_result.runtime_info if launch_result else None,
                    resumed=bench_result.resumed,
                    measured_at=bench_result.measured_at,
                )
                emitter.info("Results saved to: %s" % output_file)
                bench_result.output_yaml = output_file

                written_paths = _emit_results_outputs(results, Path(output_file), emitter)
                if "csv" in written_paths:
                    bench_result.output_csv = str(written_paths["csv"])
                if "json" in written_paths:
                    bench_result.output_json = str(written_paths["json"])
        else:
            emitter.info("[dry-run] Would parse and export results to: %s" % (output_file or "benchmark_<recipe>_<framework>.yaml"))

        # -----------------------------------------------------------------------
        # 10. Stop inference (unless --no-stop)
        # -----------------------------------------------------------------------
        if launched and not no_stop:
            emitter.info("")
            logger.log(_PROGRESS_LEVEL, "Step 3/3: Stopping inference...")
            _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
            logger.log(_PROGRESS_LEVEL, "Inference stopped.")
        elif no_stop:
            emitter.info("")
            logger.log(_PROGRESS_LEVEL, "Step 3/3: Skipping inference stop (--no-stop)")
        elif skip_run:
            emitter.info("")
            logger.log(_PROGRESS_LEVEL, "Step 3/3: Skipping inference stop (--skip-run)")

        emitter.info("")
        logger.log(_PROGRESS_LEVEL, "Benchmark complete.")
        bench_result.success = True

    except KeyboardInterrupt:
        emitter.info("")
        emitter.info("Interrupted.")
        if tasks is not None:
            emitter.info("State preserved so that you can resume later")
        if not no_stop and not skip_run:
            emitter.info("Stopping inference (cleaning up containers)...")
            _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
            emitter.info("Inference stopped.")
        raise
    finally:
        try:
            os.unlink(result_file)
        except OSError:
            pass
        lock_stack.close()

    return bench_result


def _emit_results_outputs(results: dict[str, Any], base_path: Path, emitter: _ProgressEmitter) -> dict[str, Path]:
    """Write json/csv variants of ``base_path`` and emit the artifact paths.

    Returns a mapping from format (``"json"``, ``"csv"``) to the written path.
    """
    writers = {
        "json": lambda data, path: path.write_text(json.dumps(data, indent=2)),
        "csv": lambda data, path: path.write_text(data),
    }
    written: dict[str, Path] = {}
    for fmt, writer in writers.items():
        payload = results.get(fmt)
        if not payload:
            continue
        out = base_path.with_suffix("." + fmt)
        writer(payload, out)
        emitter.info("%s output: %s" % (fmt.upper(), out))
        written[fmt] = out
    return written


def _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=None, emitter: _ProgressEmitter | None = None):
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
        api.stop(
            cluster_id=cluster_id,
            hosts=tuple(host_list) if host_list else None,
            sctx=sctx,
        )
    except Exception as e:
        logger.warning("Failed to stop inference: %s", e)
        if emitter is not None:
            emitter.warning("failed to stop inference: %s" % e)


# ---------------------------------------------------------------------------
# Resume orchestration
# ---------------------------------------------------------------------------


def resume_benchmark(
    benchmark_id: str,
    *,
    dry_run: bool = False,
    sctx: "SparkrunContext | None" = None,
    emitter: _ProgressEmitter | None = None,
) -> dict[str, Any]:
    """Resume a paused benchmark by id and return the parsed ``results`` dict.

    Full orchestration (recipe reload, host reconstruction from job
    metadata, IP detection, framework rebuild, ``run_schedule``, export,
    multi-format output) lifted out of ``cli._benchmark._resume_benchmark_run``
    so library callers get the flow with no Click / sys.exit coupling.

    The optional *emitter* surfaces banner / info lines; pass
    ``_NullProgressEmitter()`` (the default) for headless execution.  Writes
    ``consolidated.json``, ``result.yaml``, and the per-format output files
    to disk and returns the ``results`` mapping (keys: ``rows``, ``csv``,
    ``json``, etc.).

    Raises:
        NoResumableState: No state for *benchmark_id*, or the inference
            cluster is no longer running.
        BenchmarkFailed: Already-complete benchmark (nothing to resume),
            framework lookup failure, unschedulable framework, head-IP
            detection failure, or an incomplete schedule.
        KeyboardInterrupt: Re-raised after state is preserved.
    """
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
        return _resume_locked(
            benchmark_id,
            dry_run=dry_run,
            emitter=emitter,
            config=config,
            cache_dir=cache_dir,
        )


def _resume_locked(
    benchmark_id: str,
    *,
    dry_run: bool,
    emitter: _ProgressEmitter,
    config,
    cache_dir: str | None,
) -> dict[str, Any]:
    """Body of :func:`resume_benchmark`, run while holding the state-dir lock."""
    import yaml as _yaml

    from sparkrun.api._errors import NoResumableState
    from sparkrun.benchmarking.base import export_results
    from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import run_schedule
    from sparkrun.core.bootstrap import get_benchmarking_framework
    from sparkrun.core.resolve import load_recipe
    from sparkrun.orchestration.job_metadata import check_job_running, load_job_metadata
    from sparkrun.orchestration.primitives import build_ssh_kwargs, detect_host_ip
    from sparkrun.utils import is_local_host

    # Load existing state
    state = BenchmarkRunState.load(benchmark_id, cache_dir)
    if state is None:
        raise NoResumableState("no benchmark state found for id: %s" % benchmark_id)

    if state.is_complete(len(state.schedule)):
        raise BenchmarkFailed("Benchmark %s is already complete. Nothing to resume." % benchmark_id, exit_code=0)

    # Snapshot before ``run_schedule`` starts saving: this is when the tasks
    # already recorded in the state were measured.  Read afterwards it would
    # be ~now for every resume, which is exactly the conflation ``measured_at``
    # exists to prevent.
    prior_measured_at = state.updated_at or None

    # Reconstruct recipe
    recipe_name = state.recipe_qualified_name
    from sparkrun.core.recipe import RecipeError

    try:
        recipe, _recipe_path, _registry_mgr = load_recipe(config, recipe_name, resolve=False)
    except RecipeError as e:
        raise BenchmarkFailed("could not reload recipe %r: %s" % (recipe_name, e), exit_code=1) from e

    # Reconstruct hosts from job metadata
    meta = load_job_metadata(state.cluster_id, cache_dir=cache_dir)
    if not meta or not meta.get("hosts"):
        raise NoResumableState(
            "no job metadata found for cluster_id %r.\n"
            "Please relaunch inference with `sparkrun run` and then retry resume." % state.cluster_id
        )
    hosts = meta["hosts"]

    # Check if inference is currently running
    ssh_kwargs = build_ssh_kwargs(config)
    job_status = check_job_running(cluster_id=state.cluster_id, hosts=hosts, ssh_kwargs=ssh_kwargs)
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

    # Reconstruct framework
    try:
        fw = get_benchmarking_framework(state.framework)
    except ValueError as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e

    # Rebuild tasks from saved state
    tasks = fw.build_task_list(state.base_args, state.schedule)
    if tasks is None:
        raise BenchmarkFailed(
            "framework %r does not support scheduled execution (build_task_list returned None)" % state.framework,
            exit_code=1,
        )

    effective_timeout = DEFAULT_BENCHMARK_TIMEOUT

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

    try:
        with BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=benchmark_id, fw=fw, title=title) as pui:
            sched_result = run_schedule(
                fw=fw,
                tasks=tasks,
                state=state,
                target_url=base_url,
                model=recipe.model,
                timeout=effective_timeout,
                progress_ui=pui,
                cache_dir=cache_dir,
                exit_on_first_fail=False,
                skip_run=True,  # inference already running; treat first task as needing warmup by session logic
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

        # Export results
        stdout_text = json.dumps(consolidated)
        results = fw.parse_results(stdout_text, "", result_file=str(consolidated_path))

        overrides = meta.get("overrides") or {}
        effective_tp = int(overrides.get("tensor_parallel") or meta.get("tensor_parallel") or 1)

        profile_slug = state.profile.replace("/", "_").replace("@", "") if state.profile else "default"
        effective_pp = int(overrides.get("pipeline_parallel") or meta.get("pipeline_parallel") or 1)
        pp_suffix = "_pp%d" % effective_pp if effective_pp > 1 else ""

        if config:
            out_dir = config.default_benchmark_output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(
                out_dir / ("benchmark_%s_%s_tp%d%s.yaml" % (recipe.name.replace("/", "_"), profile_slug, effective_tp, pp_suffix))
            )
        else:
            output_file = "benchmark_%s_%s_tp%d%s.yaml" % (recipe.name.replace("/", "_"), profile_slug, effective_tp, pp_suffix)

        export_results(
            recipe=recipe,
            hosts=hosts,
            tp=effective_tp,
            cluster_id=state.cluster_id,
            framework_name=fw.framework_name,
            profile_name=state.profile,
            args=state.base_args,
            results=results,
            output_path=output_file,
            runtime_info=None,
            resumed=True,
            measured_at=prior_measured_at,
        )
        emitter.info("Results saved to: %s" % output_file)

        # Write additional formats
        _emit_results_outputs(results, Path(output_file), emitter)

        # Save result.yaml in state dir too
        result_yaml_path = state.state_dir(cache_dir) / "result.yaml"
        with open(result_yaml_path, "w") as _fh:
            _yaml.safe_dump(results, _fh, default_flow_style=False)

        return results

    except KeyboardInterrupt:
        emitter.info("")
        emitter.info("Interrupted. State preserved so that you can resume later.")
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
    import dataclasses

    sctx = resolve_sctx(sctx)

    # Apply arena defaults: when options.arena is True, supply the pinned profile
    # and performance category when the caller has not specified them explicitly.
    # Auth and upload are CLI-only concerns; the API caller is responsible for those.
    effective_options = options
    if options.arena:
        needs_profile = not options.profile
        needs_category = not options.category
        if needs_profile or needs_category:
            from sparkrun.core.benchmark_profiles import ARENA_BENCHMARK_PROFILE

            effective_options = dataclasses.replace(
                options,
                profile=options.profile or ARENA_BENCHMARK_PROFILE,
                category=options.category or "performance",
            )

    if effective_options.progress_callback is None:
        emitter: _ProgressEmitter = _NullProgressEmitter()
    else:
        emitter = _CallbackProgressEmitter(effective_options.progress_callback)

    try:
        bench_result = _execute_benchmark(effective_options, sctx=sctx, emitter=emitter)
    except KeyboardInterrupt:
        raise
    except SparkrunError:
        raise
    except Exception as exc:
        raise SparkrunError("benchmark failed: %s" % exc) from exc

    result = _build_result(effective_options, bench_result)
    from sparkrun.telemetry import emit_benchmark_telemetry

    emit_benchmark_telemetry(
        sctx.config,
        result=result,
        options=effective_options,
        recipe=getattr(bench_result, "recipe", None),
    )
    return result


def _build_result(options: BenchmarkOptions, bench_result: Any) -> BenchmarkResult:
    """Translate the internal ``BenchmarkResult`` into the API one."""
    outputs: dict[str, str] = {}
    raw_outputs = getattr(bench_result, "outputs", None) or {}
    for k, v in raw_outputs.items():
        if v is not None:
            outputs[k] = str(v)

    framework_plugin = getattr(bench_result, "framework", None)
    if framework_plugin is not None and not isinstance(framework_plugin, str):
        framework_str = getattr(framework_plugin, "framework_name", None) or str(framework_plugin)
    else:
        framework_str = framework_plugin or options.framework or ""

    category = options.category or ""
    if not category and framework_plugin is not None and not isinstance(framework_plugin, str):
        category = getattr(framework_plugin, "primary_category", "") or ""
    if not category and framework_str:
        try:
            from sparkrun.core.bootstrap import get_benchmarking_framework

            fw = get_benchmarking_framework(framework_str)
            category = getattr(fw, "primary_category", "") or ""
        except Exception:
            logger.debug("benchmark category resolution failed", exc_info=True)

    container_image_raw = getattr(bench_result, "container_image", None)
    container_image_str = str(container_image_raw) if container_image_raw else ""

    return BenchmarkResult(
        success=bool(getattr(bench_result, "success", False)),
        benchmark_id=str(getattr(bench_result, "benchmark_id", "") or ""),
        category=category,
        framework=framework_str,
        profile=getattr(bench_result, "profile", None) or options.profile,
        results=dict(getattr(bench_result, "results", None) or {}),
        outputs=outputs,
        run_result=None,
        cluster_id=str(getattr(bench_result, "cluster_id", "") or ""),
        host_list=tuple(getattr(bench_result, "host_list", ()) or ()),
        container_image=container_image_str,
        container_image_sha=getattr(bench_result, "container_image_sha", None),
        container_image_sha_pinned=bool(getattr(bench_result, "container_image_sha_pinned", False)),
        container_image_longterm_ref=getattr(bench_result, "longterm_image_ref", None),
        container_image_longterm_pinned=bool(getattr(bench_result, "longterm_image_pinned", False)),
        metadata={
            "framework": framework_str,
            "profile": getattr(bench_result, "profile", None) or options.profile,
            "bench_args": dict(getattr(bench_result, "benchmark_args", None) or options.bench_args),
        },
        state_dir=getattr(bench_result, "state_dir", None),
        resumed=bool(getattr(bench_result, "resumed", False)),
        submission_id=getattr(bench_result, "submission_id", None),
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
