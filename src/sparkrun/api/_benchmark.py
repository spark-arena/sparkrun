"""``sparkrun.api.benchmark`` — public Python entry point for benchmark runs.

Step 7: Orchestration body lifted from ``cli._benchmark._run_benchmark`` into
``_execute_benchmark``.  The CLI becomes a thin presentation shell; library
callers get the full flow with no Click / sys.exit coupling.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
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
    from sparkrun.orchestration.primitives import (
        build_ssh_kwargs,
        detect_host_ip,
        wait_for_healthy,
        wait_for_port,
    )
    from sparkrun.core.recipe import (
        expand_recipe_shortcut as _expand_recipe_shortcut,
        is_recipe_url as _is_recipe_url,
        simplify_recipe_ref as _simplify_recipe_ref,
    )
    from sparkrun.core.cluster_manager import resolve_cluster_config
    from sparkrun.core.resolve import apply_recipe_overrides as _apply_recipe_overrides, load_recipe as _load_recipe
    from sparkrun.api._hosts import resolve_effective_hosts, resolve_host_list
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
    image = options.overrides.get("image") if isinstance(options.overrides, dict) else None
    port = options.overrides.get("port") if isinstance(options.overrides, dict) else None

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
    missing = fw.check_prerequisites()
    if missing:
        for msg in missing:
            emitter.error("Error: %s" % msg)
        raise BenchmarkFailed("Benchmark prerequisites not met", exit_code=1)

    # -----------------------------------------------------------------------
    # 4. Build overrides and resolve runtime/hosts
    # -----------------------------------------------------------------------
    recipe, overrides = _apply_recipe_overrides(
        (),  # options tuple (CLI only)
        tensor_parallel=None,
        pipeline_parallel=None,
        data_parallel=None,
        gpu_mem=None,
        max_model_len=None,
        image=image,
        recipe=recipe,
        port=port,
    )

    issues = recipe.validate()
    for issue in issues:
        emitter.warning(issue)

    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e

    runtime_issues = runtime.validate_recipe(recipe)
    for issue in runtime_issues:
        emitter.warning(issue)

    # Resolve hosts — resolve_host_list expects a comma-separated string
    # (the raw CLI token), not a list; join any pre-resolved hosts back to string.
    hosts_str = ",".join(hosts) if hosts else ""
    try:
        host_list = resolve_host_list(hosts_str, None, cluster_name, config, sctx=sctx)
    except api.HostsUnreachable as e:
        raise BenchmarkFailed("Error: %s" % e, exit_code=1) from e
    cluster_mgr = sctx.cluster_manager

    if skip_run:
        is_solo = bool(solo) or recipe.mode == "solo" or len(host_list) <= 1
        if is_solo and len(host_list) > 1:
            host_list = host_list[:1]
    else:
        # Mirror the inputs ``api.run`` will use below (anonymous cluster from the
        # host list → default hardware) and pass the same ``exclude_intent_id`` so
        # this pre-launch estimate places identically to the authoritative
        # scheduling pass inside ``api.run`` — otherwise the banner host list could
        # disagree with what actually launches.
        from sparkrun.orchestration.job_metadata import generate_intent_id

        host_list, is_solo, _notes, _placement = resolve_effective_hosts(
            host_list,
            recipe,
            overrides,
            cluster_def=None,
            runtime=runtime,
            sctx=sctx,
            solo=solo,
            scheduler=scheduler_name,
            exclude_intent_id=generate_intent_id(recipe, overrides),
        )

    cluster_cfg = resolve_cluster_config(cluster_name, hosts, None, cluster_mgr)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = cluster_cfg.resolve_transfer_config(config)

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

    container_image = runtime.resolve_container(recipe, overrides)

    config_chain = recipe.build_config_chain(overrides)
    effective_tp = int(config_chain.get("tensor_parallel") or 1)

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

    cluster_id = _derive_cid(recipe, host_list, overrides=overrides)

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

    if tasks is not None:
        from sparkrun.benchmarking.run_state import BenchmarkRunState, derive_benchmark_id

        benchmark_id = derive_benchmark_id(
            cluster_id,
            fw.framework_name,
            profile,
            bench_args,
            [t.schedule_entry for t in tasks],
        )

        state_dir = (config.cache_dir / "benchmarks" / benchmark_id) if config else None
        state_dir_str = str(state_dir) if state_dir else "~/.cache/sparkrun/benchmarks/%s" % benchmark_id

        emitter.info("Benchmark ID:          %s" % benchmark_id)
        emitter.info("State directory:       %s" % state_dir_str)
        emitter.info("")

        existing_state = BenchmarkRunState.load(benchmark_id, cache_dir)
        if existing_state is None:
            if resume_mode == ResumeMode.REQUIRED:
                raise NoResumableState("ResumeMode.REQUIRED but no benchmark state exists for id %s" % benchmark_id)
        elif existing_state.is_complete(len(tasks)):
            if resume_mode == ResumeMode.FRESH:
                if state_dir and state_dir.exists():
                    shutil.rmtree(state_dir)
                    logger.debug("Deleted complete benchmark state at %s (--fresh)", state_dir)
                existing_state = None
        else:
            if resume_mode == ResumeMode.FRESH:
                if state_dir and state_dir.exists():
                    shutil.rmtree(state_dir)
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
                        shutil.rmtree(state_dir)
                        logger.debug("Deleted prior benchmark state at %s (user chose fresh start)", state_dir)
                    existing_state = None

        if existing_state is not None:
            state = existing_state
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

    try:
        # -----------------------------------------------------------------------
        # 6. Launch inference (unless --skip-run)
        # -----------------------------------------------------------------------
        if not skip_run:
            logger.log(_PROGRESS_LEVEL, "Step 1/3: Launching inference...")

            run_options = api.RunOptions(
                recipe=recipe,
                hosts=tuple(host_list),
                overrides=dict(overrides),
                solo=is_solo,
                dry_run=dry_run,
                follow=False,
                detached=True,
                trust=None,
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
                run_result = api.run(run_options, sctx=sctx)
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
            head_container = runtime.get_head_container_name(cluster_id, is_solo=is_solo)
            logger.log(_PROGRESS_LEVEL, "Waiting for inference server on %s:%d...", head_host, serve_port)
            logger.log(_PROGRESS_LEVEL, "Note that this could take ~5 minutes!")
            ready = wait_for_port(
                head_host,
                serve_port,
                max_retries=180,
                retry_interval=5,
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
                container_name=head_container,
            )
            if not ready:
                if launched and not no_stop:
                    _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                raise BenchmarkFailed("Error: inference server did not become ready", exit_code=1)

            health_url = "http://%s:%d/v1/models" % (target_ip, serve_port)
            logger.log(_PROGRESS_LEVEL, "Waiting for model to finish loading (%s)...", health_url)
            healthy = wait_for_healthy(
                health_url,
                max_retries=360,
                retry_interval=5,
                dry_run=dry_run,
            )
            if not healthy:
                if launched and not no_stop:
                    _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                raise BenchmarkFailed("Error: inference server health check timed out", exit_code=1)
            logger.log(_PROGRESS_LEVEL, "Inference server ready.")
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

        for k, bv in fw.prepare_benchmark_args(recipe, config_chain, overrides).items():
            bench_args.setdefault(k, bv)

        if (api_key := runtime.resolve_api_key(recipe, overrides)) and "api_key" not in bench_args:
            bench_args["api_key"] = api_key

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
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                            raise BenchmarkFailed(
                                "Error: benchmark timed out after %d seconds" % effective_timeout,
                                exit_code=1,
                            )

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
                except FileNotFoundError:
                    if launched and not no_stop:
                        _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=emitter)
                    raise BenchmarkFailed(
                        "Error: benchmark command not found: %s" % bench_cmd[0],
                        exit_code=1,
                    )

            result_file_for_parse = result_file

        # -----------------------------------------------------------------------
        # 9. Parse and export results
        # -----------------------------------------------------------------------
        if not dry_run:
            _parse_result_file = result_file_for_parse if tasks is not None else result_file
            results = fw.parse_results(stdout_text, stderr_text, result_file=_parse_result_file)
            bench_result.results = results

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

    if emitter is None:
        emitter = _NullProgressEmitter()

    sctx = resolve_sctx(sctx)
    config = sctx.config
    cache_dir = str(config.cache_dir) if config else None

    # Load existing state
    state = BenchmarkRunState.load(benchmark_id, cache_dir)
    if state is None:
        raise NoResumableState("no benchmark state found for id: %s" % benchmark_id)

    if state.is_complete(len(state.schedule)):
        raise BenchmarkFailed("Benchmark %s is already complete. Nothing to resume." % benchmark_id, exit_code=0)

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

    return _build_result(effective_options, bench_result)


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
