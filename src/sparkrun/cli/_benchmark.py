"""sparkrun benchmark command — run benchmarks against inference recipes."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import click

from ._common import (
    PROFILE_NAME,
    RECIPE_NAME,
    _apply_recipe_overrides,
    _display_vram_estimate,
    _expand_recipe_shortcut,
    _get_context,
    _is_recipe_url,
    _load_recipe,
    _resolve_hosts_or_exit,
    _simplify_recipe_ref,
    dry_run_option,
    host_options,
    recipe_override_options,
    resolve_cluster_config,
    validate_and_prepare_hosts,
    HIDE_ADVANCED_OPTIONS,
)

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_TIMEOUT: int = 14400  # 4 hours

if TYPE_CHECKING:
    pass


class _BenchmarkGroup(click.Group):
    """Group that falls back to the 'run' subcommand when no recognized
    subcommand name is present in the argument list (preserves legacy
    `sparkrun benchmark <recipe>` UX, including `benchmark --solo --dry-run
    <recipe>` forms where options precede the positional)."""

    def parse_args(self, ctx, args):
        # If none of the known subcommand names appears as a standalone token in
        # the argument list, prepend "run" so the group routes there.  We scan
        # all args but skip anything that looks like an option value (i.e. the
        # token immediately following a --flag=... would not appear here as a
        # separate element anyway), so a simple scan for tokens matching command
        # names is sufficient.
        if args:
            command_names = set(self.commands)
            has_subcommand = any(a in command_names for a in args if not a.startswith("-"))
            if not has_subcommand:
                args = ["run"] + list(args)
        return super().parse_args(ctx, args)


@click.group(cls=_BenchmarkGroup)
@click.pass_context
def benchmark(ctx):
    """Benchmark an inference recipe.

    Runs the full benchmark flow: launch inference, run benchmark, stop
    inference.

    Manage benchmark profiles via the registry subcommands:

      sparkrun registry list-benchmark-profiles

      sparkrun registry show-benchmark-profile <name>

    Examples:

      sparkrun benchmark qwen3-1.7b-sglang --solo

      sparkrun benchmark qwen3-1.7b-sglang --tp 2 --profile spark-arena-v1

      sparkrun benchmark qwen3-1.7b-sglang -b depth=0,2048,4096 -b tg=32,128

      sparkrun benchmark qwen3-1.7b-sglang --skip-run --solo
    """
    pass


@benchmark.command("run")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option("--solo", is_flag=True, help="Force single-node mode")
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--profile", default=None, type=PROFILE_NAME, help="Benchmark profile name or file path")
@click.option("--framework", default=None, help="Override benchmarking framework (default: llama-benchy)")
@click.option("--output", "output_file", default=None, type=click.Path(), help="Output file for results YAML")
@click.option("-b", "--benchmark-option", "bench_options", multiple=True, help="Override benchmark arg: -b key=value (repeatable)")
@click.option(
    "--exit-on-first-fail/--no-exit-on-first-fail",
    "exit_on_first_fail",
    default=True,
    help="Abort benchmark on first failure and skip saving results (default: enabled)",
)
@click.option("--no-stop", is_flag=True, help="Don't stop inference after benchmarking")
@click.option("--skip-run", is_flag=True, help="Skip launching inference (benchmark existing instance)")
@click.option("--sync-tuning", is_flag=True, help="Sync tuning configs from registries before benchmarking")
@click.option("--rootful", is_flag=True, help="Run with --privileged as root inside container (legacy behavior)")
@click.option(
    "--timeout",
    "bench_timeout",
    type=int,
    default=None,
    help="Benchmark timeout in seconds (default: %d, or from profile)" % DEFAULT_BENCHMARK_TIMEOUT,
)
@click.option("--fresh", is_flag=True, default=False, help="Force fresh start, deleting prior state if any")
@dry_run_option
@click.option(
    "--executor-args",
    multiple=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Arguments passed directly to the container executor (e.g. docker run)",
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def benchmark_run(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    max_model_len,
    options,
    image,
    solo,
    port,
    profile,
    framework,
    output_file,
    bench_options,
    exit_on_first_fail,
    no_stop,
    skip_run,
    sync_tuning,
    rootful,
    bench_timeout,
    fresh,
    dry_run,
    executor_args,
    extra_args,
):
    """Run a benchmark against an inference recipe."""
    return _run_benchmark(
        ctx,
        recipe_name,
        hosts,
        hosts_file,
        cluster_name,
        tensor_parallel,
        pipeline_parallel,
        data_parallel,
        gpu_mem,
        max_model_len,
        options,
        image,
        solo,
        port,
        profile,
        framework,
        output_file,
        bench_options,
        exit_on_first_fail,
        no_stop,
        skip_run,
        sync_tuning,
        rootful,
        bench_timeout,
        dry_run,
        executor_args,
        extra_args,
        fresh=fresh,
    )


@benchmark.command("resume")
@click.argument("benchmark_id")
@dry_run_option
@click.pass_context
def benchmark_resume(ctx, benchmark_id, dry_run):
    """Resume a paused benchmark by id."""
    sctx = _get_context(ctx)
    _resume_benchmark_run(ctx, benchmark_id, dry_run, sctx=sctx)


def _resume_benchmark_run(ctx, benchmark_id: str, dry_run: bool, *, sctx=None):
    """Resume a paused benchmark by id and return the parsed results dict.

    Shared by ``benchmark resume`` and ``arena benchmark resume``.  Writes
    consolidated.json, result.yaml, and the per-format output files to disk and
    returns the ``results`` mapping (keys: ``rows``, ``csv``, ``json``, etc.).
    Returns ``None`` only if we exit early (the caller should treat that as an
    error — sys.exit has already been called).
    """
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import run_schedule
    from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI
    from sparkrun.orchestration.job_metadata import load_job_metadata

    if sctx is None:
        sctx = _get_context(ctx)
    config = sctx.config
    cache_dir = str(config.cache_dir) if config else None

    # Load existing state
    state = BenchmarkRunState.load(benchmark_id, cache_dir)
    if state is None:
        click.echo("Error: no benchmark state found for id: %s" % benchmark_id, err=True)
        sys.exit(1)

    if state.is_complete(len(state.schedule)):
        click.echo("Benchmark %s is already complete. Nothing to resume." % benchmark_id)
        sys.exit(0)

    # Reconstruct recipe
    recipe_name = state.recipe_qualified_name
    try:
        recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name, resolve=False)
    except Exception as e:
        click.echo("Error: could not reload recipe %r: %s" % (recipe_name, e), err=True)
        sys.exit(1)

    # Reconstruct hosts from job metadata
    meta = load_job_metadata(state.cluster_id, cache_dir=cache_dir)
    if not meta or not meta.get("hosts"):
        click.echo(
            "Error: no job metadata found for cluster_id %r.\n"
            "Please relaunch inference with `sparkrun run` and then retry resume." % state.cluster_id,
            err=True,
        )
        sys.exit(1)
    hosts = meta["hosts"]

    # Check if inference is currently running
    from sparkrun.orchestration.job_metadata import check_job_running
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    ssh_kwargs = build_ssh_kwargs(config)
    job_status = check_job_running(cluster_id=state.cluster_id, hosts=hosts, ssh_kwargs=ssh_kwargs)
    if not job_status.running:
        click.echo(
            "Error: inference cluster %r is not currently running.\n"
            "Please relaunch with `sparkrun run %s` first, then retry resume." % (state.cluster_id, recipe_name),
            err=True,
        )
        sys.exit(1)

    # Determine the serving URL
    from sparkrun.utils import is_local_host
    from sparkrun.orchestration.primitives import detect_host_ip

    head_host = hosts[0]
    serve_port = meta.get("port") or 8000

    if is_local_host(head_host):
        target_ip = "127.0.0.1"
    else:
        if dry_run:
            target_ip = "<HEAD_IP>"
        else:
            try:
                target_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
            except RuntimeError as e:
                click.echo("Error detecting head IP: %s" % e, err=True)
                sys.exit(1)

    base_url = "http://%s:%d/v1" % (target_ip, serve_port)

    # Reconstruct framework
    from sparkrun.core.bootstrap import get_benchmarking_framework

    try:
        fw = get_benchmarking_framework(state.framework)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Rebuild tasks from saved state
    tasks = fw.build_task_list(state.base_args, state.schedule)
    if tasks is None:
        click.echo("Error: framework %r does not support scheduled execution (build_task_list returned None)" % state.framework, err=True)
        sys.exit(1)

    effective_timeout = DEFAULT_BENCHMARK_TIMEOUT

    click.echo("=" * 60)
    click.echo("sparkrun — benchmark resume")
    click.echo("=" * 60)
    click.echo("Benchmark ID:          %s" % benchmark_id)
    click.echo("Recipe:                %s" % recipe_name)
    click.echo("Framework:             %s" % state.framework)
    click.echo("Profile:               %s" % (state.profile or "(none)"))
    click.echo("Hosts:                 %s" % ", ".join(hosts))
    click.echo("Completed tasks:       %d / %d" % (len(state.completed_indices), len(tasks)))
    click.echo("State directory:       %s" % state.state_dir(cache_dir))
    click.echo("=" * 60)
    click.echo("")

    title = "%s/%s" % (recipe.name, state.profile) if state.profile else recipe.name

    try:
        with BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=benchmark_id, title=title) as pui:
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
        consolidated_path = state.state_dir(cache_dir) / "consolidated.json"
        consolidated_path.write_text(json.dumps(consolidated, indent=2))

        if not sched_result.success:
            click.echo("")
            click.echo("Benchmark incomplete; you can resume later.")
            sys.exit(1)

        click.echo("")
        click.echo("Benchmark resumed and completed successfully.")

        # Export results
        from sparkrun.benchmarking.base import export_results

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
        else:
            output_file = "benchmark_%s_%s_tp%d%s.yaml" % (
                recipe.name.replace("/", "_"),
                profile_slug,
                effective_tp,
                pp_suffix,
            )

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
        click.echo("Results saved to: %s" % output_file)

        # Write additional formats
        from pathlib import Path

        _OUTPUT_WRITERS = {
            "json": lambda data, path: path.write_text(__import__("json").dumps(data, indent=2)),
            "csv": lambda data, path: path.write_text(data),
        }
        for fmt, writer in _OUTPUT_WRITERS.items():
            content = results.get(fmt)
            if not content:
                continue
            fmt_path = Path(output_file).with_suffix(".%s" % fmt)
            writer(content, fmt_path)
            click.echo("%s results saved to: %s" % (fmt.upper(), fmt_path))

        # Save result.yaml in state dir too
        result_yaml_path = state.state_dir(cache_dir) / "result.yaml"
        import yaml as _yaml

        with open(result_yaml_path, "w") as _fh:
            _yaml.safe_dump(results, _fh, default_flow_style=False)

        return results

    except KeyboardInterrupt:
        click.echo("")
        click.echo("Interrupted. State preserved so that you can resume later.")
        sys.exit(130)


def _run_benchmark(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    max_model_len,
    options,
    image,
    solo,
    port,
    profile,
    framework,
    output_file,
    bench_options,
    exit_on_first_fail,
    no_stop,
    skip_run,
    sync_tuning,
    rootful,
    bench_timeout,
    dry_run,
    executor_args,
    extra_args,
    export_results_files=True,
    fresh: bool = False,
    submission_id_for_extras: str | None = None,
):
    """Execute the full benchmark flow: launch inference -> benchmark -> stop.

    Returns a :class:`BenchmarkResult` with output file paths on success.
    """
    from sparkrun.benchmarking.base import export_results, BenchmarkResult
    from ..core.benchmark_profiles import BenchmarkSpec
    from sparkrun.core.bootstrap import get_runtime, get_benchmarking_framework
    from sparkrun.utils import is_local_host
    from sparkrun.core.launcher import launch_inference
    from sparkrun.orchestration.primitives import (
        build_ssh_kwargs,
        detect_host_ip,
        wait_for_healthy,
        wait_for_port,
    )

    bench_result = BenchmarkResult(recipe_name=recipe_name)
    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    # ---------------------------------------------------------------
    # 1. Load recipe
    # ---------------------------------------------------------------
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name, resolve=False)

    _resolved_name = _expand_recipe_shortcut(recipe_name)
    recipe_ref = _simplify_recipe_ref(_resolved_name) if _is_recipe_url(_resolved_name) else None

    # ---------------------------------------------------------------
    # 2. Resolve benchmark configuration
    # ---------------------------------------------------------------
    bench_spec = None
    bench_args: dict = {}

    if profile:
        from ..core.benchmark_profiles import find_benchmark_profile
        from ..core.benchmark_profiles import ProfileAmbiguousError
        from ..core.benchmark_profiles import ProfileError

        try:
            profile_path = find_benchmark_profile(profile, config, registry_mgr)
        except (ProfileError, ProfileAmbiguousError) as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
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
        framework = "llama-benchy"

    try:
        fw = get_benchmarking_framework(framework)

    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    bench_result.framework = fw

    # Build layered bench args
    passthrough_layer: dict = {}
    if fw.passthrough_args:
        recipe_bench_block = recipe._raw.get("benchmark", {}) if hasattr(recipe, "_raw") else {}
        if isinstance(recipe_bench_block, dict):
            for key in fw.passthrough_args:
                if key in recipe_bench_block:
                    passthrough_layer[key] = recipe_bench_block[key]

    bench_args = {**fw.get_default_args(), **passthrough_layer, **bench_args}

    for opt_str in bench_options:
        if "=" not in opt_str:
            click.echo("Error: --bench-option must be key=value, got: %s" % opt_str, err=True)
            sys.exit(1)
        key, _, val = opt_str.partition("=")
        bench_args[key.strip()] = fw.interpret_arg(key.strip(), val.strip())

    if exit_on_first_fail:
        bench_args["exit_on_first_fail"] = True

    effective_timeout = bench_timeout or (bench_spec.timeout if bench_spec else None) or DEFAULT_BENCHMARK_TIMEOUT

    # ---------------------------------------------------------------
    # 3. Check prerequisites
    # ---------------------------------------------------------------
    missing = fw.check_prerequisites()
    if missing:
        for msg in missing:
            click.echo("Error: %s" % msg, err=True)
        sys.exit(1)

    # ---------------------------------------------------------------
    # 4. Build overrides and resolve runtime/hosts
    # ---------------------------------------------------------------
    recipe, overrides = _apply_recipe_overrides(
        options,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
        recipe=recipe,
        # custom
        port=port,
    )

    issues = recipe.validate()
    for issue in issues:
        click.echo("Warning: %s" % issue, err=True)

    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    runtime_issues = runtime.validate_recipe(recipe)
    for issue in runtime_issues:
        click.echo("Warning: %s" % issue, err=True)

    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)

    # Node count validation, max_nodes enforcement, and solo mode determination
    host_list, is_solo = validate_and_prepare_hosts(host_list, recipe, overrides, runtime, solo=solo)

    # Resolve cache dir, transfer mode, and transfer interface from cluster config
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = cluster_cfg.resolve_transfer_config(config)

    # For --skip-run, resolve port without auto-increment (server already listening)
    if skip_run:
        config_chain = recipe.build_config_chain(overrides)
        serve_port = int(config_chain.get("port") or 8000)
        overrides["port"] = serve_port
        # Derive cluster_id for skip-run (needed for stop)
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        cluster_id = generate_cluster_id(recipe, host_list, overrides=overrides)

    container_image = runtime.resolve_container(recipe, overrides)

    # Rebuild config chain for TP (used in output filename)
    config_chain = recipe.build_config_chain(overrides)
    effective_tp = int(config_chain.get("tensor_parallel") or 1)

    # ---------------------------------------------------------------
    # 5. Display summary
    # ---------------------------------------------------------------
    from sparkrun import __version__

    click.echo("=" * 60)
    click.echo("sparkrun v%s — benchmark" % __version__)
    click.echo("=" * 60)
    click.echo("Recipe:                %s" % recipe.qualified_name)
    click.echo("Model:                 %s" % recipe.model)
    click.echo("Runtime:               %s" % runtime.runtime_name)
    click.echo("Image:                 %s" % container_image)
    click.echo("Benchmark Framework:   %s" % fw.framework_name)
    if profile:
        click.echo("Benchmark Profile:     %s" % profile)
    click.echo("Hosts:                 %s" % ", ".join(host_list))
    click.echo("Mode:                  %s" % ("solo" if is_solo else "cluster (%d nodes)" % len(host_list)))
    click.echo("")
    click.echo("Benchmark args:")
    for k, bv in bench_args.items():
        click.echo("  %-35s %s" % (k + ":", bv))
    click.echo("=" * 60)
    click.echo("")

    _display_vram_estimate(recipe, cli_overrides=overrides, auto_detect=True, cache_dir=local_cache_dir)

    # ---------------------------------------------------------------
    # 6–10: Launch, benchmark, stop — wrapped so Ctrl+C always cleans up
    # ---------------------------------------------------------------
    from sparkrun.core.progress import PROGRESS as _PROGRESS_LEVEL

    launched = False
    launch_result = None
    ssh_kwargs = build_ssh_kwargs(config)
    head_host = host_list[0]

    result_file = tempfile.mktemp(suffix=".json", prefix="sparkrun_bench_")

    # Pre-compute cluster_id so we can clean up containers even if
    # launch_inference is interrupted before it returns (e.g. Ctrl+C
    # during wait_for_port inside the runtime).
    from sparkrun.orchestration.job_metadata import generate_cluster_id as _gen_cid

    cluster_id = _gen_cid(recipe, host_list, overrides=overrides)

    # Store recipe/cluster context on bench_result so callers (e.g. arena
    # benchmark) can generate metadata even when --skip-run skips the launch.
    bench_result.recipe = recipe
    bench_result.overrides = overrides
    bench_result.cluster_id = cluster_id
    bench_result.host_list = host_list
    bench_result.container_image = container_image

    # ---------------------------------------------------------------
    # Scheduled execution setup (build task list before launch)
    # ---------------------------------------------------------------
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

        click.echo("Benchmark ID:          %s" % benchmark_id)
        click.echo("State directory:       %s" % state_dir_str)
        click.echo("")

        # Check for existing state
        existing_state = BenchmarkRunState.load(benchmark_id, cache_dir)
        if existing_state is not None and not existing_state.is_complete(len(tasks)):
            if fresh:
                # Force fresh: delete prior state directory
                if state_dir and state_dir.exists():
                    shutil.rmtree(state_dir)
                    logger.debug("Deleted prior benchmark state at %s (--fresh)", state_dir)
                existing_state = None
            else:
                resume_answer = click.confirm(
                    "Found existing incomplete benchmark state (%d/%d tasks done). Resume?"
                    % (len(existing_state.completed_indices), len(tasks)),
                    default=True,
                )
                if not resume_answer:
                    if state_dir and state_dir.exists():
                        shutil.rmtree(state_dir)
                        logger.debug("Deleted prior benchmark state at %s (user chose fresh start)", state_dir)
                    existing_state = None

        # Build or load state
        if existing_state is not None:
            state = existing_state
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

        # Pin framework version on first creation so subsequent calls (and
        # resumed sessions) all use the same tool release.  Skip detection
        # if a version is already pinned (resume path) or detection fails.
        if "framework_version" not in state.extras:
            detected_version = fw.detect_version()
            if detected_version:
                state.extras["framework_version"] = detected_version
                click.echo("Pinned %s version: %s" % (fw.framework_name, detected_version))
            else:
                logger.debug("No framework version detected for %s; version will float", fw.framework_name)
        else:
            click.echo("Using pinned %s version: %s" % (fw.framework_name, state.extras["framework_version"]))

    try:
        # ---------------------------------------------------------------
        # 6. Launch inference (unless --skip-run)
        # ---------------------------------------------------------------
        if not skip_run:
            logger.log(_PROGRESS_LEVEL, "Step 1/3: Launching inference...")

            launch_result = launch_inference(
                recipe=recipe,
                runtime=runtime,
                host_list=host_list,
                overrides=overrides,
                sctx=sctx,
                is_solo=is_solo,
                cache_dir=remote_cache_dir,
                local_cache_dir=local_cache_dir,
                transfer_mode=effective_transfer_mode,
                transfer_interface=effective_transfer_interface,
                recipe_ref=recipe_ref,
                registry_mgr=registry_mgr,
                auto_port=True,
                sync_tuning=sync_tuning,
                dry_run=dry_run,
                detached=True,
                rootless=not rootful,
                auto_user=not rootful,
                extra_docker_opts=list(executor_args) if executor_args else None,
            )

            if launch_result.rc != 0 and not dry_run:
                click.echo("Error: inference launch failed (exit code %d)" % launch_result.rc, err=True)
                sys.exit(launch_result.rc)

            cluster_id = launch_result.cluster_id
            serve_port = launch_result.serve_port

            logger.info("Serve command:")
            for line in launch_result.serve_command.strip().splitlines():
                logger.info("  %s", line)
            click.echo("")

            launched = True
            bench_result.launch_result = launch_result
        else:
            logger.log(_PROGRESS_LEVEL, "Step 1/3: Skipping inference launch (--skip-run)")

        # ---------------------------------------------------------------
        # 7. Wait for readiness and build target URL
        # ---------------------------------------------------------------
        if is_local_host(head_host):
            target_ip = "127.0.0.1"
        else:
            if dry_run:
                target_ip = "<HEAD_IP>"
            else:
                try:
                    target_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
                except RuntimeError as e:
                    click.echo("Error detecting head IP: %s" % e, err=True)
                    if launched and not no_stop:
                        _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                    sys.exit(1)

        if not dry_run and not skip_run:
            head_container = runtime.get_head_container_name(cluster_id, is_solo=is_solo)
            logger.log(_PROGRESS_LEVEL, "Waiting for inference server on %s:%d...", head_host, serve_port)
            logger.log(_PROGRESS_LEVEL, "Note that this could take ~5 minutes!")
            ready = wait_for_port(
                head_host,
                serve_port,
                max_retries=180,
                retry_interval=5,  # TODO: maybe make this dynamic with model size somewhat??
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
                container_name=head_container,
            )
            if not ready:
                click.echo("Error: inference server did not become ready", err=True)
                if launched and not no_stop:
                    _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                sys.exit(1)

            health_url = "http://%s:%d/v1/models" % (target_ip, serve_port)
            logger.log(_PROGRESS_LEVEL, "Waiting for model to finish loading (%s)...", health_url)
            healthy = wait_for_healthy(
                health_url,
                max_retries=360,
                retry_interval=5,
                dry_run=dry_run,
            )
            if not healthy:
                click.echo("Error: inference server health check timed out", err=True)
                if launched and not no_stop:
                    _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                sys.exit(1)
            logger.log(_PROGRESS_LEVEL, "Inference server ready.")
        elif dry_run:
            click.echo("[dry-run] Would wait for inference server on %s:%d" % (head_host, serve_port))

        base_url = "http://%s:%d/v1" % (target_ip, serve_port)

        # -----------------------------------------------------------
        # 8. Run benchmark
        # -----------------------------------------------------------
        click.echo("")
        logger.log(_PROGRESS_LEVEL, "Step 2/3: Running benchmark (%s)...", fw.framework_name)

        est_tests = fw.estimate_test_count(bench_args)
        if est_tests is not None:
            logger.info("Estimated test iterations: %d", est_tests)

        # Pass served_model_name as an argument if defined, satisfying llama-benchy's
        # requirement to maintain the original huggingface model ID for tokenization.
        # Check config_chain to capture both CLI overrides and recipe definitions natively.
        served_model_name = config_chain.get("served_model_name")
        if served_model_name and "served_model_name" not in bench_args:
            bench_args["served_model_name"] = served_model_name

        stdout_text = ""
        stderr_text = ""

        if tasks is not None:
            # -----------------------------------------------------------
            # Scheduled execution path
            # -----------------------------------------------------------
            bench_result.profile = profile
            bench_result.benchmark_args = bench_args

            if dry_run:
                click.echo("[dry-run] Would execute %d scheduled benchmark tasks via scheduler" % len(tasks))
                for i, t in enumerate(tasks):
                    click.echo("[dry-run]   task %d: %s" % (i, t.label))
            else:
                from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI
                from sparkrun.benchmarking.scheduler import run_schedule

                title = "%s/%s" % (recipe.name, profile) if profile else recipe.name

                with BenchmarkProgressUI(total_tasks=len(tasks), benchmark_id=benchmark_id, title=title) as pui:
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

                # Write consolidated.json to state dir
                if state_dir:
                    state_dir.mkdir(parents=True, exist_ok=True)
                    consolidated_path = state_dir / "consolidated.json"
                    consolidated_path.write_text(json.dumps(consolidated, indent=2))
                    result_file_for_parse = str(consolidated_path)
                else:
                    result_file_for_parse = result_file

                if not sched_result.success:
                    click.echo("")
                    click.echo("Benchmark incomplete; you can resume later")
                    if launched and not no_stop:
                        click.echo("")
                        click.echo("Stopping inference...")
                        _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                        click.echo("Inference stopped.")
                    sys.exit(1)

                # Use consolidated JSON as "stdout_text" for parse_results
                stdout_text = json.dumps(consolidated)
                bench_result.end_time = datetime.now(tz=timezone.utc)
                bench_result.start_time = bench_result.start_time or datetime.now(tz=timezone.utc)
        else:
            # -----------------------------------------------------------
            # Legacy single-call subprocess path
            # -----------------------------------------------------------
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
            click.echo("")

            if dry_run:
                click.echo("[dry-run] Would execute benchmark command")
            else:
                import time

                click.echo("--- benchmark output ---")
                bench_start = time.monotonic()
                bench_result.start_time = datetime.now(tz=timezone.utc)
                try:
                    import os

                    bench_env = os.environ.copy()
                    bench_env["PYTHONUNBUFFERED"] = "1"
                    proc = subprocess.Popen(
                        bench_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        env=bench_env,
                    )

                    stdout_lines: list[str] = []
                    for line in proc.stdout:
                        click.echo(line, nl=False)
                        stdout_lines.append(line)

                    proc.wait(timeout=effective_timeout)
                    stdout_text = "".join(stdout_lines)
                    stderr_text = proc.stderr.read()

                    elapsed = time.monotonic() - bench_start
                    bench_result.end_time = datetime.now(tz=timezone.utc)
                    click.echo("--- end benchmark output ---")
                    click.echo("")

                    if proc.returncode != 0:
                        click.echo("Warning: benchmark exited with code %d (%.0fs elapsed)" % (proc.returncode, elapsed), err=True)
                        if stderr_text:
                            click.echo("stderr: %s" % stderr_text[:500], err=True)
                        if exit_on_first_fail:
                            click.echo("Skipping result export (--exit-on-first-fail set and benchmark failed).", err=True)
                            if launched and not no_stop:
                                click.echo("")
                                click.echo("Stopping inference...")
                                _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                                click.echo("Inference stopped.")
                            sys.exit(proc.returncode)
                    else:
                        click.echo("Benchmark completed successfully (%.0fs elapsed)." % elapsed)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    click.echo("Error: benchmark timed out after %d seconds" % effective_timeout, err=True)
                    stdout_text = ""
                    stderr_text = ""
                except FileNotFoundError:
                    click.echo("Error: benchmark command not found: %s" % bench_cmd[0], err=True)
                    if launched and not no_stop:
                        _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                    sys.exit(1)

            result_file_for_parse = result_file

        # -----------------------------------------------------------
        # 9. Parse and export results
        # -----------------------------------------------------------
        if not dry_run:
            _parse_result_file = result_file_for_parse if tasks is not None else result_file
            results = fw.parse_results(stdout_text, stderr_text, result_file=_parse_result_file)
            bench_result.results = results

            rows = results.get("rows", [])
            if rows:
                click.echo("")
                click.echo("Results: %d test row(s) collected" % len(rows))

            if export_results_files:
                # create a standard output file basis
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
                click.echo("Results saved to: %s" % output_file)
                bench_result.output_yaml = output_file

                from pathlib import Path

                _OUTPUT_WRITERS = {
                    "json": lambda data, path: path.write_text(__import__("json").dumps(data, indent=2)),
                    "csv": lambda data, path: path.write_text(data),
                }
                for fmt, writer in _OUTPUT_WRITERS.items():
                    content = results.get(fmt)
                    if not content:
                        continue
                    fmt_path = Path(output_file).with_suffix(".%s" % fmt)
                    writer(content, fmt_path)
                    click.echo("%s results saved to: %s" % (fmt.upper(), fmt_path))
                    if fmt == "csv":
                        bench_result.output_csv = str(fmt_path)
                    elif fmt == "json":
                        bench_result.output_json = str(fmt_path)
        else:
            click.echo("[dry-run] Would parse and export results to: %s" % (output_file or "benchmark_<recipe>_<framework>.yaml"))

        # -----------------------------------------------------------
        # 10. Stop inference (unless --no-stop)
        # -----------------------------------------------------------
        if launched and not no_stop:
            click.echo("")
            logger.log(_PROGRESS_LEVEL, "Step 3/3: Stopping inference...")
            _stop_inference(runtime, host_list, cluster_id, config, dry_run)
            logger.log(_PROGRESS_LEVEL, "Inference stopped.")
        elif no_stop:
            click.echo("")
            logger.log(_PROGRESS_LEVEL, "Step 3/3: Skipping inference stop (--no-stop)")
        elif skip_run:
            click.echo("")
            logger.log(_PROGRESS_LEVEL, "Step 3/3: Skipping inference stop (--skip-run)")

        click.echo("")
        logger.log(_PROGRESS_LEVEL, "Benchmark complete.")
        bench_result.success = True

    except KeyboardInterrupt:
        click.echo("")
        click.echo("Interrupted.")
        if tasks is not None:
            click.echo("State preserved so that you can resume later")
        if not no_stop and not skip_run:
            click.echo("Stopping inference (cleaning up containers)...")
            _stop_inference(runtime, host_list, cluster_id, config, dry_run)
            click.echo("Inference stopped.")
        sys.exit(130)
    finally:
        import os

        try:
            os.unlink(result_file)
        except OSError:
            pass

    return bench_result


def _stop_inference(runtime, host_list, cluster_id, config, dry_run):
    """Stop the inference workload."""
    try:
        runtime.stop(
            hosts=host_list,
            cluster_id=cluster_id,
            config=config,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.warning("Failed to stop inference: %s", e)
        click.echo("Warning: failed to stop inference: %s" % e, err=True)
