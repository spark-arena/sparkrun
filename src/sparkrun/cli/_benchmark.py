"""sparkrun benchmark command — run benchmarks against inference recipes."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile

import click

from ._common import (
    PROFILE_NAME,
    RECIPE_NAME,
    _apply_node_trimming,
    _apply_recipe_overrides,
    _display_vram_estimate,
    _expand_recipe_shortcut,
    _is_recipe_url,
    _load_recipe,
    _resolve_cluster_cache_dir,
    _resolve_hosts_or_exit,
    _setup_logging,
    _simplify_recipe_ref,
    dry_run_option,
    host_options,
    recipe_override_options,
)

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_TIMEOUT: int = 14400  # 4 hours


@click.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option("--solo", is_flag=True, help="Force single-node mode")
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory")
@click.option("--profile", default=None, type=PROFILE_NAME, help="Benchmark profile name or file path")
@click.option("--framework", default=None, help="Override benchmarking framework (default: llama-benchy)")
@click.option("--out", "--output", "output_file", default=None, type=click.Path(),
              help="Output file for results YAML")
@click.option("-b", "--benchmark-option", "bench_options", multiple=True,
              help="Override benchmark arg: -b key=value (repeatable)")
@click.option("--exit-on-first-fail/--no-exit-on-first-fail", "exit_on_first_fail", default=True,
              help="Abort benchmark on first failure and skip saving results (default: enabled)")
@click.option("--no-stop", is_flag=True, help="Don't stop inference after benchmarking")
@click.option("--skip-run", is_flag=True, help="Skip launching inference (benchmark existing instance)")
@click.option("--sync-tuning", is_flag=True, help="Sync tuning configs from registries before benchmarking")
@click.option("--timeout", "bench_timeout", type=int, default=None,
              help="Benchmark timeout in seconds (default: %d, or from profile)" % DEFAULT_BENCHMARK_TIMEOUT)
@dry_run_option
@click.pass_context
def benchmark(ctx, recipe_name, hosts, hosts_file, cluster_name,
              tensor_parallel, pipeline_parallel, gpu_mem, max_model_len,
              options, image, solo, port, cache_dir,
              profile, framework,
              output_file, bench_options, exit_on_first_fail, no_stop, skip_run,
              sync_tuning, bench_timeout, dry_run):
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
    _run_benchmark(
        ctx, recipe_name, hosts, hosts_file, cluster_name,
        tensor_parallel, pipeline_parallel, gpu_mem, max_model_len,
        options, image, solo, port, cache_dir,
        profile, framework,
        output_file, bench_options, exit_on_first_fail, no_stop, skip_run,
        sync_tuning, bench_timeout, dry_run,
    )


def _run_benchmark(
        ctx, recipe_name, hosts, hosts_file, cluster_name,
        tensor_parallel, pipeline_parallel, gpu_mem, max_model_len,
        options, image, solo, port, cache_dir,
        profile, framework_name,
        output_file, bench_options, exit_on_first_fail, no_stop, skip_run,
        sync_tuning, bench_timeout, dry_run,
):
    """Execute the full benchmark flow: launch inference -> benchmark -> stop."""
    from sparkrun.benchmarking.base import export_results
    from ..core.benchmark_profiles import BenchmarkSpec
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime, get_benchmarking_framework
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.hosts import is_local_host
    from sparkrun.core.launcher import launch_inference
    from sparkrun.orchestration.primitives import (
        build_ssh_kwargs,
        detect_host_ip,
        wait_for_healthy,
        wait_for_port,
    )

    v = init_sparkrun()
    _setup_logging(ctx.obj["verbose"])
    config = SparkrunConfig()

    # ---------------------------------------------------------------
    # 1. Load recipe
    # ---------------------------------------------------------------
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    _resolved_name = _expand_recipe_shortcut(recipe_name)
    recipe_ref = _simplify_recipe_ref(_resolved_name) if _is_recipe_url(_resolved_name) else None

    issues = recipe.validate()
    for issue in issues:
        click.echo("Warning: %s" % issue, err=True)

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
        if not framework_name and bench_spec.framework:
            framework_name = bench_spec.framework
    else:
        bench_spec = BenchmarkSpec.from_recipe(recipe)
        if bench_spec:
            bench_args = dict(bench_spec.args)
            if not framework_name and bench_spec.framework:
                framework_name = bench_spec.framework

    if not framework_name:
        framework_name = "llama-benchy"

    try:
        fw = get_benchmarking_framework(framework_name, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Build layered bench args
    passthrough_layer: dict = {}
    if fw.passthrough_args:
        recipe_bench_block = recipe._raw.get("benchmark", {}) if hasattr(recipe, '_raw') else {}
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
    overrides = _apply_recipe_overrides(
        options, tensor_parallel=tensor_parallel, pipeline_parallel=pipeline_parallel,
        gpu_mem=gpu_mem, max_model_len=max_model_len, image=image, recipe=recipe,
    )
    if port is not None:
        overrides["port"] = port

    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    runtime_issues = runtime.validate_recipe(recipe)
    for issue in runtime_issues:
        click.echo("Warning: %s" % issue, err=True)

    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v)

    # Node count validation / trimming
    if len(host_list) > 1 and not solo:
        try:
            required = runtime.compute_required_nodes(recipe, overrides)
        except ValueError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
        if required is not None:
            if required > len(host_list):
                click.echo(
                    "Error: runtime requires %d nodes, but only %d hosts provided"
                    % (required, len(host_list)),
                    err=True,
                )
                sys.exit(1)
            elif required < len(host_list):
                original_count = len(host_list)
                host_list = _apply_node_trimming(
                    host_list, recipe, overrides, runtime=runtime,
                )
                click.echo(
                    "Note: %d nodes required, using %d of %d hosts"
                    % (required, required, original_count)
                )

    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        try:
            _req = runtime.compute_required_nodes(recipe, overrides)
        except ValueError:
            _req = None
        if _req is not None and _req > recipe.max_nodes:
            click.echo(
                "Error: runtime requires %d nodes (from parallelism settings), "
                "but recipe '%s' specifies max_nodes=%d"
                % (_req, recipe.name, recipe.max_nodes),
                err=True,
            )
            sys.exit(1)

        click.echo(
            "Note: recipe max_nodes=%d, using %d of %d hosts"
            % (recipe.max_nodes, recipe.max_nodes, len(host_list))
        )
        host_list = host_list[:recipe.max_nodes]

    is_solo = solo or len(host_list) <= 1
    if recipe.mode == "solo":
        is_solo = True
    if is_solo and len(host_list) > 1:
        click.echo("Note: solo mode enabled, using 1 of %d hosts" % len(host_list))
        host_list = host_list[:1]

    # Resolve cache dir
    cluster_cache_dir = _resolve_cluster_cache_dir(cluster_name, hosts, hosts_file, cluster_mgr)
    effective_cache_dir = cache_dir or cluster_cache_dir or str(config.hf_cache_dir)

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
    click.echo("=" * 60)
    click.echo("sparkrun benchmark")
    click.echo("=" * 60)
    click.echo("Recipe:                %s" % recipe.name)
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

    _display_vram_estimate(recipe, cli_overrides=overrides, auto_detect=True, cache_dir=effective_cache_dir)

    # ---------------------------------------------------------------
    # 6. Launch inference (unless --skip-run)
    # ---------------------------------------------------------------
    launched = False
    launch_result = None
    if not skip_run:
        click.echo("Step 1/3: Launching inference...")

        launch_result = launch_inference(
            recipe=recipe,
            runtime=runtime,
            host_list=host_list,
            overrides=overrides,
            config=config,
            v=v,
            is_solo=is_solo,
            cache_dir=effective_cache_dir,
            recipe_ref=recipe_ref,
            registry_mgr=registry_mgr,
            auto_port=True,
            sync_tuning=sync_tuning,
            skip_keys={"served_model_name"},
            dry_run=dry_run,
            detached=True,
        )

        if launch_result.rc != 0 and not dry_run:
            click.echo("Error: inference launch failed (exit code %d)" % launch_result.rc, err=True)
            sys.exit(launch_result.rc)

        cluster_id = launch_result.cluster_id
        serve_port = launch_result.serve_port

        click.echo("Serve command:")
        for line in launch_result.serve_command.strip().splitlines():
            click.echo("  %s" % line)
        click.echo("")

        launched = True
    else:
        click.echo("Step 1/3: Skipping inference launch (--skip-run)")

    # ---------------------------------------------------------------
    # 7. Wait for readiness and build target URL
    # ---------------------------------------------------------------
    ssh_kwargs = build_ssh_kwargs(config)
    head_host = host_list[0]

    result_file = tempfile.mktemp(suffix=".json", prefix="sparkrun_bench_")
    try:
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
            click.echo("Waiting for inference server on %s:%d..." % (head_host, serve_port))
            ready = wait_for_port(
                head_host, serve_port,
                max_retries=180, retry_interval=5,  # TODO: maybe make this dynamic with model size somewhat??
                ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                container_name=head_container,
            )
            if not ready:
                click.echo("Error: inference server did not become ready", err=True)
                if launched and not no_stop:
                    _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                sys.exit(1)

            health_url = "http://%s:%d/v1/models" % (target_ip, serve_port)
            click.echo("Waiting for model to finish loading (%s)..." % health_url)
            healthy = wait_for_healthy(
                health_url, max_retries=360, retry_interval=5, dry_run=dry_run,
            )
            if not healthy:
                click.echo("Error: inference server health check timed out", err=True)
                if launched and not no_stop:
                    _stop_inference(runtime, host_list, cluster_id, config, dry_run)
                sys.exit(1)
            click.echo("Inference server ready.")
        elif dry_run:
            click.echo("[dry-run] Would wait for inference server on %s:%d" % (head_host, serve_port))

        base_url = "http://%s:%d/v1" % (target_ip, serve_port)

        # -----------------------------------------------------------
        # 8. Run benchmark
        # -----------------------------------------------------------
        click.echo("")
        click.echo("Step 2/3: Running benchmark (%s)..." % fw.framework_name)

        est_tests = fw.estimate_test_count(bench_args)
        if est_tests is not None:
            click.echo("Estimated test iterations: %d" % est_tests)

        bench_cmd = fw.build_benchmark_command(
            target_url=base_url,
            model=recipe.model,
            args=bench_args,
            result_file=result_file,
        )

        click.echo("Benchmark command:")
        click.echo("  %s" % " ".join(bench_cmd))
        click.echo("")

        if dry_run:
            click.echo("[dry-run] Would execute benchmark command")
            stdout_text = ""
            stderr_text = ""
        else:
            import time
            click.echo("--- benchmark output ---")
            bench_start = time.monotonic()
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

        # -----------------------------------------------------------
        # 9. Parse and export results
        # -----------------------------------------------------------
        if not dry_run:
            results = fw.parse_results(stdout_text, stderr_text, result_file=result_file)

            rows = results.get("rows", [])
            if rows:
                click.echo("")
                click.echo("Results: %d test row(s) collected" % len(rows))

            if not output_file:
                profile_slug = profile.replace("/", "_").replace("@", "") if profile else "default"
                effective_pp = int(config_chain.get("pipeline_parallel") or 1)
                pp_suffix = "_pp%d" % effective_pp if effective_pp > 1 else ""
                output_file = "benchmark_%s_%s_tp%d%s.yaml" % (
                    recipe.name.replace("/", "_"),
                    profile_slug,
                    effective_tp,
                    pp_suffix,
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
            )
            click.echo("Results saved to: %s" % output_file)

            from pathlib import Path
            _OUTPUT_WRITERS = {
                "json": lambda data, path: path.write_text(
                    __import__("json").dumps(data, indent=2)),
                "csv": lambda data, path: path.write_text(data),
            }
            for fmt, writer in _OUTPUT_WRITERS.items():
                content = results.get(fmt)
                if not content:
                    continue
                fmt_path = Path(output_file).with_suffix(".%s" % fmt)
                writer(content, fmt_path)
                click.echo("%s results saved to: %s" % (fmt.upper(), fmt_path))
        else:
            click.echo("[dry-run] Would parse and export results to: %s" % (output_file or "benchmark_<recipe>_<framework>.yaml"))

        # -----------------------------------------------------------
        # 10. Stop inference (unless --no-stop)
        # -----------------------------------------------------------
        if launched and not no_stop:
            click.echo("")
            click.echo("Step 3/3: Stopping inference...")
            _stop_inference(runtime, host_list, cluster_id, config, dry_run)
            click.echo("Inference stopped.")
        elif no_stop:
            click.echo("")
            click.echo("Step 3/3: Skipping inference stop (--no-stop)")
        elif skip_run:
            click.echo("")
            click.echo("Step 3/3: Skipping inference stop (--skip-run)")

        click.echo("")
        click.echo("Benchmark complete.")

    except KeyboardInterrupt:
        click.echo("")
        click.echo("Interrupted.")
        if launched and not no_stop:
            click.echo("Stopping inference...")
            _stop_inference(runtime, host_list, cluster_id, config, dry_run)
            click.echo("Inference stopped.")
        sys.exit(130)
    finally:
        import os
        try:
            os.unlink(result_file)
        except OSError:
            pass


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
