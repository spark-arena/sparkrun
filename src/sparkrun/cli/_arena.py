"""sparkrun arena command group — Spark Arena authentication and benchmarking."""

from __future__ import annotations

import logging
import sys

import click

from ._common import (
    # PROFILE_NAME,
    RECIPE_NAME,
    _setup_logging,
    dry_run_option,
    host_options,
    recipe_override_options,
)

logger = logging.getLogger(__name__)

_BENCHMARK_PROFILE = '@official/spark-arena-v1'


@click.group()
@click.pass_context
def arena(ctx):
    """Spark Arena leaderboard — login, benchmark, and submit results."""
    pass


@arena.command()
@click.option("--browser", "force_browser", is_flag=True, hidden=True, help="Force browser-based login")
@click.option("--device", "force_device", is_flag=True, hidden=True, help="Force device code login")
@click.pass_context
def login(ctx, force_browser, force_device):
    """Authenticate with Spark Arena via OAuth."""
    from sparkrun.arena.auth import run_login_flow

    if force_browser and force_device:
        click.echo("Error: --browser and --device are mutually exclusive.", err=True)
        sys.exit(1)

    success = run_login_flow(force_browser=force_browser, force_device=force_device)
    if not success:
        sys.exit(1)


@arena.command()
def logout():
    """Remove stored Spark Arena credentials."""
    from sparkrun.arena.auth import clear_refresh_token, load_refresh_token

    if not load_refresh_token():
        click.echo("Not logged in.")
        return

    clear_refresh_token()
    click.echo("Logged out.")


@arena.command()
def status():
    """Show Spark Arena login status."""
    from sparkrun.arena.auth import load_refresh_token, exchange_token

    token = load_refresh_token()
    if not token:
        click.echo("Not logged in.")
        click.echo("Run 'sparkrun arena login' to authenticate.")
        return

    try:
        result = exchange_token(token)
        if result.email:
            user_fmt = result.email
            if result.provider:
                user_fmt += " (via %s)" % result.provider
            click.echo("Logged in to spark-arena as %s" % user_fmt)
        else:
            click.echo("Logged in to spark-arena (user id: %s)" % result.user_id)
    except RuntimeError as e:
        click.echo("Token invalid or expired: %s" % e)
        click.echo("Run 'sparkrun arena login' to re-authenticate.")


@arena.command("benchmark")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option("--solo", is_flag=True, help="Force single-node mode", hidden=True)
@click.option("--port", type=int, default=None, help="Override serve port", hidden=True)
# @click.option("--profile", default=None, type=PROFILE_NAME, help="Benchmark profile name or file path")
# @click.option("--framework", default=None, help="Override benchmarking framework (default: llama-benchy)")
# @click.option("--out", "--output", "output_file", default=None, type=click.Path(),
#               help="Output file for results YAML")
# @click.option("-b", "--benchmark-option", "bench_options", multiple=True,
#               help="Override benchmark arg: -b key=value (repeatable)")
@click.option("--exit-on-first-fail/--no-exit-on-first-fail", "exit_on_first_fail", default=True,
              help="Abort benchmark on first failure (default: enabled)")
@click.option("--no-stop", is_flag=True, help="Don't stop inference after benchmarking", hidden=True)
@click.option("--skip-run", is_flag=True, help="Skip launching inference (benchmark existing instance)", hidden=True)
@click.option("--sync-tuning", is_flag=True, help="Sync tuning configs from registries before benchmarking", hidden=True)
@click.option("--rootful", is_flag=True, help="Run with --privileged as root inside container", hidden=True)
@click.option("--timeout", "bench_timeout", type=int, default=None,
              help="Benchmark timeout in seconds (default: 14400)")
@dry_run_option
@click.pass_context
def arena_benchmark(ctx, recipe_name, hosts, hosts_file, cluster_name,
                    tensor_parallel, pipeline_parallel, gpu_mem, max_model_len,
                    options, image, solo, port,
                    # profile, framework, output_file, bench_options,
                    exit_on_first_fail, no_stop, skip_run,
                    sync_tuning, rootful, bench_timeout, dry_run):
    """Benchmark a recipe and submit results to Spark Arena.

    Runs the full benchmark flow, then uploads results to the Spark Arena
    leaderboard. Requires prior authentication via 'sparkrun arena login'.

    Examples:

      sparkrun arena benchmark qwen3-1.7b-sglang --hosts host1,host2 --tp 2

      sparkrun arena benchmark qwen3-1.7b-sglang --tp 2
    """
    from pathlib import Path
    from sparkrun import __version__
    from sparkrun.arena.auth import load_refresh_token, exchange_token
    from sparkrun.arena.upload import upload_benchmark_results
    from ._benchmark import _run_benchmark

    # --- Pre-flight checks ---
    click.echo("sparkrun %s — Spark Arena benchmark" % __version__)
    click.echo()

    refresh_token = load_refresh_token()
    if not refresh_token:
        click.echo("Error: Not logged in. Run 'sparkrun arena login' first.", err=True)
        sys.exit(1)

    try:
        exchange_token(refresh_token)
    except RuntimeError as e:
        click.echo("Error: Authentication failed: %s" % e, err=True)
        click.echo("Run 'sparkrun arena login' to re-authenticate.", err=True)
        sys.exit(1)

    click.echo("Authentication verified.")
    click.echo()

    profile = _BENCHMARK_PROFILE
    framework = None
    output_file = None  # TODO: /tmp path? /.cache/benchmarks/...path...
    bench_options = []

    # --- Run benchmark ---
    bench_result = _run_benchmark(
        ctx, recipe_name, hosts, hosts_file, cluster_name,
        tensor_parallel, pipeline_parallel, gpu_mem, max_model_len,
        options, image, solo, port,
        profile, framework,
        output_file, bench_options, exit_on_first_fail, no_stop, skip_run,
        sync_tuning, rootful, bench_timeout, dry_run,
    )

    if dry_run:
        click.echo("[dry-run] Would upload results to Spark Arena")
        return

    # require result, success, and launch result data (cannot benchmark what we didn't start)
    if not bench_result or not bench_result.success or not bench_result.launch_result:
        click.echo("Benchmark did not complete successfully. Skipping upload.", err=True)
        return

    # TODO: resolve effective recipe w/ correct overrides
    # TODO: normalize container image (if eugr, then normalize to spark-arena image if possible;
    #       if spark-arena image otherwise, try to replace :latest with most specific tag possible)
    recipe = bench_result.launch_result.recipe

    # gather metadata/runtime info
    runtime_info = bench_result.launch_result.runtime_info
    cluster_id = bench_result.launch_result.cluster_id
    host_list = bench_result.launch_result.host_list

    # --- Upload results ---
    click.echo()
    click.echo("Uploading results to Spark Arena...")

    upload_files: list[Path] = []
    recipe_yaml_path = None

    if bench_result.output_yaml:
        upload_files.append(Path(bench_result.output_yaml))
    if bench_result.output_csv:
        upload_files.append(Path(bench_result.output_csv))
    if bench_result.output_json:
        upload_files.append(Path(bench_result.output_json))

    # Try to find the recipe YAML for upload
    from sparkrun.core.config import SparkrunConfig
    from ._common import _load_recipe
    try:
        config = SparkrunConfig()
        _recipe, recipe_path, _reg = _load_recipe(config, recipe_name)
        recipe_yaml_path = Path(recipe_path)
    except (SystemExit, Exception):
        logger.debug("Could not resolve recipe path for upload", exc_info=True)

    if not upload_files:
        click.echo("No result files to upload.", err=True)
        return

    try:
        success, submission_id = upload_benchmark_results(
            refresh_token=refresh_token,
            file_paths=upload_files,
            recipe_yaml_path=recipe_yaml_path,
        )
    except RuntimeError as e:
        click.echo("Upload failed: %s" % e, err=True)
        sys.exit(1)

    if success:
        click.echo("Results uploaded successfully (submission: %s)" % submission_id)
    else:
        click.echo("Some files failed to upload (submission: %s)" % submission_id, err=True)
        sys.exit(1)
