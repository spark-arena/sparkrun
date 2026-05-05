"""sparkrun arena command group — Spark Arena authentication and benchmarking."""

from __future__ import annotations

import json
import logging
import sys

import click

from ._common import (
    # PROFILE_NAME,
    RECIPE_NAME,
    dry_run_option,
    host_options,
    recipe_override_options,
)

logger = logging.getLogger(__name__)

_BENCHMARK_PROFILE = "@official/spark-arena-v1"

ASCII_ART = r"""
!       _____                  __      ___
!      / ___/____  ____ ______/ /__   /   |  ________  ____  ____ _
!      \__ \/ __ \/ __ `/ ___/ //_/  / /| | / ___/ _ \/ __ \/ __ `/
!     ___/ / /_/ / /_/ / /  / ,<    / ___ |/ /  /  __/ / / / /_/ /
!    /____/ .___/\__,_/_/  /_/|_|  /_/  |_/_/   \___/_/ /_/\__,_/
!        /_/
"""


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
    if success:
        click.echo(ASCII_ART)
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


class _ArenaBenchmarkGroup(click.Group):
    """Group that falls back to the 'run' subcommand when the first
    positional isn't a known subcommand name."""

    def parse_args(self, ctx, args):
        # Detect whether any subcommand name is present in args; if not,
        # prepend 'run' so legacy `arena benchmark <recipe>` keeps working.
        if args:
            command_names = set(self.commands)
            has_subcmd = any(a in command_names for a in args if not a.startswith("-"))
            if not has_subcmd:
                args = ["run"] + list(args)
        return super().parse_args(ctx, args)


@arena.group("benchmark", cls=_ArenaBenchmarkGroup)
@click.pass_context
def arena_benchmark(ctx):
    """Benchmark a recipe and submit results to Spark Arena."""
    pass


@arena_benchmark.command("run")
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
@click.option(
    "--exit-on-first-fail/--no-exit-on-first-fail",
    "exit_on_first_fail",
    default=True,
    help="Abort benchmark on first failure (default: enabled)",
    hidden=True,
)
@click.option("--no-stop", is_flag=True, help="Don't stop inference after benchmarking", hidden=True)
@click.option("--skip-run", is_flag=True, help="Skip launching inference (benchmark existing instance)", hidden=True)
@click.option("--sync-tuning", is_flag=True, help="Sync tuning configs from registries before benchmarking", hidden=True)
@click.option("--rootful", is_flag=True, help="Run with --privileged as root inside container", hidden=True)
@click.option("--timeout", "bench_timeout", type=int, default=None, help="Benchmark timeout in seconds (default: 14400)")
@click.option("--local-test", is_flag=True, hidden=True, help="Smoke test: skip profile and simulate upload without sending")
@dry_run_option
@click.pass_context
def arena_benchmark_run(
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
    # profile, framework, output_file, bench_options,
    exit_on_first_fail,
    no_stop,
    skip_run,
    sync_tuning,
    rootful,
    bench_timeout,
    local_test,
    dry_run,
):
    """Benchmark a recipe and submit results to Spark Arena.

    Runs the full benchmark flow, then uploads results to the Spark Arena
    leaderboard. Requires prior authentication via 'sparkrun arena login'.

    Examples:

      sparkrun arena benchmark qwen3-1.7b-sglang --hosts host1,host2 --tp 2

      sparkrun arena benchmark qwen3-1.7b-sglang --tp 2
    """
    from sparkrun import __version__
    from sparkrun.arena.auth import load_refresh_token, exchange_token
    from sparkrun.arena.upload import generate_submission_id, upload_benchmark_results
    from ._benchmark import _run_benchmark
    from ._common import _get_context

    # --- Pre-flight checks ---
    click.echo(ASCII_ART)
    click.echo("sparkrun v%s — Spark Arena benchmark" % __version__)
    click.echo()

    refresh_token = None
    if local_test:
        click.echo("[local-test] Skipping authentication — upload will be simulated.")
        click.echo()
    else:
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

    # Generate submission_id before calling _run_benchmark so it is stable
    # across schedule retries and persisted in state.extras.
    submission_id = generate_submission_id()

    profile = None if local_test else _BENCHMARK_PROFILE
    framework = None
    output_file = None
    bench_options = []

    # --- Run benchmark ---
    bench_result = _run_benchmark(
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
        executor_args=None,
        extra_args=None,
        export_results_files=False,
        submission_id_for_extras=submission_id,
    )

    # require result and success; launch_result may be absent with --skip-run
    if not bench_result or not bench_result.success:
        click.echo("Benchmark did not complete successfully. Skipping upload.", err=True)
        return

    # gather recipe/metadata/runtime info (works with or without launch_result)
    recipe = bench_result.launch_result.recipe if bench_result.launch_result else bench_result.recipe
    overrides = bench_result.launch_result.overrides if bench_result.launch_result else (bench_result.overrides or {})
    metadata = bench_result.generate_metadata()
    effective_recipe = recipe.export(
        overrides=overrides,
        container_image=metadata["recipe"]["container"],
    )
    # benchmark_json = bench_result.results['json']
    benchmark_csv = bench_result.results["csv"]

    if dry_run or local_test:
        from pprint import pformat

        click.echo("-" * 40)
        click.echo("Effective Recipe Export:")
        click.echo("-" * 40)
        click.echo(effective_recipe)
        click.echo("-" * 40)
        click.echo("Metadata:")
        click.echo("-" * 40)
        click.echo(pformat(metadata))
        click.echo("-" * 40)

    if dry_run:
        click.echo("[dry-run] Would upload results to Spark Arena")
        return

    # --- Write files to cache directory ---
    sctx = _get_context(ctx)
    arena_cache = sctx.config.cache_dir
    cache_dir = arena_cache / "benchmarks" / submission_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    recipe_path = cache_dir / "recipe.yaml"
    csv_path = cache_dir / "benchmark.csv"
    meta_path = cache_dir / "metadata.json"

    recipe_path.write_text(effective_recipe)
    csv_path.write_text(benchmark_csv)
    meta_path.write_text(json.dumps(metadata, indent=2))

    upload_files = [
        (recipe_path, "recipes"),
        (csv_path, "logs"),
        (meta_path, "metadata"),
    ]

    # Persist arena-specific extras in state for resume support
    _persist_arena_extras(ctx, bench_result, submission_id, effective_recipe, metadata)

    if local_test:
        click.echo()
        click.echo("[local-test] Benchmark files written to: %s" % cache_dir)
        for fpath, folder in upload_files:
            click.echo("  %s -> %s/" % (fpath.name, folder))
        click.echo("[local-test] Skipping actual upload.")
        return

    # --- Upload results ---
    click.echo()
    click.echo("Uploading results to Spark Arena...")

    try:
        success, sid = upload_benchmark_results(
            refresh_token=refresh_token,
            upload_files=upload_files,
            submission_id=submission_id,
        )
    except RuntimeError as e:
        click.echo("Upload failed: %s" % e, err=True)
        sys.exit(1)

    if success:
        click.echo("Results uploaded successfully (submission: %s)" % sid)
    else:
        click.echo("Some files failed to upload (submission: %s)" % sid, err=True)
        sys.exit(1)


def _persist_arena_extras(ctx, bench_result, submission_id: str, effective_recipe: str, metadata: dict) -> None:
    """Persist arena-specific fields into BenchmarkRunState.extras if a state exists."""
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from ._common import _get_context

    sctx = _get_context(ctx)
    config = sctx.config
    cache_dir = str(config.cache_dir) if config else None

    # cluster_id and framework are needed to derive benchmark_id; they live on bench_result
    cluster_id = bench_result.cluster_id
    if not cluster_id:
        return

    # We don't have benchmark_id directly on bench_result, so we look up by cluster_id.
    # The state was already saved by _run_benchmark; load it by scanning or by re-deriving.
    # Re-derive: we need framework + base_args — obtain from bench_result.
    fw = bench_result.framework
    bench_args = getattr(bench_result, "benchmark_args", None)
    if fw is None or bench_args is None:
        return

    # Re-derive benchmark_id from the same inputs used in _run_benchmark
    # We need the schedule; approximate by loading any existing state for this cluster_id.
    # The simplest approach: scan the benchmarks cache dir for a state whose cluster_id matches.
    if not config:
        return

    benchmarks_dir = config.cache_dir / "benchmarks"
    if not benchmarks_dir.exists():
        return

    for candidate_dir in benchmarks_dir.iterdir():
        if not candidate_dir.is_dir():
            continue
        state = BenchmarkRunState.load(candidate_dir.name, cache_dir)
        if state is None:
            continue
        if state.cluster_id != cluster_id:
            continue
        # Found the matching state; persist extras (do not overwrite existing submission_id)
        if "submission_id" not in state.extras:
            state.extras["submission_id"] = submission_id
        state.extras["effective_recipe_text"] = effective_recipe
        state.extras["metadata_json"] = metadata
        state.save(cache_dir)
        logger.debug("Persisted arena extras into benchmark state %s", state.benchmark_id)
        break


@arena_benchmark.command("resume")
@click.argument("benchmark_id")
@click.option("--local-test", is_flag=True, hidden=True, help="Skip authentication and simulate upload.")
@dry_run_option
@click.pass_context
def arena_benchmark_resume(ctx, benchmark_id, local_test, dry_run):
    """Resume a paused arena benchmark by id, then upload."""
    from sparkrun.arena.auth import load_refresh_token, exchange_token
    from sparkrun.arena.upload import generate_submission_id, upload_benchmark_results
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from ._benchmark import _resume_benchmark_run
    from ._common import _get_context

    sctx = _get_context(ctx)
    config = sctx.config
    cache_dir = str(config.cache_dir) if config else None

    # Load state to recover submission_id
    state = BenchmarkRunState.load(benchmark_id, cache_dir)
    if state is None:
        click.echo("Error: no benchmark state found for id: %s" % benchmark_id, err=True)
        sys.exit(1)

    # Recover or generate submission_id (idempotent across retries)
    submission_id: str = state.extras.get("submission_id") or generate_submission_id()

    # --- Authentication ---
    refresh_token = None
    if local_test:
        click.echo("[local-test] Skipping authentication — upload will be simulated.")
        click.echo()
    else:
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

    # --- Resume benchmark (runs scheduler, writes output files, returns results dict) ---
    results = _resume_benchmark_run(ctx, benchmark_id, dry_run, sctx=sctx)
    if results is None:
        # _resume_benchmark_run already called sys.exit; this path is unreachable but defensive
        return

    if dry_run:
        click.echo("[dry-run] Would upload results to Spark Arena (submission: %s)" % submission_id)
        return

    # --- Gather upload artefacts ---
    # Prefer extras stored on first run; fall back to re-generating from results dict.
    effective_recipe: str = state.extras.get("effective_recipe_text", "")
    metadata: dict = state.extras.get("metadata_json", {})
    benchmark_csv: str = results.get("csv", "")

    arena_cache_dir = config.cache_dir / "benchmarks" / submission_id
    arena_cache_dir.mkdir(parents=True, exist_ok=True)

    recipe_path = arena_cache_dir / "recipe.yaml"
    csv_path = arena_cache_dir / "benchmark.csv"
    meta_path = arena_cache_dir / "metadata.json"

    recipe_path.write_text(effective_recipe)
    csv_path.write_text(benchmark_csv)
    meta_path.write_text(json.dumps(metadata, indent=2))

    upload_files = [
        (recipe_path, "recipes"),
        (csv_path, "logs"),
        (meta_path, "metadata"),
    ]

    if local_test:
        click.echo()
        click.echo("[local-test] Benchmark files written to: %s" % arena_cache_dir)
        for fpath, folder in upload_files:
            click.echo("  %s -> %s/" % (fpath.name, folder))
        click.echo("[local-test] Skipping actual upload.")
        return

    # --- Upload results ---
    click.echo()
    click.echo("Uploading results to Spark Arena...")

    try:
        success, sid = upload_benchmark_results(
            refresh_token=refresh_token,
            upload_files=upload_files,
            submission_id=submission_id,
        )
    except RuntimeError as e:
        click.echo("Upload failed: %s" % e, err=True)
        sys.exit(1)

    if success:
        click.echo("Results uploaded successfully (submission: %s)" % sid)
    else:
        click.echo("Some files failed to upload (submission: %s)" % sid, err=True)
        sys.exit(1)
