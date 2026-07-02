"""sparkrun CLI — launch inference workloads on DGX Spark."""

from __future__ import annotations

import click

from sparkrun import __version__
from ._common import (
    RECIPE_NAME,
    REGISTRY_NAME,
    RUNTIME_NAME,
    _setup_logging,
    _get_context,
    dry_run_option,
    host_options,
    json_option,
)
from ._adv import adv
from ._arena import arena
from ._benchmark import benchmark
from ._cluster import cluster, cluster_status
from ._export import export
from ._proxy import proxy
from ._recipe import recipe, recipe_list, recipe_search, recipe_show
from ._registry import registry, registry_update
from ._run import run
from ._setup import setup
from ._stop_logs import logs_cmd, stop
from ._tune import tune


def _print_version(ctx, param, value):
    """Eager callback for --version: render the channel-aware display string."""
    if not value or ctx.resilient_parsing:
        return
    try:
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.version import display_version

        rendered = display_version(SparkrunConfig())
    except Exception:
        rendered = __version__
    click.echo("sparkrun, version %s" % rendered)
    ctx.exit()


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v detail, -vv timestamps, -vvv debug)")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all output except errors (for scripting)")
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_version,
    help="Show the version and exit.",
)
@click.pass_context
def main(ctx, verbose, quiet):
    """sparkrun — Launch inference workloads on NVIDIA DGX Spark systems."""
    ctx.ensure_object(dict)
    if quiet:
        verbose = -1  # sentinel: WARNING+ only
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# Register command groups and commands
main.add_command(run)
main.add_command(stop)
main.add_command(logs_cmd)
main.add_command(setup)
main.add_command(tune)
main.add_command(cluster)
main.add_command(adv)
main.add_command(recipe)
main.add_command(registry)
main.add_command(benchmark)
main.add_command(export)
main.add_command(proxy)
main.add_command(arena)


# ---------------------------------------------------------------------------
# Top-level aliases
# ---------------------------------------------------------------------------


@main.command("list")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query", required=False)
@click.option("--all", "-a", "show_all", is_flag=True, help="Include hidden registry recipes")
@json_option()
@click.pass_context
def list_cmd(ctx, registry, runtime, query, show_all, output_json):
    """List available recipes (alias for 'recipe list')."""
    ctx.invoke(recipe_list, registry=registry, runtime=runtime, query=query, show_all=show_all, output_json=output_json)


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None, help="Override tensor parallelism")
@click.option("--gpu-mem", type=float, default=None, help="Override GPU memory utilization (0.0-1.0)")
@click.pass_context
def show(ctx, recipe_name, no_vram, tensor_parallel, gpu_mem):
    """Show detailed recipe information (alias for 'recipe show')."""
    ctx.invoke(recipe_show, recipe_name=recipe_name, no_vram=no_vram, tensor_parallel=tensor_parallel, gpu_mem=gpu_mem)


@main.command("search")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query")
@click.pass_context
def search_cmd(ctx, registry, runtime, query):
    """Search for recipes by name, model, or description (alias for 'recipe search')."""
    ctx.invoke(recipe_search, registry=registry, runtime=runtime, query=query)


@main.command("status")
@host_options
@dry_run_option
@click.pass_context
def status(ctx, hosts, hosts_file, cluster_name, dry_run):
    """Show sparkrun containers running on cluster hosts (alias for 'cluster status')."""
    ctx.invoke(cluster_status, hosts=hosts, hosts_file=hosts_file, cluster_name=cluster_name, dry_run=dry_run)


@main.command("update")
@click.option("--stable", is_flag=True, help="Switch to and update the stable channel (PyPI)")
@click.option("--beta", is_flag=True, help="Switch to and update the beta channel (develop branch)")
@click.option("--alpha", is_flag=True, help="Switch to and update the alpha channel (develop-next branch)")
@click.option("--yolo", is_flag=True, help="Alias for --alpha")
@click.pass_context
def update(ctx, stable, beta, alpha, yolo):
    """Update sparkrun and recipe registries.

    Attempts to upgrade sparkrun via uv if it was installed that way, then
    always updates recipe registries from git. With no channel flag, updates
    the currently configured channel; a channel flag switches channels.

    If sparkrun was not installed via uv (e.g. pip, pipx, editable
    install), the self-upgrade step is skipped and only registries
    are updated.
    """
    import subprocess

    from sparkrun.cli._self_update import (
        capture_old_identity,
        channel_from_flags,
        describe_change,
        install_argv,
        is_uv_tool_install,
        new_binary_identity,
        resolve_uv,
        update_argv,
        warn_if_downgrade,
    )

    sctx = _get_context(ctx)
    config = sctx.config
    current = config.self_update_channel
    requested = channel_from_flags(stable, beta, alpha, yolo)
    channel = requested if requested is not None else current
    switching = requested is not None and requested != current

    old_identity = capture_old_identity()
    old_version = old_identity[0]
    new_version: str | None = old_version
    self_upgrade_attempted = False

    # --- Step 1: Try self-upgrade via uv (best-effort) ---
    uv = resolve_uv()
    upgraded = False
    if uv and is_uv_tool_install(uv):
        self_upgrade_attempted = True
        if switching:
            warn_if_downgrade(current, channel)
            click.echo("Switching to the %s channel..." % channel)
        click.echo("Checking for sparkrun updates (current: %s)..." % old_version)
        result = subprocess.run(
            install_argv(uv, channel) if switching else update_argv(uv, channel),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            config.set_self_update_channel(channel)
            new_identity = new_binary_identity()
            new_version = new_identity[0] or new_version
            click.echo(describe_change(channel, old_identity, new_identity))
            upgraded = True
        else:
            click.echo("Warning: sparkrun upgrade failed: %s" % result.stderr.strip(), err=True)
            click.echo("Continuing with registry update...", err=True)
    elif uv:
        click.echo("sparkrun not installed via uv tool — skipping self-upgrade.")
    else:
        click.echo("uv not found — skipping self-upgrade.")

    # --- Step 2: Always update registries ---
    if upgraded:
        # After uv upgrade, the running process has stale code — shell out
        # to the newly installed binary for registry update.
        click.echo()
        click.echo("Updating recipe registries...")
        reg_result = subprocess.run(
            ["sparkrun", "registry", "update"],
            capture_output=False,
        )
        if reg_result.returncode != 0:
            click.echo("Warning: registry update failed.", err=True)
    else:
        click.echo()
        ctx.invoke(registry_update)

    from sparkrun.telemetry.emit import emit_update_event

    emit_update_event(
        config,
        command="sparkrun update",
        old_version=old_version,
        new_version=new_version,
        upgraded=upgraded,
        self_upgrade_attempted=self_upgrade_attempted,
        channel=channel,
        requested_channel=requested,
    )
