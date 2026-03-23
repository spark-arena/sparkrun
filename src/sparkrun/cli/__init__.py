"""sparkrun CLI — launch inference workloads on DGX Spark."""

from __future__ import annotations

import click

from sparkrun import __version__
from ._common import (
    RECIPE_NAME,
    REGISTRY_NAME,
    RUNTIME_NAME,
    _setup_logging,
    dry_run_option,
    host_options,
)
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


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose/debug output")
@click.version_option(__version__, prog_name="sparkrun")
@click.pass_context
def main(ctx, verbose):
    """sparkrun — Launch inference workloads on NVIDIA DGX Spark systems."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# Register command groups and commands
main.add_command(run)
main.add_command(stop)
main.add_command(logs_cmd)
main.add_command(setup)
main.add_command(tune)
main.add_command(cluster)
main.add_command(recipe)
main.add_command(registry)
main.add_command(benchmark)
main.add_command(export)
main.add_command(proxy)


# ---------------------------------------------------------------------------
# Top-level aliases
# ---------------------------------------------------------------------------

@main.command("list")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query", required=False)
@click.pass_context
def list_cmd(ctx, registry, runtime, query):
    """List available recipes (alias for 'recipe list')."""
    ctx.invoke(recipe_list, registry=registry, runtime=runtime, query=query)


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.option("--gpu-mem", type=float, default=None,
              help="Override GPU memory utilization (0.0-1.0)")
@click.pass_context
def show(ctx, recipe_name, no_vram, tensor_parallel, gpu_mem):
    """Show detailed recipe information (alias for 'recipe show')."""
    ctx.invoke(recipe_show, recipe_name=recipe_name, no_vram=no_vram,
               tensor_parallel=tensor_parallel, gpu_mem=gpu_mem)


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
    ctx.invoke(cluster_status, hosts=hosts, hosts_file=hosts_file,
               cluster_name=cluster_name, dry_run=dry_run)


@main.command("update")
@click.pass_context
def update(ctx):
    """Update sparkrun and recipe registries.

    Attempts to upgrade sparkrun via ``uv tool upgrade`` if it was
    installed that way, then always updates recipe registries from git.

    If sparkrun was not installed via uv (e.g. pip, pipx, editable
    install), the self-upgrade step is skipped and only registries
    are updated.
    """
    import shutil
    import subprocess

    from sparkrun import __version__ as old_version

    # --- Step 1: Try self-upgrade via uv (best-effort) ---
    uv = shutil.which("uv")
    upgraded = False
    if uv:
        check = subprocess.run(
            [uv, "tool", "list"],
            capture_output=True, text=True,
        )
        if check.returncode == 0 and "sparkrun" in check.stdout:
            click.echo("Checking for sparkrun updates (current: %s)..." % old_version)
            result = subprocess.run(
                [uv, "tool", "upgrade", "sparkrun"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                ver_result = subprocess.run(
                    ["sparkrun", "--version"],
                    capture_output=True, text=True,
                )
                if ver_result.returncode == 0:
                    new_version = ver_result.stdout.strip().rsplit(None, 1)[-1]
                    if new_version == old_version:
                        click.echo("sparkrun %s is already the latest version." % old_version)
                    else:
                        click.echo("sparkrun updated: %s -> %s" % (old_version, new_version))
                else:
                    click.echo("sparkrun updated (could not determine new version).")
                upgraded = True
            else:
                click.echo("Warning: sparkrun upgrade failed: %s" % result.stderr.strip(), err=True)
                click.echo("Continuing with registry update...", err=True)
        else:
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
