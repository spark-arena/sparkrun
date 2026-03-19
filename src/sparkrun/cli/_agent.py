"""sparkrun agent commands — interactive AI assistant for managing workloads."""

from __future__ import annotations

import logging
import sys

import click

from ._common import (
    RECIPE_NAME,
    _load_recipe,
    _resolve_hosts_or_exit,
    _setup_logging,
    dry_run_option,
    host_options,
)

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def agent(ctx):
    """Interactive AI agent for managing sparkrun workloads.

    The agent runs a small local model on a DGX Spark and uses
    tool-calling to invoke sparkrun CLI commands via a chat interface.

    Running 'sparkrun agent' without a subcommand is equivalent to
    'sparkrun agent start'.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(start)


# ---------------------------------------------------------------------------
# agent start
# ---------------------------------------------------------------------------

@agent.command()
@click.argument("recipe_name", type=RECIPE_NAME, required=False, default=None)
@click.option("--ui", type=click.Choice(["cli", "tui", "web"], case_sensitive=False),
              default="cli", help="Chat interface (default: cli)")
@click.option("--api-base", default=None,
              help="Skip model launch, connect to existing endpoint (e.g. http://host:port/v1)")
@click.option("--port", type=int, default=None, help="Override agent model port (default: 52001)")
@click.option("--gpu-mem", type=float, default=None,
              help="Override GPU memory utilization (0.0-1.0)")
@click.option("--system-prompt", "system_prompt_file", type=click.Path(exists=True),
              default=None, help="Custom system prompt file")
@click.option("--solo", is_flag=True, help="Force single-node mode")
@host_options
@dry_run_option
@click.pass_context
def start(ctx, recipe_name, ui, api_base, port, gpu_mem,
          system_prompt_file, solo, hosts, hosts_file, cluster_name, dry_run):
    """Start the interactive agent.

    Launches a small agent model on the DGX Spark, waits for it to be
    ready, then opens a chat interface.

    If --api-base is provided, the model launch is skipped and the
    agent connects to an existing OpenAI-compatible endpoint.

    Examples:

      sparkrun agent start --solo

      sparkrun agent start --api-base http://localhost:8000/v1

      sparkrun agent start --ui tui --cluster mylab

      sparkrun agent start my-custom-agent-recipe --solo
    """
    _check_smolagents_import()
    _setup_logging(ctx.obj.get("verbose", False))

    from sparkrun.agent import DEFAULT_AGENT_PORT, DEFAULT_AGENT_RECIPE
    from sparkrun.agent.harness import create_agent
    from sparkrun.agent.state import clear_state, load_state
    from sparkrun.orchestration.primitives import wait_for_healthy

    # Load custom system prompt if provided
    custom_prompt = None
    if system_prompt_file:
        from pathlib import Path
        custom_prompt = Path(system_prompt_file).read_text().strip()

    if api_base:
        # Skip model launch — connect to existing endpoint
        endpoint = api_base
        click.echo("Connecting to existing model at %s" % endpoint)
    else:
        # Check for already-running agent
        endpoint = None
        existing = load_state()
        if existing:
            endpoint_url = existing.get("endpoint", "")
            health_url = endpoint_url.rstrip("/") + "/models" if endpoint_url else ""
            alive = wait_for_healthy(health_url, max_retries=1, max_consecutive_refused=1) if health_url else False

            if alive:
                click.echo(
                    "Agent model already running (recipe: %s, endpoint: %s). Connecting."
                    % (existing.get("recipe", "?"), endpoint_url)
                )
                endpoint = endpoint_url
            else:
                click.echo("Stale agent state found (endpoint not reachable). Clearing.")
                clear_state()

        if not endpoint:
            # Launch agent model
            endpoint = _launch_agent_model(
                recipe_name=recipe_name or DEFAULT_AGENT_RECIPE,
                port=port or DEFAULT_AGENT_PORT,
                gpu_mem=gpu_mem,
                solo=solo,
                hosts=hosts,
                hosts_file=hosts_file,
                cluster_name=cluster_name,
                dry_run=dry_run,
                ctx=ctx,
            )

    if dry_run:
        click.echo("[dry-run] Would start %s chat interface connected to %s" % (ui, endpoint))
        return

    # Create agent
    verbose = ctx.obj.get("verbose", False) if ctx.obj else False
    agent_instance = create_agent(endpoint, system_prompt=custom_prompt, verbose=verbose)

    # Start chosen interface
    if ui == "cli":
        from sparkrun.agent.interfaces.cli_chat import run_cli_chat
        run_cli_chat(agent_instance)
    elif ui == "tui":
        from sparkrun.agent.interfaces.tui_chat import run_tui_chat
        run_tui_chat(agent_instance, endpoint=endpoint)
    elif ui == "web":
        from sparkrun.agent.interfaces.web_chat import run_web_chat
        run_web_chat(agent_instance)


# ---------------------------------------------------------------------------
# agent stop
# ---------------------------------------------------------------------------

@agent.command()
@dry_run_option
def stop(dry_run):
    """Stop the agent's backing model.

    Uses stored session state to find and stop the agent model container.
    """
    from sparkrun.agent.state import clear_state, load_state

    state = load_state()
    if not state:
        click.echo("No agent session found.")
        return

    cluster_id = state.get("cluster_id")
    recipe_name = state.get("recipe", "?")

    if not cluster_id:
        click.echo("Agent state is missing cluster_id — cannot stop.")
        clear_state()
        return

    click.echo("Stopping agent model (recipe: %s, cluster: %s)..." % (recipe_name, cluster_id))

    if not dry_run:
        import shutil
        import subprocess

        sparkrun_bin = shutil.which("sparkrun")
        if sparkrun_bin:
            result = subprocess.run(
                [sparkrun_bin, "stop", recipe_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                click.echo("Agent model stopped.")
            else:
                click.echo("Warning: stop may have failed: %s" % result.stderr.strip(), err=True)
        else:
            click.echo("Warning: sparkrun not found on PATH, cannot stop container.", err=True)

        clear_state()
    else:
        click.echo("[dry-run] Would stop cluster %s and clear agent state." % cluster_id)


# ---------------------------------------------------------------------------
# agent status
# ---------------------------------------------------------------------------

@agent.command()
def status():
    """Show agent status and endpoint info."""
    from sparkrun.agent.state import load_state
    from sparkrun.orchestration.primitives import wait_for_healthy

    state = load_state()
    if not state:
        click.echo("No agent session found.")
        click.echo("Start one with: sparkrun agent start")
        return

    endpoint = state.get("endpoint", "")
    health_url = endpoint.rstrip("/") + "/models" if endpoint else ""
    alive = wait_for_healthy(health_url, max_retries=1, max_consecutive_refused=1) if health_url else False

    click.echo("Agent status:")
    click.echo("  Recipe:     %s" % state.get("recipe", "?"))
    click.echo("  Endpoint:   %s" % state.get("endpoint", "?"))
    click.echo("  Host:       %s" % state.get("host", "?"))
    click.echo("  Port:       %s" % state.get("port", "?"))
    click.echo("  Cluster ID: %s" % state.get("cluster_id", "?"))
    click.echo("  Started:    %s" % state.get("started_at", "?"))
    click.echo("  Health:     %s" % ("reachable" if alive else "NOT reachable"))

    if not alive:
        click.echo("\nThe agent endpoint is not responding. The backing model may have stopped.")
        click.echo("Run 'sparkrun agent stop' to clear stale state, or 'sparkrun agent start' to relaunch.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_smolagents_import():
    """Verify smolagents is installed, or exit with a helpful message."""
    try:
        import smolagents  # noqa: F401
    except ImportError:
        click.echo("Error: smolagents is required for the agent feature.", err=True)
        click.echo("Install it with: pip install 'sparkrun[agent]'", err=True)
        sys.exit(1)


def _launch_agent_model(
        recipe_name: str,
        port: int,
        gpu_mem: float | None,
        solo: bool,
        hosts: str | None,
        hosts_file: str | None,
        cluster_name: str | None,
        dry_run: bool,
        ctx: click.Context,
) -> str:
    """Launch the agent's backing LLM via the standard sparkrun launch pipeline.

    Returns the model endpoint URL.
    """
    from sparkrun.agent.state import save_state
    from sparkrun.cli._common import (
        _apply_node_trimming,
        _resolve_cluster_cache_dir,
        _resolve_transfer_mode,
    )
    from sparkrun.core.bootstrap import get_runtime, init_sparkrun
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
    _setup_logging(ctx.obj.get("verbose", False))
    config = SparkrunConfig()

    # Load recipe
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    # Build overrides
    overrides: dict = {}
    overrides["port"] = port
    if gpu_mem is not None:
        overrides["gpu_memory_utilization"] = gpu_mem

    # Resolve runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Resolve hosts
    host_list, cluster_mgr = _resolve_hosts_or_exit(
        hosts, hosts_file, cluster_name, config, v,
    )

    # Solo / node trimming
    is_solo = solo or len(host_list) <= 1
    if recipe.mode == "solo":
        is_solo = True
    if is_solo and len(host_list) > 1:
        host_list = host_list[:1]
    elif not is_solo and len(host_list) > 1:
        host_list = _apply_node_trimming(
            host_list, recipe, overrides, runtime=runtime,
        )

    # Resolve cache dir and transfer mode
    cluster_cache_dir = _resolve_cluster_cache_dir(cluster_name, hosts, hosts_file, cluster_mgr)
    effective_cache_dir = cluster_cache_dir or str(config.hf_cache_dir)
    cluster_transfer_mode = _resolve_transfer_mode(cluster_name, hosts, hosts_file, cluster_mgr)

    click.echo("Launching agent model (%s)..." % recipe_name)

    result = launch_inference(
        recipe=recipe,
        runtime=runtime,
        host_list=host_list,
        overrides=overrides,
        config=config,
        v=v,
        is_solo=is_solo,
        cache_dir=effective_cache_dir,
        transfer_mode=cluster_transfer_mode or "auto",
        registry_mgr=registry_mgr,
        auto_port=True,
        reuse=True,
        dry_run=dry_run,
        detached=True,
    )

    if result.rc != 0:
        click.echo("Error: failed to launch agent model (exit code %d)." % result.rc, err=True)
        sys.exit(1)

    if result.reused:
        click.echo("Found existing container for this recipe — reusing.")

    actual_port = result.serve_port
    head_host = host_list[0]

    if dry_run:
        click.echo("[dry-run] Would wait for model at http://%s:%d/v1" % (head_host, actual_port))
        return "http://%s:%d/v1" % (head_host, actual_port)

    # Two-phase readiness: port polling, then HTTP health check
    ssh_kwargs = build_ssh_kwargs(config)
    head_container = result.runtime.get_head_container_name(result.cluster_id, is_solo=result.is_solo)

    click.echo("Waiting for inference server on %s:%d..." % (head_host, actual_port))
    ready = wait_for_port(
        head_host, actual_port,
        max_retries=180, retry_interval=5,
        ssh_kwargs=ssh_kwargs,
        container_name=head_container,
    )
    if not ready:
        click.echo("Error: inference server did not become ready.", err=True)
        sys.exit(1)

    if is_local_host(head_host):
        target_ip = "127.0.0.1"
    else:
        try:
            target_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs)
        except RuntimeError:
            target_ip = head_host

    health_url = "http://%s:%d/v1/models" % (target_ip, actual_port)
    click.echo("Waiting for model to finish loading (%s)..." % health_url)
    if not wait_for_healthy(health_url, max_retries=360, retry_interval=5):
        click.echo("Error: model server health check timed out.", err=True)
        sys.exit(1)

    endpoint = "http://%s:%d/v1" % (target_ip, actual_port)
    click.echo("Model ready at %s" % endpoint)

    # Save session state
    save_state(
        endpoint=endpoint,
        cluster_id=result.cluster_id,
        recipe=recipe_name,
        host=head_host,
        port=actual_port,
    )

    return endpoint
