"""sparkrun proxy commands — unified OpenAI-compatible gateway."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    _apply_recipe_overrides,
    _load_recipe,
    _resolve_hosts_or_exit,
    dry_run_option,
    host_options,
    json_option,
    print_json,
    recipe_override_options,
    resolve_cluster_config,
    validate_and_prepare_hosts,
)


@click.group()
def proxy():
    """Manage the LiteLLM-based inference proxy.

    The proxy discovers running sparkrun inference endpoints and
    presents them through a single unified OpenAI-compatible API.
    """


# ---------------------------------------------------------------------------
# proxy start
# ---------------------------------------------------------------------------


@proxy.command()
@click.option("--port", type=int, default=None, help="Proxy listen port (default: 4000)")
@click.option("--host", "bind_host", default=None, help="Bind address (default: 0.0.0.0)")
@click.option("--master-key", default=None, help="LiteLLM master_key (default: sk-sparkrun)")
@host_options
@click.option("--foreground", is_flag=True, help="Run in foreground (default: daemonize)")
@click.option("--no-auto-discover", is_flag=True, help="Disable periodic endpoint re-scanning")
@click.option("--discover-interval", type=int, default=None, help="Seconds between discovery sweeps (default: 30)")
@dry_run_option
def start(port, bind_host, master_key, cluster_name, hosts, hosts_file, foreground, no_auto_discover, discover_interval, dry_run):
    """Start the inference proxy.

    Discovers running endpoints, generates LiteLLM config, and launches
    the proxy via ``uvx litellm``.

    Examples:

      sparkrun proxy start

      sparkrun proxy start --cluster mylab --port 4000

      sparkrun proxy start --foreground
    """
    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.discovery import discover_endpoints
    from sparkrun.proxy.engine import ProxyEngine, build_litellm_config, write_config

    proxy_cfg = ProxyConfig()

    effective_port = port or proxy_cfg.port
    effective_host = bind_host or proxy_cfg.host
    effective_key = master_key if master_key is not None else proxy_cfg.master_key

    # Resolve host filter and live discovery args
    host_filter = _resolve_host_filter(cluster_name, hosts, hosts_file)
    live_hosts, ssh_kwargs = _resolve_live_discovery_args(
        cluster_name,
        hosts,
        hosts_file,
        host_filter,
    )

    # Discover endpoints
    click.echo("Discovering inference endpoints...")
    endpoints = discover_endpoints(
        host_filter=host_filter,
        host_list=live_hosts,
        ssh_kwargs=ssh_kwargs,
    )

    healthy = [ep for ep in endpoints if ep.healthy]
    if not healthy:
        if endpoints:
            click.echo("Found %d endpoint(s) but none are healthy." % len(endpoints))
        else:
            click.echo("No inference endpoints found.")
        click.echo("Load models with: sparkrun proxy load <recipe>")
    else:
        click.echo("Discovered %d healthy endpoint(s):" % len(healthy))
        for ep in healthy:
            models_str = ", ".join(ep.actual_models) if ep.actual_models else ep.model
            click.echo("  %s:%d — %s (%s)" % (ep.host, ep.port, models_str, ep.runtime))

    # Generate config
    config_dict = build_litellm_config(healthy, effective_key)

    if dry_run:
        click.echo("")
        click.echo("[dry-run] Would write litellm config and start proxy on %s:%d" % (effective_host, effective_port))
        aliases = proxy_cfg.aliases
        if aliases:
            click.echo("[dry-run] Aliases: %s" % aliases)
        return

    config_path = write_config(config_dict)
    click.echo("")

    # Launch proxy
    engine = ProxyEngine(
        host=effective_host,
        port=effective_port,
        master_key=effective_key,
    )

    # Resolve auto-discover settings
    auto_discover = not no_auto_discover and proxy_cfg.auto_discover
    effective_interval = discover_interval or proxy_cfg.discover_interval

    ad_kwargs = None
    if auto_discover:
        ad_kwargs = {
            "interval": effective_interval,
            "host_list": live_hosts,
            "ssh_kwargs": ssh_kwargs,
        }

    # Check if already running
    if engine.is_running():
        click.echo("Proxy is already running (PID %s) on port %d." % (engine._read_pid(), engine.port))
        click.echo("Use 'sparkrun proxy stop' first, or 'sparkrun proxy load <recipe>' to add models.")
        return

    click.echo("Starting proxy on %s:%d..." % (effective_host, effective_port))
    rc = engine.start(
        config_path=config_path,
        foreground=foreground,
        autodiscover_kwargs=ad_kwargs,
    )
    if rc != 0:
        click.echo("Error: proxy failed to start (exit code %d)" % rc, err=True)
        sys.exit(rc)

    if not foreground:
        click.echo("Proxy started. API: http://localhost:%d/v1" % effective_port)
        if effective_key:
            click.echo("Management API key: %s" % effective_key)
        if auto_discover:
            click.echo("Auto-discover enabled (every %ds)" % effective_interval)

        # Apply configured aliases via management API
        aliases = proxy_cfg.aliases
        if aliases:
            import time

            time.sleep(1)  # Brief delay for proxy readiness
            added, _removed = engine.sync_aliases(aliases)
            if added:
                click.echo("Applied %d alias(es)." % added)


# ---------------------------------------------------------------------------
# proxy stop
# ---------------------------------------------------------------------------


@proxy.command()
@dry_run_option
def stop(dry_run):
    """Stop the running proxy.

    Sends SIGTERM to the proxy process using the stored PID.
    """
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine()

    if not engine.is_running():
        click.echo("No proxy is currently running.")
        return

    if engine.stop(dry_run=dry_run):
        click.echo("Proxy stopped.")
    else:
        click.echo("Failed to stop proxy.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# proxy status
# ---------------------------------------------------------------------------


@proxy.command()
@json_option()
def status(output_json):
    """Show proxy process status and registered models."""
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine()
    state = engine.get_state()

    if not state:
        if output_json:
            print_json({"running": False})
            return
        click.echo("No proxy state found.")
        return

    running = engine.is_running()

    # Autodiscover status
    ad_pid = state.get("autodiscover_pid")
    ad_running = False
    if ad_pid:
        import os

        try:
            os.kill(int(ad_pid), 0)
            ad_running = True
        except (ProcessLookupError, PermissionError, ValueError):
            pass

    # Models (only if proxy is actually running)
    model_list = []
    if running:
        for m in engine.list_models_via_api():
            name = m.get("model_name", "?")
            params = m.get("litellm_params", m.get("model_info", {}).get("litellm_params", {}))
            api_base = params.get("api_base", "?")
            model_list.append({"model_name": name, "api_base": api_base})

    if output_json:
        data = {
            "running": running,
            "pid": state.get("pid"),
            "host": state.get("host"),
            "port": state.get("port"),
            "started_at": state.get("started_at"),
            "models": model_list,
        }
        if ad_pid:
            data["autodiscover"] = {"pid": int(ad_pid), "running": ad_running}
        print_json(data)
        return

    click.echo("Proxy status: %s" % ("running" if running else "stopped (stale state)"))
    click.echo("  PID:    %s" % state.get("pid", "?"))
    click.echo("  Host:   %s" % state.get("host", "?"))
    click.echo("  Port:   %s" % state.get("port", "?"))
    click.echo("  Start:  %s" % state.get("started_at", "?"))

    if ad_pid:
        if ad_running:
            click.echo("  Auto-discover: running (PID %s)" % ad_pid)
        else:
            click.echo("  Auto-discover: stopped (stale PID %s)" % ad_pid)

    if running:
        if model_list:
            click.echo("")
            click.echo("Registered models (%d):" % len(model_list))
            for m in model_list:
                click.echo("  %s" % m["model_name"])
                if m["api_base"] != "?":
                    click.echo("    -> %s" % m["api_base"])
        else:
            click.echo("")
            click.echo("No models registered (or management API unavailable).")


# NOTE: not deleting yet, but proxy discover as a CLI command serves no purpose...
# # ---------------------------------------------------------------------------
# # proxy discover
# # ---------------------------------------------------------------------------
#
# @proxy.command()
# @host_options
# @click.option("--no-health-check", is_flag=True, help="Skip health checks")
# def discover(hosts, hosts_file, cluster_name, no_health_check):
#     """One-shot endpoint discovery (debug/inspection).
#
#     Queries running containers on cluster hosts and health-checks each
#     endpoint.  Does not start the proxy.
#
#     Examples:
#
#       sparkrun proxy discover
#
#       sparkrun proxy discover --cluster mylab
#
#       sparkrun proxy discover --no-health-check
#     """
#     from sparkrun.proxy.discovery import discover_endpoints
#
#     host_filter = _resolve_host_filter(cluster_name, hosts, hosts_file)
#
#     # Resolve hosts and SSH config for live discovery
#     live_hosts, ssh_kwargs = _resolve_live_discovery_args(
#         cluster_name, hosts, hosts_file, host_filter,
#     )
#
#     endpoints = discover_endpoints(
#         host_filter=host_filter,
#         check_health=not no_health_check,
#         host_list=live_hosts,
#         ssh_kwargs=ssh_kwargs,
#     )
#
#     if not endpoints:
#         click.echo("No inference endpoints found in job metadata.")
#         return
#
#     click.echo("Discovered %d endpoint(s):" % len(endpoints))
#     click.echo("")
#     for ep in endpoints:
#         health = "healthy" if ep.healthy else "unreachable"
#         if no_health_check:
#             health = "unchecked"
#         models_str = ", ".join(ep.actual_models) if ep.actual_models else ep.model
#         click.echo("  %-20s %s:%d" % (ep.cluster_id, ep.host, ep.port))
#         click.echo("    Recipe:   %s" % ep.recipe_name)
#         click.echo("    Model:    %s" % models_str)
#         click.echo("    Runtime:  %s" % ep.runtime)
#         click.echo("    TP:       %d" % ep.tensor_parallel)
#         click.echo("    Status:   %s" % health)
#         if ep.served_model_name:
#             click.echo("    Served:   %s" % ep.served_model_name)
#         click.echo("")
#

# ---------------------------------------------------------------------------
# proxy models
# ---------------------------------------------------------------------------


@proxy.command()
@click.option("--refresh", is_flag=True, help="Re-discover endpoints and update proxy")
@json_option()
def models(refresh, output_json):
    """List models registered with the proxy.

    Uses the management API to query the running proxy.
    With --refresh, re-discovers endpoints and adds new models.
    """
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine()

    if not engine.is_running():
        if output_json:
            print_json([])
            return
        click.echo("Proxy is not running. Start it with: sparkrun proxy start")
        return

    if refresh:
        from sparkrun.proxy.discovery import discover_endpoints

        if not output_json:
            click.echo("Re-discovering endpoints...")
        endpoints = discover_endpoints()
        healthy = [ep for ep in endpoints if ep.healthy]
        added, removed = engine.sync_models(healthy)
        if not output_json:
            if added or removed:
                parts = []
                if added:
                    parts.append("added %d" % added)
                if removed:
                    parts.append("removed %d stale" % removed)
                click.echo("Synced proxy models: %s." % ", ".join(parts))
            else:
                click.echo("Proxy models already in sync.")

    raw_models = engine.list_models_via_api()

    # Normalize model entries for both output modes
    model_list = []
    for m in raw_models:
        name = m.get("model_name", "?")
        params = m.get("litellm_params", m.get("model_info", {}).get("litellm_params", {}))
        api_base = params.get("api_base", "?")
        model_list.append({"model_name": name, "api_base": api_base})

    if output_json:
        print_json(model_list)
        return

    if not model_list:
        click.echo("No models registered with the proxy.")
        return

    click.echo("Models (%d):" % len(model_list))
    for m in model_list:
        click.echo("  %-40s -> %s" % (m["model_name"], m["api_base"]))


# ---------------------------------------------------------------------------
# proxy alias
# ---------------------------------------------------------------------------


@proxy.group()
def alias():
    """Manage model aliases."""


@alias.command("add")
@click.argument("alias_name")
@click.argument("target_model")
def alias_add(alias_name, target_model):
    """Add a model alias.

    ALIAS_NAME is the friendly name clients will use.
    TARGET_MODEL is the actual model group name to route to.

    Example:

      sparkrun proxy alias add my-model "Qwen/Qwen3-1.7B"
    """
    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.engine import ProxyEngine

    proxy_cfg = ProxyConfig()
    proxy_cfg.add_alias(alias_name, target_model)
    proxy_cfg.save()
    click.echo("Alias added: %s -> %s" % (alias_name, target_model))

    # Apply to running proxy via management API (no restart)
    engine = ProxyEngine()
    if engine.is_running():
        if engine.add_alias_via_api(alias_name, target_model):
            click.echo("Alias applied to running proxy.")
        else:
            click.echo("Warning: could not apply alias — target model '%s' not found in proxy." % target_model, err=True)
            click.echo("The alias is saved and will apply when the target model is loaded.")


@alias.command("remove")
@click.argument("alias_name")
def alias_remove(alias_name):
    """Remove a model alias.

    Example:

      sparkrun proxy alias remove my-model
    """
    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.engine import ProxyEngine

    proxy_cfg = ProxyConfig()
    if not proxy_cfg.remove_alias(alias_name):
        click.echo("Alias '%s' not found." % alias_name)
        return

    proxy_cfg.save()
    click.echo("Alias removed: %s" % alias_name)

    # Remove from running proxy via management API (no restart)
    engine = ProxyEngine()
    if engine.is_running():
        removed = engine.remove_alias_via_api(alias_name)
        if removed:
            click.echo("Removed %d alias entry/entries from running proxy." % removed)
        else:
            click.echo("Alias was not active in the running proxy.")


@alias.command("list")
@json_option()
def alias_list(output_json):
    """List all configured aliases."""
    from sparkrun.proxy.config import ProxyConfig

    proxy_cfg = ProxyConfig()
    aliases = proxy_cfg.list_aliases()

    if output_json:
        print_json(dict(aliases))
        return

    if not aliases:
        click.echo("No aliases configured.")
        click.echo("Add one with: sparkrun proxy alias add <name> <model>")
        return

    click.echo("Aliases:")
    for name, target in aliases:
        click.echo("  %-30s -> %s" % (name, target))


# ---------------------------------------------------------------------------
# proxy load / unload
# ---------------------------------------------------------------------------


@proxy.command("load")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option("--solo", is_flag=True, help="Force single-node mode")
@click.option("--port", type=int, default=None, help="Override serve port")
@dry_run_option
def load_cmd(
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tensor_parallel,
    pipeline_parallel,
    gpu_mem,
    max_model_len,
    options,
    image,
    solo,
    port,
    dry_run,
):
    """Load a model via sparkrun run and register with proxy.

    Launches inference and registers the new endpoint with the running
    proxy via the management API.

    Example:

      sparkrun proxy load qwen3-1.7b-vllm --cluster mylab

      sparkrun proxy load qwen3-1.7b-vllm --solo --gpu-mem 0.8
    """
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.core.launcher import launch_inference

    from ._common import _get_context

    sctx = _get_context(click.get_current_context())
    v = sctx.variables
    config = sctx.config

    # Load recipe (defer resolution until overrides are built)
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name, resolve=False)

    # Build overrides and resolve runtime (overrides may influence resolution)
    recipe, overrides = _apply_recipe_overrides(
        options,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
        recipe=recipe,
    )

    issues = recipe.validate()
    for issue in issues:
        click.echo("Warning: %s" % issue, err=True)
    if port is not None:
        overrides["port"] = port

    # Resolve runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Resolve hosts
    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)

    # Node count validation, max_nodes enforcement, and solo mode determination
    host_list, is_solo = validate_and_prepare_hosts(host_list, recipe, overrides, runtime, solo=solo)

    # Resolve cache dir, transfer mode, and transfer interface from cluster config
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = cluster_cfg.resolve_transfer_config(config)

    # Launch via shared pipeline (auto_port=True for conflict avoidance)
    click.echo("Loading model: %s" % recipe_name)
    result = launch_inference(
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
        registry_mgr=registry_mgr,
        auto_port=True,
        dry_run=dry_run,
        detached=True,
        # non-root user and non-privileged
        rootless=True,
        auto_user=True,
    )

    if result.rc != 0:
        click.echo("Error: failed to load model (exit code %d)." % result.rc, err=True)
        sys.exit(1)

    click.echo("Model loaded: %s (port %d)" % (recipe_name, result.serve_port))

    if not dry_run:
        # Try to register with running proxy
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine()
        if engine.is_running():
            click.echo("Registering with proxy...")
            from sparkrun.proxy.discovery import discover_endpoints
            import time

            time.sleep(2)  # Brief delay for server startup
            endpoints = discover_endpoints()
            healthy = [ep for ep in endpoints if ep.healthy]
            added, removed = engine.sync_models(healthy)
            if added:
                click.echo("Registered %d model(s) with proxy." % added)


@proxy.command("unload")
@click.argument("recipe_name")
@host_options
@dry_run_option
@click.pass_context
def unload_cmd(ctx, recipe_name, hosts, hosts_file, cluster_name, dry_run):
    """Unload a model via sparkrun stop and remove from proxy.

    Example:

      sparkrun proxy unload qwen3-1.7b-vllm --cluster mylab
    """
    from sparkrun.cli._stop_logs import _stop_recipe
    from ._common import _get_context

    config = _get_context(ctx).config
    _stop_recipe(recipe_name, hosts, hosts_file, cluster_name, config, tp_override=None, dry_run=dry_run)

    if not dry_run:
        # Sync proxy to remove the now-stale model entry
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine()
        if engine.is_running():
            click.echo("Syncing proxy models...")
            from sparkrun.proxy.discovery import discover_endpoints

            endpoints = discover_endpoints()
            healthy = [ep for ep in endpoints if ep.healthy]
            _added, removed = engine.sync_models(healthy)
            if removed:
                click.echo("Removed %d stale model(s) from proxy." % removed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_live_discovery_args(
    cluster_name: str | None,
    hosts: str | None,
    hosts_file: str | None,
    host_filter: list[str] | None,
) -> tuple[list[str] | None, dict | None]:
    """Resolve host list and SSH kwargs for live container discovery.

    Tries explicit CLI args first, then falls back to config defaults.
    Returns ``(None, None)`` when no hosts can be resolved (caller
    should fall back to metadata-only discovery).
    """
    try:
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        config = SparkrunConfig()

        live_hosts = host_filter
        if not live_hosts:
            live_hosts = config.default_hosts or None

        if not live_hosts:
            return None, None

        # Apply cluster SSH user if applicable
        if cluster_name:
            from sparkrun.cli._common import _get_cluster_manager, resolve_cluster_config

            cluster_mgr = _get_cluster_manager()
            cluster_user = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr).user
            if cluster_user:
                config.ssh_user = cluster_user

        return live_hosts, build_ssh_kwargs(config)
    except Exception:
        return None, None


def _resolve_host_filter(
    cluster_name: str | None,
    hosts: str | None,
    hosts_file: str | None,
) -> list[str] | None:
    """Resolve host filter from CLI args without exiting on empty.

    Unlike ``_resolve_hosts_or_exit``, returns None (no filter) when
    no host source is specified — discovery will scan all job metadata.
    """
    if hosts:
        return [h.strip() for h in hosts.split(",") if h.strip()]

    if hosts_file:
        try:
            from pathlib import Path

            text = Path(hosts_file).read_text()
            return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
        except OSError:
            click.echo("Warning: could not read hosts file: %s" % hosts_file, err=True)
            return None

    if cluster_name:
        try:
            from sparkrun.cli._common import _get_cluster_manager

            cluster_mgr = _get_cluster_manager()
            cluster_def = cluster_mgr.get(cluster_name)
            return cluster_def.hosts if cluster_def else None
        except Exception:
            click.echo("Warning: could not resolve cluster '%s'" % cluster_name, err=True)
            return None

    return None
