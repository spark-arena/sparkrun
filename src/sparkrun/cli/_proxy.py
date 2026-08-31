"""sparkrun proxy commands — unified OpenAI-compatible gateway."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    _apply_recipe_overrides,
    _get_context,
    _load_recipe,
    dry_run_option,
    host_options,
    json_option,
    print_json,
    recipe_override_options,
    report_launch_validation,
    resolve_cluster_config,
    resolve_effective_hosts_for_recipe,
    with_host_context,
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
@click.option(
    "--host",
    "bind_host",
    default=None,
    help="Bind address — persisted to proxy.yaml (recommended: 127.0.0.1). Unconfigured legacy default is 0.0.0.0 (warns loudly).",
)
@click.option(
    "--master-key",
    default=None,
    help="Bearer token for stateless LiteLLM auth (no DB required).",
)
@host_options
@click.option("--foreground", is_flag=True, help="Run in foreground (default: daemonize)")
@click.option("--no-auto-discover", is_flag=True, help="Disable periodic endpoint re-scanning")
@click.option("--discover-interval", type=int, default=None, help="Seconds between discovery sweeps (default: 30)")
@click.option(
    "--discover-removal-grace-sweeps",
    type=click.IntRange(min=1),
    default=None,
    help="Consecutive missed sweeps before a discovered endpoint is removed (default: 2).",
)
@click.option("--gateway", "gateway_name", default=None, help="Gateway implementation to select and persist.")
@click.option(
    "--restart",
    is_flag=True,
    default=False,
    help="If the proxy is already running, stop it and start fresh with the new settings.",
)
@dry_run_option
def start(
    port,
    bind_host,
    master_key,
    cluster_name,
    hosts,
    hosts_file,
    foreground,
    no_auto_discover,
    discover_interval,
    discover_removal_grace_sweeps,
    gateway_name,
    restart,
    dry_run,
):
    """Start the inference proxy.

    Discovers running endpoints, generates LiteLLM config, and launches
    the proxy via ``uvx litellm``.

    Examples:

      sparkrun proxy start

      sparkrun proxy start --cluster mylab --port 4000

      sparkrun proxy start --foreground
    """
    from sparkrun import api

    sctx = _get_context(click.get_current_context())

    options = api.proxy.ProxyStartOptions(
        gateway=gateway_name,
        port=port,
        host=bind_host,
        master_key=master_key,
        host_filter=_resolve_host_filter(cluster_name, hosts, hosts_file),
        cluster=cluster_name,
        # --no-auto-discover forces off; absent, proxy.yaml decides.
        auto_discover=False if no_auto_discover else None,
        discover_interval=discover_interval,
        discover_removal_grace_sweeps=discover_removal_grace_sweeps,
        foreground=foreground,
        restart=restart,
        dry_run=dry_run,
    )

    # Resolve the gateway before announcing anything: it is a pure config
    # read, and a disabled/unknown gateway should fail before the (slow,
    # SSH-backed) discovery sweep is advertised.
    try:
        api.proxy.resolve_gateway(gateway_name, sctx=sctx)
    except api.proxy.GatewayUnavailable as exc:
        click.echo("Error: %s" % exc, err=True)
        sys.exit(1)

    click.echo("Discovering inference endpoints...")
    if foreground and not dry_run:
        # Foreground blocks inside api.proxy.start until the gateway exits,
        # so say what is happening before we hand over the terminal.
        click.echo("Starting proxy in the foreground...")

    try:
        result = api.proxy.start(options, sctx=sctx)
    except api.proxy.ProxyAlreadyRunning as exc:
        # Settings supplied this run are saved even though the proxy was left
        # alone, so say so before the error.
        if exc.persisted:
            click.echo("Saved proxy.yaml: %s" % ", ".join(exc.persisted))
        click.echo(
            "Error: %s Use --restart to apply new settings, or 'sparkrun proxy stop' first." % exc,
            err=True,
        )
        sys.exit(1)
    except (api.proxy.GatewayUnavailable, api.proxy.ProxyStartFailed) as exc:
        click.echo("Error: %s" % exc, err=True)
        sys.exit(1)

    for warning in result.warnings:
        click.echo("Warning: %s" % warning, err=True)
    if result.persisted:
        click.echo("Saved proxy.yaml: %s" % ", ".join(result.persisted))

    healthy = [ep for ep in result.endpoints if ep.healthy]
    if not healthy:
        if result.endpoints:
            click.echo("Found %d endpoint(s) but none are healthy." % len(result.endpoints))
        else:
            click.echo("No inference endpoints found.")
        click.echo("Load models with: sparkrun proxy load <recipe>")
    else:
        click.echo("Discovered %d healthy endpoint(s):" % len(healthy))
        for ep in healthy:
            click.echo("  %s:%d — %s (%s)" % (ep.host, ep.port, ", ".join(ep.models), ep.runtime))

    if result.dry_run:
        click.echo("")
        click.echo("[dry-run] Would prepare %s and start proxy on %s:%d" % (result.gateway, result.host, result.port))
        aliases = sctx.proxy_config.aliases
        if aliases:
            click.echo("[dry-run] Aliases: %s" % aliases)
        return

    click.echo("")

    if result.restarted:
        click.echo("Restarting proxy: replaced the previously-running instance.")

    if foreground:
        if result.foreground_rc:
            sys.exit(result.foreground_rc)
        return

    click.echo("Proxy started on %s:%d. API: http://localhost:%d/v1" % (result.host, result.port, result.port))
    effective_key = sctx.proxy_config.master_key
    if effective_key:
        click.echo("Management API key: %s" % effective_key)
    if result.auto_discover:
        click.echo(
            "Auto-discover enabled (every %ds; remove after %d missed sweep(s))"
            % (result.discover_interval, result.discover_removal_grace_sweeps)
        )

    # Aliases are baked into the config at generation time; report the ones
    # that actually resolved to a live backend.
    if result.aliases_applied:
        click.echo("Applied %d alias(es)." % len(result.aliases_applied))
    if result.aliases_pending:
        click.echo("Alias(es) awaiting their target model: %s" % ", ".join(result.aliases_pending))


# ---------------------------------------------------------------------------
# proxy stop
# ---------------------------------------------------------------------------


@proxy.command()
@dry_run_option
def stop(dry_run):
    """Stop the running proxy.

    Sends SIGTERM to the proxy process using the stored PID.
    """
    from sparkrun import api

    sctx = _get_context(click.get_current_context())
    result = api.proxy.stop(dry_run=dry_run, sctx=sctx)

    if not result.was_running:
        click.echo("No proxy is currently running.")
        return

    if result.stopped:
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
    from sparkrun import api

    sctx = _get_context(click.get_current_context())
    result = api.proxy.status(sctx=sctx)

    if not result.known:
        if output_json:
            print_json({"running": False})
            return
        click.echo("No proxy state found.")
        return

    if output_json:
        print_json(result.to_dict())
        return

    click.echo("Proxy status: %s" % ("running" if result.running else "stopped (stale state)"))
    click.echo("  PID:     %s" % (result.pid if result.pid is not None else "?"))
    click.echo("  Gateway: %s" % (result.gateway or "?"))
    click.echo("  Host:    %s" % (result.host or "?"))
    click.echo("  Port:    %s" % (result.port if result.port is not None else "?"))
    click.echo("  Start:   %s" % (result.started_at or "?"))

    if result.autodiscover_pid:
        if result.autodiscover_running:
            click.echo("  Auto-discover: running (PID %s)" % result.autodiscover_pid)
        else:
            click.echo("  Auto-discover: stopped (stale PID %s)" % result.autodiscover_pid)

    if result.running:
        click.echo("")
        if result.models:
            click.echo("Registered models (%d):" % len(result.models))
            for m in result.models:
                click.echo("  %s" % m.model_name)
                if m.api_base:
                    click.echo("    -> %s" % m.api_base)
        elif result.model_query_error:
            # Distinct from an empty list: the models may well be serving and
            # only the management query failed.
            click.echo("Model list unavailable: %s" % result.model_query_error)
        else:
            click.echo("No models registered.")


# ---------------------------------------------------------------------------
# proxy sync
# ---------------------------------------------------------------------------


@proxy.command("sync")
@json_option()
def sync_cmd(output_json):
    """Reconcile the proxy's model list with the workloads actually running."""
    from sparkrun import api

    sctx = _get_context(click.get_current_context())
    try:
        result = api.proxy.sync(require_running=True, sctx=sctx)
    except api.proxy.ProxyUpdateFailed as exc:
        raise click.ClickException(str(exc)) from exc

    payload = {
        "proxy_running": result.proxy_running,
        "changed": result.changed,
        "added": result.added,
        "removed": result.removed,
    }
    if output_json:
        print_json(payload)
        return
    if not result.proxy_running:
        click.echo("Proxy is not running. Start it with: sparkrun proxy start")
    elif result.changed:
        click.echo("Synchronized running models: +%d, -%d." % (result.added, result.removed))
    else:
        click.echo("Running models already in sync.")


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
    from sparkrun import api

    sctx = _get_context(click.get_current_context())

    proxy_status = api.proxy.status(sctx=sctx)
    if not proxy_status.running:
        if output_json:
            print_json([])
            return
        click.echo("Proxy is not running. Start it with: sparkrun proxy start")
        return

    if refresh:
        if not output_json:
            click.echo("Re-discovering endpoints...")
        try:
            synced = api.proxy.sync(require_running=True, sctx=sctx)
        except api.proxy.ProxyUpdateFailed as exc:
            click.echo("Error: %s" % exc, err=True)
            sys.exit(1)
        if not output_json:
            if synced.changed:
                parts = []
                if synced.added:
                    parts.append("added %d" % synced.added)
                if synced.removed:
                    parts.append("removed %d stale" % synced.removed)
                click.echo("Synced proxy models: %s." % ", ".join(parts))
            else:
                click.echo("Proxy models already in sync.")

    # Reuse the status query rather than re-asking, so an authenticated
    # management failure is not collapsed into an empty list; re-read only when
    # a refresh has since changed what the gateway serves.
    if refresh:
        proxy_status = api.proxy.status(sctx=sctx)
    model_list = proxy_status.models

    if output_json:
        print_json([m.to_dict() for m in model_list])
        return

    if not model_list:
        if proxy_status.model_query_error:
            click.echo("Model list unavailable: %s" % proxy_status.model_query_error, err=True)
        else:
            click.echo("No models registered with the proxy.")
        return

    click.echo("Models (%d):" % len(model_list))
    for m in model_list:
        click.echo("  %-40s -> %s" % (m.model_name, m.api_base or "?"))


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
    from sparkrun import api

    sctx = _get_context(click.get_current_context())

    try:
        result = api.proxy.add_alias(alias_name, target_model, sctx=sctx)
    except api.proxy.ProxyUpdateFailed as exc:
        # The alias is saved; only the running proxy failed to pick it up.
        click.echo("Alias added: %s -> %s" % (alias_name, target_model))
        click.echo("Error: %s" % exc, err=True)
        sys.exit(1)

    click.echo("Alias added: %s -> %s" % (alias_name, target_model))

    if not result.proxy_running:
        return

    if result.applied:
        click.echo("Alias applied to running proxy (restarted).")
    else:
        click.echo("Note: target model '%s' is not currently served by the proxy." % target_model)
        click.echo("The alias is saved and will apply when the target model is loaded.")


@alias.command("remove")
@click.argument("alias_name")
def alias_remove(alias_name):
    """Remove a model alias.

    Example:

      sparkrun proxy alias remove my-model
    """
    from sparkrun import api

    sctx = _get_context(click.get_current_context())

    try:
        result = api.proxy.remove_alias(alias_name, sctx=sctx)
    except api.proxy.ProxyUpdateFailed as exc:
        # The alias is already removed from proxy.yaml at this point.
        click.echo("Alias removed: %s" % alias_name)
        click.echo("Error: %s" % exc, err=True)
        sys.exit(1)

    if not result.saved:
        click.echo("Alias '%s' not found." % alias_name)
        return

    click.echo("Alias removed: %s" % alias_name)

    if not result.proxy_running:
        return

    if result.removed:
        click.echo("Removed %d alias entry/entries from running proxy (restarted)." % result.removed)
    else:
        click.echo("Alias was not active in the running proxy.")


@alias.command("list")
@json_option()
def alias_list(output_json):
    """List all configured aliases."""
    from sparkrun import api

    sctx = _get_context(click.get_current_context())
    aliases = api.proxy.list_aliases(sctx=sctx)

    if output_json:
        print_json(aliases)
        return

    if not aliases:
        click.echo("No aliases configured.")
        click.echo("Add one with: sparkrun proxy alias add <name> <model>")
        return

    click.echo("Aliases:")
    for name, target in aliases.items():
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
@with_host_context
def load_cmd(
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
    dry_run,
    host_list=None,
    cluster_mgr=None,
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
        data_parallel=data_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
        recipe=recipe,
    )

    if port is not None:
        overrides["port"] = port

    # Resolve runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Same contract as ``sparkrun run`` — shared helper, so the two agree on
    # what they print and what they refuse.
    from sparkrun.core.validation import validate_for_launch

    issues, validation_failed = validate_for_launch(recipe, runtime=runtime, config=config, v=v, include_unmapped_keys=False)
    report_launch_validation(recipe.qualified_name, issues, validation_failed)
    if validation_failed:
        sys.exit(1)

    # Node count validation, max_nodes enforcement, and solo mode determination
    host_list, is_solo = resolve_effective_hosts_for_recipe(
        host_list,
        recipe,
        overrides,
        cluster_def=None,
        sctx=sctx,
        solo=solo,
    )

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
        # Try to register with a running proxy.
        from sparkrun import api

        proxy_status = api.proxy.status(sctx=sctx)
        if proxy_status.running:
            # Discovery's liveness test is an HTTP probe of /v1/models, so
            # syncing before the server is actually serving finds nothing.
            # The launch above is detached — it returned when the containers
            # came up, which for a large model is minutes early.
            from sparkrun.core.launcher import wait_for_serve_ready
            from sparkrun.orchestration.primitives import build_ssh_kwargs

            click.echo("Waiting for server to become ready...")
            readiness = wait_for_serve_ready(
                result,
                ssh_kwargs=build_ssh_kwargs(config),
                port_timeout_s=config.readiness_port_timeout_s,
                health_timeout_s=config.readiness_health_timeout_s,
            )

            if not readiness.ready:
                _warn_not_registered(readiness, proxy_status)
            else:
                click.echo("Registering with proxy...")
                try:
                    # Not a plain sync: a catalog-driven gateway persists an
                    # activatable route here.  A discovery-driven one (LiteLLM)
                    # falls through to exactly the sync this used to call.
                    synced = api.proxy.register_loaded_model(recipe_name, sctx=sctx)
                except api.proxy.ProxyUpdateFailed as exc:
                    click.echo("Error: %s" % exc, err=True)
                    sys.exit(1)
                if synced.added:
                    click.echo("Registered %d model(s) with proxy." % synced.added)
                else:
                    click.echo(
                        "Note: proxy already served this endpoint; no config change needed.",
                    )


def _warn_not_registered(readiness, proxy_status) -> None:
    """Explain why a loaded model was not registered with the proxy.

    Not an error: the workload is running either way, and the
    auto-discover daemon registers it as soon as it answers.
    """
    if readiness.reason == "port":
        click.echo(
            "Warning: server port %d never started listening on %s (container may have exited)." % (readiness.port, readiness.head_host),
            err=True,
        )
        click.echo("  Check the logs: sparkrun logs <cluster-id>", err=True)
        return

    click.echo(
        "Warning: server at %s is still not answering — it may need longer to load." % readiness.health_url,
        err=True,
    )
    if proxy_status.autodiscover_running:
        click.echo("  The proxy's auto-discover will register it once it responds.", err=True)
    else:
        click.echo("  Register it once it responds with: sparkrun proxy sync", err=True)


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

    sctx = _get_context(ctx)
    _stop_recipe(recipe_name, hosts, hosts_file, cluster_name, sctx.config, tp_override=None, dry_run=dry_run)

    if not dry_run:
        # Sync proxy to remove the now-stale model entry.
        from sparkrun import api

        if api.proxy.status(sctx=sctx).running:
            click.echo("Syncing proxy models...")
            try:
                synced = api.proxy.unregister_loaded_model(recipe_name, sctx=sctx)
            except api.proxy.ProxyUpdateFailed as exc:
                click.echo("Error: %s" % exc, err=True)
                sys.exit(1)
            if synced.removed:
                click.echo("Removed %d stale model(s) from proxy." % synced.removed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
