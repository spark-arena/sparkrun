"""sparkrun cluster group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    CLUSTER_NAME,
    TARGET,
    _get_cluster_manager,
    _get_context,
    _is_cluster_id,
    _resolve_hosts_or_exit,
    build_cluster_id_overrides,
    dry_run_option,
    host_options,
    json_option,
    print_json,
    HIDE_ADVANCED_OPTIONS,
)


@click.group()
@click.pass_context
def cluster(ctx):
    """Manage saved cluster definitions."""
    pass


@cluster.command("create")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line)")
@click.option("-d", "--description", default="", help="Cluster description")
@click.option("--user", "-u", default=None, help="SSH username for this cluster")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory for this cluster")
@click.option(
    "--transfer-mode",
    default=None,
    type=click.Choice(["auto", "local", "push", "delegated"], case_sensitive=False),
    help="Resource transfer mode (auto, local, push, delegated)",
)
@click.option(
    "--transfer-interface",
    default=None,
    type=click.Choice(["auto", "cx7", "mgmt"], case_sensitive=False),
    help="Network interface for transfers (auto=default, cx7=InfiniBand, mgmt=management)",
)
@click.option("--default", "set_default", is_flag=True, default=False, help="Set as the default cluster")
@click.pass_context
def cluster_create(ctx, name, hosts, hosts_file, description, user, cache_dir, transfer_mode, transfer_interface, set_default):
    """Create a new named cluster."""
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.hosts import parse_hosts_file

    # "auto" means unset (use default behavior)
    if transfer_interface == "auto":
        transfer_interface = None

    host_list = [h.strip() for h in hosts.split(",") if h.strip()] if hosts else []
    if hosts_file:
        host_list = parse_hosts_file(hosts_file)

    if not host_list:
        click.echo("Error: No hosts provided.", err=True)
        sys.exit(1)

    mgr = _get_cluster_manager()
    try:
        mgr.create(
            name, host_list, description, user=user, cache_dir=cache_dir, transfer_mode=transfer_mode, transfer_interface=transfer_interface
        )
        click.echo(f"Cluster '{name}' created with {len(host_list)} host(s).")
        if set_default:
            mgr.set_default(name)
            click.echo(f"Default cluster set to '{name}'.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("update")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--hosts", "-H", default=None, help="Replace host list (comma-separated)")
@click.option("--hosts-file", default=None, help="Replace host list from file (one per line)")
@click.option("--add-host", multiple=True, help="Add host(s) to the cluster (repeatable, comma-ok)")
@click.option("--remove-host", multiple=True, help="Remove host(s) from the cluster (repeatable, comma-ok)")
@click.option("-d", "--description", default=None, help="Cluster description")
@click.option("--user", "-u", default=None, help="SSH username for this cluster")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory for this cluster")
@click.option(
    "--transfer-mode",
    default=None,
    type=click.Choice(["auto", "local", "push", "delegated"], case_sensitive=False),
    help="Resource transfer mode (auto, local, push, delegated)",
)
@click.option(
    "--transfer-interface",
    default=None,
    type=click.Choice(["auto", "cx7", "mgmt"], case_sensitive=False),
    help="Network interface for transfers (auto=default, cx7=InfiniBand, mgmt=management)",
)
@click.option(
    "--topology",
    default=None,
    type=click.Choice(["none", "direct", "switch", "ring"], case_sensitive=False),
    help="CX7 topology (none=remove, direct/switch=switched fabric, ring=3-node mesh/ring)",
)
@click.pass_context
def cluster_update(
    ctx, name, hosts, hosts_file, add_host, remove_host, description, user, cache_dir, transfer_mode, transfer_interface, topology
):
    """Update an existing cluster.

    \b
    Examples:
      sparkrun cluster update mylab --add-host 10.0.0.5
      sparkrun cluster update mylab --add-host 10.0.0.5 --add-host 10.0.0.6
      sparkrun cluster update mylab --add-host 10.0.0.5,10.0.0.6
      sparkrun cluster update mylab --remove-host 10.0.0.2
      sparkrun cluster update mylab --hosts 10.0.0.1,10.0.0.2,10.0.0.3
      sparkrun cluster update mylab --user ubuntu --transfer-mode push
    """
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.hosts import parse_hosts_file

    # --hosts/--hosts-file and --add-host/--remove-host are mutually exclusive
    if (hosts or hosts_file) and (add_host or remove_host):
        click.echo(
            "Error: --hosts/--hosts-file cannot be combined with --add-host/--remove-host.",
            err=True,
        )
        sys.exit(1)

    host_list = None
    if hosts:
        host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    elif hosts_file:
        host_list = parse_hosts_file(hosts_file)

    from click.core import ParameterSource

    user_provided = ctx.get_parameter_source("user") == ParameterSource.COMMANDLINE
    cache_dir_provided = ctx.get_parameter_source("cache_dir") == ParameterSource.COMMANDLINE
    transfer_mode_provided = ctx.get_parameter_source("transfer_mode") == ParameterSource.COMMANDLINE
    transfer_interface_provided = ctx.get_parameter_source("transfer_interface") == ParameterSource.COMMANDLINE
    topology_provided = ctx.get_parameter_source("topology") == ParameterSource.COMMANDLINE

    has_host_change = host_list is not None or add_host or remove_host
    if (
        not has_host_change
        and description is None
        and not user_provided
        and not cache_dir_provided
        and not transfer_mode_provided
        and not transfer_interface_provided
        and not topology_provided
    ):
        click.echo(
            "Error: Nothing to update. Provide --hosts, --hosts-file, --add-host, "
            "--remove-host, -d, --user, --cache-dir, --transfer-mode, "
            "--transfer-interface, or --topology.",
            err=True,
        )
        sys.exit(1)

    mgr = _get_cluster_manager()

    # Handle --add-host / --remove-host by modifying the current host list
    if add_host or remove_host:
        try:
            current = mgr.get(name)
        except ClusterError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        current_hosts = list(current.hosts)
        existing_set = set(current_hosts)

        for h in add_host:
            for part in h.split(","):
                part = part.strip()
                if part and part not in existing_set:
                    current_hosts.append(part)
                    existing_set.add(part)

        for h in remove_host:
            for part in h.split(","):
                part = part.strip()
                if part in existing_set:
                    current_hosts = [x for x in current_hosts if x != part]
                    existing_set.discard(part)
                else:
                    click.echo("Warning: host '%s' not in cluster '%s', skipping." % (part, name), err=True)

        if not current_hosts:
            click.echo("Error: Cannot remove all hosts from cluster.", err=True)
            sys.exit(1)

        host_list = current_hosts

    update_kwargs = {}
    if user_provided:
        update_kwargs["user"] = user
    if cache_dir_provided:
        update_kwargs["cache_dir"] = cache_dir
    if transfer_mode_provided:
        update_kwargs["transfer_mode"] = transfer_mode
    if transfer_interface_provided:
        # "auto" means unset (use default behavior)
        update_kwargs["transfer_interface"] = None if transfer_interface == "auto" else transfer_interface
    if topology_provided:
        # none=remove, direct/switch both map to "switch", ring=ring
        if topology == "none":
            update_kwargs["topology"] = None
        elif topology in ("direct", "switch"):
            update_kwargs["topology"] = "switch"
        else:
            update_kwargs["topology"] = topology

    try:
        mgr.update(name, hosts=host_list, description=description, **update_kwargs)
        if host_list is not None:
            click.echo("Cluster '%s' updated (%d hosts: %s)." % (name, len(host_list), ", ".join(host_list)))
        else:
            click.echo(f"Cluster '{name}' updated.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("list")
@json_option()
@click.pass_context
def cluster_list(ctx, output_json):
    """List all saved clusters."""
    mgr = _get_cluster_manager()
    clusters = mgr.list_clusters()
    default_name = mgr.get_default()

    if output_json:
        data = []
        for c in clusters:
            entry = c.to_dict()
            entry["default"] = c.name == default_name
            data.append(entry)
        print_json(data)
        return

    if not clusters:
        click.echo("No saved clusters.")
        return

    click.echo(f"  {'Name':<20} {'Hosts':<40} {'Description':<30}")
    click.echo("-" * 93)
    for c in clusters:
        marker = "* " if c.name == default_name else "  "
        desc = c.description or ""
        # Break hosts into lines of 2 addresses each
        host_lines = []
        for i in range(0, len(c.hosts), 2):
            host_lines.append(", ".join(c.hosts[i : i + 2]))
        first_hosts = host_lines[0] if host_lines else ""
        click.echo(f"{marker}{c.name:<20} {first_hosts:<40} {desc:<30}")
        for extra in host_lines[1:]:
            click.echo(f"  {'':<20} {extra:<40}")

    if default_name:
        click.echo("\n* = default cluster")


@cluster.command("show")
@click.argument("name", type=CLUSTER_NAME)
@json_option()
@click.pass_context
def cluster_show(ctx, name, output_json):
    """Show details of a saved cluster."""
    from sparkrun.core.cluster_manager import ClusterError

    mgr = _get_cluster_manager()
    try:
        c = mgr.get(name)
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    default_name = mgr.get_default()

    if output_json:
        data = c.to_dict()
        data["default"] = c.name == default_name
        print_json(data)
        return

    click.echo(f"Name:        {c.name}")
    click.echo(f"Description: {c.description or '(none)'}")
    if c.user:
        click.echo(f"User:        {c.user}")
    if c.cache_dir:
        click.echo(f"Cache dir:   {c.cache_dir}")
    if c.transfer_mode:
        click.echo(f"Transfer:    {c.transfer_mode}")
    if c.transfer_interface:
        click.echo(f"Xfer iface:  {c.transfer_interface}")
    if c.topology:
        click.echo(f"Topology:    {c.topology}")
    click.echo(f"Default:     {'yes' if c.name == default_name else 'no'}")
    click.echo(f"Hosts ({len(c.hosts)}):")
    for h in c.hosts:
        click.echo(f"  - {h}")


@cluster.command("delete")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def cluster_delete(ctx, name, force):
    """Delete a saved cluster."""
    from sparkrun.core.cluster_manager import ClusterError

    mgr = _get_cluster_manager()

    if not force:
        click.confirm(f"Delete cluster '{name}'?", abort=True)

    try:
        mgr.delete(name)
        click.echo(f"Cluster '{name}' deleted.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("set-default")
@click.argument("name", type=CLUSTER_NAME)
@click.pass_context
def cluster_set_default(ctx, name):
    """Set the default cluster."""
    from sparkrun.core.cluster_manager import ClusterError

    mgr = _get_cluster_manager()
    try:
        mgr.set_default(name)
        click.echo("Default cluster set to '%s'." % name)
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("unset-default")
@click.pass_context
def cluster_unset_default(ctx):
    """Remove the default cluster setting."""
    mgr = _get_cluster_manager()
    mgr.unset_default()
    click.echo("Default cluster unset.")


@cluster.command("default")
@json_option()
@click.pass_context
def cluster_default(ctx, output_json):
    """Show the current default cluster."""
    mgr = _get_cluster_manager()
    default_name = mgr.get_default()

    if output_json:
        if not default_name:
            print_json(None)
        else:
            c = mgr.get(default_name)
            data = c.to_dict()
            data["default"] = True
            print_json(data)
        return

    if not default_name:
        click.echo("No default cluster set.")
        return

    c = mgr.get(default_name)
    click.echo(f"Name:        {c.name}")
    click.echo(f"Description: {c.description or '(none)'}")
    click.echo(f"Hosts ({len(c.hosts)}):")
    for h in c.hosts:
        click.echo(f"  - {h}")


@cluster.command("monitor")
@host_options
@dry_run_option
@click.option("--interval", "-i", default=2, type=int, help="Sampling interval in seconds")
@click.option("--simple", is_flag=True, default=False, help="Use plain-text output instead of TUI")
@json_option(help="Stream updates as newline-delimited JSON objects")
@click.option(
    "--backend",
    type=click.Choice(["bash", "nv-monitor"], case_sensitive=False),
    default=None,
    help="Monitoring backend (bash=SSH script, nv-monitor=Prometheus endpoint). Default: from config or bash.",
    hidden=True,
)
@click.pass_context
def cluster_monitor(ctx, hosts, hosts_file, cluster_name, dry_run, interval, simple, output_json, backend):
    """Live-monitor CPU, RAM, and GPU metrics across cluster hosts.

    Streams host_monitor.sh on each host via SSH and displays a refreshing
    table with key metrics.  By default launches an interactive Textual TUI;
    pass --simple for plain-text output, or --json for newline-delimited JSON
    suitable for piping into external automation.  Press q (TUI) or Ctrl-C
    to stop.

    Examples:

      sparkrun cluster monitor --hosts 192.168.11.13,192.168.11.14

      sparkrun cluster monitor --cluster mylab

      sparkrun cluster monitor --cluster mylab --interval 5

      sparkrun cluster monitor --cluster mylab --simple

      sparkrun cluster monitor --cluster mylab --json
    """
    from sparkrun.core.monitoring import ClusterMonitor, stream_cluster_monitor
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = _get_context(ctx).config
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    ssh_kwargs = build_ssh_kwargs(config)

    # Resolve monitoring backend
    if backend is None:
        backend = config.monitor_backend or "bash"
    backend = backend.lower()

    if dry_run:
        click.echo("[dry-run] Would monitor %d host(s) every %ds:" % (len(host_list), interval))
        for h in host_list:
            click.echo("  %s" % h)
        stream_cluster_monitor(host_list, ssh_kwargs, interval=interval, dry_run=True)
        if backend == "nv-monitor":
            click.echo("[dry-run] Backend: nv-monitor (Prometheus over SSH port forwarding)")
        return

    # ---- JSON streaming mode ----
    if output_json:
        import time

        def _render_json(states):
            """Emit one JSON object per update tick with all host data."""
            snapshot = {"timestamp": time.time(), "hosts": {}}
            for host in host_list:
                state = states.get(host)
                if state is None or state.latest is None:
                    snapshot["hosts"][host] = {"error": state.error} if (state and state.error) else {"connecting": True}
                    continue
                snapshot["hosts"][host] = state.latest
            print_json(snapshot)

        if backend == "nv-monitor":
            from sparkrun.core.monitoring import NvMonitorClusterMonitor

            monitor = NvMonitorClusterMonitor(host_list, ssh_kwargs, interval)
            monitor.start()
            try:
                while True:
                    time.sleep(1)
                    _render_json(monitor.states)
            except KeyboardInterrupt:
                pass
            finally:
                monitor.stop()
        else:
            stream_cluster_monitor(
                host_list,
                ssh_kwargs,
                interval=interval,
                on_update=_render_json,
            )
        return

    # Try the Textual TUI unless --simple was requested.
    if not simple:
        try:
            from sparkrun.cli._monitor_tui import ClusterMonitorApp

            if backend == "nv-monitor":
                from sparkrun.core.monitoring import NvMonitorClusterMonitor

                monitor = NvMonitorClusterMonitor(host_list, ssh_kwargs, interval)
            else:
                monitor = ClusterMonitor(host_list, ssh_kwargs, interval)
            app = ClusterMonitorApp(monitor, cache_dir=str(config.cache_dir))
            app.run()
            return
        except ImportError:
            click.echo("Textual not installed — falling back to simple mode.\n", err=True)

    # ---- simple plain-text fallback ----
    import time

    from sparkrun.utils.cli_formatters import format_monitor_table

    click.echo("Monitoring %d host(s) every %ds (Ctrl-C to stop)...\n" % (len(host_list), interval))

    # Number of lines the table occupies: header + separator + one row per host
    table_lines = len(host_list) + 2

    def _render(states):
        """Move cursor back to table start and redraw."""
        table = format_monitor_table(states, host_list)
        click.echo("\033[%dA\033[J" % table_lines, nl=False)
        click.echo(table)

    if backend == "nv-monitor":
        from sparkrun.core.monitoring import NvMonitorClusterMonitor

        monitor = NvMonitorClusterMonitor(host_list, ssh_kwargs, interval)
        monitor.start()
        click.echo(format_monitor_table({}, host_list))
        try:
            while True:
                time.sleep(1)
                table = format_monitor_table(monitor.states, host_list)
                click.echo("\033[%dA\033[J" % table_lines, nl=False)
                click.echo(table)
        except KeyboardInterrupt:
            pass
        finally:
            monitor.stop()
    else:
        click.echo(format_monitor_table({}, host_list))

        stream_cluster_monitor(
            host_list,
            ssh_kwargs,
            interval=interval,
            on_update=_render,
        )

    click.echo("\nMonitoring stopped.")


@cluster.command("status")
@host_options
@dry_run_option
@json_option()
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def cluster_status(ctx, hosts, hosts_file, cluster_name, dry_run, output_json, config_path=None):
    """Show sparkrun containers running on cluster hosts.

    Lists all Docker containers whose names start with sparkrun_ on each
    host.  Accepts the same host-resolution flags as run/stop/logs.

    Examples:

      sparkrun cluster status --hosts 192.168.11.13,192.168.11.14

      sparkrun cluster status --cluster mylab
    """
    from sparkrun.core.cluster_manager import query_cluster_status
    from sparkrun.utils.cli_formatters import format_job_label, format_job_commands, format_host_display
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = _get_context(ctx).config
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)

    ssh_kwargs = build_ssh_kwargs(config)

    if dry_run:
        docker_cmd = "docker ps --filter 'name=sparkrun_' --format '{{.Names}}\\t{{.Status}}\\t{{.Image}}'"
        click.echo("[dry-run] Would run on %d host(s): %s" % (len(host_list), docker_cmd))
        return

    # Query and classify — business logic lives in cluster_manager
    result = query_cluster_status(
        host_list,
        ssh_kwargs=ssh_kwargs,
        cache_dir=str(config.cache_dir),
    )

    if output_json:
        out = result.to_dict()
        for cid, group_data in out["groups"].items():
            group_data["label"] = format_job_label(group_data["meta"], cid)
        for entry_data in out["solo_entries"]:
            entry_data["label"] = format_job_label(entry_data["meta"], entry_data["cluster_id"])

        print_json(out)
        return

    # --- Display rendering ---

    # Display grouped clusters
    if result.groups:
        for cid, group in sorted(result.groups.items()):
            click.echo(f"Job: {format_job_label(group.meta, cid)}  ({len(group.members)} container(s))")
            for host, role, status, image in group.members:
                hdisp = format_host_display(host, group.meta)
                click.echo(f"  {role:<10s} {hdisp:<40s} {status:<25s} {image}")
            # ri = group.meta.get("runtime_info")
            # if ri and isinstance(ri, dict):
            #     click.echo("  versions: %s" % ", ".join(
            #         "%s=%s" % (k, v) for k, v in sorted(ri.items())
            #     ))
            logs_cmd, stop_cmd = format_job_commands(group.meta, cluster_id=cid)
            if logs_cmd:
                click.echo(f"  logs: {logs_cmd}")
                click.echo(f"  stop: {stop_cmd}")
            click.echo()

    # Display solo / ungrouped containers (same format as cluster jobs)
    if result.solo_entries:
        for entry in result.solo_entries:
            click.echo(f"Job: {format_job_label(entry.meta, entry.cluster_id)}  (1 container(s))")
            hdisp = format_host_display(entry.host, entry.meta)
            click.echo(f"  {'solo':<10s} {hdisp:<40s} {entry.status:<25s} {entry.image}")
            logs_cmd, stop_cmd = format_job_commands(entry.meta, cluster_id=entry.cluster_id)
            if logs_cmd:
                click.echo(f"  logs: {logs_cmd}")
                click.echo(f"  stop: {stop_cmd}")
            click.echo()

    # Display errors
    for host in host_list:
        if host in result.errors:
            click.echo(f"  {host}: Error: {result.errors[host]}")

    # Display idle hosts
    if result.idle_hosts:
        click.echo("Idle hosts (no sparkrun containers):")
        for h in result.idle_hosts:
            click.echo(f"  {h}")
        click.echo()

    # Display pending operations
    if result.pending_ops:
        click.echo("Pending operations (downloads/distributions in progress):")
        for op in result.pending_ops:
            elapsed = op.get("elapsed_seconds", 0)
            mins, secs = divmod(int(elapsed), 60)
            elapsed_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
            label = op.get("recipe") or op.get("cluster_id", "?")
            detail = op.get("operation", "unknown").replace("_", " ")
            extra = ""
            if op.get("model") and "model" in detail:
                extra = f"  model={op['model']}"
            elif op.get("image") and "image" in detail:
                extra = f"  image={op['image']}"
            click.echo(f"  {label}: {detail} ({elapsed_str}){extra}")
        click.echo()
        click.echo("  Note: pending operations will consume VRAM once launched.")
        click.echo()

    # Summary
    if result.total_containers == 0 and not result.errors and not result.pending_ops:
        click.echo("No sparkrun containers running.")
    elif result.total_containers == 0 and not result.errors and result.pending_ops:
        click.echo("No sparkrun containers running yet (pending operations above).")
    else:
        click.echo(f"Total: {result.total_containers} container(s) across {result.host_count} host(s)")


@cluster.command("check-job")
@click.argument("target", type=TARGET)
@host_options
@click.option(
    "--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallelism override (used for cluster_id generation)"
)
@click.option("--port", type=int, default=None, help="Port override (used for cluster_id generation and health check)")
@click.option("--served-model-name", default=None, help="Served model name override (used for cluster_id generation)")
@click.option(
    "--check-http-models", is_flag=True, default=False, help="Also verify the inference server responds to health checks at /v1/models"
)
@json_option()
@click.pass_context
def cluster_check_job(ctx, target, hosts, hosts_file, cluster_name, tp_override, port, served_model_name, check_http_models, output_json):
    """Check if a sparkrun job is running.

    TARGET can be a cluster ID (sparkrun_<hex>) or a recipe name.

    Exit code 0 = running (and healthy if --check-health), 1 = not running or unhealthy.

    Examples:

      sparkrun cluster check-job sparkrun_abc123def456

      sparkrun cluster check-job my-recipe --hosts 10.0.0.1,10.0.0.2

      sparkrun cluster check-job my-recipe --cluster mylab --check-health

      sparkrun cluster check-job my-recipe --cluster mylab --json
    """
    from sparkrun.orchestration.job_metadata import check_job_running
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = _get_context(ctx)
    config = sctx.config
    ssh_kwargs = build_ssh_kwargs(config)

    if _is_cluster_id(target) is not None:
        # --- Cluster ID path ---
        cid = _is_cluster_id(target)
        assert cid is not None
        from sparkrun.orchestration.job_metadata import load_job_metadata

        meta = load_job_metadata(cid, cache_dir=str(config.cache_dir))

        # Resolve hosts: CLI flags > metadata > default cluster (None means "let check_job_running decide")
        host_list = None
        if hosts or hosts_file or cluster_name:
            host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
        elif meta and meta.get("hosts"):
            host_list = meta["hosts"]

        status = check_job_running(
            cluster_id=cid,
            hosts=host_list,
            ssh_kwargs=ssh_kwargs,
            cache_dir=str(config.cache_dir),
            check_http_models=check_http_models,
            port=port,
        )
    else:
        # --- Recipe path ---
        from sparkrun.cli._common import _apply_node_trimming, _load_recipe
        from sparkrun.core.bootstrap import get_runtime
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        v = sctx.variables
        recipe, _recipe_path, _registry_mgr = _load_recipe(config, target)
        host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)

        # Resolve runtime for node trimming
        try:
            runtime = get_runtime(recipe.runtime, v)
        except ValueError:
            runtime = None

        try:
            host_list = _apply_node_trimming(
                host_list,
                recipe,
                tp_override=tp_override,
                runtime=runtime,
                quiet=True,
            )
        except ValueError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)

        # Build overrides for cluster_id generation
        cid = generate_cluster_id(
            recipe, host_list, overrides=build_cluster_id_overrides(port=port, served_model_name=served_model_name, tp_override=tp_override)
        )
        status = check_job_running(
            cluster_id=cid,
            hosts=host_list,
            ssh_kwargs=ssh_kwargs,
            cache_dir=str(config.cache_dir),
            check_http_models=check_http_models,
            port=port,
        )

    # --- Output ---
    if output_json:
        print_json(status.to_dict())
    else:
        recipe_name = status.metadata.get("recipe", "unknown") if status.metadata else "unknown"
        if status.running:
            click.echo("Job running (cluster_id: %s)" % status.cluster_id)
        else:
            click.echo("Job not running (cluster_id: %s)" % status.cluster_id)
        click.echo("  Recipe: %s" % recipe_name)
        if status.hosts:
            click.echo("  Hosts:  %s" % ", ".join(status.hosts))
        if check_http_models and status.healthy is not None:
            click.echo("  Healthy: %s" % ("yes" if status.healthy else "no"))

    # Exit code: 0 = running (and healthy if checked), 1 = not running or unhealthy
    if not status.running:
        sys.exit(1)
    if check_http_models and status.healthy is False:
        sys.exit(1)


@cluster.command("inspect", hidden=HIDE_ADVANCED_OPTIONS)
@click.argument("name", type=CLUSTER_NAME, required=False, default=None)
@host_options
@dry_run_option
@json_option()
@click.pass_context
def cluster_inspect(ctx, name, hosts, hosts_file, cluster_name, dry_run, output_json):
    """Inspect effective cluster configuration and cache directories.

    Shows resolved cluster settings (transfer mode, interface, topology,
    SSH user, cache dirs) and checks whether cache directories exist on
    each remote host.  Useful for diagnosing configuration, transfer, or
    permission issues without running a job.

    NAME is an optional cluster name (equivalent to --cluster NAME).

    \b
    Examples:
      sparkrun cluster inspect mylab
      sparkrun cluster inspect mylab --json
      sparkrun cluster inspect --hosts 192.168.11.13,192.168.11.14
    """
    # Allow positional name as shorthand for --cluster
    if name and cluster_name:
        click.echo("Error: Cannot specify both a positional cluster name and --cluster.", err=True)
        sys.exit(1)
    if name:
        cluster_name = name
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sparkrun.core.cluster_manager import resolve_cluster_config
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.ssh import run_remote_command

    config = _get_context(ctx).config
    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    ssh_kwargs = build_ssh_kwargs(config)

    # Resolve effective cluster configuration
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_hf, remote_hf, xfer_mode, xfer_iface = cluster_cfg.resolve_transfer_config(config)
    local_sparkrun = str(config.cache_dir)

    # Resolve auto transfer mode to a concrete value
    from sparkrun.orchestration.distribution import resolve_auto_transfer_mode

    xfer_result = resolve_auto_transfer_mode(xfer_mode, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
    resolved_mode = xfer_result.mode

    # Detect IB / NCCL env — reuse from transfer mode resolution if available,
    # otherwise run detection explicitly.
    ib_result = xfer_result.ib_result
    if ib_result is None and not dry_run:
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts

        ib_result = detect_ib_for_hosts(host_list, ssh_kwargs=ssh_kwargs, topology=cluster_cfg.topology)

    nccl_env = ib_result.comm_env.get_env(host_list[0]) if ib_result else {}

    # Resolve effective transfer interface
    # auto (None) → cx7 if IB is available and validated, else mgmt
    if xfer_iface == "mgmt":
        resolved_iface = "mgmt"
    elif xfer_result.ib_validated or ib_result and ib_result.ib_ip_map:
        resolved_iface = "cx7"
    elif ib_result:
        resolved_iface = "mgmt"
    else:
        resolved_iface = None

    if dry_run:
        click.echo("[dry-run] Would inspect cluster config and cache dirs on %d host(s)" % len(host_list))
        return

    # Build a script that checks existence and disk usage for both dirs.
    # We derive remote sparkrun cache the same way: ~/.cache/sparkrun on the remote user.
    remote_sparkrun = "/home/%s/.cache/sparkrun" % cluster_cfg.user if cluster_cfg.user else "$HOME/.cache/sparkrun"

    check_cmd = (
        'sr_dir="%s"; hf_dir="%s"; '
        'sr_exists="no"; hf_exists="no"; sr_du="-"; hf_du="-"; '
        'if [ -d "$sr_dir" ]; then sr_exists="yes"; sr_du=$(du -sh "$sr_dir" 2>/dev/null | cut -f1); fi; '
        'if [ -d "$hf_dir" ]; then hf_exists="yes"; hf_du=$(du -sh "$hf_dir" 2>/dev/null | cut -f1); fi; '
        'free_space=$(df -h / 2>/dev/null | awk "NR==2{print \\$4}"); '
        'echo "sr_exists=$sr_exists|sr_du=$sr_du|hf_exists=$hf_exists|hf_du=$hf_du|sr_dir=$sr_dir|hf_dir=$hf_dir|free_space=${free_space:--}"'
    ) % (remote_sparkrun, remote_hf)

    # Query all hosts in parallel
    host_info: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(host_list)) as executor:
        futures = {
            executor.submit(
                run_remote_command,
                host,
                check_cmd,
                ssh_user=ssh_kwargs.get("ssh_user"),
                ssh_key=ssh_kwargs.get("ssh_key"),
                ssh_options=ssh_kwargs.get("ssh_options"),
                timeout=15,
            ): host
            for host in host_list
        }
        for future in as_completed(futures):
            host = futures[future]
            result = future.result()
            if result.success and result.stdout.strip():
                parts = dict(kv.split("=", 1) for kv in result.stdout.strip().split("|") if "=" in kv)
                host_info[host] = parts
            else:
                host_info[host] = {"error": result.stderr.strip() or "SSH failed"}

    # Check local directories
    import os
    import subprocess as _sp

    def _local_dir_info(path: str) -> tuple[str, str]:
        if not os.path.isdir(path):
            return "no", "-"
        du_result = _sp.run(["du", "-sh", path], capture_output=True, text=True)
        size = du_result.stdout.split()[0] if du_result.returncode == 0 and du_result.stdout.strip() else "-"
        return "yes", size

    local_sr_exists, local_sr_du = _local_dir_info(local_sparkrun)
    local_hf_exists, local_hf_du = _local_dir_info(local_hf)

    # Local free space on root partition
    _df_result = _sp.run(["df", "-h", "/"], capture_output=True, text=True)
    local_free_space = "-"
    if _df_result.returncode == 0 and _df_result.stdout.strip():
        _df_lines = _df_result.stdout.strip().splitlines()
        if len(_df_lines) >= 2:
            local_free_space = _df_lines[1].split()[3]

    # Collect effective config summary
    effective_config = {
        "cluster": cluster_cfg.name,
        "ssh_user": config.ssh_user,
        "transfer_mode": xfer_mode,
        "transfer_mode_resolved": resolved_mode,
        "transfer_interface": xfer_iface or "auto",
        "transfer_interface_resolved": resolved_iface,
        "topology": cluster_cfg.topology,
        "hf_cache_local": local_hf,
        "hf_cache_remote": remote_hf,
        "sparkrun_cache": local_sparkrun,
        "nccl_env": nccl_env,
    }

    if output_json:
        data = {
            "config": effective_config,
            "hosts": list(host_list),
            "local": {
                "sparkrun_cache": {"path": local_sparkrun, "exists": local_sr_exists == "yes", "size": local_sr_du},
                "hf_cache": {"path": local_hf, "exists": local_hf_exists == "yes", "size": local_hf_du},
                "free_space": local_free_space,
            },
            "remote": {},
        }
        for h in host_list:
            info = host_info.get(h, {})
            if "error" in info:
                data["remote"][h] = {"error": info["error"]}
            else:
                data["remote"][h] = {
                    "sparkrun_cache": {
                        "path": info.get("sr_dir", "?"),
                        "exists": info.get("sr_exists") == "yes",
                        "size": info.get("sr_du", "-"),
                    },
                    "hf_cache": {"path": info.get("hf_dir", "?"), "exists": info.get("hf_exists") == "yes", "size": info.get("hf_du", "-")},
                    "free_space": info.get("free_space", "-"),
                }
        print_json(data)
        return

    # --- Text output ---

    # Cluster config section
    click.echo("Cluster Configuration:")
    if cluster_cfg.name:
        click.echo("  cluster:            %s" % cluster_cfg.name)
    else:
        click.echo("  cluster:            (none — using explicit hosts)")
    click.echo("  ssh_user:           %s" % (config.ssh_user or "(default)"))

    def _fmt_resolved(configured: str, resolved: str | None) -> str:
        if resolved and configured != resolved:
            return "%s (resolved to: %s)" % (configured, resolved)
        return configured

    click.echo("  transfer_mode:      %s" % _fmt_resolved(xfer_mode, resolved_mode))
    cfg_iface = xfer_iface or "auto"
    click.echo("  transfer_interface: %s" % _fmt_resolved(cfg_iface, resolved_iface))
    click.echo("  topology:           %s" % (cluster_cfg.topology or "(none)"))
    click.echo("  hosts:              %s" % ", ".join(host_list))
    click.echo()

    # NCCL env section
    if nccl_env:
        click.echo("NCCL Environment (head: %s):" % host_list[0])
        for k, v in sorted(nccl_env.items()):
            click.echo("  %s=%s" % (k, v))
    else:
        click.echo("NCCL Environment: (no InfiniBand detected)")
    click.echo()

    # Cache paths section
    click.echo("Cache Paths:")
    click.echo("  sparkrun (local):   %s" % local_sparkrun)
    click.echo("  HF cache (local):   %s" % local_hf)
    click.echo("  HF cache (remote):  %s" % remote_hf)
    if local_hf != remote_hf:
        click.echo("  ⚠ local and remote HF cache paths differ")
    click.echo()

    # Directory status table
    click.echo("Directory Status:")
    click.echo(
        "  %-30s %-10s %-10s %-10s %-10s %-12s %s" % ("Host", "SR exists", "SR size", "HF exists", "HF size", "Free Space", "HF path")
    )
    click.echo("  " + "-" * 112)
    click.echo(
        "  %-30s %-10s %-10s %-10s %-10s %-12s %s"
        % ("(local)", local_sr_exists, local_sr_du, local_hf_exists, local_hf_du, local_free_space, local_hf)
    )
    for h in host_list:
        info = host_info.get(h, {})
        if "error" in info:
            click.echo("  %-30s %s" % (h, "Error: %s" % info["error"]))
        else:
            sr_exists = info.get("sr_exists", "?")
            sr_du = info.get("sr_du", "-")
            hf_exists = info.get("hf_exists", "?")
            hf_du = info.get("hf_du", "-")
            free_space = info.get("free_space", "-")
            hf_dir = info.get("hf_dir", "?")
            click.echo("  %-30s %-10s %-10s %-10s %-10s %-12s %s" % (h, sr_exists, sr_du, hf_exists, hf_du, free_space, hf_dir))
