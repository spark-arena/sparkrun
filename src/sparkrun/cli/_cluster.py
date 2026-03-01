"""sparkrun cluster group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    CLUSTER_NAME,
    _get_cluster_manager,
    _resolve_hosts_or_exit,
    dry_run_option,
    host_options,
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
@click.pass_context
def cluster_create(ctx, name, hosts, hosts_file, description, user, cache_dir):
    """Create a new named cluster."""
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.hosts import parse_hosts_file

    host_list = [h.strip() for h in hosts.split(",") if h.strip()] if hosts else []
    if hosts_file:
        host_list = parse_hosts_file(hosts_file)

    if not host_list:
        click.echo("Error: No hosts provided.", err=True)
        sys.exit(1)

    mgr = _get_cluster_manager()
    try:
        mgr.create(name, host_list, description, user=user, cache_dir=cache_dir)
        click.echo(f"Cluster '{name}' created with {len(host_list)} host(s).")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("update")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line)")
@click.option("-d", "--description", default=None, help="Cluster description")
@click.option("--user", "-u", default=None, help="SSH username for this cluster")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory for this cluster")
@click.pass_context
def cluster_update(ctx, name, hosts, hosts_file, description, user, cache_dir):
    """Update an existing cluster."""
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.hosts import parse_hosts_file

    host_list = None
    if hosts:
        host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    elif hosts_file:
        host_list = parse_hosts_file(hosts_file)

    from click.core import ParameterSource

    user_provided = ctx.get_parameter_source("user") == ParameterSource.COMMANDLINE
    cache_dir_provided = ctx.get_parameter_source("cache_dir") == ParameterSource.COMMANDLINE

    if host_list is None and description is None and not user_provided and not cache_dir_provided:
        click.echo("Error: Nothing to update. Provide --hosts, --hosts-file, -d, --user, or --cache-dir.", err=True)
        sys.exit(1)

    update_kwargs = {}
    if user_provided:
        update_kwargs["user"] = user
    if cache_dir_provided:
        update_kwargs["cache_dir"] = cache_dir

    mgr = _get_cluster_manager()
    try:
        mgr.update(name, hosts=host_list, description=description, **update_kwargs)
        click.echo(f"Cluster '{name}' updated.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("list")
@click.pass_context
def cluster_list(ctx):
    """List all saved clusters."""
    mgr = _get_cluster_manager()
    clusters = mgr.list_clusters()
    default_name = mgr.get_default()

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
            host_lines.append(", ".join(c.hosts[i:i + 2]))
        first_hosts = host_lines[0] if host_lines else ""
        click.echo(f"{marker}{c.name:<20} {first_hosts:<40} {desc:<30}")
        for extra in host_lines[1:]:
            click.echo(f"  {'':<20} {extra:<40}")

    if default_name:
        click.echo("\n* = default cluster")


@cluster.command("show")
@click.argument("name", type=CLUSTER_NAME)
@click.pass_context
def cluster_show(ctx, name):
    """Show details of a saved cluster."""
    from sparkrun.core.cluster_manager import ClusterError

    mgr = _get_cluster_manager()
    try:
        c = mgr.get(name)
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    default_name = mgr.get_default()
    click.echo(f"Name:        {c.name}")
    click.echo(f"Description: {c.description or '(none)'}")
    if c.user:
        click.echo(f"User:        {c.user}")
    if c.cache_dir:
        click.echo(f"Cache dir:   {c.cache_dir}")
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
        click.echo(f"Default cluster set to '{name}'.")
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
@click.pass_context
def cluster_default(ctx):
    """Show the current default cluster."""
    mgr = _get_cluster_manager()
    default_name = mgr.get_default()
    if not default_name:
        click.echo("No default cluster set.")
        return

    c = mgr.get(default_name)
    click.echo(f"Name:        {c.name}")
    click.echo(f"Description: {c.description or '(none)'}")
    click.echo(f"Hosts ({len(c.hosts)}):")
    for h in c.hosts:
        click.echo(f"  - {h}")


@cluster.command("status")
@host_options
@dry_run_option
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def cluster_status(ctx, hosts, hosts_file, cluster_name, dry_run, config_path=None):
    """Show sparkrun containers running on cluster hosts.

    Lists all Docker containers whose names start with sparkrun_ on each
    host.  Accepts the same host-resolution flags as run/stop/logs.

    Examples:

      sparkrun cluster status --hosts 192.168.11.13,192.168.11.14

      sparkrun cluster status --cluster mylab
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.cluster_manager import query_cluster_status
    from sparkrun.utils.cli_formatters import format_job_label, format_job_commands, format_host_display
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)

    ssh_kwargs = build_ssh_kwargs(config)

    if dry_run:
        docker_cmd = (
            "docker ps --filter 'name=sparkrun_' "
            "--format '{{.Names}}\\t{{.Status}}\\t{{.Image}}'"
        )
        click.echo("[dry-run] Would run on %d host(s): %s" % (len(host_list), docker_cmd))
        return

    # Query and classify — business logic lives in cluster_manager
    result = query_cluster_status(
        host_list, ssh_kwargs=ssh_kwargs,
        cache_dir=str(config.cache_dir),
    )

    # --- Display rendering ---

    # Display grouped clusters
    if result.groups:
        for cid, group in sorted(result.groups.items()):
            click.echo(f"Job: {format_job_label(group.meta, cid)}  ({len(group.members)} container(s))")
            for host, role, status, image in group.members:
                hdisp = format_host_display(host, group.meta)
                click.echo(f"  {role:<10s} {hdisp:<40s} {status:<25s} {image}")
            logs_cmd, stop_cmd = format_job_commands(group.meta)
            if logs_cmd:
                click.echo(f"  logs: {logs_cmd}")
                click.echo(f"  stop: {stop_cmd}")
            click.echo()

    # Display solo / ungrouped containers
    if result.solo_entries:
        from sparkrun.orchestration.job_metadata import load_job_metadata
        for host, name, status, image in result.solo_entries:
            cid = name.removesuffix("_solo")
            meta = load_job_metadata(cid, cache_dir=str(config.cache_dir)) or {}
            hdisp = format_host_display(host, meta)
            click.echo(f"  {format_job_label(meta, cid):<40s} {hdisp:<40s} {status:<25s} {image}")
            logs_cmd, stop_cmd = format_job_commands(meta)
            if logs_cmd:
                click.echo(f"    logs: {logs_cmd}")
                click.echo(f"    stop: {stop_cmd}")
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
        click.echo(
            "  Note: pending operations will consume VRAM once launched."
        )
        click.echo()

    # Summary
    if result.total_containers == 0 and not result.errors and not result.pending_ops:
        click.echo("No sparkrun containers running.")
    elif result.total_containers == 0 and not result.errors and result.pending_ops:
        click.echo("No sparkrun containers running yet (pending operations above).")
    else:
        click.echo(f"Total: {result.total_containers} container(s) across {result.host_count} host(s)")
