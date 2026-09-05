"""sparkrun cluster group and subcommands."""

from __future__ import annotations

import sys

import click

import sparkrun.api as api
from sparkrun.core.scheduler import NEW_CLUSTER_DEFAULT_SCHEDULER, new_cluster_scheduler_notice

from ._common import (
    CLUSTER_NAME,
    TARGET,
    _get_cluster_manager,
    _get_context,
    _is_cluster_id,
    _resolve_hosts_or_exit,
    resolve_host_context,
    build_cluster_id_overrides,
    dry_run_option,
    host_options,
    json_option,
    print_json,
    HIDE_ADVANCED_OPTIONS,
)


def _parse_executor_opts(opts: tuple[str, ...]) -> dict:
    """Parse repeated -o/--executor-opt key=value pairs into a dict.

    Values are auto-coerced to int/float/bool where possible via
    :func:`sparkrun.utils.coerce_value`.  Exits with an error message
    on malformed entries.
    """
    from sparkrun.utils import coerce_value

    result: dict = {}
    for opt in opts:
        if "=" not in opt:
            click.echo("Error: --executor-opt must be key=value, got: %s" % opt, err=True)
            sys.exit(1)
        key, _, value = opt.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            click.echo("Error: --executor-opt has empty key: %s" % opt, err=True)
            sys.exit(1)
        result[key] = coerce_value(value)
    return result


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
    type=click.Choice(["auto", "local", "push", "delegated", "pull"], case_sensitive=False),
    help="Resource transfer mode (auto, local, push, delegated)",
)
@click.option(
    "--transfer-interface",
    default=None,
    type=click.Choice(["auto", "cx7", "mgmt"], case_sensitive=False),
    help="Network interface for transfers (auto=default, cx7=InfiniBand, mgmt=management)",
)
@click.option(
    "--executor",
    "executor_name",
    default=None,
    help="Default executor selector for workloads on this cluster (e.g. docker, local, k8s)",
)
@click.option(
    "--executor-opt",
    "-o",
    "executor_opts",
    multiple=True,
    help="Executor option (repeatable): -o key=value (e.g. -o privileged=false -o shm_size=16g)",
)
@click.option(
    "--scheduler",
    "scheduler_name",
    default=NEW_CLUSTER_DEFAULT_SCHEDULER,
    show_default=True,
    help="Default scheduler selector for workloads on this cluster (e.g. greedy, occupancy-sparse, occupancy-dense). "
    "New clusters default to occupancy-sparse; pass '--scheduler greedy' (or an empty string) for 0.2.x behavior.",
)
@click.option(
    "--max-gpu-mem-util",
    "max_gpu_mem_util",
    type=float,
    default=None,
    help="Cluster-wide cap (0.0 < x <= 1.0) on the fraction of GPU memory usable for "
    "scheduling/fit (e.g. 0.85). Overrides platform defaults. Per-type/per-host caps via cluster YAML.",
)
@click.option("--default", "set_default", is_flag=True, default=False, help="Set as the default cluster")
@click.pass_context
def cluster_create(
    ctx,
    name,
    hosts,
    hosts_file,
    description,
    user,
    cache_dir,
    transfer_mode,
    transfer_interface,
    executor_name,
    executor_opts,
    scheduler_name,
    max_gpu_mem_util,
    set_default,
):
    """Create a new named cluster."""
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.hosts import parse_hosts_file

    if max_gpu_mem_util is not None and not (0.0 < max_gpu_mem_util <= 1.0):
        click.echo("Error: --max-gpu-mem-util must be > 0.0 and <= 1.0.", err=True)
        sys.exit(1)

    # "auto" means unset (use default behavior)
    if transfer_interface == "auto":
        transfer_interface = None

    host_list = [h.strip() for h in hosts.split(",") if h.strip()] if hosts else []
    if hosts_file:
        host_list = parse_hosts_file(hosts_file)

    if not host_list:
        click.echo("Error: No hosts provided.", err=True)
        sys.exit(1)

    executor_config = _parse_executor_opts(executor_opts) if executor_opts else None

    sctx = _get_context(ctx)
    mgr = _get_cluster_manager(sctx=sctx)
    try:
        mgr.create(
            name,
            host_list,
            description,
            user=user,
            cache_dir=cache_dir,
            transfer_mode=transfer_mode,
            transfer_interface=transfer_interface,
            executor=executor_name,
            executor_config=executor_config,
            scheduler=scheduler_name,
            max_gpu_memory_utilization=max_gpu_mem_util,
        )
        click.echo(f"Cluster '{name}' created with {len(host_list)} host(s).")
        if scheduler_name and scheduler_name != "greedy":
            click.echo(new_cluster_scheduler_notice(scheduler_name))
        if set_default:
            mgr.set_default(name)
            click.echo(f"Default cluster set to '{name}'.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.group("import", invoke_without_command=True)
@click.option(
    "--from-spark-vllm-docker-env",
    "svd_env",
    default=None,
    metavar="PATH",
    help="[deprecated] Import a legacy spark-vllm-docker .env; use `cluster import svd PATH`.",
)
@click.option("--name", "name", type=CLUSTER_NAME, default=None, help="Cluster name (default: derived from the source)")
@click.option("--default", "set_default", is_flag=True, default=False, help="Set as the default cluster")
@dry_run_option
@click.pass_context
def cluster_import(ctx, svd_env, name, set_default, dry_run):
    """Import an external cluster config into a sparkrun cluster.

    Subcommands:

    \b
      svd | eugr PATH   import a spark-vllm-docker .env file

    Additional providers may be contributed by plugins (e.g. ``cluster import
    thunder`` from the sparkrun_thunder plugin).

    The legacy ``--from-spark-vllm-docker-env PATH`` flag is still accepted
    (deprecated) and forwards to ``import svd``.
    """
    if ctx.invoked_subcommand is not None:
        return
    if svd_env:
        click.echo(
            "Warning: --from-spark-vllm-docker-env is deprecated; use `sparkrun cluster import svd PATH`.",
            err=True,
        )
        _do_svd_import(ctx, svd_env, name, set_default, dry_run)
        return
    click.echo(ctx.get_help(), err=True)
    ctx.exit(0)


def _do_svd_import(ctx, svd_env, name, set_default, dry_run):
    """Import a spark-vllm-docker ``.env`` into a cluster (svd/eugr subcommand body).

    Re-running on the same file **syncs in place** — matched by the stored
    ``sync_source`` (even if the cluster was renamed) — and only ever rewrites
    the import-owned fields.  The resolved cluster name is printed to stdout.
    """
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.svd_import import build_svd_import

    if not svd_env:
        click.echo("Error: a source PATH is required.", err=True)
        sys.exit(1)

    try:
        imp = build_svd_import(svd_env)
    except ClusterError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    sctx = _get_context(ctx)
    mgr = _get_cluster_manager(sctx=sctx)

    # Re-sync identity is the sync_source, not the name — so a renamed
    # cluster still updates in place on re-import.
    existing = next((c for c in mgr.list_clusters() if c.sync_source == imp.sync_source), None)
    if existing is not None:
        target = existing.name
    else:
        target = name or imp.default_name
        clash = None
        try:
            clash = mgr.get(target)
        except ClusterError:
            clash = None
        if clash is not None and clash.sync_source != imp.sync_source:
            click.echo(
                "Error: cluster '%s' already exists from a different source; pass --name to choose another." % target,
                err=True,
            )
            sys.exit(1)

    # Human report goes to stderr; only the resolved name lands on stdout.
    for line in imp.carried:
        click.echo("  carried: %s" % line, err=True)
    for line in imp.dropped:
        click.echo("  dropped: %s" % line, err=True)
    click.echo("  env_file: %s" % imp.env_file, err=True)

    if dry_run:
        click.echo("[dry-run] would %s cluster '%s'." % ("sync" if existing else "create", target), err=True)
        click.echo(target)
        return

    try:
        if existing is not None:
            changes = [
                f
                for f, old, new in (
                    ("hosts", existing.hosts, imp.hosts),
                    ("fabric_interfaces", existing.fabric_interfaces, imp.fabric_interfaces),
                    ("env", existing.env, imp.env),
                    ("env_file", existing.env_file, imp.env_file),
                )
                if old != new
            ]
            mgr.update(
                target,
                hosts=imp.hosts,
                fabric_interfaces=imp.fabric_interfaces,
                env=imp.env,
                env_file=imp.env_file,
                sync_source=imp.sync_source,
            )
            if changes:
                click.echo("Synced cluster '%s' (updated: %s)." % (target, ", ".join(changes)), err=True)
            else:
                click.echo("Cluster '%s' already in sync; no changes." % target, err=True)
        else:
            mgr.create(
                target,
                imp.hosts,
                fabric_interfaces=imp.fabric_interfaces,
                env=imp.env,
                env_file=imp.env_file,
                sync_source=imp.sync_source,
            )
            click.echo("Imported cluster '%s' with %d host(s)." % (target, len(imp.hosts)), err=True)
        if set_default:
            mgr.set_default(target)
            click.echo("Default cluster set to '%s'." % target, err=True)
    except ClusterError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    click.echo(target)


@cluster_import.command("svd")
@click.argument("path", metavar="PATH")
@click.option("--name", "name", type=CLUSTER_NAME, default=None, help="Cluster name (default: derived from the env file path)")
@click.option("--default", "set_default", is_flag=True, default=False, help="Set as the default cluster")
@dry_run_option
@click.pass_context
def cluster_import_svd(ctx, path, name, set_default, dry_run):
    """Import a spark-vllm-docker ``.env`` file into a cluster.

    Maps ``CLUSTER_NODES`` -> hosts, ``ETH_IF`` -> fabric_interfaces, and
    ``CONTAINER_*`` -> env references (resolved from the env file at launch, so
    secrets stay out of YAML).  Re-running syncs in place.  Aliased as ``eugr``.
    """
    _do_svd_import(ctx, path, name, set_default, dry_run)


# ``eugr`` is an alias for ``svd`` (same spark-vllm-docker .env format).
cluster_import.add_command(cluster_import_svd, "eugr")


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
    type=click.Choice(["auto", "local", "push", "delegated", "pull"], case_sensitive=False),
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
@click.option(
    "--mgmt-interface",
    default=None,
    help="Pin the management/control interface name on every host (e.g. enP7s7), overriding detection. "
    "Needed only where detection can't decide — e.g. a bonded or VLAN management link. Pass empty string to clear.",
)
@click.option(
    "--infer-hardware",
    is_flag=True,
    help="SSH into each cluster host, detect accelerators (NVIDIA/AMD/Intel/Apple) + IB, and persist per-host hardware metadata",
)
@click.option(
    "--executor",
    "executor_name",
    default=None,
    help="Default executor selector for workloads on this cluster (e.g. docker, local, k8s). Pass empty string to clear.",
)
@click.option(
    "--executor-opt",
    "-o",
    "executor_opts",
    multiple=True,
    help="Executor option (repeatable): -o key=value. Pass once with no value to clear.",
)
@click.option(
    "--clear-executor-config",
    is_flag=True,
    default=False,
    help="Remove all executor config options from the cluster",
)
@click.option(
    "--scheduler",
    "scheduler_name",
    default=None,
    help="Default scheduler selector for workloads on this cluster (e.g. greedy, occupancy-sparse, occupancy-dense). Pass empty string to clear.",
)
@click.option(
    "--max-gpu-mem-util",
    "max_gpu_mem_util",
    type=float,
    default=None,
    help="Cluster-wide cap (0.0 < x <= 1.0) on the fraction of GPU memory usable for "
    "scheduling/fit (e.g. 0.85). Pass 0 to clear. Per-type/per-host caps via cluster YAML.",
)
@click.pass_context
def cluster_update(
    ctx,
    name,
    hosts,
    hosts_file,
    add_host,
    remove_host,
    description,
    user,
    cache_dir,
    transfer_mode,
    transfer_interface,
    topology,
    mgmt_interface,
    infer_hardware,
    executor_name,
    executor_opts,
    clear_executor_config,
    scheduler_name,
    max_gpu_mem_util,
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
    mgmt_interface_provided = ctx.get_parameter_source("mgmt_interface") == ParameterSource.COMMANDLINE
    executor_provided = ctx.get_parameter_source("executor_name") == ParameterSource.COMMANDLINE
    executor_opts_provided = bool(executor_opts) or clear_executor_config
    scheduler_provided = ctx.get_parameter_source("scheduler_name") == ParameterSource.COMMANDLINE
    max_gpu_mem_util_provided = ctx.get_parameter_source("max_gpu_mem_util") == ParameterSource.COMMANDLINE

    has_host_change = host_list is not None or add_host or remove_host
    if (
        not has_host_change
        and description is None
        and not user_provided
        and not cache_dir_provided
        and not transfer_mode_provided
        and not transfer_interface_provided
        and not topology_provided
        and not mgmt_interface_provided
        and not infer_hardware
        and not executor_provided
        and not executor_opts_provided
        and not scheduler_provided
        and not max_gpu_mem_util_provided
    ):
        click.echo(
            "Error: Nothing to update. Provide --hosts, --hosts-file, --add-host, "
            "--remove-host, -d, --user, --cache-dir, --transfer-mode, "
            "--transfer-interface, --topology, --mgmt-interface, --infer-hardware, --executor, "
            "--executor-opt, --clear-executor-config, --scheduler, or --max-gpu-mem-util.",
            err=True,
        )
        sys.exit(1)

    if max_gpu_mem_util_provided and max_gpu_mem_util != 0 and not (0.0 < max_gpu_mem_util <= 1.0):
        click.echo("Error: --max-gpu-mem-util must be > 0.0 and <= 1.0 (or 0 to clear).", err=True)
        sys.exit(1)

    sctx = _get_context(ctx)
    mgr = _get_cluster_manager(sctx=sctx)

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
    if mgmt_interface_provided:
        # Empty string clears the pin and restores per-host detection
        update_kwargs["mgmt_interface"] = mgmt_interface.strip() if mgmt_interface and mgmt_interface.strip() else None
    if executor_provided:
        # Empty string clears the executor selector
        update_kwargs["executor"] = executor_name if executor_name else None
    if clear_executor_config:
        update_kwargs["executor_config"] = None
    elif executor_opts:
        update_kwargs["executor_config"] = _parse_executor_opts(executor_opts)
    if scheduler_provided:
        # Empty string clears the scheduler selector
        update_kwargs["scheduler"] = scheduler_name if scheduler_name else None
    if max_gpu_mem_util_provided:
        # 0 clears the cluster-wide cap (defer to platform default / 1.0 fallback)
        update_kwargs["max_gpu_memory_utilization"] = max_gpu_mem_util if max_gpu_mem_util else None

    if infer_hardware:
        from sparkrun.core.fingerprint import fingerprint_host
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        # Resolve hosts list to probe: prefer the (possibly updated) host_list
        # for this invocation, else read the persisted cluster.
        probe_hosts: list[str]
        if host_list is not None:
            probe_hosts = host_list
        else:
            try:
                probe_hosts = list(mgr.get(name).hosts)
            except ClusterError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)

        ssh_kwargs = build_ssh_kwargs(sctx.config)
        click.echo("Fingerprinting %d host(s)..." % len(probe_hosts))
        hosts_hardware: dict = {}
        for host in probe_hosts:
            click.echo("  %s ..." % host, nl=False)
            try:
                hw = fingerprint_host(host, ssh_kwargs)
            except Exception as e:
                click.echo(" FAILED (%s)" % e)
                continue
            hosts_hardware[host] = hw
            if hw.accelerators:
                summary = ", ".join("%dx %s/%s" % (a.count, a.vendor, a.model) for a in hw.accelerators)
                click.echo(" %s" % summary)
            else:
                click.echo(" no accelerators detected (%s)" % (hw.notes or "?"))
        if hosts_hardware:
            update_kwargs["hosts_hardware"] = hosts_hardware

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

    # Resolve the scheduler the cluster would actually launch with.  A cluster
    # predating the ``scheduler`` field stores ``None`` and falls back to
    # greedy; showing only the stored value (the old behavior) printed nothing
    # at all in that case, so there was no way to tell a greedy cluster from an
    # occupancy-aware one without launching something.
    from sparkrun.core.scheduler import describe_effective_scheduler

    effective_scheduler, scheduler_defaulted = describe_effective_scheduler(cluster=c.scheduler, v=_get_context(ctx).variables)

    if output_json:
        data = c.to_dict()
        data["default"] = c.name == default_name
        data["effective_scheduler"] = effective_scheduler
        data["scheduler_defaulted"] = scheduler_defaulted
        print_json(data)
        return

    click.echo(f"Name:        {c.name}")
    click.echo(f"Description: {c.description or '(none)'}")
    if c.transport and c.transport != "ssh":
        click.echo(f"Transport:   {c.transport}")
    if c.provider_ref:
        click.echo(f"Provider:    {c.provider_ref}")
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
    if c.mgmt_interface:
        click.echo(f"Mgmt iface:  {c.mgmt_interface}")
    if c.executor:
        click.echo(f"Executor:    {c.executor}")
    if c.executor_config:
        click.echo("Executor config:")
        for k, v in sorted(c.executor_config.items()):
            click.echo(f"  {k}: {v}")
    if scheduler_defaulted:
        click.echo(f"Scheduler:   {effective_scheduler} (default — not set on this cluster)")
    else:
        click.echo(f"Scheduler:   {effective_scheduler}")
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

    # Capture the definition before deletion so a provider transport can tear
    # down out-of-band state (e.g. the managed ssh alias) after it's gone.
    _cd = None
    try:
        _cd = mgr.get(name)
    except ClusterError:
        pass

    try:
        mgr.delete(name)
        click.echo(f"Cluster '{name}' deleted.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if _cd is not None:
        from sparkrun.transports import cleanup_cluster_transport

        cleanup_cluster_transport(_cd)


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
    from sparkrun.core.monitoring import stream_cluster_monitor
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = _get_context(ctx).config
    hctx = resolve_host_context(hosts, hosts_file, cluster_name, config)
    host_list = hctx.host_list
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
        # Stream MonitorFrames (telemetry + occupancy) as newline-delimited JSON.
        # ``hosts`` is a list of per-host rows (the canonical MonitorFrame shape,
        # shared with the desktop SSE): {host, error, sample, workloads, ...}.
        from sparkrun.core.monitoring import serialize_frame

        try:
            for frame in api.live_monitor(
                host_list,
                cluster=hctx.cluster_name,
                ssh_kwargs=ssh_kwargs,
                interval=interval,
                backend=backend,
                sctx=_get_context(ctx),
            ):
                print_json({"timestamp": frame.queried_at, "hosts": serialize_frame(frame)})
        except KeyboardInterrupt:
            pass
        return

    # Try the Textual TUI unless --simple was requested.
    if not simple:
        try:
            from sparkrun.cli._monitor_tui import ClusterMonitorApp

            # Single source: a LiveMonitorSession combines the substrate's
            # telemetry stream with a background api.status occupancy poll, so
            # the TUI shows local/provider workloads (not just docker).
            session = api.open_live_monitor(
                host_list,
                cluster=hctx.cluster_name,
                ssh_kwargs=ssh_kwargs,
                interval=interval,
                backend=backend,
                sctx=_get_context(ctx),
            )
            try:
                app = ClusterMonitorApp(session, host_list, interval=interval, cache_dir=str(config.cache_dir))
                app.run()
            finally:
                session.close()
            return
        except ImportError:
            click.echo("Textual not installed — falling back to simple mode.\n", err=True)

    # ---- simple plain-text fallback ----
    # Also flows from api.live_monitor (telemetry + occupancy), so the Jobs
    # column reflects all executors' workloads, not the docker-only count.
    from sparkrun.utils.cli_formatters import format_activity_table

    click.echo("Monitoring %d host(s) every %ds (Ctrl-C to stop)...\n" % (len(host_list), interval))

    # Number of lines the table occupies: header + separator + one row per host
    table_lines = len(host_list) + 2
    click.echo(format_activity_table(None, host_list))  # initial (connecting)
    try:
        for frame in api.live_monitor(
            host_list,
            cluster=hctx.cluster_name,
            ssh_kwargs=ssh_kwargs,
            interval=interval,
            backend=backend,
            sctx=_get_context(ctx),
        ):
            click.echo("\033[%dA\033[J" % table_lines, nl=False)
            click.echo(format_activity_table(frame, host_list))
    except KeyboardInterrupt:
        pass

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
    from sparkrun.utils.cli_formatters import format_job_label, format_job_commands, format_host_display, format_pending_op
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = _get_context(ctx)
    config = sctx.config
    hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx=sctx)
    host_list = hctx.host_list

    ssh_kwargs = build_ssh_kwargs(config)

    if dry_run:
        docker_cmd = "docker ps --filter 'name=sparkrun_' --format '{{.Names}}\\t{{.Status}}\\t{{.Image}}'"
        click.echo("[dry-run] Would run on %d host(s): %s" % (len(host_list), docker_cmd))
        return

    # All status flows from the single source, ``api.status_report`` — the
    # display tier over ``api.status``.  It owns executor resolution
    # (cluster-aware, so a cluster's ``executor_config`` incl. ``pid_dir`` is
    # honored), the cross-executor merge, and classification into the
    # display-oriented ``ClusterStatusResult`` (per-container role/status/image,
    # idle hosts, pending ops).  The *effective* cluster is forwarded (see
    # ``HostContext``) so a default-cluster sweep keeps that cluster's
    # executor pin and hardware instead of resolving to an anonymous one.
    result = api.status_report(host_list, cluster=hctx.cluster_name, ssh_kwargs=ssh_kwargs, sctx=sctx)

    if output_json:
        out = result.to_dict()
        for cid, group_data in out["groups"].items():
            group_data["label"] = format_job_label(group_data["meta"], cid)
        for entry_data in out["solo_entries"]:
            entry_data["label"] = format_job_label(entry_data["meta"], entry_data["cluster_id"])

        print_json(out)
        return

    # --- Display rendering ---

    # Say what is being reported on.  With no flags the host list comes from
    # the default cluster, and without this line there was no way to tell
    # which machines the report covers.
    click.echo(hctx.describe())
    click.echo()

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

    # Display hosts a launch is staging onto — free right now, but spoken for.
    if result.preparing_hosts:
        click.echo("Preparing (launch in progress, will consume VRAM):")
        for h in result.preparing_hosts:
            for op in result.pending_by_host.get(h, []):
                click.echo(f"  {h:<20s} {format_pending_op(op, with_detail=False)}")
        click.echo()

    # Display idle hosts
    if result.idle_hosts:
        click.echo("Idle hosts (no sparkrun containers, nothing pending):")
        for h in result.idle_hosts:
            click.echo(f"  {h}")
        click.echo()

    # Display pending operations
    if result.pending_ops:
        click.echo("Pending operations (downloads/distributions in progress):")
        for op in result.pending_ops:
            click.echo(f"  {format_pending_op(op)}")
            matched = op.get("matched_hosts") or []
            other = op.get("other_hosts") or []
            if matched:
                click.echo("    hosts: %s%s" % (", ".join(matched), " (+%d outside this cluster)" % len(other) if other else ""))
            else:
                # A lock that recorded no hosts cannot be pinned to any of
                # them; say so rather than leaving the reader to assume.
                click.echo("    hosts: not recorded")
        click.echo()
        click.echo("  Note: only launches started from this machine are visible here.")
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
        # Look up persisted metadata via the API (purely on-disk job
        # cache enumeration — no executor needed).  We still fall back
        # to ``load_job_metadata`` when the API enumeration misses
        # (older job files without started_at, etc.).
        meta: dict | None = None
        try:
            jobs = api.list_jobs(sctx=sctx)
        except Exception:
            jobs = []
        for j in jobs:
            if j.cluster_id == cid:
                meta = dict(j.metadata)
                break
        if meta is None:
            from sparkrun.orchestration.job_metadata import load_job_metadata

            meta = load_job_metadata(cid, cache_dir=str(config.cache_dir))

        # Resolve hosts: CLI flags > metadata > default cluster (None means "let check_job_running decide")
        host_list = None
        if hosts or hosts_file or cluster_name:
            host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)
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
        from sparkrun.cli._common import _load_recipe
        from sparkrun.core.parallelism import extract_parallelism
        from sparkrun.core.scheduler import SchedulingRequest
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe, _recipe_path, _registry_mgr = _load_recipe(config, target)
        host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)

        # Derive the effective host list via the scheduler so the
        # cluster_id matches what ``api.run`` would have produced.
        # ``hosts_used`` IS the effective list — no separate trimming step.
        trim_overrides: dict = {}
        if tp_override is not None:
            trim_overrides["tensor_parallel"] = tp_override
        if len(host_list) > 1:
            parallelism = extract_parallelism(recipe.build_config_chain(trim_overrides))
            if any(getattr(parallelism, k) > 1 for k in ("tensor_parallel", "pipeline_parallel", "data_parallel")):
                request = SchedulingRequest(
                    parallelism=parallelism,
                    hosts=tuple(host_list),
                    host_hardware=None,
                    layout=getattr(recipe, "layout", None),
                    resources=None,
                )
                try:
                    sched_result = api.schedule(request, sctx=sctx)
                except api.SparkrunError as e:
                    click.echo("Error: %s" % e, err=True)
                    sys.exit(1)
                host_list = list(sched_result.assignment.hosts_used)

        # Build overrides for cluster_id generation
        cid = derive_cluster_id(
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


def _format_head_hardware(summary: dict) -> list[tuple[str, str]]:
    """Render a :func:`~sparkrun.diagnostics.summarize_host_diagnostics` result.

    Returns ``(label, value)`` rows in display order, collapsing the fields
    that are only meaningful together (kernel/arch qualify the OS, core counts
    qualify the CPU, the storage driver and nvidia runtime qualify Docker).  A
    row is emitted only when the summary carries its primary field, so a host
    without ``nvcc`` or without Docker simply has fewer lines.
    """
    rows: list[tuple[str, str]] = []

    def add(label: str, value: str) -> None:
        if value:
            rows.append((label, value))

    product = summary.get("product", "")
    board = summary.get("board", "")
    if product and board and board != product:
        add("platform", "%s (board: %s)" % (product, board))
    else:
        add("platform", product or board)

    os_name = summary.get("os", "")
    os_detail = []
    if summary.get("kernel"):
        os_detail.append("kernel %s" % summary["kernel"])
    if summary.get("arch"):
        os_detail.append(summary["arch"])
    add("os", "%s (%s)" % (os_name, ", ".join(os_detail)) if os_name and os_detail else os_name)

    cpu = summary.get("cpu_model", "")
    cores, threads = summary.get("cpu_cores"), summary.get("cpu_threads")
    if cpu and cores:
        counts = "%d cores" % cores + (" / %d threads" % threads if threads and threads != cores else "")
        add("cpu", "%s (%s)" % (cpu, counts))
    else:
        add("cpu", cpu)

    if summary.get("ram_total_gb"):
        add("memory", "%.1f GB" % summary["ram_total_gb"])

    gpu = summary.get("gpu_name", "")
    if gpu and summary.get("gpu_memory_gb"):
        add("gpu", "%s (%.1f GB)" % (gpu, summary["gpu_memory_gb"]))
    else:
        add("gpu", gpu)

    add("gpu driver", summary.get("gpu_driver", ""))
    add("cuda (nvcc)", summary.get("cuda_version", ""))
    add("jetpack", summary.get("jetpack_version", ""))
    add("bios", summary.get("bios_version", ""))

    docker = summary.get("docker_version", "")
    if docker:
        detail = []
        if summary.get("docker_storage_driver"):
            detail.append("storage: %s" % summary["docker_storage_driver"])
        if "docker_nvidia_runtime" in summary:
            detail.append("nvidia runtime: %s" % ("yes" if summary["docker_nvidia_runtime"] else "no"))
        add("docker", "%s (%s)" % (docker, ", ".join(detail)) if detail else docker)

    return rows


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

    Also reports the head node's hardware and driver/software versions
    (platform, OS/kernel, CPU/RAM, GPU + driver, CUDA, Docker).  Run
    `sparkrun setup diagnose` for the full per-host inventory.

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

    from sparkrun.core.cluster_manager import resolve_cluster_config
    from sparkrun.core.launcher import resolve_effective_cache_dir
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = _get_context(ctx).config
    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    ssh_kwargs = build_ssh_kwargs(config)

    # Resolve effective cluster configuration
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_hf, remote_hf, xfer_mode, xfer_iface = cluster_cfg.resolve_transfer_config(config)
    # When the cluster has no explicit cache_dir, resolve_transfer_config returns
    # None.  Probe the head node so the inspected path reflects the remote user's
    # $HOME / HF_HOME instead of falling through as a literal "None".
    remote_hf = resolve_effective_cache_dir(remote_hf, host_list, ssh_kwargs, config, dry_run=dry_run)
    local_sparkrun = str(config.cache_dir)

    # Resolve auto transfer mode to a concrete value
    from sparkrun.orchestration.distribution import resolve_auto_transfer_mode

    xfer_result = resolve_auto_transfer_mode(
        xfer_mode,
        host_list,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        topology=cluster_cfg.topology,
        mgmt_interface=cluster_cfg.mgmt_interface,
    )
    resolved_mode = xfer_result.mode

    # Detect IB / NCCL env — reuse from transfer mode resolution if available,
    # otherwise run detection explicitly.
    ib_result = xfer_result.ib_result
    if ib_result is None and not dry_run:
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts

        ib_result = detect_ib_for_hosts(
            host_list, ssh_kwargs=ssh_kwargs, topology=cluster_cfg.topology, mgmt_interface=cluster_cfg.mgmt_interface
        )

    nccl_env = ib_result.comm_env.get_env(host_list[0]) if ib_result else {}

    # Resolve effective transfer interface
    # auto (None) → cx7 if IB is available and validated, else mgmt
    if xfer_iface == "mgmt":
        resolved_iface = "mgmt"
    elif xfer_result.ib_validated:
        resolved_iface = "cx7"
    elif ib_result and ib_result.ib_ip_map:
        resolved_iface = "cx7"
    elif ib_result:
        resolved_iface = "mgmt"
    else:
        resolved_iface = None

    if dry_run:
        click.echo("[dry-run] Would inspect cluster config and cache dirs on %d host(s)" % len(host_list))
        click.echo("[dry-run] Would probe head node hardware on %s" % host_list[0])
        return

    # TODO: remote sparkrun cache path should go through same effective route as remote hf cache resolution
    # Resolve remote sparkrun cache path the same way as before: explicit
    # for a known cluster user, otherwise $HOME-relative.
    if cluster_cfg.user:
        remote_sparkrun = "/home/%s/.cache/sparkrun" % cluster_cfg.user
    else:
        remote_sparkrun = "$HOME/.cache/sparkrun"

    from sparkrun.orchestration.disk_info import probe_cache_status, probe_local_cache_status
    from sparkrun.utils.cli_formatters import format_cache_status_table

    host_status = probe_cache_status(
        host_list,
        hf_cache_dir=remote_hf,
        sparkrun_cache_dir=remote_sparkrun,
        ssh_kwargs=ssh_kwargs,
    )
    local_status = probe_local_cache_status(
        hf_cache_dir=local_hf,
        sparkrun_cache_dir=local_sparkrun,
    )

    # Head-node hardware / driver versions.  Scoped to the head for the same
    # reason the NCCL env block is: one representative host answers "what am I
    # actually launching on", and a per-host sweep is what `setup diagnose`
    # already exists for.
    from sparkrun.diagnostics import collect_spark_diagnostics, summarize_host_diagnostics

    head_host = host_list[0]
    head_diag = collect_spark_diagnostics([head_host], ssh_kwargs=ssh_kwargs).get(head_host, {})
    head_hardware = summarize_host_diagnostics(head_diag)

    # Scheduler the cluster would launch with.  ``inspect`` reports *effective*
    # configuration, and an unset cluster scheduler still resolves to one — so
    # reporting the raw ``None`` would hide the thing most likely to explain a
    # surprising placement.
    from sparkrun.core.scheduler import describe_effective_scheduler

    effective_scheduler, scheduler_defaulted = describe_effective_scheduler(cluster=cluster_cfg.scheduler, v=_get_context(ctx).variables)

    # Collect effective config summary
    effective_config = {
        "cluster": cluster_cfg.name,
        "ssh_user": config.ssh_user,
        "transport": cluster_cfg.transport,
        "provider_ref": cluster_cfg.provider_ref,
        "scheduler": cluster_cfg.scheduler,
        "scheduler_resolved": effective_scheduler,
        "scheduler_defaulted": scheduler_defaulted,
        "executor": cluster_cfg.executor,
        "transfer_mode": xfer_mode,
        "transfer_mode_resolved": resolved_mode,
        "transfer_interface": xfer_iface or "auto",
        "transfer_interface_resolved": resolved_iface,
        "topology": cluster_cfg.topology,
        "mgmt_interface": cluster_cfg.mgmt_interface,
        "hf_cache_local": local_hf,
        "hf_cache_remote": remote_hf,
        "sparkrun_cache": local_sparkrun,
        "nccl_env": nccl_env,
    }

    if output_json:
        data = {
            "config": effective_config,
            "hosts": list(host_list),
            "head_node": {"host": head_host, "hardware": head_hardware},
            "local": {
                "sparkrun_cache": {
                    "path": local_status.sparkrun_dir,
                    "exists": local_status.sparkrun_exists,
                    "size": local_status.sparkrun_size,
                },
                "hf_cache": {"path": local_status.hf_dir, "exists": local_status.hf_exists, "size": local_status.hf_size},
                "free_space": local_status.free_space,
            },
            "remote": {},
        }
        for h in host_list:
            status = host_status.get(h)
            if status is None or status.error:
                data["remote"][h] = {"error": (status.error if status else "no result")}
            else:
                data["remote"][h] = {
                    "sparkrun_cache": {"path": status.sparkrun_dir, "exists": status.sparkrun_exists, "size": status.sparkrun_size},
                    "hf_cache": {"path": status.hf_dir, "exists": status.hf_exists, "size": status.hf_size},
                    "free_space": status.free_space,
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
    click.echo("  transport:          %s" % cluster_cfg.transport)
    if cluster_cfg.provider_ref:
        click.echo("  provider_ref:       %s" % cluster_cfg.provider_ref)

    def _fmt_resolved(configured: str, resolved: str | None) -> str:
        if resolved and configured != resolved:
            return "%s (resolved to: %s)" % (configured, resolved)
        return configured

    click.echo("  transfer_mode:      %s" % _fmt_resolved(xfer_mode, resolved_mode))
    cfg_iface = xfer_iface or "auto"
    click.echo("  transfer_interface: %s" % _fmt_resolved(cfg_iface, resolved_iface))
    click.echo("  topology:           %s" % (cluster_cfg.topology or "(none)"))
    click.echo("  mgmt_interface:     %s" % (cluster_cfg.mgmt_interface or "(auto-detect)"))
    if scheduler_defaulted:
        click.echo("  scheduler:          %s (default — not set on this cluster)" % effective_scheduler)
    else:
        click.echo("  scheduler:          %s" % effective_scheduler)
    if cluster_cfg.executor:
        click.echo("  executor:           %s" % cluster_cfg.executor)
    click.echo("  hosts:              %s" % ", ".join(host_list))
    click.echo()

    # Head-node hardware section
    click.echo("Head Node Hardware (%s):" % head_host)
    if head_hardware:
        for label, value in _format_head_hardware(head_hardware):
            click.echo("  %-19s %s" % (label + ":", value))
    else:
        click.echo("  (hardware probe failed — see `sparkrun setup diagnose` for details)")
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

    # Directory status table (shared formatter — also used by the
    # distribute_model_from_local error path on out-of-space failures).
    click.echo("Directory Status:")
    click.echo(format_cache_status_table(host_status, local_status=local_status))
