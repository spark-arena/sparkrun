"""sparkrun stop and logs commands."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    _apply_node_trimming,
    _load_recipe,
    _resolve_hosts_or_exit,
    _setup_logging,
    dry_run_option,
    host_options,
)


@click.command()
@click.argument("recipe_name", type=RECIPE_NAME, required=False, default=None)
@host_options
@click.option("--all", "-a", "stop_all", is_flag=True, default=False,
              help="Stop all sparkrun containers (discovers via docker ps)")
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None,
              help="Tensor parallel (to match host trimming from run)")
@dry_run_option
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def stop(ctx, recipe_name, hosts, hosts_file, cluster_name, stop_all, tp_override, dry_run, config_path=None):
    """Stop a running workload.

    RECIPE_NAME identifies the recipe so the correct containers can be found.
    Use --all to discover and stop all sparkrun containers without specifying a recipe.

    Examples:

      sparkrun stop glm-4.7-flash-awq --hosts 192.168.11.13,192.168.11.14

      sparkrun stop glm-4.7-flash-awq --cluster mylab

      sparkrun stop --all --cluster mylab

      sparkrun stop --all --hosts 192.168.11.13,192.168.11.14
    """
    if stop_all and recipe_name:
        click.echo("Error: --all and RECIPE_NAME are mutually exclusive.", err=True)
        sys.exit(1)

    if not stop_all and not recipe_name:
        click.echo("Error: Must specify RECIPE_NAME or --all.", err=True)
        sys.exit(1)

    from sparkrun.core.config import SparkrunConfig
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    if stop_all:
        _stop_all(hosts, hosts_file, cluster_name, config, dry_run)
    else:
        _stop_recipe(recipe_name, hosts, hosts_file, cluster_name, config, tp_override, dry_run)


def _stop_all(hosts, hosts_file, cluster_name, config, dry_run):
    """Discover and stop all sparkrun containers on the target hosts."""
    from sparkrun.core.cluster_manager import query_cluster_status
    from sparkrun.orchestration.docker import docker_stop_cmd
    from sparkrun.orchestration.job_metadata import remove_job_metadata
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.ssh import run_remote_command

    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)

    ssh_kwargs = build_ssh_kwargs(config)

    click.echo("Discovering sparkrun containers on %d host(s)..." % len(host_list))
    result = query_cluster_status(host_list, ssh_kwargs=ssh_kwargs, cache_dir=str(config.cache_dir))

    if result.total_containers == 0:
        click.echo("No sparkrun containers running.")
        return

    # Summarise what was found
    jobs_count = len(result.groups) + len(result.solo_entries)
    click.echo("Found %d job(s), %d container(s):" % (jobs_count, result.total_containers))
    for cid, group in result.groups.items():
        recipe_label = group.meta.get("recipe", "unknown")
        click.echo("  %s (%s) — %d container(s)" % (cid, recipe_label, len(group.members)))
    for host, name, status, image in result.solo_entries:
        click.echo("  %s on %s" % (name, host))

    # Build per-host container name mapping
    host_containers: dict[str, list[str]] = {}
    for cid, group in result.groups.items():
        for host, role, status, image in group.members:
            container_name = "%s_%s" % (cid, role)
            host_containers.setdefault(host, []).append(container_name)
    for host, name, status, image in result.solo_entries:
        host_containers.setdefault(host, []).append(name)

    # Stop containers per host
    click.echo("Stopping all containers...")
    stopped_count = 0
    for host, names in host_containers.items():
        cmds = "; ".join(docker_stop_cmd(n) for n in names)
        run_remote_command(host, cmds, timeout=30, dry_run=dry_run, **ssh_kwargs)
        stopped_count += len(names)

    # Clean up job metadata for discovered clusters (skip in dry-run mode)
    if not dry_run:
        for cid in result.groups:
            remove_job_metadata(cid, cache_dir=str(config.cache_dir))
        for _host, name, _status, _image in result.solo_entries:
            solo_cid = name.removesuffix("_solo") if name.endswith("_solo") else name
            remove_job_metadata(solo_cid, cache_dir=str(config.cache_dir))

    hosts_touched = len(host_containers)
    click.echo("Stopped %d job(s) across %d host(s)." % (stopped_count, hosts_touched))


def _stop_recipe(recipe_name, hosts, hosts_file, cluster_name, config, tp_override, dry_run):
    """Stop containers for a specific recipe (original behaviour)."""
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime

    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)

    # Resolve runtime for accurate node trimming (accounts for PP, etc.)
    v = init_sparkrun()
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError:
        runtime = None

    # Apply runtime-aware host trimming to match what 'run' used for cluster_id
    try:
        host_list = _apply_node_trimming(
            host_list, recipe, tp_override=tp_override, runtime=runtime,
        )
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers, cleanup_containers_local
    from sparkrun.orchestration.docker import enumerate_cluster_containers
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    cluster_id = generate_cluster_id(recipe, host_list)
    ssh_kwargs = build_ssh_kwargs(config)

    container_names = enumerate_cluster_containers(cluster_id, len(host_list))

    from sparkrun.core.hosts import is_local_host
    is_local = len(host_list) == 1 and is_local_host(host_list[0])
    if is_local:
        cleanup_containers_local(container_names, dry_run=dry_run)
    else:
        cleanup_containers(host_list, container_names, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    click.echo("Workload stopped on %d host(s)." % len(host_list))
    sys.exit(0)


@click.command("logs")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@click.option("--tail", type=int, default=100, help="Number of log lines before following")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def logs_cmd(ctx, recipe_name, hosts, hosts_file, cluster_name, tp_override, tail, config_path=None):
    """Re-attach to logs of a running workload.

    RECIPE_NAME identifies the recipe so the correct containers can be found.

    Examples:

      sparkrun logs glm-4.7-flash-awq --hosts 192.168.11.13

      sparkrun logs glm-4.7-flash-awq --cluster mylab --tail 200
    """
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    v = init_sparkrun()
    _setup_logging(ctx.obj["verbose"])
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # Load recipe
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    # Resolve hosts
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v)

    # Resolve runtime so we call the correct follow_logs implementation
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Apply runtime-aware host trimming to match what 'run' used for cluster_id
    try:
        host_list = _apply_node_trimming(
            host_list, recipe, tp_override=tp_override, runtime=runtime,
        )
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    cluster_id = generate_cluster_id(recipe, host_list)

    runtime.follow_logs(
        hosts=host_list,
        cluster_id=cluster_id,
        config=config,
        tail=tail,
    )
