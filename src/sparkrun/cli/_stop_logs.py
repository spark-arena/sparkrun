"""sparkrun stop and logs commands."""

from __future__ import annotations

import sys

import click

from ._common import (
    TARGET,
    _apply_node_trimming,
    _get_context,
    _is_cluster_id,
    _load_recipe,
    _resolve_hosts_or_exit,
    build_cluster_id_overrides,
    dry_run_option,
    host_options,
    resolve_hosts_with_metadata_fallback,
)


@click.command()
@click.argument("target", type=TARGET, required=False, default=None)
@host_options
@click.option("--all", "-a", "stop_all", is_flag=True, default=False, help="Stop all sparkrun containers (discovers via docker ps)")
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@click.option("--port", type=int, default=None, help="Override port (to match run-time override)")
@click.option("--served-model-name", default=None, help="Override served model name (to match run-time override)")
@dry_run_option
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def stop(ctx, target, hosts, hosts_file, cluster_name, stop_all, tp_override, port, served_model_name, dry_run, config_path=None):
    """Stop a running workload.

    TARGET can be a recipe name or a cluster ID (from sparkrun status output).
    Use --all to discover and stop all sparkrun containers without specifying a target.

    Examples:

      sparkrun stop glm-4.7-flash-awq --hosts 192.168.11.13,192.168.11.14

      sparkrun stop glm-4.7-flash-awq --cluster mylab

      sparkrun stop e5f6a7b8

      sparkrun stop --all --cluster mylab

      sparkrun stop --all --hosts 192.168.11.13,192.168.11.14
    """
    if stop_all and target:
        click.echo("Error: --all and TARGET are mutually exclusive.", err=True)
        sys.exit(1)

    if not stop_all and not target:
        click.echo("Error: Must specify TARGET or --all.", err=True)
        sys.exit(1)

    config = _get_context(ctx).config

    if stop_all:
        _stop_all(hosts, hosts_file, cluster_name, config, dry_run)
    elif _is_cluster_id(target) is not None:
        _stop_by_cluster_id(target, hosts, hosts_file, cluster_name, config, dry_run)
    else:
        _stop_recipe(target, hosts, hosts_file, cluster_name, config, tp_override, dry_run, port=port, served_model_name=served_model_name)


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
    for entry in result.solo_entries:
        click.echo("  %s on %s" % (entry.name, entry.host))

    # Build per-host container name mapping
    host_containers: dict[str, list[str]] = {}
    for cid, group in result.groups.items():
        for host, role, _status, _image in group.members:
            container_name = "%s_%s" % (cid, role)
            host_containers.setdefault(host, []).append(container_name)
    for entry in result.solo_entries:
        host_containers.setdefault(entry.host, []).append(entry.name)

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
        for entry in result.solo_entries:
            solo_cid = entry.name.removesuffix("_solo") if entry.name.endswith("_solo") else entry.name
            remove_job_metadata(solo_cid, cache_dir=str(config.cache_dir))

    hosts_touched = len(host_containers)
    click.echo("Stopped %d job(s) across %d host(s)." % (stopped_count, hosts_touched))


def _stop_by_cluster_id(target, hosts, hosts_file, cluster_name, config, dry_run):
    """Stop containers identified by cluster ID.

    First tries to load job metadata for host info.  When metadata is
    unavailable (e.g. on a worker node that didn't launch the job),
    falls back to resolving hosts from CLI flags or the default cluster
    and stops containers by enumerating the known cluster_id patterns.
    """
    from sparkrun.orchestration.docker import enumerate_cluster_containers
    from sparkrun.orchestration.job_metadata import load_job_metadata, remove_job_metadata
    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers, cleanup_containers_local, should_run_locally

    cluster_id = _is_cluster_id(target)
    assert cluster_id is not None
    meta = load_job_metadata(cluster_id, cache_dir=str(config.cache_dir))

    # Resolve hosts: CLI flags > metadata > default cluster
    host_list = resolve_hosts_with_metadata_fallback(
        hosts,
        hosts_file,
        cluster_name,
        config,
        meta,
        target,
    )

    ssh_kwargs = build_ssh_kwargs(config)
    container_names = enumerate_cluster_containers(cluster_id, len(host_list))

    is_local = len(host_list) == 1 and should_run_locally(host_list[0], ssh_kwargs.get("ssh_user"))
    if is_local:
        cleanup_containers_local(container_names, dry_run=dry_run)
    else:
        cleanup_containers(host_list, container_names, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    if not dry_run:
        remove_job_metadata(cluster_id, cache_dir=str(config.cache_dir))

    click.echo("Workload stopped on %d host(s)." % len(host_list))


def _stop_recipe(recipe_name, hosts, hosts_file, cluster_name, config, tp_override, dry_run, port=None, served_model_name=None):
    """Stop containers for a specific recipe (original behaviour)."""
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime

    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)
    assert recipe is not None

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
            host_list,
            recipe,
            tp_override=tp_override,
            runtime=runtime,
        )
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers, cleanup_containers_local, should_run_locally
    from sparkrun.orchestration.docker import enumerate_cluster_containers
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    # Build overrides from --port and --served-model-name so cluster_id matches the run
    cluster_id = generate_cluster_id(
        recipe, host_list, overrides=build_cluster_id_overrides(port=port, served_model_name=served_model_name)
    )
    ssh_kwargs = build_ssh_kwargs(config)

    container_names = enumerate_cluster_containers(cluster_id, len(host_list))

    is_local = len(host_list) == 1 and should_run_locally(host_list[0], ssh_kwargs.get("ssh_user"))
    if is_local:
        cleanup_containers_local(container_names, dry_run=dry_run)
    else:
        cleanup_containers(host_list, container_names, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    click.echo("Workload stopped on %d host(s)." % len(host_list))


@click.command("logs")
@click.argument("target", type=TARGET)
@host_options
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@click.option("--port", type=int, default=None, help="Override port (to match run-time override)")
@click.option("--served-model-name", default=None, help="Override served model name (to match run-time override)")
@click.option("--tail", type=int, default=100, help="Number of log lines before following")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def logs_cmd(ctx, target, hosts, hosts_file, cluster_name, tp_override, port, served_model_name, tail, config_path=None):
    """Re-attach to logs of a running workload.

    TARGET can be a recipe name or a cluster ID (from sparkrun status output).

    Examples:

      sparkrun logs glm-4.7-flash-awq --hosts 192.168.11.13

      sparkrun logs glm-4.7-flash-awq --cluster mylab --tail 200

      sparkrun logs e5f6a7b8
    """
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    # Branch: cluster ID target
    if _is_cluster_id(target) is not None:
        cluster_id = _is_cluster_id(target)
        assert cluster_id is not None
        from sparkrun.orchestration.job_metadata import load_job_metadata

        meta = load_job_metadata(cluster_id, cache_dir=str(config.cache_dir))

        # Resolve runtime — from metadata if available, otherwise need hosts
        # to discover the container and fall back to generic docker logs
        runtime_name = meta.get("runtime") if meta else None
        if runtime_name:
            try:
                runtime = get_runtime(runtime_name, v)
            except ValueError as e:
                click.echo("Error: %s" % e, err=True)
                sys.exit(1)
        else:
            runtime = None

        # Resolve hosts: CLI flags > metadata > default cluster
        host_list = resolve_hosts_with_metadata_fallback(
            hosts,
            hosts_file,
            cluster_name,
            config,
            meta,
            target,
            sctx=sctx,
        )

        if runtime is not None:
            runtime.follow_logs(
                hosts=host_list,
                cluster_id=cluster_id,
                config=config,
                tail=tail,
            )
        else:
            # No metadata / unknown runtime — fall back to generic docker logs
            from sparkrun.orchestration.primitives import build_ssh_kwargs
            from sparkrun.orchestration.ssh import stream_remote_logs

            ssh_kwargs = build_ssh_kwargs(config)
            container_name = cluster_id + "_head" if len(host_list) > 1 else cluster_id + "_solo"
            stream_remote_logs(host_list[0], container_name, tail=tail, **ssh_kwargs)
        return

    # Branch: recipe name target (original path)
    recipe_name = target

    # Load recipe
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)
    assert recipe is not None

    # Resolve hosts
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)

    # Resolve runtime so we call the correct follow_logs implementation
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Apply runtime-aware host trimming to match what 'run' used for cluster_id
    try:
        host_list = _apply_node_trimming(
            host_list,
            recipe,
            tp_override=tp_override,
            runtime=runtime,
        )
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Build overrides from --port and --served-model-name so cluster_id matches the run
    cluster_id = generate_cluster_id(
        recipe, host_list, overrides=build_cluster_id_overrides(port=port, served_model_name=served_model_name)
    )

    runtime.follow_logs(
        hosts=host_list,
        cluster_id=cluster_id,
        config=config,
        tail=tail,
    )
