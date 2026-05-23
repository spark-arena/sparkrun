"""sparkrun stop and logs commands — thin wrappers around ``sparkrun.api``.

The CLI layer parses Click flags, calls ``api.stop`` / ``api.logs``,
catches typed :class:`SparkrunError` subclasses, and renders the
results to the TTY.  Business logic (cluster_id derivation, executor
selection from metadata, parallel host dispatch, log streaming)
lives in :mod:`sparkrun.api`.

The ``--all`` discovery path remains in this module because it has
no API equivalent yet — that discovery is presentation-heavy (lists
running clusters by recipe) and would expand the API surface for
limited benefit.
"""

from __future__ import annotations

import sys

import click

import sparkrun.api as api

from ._common import (
    TARGET,
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
        return

    cluster_id = _is_cluster_id(target)
    overrides = build_cluster_id_overrides(port=port, served_model_name=served_model_name, tp_override=tp_override)

    try:
        if cluster_id is not None:
            host_list = _hosts_for_cluster_id_target(target, hosts, hosts_file, cluster_name, config)
            result = api.stop(
                cluster_id=cluster_id,
                hosts=tuple(host_list) if host_list else None,
                cluster=cluster_name,
                cache_dir=str(config.cache_dir),
            )
        else:
            # Resolve the recipe at the CLI layer so cwd-discovered recipes
            # are honoured (the CLI patches ``discover_cwd_recipes`` for
            # tests; api.stop's resolver doesn't see those overrides).
            recipe, _path, _reg = _load_recipe(config, target)
            host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
            result = api.stop(
                recipe=recipe,
                hosts=tuple(host_list),
                overrides=overrides,
                cluster=cluster_name,
                cache_dir=str(config.cache_dir),
            )
    except api.JobNotFound as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except api.SparkrunError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    if result.errors:
        for line in result.errors:
            click.echo("Warning: %s" % line, err=True)
    click.echo("Workload stopped on %d host(s)." % len(result.hosts_targeted))


def _hosts_for_cluster_id_target(target, hosts, hosts_file, cluster_name, config) -> list[str]:
    """Resolve the host list for a cluster_id target.

    Mirrors the priority chain used by ``api.stop`` so we can pass an
    explicit host list when CLI flags supply one (overriding any
    metadata-recorded hosts).
    """
    from sparkrun.orchestration.job_metadata import load_job_metadata

    cluster_id = _is_cluster_id(target)
    meta = load_job_metadata(cluster_id, cache_dir=str(config.cache_dir))
    return resolve_hosts_with_metadata_fallback(
        hosts,
        hosts_file,
        cluster_name,
        config,
        meta,
        target,
    )


def _stop_all(hosts, hosts_file, cluster_name, config, dry_run):
    """Discover and stop all sparkrun containers on the target hosts.

    Kept inline because the discovery-heavy presentation has no API
    equivalent — it prints a summary of running clusters before
    issuing teardown commands.
    """
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
        for host, role, status, image in group.members:
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


@click.command("logs")
@click.argument("target", type=TARGET)
@host_options
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@click.option("--port", type=int, default=None, help="Override port (to match run-time override)")
@click.option("--served-model-name", default=None, help="Override served model name (to match run-time override)")
@click.option("--tail", type=int, default=100, help="Number of log lines before following")
@click.pass_context
def logs_cmd(ctx, target, hosts, hosts_file, cluster_name, tp_override, port, served_model_name, tail, config_path=None):
    """Re-attach to logs of a running workload.

    TARGET can be a recipe name or a cluster ID (from sparkrun status output).

    Examples:

      sparkrun logs glm-4.7-flash-awq --hosts 192.168.11.13

      sparkrun logs glm-4.7-flash-awq --cluster mylab --tail 200

      sparkrun logs e5f6a7b8
    """
    sctx = _get_context(ctx)

    cluster_id_arg = _is_cluster_id(target)
    overrides = build_cluster_id_overrides(port=port, served_model_name=served_model_name, tp_override=tp_override)

    if cluster_id_arg is not None:
        # Cluster ID target — load metadata so we can route to the same
        # runtime that launched the workload.  This keeps the existing
        # behaviour for SGLang/vLLM runtime-specific follow_logs (which
        # may attach to multiple containers) while delegating the simple
        # head-container case to api.logs.
        _follow_logs_by_cluster_id(sctx, cluster_id_arg, target, hosts, hosts_file, cluster_name, tail)
        return

    # Recipe target — derive the cluster_id via the recipe + hosts path
    # and then route to the runtime's follow_logs (multi-container aware).
    _follow_logs_by_recipe(sctx, target, hosts, hosts_file, cluster_name, overrides, tail)


def _follow_logs_by_cluster_id(sctx, cluster_id, target, hosts, hosts_file, cluster_name, tail):
    """Follow logs for a cluster_id, picking the right runtime when possible.

    Uses the recipe-aware ``runtime.follow_logs`` (attaches to multiple
    containers in cluster mode) when metadata records a runtime; falls
    back to ``api.logs`` head-container streaming for the simple case.
    """
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.orchestration.job_metadata import load_job_metadata

    config = sctx.config
    v = sctx.variables
    meta = load_job_metadata(cluster_id, cache_dir=str(config.cache_dir))

    runtime = None
    runtime_name = meta.get("runtime") if meta else None
    if runtime_name:
        try:
            runtime = get_runtime(runtime_name, v)
        except ValueError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)

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
        runtime.follow_logs(hosts=host_list, cluster_id=cluster_id, config=config, tail=tail)
        return

    # No runtime context — stream via api.logs (head container only).
    try:
        for line in api.logs(cluster_id, hosts=tuple(host_list), tail=tail, follow=True, cache_dir=str(config.cache_dir)):
            click.echo(line.text)
    except api.JobNotFound as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)


def _follow_logs_by_recipe(sctx, recipe_name, hosts, hosts_file, cluster_name, overrides, tail):
    """Follow logs for a recipe target — uses the runtime's multi-container ``follow_logs``."""
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    config = sctx.config
    v = sctx.variables

    # Use the CLI's recipe loader so cwd-discovered recipes (tests
    # monkey-patch ``discover_cwd_recipes``) and registry disambiguation
    # prompts work the same as for ``sparkrun run``.
    recipe, _path, _reg = _load_recipe(config, recipe_name)
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)

    # Derive cluster_id consistent with what api.run / api.schedule would
    # have produced — no separate trimming step; the host list as
    # supplied IS the effective list at the API boundary.
    cluster_id = generate_cluster_id(recipe, host_list, overrides=overrides)
    runtime.follow_logs(hosts=host_list, cluster_id=cluster_id, config=config, tail=tail)
