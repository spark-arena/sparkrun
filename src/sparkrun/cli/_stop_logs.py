"""sparkrun stop and logs commands — thin wrappers around ``sparkrun.api``.

The CLI layer parses Click flags, calls ``api.stop`` / ``api.logs``,
catches typed :class:`SparkrunError` subclasses, and renders the
results to the TTY.  Business logic (cluster_id derivation, executor
selection from metadata, parallel host dispatch, log streaming)
lives in :mod:`sparkrun.api`.

``--all`` is no different: discovery and teardown live in
:func:`sparkrun.api.stop_all`, and this module only prints the
discovered workloads between the two and maps the result onto an exit
code.  (It used to own that logic, which meant library callers — the
desktop sidecar included — could neither do it nor inherit its fixes.)
"""

from __future__ import annotations

import sys

import click

import sparkrun.api as api
from sparkrun.core.log_source import SCOPE_ALL, SCOPE_HEAD

from ._common import (
    TARGET,
    _get_context,
    _is_cluster_id,
    _load_recipe,
    build_cluster_id_overrides,
    dry_run_option,
    host_options,
    resolve_host_context,
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

    sctx = _get_context(ctx)
    config = sctx.config

    if stop_all:
        _stop_all(hosts, hosts_file, cluster_name, config, dry_run, sctx=sctx)
        return

    cluster_id = _is_cluster_id(target)
    overrides = build_cluster_id_overrides(port=port, served_model_name=served_model_name, tp_override=tp_override)

    try:
        if cluster_id is not None:
            host_list, effective_cluster = _hosts_for_cluster_id_target(target, hosts, hosts_file, cluster_name, config, sctx=sctx)
            result = api.stop(
                cluster_id=cluster_id,
                hosts=tuple(host_list) if host_list else None,
                cluster=effective_cluster,
                cache_dir=str(config.cache_dir),
                sctx=sctx,
            )
        else:
            # Resolve the recipe at the CLI layer so cwd-discovered recipes
            # are honoured (the CLI patches ``discover_cwd_recipes`` for
            # tests; api.stop's resolver doesn't see those overrides).
            recipe, _path, _reg = _load_recipe(config, target)
            # The *effective* cluster, not the raw --cluster flag: hosts may
            # have come from the default cluster, whose SSH user and executor
            # pin a bare host list would drop (see ``HostContext``).
            hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx=sctx)
            result = api.stop(
                recipe=recipe,
                hosts=tuple(hctx.host_list),
                overrides=overrides,
                cluster=hctx.cluster_name,
                cache_dir=str(config.cache_dir),
                sctx=sctx,
            )
    except api.AmbiguousWorkload as e:
        click.echo(
            "Error: Multiple workloads match this recipe/intent. Re-invoke with an explicit cluster_id (one of):",
            err=True,
        )
        for cid in e.cluster_ids:
            click.echo("  %s" % cid, err=True)
        sys.exit(1)
    except api.JobNotFound as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except api.SparkrunError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # A teardown that did not confirm is an error, not a warning: the
    # containers may still be running.  Same contract as ``--all``.
    for line in result.errors:
        click.echo("Error: %s" % line, err=True)
    if not result.success:
        # Never claim a stop that didn't happen.  The exit code has said so
        # since 0.3.0, but a trailing "Workload stopped on N host(s)." said
        # the opposite to every human and every log — and on a Spark the
        # workload it left behind still holds most of unified memory, so the
        # next launch fails for reasons that look unrelated (issue #277).
        scope = ", ".join(result.hosts_failed) if result.hosts_failed else "one or more of %s" % ", ".join(result.hosts_targeted)
        click.echo(
            "Workload NOT fully stopped: teardown did not confirm on %s\n"
            "  Containers may still be running and holding VRAM — check with 'sparkrun status' or 'docker ps'." % scope,
            err=True,
        )
        sys.exit(1)
    click.echo("Workload stopped on %d host(s)." % len(result.hosts_targeted))


def _hosts_for_cluster_id_target(target, hosts, hosts_file, cluster_name, config, sctx=None) -> tuple[list[str], str | None]:
    """Resolve the hosts *and effective cluster* for a cluster_id target.

    Mirrors the priority chain used by ``api.stop`` so we can pass an
    explicit host list when CLI flags supply one (overriding any
    metadata-recorded hosts).  See
    :func:`resolve_hosts_with_metadata_fallback` for why the cluster comes
    back as ``None`` on the metadata path.
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
        sctx=sctx,
    )


def _stop_all(hosts, hosts_file, cluster_name, config, dry_run, sctx=None):
    """Render ``sparkrun stop --all``.

    Discovery and teardown live in :func:`sparkrun.api.stop_all`; this
    prints the discovered workloads between the two (which is why the
    snapshot is passed back in rather than re-queried) and translates the
    result into output and an exit code.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx=sctx)
    host_list = hctx.host_list

    ssh_kwargs = build_ssh_kwargs(config)

    # Name the target before tearing anything down on it.
    click.echo(hctx.describe())
    click.echo("Discovering sparkrun containers on %d host(s)..." % len(host_list))
    # Status flows from the single source, ``api.status_report`` (cluster-aware
    # resolution + cross-executor merge + display classification).  The
    # *effective* cluster is forwarded (see ``HostContext``): teardown reaches
    # the right substrate only if discovery ran against the right one.
    discovered = api.status_report(host_list, cluster=hctx.cluster_name, ssh_kwargs=ssh_kwargs, sctx=sctx)

    # A host that errored during discovery may still be running containers —
    # it must not silently read as "nothing to stop".
    for err_host, err in sorted(discovered.errors.items()):
        click.echo("Error: could not query %s: %s" % (err_host, err), err=True)

    if discovered.total_containers == 0:
        if discovered.errors:
            click.echo("No sparkrun containers found on the hosts that could be queried.")
            sys.exit(1)
        click.echo("No sparkrun containers running.")
        return

    # Summarise what was found
    jobs_count = len(discovered.groups) + len(discovered.solo_entries)
    click.echo("Found %d job(s), %d container(s):" % (jobs_count, discovered.total_containers))
    for cid, group in discovered.groups.items():
        recipe_label = group.meta.get("recipe", "unknown")
        click.echo("  %s (%s) — %d container(s)" % (cid, recipe_label, len(group.members)))
    for entry in discovered.solo_entries:
        click.echo("  %s on %s" % (entry.name, entry.host))

    click.echo("Stopping all containers...")
    result = api.stop_all(
        host_list,
        cluster=hctx.cluster_name,
        cache_dir=str(config.cache_dir),
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        discovered=discovered,
        sctx=sctx,
    )

    click.echo(
        "Stopped %d job(s), %d container(s) across %d host(s)."
        % (result.jobs_stopped, result.containers_removed, len(result.hosts_stopped))
    )
    for failed_host, err in sorted(result.hosts_failed.items()):
        click.echo("Error: failed to stop containers on %s: %s" % (failed_host, err), err=True)
    if not result.success:
        sys.exit(1)


@click.command("logs")
@click.argument("target", type=TARGET)
@host_options
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@click.option("--port", type=int, default=None, help="Override port (to match run-time override)")
@click.option("--served-model-name", default=None, help="Override served model name (to match run-time override)")
@click.option(
    "-n",
    "--lines",
    "--tail",
    "lines",
    type=int,
    default=None,
    help="Number of log lines to show (default: all). Use with -f to set scrollback before following.",
)
@click.option("--follow", "-f", "follow", is_flag=True, default=False, help="Stay attached and stream new log lines (Ctrl-C to stop)")
@click.option(
    "--all-sources",
    "-a",
    "all_sources",
    is_flag=True,
    default=False,
    help="Read every worker/rank too, not just the head. Following interleaves them in arrival order.",
)
@click.pass_context
def logs_cmd(
    ctx, target, hosts, hosts_file, cluster_name, tp_override, port, served_model_name, lines, follow, all_sources, config_path=None
):
    """Show logs of a running workload (``docker logs`` / ``journalctl`` semantics).

    Called bare, dumps the workload's logs and exits.  Use ``-n`` to limit how
    many lines are shown, and ``-f``/``--follow`` to stay attached and stream
    new output.  ``-a``/``--all-sources`` reads every worker as well as the
    head; with ``-f`` those streams interleave in arrival order, and without it
    each source is dumped in full, head first, then workers by rank.

    TARGET can be a recipe name or a cluster ID (from sparkrun status output).

    Examples:

      sparkrun logs glm-4.7-flash-awq --hosts 192.168.11.13

      sparkrun logs glm-4.7-flash-awq --cluster mylab -n 200

      sparkrun logs e5f6a7b8 -f -a
    """
    sctx = _get_context(ctx)

    cluster_id_arg = _is_cluster_id(target)
    overrides = build_cluster_id_overrides(port=port, served_model_name=served_model_name, tp_override=tp_override)
    scope = SCOPE_ALL if all_sources else SCOPE_HEAD

    # Both targets render the same ``api.logs`` iterator; they differ only in
    # how the workload is addressed (literal id vs recipe → live intent
    # discovery).  Host resolution stays here because the cluster_id form can
    # fall back to the hosts recorded in job metadata.
    if cluster_id_arg is not None:
        host_list, effective_cluster = resolve_hosts_with_metadata_fallback(
            hosts,
            hosts_file,
            cluster_name,
            sctx.config,
            _load_job_metadata(sctx, cluster_id_arg),
            target,
            sctx=sctx,
        )
        _render_logs(sctx, cluster_id=cluster_id_arg, hosts=host_list, cluster=effective_cluster, scope=scope, lines=lines, follow=follow)
        return

    # Recipe target — resolve through the CLI's loader so cwd-discovered
    # recipes and registry disambiguation behave as they do for ``run``, then
    # let api.logs find the live workload by intent.
    recipe, _path, _reg = _load_recipe(sctx.config, target)
    hctx = resolve_host_context(hosts, hosts_file, cluster_name, sctx.config, sctx=sctx)
    _render_logs(
        sctx,
        recipe=recipe,
        hosts=hctx.host_list,
        cluster=hctx.cluster_name,
        overrides=overrides,
        scope=scope,
        lines=lines,
        follow=follow,
    )


def _load_job_metadata(sctx, cluster_id):
    from sparkrun.orchestration.job_metadata import load_job_metadata

    return load_job_metadata(cluster_id, cache_dir=str(sctx.config.cache_dir))


def _render_logs(sctx, *, cluster_id=None, recipe=None, hosts, cluster=None, overrides=None, scope, lines, follow):
    """Render the ``api.logs`` iterator — the CLI's whole job for logs.

    Lines are prefixed with their source's ``host/role`` only when more than
    one source is in play, so the common single-source case stays clean enough
    to pipe.
    """
    try:
        stream = api.logs(
            cluster_id,
            recipe=recipe,
            hosts=tuple(hosts),
            cluster=cluster,
            overrides=overrides,
            scope=scope,
            tail=lines,
            follow=follow,
            cache_dir=str(sctx.config.cache_dir),
            sctx=sctx,
        )
        multi = scope == SCOPE_ALL and len(hosts) > 1
        for line in stream:
            click.echo("[%s/%s] %s" % (line.host, line.role, line.text) if multi else line.text)
    except (api.JobNotFound, api.AmbiguousWorkload, api.SparkrunError) as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("")  # leave the shell prompt on its own line
