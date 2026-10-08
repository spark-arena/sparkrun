"""Interactive rendering and confirmation for console-free setup actions."""

from __future__ import annotations

import click

from sparkrun.core.setup_models import OK, FAIL


def run_host_steps(
    states, context, action_context, *, yes, manifest_mgr, cluster_name, ensure_password, host_password=None, only_steps=None
):
    from ._check import _render_host
    from .._common import _get_cluster_manager
    from sparkrun.api.setup import run_setup_steps, SetupFailed

    def render(event):
        if event.kind == "preview":
            click.echo("  [dry-run] %s / %s: %s" % (event.hosts[0], event.label, event.detail))
        elif event.kind == "selected":
            click.echo("%s: %s" % (event.label, ", ".join(event.hosts)))
        elif event.kind == "error":
            click.echo("  %s: %s" % (event.hosts[0], event.detail), err=True)
        elif event.kind == "result":
            click.echo("  %s: [%s] %s" % (event.hosts[0], event.status.upper(), event.detail))

    click.echo()
    click.echo("Host setup steps")
    click.echo("-" * 30)
    from sparkrun.core.cluster_manager import ClusterError

    try:
        cluster = _get_cluster_manager().get(cluster_name) if cluster_name and not action_context.dry_run else None
    except ClusterError:
        # A wizard can be evaluating an unsaved cluster. Its probed states and
        # explicit host set are sufficient for the shared runner.
        cluster = None
    try:
        result = run_setup_steps(
            states,
            context,
            action_context,
            cluster=cluster,
            cluster_name=cluster_name,
            manifest_mgr=manifest_mgr,
            only_steps=only_steps,
            credentials=ensure_password,
            host_credentials=host_password,
            approve=lambda step, hosts: yes or click.confirm("Apply this step to these hosts?", default=True),
            progress_callback=render,
        )
    except SetupFailed as exc:
        raise click.ClickException(str(exc)) from exc
    states.update(result.states)
    if not action_context.dry_run:
        for host, entries in result.plans.items():
            _render_host(host, [item for entry in entries if entry.selected for item in entry.checks])
            for entry in entries:
                if entry.reason and entry.reason != "disabled by application or user policy":
                    click.echo("  [SKIP] %s: %s" % (entry.step.label, entry.reason))
        result.steps["readiness"] = "needs attention (%d findings)" % result.findings if result.findings else OK
        click.echo("Readiness: %s" % result.steps["readiness"])
    return result.steps


def run_setup_step_command(key, hosts, hosts_file, cluster_name, user, dry_run):
    """Drive a single shared step from a standalone setup command."""
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.setup_actions import SetupActionContext
    from sparkrun.core.setup_models import HostState
    from sparkrun.core.setup_probe import probe_setup_hosts, resolve_setup_context
    from sparkrun.core.setup_manifest import ManifestManager
    from .._common import _resolve_setup_context, _get_cluster_manager
    from ._sudo import ensure_sudo_password, host_password_prompt

    config = SparkrunConfig()
    from sparkrun.core.setup_steps import step_enabled

    if not step_enabled(key, config):
        raise click.ClickException("Setup step %s is disabled; enable setup.steps.%s first" % (key, key))
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)
    manager = _get_cluster_manager()
    name = cluster_name or (manager.get_default() if not (hosts or hosts_file) else None)
    cluster = manager.get(name) if name else None
    if dry_run:
        inventory = cluster.hosts_hardware if cluster else {}
        states = {h: HostState(h, hardware=inventory.get(h)) for h in host_list}
        context = resolve_setup_context(states, config=config, cluster=cluster, cluster_name=name)
    else:
        states, context = probe_setup_hosts(host_list, ssh_kwargs=ssh_kwargs, config=config, cluster=cluster, cluster_name=name)
    action = SetupActionContext(user, ssh_kwargs, dry_run)
    password_collected = False

    def password():
        nonlocal password_collected
        if not password_collected:
            action.sudo_password, _ = ensure_sudo_password(host_list, user, ssh_kwargs, dry_run=dry_run)
            password_collected = True
        return action.sudo_password

    results = run_host_steps(
        states,
        context,
        action,
        yes=True,
        manifest_mgr=None if dry_run or not name else ManifestManager(manager.clusters_dir),
        cluster_name=name,
        ensure_password=password,
        host_password=None if dry_run else host_password_prompt(user),
        only_steps={key},
    )
    if not dry_run and results.get(key) == FAIL:
        raise click.ClickException("Setup step %s failed on one or more hosts; see results above" % key)


def require_setup_step_targets(key, host_list, ssh_kwargs, config, *, cluster_name, explicit_hosts, dry_run):
    """Require an owned plan before standalone host changes or fabric probes."""
    from .._common import _get_cluster_manager
    from sparkrun.core.setup_models import HostState
    from sparkrun.core.setup_probe import probe_setup_hosts, resolve_setup_context
    from sparkrun.core.setup_steps import setup_step_reason

    manager = _get_cluster_manager()
    name = cluster_name or (manager.get_default() if not explicit_hosts else None)
    cluster = manager.get(name) if name else None
    if dry_run:
        inventory = cluster.hosts_hardware if cluster else {}
        states = {host: HostState(host, hardware=inventory.get(host)) for host in host_list}
        context = resolve_setup_context(states, config=config, cluster=cluster, cluster_name=name, strict=False)
    else:
        states, context = probe_setup_hosts(
            host_list,
            ssh_kwargs=ssh_kwargs,
            config=config,
            cluster=cluster,
            cluster_name=name,
            discovery_only=True,
        )
    denied = ["%s: %s" % (host, reason) for host, state in states.items() if (reason := setup_step_reason(key, state, context))]
    if denied:
        raise click.ClickException("Setup step %s is unavailable; %s" % (key, "; ".join(denied)))
