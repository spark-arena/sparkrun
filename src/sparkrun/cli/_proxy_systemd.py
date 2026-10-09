"""``sparkrun proxy systemd`` — run the inference proxy as a systemd service.

A thin renderer over :mod:`sparkrun.api.proxy._service`.
"""

from __future__ import annotations

import getpass
import sys

import click

from sparkrun.core.application_profile import render_identity_text

from ._common import CLUSTER_NAME, dry_run_option, json_option, print_json
from ._proxy import proxy


def _with_sudo(call):
    """Run *call(password)* without a password first; prompt once if sudo needs one."""
    from sparkrun import api

    try:
        return call(None)
    except api.proxy.SudoPasswordRequired as exc:
        click.echo(str(exc))
        password = click.prompt("[sudo] password for %s" % getpass.getuser(), hide_input=True)
        return call(password)


def _scope_from_flags(system: bool, user: bool) -> str | None:
    if system and user:
        raise click.UsageError("--system and --user are mutually exclusive.")
    return "system" if system else "user" if user else None


@proxy.group("systemd")
def proxy_systemd():
    """Run the proxy as a systemd service that starts at boot.

    The unit runs '{app_command} proxy start --foreground' with no other flags,
    so it serves whatever proxy.yaml says each time it starts (port, host,
    gateway, discovery cluster). Once installed, '{app_command} proxy start'
    and 'proxy stop' go through the unit.

    By default this installs a system unit (/etc/systemd/system, needs sudo
    once), plus a sudoers grant that lets you start, stop and restart exactly
    that unit without a password. --user installs a user unit instead, which
    starts at boot only while lingering is enabled.
    """


@proxy_systemd.command("install")
@click.option("--user", "user_scope", is_flag=True, help="Install a user unit (no sudo; boot start needs lingering).")
@click.option(
    "--cluster",
    "cluster_name",
    type=CLUSTER_NAME,
    default=None,
    help="Discovery cluster to save in proxy.yaml. Without it, the default cluster is used at each start.",
)
@click.option("--now", is_flag=True, help="Start (or restart) the unit now, replacing a proxy running outside it.")
@dry_run_option
def install_cmd(user_scope, cluster_name, now, dry_run):
    """Install (or update) the proxy unit and enable it.

    Examples:

      {app_command} proxy systemd install --now

      {app_command} proxy systemd install --cluster mylab

      {app_command} proxy systemd install --user --now
    """
    from sparkrun import api

    sctx = click.get_current_context().ensure_object(dict).get("sparkrun_ctx")

    def call(password):
        options = api.proxy.ProxyServiceOptions(
            scope="user" if user_scope else "system",
            cluster=cluster_name,
            now=now,
            dry_run=dry_run,
            sudo_password=password,
        )
        return api.proxy.install_service(options, sctx=sctx)

    try:
        result = _with_sudo(call)
    except api.SparkrunError as exc:
        raise click.ClickException(str(exc)) from exc

    for warning in result.warnings:
        click.echo("Warning: %s" % warning, err=True)

    if result.dry_run:
        click.echo("[dry-run] Unit: %s (%s scope)" % (result.unit_path, result.scope))
        click.echo(result.unit_text)
        if result.sudoers_path:
            click.echo("[dry-run] Sudoers grant: %s" % result.sudoers_path)
            click.echo(result.sudoers_text)
        click.echo(
            "[dry-run] Would %s and enable %s%s."
            % ("update" if result.updated else "install", result.unit, ", then start it" if now else "")
        )
        return

    click.echo("%s and enabled %s (%s)." % ("Updated" if result.updated else "Installed", result.unit, result.unit_path))
    if result.sudoers_path:
        click.echo("Sudoers grant: %s (start/stop/restart of %s only)" % (result.sudoers_path, result.unit))
    if result.persisted:
        click.echo("Saved proxy.yaml: %s" % ", ".join(result.persisted))
    if result.scope == "user":
        click.echo("Lingering: %s" % ("enabled (starts at boot)" if result.linger else "not enabled (starts at login)"))
    if result.stopped_adhoc_pid:
        click.echo("Stopped the proxy that was running outside the unit (PID %d)." % result.stopped_adhoc_pid)
    if result.started:
        if result.listening:
            click.echo("Started %s; the proxy is listening." % result.unit)
        else:
            click.echo(
                render_identity_text(
                    "Started %s, but the proxy port is not answering yet. Check: {app_command} proxy systemd status" % result.unit
                )
            )
    elif result.updated:
        click.echo(render_identity_text("A running unit keeps its old settings until restarted: {app_command} proxy start --restart"))
    else:
        click.echo(render_identity_text("Start it now with: {app_command} proxy start"))


@proxy_systemd.command("status")
@click.option("--system", "system_scope", is_flag=True, help="Only look for a system unit.")
@click.option("--user", "user_scope", is_flag=True, help="Only look for a user unit.")
@click.option("--lines", type=click.IntRange(min=0), default=20, show_default=True, help="Journal lines to show.")
@json_option()
def status_cmd(system_scope, user_scope, lines, output_json):
    """Show the proxy unit's state, sudoers grant, and recent log lines."""
    from sparkrun import api

    sctx = click.get_current_context().ensure_object(dict).get("sparkrun_ctx")
    try:
        result = api.proxy.service_status(_scope_from_flags(system_scope, user_scope), journal_lines=lines, sctx=sctx)
    except api.SparkrunError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        print_json(result.to_dict())
        return
    if result.unavailable_reason:
        click.echo("systemd unavailable: %s" % result.unavailable_reason)
        return
    if not result.installed:
        click.echo(render_identity_text("No proxy unit installed. Install one with: {app_command} proxy systemd install"))
    else:
        click.echo("Unit:      %s (%s scope)" % (result.unit, result.scope))
        click.echo("File:      %s" % result.unit_path)
        click.echo("Enabled:   %s" % result.enabled)
        click.echo("Active:    %s" % result.active)
        if result.scope == "system":
            grant = {True: "ok", False: "missing (start/stop will prompt for sudo)", None: "unknown"}[result.grant_ok]
            click.echo("Sudo grant: %s" % grant)
        else:
            linger = {True: "enabled (starts at boot)", False: "not enabled (starts at login)", None: "unknown"}[result.linger]
            click.echo("Lingering: %s" % linger)
    if result.discovery is not None:
        click.echo("Discovery: %s" % result.discovery.describe())
    if result.journal:
        click.echo("")
        click.echo(result.journal)


@proxy_systemd.command("uninstall")
@click.option("--system", "system_scope", is_flag=True, help="Uninstall the system unit.")
@click.option("--user", "user_scope", is_flag=True, help="Uninstall the user unit.")
@dry_run_option
def uninstall_cmd(system_scope, user_scope, dry_run):
    """Stop, disable and delete the proxy unit (and its sudoers grant).

    Stopping alone is '{app_command} proxy stop'; the unit then stays
    enabled and starts again at the next boot.
    """
    from sparkrun import api

    sctx = click.get_current_context().ensure_object(dict).get("sparkrun_ctx")
    scope = _scope_from_flags(system_scope, user_scope)
    try:
        result = _with_sudo(lambda password: api.proxy.uninstall_service(scope, dry_run=dry_run, sudo_password=password, sctx=sctx))
    except api.SparkrunError as exc:
        raise click.ClickException(str(exc)) from exc
    if result.dry_run:
        click.echo("[dry-run] Would stop, disable and uninstall %s (%s scope)." % (result.unit, result.scope))
    elif result.uninstalled:
        click.echo("Uninstalled %s." % result.unit)
    else:
        click.echo("Nothing uninstalled.", err=True)
        sys.exit(1)
    for note in result.notes:
        click.echo(note)
