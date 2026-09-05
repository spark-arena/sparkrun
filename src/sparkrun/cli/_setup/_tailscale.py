"""``sparkrun setup tailscale`` — publish an inference endpoint on a tailnet.

Thin CLI layer over :mod:`sparkrun.api.tailscale`:

- ``setup tailscale join``   — install Tailscale + join hosts to the tailnet.
- ``setup tailscale status`` — show each host's tailnet state / IP.
- ``setup tailscale expose`` — print the tailnet URL for the proxy or a head node.
- ``setup tailscale down``   — log hosts out (optionally remove their devices).

Experimental — gated behind the ``cli.setup.tailscale`` feature flag. The api
layer holds all logic; these commands parse flags, resolve hosts, prompt for
sudo, call the api, and render results.
"""

from __future__ import annotations

import click

from .._common import _get_context, _resolve_setup_context, host_options
from . import setup
from ._sudo import ensure_sudo_password

SETUP_TAILSCALE_FEATURE = "cli.setup.tailscale"


def _setup_tailscale_enabled_at_import() -> bool:
    """Best-effort flag resolution for help-visibility (hidden when off/unknown)."""
    try:
        from sparkrun.core.config import SparkrunConfig

        return SparkrunConfig().is_feature_enabled(SETUP_TAILSCALE_FEATURE)
    except Exception:  # noqa: BLE001 — never let a config read break CLI import
        return False


@setup.group("tailscale", hidden=not _setup_tailscale_enabled_at_import())
@click.pass_context
def setup_tailscale(ctx):
    """Tailscale: join nodes to a tailnet and publish inference endpoints.

    sparkrun mints short-lived, tagged auth keys from a Tailscale OAuth client
    and joins hosts non-interactively, then surfaces a run's serve port (or the
    proxy) to the rest of your tailnet.

    Experimental — gated behind the ``cli.setup.tailscale`` feature flag.
    """
    if not _get_context(ctx).config.is_feature_enabled(SETUP_TAILSCALE_FEATURE):
        raise click.ClickException(
            "The 'setup tailscale' commands are experimental and disabled. "
            "Enable them with: sparkrun setup features enable cli.setup.tailscale"
        )


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


@setup_tailscale.command("join")
@host_options
@click.option("--user", "-u", default=None, help="SSH user (defaults to cluster / config / $USER).")
@click.option("--tag", default=None, help="ACL tag to advertise (default tailscale.tag or tag:dgx-spark).")
@click.option("--ephemeral", is_flag=True, help="Mint an ephemeral key (device auto-drops when offline).")
@click.option("--ssh", "enable_ssh", is_flag=True, help="Also enable Tailscale SSH on the joined hosts.")
@click.option(
    "--hostname",
    default=None,
    help="Tailnet device name (defaults to the cluster name for a single-host cluster; else the host's own name).",
)
@click.option("--dry-run", is_flag=True, help="Show what would happen without minting a key or touching hosts.")
@click.pass_context
def setup_tailscale_join(ctx, hosts, hosts_file, cluster_name, user, tag, ephemeral, enable_ssh, hostname, dry_run):
    """Install Tailscale and join the target hosts to your tailnet."""
    from sparkrun import api

    sctx = _get_context(ctx)
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, sctx.config, user)

    # Default the tailnet device name to the sparkrun cluster name for a
    # single-host cluster (e.g. a Thunder instance) so it isn't the opaque OS
    # hostname. Multi-host clusters keep per-node OS names unless --hostname is
    # given (a shared name would collide and get -1/-2 suffixes).
    if hostname is None and cluster_name and len(host_list) == 1:
        hostname = cluster_name
    if hostname and len(host_list) > 1:
        click.secho(
            "Note: --hostname %s applies to all %d hosts; Tailscale will append -1, -2, … to de-duplicate." % (hostname, len(host_list)),
            fg="yellow",
        )

    sudo_password = None
    if not dry_run:
        sudo_password, _ = ensure_sudo_password(host_list, user, ssh_kwargs, dry_run=dry_run)

    try:
        result = api.tailscale.join(
            sctx,
            host_list,
            ssh_kwargs,
            tag=tag,
            ephemeral=ephemeral or None,
            enable_ssh=enable_ssh,
            hostname=hostname,
            sudo_password=sudo_password,
            dry_run=dry_run,
        )
    except (api.tailscale.TailscaleNotConfigured, api.tailscale.TailscaleAuthFailed, api.tailscale.TailscaleSetupError) as exc:
        raise click.ClickException(str(exc)) from exc

    if result.dry_run:
        click.secho(
            "[dry-run] Would join %d host(s) with tag %s%s%s:"
            % (len(result.hosts), result.tag, " (ephemeral)" if result.ephemeral else "", " as '%s'" % hostname if hostname else ""),
            fg="yellow",
        )
        for h in result.hosts:
            click.echo("  %s" % h.host)
        return

    click.secho(
        "Tailscale join: %d/%d host(s) OK (tag %s%s)."
        % (result.ok_count, len(result.hosts), result.tag, ", ephemeral" if result.ephemeral else ""),
        fg="green" if result.ok_count == len(result.hosts) else "yellow",
    )
    for h in result.hosts:
        if h.ok:
            click.echo("  %-24s %s  [%s]" % (h.host, h.ip or "(no ip)", h.install or "?"))
        else:
            click.secho("  %-24s FAILED: %s" % (h.host, h.message or "unknown"), fg="red")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@setup_tailscale.command("status")
@host_options
@click.option("--user", "-u", default=None, help="SSH user (defaults to cluster / config / $USER).")
@click.pass_context
def setup_tailscale_status(ctx, hosts, hosts_file, cluster_name, user):
    """Show each host's Tailscale state and tailnet IP."""
    from sparkrun import api

    sctx = _get_context(ctx)
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, sctx.config, user)

    result = api.tailscale.status(sctx, host_list, ssh_kwargs)
    for h in result.hosts:
        color = "green" if h.joined else ("red" if h.state in ("unreachable", "not_installed") else "yellow")
        click.secho("  %-24s %-14s %s" % (h.host, h.state, h.ip or ""), fg=color)


# ---------------------------------------------------------------------------
# expose
# ---------------------------------------------------------------------------


@setup_tailscale.command("expose")
@click.option("--proxy", is_flag=True, help="Expose the local sparkrun proxy on the tailnet.")
@click.option("--head", "head_host", default=None, help="Expose a specific head host's serve port.")
@click.option("--cluster", "cluster", default=None, help="Expose the first host of a named cluster.")
@click.option("--port", type=int, default=None, help="Endpoint port (proxy: proxy.port; head: 8000).")
@click.option("--set-proxy-host", is_flag=True, help="Persist proxy.host to the local tailnet IP (proxy mode).")
@click.option("--user", "-u", default=None, help="SSH user for head/cluster resolution.")
@click.pass_context
def setup_tailscale_expose(ctx, proxy, head_host, cluster, port, set_proxy_host, user):
    """Publish the inference endpoint on the tailnet.

    ``--proxy`` reports the local sparkrun proxy's tailnet URL. ``--head`` /
    ``--cluster`` configure a ``tailscale serve --tcp`` forward on the head host
    (the inbound path that works in userspace-networking mode) and print the
    ``http://<tailnet-ip>:<port>/v1`` endpoint.
    """
    from sparkrun import api

    sctx = _get_context(ctx)
    targets = [t for t in (proxy, head_host, cluster) if t]
    if len(targets) != 1:
        raise click.ClickException("Specify exactly one of --proxy, --head <host>, or --cluster <name>.")

    ssh_kwargs = None
    sudo_password = None
    if not proxy:
        if cluster and not head_host:
            host_list, user, ssh_kwargs = _resolve_setup_context(None, None, cluster, sctx.config, user)
            if not host_list:
                raise click.ClickException("Cluster %r resolved no hosts." % cluster)
            head_host = host_list[0]
        else:
            _, user, ssh_kwargs = _resolve_setup_context(head_host, None, None, sctx.config, user)
        # Configuring `tailscale serve` on the host needs root.
        sudo_password, _ = ensure_sudo_password([head_host], user, ssh_kwargs)

    try:
        result = api.tailscale.expose(
            sctx,
            proxy=proxy,
            head_host=None if proxy else head_host,
            ssh_kwargs=ssh_kwargs,
            port=port,
            set_proxy_host=set_proxy_host,
            sudo_password=sudo_password,
        )
    except api.tailscale.TailscaleExposeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.secho("Endpoint: %s" % result.url, fg="green", bold=True)
    click.echo("  target:   %s" % result.target)
    click.echo("  address:  %s:%d" % (result.endpoint, result.port))
    if result.proxy_host_updated:
        click.echo("  proxy.host updated to %s (restart the proxy to apply)." % result.endpoint)
    for w in result.warnings:
        click.secho("  ! %s" % w, fg="yellow")


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------


@setup_tailscale.command("down")
@host_options
@click.option("--user", "-u", default=None, help="SSH user (defaults to cluster / config / $USER).")
@click.option("--remove", is_flag=True, help="Also delete the devices from the tailnet (needs OAuth creds).")
@click.option("--dry-run", is_flag=True, help="Show what would happen without touching hosts.")
@click.pass_context
def setup_tailscale_down(ctx, hosts, hosts_file, cluster_name, user, remove, dry_run):
    """Log hosts out of the tailnet (optionally remove their devices)."""
    from sparkrun import api

    sctx = _get_context(ctx)
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, sctx.config, user)

    sudo_password = None
    if not dry_run:
        sudo_password, _ = ensure_sudo_password(host_list, user, ssh_kwargs, dry_run=dry_run)

    try:
        result = api.tailscale.down(
            sctx,
            host_list,
            ssh_kwargs,
            remove=remove,
            sudo_password=sudo_password,
            dry_run=dry_run,
        )
    except (api.tailscale.TailscaleNotConfigured, api.tailscale.TailscaleAuthFailed, api.tailscale.TailscaleSetupError) as exc:
        raise click.ClickException(str(exc)) from exc

    if result.dry_run:
        click.secho(
            "[dry-run] Would log out %d host(s)%s." % (len(result.hosts), " and remove their devices" if remove else ""), fg="yellow"
        )
        return

    for h in result.hosts:
        suffix = " (device removed)" if h.removed else ""
        color = "green" if h.state == "logged_out" else "yellow"
        click.secho("  %-24s %s%s" % (h.host, h.state, suffix), fg=color)
    if remove:
        click.echo("Removed %d device(s) from the tailnet." % len(result.removed_devices))
