"""``sparkrun setup plugins`` — inspect installed plugin modules.

A thin renderer over :func:`sparkrun.core.plugin_inventory.list_plugins`; the
shape mirrors ``setup features`` (its sibling: that group says which
*capabilities* are gated on, this one says which *plugins* provide them).

Hidden from ``--help`` unless ``SPARKRUN_ADVANCED`` is set, the same
visibility-only gate ``setup throttle-gpu-clock`` and ``setup uninstall`` use —
the group stays functional either way, since a user debugging a plugin that
will not load needs it whether or not they run with advanced options on.
"""

from __future__ import annotations

import click

from .._common import HIDE_ADVANCED_OPTIONS, _get_context, json_option, print_json
from . import setup


@setup.group("plugins", invoke_without_command=True, hidden=HIDE_ADVANCED_OPTIONS)
@click.pass_context
def setup_plugins(ctx):
    """Inspect the plugin modules sparkrun knows about.

    Covers first-party in-tree integrations (sparkrun.plugins) and out-of-tree
    ones loaded from 'plugins.paths' in config.yaml. Runtimes, executors and
    transports shipped in core are not plugin modules in this sense — see
    'sparkrun list-runtimes' / 'sparkrun list-executors'.

    Defaults to 'list' when no subcommand is given.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(setup_plugins_list)


@setup_plugins.command("list")
@json_option()
@click.pass_context
def setup_plugins_list(ctx, output_json):
    """List every known plugin, with its version and gate state."""
    from sparkrun.core.plugin_inventory import SOURCE_EXTERNAL, list_plugins

    sctx = _get_context(ctx)
    plugins = list_plugins(config=sctx.config, v=sctx.variables)

    if output_json:
        # `version: null` is the machine spelling of unknown; the "unknown"
        # string belongs to the human rendering below.
        print_json([p.to_dict() for p in plugins])
        return

    if not plugins:
        click.echo("No plugins found.")
        click.echo("")
        click.echo("Out-of-tree plugins load from directories listed under 'plugins.paths' in config.yaml.")
        return

    headers = ("NAME", "VERSION", "SOURCE", "STATE", "FLAG")
    rows = [
        (
            p.name,
            p.version_display,
            p.source,
            _state(p),
            p.feature_flag or "(unbound)",
        )
        for p in plugins
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows, strict=True)]

    def _line(cells):
        return "  ".join("%-*s" % (w, c) for w, c in zip(widths, cells, strict=True)).rstrip()

    click.echo(_line(headers))
    for row in rows:
        click.echo(_line(row))

    paths = []
    for plugin in plugins:
        if plugin.source == SOURCE_EXTERNAL and plugin.path is not None and plugin.path not in paths:
            paths.append(plugin.path)
    if paths:
        click.echo("")
        click.echo("Plugin paths: %s" % ", ".join(str(p) for p in paths))

    if any(p.version is None for p in plugins):
        click.echo("")
        click.echo("'unknown' means the plugin declares no __version__; it is not an error.")


def _state(plugin) -> str:
    """Render the gate/load state.

    ``enabled`` and ``loaded`` are separate facts and the difference is the
    whole diagnostic value: a plugin whose gate is on but which never loaded
    failed to import, and the exception is in the logs (``-v``). Collapsing
    them to on/off would hide exactly the case someone runs this to find.
    """
    if not plugin.enabled:
        return "off"
    return "on" if plugin.loaded else "on (load failed)"
