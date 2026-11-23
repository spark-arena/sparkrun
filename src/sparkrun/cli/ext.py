"""Plugin-contributed CLI commands.

External plugins add Click commands/groups to the sparkrun CLI by calling
:func:`register_cli_command` at import time (or in their ``register(v)`` hook),
e.g.::

    from sparkrun.cli.ext import register_cli_command
    register_cli_command(my_command)                          # top-level: `sparkrun my-command`
    register_cli_command(import_foo, parent=("cluster", "import"))  # `sparkrun cluster import foo`

The registry itself lives in :mod:`sparkrun.core.cli_registry`, deliberately
free of Click: a plugin that also loads on the console-free ``sparkrun.api``
path should import *that* and register a **loader** rather than a built
command, so registering costs no Click import.  This module re-exports the
registry API and owns the attach half, which needs Click.

**Timing.** The CLI command tree is built at import of :mod:`sparkrun.cli`, but
external plugins load later, through :func:`sparkrun.application.initialize`.
:class:`PluggableGroup` bridges the gap: the top-level ``main`` group is a
``PluggableGroup``, so the first time Click resolves *any* subcommand it runs
:func:`ensure_cli_extensions` — which loads plugins (they call
``register_cli_command``) and attaches every registered command to its target
group (walking into nested groups like ``cluster import``) before Click
dispatches.  A plugin command therefore shows up in ``--help`` and dispatches
like a built-in.

Command selection is deliberately *not* tied to the transport/executor
abstractions — plugins map to commands however makes sense for them.  Per-command
gating (feature flags, ``hidden=``) is the command's own concern.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar

import click

# Re-exported so ``sparkrun.cli.ext`` remains the documented entry point.
from sparkrun.core.cli_registry import (  # noqa: F401 - re-export
    CliCommandSpec,
    register_cli_command,
    registered_cli_commands,
)

logger = logging.getLogger(__name__)


def _resolve_group(root: click.Group, path: tuple[str, ...]) -> "click.Group | None":
    """Walk *path* from *root* using the raw ``.commands`` dicts.

    Uses ``.commands`` rather than ``get_command`` so attaching never re-enters
    :meth:`PluggableGroup.get_command` (which would recurse into the ensure
    hook).  Returns ``None`` if any segment is missing or not a group.
    """
    group: click.Command | None = root
    for segment in path:
        if not isinstance(group, click.Group):
            return None
        group = group.commands.get(segment)
        if group is None:
            return None
    return group if isinstance(group, click.Group) else None


def render_command_identity(root: click.Command) -> None:
    from sparkrun.core.application_profile import render_identity_text

    for attr in ("help", "short_help", "epilog"):
        value = getattr(root, attr, None)
        if value:
            setattr(root, attr, render_identity_text(value))
    for parameter in root.params:
        help_text = getattr(parameter, "help", None)
        if help_text:
            setattr(parameter, "help", render_identity_text(help_text))  # noqa: B010 - optional custom Parameter attribute
    if isinstance(root, click.Group):
        for child in root.commands.values():
            render_command_identity(child)


def attach_cli_extensions(root: click.Group) -> None:
    """Attach every registered command to its target group under *root*.

    A command whose parent group is missing is logged and skipped; a command
    whose name already exists on the target group is left as-is (built-in wins,
    and re-attach passes are no-ops).
    """
    for spec in registered_cli_commands():
        parent = _resolve_group(root, spec.parent)
        if parent is None:
            logger.warning(
                "Cannot attach plugin CLI command %r: parent group %r not found",
                spec.name,
                "/".join(spec.parent) or "<root>",
            )
            continue
        if spec.name in parent.commands:
            continue
        try:
            command = spec.resolve()
        except Exception:  # noqa: BLE001 - one bad loader must not break the CLI
            logger.exception("Could not build plugin CLI command %r", spec.name)
            continue
        parent.add_command(command)


def ensure_cli_extensions(root: click.Group, *, ctx: click.Context | None = None) -> None:
    """Load external plugins, then attach their registered CLI commands.

    Bind the CLI's application/config before importing plugin modules. Their
    command registrations are then attached to *root*.
    """
    from sparkrun.cli._common import _initialize_application

    _initialize_application(ctx)
    attach_cli_extensions(root)
    render_command_identity(root)


_COMPLETING = ContextVar("sparkrun_cli_completing", default=False)
_PLUGIN_LOGGERS = (
    "sparkrun.core.bootstrap",
    "sparkrun.core.external_plugins",
    "sparkrun.core.in_tree_plugins",
    "sparkrun.core.installed_plugins",
    "sparkrun.core.registration",
    __name__,
)


@contextmanager
def _completion_diagnostics():
    """Completion emits candidates only; other invocations keep their logging."""

    def allow_record(_record):
        return not _COMPLETING.get()

    loggers = [logging.getLogger(name) for name in _PLUGIN_LOGGERS]
    token = _COMPLETING.set(True)
    try:
        for target in loggers:
            target.addFilter(allow_record)
        yield
    finally:
        for target in loggers:
            target.removeFilter(allow_record)
        _COMPLETING.reset(token)


class PluggableGroup(click.Group):
    """A Click group that lazily merges plugin-contributed commands.

    On the first command resolution (``get_command`` / ``list_commands``) it
    runs :func:`ensure_cli_extensions` once, so plugin commands are present
    before Click dispatches — including into nested groups reached through this
    one. The load is guarded per-instance. Discovery failures are logged so
    built-in recovery commands remain dispatchable.
    """

    def parse_args(self, ctx, args):
        # This runs before command resolution (and before eager nested help).
        # Root options are flags; do not inspect plugin command arguments as
        # though they were a management route.
        words = list(args)
        while words and words[0].startswith("-") and words[0] != "--":
            words.pop(0)
        if words[:1] == ["--"]:
            words.pop(0)
        ctx.meta["daemon_management"] = words[:1] == ["daemon"]
        ctx.meta["plugin_management"] = bool(
            words[:1] in (["plugins"], ["update"], ["daemon"])
            or (words[:1] == ["setup"] and words[1:2] in (["plugins"], ["update"], ["version"], ["features"]))
        )
        return super().parse_args(ctx, args)

    def invoke(self, ctx):
        from sparkrun.core.coordination import ResourceBusy

        try:
            return super().invoke(ctx)
        except ResourceBusy as exc:
            # Contention is an answer (who holds what, and what to do), not a
            # crash: no command renders it, and a traceback buried the holder.
            raise click.ClickException(str(exc)) from exc
        except RuntimeError as exc:
            # Imported only on failure: every command passes through here.
            from sparkrun.orchestration.primitives import RemoteProbeError

            if isinstance(exc, RemoteProbeError):
                # An unreachable host is an answer about the cluster, not a
                # crash; several commands probe the head before doing anything.
                raise click.ClickException(str(exc)) from exc
            raise

    def _main_shell_completion(self, *args, **kwargs):
        # Click owns the completion flag and enters this hook before building
        # contexts. Avoid loading an application profile merely to detect TAB.
        from sparkrun.core.plugin_defaults import suspend_default_installation

        with _completion_diagnostics(), suspend_default_installation():
            return super()._main_shell_completion(*args, **kwargs)

    def _ensure_cli_extensions_loaded(self, ctx: click.Context) -> None:
        from sparkrun.cli._common import _bind_application_config

        # A command tree can be reused by embedded callers. Its discovery cache
        # must not bypass validation or binding of this invocation's config.
        if not ctx.meta.get("daemon_management"):
            _bind_application_config(ctx)
        if ctx.meta.get("plugin_management"):
            render_command_identity(self)
            return
        if getattr(self, "_cli_ext_loaded", False):
            return
        # Set before running so the attach pass (which touches .commands, not
        # get_command) can't re-enter this hook.
        self._cli_ext_loaded = True
        try:
            ensure_cli_extensions(self, ctx=ctx)
        except Exception:  # noqa: BLE001 - plugin CLI loading must never break the CLI
            logger.exception("Failed to load plugin CLI commands")

    def list_commands(self, ctx):
        self._ensure_cli_extensions_loaded(ctx)
        from sparkrun.core.features import feature_gate_enabled

        if "proxy" in self.commands:
            self.commands["proxy"].hidden = not feature_gate_enabled("gateway")
        return super().list_commands(ctx)

    def get_command(self, ctx, name):
        self._ensure_cli_extensions_loaded(ctx)
        return super().get_command(ctx, name)


class ExtensibleCommand(click.Command):
    """Resolve plugin options before parsing, help rendering, and completion.

    Parameter objects belong to the Click context; the shared command object
    stays unchanged across invocations. Conflicts fail before dispatch instead
    of silently allowing one provider to shadow another's option.
    """

    def __init__(self, *args, extension_target: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.extension_target = extension_target

    def _extension_options(self, ctx):
        key = ("extension_options", id(self))
        if key not in ctx.meta:
            from sparkrun.cli._common import _initialize_application
            from sparkrun.core.cli_registry import registered_cli_options
            from sparkrun.core.application_profile import render_identity_text

            _initialize_application(ctx)
            base_params = super().get_params(ctx)
            names = {p.name for p in base_params}
            flags = {opt for p in base_params for opt in (*p.opts, *p.secondary_opts)}
            resolved = []
            for spec in registered_cli_options(self.extension_target):
                options = spec.loader()
                for option in options:
                    if not isinstance(option, click.Option):
                        raise click.UsageError("Plugin %r contributed a non-option parameter" % spec.owner)
                    declared = set(option.opts + option.secondary_opts)
                    if option.name in names or declared & flags:
                        raise click.UsageError("Plugin %r has a conflicting option: %s" % (spec.owner, option.opts))
                    names.add(option.name)
                    flags.update(declared)
                    if option.help:
                        option.help = render_identity_text(option.help)
                resolved.append((spec, options))
            ctx.meta[key] = resolved
        return ctx.meta[key]

    def get_params(self, ctx):
        return super().get_params(ctx) + [p for _, options in self._extension_options(ctx) for p in options]

    def pop_extension_values(self, ctx, values):
        selections = {}
        for spec, options in self._extension_options(ctx):
            raw = {p.name: values.pop(p.name) for p in options if p.name in values}
            try:
                settings = spec.decode(raw)
            except ValueError as exc:
                raise click.UsageError(str(exc), ctx=ctx) from exc
            if settings is not None:
                selections[spec.owner] = settings
        return selections
