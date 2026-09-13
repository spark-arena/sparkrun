# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""The hidden ``sparkrun gateway-bridge`` command.

Attached to the top-level CLI by this plugin's ``register(v)`` hook via
:func:`sparkrun.core.cli_registry.register_cli_command`, so the integration owns
its own command rather than the core CLI carrying a branch for it.

**Click is imported inside the builder, not at module scope.** Plugin
registration scans a plugin package for SAF plugin types, and that scan imports
every submodule — including this one — on the console-free ``sparkrun.api``
path. A module-level ``import click`` here would therefore drag Click into
every ``api`` caller, desktop sidecar included. Registering a lazy loader is
not enough on its own; the import has to be inside the function.

Hidden from ``--help`` on purpose: the compatibility boundary is the versioned
JSON protocol in :mod:`.protocol`, not Click's command presentation. See
``docs/SPARKROUTE_BRIDGE.md``.
"""

from __future__ import annotations


def build_gateway_bridge_command():
    """Construct the ``gateway-bridge`` Click command."""
    import click

    @click.command("gateway-bridge", hidden=True)
    @click.pass_context
    def gateway_bridge(ctx: "click.Context") -> None:
        """Run one versioned LLM Gateway bridge request over standard I/O."""
        from sparkrun.plugins.sparkroute.bridge import run_stdio

        exit_code = run_stdio()
        if exit_code:
            ctx.exit(exit_code)

    return gateway_bridge


__all__ = ["build_gateway_bridge_command"]
