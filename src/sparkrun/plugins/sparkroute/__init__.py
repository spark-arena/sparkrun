# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""LLM Gateway integration — sparkrun-supervised Go gateway, both directions.

An in-tree plugin (see :mod:`sparkrun.plugins`) because it is only coherent as
one unit while fitting no single extension point. It contributes:

* **Outbound** — :class:`~.engine.SparkrouteEngine`, registered as the
  SparkRoute gateway. Sparkrun acquires the pinned release binary
  (:mod:`.release`), manages private live admin/caller token files
  (:mod:`.credentials`), starts the gateway in SQLite configuration mode, and
  replaces its managed set through the admin API (:mod:`.admin`) from a
  projection of the binding catalog (:mod:`.projection`).
* **Inbound** — the hidden ``sparkrun gateway-bridge`` command
  (:mod:`.cli`), a one-shot JSON stdio protocol (:mod:`.protocol`,
  :mod:`.bridge`, :mod:`.operations`) the gateway drives as a child process to
  discover, activate and stop workloads.

Both halves are gated by one flag, ``gateway.sparkroute``. The host distribution
owns its channel defaults, and the flag gates loading: with it off this module is
never imported, so there is no gateway registration and no bridge command. One
flag rather than a separate presence flag because the two are not
independently useful: no point loading the plugin if the gateway will not be
used, and none in enabling the gateway without the plugin.

If disabled while a gateway is running, sparkrun's generic supervisor retains
process-level status and stop. Configuration and bridge operations require the
integration to remain enabled.
"""

from __future__ import annotations

__version__ = "0.1.1"

import logging

logger = logging.getLogger(__name__)

#: Selector for this gateway (``proxy.gateway`` in ``proxy.yaml``).
GATEWAY_NAME = "sparkroute"

#: Flag gating both halves of the integration.
FEATURE_FLAG = "gateway.sparkroute"

#: Name of the hidden bridge command this plugin contributes.
BRIDGE_COMMAND = "gateway-bridge"


def _load_engine() -> type:
    """Deferred import of the engine class for the gateway registry."""
    from sparkrun.plugins.sparkroute.engine import SparkrouteEngine

    return SparkrouteEngine


def _load_bridge_command():
    """Build the Click command, only when the CLI actually attaches it.

    Registered as a loader rather than a built command because this plugin also
    loads on the console-free ``sparkrun.api`` path, where importing Click
    would be a layering violation. See :mod:`.cli` — the import lives inside
    the builder, since plugin scanning imports every submodule regardless.
    """
    from sparkrun.plugins.sparkroute.cli import build_gateway_bridge_command

    return build_gateway_bridge_command()


def register(v) -> None:
    """Plugin entry point, invoked by the in-tree plugin loader.

    Registers the recipe item, gateway implementation and hidden bridge command.
    These registrations are cheap and import nothing heavy — the engine arrives
    through a deferred loader, and the Click command imports its runner only
    when invoked — so a stock install that never touches this gateway pays
    almost nothing for it being here.
    """
    from sparkrun.core.cli_registry import register_cli_command
    from sparkrun.plugins import register_recipe_item
    from sparkrun.proxy.gateway import register_gateway
    from .recipe_config import RECIPE_HANDLER

    register_recipe_item("sparkroute", RECIPE_HANDLER, owner=__name__, affects_fingerprint=False)
    register_gateway(GATEWAY_NAME, feature_flag=FEATURE_FLAG, loader=_load_engine)
    register_cli_command(_load_bridge_command, name=BRIDGE_COMMAND)
    logger.debug("Registered the %s gateway and its bridge command", GATEWAY_NAME)


__all__ = ["BRIDGE_COMMAND", "FEATURE_FLAG", "GATEWAY_NAME", "register"]
