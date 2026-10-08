# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""First-party ColdSnap recipe and CLI integration."""

from __future__ import annotations

__version__ = "0.1.13"

import sparkrun.plugins as _plugin_api
from sparkrun.plugins import register_cli_command, register_recipe_item
from sparkrun.plugins.coldsnap.builder import ColdSnapBuilder
from sparkrun.plugins.coldsnap.compatibility import ColdSnapCompatibilityError
from sparkrun.plugins.coldsnap.config import ColdSnapRecipeHandler
from sparkrun.plugins.coldsnap.service import ColdSnapExecutionStrategy


SPARKRUN_PLUGIN_API_VERSION = 1

_HANDLER = ColdSnapRecipeHandler()
_EXECUTION_STRATEGY = ColdSnapExecutionStrategy()


def _register_recipe_registries() -> None:
    """Contribute the qualified recipes when the host supports overlays."""
    entry_type = getattr(_plugin_api, "RegistryEntry", None)
    register_registry = getattr(_plugin_api, "register_default_registry", None)
    if entry_type is None or register_registry is None:
        return

    register_registry(
        entry_type(
            name="coldsnap",
            url="https://github.com/sparksq/sparkrun-recipes.git",
            subpath="coldsnap-recipes",
            description="Qualified ColdSnap recipes for fast, recovery-aware inference startup",
            visible=False,
        ),
        owner="coldsnap",
    )
    register_registry(
        entry_type(
            name="coldsnap-vanilla",
            url="https://github.com/sparksq/sparkrun-recipes.git",
            subpath="vanilla-recipes",
            description="Vanilla controls matched to the qualified ColdSnap recipes",
            enabled=False,
            visible=False,
        ),
        owner="coldsnap",
    )


def register(v) -> None:
    _register_recipe_registries()
    register_recipe_item(
        "coldsnap",
        _HANDLER,
        owner="sparkrun.plugins.coldsnap",
        execution_strategy=_EXECUTION_STRATEGY,
    )
    register_cli_command(
        lambda: __import__("sparkrun.plugins.coldsnap.cli", fromlist=["build_command"]).build_command(),
        name="coldsnap",
    )


__all__ = ["ColdSnapBuilder", "ColdSnapCompatibilityError", "SPARKRUN_PLUGIN_API_VERSION", "__version__", "register"]
