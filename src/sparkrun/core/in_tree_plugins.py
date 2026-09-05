"""Load first-party plugins that ship inside the sparkrun distribution.

The in-tree mate of :mod:`sparkrun.core.external_plugins`. Both walk a set of
top-level modules, register any SAF plugin subclasses they define, and run an
optional ``register(v)`` hook — the shared half is
:func:`~sparkrun.core.external_plugins.load_plugin_module`. Only *where* the
modules come from differs: a package inside the wheel here, user-configured
directories there.

``sparkrun.plugins`` is for **cross-cutting integrations** — things that span
several extension points at once and are meaningful as a single removable unit:
an integration contributing a backend implementation, a hidden CLI command and
a wire protocol together, none of which is "a runtime" or "an executor". The
existing first-party packages that *do* map cleanly onto one extension point
(``runtimes``, ``transports``, ``executors``, ``schedulers``, ``builders``,
``benchmarking``) stay where they are and keep their own
``find_types_in_modules`` scan in :mod:`sparkrun.core.bootstrap`.

**Every in-tree plugin is gated by a feature flag**, for the same reason
``executor.docker`` and ``gateway.litellm`` carry one despite shipping enabled:
every plugin surface should be controllable the same way. The flag is checked
*before* the import, so turning a plugin off costs nothing at all — no import,
no commands, no registrations.

It is the plugin's *own* feature flag rather than a separate presence flag:
there is no point loading a plugin whose capability will not be used, and no
point enabling that capability without the plugin. So an integration is gated
end-to-end by the one flag that governs the capability it contributes.

The binding lives in :data:`IN_TREE_PLUGIN_FEATURES` rather than on the plugin
because the flag has to be known *without importing* the module it gates —
which is the whole point of checking before the import.

Two deliberate differences from the external loader:

* **Per-plugin flags, not one flag for the mechanism.** Out-of-tree loading is
  all-or-nothing behind ``core.external_plugins`` because the set of plugins is
  unknown until the config is read; here the set is fixed at build time, so
  each gets its own switch.
* **A failure here is our bug, not a user's.** It is still non-fatal — a broken
  integration must not stop ``sparkrun run`` from working — but it logs at
  exception level rather than being quietly skipped.

(There is also no ``sys.path`` manipulation: these are already importable.)
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

from sparkrun.core.external_plugins import load_plugin_module
from sparkrun.core.features import feature_gate_enabled, get_feature
from sparkrun.core.registry_defaults import DeclarationTier

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

logger = logging.getLogger(__name__)

#: Package scanned for first-party plugin subpackages.
IN_TREE_PLUGIN_PACKAGE = "sparkrun.plugins"

#: In-tree plugin module name -> the feature flag gating it.
#:
#: Adding a plugin means adding an entry here. Kept as data rather than read
#: off the plugin because the flag must resolve *before* the import — a plugin
#: that declared its own gate could only be consulted by importing it, which is
#: exactly what the gate is meant to avoid.
IN_TREE_PLUGIN_FEATURES: dict[str, str] = {}


def plugin_feature_flag(name: str) -> str | None:
    """Return the feature flag gating in-tree plugin *name*, or ``None``."""
    return IN_TREE_PLUGIN_FEATURES.get(name)


def load_in_tree_plugins(v: "Variables", package: str = IN_TREE_PLUGIN_PACKAGE) -> list[str]:
    """Import and register every plugin under *package*.

    Args:
        v: The initialized SAF :class:`~scitrera_app_framework.Variables`.
        package: Dotted package to scan.  Overridable for tests.

    Returns:
        The plugin module names that loaded, in discovery order.
    """
    try:
        root = importlib.import_module(package)
    except Exception:
        logger.exception("Could not import the in-tree plugin package %r", package)
        return []

    loaded: list[str] = []
    for mod_info in pkgutil.iter_modules(getattr(root, "__path__", [])):
        name = mod_info.name
        if name.startswith("_"):
            continue
        flag = plugin_feature_flag(name)
        if flag is None or get_feature(flag) is None:
            # Skipping either way (an unregistered flag resolves off), but
            # silently — and the reason would be far from the cause. Shipping
            # an in-tree plugin means an IN_TREE_PLUGIN_FEATURES entry and a
            # flag registered in core.features.
            logger.error(
                "In-tree plugin %r is not bound to a registered feature flag (%s); "
                "add an IN_TREE_PLUGIN_FEATURES entry and register the flag in sparkrun.core.features",
                name,
                flag or "no entry",
            )
            continue
        if not feature_gate_enabled(flag, v):
            logger.debug("Skipping in-tree plugin %r (feature %r off)", name, flag)
            continue

        dotted = "%s.%s" % (package, name)
        try:
            module = importlib.import_module(dotted)
        except Exception:
            # Shipping a broken integration should not take the CLI down with
            # it, but it is a defect rather than a user misconfiguration, so
            # say so loudly.
            logger.exception("Failed to import in-tree plugin %r", dotted)
            continue
        # IN_TREE is what lets this plugin's registry declarations ship trusted
        # (sparkrun.core.registry_defaults). Asserted by the loader, never by
        # the plugin — including for a vendored plugin, where the claim rests on
        # the pinned commit in vendor/*.lock and the review of the import PR.
        load_plugin_module(module, v, tier=DeclarationTier.IN_TREE)
        loaded.append(name)

    if loaded:
        logger.debug("Loaded in-tree plugins: %s", loaded)
    return loaded


__all__ = ["IN_TREE_PLUGIN_FEATURES", "IN_TREE_PLUGIN_PACKAGE", "load_in_tree_plugins", "plugin_feature_flag"]
