"""Load first-party plugins that ship inside the sparkrun distribution.

The in-tree mate of :mod:`sparkrun.core.external_plugins`. Both walk a set of
top-level modules, register any SAF plugin subclasses they define, and run an
optional ``register(v)`` hook — the shared half is
:func:`~sparkrun.core.registration.load_and_register_plugin`. Only *where* the
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

The loading gate controls the whole integration. A plugin may additionally
expose ``FEATURE_DEFINITIONS`` for individual capabilities; those definitions
register only after the loading gate is enabled, before its implementations
are scanned. Child overrides cannot load a disabled parent.

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
from functools import partial
import logging
import pkgutil
from typing import TYPE_CHECKING

from sparkrun.core.registration import load_and_register_plugin
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
IN_TREE_PLUGIN_FEATURES: dict[str, str] = {
    "sparkroute": "gateway.sparkroute",
    "coldsnap": "plugins.coldsnap",
    "sparkarena": "integration.arena",
    "k8s": "integration.k8s",
    "lil": "registry.lil",
}


def plugin_application_profile_api(name: str) -> int | None:
    """Read the verified bundled declaration without importing the integration."""
    from pathlib import Path
    import tomllib

    path = Path(__file__).resolve().parents[1] / "plugins" / name / "VENDORED.toml"
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8")).get("application_profile_api")
    except (OSError, ValueError):
        return None
    return value if type(value) is int else None


def plugin_application_profile_failure(name: str) -> str | None:
    from sparkrun.core.application_profile import APPLICATION_PROFILE_API_VERSION, get_application_profile

    if (
        name == "sparkroute"
        and get_application_profile().id != "sparkrun"
        and plugin_application_profile_api(name) != APPLICATION_PROFILE_API_VERSION
    ):
        return (
            "The pinned SparkRoute plugin does not declare support for this host's application profile API; "
            "import a compatible plugin snapshot before using an alternate application profile"
        )
    return None


def plugin_feature_flag(name: str) -> str | None:
    """Return the feature flag gating in-tree plugin *name*, or ``None``."""
    return IN_TREE_PLUGIN_FEATURES.get(name)


def iter_in_tree_plugin_names(package: str | None = None) -> list[str]:
    """Return the plugin subpackage names shipped under *package*.

    Discovery only — nothing is imported, so this answers "what ships here?"
    even for plugins whose feature flag is off. Shared with
    :func:`load_in_tree_plugins` so a listing can never name a plugin the
    loader would skip.

    *package* defaults to :data:`IN_TREE_PLUGIN_PACKAGE` resolved at call time,
    not at definition time, so a test that redirects the package redirects both
    this and the loader together.
    """
    package = package or IN_TREE_PLUGIN_PACKAGE
    try:
        root = importlib.import_module(package)
    except Exception:
        logger.exception("Could not import the in-tree plugin package %r", package)
        return []
    return [info.name for info in pkgutil.iter_modules(getattr(root, "__path__", [])) if not info.name.startswith("_")]


def load_in_tree_plugins(v: "Variables", package: str | None = None) -> list[str]:
    """Import and register every plugin under *package*.

    Args:
        v: The initialized SAF :class:`~scitrera_app_framework.Variables`.
        package: Dotted package to scan.  Overridable for tests; ``None``
            resolves :data:`IN_TREE_PLUGIN_PACKAGE` at call time.

    Returns:
        The plugin module names that loaded, in discovery order.
    """
    package = package or IN_TREE_PLUGIN_PACKAGE
    loaded: list[str] = []
    for name in iter_in_tree_plugin_names(package):
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

        failure = plugin_application_profile_failure(name)
        if failure:
            logger.warning("Cannot load in-tree integration %s: %s", name, failure)
            continue

        dotted = "%s.%s" % (package, name)
        try:
            load_and_register_plugin(
                partial(importlib.import_module, dotted), v, tier=DeclarationTier.IN_TREE, source=("in-tree", dotted, None)
            )
        except Exception:
            logger.exception("Failed to load in-tree plugin %r", dotted)
            continue
        loaded.append(name)

    if loaded:
        logger.debug("Loaded in-tree plugins: %s", loaded)
    return loaded


__all__ = [
    "IN_TREE_PLUGIN_FEATURES",
    "IN_TREE_PLUGIN_PACKAGE",
    "iter_in_tree_plugin_names",
    "load_in_tree_plugins",
    "plugin_feature_flag",
]
