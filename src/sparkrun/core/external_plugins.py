"""Load out-of-tree plugins from user-configured directories.

sparkrun's built-in plugins are discovered by scanning packages inside the
``sparkrun`` distribution (see :func:`sparkrun.core.bootstrap.init_sparkrun`).
This module extends that discovery to directories the user lists under
``plugins.paths`` in ``config.yaml`` — a home for executors, transports,
runtimes, schedulers, builders, and benchmarking frameworks that live outside
the open-source tree.

For each configured directory:

1. The directory is prepended to ``sys.path``.
2. Every importable top-level module/package in it is imported.
3. The imported module is scanned for concrete subclasses of the SAF plugin
   base types and each is registered via ``register_plugin`` — exactly what
   ``init_sparkrun`` does for the built-in packages.
4. If the module defines a top-level ``register(v)`` callable, it is invoked.
   This is the escape hatch for the still-in-process registries
   (:mod:`sparkrun.platforms`, :mod:`sparkrun.orchestration.collectives`) whose
   plugins register by calling ``register_platform`` / ``get_backend`` wiring
   rather than by SAF subclass discovery.

Gated off by default on every channel behind the ``core.external_plugins``
feature flag: when it resolves off, the config-driven auto-load path returns
immediately without reading ``plugins.paths`` (let alone importing anything), so
a stock install pays zero cost and exposes zero extra surface.  Enable it with
``sparkrun setup features enable core.external_plugins`` (or
``features.core.external_plugins: true`` in ``config.yaml``).  Because the config
file and the plugin directories are user-owned, loading them is trusted by
definition — the same trust model as a pip-installed package or a pytest
plugin.  A single broken plugin logs and is skipped; it never breaks startup.

Separately, the ``SPARKRUN_NO_EXTERNAL_PLUGINS`` env var is a hard kill-switch
for the config-driven path — needed for test isolation, since pytest reads the
developer's real config (see ``_DISABLE_ENV`` below).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from scitrera_app_framework import ext_parse_bool, register_plugin
from scitrera_app_framework.util import find_types_in_modules

from sparkrun.core.features import FEATURE_CORE_EXTERNAL_PLUGINS, feature_gate_enabled

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.core.registry_defaults import DeclarationTier

logger = logging.getLogger(__name__)

# Hard test/CI kill-switch for the config-driven auto-load path — set truthy to
# make it inert regardless of the ``core.external_plugins`` flag. Needed because
# tests read the developer's REAL ~/.config/sparkrun (the SAF stateful root
# isn't "ready" under pytest), so a developer who enabled the flag would
# otherwise load their real plugins mid-suite. Set by ``conftest.isolate_stateful``.
# Explicitly-passed ``paths`` (a plugin's own tests) bypass this guard.
_DISABLE_ENV = "SPARKRUN_NO_EXTERNAL_PLUGINS"


def _external_plugins_disabled() -> bool:
    return bool(ext_parse_bool(os.environ.get(_DISABLE_ENV, "")))


def _plugin_base_types() -> list[type]:
    """Return the SAF plugin base classes an external module may subclass.

    Imported lazily to avoid import cycles (this module is reached from
    ``core.bootstrap``, which the plugin packages import back).
    """
    from sparkrun.runtimes.base import RuntimePlugin
    from sparkrun.benchmarking.base import BenchmarkingPlugin
    from sparkrun.builders.base import BuilderPlugin
    from sparkrun.orchestration.executors._base import Executor
    from sparkrun.core.scheduler import Scheduler
    from sparkrun.transports.base import Transport

    return [RuntimePlugin, BenchmarkingPlugin, BuilderPlugin, Executor, Scheduler, Transport]


# Plugin classes that select via a ``*_name`` attribute must set it non-blank to
# be registerable; a blank value marks a shared/abstract base (mirrors the
# scheduler/transport guards in ``core.bootstrap``).
_SELECTOR_ATTRS = ("scheduler_name", "transport_name")


def _is_registerable(cls: type) -> bool:
    for attr in _SELECTOR_ATTRS:
        if hasattr(cls, attr) and not getattr(cls, attr, ""):
            return False
    return True


def _scan_module_for_plugins(module, base: type) -> list[type]:
    """Return concrete *base* subclasses reachable from *module*.

    Scans the imported module object directly (covers single-file ``plugin.py``
    drop-ins) and, for packages, recurses into submodules via
    ``find_types_in_modules`` — which raises on a non-package, so it is only
    used when ``module`` has ``__path__``.
    """
    found: dict[str, type] = {}

    def _consider(attr) -> None:
        if inspect.isclass(attr) and issubclass(attr, base) and attr is not base and not inspect.isabstract(attr):
            found["%s.%s" % (attr.__module__, attr.__qualname__)] = attr

    for attr_name in dir(module):
        _consider(getattr(module, attr_name, None))

    if hasattr(module, "__path__"):
        for cls in find_types_in_modules(module.__name__, base):
            _consider(cls)

    return list(found.values())


def load_plugin_module(module, v: "Variables", *, tier: "DeclarationTier | None" = None) -> None:
    """Register everything *module* contributes: SAF subclasses, then ``register(v)``.

    Shared with :mod:`sparkrun.core.in_tree_plugins` — in-tree and out-of-tree
    plugins differ only in where their modules come from, so they must not
    differ in what counts as a registration.

    Args:
        module: The imported plugin module or package.
        v: The initialized SAF :class:`~scitrera_app_framework.Variables`.
        tier: Provenance to attribute this module's registry declarations to
            (see :mod:`sparkrun.core.registry_defaults`).  Supplied by the
            *loader* because a plugin must not be able to claim its own tier —
            an in-tree tier is what lets a declaration ship trusted.  Ambient
            for the duration of the load rather than an argument, since the
            plugin makes the registration call, not us.  Defaults to
            ``OUT_OF_TREE``: least privilege.
    """
    from sparkrun.core.registry_defaults import DeclarationTier, declaring_tier

    with declaring_tier(tier or DeclarationTier.OUT_OF_TREE):
        # 1) Register SAF-scanned plugin subclasses (runtimes/executors/transports/…).
        for base in _plugin_base_types():
            for cls in _scan_module_for_plugins(module, base):
                if not _is_registerable(cls):
                    continue
                try:
                    register_plugin(cls, v=v)
                    logger.debug("Registered external plugin %s from %s", cls.__name__, module.__name__)
                except (ValueError, TypeError) as e:
                    logger.debug("Skipping external plugin %s: %s", cls.__name__, e)

        # 2) Optional explicit hook — the home for in-process registrations
        #    (register_platform, collective backends, register_default_registry)
        #    and any bespoke wiring.
        hook = getattr(module, "register", None)
        if callable(hook):
            try:
                hook(v)
                logger.debug("Ran register(v) hook for external plugin module %s", module.__name__)
            except Exception:  # noqa: BLE001 - a bad third-party hook must not crash startup
                logger.exception("register(v) hook failed for external plugin module %s", module.__name__)


def _configured_paths(v: "Variables") -> list[Path]:
    """Resolve ``plugins.paths`` from the active config root.

    Reads from ``get_config_root(v)/config.yaml`` rather than the real user
    config unconditionally, so the ``isolate_stateful`` test fixture (which
    redirects the SAF stateful root to a tmp dir) keeps external plugin loading
    inert in the test suite.
    """
    from sparkrun.core.config import SparkrunConfig, get_config_root

    config_path = get_config_root(v) / "config.yaml"
    if not config_path.exists():
        return []
    return SparkrunConfig(config_path=config_path).external_plugin_paths


def load_external_plugins(v: "Variables", paths: "list[Path] | None" = None) -> list[str]:
    """Import and register plugins from user-configured directories.

    Args:
        v: The initialized SAF :class:`~scitrera_app_framework.Variables`.
        paths: Explicit directories to load (mainly for tests).  When ``None``,
            resolved from ``plugins.paths`` in the active config.

    Returns:
        The top-level module names that were loaded — ``[]`` when nothing is
        configured.
    """
    if paths is None:
        # Hard test/CI override first, then the product feature flag (off by
        # default on every channel). Either short-circuits before reading
        # plugins.paths — let alone importing anything. Explicit ``paths=``
        # (programmatic / a plugin's own tests) bypass both.
        if _external_plugins_disabled():
            logger.debug("External plugin auto-load disabled via %s", _DISABLE_ENV)
            return []
        if not feature_gate_enabled(FEATURE_CORE_EXTERNAL_PLUGINS.name, v):
            logger.debug("External plugin auto-load disabled (feature %r off)", FEATURE_CORE_EXTERNAL_PLUGINS.name)
            return []
        paths = _configured_paths(v)

    loaded: list[str] = []
    for path in paths:
        if not path.is_dir():
            logger.warning("Skipping external plugin path (not a directory): %s", path)
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        for mod_info in pkgutil.iter_modules([path_str]):
            name = mod_info.name
            if name.startswith("_"):
                continue
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 - one broken plugin shouldn't kill the CLI
                logger.exception("Failed to import external plugin module %r from %s", name, path)
                continue
            load_plugin_module(module, v)
            loaded.append(name)

    if loaded:
        logger.debug("Loaded external plugin modules from %d path(s): %s", len(paths), loaded)
    return loaded
