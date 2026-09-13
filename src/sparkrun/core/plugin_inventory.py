"""Enumerate the plugin *modules* sparkrun knows about, loaded or not.

The console-free source behind ``sparkrun setup plugins list``, and the
inventory peer of :mod:`sparkrun.core.in_tree_plugins` /
:mod:`sparkrun.core.external_plugins`: those two decide what to *load*, this
one reports what exists.  Both enumerate through the loaders' own helpers
(:func:`~sparkrun.core.in_tree_plugins.iter_in_tree_plugin_names`,
:func:`~sparkrun.core.external_plugins.iter_plugin_module_names`) so a listing
can never name a plugin the loader would skip, or omit one it would load — a
catalog that disagrees with the loader is worse than none, because it is read
as an answer.

**Scope is plugin modules**, i.e. exactly the set those two loaders govern:
in-tree subpackages of ``sparkrun.plugins`` and out-of-tree top-level modules
under ``plugins.paths``.  Deliberately *not* every SAF extension — a runtime or
an executor shipped in core has no version distinct from sparkrun's, and
``list-runtimes`` / ``list-executors`` already enumerate those.

**Nothing here imports a plugin.**  Enumeration is directory-level, and a
version is read only off a module some loader already imported.  So listing is
safe with a plugin's gate off: the module stays unimported and its version is
honestly reported as unknown rather than obtained by importing something the
user has switched off.

Version resolution, in order:

1. ``module.__version__`` on the loaded module — the declared contract (see
   ``docs/PLUGINS.md``).
2. For **out-of-tree** plugins only, the version of the installed distribution
   providing that top-level module.
3. ``None`` — *unknown*, rendered as such and never guessed.

Step 2 is restricted to out-of-tree on purpose.  Every in-tree plugin's
top-level package maps to the ``sparkrun`` distribution, so applying the
fallback there would report sparkrun's own version as the plugin's.  That is
wrong wherever it matters most: ``sparkroute`` is vendored from its own
repository at its own version, and a plugin silently inheriting the host's
version is a fabricated answer where the honest one is "the plugin did not say".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)

#: ``source`` values.
SOURCE_IN_TREE = "in-tree"
SOURCE_EXTERNAL = "external"

#: ``version_source`` values (``None`` when the version is unknown).
VERSION_FROM_MODULE = "module"
VERSION_FROM_DISTRIBUTION = "distribution"

#: The attribute a plugin sets to declare its version.
VERSION_ATTR = "__version__"


@dataclass(frozen=True)
class PluginInfo:
    """One plugin module, whether or not it loaded this process."""

    name: str
    """Plugin name as the loader spells it (the module/subpackage name)."""

    source: str
    """:data:`SOURCE_IN_TREE` or :data:`SOURCE_EXTERNAL`."""

    module: str
    """Dotted module name the loader imports."""

    enabled: bool
    """Whether the plugin's feature gate currently resolves on."""

    loaded: bool
    """Whether *this process* actually loaded it (gate on **and** import ok)."""

    feature_flag: str | None = None
    """The flag gating it; ``None`` for an in-tree plugin missing its binding."""

    version: str | None = None
    """Declared version, or ``None`` for *unknown* — never a guess."""

    version_source: str | None = None
    """How :attr:`version` was obtained; ``None`` when it is unknown."""

    path: Path | None = None
    """The ``plugins.paths`` directory it came from (out-of-tree only)."""

    @property
    def version_display(self) -> str:
        """Version for display, distinguishing *unknown* from *not declared*."""
        if self.version:
            return self.version
        return "unknown"

    def to_dict(self) -> dict:
        """Canonical mapping for ``--json``.

        Hand-written rather than left to ``dataclasses.asdict``, which would
        emit a ``Path`` the JSON encoder cannot serialize. ``version`` stays
        ``null`` when unknown — the string "unknown" is a *display* rendering,
        and emitting it here would be indistinguishable from a plugin that
        declared "unknown" as its version.
        """
        return {
            "name": self.name,
            "source": self.source,
            "module": self.module,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "feature_flag": self.feature_flag,
            "version": self.version,
            "version_source": self.version_source,
            "path": str(self.path) if self.path is not None else None,
        }


def _module_version(dotted: str) -> tuple[str | None, str | None]:
    """Read ``__version__`` off the module loaded as plugin *dotted*."""
    from sparkrun.core.external_plugins import loaded_plugin_module

    module = loaded_plugin_module(dotted)
    if module is None:
        return None, None
    raw = getattr(module, VERSION_ATTR, None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), VERSION_FROM_MODULE
    return None, None


def _distribution_version(top_level: str) -> tuple[str | None, str | None]:
    """Version of the installed distribution providing *top_level*, if any.

    Out-of-tree only (see the module docstring). Best-effort: a plugin dropped
    into ``plugins.paths`` is typically not an installed distribution at all,
    and that is a normal outcome, not an error.
    """
    try:
        from importlib.metadata import packages_distributions, version as distribution_version

        dists = packages_distributions().get(top_level) or []
        for dist in dists:
            found = distribution_version(dist)
            if found:
                return found, VERSION_FROM_DISTRIBUTION
    except Exception:  # noqa: BLE001 - metadata lookup is advisory; unknown is a fine answer
        logger.debug("Could not resolve a distribution version for %r", top_level, exc_info=True)
    return None, None


def _resolve_version(dotted: str, *, source: str, top_level: str) -> tuple[str | None, str | None]:
    version, origin = _module_version(dotted)
    if version is not None:
        return version, origin
    if source == SOURCE_EXTERNAL:
        return _distribution_version(top_level)
    return None, None


def _in_tree_plugins(v: "Variables | None") -> list[PluginInfo]:
    from sparkrun.core.external_plugins import loaded_plugin_module
    from sparkrun.core.features import feature_gate_enabled, get_feature
    from sparkrun.core.in_tree_plugins import (
        IN_TREE_PLUGIN_PACKAGE,
        iter_in_tree_plugin_names,
        plugin_feature_flag,
    )

    out: list[PluginInfo] = []
    for name in iter_in_tree_plugin_names():
        dotted = "%s.%s" % (IN_TREE_PLUGIN_PACKAGE, name)
        flag = plugin_feature_flag(name)
        # An unregistered flag resolves off — same rule the loader applies, so
        # a plugin missing its IN_TREE_PLUGIN_FEATURES binding is listed as off
        # with no flag rather than omitted, which is what makes the defect
        # visible instead of silent.
        registered = flag is not None and get_feature(flag) is not None
        enabled = registered and feature_gate_enabled(flag, v)
        version, origin = _resolve_version(dotted, source=SOURCE_IN_TREE, top_level=name)
        out.append(
            PluginInfo(
                name=name,
                source=SOURCE_IN_TREE,
                module=dotted,
                enabled=enabled,
                loaded=loaded_plugin_module(dotted) is not None,
                feature_flag=flag if registered else None,
                version=version,
                version_source=origin,
            )
        )
    return out


def _external_plugins(config: "SparkrunConfig | None", v: "Variables | None") -> list[PluginInfo]:
    from sparkrun.core.external_plugins import (
        FEATURE_CORE_EXTERNAL_PLUGINS,
        _external_plugins_disabled,
        iter_plugin_module_names,
        loaded_plugin_module,
    )
    from sparkrun.core.features import feature_gate_enabled

    if config is None:
        return []

    # Enumerating directory entries imports nothing, so it is safe (and the
    # whole point of the command) to list these even with the gate off — you
    # cannot decide whether to enable the gate without seeing what it governs.
    gate = FEATURE_CORE_EXTERNAL_PLUGINS.name
    enabled = feature_gate_enabled(gate, v) and not _external_plugins_disabled()

    out: list[PluginInfo] = []
    for path in config.external_plugin_paths:
        for name in iter_plugin_module_names(path):
            version, origin = _resolve_version(name, source=SOURCE_EXTERNAL, top_level=name)
            out.append(
                PluginInfo(
                    name=name,
                    source=SOURCE_EXTERNAL,
                    module=name,
                    enabled=enabled,
                    loaded=loaded_plugin_module(name) is not None,
                    feature_flag=gate,
                    version=version,
                    version_source=origin,
                    path=path,
                )
            )
    return out


def list_plugins(config: "SparkrunConfig | None" = None, v: "Variables | None" = None) -> list[PluginInfo]:
    """Return every known plugin module, in-tree first then out-of-tree.

    Args:
        config: Config supplying ``plugins.paths``. When ``None``, out-of-tree
            plugins are omitted rather than resolved from an implicit config —
            the caller owns which config is in effect.
        v: Initialized SAF variables, for feature-gate resolution.

    Returns:
        In-tree plugins sorted by name, then out-of-tree ones sorted by
        ``(path, name)`` so entries stay grouped by the directory they came
        from. Never raises: an unreadable source contributes no rows.
    """
    plugins = sorted(_in_tree_plugins(v), key=lambda p: p.name)
    plugins.extend(sorted(_external_plugins(config, v), key=lambda p: (str(p.path), p.name)))
    return plugins


__all__ = [
    "PluginInfo",
    "SOURCE_EXTERNAL",
    "SOURCE_IN_TREE",
    "VERSION_ATTR",
    "VERSION_FROM_DISTRIBUTION",
    "VERSION_FROM_MODULE",
    "list_plugins",
]
