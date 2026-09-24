"""Foreign recipe formats: registries whose files are not sparkrun recipes.

A registry entry names its ``format`` (default :data:`DEFAULT_RECIPE_FORMAT`,
sparkrun's own YAML). A plugin that understands another launcher's manifests
registers a :class:`RecipeFormat`. Registry lookup, catalog listing and
:meth:`Recipe.load <sparkrun.core.recipe.Recipe.load>` then dispatch on it, and
what comes out is an ordinary v2 recipe dict. Everything downstream (the
config chain, validation, fingerprint, ``export``) is unchanged.

Three rules shape the contract:

* **Listing and lookup share one enumerator.** A format supplies only
  ``iter_files`` (every manifest under a registry directory) and ``name_of``
  (the name a user types). :func:`find_in_format` is built from those two, so
  something listed is always runnable and vice versa (the ``iter_asset_files``
  rule).
* **An unregistered format is inert, never misread.** A registry whose format
  no loaded plugin provides lists nothing and resolves nothing. Scanning its
  files as sparkrun YAML would offer launch recipes that are not recipes.
* **Listing never reaches the network.** ``load(..., offline=True)`` is what
  the catalog calls. A format that needs remote facts (checkpoint config, for
  instance) must degrade to what it can say locally.

In-process like :func:`~sparkrun.models.kv.register_kv_strategy`: formats are
constructed values with callables, not stateless SAF singletons. Plugins call
:func:`register_recipe_format` from their ``register(v)`` hook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from sparkrun.core.registration import enlist_registry_state

if TYPE_CHECKING:
    from sparkrun.core.registry import RegistryManager

logger = logging.getLogger(__name__)

#: The format every registry has unless it says otherwise.
DEFAULT_RECIPE_FORMAT = "sparkrun"


@dataclass(frozen=True)
class RecipeFormat:
    """How one foreign manifest format maps onto sparkrun recipes.

    Attributes:
        name: Value of ``RegistryEntry.format`` that selects this format.
        owner: The registering plugin, used in conflict errors.
        iter_files: ``root -> [manifest paths]``: every recipe under a
            registry directory, sorted. Drafts or fragments that are not
            launchable must be left out here.
        name_of: ``(path, root) -> name`` a user types after ``@registry/``.
            ``root`` is ``None`` for a manifest loaded by path from outside
            any registry.
        load: ``(path, *, registry_manager, offline, allow_local) -> dict``
            returning v2 recipe data. ``offline=True`` is the listing path and
            must not touch the network. ``allow_local=False`` (a URL-fetched
            or imported source) forbids reading files next to *path*, the rule
            ``include:`` follows. The manifest is read as YAML before
            dispatch, so formats are YAML-based.
        claims: Optional ``(path, data) -> bool`` recognizing a manifest by
            content. Consulted **only** for a path outside every registry
            from a local source, never for a native registry's files or a
            URL-fetched / imported recipe.
    """

    name: str
    owner: str
    iter_files: Callable[[Path], list[Path]]
    name_of: Callable[[Path, Path | None], str]
    load: Callable[..., dict[str, Any]]
    claims: Callable[[Path, dict[str, Any]], bool] | None = None


_FORMATS: dict[str, RecipeFormat] = {}
enlist_registry_state(globals(), "_FORMATS")


def register_recipe_format(recipe_format: RecipeFormat) -> None:
    """Register *recipe_format*. Idempotent for the same owner; a second owner raises.

    Raises:
        ValueError: The name is the built-in format.
        PluginConflictError: Another owner already registered the name.
    """
    if recipe_format.name == DEFAULT_RECIPE_FORMAT:
        raise ValueError("recipe format %r is built in and cannot be re-registered" % DEFAULT_RECIPE_FORMAT)
    existing = _FORMATS.get(recipe_format.name)
    if existing is not None and existing.owner != recipe_format.owner:
        from sparkrun.core.installed_plugins import PluginConflictError

        raise PluginConflictError(
            "recipe format %r is already registered by %s (not %s)" % (recipe_format.name, existing.owner, recipe_format.owner)
        )
    _FORMATS[recipe_format.name] = recipe_format


def unregister_recipe_format(name: str) -> None:
    """Drop a registration (tests, and plugins that unload)."""
    _FORMATS.pop(name, None)


def reset_recipe_formats() -> None:
    """Drop every registration. Process-global, so the test suite resets it per test."""
    _FORMATS.clear()


def get_recipe_format(name: str | None) -> RecipeFormat | None:
    """The registered format called *name*; ``None`` for the default or an unknown one."""
    if not name or name == DEFAULT_RECIPE_FORMAT:
        return None
    return _FORMATS.get(name)


def is_foreign_format(name: str | None) -> bool:
    """True for any format other than sparkrun's own, registered or not."""
    return bool(name) and name != DEFAULT_RECIPE_FORMAT


def entry_recipe_format(entry: Any) -> tuple[RecipeFormat | None, str | None]:
    """``(handler, why_unavailable)`` for a registry entry's ``format``.

    ``(None, None)`` is sparkrun's own format. A foreign format is honored only
    when the registry is **trusted** or was **declared by a plugin**. A
    registry's manifest chooses its own ``format:``, and without this gate any
    third-party repo could pick which installed plugin parses its files. A
    plugin-declared registry's format came from local code, not a remote
    manifest. Unhonored means *inert*: not parsed as sparkrun YAML either,
    just unavailable, and ``why_unavailable`` says so.
    """
    name = getattr(entry, "format", DEFAULT_RECIPE_FORMAT) or DEFAULT_RECIPE_FORMAT
    if not is_foreign_format(name):
        return None, None
    if not (getattr(entry, "trusted", False) or getattr(entry, "declared_by", "") or _still_declared(entry)):
        return None, (
            "its %r recipe format is only honored for trusted registries (sparkrun registry trust %s)" % (name, getattr(entry, "name", "?"))
        )
    handler = get_recipe_format(name)
    if handler is None:
        return None, "no loaded plugin provides its %r recipe format" % name
    return handler, None


def _still_declared(entry: Any) -> bool:
    """A materialized plugin registry (``registry disable/enable/untrust`` clears ``declared_by``).

    Materializing writes the entry to ``registries.yaml``, where it looks
    exactly like one learned from a remote manifest. What distinguishes it is
    that a loaded plugin still declares the same name, URL **and** format, so
    the format still comes from local code.
    """
    from sparkrun.core.registry_defaults import iter_declared_registries

    return any(
        d.entry.name == getattr(entry, "name", None) and d.entry.url == getattr(entry, "url", None) and d.entry.format == entry.format
        for d in iter_declared_registries()
    )


def find_in_format(recipe_format: RecipeFormat, root: Path, name: str) -> list[Path]:
    """Manifests under *root* whose ``name_of`` is *name*, built from ``iter_files``."""
    return [path for path in recipe_format.iter_files(root) if recipe_format.name_of(path, root) == name]


def claiming_format(path: Path, data: dict[str, Any]) -> RecipeFormat | None:
    """The single registered format that claims this manifest by content, if any."""
    claimed = []
    for recipe_format in _FORMATS.values():
        if recipe_format.claims is None:
            continue
        try:
            if recipe_format.claims(path, data):
                claimed.append(recipe_format)
        except Exception:
            logger.debug("recipe format %s: claims() raised for %s", recipe_format.name, path, exc_info=True)
    if len(claimed) > 1:
        logger.warning("%s is claimed by several recipe formats (%s); treating it as none", path, ", ".join(f.name for f in claimed))
        return None
    return claimed[0] if claimed else None


@dataclass(frozen=True)
class PathOwnership:
    """Who owns a manifest path, for format dispatch.

    ``kind`` is ``"foreign"`` (a foreign-format registry: ``format_name`` is
    set, and ``recipe_format`` is ``None`` when it is unavailable, with
    ``unavailable`` saying why),
    ``"native"`` (a sparkrun-format registry), ``"outside"`` (no registry), or
    ``"unknown"`` (no registry manager to ask).
    """

    kind: str
    recipe_format: RecipeFormat | None = None
    root: Path | None = None
    format_name: str | None = None
    unavailable: str | None = None


def format_for_path(path: Path, registry_manager: RegistryManager | None) -> PathOwnership:
    """Classify *path* by the registry that owns it.

    Ownership failures (an orphaned cache path, a path two registries claim)
    raise :class:`~sparkrun.core.registry.RegistryError` rather than reading as
    "outside": falling through would parse a foreign manifest as sparkrun YAML.
    """
    if registry_manager is None:
        return PathOwnership("unknown")
    registry_name = registry_manager.registry_for_path(path, allow_discovery=False)
    if registry_name is None:
        return PathOwnership("outside")
    entry = registry_manager.get_registry(registry_name, allow_discovery=False)
    name = getattr(entry, "format", DEFAULT_RECIPE_FORMAT) or DEFAULT_RECIPE_FORMAT
    if not is_foreign_format(name):
        return PathOwnership("native")
    handler, unavailable = entry_recipe_format(entry)
    return PathOwnership("foreign", handler, registry_manager._recipe_dir(entry), name, unavailable)


__all__ = [
    "DEFAULT_RECIPE_FORMAT",
    "PathOwnership",
    "RecipeFormat",
    "claiming_format",
    "entry_recipe_format",
    "find_in_format",
    "format_for_path",
    "get_recipe_format",
    "is_foreign_format",
    "register_recipe_format",
    "unregister_recipe_format",
]
