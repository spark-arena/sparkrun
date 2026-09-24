"""Recipe ``include:`` — build a recipe as a small delta over another one.

A recipe may name one base recipe::

    include: qwen3.8-27b-mtp            # sibling file in the same directory
    include: "@official/qwen3.8-27b"    # fully qualified registry recipe

The base is loaded (recursively, since a base may itself include), then the
including recipe is merged over it, and the result is an ordinary recipe. So
the fingerprint, intent, validation and a flattened ``export`` all see the
merged recipe. :class:`IncludeSource` records the chain for the two consumers
that need provenance, not content: trust (a chain is only as trusted as its
least-trusted member) and display.

Merge rules (:func:`merge_recipe_data`):

* mappings merge key by key; scalars and lists replace
* an explicit ``null`` deletes the inherited key
* **values inside ``defaults`` and ``env`` are atomic**: a default is one
  setting, so an outer ``speculative_config: {...}`` replaces the base's whole
  mapping rather than blending two speculators into one config
* ``overrides:`` appends (base entries first, then the includer's), so an
  included recipe's conditional tuning survives and the includer's wins ties

Resolution is strict. A bare reference is a file **in the same directory**:
no path separators, so an include cannot walk the filesystem, and no bare
includes at all from sources without a trustworthy directory (URL-fetched or
catalog-imported recipes). A ``@registry/name`` reference resolves through the
registry manager exactly as ``sparkrun run @registry/name`` would.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import re

from sparkrun.utils import parse_scoped_name

if TYPE_CHECKING:
    from sparkrun.core.registry import RegistryManager

logger = logging.getLogger(__name__)

#: Longest include chain accepted. Chains are for "this recipe, but with X";
#: anything deeper is almost certainly a mistake, and the cap bounds the work
#: a hostile registry recipe can make a loader do.
MAX_INCLUDE_DEPTH = 8

#: Top-level keys whose *values* are single settings, replaced rather than
#: merged. See the module docstring.
_ATOMIC_VALUE_SECTIONS = frozenset({"defaults", "env"})

_RECIPE_SUFFIXES = (".yaml", ".yml")

#: One path component of an include name: starts alphanumeric (rules out
#: ``..``, dotfiles and a leading ``-``) and carries no separator or drive
#: colon. The registry-subpath charset.
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


@dataclass(frozen=True)
class IncludeSource:
    """One recipe pulled in through ``include:``, outermost base first.

    ``registry`` is set when the base was resolved through a registry
    (``@registry/name``). ``None`` means it was a sibling file and so shares
    the source (and trust) of the recipe that included it.
    """

    ref: str
    path: str
    registry: str | None = None
    registry_url: str | None = None

    def to_dict(self) -> dict[str, str]:
        d = {"ref": self.ref, "path": self.path}
        if self.registry:
            d["registry"] = self.registry
        if self.registry_url:
            d["registry_url"] = self.registry_url
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IncludeSource:
        return cls(
            ref=str(data.get("ref", "")),
            path=str(data.get("path", "")),
            registry=data.get("registry") or None,
            registry_url=data.get("registry_url") or None,
        )


def merge_recipe_data(base: dict[str, Any], outer: dict[str, Any]) -> dict[str, Any]:
    """Merge *outer* over *base* by the include rules; neither input is mutated.

    ``metadata`` describes the *model* (parameter count, dtype, architecture
    fields the memory estimate and placement read). When the outer recipe
    serves a different model the base's metadata is not inherited: stale
    architecture numbers would size the wrong model.
    """
    merged = deepcopy(base)
    merged.pop("include", None)
    if "model" in outer and outer["model"] != base.get("model"):
        merged.pop("metadata", None)
    for key, value in outer.items():
        if key == "include":
            continue
        if value is None:
            merged.pop(key, None)
        elif key == "overrides" and isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = merged[key] + deepcopy(value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value, atomic=key in _ATOMIC_VALUE_SECTIONS)
        else:
            merged[key] = deepcopy(value)
    return merged


def _merge_mapping(base: dict[str, Any], outer: dict[str, Any], *, atomic: bool) -> dict[str, Any]:
    result = dict(base)
    for key, value in outer.items():
        if value is None:
            result.pop(key, None)
        elif not atomic and isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_mapping(result[key], value, atomic=False)
        else:
            result[key] = deepcopy(value)
    return result


def listing_view(data: dict[str, Any], source_path: str | Path) -> dict[str, Any]:
    """Best-effort merged view for listing and discovery, without registries.

    Listing reads raw YAML for speed, so a recipe that is mostly ``include:``
    would show no model. Sibling includes are resolved here. A registry include
    needs the registry manager, and a catalog scan must not trigger clones, so
    such a chain keeps the outer data and full resolution happens at load.
    """
    if "include" not in data:
        return data
    try:
        return resolve_recipe_includes(data, source_path, registry_manager=_NO_REGISTRIES)[0]
    except Exception:
        logger.debug("include: could not build a listing view for %s", source_path, exc_info=True)
        return data


class _NoRegistries:
    """Registry-manager stand-in that refuses scoped lookups (listing only)."""

    def find_recipe_in_registries(self, *_args, **_kwargs):
        raise LookupError("registry includes are not resolved while listing")


_NO_REGISTRIES: Any = _NoRegistries()


def resolve_recipe_includes(
    data: dict[str, Any],
    source_path: str | Path | None,
    *,
    registry_manager: RegistryManager | None = None,
    allow_local: bool = True,
) -> tuple[dict[str, Any], tuple[IncludeSource, ...]]:
    """Return ``(merged_data, chain)`` for a recipe that may use ``include:``.

    A recipe without ``include`` is returned unchanged with an empty chain,
    so callers can run every load through this.
    """
    if "include" not in data:
        return data, ()
    visited = {str(Path(source_path).resolve())} if source_path else set()
    return _resolve(data, source_path, registry_manager, allow_local, visited, depth=0)


def _resolve(
    data: dict[str, Any],
    source_path: str | Path | None,
    registry_manager: RegistryManager | None,
    allow_local: bool,
    visited: set[str],
    *,
    depth: int,
) -> tuple[dict[str, Any], tuple[IncludeSource, ...]]:
    from sparkrun.core.recipe import RecipeError, read_yaml

    if "include" not in data:
        return data, ()
    ref = data["include"]
    if not isinstance(ref, str) or not ref.strip():
        raise RecipeError("include: must be a recipe name (a sibling file or @registry/recipe), got %r" % (ref,))
    ref = ref.strip()
    if depth >= MAX_INCLUDE_DEPTH:
        raise RecipeError("include: chain is deeper than %d recipes (at %r)" % (MAX_INCLUDE_DEPTH, ref))
    _reject_v1(data, source_path)

    path, registry, registry_url = _locate(ref, source_path, registry_manager, allow_local)
    key = str(path.resolve())
    if key in visited:
        raise RecipeError("include: cycle detected — %r is already part of this include chain" % ref)
    visited.add(key)

    base = read_yaml(str(path))
    if not isinstance(base, dict):
        raise RecipeError("include: %r (%s) is not a YAML mapping" % (ref, path))
    _reject_v1(base, path)

    # A registry base is its own source: its sibling includes stay inside it,
    # and it may itself include across registries.
    base, base_chain = _resolve(base, path, registry_manager, allow_local=True, visited=visited, depth=depth + 1)
    if registry is not None:
        base = _qualify_inherited_mods(base, registry)

    # A sibling base (registry None) shares the includer's source; any registry
    # includes further down its own chain keep the registry they recorded.
    source = IncludeSource(ref=ref, path=str(path), registry=registry, registry_url=registry_url)
    return merge_recipe_data(base, data), (*base_chain, source)


def _locate(
    ref: str,
    source_path: str | Path | None,
    registry_manager: RegistryManager | None,
    allow_local: bool,
) -> tuple[Path, str | None, str | None]:
    from sparkrun.core.recipe import RecipeError, find_recipe

    scope, name = parse_scoped_name(ref)
    if scope:
        # The name is recipe content (a third-party recipe chooses it), and the
        # registry lookup joins it onto the registry directory. An absolute or
        # ``..`` part would read any YAML on the control machine and record it
        # as coming from (and trusted as) the named registry.
        if not name or not all(_SAFE_COMPONENT.match(part) for part in name.split("/")):
            raise RecipeError("include: %r is not a valid @registry/recipe name" % ref)
        manager = registry_manager
        if manager is None:
            from sparkrun.core.config import SparkrunConfig

            manager = SparkrunConfig().get_registry_manager()
            manager.ensure_initialized()
        path = Path(find_recipe(ref, registry_manager=manager))
        entry = manager.get_registry(scope, allow_discovery=False)
        _require_within(path, _registry_root(manager, entry), ref)
        return path, entry.name, entry.url

    if not allow_local:
        raise RecipeError(
            "include: %r must be a fully qualified @registry/recipe name — sibling includes are not allowed "
            "for recipes fetched from a URL or imported into the catalog" % ref
        )
    if not _SAFE_COMPONENT.match(name):
        raise RecipeError("include: %r must name a recipe in the same directory (no path separators) or a @registry/recipe" % ref)
    if not source_path:
        raise RecipeError("include: %r is a sibling reference, but this recipe has no file location" % ref)
    directory = Path(source_path).parent
    candidates = [directory / name] if name.endswith(_RECIPE_SUFFIXES) else [directory / (name + s) for s in _RECIPE_SUFFIXES]
    for candidate in candidates:
        if candidate.is_file():
            # ``is_file`` follows symlinks, and a git registry can ship one
            # pointing anywhere; the resolved target must stay beside the includer.
            _require_within(candidate, directory, ref)
            return candidate, None, None
    raise RecipeError("include: %r not found next to %s" % (ref, source_path))


def _registry_root(manager: Any, entry: Any) -> Path | None:
    recipe_dir = getattr(manager, "_recipe_dir", None)
    return recipe_dir(entry) if callable(recipe_dir) else None


def _require_within(path: Path, root: Path | None, ref: str) -> None:
    """Refuse an include whose resolved file lies outside *root*.

    Both sides are resolved, so a registry cache dir that is itself a symlink
    to a shared clone still contains its own recipes.
    """
    from sparkrun.core.recipe import RecipeError

    if root is None:
        raise RecipeError("include: %r — could not determine where that registry's recipes live" % ref)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise RecipeError("include: %r resolves outside its registry or directory (%s)" % (ref, path.resolve()))


def _qualify_inherited_mods(data: dict[str, Any], registry: str) -> dict[str, Any]:
    """Scope unscoped ``mods:`` refs to the registry the base came from.

    Mods resolve relative to the recipe that is launched (its directory, then
    its registry). A mod inherited from a registry base would otherwise be
    looked up next to the includer, which is not where its author put it.
    Scoping it keeps it resolving where it was written, and keeps a flattened
    export self-contained.
    """
    mods = data.get("mods")
    if not isinstance(mods, list):
        return data
    qualified = []
    for ref in mods:
        if isinstance(ref, str) and parse_scoped_name(ref)[0] is None:
            qualified.append("@%s/%s" % (registry, ref))
        else:
            qualified.append(ref)
    return {**data, "mods": qualified}


def _reject_v1(data: dict[str, Any], source: str | Path | None) -> None:
    from sparkrun.core.recipe import RecipeError

    if str(data.get("recipe_version", "2")) == "1":
        raise RecipeError('include: is a v2 feature; %s is a v1 recipe (recipe_version: "1") — migrate it first' % (source or "recipe"))


__all__ = [
    "MAX_INCLUDE_DEPTH",
    "IncludeSource",
    "listing_view",
    "merge_recipe_data",
    "resolve_recipe_includes",
]
