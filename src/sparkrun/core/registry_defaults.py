"""Registries a plugin declares should exist while it is loaded.

The shipped defaults (:data:`sparkrun.core.registry.BOOTSTRAP_REGISTRY_URLS` and
:data:`~sparkrun.core.registry.FALLBACK_DEFAULT_REGISTRIES`) are closed
constants.  This module is the extension point beside them: a plugin whose
recipes are inert without it — ColdSnap is the first — declares its registry
here, and it becomes resolvable (``@coldsnap/<recipe>``) for anyone with the
plugin enabled, without the user adding anything.

An **in-process** registry, like :mod:`sparkrun.platforms` and
:func:`sparkrun.models.kv.register_kv_strategy`, rather than a SAF extension
point: registration must be cheap (it runs on every CLI invocation, including
shell completion — see below), ordered by the plugin loader, and reachable from
a plugin's ``register(v)`` hook.

**Registration performs no I/O.**  It records intent and validates; nothing is
cloned, read or written.  This is not a nicety: shell completion runs
``init_sparkrun`` on every ``<TAB>`` (Click resolves the command tree through
``PluggableGroup.get_command``), so anything expensive here lands on the
interactive path.  It is also why a plugin cannot point at a remote manifest and
have its *names* come from there — the overlay has to be computable offline.

**Declarations are an overlay, never persisted.**  ``RegistryManager`` merges
them into the list it loads; ``registries.yaml`` keeps only what the user owns.
So uninstalling or disabling a plugin removes its registries with it, and there
is no orphaned entry to reap.  User decisions still stick — mutating a declared
registry (``registry disable`` / ``trust`` / ``untrust``) *materializes* it into
the file, after which the file wins; ``registry remove`` records a tombstone.

Tier
----
:class:`DeclarationTier` records **who vouches for** a declaration, and decides
whether a ``trusted=True`` claim is honored (see
:meth:`DeclaredRegistry.effective_entry`).  It is supplied by the **loader**,
never by the plugin: ``load_plugin_module`` wraps registration in
:func:`declaring_tier`, and a call made outside any loader is treated as
``OUT_OF_TREE`` — least privilege, so nothing can acquire in-tree trust by
registering from an unexpected place.

(The design note originally proposed threading the tier as an explicit argument.
That is not implementable: the *plugin* makes the call, so the loader has no
argument to thread — the value has to be ambient for the duration of the load.)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace as _dataclass_replace
from enum import Enum
from typing import Iterator

from sparkrun.core.registry import (
    RegistryEntry,
    RegistryError,
    assert_safe_registry_entry,
    validate_registry_name,
)

logger = logging.getLogger(__name__)


class DeclarationTier(Enum):
    """Provenance of a plugin declaration."""

    #: Ships inside the sparkrun distribution (``sparkrun.plugins.*``), whether
    #: authored here or vendored in under ``vendor/*.lock``.
    IN_TREE = "in-tree"

    #: Loaded from a user-configured ``plugins.paths`` directory.
    OUT_OF_TREE = "out-of-tree"


#: Ambient tier for the duration of a plugin load.  ``OUT_OF_TREE`` is the
#: default because it is the *less* privileged of the two: a declaration made
#: outside a loader — a stray import, a test, a plugin calling in from a thread
#: — must not be able to obtain in-tree trust by accident.
_ACTIVE_TIER: ContextVar[DeclarationTier] = ContextVar(
    "sparkrun_declaring_tier",
    default=DeclarationTier.OUT_OF_TREE,
)


@contextmanager
def declaring_tier(tier: DeclarationTier) -> Iterator[None]:
    """Mark declarations made inside this block as coming from *tier*.

    Used by the plugin loaders (:mod:`sparkrun.core.in_tree_plugins`,
    :mod:`sparkrun.core.external_plugins`); plugins do not call this.
    """
    token = _ACTIVE_TIER.set(tier)
    try:
        yield
    finally:
        _ACTIVE_TIER.reset(token)


@dataclass(frozen=True)
class DeclaredRegistry:
    """One registry a plugin declared, plus who declared it."""

    entry: RegistryEntry
    owner: str
    tier: DeclarationTier

    def effective_entry(self) -> RegistryEntry:
        """A private copy of the entry, with trust resolved by tier.

        An **in-tree** declaration may ship ``trusted=True``: it arrives through
        the same review and release gate as the shipped defaults (and, when
        vendored, through a pinned commit with per-file digests), so declaring
        it trusted is the same act as adding a trusted entry to
        ``FALLBACK_DEFAULT_REGISTRIES``.

        An **out-of-tree** declaration is forced untrusted.  Installing a plugin
        says "I want this capability"; it does not say "I grant its recipe repo
        standing permission to run lifecycle hooks from whatever it contains
        next month".  Those are separable, and the user grants the second
        explicitly with ``sparkrun registry trust <name>`` — which materializes
        the entry, so it sticks.

        The copy matters: callers mutate what they are handed (``trust_registry``
        flips ``trusted``, ``disable_registry`` flips ``enabled``), and this is
        the only source of these entries, so aliasing would let one invocation
        rewrite the declaration process-wide.
        """
        trusted = bool(self.entry.trusted) and self.tier is DeclarationTier.IN_TREE
        if self.entry.trusted and not trusted:
            logger.debug(
                "Ignoring trusted=True on registry %r declared by out-of-tree plugin %r; run 'sparkrun registry trust %s' to grant it",
                self.entry.name,
                self.owner,
                self.entry.name,
            )
        return _dataclass_replace(self.entry, trusted=trusted, declared_by=self.owner)


#: name -> declaration.  Keyed by name because a name is what collides.
_DECLARED: dict[str, DeclaredRegistry] = {}


def register_default_registry(entry: RegistryEntry, *, owner: str) -> None:
    """Declare a registry that should exist while *owner* is loaded.

    Args:
        entry: The registry to contribute.  ``enabled`` / ``visible`` / trusted
            are honored subject to the tier rules above.
        owner: The declaring plugin's name.  Required and non-blank.

    Raises:
        ValueError: If *owner* is blank.
        RegistryError: If the name or a subpath is unsafe, the name impersonates
            a reserved namespace, or a *different* owner already declared it.

    Registration is idempotent for the same owner and an equal entry, so a
    re-imported plugin module is harmless.  A second owner claiming the same
    name raises — mirroring :func:`sparkrun.core.recipe_items.register_recipe_item`,
    because two plugins silently reinterpreting one registry name has no correct
    arbitration.

    Validation runs **here**, at the declaration's own call site, rather than
    where the overlay is applied — a plugin is local code we can fix, unlike a
    hostile remote manifest that must be survived, so this raises instead of
    dropping the entry.
    """
    if not owner or not owner.strip():
        raise ValueError("plugin registry declaration requires a non-blank owner")

    # Both guards are mandatory: a name becomes a directory under the cache root
    # (and _link_registry_to_shared rmtree()s a non-link cache dir, making an
    # escaping name a delete primitive), and a subpath becomes a path inside it
    # whose contents are offered to the recipe loader.
    validate_registry_name(entry.name, entry.url)
    assert_safe_registry_entry(entry)

    declaration = DeclaredRegistry(entry=entry, owner=owner, tier=_ACTIVE_TIER.get())

    existing = _DECLARED.get(entry.name)
    if existing is not None:
        if existing.owner == owner and existing.entry == entry:
            return
        if existing.owner != owner:
            raise RegistryError("registry %r is already declared by plugin %r" % (entry.name, existing.owner))
    _DECLARED[entry.name] = declaration
    logger.debug(
        "Plugin %r declared %s registry %r -> %s",
        owner,
        declaration.tier.value,
        entry.name,
        entry.url,
    )


def iter_declared_registries() -> tuple[DeclaredRegistry, ...]:
    """Every current declaration, ordered by owner then name.

    Deterministic rather than insertion-ordered so the overlay a user sees does
    not depend on plugin load order.
    """
    return tuple(sorted(_DECLARED.values(), key=lambda d: (d.owner, d.entry.name)))


def declared_registry_names() -> frozenset[str]:
    """Names currently declared by a plugin."""
    return frozenset(_DECLARED)


def reset_declared_registries() -> None:
    """Drop every declaration.

    Process-global state, so the test suite must reset it between tests exactly
    as it resets the bootstrap ``_variables`` singleton — otherwise one test's
    declaration leaks into every later test's registry list.
    """
    _DECLARED.clear()


__all__ = [
    "DeclarationTier",
    "DeclaredRegistry",
    "declared_registry_names",
    "declaring_tier",
    "iter_declared_registries",
    "register_default_registry",
    "reset_declared_registries",
]
