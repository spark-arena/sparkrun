"""Shared input-resolution helpers for the sparkrun API.

The CLI does extensive input plumbing (recipe lookup across registries,
host resolution chain, cluster definition loading, runtime
discovery).  Those concerns belong to the *library* layer so the CLI
becomes a thin click-wrapper around it.  This module hosts the pure
versions — no ``click.echo``, no ``sys.exit``, no console I/O.

Each helper accepts an optional ``sctx`` (:class:`SparkrunContext`)
that bundles SAF Variables, :class:`SparkrunConfig`, cached registry/
cluster managers.  When omitted, a fresh session is built via
:func:`sparkrun.api._context.default_sctx`.  Callers that issue
multiple ``api.*`` calls can construct one ``sctx`` and reuse it to
share state (avoid re-reading config / re-scanning registries).

The signature contract: :func:`resolve_cluster` *always* returns a
populated :class:`ClusterDefinition`.  When the caller only supplied
``hosts`` (no named cluster), the function synthesizes an *anonymous*
cluster (``name=""``) carrying those hosts and empty per-host
hardware — equivalent to "no overrides, use the DGX Spark hardware
fallback per host".  Internal code paths therefore never see
``cluster is None``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.api._errors import HostsUnreachable, RecipeNotFound

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterManager
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def resolve_recipe(
    recipe_input: "str | Recipe",
    *,
    sctx: "SparkrunContext | None" = None,
    config: "SparkrunConfig | None" = None,
    overrides: dict | None = None,
    local_files: list[Path] | None = None,
) -> "Recipe":
    """Return a resolved :class:`Recipe` from a name or pre-loaded object.

    When *recipe_input* is already a :class:`Recipe` (or any non-string
    duck-typed object), returns it unchanged (still applying *overrides*
    via ``recipe.resolve``).  When it's a string, looks up the recipe
    across the configured registries.

    Args:
        recipe_input: Recipe name or pre-loaded ``Recipe`` instance.
        sctx: Optional shared session context.  When provided, its
            ``registry_manager`` is used (avoids re-scanning registries).
        config: Explicit override for the config (takes precedence over
            ``sctx.config``).  Builds a default ``SparkrunConfig`` when
            both are absent.
        overrides: Optional override dict applied via ``recipe.resolve``.
        local_files: Optional list of local recipe paths (e.g. CWD-
            discovered recipes) consulted alongside the configured
            registries — mirrors :func:`find_recipe`'s parameter so the
            CLI's cwd-recipe shortcut works through the API.

    Raises:
        RecipeNotFound: When a string name doesn't resolve to any
            recipe in the configured registries or *local_files*.
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe, find_recipe

    # Any non-string input is treated as a pre-loaded recipe (Recipe
    # instance, or a duck-typed object — supports tests that pass
    # mocks).  Only bare strings flow through registry lookup.
    if not isinstance(recipe_input, str):
        recipe = recipe_input
    elif isinstance(recipe_input, Recipe):
        recipe = recipe_input
    else:
        # Prefer sctx.registry_manager when available — it's cached on
        # the session, so chained api calls don't re-scan registries.
        registry_mgr = None
        if sctx is not None:
            try:
                registry_mgr = sctx.registry_manager
            except Exception:
                logger.debug("sctx.registry_manager unavailable", exc_info=True)
        if registry_mgr is None:
            cfg = config or (sctx.config if sctx is not None else SparkrunConfig())
            try:
                from sparkrun.core.registry import RegistryManager

                registry_mgr = RegistryManager(cfg)
            except Exception:
                logger.debug("Failed to construct RegistryManager for recipe lookup", exc_info=True)

        try:
            recipe_path = find_recipe(recipe_input, registry_manager=registry_mgr, local_files=local_files)
        except Exception as e:
            raise RecipeNotFound("Recipe %r not found: %s" % (recipe_input, e)) from e
        if not recipe_path:
            raise RecipeNotFound("Recipe %r not found in any configured registry" % recipe_input)
        recipe = Recipe.load(recipe_path, resolve=False)

    # Apply overrides if provided so downstream callers see a fully-
    # resolved recipe (runtime selection finalized, defaults merged).
    if overrides is not None:
        recipe.resolve(overrides)
    return recipe


def resolve_cluster(
    cluster_input: "str | ClusterDefinition | None" = None,
    hosts_input: tuple[str, ...] | list[str] | None = None,
    *,
    sctx: "SparkrunContext | None" = None,
    cluster_mgr: "ClusterManager | None" = None,
    config: "SparkrunConfig | None" = None,
) -> "ClusterDefinition":
    """Always return a populated :class:`ClusterDefinition`.

    Priority:
      1. *cluster_input* is a :class:`ClusterDefinition` → return it
         (with *hosts_input* overriding ``cluster.hosts`` when both given).
      2. *cluster_input* is a string → load via ``sctx.cluster_manager``
         (or explicit *cluster_mgr*); override hosts with *hosts_input*
         when both are given.
      3. No cluster but *hosts_input* given → synthesize an anonymous
         cluster (``name=""``) carrying those hosts.
      4. No cluster, no *hosts_input*, but ``config.default_hosts`` → synthesize.
      5. Otherwise → raise :class:`HostsUnreachable`.

    Synthesized anonymous clusters have ``name=""`` (empty string) and
    empty ``hosts_hardware`` — equivalent to "no overrides, use the
    DGX Spark hardware fallback per host".  All other fields default
    to ``None`` / ``{}``.

    Args:
        cluster_input: Cluster name, pre-loaded definition, or ``None``.
        hosts_input: Explicit host list (CLI ``--hosts`` equivalent).
            When provided alongside a named/loaded cluster, overrides
            the cluster's host list.
        sctx: Optional shared session context.  Provides cluster manager
            + config for chained-call sharing.
        cluster_mgr: Per-call override of the cluster manager.  Takes
            precedence over ``sctx.cluster_manager``.  Useful for tests.
        config: Optional :class:`SparkrunConfig` override.  Used to
            consult ``default_hosts`` when no other host source exists.

    Raises:
        HostsUnreachable: No host source could be determined.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition

    # Distinguish "no hosts arg given" (None) from "explicit empty list".
    # An empty list is a valid input (e.g. ``api.status([])``) — keep it.
    explicit_hosts = list(hosts_input) if hosts_input is not None else None

    if cluster_input is not None and not isinstance(cluster_input, str):
        # Pre-loaded ClusterDefinition — return as-is (or with hosts overridden).
        if explicit_hosts is not None:
            return _replace_cluster_hosts(cluster_input, explicit_hosts)
        return cluster_input

    if isinstance(cluster_input, str):
        # Named cluster lookup.
        if cluster_mgr is None and sctx is not None:
            try:
                cluster_mgr = sctx.cluster_manager
            except Exception:
                logger.debug("sctx.cluster_manager unavailable", exc_info=True)
        if cluster_mgr is None:
            from sparkrun.core.cluster_manager import ClusterManager
            from sparkrun.core.config import get_config_root

            cluster_mgr = ClusterManager(get_config_root())
        loaded = cluster_mgr.get(cluster_input)
        if explicit_hosts is not None:
            return _replace_cluster_hosts(loaded, explicit_hosts)
        return loaded

    # No cluster.  Need a host source.
    if explicit_hosts is not None:
        return ClusterDefinition(name="", hosts=explicit_hosts)

    effective_config = config if config is not None else (sctx.config if sctx is not None else None)
    default_hosts = getattr(effective_config, "default_hosts", None) if effective_config is not None else None
    if default_hosts:
        return ClusterDefinition(name="", hosts=list(default_hosts))

    raise HostsUnreachable("No hosts provided and no default hosts configured")


def _replace_cluster_hosts(cluster: "ClusterDefinition", hosts: list[str]) -> "ClusterDefinition":
    """Return a copy of *cluster* with ``hosts`` replaced.

    Used when a caller provides both a named cluster and an explicit
    ``hosts_input`` — the explicit list wins but the cluster's other
    fields (per-host hardware, executor, user, …) are preserved.

    Note: per-host hardware entries for hosts not in the new list are
    kept in ``hosts_hardware``; the dict's purpose is *lookup by host*,
    so stale entries are harmless and dropping them would complicate
    round-tripping cluster definitions through this function.
    """
    from dataclasses import replace

    return replace(cluster, hosts=list(hosts))


def resolve_runtime(
    recipe: "Recipe",
    *,
    sctx: "SparkrunContext | None" = None,
):
    """Return the :class:`RuntimePlugin` instance for *recipe.runtime*.

    Uses ``sctx.variables`` when provided so SAF lookups consult the
    same plugin registry the caller is sharing across api calls.

    Raises:
        sparkrun.api.SparkrunError: When the runtime name doesn't map
            to any registered plugin.  (Translated from the underlying
            ``ValueError`` so callers can catch ``SparkrunError``.)
    """
    from sparkrun.api._errors import SparkrunError
    from sparkrun.core.bootstrap import get_runtime

    v = sctx.variables if sctx is not None else None
    try:
        return get_runtime(recipe.runtime, v=v)
    except ValueError as e:
        raise SparkrunError("Cannot resolve runtime %r: %s" % (recipe.runtime, e)) from e


__all__ = [
    "resolve_recipe",
    "resolve_cluster",
    "resolve_runtime",
]
