"""Shared input-resolution helpers for the sparkrun API.

The CLI does extensive input plumbing (recipe lookup across registries,
host resolution chain, cluster definition loading, runtime
discovery).  Those concerns belong to the *library* layer so the CLI
becomes a thin click-wrapper around it.  This module hosts the pure
versions — no ``click.echo``, no ``sys.exit``, no console I/O.

Each helper accepts a piece of :class:`~sparkrun.api.RunOptions`
input (which may be a raw string identifier or a pre-loaded object)
and returns a fully-resolved object, raising a typed
:class:`~sparkrun.api.SparkrunError` subclass on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.api._errors import HostsUnreachable, RecipeNotFound

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterManager
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def resolve_recipe(
    recipe_input: "str | Recipe",
    *,
    config: "SparkrunConfig | None" = None,
    overrides: dict | None = None,
    local_files: list[Path] | None = None,
) -> "Recipe":
    """Return a resolved :class:`Recipe` from a name or pre-loaded object.

    When *recipe_input* is already a :class:`Recipe`, returns it
    unchanged (still applying *overrides* via ``recipe.resolve``).
    When it's a string, looks up the recipe across the configured
    registries.

    Args:
        recipe_input: Recipe name or pre-loaded ``Recipe`` instance.
        config: Optional ``SparkrunConfig`` (built on demand if absent).
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
        # Build the registry manager from config so find_recipe can
        # consult configured registries.
        cfg = config or SparkrunConfig()
        registry_mgr = None
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


def resolve_cluster_def(
    cluster_input: "str | ClusterDefinition | None",
    *,
    cluster_mgr: "ClusterManager | None" = None,
) -> "ClusterDefinition | None":
    """Return a :class:`ClusterDefinition` or ``None``.

    Accepts a cluster name (looked up via *cluster_mgr*), a
    pre-loaded :class:`ClusterDefinition`, or ``None`` (caller is
    using explicit hosts without a named cluster).
    """
    from sparkrun.core.cluster_manager import ClusterDefinition

    if cluster_input is None:
        return None
    if isinstance(cluster_input, ClusterDefinition):
        return cluster_input
    if cluster_mgr is None:
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import get_config_root

        cluster_mgr = ClusterManager(get_config_root())
    return cluster_mgr.get(cluster_input)


def resolve_hosts(
    hosts_input: tuple[str, ...] | list[str] | None,
    *,
    cluster: "ClusterDefinition | None" = None,
    config: "SparkrunConfig | None" = None,
) -> list[str]:
    """Resolve the working host list from API inputs.

    Priority chain (matches the CLI's behaviour):
      1. Explicit *hosts_input* (CLI ``--hosts`` equivalent).
      2. Hosts from *cluster*.
      3. ``config.default_hosts``.

    Raises:
        HostsUnreachable: When no host source is configured.
    """
    if hosts_input:
        return list(hosts_input)
    if cluster is not None and cluster.hosts:
        return list(cluster.hosts)
    if config is not None and getattr(config, "default_hosts", None):
        return list(config.default_hosts)
    raise HostsUnreachable("No hosts provided and no default hosts configured")


def resolve_runtime(recipe: "Recipe"):
    """Return the :class:`RuntimePlugin` instance for *recipe.runtime*.

    Raises:
        sparkrun.api.SparkrunError: When the runtime name doesn't map
            to any registered plugin.  (Translated from the underlying
            ``ValueError`` so callers can catch ``SparkrunError``.)
    """
    from sparkrun.api._errors import SparkrunError
    from sparkrun.core.bootstrap import get_runtime

    try:
        return get_runtime(recipe.runtime)
    except ValueError as e:
        raise SparkrunError("Cannot resolve runtime %r: %s" % (recipe.runtime, e)) from e


__all__ = [
    "resolve_recipe",
    "resolve_cluster_def",
    "resolve_hosts",
    "resolve_runtime",
]
