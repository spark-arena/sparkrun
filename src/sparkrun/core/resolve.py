"""Shared, console-free input-resolution logic for sparkrun.

This module hosts the *pure* resolvers that used to live in
``sparkrun.cli._common`` but are needed by the library API as well.  None
of these helpers call ``click.echo`` / ``sys.exit`` / ``click.prompt``;
failures raise ordinary exceptions (``RecipeError`` subclasses) so both
the CLI (which translates to ``click.echo`` + ``sys.exit``) and the API
(which surfaces typed ``SparkrunError`` subclasses) can build on the same
core.

The CLI keeps thin wrappers in ``cli/_common.py`` that add the
interactive concerns (disambiguation prompts, untrusted-host
confirmation, registry-refresh retry) and translate exceptions into
console output + exit codes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def apply_recipe_overrides(
    options,
    tensor_parallel=None,
    pipeline_parallel=None,
    data_parallel=None,
    gpu_mem=None,
    max_model_len=None,
    image=None,
    *,
    recipe: Recipe,
    **kwargs,
) -> tuple[Recipe, dict]:
    """Build the overrides dict, apply it to *recipe*, and resolve the runtime.

    Returns ``(recipe, overrides)`` — *recipe* is returned to make the
    mutation explicit (runtime may change based on overrides).

    The required *recipe* is resolved with ``recipe.resolve(overrides)`` so
    that overrides can influence runtime resolution (e.g.
    ``distributed_executor_backend=ray`` switches vllm-distributed to
    vllm-ray).

    Pure logic: no console I/O.  *options* is a tuple of ``key=value``
    strings (the CLI ``--option/-o`` form); malformed entries raise
    ``ValueError`` (the CLI wrapper validates and reports these before
    calling this function, so they should not reach here in practice).
    """
    from sparkrun.utils import coerce_value

    if recipe is None:
        raise ValueError("Recipe overrides require a recipe")

    overrides: dict = {}
    for opt in options or ():
        if "=" not in opt:
            raise ValueError("--option must be key=value, got: %s" % opt)
        key, _, value = opt.partition("=")
        key = key.strip()
        if not key:
            raise ValueError("--option has empty key: %s" % opt)
        overrides[key] = coerce_value(value.strip())

    if tensor_parallel is not None:
        overrides["tensor_parallel"] = tensor_parallel
    if pipeline_parallel is not None:
        overrides["pipeline_parallel"] = pipeline_parallel
    if data_parallel is not None:
        overrides["data_parallel"] = data_parallel
    if gpu_mem is not None:
        overrides["gpu_memory_utilization"] = gpu_mem
    if max_model_len is not None:
        overrides["max_model_len"] = max_model_len
    if image:
        recipe.container = image
        # A matched `overrides:` container layer must not replace an explicit --image.
        recipe._cli_image = True
        if recipe.containers:
            # An explicit --image is a whole-launch override: anything subtler
            # (override the fallback but keep the machine-specific images) would
            # silently leave most nodes on the image the user just replaced.
            logger.info(
                "--image %s overrides the recipe's per-machine `containers:` block (%d entr%s); every node will run this image.",
                image,
                len(recipe.containers),
                "y" if len(recipe.containers) == 1 else "ies",
            )
            recipe.containers = []

    for k, v in kwargs.items():
        if v is not None:
            overrides[k] = v

    # Apply env.* overrides to recipe.env directly
    for k, v in list(overrides.items()):
        if k.startswith("env."):
            recipe.env[k[4:]] = str(v)
            recipe._cli_env_keys.add(k[4:])
            del overrides[k]

    # Resolve runtime with overrides visible to resolvers
    recipe.resolve(overrides)

    return recipe, overrides


def apply_env_overrides(recipe, env_pairs) -> dict[str, str]:
    """Apply ``KEY=VALUE`` container-env overrides to *recipe*.

    The peer of ``-o env.KEY=VALUE`` (handled inside
    :func:`apply_recipe_overrides`) for the direct ``-e/--env`` form, with one
    deliberate difference: values are taken **verbatim**. The ``-o`` path routes
    through ``coerce_value``, which turns ``-o env.X=true`` into the string
    ``"True"`` — right for a serve flag, wrong for an env var a runtime parses
    itself.

    Only the first ``=`` splits, so ``FOO=a=b`` and
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` survive intact, and
    ``FOO=`` sets an empty value (the way to blank out a lower-tier default).

    Pure logic: no console I/O. Malformed entries raise ``ValueError``; the CLI
    wrapper translates that into a ``UsageError``.

    Returns the mapping that was applied (for logging / testing).
    """
    applied: dict[str, str] = {}
    for pair in env_pairs or ():
        if "=" not in pair:
            raise ValueError("--env must be KEY=VALUE, got: %s" % pair)
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError("--env has empty key: %s" % pair)
        applied[key] = value
    if recipe is not None and applied:
        recipe.env.update(applied)
        # Matched `overrides:` env layers must not replace what -e set.
        recipe._cli_env_keys.update(applied)
    return applied


def load_recipe(
    config: "SparkrunConfig",
    recipe_name: str,
    *,
    resolve: bool = True,
):
    """Find and load a recipe without any interactive prompts or exits.

    Mirrors the non-interactive happy path of
    ``cli/_common.py:_load_recipe``: expands shortcuts, fetches URL-sourced
    recipes (refusing off-allowlist hosts rather than prompting), looks up
    the recipe across configured registries, and tags it with its source
    registry.

    Returns ``(recipe, recipe_path, registry_mgr)``.

    Raises:
        RecipeUntrustedHostError: URL host is off the allowlist (the CLI
            wrapper offers an interactive override; the API surfaces it).
        RecipeAmbiguousError: Name matches multiple registries (the CLI
            wrapper prompts; the API surfaces it).
        RecipeError: Recipe not found or failed to load.
    """
    from sparkrun.core.recipe import (
        Recipe,
        discover_cwd_recipes,
        expand_recipe_shortcut,
        fetch_and_cache_recipe,
        find_recipe,
        is_recipe_url,
    )

    from sparkrun.core._recipe_source import recipe_registry_entry, tag_recipe_source
    from sparkrun.utils import parse_scoped_name

    recipe_name = expand_recipe_shortcut(recipe_name)

    if is_recipe_url(recipe_name):
        logger.debug("Loading recipe from URL: %s", recipe_name)
        cached_path = fetch_and_cache_recipe(recipe_name)
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        recipe = Recipe.load(cached_path, resolve=resolve, registry_manager=registry_mgr, allow_local_includes=False)
        recipe.source_path = recipe_name
        # URL-sourced recipes are never auto-trusted (see
        # core.launcher.resolve_recipe_trust): their hooks require --trust
        # or interactive confirmation.
        tag_recipe_source(recipe, None, config=config, external=True)
        return recipe, cached_path, registry_mgr

    registry_mgr = config.get_registry_manager()
    registry_mgr.ensure_initialized()

    recipe_path = find_recipe(recipe_name, registry_manager=registry_mgr, local_files=discover_cwd_recipes())
    recipe = Recipe.load(recipe_path, resolve=resolve, registry_manager=registry_mgr)

    scope, _ = parse_scoped_name(recipe_name)
    registry = recipe_registry_entry(recipe_path, registry_mgr, registry_name=scope)
    tag_recipe_source(recipe, registry, config=config)
    return recipe, recipe_path, registry_mgr


__all__ = ["apply_recipe_overrides", "load_recipe"]
