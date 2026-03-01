"""sparkrun recipe group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    REGISTRY_NAME,
    RUNTIME_NAME,
    _display_recipe_detail,
    _display_vram_estimate,
    _get_config_and_registry,
    _load_recipe,
)


@click.group()
@click.pass_context
def recipe(ctx):
    """Manage recipe registries and search for recipes."""
    pass


@recipe.command("list")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.option("--all", "-a", "show_all", is_flag=True, help="Include hidden registry recipes")
@click.argument("query", required=False)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_list(ctx, registry, runtime, show_all, query, config_path=None):
    """List available recipes from all registries."""
    from sparkrun.core.recipe import list_recipes, filter_recipes
    from sparkrun.utils.cli_formatters import format_recipe_table

    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    # When a specific registry is requested, include hidden registries so the
    # user can list recipes from a non-visible registry without needing --all.
    include_hidden = show_all or (registry is not None)

    if query:
        recipes = registry_mgr.search_recipes(query, include_hidden=include_hidden)
    else:
        from sparkrun.core.recipe import discover_cwd_recipes
        recipes = list_recipes(registry_manager=registry_mgr, include_hidden=include_hidden,
                               local_files=discover_cwd_recipes())

    recipes = filter_recipes(recipes, runtime=runtime, registry=registry)
    click.echo(format_recipe_table(recipes, show_model=True))


@recipe.command("search")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.option("--all", "-a", "show_all", is_flag=True, help="Include hidden registry recipes")
@click.argument("query")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_search(ctx, registry, runtime, show_all, query, config_path=None):
    """Search for recipes by name, model, or description."""
    from sparkrun.core.recipe import filter_recipes
    from sparkrun.utils.cli_formatters import format_recipe_table

    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    include_hidden = show_all or (registry is not None)
    recipes = registry_mgr.search_recipes(query, include_hidden=include_hidden)
    recipes = filter_recipes(recipes, runtime=runtime, registry=registry)

    if not recipes:
        click.echo(f"No recipes found matching '{query}'.")
        return

    click.echo(format_recipe_table(recipes, show_model=True))


@recipe.command("show")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory for model lookups")
@click.option("--save", "save_path", default=None, type=click.Path(),
              help="Save a copy of the recipe YAML to a file")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_show(ctx, recipe_name, no_vram, tensor_parallel, cache_dir=None, save_path=None, config_path=None):
    """Show detailed recipe information."""
    import shutil

    config, _ = _get_config_and_registry(config_path)
    recipe, recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    cli_overrides = {}
    if tensor_parallel is not None:
        cli_overrides["tensor_parallel"] = tensor_parallel

    reg_name = registry_mgr.registry_for_path(recipe_path) if registry_mgr else None
    _display_recipe_detail(recipe, show_vram=not no_vram, registry_name=reg_name,
                           cli_overrides=cli_overrides or None, cache_dir=cache_dir)

    if save_path:
        from pathlib import Path
        dest = Path(save_path)
        shutil.copy2(recipe_path, dest)
        click.echo("\nRecipe saved to %s" % dest)


@recipe.command("validate")
@click.argument("recipe_name", type=RECIPE_NAME)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_validate(ctx, recipe_name, config_path=None):
    """Validate a recipe file."""
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime

    v = init_sparkrun()
    config, _ = _get_config_and_registry(config_path)
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    issues = recipe.validate()

    try:
        runtime = get_runtime(recipe.runtime, v)
        issues.extend(runtime.validate_recipe(recipe))
    except ValueError:
        issues.append(f"Unknown runtime: {recipe.runtime}")

    if issues:
        click.echo(f"Recipe '{recipe.name}' has {len(issues)} issue(s):")
        for issue in issues:
            click.echo(f"  - {issue}")
        sys.exit(1)
    else:
        click.echo(f"Recipe '{recipe.name}' is valid.")


@recipe.command("vram")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.option("--max-model-len", type=int, default=None, help="Override max sequence length")
@click.option("--gpu-mem", type=float, default=None,
              help="Override gpu_memory_utilization (0.0-1.0)")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory for model lookups")
@click.option("--no-auto-detect", is_flag=True, help="Skip HuggingFace model auto-detection")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_vram(ctx, recipe_name, tensor_parallel, max_model_len, gpu_mem, cache_dir=None, no_auto_detect=False, config_path=None):
    """Estimate VRAM usage for a recipe on DGX Spark.

    Shows model weight size, KV cache requirements, GPU memory budget,
    and whether the configuration fits within DGX Spark memory.

    Examples:

      sparkrun recipe vram glm-4.7-flash-awq

      sparkrun recipe vram glm-4.7-flash-awq --tp 2

      sparkrun recipe vram my-recipe.yaml --max-model-len 8192 --gpu-mem 0.9
    """
    config, _ = _get_config_and_registry(config_path)
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    click.echo(f"Recipe:  {recipe.name}")
    click.echo(f"Model:   {recipe.model}")
    click.echo(f"Runtime: {recipe.runtime}")

    cli_overrides = {}
    if tensor_parallel is not None:
        cli_overrides["tensor_parallel"] = tensor_parallel
    if max_model_len is not None:
        cli_overrides["max_model_len"] = max_model_len
    if gpu_mem is not None:
        cli_overrides["gpu_memory_utilization"] = gpu_mem

    _display_vram_estimate(recipe, cli_overrides=cli_overrides, auto_detect=not no_auto_detect, cache_dir=cache_dir)


@recipe.command("update", hidden=True)
@click.option("--registry", default=None, help="Update specific registry")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_update(ctx, registry, config_path=None, ):
    """Update recipe registries from git."""
    click.echo("Warning: 'sparkrun recipe update' is deprecated. Use 'sparkrun registry update' instead.", err=True)
    from sparkrun.core.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        # Count how many registries will be updated
        if registry:
            entry = registry_mgr.get_registry(registry)
            if not getattr(entry, "enabled", True):
                click.echo(
                    f"Error: Registry '{registry}' is disabled; enable it in the config before updating.",
                    err=True,
                )
                sys.exit(1)
            entries = [entry]
        else:
            entries = [e for e in registry_mgr.list_registries() if e.enabled]

        count = len(entries)
        if count == 0:
            click.echo("No enabled registries to update.")
            return

        click.echo(f"Updating {count} registr{'y' if count == 1 else 'ies'}...")

        def _progress(name: str, success: bool) -> None:
            status = "done" if success else "FAILED"
            click.echo(f"  Updating {name}... {status}")

        results = registry_mgr.update(registry, progress=_progress)
        succeeded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)

        if failed:
            click.echo(f"{succeeded} of {count} registries updated ({failed} failed).")
        else:
            click.echo(f"{succeeded} registr{'y' if succeeded == 1 else 'ies'} updated.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
