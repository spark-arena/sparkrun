"""sparkrun recipe group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    RECIPE_QUERY,
    REGISTRY_NAME,
    RUNTIME_NAME,
    _display_recipe_detail,
    _display_vram_estimate,
    _get_config_and_registry,
    _get_context,
    _load_recipe,
    json_option,
    print_json,
)
from sparkrun.core.validation import FAIL_ON_CHOICES
from sparkrun.utils.cli_formatters import format_validation_report


@click.group()
@click.pass_context
def recipe(ctx):
    """Find and manage inference recipes."""
    pass


def _search_recipes(ctx, query, *, registry, runtime, show_all, unique_names=False):
    """CLI adapter around :func:`sparkrun.api.search_recipes`.

    Renders the api's typed filter error as a Click usage error, and hands
    back the recipe-summary mappings the formatters and ``--json`` consume
    alongside the resolved ``(registry, query)`` so callers can phrase an
    empty-result message.
    """
    from sparkrun import api

    sctx = _get_context(ctx)
    try:
        # Resolved separately so the command can name *which* registry came
        # back empty; search_recipes resolves again internally, which keeps it
        # safe to call without any pre-resolution.
        scoped_registry, remaining_query = api.resolve_recipe_filter(query, registry=registry, sctx=sctx)
        recipes = api.search_recipes(
            query,
            registry=registry,
            runtime=runtime,
            include_hidden=show_all,
            unique_names=unique_names,
            sctx=sctx,
        )
    except api.InvalidRegistryFilter as e:
        raise click.UsageError(str(e)) from e

    return [r.to_dict() for r in recipes], scoped_registry, remaining_query


@recipe.command("list")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.option("--all", "-a", "show_all", is_flag=True, help="Include hidden registry recipes")
@json_option()
@click.argument("query", type=RECIPE_QUERY, required=False)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_list(ctx, registry, runtime, show_all, output_json, query, config_path=None):
    """List available recipes from all registries.

    QUERY may be scoped to a registry with the ``@registry`` shorthand:
    ``@community`` (equivalent to ``--registry community``) or
    ``@community/qwen`` (that registry, searching for ``qwen``).
    """
    from sparkrun.utils.cli_formatters import format_recipe_table

    # unique_names: `list` is the "what can I type?" view, so a name resolves
    # to exactly one row.  `search` shows every registry's copy instead.
    recipes, _registry, _query = _search_recipes(ctx, query, registry=registry, runtime=runtime, show_all=show_all, unique_names=True)

    if output_json:
        print_json(recipes)
        return

    click.echo(format_recipe_table(recipes, show_model=True))


@recipe.command("search")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.option("--all", "-a", "show_all", is_flag=True, help="Include hidden registry recipes")
@json_option()
@click.argument("query", type=RECIPE_QUERY)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_search(ctx, registry, runtime, show_all, output_json, query, config_path=None):
    """Search for recipes by name, model, or description.

    QUERY may be scoped to a registry with the ``@registry`` shorthand:
    ``@community`` (every recipe in that registry) or ``@community/qwen``
    (that registry, searching for ``qwen``).
    """
    from sparkrun.utils.cli_formatters import format_recipe_table

    recipes, scoped_registry, remaining_query = _search_recipes(ctx, query, registry=registry, runtime=runtime, show_all=show_all)

    if output_json:
        print_json(recipes)
        return

    if not recipes:
        # A bare "@registry" scope consumes the whole query, leaving nothing
        # to quote back — name the registry instead.
        if remaining_query:
            click.echo(f"No recipes found matching '{remaining_query}'.")
        else:
            click.echo(f"No recipes found in registry '{scoped_registry}'.")
        return

    click.echo(format_recipe_table(recipes, show_model=True))


@recipe.command("show")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None, help="Override tensor parallelism")
@click.option("--gpu-mem", type=float, default=None, help="Override GPU memory utilization (0.0-1.0)")
@json_option()
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_show(ctx, recipe_name, no_vram, tensor_parallel, gpu_mem, output_json, config_path=None):
    """Show detailed recipe information."""

    config, _ = _get_config_and_registry(config_path)
    recipe, recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    cli_overrides = {}
    if tensor_parallel is not None:
        cli_overrides["tensor_parallel"] = tensor_parallel
    if gpu_mem is not None:
        cli_overrides["gpu_memory_utilization"] = gpu_mem

    if output_json:
        print_json(recipe.to_dict(overrides=cli_overrides))
        return

    reg_name = registry_mgr.registry_for_path(recipe_path) if registry_mgr else None
    _display_recipe_detail(recipe, show_vram=not no_vram, registry_name=reg_name, cli_overrides=cli_overrides or None)

    return


@recipe.command("validate")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--strict", is_flag=True, help="Also fail on warnings (exit 1)")
@click.option(
    "--fail-on",
    type=click.Choice(FAIL_ON_CHOICES),
    default=None,
    hidden=True,
    help="Least-severe finding that fails (default: error; --strict means warning)",
)
@json_option()
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_validate(ctx, recipe_name, strict, fail_on, output_json, config_path=None):
    """Validate a recipe file.

    Findings come in three severities. ERRORS are things sparkrun cannot honor
    (a missing field, a runtime that rejects the recipe, a builder or executor
    that does not resolve) and always exit 1. WARNINGS mean the recipe runs but
    breaks or behaves differently off the cluster it was written on (NCCL pinned
    to one machine's devices, a bind mount only the author has). SUGGESTIONS
    mean it works as written and merely gives something up, like a serve flag
    hardcoded where the config chain cannot see it.

    Only errors are fatal by default. Pass --strict to fail on warnings too —
    the useful setting for registry CI.
    """
    from sparkrun.core.validation import ERROR, SUGGESTION, WARNING, should_fail, validate_recipe

    sctx = _get_context(ctx)
    v = sctx.variables
    config, _ = _get_config_and_registry(config_path)
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    # --strict is the typeable spelling of the one threshold most people want;
    # --fail-on is the full ladder, hidden because it is an advanced knob.
    # Explicit --fail-on wins over --strict, and both beat validation.fail_on.
    threshold = fail_on or (WARNING if strict else config.validation_fail_on)

    issues = validate_recipe(recipe, config=config, v=v)
    counts = {level: sum(1 for i in issues if i.severity == level) for level in (ERROR, WARNING, SUGGESTION)}
    failed = should_fail(issues, threshold)

    if output_json:
        print_json(
            {
                "recipe": recipe.qualified_name,
                # `valid` is a property of the recipe (no errors) and stays
                # that way whatever bar this invocation set; `failed` is the
                # exit code, i.e. this invocation's bar.
                "valid": counts[ERROR] == 0,
                "failed": failed,
                "fail_on": threshold,
                "errors": counts[ERROR],
                "warnings": counts[WARNING],
                "suggestions": counts[SUGGESTION],
                "issues": [i.to_dict() for i in issues],
            }
        )
        if failed:
            sys.exit(1)
        return

    if not issues:
        click.echo(f"Recipe '{recipe.qualified_name}' is valid.")
        return

    click.echo(format_validation_report(recipe.qualified_name, issues))
    if not failed:
        if counts[WARNING]:
            click.echo("\nNothing here blocks a launch. Use --strict to fail on warnings.")
        else:
            click.echo("\nSuggestions do not block a launch.")

    if failed:
        sys.exit(1)


@recipe.command("vram")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None, help="Override tensor parallelism")
@click.option("--max-model-len", type=int, default=None, help="Override max sequence length")
@click.option("--gpu-mem", type=float, default=None, help="Override gpu_memory_utilization (0.0-1.0)")
@click.option("--no-auto-detect", is_flag=True, help="Skip HuggingFace model auto-detection")
@json_option()
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_vram(ctx, recipe_name, tensor_parallel, max_model_len, gpu_mem, no_auto_detect, output_json, config_path=None):
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

    if not output_json:
        click.echo(f"Recipe:  {recipe.qualified_name}")
        click.echo(f"Model:   {recipe.model}")
        click.echo(f"Runtime: {recipe.runtime}")

    cli_overrides = {}
    if tensor_parallel is not None:
        cli_overrides["tensor_parallel"] = tensor_parallel
    if max_model_len is not None:
        cli_overrides["max_model_len"] = max_model_len
    if gpu_mem is not None:
        cli_overrides["gpu_memory_utilization"] = gpu_mem

    if output_json:
        est = recipe.estimate_vram(cli_overrides=cli_overrides, auto_detect=not no_auto_detect)
        print_json({"recipe": recipe.qualified_name, "model": recipe.model, "runtime": recipe.runtime, **est.to_dict()})
    else:
        _display_vram_estimate(recipe, cli_overrides=cli_overrides, auto_detect=not no_auto_detect)


@recipe.command("update", hidden=True)
@click.option("--registry", default=None, help="Update specific registry")
@click.pass_context
def recipe_update(ctx, registry):
    """Update recipe registries from git."""
    click.echo("Warning: 'sparkrun recipe update' is deprecated. Use 'sparkrun registry update' or 'sparkrun update' instead.", err=True)
    from sparkrun.cli._registry import registry_update

    ctx.invoke(registry_update, name=registry)
