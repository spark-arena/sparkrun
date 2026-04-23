"""sparkrun registry group and subcommands."""

from __future__ import annotations

import sys

import click
import yaml

from ._common import (
    PROFILE_NAME,
    REGISTRY_NAME,
    _get_config_and_registry,
    json_option,
    print_json,
    HIDE_ADVANCED_OPTIONS,
)
from sparkrun.utils.cli_formatters import RUNTIME_DISPLAY as _RUNTIME_DISPLAY


@click.group()
@click.pass_context
def registry(ctx):
    """Manage recipe registries."""
    pass


@registry.command("list")
@click.option("--show-disabled", is_flag=True, help="Also show disabled registries")
@click.option("--only-show-visible", is_flag=True, help="Only show visible registries")
@json_option()
@click.pass_context
def registry_list(ctx, show_disabled, only_show_visible, output_json, config_path=None):
    """List configured recipe registries.

    By default, shows all enabled registries (including hidden ones).
    """
    config, registry_mgr = _get_config_and_registry(config_path)
    registries = registry_mgr.list_registries()

    if not registries:
        if output_json:
            print_json([])
            return
        click.echo("No registries configured.")
        return

    # Default: all enabled registries
    display_registries = registries
    if not show_disabled:
        display_registries = [r for r in display_registries if r.enabled]
    if only_show_visible:
        display_registries = [r for r in display_registries if r.visible]

    # Sort: enabled first, then visible first, then by name
    display_registries.sort(key=lambda r: (not r.enabled, not r.visible, r.name))

    if output_json:
        print_json(display_registries)
        return

    if not display_registries:
        click.echo("No matching registries found.")
        return

    # Determine which content columns to show
    has_tuning = any(r.tuning_subpath for r in display_registries)
    has_benchmarks = any(r.benchmark_subpath for r in display_registries)

    # Table header
    header = f"{'Name':<25} {'URL':<45} {'Enabled':<9} {'Visible':<9}"
    sep_width = 88
    if has_tuning:
        header += f" {'Tuning':<8}"
        sep_width += 9
    if has_benchmarks:
        header += f" {'Bench':<7}"
        sep_width += 8
    click.echo(header)
    click.echo("-" * sep_width)

    for reg in display_registries:
        url = reg.url[:43] + ".." if len(reg.url) > 45 else reg.url
        enabled = "yes" if reg.enabled else "no"
        visible = "yes" if reg.visible else "no"
        row = f"{reg.name:<25} {url:<45} {enabled:<9} {visible:<9}"
        if has_tuning:
            tuning = "yes" if reg.tuning_subpath else "no"
            row += f" {tuning:<8}"
        if has_benchmarks:
            bench = "yes" if reg.benchmark_subpath else "no"
            row += f" {bench:<7}"
        click.echo(row)


# @registry.command("manual-add")
# @click.argument("name")
# @click.option("--url", required=True, help="Git repository URL")
# @click.option("--subpath", required=True, help="Path to recipes within repo")
# @click.option("-d", "--description", default="", help="Registry description")
# @click.option("--visible/--hidden", default=True, help="Registry visibility in default listings")
# @click.option("--tuning-subpath", default="", help="Path to tuning configs within repo")
# @click.option("--benchmark-subpath", default="", help="Path to benchmark profiles within repo")
# @click.pass_context
# def registry_add(ctx, name, url, subpath, description, visible, tuning_subpath, benchmark_subpath, config_path=None):
#     """Add a new recipe registry."""
#     from sparkrun.registry import RegistryEntry, RegistryError
#
#     config, registry_mgr = _get_config_and_registry(config_path)
#
#     try:
#         entry = RegistryEntry(
#             name=name,
#             url=url,
#             subpath=subpath,
#             description=description,
#             enabled=True,
#             visible=visible,
#             tuning_subpath=tuning_subpath,
#             benchmark_subpath=benchmark_subpath,
#         )
#         registry_mgr.add_registry(entry)
#         click.echo(f"Registry '{name}' added successfully.")
#     except RegistryError as e:
#         click.echo(f"Error: {e}", err=True)
#         sys.exit(1)


@registry.command("add")
@click.argument("url")
@click.pass_context
def registry_add_url(ctx, url, config_path=None):
    """Add registries from a repository's .sparkrun/registry.yaml manifest.

    Clones the repository, reads the manifest, and adds all declared registries.

    Examples:

      sparkrun registry add https://github.com/spark-arena/recipe-registry
    """
    from sparkrun.core.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        click.echo("Discovering registries from %s..." % url)
        added = registry_mgr.add_registry_from_url(url)
        if added:
            click.echo("Added %d registr%s:" % (len(added), "y" if len(added) == 1 else "ies"))
            for entry in added:
                vis = "" if entry.visible else " (hidden)"
                click.echo("  %s — %s%s" % (entry.name, entry.description, vis))
        else:
            click.echo("No new registries added (all may already exist).")
    except RegistryError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)


@registry.command("remove")
@click.argument("name", type=REGISTRY_NAME)
@click.pass_context
def registry_remove(ctx, name, config_path=None):
    """Remove a recipe registry."""
    from sparkrun.core.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.remove_registry(name)
        click.echo(f"Registry '{name}' removed successfully.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@registry.command("enable")
@click.argument("name", type=REGISTRY_NAME)
@click.pass_context
def registry_enable(ctx, name, config_path=None):
    """Enable a disabled registry."""
    from sparkrun.core.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.enable_registry(name)
        click.echo(f"Registry '{name}' enabled.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@registry.command("disable")
@click.argument("name", type=REGISTRY_NAME)
@click.pass_context
def registry_disable(ctx, name, config_path=None):
    """Disable a registry (recipes will not appear in searches)."""
    from sparkrun.core.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.disable_registry(name)
        click.echo(f"Registry '{name}' disabled.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@registry.command("revert-to-defaults", hidden=HIDE_ADVANCED_OPTIONS)
@click.option("--no-update", "no_run_update", is_flag=True, help="Do not run registry update after reset")
@click.pass_context
def registry_revert_to_default(ctx, no_run_update, config_path=None):
    """Reset registries to defaults (deletes config and re-initializes).

    Removes the current registries.yaml and re-discovers registries from
    the default manifest URLs.  If discovery fails (offline, etc.), falls
    back to hardcoded defaults.

    Examples:

      sparkrun registry revert-to-default

      sparkrun registry revert-to-default --update
    """
    config, registry_mgr = _get_config_and_registry(config_path)

    entries = registry_mgr.reset_to_defaults()
    click.echo("Registries reset to defaults (%d entries):" % len(entries))
    for entry in entries:
        vis = "" if entry.visible else " (hidden)"
        click.echo("  %s — %s%s" % (entry.name, entry.description or entry.url, vis))

    if not no_run_update:
        click.echo()
        ctx.invoke(registry_update)


@registry.command("update")
@click.argument("name", required=False, default=None, type=REGISTRY_NAME)
@click.pass_context
def registry_update(ctx, name, config_path=None):
    """Update recipe registries from git.

    If NAME is given, update only that registry. Otherwise update all enabled registries.
    """
    from sparkrun.core.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        # Clean up deprecated registries before updating
        if not name:
            cleaned = registry_mgr.cleanup_deprecated()
            for cname in cleaned:
                click.echo("Removed deprecated registry: %s" % cname)

            # Restore any missing default registries
            restored = registry_mgr.restore_missing_defaults()
            for rname in restored:
                click.echo("Added missing default registry: %s" % rname)

        if name:
            entry = registry_mgr.get_registry(name)
            if not entry.enabled:
                click.echo(
                    f"Error: Registry '{name}' is disabled; enable it before updating.",
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

        def _progress(prog_name: str, success: bool) -> None:
            status = "done" if success else "FAILED"
            click.echo(f"  Updating {prog_name}... {status}")

        results = registry_mgr.update(name, progress=_progress)
        succeeded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)

        if failed:
            click.echo(f"{succeeded} of {count} registries updated ({failed} failed).")
        else:
            click.echo(f"{succeeded} registr{'y' if succeeded == 1 else 'ies'} updated.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _build_raw_url(repo_url: str, subpath: str, rel_path: str) -> str:
    """Build a raw GitHub URL from a registry's git URL, subpath, and relative file path.

    ``rel_path`` is the file path relative to the recipe directory (``subpath``),
    which may include nested subdirectories discovered via rglob.
    """
    import re

    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    m = re.match(r"https?://github\.com/([^/]+/[^/]+)", url)
    if m:
        return "https://raw.githubusercontent.com/%s/main/%s/%s" % (m.group(1), subpath, rel_path)
    return ""


def _format_param_count(value) -> str | None:
    """Format a parameter count as a human-readable string (e.g. '1.7B', '480M')."""
    if value is None:
        return None
    if isinstance(value, str):
        # Already formatted (e.g. "1.7B") — pass through
        if any(value.upper().endswith(s) for s in ("B", "M", "K", "T")):
            return value
        # Raw numeric string — parse it
        try:
            value = int(float(value.replace("_", "")))
        except (ValueError, TypeError):
            return value
    if not isinstance(value, (int, float)):
        return str(value)
    n = int(value)
    if n >= 1_000_000_000:
        v = n / 1_000_000_000
        return ("%.0fB" if v == int(v) else "%.1fB") % v
    if n >= 1_000_000:
        v = n / 1_000_000
        return ("%.0fM" if v == int(v) else "%.1fM") % v
    return str(n)


@registry.command("export-metadata", hidden=True)
@click.option("--output", "-o", type=click.Path(), default="recipes.json", help="Output path for the JSON manifest")
@click.option("--include-hidden", is_flag=True, help="Include recipes from hidden registries")
@click.pass_context
def export_metadata(ctx, output, include_hidden):
    """Export recipe metadata manifest as JSON"""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from vpd.next.util import read_yaml
    from sparkrun.core.recipe import Recipe
    from sparkrun.models.download import parse_gguf_model_spec

    config, registry_mgr = _get_config_and_registry()
    registry_mgr.ensure_initialized()

    registries = registry_mgr.list_registries()
    registry_meta = []
    all_recipes = []
    all_runtimes = set()
    seen_slugs = {}  # slug -> source_path, for duplicate detection

    for entry in registries:
        if not entry.enabled:
            continue
        if not include_hidden and not entry.visible:
            continue

        recipe_dir = registry_mgr._recipe_dir(entry)
        if not recipe_dir or not recipe_dir.is_dir():
            continue

        assert recipe_dir is not None
        recipe_count = 0
        for f in sorted(recipe_dir.rglob("*.yaml"), key=lambda p, rd=recipe_dir: (len(p.relative_to(rd).parts), p.name)):
            try:
                data = read_yaml(str(f))
                if not isinstance(data, dict):
                    continue
                recipe = Recipe(data, source_path=str(f))
                recipe.resolve()
            except Exception as e:
                click.echo("  Warning: skipping %s: %s" % (f.name, e), err=True)
                continue

            # Run VRAM estimation to auto-detect model_params and model_dtype
            # from HuggingFace when not present in recipe metadata.
            try:
                recipe.estimate_vram(auto_detect=True)
            except Exception:
                pass  # best-effort — metadata fields may remain None

            rel_path = f.relative_to(recipe_dir)
            raw_url = _build_raw_url(entry.url, entry.subpath, str(rel_path))
            tp_val = recipe.defaults.get("tensor_parallel")
            gpu_mem_val = recipe.defaults.get("gpu_memory_utilization")
            port_val = recipe.defaults.get("port")

            display_runtime = _RUNTIME_DISPLAY.get(recipe.runtime, recipe.runtime)

            # Expose the clustering backend (ray/torch/rpc/mpi)
            cluster_backend = None
            if recipe.runtime == "vllm-ray":
                cluster_backend = "ray"
            elif recipe.runtime in ("vllm-distributed", "sglang"):
                cluster_backend = "torch"
            elif recipe.runtime == "llama-cpp":
                cluster_backend = "rpc"
            elif recipe.runtime == "trtllm":
                cluster_backend = "mpi"

            # TODO: validate recipe against registry recipe -- to make sure that the recipe metadata
            #       spark_arena_uuid is not spoofed or fake ?? really only applies if recipes come from external sources

            slug = "%s/%s" % (entry.name, recipe.name)
            if slug in seen_slugs:
                click.echo("  Warning: duplicate recipe slug '%s' — keeping %s, skipping %s" % (slug, seen_slugs[slug], f), err=True)
                continue
            seen_slugs[slug] = str(f)

            all_recipes.append(
                {
                    "slug": slug,
                    "name": recipe.name,
                    "registry": entry.name,
                    "model": parse_gguf_model_spec(recipe.model)[0],
                    "model_full": recipe.model,
                    "runtime": display_runtime,
                    "cluster_backend": cluster_backend,
                    "description": recipe.description,
                    "model_params": _format_param_count(recipe.metadata.get("model_params")),
                    "model_dtype": str(recipe.metadata["model_dtype"]) if recipe.metadata.get("model_dtype") else None,
                    "category": recipe.metadata.get("category"),
                    "maintainer": recipe.maintainer,
                    "min_nodes": recipe.min_nodes,
                    "max_nodes": recipe.max_nodes,
                    "mode": recipe.mode,
                    "tp": int(tp_val) if tp_val is not None else 1,
                    "gpu_mem": float(gpu_mem_val) if gpu_mem_val is not None else 0.9,
                    "port": int(port_val) if port_val is not None else 8000,
                    "container": recipe.container,  # TODO: should we replace w/ first-party if meets requirements?
                    "recipe_version": recipe.recipe_version,
                    "raw_url": raw_url,
                    "spark_arena_benchmarks": [
                        {"tp": b["tp"], "uuid": b["uuid"], "url": "https://spark-arena.com/benchmark/%s" % b["uuid"]}
                        for b in recipe.metadata.get("spark_arena_benchmarks", [])
                    ]
                    or None,
                }
            )
            all_runtimes.add(display_runtime)
            recipe_count += 1

        registry_meta.append(
            {
                "name": entry.name,
                "url": entry.url,
                "description": entry.description,
                "recipe_count": recipe_count,
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registries": registry_meta,
        "runtimes": sorted(all_runtimes),
        "recipes": all_recipes,
    }

    Path(output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    click.echo("Exported %d recipes from %d registries to %s" % (len(all_recipes), len(registry_meta), output))


# ---------------------------------------------------------------------------
# Benchmark profile subcommands
# ---------------------------------------------------------------------------


@registry.command("list-benchmark-profiles")
@click.option("--all", "-a", "show_all", is_flag=True, default=False, help="Include profiles from hidden registries")
@click.option("--registry", "registry_name", default=None, type=REGISTRY_NAME, help="Filter by registry name")
@click.pass_context
def list_benchmark_profiles(ctx, show_all, registry_name, config_path=None):
    """List available benchmark profiles across registries."""
    from sparkrun.core.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    # Validate registry name upfront so a typo gives a clear error
    if registry_name:
        try:
            registry_mgr.get_registry(registry_name)
        except RegistryError:
            available = [r.name for r in registry_mgr.list_registries() if r.enabled]
            click.echo(
                "Error: registry '%s' not found. Available: %s" % (registry_name, ", ".join(available) if available else "(none)"),
                err=True,
            )
            sys.exit(1)

    profiles = registry_mgr.list_benchmark_profiles(
        registry_name=registry_name,
        include_hidden=show_all,
    )

    if not profiles:
        click.echo("No benchmark profiles found.")
        return

    click.echo(f"{'Profile':<30} {'Registry':<25} {'Framework':<15}")
    click.echo("-" * 70)
    for p in profiles:
        click.echo(f"{p['file']:<30} {p['registry']:<25} {p.get('framework', 'n/a'):<15}")


@registry.command("show-benchmark-profile")
@click.argument("profile_name", type=PROFILE_NAME)
@click.pass_context
def show_benchmark_profile(ctx, profile_name, config_path=None):
    """Show detailed benchmark profile information."""
    from ..core.benchmark_profiles import find_benchmark_profile
    from ..core.benchmark_profiles import ProfileAmbiguousError
    from ..core.benchmark_profiles import ProfileError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        profile_path = find_benchmark_profile(profile_name, config, registry_mgr)
    except ProfileAmbiguousError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except ProfileError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    click.echo("Profile: %s" % profile_name)
    click.echo("Path:    %s" % profile_path)
    click.echo("")
    click.echo(yaml.safe_dump(yaml.safe_load(profile_path.read_text()), default_flow_style=False))
