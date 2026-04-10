"""Shared CLI infrastructure: utilities, Click types, decorators."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

from scitrera_app_framework.util import ext_parse_bool
from sparkrun.core.recipe import (
    expand_recipe_shortcut as _expand_recipe_shortcut,
    fetch_and_cache_recipe as _fetch_and_cache_recipe,
    is_recipe_url as _is_recipe_url,
    simplify_recipe_ref as _simplify_recipe_ref,  # noqa: F401 — re-exported for cli/_run.py, cli/_benchmark.py
)
from sparkrun.core.cluster_manager import ResolvedClusterConfig, resolve_cluster_config  # noqa: E402, F401 — re-exported

HIDE_ADVANCED_OPTIONS = not ext_parse_bool(os.environ.get("SPARKRUN_ADVANCED", "0"))

logger = logging.getLogger(__name__)


# noinspection PyShadowingBuiltins
def json_option(help: str = None):
    return click.option(
        "--json",
        "output_json",
        is_flag=True,
        default=False,
        help=help or "Output result as JSON",
    )


def print_json(data: Any) -> None:
    """Print an object as formatted JSON.

    Automatically handles dataclasses and objects implementing `to_dict()`.
    """
    from sparkrun.utils.json_helpers import dumps_json

    click.echo(dumps_json(data))


def _get_context(ctx) -> "SparkrunContext":
    """Lazily create and cache a :class:`SparkrunContext` on the Click context.

    Calls ``init_sparkrun()`` and creates a ``SparkrunConfig``, bundling
    them into a single context object stored in ``ctx.obj["sparkrun_ctx"]``.

    Logging is *not* re-applied here — ``_setup_logging()`` is already
    called once from the ``main()`` group callback, and SAF's
    ``fixed_logger`` parameter means ``init_framework_desktop`` skips
    its own logging setup entirely.
    """
    from sparkrun.core.context import SparkrunContext

    obj = ctx.ensure_object(dict)
    sctx = obj.get("sparkrun_ctx")
    if sctx is not None:
        return sctx

    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import SparkrunConfig

    v = init_sparkrun()
    config_path = obj.get("config_path")
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    from sparkrun.core.progress import LaunchProgress, Verbosity

    verbose_count = obj.get("verbose", 0)
    # Backward compat: bool True → 1
    if isinstance(verbose_count, bool):
        verbose_count = 1 if verbose_count else 0
    progress = LaunchProgress(verbosity=Verbosity(min(verbose_count, Verbosity.DEBUG)))

    sctx = SparkrunContext(
        variables=v,
        config=config,
        verbose=verbose_count > 0,
        progress=progress,
    )
    obj["sparkrun_ctx"] = sctx
    return sctx


def _setup_logging(verbose: int | bool):
    """Configure logging based on verbosity.

    Called once from the ``main()`` Click group callback.  No re-call
    is needed after ``init_sparkrun()`` because sparkrun passes
    ``fixed_logger`` to SAF's ``init_framework_desktop``, which skips
    SAF's own logging setup entirely (see SAF ``core.py:376``).

    Verbosity tiers::

        0 (default)  → PROGRESS (25): phase/step output only
        1 (-v)       → INFO (20): adds detail lines
        2 (-vv)      → VERBOSE (15): adds timestamps + logger names
        3+ (-vvv)    → DEBUG (10): full SSH/script diagnostics

    Uses explicit handler setup instead of ``logging.basicConfig`` which
    is silently a no-op when the root logger already has handlers (common
    when libraries like ``huggingface_hub`` configure logging on import).
    """
    from sparkrun.core.progress import PROGRESS, VERBOSE

    # Backward compat: bool True → 1, False → 0
    if isinstance(verbose, bool):
        verbose = 1 if verbose else 0

    if verbose < 0:
        level = logging.WARNING  # --quiet: errors/warnings only
        fmt = "%(message)s"
    elif verbose >= 3:
        level = logging.DEBUG
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    elif verbose >= 2:
        level = VERBOSE
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    elif verbose >= 1:
        level = logging.INFO
        fmt = "%(message)s"
    else:
        level = PROGRESS
        fmt = "%(message)s"

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers that may have been added by library imports
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)

    from sparkrun.utils import suppress_noisy_loggers

    suppress_noisy_loggers()

    return


def _parse_options(options: tuple[str, ...]) -> dict:
    """Parse --option key=value pairs into a dict.

    Values are auto-coerced to int/float/bool where possible.
    """
    from sparkrun.utils import coerce_value

    result = {}
    for opt in options:
        if "=" not in opt:
            click.echo(
                "Error: --option must be key=value, got: %s" % opt,
                err=True,
            )
            sys.exit(1)
        key, _, value = opt.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            click.echo(
                "Error: --option has empty key: %s" % opt,
                err=True,
            )
            sys.exit(1)
        result[key] = coerce_value(value)
    return result


def _get_config_and_registry(config_path=None):
    """Create SparkrunConfig and RegistryManager."""
    from sparkrun.core.config import SparkrunConfig

    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
    registry_mgr = config.get_registry_manager()
    return config, registry_mgr


def _apply_node_trimming(
    host_list: list[str],
    recipe,
    overrides: dict | None = None,
    runtime=None,
    tp_override: int | None = None,
    quiet: bool = False,
) -> list[str]:
    """Trim host list to match the runtime's required node count.

    When *runtime* is provided, delegates to
    ``runtime.compute_required_nodes()`` which accounts for all
    parallelism dimensions (TP, PP, etc.).  Falls back to TP-only
    logic when no runtime is available (legacy path).

    Used by run, stop, and logs to ensure they all derive the same
    effective host list (and therefore the same cluster_id).

    Args:
        host_list: Resolved hosts.
        recipe: Loaded recipe (used for defaults).
        overrides: Optional CLI overrides (from --option).
        runtime: Optional runtime plugin instance.
        tp_override: Explicit --tp value (takes precedence).
        quiet: Suppress logging.

    Returns:
        Possibly trimmed host list.
    """
    if len(host_list) <= 1:
        return host_list

    effective_overrides = dict(overrides or {})
    if tp_override is not None:
        effective_overrides["tensor_parallel"] = tp_override

    if runtime is not None:
        required = runtime.compute_required_nodes(recipe, effective_overrides)
    else:
        # Legacy fallback: TP-only
        if tp_override is not None:
            required = tp_override
        else:
            config_chain = recipe.build_config_chain(effective_overrides)
            tp_val = config_chain.get("tensor_parallel")
            required = int(tp_val) if tp_val is not None else None

    if required is None or required >= len(host_list):
        return host_list

    trimmed = host_list[:required]
    if not quiet:
        logger.info(
            "Required nodes=%d < %d hosts; using first %d: %s",
            required,
            len(host_list),
            required,
            ", ".join(trimmed),
        )
    return trimmed


def _apply_tp_trimming(
    host_list: list[str],
    recipe,
    overrides: dict | None = None,
    tp_override: int | None = None,
) -> list[str]:
    """Trim host list to match tensor_parallel if TP < host count.

    Backward-compatible alias for :func:`_apply_node_trimming` without
    a runtime (TP-only legacy path).

    Args:
        host_list: Resolved hosts.
        recipe: Loaded recipe (used for defaults).
        overrides: Optional CLI overrides (from --option).
        tp_override: Explicit --tp value (takes precedence).

    Returns:
        Possibly trimmed host list.
    """
    return _apply_node_trimming(
        host_list,
        recipe,
        overrides=overrides,
        tp_override=tp_override,
    )


def _get_cluster_manager(v=None, sctx: SparkrunContext | None = None):
    """Create a ClusterManager using the SAF config root.

    When *sctx* is provided, returns its cached ``cluster_manager``.
    """
    if sctx is not None:
        return sctx.cluster_manager

    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root

    return ClusterManager(get_config_root(v))


def _load_recipe(config, recipe_name, resolve=True):
    """Find, load, and return a recipe.

    Handles disambiguation when a recipe name matches multiple registries.
    Supports remote URLs and @spark-arena/ shortcuts.
    Exits with an error message on failure.

    Args:
        config: SparkrunConfig instance.
        recipe_name: Recipe name, path, or URL.
        resolve: Run the resolver chain immediately (default True).
            Pass ``False`` when CLI overrides need to influence runtime
            resolution — call ``recipe.resolve(overrides)`` later.

    Returns:
        Tuple of (recipe, recipe_path, registry_mgr).
    """
    from sparkrun.core.recipe import Recipe, find_recipe, discover_cwd_recipes, RecipeError, RecipeAmbiguousError

    # Expand shortcuts (e.g. @spark-arena/UUID -> full URL)
    recipe_name = _expand_recipe_shortcut(recipe_name)

    # Handle remote URLs (e.g. spark-arena recipe links)
    if _is_recipe_url(recipe_name):
        logger.debug("Loading recipe from URL: %s", recipe_name)
        cached_path = _fetch_and_cache_recipe(recipe_name)
        try:
            recipe = Recipe.load(cached_path, resolve=resolve)
        except RecipeError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
        # Store URL as source for display/debugging
        recipe.source_path = recipe_name
        # Registry manager still needed by callers (e.g. tuning sync)
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        return recipe, cached_path, registry_mgr

    try:
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        recipe_path = find_recipe(recipe_name, registry_manager=registry_mgr, local_files=discover_cwd_recipes())
        recipe = Recipe.load(recipe_path, resolve=resolve)
    except RecipeAmbiguousError as e:
        # Interactive disambiguation
        if sys.stdin.isatty():
            click.echo("Recipe '%s' found in multiple registries:" % e.name)
            for i, (reg, path) in enumerate(e.matches, 1):
                click.echo("  %d. @%s/%s" % (i, reg, e.name))
            click.echo()
            choice = click.prompt(
                "Select registry",
                type=click.IntRange(1, len(e.matches)),
                default=1,
            )
            _reg_name, recipe_path = e.matches[choice - 1]
            recipe = Recipe.load(recipe_path, resolve=resolve)
        else:
            raise click.ClickException(str(e))
    except RecipeError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Tag recipe with its source registry (None for local/CWD recipes)
    recipe.source_registry = registry_mgr.registry_for_path(recipe_path)
    if recipe.source_registry:
        try:
            entry = registry_mgr.get_registry(recipe.source_registry)
            recipe.source_registry_url = entry.url
        except Exception:
            pass
    return recipe, recipe_path, registry_mgr


def _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v=None, sctx: SparkrunContext | None = None):
    """Resolve hosts from CLI args; exit if none are found.

    Also applies the cluster's SSH user to *config* when a cluster is
    resolved and has a user configured.  This replaces the previous
    separate ``_apply_cluster_user()`` call.

    Returns:
        Tuple of (host_list, cluster_mgr).
    """
    from sparkrun.core.hosts import resolve_hosts

    cluster_mgr = _get_cluster_manager(v) if sctx is None else _get_cluster_manager(sctx=sctx)
    host_list = resolve_hosts(
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        cluster_manager=cluster_mgr,
        config_default_hosts=config.default_hosts,
    )
    if not host_list:
        click.echo("Error: No hosts specified. Use --hosts or configure defaults.", err=True)
        sys.exit(1)
    # Apply cluster-level SSH user (if defined) so downstream SSH calls
    # automatically use it.
    cluster_user = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr).user
    if cluster_user:
        config.ssh_user = cluster_user
    return host_list, cluster_mgr


def _resolve_setup_context(hosts, hosts_file, cluster_name, config, user=None):
    """Resolve hosts, user, and SSH kwargs for setup commands."""
    import os
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    # _resolve_hosts_or_exit now applies cluster user to config automatically
    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    if user is None:
        user = config.ssh_user or os.environ.get("USER", "root")
    ssh_kwargs = build_ssh_kwargs(config)
    if user:
        ssh_kwargs["ssh_user"] = user
    return host_list, user, ssh_kwargs


def _display_recipe_detail(recipe, show_vram=True, registry_name=None, cli_overrides=None, cache_dir=None):
    """Display recipe details (delegates to cli_formatters)."""
    from sparkrun.utils.cli_formatters import display_recipe_detail

    display_recipe_detail(recipe, show_vram=show_vram, registry_name=registry_name, cli_overrides=cli_overrides, cache_dir=cache_dir)


def _display_vram_estimate(recipe, cli_overrides=None, auto_detect=True, cache_dir=None):
    """Display VRAM estimation (delegates to cli_formatters)."""
    from sparkrun.utils.cli_formatters import display_vram_estimate

    display_vram_estimate(recipe, cli_overrides=cli_overrides, auto_detect=auto_detect, cache_dir=cache_dir)


def _shell_rc_file(shell):
    """Return the RC file path for a given shell name.

    Exits with an error for unsupported shells.
    """
    from pathlib import Path

    home = Path.home()
    rc_files = {
        "bash": home / ".bashrc",
        "zsh": home / ".zshrc",
        "fish": home / ".config" / "fish" / "config.fish",
    }
    if shell not in rc_files:
        click.echo("Error: Unsupported shell: %s" % shell, err=True)
        sys.exit(1)
    return rc_files[shell]


def _detect_shell():
    """Detect the user's login shell, returning (name, rc_file)."""
    import os
    from pathlib import Path

    login_shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in login_shell:
        return "zsh", home / ".zshrc"
    elif "fish" in login_shell:
        return "fish", home / ".config" / "fish" / "config.fish"
    else:
        return "bash", home / ".bashrc"


def _require_uv() -> str:
    """Return path to uv binary, or exit with an error message."""
    import shutil

    # noinspection PyDeprecation
    uv = shutil.which("uv")
    if not uv:
        click.echo("Error: uv is required but not found on PATH.", err=True)
        click.echo("Install uv first: pip install uv", err=True)
        sys.exit(1)
    return uv


def _complete_yaml_files(incomplete):
    """Return CompletionItems for YAML files matching an incomplete path."""
    from pathlib import Path

    items = []
    # Determine the directory to search and the prefix to match
    p = Path(incomplete)
    if incomplete.endswith("/"):
        search_dir = p
        prefix = ""
    else:
        search_dir = p.parent
        prefix = p.name

    if not search_dir.is_dir():
        return items

    try:
        for entry in sorted(search_dir.iterdir()):
            if not entry.name.startswith(prefix):
                continue
            rel = str(entry.relative_to(".")) if not incomplete.startswith("/") else str(entry)
            # Preserve leading ./ if the user typed it
            if incomplete.startswith("./") and not rel.startswith("./"):
                rel = "./" + rel
            if entry.is_dir():
                items.append(
                    click.shell_completion.CompletionItem(
                        rel + "/",
                        type="dir",
                    )
                )
            elif entry.suffix in (".yaml", ".yml"):
                items.append(
                    click.shell_completion.CompletionItem(
                        rel,
                        type="file",
                    )
                )
    except OSError:
        pass
    return items


class RecipeNameType(click.ParamType):
    """Click parameter type with shell completion for recipe names."""

    name = "recipe"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for recipe names and file paths.

        Supports @registry/name syntax:
        - @ prefix: lists registry names (all enabled, regardless of visibility)
        - @registry/: lists recipes from that registry (include hidden)

        Also completes local YAML file paths when the incomplete value
        looks like a path (starts with '.', '/', '~', or contains '/').

        Default (no @ prefix): only shows recipes from visible registries.
        """
        try:
            # Handle @registry/ prefix completion first (before file-path check,
            # since @registry/name contains '/' but is not a filesystem path)
            if incomplete.startswith("@"):
                config, registry_mgr = _get_config_and_registry()
                from sparkrun.core.recipe import list_recipes

                if "/" not in incomplete:
                    # No slash yet — try to expand directly to @registry/recipe
                    # items so the user gets full completions in one tab press.
                    # Fall back to @registry/ names when recipes can't be listed
                    # (e.g. cache not populated yet).
                    registries = registry_mgr.list_registries()
                    prefix = incomplete[1:]  # strip @
                    items = []
                    matching_registries = []
                    for reg in registries:
                        if not reg.enabled or not reg.name.startswith(prefix):
                            continue
                        matching_registries.append(reg)
                        recipe_path = registry_mgr.cache_root / reg.name / reg.subpath
                        recipes = list_recipes(search_paths=[recipe_path])
                        for r in recipes:
                            items.append(click.shell_completion.CompletionItem("@%s/%s" % (reg.name, r["file"])))
                    if not items and matching_registries:
                        # No recipes found — show registry names so the user
                        # can still discover and select the registry.
                        # type="dir" prevents the shell from appending a
                        # trailing space, so the user can continue typing
                        # the recipe name after the slash.
                        items = [click.shell_completion.CompletionItem("@%s/" % reg.name, type="dir") for reg in matching_registries]
                    return items
                else:
                    # Completing recipe after @registry/
                    from sparkrun.utils import parse_scoped_name

                    registry_name, recipe_prefix = parse_scoped_name(incomplete)
                    # Only load recipes from the target registry
                    try:
                        entry = registry_mgr.get_registry(registry_name)
                    except Exception:
                        return []
                    recipe_path = registry_mgr.cache_root / entry.name / entry.subpath
                    recipes = list_recipes(search_paths=[recipe_path])
                    return [
                        click.shell_completion.CompletionItem("@%s/%s" % (registry_name, r["file"]))
                        for r in recipes
                        if r["file"].startswith(recipe_prefix)
                    ]

            # File-path completion when input looks like a path
            if incomplete and (incomplete[0] in (".", "/", "~") or "/" in incomplete):
                return _complete_yaml_files(incomplete)

            # Default: list recipe names from visible registries only
            from sparkrun.core.recipe import list_recipes, discover_cwd_recipes

            config, registry_mgr = _get_config_and_registry()
            recipes = list_recipes(registry_manager=registry_mgr, include_hidden=False, local_files=discover_cwd_recipes())
            return [click.shell_completion.CompletionItem(r["file"]) for r in recipes if r["file"].startswith(incomplete)]
        except Exception:
            return []


RECIPE_NAME = RecipeNameType()


def _is_cluster_id(value: str) -> str | None:
    """Return normalized cluster_id if value looks like one, else None.

    Recognizes full ``sparkrun_<hex>`` IDs and short 8-12 char hex strings.
    """
    import re

    if value.startswith("sparkrun_"):
        return value
    if re.fullmatch(r"[0-9a-f]{8,12}", value):
        return "sparkrun_%s" % value
    return None


class TargetType(RecipeNameType):
    """Click parameter type that accepts either a recipe name or a cluster ID.

    Tab completion delegates to RecipeNameType (only completes recipe names).
    Cluster IDs (hex strings or sparkrun_ prefixed) pass through as-is.
    """

    name = "target"

    def convert(self, value, param, ctx):
        if _is_cluster_id(value) is not None:
            return value
        return super().convert(value, param, ctx)


TARGET = TargetType()


class ProfileNameType(click.ParamType):
    """Click parameter type with shell completion for benchmark profile names."""

    name = "profile"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for profile names and file paths.

        Supports @registry/name syntax:
        - @ prefix: lists registry names (all enabled with benchmark_subpath)
        - @registry/: lists profiles from that registry

        Also completes local YAML file paths when the incomplete value
        looks like a path (starts with '.', '/', '~', or contains '/').

        Default (no @ prefix): only shows profiles from visible registries.
        """
        try:
            # Handle @registry/ prefix completion first (before file-path check,
            # since @registry/name contains '/' but is not a filesystem path)
            if incomplete.startswith("@"):
                config, registry_mgr = _get_config_and_registry()
                if "/" not in incomplete:
                    # No slash yet — expand directly to @registry/profile items
                    registries = registry_mgr.list_registries()
                    prefix = incomplete[1:]  # strip @
                    items = []
                    for reg in registries:
                        if not reg.enabled or not reg.benchmark_subpath or not reg.name.startswith(prefix):
                            continue
                        profiles = registry_mgr.list_benchmark_profiles(registry_name=reg.name, include_hidden=True)
                        for p in profiles:
                            items.append(click.shell_completion.CompletionItem("@%s/%s" % (reg.name, p["file"])))
                    return items
                else:
                    # Completing profile name after @registry/
                    from sparkrun.utils import parse_scoped_name

                    registry_name, profile_prefix = parse_scoped_name(incomplete)
                    profiles = registry_mgr.list_benchmark_profiles(registry_name=registry_name, include_hidden=True)
                    return [
                        click.shell_completion.CompletionItem("@%s/%s" % (registry_name, p["file"]))
                        for p in profiles
                        if p["file"].startswith(profile_prefix)
                    ]

            # File-path completion when input looks like a path
            if incomplete and (incomplete[0] in (".", "/", "~") or "/" in incomplete):
                return _complete_yaml_files(incomplete)

            # Default: list profile names from visible registries only
            config, registry_mgr = _get_config_and_registry()
            profiles = registry_mgr.list_benchmark_profiles()
            return [click.shell_completion.CompletionItem(p["file"]) for p in profiles if p["file"].startswith(incomplete)]
        except Exception:
            return []


PROFILE_NAME = ProfileNameType()


class ClusterNameType(click.ParamType):
    """Click parameter type with shell completion for cluster names."""

    name = "cluster"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for cluster names."""
        try:
            mgr = _get_cluster_manager()
            clusters = mgr.list_clusters()
            return [click.shell_completion.CompletionItem(c.name) for c in clusters if c.name.startswith(incomplete)]
        except Exception:
            return []


CLUSTER_NAME = ClusterNameType()


class RegistryNameType(click.ParamType):
    """Click parameter type with shell completion for registry names."""

    name = "registry"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for registry names."""
        try:
            _, registry_mgr = _get_config_and_registry()
            return [
                click.shell_completion.CompletionItem(reg.name) for reg in registry_mgr.list_registries() if reg.name.startswith(incomplete)
            ]
        except Exception:
            return []


REGISTRY_NAME = RegistryNameType()


class RuntimeNameType(click.ParamType):
    """Click parameter type with shell completion for runtime names."""

    name = "runtime"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for known runtimes."""
        try:
            from sparkrun.core.recipe import list_recipes

            _, registry_mgr = _get_config_and_registry()
            recipes = list_recipes(registry_manager=registry_mgr)
            runtimes = sorted({r.get("runtime", "") for r in recipes if r.get("runtime")})
            return [click.shell_completion.CompletionItem(rt) for rt in runtimes if rt.startswith(incomplete)]
        except Exception:
            return []


RUNTIME_NAME = RuntimeNameType()


def host_options(f):
    """Common host-targeting options: --hosts, --hosts-file, --cluster."""
    f = click.option("--cluster", "cluster_name", default=None, type=CLUSTER_NAME, help="Use a saved cluster by name")(f)
    f = click.option("--hosts-file", default=None, help="File with hosts (one per line, # comments)")(f)
    f = click.option("--hosts", "-H", default=None, help="Comma-separated host list")(f)
    return f


def recipe_override_options(f):
    """Common recipe override options: --tp, --pp, --gpu-mem, --max-model-len, --option/-o, --image."""
    f = click.option("--option", "-o", "options", multiple=True, help="Override any recipe default: -o key=value (repeatable)")(f)
    f = click.option("--image", default=None, help="Override container image")(f)
    f = click.option("--max-model-len", type=int, default=None, help="Override maximum model context length")(f)
    f = click.option(
        "--gpu-mem", "--gpu-memory-utilization", "--mem-fraction-static", type=float, default=None, help="Override GPU memory utilization"
    )(f)
    f = click.option("--pp", "--pipeline-parallel", "pipeline_parallel", type=int, default=None, help="Override pipeline parallelism")(f)
    f = click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None, help="Override tensor parallelism")(f)
    # TODO: add options for expert parallel and data parallel and context parallel ??? and runtime arg validation
    return f


def _apply_recipe_overrides(
    options, tensor_parallel=None, pipeline_parallel=None, gpu_mem=None, max_model_len=None, image=None, recipe=None, **kwargs
):
    """Build overrides dict, apply to recipe, and resolve runtime.

    Returns ``(recipe, overrides)`` — recipe is returned to make the
    mutation explicit (runtime may change based on overrides).

    When *recipe* is provided, ``recipe.resolve(overrides)`` is called
    so that overrides can influence runtime resolution (e.g.
    ``distributed_executor_backend=ray`` switches vllm-distributed to
    vllm-ray).
    """
    overrides = _parse_options(options)
    if tensor_parallel is not None:
        overrides["tensor_parallel"] = tensor_parallel
    if pipeline_parallel is not None:
        overrides["pipeline_parallel"] = pipeline_parallel
    if gpu_mem is not None:
        overrides["gpu_memory_utilization"] = gpu_mem
    if max_model_len is not None:
        overrides["max_model_len"] = max_model_len
    if image and recipe is not None:
        recipe.container = image

    for k, v in kwargs.items():
        if v is not None:
            overrides[k] = v

    # Resolve runtime with overrides visible to resolvers
    if recipe is not None:
        recipe.resolve(overrides)

    return recipe, overrides


def dry_run_option(f):
    """Common --dry-run flag."""
    return click.option("--dry-run", "-n", is_flag=True, help="Show what would be done")(f)


def validate_and_prepare_hosts(
    host_list: list[str],
    recipe,
    overrides: dict,
    runtime,
    solo: bool = False,
) -> tuple[list[str], bool]:
    """Validate node count, enforce max_nodes, and determine solo mode.

    Returns ``(trimmed_host_list, is_solo)``.
    Exits with an error on invalid configurations.
    """
    # Node count validation / trimming
    if len(host_list) > 1 and not solo:
        try:
            required = runtime.compute_required_nodes(recipe, overrides)
        except ValueError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
        if required is not None:
            if required > len(host_list):
                click.echo(
                    "Error: runtime requires %d nodes, but only %d hosts provided" % (required, len(host_list)),
                    err=True,
                )
                sys.exit(1)
            elif required < len(host_list):
                original_count = len(host_list)
                host_list = _apply_node_trimming(
                    host_list,
                    recipe,
                    overrides,
                    runtime=runtime,
                )
                click.echo("Note: %d nodes required, using %d of %d hosts" % (required, required, original_count))

    # Enforce max_nodes
    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        try:
            req = runtime.compute_required_nodes(recipe, overrides)
        except ValueError:
            req = None
        if req is not None and req > recipe.max_nodes:
            click.echo(
                "Error: runtime requires %d nodes (from parallelism settings), "
                "but recipe '%s' specifies max_nodes=%d" % (req, recipe.qualified_name, recipe.max_nodes),
                err=True,
            )
            sys.exit(1)
        click.echo("Note: recipe max_nodes=%d, using %d of %d hosts" % (recipe.max_nodes, recipe.max_nodes, len(host_list)))
        host_list = host_list[: recipe.max_nodes]

    # Determine mode
    is_solo = solo or len(host_list) <= 1
    if recipe.mode == "solo":
        is_solo = True
    if is_solo and len(host_list) > 1:
        click.echo("Note: solo mode enabled, using 1 of %d hosts" % len(host_list))
        host_list = host_list[:1]

    return host_list, is_solo


def build_cluster_id_overrides(
    port: int | None = None,
    served_model_name: str | None = None,
    tp_override: int | None = None,
    pp_override: int | None = None,
) -> dict | None:
    """Build overrides dict for cluster_id generation from CLI flags.

    Returns dict of overrides, or None if all values are None.
    """
    overrides = {}
    if port is not None:
        overrides["port"] = port
    if served_model_name is not None:
        overrides["served_model_name"] = served_model_name
    if tp_override is not None:
        overrides["tensor_parallel"] = tp_override
    if pp_override is not None:
        overrides["pipeline_parallel"] = pp_override
    return overrides or None


def resolve_hosts_with_metadata_fallback(
    hosts,
    hosts_file,
    cluster_name,
    config,
    meta,
    target_label,
    v=None,
    sctx: SparkrunContext | None = None,
) -> list[str]:
    """Resolve hosts from CLI args, job metadata, or defaults.

    Priority: CLI flags > metadata hosts > default cluster/config.
    Exits with error if no hosts can be resolved.
    """
    if hosts or hosts_file or cluster_name:
        host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v, sctx=sctx)
        return host_list
    if meta and meta.get("hosts"):
        return meta["hosts"]
    try:
        host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v, sctx=sctx)
        return host_list
    except SystemExit:
        click.echo(
            "Error: No job metadata for '%s' and no hosts specified.\n"
            "  Specify hosts with --hosts or --cluster, or run from the machine that launched the job." % target_label,
            err=True,
        )
        sys.exit(1)
