"""Shared CLI infrastructure: utilities, Click types, decorators."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote recipe helpers (spark-arena URL support)
# ---------------------------------------------------------------------------

SPARK_ARENA_PREFIX = "@spark-arena/"
SPARK_ARENA_API_URL = "https://spark-arena.com/api/recipes/%s/raw"


def _expand_recipe_shortcut(name: str) -> str:
    """Expand known recipe shortcuts to full URLs.

    Currently supports:
        @spark-arena/UUID  ->  https://spark-arena.com/api/recipes/UUID/raw
    """
    if name.startswith(SPARK_ARENA_PREFIX):
        recipe_id = name[len(SPARK_ARENA_PREFIX):]
        return SPARK_ARENA_API_URL % recipe_id
    return name


def _simplify_recipe_ref(url: str) -> str:
    """Simplify a recipe URL to a shortcut if possible (inverse of expand).

    Currently supports:
        https://spark-arena.com/api/recipes/UUID/raw  ->  @spark-arena/UUID

    Returns the original string unchanged if no simplification applies.
    """
    import re

    m = re.match(r"https?://spark-arena\.com/api/recipes/([^/]+)/raw$", url)
    if m:
        return "%s%s" % (SPARK_ARENA_PREFIX, m.group(1))
    return url


def _is_recipe_url(name: str) -> bool:
    """Check if recipe_name looks like an HTTP(S) URL."""
    return name.startswith(("http://", "https://"))


def _url_cache_path(url: str) -> Path:
    """Return the local cache path for a remote recipe URL."""
    import hashlib

    from sparkrun.core.config import DEFAULT_CACHE_DIR

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return DEFAULT_CACHE_DIR / "remote-recipes" / ("%s.yaml" % url_hash)


def _fetch_and_cache_recipe(url: str) -> Path:
    """Fetch a recipe from URL and cache it locally.

    On success, writes/updates the cache file and returns its path.
    On network failure, falls back to cached copy if available.
    Raises click.ClickException if fetch fails and no cache exists.
    """
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    cache_path = _url_cache_path(url)

    try:
        req = Request(url, headers={"User-Agent": "sparkrun"})
        with urlopen(req, timeout=30) as resp:
            content = resp.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        return cache_path
    except (HTTPError, URLError, OSError) as e:
        if cache_path.exists():
            reason = e.code if isinstance(e, HTTPError) else e.reason
            logger.warning(
                "Failed to fetch recipe (using cached copy): %s", reason,
            )
            return cache_path
        if isinstance(e, HTTPError):
            raise click.ClickException(
                "Failed to fetch recipe from %s: HTTP %d" % (url, e.code)
            )
        raise click.ClickException(
            "Failed to fetch recipe from %s: %s"
            % (url, e.reason if isinstance(e, URLError) else e)
        )


# TODO: converge logging with SAF logging
def _setup_logging(verbose: bool):
    """Configure logging based on verbosity.

    Uses explicit handler setup instead of ``logging.basicConfig`` which
    is silently a no-op when the root logger already has handlers (common
    when libraries like ``huggingface_hub`` configure logging on import).
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = ("%(asctime)s [%(levelname)s] %(name)s: %(message)s" if verbose
           else "%(message)s")

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
    logger.info(
        "Required nodes=%d < %d hosts; using first %d: %s",
        required, len(host_list), required, ", ".join(trimmed),
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
        host_list, recipe, overrides=overrides, tp_override=tp_override,
    )


def _resolve_cluster_user(
        cluster_name: str | None,
        hosts: str | None,
        hosts_file: str | None,
        cluster_mgr,
) -> str | None:
    """Resolve the SSH user from a cluster definition, if applicable.

    Returns the cluster's configured user, or None if no cluster is
    resolved or the cluster has no user set.
    """
    resolved = cluster_name
    if not resolved and not hosts and not hosts_file:
        resolved = cluster_mgr.get_default() if cluster_mgr else None
    if resolved:
        try:
            cluster_def = cluster_mgr.get(resolved)
            return cluster_def.user
        except Exception:
            logger.debug("Failed to resolve cluster '%s'", resolved, exc_info=True)
    return None


def _resolve_transfer_mode(
        cluster_name: str | None,
        hosts: str | None,
        hosts_file: str | None,
        cluster_mgr,
) -> str | None:
    """Resolve transfer_mode from a cluster definition, if applicable.

    Returns the cluster's configured transfer_mode, or None if no cluster is
    resolved or the cluster has no transfer_mode set.

    Mirrors the priority chain of core.hosts.resolve_hosts(): if hosts or
    hosts_file is provided, the cluster is not used, so neither should
    transfer_mode.
    """
    if hosts or hosts_file:
        return None
    resolved = cluster_name
    if not resolved:
        resolved = cluster_mgr.get_default() if cluster_mgr else None
    if resolved:
        try:
            cluster_def = cluster_mgr.get(resolved)
            return cluster_def.transfer_mode
        except Exception:
            logger.debug("Failed to resolve cluster '%s'", resolved, exc_info=True)
    return None


def _resolve_cluster_cache_dir(
        cluster_name: str | None,
        hosts: str | None,
        hosts_file: str | None,
        cluster_mgr,
) -> str | None:
    """Resolve cache_dir from a cluster definition, if applicable.

    Returns the cluster's configured cache_dir, or None if no cluster is
    resolved or the cluster has no cache_dir set.

    Mirrors the priority chain of core.hosts.resolve_hosts(): if hosts or
    hosts_file is provided, the cluster is not used, so neither should cache_dir.
    """
    if hosts or hosts_file:
        return None
    resolved = cluster_name
    if not resolved:
        resolved = cluster_mgr.get_default() if cluster_mgr else None
    if resolved:
        try:
            cluster_def = cluster_mgr.get(resolved)
            return cluster_def.cache_dir
        except Exception:
            logger.debug("Failed to resolve cluster '%s'", resolved, exc_info=True)
    return None


def _get_cluster_manager(v=None):
    """Create a ClusterManager using the SAF config root."""
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root
    # TODO: switch to leveraging scitrera-app-framework plugin for ClusterManager singleton?
    return ClusterManager(get_config_root(v))


def _load_recipe(config, recipe_name):
    """Find, load, and return a recipe.

    Handles disambiguation when a recipe name matches multiple registries.
    Supports remote URLs and @spark-arena/ shortcuts.
    Exits with an error message on failure.

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
            recipe = Recipe.load(cached_path)
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
        recipe_path = find_recipe(recipe_name, registry_manager=registry_mgr,
                                  local_files=discover_cwd_recipes())
        recipe = Recipe.load(recipe_path)
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
            recipe = Recipe.load(recipe_path)
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


def _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v=None):
    """Resolve hosts from CLI args; exit if none are found.

    Also applies the cluster's SSH user to *config* when a cluster is
    resolved and has a user configured.  This replaces the previous
    separate ``_apply_cluster_user()`` call.

    Returns:
        Tuple of (host_list, cluster_mgr).
    """
    from sparkrun.core.hosts import resolve_hosts
    cluster_mgr = _get_cluster_manager(v)
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
    cluster_user = _resolve_cluster_user(cluster_name, hosts, hosts_file, cluster_mgr)
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
    display_recipe_detail(recipe, show_vram=show_vram, registry_name=registry_name,
                          cli_overrides=cli_overrides, cache_dir=cache_dir)


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
                items.append(click.shell_completion.CompletionItem(
                    rel + "/", type="dir",
                ))
            elif entry.suffix in (".yaml", ".yml"):
                items.append(click.shell_completion.CompletionItem(
                    rel, type="file",
                ))
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
                            items.append(click.shell_completion.CompletionItem(
                                "@%s/%s" % (reg.name, r["file"])))
                    if not items and matching_registries:
                        # No recipes found — show registry names so the user
                        # can still discover and select the registry.
                        # type="dir" prevents the shell from appending a
                        # trailing space, so the user can continue typing
                        # the recipe name after the slash.
                        items = [
                            click.shell_completion.CompletionItem(
                                "@%s/" % reg.name, type="dir")
                            for reg in matching_registries
                        ]
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
            recipes = list_recipes(registry_manager=registry_mgr,
                                   include_hidden=False,
                                   local_files=discover_cwd_recipes())
            return [
                click.shell_completion.CompletionItem(r["file"])
                for r in recipes
                if r["file"].startswith(incomplete)
            ]
        except Exception:
            return []


RECIPE_NAME = RecipeNameType()


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
                        profiles = registry_mgr.list_benchmark_profiles(
                            registry_name=reg.name, include_hidden=True)
                        for p in profiles:
                            items.append(click.shell_completion.CompletionItem(
                                "@%s/%s" % (reg.name, p["file"])))
                    return items
                else:
                    # Completing profile name after @registry/
                    from sparkrun.utils import parse_scoped_name
                    registry_name, profile_prefix = parse_scoped_name(incomplete)
                    profiles = registry_mgr.list_benchmark_profiles(
                        registry_name=registry_name, include_hidden=True)
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
            return [
                click.shell_completion.CompletionItem(p["file"])
                for p in profiles
                if p["file"].startswith(incomplete)
            ]
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
            return [
                click.shell_completion.CompletionItem(c.name)
                for c in clusters
                if c.name.startswith(incomplete)
            ]
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
                click.shell_completion.CompletionItem(reg.name)
                for reg in registry_mgr.list_registries()
                if reg.name.startswith(incomplete)
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
            return [
                click.shell_completion.CompletionItem(rt)
                for rt in runtimes
                if rt.startswith(incomplete)
            ]
        except Exception:
            return []


RUNTIME_NAME = RuntimeNameType()


def host_options(f):
    """Common host-targeting options: --hosts, --hosts-file, --cluster."""
    f = click.option("--cluster", "cluster_name", default=None, type=CLUSTER_NAME,
                     help="Use a saved cluster by name")(f)
    f = click.option("--hosts-file", default=None,
                     help="File with hosts (one per line, # comments)")(f)
    f = click.option("--hosts", "-H", default=None,
                     help="Comma-separated host list")(f)
    return f


def dry_run_option(f):
    """Common --dry-run flag."""
    return click.option("--dry-run", "-n", is_flag=True,
                        help="Show what would be done")(f)
