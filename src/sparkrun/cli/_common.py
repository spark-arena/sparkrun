"""Shared CLI infrastructure: utilities, Click types, decorators."""

from __future__ import annotations

import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import click
import click.shell_completion  # enables click.shell_completion.CompletionItem in helpers

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


def _get_context(ctx, config_path=None) -> "SparkrunContext":
    """Lazily create and cache a :class:`SparkrunContext` on the Click context.

    Calls ``init_sparkrun()`` and creates a ``SparkrunConfig``, bundling
    them into a single context object stored in ``ctx.obj["sparkrun_ctx"]``.

    Logging is *not* re-applied here — ``_setup_logging()`` is already
    called once from the ``main()`` group callback, and SAF's
    ``fixed_logger`` parameter means ``init_framework_desktop`` skips
    its own logging setup entirely.
    """
    obj = ctx.ensure_object(dict)
    sctx = obj.get("sparkrun_ctx", None)
    if sctx is not None:
        return sctx

    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext

    v = init_sparkrun()
    if config_path is None:
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


def _terminate_as_interrupt(signum, _frame):
    """Turn a termination signal into :class:`KeyboardInterrupt`.

    See :func:`install_termination_handlers` for why.
    """
    logger.debug("Received signal %s — unwinding as KeyboardInterrupt", signum)
    raise KeyboardInterrupt


def install_termination_handlers() -> list[int]:
    """Make ``SIGTERM`` / ``SIGHUP`` unwind like Ctrl-C.

    Two things follow from raising :class:`KeyboardInterrupt` instead of dying
    where the default handler would:

    - **Remote work is torn down.** ``subprocess.run`` kills its child from the
      bare ``except`` in its cleanup path, so the ``ssh`` client dies, sshd
      exits, and the remote session guard
      (:func:`~sparkrun.orchestration.ssh.wrap_with_session_guard`) fires.
      Without this, ``kill <sparkrun-pid>`` leaves the ``ssh`` client running
      as a local orphan, the session stays healthy, and the guard correctly
      never fires — the remote download keeps going.
    - **Existing cleanup runs.** ``KeyboardInterrupt`` is already the signal
      every state-preserving path in the codebase handles (``api._run``,
      ``api._benchmark``, the benchmark scheduler), so ``SIGTERM`` now gets the
      same treatment Ctrl-C has always had.

    ``SIGKILL`` is unreachable by definition; a launch killed with ``-9`` can
    still orphan the *streaming* distribution path, whose ``ssh`` stdout is an
    inherited terminal rather than a pipe back to us.

    Best-effort: signal handlers can only be installed from the main thread, so
    a non-main-thread caller (embedded use, a test runner) is a silent no-op.
    An already-customised handler is left alone rather than clobbered.

    Returns:
        The signal numbers for which a handler was installed.
    """
    import signal

    installed: list[int] = []
    candidates = [getattr(signal, name, None) for name in ("SIGTERM", "SIGHUP")]
    for sig in candidates:
        if sig is None:  # e.g. SIGHUP on Windows
            continue
        try:
            if signal.getsignal(sig) is not signal.SIG_DFL:
                continue  # someone else owns it — don't clobber
            signal.signal(sig, _terminate_as_interrupt)
        except (ValueError, OSError, RuntimeError):
            # Not the main thread, or the platform refuses this signal.
            continue
        installed.append(int(sig))
    return installed


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


def resolve_effective_hosts_for_recipe(
    host_list: list[str],
    recipe,
    overrides: dict | None = None,
    *,
    cluster_def=None,
    runtime=None,
    sctx: SparkrunContext | None = None,
    solo: bool = False,
    scheduler: str | None = None,
    exclude_intent_id: str | None = None,
) -> tuple[list[str], bool]:
    """CLI-layer adapter around :func:`sparkrun.api._hosts.resolve_effective_hosts`.

    Replaces the legacy ``validate_and_prepare_hosts`` helper.  Treats
    placement as a *structural* property: the scheduler's
    ``hosts_used`` IS the effective host list — there is no separate
    "required node count" step.

    The helper is responsible for the three orthogonal CLI/recipe
    constraints that sit outside the scheduler:

    * ``solo`` (or ``recipe.mode == 'solo'``): force a one-host run.
    * ``recipe.max_nodes``: hard upper bound on host count.
    * Single-host short-circuit: when only one host is supplied the
      scheduler is bypassed entirely.

    Echoes the same human-readable notes the prior CLI helpers did
    (``"Note: N nodes required, using N of M hosts"`` etc.) so console
    output remains stable for existing tests and users.

    Args:
        host_list: Resolved hosts (CLI / cluster / file).
        recipe: Loaded recipe.
        overrides: CLI overrides (``-o key=value`` flattened).
        cluster_def: Optional :class:`ClusterDefinition` carrying
            per-host hardware (used by the scheduler for multi-GPU
            placement).
        sctx: Optional shared :class:`SparkrunContext`.
        solo: ``--solo`` flag value.
        scheduler: Resolved scheduler name (CLI flag → recipe → cluster).
            **Must be the same selector the launch will use.**  This trim
            narrows the candidate host list *before* ``api.run`` re-schedules
            over the survivors, so a different scheduler here picks a
            different subset and ``api.run`` is then locked out of the hosts
            this pass discarded — e.g. defaulting to greedy while the cluster
            runs ``occupancy-sparse`` hands ``api.run`` the first N hosts
            regardless of load, and the launch fails with "insufficient free
            capacity" while idle hosts sit unused.
        exclude_intent_id: Intent whose own still-running workloads are
            subtracted from the occupancy snapshot, for the same reason:
            ``api.run`` excludes them, so this pass must too or the two
            passes disagree on a relaunch.

    Returns:
        ``(effective_host_list, is_solo)``.

    Side effects:
        ``click.echo``s human-readable summary lines and calls
        ``sys.exit(1)`` on scheduler errors (mirroring legacy
        behaviour).
    """
    import sparkrun.api as api
    from sparkrun.api._hosts import resolve_effective_hosts

    overrides = overrides or {}

    try:
        host_list, is_solo, notes, _placement = resolve_effective_hosts(
            host_list,
            recipe,
            overrides,
            cluster_def=cluster_def,
            runtime=runtime,
            sctx=sctx,
            solo=solo,
            scheduler=scheduler,
            exclude_intent_id=exclude_intent_id,
        )
    except api.InsufficientCapacity as e:
        # ``resolve_effective_hosts`` already shaped the message (host-count
        # vs occupancy) and attached the status snapshot for diagnostics.
        click.echo("Error: %s" % e, err=True)
        _render_capacity_diagnostics(getattr(e, "status", None), list(getattr(e, "host_list", ()) or host_list))
        sys.exit(1)
    except api.LayoutRequired as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except api.SparkrunError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    for note in notes:
        click.echo(note)

    return host_list, is_solo


def _render_capacity_diagnostics(cluster_status, host_list: list[str]) -> None:
    """Echo a compact rundown of what's currently running, alongside a capacity error.

    Uses the already-fetched :class:`ClusterStatus` snapshot — no new
    SSH round-trip.  When the snapshot is missing (best-effort fetch
    failed earlier) we point the user at the full ``cluster status``
    command instead.
    """
    if cluster_status is None or not getattr(cluster_status, "hosts", ()):
        click.echo("", err=True)
        click.echo("Run `sparkrun cluster status` to see what's running on the cluster.", err=True)
        return

    has_workloads = any(host_occ.workloads for host_occ in cluster_status.hosts)
    if not has_workloads:
        click.echo("", err=True)
        click.echo("No sparkrun workloads detected on these hosts (capacity may be reserved off-cluster).", err=True)
        click.echo("Run `sparkrun cluster status` for full details.", err=True)
        return

    click.echo("", err=True)
    click.echo("Currently running on this cluster:", err=True)
    for host_occ in cluster_status.hosts:
        if not host_occ.workloads:
            click.echo("  %-24s idle" % host_occ.host, err=True)
            continue
        for workload in host_occ.workloads:
            label = workload.recipe_name or workload.cluster_id
            click.echo(
                "  %-24s %s (cluster_id=%s, %d rank(s))" % (host_occ.host, label, workload.cluster_id, workload.ranks_on_host),
                err=True,
            )
    click.echo("", err=True)
    click.echo("Stop a running job with `sparkrun stop <cluster_id>` (or `sparkrun stop --all`).", err=True)


def _get_cluster_manager(v=None, sctx: SparkrunContext | None = None):
    """Create a ClusterManager using the SAF config root.

    When *sctx* is provided, returns its cached ``cluster_manager``.
    """
    if sctx is not None:
        return sctx.cluster_manager

    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root

    return ClusterManager(get_config_root(v))


def _recipe_name_looks_like_path(name: str) -> bool:
    """Return True when *name* looks like a filesystem path.

    Used to short-circuit registry refresh retries for obvious path inputs,
    since updating remote registries cannot help resolve a missing local file.
    """
    if not name:
        return False
    if name.startswith("@"):  # @registry/recipe is a registry reference, not a path
        return False
    if name.startswith((".", "/", "~")):
        return True
    if name.endswith((".yaml", ".yml")):
        return True
    return False


def _load_recipe(config, recipe_name, resolve=True, retry_after_update=False):
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
        retry_after_update: When True and the initial lookup fails with a
            "not found" error, run ``registry_mgr.update()`` once and retry
            the lookup. Useful for ``sparkrun run`` so that copy-pasted
            recipe names from newly-published sources just work.

    Returns:
        Tuple of (recipe, recipe_path, registry_mgr).
    """
    from sparkrun.core.recipe import Recipe, find_recipe, discover_cwd_recipes, RecipeError, RecipeAmbiguousError

    # Expand shortcuts (e.g. @spark-arena/UUID -> full URL)
    recipe_name = _expand_recipe_shortcut(recipe_name)

    # Handle remote URLs (e.g. spark-arena recipe links)
    if _is_recipe_url(recipe_name):
        from sparkrun.core.recipe import RecipeUntrustedHostError

        logger.debug("Loading recipe from URL: %s", recipe_name)
        try:
            cached_path = _fetch_and_cache_recipe(recipe_name)
        except RecipeUntrustedHostError as e:
            # Off-allowlist https host: confirm interactively, else abort.
            if sys.stdin.isatty() and click.confirm(
                "Recipe URL host '%s' is not in the trusted allowlist. Fetch anyway?" % e.host,
                default=False,
            ):
                cached_path = _fetch_and_cache_recipe(recipe_name, allow_untrusted_host=True)
            else:
                click.echo("Error: %s" % e, err=True)
                sys.exit(1)
        except RecipeError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
        try:
            recipe = Recipe.load(cached_path, resolve=resolve)
        except RecipeError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
        # Store URL as source for display/debugging
        recipe.source_path = recipe_name
        # URL-sourced recipes are never auto-trusted (see
        # core.launcher.resolve_recipe_trust): their hooks require
        # --trust or interactive confirmation.
        recipe.is_url_sourced = True
        # Registry manager still needed by callers (e.g. tuning sync)
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        return recipe, cached_path, registry_mgr

    registry_mgr = config.get_registry_manager()
    registry_mgr.ensure_initialized()

    def _prompt_disambiguation(err):
        # Labels are path-qualified (@reg/subdir/name), so nested matches within
        # one registry are distinguishable — a bare @reg/name would print the
        # same option twice. Each label is re-typeable as a scoped recipe name.
        click.echo("Recipe '%s' matches %d recipes:" % (err.name, len(err.matches)))
        for i, label in enumerate(err.labels, 1):
            click.echo("  %d. %s" % (i, label))
        click.echo()
        choice = click.prompt(
            "Select recipe",
            type=click.IntRange(1, len(err.matches)),
            default=1,
        )
        _reg_name, chosen = err.matches[choice - 1]
        return chosen

    # Locate the recipe file; optionally retry once after refreshing registries.
    recipe_path = None
    retried = False
    while True:
        try:
            recipe_path = find_recipe(recipe_name, registry_manager=registry_mgr, local_files=discover_cwd_recipes())
            break
        except RecipeAmbiguousError as e:
            if sys.stdin.isatty():
                recipe_path = _prompt_disambiguation(e)
                break
            raise click.ClickException(str(e))
        except RecipeError as e:
            if retried or not retry_after_update or _recipe_name_looks_like_path(recipe_name):
                click.echo("Error: %s" % e, err=True)
                sys.exit(1)
            retried = True
            click.echo("Recipe '%s' not found; refreshing registries and retrying..." % recipe_name, err=True)
            # If the user scoped the name (@registry/...), only refresh that registry.
            from sparkrun.utils import parse_scoped_name

            scoped_registry, _ = parse_scoped_name(recipe_name)
            try:
                registry_mgr.update(scoped_registry) if scoped_registry else registry_mgr.update()
            except Exception as update_err:
                logger.debug("Registry update failed during retry: %s", update_err)

    try:
        recipe = Recipe.load(recipe_path, resolve=resolve)
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


@dataclass(frozen=True)
class HostContext:
    """Resolved hosts, the cluster manager, and the cluster they came from.

    :attr:`cluster_name` is the *effective* cluster: the one named with
    ``--cluster`` or, when no hosts were supplied explicitly, the default
    cluster (``sparkrun cluster set-default``) that
    :func:`sparkrun.core.hosts.resolve_hosts` took the host list from.

    **A command that forwards a resolved host list to an ``api.*`` entry
    point must pass this name, not the raw ``--cluster`` option.**
    ``api._resolve.resolve_cluster`` short-circuits to an *anonymous*
    ``ClusterDefinition`` as soon as an explicit host list is given without a
    cluster, so ``cluster=None`` silently drops the default cluster's
    executor pin, ``executor_config`` (incl. ``pid_dir``), ``hosts_hardware``
    and transport — the sweep then runs against the wrong substrate with no
    error.
    """

    host_list: list[str]
    cluster_mgr: Any
    cluster: ResolvedClusterConfig
    source: str = "config"
    """Which link of the host-resolution chain supplied :attr:`host_list` —
    one of ``hosts`` / ``hosts-file`` / ``cluster`` / ``default-cluster`` /
    ``config``.  Commands report it so a user can tell *what* they are looking
    at when they named nothing."""

    @property
    def cluster_name(self) -> str | None:
        """The effective cluster name, or ``None`` when hosts are unattached."""
        return self.cluster.name or None

    def describe(self) -> str:
        """One-line "what am I looking at?" banner for command output."""
        n = "%d host(s)" % len(self.host_list)
        if self.source == "cluster":
            return "Cluster: %s — %s" % (self.cluster_name, n)
        if self.source == "default-cluster":
            return "Cluster: %s (default) — %s" % (self.cluster_name, n)
        if self.source == "hosts":
            return "Hosts: %s (--hosts)" % n
        if self.source == "hosts-file":
            return "Hosts: %s (--hosts-file)" % n
        return "Hosts: %s (config default_hosts)" % n


def resolve_host_context(hosts, hosts_file, cluster_name, config, v=None, sctx: SparkrunContext | None = None) -> HostContext:
    """Resolve hosts *and* the cluster they came from; exit if none are found.

    The full form of :func:`_resolve_hosts_or_exit` — same resolution, but it
    keeps the :class:`ResolvedClusterConfig` instead of discarding everything
    but the SSH user.  See :class:`HostContext` for why that matters.
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
    # Resolve the cluster once: its name identifies what the user is looking
    # at (including the default-cluster fallback), and its SSH user is applied
    # to *config* so downstream SSH calls pick it up automatically.
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    if cluster_cfg.user:
        config.ssh_user = cluster_cfg.user

    if hosts:
        source = "hosts"
    elif hosts_file:
        source = "hosts-file"
    elif cluster_name:
        source = "cluster"
    elif cluster_cfg.name:
        source = "default-cluster"
    else:
        source = "config"
    return HostContext(host_list=host_list, cluster_mgr=cluster_mgr, cluster=cluster_cfg, source=source)


def _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v=None, sctx: SparkrunContext | None = None):
    """Resolve hosts from CLI args; exit if none are found.

    Also applies the cluster's SSH user to *config* when a cluster is
    resolved and has a user configured.  This replaces the previous
    separate ``_apply_cluster_user()`` call.

    Returns:
        Tuple of (host_list, cluster_mgr).  Commands that pass a cluster on
        to the api layer want :func:`resolve_host_context` instead.
    """
    hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, v, sctx=sctx)
    return hctx.host_list, hctx.cluster_mgr


def with_host_context(func):
    """Decorator that resolves hosts and cluster manager before the command runs.

    Reads ``hosts``, ``hosts_file``, and ``cluster_name`` from the Click
    kwargs already present (supplied by :func:`host_options`), calls
    :func:`_resolve_hosts_or_exit`, and injects the results as additional
    keyword arguments:

    - ``host_list``   — resolved list of host strings
    - ``cluster_mgr`` — :class:`ClusterManager` instance

    The decorated function must accept ``**kwargs`` or declare ``host_list``
    and ``cluster_mgr`` as explicit keyword parameters.

    Usage::

        @click.command()
        @host_options
        @with_host_context
        def my_cmd(hosts, hosts_file, cluster_name, host_list, cluster_mgr):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hosts = kwargs.get("hosts")
        hosts_file = kwargs.get("hosts_file")
        cluster_name = kwargs.get("cluster_name")

        # Resolve sctx from Click context.  Two cases:
        # 1. @click.pass_context — ctx is the first positional arg.
        # 2. No @click.pass_context — use click.get_current_context() fallback.
        sctx = None
        ctx = None
        if args and hasattr(args[0], "ensure_object"):
            ctx = args[0]
        else:
            try:
                ctx = click.get_current_context()
            except RuntimeError:
                pass

        config = kwargs.get("config")
        if config is None:
            if ctx is not None:
                sctx = _get_context(ctx)
                config = sctx.config
            else:
                from sparkrun.core.config import SparkrunConfig

                config = SparkrunConfig()

        host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)
        kwargs["host_list"] = host_list
        kwargs["cluster_mgr"] = cluster_mgr
        return func(*args, **kwargs)

    return wrapper


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


def report_launch_validation(recipe_ref: str, issues, failed: bool) -> None:
    """Print launch-path validation findings, and say what they mean.

    The launch peer of ``recipe validate``'s output, sharing its renderer so
    the two cannot drift.  Two things it adds that a bare list of messages
    could not:

    * a **heading naming validation as the source**.  Amid a launch there is
      no other clue — the findings arrive before anything has started, so
      unlabelled they read as failures of whatever ran last.
    * a **verdict**.  The same three findings can be fatal or advisory
      depending on the threshold (see
      :func:`sparkrun.core.validation.validate_for_launch`), and "did this
      stop my launch?" is the one question the reader actually has.

    On failure it points at ``recipe validate``, which is the only place the
    withheld suggestions can be seen.  Everything goes to stderr, keeping
    stdout free for the launch's own output.
    """
    from sparkrun.utils.cli_formatters import format_validation_report

    if not issues:
        return
    click.echo(
        format_validation_report(recipe_ref, issues, title="Recipe validation for '%s'" % recipe_ref),
        err=True,
    )
    if failed:
        click.echo("\nCannot launch: fix the above, or see `sparkrun recipe validate %s` for the full report." % recipe_ref, err=True)
    else:
        click.echo("\nNothing above blocks the launch. Continuing.", err=True)


def _display_vram_estimate(
    recipe,
    cli_overrides=None,
    auto_detect=True,
    cache_dir=None,
    cluster=None,
    placement=None,
):
    """Display VRAM estimation (delegates to cli_formatters).

    When *cluster* + *placement* are threaded through, the formatter
    renders per-host fit alongside the legacy DGX-Spark single-line fit.
    """
    from sparkrun.utils.cli_formatters import display_vram_estimate

    display_vram_estimate(
        recipe,
        cli_overrides=cli_overrides,
        auto_detect=auto_detect,
        cache_dir=cache_dir,
        cluster=cluster,
        placement=placement,
    )


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

    Recognises two shapes:

    * **Canonical**: ``sparkrun_<intent>_<placement_token>`` — full
      intent + token.
    * **Bare digest**: 8–12 hex chars or ``<intent>_<placement>``
      digest from status output → normalised with a ``sparkrun_``
      prefix so short-form CLI shortcuts keep working.
    """
    import re

    if value.startswith("sparkrun_"):
        # API layer validates the full form at lookup time.
        return value
    if re.fullmatch(r"(?:[0-9a-f]{8,12}|[0-9a-f]{16}_[0-9a-f]{12})", value):
        return "sparkrun_%s" % value
    return None


def _describe_job(job) -> str:
    """Render a one-line description for a :class:`~sparkrun.api.JobInfo`.

    Used as the ``description`` on :class:`CompletionItem` instances so
    shells that render it (zsh, fish) show recipe + runtime + hosts
    alongside the cluster_id.
    """
    parts = []
    if job.recipe:
        parts.append(job.recipe)
    if job.runtime:
        parts.append(job.runtime)
    if job.hosts:
        parts.append("on " + ",".join(job.hosts))
    return " ".join(parts) if parts else ""


#: How many cached jobs completion considers.  Bounds both the YAML parsing
#: (each metadata file embeds a full recipe state) and the list a shell offers.
COMPLETION_JOB_LIMIT = 25

#: How long completion reuses a recorded snapshot before sweeping again.
#: Short on purpose: long enough that a burst of TABs costs one sweep, short
#: enough that a workload stopped moments ago stops being offered.  Override
#: with ``completion.cache_ttl_s``.
COMPLETION_CACHE_TTL_S = 60.0

#: Per-host ceiling for completion's live status sweep, in seconds.  Hosts are
#: swept in parallel, so this is roughly the worst-case added TAB latency —
#: paid in full only when a host is unreachable.  Override with
#: ``completion.status_timeout_s``; ``0`` disables the sweep.
COMPLETION_STATUS_TIMEOUT_S = 5.0


def _complete_targets(incomplete: str, ctx=None):
    """Complete ``logs`` / ``stop`` targets from the local job cache.

    Two shapes are offered, in this order, and the split is dictated by how
    each actually resolves rather than by preference:

    1. **Recipe names**, for jobs on the cluster this invocation will target.
       ``logs <recipe>`` resolves through live intent discovery against
       whatever ``resolve_cluster`` returns, so offering a recipe whose only
       deployment lives elsewhere would complete to something that then fails
       to resolve.  Scoping them is what keeps a dead cluster's jobs out of the
       list — and the target is read from the ``--cluster`` / ``--hosts``
       already typed on the line, so ``logs --cluster lab <TAB>`` offers that
       cluster's workloads even when no default cluster is configured.
    2. **cluster_ids**, for everything else on the target cluster — a second
       deployment of the same recipe, or a job whose recipe name would not
       resolve.  That form reads its hosts back out of the job metadata, so it
       stays valid regardless of which cluster is being addressed.

    Recipe names come first because on **bash there is no way to annotate a
    completion** — ``BashComplete.format_completion`` emits ``type,value`` and
    drops the help text entirely (zsh and fish do render it).  So the value
    itself has to carry the meaning, and a recipe name does while a hex digest
    does not.  kubectl completes pod *names* for the same reason; our analogue
    of a pod name is the recipe, not the cluster_id.

    **Filtered to what is actually running**, by querying the target cluster.
    That costs an SSH sweep on every TAB, which is a real price — but a list of
    dozens of dead hex digests is not worth having, and the cached-snapshot
    alternative is only ever as fresh as the last command that swept.  The
    sweep is hard-bounded by :func:`_completion_status_timeout` (a subprocess
    timeout per host, run in parallel), so an unreachable host costs that
    ceiling once rather than hanging the shell.  On any failure it falls back
    to the recorded snapshot, then to showing everything: completion must
    never hide a workload on the strength of information it does not have.
    """
    try:
        from sparkrun import api

        jobs = api.list_jobs(limit=COMPLETION_JOB_LIMIT)
        cluster_def = _completion_cluster(ctx)
        target_hosts = set(getattr(cluster_def, "hosts", ()) or ())
        snapshot = _completion_running(cluster_def)

        items: list = []
        seen_recipes: set[str] = set()
        offered_ids: set[str] = set()
        for job in jobs:
            if not _job_is_live(job, snapshot, target_hosts):
                continue
            recipe = job.recipe
            # A URL-sourced recipe's "name" is the URL, which `logs` accepts
            # but which is 80 characters of noise in a completion list — and
            # completing it would re-fetch the URL.  Those jobs stay reachable
            # by cluster_id, which is both shorter and unambiguous.
            if recipe and _is_recipe_url(recipe):
                recipe = None
            # A recipe name resolves only against the target cluster, so only
            # offer it for jobs that actually live there.
            if recipe and target_hosts and set(job.hosts) <= target_hosts and recipe not in seen_recipes:
                seen_recipes.add(recipe)
                if recipe.startswith(incomplete):
                    items.append(click.shell_completion.CompletionItem(recipe, help=_describe_job(job)))
                    offered_ids.add(job.cluster_id)
                    continue
            cid = job.cluster_id
            digest = cid.removeprefix("sparkrun_")
            if cid.startswith(incomplete) or digest.startswith(incomplete):
                items.append(click.shell_completion.CompletionItem(cid, help=_describe_job(job)))
                offered_ids.add(cid)

        # A workload the cluster reports running but the local cache has no
        # metadata for — launched from another machine, or pruned — is still
        # addressable by id, and is exactly what the user is reaching for.
        if snapshot is not None:
            for cid in sorted(snapshot[0] - offered_ids):
                digest = cid.removeprefix("sparkrun_")
                if cid.startswith(incomplete) or digest.startswith(incomplete):
                    items.append(click.shell_completion.CompletionItem(cid))
        return items
    except Exception:  # noqa: BLE001 — completion must never crash; degrade to empty list
        return []


def _completion_running(cluster_def):
    """Which workloads are running, for completion's purposes.

    Cache first, then a live sweep.  A TAB burst — the usual way completion is
    used — then costs one SSH round-trip rather than one per keystroke, and
    ``api.status`` records every sweep it performs, so the sweep this function
    triggers is itself what makes the next TAB instant.

    The cache is only accepted when it **covers the target's hosts**: a
    snapshot left behind by a sweep of some *other* cluster says nothing about
    this one, and treating its hosts as unobserved would put every dead job
    back in the list.

    :data:`COMPLETION_CACHE_TTL_S` is deliberately much shorter than
    :data:`~sparkrun.orchestration.job_metadata.RUNNING_SNAPSHOT_MAX_AGE_S`.
    The point of caching here is to make a burst of TABs cheap, which needs
    seconds, not minutes — and the longer window would keep offering a
    workload for ten minutes after it was stopped, which is the staleness this
    whole path exists to eliminate.  The longer window is still honoured as a
    *fallback* when a live sweep fails: stale information beats none.

    Returns ``(running_cluster_ids, hosts_covered)``, or ``None`` for "could
    not establish", which callers must treat as "show everything".  Hosts the
    sweep failed to reach are excluded from the covered set, so a workload on
    an unreachable host reads as unknown rather than dead.
    """
    from sparkrun.orchestration.job_metadata import load_running_snapshot

    target = set(getattr(cluster_def, "hosts", ()) or ())
    cached = load_running_snapshot(max_age_s=_completion_cache_ttl())
    if cached is not None and target and target <= cached[1]:
        return cached

    timeout = _completion_status_timeout()
    if cluster_def is not None and timeout > 0:
        try:
            from sparkrun import api
            from sparkrun.core.config import SparkrunConfig
            from sparkrun.orchestration.primitives import build_ssh_kwargs

            config = SparkrunConfig()
            if getattr(cluster_def, "user", None):
                config.ssh_user = cluster_def.user
            ssh_kwargs = build_ssh_kwargs(config)
            # The one hard bound on how long a TAB can take: a per-host
            # subprocess timeout, with the hosts swept in parallel.
            ssh_kwargs["timeout"] = timeout

            hosts = list(cluster_def.hosts)
            status = api.status(hosts, cluster=cluster_def, ssh_kwargs=ssh_kwargs)
            running = {w.cluster_id for entry in status.hosts for w in entry.workloads if w.cluster_id}
            covered = frozenset(h for h in hosts if h not in status.errors)
            return frozenset(running), covered
        except Exception:
            logger.debug("Completion status query failed; falling back to the cached snapshot", exc_info=True)

    # Live sweep unavailable or failed: a stale snapshot beats none.
    return load_running_snapshot()


def _completion_cache_ttl() -> float:
    """How long a recorded snapshot is reused before completion sweeps again.

    ``completion.cache_ttl_s`` in ``config.yaml``; ``0`` sweeps on every TAB.
    """
    return _completion_setting("cache_ttl_s", COMPLETION_CACHE_TTL_S)


def _completion_status_timeout() -> float:
    """Per-host ceiling for completion's status sweep, in seconds.

    ``completion.status_timeout_s`` in ``config.yaml``; ``0`` disables the live
    query entirely and falls back to the recorded snapshot.  Exists because the
    right value is a property of the user's network, not of sparkrun — and
    because someone on a flaky VPN needs a way to turn it off without losing
    completion altogether.
    """
    return _completion_setting("status_timeout_s", COMPLETION_STATUS_TIMEOUT_S)


def _completion_setting(key: str, default: float) -> float:
    """Read one ``completion.*`` float from config, never raising."""
    try:
        from sparkrun.core.config import SparkrunConfig

        return max(float(SparkrunConfig().get("completion.%s" % key, default)), 0.0)
    except Exception:
        return default


def _job_is_live(job, snapshot, target_hosts: "set[str] | None" = None) -> bool:
    """Whether *job* should be offered, given a possibly-partial snapshot.

    ``None`` (no snapshot, or a stale one) means show everything — completion
    must not hide a workload on the strength of information it does not have.

    Three cases once a snapshot exists:

    - **Running** → offer it.
    - **On a host we swept, and absent from the sweep** → dead; hide it.
    - **On a host we did not sweep** → *unknown*.  A sweep is often partial (a
      placement query covers a candidate subset), and hiding a workload nobody
      looked at is worse than showing a stale one.

    …except when the job lives entirely **outside the cluster this invocation
    targets**.  Those are not unknown so much as not this command's business:
    they are the long-dead deployments of clusters that no longer exist (a
    torn-down cloud instance keeps its jobs in the cache forever, and its
    hostnames stop resolving, so they can never be verified). Verifying them
    would mean sweeping every cluster any recent job ever touched, paying the
    full connect timeout for each dead one. Naming that cluster
    (``logs --cluster other <TAB>``) sweeps it and offers its live workloads.
    """
    if snapshot is None:
        return True
    running, covered = snapshot
    if job.cluster_id in running:
        return True
    hosts = set(job.hosts or ())
    if target_hosts and hosts and not (hosts & target_hosts):
        return False
    return not (hosts and hosts <= covered)


def _completion_cluster(ctx=None):
    """The cluster this invocation targets, or ``None`` if none can be resolved.

    Prefers the ``--cluster`` / ``--hosts`` already present on the command
    line — completion runs after Click has parsed the options it has seen, so
    ``logs --cluster lab <TAB>`` scopes to ``lab``.  Otherwise it is whatever
    :func:`~sparkrun.api._resolve.resolve_cluster` resolves with no arguments:
    the default cluster, then ``config.default_hosts``.

    Deliberately *only* those sources.  Guessing a target — from the most
    recent job, say — would silently point completion (and its SSH sweep) at a
    cluster the user never named.
    """
    params = getattr(ctx, "params", None) or {}
    try:
        from sparkrun.api._resolve import resolve_cluster

        hosts = params.get("hosts")
        host_list = [h.strip() for h in hosts.split(",") if h.strip()] if isinstance(hosts, str) else None
        return resolve_cluster(params.get("cluster_name"), host_list)
    except Exception:
        return None


class TargetType(RecipeNameType):
    """Click parameter type that accepts either a recipe name or a cluster ID.

    Tab completion offers running workloads — recipe names for the default
    cluster, cluster_ids elsewhere — falling back to recipe-name completion
    when nothing is cached, so an empty cluster still lets the user address a
    workload by recipe.
    """

    name = "target"

    def convert(self, value, param, ctx):
        if _is_cluster_id(value) is not None:
            return value
        return super().convert(value, param, ctx)

    def shell_complete(self, ctx, param, incomplete):
        """Complete the workload the user is addressing — see :func:`_complete_targets`."""
        # If the input already looks like a path or @registry ref, defer to
        # recipe/file completion — those are never cluster_ids.
        if incomplete and (incomplete[0] in (".", "/", "~") or incomplete.startswith("@")):
            return super().shell_complete(ctx, param, incomplete)

        items = _complete_targets(incomplete, ctx)
        if items:
            return items
        # Nothing cached — fall back to recipe names so the user can still
        # address a workload by recipe (logs/stop accept recipe names too).
        return super().shell_complete(ctx, param, incomplete)


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


class RecipeQueryType(click.ParamType):
    """Click parameter type for list/search queries with ``@registry`` scoping.

    Completion only kicks in for the ``@`` prefix — a free-text query has
    nothing to complete against. ``@`` lists registry names as ``@name/``
    (type ``dir`` so the shell doesn't append a space) and ``@name/`` falls
    through to recipe completion within that registry.
    """

    name = "query"

    def shell_complete(self, ctx, param, incomplete):
        if not incomplete.startswith("@"):
            return []
        try:
            _, registry_mgr = _get_config_and_registry()
            if "/" in incomplete:
                return RECIPE_NAME.shell_complete(ctx, param, incomplete)
            prefix = incomplete[1:]  # strip @
            return [
                click.shell_completion.CompletionItem("@%s/" % reg.name, type="dir")
                for reg in registry_mgr.list_registries()
                if reg.enabled and reg.name.startswith(prefix)
            ]
        except Exception:
            return []


RECIPE_QUERY = RecipeQueryType()


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
    """Common recipe override options: --tp, --pp, --gpu-mem, --max-model-len, --option/-o, --image.

    ``--dp`` / ``--data-parallel`` is registered but hidden — DP recipes are
    unusual and the flag is primarily for advanced users / tests; novice
    users should drive DP via recipe defaults.
    """
    f = click.option("--option", "-o", "options", multiple=True, help="Override any recipe default: -o key=value (repeatable)")(f)
    f = click.option("--image", default=None, help="Override container image")(f)
    f = click.option("--max-model-len", type=int, default=None, help="Override maximum model context length")(f)
    f = click.option(
        "--gpu-mem", "--gpu-memory-utilization", "--mem-fraction-static", type=float, default=None, help="Override GPU memory utilization"
    )(f)
    f = click.option(
        "--dp",
        "--data-parallel",
        "data_parallel",
        type=int,
        default=None,
        hidden=True,
        help="Override data parallelism (advanced)",
    )(f)
    f = click.option("--pp", "--pipeline-parallel", "pipeline_parallel", type=int, default=None, help="Override pipeline parallelism")(f)
    f = click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None, help="Override tensor parallelism")(f)
    # TODO: add options for expert parallel and context parallel ??? and runtime arg validation
    return f


def _apply_recipe_overrides(
    options,
    tensor_parallel=None,
    pipeline_parallel=None,
    data_parallel=None,
    gpu_mem=None,
    max_model_len=None,
    image=None,
    recipe=None,
    **kwargs,
):
    """CLI wrapper around :func:`sparkrun.core.resolve.apply_recipe_overrides`.

    Validates the ``--option/-o`` tuple first via :func:`_parse_options`
    (which echoes ``"Error: --option must be key=value..."`` and exits on
    malformed input, preserving the existing CLI behaviour), then defers
    the override construction + runtime resolution to the console-free
    core resolver.
    """
    from sparkrun.core.resolve import apply_recipe_overrides

    # Validate the option tuple up-front for the CLI's error message + exit
    # code; the core resolver re-parses the (now-valid) tuple identically.
    _parse_options(options)

    return apply_recipe_overrides(
        options,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
        recipe=recipe,
        **kwargs,
    )


def dry_run_option(f):
    """Common --dry-run flag."""
    return click.option("--dry-run", "-n", is_flag=True, help="Show what would be done")(f)


def build_cluster_id_overrides(
    port: int | None = None,
    served_model_name: str | None = None,
    tp_override: int | None = None,
    pp_override: int | None = None,
    dp_override: int | None = None,
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
    if dp_override is not None:
        overrides["data_parallel"] = dp_override
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
) -> tuple[list[str], str | None]:
    """Resolve hosts from CLI args, job metadata, or defaults.

    Priority: CLI flags > metadata hosts > default cluster/config.
    Exits with error if no hosts can be resolved.

    Returns:
        ``(host_list, cluster_name)`` — the *effective* cluster the hosts
        came from (see :class:`HostContext`), to forward to the ``api.*``
        call, or ``None`` when the hosts came from the job's own metadata.
        ``None`` there is deliberate and not a gap: it lets
        ``api._resolve.resolve_cluster_for_job`` recover the cluster the
        **job** recorded, which is a better answer than anything this
        invocation could name (issue #277).
    """
    if hosts or hosts_file or cluster_name:
        hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, v, sctx=sctx)
        return hctx.host_list, hctx.cluster_name
    if meta and meta.get("hosts"):
        return list(meta["hosts"]), None
    try:
        hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, v, sctx=sctx)
        return hctx.host_list, hctx.cluster_name
    except SystemExit:
        click.echo(
            "Error: No job metadata for '%s' and no hosts specified.\n"
            "  Specify hosts with --hosts or --cluster, or run from the machine that launched the job." % target_label,
            err=True,
        )
        sys.exit(1)
