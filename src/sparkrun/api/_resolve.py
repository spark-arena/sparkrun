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

from sparkrun.core.application_profile import resource_name

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.api._errors import HostsUnreachable, RecipeNotFound, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterManager
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def prepare_transport(cluster_def: "ClusterDefinition | None", *, dry_run: bool = False) -> None:
    """Run the cluster's transport ``prepare`` step, translating failures.

    Thin api-layer wrapper over
    :func:`sparkrun.transports.prepare_cluster_transport` that maps a
    :class:`~sparkrun.transports.TransportError` (e.g. a disabled provider
    transport, or an instance that vanished) to :class:`SparkrunError` so the
    console-free contract holds and CLI handlers surface a clean message.
    No-op for plain-SSH clusters.
    """
    from sparkrun.transports import TransportError, prepare_cluster_transport

    try:
        prepare_cluster_transport(cluster_def, dry_run=dry_run)
    except TransportError as e:
        raise SparkrunError(str(e)) from e


def scope_operation(cluster, *, sctx=None, ssh_kwargs=None, dry_run=False, prepare=True):
    """Resolve one operation's context and SSH arguments without mutating callers.

    Explicit arguments override individual configured keys, including explicit
    None/empty values. Transport preparation precedes scoping and target lookup.
    """
    from sparkrun.api._context import resolve_sctx
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    if prepare:
        prepare_transport(cluster, dry_run=dry_run)
    scoped = resolve_sctx(sctx).for_cluster(cluster)
    return scoped, {**build_ssh_kwargs(scoped.config), **(ssh_kwargs or {})}


@contextmanager
def _recipe_errors():
    """Translate known file/recipe-data failures without wrapping the whole operation."""
    from yaml import YAMLError
    from sparkrun.core.recipe import RecipeError

    try:
        yield
    except (RecipeError, OSError, YAMLError, ValueError, TypeError) as exc:
        raise SparkrunError("Recipe is invalid: %s" % type(exc).__name__) from exc


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
        SparkrunError: Known file-decoding or recipe-resolution failures,
            with the original exception retained as the cause.
    """
    from yaml import YAMLError
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe, RecipeError, find_recipe
    from sparkrun.core.registry import RegistryError
    from sparkrun.core._recipe_source import recipe_registry_entry, tag_recipe_source
    from sparkrun.utils import parse_scoped_name

    if not isinstance(recipe_input, str):
        recipe = recipe_input
    else:
        cfg = config or (sctx.config if sctx is not None else SparkrunConfig())
        try:
            registry_mgr = sctx.registry_manager if sctx is not None and config is None else cfg.get_registry_manager()
            recipe_path = find_recipe(recipe_input, registry_manager=registry_mgr, local_files=local_files)
        except (RecipeError, RegistryError, OSError, YAMLError, ValueError) as exc:
            raise RecipeNotFound("Recipe %r not found: %s" % (recipe_input, exc)) from exc
        if not recipe_path:
            raise RecipeNotFound("Recipe %r not found in any configured registry" % recipe_input)
        try:
            with _recipe_errors():
                recipe = Recipe.load(recipe_path, resolve=False, registry_manager=registry_mgr)
        except RegistryError as exc:
            # Ownership is decided while loading (it selects the recipe format);
            # an ambiguous or orphaned cache path is the same failure the
            # source attribution below reports.
            raise RecipeNotFound("Recipe source could not be established: %s" % recipe_input) from exc
        try:
            scope, _ = parse_scoped_name(recipe_input)
            registry = recipe_registry_entry(recipe_path, registry_mgr, registry_name=scope)
            tag_recipe_source(recipe, registry, config=cfg)
        except (RegistryError, OSError, YAMLError, ValueError, TypeError) as exc:
            raise RecipeNotFound("Recipe source could not be established: %s" % recipe_input) from exc

    # Keep preloaded provenance intact; resolve only when overrides are supplied.
    if overrides is not None:
        with _recipe_errors():
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
      4. No cluster, no *hosts_input*, but a **default cluster** is set
         (``sparkrun cluster set-default``) → return it, in full.
      5. …else ``config.default_hosts`` → synthesize.
      6. Otherwise → raise :class:`HostsUnreachable`.

    Steps 4 and 5 mirror :func:`sparkrun.core.hosts.resolve_hosts`, which has
    always consulted the default cluster ahead of ``config.default_hosts``.

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
        SparkrunError: A named cluster cannot be loaded.
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
        from yaml import YAMLError
        from sparkrun.api._context import resolve_sctx
        from sparkrun.core.cluster_manager import ClusterError

        try:
            if cluster_mgr is None:
                cluster_mgr = resolve_sctx(sctx).cluster_manager
            loaded = cluster_mgr.get(cluster_input)
        except (ClusterError, OSError, ValueError, YAMLError) as exc:
            raise SparkrunError("Cannot load cluster %r: %s" % (cluster_input, exc)) from exc
        if explicit_hosts is not None:
            return _replace_cluster_hosts(loaded, explicit_hosts)
        return loaded

    # No cluster.  Need a host source.
    if explicit_hosts is not None:
        return ClusterDefinition(name="", hosts=explicit_hosts)

    # The default cluster (``sparkrun cluster set-default``), which lives in a
    # marker file the ClusterManager owns rather than in ``config.yaml``.
    # Consulted ahead of ``config.default_hosts`` to match the ordering
    # :func:`sparkrun.core.hosts.resolve_hosts` has always used — without this
    # the two resolvers disagreed, and a user whose only host source was a
    # default cluster got ``HostsUnreachable`` from every ``api.*`` entry point
    # that resolves without an explicit cluster.  Returning the definition
    # rather than just its hosts also carries the cluster's SSH user, executor
    # and scheduler, which a bare host list would silently drop.
    default_cluster = _default_cluster(sctx, cluster_mgr)
    if default_cluster is not None:
        return default_cluster

    effective_config = config if config is not None else (sctx.config if sctx is not None else None)
    default_hosts = getattr(effective_config, "default_hosts", None) if effective_config is not None else None
    if default_hosts:
        return ClusterDefinition(name="", hosts=list(default_hosts))

    raise HostsUnreachable("No hosts provided, no default cluster, and no default hosts configured")


def resolve_cluster_for_job(
    cluster_input: "str | ClusterDefinition | None",
    hosts: tuple[str, ...] | list[str],
    *,
    meta: dict | None,
    sctx: "SparkrunContext | None" = None,
) -> "ClusterDefinition":
    """Recover a job's connection without changing its resource namespace.

    Explicit/current cluster settings supply transport and credentials. Saved
    user-scoped jobs retain their recorded principal; an explicit incompatible
    user is rejected. Missing named clusters fall back to recorded settings.
    """
    if cluster_input:
        cluster = resolve_cluster(cluster_input, hosts, sctx=sctx)
    else:
        cluster = None
        recorded_name = (meta or {}).get("cluster")
        if recorded_name:
            try:
                cluster = resolve_cluster(str(recorded_name), hosts, sctx=sctx)
            except Exception:
                logger.warning(
                    "Job records cluster %r, which no longer resolves; using its recorded connection details instead", recorded_name
                )
        if cluster is None:
            cluster = resolve_cluster(None, hosts, sctx=sctx)
        cluster = _with_recorded_ssh_user(cluster, meta)
    if meta and meta.get("executor"):
        from sparkrun.core._executor_destination import metadata_executor_overrides
        from sparkrun.orchestration.executor import ExecutorTarget, resolve_executor

        if type(meta.get("executor_user_scoped")) is bool:
            # The saved policy is enough to pin the principal before transport
            # preparation. Resolve live executor settings after that refresh.
            target = ExecutorTarget(meta["executor"], user_scoped=meta["executor_user_scoped"])
        else:
            # Older records did not persist the policy. Ask their executor's
            # read-only target hook; do not guess namespace semantics by name.
            executor = resolve_executor(
                cluster=cluster,
                cli_overrides=metadata_executor_overrides(meta),
                rootless=False,
                auto_user=False,
                config=sctx.config if sctx is not None else None,
                v=sctx.variables if sctx is not None else None,
            )
            target = executor.resolve_target(dry_run=True)
        cluster = bind_job_cluster(cluster, target, meta, explicit_user=bool(cluster_input and cluster.user))
    return cluster


def bind_job_cluster(cluster, target, meta, *, explicit_user=False):
    """Reapply the recorded namespace, including after provider transport refresh."""
    from dataclasses import replace
    from sparkrun.core._executor_destination import job_ssh_kwargs

    try:
        connection = job_ssh_kwargs(target, meta, {"ssh_user": cluster.user}, explicit_user=explicit_user)
    except ValueError as error:
        raise SparkrunError(str(error)) from error
    return replace(cluster, user=connection["ssh_user"])


def maybe_load_config():
    """Load :class:`SparkrunConfig` for SSH kwargs, or ``None`` on failure.

    The fallback for api entry points called without an ``sctx``: the config
    is what carries the SSH user, key and options, so skipping it means
    connecting with none of them.
    """
    try:
        from sparkrun.core.config import SparkrunConfig

        return SparkrunConfig()
    except Exception:  # pragma: no cover - defensive
        return None


def _with_recorded_ssh_user(cluster: "ClusterDefinition", meta: dict | None) -> "ClusterDefinition":
    """Fill in *cluster*'s SSH user from the job's metadata, if it lacks one."""
    if getattr(cluster, "user", None):
        return cluster
    recorded_user = (meta or {}).get("ssh_user")
    if not recorded_user:
        return cluster

    from dataclasses import replace

    return replace(cluster, user=str(recorded_user))


def _default_cluster(sctx, cluster_mgr) -> "ClusterDefinition | None":
    """Load the configured default cluster, or ``None`` when there isn't one.

    Best-effort: a missing marker file, a default naming a cluster that has
    since been deleted, or an unreadable cluster dir all fall through to the
    next host source rather than raising.
    """
    try:
        if cluster_mgr is None and sctx is not None:
            cluster_mgr = sctx.cluster_manager
        if cluster_mgr is None:
            from sparkrun.core.cluster_manager import ClusterManager
            from sparkrun.core.config import get_config_root

            cluster_mgr = ClusterManager(get_config_root())
        name = cluster_mgr.get_default()
        return cluster_mgr.get(name) if name else None
    except Exception:
        logger.debug("No default cluster available", exc_info=True)
        return None


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


def discover_cluster_id_by_intent(
    intent_id: str,
    target_hosts: list[str],
    *,
    cluster_def,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> str:
    """Find the running cluster_id whose intent prefix matches *intent_id*.

    The shared "which live workload does this recipe mean?" resolver behind
    ``api.stop(recipe=…)`` and ``api.logs(recipe=…)``.

    Status-driven rather than derived: it queries the cluster via the single
    cross-executor source (:func:`~sparkrun.orchestration.executor.query_status_for_cluster`,
    so a job launched under *any* backend is discoverable) and filters
    ``running_cluster_ids()`` for those starting with
    ``resource_name("_") + intent_id + "_"``.  Deriving the *full* cluster_id
    instead would require guessing the placement token, which a load-aware
    scheduler randomizes and a host-set change invalidates — the whole point
    of separating intent from placement.  The user's host scope is the
    authoritative discovery range.

    Raises:
        JobNotFound: no running workload matches the intent.
        AmbiguousWorkload: more than one does (carries ``cluster_ids``).
    """
    from sparkrun.api._errors import AmbiguousWorkload, JobNotFound, SparkrunError
    from sparkrun.orchestration.executor import query_status_for_cluster

    sctx, ssh_kwargs = scope_operation(cluster_def, sctx=sctx)
    config = sctx.config

    status = query_status_for_cluster(
        cluster_def,
        list(target_hosts),
        ssh_kwargs=ssh_kwargs,
        config=config,
        v=sctx.variables if sctx is not None else None,
    )

    # Missing/unreachable hosts cannot establish absence or uniqueness. In
    # particular, proxy unload may retire a saved binding on JobNotFound.
    missing = set(target_hosts) - {host.host for host in status.hosts}
    unavailable = sorted(missing | set(status.observation_errors))
    if unavailable:
        raise SparkrunError("Cannot determine running workloads: status unavailable for %s" % ", ".join(unavailable))

    prefix = resource_name("_%s_" % intent_id)
    matches = sorted({cid for cid in status.running_cluster_ids() if cid.startswith(prefix)})

    if not matches:
        raise JobNotFound("No running workload matches intent %s on hosts %s" % (intent_id, target_hosts))
    if len(matches) > 1:
        raise AmbiguousWorkload(
            "Multiple workloads match this recipe/intent on hosts %s: %s. Re-invoke with an explicit cluster_id." % (target_hosts, matches),
            cluster_ids=matches,
        )
    return matches[0]


__all__ = [
    "discover_cluster_id_by_intent",
    "resolve_recipe",
    "resolve_cluster",
    "resolve_runtime",
]


def resolve_operation_target(options=None, *, recipe, runtime=None, cluster, sctx=None):
    """Return a destination-scoped cluster and frozen high-priority overrides."""
    from dataclasses import replace
    from sparkrun.orchestration.executor import resolve_executor_target

    if runtime is None:
        # Lifecycle recovery can still use recipe/cluster executor selection
        # after a runtime plugin was uninstalled.
        try:
            runtime = resolve_runtime(recipe, sctx=sctx)
        except SparkrunError:
            logger.debug("Runtime unavailable during target resolution", exc_info=True)
    target = resolve_executor_target(
        cli_overrides=options.executor_overrides() if options is not None else None,
        recipe=recipe,
        runtime=runtime,
        cluster=cluster,
        config=sctx.config if sctx is not None else maybe_load_config(),
        v=sctx.variables if sctx is not None else None,
        dry_run=options.dry_run if options is not None else False,
    )
    settings = target.overrides
    settings.pop("executor")
    return replace(cluster, executor=target.executor, executor_config={**(cluster.executor_config or {}), **settings}), target
