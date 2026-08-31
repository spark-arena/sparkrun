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
                registry_mgr = cfg.get_registry_manager()
            except Exception:
                logger.debug("Failed to construct RegistryManager for recipe lookup", exc_info=True)

        try:
            recipe_path = find_recipe(recipe_input, registry_manager=registry_mgr, local_files=local_files)
        except Exception as e:
            raise RecipeNotFound("Recipe %r not found: %s" % (recipe_input, e)) from e
        if not recipe_path:
            raise RecipeNotFound("Recipe %r not found in any configured registry" % recipe_input)
        recipe = Recipe.load(recipe_path, resolve=False)
        if registry_mgr is not None:
            try:
                recipe.source_registry = registry_mgr.registry_for_path(recipe_path)
                if recipe.source_registry:
                    entry = registry_mgr.get_registry(recipe.source_registry)
                    recipe.source_registry_url = entry.url
            except Exception:
                logger.debug("Failed to tag recipe registry source", exc_info=True)

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
    """Resolve the cluster a *recorded* job belongs to.

    :func:`resolve_cluster` answers "which cluster is this invocation
    about?"; this answers "which cluster was this **job** launched on?" —
    the question ``stop`` / ``logs`` have to answer when they were handed a
    cluster_id and recovered the hosts from job metadata.  Those two are not
    the same: a bare host list resolves to an *anonymous* definition, which
    carries no SSH user, no executor pin, no ``executor_config`` and no
    transport, so a teardown ran as the control node's own login.  On a
    cluster whose ``user:`` differs from that login every SSH failed while
    ``stop`` still printed a success line, leaving the workload serving and
    pinning the GPUs (issue #277).

    Priority:

    1. An explicit *cluster_input* — the user naming a cluster outranks
       whatever the job recorded.
    2. The cluster named in the job's own metadata.  Its *hosts* are still
       *hosts* (the recorded placement, or the user's ``--hosts``): we are
       recovering how to connect, not where to look.
    3. Otherwise the plain :func:`resolve_cluster` answer, with the job's
       recorded ``ssh_user`` applied when nothing else supplied one.

    Step 3's overlay is also step 2's safety net: a cluster that has since
    been renamed or deleted must not take the job's connection details down
    with it, and an anonymous ``--hosts`` launch never had a cluster to
    record in the first place.  The overlay only ever *fills* a gap — a
    resolved cluster's own ``user`` always wins, because that is current
    configuration and the recorded value is history.
    """
    # Truthiness, not ``is not None``: an empty name is how the CLI spells
    # "unnamed" (``HostContext.cluster_name``), and passing it on would look
    # up a cluster called "" and raise.  A ``ClusterDefinition`` is always
    # truthy, so a pre-loaded one still takes this branch.
    if cluster_input:
        return resolve_cluster(cluster_input, hosts, sctx=sctx)

    recorded_name = (meta or {}).get("cluster") or ""
    if recorded_name:
        try:
            return _with_recorded_ssh_user(resolve_cluster(str(recorded_name), hosts, sctx=sctx), meta)
        except Exception:
            # A job outliving its cluster is ordinary (renamed, deleted, or
            # read on a control node that never had it).  Warn — the
            # executor pin and transport are genuinely lost — but carry on
            # with the recorded ssh_user, which is the part that decides
            # whether we can reach the hosts at all.
            logger.warning(
                "Job records cluster %r, which no longer resolves; using its recorded connection details instead",
                recorded_name,
            )

    return _with_recorded_ssh_user(resolve_cluster(None, hosts, sctx=sctx), meta)


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
    ``"sparkrun_" + intent_id + "_"``.  Deriving the *full* cluster_id
    instead would require guessing the placement token, which a load-aware
    scheduler randomizes and a host-set change invalidates — the whole point
    of separating intent from placement.  The user's host scope is the
    authoritative discovery range.

    Raises:
        JobNotFound: no running workload matches the intent.
        AmbiguousWorkload: more than one does (carries ``cluster_ids``).
    """
    from sparkrun.api._errors import AmbiguousWorkload, JobNotFound
    from sparkrun.orchestration.executor import query_status_for_cluster
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = sctx.config if sctx is not None else None
    ssh_kwargs = build_ssh_kwargs(config) if config else {}

    status = query_status_for_cluster(
        cluster_def,
        list(target_hosts),
        ssh_kwargs=ssh_kwargs,
        config=config,
        v=sctx.variables if sctx is not None else None,
    )

    prefix = "sparkrun_%s_" % intent_id
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
