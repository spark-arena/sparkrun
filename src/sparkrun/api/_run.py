"""``sparkrun.api.plan`` / ``sparkrun.api.run`` — launch an inference workload.

The launch path is split at the point where it stops deciding and starts
acting:

:func:`plan` — **decide** (no cluster state changes)
  1. Resolve recipe / cluster / hosts / runtime; prepare the transport.
  2. Run the scheduler once via :func:`sparkrun.api.schedule`, against live
     occupancy, applying the orthogonal constraints (solo, ``max_nodes``).
  3. Compose intent_id / placement_token / cluster_id.
  → :class:`RunPlan`

:func:`run` — **act**
  4. Evict this intent's superseded deployments.
  5. Delegate to :func:`sparkrun.core.launcher.launch_inference`.
  6. Translate the launcher's :class:`LaunchResult` into :class:`RunResult`.

``run(options)`` plans internally, so the split is invisible to callers
that don't need it.  It exists for the ones that do: anything rendering a
pre-launch summary needs the target hosts *before* launching, and the only
other way to get them is to schedule separately and pass the winners in as
``options.hosts`` — which silently makes the display pass authoritative
over which hosts ``run`` may still consider.  ``run(options, plan=plan)``
lets the decision be made exactly once.

Both functions raise :class:`~sparkrun.api.SparkrunError` (or a subclass)
for any failure.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import (
    InsufficientCapacity,
    LayoutRequired,
    SparkrunError,
)
from sparkrun.api._models import RunOptions, RunPlan, RunResult

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.scheduler import RankAssignment

logger = logging.getLogger(__name__)


def plan(options: RunOptions, *, sctx: "SparkrunContext | None" = None) -> RunPlan:
    """Decide *what* the launch described by *options* would do, without doing it.

    Resolves recipe / cluster / runtime, prepares the transport, runs the
    scheduler once against live occupancy, and composes the launch's
    identifiers.  Returns a :class:`RunPlan`; changes no cluster state.

    Hand the result to :func:`run` (``run(options, plan=plan)``) to launch
    it.  That is the *only* correct way to render a pre-launch summary: a
    caller that instead narrows ``options.hosts`` to its own placement and
    calls ``run`` leaves ``run`` re-scheduling over the survivors, unable
    to reach any host the first pass dropped — which turns a mere
    scheduler disagreement into a launch failure on a cluster with free
    capacity.

    Args:
        options: Inputs for the launch (same struct :func:`run` takes).
        sctx: Optional shared :class:`SparkrunContext`.

    Raises:
        :class:`InsufficientCapacity`: Scheduler can't fit the workload.
        :class:`LayoutRequired`: Cluster needs an explicit ``recipe.layout``.
        :class:`~sparkrun.api.RecipeNotFound`: Recipe lookup failed.
        :class:`~sparkrun.api.HostsUnreachable`: No usable host source.
        :class:`SparkrunError`: For other resolution failures.
    """
    from sparkrun.api._resolve import (
        resolve_cluster,
        resolve_recipe,
        resolve_runtime,
    )
    from sparkrun.orchestration.job_metadata import (
        derive_placement_token_from_hosts,
        derive_recipe_fingerprint,
        generate_cluster_id,
        generate_intent_id,
        generate_placement_token,
        parse_cluster_id,
    )

    sctx = resolve_sctx(sctx)
    config = sctx.config

    # 1. Resolve inputs.  `resolve_cluster` always returns a populated
    # ClusterDefinition (anonymous when only --hosts was given) so
    # downstream code never has to branch on ``cluster is None``.
    cluster_def = resolve_cluster(options.cluster, options.hosts, sctx=sctx, config=config)

    # Transport prepare: for provider-backed clusters (e.g. Thunder) this
    # refreshes ephemeral connection details (fresh IP/port, SSH key, managed
    # ssh alias) BEFORE any SSH runs — the occupancy status query inside
    # ``resolve_effective_hosts`` below is the first SSH.  No-op for plain-SSH
    # clusters, so existing clusters pay nothing.
    from sparkrun.api._resolve import prepare_transport

    prepare_transport(cluster_def, dry_run=bool(getattr(options, "dry_run", False)))

    recipe = resolve_recipe(options.recipe, sctx=sctx, overrides=options.overrides)
    hosts = list(cluster_def.hosts)
    runtime = resolve_runtime(recipe, sctx=sctx)

    # Scheduler selection chain: caller > recipe > cluster > greedy default.
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER, get_scheduler, resolve_scheduler_selector

    effective_scheduler, _scheduler_defaulted = resolve_scheduler_selector(
        cli=options.scheduler,
        recipe=getattr(recipe, "scheduler", None),
        cluster=getattr(cluster_def, "scheduler", None),
    )
    if _scheduler_defaulted:
        logger.debug("No scheduler configured (recipe/cluster); using default %r", FALLBACK_DEFAULT_SCHEDULER)

    # Apply the cluster's SSH user (if any) to the config so downstream
    # SSH operations (executor.run / distribution / build_ssh_kwargs)
    # log in as the right user.  Matches the CLI's resolution chain
    # where ``_resolve_hosts_or_exit`` applies ``cluster.user`` to
    # ``config.ssh_user`` before launch.
    if getattr(cluster_def, "user", None):
        try:
            config.ssh_user = cluster_def.user
        except Exception:
            logger.debug("Failed to apply cluster SSH user to config", exc_info=True)

    # 2. Compute placement via the single shared authority
    # (:func:`sparkrun.api._hosts.resolve_effective_hosts`).  This is the
    # same path the CLI ``run`` command and the benchmark flow use, so all
    # three place identically — the scheduler's ``hosts_used`` IS the
    # effective host list, ``runtime.world_size()`` is baked into the
    # request, and ``max_nodes`` / solo are applied as orthogonal
    # constraints.  ``notes`` (human-readable trim messages) are carried on
    # the plan for renderers to echo; the library itself never prints.
    from sparkrun.api._hosts import resolve_effective_hosts

    # Deterministic intent for this launch (recipe + overrides).  Passed to the
    # scheduler so a relaunch / resume of the same workload subtracts its own
    # still-running containers from the occupancy snapshot instead of treating
    # them as foreign load.  Reused below as the composed cluster_id's intent.
    intent_id = generate_intent_id(recipe, options.overrides)

    # Serve-configuration digest, taken here for the same reason the intent is:
    # ``launch_inference`` folds platform runtime-flag defaults into
    # recipe.defaults before it persists metadata, so a digest derived down
    # there depends on the *hardware* the job landed on and no caller could
    # reproduce it.  Deriving from the declared recipe keeps it a stable pin —
    # which is what callers that later match a job by fingerprint actually need.
    recipe_fingerprint = derive_recipe_fingerprint(recipe, options.overrides)

    placement: "RankAssignment | None"
    is_solo_request = bool(options.solo) or recipe.mode == "solo"
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        list(hosts),
        recipe,
        options.overrides,
        cluster_def=cluster_def,
        runtime=runtime,
        sctx=sctx,
        solo=is_solo_request,
        scheduler=effective_scheduler,
        exclude_intent_id=intent_id,
    )

    # 3a. Compute intent_id + placement_token; compose cluster_id.
    # The launcher honours ``cluster_id_override`` so we hand it the
    # composed cluster_id rather than letting it derive one from
    # (recipe, hosts).
    #
    # The placement token's source depends on the scheduler:
    #   * Deterministic scheduler (greedy): derive the token from the
    #     candidate host set, exactly as the lookup paths
    #     (``stop`` / ``status`` / ``--ensure`` / ``derive_cluster_id``) do.
    #     Relaunching an identical workload then yields the same cluster_id
    #     and replaces the prior deployment — sparkrun 0.2.x semantics.
    #     We hash the *input* candidate hosts (not the trimmed ``host_list``)
    #     so the launched id matches what those lookup paths compute.
    #   * Status-aware scheduler (occupancy-*): use a fresh random token so
    #     the same intent placed on different host sets across launches gets
    #     distinct identifiers and never collides.
    try:
        scheduler_plugin = get_scheduler(effective_scheduler, v=sctx.variables)
        deterministic_placement = bool(getattr(scheduler_plugin, "deterministic_placement", False))
    except ValueError:
        # Unresolvable selector (e.g. a typo, or a single-host run that
        # short-circuited the scheduler so the name was never validated):
        # fall back to a random token — it can never collide.
        deterministic_placement = False
    if deterministic_placement:
        placement_token = derive_placement_token_from_hosts(hosts)
    else:
        placement_token = generate_placement_token()
    cluster_id_for_launch = options.cluster_id_override or generate_cluster_id(intent_id, placement_token)
    # Recover intent + token from the override when one was supplied so
    # the result still carries accurate metadata.
    if options.cluster_id_override:
        try:
            parsed_intent, parsed_token = parse_cluster_id(options.cluster_id_override)
            intent_id = parsed_intent
            placement_token = parsed_token
        except ValueError:
            # Non-canonical override (e.g. a user-supplied label) — keep
            # the freshly-computed intent_id but blank the token so
            # downstream consumers don't surface a fake one.
            placement_token = ""

    return RunPlan(
        recipe=recipe,
        runtime=runtime,
        cluster=cluster_def,
        candidate_hosts=tuple(hosts),
        host_list=tuple(host_list),
        is_solo=is_solo,
        placement=placement,
        notes=tuple(notes),
        scheduler_selector=effective_scheduler,
        scheduler=_resolve_scheduler_name(effective_scheduler, sctx),
        scheduler_defaulted=_scheduler_defaulted,
        intent_id=intent_id,
        placement_token=placement_token,
        cluster_id=cluster_id_for_launch,
        recipe_fingerprint=recipe_fingerprint,
    )


def run(options: RunOptions, *, sctx: "SparkrunContext | None" = None, plan: RunPlan | None = None) -> RunResult:
    """Launch the workload described by *options* and return a :class:`RunResult`.

    Args:
        options: Inputs for the launch.
        sctx: Optional shared :class:`SparkrunContext`.  When omitted a
            fresh session is built; callers chaining multiple ``api.*``
            calls can construct one ``sctx`` and pass it to share
            config / registry-manager / cluster-manager state.
        plan: Pre-computed :class:`RunPlan` from :func:`plan`.  When given,
            resolution / transport preparation / placement are **not**
            repeated — this launches exactly what the plan describes.  Pass
            it whenever the target hosts were shown to a user first, so the
            summary and the launch cannot diverge.  ``None`` (the default)
            plans internally, which is what a caller that renders nothing
            should do.  It must have been built from the same *options* and
            *sctx*; a mismatched plan launches the plan's decisions.

    Raises:
        :class:`InsufficientCapacity`: Scheduler can't fit the workload.
        :class:`LayoutRequired`: Cluster needs an explicit ``recipe.layout``.
        :class:`~sparkrun.api.RecipeNotFound`: Recipe lookup failed.
        :class:`~sparkrun.api.HostsUnreachable`: No usable host source.
        :class:`~sparkrun.api.TrustRejected`: Recipe hooks rejected.
        :class:`SparkrunError`: For other launch failures.
    """
    from sparkrun.core.launcher import launch_inference
    from sparkrun.orchestration.job_metadata import parse_cluster_id

    sctx = resolve_sctx(sctx)
    started_at = time.time()
    config = sctx.config

    # ``_build_plan`` is a module-level alias for :func:`plan`, needed because
    # the ``plan`` parameter shadows the function name in this scope.
    if plan is None:
        plan = _build_plan(options, sctx=sctx)

    recipe = plan.recipe
    runtime = plan.runtime
    cluster_def = plan.cluster
    hosts = list(plan.candidate_hosts)
    host_list = list(plan.host_list)
    is_solo = plan.is_solo
    placement = plan.placement
    effective_scheduler = plan.scheduler_selector
    intent_id = plan.intent_id
    placement_token = plan.placement_token
    cluster_id_for_launch = plan.cluster_id

    # ``ensure``: don't launch a duplicate of a workload that's already
    # serving.  Matched on the *intent*, so the answer doesn't depend on which
    # scheduler placed the running deployment (see ``api.find_running_intent``).
    # Callers that need to skip *before* paying for a plan — the CLI's
    # ``--ensure``, which short-circuits ahead of the banner — call
    # ``find_running_intent`` themselves and leave this flag off; the query is
    # the same one either way.
    if options.ensure:
        from sparkrun.api._intent import find_running_intent

        match = find_running_intent(intent_id, hosts, cluster=cluster_def, sctx=sctx)
        if match is not None:
            logger.info("ensure: intent %s already running as %s; skipping launch", intent_id, match.cluster_id)
            return _already_running_result(match, plan=plan, options=options, started_at=started_at, sctx=sctx)

    # Re-apply the cluster's SSH user: a plan built against a different
    # ``sctx`` (or a config reset in between) would otherwise leave the
    # launch's SSH operations logging in as the wrong user.  Idempotent when
    # the plan was built from this same context.
    if getattr(cluster_def, "user", None):
        try:
            config.ssh_user = cluster_def.user
        except Exception:
            logger.debug("Failed to apply cluster SSH user to config", exc_info=True)

    # Recipe-owned execution strategies are selected only from top-level items
    # present in this recipe.  Preparation happens before the shared launcher
    # starts pulling images or distributing a model, and therefore before its
    # core-owned replacement barrier can evict a serving workload.
    from sparkrun.core.execution import ExecutionContext, resolve_recipe_execution, run_preparation_steps
    from sparkrun.core.timing import Timeline, timed

    if getattr(sctx, "timing", None) is None:
        setattr(sctx, "timing", Timeline())

    execution_context = ExecutionContext(options=options, plan=plan, sctx=sctx)
    try:
        execution_strategy, preparation_steps = resolve_recipe_execution(execution_context)
        if execution_strategy is not None or preparation_steps:
            strategy_name = execution_strategy.name if execution_strategy is not None else "recipe-hooks"
            with timed(
                sctx.timing,
                "execution.prepare",
                strategy=strategy_name,
                steps=len(preparation_steps),
            ) as preparation_span:
                preparation_receipts = run_preparation_steps(
                    execution_context,
                    preparation_steps,
                    timeline=sctx.timing,
                    parent=preparation_span,
                )
                if execution_strategy is not None:
                    with timed(
                        sctx.timing,
                        "execution.finalize",
                        parent=preparation_span,
                        strategy=strategy_name,
                    ):
                        prepared_execution = execution_strategy.finalize_preparation(execution_context, preparation_receipts)
                else:
                    prepared_execution = None
        else:
            preparation_receipts = {}
            prepared_execution = None
        if execution_strategy is not None and prepared_execution.strategy != execution_strategy.name:
            raise ValueError(
                "execution strategy prepared itself as %r, expected %r" % (prepared_execution.strategy, execution_strategy.name)
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise SparkrunError("launch preparation failed: %s" % error) from error

    # 3a-bis. Evict this intent's superseded deployments.  ``exclude_intent_id``
    # in the planning pass told the scheduler "my own containers aren't foreign
    # load, I'm replacing them" — this is the half that actually replaces them.
    # It is a no-op on the deterministic (greedy) path, where the relaunch
    # reuses the prior cluster_id and the runtime's step-1 cleanup already
    # removes those containers by name.
    #
    # Deferred to ``launch_inference``'s ``before_start`` hook rather than run
    # here: this tears down a *serving* workload, and everything between here
    # and the container start — image distribution, a multi-hundred-GB model
    # download, tuning sync — can take minutes and fail or be interrupted.
    # Evicting up front meant a `sparkrun run` killed with Ctrl-C during
    # distribution left the cluster with neither the old deployment nor the
    # new one.  By the time the hook fires, the only remaining step is
    # starting containers.
    # Every cluster_id the eviction sweep saw running, or ``None`` when no
    # sweep happened (dry run, or the status query failed).  Consumed after the
    # launch by the metadata prune, which must never delete a live workload's
    # metadata and so refuses to run at all without a trustworthy snapshot.
    observed_running: dict[str, set[str] | None] = {"ids": None}

    def _evict_before_start() -> None:
        _, running = _evict_superseded_deployments(
            intent_id=intent_id,
            cluster_id_for_launch=cluster_id_for_launch,
            candidate_hosts=hosts,
            target_hosts=host_list,
            cluster_def=cluster_def,
            config=config,
            sctx=sctx,
        )
        observed_running["ids"] = running

    # 3b. Experimental k8s JobSet path (gated by the api.run.k8s feature flag).
    # When the resolved executor is k8s AND the flag is on, route to the
    # native Kubernetes launcher instead of the SSH-oriented launch_inference.
    # Flag off → fall through to the legacy k8s-executor-over-SSH draft.
    if config.is_feature_enabled("api.run.k8s"):
        from sparkrun.orchestration.executor import ExecutorUnavailableError, resolve_executor_name

        try:
            _executor_name = resolve_executor_name(
                cli_overrides=_build_executor_overrides(options),
                recipe=recipe,
                cluster=cluster_def,
                runtime=runtime,
                config=config,
                v=sctx.variables,
            )
        except ExecutorUnavailableError:
            _executor_name = None
        if _executor_name == "k8s":
            if execution_strategy is not None:
                raise SparkrunError("execution strategy %r does not support the Kubernetes launch path" % execution_strategy.name)
            from sparkrun.api._run_k8s import run_k8s

            # This path returns without going through ``launch_inference``, so
            # it never reaches the ``before_start`` hook — evict here to keep
            # replace-my-own-deployment semantics.  It does not get the SSH
            # path's "only after distribution succeeded" guarantee; the k8s
            # launcher owns its own image/volume staging.
            if not options.dry_run:
                _evict_before_start()

            return run_k8s(
                options,
                sctx,
                recipe=recipe,
                runtime=runtime,
                cluster_def=cluster_def,
                host_list=host_list,
                placement=placement,
                is_solo=is_solo,
                cluster_id=cluster_id_for_launch,
                intent_id=intent_id,
                placement_token=placement_token,
                effective_scheduler=effective_scheduler,
                started_at=started_at,
            )

    # 4. Translate options → launch_inference kwargs.
    launch_kwargs: dict[str, Any] = {
        "recipe": recipe,
        "runtime": runtime,
        "host_list": host_list,
        "overrides": dict(options.overrides),
        "config": config,
        "v": sctx.variables,
        "sctx": sctx,
        "is_solo": is_solo,
        "transfer_mode": options.transfer_mode,
        "transfer_interface": options.transfer_interface,
        "cache_dir": options.cache_dir,
        "runtime_cache_override": (None if options.runtime_cache is None else {"enabled": options.runtime_cache}),
        "local_cache_dir": options.local_cache_dir,
        "dry_run": options.dry_run,
        "detached": options.detached,
        "follow": options.follow,
        "ray_port": options.ray_port,
        "dashboard_port": options.dashboard_port,
        "dashboard": options.dashboard,
        "init_port": options.init_port,
        "executor_config": _build_executor_overrides(options),
        "extra_docker_opts": list(options.extra_docker_opts) if options.extra_docker_opts else None,
        "rootless": not options.rootful,
        "auto_user": not options.rootful,
        "cluster": cluster_def,
        "placement": placement,
        "trust": bool(options.trust),
        "sync_tuning": options.sync_tuning,
        "topology": options.topology,
        "cluster_id_override": cluster_id_for_launch,
        "recipe_ref": options.recipe_ref,
        "preserve_model_perms": options.preserve_model_perms,
        "skip_model_fan_out": options.skip_model_fan_out,
        # ``None`` under --dry-run: the launcher also guards, but a dry run
        # must not depend on a callee honouring the contract to stay read-only.
        "before_start": None if options.dry_run else _evict_before_start,
        "recipe_fingerprint": plan.recipe_fingerprint,
        "owner": options.owner,
        "execution_context": execution_context,
        "execution_strategy": execution_strategy,
        "prepared_execution": prepared_execution,
    }

    # 5. Launch.
    try:
        result = launch_inference(**launch_kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except (InsufficientCapacity, LayoutRequired, SparkrunError):
        # Typed API errors flow through unchanged.
        raise
    except Exception as e:
        raise SparkrunError("launch_inference failed: %s" % e) from e

    # 6. Build RunResult.
    metadata: dict[str, Any] = {
        "recipe": getattr(recipe, "qualified_name", None) or getattr(recipe, "name", None),
        "model": getattr(recipe, "model", None),
        "container_image": result.container_image,
        "serve_port": result.serve_port,
        "effective_cache_dir": result.effective_cache_dir,
    }
    if result.recipe_ref:
        metadata["recipe_ref"] = result.recipe_ref
    if result.runtime_info:
        metadata["runtime_info"] = dict(result.runtime_info)

    # Recover identifier components from the launcher's final cluster_id
    # in case it differs from the one we composed (e.g. an external
    # caller passed a non-canonical cluster_id_override through).
    final_cluster_id = result.cluster_id
    final_intent_id = intent_id
    final_placement_token = placement_token
    try:
        parsed_intent, parsed_token = parse_cluster_id(final_cluster_id)
        final_intent_id = parsed_intent
        final_placement_token = parsed_token
    except ValueError:
        # Non-canonical cluster_id (manual override) — keep the values
        # we computed pre-launch so RunResult still carries something
        # meaningful.
        pass

    run_result = RunResult(
        cluster_id=final_cluster_id,
        intent_id=final_intent_id,
        placement_token=final_placement_token,
        recipe_fingerprint=plan.recipe_fingerprint,
        host_list=tuple(result.host_list),
        placement=placement,
        scheduler=plan.scheduler or _resolve_scheduler_name(effective_scheduler, sctx),
        runtime=runtime.runtime_name,
        executor=_executor_name_from_result(result),
        started_at=started_at,
        dry_run=options.dry_run,
        is_solo=result.is_solo,
        rc=int(result.rc),
        serve_command=result.serve_command or "",
        container_image=result.container_image or "",
        serve_port=int(result.serve_port or 0),
        effective_cache_dir=result.effective_cache_dir or "",
        runtime_info=dict(result.runtime_info or {}),
        metadata=metadata,
        timeline=result.timeline,
        launch_result=result,
    )
    _prune_stale_job_metadata(
        config,
        observed_running=observed_running["ids"],
        keep=(final_cluster_id,),
        sctx=sctx,
    )

    from sparkrun.telemetry import emit_run_telemetry

    emit_run_telemetry(config, result=run_result, recipe=recipe, cluster=cluster_def, options=options)
    return run_result


def _prune_stale_job_metadata(config, *, observed_running: "set[str] | None", keep: tuple[str, ...], sctx) -> None:
    """Drop stale job metadata, using the snapshot the launch already took.

    The cache is append-only — only an explicit ``stop`` removes an entry — so
    a job that crashed (the norm under ``auto_remove``, where the container is
    gone before anything asks about it) lingers forever.  Left alone it reaches
    hundreds of dead entries against a couple of dozen live intents, which is
    what makes ``logs <TAB>`` useless.

    Run here because ``run`` is the one command that both grows the cache and
    already holds a live occupancy snapshot (the eviction sweep's), so pruning
    costs no extra SSH and can be made safe: everything observed running is
    protected, as is the job just launched.

    Skipped entirely when *observed_running* is ``None`` — a dry run, or a
    failed status query.  Age is not a sufficient guard on its own: a
    long-lived server easily outlives the cutoff, and deleting its metadata
    would strand it (``stop`` / ``logs`` / proxy discovery all read this).
    Without a snapshot to check against, doing nothing is the only safe move.
    """
    if observed_running is None:
        return
    try:
        if not config.jobs_autoprune:
            return
    except Exception:
        logger.debug("Could not resolve jobs.autoprune; skipping prune", exc_info=True)
        return

    try:
        from sparkrun.orchestration.job_metadata import prune_job_metadata

        removed = prune_job_metadata(
            protected_cluster_ids=set(observed_running) | set(keep),
            sctx=sctx,
        )
        if removed:
            logger.debug("Pruned %d stale job metadata entries", len(removed))
    except Exception:
        # Housekeeping must never fail a launch that otherwise succeeded.
        logger.debug("Job metadata prune failed", exc_info=True)


#: Module-level alias so :func:`run` can call :func:`plan` despite its own
#: ``plan`` parameter shadowing the name inside that function's scope.
_build_plan = plan


def _already_running_result(match, *, plan: RunPlan, options: RunOptions, started_at: float, sctx) -> RunResult:
    """Build the :class:`RunResult` for an ``ensure`` skip.

    Describes the deployment that is *already* running, not the one the plan
    would have launched — so ``cluster_id`` / ``host_list`` / ``placement_token``
    come from *match*.  ``placement`` is ``None`` and ``launch_result`` is
    ``None``: no launch happened, and reporting the plan's intended placement
    would claim a rank layout that was never applied.
    """
    from sparkrun.orchestration.job_metadata import parse_cluster_id

    try:
        _, matched_token = parse_cluster_id(match.cluster_id)
    except ValueError:
        matched_token = ""

    return RunResult(
        cluster_id=match.cluster_id,
        intent_id=match.intent_id,
        placement_token=matched_token,
        host_list=tuple(match.hosts),
        placement=None,
        scheduler=plan.scheduler or _resolve_scheduler_name(plan.scheduler_selector, sctx),
        runtime=match.runtime or plan.runtime.runtime_name,
        executor="",
        started_at=started_at,
        dry_run=options.dry_run,
        is_solo=len(match.hosts) <= 1,
        rc=0,
        already_running=True,
        metadata={
            "recipe": match.recipe or getattr(plan.recipe, "qualified_name", None),
            "model": getattr(plan.recipe, "model", None),
            "ensure_skipped_launch": True,
            "other_cluster_ids": list(match.other_cluster_ids),
        },
        launch_result=None,
    )


def _evict_superseded_deployments(
    *,
    intent_id: str,
    cluster_id_for_launch: str,
    candidate_hosts: list[str],
    target_hosts: list[str],
    cluster_def,
    config,
    sctx: "SparkrunContext | None",
) -> "tuple[list[str], set[str] | None]":
    """Stop this intent's earlier deployments that sit on the hosts we're about to use.

    A launch's ``cluster_id`` is ``sparkrun_<intent_id>_<placement_token>``.
    Under a *deterministic* scheduler (greedy) the token is derived from the
    host set, so a relaunch reuses the prior cluster_id and the runtime's
    "Step 1: clean up existing containers" removes the previous deployment by
    name.  Under a **status-aware scheduler** (``occupancy-*``) the token is
    freshly random, so the new cluster_id can never match the old containers'
    names — step 1 becomes a no-op and the previous deployment keeps running,
    holding VRAM/RAM and the serve port (issue #223).

    ``resolve_effective_hosts(..., exclude_intent_id=...)`` has already
    subtracted this intent's occupancy from the scheduling snapshot on the
    premise that the relaunch *replaces* it.  This function is the half that
    makes that premise true.

    Scope is deliberately narrow:

    * **Same intent only.**  Foreign workloads are never touched — a second
      recipe sharing the cluster is a capacity question for the scheduler,
      not something a launch may unilaterally kill.
    * **Only deployments overlapping** *target_hosts*.  Running the same
      intent twice on disjoint host subsets is a supported use of the random
      placement token (see :func:`generate_placement_token`), so a
      non-overlapping sibling deployment is left alone.
    * An overlapping deployment is torn down **across every host it occupies**
      within *candidate_hosts*, not just the overlapping ones — half a
      distributed job is dead weight either way.

    Best-effort: discovery or teardown failures are logged and swallowed so
    they can't block a launch that may well succeed anyway.

    Returns:
        ``(evicted, observed_running)`` — the cluster_ids torn down (empty when
        there was nothing to do), and **every** cluster_id the sweep saw
        running, or ``None`` when the sweep itself failed.  The second element
        exists so the post-launch metadata prune can reuse this snapshot
        instead of paying for a second one; ``None`` vs. an empty set is the
        difference between "couldn't look" and "looked, nothing there", and
        only the latter makes deletion safe.
    """
    import sparkrun.api as api
    from sparkrun.orchestration.executor import query_status_for_cluster
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    try:
        status = query_status_for_cluster(
            cluster_def,
            list(candidate_hosts),
            ssh_kwargs=build_ssh_kwargs(config) if config else {},
            config=config,
            v=sctx.variables if sctx is not None else None,
        )
    except Exception as e:
        logger.debug("Could not query cluster status for eviction; skipping: %s", e)
        return [], None

    observed_running = {w.cluster_id for entry in status.hosts for w in entry.workloads if w.cluster_id}

    prefix = "sparkrun_%s_" % intent_id
    target = set(target_hosts)
    # cluster_id -> every host in the snapshot it occupies (insertion-ordered).
    occupied: dict[str, list[str]] = {}
    overlapping: list[str] = []
    for entry in status.hosts:
        for workload in entry.workloads:
            cid = workload.cluster_id
            if cid == cluster_id_for_launch:
                continue
            if workload.intent_id != intent_id and not cid.startswith(prefix):
                continue
            hosts_for_cid = occupied.setdefault(cid, [])
            if entry.host not in hosts_for_cid:
                hosts_for_cid.append(entry.host)
            if entry.host in target and cid not in overlapping:
                overlapping.append(cid)

    evicted: list[str] = []
    for cid in overlapping:
        logger.info(
            "Replacing earlier deployment %s of this workload on %s",
            cid,
            ", ".join(occupied[cid]),
        )
        try:
            result = api.stop(cluster_id=cid, hosts=occupied[cid], cluster=cluster_def, sctx=sctx)
        except Exception as e:
            logger.warning("Could not stop earlier deployment %s: %s — it may still hold GPU memory", cid, e)
            continue
        if result.hosts_failed:
            logger.warning(
                "Teardown of earlier deployment %s did not confirm on %s — it may still hold GPU memory",
                cid,
                ", ".join(result.hosts_failed),
            )
        evicted.append(cid)
    return evicted, observed_running


def _build_executor_overrides(options: RunOptions) -> dict[str, Any]:
    """Flatten ``options.executor`` + ``options.executor_config`` into the
    ``cli_overrides`` dict that ``launch_inference`` forwards to
    :func:`sparkrun.orchestration.executor.resolve_executor`."""
    overrides: dict[str, Any] = {}
    if options.executor:
        overrides["executor"] = options.executor
    if options.executor_config:
        for key, value in options.executor_config.items():
            overrides[key] = value
    return overrides


def _resolve_scheduler_name(effective_scheduler, sctx):
    """Return the registered ``scheduler_name`` for *effective_scheduler*.

    Looking up the scheduler plugin guarantees ``RunResult.scheduler``
    carries the *actually-used* name (e.g. ``"occupancy-sparse"`` when
    the caller relied on the project default) rather than echoing the
    possibly-``None`` selector that was passed in.
    """
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER, get_scheduler

    try:
        plugin = get_scheduler(effective_scheduler, v=sctx.variables if sctx is not None else None)
        return plugin.scheduler_name
    except Exception:
        return effective_scheduler or FALLBACK_DEFAULT_SCHEDULER


def _executor_name_from_result(result) -> str:
    """Recover the executor's name from the launcher's runtime, if it was set.

    The launcher stamps ``runtime.executor`` during launch; we read its
    ``executor_name`` attribute.  Falls back to ``"docker"`` (the
    library default) when the launcher didn't populate it (e.g. dry-run
    paths that short-circuit before executor resolution).
    """
    executor = getattr(result.runtime, "executor", None)
    if executor is None:
        return "docker"
    return getattr(executor, "executor_name", "docker")


__all__ = ["run"]
