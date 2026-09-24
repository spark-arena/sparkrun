"""``sparkrun.api.plan`` / ``sparkrun.api.run`` — launch an inference workload.

The launch path is split at the point where it stops deciding and starts
acting:

:func:`plan` — **decide** (no cluster state changes)
  1. Resolve recipe / cluster / hosts / runtime; prepare transport and probe hardware.
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

Operational failures use :class:`~sparkrun.api.SparkrunError` (or a subclass).
Invalid programmatic inputs may raise ValueError/TypeError; interrupts propagate.
"""

from __future__ import annotations

from sparkrun.core.status_observation import RunningSnapshot

import logging
from contextlib import contextmanager
from dataclasses import replace
import time
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import (
    InsufficientCapacity,
    IntegrationUnavailable,
    SparkrunError,
)
from sparkrun.api._models import RunOptions, RunPlan, RunResult

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.scheduler import RankAssignment

logger = logging.getLogger(__name__)


@contextmanager
def _launch_errors(source: str):
    """Keep both launch routes within the same public error contract."""
    try:
        yield
    except SparkrunError:
        raise
    except Exception as error:
        raise SparkrunError("%s failed: %s" % (source, error)) from error


def _observe_plan_hardware(cluster, hosts, *, sctx, dry_run):
    """Discover host facts once, before scheduling; preserve saved policy."""
    from sparkrun.orchestration.executor import cluster_status_scope

    if dry_run or cluster_status_scope(cluster, config=sctx.config, v=sctx.variables) != "host":
        return cluster, {}
    from sparkrun.core.hardware import HostHardware
    from sparkrun.core.hardware_probe import probe_hosts
    from sparkrun.core.hardware_observations import apply_hardware_observations
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    ssh_kwargs = build_ssh_kwargs(sctx.config)
    if cluster.user:
        ssh_kwargs = {**ssh_kwargs, "ssh_user": cluster.user}
    results = probe_hosts(list(hosts), ssh_kwargs=ssh_kwargs, mgmt_interface=cluster.mgmt_interface)
    observations = {host: results.get(host, HostHardware(notes="hardware probe returned no result")) for host in hosts}
    detected = {}
    for host, hardware in observations.items():
        if hardware.source == "detected":
            detected[host] = hardware
        else:
            logger.warning(
                "Host %s hardware probe unavailable (%s); using configured inventory or platform assumptions",
                host,
                hardware.notes or "no detected hardware",
            )
    if detected:
        cluster = apply_hardware_observations(cluster, detected, list(detected))
    return cluster, observations


def plan(options: RunOptions, *, sctx: "SparkrunContext | None" = None) -> RunPlan:
    """Decide *what* the launch described by *options* would do, without doing it.

    Resolves recipe / cluster / runtime, prepares the transport, runs the
    hardware probe before scheduling against live occupancy, and composes the
    launch's identifiers.  Returns a :class:`RunPlan`; changes no cluster state.

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
    # ssh alias) BEFORE the planning hardware/occupancy probes below.
    # Transport preparation is a no-op for plain-SSH clusters.
    from sparkrun.api._resolve import scope_operation

    sctx, _ = scope_operation(cluster_def, sctx=sctx, dry_run=options.dry_run)
    config = sctx.config

    recipe = resolve_recipe(options.recipe, sctx=sctx, overrides=options.overrides)
    hosts = list(cluster_def.hosts)
    runtime = resolve_runtime(recipe, sctx=sctx)
    from sparkrun.api._resolve import resolve_operation_target

    with _launch_errors("executor target resolution"):
        cluster_def, executor_target = resolve_operation_target(options, recipe=recipe, runtime=runtime, cluster=cluster_def, sctx=sctx)
        from sparkrun.core._executor_destination import resolve_destination_user
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        if executor_target.user_scoped:
            user = resolve_destination_user(executor_target, hosts, build_ssh_kwargs(config))
            cluster_def = replace(cluster_def, user=user)
            sctx = sctx.for_cluster(cluster_def)
            config = sctx.config
    from sparkrun.core.readiness import validate_readiness_policy

    try:
        validate_readiness_policy(config=config, recipe=recipe, runtime=runtime)
    except ValueError as error:
        raise SparkrunError(str(error)) from error

    # Hardware facts must reach both placement and the pre-launch summary.
    # The raw observations travel with the plan so launch/strategies can reuse
    # drivers and network discovery without another SSH sweep.
    with _launch_errors("hardware discovery"):
        cluster_def, host_hardware = _observe_plan_hardware(cluster_def, hosts, sctx=sctx, dry_run=options.dry_run)
    sctx = sctx.for_cluster(cluster_def)
    config = sctx.config

    # Scheduler selection chain: caller > recipe > cluster > greedy default.
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER, get_scheduler, resolve_scheduler_selector

    effective_scheduler, _scheduler_defaulted = resolve_scheduler_selector(
        cli=options.scheduler,
        recipe=getattr(recipe, "scheduler", None),
        cluster=getattr(cluster_def, "scheduler", None),
    )
    if _scheduler_defaulted:
        logger.debug("No scheduler configured (recipe/cluster); using default %r", FALLBACK_DEFAULT_SCHEDULER)

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

    def _place(candidates, status_snapshot=None):
        return resolve_effective_hosts(
            list(candidates),
            recipe,
            options.overrides,
            cluster_def=cluster_def,
            runtime=runtime,
            sctx=sctx,
            solo=is_solo_request,
            scheduler=effective_scheduler,
            exclude_intent_id=intent_id,
            status_snapshot=status_snapshot,
        )

    # Conditional `overrides:` layers are applied before the placement that
    # uses them (they can change max_model_len / gpu_memory_utilization, which
    # feed its memory estimate), and over hosts that agree on every hardware
    # `when:`. On a mixed cluster that means choosing a group: see
    # _place_with_overrides. The intent and fingerprint above read declared
    # values, so where the launch lands never moves them.
    override_resolution, (host_list, is_solo, notes, placement) = _place_with_overrides(
        recipe, options, _place, runtime=runtime, cluster=cluster_def, hosts=hosts, host_hardware=host_hardware, sctx=sctx
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
    from sparkrun.core._executor_destination import ExecutorDestination

    destination = ExecutorDestination.from_target(executor_target, config.ssh_user)
    if deterministic_placement and destination.known:
        placement_token = derive_placement_token_from_hosts(hosts, destination=destination.placement_key())
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
        host_hardware=host_hardware,
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
        executor_target=executor_target,
        override_resolution=override_resolution,
        _destination=destination,
    )


def _place_with_overrides(recipe, options: RunOptions, place, *, runtime, cluster, hosts, host_hardware, sctx):
    """Apply ``overrides:`` and place, choosing a hardware group on a mixed cluster.

    Returns ``(override_resolution, placement_result)`` where the second item
    is :func:`resolve_effective_hosts`' tuple.

    Launch-wide layers take one value, so the launch must land on hosts that
    agree on every hardware ``when:``. When the candidates split into several
    such groups, **the scheduler picks**: a probe placement over all candidates
    with the declared config decides which group comes first (the one holding
    the probe's rank 0), so occupancy-aware schedulers still prefer idle
    hosts. Then that group's overrides are applied and the launch is placed
    within it, falling back to the remaining groups in cluster order when one
    cannot take it. All placements share one status sweep.

    Every attempt starts from the declared recipe: overrides, resolver output
    (builder) and the metadata a memory estimate writes back are all reset,
    so a failed group leaves nothing behind for the next one.

    A layout that pins hosts is evaluated over the pinned hosts only. If those
    disagree, the split is real and is refused.
    """
    from copy import deepcopy

    from sparkrun.core.recipe_overrides import build_override_context, partition_hosts_by_hardware

    if not getattr(recipe, "overrides", None):
        return None, place(hosts)

    apply_args = dict(runtime=runtime, cluster=cluster, host_hardware=host_hardware)
    pinned = [p.host for p in (getattr(recipe.layout, "placements", None) or ())]
    if pinned:
        return _apply_recipe_overrides_for_plan(recipe, options, hosts=pinned, **apply_args), place(hosts)

    everything = build_override_context(recipe, options.overrides, hosts=list(hosts), **apply_args)
    groups = partition_hosts_by_hardware(recipe.overrides, everything.hosts)
    if len(groups) <= 1:
        return _apply_recipe_overrides_for_plan(recipe, options, hosts=hosts, **apply_args), place(hosts)

    import sparkrun.api as api
    from sparkrun.core.cluster_status import ClusterStatus

    try:
        status_snapshot = api.status(list(hosts), cluster=cluster, sctx=sctx)
    except Exception as error:
        # Same shape _gather_scheduling_inputs builds on failure. Passing None
        # instead would make every placement below sweep (and time out) again.
        logger.debug("overrides: shared status sweep failed: %s", error)
        status_snapshot = ClusterStatus(errors={host: "status query failed" for host in hosts})

    # estimate_vram writes detected facts (kv_dtype, …) back into metadata,
    # and metadata outranks defaults. A probe or a failed group's estimate
    # must not freeze its values into the next attempt.
    declared_metadata = deepcopy(recipe.metadata)

    def _reset_to_declared():
        recipe.restore_declared_values()
        recipe.resolve(recipe._applied_overrides)
        recipe.metadata = deepcopy(declared_metadata)

    _reset_to_declared()
    head = None
    try:
        probe_hosts, _solo, _notes, probe = place(hosts, status_snapshot)
        head = probe.host_for_rank(0) if probe is not None else (probe_hosts[0] if probe_hosts else None)
    except SparkrunError as error:
        # The probe only orders the groups. The declared config may fit nowhere
        # while a group's overrides would; try the groups in cluster order.
        logger.debug("overrides: probe placement failed (%s); trying groups in cluster order", error)
    ordered = sorted(groups, key=lambda group: 0 if head in group else 1)

    failures: list[tuple[tuple[str, ...], SparkrunError]] = []
    for group in ordered:
        _reset_to_declared()
        try:
            resolution = _apply_recipe_overrides_for_plan(recipe, options, hosts=group, **apply_args)
            host_list, is_solo, notes, placement = place(group, status_snapshot)
        except SparkrunError as error:
            failures.append((group, error))
            logger.debug("overrides: hardware group %s cannot take the launch: %s", ", ".join(group), error)
            continue
        notes = [
            "Note: hosts disagree on the recipe's hardware overrides; using the group %s (%d of %d hosts)"
            % (", ".join(group), len(group), len(hosts)),
            *notes,
        ]
        return resolution, (host_list, is_solo, notes, placement)

    _reset_to_declared()
    detail = "; ".join("[%s] %s" % (", ".join(group), error) for group, error in failures)
    message = "no group of hosts that agree on the recipe's hardware overrides can take this launch: %s" % detail
    capacity = [error for _group, error in failures if isinstance(error, InsufficientCapacity)]
    if len(capacity) == len(failures):
        last = capacity[-1]
        raise InsufficientCapacity(
            message,
            status=getattr(last, "status", None),
            host_list=list(hosts),
            required=getattr(last, "required", None),
            rejections=tuple(r for error in capacity for r in (getattr(error, "rejections", ()) or ())),
        ) from last
    raise SparkrunError(message) from failures[-1][1]


def _apply_recipe_overrides_for_plan(recipe, options: RunOptions, *, runtime, cluster, hosts, host_hardware):
    """Evaluate and apply the recipe's ``overrides:`` for this launch (no-op without any)."""
    from sparkrun.core.recipe_overrides import OverrideConflictError, apply_recipe_override_layers, build_override_context

    if not getattr(recipe, "overrides", None):
        return None
    original_runtime = recipe.runtime
    context_args = dict(runtime=runtime, cluster=cluster, hosts=list(hosts), host_hardware=host_hardware, solo=bool(options.solo))
    try:
        ctx = build_override_context(recipe, options.overrides, **context_args)
        resolution = apply_recipe_override_layers(recipe, ctx)
    except OverrideConflictError as error:
        raise SparkrunError(str(error)) from error
    # Re-run the resolver chain every time, matched or not: applying restores
    # the declared values first, and resolver output (the builder, notably) must
    # follow whatever is now in effect rather than a previous application's.
    recipe.resolve(recipe._applied_overrides)
    if not resolution.matched:
        return resolution
    # What the `when:` clauses matched on must still hold once the layers are
    # in: a default can steer the resolver chain (distributed_executor_backend
    # picks vllm-ray) or the world size (sglang's enable_dp_attention), and the
    # runtime plugin and shape used here were resolved before. Re-derive both
    # rather than enumerating every key that could move them.
    if recipe.runtime != original_runtime:
        raise SparkrunError(
            "overrides %s change the resolved runtime (%s → %s); set runtime: explicitly or move that setting out of overrides"
            % (list(resolution.matched), original_runtime, recipe.runtime)
        )
    after = build_override_context(recipe, options.overrides, declared=False, **context_args)
    if after.shape != ctx.shape:
        changed = sorted(k for k in ctx.shape if ctx.shape[k] != after.shape.get(k))
        raise SparkrunError(
            "overrides %s change the launch shape (%s) that their own `when:` was evaluated against; "
            "set parallelism in defaults or on the CLI instead" % (list(resolution.matched), ", ".join(changed))
        )
    for line in resolution.describe():
        logger.debug("%s", line)
    return resolution


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

    sctx = resolve_sctx(sctx)
    from sparkrun.core.installed_plugins import RequiredIntegrationError, require_integrations

    try:
        require_integrations()
    except RequiredIntegrationError as error:
        raise IntegrationUnavailable(str(error)) from error
    started_at = time.time()
    config = sctx.config

    # ``_build_plan`` is a module-level alias for :func:`plan`, needed because
    # the ``plan`` parameter shadows the function name in this scope.
    if plan is None:
        plan = _build_plan(options, sctx=sctx)

    recipe = plan.recipe
    runtime = plan.runtime
    from sparkrun.core.readiness import validate_readiness_policy

    try:
        validate_readiness_policy(config=config, recipe=recipe, runtime=runtime)
    except ValueError as error:
        raise SparkrunError(str(error)) from error
    cluster_def = plan.cluster
    if plan._destination is not None and plan._destination.user_scoped:
        cluster_def = replace(cluster_def, user=plan._destination.ssh_user)
    if plan.executor_target is not None:
        options = replace(
            options,
            executor=plan.executor_target.executor,
            executor_config={**options.executor_overrides(), **plan.executor_target.overrides},
        )
    sctx = sctx.for_cluster(cluster_def)
    config = sctx.config
    hosts = list(plan.candidate_hosts)
    host_list = list(plan.host_list)
    is_solo = plan.is_solo
    placement = plan.placement
    effective_scheduler = plan.scheduler_selector
    intent_id = plan.intent_id
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

    # Library execution must not fall through to the legacy TTY hook prompts.
    # Keep the same local/registry trust policy and reject before preparation or
    # replacement. Frontends supply explicit authorization via options.trust.
    from sparkrun.core.launcher import resolve_recipe_trust

    if any(getattr(recipe, name, None) for name in ("pre_exec", "post_exec", "post_commands")):
        if not resolve_recipe_trust(recipe, options.trust, sctx=sctx):
            raise SparkrunError("Recipe hooks require explicit authorization: pass RunOptions(trust=True) (CLI: --trust).")

    # Recipe-owned execution strategies are selected only from top-level items
    # present in this recipe.  Preparation happens before the shared launcher
    # starts pulling images or distributing a model, and therefore before its
    # core-owned replacement barrier can evict a serving workload.
    from sparkrun.core.execution import ExecutionContext, resolve_recipe_execution, run_preparation_steps
    from sparkrun.core.timing import Timeline, timed

    if sctx.timing is None:
        sctx.timing = Timeline()

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
        if execution_strategy is not None and prepared_execution is None:
            raise ValueError("execution strategy must return a prepared execution")
        if execution_strategy is not None and prepared_execution is not None and prepared_execution.strategy != execution_strategy.name:
            raise ValueError(
                "execution strategy prepared itself as %r, expected %r" % (prepared_execution.strategy, execution_strategy.name)
            )
        if prepared_execution is not None and prepared_execution.host_hardware:
            from sparkrun.core.hardware_observations import apply_hardware_observations

            cluster_def = apply_hardware_observations(cluster_def, prepared_execution.host_hardware, host_list, placement)
            plan = replace(plan, cluster=cluster_def)
            sctx = sctx.for_cluster(cluster_def)
            config = sctx.config
            execution_context = replace(execution_context, plan=plan, sctx=sctx)
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
    # The scoped observation from the eviction sweep, or ``None`` when no
    # sweep happened (dry run, or the status query failed).  Consumed after the
    # launch by the metadata prune, which must never delete a live workload's
    # metadata and so refuses to run at all without a trustworthy snapshot.
    observed_running: RunningSnapshot | None = None

    replacement_attempted = False

    def _evict_before_start(*, executor=None) -> None:
        nonlocal replacement_attempted, observed_running
        if replacement_attempted:
            return
        replacement_attempted = True
        replacement_cluster = cluster_def
        if executor is not None:
            from dataclasses import asdict

            replacement_cluster = replace(cluster_def, executor=executor.executor_name, executor_config=asdict(executor.config))
        _, running = _evict_superseded_deployments(
            intent_id=intent_id,
            cluster_id_for_launch=cluster_id_for_launch,
            candidate_hosts=hosts,
            target_hosts=host_list,
            cluster_def=replacement_cluster,
            config=config,
            sctx=sctx,
            **({"strict": True, "include_current": True} if executor is not None else {}),
        )
        observed_running = running

    # An enabled executor plugin can own its API launch path. Core supplies
    # the resolved context and preserves deployment replacement semantics.
    from sparkrun.core.run_handlers import registered_run_handlers

    run_handlers = registered_run_handlers(config)
    if run_handlers:
        from sparkrun.orchestration.executor import ExecutorUnavailableError, resolve_executor_name

        try:
            _executor_name = resolve_executor_name(
                cli_overrides=options.executor_overrides(),
                recipe=recipe,
                cluster=cluster_def,
                runtime=runtime,
                config=config,
                v=sctx.variables,
            )
        except ExecutorUnavailableError:
            _executor_name = None
        handler = run_handlers.get(_executor_name) if _executor_name is not None else None
        if handler is not None:
            if execution_strategy is not None:
                raise SparkrunError(
                    "execution strategy %r does not support the %r executor launch path" % (execution_strategy.name, _executor_name)
                )
            with _launch_errors("executor %r launch" % _executor_name):
                result = handler.run(
                    options,
                    sctx,
                    plan=plan,
                    started_at=started_at,
                    before_start=None if options.dry_run else _evict_before_start,
                )
                return _complete_run_result(result, plan=plan, options=options, sctx=sctx, started_at=started_at)

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
        "auto_port": options.auto_port,
        "ray_port": options.ray_port,
        "dashboard_port": options.dashboard_port,
        "dashboard": options.dashboard,
        "init_port": options.init_port,
        "executor_config": options.executor_overrides(),
        "extra_docker_opts": list(options.extra_docker_opts) if options.extra_docker_opts else None,
        "rootless": not options.rootful,
        "auto_user": not options.rootful,
        "cluster": cluster_def,
        "placement": placement,
        "trust": options.trust,
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
        "hardware_observations": {host: plan.host_hardware[host] for host in host_list if host in plan.host_hardware},
    }

    # 5. Launch.
    with _launch_errors("launch_inference"):
        result = launch_inference(**launch_kwargs)

    # 6. Build RunResult.
    metadata: dict[str, Any] = {
        "recipe": getattr(recipe, "qualified_name", None) or getattr(recipe, "name", None),
        "model": getattr(recipe, "model", None),
        "container_image": result.container_image,
        "serve_port": result.serve_port,
        "effective_cache_dir": result.effective_cache_dir,
    }
    if plan.host_hardware or (prepared_execution is not None and prepared_execution.host_hardware):
        from sparkrun.core.hardware_observations import hardware_evidence

        metadata["hardware_evidence"] = {host: hardware_evidence(cluster_def, host, placement) for host in host_list}
    if result.recipe_ref:
        metadata["recipe_ref"] = result.recipe_ref
    if result.runtime_info:
        metadata["runtime_info"] = dict(result.runtime_info)

    run_result = RunResult(
        cluster_id=result.cluster_id,
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
    run_result = _complete_run_result(run_result, plan=plan, options=options, sctx=sctx, started_at=started_at)
    _prune_stale_job_metadata(
        config,
        observed_running=observed_running,
        keep=(run_result.cluster_id,),
        sctx=sctx,
    )

    from sparkrun.telemetry import emit_run_telemetry

    emit_run_telemetry(config, result=run_result, recipe=recipe, cluster=cluster_def, options=options)
    return run_result


def _complete_run_result(result: RunResult, *, plan: RunPlan, options: RunOptions, sctx, started_at: float) -> RunResult:
    """Populate shared launch metadata once, for core and native handlers.

    The actual substrate identity/hosts/command remain the launcher's outcome.
    Reuse preserves verified existing metadata: a proposed plan cannot describe
    the existing deployment's fingerprint or claim a new launch timeline.
    """
    from sparkrun.orchestration.job_metadata import parse_cluster_id

    if not isinstance(result, RunResult):
        raise TypeError("Run handlers must return RunResult")
    try:
        intent_id, placement_token = parse_cluster_id(result.cluster_id)
    except ValueError:
        intent_id, placement_token = (
            (result.intent_id, result.placement_token) if result.already_running else (plan.intent_id, plan.placement_token)
        )
    return replace(
        result,
        intent_id=intent_id,
        placement_token=placement_token,
        recipe_fingerprint=result.recipe_fingerprint if result.already_running else plan.recipe_fingerprint,
        timeline=result.timeline if result.already_running or result.timeline is not None else sctx.timing,
        started_at=started_at,
        dry_run=options.dry_run,
    )


def _prune_stale_job_metadata(config, *, observed_running: "RunningSnapshot | None", keep: tuple[str, ...], sctx) -> None:
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
    if not isinstance(observed_running, RunningSnapshot) or not observed_running.coverage:
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
            protected_cluster_ids=set(keep),
            observation=observed_running,
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

    from sparkrun.orchestration.job_metadata import load_job_metadata

    saved = load_job_metadata(match.cluster_id, cache_dir=str(sctx.config.cache_dir)) or {}
    return RunResult(
        cluster_id=match.cluster_id,
        intent_id=match.intent_id,
        placement_token=matched_token,
        host_list=tuple(match.hosts),
        placement=None,
        scheduler=plan.scheduler or _resolve_scheduler_name(plan.scheduler_selector, sctx),
        runtime=match.runtime or plan.runtime.runtime_name,
        executor=saved.get("executor", ""),
        recipe_fingerprint=saved.get("recipe_fingerprint", ""),
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
    strict: bool = False,
    include_current: bool = False,
) -> "tuple[list[str], RunningSnapshot | None]":
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

    Discovery and teardown are best-effort by default so they cannot block a
    normal launch that may still succeed.  Callers that must bind the same
    host resources before their own launcher starts can set ``strict=True``;
    then an unverified replacement aborts instead of predictably colliding
    with the earlier workload.

    Returns:
        ``(evicted, observation)`` — the cluster_ids torn down (empty when
        there was nothing to do), and a ``RunningSnapshot`` with explicit
        executor/destination coverage, or ``None`` when the query failed.  The second element
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
        if strict:
            raise RuntimeError("could not query cluster status before workload replacement: %s" % e) from e
        logger.debug("Could not query cluster status for eviction; skipping: %s", e)
        return [], None

    if strict and status.observation_errors:
        raise RuntimeError("could not query cluster status before workload replacement: %s" % status.observation_errors)

    prefix = "sparkrun_%s_" % intent_id
    target = set(target_hosts)
    # cluster_id -> every host in the snapshot it occupies (insertion-ordered).
    occupied: dict[str, list[str]] = {}
    overlapping: list[str] = []
    for entry in status.hosts:
        for workload in entry.workloads:
            cid = workload.cluster_id
            if cid == cluster_id_for_launch and not include_current:
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
            # The host subset may contain only node_3 of a former four-node
            # job. Preserve observed names rather than renumbering survivors.
            result = api.stop(cluster_id=cid, hosts=occupied[cid], cluster=cluster_def, discovered=status, sctx=sctx)
        except Exception as e:
            if strict:
                raise RuntimeError("could not stop earlier deployment %s: %s" % (cid, e)) from e
            logger.warning("Could not stop earlier deployment %s: %s — it may still hold GPU memory", cid, e)
            continue
        if result.hosts_failed:
            if strict:
                raise RuntimeError("teardown of earlier deployment %s was not confirmed on %s" % (cid, ", ".join(result.hosts_failed)))
            logger.warning(
                "Teardown of earlier deployment %s did not confirm on %s — it may still hold GPU memory",
                cid,
                ", ".join(result.hosts_failed),
            )
        evicted.append(cid)
    return evicted, status.observation


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
