"""Composable cluster orchestration helpers.

Provides a :class:`ClusterContext` dataclass and reusable helper
functions that runtimes compose to build their specific cluster
launch flows.  The :func:`run_native_cluster` function implements
the full 7-step native orchestration used by SGLang and vLLM
distributed.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.backend_select import BackendBundle
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.placement import RankAssignment
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.orchestration.executor import Executor
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)


def _config_ssh_cap(config: "SparkrunConfig | None") -> int | None:
    """Return the SSH fan-out cap from *config*, or ``None`` for the default."""
    return config.max_parallel_ssh if config is not None else None


@dataclass
class ClusterContext:
    """Resolved state for a cluster launch.

    Captures the common fields every cluster orchestration needs,
    avoiding repetitive setup boilerplate across runtimes.
    """

    hosts: list[str]
    head_host: str
    worker_hosts: list[str]
    num_nodes: int
    ssh_kwargs: dict
    volumes: dict[str, str]
    all_env: dict[str, str]
    cluster_id: str
    image: str
    dry_run: bool
    config: SparkrunConfig | None
    topology: str | None = None
    cluster: ClusterDefinition | None = None
    """Named cluster definition (Phase X threading).

    When set, per-host hardware metadata is available via
    :meth:`hardware_for`.  ``None`` preserves the legacy host-list-only
    behavior so callers that pre-date the threading pass still work.
    """

    placement: RankAssignment | None = None
    """Rank-to-host placement computed via :func:`compute_placement`.

    ``None`` for callers that haven't threaded the cluster through;
    runtimes that consume it must fall back to ``enumerate(hosts)``.
    """

    def hardware_for(self, host: str):
        """Return per-host :class:`HostHardware` (DGX Spark default when unknown)."""
        from sparkrun.core.hardware import default_dgx_spark_hardware

        if self.cluster is not None:
            return self.cluster.hardware_for(host)
        return default_dgx_spark_hardware()

    @classmethod
    def build(
        cls,
        runtime: RuntimePlugin,
        hosts: list[str],
        image: str,
        cluster_id: str,
        env: dict[str, str] | None,
        cache_dir: str | None,
        config: SparkrunConfig | None,
        dry_run: bool,
        topology: str | None = None,
        *,
        cluster: ClusterDefinition | None = None,
        recipe: Recipe | None = None,
        placement: "RankAssignment | None" = None,
    ) -> ClusterContext:
        """Build context from runtime hooks and config.

        Replaces the ~8-line setup boilerplate repeated in every
        runtime's ``_run_cluster`` method.

        When *placement* is provided, it is used verbatim — this is the
        path ``api.run`` and the refactored launcher follow so the
        scheduler runs exactly once per launch.  When *placement* is
        ``None``, and *cluster* + *recipe* are both available, the
        method recomputes placement internally for back-compat with
        callers that haven't been threaded yet.
        """
        from sparkrun.orchestration.primitives import build_ssh_kwargs, build_volumes, resolved_model_volume
        from sparkrun.utils import merge_env

        num_nodes = len(hosts)
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir, extra={**runtime.get_extra_volumes(), **resolved_model_volume(recipe)})
        runtime_env = runtime.get_cluster_env(head_ip="<pending>", num_nodes=num_nodes)
        all_env = merge_env(
            runtime.get_common_env(),
            runtime_env,
            env,
            runtime.get_extra_env(),
        )

        if placement is None and cluster is not None and recipe is not None:
            try:
                import dataclasses

                from sparkrun.core.limits import resolved_hardware_for_scheduling
                from sparkrun.core.parallelism import extract_parallelism
                from sparkrun.core.scheduler import (
                    SchedulingRequest,
                    get_scheduler,
                    resolve_scheduler_selector,
                )

                config_chain = recipe.build_config_chain()
                parallelism = extract_parallelism(config_chain)
                # Bake the runtime-derived rank count so this back-compat
                # fallback matches the authoritative ``api._hosts`` scheduler
                # path (which uses ``runtime.world_size``, not raw tp·pp·dp).
                try:
                    total_ranks = runtime.world_size(parallelism, recipe=recipe, cluster=cluster)
                    if isinstance(total_ranks, int) and total_ranks > 0:
                        parallelism = dataclasses.replace(parallelism, total_ranks=total_ranks)
                except Exception:
                    logger.debug("world_size override unavailable for fallback placement", exc_info=True)
                # Pack against *capped* usable memory (same caps the scheduler
                # applies) rather than nominal memory_gb, so the fallback does
                # not over-commit GPU memory on a host.
                capped_hw = resolved_hardware_for_scheduling(cluster, list(hosts))
                # Honour the cluster/recipe-configured scheduler rather than
                # hardcoding greedy, so a cluster set to ``occupancy-sparse`` is
                # not silently downgraded on this fallback path.  No live
                # ClusterStatus is gathered here (this back-compat path runs only
                # when the caller didn't pre-thread placement), so occupancy-aware
                # schedulers degrade to their greedy whole-GPU pack — but through
                # the configured plugin, keeping the algorithm consistent with the
                # authoritative ``api._hosts`` path.
                recipe_sched = getattr(recipe, "scheduler", None)
                cluster_sched = getattr(cluster, "scheduler", None)
                selector, _defaulted = resolve_scheduler_selector(
                    recipe=recipe_sched if isinstance(recipe_sched, str) else None,
                    cluster=cluster_sched if isinstance(cluster_sched, str) else None,
                )
                request = SchedulingRequest(
                    parallelism=parallelism,
                    hosts=tuple(hosts),
                    host_hardware=capped_hw,
                    layout=recipe.layout,
                )
                placement = get_scheduler(selector).schedule(request).assignment
            except Exception as e:
                logger.warning("Placement computation skipped: %s", e)

        return cls(
            hosts=hosts,
            head_host=hosts[0],
            worker_hosts=hosts[1:],
            num_nodes=num_nodes,
            ssh_kwargs=ssh_kwargs,
            volumes=volumes,
            all_env=all_env,
            cluster_id=cluster_id,
            image=image,
            dry_run=dry_run,
            config=config,
            topology=topology,
            cluster=cluster,
            placement=placement,
        )


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def _stop_ranked_containers_parallel(
    ctx: ClusterContext,
    executor: Executor,
    target_hosts: list[str],
) -> list[str]:
    """Stop ranked node containers on *target_hosts* in parallel (best-effort).

    Ranks are assigned by position in ``ctx.hosts`` so the stopped container
    name matches what was launched, even when *target_hosts* is a subset.
    Returns the hosts whose stop command did not confirm (empty on success;
    always empty in dry-run).
    """
    from sparkrun.orchestration.ssh import resolve_parallel_cap, run_remote_command

    rank_by_host = {host: rank for rank, host in enumerate(ctx.hosts)}
    if not target_hosts:
        return []

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(target_hosts), _config_ssh_cap(ctx.config))) as pool:
        futures = {}
        for host in target_hosts:
            container_name = executor.node_container_name(ctx.cluster_id, rank_by_host.get(host, 0))
            futures[
                pool.submit(
                    run_remote_command,
                    host,
                    executor.stop_cmd(container_name),
                    timeout=30,
                    dry_run=ctx.dry_run,
                    quiet=True,
                    **ctx.ssh_kwargs,
                )
            ] = host
        for future in as_completed(futures):
            host = futures[future]
            try:
                result = future.result()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Ranked cleanup raised on %s: %s", host, e)
                failed.append(host)
                continue
            if not result.success and not ctx.dry_run:
                failed.append(host)

    if failed and not ctx.dry_run:
        logger.warning(
            "Container cleanup did not confirm on %d host(s): %s — "
            "these may still hold VRAM; check with 'sparkrun stop %s' or 'docker ps'.",
            len(failed),
            ", ".join(failed),
            ctx.cluster_id,
        )
    return failed


def cleanup_ranked_containers(ctx: ClusterContext, executor: Executor) -> list[str]:
    """Stop ranked node containers (``{cluster_id}_node_{rank}``) on all hosts.

    Parallel and best-effort; returns the hosts that did not confirm cleanup.
    """
    return _stop_ranked_containers_parallel(ctx, executor, list(ctx.hosts))


def cleanup_named_containers(ctx: ClusterContext, container_names: list[str]) -> None:
    """Stop named containers on all hosts."""
    from sparkrun.orchestration.primitives import cleanup_containers

    cleanup_containers(ctx.hosts, container_names, ssh_kwargs=ctx.ssh_kwargs, dry_run=ctx.dry_run)


def cleanup_after_failure(
    ctx: ClusterContext,
    executor: Executor,
    *,
    reason: str,
    container_names: list[str] | None = None,
    hosts: list[str] | None = None,
) -> None:
    """Stop containers after a launch failure, respecting ``--no-rm``.

    When ``--no-rm`` was passed (``executor.config.auto_remove == False``), the
    user explicitly asked to keep containers around for debugging — emit a
    warning naming the cluster id and skip cleanup.  Otherwise, stop the
    ranked node containers on every cluster host, or the explicitly named
    containers when ``container_names`` is provided (solo / partial-cluster).

    ``hosts`` restricts cleanup to a subset (e.g., only the hosts that had
    containers launched at the point of failure).  Errors during cleanup are
    logged at warning level and swallowed so the original failure remains the
    primary signal to the caller.
    """
    if not getattr(executor.config, "auto_remove", True):
        logger.warning(
            "Launch failed (%s). Containers left running because --no-rm was set. Clean up manually: sparkrun stop %s",
            reason,
            ctx.cluster_id,
        )
        return

    logger.info("Cleaning up containers after launch failure (%s)...", reason)
    try:
        if container_names is not None:
            from sparkrun.orchestration.primitives import cleanup_containers

            cleanup_containers(
                hosts or ctx.hosts,
                container_names,
                ssh_kwargs=ctx.ssh_kwargs,
                dry_run=ctx.dry_run,
                max_workers=_config_ssh_cap(ctx.config),
            )
            return
        # Ranked cleanup. Use the supplied host subset when provided so we
        # don't issue stop commands to hosts that never had a container.
        # Parallel + per-host reporting via the shared helper.
        _stop_ranked_containers_parallel(ctx, executor, list(hosts or ctx.hosts))
    except Exception as e:
        logger.warning("Cleanup encountered errors (continuing): %s", e)


def cleanup_solo_after_failure(
    executor: Executor,
    host: str,
    container_name: str,
    ssh_kwargs: dict,
    *,
    dry_run: bool,
    cluster_id: str,
    reason: str,
) -> None:
    """``cleanup_after_failure`` for the solo path (no ``ClusterContext``)."""
    if not getattr(executor.config, "auto_remove", True):
        logger.warning(
            "Launch failed (%s). Container left running because --no-rm was set. Clean up manually: sparkrun stop %s",
            reason,
            cluster_id,
        )
        return

    logger.info("Cleaning up container after launch failure (%s)...", reason)
    try:
        from sparkrun.orchestration.ssh import run_remote_command

        run_remote_command(
            host,
            executor.stop_cmd(container_name),
            timeout=30,
            dry_run=dry_run,
            **ssh_kwargs,
        )
    except Exception as e:
        logger.warning("Cleanup encountered errors (continuing): %s", e)


def dump_serve_log(host: str, container: str, ssh_kwargs: dict, *, dry_run: bool = False) -> None:
    """Log ``/tmp/sparkrun_serve.log`` from inside a container at ERROR level.

    The serve process redirects its stdout/stderr to a file inside the
    container, so ``docker logs`` (which only sees PID 1's stdio) is
    structurally blind to it.  This helper bridges the gap by running
    ``docker exec ... cat`` over SSH and emitting whatever it finds.
    """
    from sparkrun.orchestration.ssh import run_remote_command

    result = run_remote_command(
        host,
        f"docker exec {container} cat /tmp/sparkrun_serve.log 2>/dev/null || true",
        timeout=30,
        dry_run=dry_run,
        **ssh_kwargs,
    )
    if dry_run:
        return
    content = (result.stdout or "").rstrip()
    if not content:
        logger.error(
            "No content in /tmp/sparkrun_serve.log for %s on %s (container may have exited or serve never wrote to the file).",
            container,
            host,
        )
        return
    logger.error("Serve log /tmp/sparkrun_serve.log on %s (%s):", host, container)
    for line in content.splitlines():
        logger.error("  %s", line)


# ---------------------------------------------------------------------------
# InfiniBand helpers
# ---------------------------------------------------------------------------


def _refuse_unsupported_collectives(ctx: ClusterContext) -> None:
    """Raise if a placed cluster spans collective backends that aren't safe to mix.

    Walks the placed hosts (or every cluster host when no placement was
    threaded) and surfaces an actionable error when:

    - The placed set spans more than one accelerator vendor.  NCCL/RCCL/HCCL
      cannot share a process group, so the launch must either use a single
      vendor's collective backend or split work via an explicit recipe layout
      so each replica stays vendor-homogeneous.
    - The single placed vendor maps to a backend scaffold that isn't yet
      implemented (RCCL/HCCL today) — surface that now rather than waiting
      for the per-rank NCCL env to silently mislead a worker process.

    A single-vendor NVIDIA placement (the default DGX path, or any
    fingerprinted NVIDIA host) returns silently.
    """
    if ctx.cluster is None:
        return

    placed_hosts = ctx.placement.hosts_used if ctx.placement is not None else tuple(ctx.cluster.hosts)
    if not placed_hosts:
        return

    vendors: set[str] = set()
    for host in placed_hosts:
        hw = ctx.cluster.hardware_for(host)
        for a in hw.accelerators:
            if a.vendor:
                vendors.add(a.vendor)

    if not vendors or vendors == {"nvidia"}:
        return  # NCCL is the default; no extra collective bootstrap.

    if len(vendors) > 1:
        raise RuntimeError(
            "Heterogeneous-vendor cluster %s spans %s.  Sparkrun cannot compose a single "
            "collective env across NCCL/RCCL/HCCL — split work via recipe.layout so each "
            "replica stays vendor-homogeneous." % (sorted(placed_hosts), sorted(vendors))
        )

    from sparkrun.orchestration.collectives import get_backend

    vendor = next(iter(vendors))
    backend = get_backend(vendor)
    # Probe whether the backend has a real env implementation (NCCL today;
    # RCCL/HCCL raise NotImplementedError).  Touch the public API once; if it
    # raises, re-surface as a clear launch-time error rather than waiting for
    # the (eventual) per-host script generation to discover it.
    try:
        backend.env_for_host({}, topology=None)
    except NotImplementedError as e:
        raise RuntimeError(
            "%s backend not yet implemented for %s hosts (%s).  Contribute an "
            "implementation in sparkrun/orchestration/collectives/ or pin the cluster "
            "to NVIDIA hosts." % (backend.name.upper(), vendor, e)
        ) from e


def resolve_comm_env(
    ctx: ClusterContext,
    comm_env: ClusterCommEnv | None,
    backends: "dict[str, BackendBundle] | None" = None,
) -> ClusterCommEnv:
    """Resolve the cluster comm env: reuse pre-detected or probe.

    Emits per-host env via ``backends[host].collective.env_for_host``
    when *backends* is supplied; otherwise falls back to the legacy
    NCCL generator (byte-identical for NVIDIA hosts).
    """
    from sparkrun.orchestration.comm_env import ClusterCommEnv as _CCE

    _refuse_unsupported_collectives(ctx)

    if comm_env is not None:
        logger.info("Using pre-detected comm env (%d vars)", len(comm_env))
        return comm_env

    from sparkrun.orchestration.infiniband import detect_ib_for_hosts

    logger.info("Detecting InfiniBand on %d host(s)...", len(ctx.hosts))
    ib_result = detect_ib_for_hosts(
        ctx.hosts,
        ssh_kwargs=ctx.ssh_kwargs,
        dry_run=ctx.dry_run,
        topology=ctx.topology,
        backends=backends,
    )
    if ib_result.comm_env.is_empty():
        logger.info("  No InfiniBand detected, using default networking")
        return _CCE.empty()
    return ib_result.comm_env


def detect_ib_with_ips(
    ctx: ClusterContext,
    comm_env: ClusterCommEnv | None,
    ib_ip_map: dict[str, str] | None,
    backends: "dict[str, BackendBundle] | None" = None,
) -> tuple[ClusterCommEnv, dict[str, str]]:
    """Detect comm env and IP map (for runtimes needing IB addresses).

    Args:
        ctx: Cluster context.
        comm_env: Pre-detected comm env (skip probe if non-None).
        ib_ip_map: Pre-detected IB IPs (preserved if non-None).
        backends: Optional per-host :class:`BackendBundle`.  When
            provided, IB env vars are emitted through the host's
            collective backend (NCCL/RCCL/HCCL); otherwise the legacy
            NCCL generator is used (byte-identical for NVIDIA hosts).

    Returns ``(comm_env, ib_ip_map)``.
    """
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts

    _refuse_unsupported_collectives(ctx)

    if ib_ip_map is None:
        ib_ip_map = {}
    if comm_env is not None:
        logger.info("Using pre-detected comm env (%d vars)", len(comm_env))
        if ib_ip_map:
            logger.info("  Pre-detected IB IPs for %d host(s)", len(ib_ip_map))
        return comm_env, ib_ip_map

    logger.info("Detecting InfiniBand on all hosts...")
    ib_result = detect_ib_for_hosts(
        ctx.hosts,
        ssh_kwargs=ctx.ssh_kwargs,
        dry_run=ctx.dry_run,
        topology=ctx.topology,
        backends=backends,
    )
    return ib_result.comm_env, ib_result.ib_ip_map


# ---------------------------------------------------------------------------
# IP / port helpers
# ---------------------------------------------------------------------------


def detect_head_ip(ctx: ClusterContext) -> str:
    """Detect the management IP of the head host.

    Raises ``RuntimeError`` on failure; returns ``"<HEAD_IP>"`` in
    dry-run mode.
    """
    from sparkrun.orchestration.primitives import detect_host_ip

    return detect_host_ip(ctx.head_host, ssh_kwargs=ctx.ssh_kwargs, dry_run=ctx.dry_run)


def resolve_hosts_for_init(ctx: ClusterContext, head_ip: str) -> list[str]:
    """Return ``ctx.hosts`` with loopback entries replaced by routable IPs.

    The cluster config conventionally lists the control machine's own
    host as ``127.0.0.1`` for SSH convenience.  That loopback must not
    appear in NCCL / torch-distributed master addresses, otherwise
    workers connect to their own loopback instead of the head.
    """
    from sparkrun.orchestration.primitives import detect_host_ip
    from sparkrun.utils import is_local_host

    resolved: list[str] = []
    for h in ctx.hosts:
        if not is_local_host(h):
            resolved.append(h)
            continue
        if h == ctx.head_host:
            resolved.append(head_ip)
        else:
            resolved.append(detect_host_ip(h, ssh_kwargs=ctx.ssh_kwargs, dry_run=ctx.dry_run))
    return resolved


def find_port(ctx: ClusterContext, host: str, preferred: int) -> int:
    """Find an available port, avoiding collisions with running instances."""
    from sparkrun.orchestration.primitives import find_available_port

    return find_available_port(host, preferred, ssh_kwargs=ctx.ssh_kwargs, dry_run=ctx.dry_run)


# ---------------------------------------------------------------------------
# Container launch
# ---------------------------------------------------------------------------


def launch_containers_parallel(
    ctx: ClusterContext,
    containers: list[tuple[str, str]],
    executor: Executor,
    comm_env: ClusterCommEnv | None,
    extra_docker_opts: list[str] | None = None,
    *,
    runtime: RuntimePlugin | None = None,
    recipe: Recipe | None = None,
    container_ranks: dict[str, int] | None = None,
) -> int:
    """Launch sleep-infinity containers in parallel.

    Each host receives ``comm_env.get_env(host)`` for its docker ``-e``
    block, so heterogeneous management interfaces (e.g. wired on the
    head, wifi on a worker) each see the right ``*_SOCKET_IFNAME``
    values instead of the head's interface being broadcast cluster-wide.

    When *runtime* and *recipe* are provided, each container is tagged
    with the canonical sparkrun label set via
    :meth:`Executor.workload_labels_for_cluster` so ``docker ps --filter
    "label=sparkrun.intent_id=<x>"`` becomes a working discovery
    surface.  Per-container rank can be supplied via *container_ranks*
    (keyed by container name); when absent the rank label is omitted
    for that container.

    Returns 0 on success, 1 on first failure.
    """
    from sparkrun.orchestration.ssh import resolve_parallel_cap, run_remote_script

    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(containers), _config_ssh_cap(ctx.config))) as pool:
        futures = {}
        for host, cname in containers:
            host_nccl_env = comm_env.get_env(host) if comm_env else None
            rank = (container_ranks or {}).get(cname)
            sparkrun_labels = executor.workload_labels_for_cluster(
                cluster_id=ctx.cluster_id,
                recipe=recipe,
                runtime=runtime,
                rank=rank,
            )
            script = executor.generate_launch_script(
                image=ctx.image,
                container_name=cname,
                command="sleep infinity",
                env=ctx.all_env,
                volumes=ctx.volumes,
                nccl_env=host_nccl_env,
                extra_docker_opts=extra_docker_opts,
                sparkrun_labels=sparkrun_labels or None,
            )
            future = pool.submit(
                run_remote_script,
                host,
                script,
                timeout=120,
                dry_run=ctx.dry_run,
                **ctx.ssh_kwargs,
            )
            futures[future] = (host, cname)

        for future in as_completed(futures):
            host, cname = futures[future]
            result = future.result()
            if not result.success and not ctx.dry_run:
                logger.error("Failed to launch container %s on %s: %s", cname, host, result.stderr[:200])
                return 1

    return 0


# ---------------------------------------------------------------------------
# Pre-serve hooks
# ---------------------------------------------------------------------------


def run_pre_serve_hooks(
    runtime: RuntimePlugin,
    ctx: ClusterContext,
    hosts_containers: list[tuple[str, str]],
    recipe: Recipe | None,
    overrides: dict[str, Any] | None,
    trust: bool = False,
    cache_dir: str | None = None,
) -> None:
    """Build config chain and invoke runtime._pre_serve.

    *trust* controls whether the pre_exec confirmation prompt is
    suppressed (see :meth:`RuntimePlugin._pre_serve`).

    *cache_dir* is the effective HuggingFace cache directory on remote
    hosts, threaded from the launcher so disk-space failure messages
    show the correct path.
    """
    config_chain = recipe.build_config_chain(overrides) if recipe else None
    runtime._pre_serve(
        hosts_containers,
        ctx.ssh_kwargs,
        ctx.dry_run,
        recipe=recipe,
        config_chain=config_chain,
        trust=trust,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Exec helpers
# ---------------------------------------------------------------------------


def exec_serve_on_container(
    ctx: ClusterContext,
    executor: Executor,
    host: str,
    container_name: str,
    command: str,
    *,
    detached: bool = True,
    sparkrun_labels: dict[str, str] | None = None,
) -> int:
    """Generate and run an exec-serve script on a container.

    *sparkrun_labels* is forwarded to
    :meth:`Executor.generate_exec_serve_script`.  For Docker this is a
    no-op (labels live on the parent container created by
    ``launch_containers_parallel``); for K8s the Pod is created here,
    so labels must flow through.

    Returns 0 on success, 1 on failure.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    script = executor.generate_exec_serve_script(
        container_name=container_name,
        serve_command=command,
        env=ctx.all_env,
        detached=detached,
        sparkrun_labels=sparkrun_labels or None,
    )
    result = run_remote_script(
        host,
        script,
        timeout=60,
        dry_run=ctx.dry_run,
        **ctx.ssh_kwargs,
    )
    if not result.success and not ctx.dry_run:
        logger.error("Failed to exec serve on %s (%s, rc=%d):", host, container_name, result.returncode)
        for line in (result.stderr or "").rstrip().splitlines():
            logger.error("  %s", line)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Full native cluster orchestration
# ---------------------------------------------------------------------------


def _attach_foreground(runtime: RuntimePlugin, ctx: ClusterContext, follow: bool) -> None:
    """Block until interrupted, optionally following logs. Stop cluster on exit."""
    logger.info("Foreground mode: press Ctrl+C to stop cluster.")
    try:
        if follow:
            runtime.follow_logs(ctx.hosts, cluster_id=ctx.cluster_id, config=ctx.config)
        else:
            threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Stopping cluster '%s'...", ctx.cluster_id)
    finally:
        runtime.stop(ctx.hosts, cluster_id=ctx.cluster_id, config=ctx.config)


def run_native_cluster(
    runtime: RuntimePlugin,
    ctx: ClusterContext,
    recipe: Recipe | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    comm_env: ClusterCommEnv | None = None,
    ib_ip_map: dict[str, str] | None = None,
    init_port: int = 25000,
    skip_keys: set[str] | frozenset[str] = frozenset(),
    banner_title: str = "Native Cluster Launcher",
    port_label: str = "Init Port",
    node_label: str = "node",
    detached: bool = True,
    follow: bool = True,
    progress=None,
    extra_docker_opts: list[str] | None = None,
    backends: "dict[str, BackendBundle] | None" = None,
    trust: bool = False,
    cache_dir: str | None = None,
) -> int:
    """Orchestrate a multi-node native cluster.

    Shared by SGLang and vLLM distributed.  Uses the two-phase launch
    pattern (sleep infinity + exec) so that ``_pre_serve`` hooks run
    between container startup and serve execution.

    Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Detect head node IP.
        4. Launch ALL containers with ``sleep infinity``.
        5. Run pre-serve hooks (pre_exec) on all containers.
        6. Exec head serve command, wait for init port.
        7. Exec worker serve commands in parallel.
    """
    from sparkrun.orchestration.primitives import wait_for_port
    from sparkrun.orchestration.ssh import (
        run_remote_script,
        start_log_capture,
        stop_log_capture,
    )

    executor = runtime._resolve_executor()

    if progress:
        progress.begin_runtime_steps(7)

    # Step 1: Cleanup
    t0 = time.monotonic()
    if progress:
        progress.step("Cleaning up existing containers")
    else:
        logger.info("Step 1/7: Cleaning up existing containers for cluster '%s'...", ctx.cluster_id)
    cleanup_ranked_containers(ctx, executor)
    logger.info("Step 1/7: Cleanup done (%.1fs)", time.monotonic() - t0)

    # Step 2: InfiniBand detection (also resolves IB IPs for runtimes
    # that opt into IB-routed bootstrap via prefer_ib_for_init_addr).
    t0 = time.monotonic()
    if progress:
        progress.step("Detecting InfiniBand")
    else:
        logger.info("Step 2/7: InfiniBand detection...")
    comm_env, ib_ip_map = detect_ib_with_ips(ctx, comm_env, ib_ip_map, backends=backends)
    logger.info("Step 2/7: IB step done (%.1fs)", time.monotonic() - t0)

    # Step 3: Detect head node IP
    t0 = time.monotonic()
    if progress:
        progress.step("Detecting head node IP")
    else:
        logger.info("Step 3/7: Detecting head node IP on %s...", ctx.head_host)
    try:
        head_ip = detect_head_ip(ctx)
    except RuntimeError as e:
        logger.error("%s", e)
        return 1
    logger.info("  Head IP: %s", head_ip)
    logger.info("Step 3/7: IP detection done (%.1fs)", time.monotonic() - t0)

    # Substitute loopback entries (e.g. ``127.0.0.1`` from cluster config)
    # with routable IPs so NCCL / torch-distributed master addresses are
    # never broadcast as loopback to remote workers.
    resolved_hosts = resolve_hosts_for_init(ctx, head_ip)
    if resolved_hosts != ctx.hosts:
        logger.info("  Resolved init hosts: %s", resolved_hosts)

    # Optional IB-routed bootstrap: overlay IB IPs on head_ip and
    # resolved_hosts when the runtime opts in.  Hosts without an IB
    # entry keep their mgmt-resolved address.
    if runtime.prefer_ib_for_init_addr() and ib_ip_map:
        head_ip = ib_ip_map.get(ctx.head_host, head_ip)
        resolved_hosts = [ib_ip_map.get(orig, fallback) for orig, fallback in zip(ctx.hosts, resolved_hosts)]
        logger.info("  IB-routed init: head=%s, hosts=%s", head_ip, resolved_hosts)

    # Auto-detect available init port
    init_port = find_port(ctx, ctx.head_host, init_port)

    # Print banner after finalizing ports
    runtime._print_cluster_banner(
        banner_title,
        ctx.hosts,
        ctx.image,
        ctx.cluster_id,
        {port_label: init_port},
        ctx.dry_run,
    )

    # Generate per-node commands
    head_command = runtime.generate_node_command(
        recipe=recipe,
        overrides=overrides,
        head_ip=head_ip,
        num_nodes=ctx.num_nodes,
        node_rank=0,
        init_port=init_port,
        skip_keys=skip_keys,
        hosts=resolved_hosts,
        placement=ctx.placement,
    )
    logger.info("Serve command (head, rank 0):")
    for line in head_command.strip().splitlines():
        logger.info("  %s", line)

    # Step 4: Launch ALL containers with sleep infinity
    t0 = time.monotonic()
    if progress:
        progress.step("Launching containers")
    else:
        logger.info("Step 4/7: Launching containers with sleep infinity on all %d host(s)...", ctx.num_nodes)

    all_nodes: list[tuple[str, int, str]] = []
    for rank, host in enumerate(ctx.hosts):
        all_nodes.append((host, rank, executor.node_container_name(ctx.cluster_id, rank)))

    containers = [(host, cname) for host, _rank, cname in all_nodes]
    container_ranks = {cname: rank for _host, rank, cname in all_nodes}
    combined_docker_opts = (runtime.get_extra_docker_opts() or []) + (extra_docker_opts or [])
    rc = launch_containers_parallel(
        ctx,
        containers,
        executor,
        comm_env,
        extra_docker_opts=combined_docker_opts or None,
        runtime=runtime,
        recipe=recipe,
        container_ranks=container_ranks,
    )
    if rc != 0:
        return rc
    logger.info("Step 4/7: All containers launched (%.1fs)", time.monotonic() - t0)

    # Step 5: Pre-serve hooks
    t0 = time.monotonic()
    if progress:
        progress.step("Running pre-serve hooks")
    else:
        logger.info("Step 5/7: Running pre-serve hooks...")
    hosts_containers = [(host, cname) for host, _rank, cname in all_nodes]
    run_pre_serve_hooks(runtime, ctx, hosts_containers, recipe, overrides, trust=trust, cache_dir=cache_dir)
    logger.info("Step 5/7: Pre-serve hooks done (%.1fs)", time.monotonic() - t0)

    # Step 6: Exec head serve command and wait for init port
    t0 = time.monotonic()
    head_container = all_nodes[0][2]
    if progress:
        progress.step("Starting head node serve")
    else:
        logger.info("Step 6/7: Executing serve command on head node (rank 0) %s...", ctx.head_host)
    head_sparkrun_labels = executor.workload_labels_for_cluster(
        cluster_id=ctx.cluster_id,
        recipe=recipe,
        runtime=runtime,
        rank=0,
    )
    head_exec_script = executor.generate_exec_serve_script(
        container_name=head_container,
        serve_command=head_command,
        env=ctx.all_env,
        detached=True,
        sparkrun_labels=head_sparkrun_labels or None,
    )
    head_result = run_remote_script(
        ctx.head_host,
        head_exec_script,
        timeout=60,
        dry_run=ctx.dry_run,
        **ctx.ssh_kwargs,
    )
    if not head_result.success and not ctx.dry_run:
        logger.error("Failed to exec serve on head node %s (rc=%d):", ctx.head_host, head_result.returncode)
        for line in (head_result.stderr or "").rstrip().splitlines():
            logger.error("  %s", line)
        if head_result.stdout:
            logger.error("Head node stdout:")
            for line in head_result.stdout.rstrip().splitlines():
                logger.error("  %s", line)
        cleanup_after_failure(ctx, executor, reason="head serve exec failed")
        return 1

    # Wait for head init port
    if not ctx.dry_run:
        logger.info("  Waiting for head node %s %s:%d...", port_label.lower(), ctx.head_host, init_port)

        log_proc = start_log_capture(ctx.head_host, head_container, ctx.ssh_kwargs)
        try:
            ready = wait_for_port(
                ctx.head_host,
                init_port,
                max_retries=60,
                retry_interval=2,
                ssh_kwargs=ctx.ssh_kwargs,
                dry_run=ctx.dry_run,
                container_name=head_container,
            )
        finally:
            captured = stop_log_capture(log_proc)

        if not ready:
            logger.error("Head node failed to become ready on %s.", ctx.head_host)
            if captured:
                logger.error("Container logs for %s:", head_container)
                for line in captured[-150:]:
                    logger.error("  %s", line)
            # `start_log_capture` uses `docker logs -f`, which is blind to the
            # serve process's redirected output file. Always emit the real log
            # so the user sees the actual error, not just sleep-infinity stdio.
            dump_serve_log(ctx.head_host, head_container, ctx.ssh_kwargs, dry_run=ctx.dry_run)
            cleanup_after_failure(ctx, executor, reason="head node did not become ready")
            return 1
        logger.info("Step 6/7: Head node ready (%.1fs)", time.monotonic() - t0)
    else:
        logger.info("Step 6/7: [dry-run] Would wait for %s %d", port_label.lower(), init_port)

    # Step 7: Exec worker serve commands in parallel
    t0 = time.monotonic()
    if ctx.worker_hosts:
        if progress:
            progress.step("Starting worker nodes")
        else:
            logger.info(
                "Step 7/7: Executing serve on %d worker node(s) on %s...",
                len(ctx.worker_hosts),
                ", ".join(ctx.worker_hosts),
            )
        from sparkrun.orchestration.ssh import resolve_parallel_cap

        with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(ctx.worker_hosts), _config_ssh_cap(ctx.config))) as pool:
            futures = {}
            for i, host in enumerate(ctx.worker_hosts):
                rank = i + 1
                worker_command = runtime.generate_node_command(
                    recipe=recipe,
                    overrides=overrides,
                    head_ip=head_ip,
                    num_nodes=ctx.num_nodes,
                    node_rank=rank,
                    init_port=init_port,
                    skip_keys=skip_keys,
                    hosts=resolved_hosts,
                    placement=ctx.placement,
                )
                worker_container = all_nodes[rank][2]
                worker_sparkrun_labels = executor.workload_labels_for_cluster(
                    cluster_id=ctx.cluster_id,
                    recipe=recipe,
                    runtime=runtime,
                    rank=rank,
                )
                worker_exec_script = executor.generate_exec_serve_script(
                    container_name=worker_container,
                    serve_command=worker_command,
                    env=ctx.all_env,
                    detached=True,
                    sparkrun_labels=worker_sparkrun_labels or None,
                )
                future = pool.submit(
                    run_remote_script,
                    host,
                    worker_exec_script,
                    timeout=60,
                    dry_run=ctx.dry_run,
                    **ctx.ssh_kwargs,
                )
                futures[future] = (host, rank)

            worker_failures: list[tuple[str, int, Any]] = []
            for future in as_completed(futures):
                host, rank = futures[future]
                result = future.result()
                if not result.success and not ctx.dry_run:
                    worker_failures.append((host, rank, result))

            if worker_failures:
                for host, rank, result in worker_failures:
                    logger.error(
                        "Worker rank %d on %s failed to exec serve (rc=%d):",
                        rank,
                        host,
                        result.returncode,
                    )
                    for line in (result.stderr or "").rstrip().splitlines():
                        logger.error("  %s", line)
                cleanup_after_failure(
                    ctx,
                    executor,
                    reason=f"{len(worker_failures)} worker(s) failed serve exec",
                )
                return 1

        if not progress:
            logger.info("Step 7/7: Workers launched (%.1fs)", time.monotonic() - t0)
    else:
        if progress:
            progress.step("Starting worker nodes")
            progress.detail("  No worker hosts, skipping")
        else:
            logger.info("Step 7/7: No worker hosts, skipping")

    runtime._print_connection_info(ctx.hosts, ctx.cluster_id, per_node_logs=True)

    if not detached and not ctx.dry_run:
        _attach_foreground(runtime, ctx, follow)

    return 0
