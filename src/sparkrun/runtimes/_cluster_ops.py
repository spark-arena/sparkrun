"""Composable cluster orchestration helpers.

Provides a :class:`ClusterContext` dataclass and reusable helper
functions that runtimes compose to build their specific cluster
launch flows.  The :func:`run_native_cluster` function implements
the full 7-step native orchestration used by SGLang and vLLM
distributed.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.orchestration.executor import Executor
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)


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
    ) -> ClusterContext:
        """Build context from runtime hooks and config.

        Replaces the ~8-line setup boilerplate repeated in every
        runtime's ``_run_cluster`` method.
        """
        from sparkrun.orchestration.primitives import build_ssh_kwargs, build_volumes
        from sparkrun.utils import merge_env

        num_nodes = len(hosts)
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir, extra=runtime.get_extra_volumes())
        runtime_env = runtime.get_cluster_env(head_ip="<pending>", num_nodes=num_nodes)
        all_env = merge_env(
            runtime.get_common_env(),
            runtime_env,
            env,
            runtime.get_extra_env(),
        )
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
        )


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def cleanup_ranked_containers(ctx: ClusterContext, executor: Executor) -> None:
    """Stop ranked node containers (``{cluster_id}_node_{rank}``) on all hosts."""
    from sparkrun.orchestration.ssh import run_remote_command

    for rank, host in enumerate(ctx.hosts):
        container_name = executor.node_container_name(ctx.cluster_id, rank)
        run_remote_command(
            host,
            executor.stop_cmd(container_name),
            timeout=30,
            dry_run=ctx.dry_run,
            **ctx.ssh_kwargs,
        )


def cleanup_named_containers(ctx: ClusterContext, container_names: list[str]) -> None:
    """Stop named containers on all hosts."""
    from sparkrun.orchestration.primitives import cleanup_containers

    cleanup_containers(ctx.hosts, container_names, ssh_kwargs=ctx.ssh_kwargs, dry_run=ctx.dry_run)


# ---------------------------------------------------------------------------
# InfiniBand helpers
# ---------------------------------------------------------------------------


def resolve_ib_env(
    ctx: ClusterContext,
    comm_env: ClusterCommEnv | None,
) -> ClusterCommEnv:
    """Resolve the cluster comm env: reuse pre-detected or probe.

    Returns a :class:`ClusterCommEnv` carrying both shared and
    per-host inter-node comm env vars.  Per-host entries let
    heterogeneous management interfaces (e.g. wired on the head, wifi
    on a worker) each bind the correct ``*_SOCKET_IFNAME`` values.
    """
    from sparkrun.orchestration.comm_env import ClusterCommEnv as _CCE

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
    )
    if ib_result.comm_env.is_empty():
        logger.info("  No InfiniBand detected, using default networking")
        return _CCE.empty()
    return ib_result.comm_env


def detect_ib_with_ips(
    ctx: ClusterContext,
    comm_env: ClusterCommEnv | None,
    ib_ip_map: dict[str, str] | None,
) -> tuple[ClusterCommEnv, dict[str, str]]:
    """Detect comm env and IP map (for runtimes needing IB addresses).

    Returns ``(comm_env, ib_ip_map)``.
    """
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts

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
) -> int:
    """Launch sleep-infinity containers in parallel.

    Each host receives ``comm_env.get_env(host)`` for its docker ``-e``
    block, so heterogeneous management interfaces (e.g. wired on the
    head, wifi on a worker) each see the right ``*_SOCKET_IFNAME``
    values instead of the head's interface being broadcast cluster-wide.

    Returns 0 on success, 1 on first failure.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    with ThreadPoolExecutor(max_workers=len(containers)) as pool:
        futures = {}
        for host, cname in containers:
            host_nccl_env = comm_env.get_env(host) if comm_env else None
            script = executor.generate_launch_script(
                image=ctx.image,
                container_name=cname,
                command="sleep infinity",
                env=ctx.all_env,
                volumes=ctx.volumes,
                nccl_env=host_nccl_env,
                extra_docker_opts=extra_docker_opts,
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
) -> None:
    """Build config chain and invoke runtime._pre_serve."""
    config_chain = recipe.build_config_chain(overrides) if recipe else None
    runtime._pre_serve(hosts_containers, ctx.ssh_kwargs, ctx.dry_run, recipe=recipe, config_chain=config_chain)


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
) -> int:
    """Generate and run an exec-serve script on a container.

    Returns 0 on success, 1 on failure.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    script = executor.generate_exec_serve_script(
        container_name=container_name,
        serve_command=command,
        env=ctx.all_env,
        detached=detached,
    )
    result = run_remote_script(
        host,
        script,
        timeout=60,
        dry_run=ctx.dry_run,
        **ctx.ssh_kwargs,
    )
    if not result.success and not ctx.dry_run:
        logger.error("Failed to exec serve on %s (%s): %s", host, container_name, result.stderr[:200])
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
            import threading

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
    init_port: int = 25000,
    skip_keys: set[str] | frozenset[str] = frozenset(),
    banner_title: str = "Native Cluster Launcher",
    port_label: str = "Init Port",
    node_label: str = "node",
    detached: bool = True,
    follow: bool = True,
    progress=None,
    extra_docker_opts: list[str] | None = None,
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

    executor = runtime.executor

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

    # Step 2: InfiniBand detection
    t0 = time.monotonic()
    if progress:
        progress.step("Detecting InfiniBand")
    else:
        logger.info("Step 2/7: InfiniBand detection...")
    comm_env = resolve_ib_env(ctx, comm_env)
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
    combined_docker_opts = (runtime.get_extra_docker_opts() or []) + (extra_docker_opts or [])
    rc = launch_containers_parallel(
        ctx,
        containers,
        executor,
        comm_env,
        extra_docker_opts=combined_docker_opts or None,
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
    run_pre_serve_hooks(runtime, ctx, hosts_containers, recipe, overrides)
    logger.info("Step 5/7: Pre-serve hooks done (%.1fs)", time.monotonic() - t0)

    # Step 6: Exec head serve command and wait for init port
    t0 = time.monotonic()
    head_container = all_nodes[0][2]
    if progress:
        progress.step("Starting head node serve")
    else:
        logger.info("Step 6/7: Executing serve command on head node (rank 0) %s...", ctx.head_host)
    head_exec_script = executor.generate_exec_serve_script(
        container_name=head_container,
        serve_command=head_command,
        env=ctx.all_env,
        detached=True,
    )
    head_result = run_remote_script(
        ctx.head_host,
        head_exec_script,
        timeout=60,
        dry_run=ctx.dry_run,
        **ctx.ssh_kwargs,
    )
    if not head_result.success and not ctx.dry_run:
        logger.error("Failed to exec serve on head node: %s", head_result.stderr[:200])
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
            else:
                logger.error(
                    "No logs captured. Check manually: ssh %s 'docker logs %s'",
                    ctx.head_host,
                    head_container,
                )
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
        with ThreadPoolExecutor(max_workers=len(ctx.worker_hosts)) as pool:
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
                )
                worker_container = all_nodes[rank][2]
                worker_exec_script = executor.generate_exec_serve_script(
                    container_name=worker_container,
                    serve_command=worker_command,
                    env=ctx.all_env,
                    detached=True,
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

            for future in as_completed(futures):
                host, rank = futures[future]
                result = future.result()
                if not result.success and not ctx.dry_run:
                    logger.warning(
                        "  Worker rank %d on %s may have failed: %s",
                        rank,
                        host,
                        result.stderr[:100],
                    )

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
