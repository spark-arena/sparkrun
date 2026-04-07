"""Native vLLM runtime for sparkrun."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin
from sparkrun.runtimes._vllm_common import VllmMixin, VLLM_FLAG_MAP, VLLM_BOOL_FLAGS

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


class VllmRayRuntime(VllmMixin, RuntimePlugin):
    """vLLM runtime using Ray for multi-node clustering.

    Uses Ray head/worker orchestration for distributed inference.
    Registered as ``"vllm-ray"``; the virtual ``"vllm"`` alias
    routes here when the recipe hints at Ray usage.
    """

    runtime_name = "vllm-ray"
    default_image_prefix = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5"

    def get_family(self) -> str:
        return "vllm"

    def generate_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the vllm serve command."""
        config = recipe.build_config_chain(overrides)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            # Ensure --distributed-executor-backend ray is present for cluster mode
            if is_cluster and "--distributed-executor-backend" not in rendered:
                rendered = rendered.rstrip() + " --distributed-executor-backend ray"
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--served-model-name",
                skip_keys,
            )
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    VLLM_FLAG_MAP,
                    VLLM_BOOL_FLAGS,
                )
            return rendered

        # Otherwise, build command from structured defaults
        return self._build_command(recipe, config, is_cluster, num_nodes, skip_keys=skip_keys)

    def _build_command(
        self, recipe: Recipe, config, is_cluster: bool, num_nodes: int, skip_keys: set[str] | frozenset[str] = frozenset()
    ) -> str:
        """Build the vllm serve command from structured config."""
        parts = ["vllm", "serve", recipe.model]

        # Auto-inject cluster args
        if is_cluster:
            tp = config.get("tensor_parallel")
            if tp:
                parts.extend(["-tp", str(tp)])
            parts.extend(["--distributed-executor-backend", "ray"])
        else:
            tp = config.get("tensor_parallel")
            if tp:
                parts.extend(["-tp", str(tp)])

        # Add flags from defaults (skip tp since handled above)
        skip = {"tensor_parallel"}
        if is_cluster:
            skip.add("distributed_executor_backend")
        skip.update(skip_keys)
        parts.extend(
            self.build_flags_from_map(
                config,
                VLLM_FLAG_MAP,
                bool_keys=VLLM_BOOL_FLAGS,
                skip_keys=skip,
            )
        )

        return " ".join(parts)

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return ray vLLM-specific cluster environment variables."""
        return {
            **RuntimePlugin.get_cluster_env(self, head_ip, num_nodes),
            "RAY_memory_monitor_refresh_ms": "0",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        }

    # --- Log following hooks ---

    def _cluster_log_mode(self) -> str:
        """vLLM uses sleep-infinity + exec, so tail the serve log file."""
        return "file"

    # --- Cluster launch / stop ---

    def _stop_cluster(
        self,
        hosts: list[str],
        cluster_id: str,
        config=None,
        dry_run: bool = False,
    ) -> int:
        """Stop a vLLM Ray cluster."""
        from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers

        head_container = self.executor.container_name(cluster_id, "head")
        worker_container = self.executor.container_name(cluster_id, "worker")
        ssh_kwargs = build_ssh_kwargs(config)

        cleanup_containers(
            hosts,
            [head_container, worker_container],
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
        )
        logger.info("Cluster '%s' stopped on %d host(s)", cluster_id, len(hosts))
        return 0

    def _run_cluster(
        self,
        hosts: list[str],
        image: str,
        serve_command: str,
        recipe=None,
        overrides=None,
        *,
        cluster_id: str = "sparkrun0",
        env: dict[str, str] | None = None,
        cache_dir: str | None = None,
        config=None,
        dry_run: bool = False,
        detached: bool = True,
        nccl_env: dict[str, str] | None = None,
        ray_port: int = 46379,
        dashboard_port: int = 8265,
        dashboard: bool = False,
        extra_docker_opts: list[str] | None = None,
        **kwargs,
    ) -> int:
        """Orchestrate a multi-node Ray cluster for vLLM.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Launch Ray head on first host.
        4. Launch Ray workers on remaining hosts (parallel).
        5. Execute serve command on head node.
        """
        import time
        from sparkrun.runtimes._cluster_ops import (
            ClusterContext,
            cleanup_named_containers,
            resolve_ib_env,
            find_port,
            run_pre_serve_hooks,
        )
        from sparkrun.orchestration.primitives import is_valid_ip, wait_for_port
        from sparkrun.orchestration.ssh import run_remote_script, run_remote_scripts_parallel

        progress = kwargs.pop("progress", None)
        combined_docker_opts = (self.get_extra_docker_opts() or []) + (extra_docker_opts or [])

        ctx = ClusterContext.build(self, hosts, image, cluster_id, env, cache_dir, config, dry_run)
        head_container = self.executor.container_name(cluster_id, "head")
        worker_container = self.executor.container_name(cluster_id, "worker")

        if progress:
            progress.begin_runtime_steps(5)

        # Step 1: Cleanup
        t0 = time.monotonic()
        if progress:
            progress.step("Cleaning up existing containers")
        else:
            logger.info("Step 1/5: Cleaning up existing containers for cluster '%s'...", cluster_id)
        cleanup_named_containers(ctx, [head_container, worker_container])
        logger.info("Step 1/5: Cleanup done (%.1fs)", time.monotonic() - t0)

        # Step 2: InfiniBand detection (skip if pre-detected nccl_env provided)
        t0 = time.monotonic()
        if progress:
            progress.step("Detecting InfiniBand")
        else:
            logger.info("Step 2/5: InfiniBand detection...")
        nccl_env = resolve_ib_env(ctx, nccl_env)
        logger.info("Step 2/5: IB step done (%.1fs)", time.monotonic() - t0)

        # Auto-detect available ports to avoid collisions with running instances
        ray_port = find_port(ctx, ctx.head_host, ray_port)
        if dashboard:
            dashboard_port = find_port(ctx, ctx.head_host, dashboard_port)

        # print banner AFTER finalizing ports
        self._print_cluster_banner(
            "Ray Cluster Launcher",
            hosts,
            image,
            cluster_id,
            {"Ray Port": ray_port, "Dashboard Port": dashboard_port, "Command": serve_command[:80]},
            dry_run,
        )

        # Step 3: Launch Ray head
        t0 = time.monotonic()
        if progress:
            progress.step("Launching Ray head")
        else:
            logger.info("Step 3/5: Launching Ray head on %s...", ctx.head_host)
        head_script = self.executor.generate_ray_head_script(
            image=image,
            container_name=head_container,
            ray_port=ray_port,
            dashboard_port=dashboard_port,
            dashboard=dashboard,
            env=ctx.all_env,
            volumes=ctx.volumes,
            nccl_env=nccl_env,
            extra_docker_opts=combined_docker_opts or None,
        )
        head_result = run_remote_script(
            ctx.head_host,
            head_script,
            timeout=120,
            dry_run=dry_run,
            **ctx.ssh_kwargs,
        )
        if not head_result.success and not dry_run:
            logger.error("Failed to launch Ray head: %s", head_result.stderr)
            return 1

        head_ip = head_result.last_line if not dry_run else "<HEAD_IP>"
        if not dry_run and not is_valid_ip(head_ip):
            logger.error(
                "Could not determine head IP from output: %s",
                head_result.stdout[-200:],
            )
            return 1
        logger.info("  Ray head launched. HEAD_IP=%s", head_ip)

        if not dry_run:
            logger.info("  Waiting for Ray head port %s:%d...", ctx.head_host, ray_port)
            ready = wait_for_port(
                ctx.head_host,
                ray_port,
                max_retries=30,
                retry_interval=2,
                ssh_kwargs=ctx.ssh_kwargs,
                container_name=head_container,
            )
            if not ready:
                logger.error(
                    "Ray head failed to become ready. Check logs: ssh %s 'docker logs %s'",
                    ctx.head_host,
                    head_container,
                )
                return 1
        logger.info("Step 3/5: Ray head ready (%.1fs)", time.monotonic() - t0)

        # Step 4: Launch Ray workers (parallel)
        t0 = time.monotonic()
        if ctx.worker_hosts:
            if progress:
                progress.step("Launching Ray workers")
            else:
                logger.info(
                    "Step 4/5: Launching %d Ray worker(s) on %s...",
                    len(ctx.worker_hosts),
                    ", ".join(ctx.worker_hosts),
                )
            worker_script = self.executor.generate_ray_worker_script(
                image=image,
                container_name=worker_container,
                head_ip=head_ip,
                ray_port=ray_port,
                env=ctx.all_env,
                volumes=ctx.volumes,
                nccl_env=nccl_env,
                extra_docker_opts=combined_docker_opts or None,
            )
            worker_results = run_remote_scripts_parallel(
                ctx.worker_hosts,
                worker_script,
                timeout=120,
                dry_run=dry_run,
                **ctx.ssh_kwargs,
            )
            failed = [r for r in worker_results if not r.success and not dry_run]
            for r in failed:
                logger.warning(
                    "  Worker launch may have failed on %s: %s",
                    r.host,
                    r.stderr[:100],
                )
            if not dry_run:
                logger.info(
                    "  Waiting 3s for workers to connect to head at %s:%d...",
                    head_ip,
                    ray_port,
                )
                time.sleep(3)
            if not progress:
                logger.info("Step 4/5: Workers launched (%.1fs)", time.monotonic() - t0)
        else:
            if progress:
                progress.step("Launching Ray workers")
                progress.detail("  No worker hosts, skipping")
            else:
                logger.info("Step 4/5: No worker hosts, skipping")

        # Pre-serve hook (e.g., apply mods to containers, run pre_exec)
        all_containers = [(ctx.head_host, head_container)]
        for worker in ctx.worker_hosts:
            all_containers.append((worker, worker_container))
        run_pre_serve_hooks(self, ctx, all_containers, recipe, overrides)

        # Step 5: Execute serve command on head
        t0 = time.monotonic()
        if progress:
            progress.step("Executing serve command on head")
        else:
            logger.info(
                "Step 5/5: Executing serve command on head node %s (container: %s)...",
                ctx.head_host,
                head_container,
            )
        exec_script = self.executor.generate_exec_serve_script(
            container_name=head_container,
            serve_command=serve_command,
            env=ctx.all_env,
            detached=detached,
        )

        self._print_connection_info(hosts, cluster_id, head_ip=head_ip, dashboard_port=dashboard_port)

        exec_result = run_remote_script(
            ctx.head_host,
            exec_script,
            timeout=60,
            dry_run=dry_run,
            **ctx.ssh_kwargs,
        )
        logger.info("Step 5/5: Serve command dispatched (%.1fs)", time.monotonic() - t0)

        if dry_run:
            return 0
        return exec_result.returncode

    def _print_connection_info(self, hosts, cluster_id, head_ip=None, dashboard_port=8265):
        """Print vLLM-specific connection info including Dashboard URL."""
        logger.info("=" * 60)
        logger.info("Cluster launched successfully. Nodes: %d", len(hosts))
        logger.info("")
        logger.info("To view logs:    sparkrun logs <recipe> --hosts %s", ",".join(hosts))
        logger.info("To stop cluster: sparkrun stop <recipe> --hosts %s", ",".join(hosts))
        if head_ip:
            logger.info("Dashboard:       http://%s:%d", head_ip, dashboard_port)
        logger.info("=" * 60)
