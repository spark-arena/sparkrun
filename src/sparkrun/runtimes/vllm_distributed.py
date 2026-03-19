"""Native vLLM distributed runtime for sparkrun.

Uses vLLM's built-in multi-node support (``--nnodes``, ``--node-rank``,
``--master-addr``, ``--master-port``, ``--headless``) instead of Ray.
Follows the same orchestration pattern as SGLang's native distribution.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin
from sparkrun.runtimes.vllm_ray import _VLLM_FLAG_MAP, _VLLM_BOOL_FLAGS

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


class VllmDistributedRuntime(RuntimePlugin):
    """vLLM runtime using native distributed mode (no Ray).

    Each node runs the full ``vllm serve`` command with node-specific
    ``--nnodes``, ``--node-rank``, ``--master-addr``, and ``--master-port``
    arguments.  Worker nodes additionally receive ``--headless``.
    """

    runtime_name = "vllm-distributed"
    default_image_prefix = "scitrera/dgx-spark-vllm"

    def cluster_strategy(self) -> str:
        """vLLM distributed uses native multi-node distribution, not Ray."""
        return "native"

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None,
                         skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Generate the vllm serve command.

        For cluster mode this produces the *base* command without
        ``--node-rank``.  Use :meth:`generate_node_command` to get the
        per-node variant.
        """
        config = recipe.build_config_chain(overrides)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered, config, "--served-model-name", skip_keys,
            )
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered, skip_keys, _VLLM_FLAG_MAP, _VLLM_BOOL_FLAGS,
                )
            return rendered

        return self._build_command(recipe, config, is_cluster, num_nodes, head_ip, skip_keys=skip_keys)

    def generate_node_command(
            self,
            recipe: Recipe,
            overrides: dict[str, Any],
            head_ip: str,
            num_nodes: int,
            node_rank: int,
            init_port: int = 25000,
            skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the vllm serve command for a specific node.

        Produces the full ``vllm serve`` invocation with the node-specific
        ``--nnodes``, ``--node-rank``, ``--master-addr``, and
        ``--master-port`` flags appended.  Workers (rank > 0) also get
        ``--headless``.
        """
        config = recipe.build_config_chain(overrides)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered, config, "--served-model-name", skip_keys,
            )
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered, skip_keys, _VLLM_FLAG_MAP, _VLLM_BOOL_FLAGS,
                )
            base = rendered
        else:
            base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        # Append vLLM native multi-node arguments
        parts = [
            base,
            "--nnodes %d" % num_nodes,
            "--node-rank %d" % node_rank,
            "--master-addr %s" % head_ip,
            "--master-port %d" % init_port,
        ]
        if node_rank > 0:
            parts.append("--headless")
        return " ".join(parts)

    def _build_base_command(self, recipe: Recipe, config,
                            skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the vllm serve command without cluster-specific arguments."""
        parts = ["vllm", "serve", recipe.model]

        tp = config.get("tensor_parallel")
        if tp:
            parts.extend(["-tp", str(tp)])

        # Add flags from defaults (skip tp and distributed_executor_backend)
        skip = {"tensor_parallel", "distributed_executor_backend"}
        skip.update(skip_keys)
        parts.extend(self.build_flags_from_map(
            config, _VLLM_FLAG_MAP, bool_keys=_VLLM_BOOL_FLAGS, skip_keys=skip,
        ))

        return " ".join(parts)

    def _build_command(self, recipe: Recipe, config, is_cluster: bool,
                       num_nodes: int, head_ip: str | None = None,
                       skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the vllm serve command from structured config.

        For cluster mode, includes ``--nnodes``, ``--master-addr``, and
        ``--master-port`` but NOT ``--node-rank`` (that is added per-node
        by :meth:`generate_node_command`).
        """
        base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        if is_cluster and head_ip:
            base += " --nnodes %d --master-addr %s --master-port 25000" % (num_nodes, head_ip)

        return base

    # --- Tuning config auto-mount ---

    def get_extra_volumes(self) -> dict[str, str]:
        """Mount vLLM tuning configs if available."""
        from sparkrun.tuning.vllm import get_vllm_tuning_volumes
        return get_vllm_tuning_volumes() or {}

    def get_extra_env(self) -> dict[str, str]:
        """Set VLLM_TUNED_CONFIG_FOLDER if tuning configs exist."""
        from sparkrun.tuning.vllm import get_vllm_tuning_env
        return get_vllm_tuning_env() or {}

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return vLLM distributed-specific cluster environment variables.

        Sets ``OMP_NUM_THREADS=4`` by default to avoid thread
        over-subscription on multi-node clusters.  Recipe ``env`` can
        override any of these values (runtime defaults are merged first,
        recipe env wins).
        """
        return {
            "NCCL_CUMEM_ENABLE": "0",
            "OMP_NUM_THREADS": "4",
        }

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate vLLM distributed-specific recipe fields."""
        return super().validate_recipe(recipe)

    # --- Log following hooks ---

    def _head_container_name(self, cluster_id: str) -> str:
        """vLLM distributed names the head container ``{cluster_id}_node_0``."""
        from sparkrun.orchestration.docker import generate_node_container_name
        return generate_node_container_name(cluster_id, 0)

    # --- Cluster stop ---

    def _stop_cluster(
            self,
            hosts: list[str],
            cluster_id: str,
            config=None,
            dry_run: bool = False,
    ) -> int:
        """Stop a vLLM distributed native cluster."""
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)

    # --- Cluster launch ---

    def _run_cluster(
            self,
            hosts: list[str],
            image: str,
            serve_command: str = "",
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
            init_port: int = 25000,
            skip_keys: set[str] | frozenset[str] = frozenset(),
            auto_remove: bool = True,
            restart_policy: str | None = None,
            **kwargs,
    ) -> int:
        """Orchestrate a multi-node vLLM cluster using native distribution.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Detect head node IP.
        4. Launch head node (rank 0).
        5. Wait for master port to be ready.
        6. Launch worker nodes in parallel (with --headless).
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            build_volumes,
            merge_env,
            detect_host_ip,
            wait_for_port,
            resolve_nccl_env,
        )
        from sparkrun.orchestration.ssh import (
            run_remote_script, run_remote_command,
            start_log_capture, stop_log_capture,
        )
        from sparkrun.orchestration.docker import (
            docker_stop_cmd, generate_node_container_name,
        )

        num_nodes = len(hosts)
        head_host = hosts[0]
        worker_hosts = hosts[1:]
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir, extra=self.get_extra_volumes())
        runtime_env = self.get_cluster_env(head_ip="<pending>", num_nodes=num_nodes)
        # Runtime defaults first, recipe env overrides (power users can tweak)
        all_env = merge_env(runtime_env, self.get_extra_env(), env)

        self._print_cluster_banner(
            "vLLM Distributed Cluster Launcher", hosts, image, cluster_id,
            {"Master Port": init_port}, dry_run,
        )

        # Step 1: Cleanup
        t0 = time.monotonic()
        logger.info("Step 1/6: Cleaning up existing containers for cluster '%s'...", cluster_id)
        for rank, host in enumerate(hosts):
            container_name = generate_node_container_name(cluster_id, rank)
            run_remote_command(
                host, docker_stop_cmd(container_name),
                timeout=30, dry_run=dry_run, **ssh_kwargs,
            )
        logger.info("Step 1/6: Cleanup done (%.1fs)", time.monotonic() - t0)

        # Step 2: InfiniBand detection (skip if pre-detected nccl_env provided)
        t0 = time.monotonic()
        logger.info("Step 2/6: InfiniBand detection...")
        nccl_env = resolve_nccl_env(
            nccl_env, hosts,
            head_host=head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )
        logger.info("Step 2/6: IB step done (%.1fs)", time.monotonic() - t0)

        # Step 3: Detect head node IP
        t0 = time.monotonic()
        logger.info("Step 3/6: Detecting head node IP on %s...", head_host)
        try:
            head_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
        except RuntimeError as e:
            logger.error("%s", e)
            return 1
        logger.info("  Head IP: %s", head_ip)
        logger.info("Step 3/6: IP detection done (%.1fs)", time.monotonic() - t0)

        # Auto-detect available init port to avoid collisions with running instances
        from sparkrun.orchestration.primitives import find_available_port
        init_port = find_available_port(head_host, init_port, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

        # Generate per-node commands
        head_command = self.generate_node_command(
            recipe=recipe, overrides=overrides,
            head_ip=head_ip, num_nodes=num_nodes,
            node_rank=0, init_port=init_port,
            skip_keys=skip_keys,
        )
        logger.info("Serve command (head, rank 0):")
        for line in head_command.strip().splitlines():
            logger.info("  %s", line)

        # Step 4: Launch head node (rank 0)
        t0 = time.monotonic()
        head_container = generate_node_container_name(cluster_id, 0)
        logger.info(
            "Step 4/6: Launching head node (rank 0) on %s as %s...",
            head_host, head_container,
        )
        head_script = self._generate_node_script(
            image=image, container_name=head_container,
            serve_command=head_command, label="vllm node",
            env=all_env, volumes=volumes, nccl_env=nccl_env,
            auto_remove=auto_remove, restart_policy=restart_policy,
        )
        head_result = run_remote_script(
            head_host, head_script, timeout=120, dry_run=dry_run, **ssh_kwargs,
        )
        if not head_result.success and not dry_run:
            logger.error("Failed to launch head node: %s", head_result.stderr[:200])
            return 1
        logger.info("Step 4/6: Head node launched (%.1fs)", time.monotonic() - t0)

        # Step 5: Wait for master port
        # Capture head container logs in the background so we can surface
        # them if the node fails to become ready (avoids manual SSH).
        t0 = time.monotonic()
        if not dry_run:
            logger.info("Step 5/6: Waiting for head node master port %s:%d...", head_host, init_port)

            log_proc = start_log_capture(head_host, head_container, ssh_kwargs)
            try:
                ready = wait_for_port(
                    head_host, init_port,
                    max_retries=60, retry_interval=2,
                    ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                    container_name=head_container,
                )
            finally:
                captured = stop_log_capture(log_proc)

            if not ready:
                logger.error("Head node failed to become ready on %s.", head_host)
                if captured:
                    logger.error("Container logs for %s:", head_container)
                    for line in captured[-150:]:
                        logger.error("  %s", line)
                else:
                    logger.error(
                        "No logs captured. Check manually: ssh %s 'docker logs %s'",
                        head_host, head_container,
                    )
                return 1
            logger.info("Step 5/6: Head node ready (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 5/6: [dry-run] Would wait for master port %d", init_port)

        # Step 6: Launch worker nodes in parallel
        t0 = time.monotonic()
        if worker_hosts:
            logger.info(
                "Step 6/6: Launching %d worker node(s) on %s...",
                len(worker_hosts), ", ".join(worker_hosts),
            )
            with ThreadPoolExecutor(max_workers=len(worker_hosts)) as executor:
                futures = {}
                for i, host in enumerate(worker_hosts):
                    rank = i + 1
                    worker_command = self.generate_node_command(
                        recipe=recipe, overrides=overrides,
                        head_ip=head_ip, num_nodes=num_nodes,
                        node_rank=rank, init_port=init_port,
                        skip_keys=skip_keys,
                    )
                    worker_container = generate_node_container_name(cluster_id, rank)
                    worker_script = self._generate_node_script(
                        image=image, container_name=worker_container,
                        serve_command=worker_command, label="vllm node",
                        env=all_env, volumes=volumes, nccl_env=nccl_env,
                        auto_remove=auto_remove, restart_policy=restart_policy,
                    )
                    future = executor.submit(
                        run_remote_script, host, worker_script,
                        timeout=120, dry_run=dry_run, **ssh_kwargs,
                    )
                    futures[future] = (host, rank)

                for future in as_completed(futures):
                    host, rank = futures[future]
                    result = future.result()
                    if not result.success and not dry_run:
                        logger.warning(
                            "  Worker rank %d on %s may have failed: %s",
                            rank, host, result.stderr[:100],
                        )

            logger.info("Step 6/6: Workers launched (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 6/6: No worker hosts, skipping")

        self._print_connection_info(hosts, cluster_id, per_node_logs=True)
        return 0
