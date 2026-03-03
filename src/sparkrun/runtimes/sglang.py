"""Native SGLang runtime for sparkrun."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# SGLang CLI flag mapping
_SGLANG_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "tensor_parallel": "--tp-size",
    "pipeline_parallel": "--pp-size",
    "gpu_memory_utilization": "--mem-fraction-static",
    "max_model_len": "--context-length",
    "max_num_seqs": "--max-running-requests",
    "served_model_name": "--served-model-name",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "trust_remote_code": "--trust-remote-code",
    "chunked_prefill": "--chunked-prefill-size",
    "kv_cache_dtype": "--kv-cache-dtype",
    "tokenizer_path": "--tokenizer-path",
}

_SGLANG_BOOL_FLAGS = {
    "trust_remote_code", "enable_torch_compile", "disable_radix_cache",
}


class SglangRuntime(RuntimePlugin):
    """Native SGLang runtime using prebuilt container images.

    SGLang uses its own distributed init mechanism for multi-node inference,
    not Ray.  Each node runs the full ``sglang.launch_server`` command with
    ``--dist-init-addr``, ``--nnodes``, and ``--node-rank`` arguments.
    """

    runtime_name = "sglang"
    default_image_prefix = "scitrera/dgx-spark-sglang"

    def cluster_strategy(self) -> str:
        """SGLang uses native multi-node distribution, not Ray."""
        return "native"

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None,
                         skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Generate the sglang launch_server command.

        For cluster mode this produces the *base* command without
        ``--node-rank``.  Use :meth:`generate_node_command` to get the
        per-node variant.
        """
        config = recipe.build_config_chain(overrides)
        self._inject_gguf_model(config)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered, skip_keys, _SGLANG_FLAG_MAP, _SGLANG_BOOL_FLAGS,
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
        """Generate the sglang command for a specific node.

        Produces the full ``sglang.launch_server`` invocation with the
        node-specific ``--dist-init-addr``, ``--nnodes``, and
        ``--node-rank`` flags appended.
        """
        config = recipe.build_config_chain(overrides)
        self._inject_gguf_model(config)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered, skip_keys, _SGLANG_FLAG_MAP, _SGLANG_BOOL_FLAGS,
                )
            base = rendered
        else:
            base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        # Append sglang multi-node arguments
        parts = [
            base,
            "--dist-init-addr %s:%d" % (head_ip, init_port),
            "--nnodes %d" % num_nodes,
            "--node-rank %d" % node_rank,
        ]
        return " ".join(parts)

    @staticmethod
    def _inject_gguf_model(config) -> None:
        """Ensure ``{model}`` in command templates resolves to the GGUF file path.

        When a GGUF model has been pre-synced, the CLI stores the
        container-internal path as ``_gguf_model_path`` in overrides.
        This helper copies that value into the ``model`` key so that
        ``{model}`` in recipe command templates renders the local file
        path instead of the raw HF repo spec (which includes the
        sparkrun-specific ``:quant`` suffix that runtimes cannot parse).
        """
        gguf_path = config.get("_gguf_model_path")
        if gguf_path:
            config.put("model", str(gguf_path))

    def _build_base_command(self, recipe: Recipe, config,
                            skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the sglang command without cluster-specific arguments."""
        # For GGUF models, use the resolved file path instead of the HF repo name
        model_path = config.get("_gguf_model_path") or recipe.model
        parts = ["python3", "-m", "sglang.launch_server", "--model-path", str(model_path)]

        tp = config.get("tensor_parallel")
        if tp:
            parts.extend(["--tp-size", str(tp)])

        skip = {"tensor_parallel"}
        skip.update(skip_keys)
        parts.extend(self.build_flags_from_map(
            config, _SGLANG_FLAG_MAP, bool_keys=_SGLANG_BOOL_FLAGS,
            skip_keys=skip,
        ))

        return " ".join(parts)

    def _build_command(self, recipe: Recipe, config, is_cluster: bool,
                       num_nodes: int, head_ip: str | None = None,
                       skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the sglang launch_server command from structured config.

        For cluster mode, includes ``--dist-init-addr`` and ``--nnodes`` but
        NOT ``--node-rank`` (that is added per-node by the orchestrator or
        by :meth:`generate_node_command`).
        """
        base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        if is_cluster and head_ip:
            base += " --dist-init-addr %s:25000 --nnodes %d" % (head_ip, num_nodes)

        return base

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return SGLang-specific cluster environment variables."""
        return {
            "NCCL_CUMEM_ENABLE": "0",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",  # confirmed for v0.5.9 on 20260205 by DB
        }

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate SGLang-specific recipe fields."""
        from sparkrun.models.download import is_gguf_model

        issues = super().validate_recipe(recipe)

        if recipe.model and is_gguf_model(recipe.model):
            tokenizer = (recipe.defaults or {}).get("tokenizer_path")
            cmd = recipe.command or ""
            cmd_has_tokenizer = "--tokenizer-path" in cmd or "{tokenizer_path}" in cmd

            if not tokenizer and not cmd_has_tokenizer:
                issues.append(
                    "[sglang] GGUF model detected but no tokenizer path configured. "
                    "SGLang requires --tokenizer-path pointing to the base (non-GGUF) HF model. "
                    "Set 'tokenizer_path' in defaults (e.g. tokenizer_path: Qwen/Qwen3-1.7B) "
                    "or add --tokenizer-path to the command template."
                )
            if tokenizer and cmd and not cmd_has_tokenizer:
                issues.append(
                    "[sglang] GGUF recipe has 'tokenizer_path' in defaults but the command "
                    "template does not reference {tokenizer_path} or --tokenizer-path. "
                    "Add '--tokenizer-path {tokenizer_path}' to the command template."
                )

        return issues

    def compute_required_nodes(self, recipe, overrides=None):
        """Compute required nodes as ``tp * pp``.

        SGLang supports pipeline parallelism via ``--pp-size``.  On
        DGX Spark (1 GPU per node), the total node count is the product
        of tensor and pipeline parallelism.

        Returns ``None`` when neither dimension is configured (meaning
        "use all provided hosts, no trimming").
        """
        config = recipe.build_config_chain(overrides or {})
        tp_val = config.get("tensor_parallel")
        pp_val = config.get("pipeline_parallel")
        if tp_val is None and pp_val is None:
            return None
        tp = int(tp_val) if tp_val is not None else 1
        pp = int(pp_val) if pp_val is not None else 1
        return tp * pp

    # --- Tuning config auto-mount ---

    def get_extra_volumes(self) -> dict[str, str]:
        """Mount SGLang tuning configs if available."""
        from sparkrun.tuning.sglang import get_sglang_tuning_volumes
        return get_sglang_tuning_volumes() or {}

    def get_extra_env(self) -> dict[str, str]:
        """Set SGLANG_MOE_CONFIG_DIR if tuning configs exist."""
        from sparkrun.tuning.sglang import get_sglang_tuning_env
        return get_sglang_tuning_env() or {}

    # --- Log following hooks ---

    def _head_container_name(self, cluster_id: str) -> str:
        """SGLang names the head container ``{cluster_id}_node_0``."""
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
        """Stop an SGLang native cluster."""
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
            **kwargs,
    ) -> int:
        """Orchestrate a multi-node SGLang cluster using native distribution.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Detect head node IP.
        4. Launch head node (rank 0).
        5. Wait for head init port to be ready.
        6. Launch worker nodes in parallel.
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
            "SGLang Cluster Launcher", hosts, image, cluster_id,
            {"Init Port": init_port}, dry_run,
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
            serve_command=head_command, label="sglang node",
            env=all_env, volumes=volumes, nccl_env=nccl_env,
        )
        head_result = run_remote_script(
            head_host, head_script, timeout=120, dry_run=dry_run, **ssh_kwargs,
        )
        if not head_result.success and not dry_run:
            logger.error("Failed to launch head node: %s", head_result.stderr[:200])
            return 1
        logger.info("Step 4/6: Head node launched (%.1fs)", time.monotonic() - t0)

        # Step 5: Wait for head init port
        # Capture head container logs in the background so we can surface
        # them if the node fails to become ready (avoids manual SSH).
        t0 = time.monotonic()
        if not dry_run:
            logger.info("Step 5/6: Waiting for head node init port %s:%d...", head_host, init_port)

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
            logger.info("Step 5/6: [dry-run] Would wait for head init port %d", init_port)

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
                        serve_command=worker_command, label="sglang node",
                        env=all_env, volumes=volumes, nccl_env=nccl_env,
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
