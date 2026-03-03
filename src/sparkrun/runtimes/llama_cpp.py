"""llama.cpp runtime for sparkrun via llama-server."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# llama-server CLI flag mapping (recipe key -> CLI flag)
_LLAMA_CPP_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "ctx_size": "--ctx-size",
    "n_gpu_layers": "--n-gpu-layers",
    "parallel": "--parallel",
    "threads": "--threads",
    "chat_template": "--chat-template",
    "reasoning_format": "--reasoning-format",
    "split_mode": "--split-mode",
    "served_model_name": "--alias",
}

# Defaults injected when not set in recipe config
_LLAMA_CPP_DEFAULTS = {
    "split_mode": "layer",
}

# Boolean flags (present when truthy, absent when falsy)
_LLAMA_CPP_BOOL_FLAGS = {
    "flash_attn": "--flash-attn",
    "cont_batching": "--cont-batching",
    "no_webui": "--no-webui",
    "jinja": "--jinja",
}

# Short-form flag aliases recognised when stripping flags from rendered
# command templates (e.g. ``-a`` is the short form of ``--alias``).
_LLAMA_CPP_FLAG_ALIASES: dict[str, list[str]] = {
    "served_model_name": ["-a"],
}

# Default RPC port for llama.cpp distributed inference
_DEFAULT_RPC_PORT = 50052


class LlamaCppRuntime(RuntimePlugin):
    """llama.cpp runtime using llama-server for GGUF model inference.

    Provides lightweight inference via llama-server with OpenAI-compatible
    API.  Supports GGUF quantized models loaded directly from HuggingFace
    (e.g. ``Qwen/Qwen3-1.7B-GGUF:Q4_K_M``) or from local paths.

    **Solo mode**: Single-node inference using ``_run_solo`` (sleep infinity
    + exec).

    **Cluster mode** (experimental): Multi-node tensor-parallel inference
    via llama.cpp RPC.  Worker nodes run ``rpc-server`` and the head node
    runs ``llama-server --rpc worker1:port,worker2:port,...``.

    **Parallelism mapping**: In llama.cpp ``--split-mode`` selects the
    strategy.  ``--tp`` (tensor parallel) maps to ``--split-mode row``
    and ``--pp`` (pipeline parallel) maps to ``--split-mode layer``.
    They cannot be used simultaneously.
    """

    runtime_name = "llama-cpp"
    default_image_prefix = "scitrera/dgx-spark-llama-cpp"

    def cluster_strategy(self) -> str:
        """llama.cpp uses native RPC-based distribution, not Ray."""
        return "native"

    # --- Parallelism helpers ---

    @staticmethod
    def _resolve_split_mode(config) -> str | None:
        """Derive ``--split-mode`` from tensor/pipeline parallelism settings.

        In llama.cpp ``--split-mode row`` ≈ tensor parallelism and
        ``--split-mode layer`` ≈ pipeline parallelism.  They are
        mutually exclusive.

        Returns:
            ``"row"`` when tensor_parallel is set, ``"layer"`` when
            pipeline_parallel is set, or ``None`` to fall through to
            recipe defaults.

        Raises:
            ValueError: If both tensor_parallel and pipeline_parallel
                are present in the config.
        """
        tp = config.get("tensor_parallel")
        pp = config.get("pipeline_parallel")
        if tp is not None and pp is not None:
            raise ValueError(
                "llama.cpp does not support tensor_parallel and pipeline_parallel "
                "simultaneously; use --tp for row splitting or --pp for layer "
                "splitting, not both"
            )
        if tp is not None:
            return "row"
        if pp is not None:
            return "layer"
        return None

    def compute_required_nodes(self, recipe, overrides=None):
        """Compute required nodes from TP or PP (mutually exclusive).

        In llama.cpp, ``--split-mode row`` (TP) and ``--split-mode layer``
        (PP) both distribute across N nodes but cannot be combined.

        Returns ``None`` when neither is configured.

        Raises:
            ValueError: If both tensor_parallel and pipeline_parallel
                are set.
        """
        config = recipe.build_config_chain(overrides or {})
        # _resolve_split_mode validates mutual exclusivity
        self._resolve_split_mode(config)

        tp = config.get("tensor_parallel")
        pp = config.get("pipeline_parallel")
        if tp is not None:
            return int(tp)
        if pp is not None:
            return int(pp)
        return None

    @staticmethod
    def _inject_split_mode_in_command(command: str, split_mode: str) -> str:
        """Strip existing ``--split-mode`` from *command* and append the correct one."""
        import re
        command = re.sub(r"--split-mode\s+\S+", "", command).strip()
        return "%s --split-mode %s" % (command, split_mode)

    # --- Command generation ---

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None,
                         skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Generate the llama-server command.

        When a pre-resolved GGUF path is available (``_gguf_model_path``
        in overrides / config), the ``model`` override contains the
        container-internal cache path.  The recipe command template is
        still rendered normally (``{model}`` resolves to the cache path),
        but ``-hf`` is switched to ``-m`` since the model is now a local
        file rather than a HuggingFace download spec.

        TP/PP parallelism settings are resolved into ``--split-mode``
        automatically (row for TP, layer for PP).
        """
        config = recipe.build_config_chain(overrides)
        split_mode = self._resolve_split_mode(config)
        gguf_path = config.get("_gguf_model_path")

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            if gguf_path:
                # Template rendered with the GGUF cache path as {model},
                # but -hf expects a HF repo spec, not a local file.
                # Switch to -m (model file path) instead.
                rendered = rendered.replace("-hf ", "-m ", 1)
            # Override --split-mode when TP/PP forces a specific strategy
            if split_mode is not None:
                rendered = self._inject_split_mode_in_command(rendered, split_mode)
            if skip_keys:
                all_flags = {**_LLAMA_CPP_FLAG_MAP, **_LLAMA_CPP_BOOL_FLAGS}
                rendered = self.strip_flags_from_command(
                    rendered, skip_keys, all_flags, set(_LLAMA_CPP_BOOL_FLAGS),
                    flag_aliases=_LLAMA_CPP_FLAG_ALIASES,
                )
            return rendered

        # Otherwise, build command from structured defaults
        return self._build_command(recipe, config, skip_keys=skip_keys,
                                   split_mode_override=split_mode)

    def _build_command(self, recipe: Recipe, config,
                       skip_keys: set[str] | frozenset[str] = frozenset(),
                       split_mode_override: str | None = None) -> str:
        """Build the llama-server command from structured config."""
        from vpd.legacy.yaml_dict import vpd_chain

        # TP/PP → split_mode takes highest priority, then config, then defaults
        parallelism_layer = {"split_mode": split_mode_override} if split_mode_override else {}
        config = vpd_chain(parallelism_layer, config, _LLAMA_CPP_DEFAULTS)

        model = recipe.model

        # Check for pre-resolved GGUF path from distribution pre-sync
        gguf_path = config.get("_gguf_model_path")

        # Determine model source flag:
        #   - Pre-synced GGUF -> -m <container_cache_path>
        #   - Local .gguf path -> -m <path>
        #   - HuggingFace repo (contains '/') -> -hf <repo>
        if gguf_path:
            parts = ["llama-server", "-m", str(gguf_path)]
        elif model and model.lower().endswith(".gguf"):
            parts = ["llama-server", "-m", model]
        elif model and "/" in model:
            parts = ["llama-server", "-hf", model]
        else:
            parts = ["llama-server", "-m", model or ""]

        # Add valued and boolean flags from config
        all_flags = {**_LLAMA_CPP_FLAG_MAP, **_LLAMA_CPP_BOOL_FLAGS}
        parts.extend(self.build_flags_from_map(
            config, all_flags, bool_keys=set(_LLAMA_CPP_BOOL_FLAGS),
            skip_keys=skip_keys,
        ))

        return " ".join(parts)

    def _build_rpc_head_command(self, recipe: Recipe, config,
                                worker_hosts: list[str],
                                rpc_port: int,
                                skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the llama-server head command with --rpc for worker nodes."""
        base = self._build_command(recipe, config, skip_keys=skip_keys)
        rpc_addrs = ",".join("%s:%d" % (h, rpc_port) for h in worker_hosts)
        return "%s --rpc %s" % (base, rpc_addrs)

    @staticmethod
    def _build_rpc_worker_command(rpc_port: int) -> str:
        """Build the rpc-server command for a worker node."""
        return "rpc-server --host 0.0.0.0 --port %d" % rpc_port

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate llama.cpp-specific recipe fields."""
        issues = super().validate_recipe(recipe)
        defaults = recipe.defaults or {}
        if defaults.get("tensor_parallel") and defaults.get("pipeline_parallel"):
            issues.append(
                "[llama-cpp] tensor_parallel and pipeline_parallel are mutually "
                "exclusive; use one for --split-mode row (TP) or layer (PP), not both"
            )
        return issues

    # --- Log following hooks ---

    def _head_container_name(self, cluster_id: str) -> str:
        """llama.cpp names the head container ``{cluster_id}_head``."""
        return self._container_name(cluster_id, "head")

    # --- Cluster stop ---

    def _stop_cluster(
            self,
            hosts: list[str],
            cluster_id: str,
            config=None,
            dry_run: bool = False,
    ) -> int:
        """Stop a llama.cpp RPC cluster."""
        from sparkrun.orchestration.primitives import build_ssh_kwargs
        from sparkrun.orchestration.ssh import run_remote_command
        from sparkrun.orchestration.docker import docker_stop_cmd

        ssh_kwargs = build_ssh_kwargs(config)

        # Stop head
        head_container = self._container_name(cluster_id, "head")
        run_remote_command(
            hosts[0], docker_stop_cmd(head_container),
            timeout=30, dry_run=dry_run, **ssh_kwargs,
        )

        # Stop workers
        for host in hosts[1:]:
            worker_container = self._container_name(cluster_id, "worker")
            run_remote_command(
                host, docker_stop_cmd(worker_container),
                timeout=30, dry_run=dry_run, **ssh_kwargs,
            )

        logger.info("llama.cpp cluster '%s' stopped on %d host(s)", cluster_id, len(hosts))
        return 0

    @staticmethod
    def _container_name(cluster_id: str, role: str) -> str:
        """Generate container name: ``{cluster_id}_{role}``."""
        return "%s_%s" % (cluster_id, role)

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
            ib_ip_map: dict[str, str] | None = None,
            rpc_port: int = _DEFAULT_RPC_PORT,
            skip_keys: set[str] | frozenset[str] = frozenset(),
            **kwargs,
    ) -> int:
        """Orchestrate a multi-node llama.cpp cluster using RPC.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Launch RPC workers on non-head hosts.
        4. Wait for RPC ports to be ready.
        5. Launch llama-server on head with --rpc pointing to workers.

        .. note:: Experimental. The llama.cpp RPC backend is still evolving.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            build_volumes,
            merge_env,
            wait_for_port,
        )
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts
        from sparkrun.orchestration.ssh import run_remote_script, run_remote_command
        from sparkrun.orchestration.docker import docker_stop_cmd

        logger.warning(
            "llama.cpp RPC clustering is EXPERIMENTAL. "
            "Behavior may change in future versions."
        )

        head_host = hosts[0]
        worker_hosts = hosts[1:]
        head_container = self._container_name(cluster_id, "head")
        worker_container_name = self._container_name(cluster_id, "worker")
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir)
        all_env = merge_env(env)

        self._print_cluster_banner(
            "llama.cpp RPC Cluster Launcher (EXPERIMENTAL)", hosts, image, cluster_id,
            {"RPC Port": rpc_port}, dry_run,
        )

        # Step 1: Cleanup
        t0 = time.monotonic()
        logger.info("Step 1/5: Cleaning up existing containers for cluster '%s'...", cluster_id)
        run_remote_command(
            head_host, docker_stop_cmd(head_container),
            timeout=30, dry_run=dry_run, **ssh_kwargs,
        )
        for host in worker_hosts:
            run_remote_command(
                host, docker_stop_cmd(worker_container_name),
                timeout=30, dry_run=dry_run, **ssh_kwargs,
            )
        logger.info("Step 1/5: Cleanup done (%.1fs)", time.monotonic() - t0)

        # Step 2: InfiniBand detection (also resolves IB IPs for RPC routing)
        t0 = time.monotonic()
        if ib_ip_map is None:
            ib_ip_map = {}
        if nccl_env is not None:
            logger.info("Step 2/5: Using pre-detected NCCL env (%d vars)", len(nccl_env))
            if ib_ip_map:
                logger.info("  Pre-detected IB IPs for %d host(s)", len(ib_ip_map))
        else:
            logger.info("Step 2/5: Detecting InfiniBand on all hosts...")
            ib_result = detect_ib_for_hosts(
                hosts, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
            )
            nccl_env = ib_result.nccl_env
            ib_ip_map = ib_result.ib_ip_map
            logger.info("Step 2/5: IB detection done (%.1fs)", time.monotonic() - t0)

        # Resolve worker RPC addresses: prefer IB IPs for high-speed fabric
        rpc_hosts = []
        for h in worker_hosts:
            ib_ip = ib_ip_map.get(h)
            if ib_ip:
                logger.info("  Worker %s RPC via IB: %s", h, ib_ip)
                rpc_hosts.append(ib_ip)
            else:
                logger.info("  Worker %s RPC via management IP (no IB)", h)
                rpc_hosts.append(h)

        # Step 3: Launch RPC workers in parallel
        t0 = time.monotonic()
        if worker_hosts:
            logger.info(
                "Step 3/5: Launching %d RPC worker(s) on %s...",
                len(worker_hosts), ", ".join(worker_hosts),
            )
            rpc_worker_command = self._build_rpc_worker_command(rpc_port)

            with ThreadPoolExecutor(max_workers=len(worker_hosts)) as executor:
                futures = {}
                for host in worker_hosts:
                    script = self._generate_node_script(
                        image=image, container_name=worker_container_name,
                        serve_command=rpc_worker_command, label="llama.cpp node",
                        env=all_env, volumes=volumes, nccl_env=nccl_env,
                    )
                    future = executor.submit(
                        run_remote_script, host, script,
                        timeout=120, dry_run=dry_run, **ssh_kwargs,
                    )
                    futures[future] = host

                for future in as_completed(futures):
                    host = futures[future]
                    result = future.result()
                    if not result.success and not dry_run:
                        logger.warning(
                            "  RPC worker on %s may have failed: %s",
                            host, result.stderr[:100],
                        )

            logger.info("Step 3/5: RPC workers launched (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 3/5: No worker hosts, skipping")

        # Step 4: Wait for RPC ports (probe via management IPs -- SSH
        # connectivity is guaranteed there; the IB IPs are used for the
        # actual RPC data path in step 5)
        t0 = time.monotonic()
        if worker_hosts and not dry_run:
            logger.info("Step 4/5: Waiting for RPC workers to be ready...")
            for host in worker_hosts:
                ready = wait_for_port(
                    host, rpc_port,
                    max_retries=30, retry_interval=2,
                    ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                    container_name=worker_container_name,
                )
                if not ready:
                    logger.error(
                        "RPC worker on %s failed to become ready. "
                        "Check logs: ssh %s 'docker logs %s'",
                        host, host, worker_container_name,
                    )
                    return 1
            logger.info("Step 4/5: RPC workers ready (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 4/5: [dry-run] Would wait for RPC workers")

        # Step 5: Launch head with --rpc (uses IB IPs when available)
        t0 = time.monotonic()
        config_chain = recipe.build_config_chain(overrides)
        head_command = self._build_rpc_head_command(
            recipe, config_chain, rpc_hosts, rpc_port,
            skip_keys=skip_keys,
        )
        logger.info("Step 5/5: Launching llama-server on head %s...", head_host)
        logger.info("  Command: %s", head_command[:120])

        head_script = self._generate_node_script(
            image=image, container_name=head_container,
            serve_command=head_command, label="llama.cpp node",
            env=all_env, volumes=volumes, nccl_env=nccl_env,
        )
        head_result = run_remote_script(
            head_host, head_script, timeout=120, dry_run=dry_run, **ssh_kwargs,
        )
        if not head_result.success and not dry_run:
            logger.error("Failed to launch head: %s", head_result.stderr[:200])
            return 1
        logger.info("Step 5/5: Head launched (%.1fs)", time.monotonic() - t0)

        self._print_connection_info(hosts, cluster_id)
        return 0
