"""llama.cpp runtime for sparkrun via llama-server."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv

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

# first party images for special handling -- TODO: ensure specific tag for image available as label for spark-arena snapshot
_FIRST_PARTY_IMAGES = (
    "scitrera/dgx-spark-llama-cpp:latest",
    "ghcr.io/spark-arena/dgx-llama-cpp:latest",
)


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
            tp_val, pp_val = int(str(tp)), int(str(pp))
            # Both > 1 is genuinely mutually exclusive
            if tp_val > 1 and pp_val > 1:
                raise ValueError(
                    "llama.cpp does not support tensor_parallel and pipeline_parallel "
                    "simultaneously; use --tp for row splitting or --pp for layer "
                    "splitting, not both"
                )
            # One > 1 and the other == 1: the 1 is a no-op, use the active one
            if tp_val > 1:
                return "row"
            if pp_val > 1:
                return "layer"
            # Both are 1 — no override needed (falls through to _LLAMA_CPP_DEFAULTS)
            return None

        if tp is not None:
            return "row" if int(str(tp)) > 1 else None
        if pp is not None:
            return "layer" if int(str(pp)) > 1 else None
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

        if tp is not None and pp is not None:
            # Both set — use whichever is > 1; if both are 1, return 1
            tp_val, pp_val = int(str(tp)), int(str(pp))
            if tp_val > 1:
                return tp_val
            if pp_val > 1:
                return pp_val
            return 1

        if tp is not None:
            return int(str(tp))
        if pp is not None:
            return int(str(pp))
        return None

    @staticmethod
    def _inject_split_mode_in_command(command: str, split_mode: str) -> str:
        """Strip existing ``--split-mode`` from *command* and append the correct one."""
        import re

        command = re.sub(r"--split-mode\s+\S+", "", command).strip()
        return "%s --split-mode %s" % (command, split_mode)

    # --- Command generation ---

    def generate_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
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
        # Translate max_model_len → ctx_size so the cross-runtime CLI flag
        # works transparently (vLLM/SGLang use max_model_len, llama.cpp uses ctx_size).
        # This covers both CLI overrides and recipe defaults, and applies to
        # both template-rendered and structured command paths.
        if "ctx_size" not in overrides:
            if "max_model_len" in overrides:
                overrides = {**overrides, "ctx_size": overrides["max_model_len"]}
            else:
                # Check recipe defaults for max_model_len
                config = recipe.build_config_chain(overrides)
                max_model_len = config.get("max_model_len")
                if max_model_len is not None and config.get("ctx_size") is None:
                    overrides = {**overrides, "ctx_size": max_model_len}

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
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--alias",
                skip_keys,
            )
            if skip_keys:
                all_flags = {**_LLAMA_CPP_FLAG_MAP, **_LLAMA_CPP_BOOL_FLAGS}
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    all_flags,
                    set(_LLAMA_CPP_BOOL_FLAGS),
                    flag_aliases=_LLAMA_CPP_FLAG_ALIASES,
                )
            return rendered

        # Otherwise, build command from structured defaults
        return self._build_command(recipe, config, skip_keys=skip_keys, split_mode_override=split_mode)

    def _build_command(
        self, recipe: Recipe, config, skip_keys: set[str] | frozenset[str] = frozenset(), split_mode_override: str | None = None
    ) -> str:
        """Build the llama-server command from structured config."""
        from scitrera_app_framework.api import Variables, EnvPlacement

        # TP/PP → split_mode takes highest priority, then config, then defaults.
        # Export config to dict first — Variables cannot nest as a source
        # because its .get() returns None for missing keys instead of raising
        # KeyError, which stops the outer chain from checking later sources.
        config_dict = config.export_all_variables() if isinstance(config, Variables) else config
        parallelism_layer = {"split_mode": split_mode_override} if split_mode_override else {}
        config = Variables(sources=(parallelism_layer, config_dict, _LLAMA_CPP_DEFAULTS), env_placement=EnvPlacement.IGNORED)

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
        parts.extend(
            self.build_flags_from_map(
                config,
                all_flags,
                bool_keys=set(_LLAMA_CPP_BOOL_FLAGS),
                skip_keys=skip_keys,
            )
        )

        return " ".join(parts)

    def _build_rpc_head_command(
        self, recipe: Recipe, config, worker_hosts: list[str], rpc_port: int, skip_keys: set[str] | frozenset[str] = frozenset()
    ) -> str:
        """Build the llama-server head command with --rpc for worker nodes."""
        base = self._build_command(recipe, config, skip_keys=skip_keys)
        rpc_addrs = ",".join("%s:%d" % (h, rpc_port) for h in worker_hosts)
        return "%s --rpc %s" % (base, rpc_addrs)

    @staticmethod
    def _build_rpc_worker_command(rpc_port: int) -> str:
        """Build the rpc-server command for a worker node."""
        return "rpc-server --host 0.0.0.0 --port %d" % rpc_port

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        cmds["llama_cpp"] = "llama-server --version 2>/dev/null | head -1 || echo unknown"
        return cmds

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate llama.cpp-specific recipe fields."""
        issues = super().validate_recipe(recipe)
        defaults = recipe.defaults or {}
        tp = defaults.get("tensor_parallel")
        pp = defaults.get("pipeline_parallel")
        if tp and pp and int(tp) > 1 and int(pp) > 1:
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

        ssh_kwargs = build_ssh_kwargs(config)

        # Stop head
        head_container = self._container_name(cluster_id, "head")
        run_remote_command(
            hosts[0],
            self.executor.stop_cmd(head_container),
            timeout=30,
            dry_run=dry_run,
            **ssh_kwargs,
        )

        # Stop workers
        for host in hosts[1:]:
            worker_container = self._container_name(cluster_id, "worker")
            run_remote_command(
                host,
                self.executor.stop_cmd(worker_container),
                timeout=30,
                dry_run=dry_run,
                **ssh_kwargs,
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
        comm_env: "ClusterCommEnv | None" = None,
        ib_ip_map: dict[str, str] | None = None,
        rpc_port: int = _DEFAULT_RPC_PORT,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        extra_docker_opts: list[str] | None = None,
        **kwargs,
    ) -> int:
        """Orchestrate a multi-node llama.cpp cluster using RPC.

        Uses the two-phase launch pattern (sleep infinity + exec) so that
        ``_pre_serve`` hooks (e.g. ``pre_exec`` from recipes) run between
        container startup and serve execution.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Launch ALL containers with ``sleep infinity``.
        4. Run pre-serve hooks (pre_exec) on all containers.
        5. Exec RPC workers, wait for RPC ports.
        6. Exec llama-server on head with --rpc pointing to workers.

        .. note:: Experimental. The llama.cpp RPC backend is still evolving.
        """
        import time
        from sparkrun.runtimes._cluster_ops import (
            ClusterContext,
            cleanup_named_containers,
            detect_ib_with_ips,
            launch_containers_parallel,
            run_pre_serve_hooks,
            exec_serve_on_container,
        )
        from sparkrun.orchestration.primitives import wait_for_port
        from sparkrun.orchestration.ssh import run_remote_script

        logger.warning("llama.cpp RPC clustering is EXPERIMENTAL. Behavior may change in future versions.")

        progress = kwargs.pop("progress", None)

        assert recipe is not None

        ctx = ClusterContext.build(self, hosts, image, cluster_id, env, cache_dir, config, dry_run, overrides=overrides)
        head_container = self._container_name(cluster_id, "head")
        worker_container_name = self._container_name(cluster_id, "worker")

        self._print_cluster_banner(
            "llama.cpp RPC Cluster Launcher (EXPERIMENTAL)",
            hosts,
            image,
            cluster_id,
            {"RPC Port": rpc_port},
            dry_run,
        )

        if progress:
            progress.begin_runtime_steps(6)

        # Step 1: Cleanup
        t0 = time.monotonic()
        if progress:
            progress.step("Cleaning up existing containers")
        else:
            logger.info("Step 1/6: Cleaning up existing containers for cluster '%s'...", cluster_id)
        cleanup_named_containers(ctx, [head_container, worker_container_name])
        logger.info("Step 1/6: Cleanup done (%.1fs)", time.monotonic() - t0)

        # Step 2: InfiniBand detection (also resolves IB IPs for RPC routing)
        t0 = time.monotonic()
        if progress:
            progress.step("Detecting InfiniBand")
        else:
            logger.info("Step 2/6: InfiniBand detection...")
        comm_env, ib_ip_map = detect_ib_with_ips(ctx, comm_env, ib_ip_map)
        logger.info("Step 2/6: IB step done (%.1fs)", time.monotonic() - t0)

        # Resolve worker RPC addresses: prefer IB IPs for high-speed fabric
        rpc_hosts = []
        for h in ctx.worker_hosts:
            ib_ip = ib_ip_map.get(h)
            if ib_ip:
                logger.info("  Worker %s RPC via IB: %s", h, ib_ip)
                rpc_hosts.append(ib_ip)
            else:
                logger.info("  Worker %s RPC via management IP (no IB)", h)
                rpc_hosts.append(h)

        # Step 3: Launch ALL containers with sleep infinity
        t0 = time.monotonic()
        if progress:
            progress.step("Launching containers")
        else:
            logger.info("Step 3/6: Launching containers with sleep infinity on all %d host(s)...", len(hosts))

        all_containers: list[tuple[str, str]] = [(ctx.head_host, head_container)]
        for host in ctx.worker_hosts:
            all_containers.append((host, worker_container_name))

        combined_docker_opts = (self.get_extra_docker_opts() or []) + (extra_docker_opts or [])
        rc = launch_containers_parallel(ctx, all_containers, self.executor, comm_env, extra_docker_opts=combined_docker_opts or None)
        if rc != 0:
            return rc
        logger.info("Step 3/6: All containers launched (%.1fs)", time.monotonic() - t0)

        # Step 4: Pre-serve hooks (pre_exec)
        t0 = time.monotonic()
        if progress:
            progress.step("Running pre-serve hooks")
        else:
            logger.info("Step 4/6: Running pre-serve hooks...")
        run_pre_serve_hooks(self, ctx, all_containers, recipe, overrides)
        logger.info("Step 4/6: Pre-serve hooks done (%.1fs)", time.monotonic() - t0)

        # Step 5: Exec RPC workers and wait for RPC ports
        t0 = time.monotonic()
        config_chain = recipe.build_config_chain(overrides) if recipe else None
        if ctx.worker_hosts:
            if progress:
                progress.step("Starting RPC workers")
            else:
                logger.info(
                    "Step 5/6: Executing RPC server on %d worker(s) on %s...",
                    len(ctx.worker_hosts),
                    ", ".join(ctx.worker_hosts),
                )
            rpc_worker_command = self._build_rpc_worker_command(rpc_port)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=len(ctx.worker_hosts)) as pool:
                futures = {}
                for host in ctx.worker_hosts:
                    exec_script = self.executor.generate_exec_serve_script(
                        container_name=worker_container_name,
                        serve_command=rpc_worker_command,
                        env=ctx.all_env,
                        detached=True,
                    )
                    future = pool.submit(
                        run_remote_script,
                        host,
                        exec_script,
                        timeout=60,
                        dry_run=dry_run,
                        **ctx.ssh_kwargs,
                    )
                    futures[future] = host

                for future in as_completed(futures):
                    host = futures[future]
                    result = future.result()
                    if not result.success and not dry_run:
                        logger.warning(
                            "  RPC worker on %s may have failed: %s",
                            host,
                            result.stderr[:100],
                        )

            # Wait for RPC ports (probe via management IPs -- SSH
            # connectivity is guaranteed there; the IB IPs are used for the
            # actual RPC data path in step 6)
            if not dry_run:
                logger.info("  Waiting for RPC workers to be ready...")
                for host in ctx.worker_hosts:
                    ready = wait_for_port(
                        host,
                        rpc_port,
                        max_retries=30,
                        retry_interval=2,
                        ssh_kwargs=ctx.ssh_kwargs,
                        dry_run=dry_run,
                        container_name=worker_container_name,
                    )
                    if not ready:
                        logger.error(
                            "RPC worker on %s failed to become ready. Check logs: ssh %s 'docker logs %s'",
                            host,
                            host,
                            worker_container_name,
                        )
                        return 1

            if not progress:
                logger.info("Step 5/6: RPC workers ready (%.1fs)", time.monotonic() - t0)
        else:
            if progress:
                progress.step("Starting RPC workers")
                progress.detail("  No worker hosts, skipping")
            else:
                logger.info("Step 5/6: No worker hosts, skipping")

        # Step 6: Exec llama-server on head with --rpc (uses IB IPs when available)
        t0 = time.monotonic()
        head_command = self._build_rpc_head_command(
            recipe,
            config_chain,
            rpc_hosts,
            rpc_port,
            skip_keys=skip_keys,
        )
        if progress:
            progress.step("Executing llama-server on head")
        else:
            logger.info("Step 6/6: Executing llama-server on head %s...", ctx.head_host)
        logger.info("  Command: %s", head_command[:120])

        rc = exec_serve_on_container(ctx, self.executor, ctx.head_host, head_container, head_command)
        if rc != 0:
            return rc
        logger.info("Step 6/6: Head launched (%.1fs)", time.monotonic() - t0)

        self._print_connection_info(hosts, cluster_id)

        if not detached and not dry_run:
            from sparkrun.runtimes._cluster_ops import _attach_foreground

            _attach_foreground(self, ctx, kwargs.get("follow", True))

        return 0
