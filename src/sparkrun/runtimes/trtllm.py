"""TensorRT-LLM runtime for sparkrun.

Uses MPI (OpenMPI) for multi-node tensor parallelism.  Instead of
installing openssh-server inside containers, a custom mpirun rsh agent
routes through host-level SSH + ``docker exec`` into worker containers.

Solo mode (tp=1) uses the standard sleep-infinity + exec pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# TRT-LLM CLI flag mapping (trtllm-serve flags)
_TRTLLM_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "tensor_parallel": "--tp_size",
    "pipeline_parallel": "--pp_size",
    "expert_parallel": "--ep_size",
    "max_num_tokens": "--max_num_tokens",
    "max_batch_size": "--max_batch_size",
    "max_model_len": "--max_seq_len",
    "backend": "--backend",
    "tokenizer": "--tokenizer",
    "kv_cache_free_gpu_memory_fraction": "--kv_cache_free_gpu_memory_fraction",
    "trust_remote_code": "--trust_remote_code",
}

_TRTLLM_BOOL_FLAGS = {"trust_remote_code"}

# Extra docker options required by TRT-LLM for large memory allocations
_TRTLLM_EXTRA_DOCKER_OPTS = ["--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]

# Fixed container-internal path for the extra LLM API config YAML
_EXTRA_CONFIG_PATH = "/tmp/extra-llm-api-config.yml"
_EXTRA_CONFIG_FLAG = "--extra_llm_api_options"

# Keys in recipe defaults that map to extra-llm-api-config.yml
_EXTRA_CONFIG_KEYS = {
    "free_gpu_memory_fraction", "kv_cache_dtype", "kv_cache_enable_block_reuse",
    "cuda_graph_padding", "cuda_graph_max_batch_size",
    "moe_backend",
    "print_iter_log",
}


class TrtllmRuntime(RuntimePlugin):
    """TensorRT-LLM runtime using MPI for multi-node inference.

    Multi-node orchestration uses ``mpirun`` with a custom rsh wrapper
    script that routes through host-level SSH and ``docker exec``.
    This avoids installing openssh-server inside containers.

    Solo mode (single host) uses the standard sleep-infinity + exec
    pattern inherited from the base class.
    """

    runtime_name = "trtllm"
    default_image_prefix = "nvcr.io/nvidia/tensorrt-llm/release"

    def cluster_strategy(self) -> str:
        """TRT-LLM uses native clustering with MPI orchestration."""
        return "native"

    def _augment_extra_config_flag(self, command: str, recipe: Recipe,
                                    overrides: dict[str, Any] | None = None) -> str:
        """Append ``--extra_llm_api_options`` if extra config keys are present.

        When ``_build_extra_config`` produces YAML content but the flag
        is not already in the command string, append it so the config
        file written by ``_pre_serve`` (solo) or ``_run_cluster`` is
        actually consumed by ``trtllm-serve``.
        """
        if _EXTRA_CONFIG_FLAG in command:
            return command
        if self._build_extra_config(recipe, overrides) is None:
            return command
        return "%s %s %s" % (command.rstrip(), _EXTRA_CONFIG_FLAG, _EXTRA_CONFIG_PATH)

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None,
                         skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Generate the trtllm-serve command.

        For cluster mode, the command is later wrapped by ``mpirun``
        via ``_build_mpirun_command``.  This method produces just the
        ``trtllm-serve`` invocation.
        """
        config = recipe.build_config_chain(overrides)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_extra_config_flag(rendered, recipe, overrides)
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered, skip_keys, _TRTLLM_FLAG_MAP, _TRTLLM_BOOL_FLAGS,
                )
            return rendered

        cmd = self._build_command(recipe, config, skip_keys=skip_keys)
        return self._augment_extra_config_flag(cmd, recipe, overrides)

    def _build_command(self, recipe: Recipe, config,
                       skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the trtllm-serve command from structured config."""
        parts = ["trtllm-serve", recipe.model]

        # Default to pytorch backend if not specified
        backend = config.get("backend")
        if not backend:
            parts.extend(["--backend", "pytorch"])

        parts.extend(self.build_flags_from_map(
            config, _TRTLLM_FLAG_MAP,
            bool_keys=_TRTLLM_BOOL_FLAGS,
            skip_keys=skip_keys,
        ))

        return " ".join(parts)

    def _build_mpirun_command(
            self,
            serve_cmd: str,
            host_ips: list[str],
            nccl_env: dict[str, str] | None = None,
            extra_env_keys: list[str] | None = None,
    ) -> str:
        """Wrap a trtllm-serve command with mpirun for multi-node execution.

        Args:
            serve_cmd: The ``trtllm-serve`` command string.
            host_ips: Management IPs for all hosts (order = rank order).
            nccl_env: NCCL environment variables to propagate via ``-x``.
            extra_env_keys: Additional env var names to propagate.

        Returns:
            Complete ``mpirun`` command string.
        """
        parts = [
            "mpirun", "--allow-run-as-root",
            "--mca", "plm_rsh_agent", "/tmp/sparkrun-rsh-wrapper.sh",
            "--mca", "rmaps_ppr_n_pernode", "1",
            "-H", ",".join(host_ips),
        ]

        # Propagate NCCL env vars
        env_keys_to_pass = set()
        if nccl_env:
            env_keys_to_pass.update(nccl_env.keys())
        # Always propagate common keys
        env_keys_to_pass.update([
            "NCCL_SOCKET_IFNAME", "UCX_NET_DEVICES",
            "OMPI_MCA_btl_tcp_if_include", "HF_TOKEN",
            "NCCL_CUMEM_ENABLE",
        ])
        if extra_env_keys:
            env_keys_to_pass.update(extra_env_keys)

        for key in sorted(env_keys_to_pass):
            parts.extend(["-x", key])

        # Wrap serve command with trtllm-llmapi-launch
        parts.extend(["trtllm-llmapi-launch", serve_cmd])

        return " ".join(parts)

    @staticmethod
    def _generate_rsh_wrapper(
            host_ip_map: dict[str, str],
            cluster_id: str,
            ssh_key_path: str = "/tmp/.ssh/id_ed25519",
    ) -> str:
        """Generate the bash rsh wrapper script for mpirun.

        The wrapper is written into the head container and used as
        ``--mca plm_rsh_agent``.  It SSHes to the worker HOST
        (using existing host-level sshd) and ``docker exec``s
        the MPI command into the worker container.

        Args:
            host_ip_map: Mapping of management IP to container name.
            cluster_id: Cluster ID for container naming (unused here,
                names are in host_ip_map values).
            ssh_key_path: Path to SSH private key inside the container.

        Returns:
            Complete bash script as a string.
        """
        lines = [
            "#!/bin/bash",
            "# mpirun rsh agent: SSH to worker HOST, docker exec into container",
            "HOST=$1; shift",
            "case $HOST in",
        ]
        for ip, container_name in sorted(host_ip_map.items()):
            lines.append('    %s) CONTAINER="%s" ;;' % (ip, container_name))
        lines.extend([
            '    *) echo "Unknown host: $HOST" >&2; exit 1 ;;',
            "esac",
            'exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\',
            '  -i %s "$HOST" docker exec "$CONTAINER" "$@"' % ssh_key_path,
        ])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_extra_config(recipe: Recipe, overrides: dict[str, Any] | None = None) -> str | None:
        """Generate extra-llm-api-config.yml YAML content from recipe defaults.

        Returns ``None`` if no extra config keys are present in the recipe.
        """
        config = recipe.build_config_chain(overrides)
        extra: dict[str, Any] = {}

        # print_iter_log
        print_log = config.get("print_iter_log")
        if print_log is not None:
            extra["print_iter_log"] = bool(print_log)

        # kv_cache_config
        kv_cache: dict[str, Any] = {}
        frac = config.get("free_gpu_memory_fraction")
        if frac is not None:
            kv_cache["free_gpu_memory_fraction"] = float(frac)
        kv_dtype = config.get("kv_cache_dtype")
        if kv_dtype is not None:
            kv_cache["dtype"] = str(kv_dtype)
        block_reuse = config.get("kv_cache_enable_block_reuse")
        if block_reuse is not None:
            kv_cache["enable_block_reuse"] = bool(block_reuse)
        if kv_cache:
            extra["kv_cache_config"] = kv_cache

        # cuda_graph_config
        cuda_graph: dict[str, Any] = {}
        padding = config.get("cuda_graph_padding")
        if padding is not None:
            cuda_graph["enable_padding"] = bool(padding)
        cg_max_batch = config.get("cuda_graph_max_batch_size")
        if cg_max_batch is not None:
            cuda_graph["max_batch_size"] = int(cg_max_batch)
        if cuda_graph:
            extra["cuda_graph_config"] = cuda_graph

        # moe_config
        moe: dict[str, Any] = {}
        moe_backend = config.get("moe_backend")
        if moe_backend is not None:
            moe["backend"] = str(moe_backend)
        if moe:
            extra["moe_config"] = moe

        if not extra:
            return None

        return yaml.safe_dump(extra, default_flow_style=False)

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return TRT-LLM cluster environment variables."""
        return {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "OMPI_MCA_rmaps_ppr_n_pernode": "1",
        }

    def get_extra_docker_opts(self) -> list[str]:
        """Return ulimit flags required by TRT-LLM."""
        return list(_TRTLLM_EXTRA_DOCKER_OPTS)

    def get_extra_volumes(self) -> dict[str, str]:
        """Mount SSH keys for mpirun inter-node communication."""
        ssh_dir = Path.home() / ".ssh"
        if ssh_dir.is_dir():
            return {str(ssh_dir): "/tmp/.ssh:ro"}
        return {}

    def _pre_serve(
            self,
            hosts_containers: list[tuple[str, str]],
            ssh_kwargs: dict,
            dry_run: bool,
            recipe: Recipe | None = None,
            config_chain=None,
    ) -> None:
        """Write extra-llm-api-config.yml into containers before serve.

        Extends the base ``_pre_serve`` to inject the extra LLM API config
        YAML (kv_cache_config, cuda_graph_config, moe_config, etc.) into
        every container.  This is the solo-mode counterpart of the config
        injection that ``_run_cluster`` performs in step 6.
        """
        super()._pre_serve(hosts_containers, ssh_kwargs, dry_run, recipe=recipe, config_chain=config_chain)

        if recipe is None:
            return
        extra_yaml = self._build_extra_config(recipe)
        if extra_yaml is None:
            return

        from sparkrun.orchestration.ssh import run_remote_command
        from sparkrun.orchestration.docker import docker_exec_cmd

        write_cmd = (
            "cat > %s << 'SPARKRUN_EOF'\n"
            "%s"
            "SPARKRUN_EOF"
        ) % (_EXTRA_CONFIG_PATH, extra_yaml)

        for host, container_name in hosts_containers:
            exec_cmd = docker_exec_cmd(container_name, write_cmd)
            result = run_remote_command(
                host, exec_cmd, timeout=30, dry_run=dry_run, **ssh_kwargs,
            )
            if not result.success and not dry_run:
                logger.warning(
                    "Failed to write extra-llm-api-config.yml on %s: %s",
                    host, result.stderr[:100],
                )
            else:
                logger.info("Wrote extra-llm-api-config.yml into %s on %s", container_name, host)

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate TRT-LLM-specific recipe fields."""
        issues = super().validate_recipe(recipe)

        # Warn about SSH keys for multi-node
        if recipe.min_nodes and recipe.min_nodes > 1:
            ssh_dir = Path.home() / ".ssh"
            if not ssh_dir.is_dir():
                issues.append(
                    "[trtllm] Multi-node requires SSH keys at ~/.ssh for "
                    "mpirun inter-node communication. No ~/.ssh directory found."
                )

        return issues

    def _head_container_name(self, cluster_id: str) -> str:
        """TRT-LLM names the head container ``{cluster_id}_node_0``."""
        from sparkrun.orchestration.docker import generate_node_container_name
        return generate_node_container_name(cluster_id, 0)

    def _cluster_log_mode(self) -> str:
        """TRT-LLM cluster uses docker logs (mpirun output goes to stdout)."""
        return "docker"

    # --- Cluster stop ---

    def _stop_cluster(
            self,
            hosts: list[str],
            cluster_id: str,
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> int:
        """Stop a TRT-LLM native cluster."""
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)

    # --- Cluster launch ---

    def _run_cluster(
            self,
            hosts: list[str],
            image: str,
            serve_command: str = "",
            recipe: Recipe | None = None,
            overrides: dict[str, Any] | None = None,
            *,
            cluster_id: str = "sparkrun0",
            env: dict[str, str] | None = None,
            cache_dir: str | None = None,
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            detached: bool = True,
            nccl_env: dict[str, str] | None = None,
            skip_keys: set[str] | frozenset[str] = frozenset(),
            **kwargs,
    ) -> int:
        """Orchestrate a multi-node TRT-LLM cluster using MPI.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand (resolve_nccl_env).
        3. Detect management IPs on all hosts.
        4. Launch all containers with sleep infinity (parallel).
        5. Verify containers are running on all hosts.
        6. Write rsh wrapper + extra config into head container.
        7. Exec mpirun on head container.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            build_volumes,
            merge_env,
            detect_host_ip,
            is_container_running,
            resolve_nccl_env,
        )
        from sparkrun.orchestration.ssh import (
            run_remote_script, run_remote_command,
        )
        from sparkrun.orchestration.docker import (
            docker_stop_cmd, docker_exec_cmd,
            generate_node_container_name,
        )
        from sparkrun.orchestration.scripts import generate_container_launch_script

        num_nodes = len(hosts)
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir, extra=self.get_extra_volumes())
        runtime_env = self.get_cluster_env(head_ip="<pending>", num_nodes=num_nodes)
        all_env = merge_env(runtime_env, env)
        extra_docker_opts = self.get_extra_docker_opts()

        self._print_cluster_banner(
            "TRT-LLM MPI Cluster Launcher", hosts, image, cluster_id,
            {"Nodes": num_nodes}, dry_run,
        )

        # Step 1: Cleanup
        t0 = time.monotonic()
        logger.info("Step 1/7: Cleaning up existing containers for cluster '%s'...", cluster_id)
        for rank, host in enumerate(hosts):
            container_name = generate_node_container_name(cluster_id, rank)
            run_remote_command(
                host, docker_stop_cmd(container_name),
                timeout=30, dry_run=dry_run, **ssh_kwargs,
            )
        logger.info("Step 1/7: Cleanup done (%.1fs)", time.monotonic() - t0)

        # Step 2: InfiniBand detection
        t0 = time.monotonic()
        logger.info("Step 2/7: InfiniBand detection...")
        nccl_env = resolve_nccl_env(
            nccl_env, hosts,
            head_host=hosts[0], ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )
        logger.info("Step 2/7: IB step done (%.1fs)", time.monotonic() - t0)

        # Step 3: Detect management IPs on all hosts
        t0 = time.monotonic()
        logger.info("Step 3/7: Detecting management IPs on %d host(s)...", num_nodes)
        host_ip_map: dict[str, str] = {}  # management_ip -> container_name
        host_ips: list[str] = []  # ordered by rank
        for rank, host in enumerate(hosts):
            try:
                ip = detect_host_ip(host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
            except RuntimeError as e:
                logger.error("%s", e)
                return 1
            container_name = generate_node_container_name(cluster_id, rank)
            host_ip_map[ip] = container_name
            host_ips.append(ip)
            logger.info("  Rank %d: %s -> %s (IP: %s)", rank, host, container_name, ip)
        logger.info("Step 3/7: IP detection done (%.1fs)", time.monotonic() - t0)

        # Step 4: Launch all containers with sleep infinity (parallel)
        t0 = time.monotonic()
        logger.info("Step 4/7: Launching %d container(s) with sleep infinity...", num_nodes)
        with ThreadPoolExecutor(max_workers=num_nodes) as executor:
            futures = {}
            for rank, host in enumerate(hosts):
                container_name = generate_node_container_name(cluster_id, rank)
                launch_script = generate_container_launch_script(
                    image=image,
                    container_name=container_name,
                    command="sleep infinity",
                    env=all_env,
                    volumes=volumes,
                    nccl_env=nccl_env,
                    extra_docker_opts=extra_docker_opts or None,
                )
                future = executor.submit(
                    run_remote_script, host, launch_script,
                    timeout=120, dry_run=dry_run, **ssh_kwargs,
                )
                futures[future] = (host, rank)

            for future in as_completed(futures):
                host, rank = futures[future]
                result = future.result()
                if not result.success and not dry_run:
                    logger.error(
                        "Failed to launch container on %s (rank %d): %s",
                        host, rank, result.stderr[:200],
                    )
                    return 1
        logger.info("Step 4/7: Containers launched (%.1fs)", time.monotonic() - t0)

        # Step 5: Verify containers are running
        t0 = time.monotonic()
        logger.info("Step 5/7: Verifying containers are running...")
        if not dry_run:
            for rank, host in enumerate(hosts):
                container_name = generate_node_container_name(cluster_id, rank)
                if not is_container_running(host, container_name, ssh_kwargs=ssh_kwargs):
                    logger.error(
                        "Container %s not running on %s (rank %d)",
                        container_name, host, rank,
                    )
                    return 1
            logger.info("Step 5/7: All containers verified (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 5/7: [dry-run] Would verify containers")

        # Step 6: Write rsh wrapper + extra config into head container
        t0 = time.monotonic()
        head_host = hosts[0]
        head_container = generate_node_container_name(cluster_id, 0)
        logger.info("Step 6/7: Writing rsh wrapper into head container %s...", head_container)

        # Determine SSH key path inside container
        ssh_key_path = "/tmp/.ssh/id_ed25519"

        # Generate rsh wrapper
        rsh_wrapper = self._generate_rsh_wrapper(
            host_ip_map, cluster_id, ssh_key_path=ssh_key_path,
        )

        # Write wrapper into head container
        write_wrapper_cmd = (
            "cat > /tmp/sparkrun-rsh-wrapper.sh << 'SPARKRUN_EOF'\n"
            "%s"
            "SPARKRUN_EOF\n"
            "chmod +x /tmp/sparkrun-rsh-wrapper.sh"
        ) % rsh_wrapper

        exec_write = docker_exec_cmd(head_container, write_wrapper_cmd)
        result = run_remote_command(
            head_host, exec_write, timeout=30, dry_run=dry_run, **ssh_kwargs,
        )
        if not result.success and not dry_run:
            logger.error("Failed to write rsh wrapper: %s", result.stderr[:200])
            return 1

        # Write extra-llm-api-config.yml if needed
        if recipe is not None:
            extra_config_yaml = self._build_extra_config(recipe, overrides)
            if extra_config_yaml:
                write_config_cmd = (
                    "cat > %s << 'SPARKRUN_EOF'\n"
                    "%s"
                    "SPARKRUN_EOF"
                ) % (_EXTRA_CONFIG_PATH, extra_config_yaml)
                exec_config = docker_exec_cmd(head_container, write_config_cmd)
                result = run_remote_command(
                    head_host, exec_config, timeout=30, dry_run=dry_run, **ssh_kwargs,
                )
                if not result.success and not dry_run:
                    logger.error("Failed to write extra config: %s", result.stderr[:200])
                    return 1
                logger.info("  Extra LLM API config written to %s", _EXTRA_CONFIG_PATH)

        logger.info("Step 6/7: Rsh wrapper written (%.1fs)", time.monotonic() - t0)

        # Step 7: Exec mpirun on head container
        t0 = time.monotonic()
        logger.info("Step 7/7: Executing mpirun on head container...")

        # Build the trtllm-serve command (generate_command auto-appends
        # --extra_llm_api_options when extra config keys are present)
        if serve_command:
            trtllm_cmd = serve_command
        elif recipe is not None:
            trtllm_cmd = self.generate_command(
                recipe, overrides or {}, is_cluster=True,
                num_nodes=num_nodes, head_ip=host_ips[0],
                skip_keys=skip_keys,
            )
        else:
            logger.error("No serve command or recipe provided")
            return 1

        # Build mpirun command
        mpirun_cmd = self._build_mpirun_command(
            trtllm_cmd, host_ips, nccl_env=nccl_env,
        )

        logger.info("mpirun command:")
        for line in mpirun_cmd.strip().splitlines():
            logger.info("  %s", line)

        # Exec mpirun on head container (detached)
        detach_flag = "-d" if detached else ""
        escaped_mpirun = mpirun_cmd.replace("'", "'\\''")
        exec_mpirun = "docker exec %s %s bash -c '%s'" % (
            detach_flag, head_container, escaped_mpirun,
        )
        result = run_remote_command(
            head_host, exec_mpirun, timeout=60, dry_run=dry_run, **ssh_kwargs,
        )
        logger.info("Step 7/7: mpirun dispatched (%.1fs)", time.monotonic() - t0)

        if not dry_run and not result.success:
            logger.error("mpirun exec failed: %s", result.stderr[:200])
            return 1

        self._print_connection_info(hosts, cluster_id, per_node_logs=True)
        return 0
