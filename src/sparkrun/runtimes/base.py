"""Base class for sparkrun runtimes."""

from __future__ import annotations

import logging
from abc import abstractmethod
from logging import Logger
from typing import Any, TYPE_CHECKING

from scitrera_app_framework import Plugin, Variables, ext_parse_bool

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

EXT_RUNTIME = "sparkrun.runtime"


class RuntimePlugin(Plugin):
    """Abstract base class for sparkrun inference runtimes.

    Each runtime is an SAF Plugin that registers as a multi-extension
    under the 'sparkrun.runtime' extension point. Multiple runtimes
    can coexist simultaneously.

    Subclasses must define:
        - runtime_name: str identifier (e.g. "vllm", "sglang")
        - generate_command(): produce the serve command from a recipe
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    runtime_name: str = ""
    default_image_prefix: str = ""

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.runtime.%s" % self.runtime_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_RUNTIME

    def is_enabled(self, v: Variables) -> bool:
        # Must return False for multi-extension plugins to prevent SAF's
        # single-extension cache (er[ext_name]) from short-circuiting
        # subsequent plugin initializations under the same extension point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: Logger) -> RuntimePlugin:
        return self

    # --- Runtime interface ---

    @abstractmethod
    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None,
                         skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Generate the serve command string from recipe + CLI overrides.

        Args:
            recipe: The loaded recipe
            overrides: CLI override values (e.g. --port 9000)
            is_cluster: Whether running in multi-node mode
            num_nodes: Total number of nodes in the cluster
            head_ip: Head node IP (only set for cluster mode)
            skip_keys: Config keys to omit from the generated command.
                Used by the benchmark flow to suppress ``served_model_name``
                so the server responds to the raw HF model ID.

        Returns:
            The full command string to execute inside the container
        """
        ...

    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve the container image to use.

        Returns the recipe's explicit container if set, otherwise
        falls back to ``{default_image_prefix}:latest``.

        Subclasses may override for custom resolution logic.
        """
        if recipe.container:
            return recipe.container
        return "%s:latest" % self.default_image_prefix

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return runtime-specific environment variables for cluster mode.

        Override in subclasses to inject runtime-specific cluster config.
        """
        return {}

    def cluster_strategy(self) -> str:
        """Return the clustering strategy for multi-node mode.

        Returns:
            ``"ray"`` — use Ray cluster orchestration (start Ray head/workers,
            then exec serve command on head). This is the default.

            ``"native"`` — the runtime handles its own distribution. Each node
            runs the serve command directly with node-rank arguments appended.
            Used by sglang, which has built-in multi-node support via
            ``--dist-init-addr``, ``--nnodes``, ``--node-rank``.
        """
        return "ray"

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
        """Generate the serve command for a specific node in native clustering.

        Only called when :meth:`cluster_strategy` returns ``"native"``.

        Args:
            recipe: The loaded recipe.
            overrides: CLI override values.
            head_ip: Head node IP address.
            num_nodes: Total number of nodes.
            node_rank: This node's rank (0 = head).
            init_port: Coordination port for distributed init.
            skip_keys: Config keys to omit from the generated command.

        Returns:
            The full command string for this node.
        """
        raise NotImplementedError(
            "%s does not implement native clustering" % type(self).__name__
        )

    def prepare(
            self,
            recipe: Recipe,
            hosts: list[str],
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> None:
        """Pre-launch preparation (e.g., building container images).

        Called by the CLI before resource distribution.  Override in
        subclasses that need to build or transform images before they
        can be distributed to hosts.
        """
        pass

    def _pre_serve(
            self,
            hosts_containers: list[tuple[str, str]],
            ssh_kwargs: dict,
            dry_run: bool,
    ) -> None:
        """Hook called after containers are launched but before serve command.

        Override in subclasses to apply modifications (e.g., eugr mods)
        to containers before the inference server starts.

        Args:
            hosts_containers: List of (host, container_name) pairs.
            ssh_kwargs: SSH connection kwargs.
            dry_run: Dry-run mode.
        """
        pass

    def get_extra_volumes(self) -> dict[str, str]:
        """Return additional volume mounts for this runtime.

        Override in subclasses to inject runtime-specific volumes
        (e.g. tuning config directories).  Called by ``_run_solo``
        and cluster launch methods.

        Returns:
            Dict of host_path -> container_path (empty by default).
        """
        return {}

    def get_extra_env(self) -> dict[str, str]:
        """Return additional environment variables for this runtime.

        Override in subclasses to inject runtime-specific env vars
        (e.g. tuning config path env vars).  Called by ``_run_solo``
        and cluster launch methods.

        Returns:
            Dict of env var name -> value (empty by default).
        """
        return {}

    def get_extra_docker_opts(self) -> list[str]:
        """Return additional ``docker run`` options for this runtime.

        Override in subclasses to inject runtime-specific docker flags
        (e.g. ulimit settings).  Called by ``_run_solo`` and
        ``_generate_node_script``.

        Returns:
            List of extra docker CLI arguments (empty by default).
        """
        return []

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Return list of warnings/errors for runtime-specific fields.

        The base implementation checks that a model is specified.
        Subclasses should call ``super().validate_recipe(recipe)`` and
        extend the returned list with runtime-specific checks.
        """
        issues = []
        if not recipe.model:
            issues.append("[%s] model is required" % self.runtime_name)
        return issues

    def compute_required_nodes(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> int | None:
        """Compute the number of nodes required to run this recipe.

        The base implementation reads ``tensor_parallel`` from the
        recipe config chain.  Subclasses override to account for
        additional parallelism dimensions (e.g. pipeline parallelism).

        Args:
            recipe: The loaded recipe.
            overrides: CLI override values (merged into config chain).

        Returns:
            Required node count, or ``None`` if no parallelism config
            is set (meaning "use all provided hosts, no trimming").
        """
        config = recipe.build_config_chain(overrides or {})
        tp_val = config.get("tensor_parallel")
        if tp_val is None:
            return None
        return int(tp_val)

    @staticmethod
    def build_flags_from_map(
            config,
            flag_map: dict[str, str],
            bool_keys: set[str] | frozenset[str] = frozenset(),
            skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> list[str]:
        """Build CLI flag list from a config-key to CLI-flag mapping.

        Iterates *flag_map* and looks up each key in *config*.  Keys in
        *bool_keys* are treated as boolean toggles (flag appended when
        truthy, omitted otherwise).  All other keys emit ``[flag, value]``
        pairs.  Keys listed in *skip_keys* are skipped entirely.

        Args:
            config: Config chain object (must support ``.get(key)``).
            flag_map: Mapping of recipe config key to CLI flag string.
            bool_keys: Set of keys that should be treated as boolean flags.
            skip_keys: Keys to skip (already handled by the caller).

        Returns:
            Flat list of CLI argument strings.
        """
        parts: list[str] = []
        for key, flag in flag_map.items():
            if key in skip_keys:
                continue
            value = config.get(key)
            if value is None:
                continue
            if key in bool_keys:
                if ext_parse_bool(value):
                    parts.append(flag)
            else:
                parts.extend([flag, str(value)])
        return parts

    @staticmethod
    def strip_flags_from_command(
            command: str,
            skip_keys: set[str] | frozenset[str],
            flag_map: dict[str, str],
            bool_keys: set[str] | frozenset[str] = frozenset(),
            flag_aliases: dict[str, list[str]] | None = None,
    ) -> str:
        """Strip CLI flags for *skip_keys* from a rendered command string.

        Used when ``recipe.render_command()`` produces the command via template
        substitution, bypassing ``build_flags_from_map()``'s skip_keys support.
        Each runtime calls this with its own flag_map.

        Args:
            command: The rendered command string.
            skip_keys: Config keys whose flags should be removed.
            flag_map: Mapping of config key to CLI flag string.
            bool_keys: Set of keys treated as boolean (flag-only, no value).
            flag_aliases: Optional mapping of config key to additional flag
                forms (e.g. short flags) that should also be stripped.

        Returns:
            Command string with the specified flags removed.
        """
        import re
        for key in skip_keys:
            # Collect all flag forms for this key: canonical + aliases
            flags_to_strip: list[str] = []
            canonical = flag_map.get(key)
            if canonical:
                flags_to_strip.append(canonical)
            if flag_aliases and key in flag_aliases:
                flags_to_strip.extend(flag_aliases[key])
            if not flags_to_strip:
                continue

            for flag in flags_to_strip:
                escaped = re.escape(flag)
                if key in bool_keys:
                    command = re.sub(r'\s*' + escaped + r'(?=\s|$)', '', command)
                else:
                    # Match the flag, its value, and an optional trailing
                    # backslash continuation on the same line.
                    command = re.sub(
                        escaped + r'\s+\S+\s*\\?\s*\n?', '', command,
                    )

        # Clean up artifacts from removed lines:
        # - collapse double backslash-continuations (``\ \``) into one
        # - remove blank continuation lines (``\`` followed by only whitespace)
        command = re.sub(r'\\\s*\\\s*\n', '\\\n', command)
        command = re.sub(r'\\\s*\n(\s*\\\s*\n)', r'\\\n', command)
        # Remove lines that are only whitespace (left behind after removal)
        command = re.sub(r'\n\s*\n', '\n', command)
        return command

    def is_delegating_runtime(self) -> bool:
        """True if this runtime delegates entirely to external scripts.

        Delegating runtimes bypass sparkrun's orchestration layer and
        instead call external tools directly.  No built-in runtimes
        currently delegate — all use native orchestration.
        """
        return False

    # --- Log following interface ---

    def follow_logs(
            self,
            hosts: list[str],
            cluster_id: str = "sparkrun0",
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            tail: int = 100,
    ) -> None:
        """Follow container logs after a successful launch.

        Solo mode tails the serve log file inside the container
        (``/tmp/sparkrun_serve.log``), which is the correct approach
        for all runtimes using the sleep-infinity + exec pattern.

        Cluster mode delegates to :meth:`_follow_cluster_logs`, which
        subclasses should override.
        """
        if len(hosts) <= 1:
            from sparkrun.orchestration.primitives import build_ssh_kwargs
            from sparkrun.orchestration.docker import generate_container_name
            from sparkrun.orchestration.ssh import stream_container_file_logs

            host = hosts[0] if hosts else "localhost"
            container_name = generate_container_name(cluster_id, "solo")
            ssh_kwargs = build_ssh_kwargs(config)
            stream_container_file_logs(
                host, container_name, tail=tail, dry_run=dry_run, **ssh_kwargs,
            )
            return

        self._follow_cluster_logs(hosts, cluster_id, config, dry_run, tail)

    def _follow_cluster_logs(
            self,
            hosts: list[str],
            cluster_id: str,
            config: SparkrunConfig | None,
            dry_run: bool,
            tail: int,
    ) -> None:
        """Follow logs for a multi-node cluster.

        Uses :meth:`_cluster_log_mode` and :meth:`_head_container_name`
        to determine the log tailing strategy and target container.
        Subclasses control behaviour by overriding those hooks rather
        than this method.
        """
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        ssh_kwargs = build_ssh_kwargs(config)
        container_name = self._head_container_name(cluster_id)

        if self._cluster_log_mode() == "file":
            from sparkrun.orchestration.ssh import stream_container_file_logs
            stream_container_file_logs(
                hosts[0], container_name, tail=tail, dry_run=dry_run, **ssh_kwargs,
            )
        else:
            from sparkrun.orchestration.ssh import stream_remote_logs
            stream_remote_logs(
                hosts[0], container_name, tail=tail, dry_run=dry_run, **ssh_kwargs,
            )

    def get_head_container_name(self, cluster_id: str, is_solo: bool = False) -> str:
        """Return the expected head/solo container name for *cluster_id*.

        Solo mode always uses ``{cluster_id}_solo``.  Cluster mode
        delegates to :meth:`_head_container_name` which subclasses
        override when they use non-standard naming (e.g.
        ``{cluster_id}_node_0`` for SGLang and vLLM distributed).
        """
        from sparkrun.orchestration.docker import generate_container_name
        if is_solo:
            return generate_container_name(cluster_id, "solo")
        return self._head_container_name(cluster_id)

    def _head_container_name(self, cluster_id: str) -> str:
        """Return the head container name for log following.

        Override in subclasses that use non-standard naming.
        The default returns ``{cluster_id}_head``.
        """
        from sparkrun.orchestration.docker import generate_container_name
        return generate_container_name(cluster_id, "head")

    def _cluster_log_mode(self) -> str:
        """Return the log tailing mode for cluster containers.

        ``"file"`` uses :func:`stream_container_file_logs` (tails a log
        file inside the container).  ``"docker"`` uses
        :func:`stream_remote_logs` (``docker logs``).

        Default is ``"docker"``.  Override to ``"file"`` for runtimes
        that use the sleep-infinity + exec pattern in cluster mode.
        """
        return "docker"

    # --- Launch / Stop interface ---
    #
    # The base class handles the solo-vs-cluster dispatch.  Runtimes that
    # support multi-node clustering override ``_run_cluster`` and
    # ``_stop_cluster`` to compose their specific flow from orchestration
    # primitives.

    def run(
            self,
            hosts: list[str],
            image: str,
            serve_command: str,
            recipe: Recipe,
            overrides: dict[str, Any],
            *,
            cluster_id: str = "sparkrun0",
            env: dict[str, str] | None = None,
            cache_dir: str | None = None,
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            detached: bool = True,
            nccl_env: dict[str, str] | None = None,
            ib_ip_map: dict[str, str] | None = None,
            skip_keys: set[str] | frozenset[str] = frozenset(),
            **kwargs,
    ) -> int:
        """Launch a workload -- delegates to solo or cluster implementation.

        Args:
            hosts: List of hostnames/IPs (first = head).
            image: Container image to use.
            serve_command: The inference serve command to run.
            recipe: The loaded recipe.
            overrides: CLI override values.
            cluster_id: Identifier for container naming.
            env: Additional environment variables from the recipe.
            cache_dir: HuggingFace cache directory path.
            config: SparkrunConfig instance for SSH settings.
            dry_run: Show what would be done without executing.
            detached: Run serve command in background.
            nccl_env: Pre-detected NCCL environment variables.  When
                provided (not ``None``), skips runtime IB detection and
                uses this env directly.
            ib_ip_map: Pre-detected InfiniBand IP mapping
                (management host -> IB IP).  Used by runtimes that need
                IB addresses for inter-node communication (e.g. llama.cpp
                RPC).  When ``None``, the runtime may detect IB IPs
                itself if ``nccl_env`` is also ``None``.
            skip_keys: Config keys to omit when the runtime regenerates
                serve commands internally (e.g. native-cluster runtimes
                that call ``generate_node_command()`` instead of using
                the pre-built *serve_command*).
            **kwargs: Runtime-specific keyword arguments (e.g. ray_port,
                dashboard_port, init_port, rpc_port).

        Returns:
            Exit code (0 = success).
        """
        if len(hosts) <= 1:
            return self._run_solo(
                host=hosts[0] if hosts else "localhost",
                image=image,
                serve_command=serve_command,
                cluster_id=cluster_id,
                env=env,
                cache_dir=cache_dir,
                config=config,
                dry_run=dry_run,
                detached=detached,
                nccl_env=nccl_env,
            )
        return self._run_cluster(
            hosts=hosts,
            image=image,
            serve_command=serve_command,
            recipe=recipe,
            overrides=overrides,
            cluster_id=cluster_id,
            env=env,
            cache_dir=cache_dir,
            config=config,
            dry_run=dry_run,
            detached=detached,
            nccl_env=nccl_env,
            ib_ip_map=ib_ip_map,
            skip_keys=skip_keys,
            **kwargs,
        )

    def _run_cluster(
            self,
            hosts: list[str],
            image: str,
            serve_command: str,
            recipe: Recipe,
            overrides: dict[str, Any],
            **kwargs,
    ) -> int:
        """Launch a multi-node cluster workload.

        Override in subclasses to implement cluster launch.
        The default raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "Cluster mode not supported by %s" % self.runtime_name
        )

    def stop(
            self,
            hosts: list[str],
            cluster_id: str = "sparkrun0",
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> int:
        """Stop a running workload -- delegates to solo or cluster implementation.

        Args:
            hosts: List of hostnames/IPs in the workload.
            cluster_id: Cluster identifier used when launching.
            config: SparkrunConfig instance for SSH settings.
            dry_run: Show what would be done without executing.

        Returns:
            Exit code (0 = success).
        """
        if len(hosts) <= 1:
            return self._stop_solo(
                host=hosts[0] if hosts else "localhost",
                cluster_id=cluster_id,
                config=config,
                dry_run=dry_run,
            )
        return self._stop_cluster(
            hosts=hosts,
            cluster_id=cluster_id,
            config=config,
            dry_run=dry_run,
        )

    def _stop_cluster(
            self,
            hosts: list[str],
            cluster_id: str,
            config: SparkrunConfig | None,
            dry_run: bool,
    ) -> int:
        """Stop a multi-node cluster workload.

        Override in subclasses to implement cluster teardown.
        The default raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "Cluster stop not supported by %s" % self.runtime_name
        )

    # --- Default solo implementation (used by base and simple runtimes) ---

    def _run_solo(
            self,
            host: str,
            image: str,
            serve_command: str,
            cluster_id: str = "sparkrun0",
            env: dict[str, str] | None = None,
            cache_dir: str | None = None,
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            detached: bool = True,
            nccl_env: dict[str, str] | None = None,
    ) -> int:
        """Launch a single-node inference workload.

        Steps:
        1. Detect InfiniBand on the target host (optional).
        2. Launch container with ``sleep infinity``.
        3. Execute the serve command inside the container.
        """
        import time
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            build_volumes,
            merge_env,
            detect_infiniband,
            detect_infiniband_local,
            run_script_on_host,
        )
        from sparkrun.orchestration.docker import generate_container_name
        from sparkrun.orchestration.scripts import (
            generate_container_launch_script,
            generate_exec_serve_script,
        )
        from sparkrun.core.hosts import is_local_host

        is_local = is_local_host(host)
        container_name = generate_container_name(cluster_id, "solo")
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir, extra=self.get_extra_volumes())
        all_env = merge_env(env, self.get_extra_env())

        # Step 1: InfiniBand detection (skip if pre-detected nccl_env provided)
        t0 = time.monotonic()
        if nccl_env is not None:
            logger.info("Step 1/3: Using pre-detected NCCL env (%d vars)", len(nccl_env))
        else:
            logger.info("Step 1/3: Detecting InfiniBand on %s...", host)
            if is_local:
                nccl_env = detect_infiniband_local(dry_run=dry_run)
            else:
                nccl_env = detect_infiniband(
                    [host], ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                )
            logger.info("Step 1/3: IB detection done (%.1fs)", time.monotonic() - t0)

        # Step 2: Launch container
        t0 = time.monotonic()
        logger.info(
            "Step 2/3: Launching container %s on %s (image: %s)...",
            container_name, host, image,
        )
        launch_script = generate_container_launch_script(
            image=image,
            container_name=container_name,
            command="sleep infinity",
            env=all_env,
            volumes=volumes,
            nccl_env=nccl_env,
            extra_docker_opts=self.get_extra_docker_opts() or None,
        )
        result = run_script_on_host(
            host, launch_script, ssh_kwargs=ssh_kwargs, timeout=120, dry_run=dry_run,
        )
        if not result.success and not dry_run:
            logger.error("Failed to launch container: %s", result.stderr)
            return 1
        logger.info("Step 2/3: Container launched (%.1fs)", time.monotonic() - t0)

        # Pre-serve hook (e.g., apply mods to container)
        self._pre_serve([(host, container_name)], ssh_kwargs, dry_run)

        # Step 3: Execute serve command
        t0 = time.monotonic()
        logger.info("Step 3/3: Executing serve command in %s...", container_name)
        logger.debug("Serve command: %s", serve_command)
        exec_script = generate_exec_serve_script(
            container_name=container_name,
            serve_command=serve_command,
            env=all_env,
            detached=detached,
        )
        result = run_script_on_host(
            host, exec_script, ssh_kwargs=ssh_kwargs, timeout=60, dry_run=dry_run,
        )
        logger.info("Step 3/3: Serve command dispatched (%.1fs)", time.monotonic() - t0)

        if dry_run:
            return 0
        return result.returncode

    def _stop_solo(
            self,
            host: str,
            cluster_id: str = "sparkrun0",
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> int:
        """Stop a solo workload by removing the container."""
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            cleanup_containers,
            cleanup_containers_local,
        )
        from sparkrun.orchestration.docker import generate_container_name
        from sparkrun.core.hosts import is_local_host

        container_name = generate_container_name(cluster_id, "solo")
        is_local = is_local_host(host)

        if is_local:
            cleanup_containers_local([container_name], dry_run=dry_run)
        else:
            ssh_kwargs = build_ssh_kwargs(config)
            cleanup_containers([host], [container_name], ssh_kwargs=ssh_kwargs, dry_run=dry_run)

        logger.info("Solo workload '%s' stopped on %s", cluster_id, host)
        return 0

    @staticmethod
    def _generate_node_script(
            image: str,
            container_name: str,
            serve_command: str,
            label: str = "node",
            env: dict[str, str] | None = None,
            volumes: dict[str, str] | None = None,
            nccl_env: dict[str, str] | None = None,
            extra_docker_opts: list[str] | None = None,
    ) -> str:
        """Generate a script that launches a container with a direct entrypoint command.

        Unlike the sleep-infinity + exec pattern used in solo mode, the
        serve command runs as the container's entrypoint.  Used for native
        and RPC cluster nodes where each container runs its own serve process.

        Args:
            image: Container image reference.
            container_name: Name for the container.
            serve_command: Command to run as the container entrypoint.
            label: Human-readable label for log messages (e.g. "sglang node").
            env: Additional environment variables.
            volumes: Volume mounts (host_path -> container_path).
            nccl_env: NCCL-specific environment variables.
            extra_docker_opts: Additional ``docker run`` options.

        Returns:
            Complete bash script as a string.
        """
        from sparkrun.orchestration.docker import docker_run_cmd, docker_stop_cmd
        from sparkrun.orchestration.primitives import merge_env

        all_env = merge_env(nccl_env, env)
        cleanup = docker_stop_cmd(container_name)
        run_cmd = docker_run_cmd(
            image=image,
            command=serve_command,
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
        )

        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            "\n"
            "echo 'Cleaning up existing container: %(name)s'\n"
            "%(cleanup)s\n"
            "\n"
            "echo 'Launching %(label)s: %(name)s'\n"
            "%(run_cmd)s\n"
            "\n"
            "# Verify container started\n"
            "sleep 1\n"
            "if docker ps --format '{{.Names}}' | grep -q '^%(name)s$'; then\n"
            "    echo 'Container %(name)s launched successfully'\n"
            "else\n"
            "    echo 'ERROR: Container %(name)s failed to start' >&2\n"
            "    docker logs %(name)s 2>&1 | tail -20 || true\n"
            "    exit 1\n"
            "fi\n"
        ) % {"name": container_name, "cleanup": cleanup, "run_cmd": run_cmd, "label": label}

    # --- Banner / connection info ---

    def _print_cluster_banner(self, title, hosts, image, cluster_id, ports, dry_run):
        """Print standardized cluster launch banner.

        Args:
            title: Banner title (e.g. "Ray Cluster Launcher").
            hosts: All hosts in the cluster.
            image: Container image reference.
            cluster_id: Cluster identifier.
            ports: Mapping of label to value for port lines.
            dry_run: Whether this is a dry-run invocation.
        """
        mode = "DRY-RUN" if dry_run else "LIVE"
        logger.info("=" * 60)
        logger.info("sparkrun %s", title)
        logger.info("=" * 60)
        logger.info("Cluster ID:     %s", cluster_id)
        logger.info("Image:          %s", image)
        logger.info("Head Node:      %s", hosts[0])
        logger.info(
            "Worker Nodes:   %s",
            ", ".join(hosts[1:]) if len(hosts) > 1 else "<none>",
        )
        for label, value in ports.items():
            logger.info("%-16s%s", label + ":", value)
        logger.info("Mode:           %s", mode)
        logger.info("=" * 60)

    def _stop_native_cluster(
            self,
            hosts: list[str],
            cluster_id: str,
            config=None,
            dry_run: bool = False,
    ) -> int:
        """Stop a native cluster by iterating ranked node containers.

        Shared implementation for runtimes using the native clustering
        strategy (SGLang, vllm-distributed) where each node has a
        ``{cluster_id}_node_{rank}`` container.

        Args:
            hosts: All hosts in the cluster.
            cluster_id: Cluster identifier.
            config: SparkrunConfig instance for SSH settings.
            dry_run: Show what would be done without executing.

        Returns:
            Exit code (0 = success).
        """
        from sparkrun.orchestration.primitives import build_ssh_kwargs
        from sparkrun.orchestration.ssh import run_remote_command
        from sparkrun.orchestration.docker import docker_stop_cmd, generate_node_container_name

        ssh_kwargs = build_ssh_kwargs(config)
        for rank, host in enumerate(hosts):
            container_name = generate_node_container_name(cluster_id, rank)
            run_remote_command(
                host, docker_stop_cmd(container_name),
                timeout=30, dry_run=dry_run, **ssh_kwargs,
            )

        logger.info("Cluster '%s' stopped on %d host(s)", cluster_id, len(hosts))
        return 0

    def _print_connection_info(self, hosts, cluster_id, *, per_node_logs=False):
        """Print standardized post-launch connection info.

        Args:
            hosts: All hosts in the cluster.
            cluster_id: Cluster identifier.
            per_node_logs: If True, print per-node ``docker logs`` commands
                using ranked container names (for native-cluster runtimes).
        """
        logger.info("=" * 60)
        logger.info("Cluster launched successfully. Nodes: %d", len(hosts))
        logger.info("")
        logger.info("To view logs:    sparkrun logs <recipe> --hosts %s", ",".join(hosts))
        logger.info("To stop cluster: sparkrun stop <recipe> --hosts %s", ",".join(hosts))
        if per_node_logs:
            from sparkrun.orchestration.docker import generate_node_container_name
            logger.info("")
            for rank, host in enumerate(hosts):
                logger.info(
                    "  Node %d: ssh %s 'docker logs %s'",
                    rank, host, generate_node_container_name(cluster_id, rank),
                )
        logger.info("=" * 60)

    def __repr__(self) -> str:
        return "%s(runtime_name=%r)" % (self.__class__.__name__, self.runtime_name)
