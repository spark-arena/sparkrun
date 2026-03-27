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
    from sparkrun.orchestration.executor import Executor

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

    # --- Executor ---
    _executor: Executor | None = None

    @property
    def executor(self) -> Executor:
        """Return the executor, lazily defaulting to DockerExecutor."""
        if self._executor is None:
            from sparkrun.orchestration.executor_docker import DockerExecutor

            self._executor = DockerExecutor()
        return self._executor

    @executor.setter
    def executor(self, value: Executor) -> None:
        self._executor = value

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
    def generate_command(
            self,
            recipe: Recipe,
            overrides: dict[str, Any],
            is_cluster: bool,
            num_nodes: int = 1,
            head_ip: str | None = None,
            skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
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

    # noinspection PyMethodMayBeStatic
    def get_common_env(self):
        """Return environment variables common to either solo or cluster mode for this runtime."""
        return {}

    # noinspection PyMethodMayBeStatic
    def get_solo_env(self):
        """Return runtime-specific environment variables for solo mode."""
        return {}

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return runtime-specific environment variables for cluster mode.

        Override in subclasses to inject runtime-specific cluster config.
        """
        return {}

    def cluster_strategy(self) -> str:
        """Return the clustering strategy for multi-node mode.

        Returns:
            ``"ray"`` — use Ray cluster orchestration (start Ray head/workers,
            then exec serve command on head). This was the original default.

            ``"native"`` — the runtime handles its own distribution. Each node
            runs the serve command directly with node-rank arguments appended.
            Used by sglang, which has built-in multi-node support via
            ``--dist-init-addr``, ``--nnodes``, ``--node-rank``.
        """
        return "ray"

    def get_family(self) -> str:
        """Return the canonical runtime family name.

        Defaults to runtime_name. Override in subclasses to map
        variants to their canonical family (e.g. vllm-ray -> vllm).
        """
        return self.runtime_name

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
        raise NotImplementedError("%s does not implement native clustering" % type(self).__name__)

    def prepare(
            self,
            recipe: Recipe,
            hosts: list[str],
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            transfer_mode: str = "local",
    ) -> None:
        """Pre-launch preparation (e.g., building container images).

        Called by the CLI before resource distribution.  Override in
        subclasses that need to build or transform images before they
        can be distributed to hosts.

        Args:
            recipe: The loaded recipe.
            hosts: Target host list.
            config: SparkrunConfig instance.
            dry_run: Show what would be done without executing.
            transfer_mode: ``"local"`` or ``"delegated"``.
        """
        pass

    def _pre_serve(
            self,
            hosts_containers: list[tuple[str, str]],
            ssh_kwargs: dict,
            dry_run: bool,
            recipe: Recipe | None = None,
            config_chain=None,
    ) -> None:
        """Hook called after containers are launched but before serve command.

        Processes ``pre_exec`` commands from the recipe (if any) by running
        them inside each container via ``docker exec``.  Subclasses can
        override to add additional pre-serve logic (call ``super()`` to
        preserve pre_exec processing).

        Args:
            hosts_containers: List of (host, container_name) pairs.
            ssh_kwargs: SSH connection kwargs.
            dry_run: Dry-run mode.
            recipe: The loaded recipe (for pre_exec commands).
            config_chain: Config chain for template substitution.
        """
        if recipe and recipe.pre_exec:
            from sparkrun.orchestration.hooks import run_pre_exec

            run_pre_exec(hosts_containers, recipe.pre_exec, config_chain, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

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

        The base implementation sets ``HF_HOME`` so HuggingFace
        libraries find the cache at the rootless-compatible mount
        point (``/cache/huggingface``).

        Returns:
            Dict of env var name -> value.
        """
        return {"HF_HOME": "/cache/huggingface"}

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
    def _augment_served_model_name(
            command: str,
            config,
            flag: str,
            skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Append ``served_model_name`` to a rendered command if missing.

        When a recipe uses an explicit command template that omits the
        ``{served_model_name}`` placeholder, CLI overrides for
        ``--served-model-name`` are silently dropped.  This helper
        checks whether the override was consumed and appends the
        appropriate flag if not.

        Args:
            command: The rendered command string.
            config: Config chain (must support ``.get(key)``).
            flag: The CLI flag to use (e.g. ``"--served-model-name"``
                or ``"--alias"`` for llama.cpp).
            skip_keys: Keys being suppressed (e.g. by benchmark flow).

        Returns:
            The command string, possibly with the flag appended.
        """
        if "served_model_name" in skip_keys:
            return command
        value = config.get("served_model_name")
        if value is None:
            return command
        if flag in command:
            return command
        return "%s %s %s" % (command.rstrip(), flag, value)

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
                    command = re.sub(r"\s*" + escaped + r"(?=\s|$)", "", command)
                else:
                    # Match the flag, its value, and an optional trailing
                    # backslash continuation on the same line.
                    command = re.sub(
                        escaped + r"\s+\S+\s*\\?\s*\n?",
                        "",
                        command,
                    )

        # Clean up artifacts from removed lines:
        # - collapse double backslash-continuations (``\ \``) into one
        # - remove blank continuation lines (``\`` followed by only whitespace)
        command = re.sub(r"\\\s*\\\s*\n", "\\\n", command)
        command = re.sub(r"\\\s*\n(\s*\\\s*\n)", r"\\\n", command)
        # Remove lines that are only whitespace (left behind after removal)
        command = re.sub(r"\n\s*\n", "\n", command)
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
            from sparkrun.orchestration.ssh import stream_container_file_logs

            host = hosts[0] if hosts else "localhost"
            container_name = self.executor.container_name(cluster_id, "solo")
            ssh_kwargs = build_ssh_kwargs(config)
            stream_container_file_logs(
                host,
                container_name,
                tail=tail,
                dry_run=dry_run,
                **ssh_kwargs,
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
                hosts[0],
                container_name,
                tail=tail,
                dry_run=dry_run,
                **ssh_kwargs,
            )
        else:
            from sparkrun.orchestration.ssh import stream_remote_logs

            stream_remote_logs(
                hosts[0],
                container_name,
                tail=tail,
                dry_run=dry_run,
                **ssh_kwargs,
            )

    def get_head_container_name(self, cluster_id: str, is_solo: bool = False) -> str:
        """Return the expected head/solo container name for *cluster_id*.

        Solo mode always uses ``{cluster_id}_solo``.  Cluster mode
        delegates to :meth:`_head_container_name` which subclasses
        override when they use non-standard naming (e.g.
        ``{cluster_id}_node_0`` for SGLang and vLLM distributed).
        """
        if is_solo:
            return self.executor.container_name(cluster_id, "solo")
        return self._head_container_name(cluster_id)

    def _head_container_name(self, cluster_id: str) -> str:
        """Return the head container name for log following.

        Native-cluster runtimes (``cluster_strategy() == "native"``)
        default to ``{cluster_id}_node_0``.  Ray-based runtimes default
        to ``{cluster_id}_head``.  Subclasses can still override.
        """
        if self.cluster_strategy() == "native":
            return self.executor.node_container_name(cluster_id, 0)
        return self.executor.container_name(cluster_id, "head")

    def _cluster_log_mode(self) -> str:
        """Return the log tailing mode for cluster containers.

        ``"file"`` uses :func:`stream_container_file_logs` (tails a log
        file inside the container).  ``"docker"`` uses
        :func:`stream_remote_logs` (``docker logs``).

        Default is ``"file"`` for native-cluster runtimes (which use
        the sleep-infinity + exec pattern) and ``"docker"`` for others.
        Override in subclasses to change.
        """
        if self.cluster_strategy() == "native":
            return "file"
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
            executor: Executor | None = None,
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
            executor: Container executor (defaults to DockerExecutor).
            **kwargs: Runtime-specific keyword arguments (e.g. ray_port,
                dashboard_port, init_port, rpc_port).

        Returns:
            Exit code (0 = success).
        """
        if executor is not None:
            self._executor = executor

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
                recipe=recipe,
                overrides=overrides,
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
        raise NotImplementedError("Cluster mode not supported by %s" % self.runtime_name)

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
        raise NotImplementedError("Cluster stop not supported by %s" % self.runtime_name)

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
            recipe: Recipe | None = None,
            overrides: dict[str, Any] | None = None,
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
            detect_infiniband,
            detect_infiniband_local,
            run_script_on_host,
        )
        from sparkrun.utils import is_local_host, merge_env

        is_local = is_local_host(host)
        container_name = self.executor.container_name(cluster_id, "solo")
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir, extra=self.get_extra_volumes())
        all_env = merge_env(
            self.get_common_env(),  # base env
            self.get_solo_env(),  # solo-specific
            env,  # recipe
            self.get_extra_env()  # tuning/other overrides
        )

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
                    [host],
                    ssh_kwargs=ssh_kwargs,
                    dry_run=dry_run,
                )
            logger.info("Step 1/3: IB detection done (%.1fs)", time.monotonic() - t0)

        # Step 2: Launch container
        t0 = time.monotonic()
        logger.info(
            "Step 2/3: Launching container %s on %s (image: %s)...",
            container_name,
            host,
            image,
        )
        launch_script = self.executor.generate_launch_script(
            image=image,
            container_name=container_name,
            command="sleep infinity",
            env=all_env,
            volumes=volumes,
            nccl_env=nccl_env,
            extra_docker_opts=self.get_extra_docker_opts() or None,
        )
        result = run_script_on_host(
            host,
            launch_script,
            ssh_kwargs=ssh_kwargs,
            timeout=120,
            dry_run=dry_run,
        )
        if not result.success and not dry_run:
            logger.error("Failed to launch container: %s", result.stderr)
            return 1
        logger.info("Step 2/3: Container launched (%.1fs)", time.monotonic() - t0)

        # Pre-serve hook (e.g., apply mods to container, run pre_exec)
        config_chain = recipe.build_config_chain(overrides) if recipe else None
        self._pre_serve([(host, container_name)], ssh_kwargs, dry_run, recipe=recipe, config_chain=config_chain)

        # Step 3: Execute serve command
        t0 = time.monotonic()
        logger.info("Step 3/3: Executing serve command in %s...", container_name)
        logger.debug("Serve command: %s", serve_command)
        exec_script = self.executor.generate_exec_serve_script(
            container_name=container_name,
            serve_command=serve_command,
            env=all_env,
            detached=detached,
        )
        result = run_script_on_host(
            host,
            exec_script,
            ssh_kwargs=ssh_kwargs,
            timeout=60,
            dry_run=dry_run,
        )
        logger.info("Step 3/3: Serve command dispatched (%.1fs)", time.monotonic() - t0)

        if dry_run:
            return 0

        if result.returncode != 0:
            # Serve process failed to start — print captured output
            logger.error("Serve process failed to start (rc=%d)", result.returncode)
            if result.stderr:
                for line in result.stderr.rstrip().splitlines():
                    logger.error("  %s", line)
            elif result.stdout:
                for line in result.stdout.rstrip().splitlines():
                    logger.error("  %s", line)

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
        from sparkrun.utils import is_local_host

        container_name = self.executor.container_name(cluster_id, "solo")
        is_local = is_local_host(host)

        if is_local:
            cleanup_containers_local([container_name], dry_run=dry_run)
        else:
            ssh_kwargs = build_ssh_kwargs(config)
            cleanup_containers([host], [container_name], ssh_kwargs=ssh_kwargs, dry_run=dry_run)

        logger.info("Solo workload '%s' stopped on %s", cluster_id, host)
        return 0

    def _generate_node_script(
            self,
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
        return self.executor.generate_node_script(
            image=image,
            container_name=container_name,
            serve_command=serve_command,
            label=label,
            env=env,
            volumes=volumes,
            nccl_env=nccl_env,
            extra_docker_opts=extra_docker_opts,
        )

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

        ssh_kwargs = build_ssh_kwargs(config)
        for rank, host in enumerate(hosts):
            container_name = self.executor.node_container_name(cluster_id, rank)
            run_remote_command(
                host,
                self.executor.stop_cmd(container_name),
                timeout=30,
                dry_run=dry_run,
                **ssh_kwargs,
            )

        logger.info("Cluster '%s' stopped on %d host(s)", cluster_id, len(hosts))
        return 0

    def _run_native_cluster(
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
            banner_title: str = "Native Cluster Launcher",
            port_label: str = "Init Port",
            node_label: str = "node",
            **kwargs,
    ) -> int:
        """Orchestrate a multi-node native cluster (shared by SGLang, vLLM distributed).

        Uses the two-phase launch pattern (sleep infinity + exec) so that
        ``_pre_serve`` hooks (e.g. ``pre_exec`` from recipes) run between
        container startup and serve execution — identical to solo mode.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Detect head node IP.
        4. Launch ALL containers with ``sleep infinity``.
        5. Run pre-serve hooks (pre_exec) on all containers.
        6. Exec head serve command, wait for init port.
        7. Exec worker serve commands in parallel.

        Args:
            hosts: All hosts in the cluster (first = head).
            image: Container image reference.
            serve_command: Unused (commands are generated per-node).
            recipe: The loaded recipe.
            overrides: CLI override values.
            cluster_id: Cluster identifier for container naming.
            env: Additional environment variables from the recipe.
            cache_dir: HuggingFace cache directory path.
            config: SparkrunConfig instance for SSH settings.
            dry_run: Show what would be done without executing.
            detached: Run serve command in background.
            nccl_env: Pre-detected NCCL environment variables.
            init_port: Coordination port for distributed init.
            skip_keys: Config keys to omit from generated commands.
            banner_title: Title for the launch banner.
            port_label: Label for the port in the banner (e.g. "Init Port").
            node_label: Label for nodes in log messages (e.g. "sglang node").
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            build_volumes,
            detect_host_ip,
            wait_for_port,
            resolve_nccl_env,
        )
        from sparkrun.utils import merge_env
        from sparkrun.orchestration.ssh import (
            run_remote_script,
            run_remote_command,
            start_log_capture,
            stop_log_capture,
        )

        num_nodes = len(hosts)
        head_host = hosts[0]
        worker_hosts = hosts[1:]
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir, extra=self.get_extra_volumes())
        runtime_env = self.get_cluster_env(head_ip="<pending>", num_nodes=num_nodes)
        # Runtime defaults first, recipe env overrides (power users can tweak)
        all_env = merge_env(
            self.get_common_env(),  # common env
            runtime_env,  # cluster-specific env
            env,  # recipe env
            self.get_extra_env()  # tuning/other overrides
        )

        self._print_cluster_banner(
            banner_title,
            hosts,
            image,
            cluster_id,
            {port_label: init_port},
            dry_run,
        )

        # Step 1: Cleanup
        t0 = time.monotonic()
        logger.info("Step 1/7: Cleaning up existing containers for cluster '%s'...", cluster_id)
        for rank, host in enumerate(hosts):
            container_name = self.executor.node_container_name(cluster_id, rank)
            run_remote_command(
                host,
                self.executor.stop_cmd(container_name),
                timeout=30,
                dry_run=dry_run,
                **ssh_kwargs,
            )
        logger.info("Step 1/7: Cleanup done (%.1fs)", time.monotonic() - t0)

        # Step 2: InfiniBand detection (skip if pre-detected nccl_env provided)
        t0 = time.monotonic()
        logger.info("Step 2/7: InfiniBand detection...")
        nccl_env = resolve_nccl_env(
            nccl_env,
            hosts,
            head_host=head_host,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
        )
        logger.info("Step 2/7: IB step done (%.1fs)", time.monotonic() - t0)

        # Step 3: Detect head node IP
        t0 = time.monotonic()
        logger.info("Step 3/7: Detecting head node IP on %s...", head_host)
        try:
            head_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
        except RuntimeError as e:
            logger.error("%s", e)
            return 1
        logger.info("  Head IP: %s", head_ip)
        logger.info("Step 3/7: IP detection done (%.1fs)", time.monotonic() - t0)

        # Auto-detect available init port to avoid collisions with running instances
        from sparkrun.orchestration.primitives import find_available_port

        init_port = find_available_port(head_host, init_port, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

        # Generate per-node commands (needed later for exec steps)
        head_command = self.generate_node_command(
            recipe=recipe,
            overrides=overrides,
            head_ip=head_ip,
            num_nodes=num_nodes,
            node_rank=0,
            init_port=init_port,
            skip_keys=skip_keys,
        )
        logger.info("Serve command (head, rank 0):")
        for line in head_command.strip().splitlines():
            logger.info("  %s", line)

        # Step 4: Launch ALL containers with sleep infinity
        t0 = time.monotonic()
        logger.info("Step 4/7: Launching containers with sleep infinity on all %d host(s)...", num_nodes)

        # Build (host, rank, container_name) list for all nodes
        all_nodes: list[tuple[str, int, str]] = []
        for rank, host in enumerate(hosts):
            all_nodes.append((host, rank, self.executor.node_container_name(cluster_id, rank)))

        # Launch all containers in parallel
        with ThreadPoolExecutor(max_workers=num_nodes) as pool:
            launch_futures = {}
            for host, rank, cname in all_nodes:
                launch_script = self.executor.generate_launch_script(
                    image=image,
                    container_name=cname,
                    command="sleep infinity",
                    env=all_env,
                    volumes=volumes,
                    nccl_env=nccl_env,
                    extra_docker_opts=self.get_extra_docker_opts() or None,
                )
                future = pool.submit(
                    run_remote_script,
                    host,
                    launch_script,
                    timeout=120,
                    dry_run=dry_run,
                    **ssh_kwargs,
                )
                launch_futures[future] = (host, rank, cname)

            for future in as_completed(launch_futures):
                host, rank, cname = launch_futures[future]
                result = future.result()
                if not result.success and not dry_run:
                    logger.error("Failed to launch container %s (rank %d) on %s: %s", cname, rank, host, result.stderr[:200])
                    return 1

        logger.info("Step 4/7: All containers launched (%.1fs)", time.monotonic() - t0)

        # Step 5: Pre-serve hooks (pre_exec)
        t0 = time.monotonic()
        logger.info("Step 5/7: Running pre-serve hooks...")
        hosts_containers = [(host, cname) for host, _rank, cname in all_nodes]
        config_chain = recipe.build_config_chain(overrides) if recipe else None
        self._pre_serve(hosts_containers, ssh_kwargs, dry_run, recipe=recipe, config_chain=config_chain)
        logger.info("Step 5/7: Pre-serve hooks done (%.1fs)", time.monotonic() - t0)

        # Step 6: Exec head serve command and wait for init port
        t0 = time.monotonic()
        head_container = all_nodes[0][2]
        logger.info("Step 6/7: Executing serve command on head node (rank 0) %s...", head_host)
        head_exec_script = self.executor.generate_exec_serve_script(
            container_name=head_container,
            serve_command=head_command,
            env=all_env,
            detached=True,
        )
        head_result = run_remote_script(
            head_host,
            head_exec_script,
            timeout=60,
            dry_run=dry_run,
            **ssh_kwargs,
        )
        if not head_result.success and not dry_run:
            logger.error("Failed to exec serve on head node: %s", head_result.stderr[:200])
            return 1

        # Wait for head init port
        if not dry_run:
            logger.info("  Waiting for head node %s %s:%d...", port_label.lower(), head_host, init_port)

            log_proc = start_log_capture(head_host, head_container, ssh_kwargs)
            try:
                ready = wait_for_port(
                    head_host,
                    init_port,
                    max_retries=60,
                    retry_interval=2,
                    ssh_kwargs=ssh_kwargs,
                    dry_run=dry_run,
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
                        head_host,
                        head_container,
                    )
                return 1
            logger.info("Step 6/7: Head node ready (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 6/7: [dry-run] Would wait for %s %d", port_label.lower(), init_port)

        # Step 7: Exec worker serve commands in parallel
        t0 = time.monotonic()
        if worker_hosts:
            logger.info(
                "Step 7/7: Executing serve on %d worker node(s) on %s...",
                len(worker_hosts),
                ", ".join(worker_hosts),
            )
            with ThreadPoolExecutor(max_workers=len(worker_hosts)) as pool:
                futures = {}
                for i, host in enumerate(worker_hosts):
                    rank = i + 1
                    worker_command = self.generate_node_command(
                        recipe=recipe,
                        overrides=overrides,
                        head_ip=head_ip,
                        num_nodes=num_nodes,
                        node_rank=rank,
                        init_port=init_port,
                        skip_keys=skip_keys,
                    )
                    worker_container = all_nodes[rank][2]
                    worker_exec_script = self.executor.generate_exec_serve_script(
                        container_name=worker_container,
                        serve_command=worker_command,
                        env=all_env,
                        detached=True,
                    )
                    future = pool.submit(
                        run_remote_script,
                        host,
                        worker_exec_script,
                        timeout=60,
                        dry_run=dry_run,
                        **ssh_kwargs,
                    )
                    futures[future] = (host, rank)

                for future in as_completed(futures):
                    host, rank = futures[future]
                    result = future.result()
                    if not result.success and not dry_run:
                        logger.warning(
                            "  Worker rank %d on %s may have failed: %s",
                            rank,
                            host,
                            result.stderr[:100],
                        )

            logger.info("Step 7/7: Workers launched (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 7/7: No worker hosts, skipping")

        self._print_connection_info(hosts, cluster_id, per_node_logs=True)
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
            logger.info("")
            for rank, host in enumerate(hosts):
                logger.info(
                    "  Node %d: ssh %s 'docker logs %s'",
                    rank,
                    host,
                    self.executor.node_container_name(cluster_id, rank),
                )
        logger.info("=" * 60)

    # --- Runtime version detection ---

    def version_commands(self) -> dict[str, str]:
        """Return label→shell command pairs for version detection.

        Base implementation provides common GPU stack versions.
        Subclasses should call super() and add runtime-specific entries.
        """
        return {
            "cuda": "nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//' || nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown",
            "python": "python3 --version 2>/dev/null | awk '{print $2}' || echo unknown",
            "torch": "python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown",
            "nccl": "python3 -c 'import torch; print(torch.cuda.nccl.version())' 2>/dev/null || echo unknown",
        }

    def _collect_runtime_info(
            self,
            host: str,
            container_name: str,
            ssh_kwargs: dict,
            dry_run: bool = False,
            builder=None,
    ) -> dict[str, str]:
        """Run version commands inside a container, return {label: version}.

        Builds a single bash script from :meth:`version_commands`, executes
        it inside the container via ``docker exec``, and parses the output.
        When a *builder* is provided, its :meth:`version_info_commands` are
        appended to the same script using delimited blocks, and the raw
        output is post-processed via :meth:`builder.process_version_info`.
        All exceptions are caught — version capture never blocks a launch.
        """
        if dry_run:
            return {}
        cmds = self.version_commands()
        builder_cmds = builder.version_info_commands() if builder else {}
        if not cmds and not builder_cmds:
            return {}

        # Build an inner script that runs inside the container.
        # Each command outputs a SPARKRUN_VER_KEY=<value> line.
        inner_lines = ["#!/bin/bash"]
        for key, cmd in sorted(cmds.items()):
            inner_lines.append('echo "SPARKRUN_VER_%s=$(%s)"' % (key.upper(), cmd))

        # Builder commands use delimited blocks for multi-line output.
        for label, cmd in sorted(builder_cmds.items()):
            inner_lines.append('echo "SPARKRUN_BUILDER_START_%s"' % label)
            inner_lines.append(cmd)
            inner_lines.append('echo "SPARKRUN_BUILDER_END_%s"' % label)

        inner_script = "\n".join(inner_lines)

        # Pipe the inner script into `docker exec <container> bash -s`
        # via a here-document.  This avoids quoting issues that arise
        # when embedding complex shell commands in bash -c '...'.
        # TODO: using 'docker exec' implies executor should be involved
        outer_script = ("docker exec -i %s bash -s <<'SPARKRUN_VER_EOF'\n%s\nSPARKRUN_VER_EOF") % (container_name, inner_script)

        try:
            from sparkrun.orchestration.primitives import run_script_on_host

            result = run_script_on_host(host, outer_script, ssh_kwargs=ssh_kwargs, timeout=30, dry_run=False)
            if result.returncode != 0:
                logger.debug("Version collection failed (rc=%d): %s", result.returncode, result.stderr)
                return {}
            info = {}
            # Parse runtime SPARKRUN_VER_ lines
            for line in result.stdout.splitlines():
                if line.startswith("SPARKRUN_VER_"):
                    key_val = line.removeprefix("SPARKRUN_VER_")
                    if "=" in key_val:
                        k, v = key_val.split("=", 1)
                        v = v.strip()
                        if v and v != "unknown":
                            info[k.lower()] = v

            # Extract builder delimited blocks and post-process
            if builder and builder_cmds:
                raw_builder: dict[str, str] = {}
                stdout = result.stdout
                for label in builder_cmds:
                    start_marker = "SPARKRUN_BUILDER_START_%s" % label
                    end_marker = "SPARKRUN_BUILDER_END_%s" % label
                    start_idx = stdout.find(start_marker)
                    end_idx = stdout.find(end_marker)
                    if start_idx >= 0 and end_idx > start_idx:
                        block = stdout[start_idx + len(start_marker): end_idx]
                        # Strip the leading newline from the marker line
                        if block.startswith("\n"):
                            block = block[1:]
                        # Strip trailing newline before end marker
                        if block.endswith("\n"):
                            block = block[:-1]
                        raw_builder[label] = block
                try:
                    builder_info = builder.process_version_info(raw_builder)
                    # Merge builder results (don't overwrite runtime keys)
                    for k, v in builder_info.items():
                        if k not in info:
                            info[k] = v
                except Exception:
                    logger.debug("Builder version info processing failed", exc_info=True)

            logger.debug("Collected runtime info: %s", info)
            return info
        except Exception:
            logger.debug("Version collection error", exc_info=True)
            return {}

    def __repr__(self) -> str:
        return "%s(runtime_name=%r)" % (self.__class__.__name__, self.runtime_name)
