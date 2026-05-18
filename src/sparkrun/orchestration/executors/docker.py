"""Docker implementation of the Executor abstraction.

``DockerExecutor`` generates Docker CLI command strings from
``ExecutorConfig`` settings.  The Docker-flavoured global defaults
(``privileged``, ``ipc=host``, ``shm_size``, ...) and the
``rootless``/``auto_user`` adjustment layer live here — they are not
shared concerns of other executors.
"""

from __future__ import annotations

import logging

from sparkrun.orchestration.executors._base import Executor
from sparkrun.utils.shell import args_list_to_shell_str, b64_wrap_bash, quote

logger = logging.getLogger(__name__)


# Per-executor defaults for the resolution chain — sits just above
# the :class:`ExecutorConfig` dataclass field defaults and below
# everything else.  Lives with :class:`DockerExecutor` because every
# value here is Docker-specific (``--privileged``, ``--shm-size``,
# ``--ipc=host`` etc.).
DOCKER_DEFAULTS = {
    "auto_remove": True,
    "restart_policy": None,
    "privileged": True,
    "gpus": "all",
    "ipc": "host",
    "shm_size": "32gb",
    "network": "host",
    "user": None,
    "security_opt": None,
    "cap_add": None,
    "ulimit": None,
    "devices": None,
}


class DockerExecutor(Executor):
    """Docker-based executor for container operations."""

    executor_name = "docker"

    # --- Resolution chain hooks ---

    @classmethod
    def default_config(cls) -> dict:
        """Docker-flavoured defaults — shm_size, ipc=host, network=host, ...."""
        return dict(DOCKER_DEFAULTS)

    @classmethod
    def apply_runtime_adjustments(cls, *, rootless: bool = True, auto_user: bool = True, **kwargs) -> dict:
        """Docker reads ``rootless`` and ``auto_user`` here.

        Sits above SparkrunConfig and below recipe overrides in the
        resolution chain, so users can still pin specific values in
        the recipe to override the rootless/auto_user defaults.
        """
        adjustments: dict = {}
        if rootless:
            adjustments["privileged"] = False
            adjustments["security_opt"] = ["no-new-privileges"]
            adjustments["cap_add"] = []
            adjustments["ulimit"] = [
                "memlock=-1:-1",
                "stack=67108864",
            ]
            # TODO: confirm existence and/or adjust? (for future heterogeneous support??)
            adjustments["devices"] = [
                "/dev/infiniband",
            ]
        if auto_user:
            adjustments["user"] = "$SHELL_USER"  # auto hint to use ssh user+group
        return adjustments

    # --- Internal command-string builders ---

    def _accelerator_opts(self) -> list[str]:
        """Emit accelerator device flags based on ``config.accelerator_vendor``.

        - ``None`` (default) or ``"nvidia"`` → ``--gpus <cfg.gpus>``.
        - ``"amd"`` → ROCm device + render-group flags.
        - ``"intel"`` → Intel Gaudi device flag.
        - ``"apple"`` / ``"cpu"`` → no device flag.  Apple MLX cannot
          run inside Docker; callers should route such hosts to a
          non-Docker executor.
        """
        cfg = self.config
        vendor = (cfg.accelerator_vendor or "").lower()

        if not vendor or vendor == "nvidia":
            if cfg.gpus:
                return ["--gpus", quote(cfg.gpus)]
            return []
        if vendor == "amd":
            return [
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "video",
            ]
        if vendor == "intel":
            return ["--device", "/dev/accel"]
        if vendor in ("apple", "cpu"):
            return []
        logger.warning(
            "DockerExecutor: unknown accelerator_vendor %r — emitting no device flag",
            cfg.accelerator_vendor,
        )
        return []

    def _build_default_opts(self) -> list[str]:
        """Build the default ``docker run`` option list from config."""
        cfg = self.config
        opts: list[str] = []

        if cfg.privileged:
            opts.append("--privileged")
        opts.extend(self._accelerator_opts())
        if cfg.ipc:
            opts.append("--ipc=%s" % quote(cfg.ipc))
        if cfg.shm_size:
            opts.append("--shm-size=%s" % quote(cfg.shm_size))
        if cfg.network:
            logger.debug("DockerExecutor using network: %s", cfg.network)
            opts.append("--network=%s" % quote(cfg.network))
        if cfg.user:
            if cfg.user == "$SHELL_USER":
                opts.extend(["--user", "$(id -u):$(id -g)"])
                opts.extend(["-v", "/etc/passwd:/etc/passwd:ro"])
                opts.extend(["-v", "/etc/group:/etc/group:ro"])
                opts.extend(["-e", "HOME=/tmp"])
            else:
                opts.extend(["--user", quote(cfg.user)])
        if cfg.security_opt:
            for opt in cfg.security_opt:
                opts.extend(["--security-opt", quote(opt)])
        if cfg.cap_add:
            for cap in cfg.cap_add:
                opts.extend(["--cap-add", quote(cap)])
        if cfg.ulimit:
            for ul in cfg.ulimit:
                opts.extend(["--ulimit", quote(ul)])
        if cfg.devices:
            for dev in cfg.devices:
                opts.extend(["--device", quote(dev)])
        if cfg.memory_limit:
            opts.append("--memory=%s" % quote(cfg.memory_limit))
        if cfg.labels:
            for lbl in cfg.labels:
                opts.extend(["--label", quote(lbl)])

        return opts

    # --- Low-level command generators (Executor ABC) ---

    def run_cmd(
        self,
        image: str,
        command: str = "",
        container_name: str | None = None,
        detach: bool = True,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        extra_opts: list[str] | None = None,
    ) -> str:
        """Generate a ``docker run`` command string."""
        cfg = self.config
        parts = ["docker", "run"]

        if detach:
            parts.append("-d")

        parts.extend(self._build_default_opts())

        if cfg.auto_remove:
            parts.append("--rm")

        if cfg.restart_policy:
            parts.extend(["--restart", cfg.restart_policy])

        if container_name:
            parts.extend(["--name", quote(container_name)])

        if env:
            for key, value in sorted(env.items()):
                parts.extend(["-e", quote("%s=%s" % (key, value))])

        if volumes:
            # TODO: option for ro/rw on volumes?
            for host_path, container_path in sorted(volumes.items()):
                parts.extend(["-v", quote("%s:%s" % (host_path, container_path))])

        if extra_opts:
            from shlex import split as shlex_split

            for opt in extra_opts:
                parts.extend(quote(token) for token in shlex_split(opt))

        parts.append(quote(image))

        if command:
            parts.extend(["bash", "-c", b64_wrap_bash(command)])

        result = " ".join(parts)

        if env:
            logger.debug("docker run %s env (%d vars):", container_name or image, len(env))
            for key, value in sorted(env.items()):
                logger.debug("  %s=%s", key, value)
        logger.debug("docker run command: %s", result)

        return result

    def exec_cmd(
        self,
        container_name: str,
        command: str,
        detach: bool = False,
        env: dict[str, str] | None = None,
    ) -> str:
        """Generate a ``docker exec`` command string."""
        parts = ["docker", "exec"]
        if detach:
            parts.append("-d")
        if env:
            for key, value in sorted(env.items()):
                parts.extend(["-e", "%s=%s" % (key, value)])
        parts.extend([container_name, "bash", "-c", b64_wrap_bash(command)])
        return args_list_to_shell_str(parts)

    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Generate a docker stop/rm command string."""
        quoted = quote(container_name)
        if force:
            return "docker rm -f %s 2>/dev/null || true" % quoted
        return "docker stop %s 2>/dev/null || true" % quoted

    def logs_cmd(
        self,
        container_name: str,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Generate a ``docker logs`` command string."""
        parts = ["docker", "logs"]
        if follow:
            parts.append("-f")
        if tail is not None:
            parts.extend(["--tail", str(tail)])
        parts.append(container_name)
        return args_list_to_shell_str(parts)

    def status_cmd(self, container_name: str) -> str:
        """Exit 0 iff a container named *container_name* is currently running."""
        # Anchored filter so name=foo doesn't match foo_solo etc.
        filter_arg = quote("name=^%s$" % container_name)
        return "[ -n \"$(docker ps --filter %s --format '{{.ID}}')\" ]" % filter_arg

    def inspect_exists_cmd(self, image: str) -> str:
        """Generate a command to check if a docker image exists locally."""
        return "docker image inspect %s >/dev/null 2>&1" % quote(image)

    def pull_cmd(self, image: str) -> str:
        """Generate a ``docker pull`` command."""
        return "docker pull %s" % quote(image)
