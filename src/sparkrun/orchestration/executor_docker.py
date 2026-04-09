"""Docker implementation of the Executor abstraction.

``DockerExecutor`` generates Docker CLI command strings from
``ExecutorConfig`` settings.  It replaces the hardcoded
``_DEFAULT_DOCKER_OPTS`` list in ``docker.py``.
"""

from __future__ import annotations

import logging
import shlex
from sparkrun.orchestration.executor import Executor
from sparkrun.utils.shell import b64_wrap_bash, quote

logger = logging.getLogger(__name__)


class DockerExecutor(Executor):
    """Docker-based executor for container operations."""

    def _build_default_opts(self) -> list[str]:
        """Build the default ``docker run`` option list from config."""
        cfg = self.config
        opts: list[str] = []

        if cfg.privileged:
            opts.append("--privileged")
        if cfg.gpus:
            opts.extend(["--gpus", cfg.gpus])
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
            opts.append("--memory=%s" % cfg.memory_limit)
        if cfg.labels:
            for lbl in cfg.labels:
                opts.extend(["--label", quote(lbl)])

        return opts

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
            for host_path, container_path in sorted(volumes.items()):
                parts.extend(["-v", quote("%s:%s" % (host_path, container_path))])

        if extra_opts:
            for opt in extra_opts:
                parts.extend(quote(token) for token in shlex.split(opt))

        parts.append(quote(image))

        if command:
            parts.extend(["bash", "-c", quote(b64_wrap_bash(command))])

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
                parts.extend(["-e", quote("%s=%s" % (key, value))])
        parts.extend([quote(container_name), "bash", "-c", quote(b64_wrap_bash(command))])
        return " ".join(parts)

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
        parts.append(quote(container_name))
        return " ".join(parts)

    def inspect_exists_cmd(self, image: str) -> str:
        """Generate a command to check if a docker image exists locally."""
        return "docker image inspect %s >/dev/null 2>&1" % quote(image)

    def pull_cmd(self, image: str) -> str:
        """Generate a ``docker pull`` command."""
        return "docker pull %s" % quote(image)
