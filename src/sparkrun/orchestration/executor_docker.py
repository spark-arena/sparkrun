"""Docker implementation of the Executor abstraction.

``DockerExecutor`` generates Docker CLI command strings from
``ExecutorConfig`` settings.  It replaces the hardcoded
``_DEFAULT_DOCKER_OPTS`` list in ``docker.py``.
"""

from __future__ import annotations

import logging
import shlex

from sparkrun.orchestration.executor import Executor

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
            opts.append("--ipc=%s" % cfg.ipc)
        if cfg.shm_size:
            opts.append("--shm-size=%s" % cfg.shm_size)
        if cfg.network:
            opts.append("--network %s" % cfg.network)
        if cfg.user:
            if cfg.user == "$SHELL_USER":
                opts.extend(["--user", "$(id -u):$(id -g)"])
                opts.extend(["-v", "/etc/passwd:/etc/passwd:ro"])
                opts.extend(["-v", "/etc/group:/etc/group:ro"])
                opts.extend(["-e", "HOME=/tmp"])
            else:
                opts.extend(["--user", cfg.user])
        if cfg.security_opt:
            for opt in cfg.security_opt:
                opts.extend(["--security-opt", opt])
        if cfg.cap_add:
            for cap in cfg.cap_add:
                opts.extend(["--cap-add", cap])
        if cfg.ulimit:
            for ul in cfg.ulimit:
                opts.extend(["--ulimit", ul])
        if cfg.devices:
            for dev in cfg.devices:
                opts.extend(["--device", dev])

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
            parts.extend(["--name", container_name])

        if env:
            for key, value in sorted(env.items()):
                parts.extend(["-e", "%s=%s" % (key, value)])

        if volumes:
            for host_path, container_path in sorted(volumes.items()):
                parts.extend(["-v", "%s:%s" % (host_path, container_path)])

        if extra_opts:
            parts.extend(extra_opts)

        parts.append(shlex.quote(image))

        if command:
            parts.append(command)

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
        escaped_cmd = command.replace("'", "'\\''")
        parts.extend([shlex.quote(container_name), "bash", "-c", "'%s'" % escaped_cmd])
        return " ".join(parts)

    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Generate a docker stop/rm command string."""
        quoted = shlex.quote(container_name)
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
        return " ".join(parts)

    def inspect_exists_cmd(self, image: str) -> str:
        """Generate a command to check if a docker image exists locally."""
        return "docker image inspect %s >/dev/null 2>&1" % shlex.quote(image)

    def pull_cmd(self, image: str) -> str:
        """Generate a ``docker pull`` command."""
        return "docker pull %s" % shlex.quote(image)
