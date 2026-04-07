"""Executor abstraction for container engine operations.

Provides ``ExecutorConfig`` (typed config from config chain resolution)
and ``Executor`` (abstract base for container engines like Docker/Podman).

Runtimes call ``self.executor.*`` instead of importing ``docker.py``
directly, making them container-engine-agnostic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from scitrera_app_framework.util import ext_parse_bool

logger = logging.getLogger(__name__)

# Default executor settings for DGX Spark GPU workloads.
# Lowest priority in the config chain: CLI → recipe → these defaults.
EXECUTOR_DEFAULTS = {
    "auto_remove": True,
    "restart_policy": None,
    "privileged": True,
    "gpus": "all",
    "ipc": "host",
    "shm_size": "10.24gb",
    "network": "host",
    "user": None,
    "security_opt": None,
    "cap_add": None,
    "ulimit": None,
    "devices": None,
}


@dataclass
class ExecutorConfig:
    """Typed view of resolved executor settings.

    Constructed from a config chain (or plain dict) after CLI → recipe →
    defaults layering.
    """

    auto_remove: bool = True
    restart_policy: str | None = None
    privileged: bool = True
    gpus: str = "all"
    ipc: str = "host"
    shm_size: str = "10.24gb"
    network: str = "host"
    user: str | None = None
    security_opt: list[str] | None = None
    cap_add: list[str] | None = None
    ulimit: list[str] | None = None
    devices: list[str] | None = None
    memory_limit: str | None = None
    labels: list[str] | None = None

    @classmethod
    def from_chain(cls, chain) -> ExecutorConfig:
        """Build from a config chain or plain dict."""
        raw_sec = chain.get("security_opt")
        if isinstance(raw_sec, str):
            raw_sec = [raw_sec]
        raw_cap = chain.get("cap_add")
        if isinstance(raw_cap, str):
            raw_cap = [raw_cap]
        raw_ulimit = chain.get("ulimit")
        if isinstance(raw_ulimit, str):
            raw_ulimit = [raw_ulimit]
        raw_devices = chain.get("devices")
        if isinstance(raw_devices, str):
            raw_devices = [raw_devices]
        raw_labels = chain.get("labels")
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]

        # Fallback to EXECUTOR_DEFAULTS for None values. With Variables,
        # falsy values like False/0 are preserved correctly, but None
        # still means "not set" and should fall back.
        def _get(key):
            v = chain.get(key)
            return v if v is not None else EXECUTOR_DEFAULTS.get(key)

        return cls(
            auto_remove=ext_parse_bool(_get("auto_remove")),
            restart_policy=chain.get("restart_policy") or None,
            privileged=ext_parse_bool(_get("privileged")),
            gpus=str(_get("gpus")),
            ipc=str(_get("ipc")),
            shm_size=str(_get("shm_size")),
            network=str(_get("network")),
            user=chain.get("user") or None,
            security_opt=raw_sec or None,
            cap_add=raw_cap or None,
            ulimit=raw_ulimit or None,
            devices=raw_devices or None,
            memory_limit=chain.get("memory_limit") or None,
            labels=raw_labels or None,
        )

    def __post_init__(self):
        # Docker does not allow --rm with --restart
        if self.restart_policy:
            self.auto_remove = False


class Executor(ABC):
    """Abstract base for container engine executors.

    Low-level methods (abstract) generate engine-specific command strings.
    High-level methods (concrete) compose scripts from low-level methods
    and bash templates.
    """

    def __init__(self, config: ExecutorConfig | None = None):
        self.config = config or ExecutorConfig()

    # --- Low-level command generators (abstract) ---

    @abstractmethod
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
        """Generate a container run command string."""
        ...

    @abstractmethod
    def exec_cmd(
        self,
        container_name: str,
        command: str,
        detach: bool = False,
        env: dict[str, str] | None = None,
    ) -> str:
        """Generate a container exec command string."""
        ...

    @abstractmethod
    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Generate a container stop/remove command string."""
        ...

    @abstractmethod
    def logs_cmd(
        self,
        container_name: str,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Generate a container logs command string."""
        ...

    @abstractmethod
    def inspect_exists_cmd(self, image: str) -> str:
        """Generate a command to check if an image exists locally."""
        ...

    @abstractmethod
    def pull_cmd(self, image: str) -> str:
        """Generate an image pull command string."""
        ...

    # --- Naming helpers (concrete, shared across executors) ---

    @staticmethod
    def container_name(cluster_id: str, role: str = "head") -> str:
        """Generate a deterministic container name: ``{cluster_id}_{role}``."""
        return "%s_%s" % (cluster_id, role)

    @staticmethod
    def node_container_name(cluster_id: str, rank: int) -> str:
        """Generate a ranked node container name: ``{cluster_id}_node_{rank}``."""
        return "%s_node_%d" % (cluster_id, rank)

    @staticmethod
    def enumerate_containers(cluster_id: str, num_hosts: int) -> list[str]:
        """Return all possible container names for a cluster.

        Covers solo, Ray (head/worker), and native (node_N) patterns.
        """
        names = [
            "%s_solo" % cluster_id,
            "%s_head" % cluster_id,
            "%s_worker" % cluster_id,
        ]
        for rank in range(num_hosts):
            names.append("%s_node_%d" % (cluster_id, rank))
        return names

    # --- High-level script generators (concrete) ---

    def generate_launch_script(
        self,
        image: str,
        container_name: str,
        command: str,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
        detach: bool = True,
        extra_docker_opts: list[str] | None = None,
    ) -> str:
        """Generate a script that cleans up then launches a container.

        Absorbs ``scripts.py::generate_container_launch_script``.
        """
        from sparkrun.utils import merge_env
        from sparkrun.scripts import read_script

        all_env = merge_env(nccl_env, env)
        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=command,
            container_name=container_name,
            detach=detach,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
        )

        template = read_script("container_launch.sh")
        return template.format(
            container_name=container_name,
            image=image,
            cleanup_cmd=cleanup,
            run_cmd=run,
        )

    def generate_exec_serve_script(
        self,
        container_name: str,
        serve_command: str,
        env: dict[str, str] | None = None,
        detached: bool = True,
    ) -> str:
        """Generate a script that executes the serve command inside a running container.

        Absorbs ``scripts.py::generate_exec_serve_script``.
        """
        from sparkrun.scripts import read_script

        env_exports = ""
        if env:
            for key, value in sorted(env.items()):
                env_exports += "export %s='%s'; " % (key, value)

        escaped_cmd = serve_command.replace("'", "'\\''")
        full_cmd = "%s%s" % (env_exports, escaped_cmd)

        script_name = "exec_serve_detached.sh" if detached else "exec_serve_foreground.sh"
        template = read_script(script_name)
        return template.format(
            container_name=container_name,
            full_cmd=full_cmd,
        )

    def generate_ray_head_script(
        self,
        image: str,
        container_name: str,
        ray_port: int = 46379,
        dashboard_port: int = 8265,
        dashboard: bool = False,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
        extra_docker_opts: list[str] | None = None,
    ) -> str:
        """Generate a script that starts a Ray head node in a container.

        Absorbs ``scripts.py::generate_ray_head_script``.
        """
        from sparkrun.utils import merge_env
        from sparkrun.scripts import read_script

        all_env = merge_env({"RAY_memory_monitor_refresh_ms": "0"}, nccl_env, env)

        dashboard_flags = ""
        if dashboard:
            dashboard_flags = "--include-dashboard=True --dashboard-host 0.0.0.0 --dashboard-port %d " % dashboard_port

        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=("ray start --block --head --port %d --node-ip-address $NODE_IP %s--disable-usage-stats" % (ray_port, dashboard_flags)),
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
        )

        template = read_script("ray_head.sh")
        return template.format(
            cleanup_cmd=cleanup,
            run_cmd=run,
        )

    def generate_ray_worker_script(
        self,
        image: str,
        container_name: str,
        head_ip: str,
        ray_port: int = 46379,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
        extra_docker_opts: list[str] | None = None,
    ) -> str:
        """Generate a script that starts a Ray worker node.

        Absorbs ``scripts.py::generate_ray_worker_script``.
        """
        from sparkrun.utils import merge_env
        from sparkrun.scripts import read_script

        all_env = merge_env({"RAY_memory_monitor_refresh_ms": "0"}, nccl_env, env)

        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=("ray start --block --address=%s:%d --node-ip-address $NODE_IP" % (head_ip, ray_port)),
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
        )

        template = read_script("ray_worker.sh")
        return template.format(
            cleanup_cmd=cleanup,
            run_cmd=run,
            head_ip=head_ip,
            ray_port=ray_port,
        )

    def generate_node_script(
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
        and RPC cluster nodes.

        Absorbs ``base.py::_generate_node_script``.
        """
        from sparkrun.utils import merge_env

        all_env = merge_env(nccl_env, env)
        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
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
        ) % {"name": container_name, "cleanup": cleanup, "run_cmd": run, "label": label}
