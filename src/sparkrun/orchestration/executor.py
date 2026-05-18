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

from sparkrun.scripts import read_script
from sparkrun.utils import merge_env
from sparkrun.utils.shell import b64_encode_cmd, quote

logger = logging.getLogger(__name__)

# Default executor settings for DGX Spark GPU workloads.
# Lowest priority in the config chain: CLI → recipe → these defaults.
EXECUTOR_DEFAULTS = {
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


def build_executor(
    executor_selector: str | None,
    executor_config_dict: dict | None = None,
) -> "Executor":
    """Build an :class:`Executor` from a recipe-style selector + config dict.

    Mirrors the launcher's resolution path but operates from cached
    job-metadata (or other downstream sources) where the full
    rootless/auto_user adjustment layer isn't available.  Returns a
    :class:`DockerExecutor` by default; ``"local"`` opts into
    :class:`LocalExecutor`.
    """
    from scitrera_app_framework.api import Variables, EnvPlacement

    cfg = dict(executor_config_dict or {})
    if executor_selector and "executor" not in cfg:
        cfg["executor"] = executor_selector
    chain = Variables(sources=(cfg, EXECUTOR_DEFAULTS), env_placement=EnvPlacement.IGNORED)
    exec_cfg = ExecutorConfig.from_chain(chain)
    if exec_cfg.executor_type == "local":
        from sparkrun.orchestration.executor_local import LocalExecutor

        return LocalExecutor(exec_cfg)
    if exec_cfg.executor_type == "k8s":
        from sparkrun.orchestration.executor_k8s import K8sExecutor

        return K8sExecutor(exec_cfg)
    from sparkrun.orchestration.executor_docker import DockerExecutor

    return DockerExecutor(exec_cfg)


def accelerator_vendor_for(host_hardware) -> str | None:
    """Return the single shared accelerator vendor on *host_hardware*, or ``None``.

    Used by per-host executor config resolution to decide whether to
    emit NVIDIA ``--gpus``, ROCm device flags, etc.  Returns ``None``
    when the host has zero accelerators or advertises more than one
    vendor (e.g. an Apple M5 host with a discrete NVIDIA GPU); the
    caller is expected to fall back to recipe-level overrides or refuse
    to auto-pack.
    """
    if host_hardware is None:
        return None
    vendors = getattr(host_hardware, "vendors", None)
    if not vendors or len(vendors) != 1:
        return None
    return next(iter(vendors))


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
    shm_size: str = "25gb"
    network: str = "host"
    user: str | None = None
    security_opt: list[str] | None = None
    cap_add: list[str] | None = None
    ulimit: list[str] | None = None
    devices: list[str] | None = None
    memory_limit: str | None = None
    labels: list[str] | None = None
    accelerator_vendor: str | None = None
    """Per-host accelerator vendor that drives device-flag emission.

    ``None`` (default) preserves legacy NVIDIA behavior: the executor
    emits ``--gpus <gpus>``.  Set to ``"nvidia"`` explicitly when a
    per-host config is computed from :class:`HostHardware`.  Other
    supported values:

    - ``"amd"``: ROCm — emits ``--device /dev/kfd --device /dev/dri --group-add video``.
    - ``"intel"``: Intel Gaudi — emits ``--device /dev/accel``.
    - ``"apple"`` / ``"cpu"``: no device flag (Apple MLX requires a
      non-Docker executor; this leaves CPU-only containers untouched).

    For all non-NVIDIA values, ``gpus`` is ignored.
    """

    # Executor selector (experimental).  ``"docker"`` (default) uses
    # :class:`DockerExecutor`; ``"local"`` opts into the experimental
    # :class:`LocalExecutor` that runs the serve command as a native
    # subprocess (no container).
    executor_type: str = "docker"

    # LocalExecutor-only fields (ignored by DockerExecutor).
    working_dir: str | None = None
    log_dir: str | None = None
    log_file: str | None = None
    pid_dir: str | None = None
    pid_file: str | None = None
    env_file: str | None = None
    command_prefix: str | None = None

    # K8sExecutor-only fields (ignored by Docker/Local). All
    # experimental — the K8s executor is a draft pending real-world
    # validation.  Empty values fall back to whatever ``kubectl`` picks
    # up from its current context.
    k8s_namespace: str | None = None
    k8s_context: str | None = None
    k8s_node_selector: str | None = None
    k8s_image_pull_policy: str | None = None
    kubeconfig: str | None = None

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
            val = v if v is not None else EXECUTOR_DEFAULTS.get(key)
            logger.debug("ExecutorConfig resolve: %s=%r (from chain: %r)", key, val, v)
            return val

        # Executor selector: prefer the recipe-level ``executor`` key,
        # fall back to ``executor_type`` for forward compat.  Unknown
        # values warn and degrade to ``"docker"``.
        exec_type_raw = chain.get("executor")
        if exec_type_raw is None:
            exec_type_raw = chain.get("executor_type")
        executor_type = str(exec_type_raw).strip().lower() if exec_type_raw else "docker"
        if executor_type not in ("docker", "local", "k8s"):
            logger.warning("Unknown executor type %r; falling back to 'docker'", executor_type)
            executor_type = "docker"

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
            executor_type=executor_type,
            working_dir=chain.get("working_dir") or None,
            log_dir=chain.get("log_dir") or None,
            log_file=chain.get("log_file") or None,
            pid_dir=chain.get("pid_dir") or None,
            pid_file=chain.get("pid_file") or None,
            env_file=chain.get("env_file") or None,
            command_prefix=chain.get("command_prefix") or None,
            k8s_namespace=chain.get("k8s_namespace") or None,
            k8s_context=chain.get("k8s_context") or None,
            k8s_node_selector=chain.get("k8s_node_selector") or None,
            k8s_image_pull_policy=chain.get("k8s_image_pull_policy") or None,
            kubeconfig=chain.get("kubeconfig") or None,
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
    def status_cmd(self, container_name: str) -> str:
        """Generate a command that exits 0 iff *container_name* is alive.

        Docker: runs ``docker ps --filter name=... -q`` and checks output.
        Local: runs ``kill -0 $(cat <pid_file>)``.

        The command must be safe to run inside ``bash -c``.
        """
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
            container_name=quote(container_name),
            image=quote(image),
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

        env_exports = ""
        if env:
            for key, value in sorted(env.items()):
                env_exports += "export %s=%s; " % (key, quote(str(value)))

        full_cmd = "%s%s" % (env_exports, serve_command)

        # Base64 encode the command to avoid all bash string-escaping/quoting bugs
        # when passing it into `docker exec ... bash -c "..."`
        b64_cmd = b64_encode_cmd(full_cmd)

        script_name = "exec_serve_detached.sh" if detached else "exec_serve_foreground.sh"
        template = read_script(script_name)
        return template.format(
            container_name=quote(container_name),
            b64_cmd=b64_cmd,
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
            "printf 'Cleaning up existing container: %%s\\n' %(name)s\n"
            "%(cleanup)s\n"
            "\n"
            "printf 'Launching %%s: %%s\\n' %(label)s %(name)s\n"
            "%(run_cmd)s\n"
            "\n"
            "# Verify container started\n"
            "sleep 1\n"
            "if docker ps --format '{{.Names}}' | grep -q '^%(name)s$'; then\n"
            "    printf 'Container %%s launched successfully\\n' %(name)s\n"
            "else\n"
            "    printf 'ERROR: Container %%s failed to start\\n' %(name)s >&2\n"
            "    docker logs %(name)s 2>&1 | tail -20 || true\n"
            "    exit 1\n"
            "fi\n"
        ) % {
            "name": quote(container_name),
            "cleanup": cleanup,
            "run_cmd": run,
            "label": quote(label),
        }
