"""Executor ABC, ExecutorConfig dataclass, and SAF extension point.

This module is the canonical home for the :class:`Executor` plugin
interface.  Concrete implementations live alongside in
``executors/docker.py``, ``executors/local.py``, ``executors/k8s.py``.

The public facade :mod:`sparkrun.orchestration.executor` re-exports
the names defined here, plus the resolution helpers
(:func:`resolve_executor` / :func:`get_executor`).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping, TYPE_CHECKING

from scitrera_app_framework import Plugin, Variables, ext_parse_bool, get_extensions

from sparkrun.scripts import read_script
from sparkrun.utils import merge_env
from sparkrun.utils.shell import b64_encode_cmd, quote

if TYPE_CHECKING:
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.hardware import HostHardware

logger = logging.getLogger(__name__)

EXT_EXECUTOR = "sparkrun.executor"

# Canonical container/pod label schema for sparkrun-launched workloads.
# Concrete executors emit these in their native annotation system
# (Docker --label, k8s metadata.labels, etc.) and query_status reads
# them back to enrich the ClusterStatus snapshot.
LABEL_CLUSTER_ID = "sparkrun.cluster_id"
LABEL_RECIPE = "sparkrun.recipe"
LABEL_RANK = "sparkrun.rank"
LABEL_RUNTIME = "sparkrun.runtime"


def _registered_executor_names() -> set[str]:
    """Return executor selectors registered with SAF under :data:`EXT_EXECUTOR`.

    Falls back to the built-in static set when SAF isn't initialized
    (e.g. test paths that build :class:`ExecutorConfig` directly without
    going through ``init_sparkrun``).
    """
    try:
        from sparkrun.core.bootstrap import get_variables

        v = get_variables()
    except Exception:  # pragma: no cover - degraded path
        v = None

    if v is not None:
        try:
            plugins = get_extensions(EXT_EXECUTOR, v=v)
            names = {p.executor_name for p in plugins.values() if getattr(p, "executor_name", "")}
            if names:
                return names
        except Exception:
            logger.debug("Falling back to static executor name set", exc_info=True)

    # Static fallback — keeps the validation logic functioning for
    # in-tree tests that import :class:`ExecutorConfig` directly without
    # bootstrapping the SAF plugin registry.
    return {"docker", "local", "k8s"}


@dataclass
class ExecutorConfig:
    """Typed view of resolved executor settings.

    Constructed from a config chain (or plain dict) after CLI → recipe
    → runtime → per-executor adjustments → SparkrunConfig →
    per-executor defaults layering, driven by
    :func:`sparkrun.orchestration.executor.resolve_executor`.
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

    # Executor selector.  ``"docker"`` (default) uses
    # :class:`DockerExecutor`; ``"local"`` opts into the experimental
    # :class:`LocalExecutor` that runs the serve command as a native
    # subprocess; ``"k8s"`` opts into the experimental
    # :class:`K8sExecutor` draft.
    executor_type: str = "docker"

    # LocalExecutor-only fields (ignored by Docker/K8s).
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
        """Build from a config chain or plain dict.

        Only keys present (non-None) in *chain* override the
        dataclass-level field defaults.  Per-executor defaults are
        applied **upstream** (the bottom layer of
        :func:`resolve_executor`'s chain), not inside this method —
        this keeps ``ExecutorConfig`` executor-agnostic.
        """
        kwargs: dict = {}

        # Bool fields — only forward when explicitly present.
        for key in ("auto_remove", "privileged"):
            v = chain.get(key)
            if v is not None:
                kwargs[key] = ext_parse_bool(v)

        # Plain (non-nullable) string fields — only forward when present.
        for key in ("gpus", "ipc", "shm_size", "network"):
            v = chain.get(key)
            if v is not None:
                kwargs[key] = str(v)

        # Nullable string fields — falsy values map to dataclass default.
        for key in (
            "restart_policy",
            "user",
            "memory_limit",
            "accelerator_vendor",
            "working_dir",
            "log_dir",
            "log_file",
            "pid_dir",
            "pid_file",
            "env_file",
            "command_prefix",
            "k8s_namespace",
            "k8s_context",
            "k8s_node_selector",
            "k8s_image_pull_policy",
            "kubeconfig",
        ):
            v = chain.get(key)
            if v:
                kwargs[key] = v

        # List-or-string fields — promote bare strings to single-item lists.
        for key in ("security_opt", "cap_add", "ulimit", "devices", "labels"):
            raw = chain.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                raw = [raw]
            kwargs[key] = raw or None

        # Executor selector: prefer ``executor``, fall back to
        # ``executor_type`` for forward compat.  Validity is checked
        # against the SAF plugin registry (see
        # :func:`_registered_executor_names`); unknown values warn and
        # degrade to ``"docker"``.
        exec_type_raw = chain.get("executor")
        if exec_type_raw is None:
            exec_type_raw = chain.get("executor_type")
        if exec_type_raw:
            executor_type = str(exec_type_raw).strip().lower()
            known = _registered_executor_names()
            if executor_type not in known:
                logger.warning(
                    "Unknown executor type %r; falling back to 'docker'. Known: %s",
                    executor_type,
                    sorted(known),
                )
                executor_type = "docker"
            kwargs["executor_type"] = executor_type

        return cls(**kwargs)

    def __post_init__(self):
        # Docker does not allow --rm with --restart
        if self.restart_policy:
            self.auto_remove = False


class Executor(Plugin):
    """Abstract base for executors.

    Each concrete executor (Docker, local-native-subprocess, K8s, …) is
    a SAF :class:`Plugin` registered at the ``sparkrun.executor``
    extension point and discovered via :func:`find_types_in_modules`.
    The plugin instance returned from the extension registry acts as a
    *class registry* — callers look up the type and instantiate a
    fresh per-launch ``Executor`` via
    :func:`sparkrun.orchestration.executor.resolve_executor`.
    Per-launch config lives on ``self.config``; the registered SAF
    singleton uses a default ``ExecutorConfig`` only to keep its
    method signatures uniform.

    Subclasses must define:

    - ``executor_name``: identifier used to select this executor
      (matched against ``recipe.executor`` / ``SparkrunConfig.default_executor``).
    - The abstract low-level command generators below.

    Subclasses *should* override:

    - :meth:`default_config` — per-executor defaults dict.  Lowest
      layer of the resolution chain (after the dataclass defaults).
    - :meth:`apply_runtime_adjustments` — runtime-driven adjustments
      (e.g. Docker reads ``rootless`` / ``auto_user`` flags here).
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    executor_name: ClassVar[str] = ""

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.executor.%s" % self.executor_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_EXECUTOR

    def is_enabled(self, v: Variables) -> bool:
        # Must return False for multi-extension plugins to prevent SAF's
        # single-extension cache from short-circuiting subsequent plugin
        # initializations under the same extension point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger=None) -> "Executor":
        return self

    def __init__(self, config: ExecutorConfig | None = None):
        super().__init__()
        self.config = config or ExecutorConfig()

    # --- Per-executor layering hooks ---

    @classmethod
    def default_config(cls) -> dict:
        """Per-executor default settings — lowest-priority chain layer.

        Override to ship sensible defaults (e.g. DockerExecutor's
        ``shm_size``/``ipc=host``).  Bare base class contributes
        nothing — the dataclass field defaults already provide the
        bottom layer.
        """
        return {}

    @classmethod
    def apply_runtime_adjustments(cls, *, rootless: bool = True, auto_user: bool = True, **kwargs) -> dict:
        """Runtime-driven adjustments — sits above SparkrunConfig.

        Only executors that care about a given knob need to consume
        it.  Docker reads ``rootless``/``auto_user`` to flip
        ``privileged`` and inject ``--user $SHELL_USER``; Local and
        K8s ignore both.  ``**kwargs`` is accepted for forward
        compatibility (future signals like ``cluster=…``).
        """
        return {}

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
        """Generate a command that exits 0 iff *container_name* is alive."""
        ...

    @abstractmethod
    def inspect_exists_cmd(self, image: str) -> str:
        """Generate a command to check if an image exists locally."""
        ...

    @abstractmethod
    def pull_cmd(self, image: str) -> str:
        """Generate an image pull command string."""
        ...

    # --- Status introspection ---

    def query_status(
        self,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
        host_hardware: "Mapping[str, HostHardware] | None" = None,
    ) -> "ClusterStatus":
        """Inspect sparkrun-launched workloads running on *hosts*.

        Default implementation returns a zero-occupancy snapshot — every
        host is treated as fully free with no running workloads.  This
        is a safe degradation for executors that don't yet implement
        introspection (e.g. the K8sExecutor draft); they satisfy the
        contract without lying about state they can't see.

        Concrete executors override to query their backend
        (``docker ps``, ``kubectl get pods``, local process state).
        """
        from sparkrun.core.cluster_status import empty_status

        return empty_status(hosts, executor=self.executor_name)

    @classmethod
    def workload_labels(
        cls,
        cluster_id: str,
        recipe_name: str | None = None,
        runtime_name: str | None = None,
        rank: int | None = None,
    ) -> dict[str, str]:
        """Return the canonical sparkrun label set for a workload.

        Used by run-script generators to tag containers/pods so
        :meth:`query_status` can recover workload identity from the
        backend's native introspection (``docker ps``, ``kubectl get``).

        Returns a plain ``dict[str, str]``; concrete executors choose
        how to emit them (``--label key=value`` for Docker, k8s
        ``metadata.labels``, …).
        """
        labels: dict[str, str] = {LABEL_CLUSTER_ID: cluster_id}
        if recipe_name:
            labels[LABEL_RECIPE] = recipe_name
        if runtime_name:
            labels[LABEL_RUNTIME] = runtime_name
        if rank is not None:
            labels[LABEL_RANK] = str(rank)
        return labels

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
        """Generate a script that cleans up then launches a container."""

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
        """Generate a script that executes the serve command inside a running container."""

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
        """Generate a script that starts a Ray head node in a container."""

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
        """Generate a script that starts a Ray worker node."""

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
        """Generate a script that launches a container with a direct entrypoint command."""

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
