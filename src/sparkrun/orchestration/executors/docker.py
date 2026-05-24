"""Docker implementation of the Executor abstraction.

``DockerExecutor`` generates Docker CLI command strings from
``ExecutorConfig`` settings.  The Docker-flavoured global defaults
(``privileged``, ``ipc=host``, ``shm_size``, ...) and the
``rootless``/``auto_user`` adjustment layer live here — they are not
shared concerns of other executors.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Mapping, TYPE_CHECKING

from sparkrun.orchestration.executors._base import (
    LABEL_INTENT_ID,
    LABEL_RANK,
    LABEL_RECIPE,
    LABEL_RUNTIME,
    Executor,
)
from sparkrun.orchestration.job_metadata import INTENT_ID_LEN, PLACEMENT_TOKEN_LEN
from sparkrun.utils.shell import args_list_to_shell_str, b64_wrap_bash, quote

if TYPE_CHECKING:
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.hardware import HostHardware

logger = logging.getLogger(__name__)


# Matches the deterministic sparkrun container-name convention emitted by
# :class:`Executor.container_name` / :class:`Executor.node_container_name`:
# ``sparkrun_<intent>_<placement_token>_(solo|head|worker|node_<rank>)``
# where ``intent`` is :data:`INTENT_ID_LEN` hex chars and
# ``placement_token`` is :data:`PLACEMENT_TOKEN_LEN` hex chars.
#
# ``cluster`` captures the full ``sparkrun_...`` cluster_id; ``intent``
# captures the intent_id prefix.
_CONTAINER_NAME_RE = re.compile(
    r"^(?P<cluster>sparkrun_(?P<intent>[0-9a-f]{%d})_[0-9a-f]{%d})_(?P<role>solo|head|worker|node_(?P<rank>\d+))$"
    % (INTENT_ID_LEN, PLACEMENT_TOKEN_LEN)
)


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
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Generate a ``docker run`` command string.

        ``sparkrun_labels`` is emitted as additional ``--label key=value``
        flags so :meth:`query_status` (and any external observer using
        ``docker ps --filter "label=sparkrun.intent_id=..."``) can recover
        workload identity from the Docker daemon itself.  User-supplied
        ``cfg.labels`` is still emitted in :meth:`_build_default_opts`;
        both sets coexist on the resulting container.
        """
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

        if sparkrun_labels:
            for key, value in sorted(sparkrun_labels.items()):
                parts.extend(["--label", quote("%s=%s" % (key, value))])

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

    # --- Status introspection ---

    def query_status(
        self,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
        host_hardware: "Mapping[str, HostHardware] | None" = None,
    ) -> "ClusterStatus":
        """Snapshot sparkrun-launched Docker containers across *hosts*.

        Implementation: ``docker ps --no-trunc --format '{{json .}}'`` over
        SSH (one parallel script per host), filtered by the canonical
        sparkrun container-name pattern.  Workload identity is recovered
        from the name (cluster_id + rank); recipe/runtime are read from
        the optional sparkrun labels when present and enriched from
        ``~/.cache/sparkrun/jobs/`` job metadata when the labels haven't
        been emitted yet.

        Unreachable hosts are omitted from :attr:`ClusterStatus.hosts`;
        callers can detect this via ``status.for_host(h) is None``.
        """
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
        from sparkrun.core.hardware import default_dgx_spark_hardware
        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        if not hosts:
            return ClusterStatus(hosts=(), queried_at=time.time(), executor=self.executor_name)

        ssh_kwargs = ssh_kwargs or {}
        script = "docker ps --no-trunc --format '{{json .}}' 2>/dev/null || true\n"
        results = run_remote_scripts_parallel(
            hosts,
            script,
            ssh_user=ssh_kwargs.get("ssh_user"),
            ssh_key=ssh_kwargs.get("ssh_key"),
            ssh_options=ssh_kwargs.get("ssh_options"),
            timeout=ssh_kwargs.get("timeout", 15),
            quiet=True,
        )

        # Map results back to input host order.
        by_host = {r.host: r for r in results}
        host_entries: list[HostOccupancy] = []

        for host in hosts:
            r = by_host.get(host)
            if r is None or r.returncode != 0:
                logger.debug("query_status: skipping unreachable host %r (rc=%s)", host, getattr(r, "returncode", "n/a"))
                continue

            hw = (host_hardware or {}).get(host) or default_dgx_spark_hardware()
            capacity = hw.total_gpus

            workloads, used = _parse_docker_ps_output(r.stdout, host)
            host_entries.append(
                HostOccupancy(
                    host=host,
                    workloads=tuple(workloads),
                    used_slots=used,
                    free_slots=max(capacity - used, 0),
                )
            )

        return ClusterStatus(
            hosts=tuple(host_entries),
            queried_at=time.time(),
            executor=self.executor_name,
        )


# --------------------------------------------------------------------------
# query_status helpers (module-level so they're unit-testable)
# --------------------------------------------------------------------------


def _parse_docker_labels(raw: str) -> dict[str, str]:
    """Parse Docker's ``--format`` Labels field: ``k1=v1,k2=v2``."""
    out: dict[str, str] = {}
    if not raw:
        return out
    for token in raw.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, _, value = token.partition("=")
        out[key.strip()] = value.strip()
    return out


def _parse_docker_ps_output(stdout: str, host: str) -> tuple[list, int]:
    """Parse ``docker ps --format '{{json .}}'`` output into RunningWorkloads.

    Returns ``(workloads, used_slots)``.  Containers whose names don't
    match the sparkrun convention are ignored.  Workloads are deduplicated
    by ``(cluster_id, container_id)`` and aggregated by cluster so that a
    cluster with multiple ranks on this host contributes one
    :class:`RunningWorkload` with ``ranks_on_host`` reflecting the count.
    """
    from sparkrun.core.cluster_status import RunningWorkload

    # Group sightings by cluster_id so we can aggregate ranks_on_host.
    by_cluster: dict[str, dict] = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            logger.debug("query_status: ignoring non-JSON docker ps line on %r: %r", host, line[:80])
            continue

        # ``Names`` can be a comma-separated list; sparkrun names don't
        # include commas so any comma is an alias we ignore.
        name = (entry.get("Names") or "").split(",", 1)[0].strip()
        if not name:
            continue
        m = _CONTAINER_NAME_RE.match(name)
        if not m:
            continue

        cluster_id = m.group("cluster")
        rank_str = m.group("rank")
        rank_from_name = int(rank_str) if rank_str is not None else 0
        intent_from_name = m.group("intent")

        labels = _parse_docker_labels(entry.get("Labels") or "")
        # Labels take precedence when present (future-proof for richer
        # tagging); fall back to name-derived rank otherwise.
        rank = int(labels[LABEL_RANK]) if LABEL_RANK in labels else rank_from_name
        recipe_name = labels.get(LABEL_RECIPE)
        runtime_name = labels.get(LABEL_RUNTIME)
        intent_id = labels.get(LABEL_INTENT_ID) or intent_from_name
        container_id = entry.get("ID") or ""

        bucket = by_cluster.setdefault(
            cluster_id,
            {
                "ranks": set(),
                "container_ids": [],
                "recipe_name": None,
                "runtime_name": None,
                "intent_id": None,
            },
        )
        bucket["ranks"].add(rank)
        if container_id:
            bucket["container_ids"].append(container_id)
        if recipe_name and bucket["recipe_name"] is None:
            bucket["recipe_name"] = recipe_name
        if runtime_name and bucket["runtime_name"] is None:
            bucket["runtime_name"] = runtime_name
        if intent_id and bucket["intent_id"] is None:
            bucket["intent_id"] = intent_id

    # Enrich missing recipe/runtime/intent_id from cached job metadata
    # when the labels haven't been emitted yet.
    workloads: list[RunningWorkload] = []
    total_ranks_on_host = 0
    for cluster_id, bucket in by_cluster.items():
        if bucket["recipe_name"] is None or bucket["runtime_name"] is None or bucket["intent_id"] is None:
            meta = _load_metadata_safely(cluster_id)
            if meta is not None:
                bucket["recipe_name"] = bucket["recipe_name"] or meta.get("recipe")
                bucket["runtime_name"] = bucket["runtime_name"] or meta.get("runtime")
                bucket["intent_id"] = bucket["intent_id"] or meta.get("intent_id")

        ranks_on_host = len(bucket["ranks"])
        total_ranks_on_host += ranks_on_host
        workloads.append(
            RunningWorkload(
                cluster_id=cluster_id,
                intent_id=bucket["intent_id"],
                recipe_name=bucket["recipe_name"],
                runtime_name=bucket["runtime_name"],
                ranks_on_host=ranks_on_host,
                container_ids=tuple(bucket["container_ids"]),
            )
        )

    return workloads, total_ranks_on_host


def _load_metadata_safely(cluster_id: str) -> dict | None:
    """Best-effort job-metadata lookup that never raises."""
    try:
        from sparkrun.orchestration.job_metadata import load_job_metadata

        return load_job_metadata(cluster_id)
    except Exception:  # pragma: no cover - defensive
        logger.debug("query_status: load_job_metadata failed for %s", cluster_id, exc_info=True)
        return None
