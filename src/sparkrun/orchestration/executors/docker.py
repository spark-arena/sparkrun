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
from sparkrun.core.log_source import MODE_FILE, SERVE_LOG_PATH
from sparkrun.orchestration.job_metadata import INTENT_ID_LEN, PLACEMENT_TOKEN_LEN
from sparkrun.utils.shell import args_list_to_shell_str, assert_safe_mount_source, b64_wrap_bash, quote

if TYPE_CHECKING:
    from sparkrun.containers.entrypoint import EntrypointProbe
    from sparkrun.core.cluster_status import ClusterStatus, TerminationInfo
    from sparkrun.core.hardware import HostHardware
    from sparkrun.core.log_source import LogSource
    from sparkrun.core.runtime_cache import RuntimeCacheMounts

logger = logging.getLogger(__name__)


#: ``gpu_access_mode`` — request GPUs via the Container Device Interface,
#: ``--device nvidia.com/gpu=<id>``.
GPU_ACCESS_CDI = "cdi"

#: ``gpu_access_mode`` — request GPUs via the classic nvidia-container-runtime
#: flag, ``--gpus <spec>``.
GPU_ACCESS_GPUS = "gpus"

GPU_ACCESS_MODES = frozenset({GPU_ACCESS_CDI, GPU_ACCESS_GPUS})

#: Used when ``gpu_access_mode`` is unset or unrecognised.  CDI is the portable
#: choice and the only one some daemons accept (see :func:`_nvidia_gpu_args`);
#: hardware that prefers ``--gpus`` says so at the platform tier.
GPU_ACCESS_DEFAULT = GPU_ACCESS_CDI


def _nvidia_gpu_args(gpus: str | None, mode: str | None = None) -> list[str]:
    """Emit the NVIDIA GPU request flags for *gpus* in the given access *mode*.

    Two spellings exist and neither works everywhere:

    - :data:`GPU_ACCESS_CDI` — ``--device nvidia.com/gpu=<id>`` (Container
      Device Interface).  The modern, portable path (Docker >= 25) and
      *required* on environments whose custom docker rejects ``--gpus`` (e.g.
      Thunder Compute).  It depends on a present, non-stale
      ``/etc/cdi/nvidia.yaml`` — a versioned driver upgrade leaves the spec
      pointing at paths that no longer exist and containers then fail to start,
      which is why some GB10 hosts do better on the classic flag.
    - :data:`GPU_ACCESS_GPUS` — ``--gpus <spec>``, the nvidia-container-runtime
      flag.  Passes the value straight through.

    Maps the ``gpus`` value for CDI — ``"all"`` → ``nvidia.com/gpu=all``;
    ``"device=0,1"`` / ``"0,1"`` → one ``--device`` per id.
    """
    # Falsy gpus (None / "") means "no GPU request" — preserve that (byte-identical
    # to the legacy `if cfg.gpus:` guard) rather than forcing all GPUs.
    if not gpus:
        return []
    resolved = (mode or GPU_ACCESS_DEFAULT).strip().lower()
    if resolved not in GPU_ACCESS_MODES:
        logger.warning(
            "DockerExecutor: unknown gpu_access_mode %r — falling back to %r. Known: %s",
            mode,
            GPU_ACCESS_DEFAULT,
            sorted(GPU_ACCESS_MODES),
        )
        resolved = GPU_ACCESS_DEFAULT
    if resolved == GPU_ACCESS_GPUS:
        return ["--gpus", quote(gpus)]
    spec = gpus.strip().strip('"')
    if not spec or spec == "all":
        return ["--device", "nvidia.com/gpu=all"]
    spec = spec.removeprefix("device=")
    args: list[str] = []
    for dev in spec.split(","):
        dev = dev.strip()
        if dev:
            args += ["--device", quote("nvidia.com/gpu=%s" % dev)]
    return args or ["--device", "nvidia.com/gpu=all"]


# Devices that sparkrun requests but that may legitimately be absent on a host
# (e.g. a cloud GPU with no InfiniBand).  These are emitted existence-guarded so
# ``docker run`` doesn't fail with "no such file or directory" when the node
# isn't present; hosts that have it (IB-equipped DGX Spark clusters) still get it.
_OPTIONAL_DEVICES = frozenset({"/dev/infiniband"})


def _optional_device_arg(dev: str) -> str:
    """Emit a host-existence-guarded ``--device`` for a possibly-absent device.

    Returns a raw, *unquoted* shell command substitution that is evaluated on the
    remote host at launch — the same controlled-substitution pattern already used
    for ``--user $(id -u):$(id -g)``.  The flag is added only when the device node
    exists, so a host without it doesn't fail ``docker run`` while a host with it
    (IB fabric) mounts it normally.

    *dev* is always an internally-sourced literal from :data:`_OPTIONAL_DEVICES`
    (never user/recipe input), so interpolating it into the substitution carries
    no injection risk.
    """
    return "$( [ -e %s ] && printf -- '--device %s' )" % (dev, dev)


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


#: First column of the post-mortem log lines, so ``_parse_terminated_probe``
#: can tell them from the ``docker ps -a`` lines sharing the same stdout.
POST_MORTEM_LOG_MARKER = "SPARKRUN_POSTMORTEM_LOG"

#: How much of a dead workload's tail to retrieve.  Both bounds apply: the line
#: count keeps a chatty engine's log from dominating the SSH round-trip, the
#: byte cap keeps one pathological line from doing the same.  A crash signature
#: is at the very end of the log, so a generous tail buys nothing.
POST_MORTEM_LOG_LINES = 400
POST_MORTEM_LOG_BYTES = 32768


# Env-var keys whose values are likely secrets and must be masked in DEBUG
# logs (case-insensitive substring match on token/key/password/secret).
_SENSITIVE_ENV_KEY_RE = re.compile(r"token|key|password|secret", re.IGNORECASE)


def _mask_sensitive_env_in_command(command: str, env: dict[str, str] | None) -> str:
    """Return *command* with sensitive ``-e KEY=value`` values masked.

    The per-var DEBUG dump masks secrets, but the full assembled command
    line also contains ``-e KEY=value`` tokens — mask those too so a
    secret never reaches logs via the command-line debug entry. Only the
    log representation is masked; the returned command is unchanged.
    """
    if not env:
        return command
    masked = command
    for key, value in env.items():
        if value and _SENSITIVE_ENV_KEY_RE.search(key):
            masked = masked.replace(quote("%s=%s" % (key, value)), quote("%s=***" % key))
    return masked


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
    "gpu_access_mode": GPU_ACCESS_DEFAULT,
    "ipc": "host",
    "shm_size": "32gb",
    "network": "host",
    "user": None,
    "security_opt": None,
    "cap_add": None,
    "ulimit": ["nofile=65535:65535"],
    "devices": None,
    "volumes": None,
    "entrypoint": None,
}


class DockerExecutor(Executor):
    """Docker-based executor for container operations."""

    executor_name = "docker"
    # Gated like every other executor (uniformity) but ships enabled on every
    # channel — ``executor.docker`` defaults on.  The flag exists so all
    # executors self-gate the same way and to leave room to disable docker on
    # hosts/clusters that don't need it.  See ``core.features``.
    required_feature_flag = "executor.docker"

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
            # NOTE: deliberately no ``cap_add`` entry.  Docker grants no extra
            # capabilities unless asked, so ``[]`` here would be identical to
            # ``None`` in the emitted flags — it hardened nothing.  All it did
            # was *suppress* the three layers below this one (runtime /
            # SparkrunConfig / platform defaults) while the four above it
            # (CLI / recipe / builder / cluster) set ``cap_add`` freely under
            # rootless anyway.  That asymmetry left a runtime with no way to
            # declare a capability it genuinely needs — see
            # ``runtimes._util.ptrace_executor_config``.  Untrusted recipes are
            # still blocked from ``cap_add`` by the launcher's trust gate
            # (``_TRUST_GATED_EXECUTOR_KEYS``), which is the real control here.
            adjustments["ulimit"] = [
                "memlock=-1:-1",
                "stack=67108864",
                "nofile=65535:65535",
            ]
            # Request the IB fabric device for rootless (non-privileged) NCCL
            # over InfiniBand.  Emitted existence-guarded at build time (see
            # _OPTIONAL_DEVICES) so hosts without it — solo runs, cloud GPUs — do
            # not fail docker run; IB-equipped hosts still get it.
            adjustments["devices"] = [
                "/dev/infiniband",
            ]
        if auto_user:
            adjustments["user"] = "$SHELL_USER"  # auto hint to use ssh user+group
        return adjustments

    # --- Internal command-string builders ---

    def _accelerator_opts(self) -> list[str]:
        """Emit accelerator device flags based on ``config.accelerator_vendor``.

        - ``None`` (default) or ``"nvidia"`` → the GPU request spelled per
          ``config.gpu_access_mode`` (CDI ``--device nvidia.com/gpu=…`` or
          classic ``--gpus``).
        - ``"amd"`` → ROCm device + render-group flags.
        - ``"intel"`` → Intel Gaudi device flag.
        - ``"apple"`` / ``"cpu"`` → no device flag.  Apple MLX cannot
          run inside Docker; callers should route such hosts to a
          non-Docker executor.
        """
        cfg = self.config
        vendor = (cfg.accelerator_vendor or "").lower()

        if not vendor or vendor == "nvidia":
            return _nvidia_gpu_args(cfg.gpus, cfg.gpu_access_mode)
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

        if cfg.entrypoint is not None:
            opts.extend(["--entrypoint", quote(cfg.entrypoint)])
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
                if dev in _OPTIONAL_DEVICES:
                    # Emitted only when the device exists on the host (see helper).
                    opts.append(_optional_device_arg(dev))
                else:
                    opts.extend(["--device", quote(dev)])
        if cfg.volumes:
            for vol in cfg.volumes:
                # Bare path → identity mount; src:dst / src:dst:ro pass through.
                spec = vol if ":" in vol else "%s:%s" % (vol, vol)
                # Defense in depth: refuse catastrophic host mount sources (root
                # fs, docker socket, SSH keys) regardless of where the spec came
                # from. Untrusted recipes are already blocked upstream.
                assert_safe_mount_source(spec.split(":", 1)[0])
                opts.extend(["-v", quote(spec)])
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
                # Mask values for sensitive keys so secrets never reach logs.
                shown = "***" if _SENSITIVE_ENV_KEY_RE.search(key) else value
                logger.debug("  %s=%s", key, shown)
        logger.debug("docker run command: %s", _mask_sensitive_env_in_command(result, env))

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

    def read_logs_cmd(
        self,
        source: "LogSource",
        *,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Read *source* via ``docker logs`` or an in-container ``tail``.

        Docker is the one substrate that needs the distinction.  sparkrun's
        sleep-infinity + exec launch makes container PID 1 ``sleep infinity``
        and redirects the serve process to a file *inside* the container, so
        ``docker logs`` is structurally blind to it (``docker logs`` shows
        PID 1's stdout — nothing).  ``scripts/exec_serve_detached.sh`` says
        as much at the point of the redirect.  A :data:`MODE_FILE` source is
        therefore read with ``docker exec … tail``; :data:`MODE_STDOUT`
        sources (TRT-LLM cluster mode, Ray worker containers whose PID 1 *is*
        ``ray start --block``) use ``docker logs``.

        ``tail -F`` rather than ``-f``: the serve log is created by the
        exec'd process slightly after the container starts, and ``-F`` waits
        for a not-yet-existing file instead of erroring out.
        """
        if source.mode != MODE_FILE:
            return self.logs_cmd(source.container, follow=follow, tail=tail)

        path = source.path or SERVE_LOG_PATH
        inner = ["tail"]
        if follow:
            inner.append("-F")
        # ``-n +1`` emits the whole file from line 1; a concrete N emits the
        # last N lines — matching stream_container_file_logs' semantics.
        inner.extend(["-n", str(int(tail)) if tail is not None else "+1"])
        inner.append(path)
        return "docker exec %s %s" % (quote(source.container), " ".join(quote(part) for part in inner))

    def verify_command_passthrough(
        self,
        image: str,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> "EntrypointProbe | None":
        """Probe whether *image*'s ENTRYPOINT swallows the appended ``bash -c``.

        Only one host is probed.  Image distribution has already established
        that every host carries the same image (by Id or RepoDigest), and the
        verdict is a property of the image, not of the host — so a second probe
        would spend another container start to re-derive the same answer.

        The executor's own :meth:`_accelerator_opts` are forwarded so the probe
        container starts under the same device conditions the real launch will
        use.  Without them an entrypoint that hard-fails on a missing GPU
        *before* reaching ``exec "$@"`` would look indistinguishable from one
        that consumed the command.

        A resolved ``entrypoint`` override short-circuits the probe: the launch
        will emit ``--entrypoint`` and the image's own ENTRYPOINT never runs, so
        there is nothing left to consume the command.  Without this the probe
        would reject the very fix it recommends — ``entrypoint: ""`` — since the
        image keeps declaring a consuming ENTRYPOINT either way.
        """
        from sparkrun.containers.entrypoint import probe_image_entrypoint

        if not image or not hosts:
            return None
        if self.config.entrypoint is not None:
            logger.debug("Skipping entrypoint probe for %s: launch overrides entrypoint to %r", image, self.config.entrypoint)
            return None
        return probe_image_entrypoint(
            image,
            hosts[0],
            ssh_kwargs=ssh_kwargs,
            accel_opts=self._accelerator_opts(),
        )

    def status_cmd(self, container_name: str) -> str:
        """Exit 0 iff a container named *container_name* is currently running."""
        # Anchored filter so name=foo doesn't match foo_solo etc.
        filter_arg = quote("name=^%s$" % container_name)
        return "[ -n \"$(docker ps --filter %s --format '{{.ID}}')\" ]" % filter_arg

    def exists_cmd(self, container_name: str) -> str:
        """Exit 0 iff the container is present — **running or exited**.

        Diverges from :meth:`status_cmd` (``docker ps``) because a container
        that exited without ``--rm`` still occupies its name and must still be
        removed by teardown.
        """
        filter_arg = quote("name=^%s$" % container_name)
        return "[ -n \"$(docker ps -a --filter %s --format '{{.ID}}')\" ]" % filter_arg

    def teardown_script(self, container_names: list[str] | tuple[str, ...]) -> str:
        """Remove *container_names* via docker, verifying the daemon answered.

        Overrides the ABC's generic composition because Docker's substrate can
        be *unavailable* rather than merely empty: a stopped daemon, a
        permission error or an absent binary makes every ``exists_cmd`` probe
        report "not present", which the generic verification would read as a
        successful teardown.  :func:`~sparkrun.orchestration.docker.docker_teardown_script`
        checks that ``docker ps`` itself succeeded and fails the teardown when
        it did not.  It also does the census in one call instead of one per
        candidate name.
        """
        from sparkrun.orchestration.docker import docker_teardown_script

        return docker_teardown_script(list(container_names))

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
        # ``allow_local=True``: a bare SSH to localhost fails on a host
        # without self-SSH configured, which would make every local
        # workload invisible to status discovery — and to everything
        # downstream of it (stop --all, cluster status, scheduler
        # occupancy, the monitor TUI).
        results = run_remote_scripts_parallel(
            hosts,
            script,
            ssh_user=ssh_kwargs.get("ssh_user"),
            ssh_key=ssh_kwargs.get("ssh_key"),
            ssh_options=ssh_kwargs.get("ssh_options"),
            timeout=ssh_kwargs.get("timeout", 15),
            quiet=True,
            allow_local=True,
        )

        # Map results back to input host order.
        by_host = {r.host: r for r in results}
        host_entries: list[HostOccupancy] = []
        errors: dict[str, str] = {}

        for host in hosts:
            r = by_host.get(host)
            if r is None or r.returncode != 0:
                logger.debug("query_status: skipping unreachable host %r (rc=%s)", host, getattr(r, "returncode", "n/a"))
                errors[host] = (getattr(r, "stderr", "") or "").strip() or "unreachable"
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
            errors=errors,
        )

    def describe_terminated(
        self,
        sources: "list[LogSource]",
        *,
        ssh_kwargs: dict | None = None,
    ) -> "dict[str, TerminationInfo]":
        """Look for stopped containers behind *sources* via ``docker ps -a``.

        ``query_status`` runs ``docker ps`` (running only), so it structurally
        cannot distinguish "stopped but still inspectable" from "gone".  This
        does the ``-a`` pass, batched one script per host through the same
        parallel SSH fan-out ``query_status`` uses.

        **``--rm`` is the reason this is not just an existence check.**
        :attr:`~sparkrun.orchestration.executors._base.ExecutorConfig.auto_remove`
        defaults to ``True``, so the daemon deletes a container the moment it
        exits: absence is then the *normal* outcome of a crash, not evidence
        that a workload never ran.  Reporting it as a bare ``exists=False``
        invites the caller to treat the most interesting failure as stale
        bookkeeping.  We know our own config, so we say which it was and point
        at the setting that would have preserved the evidence next time.

        A container that *did* survive also gets its log tail retrieved in the
        same round-trip, so a failure sparkrun can attribute
        (:func:`~sparkrun.utils.log_diagnostics.detect_in_place_write_failure`)
        becomes a hint naming the fix rather than a ``docker logs`` invitation
        to go read the traceback again.  Nothing is retrieved when the
        container is gone — under ``--rm`` there is no log left to read, which
        is precisely what the ``auto_remove=false`` hint exists to change.
        """
        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        if not sources:
            return {}

        ssh_kwargs = ssh_kwargs or {}
        by_host: dict[str, list[str]] = {}
        for source in sources:
            by_host.setdefault(source.host, []).append(source.container)

        hosts = list(by_host)
        # One script for every host (the fan-out helper takes a single script),
        # probing the union of the names.  Attribution stays per-host below, so
        # a name that legitimately exists on several hosts — Ray worker
        # containers share one name across nodes — is never cross-reported.
        script = self._terminated_probe_script(sorted({s.container for s in sources})) + self._post_mortem_log_script(sources)
        try:
            results = run_remote_scripts_parallel(
                hosts,
                script,
                ssh_user=ssh_kwargs.get("ssh_user"),
                ssh_key=ssh_kwargs.get("ssh_key"),
                ssh_options=ssh_kwargs.get("ssh_options"),
                timeout=ssh_kwargs.get("timeout", 15),
                quiet=True,
                allow_local=True,
            )
        except Exception:  # noqa: BLE001 — best-effort, exactly like query_status
            logger.debug("describe_terminated: probe failed", exc_info=True)
            return {}

        found: dict[tuple[str, str], TerminationInfo] = {}
        result_by_host = {r.host: r for r in results}
        for host in hosts:
            r = result_by_host.get(host)
            # Any non-zero rc (255 SSH failure, 127 no docker, timeout) is
            # inconclusive, not "gone" — leave those containers unreported so
            # the caller keeps the metadata.
            if r is None or r.returncode != 0:
                logger.debug("describe_terminated: inconclusive for %r (rc=%s)", host, getattr(r, "returncode", "n/a"))
                continue
            states = _parse_terminated_probe(r.stdout)
            logs = _parse_post_mortem_logs(r.stdout)
            for container in by_host[host]:
                found[(host, container)] = self._termination_info(container, states.get(container), logs.get(container))
        return found

    def _terminated_probe_script(self, containers: list[str]) -> str:
        """One ``docker ps -a`` per container, emitting ``<name>\\t<status>``.

        Anchored name filters (``^name$``) so ``foo`` cannot match ``foo_solo``
        — the same anchoring :meth:`status_cmd` uses.  A container that no
        longer exists contributes no line at all.
        """
        fmt = quote("{{.Names}}\t{{.Status}}")
        lines = ["docker ps -a --filter %s --format %s 2>/dev/null || true" % (quote("name=^%s$" % c), fmt) for c in containers]
        return "\n".join(lines) + "\n"

    def _post_mortem_log_script(self, sources: "list[LogSource]") -> str:
        """Emit ``<marker>\\t<name>\\t<base64 log tail>`` for each source.

        Retrieval is mode-aware for the same reason :meth:`read_logs_cmd` is:
        sparkrun's sleep-infinity + exec launch makes container PID 1 ``sleep
        infinity`` and redirects the serve process to a file *inside* the
        container, so ``docker logs`` on those workloads is empty — the
        traceback we want to attribute is in :data:`SERVE_LOG_PATH`.  Reading
        it back is ``docker cp`` rather than the ``docker exec`` that
        :meth:`read_logs_cmd` uses, because this runs on a container that has
        already exited and ``exec`` needs a running one.

        Base64 rather than a delimited block: engine logs carry control
        characters and partial UTF-8, and a container is free to print anything
        that looks like our marker.  Encoding makes each source exactly one
        line whose payload cannot be confused for framing.

        Deduplicated by ``(container, mode, path)``, mirroring the ``ps`` half's
        union-of-names: one script serves every host, and per-host attribution
        happens in :meth:`describe_terminated`.
        """
        seen: set[tuple[str, str, str]] = set()
        lines: list[str] = []
        for source in sources:
            path = source.path or SERVE_LOG_PATH
            key = (source.container, source.mode, path)
            if key in seen:
                continue
            seen.add(key)

            if source.mode == MODE_FILE:
                # ``docker cp <c>:<path> -`` writes a tar stream; ``tar -xO``
                # unpacks it to stdout.  Works on a stopped container.
                read = "docker cp %s - 2>/dev/null | tar -xO 2>/dev/null" % quote("%s:%s" % (source.container, path))
            else:
                # ``2>&1`` is load-bearing: ``docker logs`` demultiplexes, so
                # the container's *stderr* — where the traceback is — arrives on
                # this command's stderr.  It also absorbs "No such container".
                read = "docker logs --tail %d %s 2>&1" % (POST_MORTEM_LOG_LINES, quote(source.container))

            lines.append(
                "printf '%%s\\t%%s\\t' %s %s; %s | tail -c %d | base64 | tr -d '\\n' || true; echo"
                % (quote(POST_MORTEM_LOG_MARKER), quote(source.container), read, POST_MORTEM_LOG_BYTES)
            )
        return "\n".join(lines) + "\n" if lines else ""

    def _termination_info(self, container: str, status: str | None, log_text: str | None = None) -> "TerminationInfo":
        """Shape one container's ``docker ps -a`` result into a verdict."""
        from sparkrun.core.cluster_status import TerminationInfo

        if status is not None:
            return TerminationInfo(
                exists=True,
                detail=status,
                investigate_hints=(
                    *self._attribution_hints(log_text),
                    "docker logs %s" % container,
                    "docker inspect %s" % container,
                ),
            )
        if self.config.auto_remove:
            return TerminationInfo(
                exists=False,
                detail="container auto-removed on exit (executor_config.auto_remove is on, so docker run used --rm)",
                investigate_hints=("relaunch with `-o auto_remove=false` to keep the container for inspection",),
            )
        return TerminationInfo(exists=False, detail="no container by that name exists on the host")

    def _attribution_hints(self, log_text: str | None) -> tuple[str, ...]:
        """Turn a recognised log signature into Docker's remedy for it.

        The *detection* is substrate-independent and lives in
        :mod:`sparkrun.utils.log_diagnostics`; the *wording* is ours, because
        ``--user`` is a Docker concept.  A ``local`` job has no container user
        to change and would need entirely different advice, which is the same
        reason ``docker logs`` is not authored in ``api.logs``.

        ``-o user=root`` rather than ``--rootful``: both make the container run
        as root, but ``--rootful`` *also* drops ``--security-opt
        no-new-privileges`` and adds ``--privileged`` (see
        :meth:`apply_runtime_adjustments`).  The image needs to write inside
        itself, not to own the host — recommending the wider flag would trade a
        launch failure for a standing privilege grant.

        Hints are prepended to the generic ``docker logs`` / ``docker inspect``
        pair rather than replacing it: this names a *likely* cause, and the
        operator still wants the raw log.
        """
        if not log_text:
            return ()
        try:
            from sparkrun.utils.log_diagnostics import detect_in_place_write_failure

            failure = detect_in_place_write_failure(log_text)
        except Exception:  # noqa: BLE001 — a post-mortem hint must never raise
            logger.debug("post-mortem attribution failed", exc_info=True)
            return ()
        if failure is None:
            return ()

        logger.debug("post-mortem: in-place write failure at %s (errno %d)", failure.path, failure.errno)
        return (
            "likely cause: the workload failed writing inside its own image installation "
            "(%s: %s) — sparkrun runs containers as the invoking uid, which cannot write there" % (failure.message, failure.path),
            "relaunch with `-o user=root` to run this container as root (narrower than `--rootful`, which also adds --privileged)",
        )

    def verify_mount_sources(
        self,
        paths: list[str],
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> dict[str, list[str]]:
        """Docker binds host paths directly, so identity-mount sources must
        exist on the host FS — SSH-probe them (shared host-substrate impl)."""
        from sparkrun.orchestration.ssh import verify_host_paths

        return verify_host_paths(hosts, list(paths), ssh_kwargs)

    def bind_mount_sources(self) -> list[str]:
        """Docker emits every ``config.volumes`` entry as a ``-v`` bind, so each
        source is a real claim about the host filesystem."""
        return self._parse_bind_sources(self.config.volumes)

    def ensure_runtime_cache(
        self,
        mounts: "RuntimeCacheMounts",
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> None:
        """Docker bind-mounts the cache from the host FS, so create/stamp/sweep it
        there (shared host-substrate impl)."""
        from sparkrun.orchestration.runtime_cache import ensure_runtime_cache_on_hosts

        ensure_runtime_cache_on_hosts(mounts, hosts, ssh_kwargs)


# --------------------------------------------------------------------------
# query_status helpers (module-level so they're unit-testable)
# --------------------------------------------------------------------------


def _parse_terminated_probe(stdout: str) -> dict[str, str]:
    """Parse ``<name>\\t<status>`` lines from the ``docker ps -a`` probe.

    A container that no longer exists contributes no line, so absence from the
    returned mapping is what "gone" looks like.

    The post-mortem log lines share this stdout and have the same tab-separated
    shape, so they are skipped by their marker column — a container cannot be
    named :data:`POST_MORTEM_LOG_MARKER` (sparkrun generates the names, and the
    filters are anchored), so this cannot drop a real status line.
    """
    states: dict[str, str] = {}
    for line in (stdout or "").splitlines():
        name, sep, status = line.partition("\t")
        name = name.strip()
        if not name or not sep or name == POST_MORTEM_LOG_MARKER:
            continue
        states[name] = status.strip()
    return states


def _parse_post_mortem_logs(stdout: str) -> dict[str, str]:
    """Decode the ``<marker>\\t<name>\\t<base64>`` lines into per-container text.

    Best-effort in both directions: a line that does not decode is dropped
    rather than raised on (the payload came off a crashing workload, and the
    worst case for a mangled one is that no hint is offered), and the decoded
    bytes are read with ``errors="replace"`` because engine logs routinely end
    mid-UTF-8 sequence where ``tail -c`` cut them.
    """
    import base64
    import binascii

    logs: dict[str, str] = {}
    for line in (stdout or "").splitlines():
        marker, sep, rest = line.partition("\t")
        if not sep or marker.strip() != POST_MORTEM_LOG_MARKER:
            continue
        name, sep, payload = rest.partition("\t")
        name = name.strip()
        if not name or not sep:
            continue
        payload = payload.strip()
        if not payload:
            continue
        try:
            logs[name] = base64.b64decode(payload, validate=True).decode("utf-8", errors="replace")
        except (binascii.Error, ValueError):
            logger.debug("post-mortem: undecodable log payload for %r", name)
    return logs


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
    from sparkrun.core.cluster_status import ContainerDetail, RunningWorkload

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
        role = m.group("role") or "?"

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
                "containers": [],
                "recipe_name": None,
                "runtime_name": None,
                "intent_id": None,
            },
        )
        bucket["ranks"].add(rank)
        if container_id:
            bucket["container_ids"].append(container_id)
        bucket["containers"].append(
            ContainerDetail(
                name=name,
                role=role,
                status=entry.get("Status") or "",
                image=entry.get("Image") or "",
            )
        )
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
                containers=tuple(bucket["containers"]),
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
