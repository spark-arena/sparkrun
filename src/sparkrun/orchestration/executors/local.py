"""Experimental local (no-container) executor.

:class:`LocalExecutor` runs the runtime's serve command as a native
subprocess on the target host — there is no Docker container in the
loop.  The orchestration / SSH dispatch layer is unchanged: the
executor still emits *scripts*, those scripts still get piped via
``ssh <host> bash -s`` (or run locally via
:func:`should_run_locally`).  The only thing that changes is what the
script does — instead of ``docker run``, it ``setsid``-launches the
serve command, writes its PID to a pidfile, and redirects stdout/stderr
to a logfile.

Selected via the recipe-level ``executor: local`` field (or the
equivalent dict key in ``executor_config``).  Defaults to
``DockerExecutor`` for backward compatibility.

Out of scope (raises if used): Ray strategy, image distribution,
``docker pull`` / ``docker image inspect``.  Multi-host native cluster
runtimes (``vllm-distributed``, ``sglang``) work — per-rank scripts
land on each host with deterministic per-rank pid/log paths derived
from the container name.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Mapping, TYPE_CHECKING

from sparkrun.orchestration.executors._base import Executor
from sparkrun.orchestration.job_metadata import INTENT_ID_LEN, PLACEMENT_TOKEN_LEN
from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.core.cluster_status import ClusterStatus, TerminationInfo
    from sparkrun.core.hardware import HostHardware
    from sparkrun.core.log_source import LogSource
    from sparkrun.core.runtime_cache import RuntimeCacheMounts

logger = logging.getLogger(__name__)

# Same name pattern as DockerExecutor — sparkrun's container_name helpers
# emit ``sparkrun_<intent>_<placement_token>_(solo|head|worker|node_<rank>)``.
# LocalExecutor uses the container_name as the pidfile basename so the
# same parse works.
_PID_NAME_RE = re.compile(
    r"^(?P<cluster>sparkrun_(?P<intent>[0-9a-f]{%d})_[0-9a-f]{%d})_(?P<role>solo|head|worker|node_(?P<rank>\d+))$"
    % (INTENT_ID_LEN, PLACEMENT_TOKEN_LEN)
)

# Where pidfiles/logfiles land when no explicit override is provided.
# Lives under ``~/.cache/sparkrun/local/`` so it follows the same
# convention as the rest of the sparkrun runtime state.
_DEFAULT_PID_DIR = "$HOME/.cache/sparkrun/local/pids"
_DEFAULT_LOG_DIR = "$HOME/.cache/sparkrun/local/logs"

# ``--gpus device=0,2`` → CUDA_VISIBLE_DEVICES=0,2.  Anything fancier
# (``count=2``, capability filters) is ignored with a warning.
_GPUS_DEVICE_RE = re.compile(r"device=([0-9,]+)")


def _shell_path(path: str) -> str:
    """Render a config path (``working_dir`` / ``env_file``) for bash, expanding
    only a leading ``$HOME`` / ``~/`` prefix and shlex-quoting everything else.

    The goal is narrow: let a builder self-determine a ``$HOME``-relative path
    (e.g. a venv activation ``env_file``). Only a leading ``~/``, ``$HOME/`` or
    ``${HOME}/`` is honored — rewritten to an expanding unquoted ``"$HOME"`` —
    and the *remainder* is shlex-quoted so nothing else in the path is
    interpreted by the shell. This keeps a path containing ``$var`` / ``$(...)``
    / backticks / spaces a literal (no command substitution, no accidental
    variable expansion, no injection), unlike a bare double-quote wrapper.
    """
    for prefix in ("~/", "$HOME/", "${HOME}/"):
        if path.startswith(prefix):
            rest = path[len(prefix) :]
            return '"$HOME"/' + quote(rest) if rest else '"$HOME"'
    return quote(path)


class LocalExecutor(Executor):
    """Native-subprocess executor (experimental, no container).

    The bash scripts this class generates assume ``setsid`` is
    available (it is part of util-linux on every modern Linux distro).
    Process-group kill (``kill -- -<pgid>``) is used to clean up the
    whole tree including any workers the runtime forks.
    """

    executor_name = "local"
    required_feature_flag = "executor.local"
    needs_image = False

    # No Docker-style defaults; the dataclass field defaults are
    # appropriate.  No rootless/auto_user concerns either.

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def _resolve_pid_file(self, container_name: str) -> str:
        """Return the pidfile path for *container_name* (single workload)."""
        cfg = self.config
        if cfg.pid_file:
            return cfg.pid_file
        directory = cfg.pid_dir or _DEFAULT_PID_DIR
        return "%s/%s.pid" % (directory, container_name)

    def _resolve_log_file(self, container_name: str) -> str:
        """Return the logfile path for *container_name*."""
        cfg = self.config
        if cfg.log_file:
            return cfg.log_file
        directory = cfg.log_dir or _DEFAULT_LOG_DIR
        return "%s/%s.log" % (directory, container_name)

    # ------------------------------------------------------------------
    # Bash fragment helpers
    # ------------------------------------------------------------------

    def _env_prelude(self, env: dict[str, str] | None = None) -> str:
        """Emit the bash setup lines shared by ``run_cmd`` and ``exec_cmd``.

        Order matters: cd → source env_file → export gpu vars → export
        explicit env.  Returns a fragment ending in a newline (or empty).
        """
        cfg = self.config
        lines: list[str] = []
        if cfg.working_dir:
            lines.append("cd %s" % _shell_path(cfg.working_dir))
        if cfg.env_file:
            # 'set -a' so sourced KEY=VAL lines become exports — matches
            # docker --env-file semantics.
            lines.append("set -a")
            lines.append(". %s" % _shell_path(cfg.env_file))
            lines.append("set +a")

        gpus_export = self._cuda_visible_devices_export()
        if gpus_export:
            lines.append(gpus_export)

        if env:
            for key, value in sorted(env.items()):
                lines.append("export %s=%s" % (key, quote(str(value))))

        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    def _cuda_visible_devices_export(self) -> str | None:
        """Translate ``--gpus`` into a ``CUDA_VISIBLE_DEVICES`` export.

        - ``"all"`` / empty / ``None`` → no export (use whatever's visible).
        - ``"device=0,2"`` → ``export CUDA_VISIBLE_DEVICES=0,2``.
        - Anything else (``count=2``, capability filters) → warn + skip.
        """
        gpus = (self.config.gpus or "").strip()
        if not gpus or gpus.lower() == "all":
            return None
        m = _GPUS_DEVICE_RE.match(gpus)
        if m:
            return "export CUDA_VISIBLE_DEVICES=%s" % quote(m.group(1))
        logger.warning(
            "LocalExecutor: gpus=%r is not translatable to CUDA_VISIBLE_DEVICES; leaving GPU visibility to the workload itself.",
            gpus,
        )
        return None

    def _full_command(self, command: str) -> str:
        """Prepend ``command_prefix`` to *command* when set."""
        prefix = (self.config.command_prefix or "").strip()
        if not prefix:
            return command
        return "%s %s" % (prefix, command)

    # ------------------------------------------------------------------
    # Low-level command generators (Executor ABC)
    # ------------------------------------------------------------------

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
        """Emit a setsid-based native launcher.

        *image* is ignored — there is no container.  *volumes* mounts
        nothing (native execution), but it *is* consulted to reverse-map
        container-path env values back to their host source paths via
        :func:`_hostify_env` (e.g. ``HF_HOME=/cache/huggingface`` →
        ``HF_HOME=/home/ubuntu/.cache/huggingface``), so a natively-run
        serve command finds host-side resources the Docker path would
        have bind-mounted.  *extra_opts* are docker-only and are silently
        dropped.  *sparkrun_labels* is accepted for API symmetry but
        ignored — there is no container to tag.  Workload identity for
        the LocalExecutor flows through the pidfile name + job metadata
        cache instead (see :func:`_parse_local_pidfile_output`).
        """
        del sparkrun_labels  # accepted but unused — no container to tag
        if not container_name:
            raise ValueError("LocalExecutor.run_cmd requires container_name")
        if not command:
            raise ValueError("LocalExecutor.run_cmd requires a non-empty command")

        pid_file = self._resolve_pid_file(container_name)
        log_file = self._resolve_log_file(container_name)
        full_cmd = self._full_command(command)

        # NOTE: ``setsid`` makes the child a session leader → its own
        # process group.  ``kill -TERM -<pgid>`` (in stop_cmd) reaps the
        # whole tree without needing tini.
        prelude = self._env_prelude(_hostify_env(env, volumes))
        body = (
            "mkdir -p %(pid_dir_dq)s %(log_dir_dq)s\n"
            "%(prelude)s"
            "setsid bash -c %(b64_cmd)s >>%(log)s 2>&1 </dev/null &\n"
            "_pid=$!\n"
            'echo "$_pid" > %(pid)s\n'
            'printf "Launched %%s (pid=%%s, log=%%s)\\n" %(name)s "$_pid" %(log)s\n'
        ) % {
            "pid_dir_dq": '"$(dirname %s)"' % pid_file,
            "log_dir_dq": '"$(dirname %s)"' % log_file,
            "prelude": prelude,
            "b64_cmd": _bash_safe_command(full_cmd),
            "log": log_file,
            "pid": pid_file,
            "name": quote(container_name),
        }
        return body

    def exec_cmd(
        self,
        container_name: str,
        command: str,
        detach: bool = False,
        env: dict[str, str] | None = None,
    ) -> str:
        """Run *command* in a subshell with the same env prelude.

        Used by ``pre_exec`` / ``post_exec`` hooks.  Always foreground
        — ``detach`` is ignored for the local path (hooks are meant to
        complete before the next phase).  *container_name* is unused
        here but kept for ABC parity.
        """
        prelude = self._env_prelude(env)
        if prelude:
            return "( %sbash -c %s )" % (prelude, _bash_safe_command(command))
        return "bash -c %s" % _bash_safe_command(command)

    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Signal the process group, wait briefly, SIGKILL, then prune pidfile."""
        pid_file = self._resolve_pid_file(container_name)
        # Send to the negative PID to target the whole process group.
        # ``kill -0`` precheck avoids spurious "no such process" noise.
        # 2>/dev/null on the read guards against missing pidfile.
        return (
            "{ "
            "_pid=$(cat %(pid)s 2>/dev/null || true); "
            'if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then '
            '  kill -TERM -- -"$_pid" 2>/dev/null || kill -TERM "$_pid" 2>/dev/null || true; '
            "  for _i in 1 2 3 4 5 6 7 8 9 10; do "
            '    kill -0 "$_pid" 2>/dev/null || break; '
            "    sleep 1; "
            "  done; "
            '  kill -0 "$_pid" 2>/dev/null && '
            '    { kill -KILL -- -"$_pid" 2>/dev/null || kill -KILL "$_pid" 2>/dev/null || true; }; '
            "fi; "
            "rm -f %(pid)s 2>/dev/null || true; "
            "}"
        ) % {"pid": pid_file}

    def logs_cmd(
        self,
        container_name: str,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Tail the logfile that ``run_cmd`` writes to."""
        log_file = self._resolve_log_file(container_name)
        parts = ["tail"]
        if follow:
            parts.append("-F")  # -F survives logfile rotation/recreation
        if tail is not None:
            parts.extend(["-n", str(int(tail))])
        parts.append(log_file)
        return " ".join(parts)

    def status_cmd(self, container_name: str) -> str:
        """Exit 0 iff the workload's PID is still alive."""
        pid_file = self._resolve_pid_file(container_name)
        return ('{ _pid=$(cat %(pid)s 2>/dev/null || true); [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; }') % {"pid": pid_file}

    def describe_terminated(
        self,
        sources: "list[LogSource]",
        *,
        ssh_kwargs: dict | None = None,
    ) -> "dict[tuple[str, str], TerminationInfo]":
        """Look for a dead workload's leftover pidfile and logfile.

        There is no container here, so "does it still exist" means "did the
        launch leave anything behind": the logfile is what an operator would
        actually read, and unlike a Docker container nothing auto-removes it —
        so the answer is usually yes, and the hint is a plain ``cat``.
        """
        from sparkrun.core.cluster_status import TerminationInfo
        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        if not sources:
            return {}

        ssh_kwargs = ssh_kwargs or {}
        by_host: dict[str, list[str]] = {}
        for source in sources:
            by_host.setdefault(source.host, []).append(source.container)

        hosts = list(by_host)
        names = sorted({s.container for s in sources})
        # ``<name>\t<log_file>`` for each workload whose logfile survives.
        script = (
            "\n".join(
                "[ -e %s ] && printf '%%s\\t%%s\\n' %s %s || true"
                % (quote(self._resolve_log_file(n)), quote(n), quote(self._resolve_log_file(n)))
                for n in names
            )
            + "\n"
        )
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
        except Exception:  # noqa: BLE001 — best-effort, like query_status
            logger.debug("describe_terminated: probe failed", exc_info=True)
            return {}

        found: dict[tuple[str, str], TerminationInfo] = {}
        result_by_host = {r.host: r for r in results}
        for host in hosts:
            r = result_by_host.get(host)
            if r is None or r.returncode != 0:
                logger.debug("describe_terminated: inconclusive for %r (rc=%s)", host, getattr(r, "returncode", "n/a"))
                continue
            logs = {}
            for line in (r.stdout or "").splitlines():
                name, sep, path = line.partition("\t")
                if sep and name.strip():
                    logs[name.strip()] = path.strip()
            for container in by_host[host]:
                log_file = logs.get(container)
                if log_file:
                    found[(host, container)] = TerminationInfo(
                        exists=True,
                        detail="process is gone; its log file remains",
                        investigate_hints=("cat %s" % log_file,),
                    )
                else:
                    found[(host, container)] = TerminationInfo(exists=False, detail="no log file remains on the host")
        return found

    def inspect_exists_cmd(self, image: str) -> str:
        """No-op: there is no image concept for native execution."""
        return "true"

    def pull_cmd(self, image: str) -> str:
        """No-op: there is no image concept for native execution."""
        return "true"

    # ------------------------------------------------------------------
    # High-level script generators (override Executor defaults)
    # ------------------------------------------------------------------

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
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Preflight only — actual launch happens in :meth:`generate_exec_serve_script`.

        Solo-mode invokes ``generate_launch_script`` with a placeholder
        command (``sleep infinity``) to start the container, then
        ``generate_exec_serve_script`` to inject the real serve command.
        For LocalExecutor there is no container to hold; we just clean
        up any stale pidfile so the subsequent launch is well-defined.
        ``sparkrun_labels`` is ignored (no container).
        """
        del sparkrun_labels  # accepted but unused — no container to tag
        cleanup = self.stop_cmd(container_name)
        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            "# LocalExecutor preflight: ensure no stale process owns this name.\n"
            "%(cleanup)s\n"
            'printf "LocalExecutor: preflight complete for %%s\\n" %(name)s\n'
        ) % {
            "cleanup": cleanup,
            "name": quote(container_name),
        }

    def generate_exec_serve_script(
        self,
        container_name: str,
        serve_command: str,
        env: dict[str, str] | None = None,
        detached: bool = True,
        volumes: dict[str, str] | None = None,
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Actually launch the serve command via setsid.

        This is where the native subprocess starts.  ``detached`` is
        honored to match the docker behavior (always true in practice
        for sparkrun's solo flow).  ``volumes`` mounts nothing here, but
        is forwarded to :meth:`run_cmd` so container-path env values get
        reverse-mapped to their host source (see :func:`_hostify_env`).
        ``sparkrun_labels`` is ignored (no container to tag).
        """
        del sparkrun_labels  # accepted but unused — no container to tag
        # ``run_cmd`` already writes the launcher.  Detached / foreground
        # is the same shape for native — the setsid + & ensures the
        # parent script exits while the workload keeps running.
        return "#!/bin/bash\nset -uo pipefail\n%s" % self.run_cmd(
            image="",
            command=serve_command,
            container_name=container_name,
            detach=detached,
            env=env,
            volumes=volumes,
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
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Per-rank native launch script (used by native cluster runtimes).

        Multi-host falls out for free because each host's
        ``container_name`` is ``<cluster_id>_node_<rank>`` — that's the
        basename for the per-rank pidfile and logfile.  ``volumes`` mounts
        nothing here, but is forwarded to :meth:`run_cmd` so container-path
        env values get reverse-mapped to their host source (see
        :func:`_hostify_env`).  ``sparkrun_labels`` is ignored (no
        container to tag).
        """
        del sparkrun_labels  # accepted but unused — no container to tag
        from sparkrun.utils import merge_env

        all_env = merge_env(nccl_env, env)
        cleanup = self.stop_cmd(container_name)
        launcher = self.run_cmd(
            image="",
            command=serve_command,
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
        )
        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            'printf "Cleaning up existing process: %%s\\n" %(name)s\n'
            "%(cleanup)s\n"
            "\n"
            'printf "Launching %%s: %%s\\n" %(label)s %(name)s\n'
            "%(launcher)s\n"
        ) % {
            "name": quote(container_name),
            "label": quote(label),
            "cleanup": cleanup,
            "launcher": launcher,
        }

    def generate_ray_head_script(self, *args, **kwargs) -> str:  # noqa: D401
        """LocalExecutor does not support Ray clustering."""
        raise NotImplementedError(
            "LocalExecutor does not support Ray cluster strategy. Use a native runtime (e.g. vllm-distributed, sglang) or DockerExecutor."
        )

    def generate_ray_worker_script(self, *args, **kwargs) -> str:  # noqa: D401
        """LocalExecutor does not support Ray clustering."""
        raise NotImplementedError(
            "LocalExecutor does not support Ray cluster strategy. Use a native runtime (e.g. vllm-distributed, sglang) or DockerExecutor."
        )

    # ------------------------------------------------------------------
    # Status introspection
    # ------------------------------------------------------------------

    def query_status(
        self,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
        host_hardware: "Mapping[str, HostHardware] | None" = None,
    ) -> "ClusterStatus":
        """Snapshot sparkrun-launched native subprocesses across *hosts*.

        Reads pidfiles under :data:`_DEFAULT_PID_DIR` over SSH and
        ``kill -0``-checks each PID.  Workloads whose pidfile name
        matches the canonical ``sparkrun_<digest>_<role>`` convention
        are surfaced.  Unreachable hosts are omitted.
        """
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
        from sparkrun.core.hardware import default_dgx_spark_hardware
        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        if not hosts:
            return ClusterStatus(hosts=(), queried_at=time.time(), executor=self.executor_name)

        pid_dir = self.config.pid_dir or _DEFAULT_PID_DIR
        # Print "<name>\t<pid>" for each pidfile whose PID is alive.
        script = (
            "shopt -s nullglob\n"
            "for f in %s/*.pid; do\n"
            '  [ -f "$f" ] || continue\n'
            '  name=$(basename "$f" .pid)\n'
            '  pid=$(cat "$f" 2>/dev/null || true)\n'
            '  [ -n "$pid" ] || continue\n'
            '  kill -0 "$pid" 2>/dev/null || continue\n'
            '  printf "%%s\\t%%s\\n" "$name" "$pid"\n'
            "done\n"
        ) % pid_dir

        ssh_kwargs = ssh_kwargs or {}
        # ``allow_local=True`` for the same reason as the docker executor:
        # both share the ``"host"`` status scope and are merged, so a host
        # without self-SSH must not hide its native workloads either.
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

            workloads, used = _parse_local_pidfile_output(r.stdout)
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

    def verify_mount_sources(
        self,
        paths: list[str],
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> dict[str, list[str]]:
        """Local (container-less) runs read the path straight off the host FS,
        so pre-placed weights must exist there — SSH-probe them (shared
        host-substrate impl, identical to the docker executor)."""
        from sparkrun.orchestration.ssh import verify_host_paths

        return verify_host_paths(hosts, list(paths), ssh_kwargs)

    def ensure_runtime_cache(
        self,
        mounts: "RuntimeCacheMounts",
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> None:
        """Native runs read the cache straight off the host FS (``_hostify_env``
        rewrites the container paths back to host ones), so it is created and
        swept exactly as for docker."""
        from sparkrun.orchestration.runtime_cache import ensure_runtime_cache_on_hosts

        ensure_runtime_cache_on_hosts(mounts, hosts, ssh_kwargs)


# --------------------------------------------------------------------------
# query_status helpers (module-level so they're unit-testable)
# --------------------------------------------------------------------------


def _parse_local_pidfile_output(stdout: str) -> tuple[list, int]:
    """Parse ``<name>\\t<pid>`` lines into RunningWorkloads.

    Returns ``(workloads, used_slots)``.  Lines whose name doesn't
    match the sparkrun convention are ignored.  Workloads are
    aggregated by cluster_id so a multi-rank workload on this host
    contributes a single :class:`RunningWorkload` with
    ``ranks_on_host`` reflecting the count.
    """
    from sparkrun.core.cluster_status import ContainerDetail, RunningWorkload

    by_cluster: dict[str, dict] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, _pid = line.partition("\t")
        m = _PID_NAME_RE.match(name)
        if not m:
            continue
        cluster_id = m.group("cluster")
        rank_str = m.group("rank")
        rank = int(rank_str) if rank_str is not None else 0
        bucket = by_cluster.setdefault(cluster_id, {"ranks": set(), "intent_id": m.group("intent"), "containers": []})
        bucket["ranks"].add(rank)
        bucket["containers"].append(
            ContainerDetail(
                name=name,
                role=m.group("role") or "?",
                status="Up (pid %s)" % _pid,
                image="(local process)",
            )
        )

    workloads: list[RunningWorkload] = []
    total = 0
    for cluster_id, bucket in by_cluster.items():
        ranks_on_host = len(bucket["ranks"])
        total += ranks_on_host
        meta = _load_metadata_safely(cluster_id)
        recipe_name = meta.get("recipe") if meta else None
        runtime_name = meta.get("runtime") if meta else None
        intent_id = (meta.get("intent_id") if meta else None) or bucket.get("intent_id")
        workloads.append(
            RunningWorkload(
                cluster_id=cluster_id,
                intent_id=intent_id,
                recipe_name=recipe_name,
                runtime_name=runtime_name,
                ranks_on_host=ranks_on_host,
                containers=tuple(bucket["containers"]),
            )
        )
    return workloads, total


def _load_metadata_safely(cluster_id: str) -> dict | None:
    """Best-effort job-metadata lookup that never raises."""
    try:
        from sparkrun.orchestration.job_metadata import load_job_metadata

        return load_job_metadata(cluster_id)
    except Exception:  # pragma: no cover - defensive
        logger.debug("query_status: load_job_metadata failed for %s", cluster_id, exc_info=True)
        return None


def _hostify_env(env: dict[str, str] | None, volumes: dict[str, str] | None) -> dict[str, str] | None:
    """Reverse-map container-path env VALUES back to their host source paths.

    The DockerExecutor mounts host dirs into the container (``volumes`` is
    ``{host_source: container_target}``) and the runtime emits env pointing at
    the *container* target (e.g. ``HF_HOME=/cache/huggingface``).  The
    LocalExecutor runs the serve command natively — those container paths do
    not exist on the host — so any env value that references a mount target is
    rewritten to point at the host source instead.

    A value maps when it equals a container target or starts with
    ``target + "/"``; the matched target prefix is replaced with the host
    source.  When several targets match, the longest (most specific) target
    wins.  Non-string / non-path values pass through untouched.  Returns
    *env* unchanged when *volumes* is falsy.
    """
    if not env or not volumes:
        return env

    # Longest container target first so the most specific mount wins.
    targets = sorted(
        ((container_target, host_source) for host_source, container_target in volumes.items() if container_target),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    result: dict[str, str] = {}
    for key, value in env.items():
        new_value = value
        if isinstance(value, str):
            for container_target, host_source in targets:
                if value == container_target:
                    new_value = host_source
                    break
                if value.startswith(container_target + "/"):
                    new_value = host_source + value[len(container_target) :]
                    break
        result[key] = new_value
    return result


def _bash_safe_command(command: str) -> str:
    """Wrap *command* into a base64-decoded bash invocation argument.

    Mirrors :func:`sparkrun.utils.shell.b64_wrap_bash` semantics so we
    don't have to worry about quoting the serve command (which often
    contains single quotes, embedded newlines, etc.).  Returns a token
    suitable as the argument to ``bash -c``.
    """
    from sparkrun.utils.shell import b64_wrap_bash

    return b64_wrap_bash(command)
