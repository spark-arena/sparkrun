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

from sparkrun.orchestration.executors._base import Executor
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)

# Where pidfiles/logfiles land when no explicit override is provided.
# Lives under ``~/.cache/sparkrun/local/`` so it follows the same
# convention as the rest of the sparkrun runtime state.
_DEFAULT_PID_DIR = "$HOME/.cache/sparkrun/local/pids"
_DEFAULT_LOG_DIR = "$HOME/.cache/sparkrun/local/logs"

# ``--gpus device=0,2`` → CUDA_VISIBLE_DEVICES=0,2.  Anything fancier
# (``count=2``, capability filters) is ignored with a warning.
_GPUS_DEVICE_RE = re.compile(r"device=([0-9,]+)")


class LocalExecutor(Executor):
    """Native-subprocess executor (experimental, no container).

    The bash scripts this class generates assume ``setsid`` is
    available (it is part of util-linux on every modern Linux distro).
    Process-group kill (``kill -- -<pgid>``) is used to clean up the
    whole tree including any workers the runtime forks.
    """

    executor_name = "local"

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
            lines.append("cd %s" % quote(cfg.working_dir))
        if cfg.env_file:
            # 'set -a' so sourced KEY=VAL lines become exports — matches
            # docker --env-file semantics.
            lines.append("set -a")
            lines.append(". %s" % quote(cfg.env_file))
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
    ) -> str:
        """Emit a setsid-based native launcher.

        *image* and *volumes* are ignored — there is no container.
        *extra_opts* are docker-only and are silently dropped.
        """
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
        prelude = self._env_prelude(env)
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
    ) -> str:
        """Preflight only — actual launch happens in :meth:`generate_exec_serve_script`.

        Solo-mode invokes ``generate_launch_script`` with a placeholder
        command (``sleep infinity``) to start the container, then
        ``generate_exec_serve_script`` to inject the real serve command.
        For LocalExecutor there is no container to hold; we just clean
        up any stale pidfile so the subsequent launch is well-defined.
        """
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
    ) -> str:
        """Actually launch the serve command via setsid.

        This is where the native subprocess starts.  ``detached`` is
        honored to match the docker behavior (always true in practice
        for sparkrun's solo flow).
        """
        # ``run_cmd`` already writes the launcher.  Detached / foreground
        # is the same shape for native — the setsid + & ensures the
        # parent script exits while the workload keeps running.
        return "#!/bin/bash\nset -uo pipefail\n%s" % self.run_cmd(
            image="",
            command=serve_command,
            container_name=container_name,
            detach=detached,
            env=env,
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
        """Per-rank native launch script (used by native cluster runtimes).

        Multi-host falls out for free because each host's
        ``container_name`` is ``<cluster_id>_node_<rank>`` — that's the
        basename for the per-rank pidfile and logfile.
        """
        from sparkrun.utils import merge_env

        all_env = merge_env(nccl_env, env)
        cleanup = self.stop_cmd(container_name)
        launcher = self.run_cmd(
            image="",
            command=serve_command,
            container_name=container_name,
            detach=True,
            env=all_env,
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


def _bash_safe_command(command: str) -> str:
    """Wrap *command* into a base64-decoded bash invocation argument.

    Mirrors :func:`sparkrun.utils.shell.b64_wrap_bash` semantics so we
    don't have to worry about quoting the serve command (which often
    contains single quotes, embedded newlines, etc.).  Returns a token
    suitable as the argument to ``bash -c``.
    """
    from sparkrun.utils.shell import b64_wrap_bash

    return b64_wrap_bash(command)
