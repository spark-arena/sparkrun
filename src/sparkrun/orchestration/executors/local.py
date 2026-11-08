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

from sparkrun.core.application_profile import remote_cache_path, get_application_profile
from sparkrun.core.ownership import OWNER_LABEL, owns_resource, assert_resource_namespace

import logging
import posixpath
import re
import time
from typing import Mapping, TYPE_CHECKING

from sparkrun.orchestration.executors._base import Executor, ExecutorTarget
from sparkrun.orchestration.executors._local_paths import home_relative as _home_relative, validate_managed_paths
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
    r"^(?P<cluster>[a-z][a-z0-9-]{0,47}_(?P<intent>[0-9a-f]{%d})_[0-9a-f]{%d})_(?P<role>solo|head|worker|node_(?P<rank>\d+))$"
    % (INTENT_ID_LEN, PLACEMENT_TOKEN_LEN)
)


# Where pidfiles/logfiles land when no explicit override is provided.
# Lives under ``~/.cache/sparkrun/local/`` so it follows the same
# convention as the rest of the sparkrun runtime state.
def default_pid_dir():
    return remote_cache_path("local/pids")


def default_log_dir():
    return remote_cache_path("local/logs")


# ``--gpus device=0,2`` → CUDA_VISIBLE_DEVICES=0,2.  Anything fancier
# (``count=2``, capability filters) is ignored with a warning.
_GPUS_DEVICE_RE = re.compile(r"device=([0-9,]+)")


def _remote_path(path: str, *, directory: bool = False) -> str:
    """Normalize spelling without changing traversal or expansion eligibility."""
    home_relative = _home_relative(path)
    if home_relative is not None:
        path = "$HOME/" + home_relative if home_relative or path.endswith("/") else "$HOME"
    root = "//" if path.startswith("//") and not path.startswith("///") else "/" if path.startswith("/") else ""
    result = root + "/".join(part for part in path.split("/") if part and part != ".")
    result = result or "."
    # Removing './' must not turn a literal directory named '$HOME' or '~'
    # into an expanding prefix. Retain this distinction in target serialization.
    if home_relative is None and _home_relative(result) is not None:
        result = "./" + result
    # A trailing slash/dot on a file path requires a directory. Do not turn a
    # bad file path into a different successful read by dropping that constraint.
    if not directory and path.endswith(("/", "/.")) and not result.endswith("/"):
        result += "/"
    return result


def _state_helpers() -> str:
    from sparkrun.scripts import read_script

    return read_script("local_state.sh")


def _shell_path(path: str) -> str:
    """Expand an opted-in leading remote home; quote every other path component."""
    rest = _home_relative(path)
    if rest is not None:
        return '"$HOME"/' + quote(rest) if rest else '"$HOME"'
    return quote("./" + path if path.startswith("-") else path)


def _bind_launch_path(variable: str, path: str) -> str:
    """Bind control state to the entry directory/home before workload setup."""
    return ('%(var)s=%(path)s\ncase "$%(var)s" in /*) ;; *) %(var)s="$PWD/$%(var)s" ;; esac\nreadonly %(var)s\n') % {
        "var": variable,
        "path": _shell_path(path),
    }


class LocalExecutor(Executor):
    """Native-subprocess executor (experimental, no container).

    The bash scripts this class generates assume ``setsid`` is
    available (it is part of util-linux on every modern Linux distro).
    Status and teardown include live workers in the launched process group,
    even after its leader exits. Descendants that leave the group are outside
    this executor's lifecycle tracking; it is not a process supervisor.
    """

    executor_name = "local"
    required_feature_flag = "executor.local"
    needs_image = False

    # No Docker-style defaults; the dataclass field defaults are
    # appropriate.  No rootless/auto_user concerns either.

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def _require_managed_paths(self):
        validate_managed_paths(vars(self.config))

    @staticmethod
    def _require_detached(detach):
        if not detach:
            raise ValueError("LocalExecutor supports detached workload launch only; foreground launch is unsupported")

    def _pid_directory(self) -> str:
        return _remote_path(self.config.pid_dir or default_pid_dir(), directory=True)

    def _log_directory(self) -> str:
        return _remote_path(self.config.log_dir or default_log_dir(), directory=True)

    def resolve_target(self, *, dry_run=False) -> ExecutorTarget:
        self._require_managed_paths()
        pid_dir = self._pid_directory()
        return ExecutorTarget(
            self.executor_name,
            {"pid_dir": pid_dir, "log_dir": self._log_directory()},
            destination_key=pid_dir if pid_dir != _remote_path(default_pid_dir(), directory=True) else "",
            user_scoped=True,
        )

    def _resolve_pid_file(self, container_name: str) -> str:
        """Raw remote path; fixed files are retained only for legacy recovery."""
        if self.config.pid_file:
            return _remote_path(self.config.pid_file)
        return posixpath.join(self._pid_directory(), container_name + ".pid")

    def _resolve_log_file(self, container_name: str) -> str:
        """Raw remote logfile path, before shell rendering."""
        if self.config.log_file:
            return _remote_path(self.config.log_file)
        return posixpath.join(self._log_directory(), container_name + ".log")

    # ------------------------------------------------------------------
    # Bash fragment helpers
    # ------------------------------------------------------------------

    def _env_prelude(self, env: dict[str, str] | None = None) -> str:
        """Emit the bash setup lines shared by ``run_cmd`` and ``exec_cmd``.

        Order matters: cd → source env_file → export gpu vars → export
        explicit env. Required setup failures abort the subshell.
        env_file is relative to working_dir (or the entry directory when unset).
        Returns a fragment ending in a newline (or empty).
        """
        cfg = self.config
        lines: list[str] = []
        if cfg.working_dir:
            lines.append("cd -- %s || exit $?" % _shell_path(cfg.working_dir))
        if cfg.env_file:
            # 'set -a' so sourced KEY=VAL lines become exports — matches
            # docker --env-file semantics.
            # Bash source searches PATH for a bare name. A configured file is
            # relative to the workload directory, not a PATH-selected script.
            env_file = cfg.env_file
            if "/" not in env_file and _home_relative(env_file) is None:
                env_file = "./" + env_file
            lines.append("set -a")
            lines.append(". %s || exit $?" % _shell_path(env_file))
            lines.append("set +a")

        gpus_export = self._visible_devices_export()
        if gpus_export:
            lines.append(gpus_export + " || exit $?")

        if env:
            for key, value in sorted(env.items()):
                lines.append("export %s=%s || exit $?" % (key, quote(str(value))))

        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    def _visible_devices_variable(self) -> str | None:
        return {"nvidia": "CUDA_VISIBLE_DEVICES", "amd": "ROCR_VISIBLE_DEVICES", "intel": "HABANA_VISIBLE_MODULES"}.get(
            self.config.accelerator_vendor or "nvidia"
        )

    def _visible_devices_export(self) -> str | None:
        """Translate ``--gpus`` into a ``CUDA_VISIBLE_DEVICES`` export.

        - ``"all"`` / empty / ``None`` → no export (use whatever's visible).
        - ``"device=0,2"`` → ``export CUDA_VISIBLE_DEVICES=0,2``.
        - Anything else (``count=2``, capability filters) → warn + skip.
        """
        variable = self._visible_devices_variable()
        if variable is None:
            return ""
        gpus = (self.config.gpus or "").strip()
        if not gpus or gpus.lower() == "all":
            return None
        m = _GPUS_DEVICE_RE.match(gpus)
        if m:
            return "export %s=%s" % (variable, quote(m.group(1)))
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
        """Emit a detached setsid launcher; ``detach=False`` is unsupported.

        Success means the complete PID claim was committed, not that the
        serving endpoint is ready. PID-commit failure rolls back the child.

        *image* is ignored — there is no container.  *volumes* mounts
        nothing (native execution), but it *is* consulted to reverse-map
        container-path env values back to their host source paths via
        :func:`_hostify_env` (e.g. ``HF_HOME=/cache/huggingface`` →
        ``HF_HOME=/home/ubuntu/.cache/huggingface``), so a natively-run
        serve command finds host-side resources the Docker path would
        have bind-mounted.  *extra_opts* are docker-only and are silently
        dropped.  *sparkrun_labels* carries GPU assignments persisted beside the PID record.  Workload identity for
        the LocalExecutor flows through the pidfile name + job metadata
        cache instead (see :func:`_parse_local_pidfile_output`).
        """
        self._require_detached(detach)
        self._require_managed_paths()
        from sparkrun.core.allocations import ALLOCATION_LABEL, decode_allocations

        allocation_record = (sparkrun_labels or {}).get(ALLOCATION_LABEL)
        if not container_name:
            raise ValueError("LocalExecutor.run_cmd requires container_name")
        if not command:
            raise ValueError("LocalExecutor.run_cmd requires a non-empty command")

        paths = _bind_launch_path("_sr_pid_path", self._resolve_pid_file(container_name)) + _bind_launch_path(
            "_sr_log_path", self._resolve_log_file(container_name)
        )
        pid_file, log_file = '"$_sr_pid_path"', '"$_sr_log_path"'
        full_cmd = self._full_command(command)

        # NOTE: ``setsid`` makes the child a session leader → its own
        # process group. Status and teardown track its surviving members.
        # Control paths are fixed before setup changes the directory or HOME.
        prelude = self._env_prelude(_hostify_env(env, volumes))
        allocations = decode_allocations(allocation_record)
        variable = self._visible_devices_variable()
        if allocations and variable is not None:
            prelude += "export %s=%s || exit $?\n" % (
                variable,
                quote(",".join(str(i) for i in dict.fromkeys(a.gpu_index for a in allocations))),
            )
        body = (
            "(\n"
            "%(helpers)s\n"
            "%(paths)s"
            "%(lock)s\n"
            "_sr_local_state %(pid)s || exit $?\n"
            "%(guard)s\n"
            'if _sr_local_alive "$_sr_pid"; then\n'
            '    echo "Refusing to replace a running native workload" >&2; exit 1\n'
            'else _rc=$?; [ "$_rc" -eq 1 ] || exit "$_rc"; fi\n'
            "mkdir -p -- %(log_dir_dq)s || exit 1\n"
            "%(prelude)s"
            "_sr_local_write_record %(owner)s %(pid)s.owner || exit 1\n"
            "%(allocation_commit)s"
            "set +m\n"
            "setsid bash -c %(b64_cmd)s >>%(log)s 2>&1 </dev/null 9>&- &\n"
            "_pid=$!\n"
            '_sr_local_commit_pid "$_pid" %(pid)s || exit $?\n'
            'printf "Launched %%s (pid=%%s, log=%%s)\\n" %(name)s "$_pid" %(log)s\n'
            ") || exit $?\n"
        ) % {
            "helpers": _state_helpers(),
            "paths": paths,
            "lock": self._pid_lock(pid_file),
            "guard": self._owner_guard(container_name),
            "log_dir_dq": '"$(dirname -- %s)"' % log_file,
            "prelude": prelude,
            "b64_cmd": _bash_safe_command(full_cmd),
            "log": log_file,
            "pid": pid_file,
            "name": quote(container_name),
            "owner": quote(get_application_profile().id),
            "allocation_commit": (
                "_sr_local_write_record %s %s.allocations || exit 1\n" % (quote(allocation_record), pid_file)
                if allocation_record and allocations
                else "rm -f -- %s.allocations || exit 1\n" % pid_file
            ),
        }
        return body

    def exec_cmd(
        self,
        container_name: str,
        command: str,
        detach: bool = False,
        env: dict[str, str] | None = None,
        *,
        user: str | None = None,
    ) -> str:
        """Run *command* in a subshell with the same env prelude.

        Used by ``pre_exec`` / ``post_exec`` hooks.  Always foreground
        — ``detach`` is ignored for the local path (hooks are meant to
        complete before the next phase).  *container_name* is unused
        here but kept for ABC parity, and *user* is ignored: a native
        workload runs as the connection's user, so its hooks do too.
        """
        if user is not None:
            logger.debug("LocalExecutor: running as the connection's user, not %r (a native workload has no user switch)", user)
        prelude = self._env_prelude(env)
        if prelude:
            return "( %sbash -c %s )" % (prelude, _bash_safe_command(command))
        return "bash -c %s" % _bash_safe_command(command)

    def interactive_exec_argv(self, host, container_name, command=None, *, tty=False, ssh_kwargs=None, resource=None):
        """``ssh [-t] host bash -c <script>`` that enters the native workload's environment.

        There is no container to step into, so the equivalent of ``docker exec``
        is the workload process's own environment: the script verifies the PID
        record (and its owner, as stop does), then imports ``/proc/<pid>/environ``
        and the cwd from a live member of the process group. That is exactly
        what the serve command got, including an environment builder's venv
        activation (``uv-venv``'s ``env_file``) and the launch env. When no
        member is readable it falls back to this executor's own setup
        (working_dir, env_file, GPU visibility) and says so, since that lacks
        the launch env. The terminal's own ``TERM`` / ``SSH_*`` are kept. As
        with ``docker exec``, this is the environment the process was *started*
        with: a variable the serve command exports inside itself is not seen.

        The script travels base64-encoded and is decoded by the remote shell,
        so it is not re-parsed by whatever login shell the host user has.
        """
        from sparkrun.orchestration.ssh import interactive_ssh_argv
        from sparkrun.utils.shell import b64_encode_cmd

        assert_resource_namespace(container_name)
        pid_file = _shell_path(self._resolve_pid_file(container_name))
        if command is None:
            # ~/.bashrc may prepend to PATH, which would put another python
            # ahead of the workload's venv. Re-assert the workload's PATH after it.
            final = (
                "_SR_WORKLOAD_PATH=$PATH exec bash --rcfile <(printf '%s\\n' "
                "'[ -f ~/.bashrc ] && . ~/.bashrc' 'PATH=$_SR_WORKLOAD_PATH; unset _SR_WORKLOAD_PATH') -i"
            )
        else:
            final = "exec bash -c %s" % quote(command)
        script = (
            "%(helpers)s\n"
            "_sr_local_state %(pid)s || exit $?\n"
            "%(guard)s\n"
            'if _sr_local_alive "$_sr_pid"; then :; else\n'
            '    _rc=$?; [ "$_rc" -eq 1 ] && echo "Native workload "%(name)s" is not running" >&2; exit "$_rc"\n'
            "fi\n"
            "_sr_env_pid=$_sr_pid\n"
            '[ -r "/proc/$_sr_env_pid/environ" ] || _sr_env_pid=$(ps -e -o pid=,pgid=,stat= | awk -v g="$_sr_pid" \'$2 == g && $3 !~ /^[ZX]/ {print $1; exit}\')\n'
            'if [ -n "$_sr_env_pid" ] && [ -r "/proc/$_sr_env_pid/environ" ] && _sr_cwd=$(readlink -- "/proc/$_sr_env_pid/cwd"); then\n'
            "    while IFS= read -r -d '' _sr_kv; do\n"
            "        [[ $_sr_kv =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue\n"
            "        case ${_sr_kv%%%%=*} in TERM|SHLVL|_|PWD|OLDPWD|COLUMNS|LINES|SSH_*) continue ;; esac\n"
            '        export -- "$_sr_kv"\n'
            '    done < "/proc/$_sr_env_pid/environ"\n'
            '    cd -- "$_sr_cwd" 2>/dev/null || echo "The workload\'s working directory is gone ($_sr_cwd); staying in $PWD" >&2\n'
            "else\n"
            '    echo "Cannot read the workload process environment; using the executor setup (no launch env)" >&2\n'
            "%(prelude)s"
            "fi\n"
            "%(final)s\n"
        ) % {
            "helpers": _state_helpers(),
            "pid": pid_file,
            "guard": self._owner_guard(container_name, verb="enter"),
            "name": quote(container_name),
            "prelude": self._env_prelude() or ":\n",
            "final": final,
        }
        remote = 'bash --noprofile --norc -c "$(printf %%s %s | base64 -d)"' % b64_encode_cmd(script)
        return interactive_ssh_argv(host, remote, tty=tty, **(ssh_kwargs or {}))

    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Stop a verified workload group; retain records on an unknown outcome."""
        return self._stop_command(container_name)

    def _stop_command(self, container_name: str, *, report_removed=False) -> str:
        assert_resource_namespace(container_name)
        pid_file = _shell_path(self._resolve_pid_file(container_name))
        return (
            "(\n"
            "%(helpers)s\n"
            "%(lock)s\n"
            "_sr_local_state %(pid)s || exit $?\n"
            "%(guard)s\n"
            '_sr_local_stop "$_sr_pid" %(name)s || exit $?\n'
            "rm -f -- %(pid)s %(pid)s.owner %(pid)s.allocations || exit 1\n"
            "%(report)s\n"
            ") || exit $?"
        ) % {
            "helpers": _state_helpers(),
            "pid": pid_file,
            "name": quote(container_name),
            "guard": self._owner_guard(container_name),
            "lock": self._pid_lock(pid_file),
            "report": 'printf "%s\\n" "$_sr_was_running"' if report_removed else "",
        }

    def teardown_script(self, container_names: list[str] | tuple[str, ...]) -> str:
        """Count and verify each stop under its PID lock; unknowns fail teardown."""
        from sparkrun.orchestration.teardown import TEARDOWN_REMOVED_MARKER

        lines = ["_sr_removed=0"]
        for name in container_names:
            lines.append("_sr_count=$( %s\n) || exit $?" % self._stop_command(name, report_removed=True))
            lines.append("_sr_removed=$((_sr_removed + _sr_count))")
        lines.append('printf "%%s%%s\\n" %s "$_sr_removed"' % quote(TEARDOWN_REMOVED_MARKER))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _pid_lock(pid_file):
        # Keep the inode stable, and close fd 9 in the workload child.
        return 'mkdir -p -- "$(dirname -- %s)" && exec 9>%s.lock && flock -x 9 || exit 1' % (pid_file, pid_file)

    def _owner_guard(self, name, verb="modify"):
        # _sr_local_state has already distinguished missing and invalid markers.
        condition = '[ "$_sr_owner_present" -eq 1 ] && [ "$_sr_owner" = %s ]' % quote(get_application_profile().id)
        absent_owner = '[ "$_sr_owner_present" -eq 0 ]'
        if not owns_resource(name):
            absent_owner += ' && [ "$_sr_pid_present" -eq 0 ]'
        return ('{ { %s; } || { %s; }; } || { echo "Refusing to %s a native workload owned by another application" >&2; exit 1; };') % (
            condition,
            absent_owner,
            verb,
        )

    def logs_cmd(
        self,
        container_name: str,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Tail the logfile that ``run_cmd`` writes to."""
        log_file = _shell_path(self._resolve_log_file(container_name))
        parts = ["tail"]
        if follow:
            parts.append("-F")  # -F survives logfile rotation/recreation
        parts.extend(["-n", str(int(tail)) if tail is not None else "+1"])
        parts.append(log_file)
        return " ".join(parts)

    def status_cmd(self, container_name: str) -> str:
        """Exit 0 for a live PID/group, 1 for absent/dead, 2 for acquisition failure."""
        pid_file = _shell_path(self._resolve_pid_file(container_name))
        return '(\n%s\n_sr_local_state %s || exit $?\n_sr_local_alive "$_sr_pid"\n)' % (_state_helpers(), pid_file)

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
                % (_shell_path(self._resolve_log_file(n)), quote(n), _shell_path(self._resolve_log_file(n)))
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
                        investigate_hints=("cat %s" % _shell_path(log_file),),
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
        self._require_detached(detach)
        self._require_managed_paths()
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

        This is where the native subprocess starts. Only ``detached=True``
        is supported; foreground requests raise before generating a script.
        ``volumes`` mounts nothing here, but
        is forwarded to :meth:`run_cmd` so container-path env values get
        reverse-mapped to their host source (see :func:`_hostify_env`).
        ``sparkrun_labels`` carries the accepted GPU allocation to the process record.
        """
        self._require_detached(detached)
        self._require_managed_paths()
        # ``run_cmd`` owns the detached launch and PID-commit contract.
        return "#!/bin/bash\nset -uo pipefail\n%s" % self.run_cmd(
            image="",
            command=serve_command,
            container_name=container_name,
            detach=detached,
            env=env,
            volumes=volumes,
            sparkrun_labels=sparkrun_labels,
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
        :func:`_hostify_env`).  ``sparkrun_labels`` carries the accepted GPU allocation to the process record.
        """
        self._require_managed_paths()
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
            sparkrun_labels=sparkrun_labels,
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

        Reads the configured PID directory and validates process and owner
        records on each host. Canonically named workloads belonging to the
        application are surfaced. Unreachable hosts and failed state acquisition
        are reported as errors without claiming complete coverage for that host.
        """
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, with_gpu_allocations
        from sparkrun.core.hardware import resolve_hardware
        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        self._require_managed_paths()
        if not hosts:
            return ClusterStatus(hosts=(), queried_at=time.time(), executor=self.executor_name)

        pid_dir = _shell_path(self._pid_directory())
        # A glob miss is meaningful only after acquiring a readable namespace.
        script = _state_helpers() + (
            "\nif _sr_local_directory %s; then :; else\n"
            '  _rc=$?; [ "$_rc" -eq 1 ] && exit 0; exit "$_rc"\n'
            "fi\n"
            "shopt -s nullglob\n"
            "for f in %s/*.pid; do\n"
            '  name=$(basename -- "$f" .pid) || exit 2\n'
            '  _sr_local_state "$f" || exit $?\n'
            '  if _sr_local_alive "$_sr_pid"; then\n'
            "    _allocation=\n"
            '    if _sr_local_read "$f.allocations" allocation; then _allocation=$_sr_value;\n'
            '    else _rc=$?; [ "$_rc" -eq 1 ] || exit "$_rc"; fi\n'
            '    printf "%%s\\t%%s\\t%%s\\t%%s\\n" "$name" "$_sr_pid" "$_sr_owner" "$_allocation"\n'
            '  else _rc=$?; [ "$_rc" -eq 1 ] || exit "$_rc"; fi\n'
            "done\n"
        ) % (pid_dir, pid_dir)

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

            hw = (host_hardware or {}).get(host) or resolve_hardware()
            capacity = hw.total_gpus

            workloads, used = _parse_local_pidfile_output(r.stdout)
            host_entries.append(
                with_gpu_allocations(
                    HostOccupancy(
                        host=host,
                        workloads=tuple(workloads),
                        used_slots=used,
                        free_slots=max(capacity - used, 0),
                    ),
                    capacity,
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

    def bind_mount_sources(self) -> list[str]:
        """Nothing: a native process has no mounts.

        ``executor_config.volumes`` is inert under this executor (the workload
        reads the host filesystem directly), so the base ``[]`` is the honest
        answer — checking those paths would fail a launch over a setting that
        does nothing here.  Stated explicitly rather than inherited because it
        is a real property of the substrate, not an unimplemented hook.
        """
        return []

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
    from sparkrun.core.allocations import decode_allocations

    by_cluster: dict[str, dict] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, rest = line.partition("\t")
        _pid, _, owner_and_allocation = rest.partition("\t")
        owner, _, allocation_record = owner_and_allocation.partition("\t")
        if not owns_resource(name, {OWNER_LABEL: owner} if owner else None):
            continue
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
                allocations=decode_allocations(allocation_record),
            )
        )

    workloads: list[RunningWorkload] = []
    total = 0
    for cluster_id, bucket in by_cluster.items():
        allocated_ranks = {a.rank for c in bucket["containers"] for a in c.allocations}
        ranks_on_host = (
            len(allocated_ranks) if allocated_ranks and all(c.allocations for c in bucket["containers"]) else len(bucket["ranks"])
        )
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
