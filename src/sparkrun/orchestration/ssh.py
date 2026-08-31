"""SSH remote execution via bash -s stdin piping.

All remote operations in sparkrun are executed by generating scripts
as Python strings and piping them to `ssh <host> bash -s` via stdin.
No files are ever copied to remote hosts.

A host in the target list may *be* the control machine (the single-node
DGX Spark case, or a control node that is also a cluster member).  SSH to
such a host only works when self-SSH has been configured, so the
local-vs-SSH dispatch primitives (:func:`should_run_locally`,
:func:`run_local_script`) live here alongside :class:`RemoteResult`, and
the fan-out helpers can honour them via ``allow_local=True``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

from sparkrun.utils.shell import quote, quote_list, args_list_to_shell_str, stdin_bytes

logger = logging.getLogger(__name__)

# The three attribute classes rsync cannot apply to a *destination directory it
# does not own* — the common case for a cache on NFS (root_squash, a differing
# uid mapping, or a directory a container created as root).  rsync transfers the
# data fine, then the generator EPERMs applying attributes to the destination
# root and exits 23, so a complete transfer is reported as a failure:
#
#     rsync: [generator] chgrp "/cache/." failed: Operation not permitted (1)
#     rsync error: some files/attrs were not transferred (code 23)
#
# ``-a`` implies ``-rlptgoD``; of those only ``-p``, ``-g`` and directory times
# can fail this way (``-o`` is already a no-op for a non-root user).  Dropping
# them costs nothing we rely on: rsync still creates *new* files with the
# source's mode masked by the receiving umask, so executability survives (which
# matters for the hook/mod paths that stage scripts) — ``--no-perms`` only
# declines to chmod files that already exist.  File times (``-t``) are kept,
# since rsync writes via temp-file+rename and therefore owns what it creates.
#
# This is the *default* relaxation, applied everywhere.  ``preserve_perms:
# false`` remains the harder one (it also drops ``-t``) but is no longer needed
# for the ordinary shared-cache case.
NFS_SAFE_ATTR_OPTS = ["--no-perms", "--no-group", "--omit-dir-times"]

_DEFAULT_RSYNC_OPTIONS = ["-az", "--mkpath", "--partial", "--links", *NFS_SAFE_ATTR_OPTS]

# What the automatic retry falls back to: everything in NFS_SAFE_ATTR_OPTS plus
# file times.  Times are *not* in the default set because rsync writes via
# temp-file+rename and so owns what it creates — but a destination file that
# already exists and belongs to someone else still refuses ``utime``, and no
# amount of anticipation covers every filesystem.  Equivalent to what
# ``preserve_perms: false`` asks for, reached automatically instead of by
# configuration.
RSYNC_RELAXED_ATTR_OPTS = [*NFS_SAFE_ATTR_OPTS, "--no-times"]

# Env kill-switch for the relaxed retry (see :func:`relax_rsync_options`).
NO_RSYNC_RETRY_ENV = "SPARKRUN_NO_RSYNC_RETRY"


def rsync_retry_disabled() -> bool:
    """True when the relaxed rsync retry is switched off via the environment."""
    return os.environ.get(NO_RSYNC_RETRY_ENV, "").strip().lower() not in ("", "0", "false", "no")


def relax_rsync_options(rsync_options: list[str]) -> list[str]:
    """Append :data:`RSYNC_RELAXED_ATTR_OPTS`, or return ``None``-equivalent input.

    Appending rather than rewriting is deliberate: rsync resolves repeated
    attribute flags last-wins, so ``-a … --no-times`` disables times without
    parsing what the caller asked for — which means this works for any option
    set, including ones added later that this function has never seen.
    """
    return [*rsync_options, *(o for o in RSYNC_RELAXED_ATTR_OPTS if o not in rsync_options)]


def rsync_options_are_relaxed(rsync_options: list[str]) -> bool:
    """True when *rsync_options* already carries every relaxation the retry adds."""
    return all(o in rsync_options for o in RSYNC_RELAXED_ATTR_OPTS)


# Default cap on concurrent SSH/rsync fan-out workers.  At 32+ hosts an
# uncapped ``max_workers=len(hosts)`` spawns one SSH (or ``docker save|ssh
# docker load`` pipeline) per host simultaneously, hitting sshd's
# ``MaxStartups`` (default 10) and saturating the control node.  Capping the
# thread pool bounds concurrency while leaving small clusters (<= cap hosts)
# byte-for-byte unchanged.  Overridable via ``SparkrunConfig.max_parallel_ssh``
# (``ssh.max_parallel_ssh`` in config.yaml).
DEFAULT_MAX_PARALLEL_SSH = 20

# Concurrency cap for head→worker fan-out inside the embedded
# ``image_distribute.sh`` / ``model_distribute.sh`` scripts.  These run ON THE
# HEAD and stream multi-GB transfers (``docker save | ssh docker load``, rsync)
# out a single NIC/disk, so the cap is intentionally small — overlap connect
# latency without saturating the head.  Distinct from
# ``DEFAULT_MAX_PARALLEL_SSH`` (control-node SSH fan-out).
HEAD_DISTRIBUTE_MAX_PARALLEL = 4


# Env kill-switch for the remote session guard (see
# :func:`wrap_with_session_guard`).  The guard relies on ``ps`` and on sshd
# exiting when the client goes away; set this to opt out on a host where that
# doesn't hold, at the cost of orphaned work on a killed launch.
NO_SESSION_GUARD_ENV = "SPARKRUN_NO_SESSION_GUARD"

# Sentinel line in ``scripts/session_guard.sh`` replaced by the payload.
_GUARD_PAYLOAD_SENTINEL = "__SPARKRUN_PAYLOAD__"


def session_guard_disabled() -> bool:
    """True when the session guard is switched off via the environment."""
    return os.environ.get(NO_SESSION_GUARD_ENV, "").strip().lower() not in ("", "0", "false", "no")


def wrap_with_session_guard(script: str) -> str:
    """Wrap *script* so it dies with its SSH session.

    Remote payloads run via ``ssh <host> bash -s``, i.e. **without a PTY**.  On
    disconnect sshd's session process exits without signalling its child (the
    SIGHUP-on-disconnect path is PTY-only), so a killed ``sparkrun`` on the
    control node leaves the payload running on the remote host — invisible from
    the control node, holding HF cache locks, consuming WAN, and stacking across
    kill-and-retry cycles.

    The wrapper backgrounds *script* in its own process group and polls its own
    parent PID; when the session dies the shell is reparented (to init/systemd)
    and the payload's whole process group is TERMed, then KILLed.  It is
    transparent otherwise: stdout, stderr and the exit code pass through
    unchanged.

    Opt-in per call site (see the ``session_guard`` argument on
    :func:`run_remote_script` and friends) — it is meant for long-running,
    resource-consuming work (model downloads, image pulls, head→worker
    fan-outs), not for short status probes where an orphan is harmless.

    Returns *script* unchanged when :data:`NO_SESSION_GUARD_ENV` is set.
    """
    if session_guard_disabled():
        logger.debug("Session guard disabled via %s", NO_SESSION_GUARD_ENV)
        return script

    from sparkrun.scripts import read_script

    guard = read_script("session_guard.sh")
    # Splice on the sentinel *line*, not a substring: the guard's own header
    # comment names the token, and a substring replace would splice the payload
    # into that comment instead.
    lines = guard.splitlines()
    try:
        at = next(i for i, line in enumerate(lines) if line.strip() == _GUARD_PAYLOAD_SENTINEL)
    except StopIteration:  # pragma: no cover — would mean a corrupt install
        logger.error("session_guard.sh is missing its payload sentinel; running unguarded")
        return script
    # The payload is spliced in verbatim; the explicit newline keeps a payload
    # without a trailing one from running into the subshell's closing paren.
    lines[at : at + 1] = (script.rstrip("\n") + "\n").splitlines()
    return "\n".join(lines) + "\n"


def resolve_parallel_cap(n: int, cap: int | None = None) -> int:
    """Return the worker count for a fan-out over *n* items.

    ``min(n, cap)`` with ``cap`` defaulting to
    :data:`DEFAULT_MAX_PARALLEL_SSH`.  Always returns at least ``1`` so a
    ``ThreadPoolExecutor`` is never constructed with ``max_workers=0``.
    """
    if cap is None or cap <= 0:
        cap = DEFAULT_MAX_PARALLEL_SSH
    return max(1, min(n, cap))


@dataclass
class RemoteResult:
    """Result of a remote script execution."""

    host: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def last_line(self) -> str:
        """Get the last non-empty line of stdout (useful for extracting IPs etc)."""
        lines = [line for line in self.stdout.strip().splitlines() if line.strip()]
        return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Local execution / dispatch
# ---------------------------------------------------------------------------


def should_run_locally(host: str, ssh_user: str | None = None) -> bool:
    """True if *host* is local AND no cross-user SSH is needed.

    Use this instead of :func:`~sparkrun.utils.is_local_host` at
    execution dispatch points (where the code decides "run locally via
    subprocess" vs "run via SSH").  Keep ``is_local_host`` for pure
    address-identity checks (e.g. "is this IP me?").

    Returns ``True`` when the host is local and *ssh_user* is ``None``
    or matches the current OS user.
    """
    from sparkrun.utils import is_local_host

    if not is_local_host(host):
        return False
    if ssh_user is None:
        return True
    return ssh_user == os.environ.get("USER", "root")


def run_local_script(script: str, dry_run: bool = False, timeout: int | None = None) -> RemoteResult:
    """Execute a script locally via subprocess.

    Args:
        script: Bash script content to execute.
        dry_run: If True, log the script but don't execute.
        timeout: Wall-clock cap in seconds.  Mirrors the SSH path's
            ``timeout``: a local dispatch must not be able to hang a
            caller (status polling, teardown) that bounded the remote
            path.  A timeout is reported as rc 124, like ``timeout(1)``.

    Returns:
        RemoteResult with host set to ``"localhost"``.
    """
    if dry_run:
        script_lines = script.count("\n")
        logger.info("[dry-run] Would execute locally (%d lines, %d bytes)", script_lines, len(script))
        return RemoteResult(host="localhost", returncode=0, stdout="[dry-run]", stderr="")

    try:
        proc = subprocess.run(
            ["bash", "-s"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        logger.warning("Local execution timed out after %ss", timeout)
        return RemoteResult(
            host="localhost",
            returncode=124,
            stdout=_decode(e.stdout),
            stderr=(_decode(e.stderr) + ("\n" if e.stderr else "")) + "local execution timed out after %ss" % timeout,
        )
    return RemoteResult(
        host="localhost",
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def run_local_script_streaming(
    script: str,
    dry_run: bool = False,
    timeout: int | None = None,
    quiet: bool = False,
) -> RemoteResult:
    """Execute a local script with the same output contract as remote streaming.

    The local peer of :func:`run_remote_script_streaming`, so a caller that
    dispatches local-or-remote (``run_script_on_host_streaming``) gets one
    behaviour either way.  Non-``quiet`` inherits the terminal, which is what
    keeps a long build visibly alive; ``quiet`` keeps the captured result for
    callers that render their own progress.
    """
    if quiet:
        return run_local_script(script, dry_run=dry_run, timeout=timeout)
    if dry_run:
        logger.info("[dry-run] Would execute locally (streaming; %d bytes)", len(script))
        return RemoteResult(host="localhost", returncode=0, stdout="[dry-run]", stderr="")

    started = time.monotonic()
    try:
        proc = subprocess.run(
            ["bash", "-s"],
            input=stdin_bytes(script),
            text=False,
            timeout=timeout,
            stdout=None,
            stderr=None,
        )
        elapsed = time.monotonic() - started
        if proc.returncode == 0:
            logger.debug("  Local script (streaming) <- OK (%.1fs)", elapsed)
        else:
            logger.warning("  Local script (streaming) <- FAILED rc=%d (%.1fs)", proc.returncode, elapsed)
        return RemoteResult(host="localhost", returncode=proc.returncode, stdout="", stderr="")
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        logger.error("  Local script (streaming) <- TIMEOUT after %.0fs", elapsed)
        return RemoteResult(host="localhost", returncode=124, stdout="", stderr="Execution timed out")
    except Exception as error:
        elapsed = time.monotonic() - started
        logger.error("  Local script (streaming) <- ERROR (%.1fs): %s", elapsed, error)
        return RemoteResult(host="localhost", returncode=-1, stdout="", stderr=str(error))


def _decode(raw: bytes | str | None) -> str:
    """Normalize subprocess output to text.

    Output is captured in binary mode (see :func:`_run_subprocess`), so this is
    a decode in practice; ``errors="replace"`` keeps an odd byte from turning a
    command failure into a UnicodeDecodeError. ``str`` is accepted unchanged so
    a caller (or a test double) that already has text isn't a special case.
    """
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")


# Truncation budget for a failed command's captured output.  The default suits
# a probe that fails on one line.  rsync does not: it reports one line *per
# problem file* and its most diagnostic line is rarely the first, so 200 chars
# routinely cut a message mid-path — leaving ``mkdir ".../sglang/c`` with the
# reason it failed truncated away, which is precisely the byte that decides
# whether a transfer completed.
_DEFAULT_FAILURE_DETAIL_LIMIT = 200
RSYNC_FAILURE_DETAIL_LIMIT = 2000


def _failure_detail(result: "RemoteResult", limit: int = _DEFAULT_FAILURE_DETAIL_LIMIT) -> str:
    """Best available explanation for a failed remote script.

    Falls back from stderr to stdout, then to an explicit ``(no output)``.
    Logging stderr alone produced a bare ``FAILED rc=1 (0.3s):`` with nothing
    after the colon whenever the remote payload reported its problem on
    *stdout* — which the embedded scripts routinely do, since they ``echo``
    diagnostics. An empty reason reads as a tool malfunction rather than as a
    remote command that failed for a stated reason.  Mirrors the fallback
    :func:`sparkrun.orchestration.hooks._run_exec_command` already applies.

    Truncation is marked when it happens, so a reader can tell a complete
    message from a clipped one rather than guessing at the tail.
    """
    for stream in (result.stderr, result.stdout):
        text = (stream or "").strip()
        if text:
            if len(text) > limit:
                return text[:limit] + "… (truncated)"
            return text
    return "(no output)"


def _run_subprocess(
    cmd: list[str] | str,
    host: str,
    label: str,
    timeout: int | None = None,
    input_data: str | None = None,
    shell: bool = False,
    quiet: bool = False,
    detail_limit: int = _DEFAULT_FAILURE_DETAIL_LIMIT,
) -> RemoteResult:
    """Run a subprocess and return a RemoteResult with standard error handling.

    Centralizes the try/subprocess.run/TimeoutExpired/Exception pattern
    used by all SSH, rsync, and pipeline execution functions.

    Args:
        cmd: Command to execute (list or string for shell=True).
        host: Host identifier for the result and log messages.
        label: Human-readable label for log messages (e.g. "SSH script", "Rsync").
        timeout: Execution timeout in seconds.
        input_data: Optional stdin data.
        shell: Whether to use shell=True.
        quiet: If True, downgrade failure logging from WARNING to DEBUG.
            Used for expected-failure probes (e.g. NOPASSWD sudo checks).
        detail_limit: Truncation budget for the logged failure output.
            Commands whose diagnostics run to many lines (rsync) pass a
            larger budget — see :data:`RSYNC_FAILURE_DETAIL_LIMIT`.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    t0 = time.monotonic()
    try:
        # Bytes in, bytes out: text mode would rewrite every "\n" in the script
        # to os.linesep, which breaks the remote bash on a Windows control
        # machine (see utils.shell.stdin_bytes).
        proc = subprocess.run(
            cmd,
            input=stdin_bytes(input_data) if input_data is not None else None,
            capture_output=True,
            text=False,
            timeout=timeout,
            shell=shell,
        )
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host,
            returncode=proc.returncode,
            stdout=_decode(proc.stdout),
            stderr=_decode(proc.stderr),
        )
        if result.success:
            logger.debug("  %s <- %s OK (%.1fs)", label, host, elapsed)
        else:
            log_fn = logger.debug if quiet else logger.warning
            log_fn(
                "  %s <- %s FAILED rc=%d (%.1fs): %s",
                label,
                host,
                proc.returncode,
                elapsed,
                _failure_detail(result, limit=detail_limit),
            )
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  %s <- %s TIMEOUT after %.0fs", label, host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  %s <- %s ERROR (%.1fs): %s", label, host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


def build_ssh_cmd(
    host: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
) -> list[str]:
    """Build the base SSH command with standard options.

    Args:
        host: Remote hostname or IP address.
        ssh_user: Optional SSH username (prepended as user@host).
        ssh_key: Optional path to SSH private key file.
        ssh_options: Additional SSH command-line options.
        connect_timeout: SSH connection timeout in seconds.

    Returns:
        List of command parts suitable for subprocess.
    """
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}"]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    if ssh_options:
        cmd.extend(ssh_options)
    target = f"{ssh_user}@{host}" if ssh_user else host
    cmd.append(target)
    return cmd


def run_remote_script(
    host: str,
    script: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    timeout: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
    allow_local: bool = False,
    session_guard: bool = False,
) -> RemoteResult:
    """Execute a script on a remote host via stdin piping.

    The script is generated in-process and piped directly to
    ``ssh <host> bash -s`` on the remote. No files are copied.

    Args:
        host: Remote hostname or IP.
        script: Bash script content to execute.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the script but don't execute.
        quiet: If True, downgrade failure logging from WARNING to DEBUG.
        allow_local: Run the script directly (no SSH) when *host* is this
            machine — see :func:`run_remote_scripts_parallel` for why this
            is opt-in.
        session_guard: Wrap the script so it dies with its SSH session — see
            :func:`wrap_with_session_guard`.  Opt-in, for long-running work
            that must not be orphaned by a killed launch.  Ignored on the
            local-dispatch path, which has no SSH session to lose.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    script_lines = script.count("\n")
    if dry_run:
        logger.info("[dry-run] Would execute on %s (%d lines, %d bytes)", host, script_lines, len(script))
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    if allow_local and should_run_locally(host, ssh_user):
        logger.debug("  Dispatching locally (no SSH) for: %s", host)
        return replace(run_local_script(script, timeout=timeout), host=host)

    if session_guard:
        script = wrap_with_session_guard(script)
        script_lines = script.count("\n")

    cmd = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options, connect_timeout)
    cmd.extend(["bash", "-s"])

    logger.debug("  SSH script -> %s (%d bytes)%s", host, len(script), f" [timeout={timeout}s]" if timeout else "")
    logger.debug("SSH command: %s", " ".join(cmd))
    logger.debug("Script: %d lines, %d bytes", script_lines, len(script))

    result = _run_subprocess(quote_list(cmd), host, "SSH script", timeout=timeout, input_data=script, quiet=quiet)
    if result.success:
        if result.stdout.strip():
            logger.debug("Remote script stdout on %s:\n%s", host, result.stdout.strip())
        if result.stderr.strip():
            logger.debug("Remote script stderr on %s:\n%s", host, result.stderr.strip())
    else:
        if result.stdout.strip():
            logger.debug("Remote script stdout on %s:\n%s", host, result.stdout.strip())
    return result


def run_remote_script_streaming(
    host: str,
    script: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    timeout: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
    session_guard: bool = False,
) -> RemoteResult:
    """Execute a script on a remote host with real-time stdout/stderr.

    Like :func:`run_remote_script` but connects the remote process's
    stdout and stderr directly to the terminal so output streams in
    real time.  Useful for long-running operations like container builds.

    When *quiet* is True, stdout/stderr are captured instead of
    streamed to the terminal.  Captured output is logged at DEBUG
    level.

    Args:
        host: Remote hostname or IP.
        script: Bash script content to execute.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the script but don't execute.
        quiet: If True, capture output instead of streaming to terminal.
        session_guard: Wrap the script so it dies with its SSH session — see
            :func:`wrap_with_session_guard`.

    Returns:
        RemoteResult with returncode (stdout/stderr are empty when
        streaming, or captured when quiet).
    """
    if dry_run:
        logger.info("[dry-run] Would execute (streaming) on %s (%d bytes)", host, len(script))
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    if session_guard:
        script = wrap_with_session_guard(script)

    cmd = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options, connect_timeout)
    cmd.extend(["bash", "-s"])

    logger.debug("  SSH script (streaming) -> %s (%d bytes)%s", host, len(script), " [timeout=%ds]" % timeout if timeout else "")

    t0 = time.monotonic()
    try:
        if quiet:
            proc = subprocess.run(
                cmd,
                input=stdin_bytes(script),
                text=False,
                timeout=timeout,
                capture_output=True,
            )
        else:
            proc = subprocess.run(
                cmd,
                input=stdin_bytes(script),
                text=False,
                timeout=timeout,
                # stdout/stderr go to terminal (no capture)
                stdout=None,
                stderr=None,
            )
        elapsed = time.monotonic() - t0
        if proc.returncode == 0:
            logger.debug("  SSH script (streaming) <- %s OK (%.1fs)", host, elapsed)
        else:
            logger.warning("  SSH script (streaming) <- %s FAILED rc=%d (%.1fs)", host, proc.returncode, elapsed)
        # Decoded, not used raw: the streaming path runs in binary mode like
        # every other subprocess here, so `quiet` callers logging or matching on
        # these would otherwise be handed bytes.
        stdout = _decode(getattr(proc, "stdout", ""))
        stderr = _decode(getattr(proc, "stderr", ""))
        if quiet and stdout:
            logger.debug("Captured stdout on %s:\n%s", host, stdout[-2000:])
        if quiet and stderr:
            logger.debug("Captured stderr on %s:\n%s", host, stderr[-2000:])
        return RemoteResult(host=host, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  SSH script (streaming) <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  SSH script (streaming) <- %s ERROR (%.1fs): %s", host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


def run_remote_command(
    host: str,
    command: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    timeout: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> RemoteResult:
    """Execute a single command on a remote host (not via bash -s).

    For simple one-liners where piping a script is overkill.

    Args:
        host: Remote hostname or IP.
        command: Command string to execute remotely.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the command but don't execute.
        quiet: If True, downgrade failure logging from WARNING to DEBUG.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    if dry_run:
        logger.info("[dry-run] Would run on %s: %s", host, command)
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    cmd = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options, connect_timeout)
    cmd.append(command)

    logger.debug("  SSH cmd -> %s: %s", host, command[:80])
    logger.debug("SSH command: %s", " ".join(cmd))

    result = _run_subprocess(cmd, host, "SSH cmd", timeout=timeout, quiet=quiet)
    if result.stdout.strip():
        logger.debug("Remote command stdout on %s:\n%s", host, result.stdout.strip())
    if result.stderr.strip():
        logger.debug("Remote command stderr on %s:\n%s", host, result.stderr.strip())
    return result


def stream_remote_logs(
    host: str,
    container_name: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    tail: int | None = 100,
    dry_run: bool = False,
    follow: bool = True,
) -> None:
    """Print ``docker logs`` output to the terminal, optionally following.

    For remote hosts, runs ``ssh <host> docker logs [-f] [--tail N] <container>``.
    For local hosts, runs ``docker logs [-f] [--tail N] <container>`` directly.

    The process's stdout/stderr are connected directly to the terminal
    (no capture), so log output flows in real time.  When *follow* is
    ``True`` a ``KeyboardInterrupt`` is caught so the user can press
    Ctrl-C to stop following without a traceback; when ``False`` the
    command dumps the requested lines and returns (``docker logs``
    semantics).

    Args:
        host: Target hostname or IP.  ``"localhost"``, ``"127.0.0.1"``,
            or ``""`` are treated as local.
        container_name: Name of the Docker container to read.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        tail: Number of existing log lines to show.  ``None`` shows the
            whole log (no ``--tail``).
        dry_run: If True, print the command that would run and return.
        follow: When ``True`` (default), keep streaming new lines
            (``-f``); when ``False``, dump and exit.
    """
    from sparkrun.orchestration.docker import docker_logs_cmd
    from sparkrun.orchestration.primitives import should_run_locally

    logs_cmd = docker_logs_cmd(container_name, follow=follow, tail=tail)

    if should_run_locally(host, ssh_user):
        cmd = logs_cmd.split()
    else:
        ssh_base = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options)
        cmd = ssh_base + logs_cmd.split()

    if dry_run:
        logger.info("[dry-run] Would stream logs: %s", " ".join(cmd))
        return

    if follow:
        logger.info("Following logs for container '%s' on %s (Ctrl-C to stop)...", container_name, host or "localhost")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("\nLog following stopped.")


def stream_container_file_logs(
    host: str,
    container_name: str,
    log_file: str = "/tmp/sparkrun_serve.log",
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    tail: int | None = 100,
    dry_run: bool = False,
    follow: bool = True,
) -> None:
    """Print a log file from inside a running container, optionally following.

    Runs ``docker exec <container> tail [-f] -n <N|+1> <file>``.  Used for
    runtimes that exec the serve command inside a long-running container
    (e.g. vLLM's ``sleep infinity`` + ``nohup serve``).

    Args:
        host: Target hostname or IP.
        container_name: Name of the Docker container.
        log_file: Path to the log file inside the container.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        tail: Number of existing log lines to show.  ``None`` shows the
            whole file (``tail -n +1``).
        dry_run: If True, print the command that would run and return.
        follow: When ``True`` (default), keep streaming new lines
            (``-f``); when ``False``, dump and exit.
    """
    # ``tail -n +1`` emits the whole file from line 1; a concrete N emits
    # the last N lines.  ``-f`` follows in either case.
    lines_arg = "+1" if tail is None else str(tail)
    tail_cmd = ["docker", "exec", container_name, "tail"]
    if follow:
        tail_cmd.append("-f")
    tail_cmd += ["-n", lines_arg, log_file]

    from sparkrun.orchestration.primitives import should_run_locally

    if should_run_locally(host, ssh_user):
        cmd = tail_cmd
    else:
        ssh_base = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options)
        cmd = ssh_base + tail_cmd

    if dry_run:
        logger.info("[dry-run] Would stream container file logs: %s", " ".join(cmd))
        return

    if follow:
        logger.info("Following serve logs in container '%s' on %s (Ctrl-C to stop)...", container_name, host or "localhost")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("\nLog following stopped.")


def start_log_capture(
    host: str,
    container_name: str,
    ssh_kwargs: dict,
    tail: int = 200,
) -> subprocess.Popen | None:
    """Start a background ``docker logs -f`` process, capturing output.

    Returns the Popen handle (or ``None`` if the process couldn't start).
    The caller should later pass this to :func:`stop_log_capture`.

    Args:
        host: Target hostname or IP.
        container_name: Name of the Docker container to follow.
        ssh_kwargs: SSH connection kwargs (ssh_user, ssh_key, ssh_options).
        tail: Number of existing log lines to include.

    Returns:
        A :class:`subprocess.Popen` handle, or ``None`` on failure.
    """
    from sparkrun.orchestration.docker import docker_logs_cmd
    from sparkrun.orchestration.primitives import should_run_locally

    logs_cmd = docker_logs_cmd(container_name, follow=True, tail=tail)

    if should_run_locally(host, ssh_kwargs.get("ssh_user")):
        cmd = logs_cmd.split()
    else:
        ssh_base = build_ssh_cmd(
            host,
            ssh_user=ssh_kwargs.get("ssh_user"),
            ssh_key=ssh_kwargs.get("ssh_key"),
            ssh_options=ssh_kwargs.get("ssh_options"),
        )
        cmd = ssh_base + logs_cmd.split()

    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        logger.debug("Failed to start background log capture for %s", container_name)
        return None


def stop_log_capture(proc: subprocess.Popen | None) -> list[str]:
    """Terminate a background log capture and return captured lines.

    Args:
        proc: The Popen handle returned by :func:`start_log_capture`,
            or ``None`` (in which case an empty list is returned).

    Returns:
        List of captured log lines.
    """
    if proc is None:
        return []
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)

    lines: list[str] = []
    if proc.stdout:
        try:
            raw = proc.stdout.read()
            lines = raw.splitlines()
        except (OSError, ValueError):
            pass
        finally:
            proc.stdout.close()
    return lines


def run_remote_scripts_parallel(
    hosts: list[str],
    script: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
    max_workers: int | None = None,
    allow_local: bool = False,
    session_guard: bool = False,
) -> list[RemoteResult]:
    """Execute the same script on multiple hosts in parallel using threads.

    Args:
        hosts: List of remote hostnames or IPs.
        script: Bash script content to execute on each host.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Per-host execution timeout in seconds.
        dry_run: If True, log the script but don't execute.
        quiet: If True, downgrade failure logging from WARNING to DEBUG.
        max_workers: Cap on concurrent SSH workers.  Defaults to
            :data:`DEFAULT_MAX_PARALLEL_SSH`; the effective pool size is
            ``min(len(hosts), max_workers)``.
        allow_local: Run the script directly (no SSH) on any host that
            :func:`should_run_locally` accepts.  **Opt-in**, because for
            some callers SSH-to-self is the point, not an accident:
            ``api.setup`` probes and meshes SSH credentials and must
            really connect.  Callers that just want the script's output
            (status discovery, hardware probes, teardown) pass ``True``
            so they work on a host without self-SSH configured.  Results
            are re-keyed to the caller's host string either way.
        session_guard: Wrap the script so it dies with its SSH session — see
            :func:`wrap_with_session_guard`.  Applies to the SSH-dispatched
            hosts only; a locally-dispatched host has no session to lose.

    Returns:
        List of RemoteResult, one per host (order not guaranteed).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Running script in parallel on %d hosts: %s", len(hosts), ", ".join(hosts))

    local_hosts = {h for h in hosts if should_run_locally(h, ssh_user)} if allow_local else set()
    if local_hosts:
        logger.debug("  Dispatching locally (no SSH) for: %s", ", ".join(sorted(local_hosts)))

    def _dispatch(host: str) -> RemoteResult:
        if host in local_hosts:
            return replace(run_local_script(script, dry_run=dry_run, timeout=timeout), host=host)
        return run_remote_script(
            host,
            script,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            timeout=timeout,
            dry_run=dry_run,
            quiet=quiet,
            session_guard=session_guard,
        )

    t0 = time.monotonic()
    results: list[RemoteResult] = []
    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(hosts), max_workers)) as executor:
        futures = {executor.submit(_dispatch, host): host for host in hosts}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if r.success)
    logger.info("  Parallel execution done: %d/%d OK (%.1fs total)", ok, len(results), elapsed)

    return results


def verify_host_paths(
    hosts: list[str],
    paths: list[str],
    ssh_kwargs: dict | None = None,
) -> dict[str, list[str]]:
    """Report host-filesystem *paths* that are absent on each of *hosts*.

    Runs a single ``test -e`` sweep per host over SSH (in parallel), echoing
    back every path that does not exist.  Returns ``{host: [missing, ...]}``
    containing **only** hosts with at least one confirmed-missing path.

    Tolerant by design — the same "safe degradation" contract as
    :meth:`Executor.query_status`: a host whose SSH probe is unreachable or
    errors is *omitted* (treated as "couldn't verify", never a false block), so
    callers should raise only on the paths this function actively reports
    missing.  This is the host-substrate implementation shared by the docker and
    local executors' :meth:`Executor.verify_mount_sources`.

    Args:
        hosts: Target hostnames/IPs.
        paths: Absolute host paths that must exist on every host.
        ssh_kwargs: Connection settings (``ssh_user`` / ``ssh_key`` /
            ``ssh_options`` / ``timeout``), as built by ``build_ssh_kwargs``.
    """
    if not hosts or not paths:
        return {}

    ssh_kwargs = ssh_kwargs or {}
    requested = list(dict.fromkeys(paths))  # de-dupe, preserve order
    # For each path, echo it verbatim (on its own line) iff it is missing.
    script = "".join("if [ ! -e %s ]; then printf '%%s\\n' %s; fi\n" % (quote(p), quote(p)) for p in requested)

    results = run_remote_scripts_parallel(
        hosts,
        script,
        ssh_user=ssh_kwargs.get("ssh_user"),
        ssh_key=ssh_kwargs.get("ssh_key"),
        ssh_options=ssh_kwargs.get("ssh_options"),
        timeout=ssh_kwargs.get("timeout", 15),
        quiet=True,
    )

    requested_set = set(requested)
    missing_by_host: dict[str, list[str]] = {}
    for r in results:
        # Unreachable / non-zero probe → couldn't verify → don't block.
        if r.returncode != 0:
            logger.debug("verify_host_paths: skipping unverifiable host %r (rc=%s)", r.host, r.returncode)
            continue
        missing = [ln for ln in (line.strip() for line in r.stdout.splitlines()) if ln in requested_set]
        if missing:
            # Preserve the requested order for a stable, readable error.
            order = {p: i for i, p in enumerate(requested)}
            missing_by_host[r.host] = sorted(set(missing), key=order.get)
    return missing_by_host


def run_remote_sudo_script(
    host: str,
    script: str,
    password: str | None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int = 60,
    dry_run: bool = False,
) -> RemoteResult:
    """Execute a script on a remote host as root via SSH.

    With a *password*, uses ``sudo -S bash -s`` and prepends the password to
    stdin so ``sudo -S`` can read it; the remaining stdin is consumed by
    ``bash -s`` as the script.  With ``password=None`` — which is what the
    sudo helpers return once NOPASSWD is confirmed on every host — uses
    ``sudo -n bash -s`` and feeds only the script, mirroring
    :func:`sparkrun.orchestration.sudo._run_local_sudo_script`.

    The password branch must not be used on a NOPASSWD host: ``sudo -S``
    would not consume the password line, leaving it in stdin for bash to
    misinterpret as a command.

    Args:
        host: Remote hostname or IP.
        script: Bash script content to execute.
        password: Sudo password for the remote user, or None for NOPASSWD.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the script but don't execute.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    if dry_run:
        logger.info("[dry-run] Would execute with sudo on %s", host)
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    cmd = build_ssh_cmd(host, ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options)
    if password is None:
        cmd.extend(["sudo", "-n", "bash", "-s"])
        full_input = script
    else:
        cmd.extend(["sudo", "-S", "bash", "-s"])
        full_input = password + "\n" + script

    logger.debug("  SSH sudo script -> %s (%d bytes)", host, len(script))

    result = _run_subprocess(cmd, host, "SSH sudo script", timeout=timeout, input_data=full_input)
    # Upgrade success log to INFO for sudo operations
    if result.success:
        logger.info("  SSH sudo script <- %s OK", host)
    return result


def detect_sudo_on_hosts(
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> set[str]:
    """Detect which hosts have passwordless sudo.

    Runs ``sudo -n true`` on each host in parallel to check whether
    the SSH user can execute sudo commands without a password prompt.

    Args:
        hosts: List of remote hostnames or IPs.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        dry_run: If True, return empty set without executing.

    Returns:
        Set of hostnames that have passwordless (NOPASSWD) sudo.
    """
    if not hosts:
        return set()

    script = 'sudo -n true 2>/dev/null && echo "SUDO_OK=1" || echo "SUDO_OK=0"'
    results = run_remote_scripts_parallel(
        hosts,
        script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=15,
        dry_run=dry_run,
    )

    nopasswd_hosts: set[str] = set()
    for r in results:
        if r.success and "SUDO_OK=1" in r.stdout:
            nopasswd_hosts.add(r.host)
            logger.debug("  %s: passwordless sudo available", r.host)
        else:
            logger.debug("  %s: passwordless sudo NOT available", r.host)

    return nopasswd_hosts


def build_ssh_opts_string(
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
) -> str:
    """Build a flat SSH options string for embedding in bash script templates.

    Unlike :func:`build_ssh_cmd`, this returns a single string of options
    (without the ``ssh`` command or target host) suitable for interpolation
    into shell scripts that construct their own ``ssh`` or ``rsync -e`` calls.

    Args:
        ssh_user: Optional SSH username (not included here — handle in the script).
        ssh_key: Optional path to SSH private key file.
        ssh_options: Additional SSH command-line options.
        connect_timeout: SSH connection timeout in seconds.

    Returns:
        Space-separated options string, e.g.
        ``"-o BatchMode=yes -o ConnectTimeout=10 -i /path/key"``.
    """
    parts = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}"]
    if ssh_key:
        parts.extend(["-i", ssh_key])
    if ssh_options:
        parts.extend(ssh_options)
    return args_list_to_shell_str(parts)


def run_pipeline_to_remote(
    host: str,
    local_cmd: str,
    remote_cmd: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    timeout: int | None = None,
    dry_run: bool = False,
) -> RemoteResult:
    """Run a shell pipeline that streams data from a local command to a remote command.

    Executes ``{local_cmd} | ssh {host} '{remote_cmd}'`` as a single shell
    pipeline via :func:`subprocess.run`.  Useful for streaming transfers like
    ``docker save img | ssh host 'docker load'``.

    Args:
        host: Remote hostname or IP.
        local_cmd: Command to run locally (producer side of pipe).
        remote_cmd: Command to run on the remote host (consumer side).
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the pipeline but don't execute.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    ssh_opts = build_ssh_opts_string(
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        connect_timeout=connect_timeout,
    )
    target = f"{ssh_user}@{host}" if ssh_user else host
    pipeline = f"{local_cmd} | ssh {ssh_opts} {quote(target)} {quote(remote_cmd)}"

    if dry_run:
        logger.info("[dry-run] Would run pipeline to %s: %s", host, pipeline)
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    logger.info("  Pipeline -> %s%s", host, f" [timeout={timeout}s]" if timeout else "")
    logger.debug("Pipeline command: %s", pipeline)

    result = _run_subprocess(pipeline, host, "Pipeline", timeout=timeout, shell=True)
    if result.success:
        logger.info("  Pipeline <- %s OK", host)
    return result


def _run_rsync_impl(
    source: str,
    dest: str,
    host: str,
    direction: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    rsync_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
) -> RemoteResult:
    """Shared rsync implementation for both push and pull directions.

    Args:
        source: Source path (with trailing ``/`` for directory contents).
        dest: Destination path.
        host: Remote hostname (for logging and result).
        direction: ``"->"`` for push, ``"<-"`` for pull (for log messages).
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        rsync_options: Override rsync flags.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the command but don't execute.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    if rsync_options is None:
        rsync_options = list(_DEFAULT_RSYNC_OPTIONS)

    ssh_opts = build_ssh_opts_string(
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        connect_timeout=connect_timeout,
    )

    cmd = ["rsync"] + rsync_options + ["-e", f"ssh {ssh_opts}", source, dest]

    if dry_run:
        logger.info("[dry-run] Would rsync %s %s: %s", direction, host, " ".join(cmd))
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    logger.info("  Rsync %s %s%s", direction, host, f" [timeout={timeout}s]" if timeout else "")
    logger.debug("Rsync command: %s", " ".join(cmd))

    result = _run_subprocess(cmd, host, "Rsync", timeout=timeout, detail_limit=RSYNC_FAILURE_DETAIL_LIMIT)
    if result.success:
        logger.info("  Rsync %s %s OK", direction, host)
        return result

    # One relaxed retry when the failure was an attribute the destination
    # refused.  NFS_SAFE_ATTR_OPTS covers the attributes we can anticipate;
    # this covers the ones we cannot, so an unfamiliar filesystem costs a
    # second pass rather than a failed launch and a config change.  Cheap:
    # rsync is incremental, so the retry re-walks the tree but re-sends
    # almost nothing.
    #
    # Deferred import — transfer.py imports RemoteResult from this module, so
    # a module-level import here would be circular.
    from sparkrun.orchestration.transfer import rsync_attribute_errors_only, rsync_has_attribute_permission_error

    if rsync_attribute_errors_only(result):
        # Every byte arrived and only attributes were refused.  The caller's
        # mapping boundary already accepts this, so a retry would re-walk the
        # whole tree to reach a state we are in — and on a model cache that
        # walk is the expensive part, not the bytes.
        return result
    if rsync_options_are_relaxed(rsync_options) or rsync_retry_disabled():
        return result
    if not rsync_has_attribute_permission_error(result):
        # A destination we cannot write to at all is not fixed by asking for
        # fewer attributes; retrying would double the wait and change nothing.
        return result

    retry_options = relax_rsync_options(rsync_options)
    logger.warning(
        "  Rsync %s %s could not set file attributes on the destination; retrying without owner/group/permission/time preservation.",
        direction,
        host,
    )
    retry_cmd = ["rsync"] + retry_options + ["-e", f"ssh {ssh_opts}", source, dest]
    logger.debug("Rsync retry command: %s", " ".join(retry_cmd))

    retry = _run_subprocess(retry_cmd, host, "Rsync", timeout=timeout, detail_limit=RSYNC_FAILURE_DETAIL_LIMIT)
    if retry.success:
        logger.info("  Rsync %s %s OK (after relaxing attribute preservation)", direction, host)
        return retry

    # The retry is strictly more permissive, so its failure is the more
    # informative one — reporting the first would point at attributes that are
    # no longer being requested.
    return retry


def guard_rsync_delete(rsync_options: list[str], local_source: str) -> list[str]:
    """Strip ``--delete*`` when *local_source* is missing or empty.

    ``--delete`` makes the destination match the source, so an empty or absent
    source turns a sync into "erase the destination".  Nothing in sparkrun ever
    means that: an empty source directory is a bug, a race, or a resource that
    was never staged — never an instruction to clear the far side.  The cost of
    being wrong is asymmetric and unrecoverable (the destination is a user's
    model or tuning cache), while the cost of the guard is a stale file that
    the next non-empty sync prunes anyway.

    Only meaningful for the push direction, where the source is local and can
    be inspected; :func:`run_rsync_from_remote` cannot use it.
    """
    if not any(o.startswith("--delete") for o in rsync_options):
        return rsync_options

    src = Path(local_source)
    try:
        populated = src.is_dir() and any(src.iterdir())
    except OSError:
        # Unreadable source: we cannot show it is safe, so we do not assume it.
        populated = False
    if populated:
        return rsync_options

    logger.warning(
        "  Refusing to rsync --delete from an empty or unreadable source (%s) — dropping --delete so the destination is not cleared.",
        local_source,
    )
    return [o for o in rsync_options if not o.startswith("--delete")]


def run_rsync(
    source_path: str,
    host: str,
    dest_path: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    rsync_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
) -> RemoteResult:
    """Rsync a local path to a remote host.

    Runs ``rsync {rsync_options} -e "ssh {opts}" source user@host:dest``.
    Default *rsync_options* are ``["-az", "--mkpath", "--partial", "--links"]``
    plus :data:`NFS_SAFE_ATTR_OPTS`, which create the destination path and
    preserve symlinks (important for HuggingFace cache layout).

    Any ``--delete`` is gated by :func:`guard_rsync_delete`.
    """
    if rsync_options is not None:
        rsync_options = guard_rsync_delete(rsync_options, source_path)
    src = source_path.rstrip("/") + "/"
    target = f"{ssh_user}@{host}:{dest_path}" if ssh_user else f"{host}:{dest_path}"
    return _run_rsync_impl(
        src,
        target,
        host,
        "->",
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        connect_timeout=connect_timeout,
        rsync_options=rsync_options,
        timeout=timeout,
        dry_run=dry_run,
    )


def run_pipeline_to_remotes_parallel(
    hosts: list[str],
    local_cmd: str,
    remote_cmd: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    timeout: int | None = None,
    dry_run: bool = False,
    max_workers: int | None = None,
) -> list[RemoteResult]:
    """Run a local-to-remote pipeline on multiple hosts in parallel.

    Wrapper over :func:`run_pipeline_to_remote` using a thread pool,
    matching the pattern of :func:`run_remote_scripts_parallel`.

    Args:
        hosts: List of remote hostnames or IPs.
        local_cmd: Command to run locally (producer side).
        remote_cmd: Command to run on each remote host (consumer side).
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        timeout: Per-host execution timeout in seconds.
        dry_run: If True, log but don't execute.
        max_workers: Cap on concurrent pipeline workers.  Defaults to
            :data:`DEFAULT_MAX_PARALLEL_SSH`; the effective pool size is
            ``min(len(hosts), max_workers)``.  Pipelines stream multi-GB
            ``docker save | ssh docker load`` data, so a tight cap also
            protects control-node I/O.

    Returns:
        List of RemoteResult, one per host.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Running pipeline in parallel to %d hosts: %s", len(hosts), ", ".join(hosts))

    t0 = time.monotonic()
    results: list[RemoteResult] = []
    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(hosts), max_workers)) as executor:
        futures = {
            executor.submit(
                run_pipeline_to_remote,
                host,
                local_cmd,
                remote_cmd,
                ssh_user=ssh_user,
                ssh_key=ssh_key,
                ssh_options=ssh_options,
                connect_timeout=connect_timeout,
                timeout=timeout,
                dry_run=dry_run,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if r.success)
    logger.info("  Parallel pipeline done: %d/%d OK (%.1fs total)", ok, len(results), elapsed)
    return results


def run_rsync_from_remote(
    host: str,
    source_path: str,
    dest_path: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    rsync_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
) -> RemoteResult:
    """Rsync a remote path to the local machine.

    Inverse of :func:`run_rsync` — pulls ``user@host:source/`` to local
    *dest_path*.
    """
    remote_src = source_path.rstrip("/") + "/"
    remote = f"{ssh_user}@{host}:{remote_src}" if ssh_user else f"{host}:{remote_src}"
    return _run_rsync_impl(
        remote,
        dest_path,
        host,
        "<-",
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        connect_timeout=connect_timeout,
        rsync_options=rsync_options,
        timeout=timeout,
        dry_run=dry_run,
    )


def run_rsync_parallel(
    source_path: str,
    hosts: list[str],
    dest_path: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    rsync_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    max_workers: int | None = None,
) -> list[RemoteResult]:
    """Rsync a local path to multiple hosts in parallel.

    Wrapper over :func:`run_rsync` using a thread pool,
    matching the pattern of :func:`run_remote_scripts_parallel`.

    Args:
        source_path: Local source directory.
        hosts: List of remote hostnames or IPs.
        dest_path: Remote destination directory.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        rsync_options: Override rsync flags.
        timeout: Per-host execution timeout in seconds.
        dry_run: If True, log but don't execute.
        max_workers: Cap on concurrent rsync workers.  Defaults to
            :data:`DEFAULT_MAX_PARALLEL_SSH`; the effective pool size is
            ``min(len(hosts), max_workers)``.

    Returns:
        List of RemoteResult, one per host.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Running rsync in parallel to %d hosts: %s", len(hosts), ", ".join(hosts))

    t0 = time.monotonic()
    results: list[RemoteResult] = []
    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(hosts), max_workers)) as executor:
        futures = {
            executor.submit(
                run_rsync,
                source_path,
                host,
                dest_path,
                ssh_user=ssh_user,
                ssh_key=ssh_key,
                ssh_options=ssh_options,
                connect_timeout=connect_timeout,
                rsync_options=rsync_options,
                timeout=timeout,
                dry_run=dry_run,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if r.success)
    logger.info("  Parallel rsync done: %d/%d OK (%.1fs total)", ok, len(results), elapsed)
    return results
