"""Sudo fallback orchestration for sparkrun setup commands."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from sparkrun.orchestration import ssh as _ssh
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.utils.shell import b64_encode_cmd, b64_wrap_python, validate_unix_username

logger = logging.getLogger(__name__)


def _run_local_sudo_script(
    script: str,
    password: str | None = None,
    timeout: int = 300,
    dry_run: bool = False,
) -> RemoteResult:
    """Execute a script locally via ``sudo bash -s``.

    When *password* is provided, uses ``sudo -S`` (reads password from stdin).
    Otherwise uses ``sudo -n`` (non-interactive, relies on NOPASSWD sudoers).

    Args:
        script: Bash script content to execute as root.
        password: Optional sudo password.
        timeout: Execution timeout in seconds.
        dry_run: If True, log the script but don't execute.

    Returns:
        RemoteResult with host set to ``"localhost"``.
    """
    if dry_run:
        logger.info("[dry-run] Would execute locally with sudo (%d bytes)", len(script))
        return RemoteResult(host="localhost", returncode=0, stdout="[dry-run]", stderr="")

    if password is not None:
        cmd = ["sudo", "-S", "bash", "-s"]
        full_input = password + "\n" + script
    else:
        cmd = ["sudo", "-n", "bash", "-s"]
        full_input = script

    try:
        proc = subprocess.run(
            cmd,
            input=full_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return RemoteResult(
            host="localhost",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired:
        return RemoteResult(
            host="localhost",
            returncode=1,
            stdout="",
            stderr="Timeout after %ds" % timeout,
        )


def run_sudo_script_on_host(
    host: str,
    script: str,
    password: str | None,
    ssh_kwargs: dict | None = None,
    timeout: int = 300,
    dry_run: bool = False,
) -> RemoteResult:
    """Run a sudo script on a single host, dispatching local vs SSH.

    For local hosts (localhost, 127.0.0.1), executes via ``sudo bash -s``
    directly.  For remote hosts, delegates to
    :func:`~sparkrun.orchestration.ssh.run_remote_sudo_script`.

    Both paths accept ``password=None`` (NOPASSWD) and run non-interactively.

    Args:
        host: Target hostname or IP.
        script: Bash script content to execute as root.
        password: Sudo password (fed via stdin), or None for NOPASSWD hosts.
        ssh_kwargs: SSH connection parameters (forwarded for remote hosts).
        timeout: Execution timeout in seconds.
        dry_run: If True, skip actual execution.

    Returns:
        RemoteResult with the original host label preserved.
    """
    from sparkrun.orchestration.primitives import should_run_locally

    kw = ssh_kwargs or {}
    if should_run_locally(host, kw.get("ssh_user")):
        r = _run_local_sudo_script(script, password=password, timeout=timeout, dry_run=dry_run)
        return RemoteResult(host=host, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    return _ssh.run_remote_sudo_script(host, script, password, timeout=timeout, dry_run=dry_run, **kw)


def run_indirect_sudo_script(
    host: str,
    script: str,
    sudo_user: str,
    sudo_password: str,
    ssh_kwargs: dict | None = None,
    timeout: int = 300,
    dry_run: bool = False,
) -> RemoteResult:
    """Run a sudo script on a host via an intermediate SSH user.

    SSHs as the cluster user (from *ssh_kwargs*), then uses ``su`` on the
    remote side to switch to *sudo_user* who has sudo access.  A small
    Python helper using ``pty.fork()`` handles the non-interactive ``su``
    password prompt, then pipes the script through ``sudo -S bash -s``.

    Use this when the SSH user lacks sudo access but a different user on
    the remote host does.

    Args:
        host: Remote hostname or IP.
        script: Bash script content to execute as root.
        sudo_user: The user with sudo access on the remote host.
        sudo_password: Password for *sudo_user* (used for both ``su`` and ``sudo``).
        ssh_kwargs: SSH connection parameters (SSH user is the cluster user).
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log without executing.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    validate_unix_username(sudo_user)

    if dry_run:
        logger.info("[dry-run] Would execute indirect sudo on %s (via su %s)", host, sudo_user)
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    # Build a Python wrapper that runs on the remote host.
    # It uses pty.fork() to feed the su password, then pipes the
    # script through sudo -S bash -s.
    #
    # The password and script are base64-encoded into the wrapper itself
    # (not read from stdin) because stdin is used as the pipe to python3.

    b64_password = b64_encode_cmd(sudo_password)
    b64_script = b64_encode_cmd(script)

    wrapper = (
        "import base64, os, pty, select, sys, time\n"
        "try:\n"
        "    password = base64.b64decode('%s').decode('utf-8')\n"
        "    script = base64.b64decode('%s').decode('utf-8')\n"
        "    sudo_user = %r\n"
        "    pid, fd = pty.fork()\n"
        "    if pid == 0:\n"
        "        os.execlp('su', 'su', '-', sudo_user, '-c', 'sudo -S bash -s')\n"
        "    else:\n"
        "        buf = b''\n"
        "        deadline = time.time() + 10\n"
        "        fed_su = False\n"
        "        while time.time() < deadline:\n"
        "            r, _, _ = select.select([fd], [], [], 0.5)\n"
        "            if r:\n"
        "                try:\n"
        "                    data = os.read(fd, 4096)\n"
        "                except OSError:\n"
        "                    break\n"
        "                buf += data\n"
        "                low = buf.lower()\n"
        "                if not fed_su and (b'password' in low or b'passwort' in low):\n"
        "                    os.write(fd, (password + '\\n').encode())\n"
        "                    fed_su = True\n"
        "                    buf = b''\n"
        "                elif fed_su and (b'password' in low or b'passwort' in low):\n"
        "                    os.write(fd, (password + '\\n').encode())\n"
        "                    time.sleep(0.2)\n"
        "                    os.write(fd, script.encode())\n"
        "                    break\n"
        "        else:\n"
        "            os.close(fd)\n"
        "            sys.stderr.write('Timeout waiting for su/sudo prompts\\n')\n"
        "            sys.exit(1)\n"
        "        # Drain remaining output\n"
        "        out = b''\n"
        "        while True:\n"
        "            r, _, _ = select.select([fd], [], [], 2)\n"
        "            if not r:\n"
        "                break\n"
        "            try:\n"
        "                chunk = os.read(fd, 4096)\n"
        "                if not chunk:\n"
        "                    break\n"
        "                out += chunk\n"
        "            except OSError:\n"
        "                break\n"
        "        os.close(fd)\n"
        "        _, status = os.waitpid(pid, 0)\n"
        "        sys.stdout.buffer.write(out)\n"
        "        sys.exit(os.WEXITSTATUS(status))\n"
        "except Exception as e:\n"
        "    sys.stderr.write('indirect-sudo wrapper error: ' + str(e) + '\\n')\n"
        "    sys.exit(1)\n"
    ) % (b64_password, b64_script, sudo_user)

    # Deliver the wrapper via a base64 pipeline (avoids shell escaping issues
    # that occur with python3 -c when SSH joins args for the remote shell).
    kw = ssh_kwargs or {}
    cmd = _ssh.build_ssh_cmd(host, **{k: v for k, v in kw.items() if k in ("ssh_user", "ssh_key", "ssh_options")})
    cmd.append(b64_wrap_python(wrapper, quoted=False))

    logger.debug("  SSH indirect sudo -> %s (su %s, %d bytes)", host, sudo_user, len(script))

    result = _ssh._run_subprocess(cmd, host, "SSH indirect sudo", timeout=timeout)
    if result.success:
        logger.info("  SSH indirect sudo <- %s OK", host)
    return result


def dispatch_sudo_script(
    host: str,
    script: str,
    sudo_password: str | None,
    ssh_kwargs: dict | None = None,
    indirect_sudo_user: str | None = None,
    timeout: int = 300,
    dry_run: bool = False,
) -> RemoteResult:
    """Run a sudo script on a host, dispatching direct vs indirect.

    When *indirect_sudo_user* is set, uses :func:`run_indirect_sudo_script`
    (SSH as cluster user, ``su`` to sudo user).  Otherwise uses
    :func:`run_sudo_script_on_host` directly.

    Args:
        host: Target hostname or IP.
        script: Bash script content to execute as root.
        sudo_password: Sudo password.
        ssh_kwargs: SSH connection parameters.
        indirect_sudo_user: If set, SSH as cluster user and ``su`` to this
            user for sudo access.
        timeout: Execution timeout in seconds.
        dry_run: If True, skip actual execution.

    Returns:
        RemoteResult from the executed script.
    """
    if indirect_sudo_user:
        return run_indirect_sudo_script(
            host,
            script,
            sudo_user=indirect_sudo_user,
            sudo_password=sudo_password,
            ssh_kwargs=ssh_kwargs,
            timeout=timeout,
            dry_run=dry_run,
        )
    return run_sudo_script_on_host(
        host,
        script,
        sudo_password,
        ssh_kwargs=ssh_kwargs,
        timeout=timeout,
        dry_run=dry_run,
    )


def run_with_sudo_fallback(
    host_list: list[str],
    script: str,
    fallback_script: str,
    ssh_kwargs: dict,
    dry_run: bool = False,
    sudo_password: str | None = None,
    timeout: int = 300,
) -> tuple[dict[str, object], list[str]]:
    """Run script with sudo fallback. Returns (result_map, still_failed_hosts).

    Local hosts (localhost, 127.0.0.1) are executed directly via
    ``sudo bash -s`` without SSH.  Remote hosts use SSH as before.

    Steps:
    1. Try non-interactive sudo on all hosts in parallel.
    2. Partition into successes/failures.
    3. For failures, fall back to password-based sudo using provided password.
    4. Return result map and list of hosts that still failed after fallback.

    The CLI handler is responsible for prompting for passwords and handling
    per-host retries on still_failed_hosts.

    Args:
        host_list: Hosts to run on.
        script: Initial script to run (non-interactive sudo).
        fallback_script: Script for password-based sudo fallback.
        ssh_kwargs: SSH connection parameters.
        dry_run: If True, skip actual execution.
        sudo_password: Optional sudo password for fallback.
        timeout: Timeout in seconds for each operation.

    Returns:
        Tuple of (result_map, still_failed_hosts) where result_map is
        {host: SSHResult} and still_failed_hosts is a list of hosts
        that failed even after password-based sudo.
    """
    from sparkrun.orchestration.primitives import should_run_locally

    # Separate local and remote hosts
    ssh_user = ssh_kwargs.get("ssh_user")
    local_hosts = [h for h in host_list if should_run_locally(h, ssh_user)]
    remote_hosts = [h for h in host_list if not should_run_locally(h, ssh_user)]

    result_map: dict[str, object] = {}
    failed_hosts: list[str] = []

    # Step 1a: Run locally for local hosts (sudo -n, non-interactive)
    for h in local_hosts:
        r = _run_local_sudo_script(script, password=None, timeout=timeout, dry_run=dry_run)
        # Preserve original host label in the result
        r = RemoteResult(host=h, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        # Always record the result (even on failure) so callers keep the script's
        # output for diagnostics; a failure that isn't a sudo issue (and so has no
        # password retry) would otherwise vanish. Success/failure is read via
        # ``.success``; ``still_failed`` is derived from ``failed_hosts`` below.
        result_map[h] = r
        if not r.success:
            failed_hosts.append(h)

    # Step 1b: Try non-interactive sudo on remote hosts in parallel.
    # Use quiet=True to suppress scary "FAILED" warnings — these are
    # expected when hosts don't have NOPASSWD and will be retried with
    # a password in Step 2.
    if remote_hosts:
        parallel_results = _ssh.run_remote_scripts_parallel(
            remote_hosts,
            script,
            timeout=timeout,
            dry_run=dry_run,
            quiet=True,
            **ssh_kwargs,
        )
        for r in parallel_results:
            # Record every result (see Step 1a) so a non-sudo failure's output
            # survives even when there's no password to trigger a fallback.
            result_map[r.host] = r
            if not r.success:
                failed_hosts.append(r.host)

    if failed_hosts and not dry_run:
        logger.info("Sudo password required on %d host(s).", len(failed_hosts))

    # Step 2: For failed hosts, fall back to password-based sudo
    if failed_hosts and not dry_run and sudo_password is not None:
        for h in failed_hosts:
            if should_run_locally(h, ssh_user):
                r = _run_local_sudo_script(
                    fallback_script,
                    password=sudo_password,
                    timeout=timeout,
                    dry_run=dry_run,
                )
                r = RemoteResult(host=h, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
            else:
                r = _ssh.run_remote_sudo_script(
                    h,
                    fallback_script,
                    sudo_password,
                    timeout=timeout,
                    dry_run=dry_run,
                    **ssh_kwargs,
                )
            result_map[h] = r

    # Return results and hosts that still failed after fallback
    still_failed = [h for h in failed_hosts if h not in result_map or not result_map[h].success]
    return result_map, still_failed


# ---------------------------------------------------------------------------
# Directory-ownership repair
# ---------------------------------------------------------------------------

# Best-effort ownership fix for a sparkrun-managed cache directory.
#
# The recurring way one of these ends up root-owned is Docker: a bind mount
# whose source does not exist on the host is *materialized by the daemon*, and
# it creates the whole missing path chain owned by root.  From then on the SSH
# user cannot write into it, so every later rsync into that tree fails with
# EACCES — the transfer never happens, and no rsync flag can fix it because the
# problem is the directory, not the attributes being requested.
#
# `sudo -n` only: this runs on the hot launch path, so it must never block on a
# password prompt.  A host without NOPASSWD sudo simply reports failure and the
# caller carries on — the point is to self-heal where we can, not to gate a
# launch on sudo.
_FIX_OWNERSHIP_SCRIPT = """set -euo pipefail
TARGET="{path}"
[ -d "$TARGET" ] || exit 0
OWNER=$(stat -c "%U" "$TARGET" 2>/dev/null || echo "")
ME=$(id -un)
[ "$OWNER" = "$ME" ] && exit 0
sudo -n /usr/bin/chown -R "$ME" "$TARGET" 2>/dev/null
"""


def ensure_remote_dir_ownership(
    dir_path: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
    resource_label: str = "cache",
    session: Any | None = None,
) -> list[str]:
    """Best-effort ``chown`` of *dir_path* to the SSH user on each host.

    A no-op where the directory is absent or already owned by the SSH user, so
    it is cheap to call unconditionally.  Never raises and never prompts.

    Returns:
        Hosts where ownership could not be fixed (empty = nothing to do, or
        everything succeeded).  Callers treat a non-empty list as a warning,
        not a failure: the repair is opportunistic, and the operation it is
        clearing the way for will report its own error if it still cannot
        proceed.
    """
    if not hosts:
        return []

    from sparkrun.utils.shell import safe_remote_path

    # safe_remote_path, not assert_safe_path: a cache dir may be spelled "~/..."
    # and tilde expansion does not happen inside the double quotes this script
    # interpolates into, so it has to become "$HOME/...".
    script = _FIX_OWNERSHIP_SCRIPT.format(path=safe_remote_path(dir_path))
    if session is None:
        results = _ssh.run_remote_scripts_parallel(
            hosts,
            script,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            timeout=60,
            dry_run=dry_run,
        )
        failed = [result.host for result in results if not result.success]
    elif dry_run:
        failed = []
    else:
        failed = []
        for host in hosts:
            try:
                result = session.execute(host, ["bash", "-c", script], timeout=60)
            except Exception:
                logger.debug("Could not repair %s ownership on %s", resource_label, host, exc_info=True)
                failed.append(host)
            else:
                if result.returncode != 0:
                    failed.append(host)
    if failed:
        logger.warning(
            "Could not fix %s ownership on %d host(s) (%s) — a root-owned "
            "directory blocks writes by the SSH user.  Run 'sparkrun setup "
            "fix-permissions --save-sudo' to allow passwordless chown, or "
            "chown %s manually.",
            resource_label,
            len(failed),
            ", ".join(failed),
            dir_path,
        )
    return failed
