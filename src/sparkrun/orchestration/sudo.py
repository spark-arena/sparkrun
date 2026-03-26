"""Sudo fallback orchestration for sparkrun setup commands."""

from __future__ import annotations

import logging
import subprocess

from sparkrun.orchestration import ssh as _ssh
from sparkrun.orchestration.ssh import RemoteResult

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
    password: str,
    ssh_kwargs: dict | None = None,
    timeout: int = 300,
    dry_run: bool = False,
) -> RemoteResult:
    """Run a sudo script on a single host, dispatching local vs SSH.

    For local hosts (localhost, 127.0.0.1), executes via ``sudo -S bash -s``
    directly.  For remote hosts, delegates to
    :func:`~sparkrun.orchestration.ssh.run_remote_sudo_script`.

    Args:
        host: Target hostname or IP.
        script: Bash script content to execute as root.
        password: Sudo password (fed via stdin).
        ssh_kwargs: SSH connection parameters (forwarded for remote hosts).
        timeout: Execution timeout in seconds.
        dry_run: If True, skip actual execution.

    Returns:
        RemoteResult with the original host label preserved.
    """
    from sparkrun.utils import is_local_host

    if is_local_host(host):
        r = _run_local_sudo_script(script, password=password, timeout=timeout, dry_run=dry_run)
        return RemoteResult(host=host, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    kw = ssh_kwargs or {}
    return _ssh.run_remote_sudo_script(host, script, password, timeout=timeout, dry_run=dry_run, **kw)


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
    from sparkrun.utils import is_local_host

    # Separate local and remote hosts
    local_hosts = [h for h in host_list if is_local_host(h)]
    remote_hosts = [h for h in host_list if not is_local_host(h)]

    result_map: dict[str, object] = {}
    failed_hosts: list[str] = []

    # Step 1a: Run locally for local hosts (sudo -n, non-interactive)
    for h in local_hosts:
        r = _run_local_sudo_script(script, password=None, timeout=timeout, dry_run=dry_run)
        # Preserve original host label in the result
        r = RemoteResult(host=h, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        if r.success:
            result_map[h] = r
        else:
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
            if r.success:
                result_map[r.host] = r
            else:
                failed_hosts.append(r.host)

    if failed_hosts and not dry_run:
        logger.info("Sudo password required on %d host(s).", len(failed_hosts))

    # Step 2: For failed hosts, fall back to password-based sudo
    if failed_hosts and not dry_run and sudo_password is not None:
        for h in failed_hosts:
            if is_local_host(h):
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
