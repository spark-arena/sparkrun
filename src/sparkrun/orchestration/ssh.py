"""SSH remote execution via bash -s stdin piping.

All remote operations in sparkrun are executed by generating scripts
as Python strings and piping them to `ssh <host> bash -s` via stdin.
No files are ever copied to remote hosts.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    script_lines = script.count('\n')
    if dry_run:
        logger.info("[dry-run] Would execute on %s (%d lines, %d bytes)",
                    host, script_lines, len(script))
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    cmd = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options, connect_timeout)
    cmd.extend(["bash", "-s"])

    logger.debug("  SSH script -> %s (%d bytes)%s",
                 host, len(script),
                 f" [timeout={timeout}s]" if timeout else "")
    logger.debug("SSH command: %s", " ".join(cmd))
    logger.debug("Script: %d lines, %d bytes", script_lines, len(script))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if result.success:
            logger.debug("  SSH script <- %s OK (%.1fs)", host, elapsed)
            if proc.stdout.strip():
                logger.debug("Remote script stdout on %s:\n%s", host, proc.stdout.strip())
            if proc.stderr.strip():
                logger.debug("Remote script stderr on %s:\n%s", host, proc.stderr.strip())
        else:
            logger.warning(
                "  SSH script <- %s FAILED rc=%d (%.1fs): %s",
                host,
                proc.returncode,
                elapsed,
                proc.stderr.strip()[:200],
            )
            if proc.stdout.strip():
                logger.debug("Remote script stdout on %s:\n%s", host, proc.stdout.strip())
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  SSH script <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  SSH script <- %s ERROR (%.1fs): %s", host, elapsed, e)
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

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        logger.debug("  SSH cmd <- %s rc=%d (%.1fs)", host, proc.returncode, elapsed)
        if proc.stdout.strip():
            logger.debug("Remote command stdout on %s:\n%s", host, proc.stdout.strip())
        if proc.stderr.strip():
            logger.debug("Remote command stderr on %s:\n%s", host, proc.stderr.strip())
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  SSH cmd <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  SSH cmd <- %s ERROR (%.1fs): %s", host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


def stream_remote_logs(
        host: str,
        container_name: str,
        ssh_user: str | None = None,
        ssh_key: str | None = None,
        ssh_options: list[str] | None = None,
        tail: int = 100,
        dry_run: bool = False,
) -> None:
    """Stream ``docker logs -f`` output to the terminal.

    For remote hosts, runs ``ssh <host> docker logs -f --tail N <container>``.
    For local hosts, runs ``docker logs -f --tail N <container>`` directly.

    The process's stdout/stderr are connected directly to the terminal
    (no capture), so log output flows in real time.  A ``KeyboardInterrupt``
    is caught so the user can press Ctrl-C to stop following without a
    traceback.

    Args:
        host: Target hostname or IP.  ``"localhost"``, ``"127.0.0.1"``,
            or ``""`` are treated as local.
        container_name: Name of the Docker container to follow.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        tail: Number of existing log lines to show before following.
        dry_run: If True, print the command that would run and return.
    """
    from sparkrun.orchestration.docker import docker_logs_cmd
    from sparkrun.core.hosts import is_local_host

    logs_cmd = docker_logs_cmd(container_name, follow=True, tail=tail)

    if is_local := is_local_host(host):
        cmd = logs_cmd.split()
    else:
        ssh_base = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options)
        cmd = ssh_base + logs_cmd.split()

    if dry_run:
        logger.info("[dry-run] Would stream logs: %s", " ".join(cmd))
        return

    logger.info("Following logs for container '%s' on %s (Ctrl-C to stop)...",
                container_name, host or "localhost")
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
        tail: int = 100,
        dry_run: bool = False,
) -> None:
    """Stream a log file from inside a running container.

    Runs ``docker exec <container> tail -f --lines <N> <file>``.
    Used for runtimes that exec the serve command inside a long-running
    container (e.g. vLLM's ``sleep infinity`` + ``nohup serve``).

    Args:
        host: Target hostname or IP.
        container_name: Name of the Docker container.
        log_file: Path to the log file inside the container.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        tail: Number of existing log lines to show before following.
        dry_run: If True, print the command that would run and return.
    """
    tail_cmd = [
        "docker", "exec", container_name,
        "tail", "-f", "--lines", str(tail), log_file,
    ]

    from sparkrun.core.hosts import is_local_host

    if is_local := is_local_host(host):
        cmd = tail_cmd
    else:
        ssh_base = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options)
        cmd = ssh_base + tail_cmd

    if dry_run:
        logger.info("[dry-run] Would stream container file logs: %s", " ".join(cmd))
        return

    logger.info("Following serve logs in container '%s' on %s (Ctrl-C to stop)...",
                container_name, host or "localhost")
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
    from sparkrun.core.hosts import is_local_host

    logs_cmd = docker_logs_cmd(container_name, follow=True, tail=tail)

    if is_local_host(host):
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

    Returns:
        List of RemoteResult, one per host (order not guaranteed).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Running script in parallel on %d hosts: %s",
                len(hosts), ", ".join(hosts))

    t0 = time.monotonic()
    results: list[RemoteResult] = []
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {
            executor.submit(
                run_remote_script,
                host,
                script,
                ssh_user=ssh_user,
                ssh_key=ssh_key,
                ssh_options=ssh_options,
                timeout=timeout,
                dry_run=dry_run,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if r.success)
    logger.info("  Parallel execution done: %d/%d OK (%.1fs total)",
                ok, len(results), elapsed)

    return results


def run_remote_sudo_script(
        host: str,
        script: str,
        password: str,
        ssh_user: str | None = None,
        ssh_key: str | None = None,
        ssh_options: list[str] | None = None,
        timeout: int = 60,
        dry_run: bool = False,
) -> RemoteResult:
    """Execute a script on a remote host via ``sudo -S bash -s``.

    Prepends the sudo password to stdin so ``sudo -S`` can read it,
    then the remaining stdin is consumed by ``bash -s`` as the script.

    Only use this for hosts that do NOT have passwordless sudo.
    For NOPASSWD hosts, use :func:`run_remote_script` instead — ``sudo -S``
    on a NOPASSWD host would leave the password line in stdin for bash
    to misinterpret as a command.

    Args:
        host: Remote hostname or IP.
        script: Bash script content to execute.
        password: Sudo password for the remote user.
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
    cmd.extend(["sudo", "-S", "bash", "-s"])
    full_input = password + "\n" + script

    logger.debug("  SSH sudo script -> %s (%d bytes)", host, len(script))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, input=full_input, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host, returncode=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr,
        )
        if result.success:
            logger.info("  SSH sudo script <- %s OK (%.1fs)", host, elapsed)
        else:
            # Filter out the sudo password prompt from stderr for cleaner logging
            stderr_clean = proc.stderr.replace("[sudo] password for %s: " % (ssh_user or ""), "").strip()
            logger.warning("  SSH sudo script <- %s FAILED rc=%d (%.1fs): %s",
                           host, proc.returncode, elapsed, stderr_clean[:200])
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  SSH sudo script <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  SSH sudo script <- %s ERROR (%.1fs): %s", host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


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
        hosts, script,
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        timeout=15, dry_run=dry_run,
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
    return " ".join(parts)


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
        ssh_user=ssh_user, ssh_key=ssh_key,
        ssh_options=ssh_options, connect_timeout=connect_timeout,
    )
    target = f"{ssh_user}@{host}" if ssh_user else host
    pipeline = f"{local_cmd} | ssh {ssh_opts} {target} '{remote_cmd}'"

    if dry_run:
        logger.info("[dry-run] Would run pipeline to %s: %s", host, pipeline)
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    logger.info("  Pipeline -> %s%s", host, f" [timeout={timeout}s]" if timeout else "")
    logger.debug("Pipeline command: %s", pipeline)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            pipeline,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host, returncode=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr,
        )
        if result.success:
            logger.info("  Pipeline <- %s OK (%.1fs)", host, elapsed)
        else:
            logger.warning(
                "  Pipeline <- %s FAILED rc=%d (%.1fs): %s",
                host, proc.returncode, elapsed, proc.stderr.strip()[:200],
            )
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  Pipeline <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  Pipeline <- %s ERROR (%.1fs): %s", host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


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
    which create the destination path and preserve symlinks (important for
    HuggingFace cache layout).

    Args:
        source_path: Local source directory (trailing ``/`` is appended if missing).
        host: Remote hostname or IP.
        dest_path: Remote destination directory.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        rsync_options: Override rsync flags (default ``["-az", "--mkpath", "--partial", "--links"]``).
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the command but don't execute.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    if rsync_options is None:
        rsync_options = ["-az", "--mkpath", "--partial", "--links"]

    ssh_opts = build_ssh_opts_string(
        ssh_user=ssh_user, ssh_key=ssh_key,
        ssh_options=ssh_options, connect_timeout=connect_timeout,
    )

    # Ensure trailing slash so rsync copies directory contents
    src = source_path.rstrip("/") + "/"
    target = f"{ssh_user}@{host}:{dest_path}" if ssh_user else f"{host}:{dest_path}"

    cmd = ["rsync"] + rsync_options + ["-e", f"ssh {ssh_opts}", src, target]

    if dry_run:
        logger.info("[dry-run] Would rsync to %s: %s", host, " ".join(cmd))
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    logger.info("  Rsync -> %s%s", host, f" [timeout={timeout}s]" if timeout else "")
    logger.debug("Rsync command: %s", " ".join(cmd))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host, returncode=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr,
        )
        if result.success:
            logger.info("  Rsync <- %s OK (%.1fs)", host, elapsed)
        else:
            logger.warning(
                "  Rsync <- %s FAILED rc=%d (%.1fs): %s",
                host, proc.returncode, elapsed, proc.stderr.strip()[:200],
            )
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  Rsync <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  Rsync <- %s ERROR (%.1fs): %s", host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


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

    Returns:
        List of RemoteResult, one per host.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Running pipeline in parallel to %d hosts: %s",
                len(hosts), ", ".join(hosts))

    t0 = time.monotonic()
    results: list[RemoteResult] = []
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {
            executor.submit(
                run_pipeline_to_remote,
                host, local_cmd, remote_cmd,
                ssh_user=ssh_user, ssh_key=ssh_key,
                ssh_options=ssh_options, connect_timeout=connect_timeout,
                timeout=timeout, dry_run=dry_run,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if r.success)
    logger.info("  Parallel pipeline done: %d/%d OK (%.1fs total)",
                ok, len(results), elapsed)
    return results


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

    Returns:
        List of RemoteResult, one per host.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Running rsync in parallel to %d hosts: %s",
                len(hosts), ", ".join(hosts))

    t0 = time.monotonic()
    results: list[RemoteResult] = []
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {
            executor.submit(
                run_rsync,
                source_path, host, dest_path,
                ssh_user=ssh_user, ssh_key=ssh_key,
                ssh_options=ssh_options, connect_timeout=connect_timeout,
                rsync_options=rsync_options,
                timeout=timeout, dry_run=dry_run,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if r.success)
    logger.info("  Parallel rsync done: %d/%d OK (%.1fs total)",
                ok, len(results), elapsed)
    return results
