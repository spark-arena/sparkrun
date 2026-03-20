"""Reusable orchestration primitives for sparkrun.

Higher-level building blocks composed from the low-level modules
(ssh, docker, infiniband, scripts).  Runtimes use these to assemble
their particular launch and teardown flows.
"""

from __future__ import annotations

import logging
import subprocess
import time

from sparkrun.core.config import SparkrunConfig, resolve_cache_dir
from sparkrun.utils import is_valid_ip  # noqa: F401 — re-exported for callers
from sparkrun.utils import merge_env  # noqa: F401 — re-exported for callers
from sparkrun.orchestration.ssh import (
    RemoteResult,
    run_remote_command,
    run_remote_script,
    run_remote_scripts_parallel,
)
from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
)
from sparkrun.orchestration.docker import docker_stop_cmd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def build_ssh_kwargs(config: SparkrunConfig | None) -> dict:
    """Extract SSH connection parameters from a SparkrunConfig.

    Returns a dict suitable for ``**kwargs`` into :func:`run_remote_script`
    and friends.
    """
    if not config:
        return {}
    return {
        "ssh_user": config.ssh_user,
        "ssh_key": config.ssh_key,
        "ssh_options": config.ssh_options,
    }


def build_volumes(
        cache_dir: str | None = None,
        extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the standard volume mapping for HuggingFace cache + extras.

    Args:
        cache_dir: Host-side HF cache path (defaults to
            :data:`sparkrun.config.DEFAULT_HF_CACHE_DIR`).
        extra: Additional host→container volume mappings.

    Returns:
        Merged volume dict.
    """
    hf_cache = resolve_cache_dir(cache_dir)
    volumes: dict[str, str] = {hf_cache: "/root/.cache/huggingface"}
    if extra:
        volumes.update(extra)
    return volumes


# ---------------------------------------------------------------------------
# Resource sync helpers
# ---------------------------------------------------------------------------

def sync_resource_to_hosts(
        script: str,
        hosts: list[str],
        resource_label: str,
        ssh_user: str | None = None,
        ssh_key: str | None = None,
        dry_run: bool = False,
) -> list[str]:
    """Run a sync script on all hosts in parallel and return failures.

    Args:
        script: Pre-formatted bash script to execute on each host.
        hosts: Target hostnames or IPs.
        resource_label: Human-readable label for log messages (e.g. "Model", "Image").
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        dry_run: If True, show what would be done without executing.

    Returns:
        List of hostnames where the sync failed.
    """
    results = run_remote_scripts_parallel(
        hosts,
        script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )

    failed = [r.host for r in results if not r.success]
    if failed:
        logger.warning("%s sync failed on hosts: %s", resource_label, failed)
    else:
        logger.info("%s synced to all %d hosts", resource_label, len(hosts))

    return failed


def map_transfer_failures(
        results: list[RemoteResult],
        transfer_hosts: list[str],
        management_hosts: list[str],
) -> list[str]:
    """Map failed transfer-host results back to management hostnames.

    When fast-network IPs (InfiniBand) are used for data transfer,
    failures are reported against those IPs. This maps them back to
    the corresponding management hostnames for user-facing reporting.

    Args:
        results: Remote execution results (keyed by transfer host).
        transfer_hosts: IPs/hostnames used for the actual transfer.
        management_hosts: Corresponding management hostnames for reporting.

    Returns:
        List of management hostnames where transfer failed.
    """
    xfer_to_host = dict(zip(transfer_hosts, management_hosts))
    failed = [xfer_to_host.get(r.host, r.host) for r in results if not r.success]
    return failed


# ---------------------------------------------------------------------------
# InfiniBand detection flow
# ---------------------------------------------------------------------------

def detect_infiniband(
        hosts: list[str],
        head_host: str | None = None,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> dict[str, str]:
    """Run InfiniBand detection on *hosts* and return NCCL env vars.

    Detects IB on all hosts in parallel and uses the head node's (or
    the first host's) result to generate NCCL configuration.

    Args:
        hosts: Hosts to probe.
        head_host: Which host's IB config to use (defaults to ``hosts[0]``).
        ssh_kwargs: SSH connection parameters.
        dry_run: Log what would happen without executing.

    Returns:
        Dict of NCCL environment variables (empty if no IB found).
    """
    if not hosts:
        return {}

    target_host = head_host or hosts[0]
    kw = ssh_kwargs or {}

    logger.info("Detecting InfiniBand on %d host(s)...", len(hosts))
    ib_script = generate_ib_detect_script()
    ib_results = run_remote_scripts_parallel(
        hosts, ib_script, timeout=30, dry_run=dry_run, **kw,
    )

    nccl_env: dict[str, str] = {}
    for result in ib_results:
        if result.host == target_host and result.success:
            ib_info = parse_ib_detect_output(result.stdout)
            nccl_env = generate_nccl_env(ib_info)
            if nccl_env:
                logger.info("  InfiniBand detected on %s, NCCL configured", target_host)
            break

    if not nccl_env:
        logger.info("  No InfiniBand detected, using default networking")

    return nccl_env


def detect_infiniband_local(
        dry_run: bool = False,
) -> dict[str, str]:
    """Run InfiniBand detection locally and return NCCL env vars."""
    ib_script = generate_ib_detect_script()
    result = run_local_script(ib_script, dry_run=dry_run)
    if result.success:
        ib_info = parse_ib_detect_output(result.stdout)
        nccl_env = generate_nccl_env(ib_info)
        if nccl_env:
            logger.info("  InfiniBand detected locally, NCCL configured")
            return nccl_env
        logger.info("  No InfiniBand detected, using default networking")
    else:
        logger.warning(
            "  InfiniBand detection failed, continuing without: %s",
            result.stderr[:100],
        )
    return {}


def resolve_nccl_env(
        nccl_env: dict[str, str] | None,
        hosts: list[str],
        head_host: str | None = None,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> dict[str, str]:
    """Resolve NCCL environment: reuse pre-detected or detect.

    Encapsulates the common IB detection pattern used by cluster launch methods.

    Args:
        nccl_env: Pre-detected NCCL env, or ``None`` to trigger detection.
        hosts: Hosts to probe for InfiniBand.
        head_host: Which host's IB config to use (defaults to hosts[0]).
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.

    Returns:
        Dict of NCCL environment variables (empty if no IB found).
    """
    if nccl_env is not None:
        logger.info("Using pre-detected NCCL env (%d vars)", len(nccl_env))
        return nccl_env
    logger.info("Detecting InfiniBand on %d host(s)...", len(hosts))
    return detect_infiniband(
        hosts, head_host=head_host,
        ssh_kwargs=ssh_kwargs, dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Host preparation (pre-launch)
# ---------------------------------------------------------------------------

def try_clear_page_cache(
        hosts: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> None:
    """Best-effort drop of the Linux page cache on hosts before container launch.

    Frees cached file data so GPU-intensive inference containers have
    maximum available memory on DGX Spark's unified CPU/GPU memory.
    Uses ``sudo -n tee`` to write ``3`` to ``/proc/sys/vm/drop_caches``.
    Failures are non-fatal — a warning is logged with a hint about
    ``sparkrun setup clear-cache --save-sudo``.
    """
    from sparkrun.core.hosts import is_local_host
    from sparkrun.scripts import read_script

    script = read_script("clear_cache.sh")

    kw = ssh_kwargs or {}
    local_hosts = [h for h in hosts if is_local_host(h)]
    remote_hosts = [h for h in hosts if not is_local_host(h)]

    if local_hosts:
        result = run_local_script(script, dry_run=dry_run)
        if not result.success and not dry_run:
            logger.warning(
                "Could not clear page cache locally — run "
                "'sparkrun setup clear-cache --save-sudo' to enable "
                "passwordless cache clearing for future runs."
            )

    if remote_hosts:
        results = run_remote_scripts_parallel(
            remote_hosts, script, timeout=30, dry_run=dry_run, **kw,
        )
        failed = [r.host for r in results if not r.success]
        if failed:
            logger.warning(
                "Could not clear page cache on %d host(s) — run "
                "'sparkrun setup clear-cache --save-sudo' to enable "
                "passwordless cache clearing for future runs.",
                len(failed),
            )


# ---------------------------------------------------------------------------
# Container cleanup
# ---------------------------------------------------------------------------

def check_tcp_reachability(
        ips: list[str],
        port: int = 22,
        timeout: float = 3.0,
) -> dict[str, bool]:
    """Test TCP port reachability from the control machine to each IP.

    Uses raw TCP socket connect (no SSH, no auth needed). Runs in parallel.

    Args:
        ips: IP addresses to check.
        port: TCP port to test (default 22 for SSH).
        timeout: Connection timeout in seconds.

    Returns:
        Dict mapping IP -> bool (reachable).
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor

    if not ips:
        return {}

    def _check(ip: str) -> tuple[str, bool]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((ip, port))
                return ip, True
        except (OSError, socket.timeout):
            return ip, False

    with ThreadPoolExecutor(max_workers=min(len(ips), 20)) as pool:
        results = dict(pool.map(_check, ips))

    return results


def cleanup_containers(
        hosts: list[str],
        container_names: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> None:
    """Stop and remove named containers on every host.

    Uses local execution when a host is localhost, SSH otherwise.

    Args:
        hosts: Target hosts.
        container_names: Container names to remove on each host.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
    """
    cmds = "; ".join(docker_stop_cmd(name) for name in container_names)
    for host in hosts:
        run_command_on_host(host, cmds, ssh_kwargs=ssh_kwargs, timeout=30, dry_run=dry_run)


def cleanup_containers_local(
        container_names: list[str],
        dry_run: bool = False,
) -> None:
    """Stop and remove named containers locally."""
    cmds = "; ".join(docker_stop_cmd(name) for name in container_names)
    run_local_script("#!/bin/bash\n" + cmds, dry_run=dry_run)


# ---------------------------------------------------------------------------
# IP detection
# ---------------------------------------------------------------------------

def local_ip_for(target_host: str) -> str | None:
    """Return the local IP address on the interface that routes to *target_host*.

    Uses a UDP connect (no packets sent) to let the OS pick the right
    source address.  Falls back to ``socket.gethostname()`` on failure.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((target_host, 1))  # port is arbitrary; no traffic sent
            return s.getsockname()[0]
    except Exception:
        # Fall back to hostname if routing lookup fails (e.g. target
        # is not resolvable from the control machine itself).
        return socket.gethostname() or None


def detect_host_ip(
        host: str,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> str:
    """Detect the management IP of a remote host.

    Returns:
        The detected IPv4 address string.

    Raises:
        RuntimeError: If detection fails or result is not a valid IP.
    """
    from sparkrun.orchestration.scripts import generate_ip_detect_script

    kw = ssh_kwargs or {}
    ip_script = generate_ip_detect_script()
    result = run_remote_script(host, ip_script, timeout=15, dry_run=dry_run, **kw)

    if dry_run:
        return "<HEAD_IP>"

    if not result.success:
        raise RuntimeError("Failed to detect IP on %s: %s" % (host, result.stderr[:200]))

    ip = result.last_line.strip()
    if not is_valid_ip(ip):
        raise RuntimeError(
            "Could not determine IP from output on %s: %s" % (host, result.stdout[-200:])
        )
    return ip


# ---------------------------------------------------------------------------
# Container liveness
# ---------------------------------------------------------------------------

def is_container_running(
        host: str,
        container_name: str,
        ssh_kwargs: dict | None = None,
) -> bool:
    """Check whether a Docker container is running on a host.

    Uses local execution when *host* is localhost, SSH otherwise.

    Args:
        host: Hostname (local or remote).
        container_name: Docker container name.
        ssh_kwargs: SSH connection parameters.

    Returns:
        True if the container is currently running.
    """
    cmd = "docker inspect -f '{{.State.Running}}' %s 2>/dev/null" % container_name
    result = run_command_on_host(host, cmd, ssh_kwargs=ssh_kwargs, timeout=10)
    return result.success and "true" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Port availability detection
# ---------------------------------------------------------------------------

def find_available_port(
        host: str,
        port: int,
        max_attempts: int = 24,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> int:
    """Find an available TCP port on a host, starting from *port*.

    Uses local execution when *host* is localhost, SSH otherwise.
    Checks if *port* is free using ``nc -z``.  If occupied, increments
    and retries up to *max_attempts* times.

    Returns the first available port, or the original port if *dry_run*
    or all attempts fail (with a warning).
    """
    if dry_run:
        return port

    original = port

    for _ in range(max_attempts):
        result = run_command_on_host(host, "nc -z localhost %d" % port, ssh_kwargs=ssh_kwargs, timeout=5)
        if not result.success:
            # nc failed → port is free
            if port != original:
                logger.info("Port %d in use on %s, using %d instead", original, host, port)
            return port
        port += 1

    logger.warning(
        "All %d ports starting from %d are in use on %s; using %d anyway",
        max_attempts, original, host, original,
    )
    return original


# ---------------------------------------------------------------------------
# Port readiness polling
# ---------------------------------------------------------------------------

def wait_for_port(
        host: str,
        port: int,
        max_retries: int = 60,
        retry_interval: int = 2,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        container_name: str | None = None,
) -> bool:
    """Poll until a TCP port is listening on a host.

    Uses local execution when *host* is localhost, SSH otherwise.

    Args:
        host: Hostname (local or remote).
        port: Port to check.
        max_retries: Maximum number of retries.
        retry_interval: Seconds between retries.
        ssh_kwargs: SSH connection parameters.
        dry_run: Skip waiting in dry-run mode.
        container_name: If provided, verify the container is still
            running on each iteration.  Aborts early if the container
            has exited (e.g. crashed on startup).

    Returns:
        True if port became reachable, False if timed out or the
        container exited.
    """
    if dry_run:
        return True

    check_cmd = "nc -z localhost %d" % port
    for attempt in range(1, max_retries + 1):
        # Check container liveness before polling the port
        if container_name and attempt > 1:
            if not is_container_running(host, container_name, ssh_kwargs=ssh_kwargs):
                logger.error(
                    "  Container %s is no longer running on %s — aborting wait",
                    container_name, host,
                )
                return False

        result = run_command_on_host(host, check_cmd, ssh_kwargs=ssh_kwargs, timeout=5)
        if result.success:
            logger.info("  Port %d ready after %ds", port, attempt * retry_interval)
            return True
        if attempt % 10 == 0:
            logger.info(
                "  Still waiting for port %d (%ds elapsed)...",
                port, attempt * retry_interval,
            )
        time.sleep(retry_interval)

    return False


def wait_for_healthy(
        url: str,
        max_retries: int = 120,
        retry_interval: int = 5,
        dry_run: bool = False,
        max_consecutive_refused=2,
) -> bool:
    """Poll an HTTP endpoint until it returns 200.

    Inference servers (vLLM, SGLang, llama-server) bind their port
    immediately on startup but only return HTTP 200 on ``/v1/models``
    once the model is fully loaded and the server is ready to serve
    requests.

    Args:
        url: Full URL to poll (e.g. ``http://host:port/v1/models``).
        max_retries: Maximum number of retries.
        retry_interval: Seconds between retries.
        dry_run: Skip waiting in dry-run mode.
        max_consecutive_refused: Maximum number of consecutive refused connections before giving up.

    Returns:
        True if the endpoint returned 200, False if timed out.
    """
    if dry_run:
        return True

    import urllib.request
    import urllib.error

    consecutive_refused = 0
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info(
                        "  Health check passed after %ds",
                        attempt * retry_interval,
                    )
                    return True
            # Got a response but not 200 — server is alive, reset counter
            consecutive_refused = 0
        except urllib.error.HTTPError:
            # Server responded with an error status — alive but not ready
            consecutive_refused = 0
        except (urllib.error.URLError, OSError):
            # Connection refused / unreachable — port may have closed
            consecutive_refused += 1
            if consecutive_refused >= max_consecutive_refused:
                logger.error(
                    "  Server appears to have died (%d consecutive connection failures)",
                    consecutive_refused,
                )
                return False

        if attempt % 12 == 0:
            logger.info(
                "  Still waiting for server to be ready (%ds elapsed)...",
                attempt * retry_interval,
            )
        time.sleep(retry_interval)

    return False


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

def run_local_script(script: str, dry_run: bool = False) -> RemoteResult:
    """Execute a script locally via subprocess.

    Args:
        script: Bash script content to execute.
        dry_run: If True, log the script but don't execute.

    Returns:
        RemoteResult with host set to ``"localhost"``.
    """
    if dry_run:
        script_lines = script.count('\n')
        logger.info("[dry-run] Would execute locally (%d lines, %d bytes)",
                    script_lines, len(script))
        return RemoteResult(host="localhost", returncode=0, stdout="[dry-run]", stderr="")

    proc = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
    )
    return RemoteResult(
        host="localhost",
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


# ---------------------------------------------------------------------------
# Execution helpers (local-or-remote dispatch)
# ---------------------------------------------------------------------------

def run_script_on_host(
        host: str,
        script: str,
        ssh_kwargs: dict | None = None,
        timeout: int | None = None,
        dry_run: bool = False,
) -> RemoteResult:
    """Run a script on a host — dispatches to local or remote execution.

    If *host* is ``"localhost"``, ``"127.0.0.1"``, or empty, runs locally.
    Otherwise runs via SSH.
    """
    from sparkrun.core.hosts import is_local_host
    if is_local_host(host):
        return run_local_script(script, dry_run=dry_run)
    kw = ssh_kwargs or {}
    return run_remote_script(host, script, timeout=timeout, dry_run=dry_run, **kw)


def run_command_on_host(
        host: str,
        command: str,
        ssh_kwargs: dict | None = None,
        timeout: int | None = None,
        dry_run: bool = False,
) -> RemoteResult:
    """Run a command on a host — dispatches to local or remote execution."""
    from sparkrun.core.hosts import is_local_host
    if is_local_host(host):
        return run_local_script("#!/bin/bash\n" + command, dry_run=dry_run)
    kw = ssh_kwargs or {}
    return run_remote_command(host, command, timeout=timeout, dry_run=dry_run, **kw)
