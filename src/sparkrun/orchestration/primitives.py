"""Reusable orchestration primitives for sparkrun.

Higher-level building blocks composed from the low-level modules
(ssh, docker, infiniband, scripts).  Runtimes use these to assemble
their particular launch and teardown flows.
"""

from __future__ import annotations

import logging
import subprocess

from sparkrun.core.config import SparkrunConfig, resolve_hf_cache_home
from sparkrun.utils import is_valid_ip
from sparkrun.orchestration.ssh import (
    RemoteResult,
    run_remote_command,
    run_remote_script,
    run_remote_scripts_parallel,
)
from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
)
from sparkrun.orchestration.docker import docker_stop_cmd

logger = logging.getLogger(__name__)

# Orchestration constants
MAX_PARALLEL_SSH = 20
PORT_SCAN_MAX_ATTEMPTS = 24


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
    hf_cache = resolve_hf_cache_home(cache_dir)
    volumes: dict[str, str] = {hf_cache: "/cache/huggingface"}
    if extra:
        volumes.update(extra)
    return volumes


def probe_remote_hf_cache(
    host: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int = 10,
    dry_run: bool = False,
) -> str:
    """SSH-probe *host* for its resolved HuggingFace cache directory.

    Runs ``echo "${HF_HOME:-$HOME/.cache/huggingface}"`` on the target so the
    returned path reflects the SSH login user's environment, not the control
    machine's.  Used to populate ``cache_dir`` when no cluster ``cache_dir``
    is configured and the target may have a different ``$HOME`` or ``HF_HOME``.

    The result is validated against shell-injection metacharacters before being
    returned, since callers feed it to ``shlex.quote``-aware code paths
    (volume mounts, rsync targets) that would silently break if the path
    contained ``$``, ``{``, ``}`` etc.

    Args:
        host: Remote host to probe.
        ssh_user, ssh_key, ssh_options: Standard SSH parameters.
        timeout: SSH command timeout in seconds.
        dry_run: When True, returns ``DEFAULT_HF_CACHE_DIR`` without an SSH call.

    Returns:
        Concrete absolute path on the remote host.

    Raises:
        RuntimeError: If the probe fails or returns a path with unsafe characters.
    """
    if dry_run:
        return resolve_hf_cache_home(None)

    cmd = 'echo "${HF_HOME:-$HOME/.cache/huggingface}"'
    result = run_remote_command(
        host,
        cmd,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
    )
    if not result.success or not result.stdout.strip():
        raise RuntimeError(
            "Could not resolve remote HF cache on %s (rc=%d): %s" % (host, result.returncode, result.stderr.strip() or "no output")
        )

    from sparkrun.utils.shell import assert_safe_path

    return assert_safe_path(result.stdout.strip())


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
    topology: str | None = None,
) -> ClusterCommEnv:
    """Run InfiniBand detection on *hosts* and return a :class:`ClusterCommEnv`.

    Probes IB on all hosts in parallel and builds a comm env with
    shared keys factored out and per-host interface overrides kept
    separate.
    """
    if not hosts:
        return ClusterCommEnv.empty()

    from sparkrun.orchestration.infiniband import detect_ib_for_hosts

    ib_result = detect_ib_for_hosts(
        hosts,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        topology=topology,
    )
    # ``head_host`` is accepted for backward-compat with older callers
    # but the per-host map is now the source of truth — logging is
    # handled inside ``detect_ib_for_hosts``.
    _ = head_host
    return ib_result.comm_env


def detect_infiniband_local(
    dry_run: bool = False,
) -> ClusterCommEnv:
    """Run InfiniBand detection locally and return a :class:`ClusterCommEnv`."""
    ib_script = generate_ib_detect_script()
    result = run_local_script(ib_script, dry_run=dry_run)
    if result.success:
        ib_info = parse_ib_detect_output(result.stdout)
        env = generate_nccl_env(ib_info)
        if env:
            logger.info("  InfiniBand detected locally, comm env configured")
            return ClusterCommEnv.from_shared(env)
        logger.info("  No InfiniBand detected, using default networking")
    else:
        logger.warning(
            "  InfiniBand detection failed, continuing without: %s",
            result.stderr[:100],
        )
    return ClusterCommEnv.empty()


def resolve_nccl_env(
    comm_env: ClusterCommEnv | None,
    hosts: list[str],
    head_host: str | None = None,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    topology: str | None = None,
) -> ClusterCommEnv:
    """Resolve comm env: reuse pre-detected or probe.

    Args:
        comm_env: Pre-detected :class:`ClusterCommEnv`, or ``None`` to
            trigger detection.
        hosts: Hosts to probe for InfiniBand.
        head_host: Which host's IB config to log about (defaults to
            ``hosts[0]``).  Informational only — the per-host map
            captures the full picture.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
    """
    if comm_env is not None:
        logger.info("Using pre-detected comm env (%d vars)", len(comm_env))
        return comm_env
    logger.info("Detecting InfiniBand on %d host(s)...", len(hosts))
    return detect_infiniband(
        hosts,
        head_host=head_host,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        topology=topology,
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
    from sparkrun.scripts import read_script

    script = read_script("clear_cache.sh")

    kw = ssh_kwargs or {}
    ssh_user = kw.get("ssh_user")
    local_hosts = [h for h in hosts if should_run_locally(h, ssh_user)]
    remote_hosts = [h for h in hosts if not should_run_locally(h, ssh_user)]

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
            remote_hosts,
            script,
            timeout=30,
            dry_run=dry_run,
            **kw,
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

    with ThreadPoolExecutor(max_workers=min(len(ips), MAX_PARALLEL_SSH)) as pool:
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
        raise RuntimeError("Could not determine IP from output on %s: %s" % (host, result.stdout[-200:]))
    return ip


# ---------------------------------------------------------------------------
# Container liveness
# ---------------------------------------------------------------------------

from sparkrun.orchestration.health import (  # noqa: F401, E402
    is_container_running,
    wait_for_port,
    wait_for_healthy,
)


# ---------------------------------------------------------------------------
# Port availability detection
# ---------------------------------------------------------------------------


def find_available_port(
    host: str,
    port: int,
    max_attempts: int = PORT_SCAN_MAX_ATTEMPTS,
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
        result = run_command_on_host(host, "nc -z localhost %d" % port, ssh_kwargs=ssh_kwargs, timeout=5, quiet=True)
        if not result.success:
            # nc failed → port is free
            if port != original:
                logger.info("Port %d in use on %s, using %d instead", original, host, port)
            return port
        port += 1

    logger.warning(
        "All %d ports starting from %d are in use on %s; using %d anyway",
        max_attempts,
        original,
        host,
        original,
    )
    return original


# ---------------------------------------------------------------------------
# Execution dispatch predicate
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
    import os
    from sparkrun.utils import is_local_host

    if not is_local_host(host):
        return False
    if ssh_user is None:
        return True
    return ssh_user == os.environ.get("USER", "root")


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
        script_lines = script.count("\n")
        logger.info("[dry-run] Would execute locally (%d lines, %d bytes)", script_lines, len(script))
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

    Uses :func:`should_run_locally` so that a local host with a
    different ``ssh_user`` is still reached via SSH.
    """
    kw = ssh_kwargs or {}
    if should_run_locally(host, kw.get("ssh_user")):
        return run_local_script(script, dry_run=dry_run)
    return run_remote_script(host, script, timeout=timeout, dry_run=dry_run, **kw)


def run_command_on_host(
    host: str,
    command: str,
    ssh_kwargs: dict | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> RemoteResult:
    """Run a command on a host — dispatches to local or remote execution."""
    kw = ssh_kwargs or {}
    if should_run_locally(host, kw.get("ssh_user")):
        return run_local_script("#!/bin/bash\n" + command, dry_run=dry_run)
    return run_remote_command(host, command, timeout=timeout, dry_run=dry_run, quiet=quiet, **kw)
