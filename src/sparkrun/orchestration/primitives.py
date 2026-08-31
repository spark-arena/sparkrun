"""Reusable orchestration primitives for sparkrun.

Higher-level building blocks composed from the low-level modules
(ssh, docker, infiniband, scripts).  Runtimes use these to assemble
their particular launch and teardown flows.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sparkrun.core.config import DEFAULT_CACHE_DIR, SparkrunConfig, resolve_hf_cache_home
from sparkrun.utils import is_valid_ip
from sparkrun.orchestration.ssh import (
    DEFAULT_MAX_PARALLEL_SSH,
    RemoteResult,
    resolve_parallel_cap,
    run_local_script,
    run_local_script_streaming,
    run_remote_command,
    run_remote_script,
    run_remote_script_streaming,
    run_remote_scripts_parallel,
    should_run_locally,
)
from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
)

if TYPE_CHECKING:
    from sparkrun.orchestration.executors._base import Executor

logger = logging.getLogger(__name__)

# Orchestration constants.
# ``MAX_PARALLEL_SSH`` is retained here for backward-compatible imports;
# the canonical default now lives in ``orchestration.ssh`` so the fan-out
# helpers and ``SparkrunConfig.max_parallel_ssh`` share one source of truth.
MAX_PARALLEL_SSH = DEFAULT_MAX_PARALLEL_SSH
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


def resolved_model_volume(recipe) -> dict[str, str]:
    """Identity bind-mount for a recipe's pre-placed on-disk model weights.

    Two surfaces feed this: the ``cluster_config.resolved_model_path`` escape
    hatch, and — as user-facing sugar — an absolute path in the recipe's
    ``model:`` field (:func:`sparkrun.core.recipe.is_local_model_path`).  Either
    way the weights directory is already present on every node (e.g. a shared
    NFS mount) and is mounted into the container at the *same* path so the
    serving runtime can read it directly (the serve argument points at this path
    already).  Returns an empty dict when neither is configured.
    """
    path = getattr(getattr(recipe, "cluster_config", None), "resolved_model_path", None)
    if not path:
        from sparkrun.core.recipe import is_local_model_path

        model = getattr(recipe, "model", None)
        if is_local_model_path(model):
            path = model
    if not path or not isinstance(path, str):
        return {}
    from sparkrun.utils.shell import assert_safe_mount_source

    assert_safe_mount_source(path)
    return {path: path}


def probe_remote_path(
    host: str,
    expr: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int = 10,
) -> str:
    """SSH-probe *host*, echoing shell expression *expr*, and validate the result.

    The generic form behind :func:`probe_remote_hf_cache`: any path sparkrun
    needs on a target must be resolved in the *login user's* environment, not
    the control machine's — ``$HOME`` differs, and on a cross-user or remote
    launch the control machine's answer is simply wrong.

    *expr* is embedded in a double-quoted ``echo`` so parameter expansion
    happens on the host.  It is caller-supplied and never user input; the
    *result* is validated against shell metacharacters because callers feed it
    to ``shlex.quote``-aware code (volume mounts, generated scripts) that would
    silently break on ``$``/``{``/``}``.
    """
    from sparkrun.utils.shell import assert_safe_path

    result = run_remote_command(
        host,
        'echo "%s"' % expr,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
    )
    if not result.success or not result.stdout.strip():
        raise RuntimeError(
            "Could not resolve remote path on %s (rc=%d): %s" % (host, result.returncode, result.stderr.strip() or "no output")
        )
    return assert_safe_path(result.stdout.strip())


def probe_remote_sparkrun_cache(
    host: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int = 10,
    dry_run: bool = False,
) -> str:
    """SSH-probe *host* for its sparkrun cache directory (``~/.cache/sparkrun``).

    The runtime-cache peer of :func:`probe_remote_hf_cache`.  Honors
    ``SPARKRUN_CACHE_DIR`` / ``XDG_CACHE_HOME`` on the target so a host that
    relocates its caches is respected.
    """
    if dry_run:
        return str(DEFAULT_CACHE_DIR)

    return probe_remote_path(
        host,
        "${SPARKRUN_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/sparkrun}",
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
    )


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

    The script runs under the session guard
    (:func:`~sparkrun.orchestration.ssh.wrap_with_session_guard`): every caller
    here is a model download or image pull, which must not keep running on the
    hosts after the launch that started it was killed.

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
        session_guard=True,
    )

    failed = [r.host for r in results if not r.success]
    if failed:
        logger.warning("%s sync failed on hosts: %s", resource_label, failed)
    else:
        logger.info("%s synced to all %d hosts", resource_label, len(hosts))

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
    mgmt_interface: str | None = None,
) -> ClusterCommEnv:
    """Run InfiniBand detection on *hosts* and return a :class:`ClusterCommEnv`.

    Probes IB on all hosts in parallel and builds a comm env with
    shared keys factored out and per-host interface overrides kept
    separate.

    *mgmt_interface* pins the management interface on every host, overriding
    detection (see
    :attr:`~sparkrun.core.cluster_manager.ClusterDefinition.mgmt_interface`).
    """
    if not hosts:
        return ClusterCommEnv.empty()

    from sparkrun.orchestration.infiniband import detect_ib_for_hosts

    ib_result = detect_ib_for_hosts(
        hosts,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        topology=topology,
        mgmt_interface=mgmt_interface,
    )
    # ``head_host`` is accepted for backward-compat with older callers
    # but the per-host map is now the source of truth — logging is
    # handled inside ``detect_ib_for_hosts``.
    _ = head_host
    return ib_result.comm_env


def detect_infiniband_local(
    dry_run: bool = False,
    mgmt_interface: str | None = None,
) -> ClusterCommEnv:
    """Run InfiniBand detection locally and return a :class:`ClusterCommEnv`."""
    ib_script = generate_ib_detect_script(mgmt_interface)
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


def cleanup_containers_by_host(
    host_containers: dict[str, list[str]],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    max_workers: int | None = None,
    *,
    executor: "Executor | None" = None,
) -> dict[str, RemoteResult]:
    """Tear down a *different* container set per host, in parallel.

    The shared teardown primitive: one dispatching, verifying removal per
    host.  The script comes from
    :meth:`~sparkrun.orchestration.executors._base.Executor.teardown_script`,
    so a workload is torn down by the substrate that started it — ``docker rm
    -f`` for a container, a process-group kill for a ``local`` native process,
    a Pod delete for k8s.  Unlike a bare ``docker rm -f ... || true`` chain it
    confirms the workloads are actually gone and reports how many were there,
    so callers can report a truthful count instead of assuming the command
    that returned 0 did anything.

    Best-effort per host: an exception or a failed teardown on one host
    is recorded, never raised, so it can't block or mask cleanup of the
    rest.

    Args:
        host_containers: Mapping of host → container names on that host.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
        max_workers: Cap on concurrent cleanup workers (defaults to the
            shared SSH fan-out cap).
        executor: The executor that owns these workloads.  ``None`` falls
            back to Docker, preserving the historical behaviour for callers
            that have no executor in hand — but a caller that *does* know
            must pass it: asking Docker about a ``local`` executor's native
            process gets a truthful "no such container" and leaves the
            workload running.  Hosts needing more than one executor are
            handled by calling this once per executor and folding the results
            with :func:`merge_teardown_results`.

    Returns:
        Mapping of host → :class:`RemoteResult`.  ``result.success`` is
        the per-host verdict; the removed count is recoverable via
        :func:`~sparkrun.orchestration.teardown.parse_teardown_removed`.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not host_containers:
        return {}

    if executor is None:
        from sparkrun.orchestration.executors.docker import DockerExecutor

        executor = DockerExecutor()

    results: dict[str, RemoteResult] = {}
    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(host_containers), max_workers)) as pool:
        futures = {
            pool.submit(
                run_command_on_host,
                host,
                executor.teardown_script(names),
                ssh_kwargs=ssh_kwargs,
                timeout=30,
                dry_run=dry_run,
                quiet=True,
            ): host
            for host, names in host_containers.items()
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                results[host] = future.result()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Cleanup raised on %s: %s", host, e)
                results[host] = RemoteResult(host=host, returncode=255, stdout="", stderr=str(e))

    failed = [h for h, r in results.items() if not r.success] if not dry_run else []
    if failed:
        logger.warning(
            "Container cleanup did not confirm on %d host(s): %s — these may still hold VRAM; check with 'sparkrun stop' or 'docker ps'.",
            len(failed),
            ", ".join(sorted(failed)),
        )
    return results


def cleanup_containers(
    hosts: list[str],
    container_names: list[str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    max_workers: int | None = None,
    *,
    executor: "Executor | None" = None,
) -> list[str]:
    """Stop and remove the same named containers on every host, in parallel.

    Thin wrapper over :func:`cleanup_containers_by_host` for the common
    "same candidate names everywhere" case (see it for the dispatch and
    verification semantics, and for what ``executor`` means).

    Args:
        hosts: Target hosts.
        container_names: Container names to remove on each host.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
        max_workers: Cap on concurrent cleanup workers (defaults to the
            shared SSH fan-out cap).
        executor: The executor that owns these workloads (``None`` → Docker).

    Returns:
        List of hosts where the teardown did not confirm (empty on full
        success).  Always empty in dry-run mode.
    """
    if not hosts:
        return []

    results = cleanup_containers_by_host(
        {host: list(container_names) for host in hosts},
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        max_workers=max_workers,
        executor=executor,
    )
    if dry_run:
        return []
    # Preserve the caller's host order in the failure report.
    return [host for host in hosts if not results[host].success]


def cleanup_containers_local(
    container_names: list[str],
    dry_run: bool = False,
    *,
    executor: "Executor | None" = None,
) -> None:
    """Stop and remove named workloads on the local machine.

    Uses the executor's own teardown script (``None`` → Docker) for the same
    reason the remote path does: the local dispatch is a different *transport*,
    not a different substrate.
    """
    if executor is None:
        from sparkrun.orchestration.executors.docker import DockerExecutor

        executor = DockerExecutor()
    run_local_script("#!/bin/bash\n" + executor.teardown_script(list(container_names)), dry_run=dry_run)


def merge_teardown_results(*result_maps: dict[str, RemoteResult]) -> dict[str, RemoteResult]:
    """Fold several per-host teardown result maps into one.

    A single host can need more than one executor — ``docker`` and ``local``
    share the ``"host"`` status scope and are merged into one status snapshot,
    so ``stop --all`` on a mixed cluster legitimately discovers both kinds of
    workload on the same machine.  Each executor is dispatched separately (its
    teardown script speaks only for its own substrate); this recombines them
    so callers keep seeing one verdict per host.

    Per host: the result is successful only if *every* map's result was, the
    output streams are concatenated, and the removed counts are **summed** and
    re-emitted as a single trailing marker line — otherwise
    :func:`~sparkrun.orchestration.teardown.parse_teardown_removed`, which
    reads the last marker it finds, would report only the final executor's
    count.
    """
    from sparkrun.orchestration.teardown import format_teardown_removed, parse_teardown_removed

    merged: dict[str, RemoteResult] = {}
    for results in result_maps:
        for host, result in results.items():
            existing = merged.get(host)
            if existing is None:
                merged[host] = result
                continue
            total = parse_teardown_removed(existing.stdout) + parse_teardown_removed(result.stdout)
            merged[host] = RemoteResult(
                host=host,
                returncode=existing.returncode or result.returncode,
                stdout="".join(
                    part
                    for part in (
                        existing.stdout,
                        "" if existing.stdout.endswith("\n") or not existing.stdout else "\n",
                        result.stdout,
                        "" if result.stdout.endswith("\n") or not result.stdout else "\n",
                        format_teardown_removed(total) + "\n",
                    )
                ),
                stderr="".join(s for s in (existing.stderr, result.stderr) if s),
            )
    return merged


# ---------------------------------------------------------------------------
# IP detection
# ---------------------------------------------------------------------------


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
# Execution helpers (local-or-remote dispatch)
# ---------------------------------------------------------------------------
#
# ``should_run_locally`` / ``run_local_script`` are defined in
# ``orchestration.ssh`` (alongside ``RemoteResult``, and so the fan-out
# helpers there can honour ``allow_local``); they are re-exported above
# for the many callers that import them from here.


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
        return run_local_script(script, dry_run=dry_run, timeout=timeout)
    return run_remote_script(host, script, timeout=timeout, dry_run=dry_run, **kw)


def run_script_on_host_streaming(
    host: str,
    script: str,
    ssh_kwargs: dict | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
    session_guard: bool = False,
) -> RemoteResult:
    """Run a local-or-remote script with live output when ``quiet`` is false.

    The streaming peer of :func:`run_script_on_host`, dispatching on the same
    :func:`should_run_locally` rule so a payload does not change behaviour
    depending on which side of that boundary it lands.
    """
    kw = ssh_kwargs or {}
    if should_run_locally(host, kw.get("ssh_user")):
        return run_local_script_streaming(
            script,
            dry_run=dry_run,
            timeout=timeout,
            quiet=quiet,
        )
    return run_remote_script_streaming(
        host,
        script,
        timeout=timeout,
        dry_run=dry_run,
        quiet=quiet,
        session_guard=session_guard,
        **kw,
    )


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
        return run_local_script("#!/bin/bash\n" + command, dry_run=dry_run, timeout=timeout)
    return run_remote_command(host, command, timeout=timeout, dry_run=dry_run, quiet=quiet, **kw)


def resolve_image_sha(
    image_ref: str,
    hosts: list[str] | tuple[str, ...],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> str | None:
    """Resolve a container image reference to its content-addressable image ID.

    Runs ``docker image inspect --format '{{.Id}}' <ref>`` on each host in
    *hosts* until one succeeds; returns the ``sha256:...`` ID string. Returns
    ``None`` when no host can resolve it (image not pulled, docker unavailable,
    network error). Returns ``None`` on ``dry_run`` to avoid live calls.

    Used by benchmark resume to lock the exact image bits across invocations
    so a re-pushed tag or rebuilt local image cannot silently change the
    workload between sessions.
    """
    if dry_run or not hosts:
        return None
    from sparkrun.utils.shell import quote as _q

    cmd = "docker image inspect --format '{{.Id}}' %s" % _q(image_ref)
    for host in hosts:
        try:
            result = run_command_on_host(host, cmd, ssh_kwargs=ssh_kwargs, dry_run=False, quiet=True)
        except Exception:
            logger.debug("resolve_image_sha: exception on %s", host, exc_info=True)
            continue
        if result.returncode == 0 and result.stdout.strip():
            sha = result.stdout.strip().splitlines()[0].strip()
            if sha.startswith("sha256:"):
                return sha
            logger.debug("resolve_image_sha: unexpected output on %s: %r", host, sha)
    return None
