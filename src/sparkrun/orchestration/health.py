"""Health and readiness checks for sparkrun containers and services.

Extracted from orchestration/primitives.py to reduce scope creep in that module.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

#: Kernel socket tables read by the ``wait_for_port`` fallback probe.
PROC_NET_TCP_PATHS = "/proc/net/tcp /proc/net/tcp6"

#: ``/proc/net/tcp`` ``st`` column value for ``TCP_LISTEN``.
_PROC_TCP_LISTEN = "0A"


def proc_listen_probe_cmd(port: int, proc_paths: str = PROC_NET_TCP_PATHS) -> str:
    """Build the ``/proc/net/tcp`` half of the listening-port probe.

    The fallback for hosts without ``ss``.  Three details are
    load-bearing, and getting any of them wrong reports a port as ready
    when nothing is listening on it:

    - **Filter on the state column.**  ``st`` must be ``0A``
      (``TCP_LISTEN``).  A plain port match also hits ``ESTABLISHED``
      and — the one that bites — ``TIME_WAIT`` rows, which linger for
      ~60s after a workload is stopped.  Relaunching inside that window
      would report the *new* server ready before it had bound.
    - **Anchor to the local-address column.**  ``$3`` is the *remote*
      address; matching it means any outbound connection to a host
      serving on the same port number reads as "we are listening".
    - **Read ``tcp6`` too**, via ``cat`` rather than as extra ``awk``
      operands — awk aborts with a fatal error (ignoring ``END``) when a
      file cannot be opened, and ``/proc/net/tcp6`` is absent on hosts
      with IPv6 disabled.

    Args:
        port: TCP port to look for.
        proc_paths: Space-separated socket tables to scan.  Overridden
            by tests to run the probe against a fixture.

    Returns:
        A shell command that exits 0 iff *port* is in ``LISTEN`` state.
    """
    return "cat %s 2>/dev/null | awk '$4==\"%s\" && $2 ~ /:%04X$/{f=1} END{exit !f}'" % (
        proc_paths,
        _PROC_TCP_LISTEN,
        port,
    )


def listen_probe_cmd(port: int, proc_paths: str = PROC_NET_TCP_PATHS) -> str:
    """Build the side-effect-free "is *port* listening?" probe command.

    Deliberately *not* ``nc -z``: that opens a real TCP connection,
    which consumes one-shot rendezvous accepts.  Atlas's NCCL bootstrap
    (rank 0) accepts exactly ``world_size - 1`` connections, hands out
    its unique NCCL ID, then drops the listener — a probe connection
    steals that single accept, the real worker gets "Connection refused"
    forever, and the head spins in ``ncclCommInitRank``.

    ``ss``'s own ``sport`` filter does the port matching, so no output
    parsing is involved: grepping ``ss`` output for the port number has
    to defend against the Recv-Q/Send-Q columns and against ``:18000``
    matching a probe for ``8000``.  ``LISTEN`` is matched instead of
    passing ``-H`` (added in iproute2 4.x) so the probe also works on
    older hosts; the header line never contains it.  An ``ss`` that is
    missing — or too old to parse the filter — exits non-zero and falls
    through to :func:`proc_listen_probe_cmd`.

    Note this matches a listener on *any* local address, where
    ``nc -z localhost`` only reached loopback.  That is the intent: the
    health check that follows connects via the head's routable IP, not
    via loopback.

    Args:
        port: TCP port to look for.
        proc_paths: Forwarded to :func:`proc_listen_probe_cmd`.

    Returns:
        A shell command that exits 0 iff *port* is in ``LISTEN`` state.
    """
    return 'ss -tln "sport = :%d" 2>/dev/null | grep -q LISTEN || %s' % (
        port,
        proc_listen_probe_cmd(port, proc_paths),
    )


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
    from sparkrun.orchestration.primitives import run_command_on_host
    from sparkrun.utils.shell import quote

    cmd = "docker inspect -f '{{.State.Running}}' %s 2>/dev/null" % quote(container_name)
    result = run_command_on_host(host, cmd, ssh_kwargs=ssh_kwargs, timeout=10)
    return result.success and "true" in result.stdout.lower()


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

    from sparkrun.orchestration.primitives import run_command_on_host

    # Reads the kernel socket table rather than connecting; see
    # `listen_probe_cmd` for why a connecting probe breaks NCCL bootstrap.
    check_cmd = listen_probe_cmd(port)
    for attempt in range(1, max_retries + 1):
        # Check container liveness before polling the port
        if container_name and attempt > 1:
            if not is_container_running(host, container_name, ssh_kwargs=ssh_kwargs):
                logger.error(
                    "  Container %s is no longer running on %s — aborting wait",
                    container_name,
                    host,
                )
                return False

        result = run_command_on_host(host, check_cmd, ssh_kwargs=ssh_kwargs, timeout=5, quiet=True)
        if result.success:
            logger.info("  Port %d ready after %ds", port, attempt * retry_interval)
            return True
        if attempt % 10 == 0:
            logger.info(
                "  Still waiting for port %d (%ds elapsed)...",
                port,
                attempt * retry_interval,
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
