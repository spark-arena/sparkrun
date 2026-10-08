"""Host-local network identity primitives.

Leaf-level helpers that answer questions about *this* machine's networking
(its IPs, whether an address refers to it) using the standard library and
OS tools.  Kept here — not in ``orchestration/`` — because low layers such
as ``core.hosts`` consume them; placing them in orchestration would invert
the dependency direction.  Multi-host network *orchestration* (CX7/IB,
host-key distribution) lives in ``orchestration/networking.py``.

The only OS-specific piece is interface enumeration (``ip -o addr show``,
Linux/iproute2).  It is isolated so a future ``sys.platform`` branch (macOS
``ifconfig``, Windows) can be added without touching callers.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import time
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

# Matches the address in ``ip -o addr show`` lines: ``... inet 10.0.0.1/24 ...``
# or ``... inet6 fe80::1/64 ...``.  The CIDR suffix is excluded by the char class.
_IP_ADDR_RE = re.compile(r"\binet6?\s+([0-9a-fA-F.:]+)")


def is_valid_ip(ip: str) -> bool:
    """Basic check if a string looks like an IPv4 address."""
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def get_local_ips() -> set[str]:
    """Return every IP address assigned to a local network interface.

    Enumerates interfaces via ``ip -o addr show`` rather than relying on
    ``socket.getaddrinfo(gethostname())``, which only sees addresses mapped
    in DNS/hosts.  On DGX Spark the hostname is pinned to ``127.0.0.1`` in
    ``/etc/hosts`` and the LAN IPs live only on the interfaces, so the
    getaddrinfo approach misses them entirely.

    Covers both IPv4 and IPv6 addresses; loopback is always included.
    Returns just the loopback set on any failure (``ip`` missing, non-zero
    exit, timeout) so callers can treat the result as a safe lower bound.
    """
    ips: set[str] = {"127.0.0.1", "::1"}
    try:
        result = subprocess.run(
            ["ip", "-o", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ips
    if result.returncode != 0:
        return ips
    for match in _IP_ADDR_RE.finditer(result.stdout):
        # Strip any zone id on link-local addresses (e.g. ``fe80::1%eth0``).
        ips.add(match.group(1).split("%")[0])
    return ips


def is_local_host(host: str) -> bool:
    """Check if a host string refers to the local machine.

    Checks against localhost, the system hostname/FQDN, and every IP
    (IPv4 and IPv6) assigned to a local network interface.  Falls back to
    a bind probe for hostnames or edge cases interface enumeration misses.
    """
    if host in ("localhost", "127.0.0.1", "::1", ""):
        return True

    # Check hostname match
    try:
        if host == socket.gethostname() or host == socket.getfqdn():
            return True
    except OSError:
        logger.debug("hostname lookup failed in is_local_host(%r)", host, exc_info=True)

    # Match against enumerated local interface IPs (covers DGX Spark LAN IPs
    # that aren't mapped in DNS/hosts).
    if host in get_local_ips():
        return True

    # Fallback: try to bind to the address — succeeds only for local IPs.
    # Catches hostnames and addresses the interface enumeration didn't list.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return True
    except OSError:
        return False


def local_ip_for(target_host: str) -> str | None:
    """Return the local IP address on the interface that routes to *target_host*.

    Uses a UDP connect (no packets sent) to let the OS pick the right
    source address.  Falls back to ``socket.gethostname()`` on failure.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((target_host, 1))  # port is arbitrary; no traffic sent
            return s.getsockname()[0]
    except OSError:
        # Fall back to hostname if routing lookup fails (e.g. target
        # is not resolvable from the control machine itself).
        logger.debug("routing lookup failed in local_ip_for(%r)", target_host, exc_info=True)
        return socket.gethostname() or None


def accepts_tcp(address: str, port: int, timeout: float = 1.0) -> bool:
    """Whether something on this machine accepts TCP connections at *address*:*port*."""
    try:
        with socket.create_connection((address, port), timeout=timeout):
            return True
    except OSError:
        return False


def has_route_to(host: str) -> bool | None:
    """Whether this machine currently has a route to *host*.

    A UDP connect performs the kernel's route lookup without sending
    anything, and fails with ``ENETUNREACH`` while no route exists.

    ``None`` means *cannot tell*: the name does not resolve here, which is
    normal for an ``~/.ssh/config`` alias (ssh resolves ``HostName`` itself)
    and says nothing about routes.
    """
    try:
        infos = socket.getaddrinfo(host, 22, type=socket.SOCK_DGRAM)
    except OSError:
        return None
    for family, socktype, proto, _canon, sockaddr in infos:
        try:
            with socket.socket(family, socktype, proto) as s:
                s.connect(sockaddr)
                return True
        except OSError:
            continue
    return False


def wait_for_routes(
    hosts: Iterable[str],
    timeout_s: float = 30.0,
    interval_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Wait until this machine has a route to every host in *hosts*.

    For use after reconfiguring local networking (``netplan apply``), which
    can briefly drop the management interface's routes.  Returns the hosts
    still unroutable when *timeout_s* ran out (empty when all recovered).
    A host whose route cannot be checked (:func:`has_route_to` ``None``) is
    not waited for.
    """
    pending = list(dict.fromkeys(hosts))
    deadline = time.monotonic() + timeout_s
    while True:
        pending = [h for h in pending if has_route_to(h) is False]
        if not pending or time.monotonic() >= deadline:
            return pending
        sleep(interval_s)
