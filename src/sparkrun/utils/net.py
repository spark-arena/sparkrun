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

import re
import socket
import subprocess

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
    except Exception:
        pass

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
    except Exception:
        # Fall back to hostname if routing lookup fails (e.g. target
        # is not resolvable from the control machine itself).
        return socket.gethostname() or None
