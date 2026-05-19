"""Transfer-failure mapping, classification, and presentation.

Helpers for turning a list of :class:`RemoteResult` from
:func:`run_rsync_parallel` into user-facing diagnostics.  When fast-network
IPs (InfiniBand) are used for the actual data transfer, failures are
reported against those IPs; this module is responsible for mapping them
back to management hostnames, classifying the rsync stderr into a short
human-readable reason (e.g. ``"out of disk space on destination"``), and
formatting summaries for error messages and logs.
"""

from __future__ import annotations

from dataclasses import dataclass

from sparkrun.orchestration.ssh import RemoteResult


@dataclass
class TransferFailure:
    """Classified information about a single failed transfer.

    Used by callers that want a human-readable reason (e.g. "out of
    disk space") alongside the host name, rather than just a bare list
    of failed hosts.
    """

    host: str
    """Management hostname (after transfer→management mapping)."""

    reason: str
    """Short classified reason — see :func:`classify_rsync_failure`."""

    detail: str = ""
    """Truncated stderr excerpt for diagnostics; may be empty."""


# Common rsync stderr fragments mapped to short classified reasons.  Order
# matters — earlier patterns take precedence so the most specific match
# wins.  All patterns are matched case-insensitively.
_RSYNC_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("no space left on device", "out of disk space"),
    ("disk quota exceeded", "disk quota exceeded"),
    ("permission denied", "permission denied"),
    ("connection refused", "SSH connection refused"),
    ("connection timed out", "SSH connection timed out"),
    ("connection closed", "SSH connection closed unexpectedly"),
    ("host key verification failed", "SSH host key verification failed"),
    ("could not resolve hostname", "hostname resolution failed"),
    ("rsync error: error in rsync protocol", "rsync protocol error"),
)


def classify_rsync_failure(result: RemoteResult) -> str:
    """Return a short human-readable reason for an rsync failure.

    Inspects the captured stderr for well-known fragments and returns a
    short classification.  When no pattern matches, returns a generic
    message that includes the return code so the user at least sees the
    failure was real, even if its specific cause is unfamiliar.
    """
    stderr = (result.stderr or "").lower()
    for needle, reason in _RSYNC_FAILURE_PATTERNS:
        if needle in stderr:
            return reason
    return "rsync failed (rc=%d)" % result.returncode


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


def map_transfer_failures_detailed(
    results: list[RemoteResult],
    transfer_hosts: list[str],
    management_hosts: list[str],
) -> list[TransferFailure]:
    """Like :func:`map_transfer_failures` but with classified per-host detail.

    Returns one :class:`TransferFailure` per failed result, with the
    management hostname (after IB→mgmt mapping) and a short classified
    reason derived from the rsync stderr.  The ``detail`` field carries
    a truncated stderr excerpt so callers can include it in logs
    without dumping multi-KB blobs.
    """
    xfer_to_host = dict(zip(transfer_hosts, management_hosts))
    failures: list[TransferFailure] = []
    for r in results:
        if r.success:
            continue
        host = xfer_to_host.get(r.host, r.host)
        reason = classify_rsync_failure(r)
        stderr = (r.stderr or "").strip()
        detail = stderr[-400:] if len(stderr) > 400 else stderr
        failures.append(TransferFailure(host=host, reason=reason, detail=detail))
    return failures


def format_transfer_failures(failures: list[TransferFailure]) -> str:
    """Render a list of :class:`TransferFailure` as a multi-host summary line.

    Hosts sharing the same reason are grouped so the summary stays
    readable even with large clusters.
    """
    by_reason: dict[str, list[str]] = {}
    for f in failures:
        by_reason.setdefault(f.reason, []).append(f.host)
    parts: list[str] = []
    for reason, hosts in by_reason.items():
        parts.append("%s on %s" % (reason, ", ".join(hosts)))
    return "; ".join(parts)
