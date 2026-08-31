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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from sparkrun.orchestration.ssh import RemoteResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TransferError(Exception):
    """User-facing error for any file-transfer (rsync) operation.

    Raised by hooks, mods, tuning, and distribution layers when an
    rsync-like operation fails.  The exception message is expected to
    be already classified via :func:`format_transfer_failures` or
    :func:`classify_rsync_failure` so the CLI can display it directly.
    """


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


def present_and_raise_transfer_failure(
    failures: list[TransferFailure],
    *,
    operation: str,
    cache_status_hosts: list[str] | None = None,
    cache_dir: str | None = None,
    ssh_kwargs: dict | None = None,
    label: str = "transfer",
    exc_class: type[TransferError] = TransferError,
    _logger: logging.Logger | None = None,
) -> NoReturn:
    """Log classified per-host failure lines, optionally emit a disk-space table, then raise.

    This is the canonical "polish" path shared by model distribution,
    hook copy commands, and mod staging.  It replaces the repetitive
    inline ``for f in failures: logger.error(...)`` / OOS table blocks
    that used to live in each call site.

    Args:
        failures: Non-empty list of classified failures.
        operation: Human-readable operation label used in the raised
            exception message, e.g. ``"pre_exec[1] copy failed"`` or
            ``"Model distribution failed"``.
        cache_status_hosts: Full host list to probe for the disk-space
            table (may be larger than the failing hosts).  When ``None``
            the table is skipped even for OOS failures.
        cache_dir: Remote HuggingFace cache directory to probe.  When
            ``None`` the table is skipped.
        ssh_kwargs: SSH connection kwargs forwarded to
            :func:`~sparkrun.orchestration.disk_info.probe_cache_status`.
        label: Noun used in per-host log lines, e.g. ``"rsync"`` or
            ``"copy"``.
        exc_class: Exception class to raise — defaults to
            :class:`TransferError` but callers that need
            ``DistributionError`` pass it here.
        _logger: Override the module-level logger (used by tests).

    Raises:
        TransferError: (or *exc_class*) Always.  The message is
            ``"<operation>: <format_transfer_failures(failures)>"``.
    """
    log = _logger or logger

    # Per-host classified lines (e.g. "ERROR:  rsync failed on h1: out of disk space")
    for f in failures:
        log.error("  %s failed on %s: %s", label, f.host, f.reason)

    # If any failure is OOS and we have enough info, emit the cache-status table.
    oos_hosts = [f.host for f in failures if "disk space" in f.reason or "quota" in f.reason]
    if oos_hosts and cache_status_hosts is not None and cache_dir is not None:
        from sparkrun.orchestration.disk_info import probe_cache_status
        from sparkrun.utils.cli_formatters import format_cache_status_table

        kw = ssh_kwargs or {}
        cache_status = probe_cache_status(
            cache_status_hosts,
            hf_cache_dir=cache_dir,
            ssh_kwargs=kw,
        )
        if cache_status:
            log.error(
                "  Cluster cache status:\n%s",
                format_cache_status_table(cache_status, highlight_hosts=oos_hosts),
            )

    raise exc_class("%s: %s" % (operation, format_transfer_failures(failures)))
