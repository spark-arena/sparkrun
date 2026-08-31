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
    # rsync reports an EPERM (as opposed to EACCES) with this wording, which
    # the "permission denied" pattern above does not match — such failures used
    # to fall all the way through to the generic "rsync failed (rc=23)".
    ("operation not permitted", "permission denied (could not change file attributes)"),
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


# rsync exit code for "partial transfer due to error".  It is what rsync
# returns both when data genuinely failed to transfer *and* when every byte
# arrived but the generator could not apply attributes to the destination.
RSYNC_PARTIAL_TRANSFER_RC = 23

# Truncation budget for the stderr excerpt carried on a :class:`TransferFailure`.
_TRANSFER_DETAIL_LIMIT = 1200

# Generator messages that mean "the data is there, I just could not set an
# attribute on the destination".  Each is emitted when the SSH user does not
# *own* the destination directory — routine for a cache on NFS (root_squash, a
# differing uid mapping) or one a container created as root.  See
# :data:`~sparkrun.orchestration.ssh.NFS_SAFE_ATTR_OPTS`, which stops sparkrun
# asking for these in the first place; this set is the backstop for the ones it
# cannot anticipate.
_RSYNC_ATTRIBUTE_ERROR_PATTERNS: tuple[str, ...] = (
    "failed to set times on",
    "failed to set permissions on",
    "failed to set modification time",
    "failed to set ownership",
    "chgrp ",
    "chown ",
    "chmod ",
)

# Trailing summary line rsync always prints alongside a nonzero exit.  It
# restates the exit code rather than reporting an additional problem, so it
# must not be counted as evidence of a data failure.
_RSYNC_SUMMARY_PREFIX = "rsync error:"


def rsync_attribute_errors_only(result: RemoteResult) -> bool:
    """True when *result* failed **only** because attributes could not be set.

    rsync exits 23 for two very different situations: files that did not
    transfer, and files that all transferred while the generator hit ``EPERM``
    applying owner/group/permissions/times to a destination directory it does
    not own.  The second is a complete, usable transfer — treating it as a
    failure is what made a successful model or tuning-config distribution abort
    the launch on an NFS-backed cache.

    Conservative by construction: it demands exit code 23, at least one
    recognised attribute error, and that *every* ``rsync:`` diagnostic line be
    one.  Anything else — an ``mkdir``/``mkstemp`` that failed, a read error, an
    empty stderr we cannot reason about — is reported as the failure it may
    well be, because the cost of being wrong here is launching a workload
    against weights that never arrived.
    """
    if result.returncode != RSYNC_PARTIAL_TRANSFER_RC:
        return False

    matched_attribute_error = False
    for raw_line in (result.stderr or "").splitlines():
        line = raw_line.strip().lower()
        # The summary line restates the exit code; it is not a second problem.
        # Checked before the "rsync:" filter below, which would not match it.
        if line.startswith(_RSYNC_SUMMARY_PREFIX):
            continue
        # Only rsync's own diagnostics carry a verdict; blank lines and any
        # unrelated chatter on the channel (e.g. an SSH banner) are not
        # evidence either way.
        if not line.startswith("rsync:"):
            continue
        if any(needle in line for needle in _RSYNC_ATTRIBUTE_ERROR_PATTERNS):
            matched_attribute_error = True
            continue
        return False

    return matched_attribute_error


def rsync_has_attribute_permission_error(result: RemoteResult) -> bool:
    """True when *any* rsync error is an attribute op refused for permissions.

    The trigger for the relaxed retry (see
    :func:`~sparkrun.orchestration.ssh.relax_rsync_options`).  Weaker than
    :func:`rsync_attribute_errors_only` on purpose: that one answers "did
    everything else succeed?", this one answers "is relaxing attributes worth
    another pass?", and a transfer can have a genuine failure *and* an
    attribute failure the retry would fix.

    Requires the permission wording as well as the attribute verb, so a
    ``chown`` that failed for some other reason (a read-only filesystem, a
    vanished path) does not buy a retry that cannot help.
    """
    if result.success:
        return False
    for raw_line in (result.stderr or "").splitlines():
        line = raw_line.strip().lower()
        if not line.startswith("rsync:"):
            continue
        if not any(needle in line for needle in _RSYNC_ATTRIBUTE_ERROR_PATTERNS):
            continue
        if "operation not permitted" in line or "permission denied" in line:
            return True
    return False


def rsync_transfer_ok(result: RemoteResult) -> bool:
    """``result.success``, widened to accept an attribute-only rc=23.

    The predicate every rsync call site should use in place of
    ``result.success``.  See :func:`rsync_attribute_errors_only`.
    """
    return result.success or rsync_attribute_errors_only(result)


def split_rsync_results(
    results: list[RemoteResult],
) -> tuple[list[RemoteResult], list[RemoteResult]]:
    """Partition *results* into (genuinely failed, benign attribute-only)."""
    failed: list[RemoteResult] = []
    benign: list[RemoteResult] = []
    for r in results:
        if r.success:
            continue
        (benign if rsync_attribute_errors_only(r) else failed).append(r)
    return failed, benign


def log_benign_rsync_results(
    benign: list[RemoteResult],
    *,
    label: str = "rsync",
    _logger: logging.Logger | None = None,
) -> None:
    """Warn once per host whose transfer completed but could not set attributes.

    Deliberately a warning rather than silence: the transfer is usable, but the
    destination's ownership is worth knowing about — it is the same condition
    that will make a *future* stricter operation fail.
    """
    if not benign:
        return
    log = _logger or logger
    for r in benign:
        log.warning(
            "  %s to %s completed, but some file attributes could not be set "
            "(destination is not owned by the SSH user — typical on NFS). "
            "The data transferred successfully; continuing.",
            label,
            r.host,
        )
        detail = (r.stderr or "").strip()
        if detail:
            log.debug("  %s attribute errors on %s:\n%s", label, r.host, detail)


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
    genuinely_failed, benign = split_rsync_results(results)
    log_benign_rsync_results(benign)
    return [xfer_to_host.get(r.host, r.host) for r in genuinely_failed]


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

    Results that failed *only* because attributes could not be applied to the
    destination are not failures at all (see
    :func:`rsync_attribute_errors_only`); they are logged and dropped, so a
    transfer that actually completed never reaches a caller as an error.
    """
    xfer_to_host = dict(zip(transfer_hosts, management_hosts))
    genuinely_failed, benign = split_rsync_results(results)
    log_benign_rsync_results(benign)
    failures: list[TransferFailure] = []
    for r in genuinely_failed:
        host = xfer_to_host.get(r.host, r.host)
        reason = classify_rsync_failure(r)
        stderr = (r.stderr or "").strip()
        # Keep the *head*, not the tail.  rsync's generator reports the first
        # thing that went wrong first and closes with a fixed "rsync error:
        # some files/attrs were not transferred" summary — so a tail excerpt
        # spends its budget restating the exit code we already have and clips
        # away the one line naming the path that failed.
        detail = stderr[:_TRANSFER_DETAIL_LIMIT] + "… (truncated)" if len(stderr) > _TRANSFER_DETAIL_LIMIT else stderr
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

    # Per-host classified lines (e.g. "ERROR:  rsync failed on h1: out of disk space").
    # The captured stderr excerpt follows the classified reason rather than being
    # discarded: the classification is a guess from a pattern table, and when it
    # guesses wrong (or falls through to the generic "rc=N") the excerpt is the
    # only thing that tells the user which path failed and why.
    for f in failures:
        log.error("  %s failed on %s: %s", label, f.host, f.reason)
        if f.detail:
            log.error("    %s", f.detail.replace("\n", "\n    "))

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
