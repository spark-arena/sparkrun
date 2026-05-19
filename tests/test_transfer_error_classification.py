"""Tests for sparkrun.orchestration.transfer — rsync error classification."""

from __future__ import annotations

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.orchestration.transfer import (
    TransferFailure,
    classify_rsync_failure,
    format_transfer_failures,
    map_transfer_failures,
    map_transfer_failures_detailed,
)


# ---------------------------------------------------------------------------
# classify_rsync_failure
# ---------------------------------------------------------------------------


def _r(host: str, returncode: int, stderr: str = "") -> RemoteResult:
    return RemoteResult(host=host, returncode=returncode, stdout="", stderr=stderr)


def test_classify_no_space():
    """Issue #168: rsync's 'No space left on device' is recognized."""
    result = _r(
        "h1",
        11,
        'rsync: [receiver] write failed on "/path/blob": No space left on device (28)',
    )
    assert classify_rsync_failure(result) == "out of disk space"


def test_classify_disk_quota():
    result = _r("h1", 11, "rsync: write failed: Disk quota exceeded (122)")
    assert classify_rsync_failure(result) == "disk quota exceeded"


def test_classify_permission_denied():
    result = _r("h1", 23, "rsync: change_dir failed: Permission denied (13)")
    assert classify_rsync_failure(result) == "permission denied"


def test_classify_connection_refused():
    result = _r("h1", 255, "ssh: connect to host h1 port 22: Connection refused")
    assert classify_rsync_failure(result) == "SSH connection refused"


def test_classify_connection_timed_out():
    result = _r("h1", 255, "ssh: connect to host h1 port 22: Connection timed out")
    assert classify_rsync_failure(result) == "SSH connection timed out"


def test_classify_unknown_returns_generic_with_rc():
    """Unknown stderr falls back to a generic rc-tagged reason."""
    result = _r("h1", 99, "something weird happened")
    assert classify_rsync_failure(result) == "rsync failed (rc=99)"


def test_classify_empty_stderr_returns_generic():
    result = _r("h1", 23, "")
    assert classify_rsync_failure(result) == "rsync failed (rc=23)"


def test_classify_case_insensitive():
    """Patterns match regardless of stderr casing (rsync varies by version)."""
    result = _r("h1", 11, "NO SPACE LEFT ON DEVICE")
    assert classify_rsync_failure(result) == "out of disk space"


# ---------------------------------------------------------------------------
# map_transfer_failures (legacy)
# ---------------------------------------------------------------------------


def test_map_transfer_failures_returns_host_names_only():
    results = [
        _r("10.0.0.1", 0),
        _r("10.0.0.2", 11, "No space left on device"),
    ]
    failed = map_transfer_failures(results, ["10.0.0.1", "10.0.0.2"], ["m1", "m2"])
    assert failed == ["m2"]


def test_map_transfer_failures_unmapped_host_returned_as_is():
    """If a result host isn't in the transfer→mgmt mapping it's returned verbatim."""
    results = [_r("rogue", 1, "err")]
    failed = map_transfer_failures(results, ["10.0.0.1"], ["m1"])
    assert failed == ["rogue"]


# ---------------------------------------------------------------------------
# map_transfer_failures_detailed
# ---------------------------------------------------------------------------


def test_map_transfer_failures_detailed_classifies_per_host():
    results = [
        _r("10.0.0.1", 0),
        _r("10.0.0.2", 11, "rsync: write failed: No space left on device (28)"),
        _r("10.0.0.3", 23, "rsync: change_dir failed: Permission denied (13)"),
    ]
    failures = map_transfer_failures_detailed(
        results,
        ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        ["m1", "m2", "m3"],
    )
    by_host = {f.host: f for f in failures}
    assert set(by_host) == {"m2", "m3"}
    assert by_host["m2"].reason == "out of disk space"
    assert by_host["m3"].reason == "permission denied"


def test_map_transfer_failures_detailed_truncates_long_stderr():
    """Detail field is bounded so callers don't dump multi-KB blobs in logs."""
    big_stderr = "x" * 5000
    results = [_r("10.0.0.1", 23, big_stderr)]
    failures = map_transfer_failures_detailed(results, ["10.0.0.1"], ["m1"])
    assert len(failures) == 1
    assert len(failures[0].detail) <= 400


def test_map_transfer_failures_detailed_empty_on_success():
    results = [_r("10.0.0.1", 0)]
    failures = map_transfer_failures_detailed(results, ["10.0.0.1"], ["m1"])
    assert failures == []


# ---------------------------------------------------------------------------
# format_transfer_failures
# ---------------------------------------------------------------------------


def test_format_transfer_failures_groups_by_reason():
    failures = [
        TransferFailure(host="m1", reason="out of disk space on destination"),
        TransferFailure(host="m2", reason="out of disk space on destination"),
        TransferFailure(host="m3", reason="permission denied on destination"),
    ]
    rendered = format_transfer_failures(failures)
    assert "out of disk space on destination on m1, m2" in rendered
    assert "permission denied on destination on m3" in rendered


def test_format_transfer_failures_empty_returns_empty_string():
    assert format_transfer_failures([]) == ""


def test_format_transfer_failures_single_host():
    failures = [TransferFailure(host="m1", reason="out of disk space on destination")]
    assert format_transfer_failures(failures) == "out of disk space on destination on m1"
