"""Tests for the vanished-source (rc=24) rsync classification and retry."""

from __future__ import annotations

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.orchestration.transfer import (
    classify_rsync_failure,
    rsync_had_vanished_files,
)

# A live cache mid-write: blobs the file list saw, gone when the generator
# read them — the case the retry exists for.
_VANISHED = (
    "file has vanished: '/cache/hub/models--org--name/blobs/0de9'\n"
    "file has vanished: '/cache/hub/models--org--name/blobs/0f6f'\n"
    "rsync error: some files/attrs were not transferred (code 24) at main.c(1338) [sender=3.2.7]\n"
)

# rc=24 without the marker: nothing identifies vanished files, so a blind
# retry could churn on a failure no re-walk can fix.
_UNMARKED_RC24 = "rsync error: some files/attrs were not transferred (code 24) at main.c(1338) [sender=3.2.7]\n"


def _res(rc: int, stderr: str = "") -> RemoteResult:
    return RemoteResult(host="h1", returncode=rc, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# detection and classification
# ---------------------------------------------------------------------------


def test_vanished_failure_is_classified():
    assert classify_rsync_failure(_res(24, _VANISHED)) == "source files changed during transfer"


def test_vanished_detection_requires_rc_24_and_the_marker():
    assert rsync_had_vanished_files(_res(24, _VANISHED)) is True
    assert rsync_had_vanished_files(_res(24, _UNMARKED_RC24)) is False
    assert rsync_had_vanished_files(_res(23, _VANISHED)) is False
    assert rsync_had_vanished_files(_res(0, _VANISHED)) is False


# ---------------------------------------------------------------------------
# end-to-end through run_rsync
# ---------------------------------------------------------------------------


def _run_with_results(results, monkeypatch, tmp_path):
    """Drive run_rsync with a scripted sequence of subprocess outcomes."""
    calls = []

    class P:
        def __init__(self, rc, err):
            self.returncode, self.stdout, self.stderr = rc, b"", err.encode()

    seq = list(results)

    def fake_run(cmd, **kw):
        calls.append(cmd)
        rc, err = seq.pop(0)
        return P(rc, err)

    monkeypatch.setattr("sparkrun.orchestration.ssh.subprocess.run", fake_run)
    from sparkrun.orchestration.ssh import run_rsync

    return run_rsync(str(tmp_path), "h1", "/remote"), calls


def test_retry_fires_once_and_succeeds(monkeypatch, tmp_path):
    result, calls = _run_with_results([(24, _VANISHED), (0, "")], monkeypatch, tmp_path)
    assert len(calls) == 2
    assert result.success is True


def test_unmarked_rc24_is_not_retried(monkeypatch, tmp_path):
    result, calls = _run_with_results([(24, _UNMARKED_RC24)], monkeypatch, tmp_path)
    assert len(calls) == 1
    assert result.returncode == 24


def test_failed_retry_reports_the_retry(monkeypatch, tmp_path):
    result, calls = _run_with_results([(24, _VANISHED), (24, _VANISHED)], monkeypatch, tmp_path)
    assert len(calls) == 2
    assert result.returncode == 24
    assert "file has vanished" in result.stderr


def test_kill_switch_disables_the_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKRUN_NO_RSYNC_RETRY", "1")
    result, calls = _run_with_results([(24, _VANISHED)], monkeypatch, tmp_path)
    assert len(calls) == 1


def test_dry_run_never_executes_or_retries(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.subprocess.run",
        lambda cmd, **kw: called.append(cmd),
    )
    from sparkrun.orchestration.ssh import run_rsync

    result = run_rsync(str(tmp_path), "h1", "/remote", dry_run=True)
    assert called == []
    assert result.success is True
