"""Tests for the automatic relaxed-attribute rsync retry."""

from __future__ import annotations


import pytest

from sparkrun.orchestration.ssh import (
    NFS_SAFE_ATTR_OPTS,
    RSYNC_RELAXED_ATTR_OPTS,
    RemoteResult,
    relax_rsync_options,
    rsync_options_are_relaxed,
)
from sparkrun.orchestration.transfer import rsync_has_attribute_permission_error

# An attribute op refused for permissions, alongside a data failure the
# relaxation would not fix on its own — the case the retry exists for.
_ATTR_PERM_PLUS_DATA = (
    'rsync: [receiver] failed to set times on "/cache/blobs/abc": Operation not permitted (1)\n'
    'rsync: [receiver] mkstemp "/cache/blobs/.def" failed: Operation not permitted (1)\n'
    "rsync error: some files/attrs were not transferred (code 23) at main.c(1338) [sender=3.2.7]\n"
)

# Attribute-only: complete transfer, accepted without a retry.
_ATTR_ONLY = (
    'rsync: [generator] chgrp "/cache/." failed: Operation not permitted (1)\n'
    "rsync error: some files/attrs were not transferred (code 23) at main.c(1338) [sender=3.2.7]\n"
)

# No attribute verb at all — relaxing attributes cannot help.
_HARD_DENIED = (
    'rsync: [generator] recv_generator: mkdir "/cache/configs" failed: Permission denied (13)\n'
    "rsync error: some files/attrs were not transferred (code 23) at main.c(1338) [sender=3.2.7]\n"
)


def _res(rc: int, stderr: str = "") -> RemoteResult:
    return RemoteResult(host="h1", returncode=rc, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# option helpers
# ---------------------------------------------------------------------------


def test_relax_appends_rather_than_rewrites():
    """rsync resolves repeated attribute flags last-wins, so appending suffices.

    That is what lets the relaxation work on an option set this function has
    never seen, including ones added later.
    """
    out = relax_rsync_options(["-a", "--size-only"])
    assert out[:2] == ["-a", "--size-only"]
    assert set(RSYNC_RELAXED_ATTR_OPTS) <= set(out)


def test_relax_is_idempotent():
    once = relax_rsync_options(["-a"])
    assert relax_rsync_options(once) == once


def test_relaxed_set_extends_the_default_nfs_safe_set():
    """The retry adds times to what the default already drops."""
    assert set(NFS_SAFE_ATTR_OPTS) < set(RSYNC_RELAXED_ATTR_OPTS)
    assert "--no-times" in RSYNC_RELAXED_ATTR_OPTS


def test_rsync_options_are_relaxed_detects_a_full_set():
    assert rsync_options_are_relaxed(relax_rsync_options(["-a"])) is True
    assert rsync_options_are_relaxed(["-a", *NFS_SAFE_ATTR_OPTS]) is False


# ---------------------------------------------------------------------------
# the retry trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stderr, expected",
    [
        (_ATTR_PERM_PLUS_DATA, True),
        (_ATTR_ONLY, True),
        (_HARD_DENIED, False),  # no attribute verb — relaxing cannot help
        ('rsync: [receiver] chown "/x" failed: Read-only file system (30)\n', False),
        ("", False),
    ],
)
def test_attribute_permission_error_detection(stderr, expected):
    assert rsync_has_attribute_permission_error(_res(23, stderr)) is expected


def test_success_never_triggers_a_retry():
    assert rsync_has_attribute_permission_error(_res(0, _ATTR_ONLY)) is False


# ---------------------------------------------------------------------------
# end-to-end through run_rsync
# ---------------------------------------------------------------------------


def _run_with_results(results, monkeypatch, tmp_path, **kwargs):
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

    result = run_rsync(str(tmp_path), "h1", "/remote", **kwargs)
    return result, calls


def test_retry_fires_and_succeeds(monkeypatch, tmp_path, caplog):
    result, calls = _run_with_results([(23, _ATTR_PERM_PLUS_DATA), (0, "")], monkeypatch, tmp_path)
    assert len(calls) == 2
    assert "--no-times" not in calls[0]
    assert "--no-times" in calls[1]
    assert result.success is True


def test_attribute_only_failure_is_not_retried(monkeypatch, tmp_path):
    """The data is already there; a retry would re-walk the tree for nothing."""
    result, calls = _run_with_results([(23, _ATTR_ONLY)], monkeypatch, tmp_path)
    assert len(calls) == 1
    assert result.returncode == 23


def test_hard_permission_failure_is_not_retried(monkeypatch, tmp_path):
    """A destination we cannot write to is not fixed by asking for less."""
    result, calls = _run_with_results([(23, _HARD_DENIED)], monkeypatch, tmp_path)
    assert len(calls) == 1
    assert result.returncode == 23


def test_already_relaxed_options_are_not_retried(monkeypatch, tmp_path):
    result, calls = _run_with_results(
        [(23, _ATTR_PERM_PLUS_DATA)],
        monkeypatch,
        tmp_path,
        rsync_options=relax_rsync_options(["-a"]),
    )
    assert len(calls) == 1


def test_failed_retry_reports_the_retry_not_the_first_attempt(monkeypatch, tmp_path):
    """The retry is strictly more permissive, so its stderr is the useful one."""
    result, calls = _run_with_results(
        [(23, _ATTR_PERM_PLUS_DATA), (11, "rsync: write failed: No space left on device (28)\n")],
        monkeypatch,
        tmp_path,
    )
    assert len(calls) == 2
    assert result.returncode == 11
    assert "No space left" in result.stderr


def test_kill_switch_disables_the_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKRUN_NO_RSYNC_RETRY", "1")
    result, calls = _run_with_results([(23, _ATTR_PERM_PLUS_DATA)], monkeypatch, tmp_path)
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
