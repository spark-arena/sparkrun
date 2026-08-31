"""Tests for the remote-failure log detail (spark-arena/sparkrun#257, minor).

``_run_subprocess`` logged only ``stderr`` on failure, so a payload that
reported its problem on *stdout* — which the embedded scripts routinely do,
since they ``echo`` diagnostics — produced a bare::

    WARNING   SSH script <- spark2 FAILED rc=1 (0.3s):

with nothing after the colon.  An empty reason reads as a tool malfunction
rather than a remote command that failed for a stated reason.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from sparkrun.orchestration.ssh import RemoteResult, _failure_detail, run_remote_script


# ---------------------------------------------------------------------------
# _failure_detail — fallback order
# ---------------------------------------------------------------------------


def _res(stdout="", stderr=""):
    return RemoteResult(host="h", returncode=1, stdout=stdout, stderr=stderr)


def test_failure_detail_prefers_stderr():
    assert _failure_detail(_res(stdout="on stdout", stderr="on stderr")) == "on stderr"


def test_failure_detail_falls_back_to_stdout():
    assert _failure_detail(_res(stdout="permission denied", stderr="")) == "permission denied"


def test_failure_detail_ignores_whitespace_only_stderr():
    """A stderr of only newlines must not win over a real stdout message."""
    assert _failure_detail(_res(stdout="the actual reason", stderr="\n  \n")) == "the actual reason"


def test_failure_detail_no_output_is_explicit():
    assert _failure_detail(_res()) == "(no output)"


def test_failure_detail_truncates():
    assert _failure_detail(_res(stderr="x" * 500), limit=200) == "x" * 200


# ---------------------------------------------------------------------------
# End-to-end through run_remote_script's failure logging
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_stdout_only_failure_is_logged(mock_run, caplog):
    mock_run.return_value = MagicMock(returncode=1, stdout=b"ERROR: sudo password required", stderr=b"")

    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.ssh"):
        result = run_remote_script("spark2", "#!/bin/bash\nexit 1")

    assert not result.success
    assert "ERROR: sudo password required" in caplog.text


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_silent_failure_says_no_output(mock_run, caplog):
    """rc!=0 with nothing on either stream must still name the empty case."""
    mock_run.return_value = MagicMock(returncode=1, stdout=b"", stderr=b"")

    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.ssh"):
        run_remote_script("spark2", "#!/bin/bash\nexit 1")

    assert "(no output)" in caplog.text
    # The pre-fix log ended at the colon with nothing after it.
    assert not caplog.text.rstrip().endswith(":")


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_stderr_still_wins_end_to_end(mock_run, caplog):
    mock_run.return_value = MagicMock(
        returncode=255,
        stdout=b"",
        stderr=b"ssh: Could not resolve hostname s: Temporary failure in name resolution",
    )

    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.ssh"):
        run_remote_script("s", "#!/bin/bash\ntrue")

    assert "Could not resolve hostname s" in caplog.text
