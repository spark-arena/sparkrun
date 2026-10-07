"""Bounded retries retain the first failure and require a complete source."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from sparkrun.models.transport import builtin_copy
from sparkrun.models.artifacts import CacheValidationReport
from sparkrun.transports.session import HostCommandResult
from test_model_distribution_provider import request


def result(rc, stderr=b""):
    return HostCommandResult("h", rc, b"", stderr)


@pytest.mark.parametrize("complete,expected", [(True, 2), (False, 1)])
def test_vanished_retry_requires_inventory(complete, expected, caplog):
    req = request()
    io = Mock()
    io.execute.side_effect = [
        result(24, b'file has vanished: "blob"\nrsync warning: some files vanished before they could be transferred (code 24)'),
        result(0),
    ]
    io.observe.return_value = CacheValidationReport("control", req.manifest.identity, "metadata", "complete" if complete else "incomplete")
    outcome = builtin_copy(req, io, {})
    assert io.execute.call_count == expected
    assert outcome.outcomes["h"] == ("complete" if complete else "failed")
    assert "file has vanished" in caplog.text
    if complete:
        assert io.execute.call_args_list[1].kwargs["timeout"] <= io.execute.call_args_list[0].kwargs["timeout"]


def test_kill_switch_and_deadline(monkeypatch):
    req = request()
    io = Mock()
    io.execute.return_value = result(24, b"file has vanished: blob")
    monkeypatch.setenv("SPARKRUN_NO_RSYNC_RETRY", "1")
    assert builtin_copy(req, io, {}).outcomes["h"] == "failed"
    assert io.execute.call_count == 1
    io.observe.assert_not_called()
    io.reset_mock()
    assert builtin_copy(replace(req, timeout=0), io, {}).outcomes["h"] == "failed"
    io.execute.assert_not_called()


@pytest.mark.parametrize("guard_result,expected_calls", [(True, 2), (False, 1)])
def test_generic_rsync_retry_still_runs_with_relaxed_attributes(monkeypatch, guard_result, expected_calls):
    from sparkrun.orchestration import ssh

    first = ssh.RemoteResult("remote", 24, "", "file has vanished: weights\n")
    execute = Mock(side_effect=[first, ssh.RemoteResult("remote", 0, "", "")])
    guard = Mock(return_value=guard_result)
    monkeypatch.setattr(ssh, "_run_subprocess", execute)
    result = ssh.run_rsync("/source", "remote", "/target", rsync_options=["-r"], timeout=20, vanished_retry_guard=guard)
    assert execute.call_count == expected_calls
    guard.assert_called_once_with()
    assert result.returncode == (0 if guard_result else 24)
    if guard_result:
        assert execute.call_args.kwargs["timeout"] <= 20


def test_no_vanished_retry_without_inventory_guard(monkeypatch):
    from sparkrun.orchestration import ssh

    execute = Mock(return_value=ssh.RemoteResult("remote", 24, "", "file has vanished: weights\n"))
    monkeypatch.setattr(ssh, "_run_subprocess", execute)
    assert ssh.run_rsync("/source", "remote", "/target").returncode == 24
    assert execute.call_count == 1
