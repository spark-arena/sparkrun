"""Process status checks must never signal a Windows process."""

import subprocess
import sys
from unittest.mock import Mock

import pytest

from sparkrun.utils import process


def test_probe_leaves_real_child_alive_until_explicit_termination():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert process.process_exists(child.pid)
        assert child.poll() is None
        assert process.process_exists(child.pid)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert not process.process_exists(child.pid)


@pytest.mark.parametrize("pid", [None, 0, -1, True])
def test_invalid_pid_cannot_target_a_process_group(pid, monkeypatch):
    kill = Mock(side_effect=AssertionError("must not signal"))
    monkeypatch.setattr(process.os, "kill", kill)
    assert not process.process_exists(pid)
    kill.assert_not_called()


def test_windows_dispatch_never_uses_kill_zero(monkeypatch):
    monkeypatch.setattr(process.sys, "platform", "win32")
    monkeypatch.setattr(process, "_windows_process_exists", lambda pid: pid == 42)
    kill = Mock(side_effect=AssertionError("must not signal"))
    monkeypatch.setattr(process.os, "kill", kill)
    assert process.process_exists(42)
    assert not process.process_exists(43)
    kill.assert_not_called()


def test_inaccessible_probe_preserves_callers_policy(monkeypatch):
    monkeypatch.setattr(process.sys, "platform", "win32")
    monkeypatch.setattr(process, "_windows_process_exists", Mock(side_effect=PermissionError()))
    assert not process.process_exists(42)
    assert process.process_exists(42, inaccessible=True)
