"""Tests for the remote session guard (issue #240).

Remote payloads run via ``ssh <host> bash -s`` — without a PTY — so sshd sends
no signal to its child on disconnect and a killed ``sparkrun`` on the control
node would otherwise leave ``hf download`` / ``docker pull`` / rsync fan-outs
running on the cluster.  See ``orchestration.ssh.wrap_with_session_guard``.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from sparkrun.orchestration.ssh import (
    NO_SESSION_GUARD_ENV,
    RemoteResult,
    run_remote_script,
    run_remote_script_streaming,
    run_remote_scripts_parallel,
    session_guard_disabled,
    wrap_with_session_guard,
)

GUARD_MARKER = "sparkrun session guard"

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="requires a POSIX shell",
)


# ---------------------------------------------------------------------------
# wrap_with_session_guard
# ---------------------------------------------------------------------------


def test_wrap_embeds_payload_and_consumes_sentinel():
    wrapped = wrap_with_session_guard("echo hello\n")

    assert "echo hello" in wrapped
    assert GUARD_MARKER in wrapped
    # The sentinel must be substituted, not left behind as a stray command.
    assert "__SPARKRUN_PAYLOAD__" not in wrapped


def test_wrap_separates_payload_without_trailing_newline():
    """A payload with no trailing newline must not run into the closing paren."""
    wrapped = wrap_with_session_guard("echo hello")

    assert "echo hello\n)" in wrapped


def test_wrap_nests_rather_than_deduplicating():
    """Wrapping twice nests, so each call site must wrap exactly once.

    Enforced structurally: the only wrap points are the three dispatch
    functions in ``orchestration.ssh``, and none of them calls another.
    """
    once = wrap_with_session_guard("echo hi")
    twice = wrap_with_session_guard(once)

    assert once.count(GUARD_MARKER) == 1
    assert twice.count(GUARD_MARKER) == 2


def test_guard_script_has_no_curly_braces():
    """Repo convention: embedded scripts stay safe to run through str.format()."""
    from sparkrun.scripts import read_script

    guard = read_script("session_guard.sh")
    assert "{" not in guard
    assert "}" not in guard


def test_kill_switch_returns_script_unchanged(monkeypatch):
    monkeypatch.setenv(NO_SESSION_GUARD_ENV, "1")

    assert session_guard_disabled() is True
    assert wrap_with_session_guard("echo hello") == "echo hello"


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_kill_switch_falsy_values_keep_guard_on(monkeypatch, value):
    monkeypatch.setenv(NO_SESSION_GUARD_ENV, value)

    assert session_guard_disabled() is False
    assert GUARD_MARKER in wrap_with_session_guard("echo hello")


# ---------------------------------------------------------------------------
# Real-shell behaviour: the guard must be transparent
# ---------------------------------------------------------------------------


def _run_guarded(payload: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-s"],
        input=wrap_with_session_guard(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )


@needs_bash
def test_guard_passes_through_stdout_stderr_and_rc():
    proc = _run_guarded("echo to-out\necho to-err >&2\n")

    assert proc.returncode == 0
    assert proc.stdout.strip() == "to-out"
    assert proc.stderr.strip() == "to-err"


@needs_bash
def test_guard_propagates_nonzero_rc():
    proc = _run_guarded("echo failing\nexit 17\n")

    assert proc.returncode == 17
    assert proc.stdout.strip() == "failing"


@needs_bash
def test_guard_propagates_early_exit_zero():
    """model_sync.sh's cache-hit fast path exits 0 from inside the payload."""
    proc = _run_guarded('echo cached\nexit 0\necho "NOT REACHED"\n')

    assert proc.returncode == 0
    assert "NOT REACHED" not in proc.stdout


@needs_bash
def test_guard_preserves_payload_shell_options():
    """`set -uo pipefail` inside the payload must not leak out, or vice versa."""
    proc = _run_guarded("set -uo pipefail\necho ${MISSING_VAR:-default}\n")

    assert proc.returncode == 0
    assert proc.stdout.strip() == "default"


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_run_remote_script_wraps_when_enabled(mock_run):
    mock_run.return_value = RemoteResult(host="h1", returncode=0, stdout="", stderr="")

    run_remote_script("h1", "echo payload", session_guard=True)

    sent = mock_run.call_args[1]["input_data"]
    assert GUARD_MARKER in sent
    assert "echo payload" in sent


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_run_remote_script_unwrapped_by_default(mock_run):
    mock_run.return_value = RemoteResult(host="h1", returncode=0, stdout="", stderr="")

    run_remote_script("h1", "echo payload")

    assert mock_run.call_args[1]["input_data"] == "echo payload"


@patch("sparkrun.orchestration.ssh.run_local_script")
@patch("sparkrun.orchestration.ssh.should_run_locally", return_value=True)
def test_local_dispatch_is_never_wrapped(_mock_local, mock_run_local):
    """A local run has no SSH session to lose — wrapping would be dead weight."""
    mock_run_local.return_value = RemoteResult(host="localhost", returncode=0, stdout="", stderr="")

    run_remote_script("localhost", "echo payload", allow_local=True, session_guard=True)

    assert mock_run_local.call_args[0][0] == "echo payload"


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_dry_run_does_not_wrap(mock_run):
    result = run_remote_script("h1", "echo payload", session_guard=True, dry_run=True)

    assert result.stdout == "[dry-run]"
    mock_run.assert_not_called()


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_streaming_wraps_when_enabled(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")

    run_remote_script_streaming("h1", "echo payload", session_guard=True, quiet=True)

    sent = mock_run.call_args[1]["input"].decode()
    assert GUARD_MARKER in sent
    assert "echo payload" in sent


@patch("sparkrun.orchestration.ssh.run_remote_script")
def test_parallel_propagates_session_guard(mock_one):
    mock_one.return_value = RemoteResult(host="h1", returncode=0, stdout="", stderr="")

    run_remote_scripts_parallel(["h1", "h2"], "echo payload", session_guard=True)

    assert all(call[1]["session_guard"] is True for call in mock_one.call_args_list)


# ---------------------------------------------------------------------------
# Call sites that must opt in
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.primitives.run_remote_scripts_parallel")
def test_sync_resource_to_hosts_opts_in(mock_parallel):
    from sparkrun.orchestration.primitives import sync_resource_to_hosts

    mock_parallel.return_value = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]

    sync_resource_to_hosts("echo sync", ["h1"], "Model")

    assert mock_parallel.call_args[1]["session_guard"] is True


@patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
def test_distribute_from_head_opts_in_on_both_steps(mock_stream):
    from sparkrun.orchestration.distribution import _distribute_from_head

    mock_stream.return_value = RemoteResult(host="head", returncode=0, stdout="", stderr="")

    _distribute_from_head(
        head="head",
        hosts=["head", "w1"],
        ensure_script="echo ensure",
        distribute_script="echo distribute",
        resource_label="Model 'x'",
    )

    assert len(mock_stream.call_args_list) == 2
    assert all(call[1]["session_guard"] is True for call in mock_stream.call_args_list)


# ---------------------------------------------------------------------------
# Layer 2: SIGTERM/SIGHUP unwind like Ctrl-C
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_signal_handlers():
    saved = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            saved[sig] = signal.getsignal(sig)
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_install_termination_handlers_installs_on_default(restore_signal_handlers):
    from sparkrun.cli._common import _terminate_as_interrupt, install_termination_handlers

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    installed = install_termination_handlers()

    assert int(signal.SIGTERM) in installed
    assert signal.getsignal(signal.SIGTERM) is _terminate_as_interrupt


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_install_termination_handlers_does_not_clobber(restore_signal_handlers):
    from sparkrun.cli._common import install_termination_handlers

    sentinel = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        installed = install_termination_handlers()
        assert int(signal.SIGTERM) not in installed
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, sentinel)


def test_termination_handler_raises_keyboard_interrupt():
    """KeyboardInterrupt is what subprocess.run's cleanup path kills ssh on."""
    from sparkrun.cli._common import _terminate_as_interrupt

    with pytest.raises(KeyboardInterrupt):
        _terminate_as_interrupt(signal.SIGTERM, None)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
@pytest.mark.skipif(shutil.which("pgrep") is None, reason="requires pgrep")
def test_sigterm_reaps_the_subprocess_run_child():
    """End-to-end for layer 2: SIGTERM to the parent must reap the child.

    ``subprocess.run`` kills its child from the bare ``except`` in its cleanup
    path — that is the mechanism the handler relies on to drop the SSH session
    (and so trigger the remote guard on the far end).  ``sleep`` stands in for
    the ``ssh`` client here.
    """
    import time

    harness = (
        "import subprocess, sys\n"
        "from sparkrun.cli._common import install_termination_handlers\n"
        "assert install_termination_handlers()\n"
        "try:\n"
        "    subprocess.run(['sleep', '120'])\n"
        "except KeyboardInterrupt:\n"
        "    sys.exit(143)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", harness])
    try:
        children: list[str] = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not children:
            children = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
            if not children:
                time.sleep(0.1)
        assert children, "harness never spawned its `sleep` child"

        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 143, "SIGTERM did not unwind as KeyboardInterrupt"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            alive = [pid for pid in children if subprocess.run(["kill", "-0", pid], capture_output=True).returncode == 0]
            if not alive:
                break
            time.sleep(0.1)
        assert not alive, "subprocess.run child survived SIGTERM: %s" % alive
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@needs_bash
def test_streaming_guard_preserves_binary_stdin():
    payload = bytes(range(256)) * 1024
    result = subprocess.run(
        ["bash", "-c", wrap_with_session_guard("cat", preserve_stdin=True)],
        input=payload,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout == payload
