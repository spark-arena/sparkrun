"""Unit tests for sparkrun.orchestration.ssh module."""

import subprocess
from unittest.mock import MagicMock, patch

from sparkrun.orchestration.ssh import (
    build_ssh_cmd,
    RemoteResult,
    run_remote_script,
    run_remote_command,
    run_remote_scripts_parallel,
    run_remote_sudo_script,
    detect_sudo_on_hosts,
)


def test_build_ssh_cmd_basic():
    """Basic SSH command with just host."""
    cmd = build_ssh_cmd("192.168.1.100")

    assert cmd[0] == "ssh"
    assert "-o" in cmd
    assert "BatchMode=yes" in cmd
    assert "ConnectTimeout=10" in cmd
    assert "192.168.1.100" in cmd


def test_build_ssh_cmd_with_user():
    """With ssh_user."""
    cmd = build_ssh_cmd("192.168.1.100", ssh_user="root")

    assert "root@192.168.1.100" in cmd


def test_build_ssh_cmd_with_key():
    """With ssh_key."""
    cmd = build_ssh_cmd("192.168.1.100", ssh_key="/path/to/key.pem")

    assert "-i" in cmd
    assert "/path/to/key.pem" in cmd


def test_build_ssh_cmd_with_options():
    """With extra ssh_options."""
    cmd = build_ssh_cmd("192.168.1.100", ssh_options=["-v", "-o", "StrictHostKeyChecking=no"])

    assert "-v" in cmd
    assert "StrictHostKeyChecking=no" in cmd


def test_remote_result_success():
    """RemoteResult with returncode=0 is success."""
    result = RemoteResult(host="host1", returncode=0, stdout="OK", stderr="")
    assert result.success is True


def test_remote_result_failure():
    """RemoteResult with returncode=1 is not success."""
    result = RemoteResult(host="host1", returncode=1, stdout="", stderr="error")
    assert result.success is False


def test_remote_result_last_line():
    """Test last_line extraction from stdout."""
    result = RemoteResult(
        host="host1",
        returncode=0,
        stdout="line1\nline2\nline3\n",
        stderr="",
    )
    assert result.last_line == "line3"


def test_remote_result_last_line_empty():
    """Empty stdout returns empty string."""
    result = RemoteResult(host="host1", returncode=0, stdout="", stderr="")
    assert result.last_line == ""


def test_remote_result_last_line_with_blank_lines():
    """Test last_line ignores trailing blank lines."""
    result = RemoteResult(
        host="host1",
        returncode=0,
        stdout="line1\nline2\n\n\n",
        stderr="",
    )
    assert result.last_line == "line2"


def test_run_remote_script_dry_run():
    """Dry run returns success without subprocess."""
    result = run_remote_script(
        "192.168.1.100",
        "#!/bin/bash\necho test",
        dry_run=True,
    )

    assert result.success
    assert result.host == "192.168.1.100"
    assert result.stdout == "[dry-run]"
    assert result.returncode == 0


def test_run_remote_command_dry_run():
    """Dry run returns success without subprocess."""
    result = run_remote_command(
        "192.168.1.100",
        "echo test",
        dry_run=True,
    )

    assert result.success
    assert result.host == "192.168.1.100"
    assert result.stdout == "[dry-run]"
    assert result.returncode == 0


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_script_mocks_subprocess(mock_run):
    """Mock subprocess.run, verify ssh bash -s is called with script as input."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "output"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    script = "#!/bin/bash\necho hello"
    result = run_remote_script("192.168.1.100", script)

    # Verify subprocess.run was called
    assert mock_run.called
    call_args = mock_run.call_args

    # Check command structure
    cmd = call_args[0][0]
    assert cmd[0] == "ssh"
    assert "bash" in cmd
    assert "-s" in cmd

    # Check script was passed as input
    assert call_args[1]["input"] == script
    assert call_args[1]["text"] is True
    assert call_args[1]["capture_output"] is True

    # Verify result
    assert result.success
    assert result.stdout == "output"


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_command_mocks_subprocess(mock_run):
    """Mock subprocess.run, verify command execution."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "hello"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    result = run_remote_command("192.168.1.100", "echo hello")

    # Verify subprocess.run was called
    assert mock_run.called
    call_args = mock_run.call_args

    # Check command structure
    cmd = call_args[0][0]
    assert cmd[0] == "ssh"
    assert "echo hello" in cmd

    # Verify result
    assert result.success
    assert result.stdout == "hello"


def test_run_remote_scripts_parallel_dry_run():
    """Parallel dry run for multiple hosts."""
    hosts = ["host1", "host2", "host3"]
    script = "#!/bin/bash\necho test"

    results = run_remote_scripts_parallel(hosts, script, dry_run=True)

    assert len(results) == 3
    assert all(r.success for r in results)
    assert {r.host for r in results} == set(hosts)


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_scripts_parallel_mocks_subprocess(mock_run):
    """Test parallel execution with mocked subprocess."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "output"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    hosts = ["host1", "host2"]
    script = "#!/bin/bash\necho test"

    results = run_remote_scripts_parallel(hosts, script)

    # Should have called subprocess.run twice (once per host)
    assert mock_run.call_count == 2
    assert len(results) == 2
    assert all(r.success for r in results)


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_script_timeout(mock_run):
    """Test timeout handling."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ssh"], timeout=10)

    result = run_remote_script("192.168.1.100", "sleep 100", timeout=10)

    assert not result.success
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_script_exception(mock_run):
    """Test exception handling."""
    mock_run.side_effect = Exception("Connection refused")

    result = run_remote_script("192.168.1.100", "echo test")

    assert not result.success
    assert result.returncode == -1
    assert "Connection refused" in result.stderr


# ---------------------------------------------------------------------------
# run_remote_sudo_script tests
# ---------------------------------------------------------------------------


def test_run_remote_sudo_script_dry_run():
    """Dry run returns success without subprocess."""
    result = run_remote_sudo_script(
        "192.168.1.100",
        "#!/bin/bash\nchown -R user /cache",
        "mypassword",
        dry_run=True,
    )

    assert result.success
    assert result.host == "192.168.1.100"
    assert result.stdout == "[dry-run]"
    assert result.returncode == 0


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_sudo_script_mocks_subprocess(mock_run):
    """Mock subprocess.run, verify sudo -S bash -s is called with password+script."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "OK: fixed permissions"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    script = "chown -R user /cache"
    password = "secret123"
    result = run_remote_sudo_script("192.168.1.100", script, password, ssh_user="testuser")

    assert mock_run.called
    call_args = mock_run.call_args

    # Check command structure: ssh ... sudo -S bash -s
    cmd = call_args[0][0]
    assert cmd[0] == "ssh"
    assert "sudo" in cmd
    assert "-S" in cmd
    assert "bash" in cmd
    assert "-s" in cmd

    # Check password + script passed as input
    full_input = call_args[1]["input"]
    assert full_input.startswith(password + "\n")
    assert script in full_input

    assert result.success
    assert result.stdout == "OK: fixed permissions"


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_sudo_script_timeout(mock_run):
    """Test timeout handling for sudo script."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ssh"], timeout=60)

    result = run_remote_sudo_script(
        "192.168.1.100",
        "chown -R user /cache",
        "password",
        timeout=60,
    )

    assert not result.success
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_sudo_script_failure(mock_run):
    """Test handling of failed sudo command."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "[sudo] password for user: Sorry, try again."
    mock_run.return_value = mock_proc

    result = run_remote_sudo_script(
        "192.168.1.100",
        "chown -R user /cache",
        "wrongpassword",
        ssh_user="user",
    )

    assert not result.success
    assert result.returncode == 1


# ---------------------------------------------------------------------------
# detect_sudo_on_hosts tests
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_detect_sudo_on_hosts_mixed(mock_run):
    """Test detection with mixed NOPASSWD/password hosts."""

    def side_effect(cmd, **kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = ""
        # Determine host from the command — target is the arg before "bash"
        # cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "host1", "bash", "-s"]
        target = cmd[-3]  # host is 3 before end: [..., "host", "bash", "-s"]
        if "host1" in target:
            proc.stdout = "SUDO_OK=1\n"
        else:
            proc.stdout = "SUDO_OK=0\n"
        return proc

    mock_run.side_effect = side_effect

    result = detect_sudo_on_hosts(["host1", "host2"])

    assert result == {"host1"}


def test_detect_sudo_on_hosts_empty():
    """Empty host list returns empty set."""
    result = detect_sudo_on_hosts([])
    assert result == set()


def test_detect_sudo_on_hosts_dry_run():
    """Dry run returns empty set without executing."""
    result = detect_sudo_on_hosts(["host1", "host2"], dry_run=True)
    # Dry run: all results have stdout "[dry-run]" which doesn't contain SUDO_OK=1
    assert result == set()


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_detect_sudo_on_hosts_all_nopasswd(mock_run):
    """Test when all hosts have passwordless sudo."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "SUDO_OK=1\n"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    result = detect_sudo_on_hosts(["host1", "host2", "host3"])

    assert result == {"host1", "host2", "host3"}


# ---------------------------------------------------------------------------
# C1: SSH/rsync fan-out cap
# ---------------------------------------------------------------------------

from sparkrun.orchestration.ssh import (  # noqa: E402
    DEFAULT_MAX_PARALLEL_SSH,
    resolve_parallel_cap,
    run_rsync_parallel,
    run_pipeline_to_remotes_parallel,
)


def test_resolve_parallel_cap_small_cluster_unchanged():
    """Small clusters (<= cap) get one worker per host (behavior unchanged)."""
    assert resolve_parallel_cap(5) == 5
    assert resolve_parallel_cap(1) == 1


def test_resolve_parallel_cap_large_cluster_capped():
    """Large clusters are capped at the default."""
    assert resolve_parallel_cap(32) == DEFAULT_MAX_PARALLEL_SSH
    assert resolve_parallel_cap(100) == DEFAULT_MAX_PARALLEL_SSH


def test_resolve_parallel_cap_custom_override():
    """Explicit cap overrides the default."""
    assert resolve_parallel_cap(32, 8) == 8
    assert resolve_parallel_cap(4, 8) == 4


def test_resolve_parallel_cap_never_zero():
    """A zero/empty fan-out never produces max_workers=0."""
    assert resolve_parallel_cap(0) == 1
    assert resolve_parallel_cap(5, 0) == 5  # cap<=0 falls back to default, then min(5,20)=5


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_scripts_parallel_caps_workers(mock_run, monkeypatch):
    """With 32 hosts, the pool is capped at DEFAULT_MAX_PARALLEL_SSH."""
    import concurrent.futures

    captured = {}
    real_pool = concurrent.futures.ThreadPoolExecutor

    def _spy(max_workers=None, **kw):
        captured["max_workers"] = max_workers
        return real_pool(max_workers=max_workers, **kw)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _spy)

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")
    mock_run.return_value = mock_proc

    hosts = [f"host{i}" for i in range(32)]
    run_remote_scripts_parallel(hosts, "echo x")
    assert captured["max_workers"] == DEFAULT_MAX_PARALLEL_SSH


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_rsync_parallel_caps_workers(mock_run, monkeypatch):
    """run_rsync_parallel also caps the pool size."""
    import concurrent.futures

    captured = {}
    real_pool = concurrent.futures.ThreadPoolExecutor

    def _spy(max_workers=None, **kw):
        captured["max_workers"] = max_workers
        return real_pool(max_workers=max_workers, **kw)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _spy)
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    hosts = [f"host{i}" for i in range(40)]
    run_rsync_parallel("/src", hosts, "/dst")
    assert captured["max_workers"] == DEFAULT_MAX_PARALLEL_SSH


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_pipeline_parallel_caps_workers(mock_run, monkeypatch):
    """run_pipeline_to_remotes_parallel caps the pool size."""
    import concurrent.futures

    captured = {}
    real_pool = concurrent.futures.ThreadPoolExecutor

    def _spy(max_workers=None, **kw):
        captured["max_workers"] = max_workers
        return real_pool(max_workers=max_workers, **kw)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _spy)
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    hosts = [f"host{i}" for i in range(25)]
    run_pipeline_to_remotes_parallel(hosts, "docker save x", "docker load")
    assert captured["max_workers"] == DEFAULT_MAX_PARALLEL_SSH


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_scripts_parallel_explicit_cap(mock_run, monkeypatch):
    """An explicit max_workers overrides the default cap."""
    import concurrent.futures

    captured = {}
    real_pool = concurrent.futures.ThreadPoolExecutor

    def _spy(max_workers=None, **kw):
        captured["max_workers"] = max_workers
        return real_pool(max_workers=max_workers, **kw)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _spy)
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    hosts = [f"host{i}" for i in range(16)]
    run_remote_scripts_parallel(hosts, "echo x", max_workers=4)
    assert captured["max_workers"] == 4


# ---------------------------------------------------------------------------
# C3: execution timeout on a hung host
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_parallel_hung_host_hits_timeout(mock_run):
    """A host that connects then hangs surfaces as a timeout, not a block.

    subprocess.run raising TimeoutExpired for one host must not prevent the
    other hosts' results from being collected, and the hung host's result
    must report failure.
    """

    def _side_effect(*args, **kwargs):
        # Simulate: host with 'hang' in stdin/script times out, others OK.
        # We can't easily inspect the host here, so alternate behavior by a
        # mutable counter.
        if not hasattr(_side_effect, "n"):
            _side_effect.n = 0
        _side_effect.n += 1
        if _side_effect.n == 1:
            raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _side_effect

    hosts = ["h1", "h2", "h3"]
    results = run_remote_scripts_parallel(hosts, "sleep 100", timeout=5)
    assert len(results) == 3
    # Exactly one host timed out (rc=-1), the rest succeeded.
    timed_out = [r for r in results if r.returncode == -1]
    assert len(timed_out) == 1
    assert "timed out" in timed_out[0].stderr.lower()
