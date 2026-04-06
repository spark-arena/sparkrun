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
