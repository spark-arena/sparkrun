"""The executable half of the transport seam.

``Transport.prepare`` makes a host reachable; a ``HostSession`` executes on it.
Everything sparkrun itself runs remotely goes through ``orchestration.ssh`` as
a *script* piped to ``bash -s``. A session exists for the callers that need
**exact argv** instead — a managed binary invoked with structured arguments,
where re-quoting through a generated shell script is a correctness hazard
rather than a convenience.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.transports import open_cluster_host_session, resolve_transport
from sparkrun.transports.session import HostCommandResult, HostSessionError, SshHostSession


def _popen(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate.return_value = (stdout, stderr)
    proc.poll.return_value = returncode
    proc.pid = 4242
    return proc


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_default_transport_supplies_an_ssh_session():
    """Every existing cluster gets a session without declaring anything."""
    session = resolve_transport("ssh").open_host_session(None)
    assert isinstance(session, SshHostSession)
    assert session.provider_name == "sparkrun-ssh"


def test_open_cluster_host_session_threads_ssh_settings():
    cluster = ClusterDefinition(name="c", hosts=["h1"], user="dgxuser")
    session = open_cluster_host_session(cluster, ssh_kwargs={"ssh_user": "dgxuser", "ssh_key": "/k"})
    assert session.ssh_user == "dgxuser"
    assert session.ssh_key == "/k"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


@patch("sparkrun.transports.session.should_run_locally", return_value=False)
@patch("subprocess.Popen")
def test_remote_execute_quotes_every_argument(mock_popen, _local):
    mock_popen.return_value = _popen(stdout=b"out")
    session = SshHostSession(ssh_user="dgxuser")

    result = session.execute("h1", ["mytool", "--path", "/a b/c", "--json", '{"k": 1}'])

    assert result == HostCommandResult("h1", 0, b"out", b"")
    # argv is collapsed into one remote command string, so each element has to
    # survive the remote shell's own word splitting.
    remote_command = mock_popen.call_args.args[0][-1]
    assert "'/a b/c'" in remote_command
    assert "'{\"k\": 1}'" in remote_command


@patch("sparkrun.transports.session.should_run_locally", return_value=True)
@patch("subprocess.Popen")
def test_local_execute_bypasses_the_shell_entirely(mock_popen, _local):
    """No SSH hop means no shell, so argv is passed through verbatim."""
    mock_popen.return_value = _popen()
    SshHostSession().execute("localhost", ["mytool", "--path", "/a b/c"])

    assert mock_popen.call_args.args[0] == ["mytool", "--path", "/a b/c"]


@pytest.mark.parametrize(
    "arguments",
    [[], ["ok", "nul\x00byte"], ["ok", 7]],
    ids=["empty", "nul-byte", "non-string"],
)
def test_execute_rejects_unrunnable_argv(arguments):
    with pytest.raises(HostSessionError):
        SshHostSession().execute("h1", arguments)


@patch("sparkrun.transports.session.should_run_locally", return_value=False)
@patch("subprocess.Popen")
def test_execute_reports_the_exit_code_rather_than_raising(mock_popen, _local):
    """A non-zero command is data; only an unusable session is an error."""
    mock_popen.return_value = _popen(returncode=3, stderr=b"nope")

    result = SshHostSession().execute("h1", ["mytool"])

    assert result.returncode == 3
    assert result.stderr == b"nope"


# ---------------------------------------------------------------------------
# upload / docker_registry
# ---------------------------------------------------------------------------


@patch("sparkrun.transports.session.should_run_locally", return_value=True)
@patch("subprocess.Popen")
def test_local_upload_is_a_copy(mock_popen, _local):
    mock_popen.return_value = _popen()
    SshHostSession().upload("localhost", ["/src"], "/dst", recursive=True)

    assert mock_popen.call_args.args[0] == ["cp", "-R", "/src", "/dst"]


@patch("sparkrun.transports.session.should_run_locally", return_value=False)
@patch("subprocess.Popen")
def test_failed_upload_raises_with_the_host_and_output(mock_popen, _local):
    mock_popen.return_value = _popen(returncode=1, stderr=b"permission denied")

    with pytest.raises(HostSessionError, match="upload on h1 exited 1.*permission denied"):
        SshHostSession().upload("h1", ["/src"], "/dst")


@patch("sparkrun.transports.session.should_run_locally", return_value=False)
@patch("subprocess.Popen")
def test_docker_registry_runs_against_the_host_daemon(mock_popen, _local):
    """The pull happens *on the host*, using that host's credentials.

    Routing it through the control machine's daemon would pull the image to
    the wrong place, and a private image the node can reach may be one the
    control machine cannot.
    """
    mock_popen.return_value = _popen()
    SshHostSession(ssh_user="dgx user").docker_registry("h1", "pull", "org/img:tag")

    argv = mock_popen.call_args.args[0]
    assert argv[:2] == ["docker", "--host"]
    # The user becomes part of a URL, so it is percent-encoded, not shell-quoted.
    assert argv[2] == "ssh://dgx%20user@h1"
    assert argv[3:] == ["image", "pull", "org/img:tag"]
    assert "DOCKER_SSH_COMMAND" in mock_popen.call_args.kwargs["env"]


@pytest.mark.parametrize("operation", ["rmi", "push --force", ""])
def test_docker_registry_refuses_operations_it_does_not_define(operation):
    with pytest.raises(HostSessionError):
        SshHostSession().docker_registry("h1", operation, "org/img:tag")


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@patch("sparkrun.transports.session.should_run_locally", return_value=False)
@patch("subprocess.Popen")
def test_close_terminates_processes_still_in_flight(mock_popen, _local):
    """A session is the cancellation handle for the work it started.

    Without it a caller interrupted mid-operation leaves the remote work
    running with nothing left to observe or stop it — the orphaned-work
    failure the session guard exists to prevent, one layer up.
    """
    proc = _popen()
    proc.poll.return_value = None
    mock_popen.return_value = proc
    session = SshHostSession()
    session.execute("h1", ["mytool"])
    session._processes.add(proc)

    with patch("os.killpg") as mock_killpg:
        session.close()

    assert mock_killpg.called


@patch("sparkrun.transports.session.should_run_locally", return_value=False)
def test_a_closed_session_starts_nothing_further(_local):
    session = SshHostSession()
    session.close()

    with pytest.raises(HostSessionError, match="closed"):
        session.execute("h1", ["mytool"])
