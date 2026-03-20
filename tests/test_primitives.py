"""Unit tests for sparkrun.orchestration.primitives module."""

from unittest.mock import patch, MagicMock

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.orchestration.primitives import check_tcp_reachability, find_available_port


# ---------------------------------------------------------------------------
# check_tcp_reachability tests
# ---------------------------------------------------------------------------


def test_check_tcp_reachability_all_reachable():
    """All IPs reachable returns all True."""
    def mock_connect(addr):
        pass  # success

    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock.__enter__ = lambda s: mock_sock
        mock_sock.__exit__ = lambda s, *a: None
        mock_sock.connect = mock_connect
        mock_sock_cls.return_value = mock_sock

        result = check_tcp_reachability(["10.0.0.1", "10.0.0.2"])

    assert result == {"10.0.0.1": True, "10.0.0.2": True}


def test_check_tcp_reachability_some_unreachable():
    """Mixed results: some reachable, some not."""
    call_count = {"n": 0}

    def mock_connect(addr):
        call_count["n"] += 1
        if addr[0] == "10.0.0.2":
            raise OSError("Connection refused")

    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock.__enter__ = lambda s: mock_sock
        mock_sock.__exit__ = lambda s, *a: None
        mock_sock.connect = mock_connect
        mock_sock_cls.return_value = mock_sock

        result = check_tcp_reachability(["10.0.0.1", "10.0.0.2"])

    assert result["10.0.0.1"] is True
    assert result["10.0.0.2"] is False


def test_check_tcp_reachability_empty():
    """Empty input returns empty dict."""
    result = check_tcp_reachability([])
    assert result == {}


def _make_result(success: bool) -> RemoteResult:
    """Helper to create a RemoteResult for port check mocking."""
    return RemoteResult(
        host="testhost",
        returncode=0 if success else 1,
        stdout="" if success else "",
        stderr="" if not success else "",
    )


# ---------------------------------------------------------------------------
# find_available_port tests
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_first_free(mock_cmd):
    """Port available on first try returns original port."""
    mock_cmd.return_value = _make_result(success=False)  # nc fails = port free

    result = find_available_port("myhost", 46379)

    assert result == 46379
    mock_cmd.assert_called_once()


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_second_free(mock_cmd):
    """Port occupied on first try, free on second returns port+1."""
    mock_cmd.side_effect = [
        _make_result(success=True),   # 46379 occupied
        _make_result(success=False),  # 46380 free
    ]

    result = find_available_port("myhost", 46379)

    assert result == 46380
    assert mock_cmd.call_count == 2


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_third_free(mock_cmd):
    """Two ports occupied, third free returns port+2."""
    mock_cmd.side_effect = [
        _make_result(success=True),   # 46379 occupied
        _make_result(success=True),   # 46380 occupied
        _make_result(success=False),  # 46381 free
    ]

    result = find_available_port("myhost", 46379)

    assert result == 46381
    assert mock_cmd.call_count == 3


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_all_occupied(mock_cmd):
    """All ports occupied returns original port with warning."""
    mock_cmd.return_value = _make_result(success=True)  # always occupied

    result = find_available_port("myhost", 46379, max_attempts=10)

    assert result == 46379  # falls back to original
    assert mock_cmd.call_count == 10


def test_find_available_port_dry_run():
    """Dry run returns original port without any SSH calls."""
    result = find_available_port("myhost", 46379, dry_run=True)

    assert result == 46379


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_passes_ssh_kwargs(mock_cmd):
    """SSH kwargs are forwarded to run_remote_command."""
    mock_cmd.return_value = _make_result(success=False)  # port free

    ssh_kw = {"ssh_user": "admin", "ssh_key": "/path/to/key"}
    find_available_port("myhost", 8000, ssh_kwargs=ssh_kw)

    mock_cmd.assert_called_once()
    _, kwargs = mock_cmd.call_args
    assert kwargs["ssh_user"] == "admin"
    assert kwargs["ssh_key"] == "/path/to/key"


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_custom_max_attempts(mock_cmd):
    """Custom max_attempts limits the number of checks."""
    mock_cmd.return_value = _make_result(success=True)  # always occupied

    result = find_available_port("myhost", 5000, max_attempts=3)

    assert result == 5000  # falls back to original
    assert mock_cmd.call_count == 3


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_checks_correct_ports(mock_cmd):
    """Verify the correct port numbers are checked in sequence."""
    mock_cmd.side_effect = [
        _make_result(success=True),   # 25000 occupied
        _make_result(success=True),   # 25001 occupied
        _make_result(success=False),  # 25002 free
    ]

    result = find_available_port("myhost", 25000)

    assert result == 25002
    # Verify the nc commands checked the right ports
    calls = mock_cmd.call_args_list
    assert "nc -z localhost 25000" in calls[0].args[1]
    assert "nc -z localhost 25001" in calls[1].args[1]
    assert "nc -z localhost 25002" in calls[2].args[1]
