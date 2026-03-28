"""Unit tests for sparkrun.orchestration.primitives module."""

from unittest.mock import patch, MagicMock

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.orchestration.distribution import _is_cross_user
from sparkrun.orchestration.primitives import check_tcp_reachability, find_available_port, should_run_locally


# ---------------------------------------------------------------------------
# should_run_locally tests
# ---------------------------------------------------------------------------


@patch.dict("os.environ", {"USER": "drew"})
def test_should_run_locally_local_no_user():
    """Local host with no ssh_user → True."""
    assert should_run_locally("127.0.0.1") is True
    assert should_run_locally("localhost") is True
    assert should_run_locally("") is True


@patch.dict("os.environ", {"USER": "drew"})
def test_should_run_locally_local_same_user():
    """Local host with ssh_user matching OS user → True."""
    assert should_run_locally("127.0.0.1", "drew") is True
    assert should_run_locally("localhost", "drew") is True


@patch.dict("os.environ", {"USER": "drew"})
def test_should_run_locally_local_different_user():
    """Local host with different ssh_user → False (needs SSH)."""
    assert should_run_locally("127.0.0.1", "dgxuser") is False
    assert should_run_locally("localhost", "dgxuser") is False


def test_should_run_locally_remote_host():
    """Remote host → always False regardless of user."""
    assert should_run_locally("10.0.0.1") is False
    assert should_run_locally("10.0.0.1", "drew") is False
    assert should_run_locally("10.0.0.1", None) is False


@patch.dict("os.environ", {"USER": "drew"})
def test_should_run_locally_none_user_explicit():
    """Explicit None ssh_user on local host → True."""
    assert should_run_locally("127.0.0.1", None) is True


# ---------------------------------------------------------------------------
# run_script_on_host / run_command_on_host cross-user dispatch tests
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.primitives.run_remote_script")
@patch("sparkrun.orchestration.primitives.run_local_script")
@patch.dict("os.environ", {"USER": "drew"})
def test_run_script_on_host_local_different_user_uses_ssh(mock_local, mock_remote):
    """localhost with different ssh_user dispatches to SSH, not local."""
    from sparkrun.orchestration.primitives import run_script_on_host

    mock_remote.return_value = RemoteResult(host="127.0.0.1", returncode=0, stdout="ok", stderr="")

    run_script_on_host("127.0.0.1", "echo test", ssh_kwargs={"ssh_user": "dgxuser"})

    mock_local.assert_not_called()
    mock_remote.assert_called_once()
    _, kwargs = mock_remote.call_args
    assert kwargs.get("ssh_user") == "dgxuser" or mock_remote.call_args[1].get("ssh_user") == "dgxuser"


@patch("sparkrun.orchestration.primitives.run_remote_script")
@patch("sparkrun.orchestration.primitives.run_local_script")
@patch.dict("os.environ", {"USER": "drew"})
def test_run_script_on_host_local_same_user_runs_locally(mock_local, mock_remote):
    """localhost with same ssh_user dispatches locally."""
    from sparkrun.orchestration.primitives import run_script_on_host

    mock_local.return_value = RemoteResult(host="localhost", returncode=0, stdout="ok", stderr="")

    run_script_on_host("127.0.0.1", "echo test", ssh_kwargs={"ssh_user": "drew"})

    mock_local.assert_called_once()
    mock_remote.assert_not_called()


@patch("sparkrun.orchestration.primitives.run_remote_command")
@patch("sparkrun.orchestration.primitives.run_local_script")
@patch.dict("os.environ", {"USER": "drew"})
def test_run_command_on_host_local_different_user_uses_ssh(mock_local, mock_remote):
    """localhost with different ssh_user dispatches to SSH."""
    from sparkrun.orchestration.primitives import run_command_on_host

    mock_remote.return_value = RemoteResult(host="127.0.0.1", returncode=0, stdout="ok", stderr="")

    run_command_on_host("127.0.0.1", "echo test", ssh_kwargs={"ssh_user": "dgxuser"})

    mock_local.assert_not_called()
    mock_remote.assert_called_once()


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


# ---------------------------------------------------------------------------
# _is_cross_user tests (distribution module)
# ---------------------------------------------------------------------------

@patch.dict("os.environ", {"USER": "drew"})
def test_is_cross_user_same_user():
    """Same user → False."""
    assert _is_cross_user({"ssh_user": "drew"}) is False


@patch.dict("os.environ", {"USER": "drew"})
def test_is_cross_user_different_user():
    """Different user → True."""
    assert _is_cross_user({"ssh_user": "dgxuser"}) is True


def test_is_cross_user_none():
    """No ssh_user → False."""
    assert _is_cross_user({"ssh_user": None}) is False
    assert _is_cross_user({}) is False
    assert _is_cross_user(None) is False


# ---------------------------------------------------------------------------
# distribute_resources cross-user auto-detection tests
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=True)
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value=None)
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.primitives.build_ssh_kwargs")
@patch("sparkrun.containers.distribute.distribute_image_from_head")
@patch("sparkrun.core.pending_ops.pending_op")
@patch.dict("os.environ", {"USER": "drew"})
def test_auto_detection_cross_user_forces_delegated(
    mock_pop, mock_dist_head, mock_ssh_kw, mock_detect, mock_validate, mock_in_cluster
):
    """Control in cluster + cross-user → auto-detects 'delegated', not 'local'."""
    from sparkrun.orchestration.distribution import distribute_resources

    # Setup mocks
    mock_ssh_kw.return_value = {"ssh_user": "dgxuser", "ssh_key": None, "ssh_options": None}
    ib_mock = MagicMock()
    ib_mock.nccl_env = {}
    ib_mock.ib_ip_map = {}
    ib_mock.mgmt_ip_map = {}
    mock_detect.return_value = ib_mock
    mock_dist_head.return_value = []  # success
    mock_pop.return_value.__enter__ = lambda s: s
    mock_pop.return_value.__exit__ = lambda s, *a: None

    config_mock = MagicMock()
    config_mock.cache_dir = "/tmp/cache"

    distribute_resources(
        image="test:latest",
        model="",
        host_list=["10.0.0.5", "10.0.0.6"],
        cache_dir="/tmp/cache",
        config=config_mock,
        dry_run=True,
        transfer_mode="auto",
    )

    # Should have called distribute_image_from_head (delegated), not distribute_image_from_local
    mock_dist_head.assert_called_once()


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=True)
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value=None)
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.primitives.build_ssh_kwargs")
@patch("sparkrun.containers.distribute.distribute_image_from_local")
@patch("sparkrun.core.pending_ops.pending_op")
@patch.dict("os.environ", {"USER": "drew"})
def test_auto_detection_same_user_stays_local(
    mock_pop, mock_dist_local, mock_ssh_kw, mock_detect, mock_validate, mock_in_cluster
):
    """Control in cluster + same user → auto-detects 'local'."""
    from sparkrun.orchestration.distribution import distribute_resources

    mock_ssh_kw.return_value = {"ssh_user": "drew", "ssh_key": None, "ssh_options": None}
    ib_mock = MagicMock()
    ib_mock.nccl_env = {}
    ib_mock.ib_ip_map = {}
    ib_mock.mgmt_ip_map = {}
    mock_detect.return_value = ib_mock
    mock_dist_local.return_value = []  # success
    mock_pop.return_value.__enter__ = lambda s: s
    mock_pop.return_value.__exit__ = lambda s, *a: None

    config_mock = MagicMock()
    config_mock.cache_dir = "/tmp/cache"

    distribute_resources(
        image="test:latest",
        model="",
        host_list=["10.0.0.5", "10.0.0.6"],
        cache_dir="/tmp/cache",
        config=config_mock,
        dry_run=True,
        transfer_mode="auto",
    )

    # Should have called distribute_image_from_local (local mode)
    mock_dist_local.assert_called_once()


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=True)
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value=None)
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.primitives.build_ssh_kwargs")
@patch("sparkrun.containers.distribute.distribute_image_from_local")
@patch("sparkrun.core.pending_ops.pending_op")
@patch.dict("os.environ", {"USER": "drew"})
def test_explicit_local_cross_user_warns(
    mock_pop, mock_dist_local, mock_ssh_kw, mock_detect, mock_validate, mock_in_cluster, caplog
):
    """Explicit transfer_mode='local' + cross-user → warning logged."""
    import logging
    from sparkrun.orchestration.distribution import distribute_resources

    mock_ssh_kw.return_value = {"ssh_user": "dgxuser", "ssh_key": None, "ssh_options": None}
    ib_mock = MagicMock()
    ib_mock.nccl_env = {}
    ib_mock.ib_ip_map = {}
    ib_mock.mgmt_ip_map = {}
    mock_detect.return_value = ib_mock
    mock_dist_local.return_value = []
    mock_pop.return_value.__enter__ = lambda s: s
    mock_pop.return_value.__exit__ = lambda s, *a: None

    config_mock = MagicMock()
    config_mock.cache_dir = "/tmp/cache"

    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.distribution"):
        distribute_resources(
            image="test:latest",
            model="",
            host_list=["10.0.0.5", "10.0.0.6"],
            cache_dir="/tmp/cache",
            config=config_mock,
            dry_run=True,
            transfer_mode="local",
        )

    assert any("transfer_mode='local'" in r.message and "dgxuser" in r.message for r in caplog.records)
    # Should still use local mode (explicit choice respected)
    mock_dist_local.assert_called_once()


# ---------------------------------------------------------------------------
# _stop_by_cluster_id cross-user dispatch test
# ---------------------------------------------------------------------------


@patch.dict("os.environ", {"USER": "drew"})
def test_stop_by_cluster_id_cross_user_uses_ssh():
    """_stop_by_cluster_id: localhost + different ssh_user → SSH cleanup path."""
    from sparkrun.cli._stop_logs import _stop_by_cluster_id

    config_mock = MagicMock()
    config_mock.cache_dir = "/tmp/cache"
    config_mock.ssh_user = "dgxuser"
    config_mock.ssh_key = None
    config_mock.ssh_options = None

    with (
        patch("sparkrun.orchestration.job_metadata.load_job_metadata", return_value={"hosts": ["127.0.0.1"]}),
        patch("sparkrun.orchestration.job_metadata.remove_job_metadata"),
        patch("sparkrun.cli._stop_logs.resolve_hosts_with_metadata_fallback", return_value=["127.0.0.1"]),
        patch("sparkrun.orchestration.docker.enumerate_cluster_containers", return_value=["c1_solo"]),
        patch("sparkrun.orchestration.primitives.cleanup_containers") as mock_remote,
        patch("sparkrun.orchestration.primitives.cleanup_containers_local") as mock_local,
    ):
        _stop_by_cluster_id("abc12345", None, None, None, config_mock, dry_run=True)

        mock_local.assert_not_called()
        mock_remote.assert_called_once()
