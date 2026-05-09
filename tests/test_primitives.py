"""Unit tests for sparkrun.orchestration.primitives module."""

from unittest.mock import patch, MagicMock

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.orchestration.distribution import _is_cross_user, resolve_auto_transfer_mode, TransferModeResult
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
        _make_result(success=True),  # 46379 occupied
        _make_result(success=False),  # 46380 free
    ]

    result = find_available_port("myhost", 46379)

    assert result == 46380
    assert mock_cmd.call_count == 2


@patch("sparkrun.orchestration.primitives.run_remote_command")
def test_find_available_port_third_free(mock_cmd):
    """Two ports occupied, third free returns port+2."""
    mock_cmd.side_effect = [
        _make_result(success=True),  # 46379 occupied
        _make_result(success=True),  # 46380 occupied
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
        _make_result(success=True),  # 25000 occupied
        _make_result(success=True),  # 25001 occupied
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
# resolve_auto_transfer_mode tests
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=True)
@patch.dict("os.environ", {"USER": "drew"})
def test_resolve_auto_cross_user_returns_delegated(mock_in_cluster):
    """Auto + control in cluster + cross-user → delegated."""
    result = resolve_auto_transfer_mode("auto", ["10.0.0.5"], ssh_kwargs={"ssh_user": "dgxuser"})
    assert isinstance(result, TransferModeResult)
    assert result.mode == "delegated"
    assert result.ib_result is None


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=True)
@patch.dict("os.environ", {"USER": "drew"})
def test_resolve_auto_same_user_returns_local(mock_in_cluster):
    """Auto + control in cluster + same user → local."""
    result = resolve_auto_transfer_mode("auto", ["10.0.0.5"], ssh_kwargs={"ssh_user": "drew"})
    assert result.mode == "local"
    assert result.ib_result is None


def test_resolve_explicit_mode_passthrough():
    """Explicit modes are returned unchanged."""
    assert resolve_auto_transfer_mode("local", ["10.0.0.5"]).mode == "local"
    assert resolve_auto_transfer_mode("delegated", ["10.0.0.5"]).mode == "delegated"
    assert resolve_auto_transfer_mode("push", ["10.0.0.5"]).mode == "push"


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=False)
@patch("sparkrun.orchestration.distribution._has_local_ib", return_value=False)
@patch.dict("os.environ", {"USER": "drew"})
def test_resolve_auto_external_no_local_ib_returns_delegated(mock_ib, mock_in_cluster):
    """Auto + external + same user + no local IB → delegated."""
    result = resolve_auto_transfer_mode("auto", ["10.0.0.5"], ssh_kwargs={"ssh_user": "drew"})
    assert result.mode == "delegated"
    assert result.ib_result is None


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=False)
@patch("sparkrun.orchestration.distribution._has_local_ib", return_value=True)
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity")
@patch.dict("os.environ", {"USER": "drew"})
def test_resolve_auto_external_with_local_ib_reachable(mock_validate, mock_detect, mock_ib, mock_in_cluster):
    """Auto + external + same user + has local IB + IB reachable → local with IB results."""
    from sparkrun.orchestration.infiniband import IBDetectionResult

    from sparkrun.orchestration.comm_env import ClusterCommEnv

    mock_detect.return_value = IBDetectionResult(
        comm_env=ClusterCommEnv(shared={"NCCL_IB_HCA": "mlx5_0"}),
        ib_ip_map={"10.0.0.5": "192.168.1.5"},
        mgmt_ip_map={"10.0.0.5": "10.0.0.5"},
    )
    mock_validate.return_value = {"10.0.0.5": "192.168.1.5"}
    result = resolve_auto_transfer_mode("auto", ["10.0.0.5"], ssh_kwargs={"ssh_user": "drew"})
    assert result.mode == "local"
    assert result.ib_result is not None
    assert result.ib_validated == {"10.0.0.5": "192.168.1.5"}
    assert not result.auto_delegated


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=False)
@patch("sparkrun.orchestration.distribution._has_local_ib", return_value=True)
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value={})
@patch.dict("os.environ", {"USER": "drew"})
def test_resolve_auto_external_with_local_ib_unreachable(mock_validate, mock_detect, mock_ib, mock_in_cluster):
    """Auto + external + same user + has local IB + IB unreachable → delegated with IB results."""
    from sparkrun.orchestration.infiniband import IBDetectionResult

    from sparkrun.orchestration.comm_env import ClusterCommEnv

    mock_detect.return_value = IBDetectionResult(
        comm_env=ClusterCommEnv(shared={"NCCL_IB_HCA": "mlx5_0"}),
        ib_ip_map={"10.0.0.5": "192.168.1.5"},
        mgmt_ip_map={"10.0.0.5": "10.0.0.5"},
    )
    result = resolve_auto_transfer_mode("auto", ["10.0.0.5"], ssh_kwargs={"ssh_user": "drew"})
    assert result.mode == "delegated"
    assert result.ib_result is not None
    assert result.ib_validated == {}
    assert result.auto_delegated


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=False)
@patch.dict("os.environ", {"USER": "drew"})
def test_resolve_auto_external_control_cross_user_returns_delegated(mock_in_cluster):
    """Auto + not in cluster + cross-user → delegated."""
    result = resolve_auto_transfer_mode("auto", ["10.0.0.5"], ssh_kwargs={"ssh_user": "dgxuser"})
    assert result.mode == "delegated"


@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=False)
@patch("sparkrun.orchestration.distribution._has_local_ib", return_value=True)
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value={"10.0.0.5": "192.168.1.5"})
@patch.dict("os.environ", {"USER": "drew"})
def test_resolve_auto_forwards_topology_to_ib_detect(mock_validate, mock_detect, mock_ib, mock_in_cluster):
    """Topology kwarg is forwarded into detect_ib_for_hosts so ring overrides apply."""
    from sparkrun.orchestration.infiniband import IBDetectionResult
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    mock_detect.return_value = IBDetectionResult(
        comm_env=ClusterCommEnv(shared={"NCCL_IB_HCA": "mlx5_0"}),
        ib_ip_map={"10.0.0.5": "192.168.1.5"},
        mgmt_ip_map={"10.0.0.5": "10.0.0.5"},
    )
    resolve_auto_transfer_mode("auto", ["10.0.0.5"], ssh_kwargs={"ssh_user": "drew"}, topology="ring")
    assert mock_detect.call_args.kwargs.get("topology") == "ring"


@patch("sparkrun.orchestration.distribution.is_local_host", return_value=False)
@patch("sparkrun.orchestration.distribution._is_cross_user", return_value=False)
@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=True)
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value={})
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
def test_distribute_from_config_forwards_topology_to_ib_detect(
    mock_ssh, mock_detect, mock_validate, mock_in_cluster, mock_cross, mock_local
):
    """distribute_from_config forwards topology to detect_ib_for_hosts when pre_ib lacks ib_result."""
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.orchestration.distribution import distribute_from_config
    from sparkrun.orchestration.infiniband import IBDetectionResult

    mock_detect.return_value = IBDetectionResult(
        comm_env=ClusterCommEnv.empty(),
        ib_ip_map={},
        mgmt_ip_map={},
    )

    # Mock recipe with disabled distribution so the function just runs IB detect and returns.
    recipe = MagicMock()
    dist_cfg = MagicMock()
    dist_cfg.containers.enabled = False
    dist_cfg.models.enabled = False
    recipe.distribution_config.resolve.return_value = dist_cfg

    config = MagicMock()
    config.cache_dir = "/tmp/cache"

    # pre_ib without ib_result forces the in-function IB detection path.
    pre_ib = TransferModeResult(mode="local", ib_result=None)

    distribute_from_config(
        recipe,
        "img:latest",
        ["h1", "h2"],
        "/cache",
        config,
        dry_run=True,
        transfer_mode="local",
        pre_ib=pre_ib,
        topology="ring",
    )

    assert mock_detect.call_args.kwargs.get("topology") == "ring"


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
def test_auto_detection_cross_user_forces_delegated(mock_pop, mock_dist_head, mock_ssh_kw, mock_detect, mock_validate, mock_in_cluster):
    """Control in cluster + cross-user → auto-detects 'delegated', not 'local'."""
    from sparkrun.orchestration.distribution import distribute_resources

    # Setup mocks
    mock_ssh_kw.return_value = {"ssh_user": "dgxuser", "ssh_key": None, "ssh_options": None}
    ib_mock = MagicMock()
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    ib_mock.comm_env = ClusterCommEnv.empty()
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
def test_auto_detection_same_user_stays_local(mock_pop, mock_dist_local, mock_ssh_kw, mock_detect, mock_validate, mock_in_cluster):
    """Control in cluster + same user → auto-detects 'local'."""
    from sparkrun.orchestration.distribution import distribute_resources

    mock_ssh_kw.return_value = {"ssh_user": "drew", "ssh_key": None, "ssh_options": None}
    ib_mock = MagicMock()
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    ib_mock.comm_env = ClusterCommEnv.empty()
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
def test_explicit_local_cross_user_warns(mock_pop, mock_dist_local, mock_ssh_kw, mock_detect, mock_validate, mock_in_cluster, caplog):
    """Explicit transfer_mode='local' + cross-user → warning logged."""
    import logging
    from sparkrun.orchestration.distribution import distribute_resources

    mock_ssh_kw.return_value = {"ssh_user": "dgxuser", "ssh_key": None, "ssh_options": None}
    ib_mock = MagicMock()
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    ib_mock.comm_env = ClusterCommEnv.empty()
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


# ---------------------------------------------------------------------------
# probe_remote_hf_cache tests
# ---------------------------------------------------------------------------


class TestProbeRemoteHfCache:
    """Tests for ``probe_remote_hf_cache`` — SSHes the head and returns the
    resolved HF cache path so callers don't leak the control machine's $HOME."""

    def test_returns_remote_path_on_success(self):
        from sparkrun.orchestration.primitives import probe_remote_hf_cache

        with patch(
            "sparkrun.orchestration.primitives.run_remote_command",
            return_value=RemoteResult(host="h", returncode=0, stdout="/home/user1/.cache/huggingface\n", stderr=""),
        ) as mock_run:
            path = probe_remote_hf_cache("h", ssh_user="user1")

        assert path == "/home/user1/.cache/huggingface"
        # Verify the probe command embeds HF_HOME fallback expression
        call_kwargs = mock_run.call_args
        assert "HF_HOME" in call_kwargs.args[1]
        assert "$HOME" in call_kwargs.args[1]

    def test_dry_run_skips_ssh(self):
        from sparkrun.orchestration.primitives import probe_remote_hf_cache

        with patch("sparkrun.orchestration.primitives.run_remote_command") as mock_run:
            path = probe_remote_hf_cache("h", dry_run=True)

        assert path  # falls back to local default, not None
        mock_run.assert_not_called()

    def test_failure_raises_runtime_error(self):
        import pytest
        from sparkrun.orchestration.primitives import probe_remote_hf_cache

        with patch(
            "sparkrun.orchestration.primitives.run_remote_command",
            return_value=RemoteResult(host="h", returncode=255, stdout="", stderr="ssh: connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Could not resolve remote HF cache"):
                probe_remote_hf_cache("h")

    def test_empty_output_raises_runtime_error(self):
        import pytest
        from sparkrun.orchestration.primitives import probe_remote_hf_cache

        with patch(
            "sparkrun.orchestration.primitives.run_remote_command",
            return_value=RemoteResult(host="h", returncode=0, stdout="\n", stderr=""),
        ):
            with pytest.raises(RuntimeError):
                probe_remote_hf_cache("h")

    def test_unsafe_path_rejected(self):
        """A remote that returns a path with shell metacharacters must be
        rejected before that path is fed to shlex.quote-aware downstream code."""
        import pytest
        from sparkrun.orchestration.primitives import probe_remote_hf_cache

        with patch(
            "sparkrun.orchestration.primitives.run_remote_command",
            return_value=RemoteResult(host="h", returncode=0, stdout="/tmp/$(whoami)/.cache/huggingface\n", stderr=""),
        ):
            with pytest.raises(ValueError, match="Unsafe character"):
                probe_remote_hf_cache("h")


# ---------------------------------------------------------------------------
# resolve_effective_cache_dir tests — launcher.py
# ---------------------------------------------------------------------------


class TestResolveEffectiveCacheDir:
    """Tests for ``core.launcher.resolve_effective_cache_dir`` — the single
    decision point that turns the (possibly None) cluster cache_dir into a
    concrete absolute path before it reaches build_volumes / docker -v / rsync."""

    def _config(self, hf_cache="/home/local/.cache/huggingface"):
        cfg = MagicMock()
        cfg.hf_cache_dir = hf_cache
        return cfg

    def test_explicit_cache_dir_skips_probe(self):
        """When the cluster sets cache_dir explicitly, that value is returned
        as-is — no probe, no fallback."""
        from sparkrun.core.launcher import resolve_effective_cache_dir

        with patch("sparkrun.orchestration.primitives.probe_remote_hf_cache") as mock_probe:
            result = resolve_effective_cache_dir(
                "/mnt/shared/hf",
                ["remote-host"],
                {"ssh_user": "user1"},
                self._config(),
            )

        assert result == "/mnt/shared/hf"
        mock_probe.assert_not_called()

    def test_local_same_user_uses_local_cache(self):
        """Single localhost target with same user → use control's hf_cache_dir,
        no SSH probe needed."""
        from sparkrun.core.launcher import resolve_effective_cache_dir

        with (
            patch("sparkrun.utils.is_local_host", return_value=True),
            patch("sparkrun.orchestration.primitives.probe_remote_hf_cache") as mock_probe,
            patch.dict("os.environ", {"USER": "drew"}),
        ):
            result = resolve_effective_cache_dir(
                None,
                ["localhost"],
                {"ssh_user": None},
                self._config("/home/drew/.cache/huggingface"),
            )

        assert result == "/home/drew/.cache/huggingface"
        mock_probe.assert_not_called()

    def test_remote_host_triggers_probe(self):
        """Non-local host with no explicit cache → probe the head."""
        from sparkrun.core.launcher import resolve_effective_cache_dir

        with (
            patch("sparkrun.utils.is_local_host", return_value=False),
            patch(
                "sparkrun.orchestration.primitives.probe_remote_hf_cache",
                return_value="/home/user1/.cache/huggingface",
            ) as mock_probe,
        ):
            result = resolve_effective_cache_dir(
                None,
                ["spark-01"],
                {"ssh_user": "user1"},
                self._config(),
            )

        assert result == "/home/user1/.cache/huggingface"
        mock_probe.assert_called_once()
        # Probe receives the head host
        assert mock_probe.call_args.args[0] == "spark-01"

    def test_local_host_cross_user_triggers_probe(self):
        """Even when target is localhost, a different SSH user means we need
        the probe (the target user's $HOME differs from ours)."""
        from sparkrun.core.launcher import resolve_effective_cache_dir

        with (
            patch("sparkrun.utils.is_local_host", return_value=True),
            patch(
                "sparkrun.orchestration.primitives.probe_remote_hf_cache",
                return_value="/home/other/.cache/huggingface",
            ) as mock_probe,
            patch.dict("os.environ", {"USER": "drew"}),
        ):
            result = resolve_effective_cache_dir(
                None,
                ["localhost"],
                {"ssh_user": "other"},
                self._config(),
            )

        assert result == "/home/other/.cache/huggingface"
        mock_probe.assert_called_once()
