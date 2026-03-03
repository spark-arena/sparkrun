"""Unit tests for sparkrun.orchestration.infiniband module."""

from unittest.mock import patch

import pytest
from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
    validate_ib_connectivity,
)
from sparkrun.orchestration.ssh import RemoteResult


def test_generate_ib_detect_script_is_bash():
    """Script starts with #!/bin/bash."""
    script = generate_ib_detect_script()
    assert script.startswith("#!/bin/bash")


def test_generate_ib_detect_script_contains_key_checks():
    """Script checks for show_gids and /sys/class/infiniband."""
    script = generate_ib_detect_script()

    assert "show_gids" in script
    assert "/sys/class/infiniband" in script
    assert "IB_DETECTED" in script


def test_parse_ib_detect_output_with_ib():
    """Parse output with IB_DETECTED=1 and all DETECTED_* values."""
    output = """
IB_DETECTED=1
DETECTED_GID_INDEX=3
DETECTED_HCA_LIST=mlx5_0,mlx5_1
DETECTED_SOCKET_IFNAME=eth0
DETECTED_NET_LIST=ib0,ib1
DETECTED_UCX_LIST=mlx5_0:1,mlx5_1:1
"""
    result = parse_ib_detect_output(output)

    assert result["IB_DETECTED"] == "1"
    assert result["DETECTED_GID_INDEX"] == "3"
    assert result["DETECTED_HCA_LIST"] == "mlx5_0,mlx5_1"
    assert result["DETECTED_SOCKET_IFNAME"] == "eth0"
    assert result["DETECTED_NET_LIST"] == "ib0,ib1"
    assert result["DETECTED_UCX_LIST"] == "mlx5_0:1,mlx5_1:1"


def test_parse_ib_detect_output_without_ib():
    """Parse output with IB_DETECTED=0."""
    output = "IB_DETECTED=0\n"
    result = parse_ib_detect_output(output)

    assert result["IB_DETECTED"] == "0"
    assert len(result) == 1


def test_parse_ib_detect_output_empty():
    """Empty string returns empty dict."""
    result = parse_ib_detect_output("")
    assert result == {}


def test_parse_ib_detect_output_ignores_comments():
    """Lines starting with # are ignored."""
    output = """
# This is a comment
IB_DETECTED=1
# Another comment
DETECTED_GID_INDEX=0
"""
    result = parse_ib_detect_output(output)

    assert result["IB_DETECTED"] == "1"
    assert result["DETECTED_GID_INDEX"] == "0"
    assert len(result) == 2


def test_generate_nccl_env_with_ib():
    """Full IB info produces NCCL_NET, NCCL_IB_DISABLE=0, NCCL_IB_GID_INDEX, NCCL_IB_HCA, etc."""
    ib_info = {
        "IB_DETECTED": "1",
        "DETECTED_GID_INDEX": "3",
        "DETECTED_HCA_LIST": "mlx5_0,mlx5_1",
        "DETECTED_SOCKET_IFNAME": "eth0",
        "DETECTED_NET_LIST": "ib0,ib1",
        "DETECTED_UCX_LIST": "mlx5_0:1,mlx5_1:1",
    }
    env = generate_nccl_env(ib_info)

    # Check mandatory NCCL vars
    assert env["NCCL_IGNORE_CPU_AFFINITY"] == "1"
    assert env["NCCL_NET"] == "IB"
    assert env["NCCL_IB_DISABLE"] == "0"
    assert env["NCCL_CROSS_NIC"] == "1"

    # Check detected values
    assert env["NCCL_IB_GID_INDEX"] == "3"
    assert env["NCCL_IB_HCA"] == "mlx5_0,mlx5_1"
    assert env["NCCL_SOCKET_IFNAME"] == "=ib0,=ib1"
    assert env["MN_IF_NAME"] == "ib0,ib1"
    assert env["OMPI_MCA_btl_tcp_if_include"] == "ib0,ib1"
    assert env["GLOO_SOCKET_IFNAME"] == "ib0,ib1"
    assert env["TP_SOCKET_IFNAME"] == "ib0,ib1"
    assert env["UCX_NET_DEVICES"] == "mlx5_0:1,mlx5_1:1"


def test_generate_nccl_env_without_ib():
    """IB_DETECTED=0 returns empty dict."""
    ib_info = {"IB_DETECTED": "0"}
    env = generate_nccl_env(ib_info)

    assert env == {}


def test_generate_nccl_env_partial_info():
    """Only some DETECTED_* values present."""
    ib_info = {
        "IB_DETECTED": "1",
        "DETECTED_GID_INDEX": "0",
        "DETECTED_HCA_LIST": "mlx5_0",
        # Missing DETECTED_NET_LIST and others
    }
    env = generate_nccl_env(ib_info)

    # Should have base NCCL vars
    assert env["NCCL_NET"] == "IB"
    assert env["NCCL_IB_DISABLE"] == "0"

    # Should have available fields
    assert env["NCCL_IB_GID_INDEX"] == "0"
    assert env["NCCL_IB_HCA"] == "mlx5_0"

    # Should not have missing fields
    assert "MN_IF_NAME" not in env
    assert "UCX_NET_DEVICES" not in env


def test_generate_nccl_env_missing_ib_detected():
    """Missing IB_DETECTED key returns empty dict."""
    ib_info = {"DETECTED_GID_INDEX": "3"}
    env = generate_nccl_env(ib_info)

    assert env == {}


def test_parse_ib_detect_output_with_whitespace():
    """Parse output handles whitespace correctly."""
    output = """
  IB_DETECTED=1
DETECTED_GID_INDEX  =  0
DETECTED_HCA_LIST=mlx5_0
"""
    result = parse_ib_detect_output(output)

    assert result["IB_DETECTED"] == "1"
    assert result["DETECTED_GID_INDEX"] == "0"
    assert result["DETECTED_HCA_LIST"] == "mlx5_0"


# ---------------------------------------------------------------------------
# validate_ib_connectivity tests
# ---------------------------------------------------------------------------

class TestValidateIbConnectivity:
    """Tests for validate_ib_connectivity."""

    def test_empty_map_returns_empty(self):
        """Empty ib_ip_map returns empty dict without any SSH calls."""
        result = validate_ib_connectivity({})
        assert result == {}

    def test_dry_run_returns_map_unchanged(self):
        """Dry-run mode skips the connectivity check."""
        ib_map = {"spark1": "10.0.0.1", "spark2": "10.0.0.2"}
        result = validate_ib_connectivity(ib_map, dry_run=True)
        assert result == ib_map

    @patch("sparkrun.orchestration.ssh.run_remote_command")
    def test_reachable_returns_original_map(self, mock_cmd):
        """When IB IP is reachable, returns the original map."""
        mock_cmd.return_value = RemoteResult(
            host="10.0.0.1", returncode=0, stdout="", stderr="",
        )
        ib_map = {"spark1": "10.0.0.1", "spark2": "10.0.0.2"}
        result = validate_ib_connectivity(ib_map, ssh_kwargs={"ssh_user": "user"})

        assert result == ib_map
        mock_cmd.assert_called_once_with(
            "10.0.0.1", "true",
            connect_timeout=5, timeout=10,
            ssh_user="user",
        )

    @patch("sparkrun.orchestration.ssh.run_remote_command")
    def test_unreachable_returns_empty(self, mock_cmd):
        """When IB IP is unreachable, returns empty dict (fallback)."""
        mock_cmd.return_value = RemoteResult(
            host="10.0.0.1", returncode=255,
            stdout="", stderr="Connection timed out",
        )
        ib_map = {"spark1": "10.0.0.1", "spark2": "10.0.0.2"}
        result = validate_ib_connectivity(ib_map)

        assert result == {}

    @patch("sparkrun.orchestration.ssh.run_remote_command")
    def test_ssh_kwargs_passed_through(self, mock_cmd):
        """SSH kwargs are forwarded to the connectivity check."""
        mock_cmd.return_value = RemoteResult(
            host="10.0.0.1", returncode=0, stdout="", stderr="",
        )
        ssh_kw = {"ssh_user": "drew", "ssh_key": "/path/to/key", "ssh_options": ["-o", "Foo=bar"]}
        ib_map = {"spark1": "10.0.0.1"}
        validate_ib_connectivity(ib_map, ssh_kwargs=ssh_kw)

        mock_cmd.assert_called_once_with(
            "10.0.0.1", "true",
            connect_timeout=5, timeout=10,
            ssh_user="drew", ssh_key="/path/to/key", ssh_options=["-o", "Foo=bar"],
        )
