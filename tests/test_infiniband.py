"""Unit tests for sparkrun.orchestration.infiniband module."""

from unittest.mock import patch

from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
    generate_ring_nccl_overrides,
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
    # NCCL_SOCKET_IFNAME = mgmt first, then IB adapters as fallback
    assert env["NCCL_SOCKET_IFNAME"] == "eth0,ib0,ib1"
    assert env["MN_IF_NAME"] == "eth0"
    assert env["OMPI_MCA_btl_tcp_if_include"] == "eth0"
    assert env["GLOO_SOCKET_IFNAME"] == "eth0"
    assert env["TP_SOCKET_IFNAME"] == "eth0"
    assert env["UCX_NET_DEVICES"] == "mlx5_0:1,mlx5_1:1"


def test_nccl_socket_ifname_dedupes_mgmt_in_ib_list():
    """When mgmt iface also appears in DETECTED_NET_LIST, it isn't duplicated."""
    ib_info = {
        "IB_DETECTED": "1",
        "DETECTED_SOCKET_IFNAME": "ib0",  # mgmt path happens to be ib0
        "DETECTED_NET_LIST": "ib0,ib1",
    }
    env = generate_nccl_env(ib_info)
    assert env["NCCL_SOCKET_IFNAME"] == "ib0,ib1"


def test_nccl_socket_ifname_falls_back_to_ib_list_without_mgmt():
    """When no DETECTED_SOCKET_IFNAME is found, fall back to IB list only."""
    ib_info = {
        "IB_DETECTED": "1",
        "DETECTED_NET_LIST": "ib0,ib1",
    }
    env = generate_nccl_env(ib_info)
    assert env["NCCL_SOCKET_IFNAME"] == "ib0,ib1"


def test_nccl_socket_ifname_mgmt_first_heterogeneous():
    """Real-world scenario: head on wired, worker on wifi, same IB fabric.

    Each host's detection produces a different NCCL_SOCKET_IFNAME
    first-entry (mgmt) but the same IB tail — exactly what we want
    for ClusterCommEnv.from_per_host to split into shared + per-host.
    """
    head_env = generate_nccl_env(
        {
            "IB_DETECTED": "1",
            "DETECTED_SOCKET_IFNAME": "enP7s7",
            "DETECTED_NET_LIST": "enp1s0f0np0,enP2p1s0f0np0",
        }
    )
    worker_env = generate_nccl_env(
        {
            "IB_DETECTED": "1",
            "DETECTED_SOCKET_IFNAME": "wlP9s9",
            "DETECTED_NET_LIST": "enp1s0f0np0,enP2p1s0f0np0",
        }
    )
    assert head_env["NCCL_SOCKET_IFNAME"] == "enP7s7,enp1s0f0np0,enP2p1s0f0np0"
    assert worker_env["NCCL_SOCKET_IFNAME"] == "wlP9s9,enp1s0f0np0,enP2p1s0f0np0"

    from sparkrun.orchestration.comm_env import ClusterCommEnv

    cce = ClusterCommEnv.from_per_host({"head": head_env, "worker": worker_env})
    # NCCL_SOCKET_IFNAME differs per host → stays in per_host, not shared
    assert "NCCL_SOCKET_IFNAME" not in cce.shared
    assert cce.get_env("head")["NCCL_SOCKET_IFNAME"] == "enP7s7,enp1s0f0np0,enP2p1s0f0np0"
    assert cce.get_env("worker")["NCCL_SOCKET_IFNAME"] == "wlP9s9,enp1s0f0np0,enP2p1s0f0np0"


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
            host="10.0.0.1",
            returncode=0,
            stdout="",
            stderr="",
        )
        ib_map = {"spark1": "10.0.0.1", "spark2": "10.0.0.2"}
        result = validate_ib_connectivity(ib_map, ssh_kwargs={"ssh_user": "user"})

        assert result == ib_map
        mock_cmd.assert_called_once_with(
            "10.0.0.1",
            "true",
            connect_timeout=5,
            timeout=10,
            ssh_user="user",
        )

    @patch("sparkrun.orchestration.ssh.run_remote_command")
    def test_unreachable_returns_empty(self, mock_cmd):
        """When IB IP is unreachable, returns empty dict (fallback)."""
        mock_cmd.return_value = RemoteResult(
            host="10.0.0.1",
            returncode=255,
            stdout="",
            stderr="Connection timed out",
        )
        ib_map = {"spark1": "10.0.0.1", "spark2": "10.0.0.2"}
        result = validate_ib_connectivity(ib_map)

        assert result == {}


# ---------------------------------------------------------------------------
# generate_ring_nccl_overrides tests
# ---------------------------------------------------------------------------


def test_ring_nccl_overrides_keys():
    """Ring overrides contain the expected NCCL variables."""
    overrides = generate_ring_nccl_overrides({})
    assert overrides["NCCL_NET_PLUGIN"] == "none"
    assert overrides["NCCL_IB_SUBNET_AWARE_ROUTING"] == "1"
    assert overrides["NCCL_IB_MERGE_NICS"] == "0"
    assert len(overrides) == 3


def test_generate_nccl_env_ring_topology():
    """Ring topology adds ring-specific overrides."""
    ib_info = {
        "IB_DETECTED": "1",
        "DETECTED_GID_INDEX": "3",
        "DETECTED_HCA_LIST": "mlx5_0,mlx5_1",
    }
    env = generate_nccl_env(ib_info, topology="ring")
    assert env["NCCL_NET_PLUGIN"] == "none"
    assert env["NCCL_IB_SUBNET_AWARE_ROUTING"] == "1"
    assert env["NCCL_IB_MERGE_NICS"] == "0"
    # Standard IB vars still present
    assert env["NCCL_NET"] == "IB"
    assert env["NCCL_IB_HCA"] == "mlx5_0,mlx5_1"


def test_generate_nccl_env_no_ring_topology():
    """Non-ring topology does NOT add ring overrides."""
    ib_info = {
        "IB_DETECTED": "1",
        "DETECTED_GID_INDEX": "3",
        "DETECTED_HCA_LIST": "mlx5_0,mlx5_1",
    }
    env = generate_nccl_env(ib_info, topology=None)
    assert "NCCL_NET_PLUGIN" not in env
    assert "NCCL_IB_SUBNET_AWARE_ROUTING" not in env
    assert "NCCL_IB_MERGE_NICS" not in env

    env_switch = generate_nccl_env(ib_info, topology="switch")
    assert "NCCL_NET_PLUGIN" not in env_switch


# ---------------------------------------------------------------------------
# validate_ib_connectivity — ssh_kwargs forwarding
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.ssh.run_remote_command")
def test_ssh_kwargs_passed_through(mock_cmd):
    """SSH kwargs are forwarded to the connectivity check."""
    mock_cmd.return_value = RemoteResult(
        host="10.0.0.1",
        returncode=0,
        stdout="",
        stderr="",
    )
    ssh_kw = {"ssh_user": "drew", "ssh_key": "/path/to/key", "ssh_options": ["-o", "Foo=bar"]}
    ib_map = {"spark1": "10.0.0.1"}
    validate_ib_connectivity(ib_map, ssh_kwargs=ssh_kw)

    mock_cmd.assert_called_once_with(
        "10.0.0.1",
        "true",
        connect_timeout=5,
        timeout=10,
        ssh_user="drew",
        ssh_key="/path/to/key",
        ssh_options=["-o", "Foo=bar"],
    )


# ---------------------------------------------------------------------------
# ClusterCommEnv tests
# ---------------------------------------------------------------------------


class TestClusterCommEnv:
    """Tests for the ClusterCommEnv value object used across cluster ops."""

    def test_empty_returns_empty(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv.empty()
        assert cce.is_empty()
        assert cce.get_env("h1") == {}
        assert cce.all_keys() == set()
        assert len(cce) == 0
        assert not cce

    def test_from_shared_round_trip(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv.from_shared({"NCCL_NET": "IB"})
        assert not cce.is_empty()
        assert cce.get_env("h1") == {"NCCL_NET": "IB"}
        assert cce.get_env("h2") == {"NCCL_NET": "IB"}

    def test_get_env_merges_shared_and_override(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv(
            shared={"NCCL_NET": "IB", "NCCL_IB_HCA": "mlx5_0"},
            per_host={"h1": {"GLOO_SOCKET_IFNAME": "enp1s0"}},
        )
        assert cce.get_env("h1") == {
            "NCCL_NET": "IB",
            "NCCL_IB_HCA": "mlx5_0",
            "GLOO_SOCKET_IFNAME": "enp1s0",
        }
        # Unknown host falls back to shared only
        assert cce.get_env("unknown") == {"NCCL_NET": "IB", "NCCL_IB_HCA": "mlx5_0"}

    def test_per_host_overrides_shared(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv(
            shared={"NCCL_NET": "IB"},
            per_host={"h1": {"NCCL_NET": "Socket"}},
        )
        assert cce.get_env("h1") == {"NCCL_NET": "Socket"}
        assert cce.get_env("h2") == {"NCCL_NET": "IB"}

    def test_get_env_returns_fresh_dict(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv(shared={"NCCL_NET": "IB"})
        env1 = cce.get_env("h1")
        env1["MUTATED"] = "yes"
        assert "MUTATED" not in cce.get_env("h1")

    def test_from_per_host_factors_out_shared_keys(self):
        """Keys identical across hosts move to shared; differing keys stay per-host.

        This is the real-world bug scenario: head on enP7s7, worker on
        wlP9s9, everything else identical.
        """
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        per_host = {
            "head": {
                "NCCL_NET": "IB",
                "NCCL_IB_HCA": "mlx5_0",
                "GLOO_SOCKET_IFNAME": "enP7s7",
                "TP_SOCKET_IFNAME": "enP7s7",
            },
            "worker": {
                "NCCL_NET": "IB",
                "NCCL_IB_HCA": "mlx5_0",
                "GLOO_SOCKET_IFNAME": "wlP9s9",
                "TP_SOCKET_IFNAME": "wlP9s9",
            },
        }
        cce = ClusterCommEnv.from_per_host(per_host)
        assert cce.shared == {"NCCL_NET": "IB", "NCCL_IB_HCA": "mlx5_0"}
        assert cce.get_env("head")["GLOO_SOCKET_IFNAME"] == "enP7s7"
        assert cce.get_env("worker")["GLOO_SOCKET_IFNAME"] == "wlP9s9"
        assert cce.get_env("head")["NCCL_NET"] == "IB"
        assert cce.get_env("worker")["NCCL_NET"] == "IB"

    def test_from_per_host_single_host_becomes_shared(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv.from_per_host({"only": {"A": "1", "B": "2"}})
        assert cce.shared == {"A": "1", "B": "2"}
        assert cce.per_host == {}
        assert cce.get_env("only") == {"A": "1", "B": "2"}

    def test_from_per_host_empty(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv.from_per_host({})
        assert cce.is_empty()

    def test_from_per_host_key_missing_on_one_host_stays_per_host(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv.from_per_host(
            {
                "a": {"NCCL_NET": "IB", "ONLY_A": "1"},
                "b": {"NCCL_NET": "IB"},
            }
        )
        assert cce.shared == {"NCCL_NET": "IB"}
        assert "ONLY_A" in cce.get_env("a")
        assert "ONLY_A" not in cce.get_env("b")

    def test_all_keys_union(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv(
            shared={"A": "1"},
            per_host={"h1": {"B": "2"}, "h2": {"C": "3"}},
        )
        assert cce.all_keys() == {"A", "B", "C"}
        assert len(cce) == 3

    def test_per_host_keys_returns_override_keys_only(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv(
            shared={"NCCL_NET": "IB", "NCCL_IB_HCA": "mlx5_0"},
            per_host={
                "head": {"GLOO_SOCKET_IFNAME": "enP7s7", "NCCL_IB_HCA": "rocep1s0f0"},
                "worker": {"GLOO_SOCKET_IFNAME": "wlP9s9", "NCCL_IB_HCA": "rocep1s0f1"},
            },
        )
        assert cce.per_host_keys() == {"GLOO_SOCKET_IFNAME", "NCCL_IB_HCA"}

    def test_per_host_keys_empty_when_no_overrides(self):
        from sparkrun.orchestration.comm_env import ClusterCommEnv

        cce = ClusterCommEnv(shared={"NCCL_NET": "IB"})
        assert cce.per_host_keys() == set()


def test_cross_port_cabling_produces_per_host_nccl_vars():
    """Issue #135: head port 1 (f0) + worker port 2 (f1) → per-host NCCL vars.

    When DGX Sparks are cabled on different CX-7 ports, the HCA names,
    network interfaces, and UCX devices differ per host.  These must
    stay in ``per_host`` so each container receives its own values.
    """
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.orchestration.infiniband import generate_nccl_env

    head_ib = {
        "IB_DETECTED": "1",
        "DETECTED_HCA_LIST": "rocep1s0f0,roceP2p1s0f0",
        "DETECTED_SOCKET_IFNAME": "enP7s7",
        "DETECTED_NET_LIST": "enp1s0f0np0,enP2p1s0f0np0",
        "DETECTED_UCX_LIST": "rocep1s0f0:1,roceP2p1s0f0:1",
        "DETECTED_MGMT_IP": "192.168.0.192",
    }
    worker_ib = {
        "IB_DETECTED": "1",
        "DETECTED_HCA_LIST": "rocep1s0f1,roceP2p1s0f1",
        "DETECTED_SOCKET_IFNAME": "enP7s7",
        "DETECTED_NET_LIST": "enp1s0f1np1,enP2p1s0f1np1",
        "DETECTED_UCX_LIST": "rocep1s0f1:1,roceP2p1s0f1:1",
        "DETECTED_MGMT_IP": "192.168.1.116",
    }

    head_env = generate_nccl_env(head_ib)
    worker_env = generate_nccl_env(worker_ib)
    cce = ClusterCommEnv.from_per_host({"head": head_env, "worker": worker_env})

    per_host = cce.per_host_keys()
    assert "NCCL_IB_HCA" in per_host
    assert "UCX_NET_DEVICES" in per_host
    assert "NCCL_SOCKET_IFNAME" in per_host
    # NODE_IP also differs (different mgmt IPs)
    assert "NODE_IP" in per_host

    # Verify each host gets its own values back
    assert cce.get_env("head")["NCCL_IB_HCA"] == "rocep1s0f0,roceP2p1s0f0"
    assert cce.get_env("worker")["NCCL_IB_HCA"] == "rocep1s0f1,roceP2p1s0f1"
