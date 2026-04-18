"""Tests for VllmRayRuntime per-host NCCL env fix (issue #135)."""

from unittest.mock import patch

from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.runtimes.vllm_ray import VllmRayRuntime


class TestRayNonCarryOver:
    """Tests for ray_non_carry_over_env_vars.json generation."""

    @patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_writes_per_host_keys_as_json(self, mock_run):
        """Per-host keys are written to ray_non_carry_over_env_vars.json."""
        mock_run.return_value = RemoteResult(
            host="head",
            returncode=0,
            stdout="",
            stderr="",
        )

        comm_env = ClusterCommEnv(
            shared={"NCCL_NET": "IB"},
            per_host={
                "head": {"NCCL_IB_HCA": "rocep1s0f0"},
                "worker": {"NCCL_IB_HCA": "rocep1s0f1"},
            },
        )
        VllmRayRuntime._write_ray_non_carry_over(
            "head",
            "sparkrun0_head",
            comm_env,
            {},
            dry_run=False,
        )
        mock_run.assert_called_once()
        script_arg = mock_run.call_args[0][1]
        assert "ray_non_carry_over_env_vars.json" in script_arg
        assert "NCCL_IB_HCA" in script_arg

    @patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_skips_when_no_per_host_keys(self, mock_run):
        """No file written when all vars are shared."""
        comm_env = ClusterCommEnv(shared={"NCCL_NET": "IB"})
        VllmRayRuntime._write_ray_non_carry_over(
            "head",
            "sparkrun0_head",
            comm_env,
            {},
            dry_run=False,
        )
        mock_run.assert_not_called()

    @patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_dry_run_passes_through(self, mock_run):
        """Dry-run mode passes dry_run=True to run_remote_script."""
        mock_run.return_value = RemoteResult(
            host="head",
            returncode=0,
            stdout="",
            stderr="",
        )
        comm_env = ClusterCommEnv(
            shared={},
            per_host={"head": {"X": "1"}, "worker": {"X": "2"}},
        )
        VllmRayRuntime._write_ray_non_carry_over(
            "head",
            "sparkrun0_head",
            comm_env,
            {},
            dry_run=True,
        )
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("dry_run") is True

    @patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_multiple_per_host_vars_all_excluded(self, mock_run):
        """All per-host var names appear in the exclusion script."""
        mock_run.return_value = RemoteResult(
            host="head",
            returncode=0,
            stdout="",
            stderr="",
        )
        comm_env = ClusterCommEnv(
            shared={"NCCL_NET": "IB"},
            per_host={
                "head": {
                    "NCCL_IB_HCA": "rocep1s0f0",
                    "UCX_NET_DEVICES": "rocep1s0f0:1",
                    "NCCL_SOCKET_IFNAME": "enP7s7,enp1s0f0np0",
                },
                "worker": {
                    "NCCL_IB_HCA": "rocep1s0f1",
                    "UCX_NET_DEVICES": "rocep1s0f1:1",
                    "NCCL_SOCKET_IFNAME": "enP7s7,enp1s0f1np1",
                },
            },
        )
        VllmRayRuntime._write_ray_non_carry_over(
            "head",
            "sparkrun0_head",
            comm_env,
            {},
            dry_run=False,
        )
        script_arg = mock_run.call_args[0][1]
        assert "NCCL_IB_HCA" in script_arg
        assert "UCX_NET_DEVICES" in script_arg
        assert "NCCL_SOCKET_IFNAME" in script_arg
