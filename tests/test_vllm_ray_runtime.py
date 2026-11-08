"""Tests for VllmRayRuntime per-host NCCL env fix (issue #135)."""

from unittest.mock import patch

from _hook_scripts import hook_payload

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
        assert script_arg.startswith("docker exec sparkrun0_head bash -c ")
        payload = hook_payload(script_arg)
        assert "ray_non_carry_over_env_vars.json" in payload
        assert """'["NCCL_IB_HCA"]'""" in payload

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
        payload = hook_payload(mock_run.call_args[0][1])
        assert "NCCL_IB_HCA" in payload
        assert "UCX_NET_DEVICES" in payload
        assert "NCCL_SOCKET_IFNAME" in payload


class _StubRecipe:
    def __init__(self, runtime_config=None):
        self.runtime_config = runtime_config or {}


class TestResolveDashboard:
    """VllmRayRuntime._resolve_dashboard tri-state resolution."""

    def test_explicit_true_wins(self):
        assert VllmRayRuntime._resolve_dashboard(True, _StubRecipe({"dashboard": False})) is True

    def test_explicit_false_wins(self):
        # Explicit --no-dashboard overrides a recipe that asks for the dashboard.
        assert VllmRayRuntime._resolve_dashboard(False, _StubRecipe({"dashboard": True})) is False

    def test_none_defaults_on_when_recipe_silent(self):
        assert VllmRayRuntime._resolve_dashboard(None, _StubRecipe()) is True

    def test_none_uses_recipe_bool_false(self):
        assert VllmRayRuntime._resolve_dashboard(None, _StubRecipe({"dashboard": False})) is False

    def test_none_uses_recipe_bool_true(self):
        assert VllmRayRuntime._resolve_dashboard(None, _StubRecipe({"dashboard": True})) is True

    def test_none_coerces_recipe_string(self):
        # runtime_config values may arrive as strings; ext_parse_bool handles them.
        assert VllmRayRuntime._resolve_dashboard(None, _StubRecipe({"dashboard": "off"})) is False
        assert VllmRayRuntime._resolve_dashboard(None, _StubRecipe({"dashboard": "yes"})) is True

    def test_none_recipe_is_safe(self):
        assert VllmRayRuntime._resolve_dashboard(None, None) is True
