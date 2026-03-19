"""Tests for sparkrun agent package."""

from __future__ import annotations

import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Mock smolagents if not installed (it's an optional dependency)
# ---------------------------------------------------------------------------

_mock_smolagents = None


def _ensure_smolagents_mock():
    """Install a minimal smolagents mock into sys.modules if not already importable."""
    global _mock_smolagents
    if _mock_smolagents is not None:
        return
    try:
        import smolagents  # noqa: F401
        _mock_smolagents = False  # real module available
    except ImportError:
        mod = types.ModuleType("smolagents")

        class _MockTool:
            name = ""
            description = ""
            inputs = {}
            output_type = "string"

            def forward(self, *args, **kwargs):
                pass

        mod.Tool = _MockTool
        mod.CodeAgent = MagicMock
        mod.OpenAIServerModel = MagicMock
        mod.GradioUI = MagicMock
        sys.modules["smolagents"] = mod
        _mock_smolagents = True


# Install mock before any test collection touches agent imports
_ensure_smolagents_mock()


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


class TestAgentState:
    """Tests for agent session state persistence."""

    def test_save_and_load_state(self, tmp_path, monkeypatch):
        from sparkrun.agent import state

        state_file = tmp_path / "agent-state.yaml"
        monkeypatch.setattr(state, "STATE_FILE", state_file)

        state.save_state(
            endpoint="http://10.0.0.1:52001/v1",
            cluster_id="sparkrun_abc123",
            recipe="agent-qwen3.5-4b-awq-vllm",
            host="10.0.0.1",
            port=52001,
        )

        assert state_file.exists()
        loaded = state.load_state()
        assert loaded is not None
        assert loaded["endpoint"] == "http://10.0.0.1:52001/v1"
        assert loaded["cluster_id"] == "sparkrun_abc123"
        assert loaded["recipe"] == "agent-qwen3.5-4b-awq-vllm"
        assert loaded["host"] == "10.0.0.1"
        assert loaded["port"] == 52001
        assert "started_at" in loaded

    def test_load_state_missing(self, tmp_path, monkeypatch):
        from sparkrun.agent import state

        state_file = tmp_path / "agent-state.yaml"
        monkeypatch.setattr(state, "STATE_FILE", state_file)

        assert state.load_state() is None

    def test_clear_state(self, tmp_path, monkeypatch):
        from sparkrun.agent import state

        state_file = tmp_path / "agent-state.yaml"
        monkeypatch.setattr(state, "STATE_FILE", state_file)

        state.save_state(
            endpoint="http://10.0.0.1:52001/v1",
            cluster_id="sparkrun_abc123",
            recipe="test",
            host="10.0.0.1",
            port=52001,
        )
        assert state_file.exists()

        state.clear_state()
        assert not state_file.exists()

    def test_clear_state_missing(self, tmp_path, monkeypatch):
        """clear_state should not raise when no state file exists."""
        from sparkrun.agent import state

        state_file = tmp_path / "agent-state.yaml"
        monkeypatch.setattr(state, "STATE_FILE", state_file)

        state.clear_state()  # should not raise


# ---------------------------------------------------------------------------
# Readiness tests
# ---------------------------------------------------------------------------


class TestReadiness:
    """Tests for model readiness polling via orchestration.primitives.wait_for_healthy."""

    def test_wait_for_healthy_success(self):
        from sparkrun.orchestration.primitives import wait_for_healthy

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = wait_for_healthy(
                "http://10.0.0.1:52001/v1/models",
                max_retries=3,
                retry_interval=0,
            )
            assert result is True

    def test_wait_for_healthy_timeout(self):
        import urllib.error

        from sparkrun.orchestration.primitives import wait_for_healthy

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            result = wait_for_healthy(
                "http://10.0.0.1:52001/v1/models",
                max_retries=1,
                retry_interval=0,
                max_consecutive_refused=1,
            )
            assert result is False


# ---------------------------------------------------------------------------
# Container reuse tests
# ---------------------------------------------------------------------------


class TestContainerReuse:
    """Tests for sparkrun container reuse detection."""

    def test_check_sparkrun_container_found(self):
        from sparkrun.orchestration.primitives import check_sparkrun_container

        with patch("sparkrun.orchestration.primitives.run_command_on_host") as mock_run:
            from sparkrun.orchestration.ssh import RemoteResult

            mock_run.return_value = RemoteResult(
                host="10.0.0.1", returncode=0,
                stdout="sparkrun_abc123_solo\n", stderr="",
            )
            result = check_sparkrun_container("10.0.0.1", "sparkrun_abc123")
            assert result == "sparkrun_abc123_solo"

    def test_check_sparkrun_container_not_found(self):
        from sparkrun.orchestration.primitives import check_sparkrun_container

        with patch("sparkrun.orchestration.primitives.run_command_on_host") as mock_run:
            from sparkrun.orchestration.ssh import RemoteResult

            mock_run.return_value = RemoteResult(
                host="10.0.0.1", returncode=0,
                stdout="", stderr="",
            )
            result = check_sparkrun_container("10.0.0.1", "sparkrun_abc123")
            assert result is None

    def test_launch_reuse_existing_container(self):
        """When reuse=True and a matching container exists, return early with reused=True."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe = MagicMock()
        mock_recipe.name = "test-recipe"
        mock_recipe.runtime = "vllm"
        mock_recipe.model = "test-model"
        mock_recipe.defaults = {}
        mock_recipe.build_config_chain.return_value = {"port": 52001}

        mock_runtime = MagicMock()
        mock_runtime.resolve_container.return_value = "test-image:latest"

        mock_config = MagicMock()
        mock_config.hf_cache_dir = "/tmp/cache"
        mock_config.ssh_user = None
        mock_config.ssh_key = None
        mock_config.ssh_options = None

        with patch("sparkrun.orchestration.primitives.check_sparkrun_container",
                    return_value="sparkrun_abc123_solo"), \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id",
                    return_value="sparkrun_abc123"):
            result = launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={"port": 52001},
                config=mock_config,
                auto_port=True,
                reuse=True,
            )

        assert result.rc == 0
        assert result.reused is True
        assert result.cluster_id == "sparkrun_abc123"
        assert result.serve_port == 52001
        # Should NOT have called runtime.run()
        mock_runtime.run.assert_not_called()

    def test_launch_no_reuse_skips_container_check(self):
        """When reuse=False, container check is skipped and auto-increment proceeds normally."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe = MagicMock()
        mock_recipe.name = "test-recipe"
        mock_recipe.runtime = "vllm"
        mock_recipe.model = "test-model"
        mock_recipe.defaults = {}
        mock_recipe.builder = None
        mock_recipe.env = {}
        mock_recipe.build_config_chain.return_value = {"port": 52001}
        mock_recipe.mode = "solo"
        mock_recipe.model_revision = None
        mock_recipe.source_registry = None

        mock_runtime = MagicMock()
        mock_runtime.resolve_container.return_value = "test-image:latest"
        mock_runtime.is_delegating_runtime.return_value = True
        mock_runtime.run.return_value = 0
        mock_runtime.generate_command.return_value = "serve cmd"

        mock_config = MagicMock()
        mock_config.hf_cache_dir = "/tmp/cache"
        mock_config.cache_dir = "/tmp/cache"
        mock_config.ssh_user = None
        mock_config.ssh_key = None
        mock_config.ssh_options = None

        with patch("sparkrun.orchestration.primitives.check_sparkrun_container") as mock_check, \
             patch("sparkrun.orchestration.primitives.find_available_port",
                    return_value=52001), \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id",
                    return_value="sparkrun_abc123"), \
             patch("sparkrun.orchestration.job_metadata.save_job_metadata"), \
             patch("sparkrun.models.download.is_gguf_model", return_value=False):
            result = launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={"port": 52001},
                config=mock_config,
                auto_port=True,
                reuse=False,
            )

        # Container check should NOT have been called
        mock_check.assert_not_called()
        assert result.rc == 0
        assert result.reused is False
        mock_runtime.run.assert_called_once()

    def test_launch_no_existing_container_proceeds(self):
        """When no matching container exists, proceed with normal launch."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe = MagicMock()
        mock_recipe.name = "test-recipe"
        mock_recipe.runtime = "vllm"
        mock_recipe.model = "test-model"
        mock_recipe.defaults = {}
        mock_recipe.builder = None
        mock_recipe.env = {}
        mock_recipe.build_config_chain.return_value = {"port": 52001}
        mock_recipe.mode = "solo"
        mock_recipe.model_revision = None
        mock_recipe.source_registry = None

        mock_runtime = MagicMock()
        mock_runtime.resolve_container.return_value = "test-image:latest"
        mock_runtime.is_delegating_runtime.return_value = True
        mock_runtime.run.return_value = 0
        mock_runtime.generate_command.return_value = "serve cmd"

        mock_config = MagicMock()
        mock_config.hf_cache_dir = "/tmp/cache"
        mock_config.cache_dir = "/tmp/cache"
        mock_config.ssh_user = None
        mock_config.ssh_key = None
        mock_config.ssh_options = None

        with patch("sparkrun.orchestration.primitives.check_sparkrun_container",
                    return_value=None), \
             patch("sparkrun.orchestration.primitives.find_available_port",
                    return_value=52001), \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id",
                    return_value="sparkrun_abc123"), \
             patch("sparkrun.orchestration.job_metadata.save_job_metadata"), \
             patch("sparkrun.models.download.is_gguf_model", return_value=False):
            result = launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={"port": 52001},
                config=mock_config,
                auto_port=True,
                reuse=True,
            )

        assert result.rc == 0
        assert result.reused is False
        mock_runtime.run.assert_called_once()

    def test_launch_result_reused_default_false(self):
        """LaunchResult.reused defaults to False."""
        from sparkrun.core.launcher import LaunchResult

        result = LaunchResult(
            rc=0,
            cluster_id="sparkrun_test",
            host_list=["h1"],
            is_solo=True,
            runtime=MagicMock(),
            recipe=MagicMock(),
            overrides={},
            container_image="img",
            effective_cache_dir="/tmp",
            serve_port=8000,
            config=MagicMock(),
        )
        assert result.reused is False


# ---------------------------------------------------------------------------
# Tool base tests
# ---------------------------------------------------------------------------


class TestSparkrunBaseTool:
    """Tests for the SparkrunBaseTool subprocess wrapper."""

    def test_run_sparkrun_success(self):
        from sparkrun.agent.tools._base import SparkrunBaseTool

        tool = SparkrunBaseTool()

        with patch("shutil.which", return_value="/usr/bin/sparkrun"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["sparkrun", "list"],
                returncode=0,
                stdout="recipe1\nrecipe2",
                stderr="",
            )

            result = tool._run_sparkrun("list")
            assert "recipe1" in result
            assert "recipe2" in result
            mock_run.assert_called_once()

    def test_run_sparkrun_failure(self):
        from sparkrun.agent.tools._base import SparkrunBaseTool

        tool = SparkrunBaseTool()

        with patch("shutil.which", return_value="/usr/bin/sparkrun"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["sparkrun", "bad-cmd"],
                returncode=1,
                stdout="",
                stderr="Error: unknown command",
            )

            result = tool._run_sparkrun("bad-cmd")
            assert "Command failed" in result
            assert "Error: unknown command" in result

    def test_run_sparkrun_not_found(self):
        from sparkrun.agent.tools._base import SparkrunBaseTool

        tool = SparkrunBaseTool()

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="sparkrun not found"):
                tool._run_sparkrun("list")


# ---------------------------------------------------------------------------
# Tool forward tests (mocking _run_sparkrun)
# ---------------------------------------------------------------------------


class TestRunStopTools:
    """Tests for RunInferenceTool and StopInferenceTool."""

    def test_run_inference_basic(self):
        from sparkrun.agent.tools.run_stop import RunInferenceTool

        tool = RunInferenceTool()
        with patch.object(tool, "_run_sparkrun", return_value="Launched") as mock:
            result = tool.forward("qwen3-1.7b-vllm")
            mock.assert_called_once_with(
                "run", "qwen3-1.7b-vllm", "--no-follow", timeout=300,
            )
            assert result == "Launched"

    def test_run_inference_with_options(self):
        from sparkrun.agent.tools.run_stop import RunInferenceTool

        tool = RunInferenceTool()
        with patch.object(tool, "_run_sparkrun", return_value="ok") as mock:
            tool.forward(
                "my-recipe",
                hosts="10.0.0.1,10.0.0.2",
                cluster="mylab",
                solo=True,
                tensor_parallel=2,
                gpu_mem=0.8,
            )
            args = mock.call_args[0]
            assert "--hosts" in args
            assert "--cluster" in args
            assert "--solo" in args
            assert "--tp" in args
            assert "--gpu-mem" in args

    def test_stop_inference_recipe(self):
        from sparkrun.agent.tools.run_stop import StopInferenceTool

        tool = StopInferenceTool()
        with patch.object(tool, "_run_sparkrun", return_value="Stopped") as mock:
            result = tool.forward(recipe_name="qwen3-1.7b-vllm")
            mock.assert_called_once_with("stop", "qwen3-1.7b-vllm")
            assert result == "Stopped"

    def test_stop_inference_all(self):
        from sparkrun.agent.tools.run_stop import StopInferenceTool

        tool = StopInferenceTool()
        with patch.object(tool, "_run_sparkrun", return_value="All stopped") as mock:
            tool.forward(stop_all=True)
            mock.assert_called_once_with("stop", "--all")

    def test_stop_inference_no_args(self):
        from sparkrun.agent.tools.run_stop import StopInferenceTool

        tool = StopInferenceTool()
        result = tool.forward()
        assert "Error" in result


class TestStatusLogsTools:
    """Tests for ClusterStatusTool and ContainerLogsTool."""

    def test_cluster_status(self):
        from sparkrun.agent.tools.status_logs import ClusterStatusTool

        tool = ClusterStatusTool()
        with patch.object(tool, "_run_sparkrun", return_value="No containers") as mock:
            tool.forward()
            mock.assert_called_once_with("status")

    def test_container_logs(self):
        from sparkrun.agent.tools.status_logs import ContainerLogsTool

        tool = ContainerLogsTool()
        with patch.object(tool, "_run_sparkrun", return_value="log line") as mock:
            tool.forward("my-recipe", tail=20)
            mock.assert_called_once_with(
                "logs", "my-recipe", "--no-follow", "--tail", "20",
                timeout=30,
            )


class TestRecipeTools:
    """Tests for recipe-related tools."""

    def test_recipe_search(self):
        from sparkrun.agent.tools.recipes import RecipeSearchTool

        tool = RecipeSearchTool()
        with patch.object(tool, "_run_sparkrun", return_value="found 3") as mock:
            tool.forward("llama")
            mock.assert_called_once_with("recipe", "search", "llama")

    def test_recipe_list(self):
        from sparkrun.agent.tools.recipes import RecipeListTool

        tool = RecipeListTool()
        with patch.object(tool, "_run_sparkrun", return_value="recipes...") as mock:
            tool.forward(runtime="vllm")
            args = mock.call_args[0]
            assert "--runtime" in args
            assert "vllm" in args

    def test_recipe_show(self):
        from sparkrun.agent.tools.recipes import RecipeShowTool

        tool = RecipeShowTool()
        with patch.object(tool, "_run_sparkrun", return_value="details") as mock:
            tool.forward("qwen3-1.7b-vllm")
            mock.assert_called_once_with("recipe", "show", "qwen3-1.7b-vllm")

    def test_recipe_create(self, tmp_path):
        from sparkrun.agent.tools.recipes import RecipeCreateTool

        tool = RecipeCreateTool()
        with patch.object(tool, "_run_sparkrun", return_value="Valid recipe"):
            result = tool.forward(
                name="test-recipe",
                model="meta-llama/Llama-2-7b-hf",
                runtime="vllm",
                container="scitrera/dgx-spark-vllm:latest",
                output_dir=str(tmp_path),
            )
            assert "Recipe written" in result

            recipe_path = tmp_path / "test-recipe.yaml"
            assert recipe_path.exists()
            data = yaml.safe_load(recipe_path.read_text())
            assert data["model"] == "meta-llama/Llama-2-7b-hf"
            assert data["runtime"] == "vllm"
            assert data["sparkrun_version"] == "2"

    def test_recipe_validate(self):
        from sparkrun.agent.tools.recipes import RecipeValidateTool

        tool = RecipeValidateTool()
        with patch.object(tool, "_run_sparkrun", return_value="Valid") as mock:
            tool.forward("/path/to/recipe.yaml")
            mock.assert_called_once_with("recipe", "validate", "/path/to/recipe.yaml")


class TestClusterTools:
    """Tests for cluster management tools."""

    def test_cluster_list(self):
        from sparkrun.agent.tools.clusters import ClusterListTool

        tool = ClusterListTool()
        with patch.object(tool, "_run_sparkrun", return_value="mylab") as mock:
            tool.forward()
            mock.assert_called_once_with("cluster", "list")

    def test_cluster_show(self):
        from sparkrun.agent.tools.clusters import ClusterShowTool

        tool = ClusterShowTool()
        with patch.object(tool, "_run_sparkrun", return_value="details") as mock:
            tool.forward("mylab")
            mock.assert_called_once_with("cluster", "show", "mylab")

    def test_cluster_create(self):
        from sparkrun.agent.tools.clusters import ClusterCreateTool

        tool = ClusterCreateTool()
        with patch.object(tool, "_run_sparkrun", return_value="Created") as mock:
            tool.forward("mylab", "10.0.0.1,10.0.0.2", set_default=True)
            args = mock.call_args[0]
            assert "cluster" in args
            assert "create" in args
            assert "mylab" in args
            assert "--default" in args


class TestSetupTools:
    """Tests for setup tools."""

    def test_setup_ssh_mesh(self):
        from sparkrun.agent.tools.setup import SetupSSHMeshTool

        tool = SetupSSHMeshTool()
        with patch.object(tool, "_run_sparkrun", return_value="Done") as mock:
            tool.forward(cluster="mylab")
            args = mock.call_args[0]
            assert "setup" in args
            assert "ssh-mesh" in args
            assert "--cluster" in args

    def test_setup_cx7(self):
        from sparkrun.agent.tools.setup import SetupCX7Tool

        tool = SetupCX7Tool()
        with patch.object(tool, "_run_sparkrun", return_value="Done") as mock:
            tool.forward(hosts="10.0.0.1")
            args = mock.call_args[0]
            assert "setup" in args
            assert "cx7" in args

    def test_setup_permissions(self):
        from sparkrun.agent.tools.setup import SetupPermissionsTool

        tool = SetupPermissionsTool()
        with patch.object(tool, "_run_sparkrun", return_value="Done") as mock:
            tool.forward()
            mock.assert_called_once_with("setup", "permissions", timeout=120)


class TestOpenShellTool:
    """Tests for the OpenShell stub tool."""

    def test_openshell_stub(self):
        from sparkrun.agent.tools.openshell import OpenShellExecuteTool

        tool = OpenShellExecuteTool()
        result = tool.forward("print('hello')")
        assert "not yet available" in result


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    """Tests for discover_tools()."""

    def test_discover_tools_returns_list(self):
        from sparkrun.agent.tools import discover_tools

        tools = discover_tools()
        assert isinstance(tools, list)
        assert len(tools) == 16  # all tools from the plan

    def test_discover_tools_unique_names(self):
        from sparkrun.agent.tools import discover_tools

        tools = discover_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names)), "Duplicate tool names found: %s" % names


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestAgentCLI:
    """Tests for sparkrun agent CLI commands."""

    def test_agent_group_help(self):
        from click.testing import CliRunner

        from sparkrun.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["agent", "--help"])
        assert result.exit_code == 0
        assert "Interactive AI agent" in result.output

    def test_agent_start_help(self):
        from click.testing import CliRunner

        from sparkrun.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["agent", "start", "--help"])
        assert result.exit_code == 0
        assert "--ui" in result.output
        assert "--api-base" in result.output

    def test_agent_stop_no_session(self):
        from click.testing import CliRunner

        from sparkrun.cli import main

        runner = CliRunner()
        with patch("sparkrun.agent.state.load_state", return_value=None):
            result = runner.invoke(main, ["agent", "stop"])
            assert result.exit_code == 0
            assert "No agent session" in result.output

    def test_agent_status_no_session(self):
        from click.testing import CliRunner

        from sparkrun.cli import main

        runner = CliRunner()
        with patch("sparkrun.agent.state.load_state", return_value=None):
            result = runner.invoke(main, ["agent", "status"])
            assert result.exit_code == 0
            assert "No agent session" in result.output

    def test_agent_status_with_session(self):
        from click.testing import CliRunner

        from sparkrun.cli import main

        state = {
            "endpoint": "http://10.0.0.1:52001/v1",
            "cluster_id": "sparkrun_abc123",
            "recipe": "agent-qwen3.5-4b-awq-vllm",
            "host": "10.0.0.1",
            "port": 52001,
            "started_at": "2026-03-19T10:30:00",
        }
        runner = CliRunner()
        with patch("sparkrun.agent.state.load_state", return_value=state), \
             patch("sparkrun.orchestration.primitives.wait_for_healthy", return_value=True):
            result = runner.invoke(main, ["agent", "status"])
            assert result.exit_code == 0
            assert "10.0.0.1" in result.output
            assert "52001" in result.output
            assert "reachable" in result.output

    def test_agent_start_missing_smolagents(self):
        from click.testing import CliRunner

        from sparkrun.cli import main

        runner = CliRunner()
        with patch("sparkrun.cli._agent._check_smolagents_import", side_effect=SystemExit(1)):
            result = runner.invoke(main, ["agent", "start", "--api-base", "http://localhost:8000/v1"])
            assert result.exit_code == 1
