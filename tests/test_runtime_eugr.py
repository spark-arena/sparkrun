"""Unit tests for sparkrun.runtimes.eugr_vllm_ray (EugrVllmRayRuntime)."""

from unittest import mock

import pytest
from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.vllm_ray import VllmRayRuntime
from sparkrun.runtimes.eugr_vllm_ray import EugrVllmRayRuntime


# --- EugrVllmRuntime Tests ---


def test_eugr_inherits_vllm():
    """EugrVllmRuntime extends VllmRuntime."""
    runtime = EugrVllmRayRuntime()
    assert isinstance(runtime, VllmRayRuntime)


def test_eugr_is_not_delegating():
    """EugrVllmRuntime.is_delegating_runtime() returns False (native orchestration)."""
    runtime = EugrVllmRayRuntime()
    assert runtime.is_delegating_runtime() is False


def test_eugr_runtime_name():
    """runtime_name == 'eugr-vllm'."""
    runtime = EugrVllmRayRuntime()
    assert runtime.runtime_name == "eugr-vllm"


def test_eugr_resolve_container():
    """Resolve container for eugr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "container": "custom-eugr:latest",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-eugr:latest"


def test_eugr_resolve_container_default():
    """Default container for eugr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "vllm-node"


def test_eugr_generate_command_from_template():
    """Generate command renders recipe command template (inherited from VllmRuntime)."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "command": "vllm serve {model} --port {port}",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "vllm serve meta-llama/Llama-2-7b-hf --port 8000"


def test_eugr_generate_command_structured():
    """Without a command template, generates vllm serve from defaults (inherited)."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "defaults": {"port": 8000, "tensor_parallel": 2},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "-tp 2" in cmd
    assert "--port 8000" in cmd


def test_eugr_validate_recipe():
    """Validate eugr recipe."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "command": "vllm serve model",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    issues = runtime.validate_recipe(recipe)
    # Should pass validation
    assert all("model is required" not in issue for issue in issues)


class TestEugrPrepare:
    """Test EugrBuilder.prepare_image() — container build and mod injection."""

    @pytest.fixture
    def eugr_builder(self, tmp_path):
        """Create builder with a fake repo containing build-and-copy.sh."""
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        repo_dir = tmp_path / "eugr-repo"
        repo_dir.mkdir()
        (repo_dir / "build-and-copy.sh").write_text("#!/bin/bash\nexit 0\n")
        (repo_dir / "build-and-copy.sh").chmod(0o755)
        # Create a sample mod directory
        mod_dir = repo_dir / "mods" / "flash-attn"
        mod_dir.mkdir(parents=True)
        (mod_dir / "run.sh").write_text("#!/bin/bash\necho applied\n")
        return builder, repo_dir

    def test_prepare_with_build_args(self, eugr_builder):
        """prepare_image() calls build-and-copy.sh when build_args present."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--some-flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")) as mock_build:
                    with mock.patch.object(builder, "_verify_image_imports"):
                        with mock.patch.object(builder, "_save_build_metadata"):
                            builder.prepare_image("my-image", recipe, ["10.0.0.1"])

                    cmd = mock_build.call_args[0][0]
                    assert str(repo_dir / "build-and-copy.sh") in cmd[0]
                    assert "-t" in cmd
                    assert "my-image" in cmd
                    assert "--some-flag" in cmd

    def test_prepare_without_build_args_or_mods_image_exists(self, eugr_builder):
        """prepare_image() is a no-op when no build_args/mods and image exists."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch.object(builder, "ensure_repo") as mock_ensure:
                builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])
                # ensure_repo should not be called when nothing to prepare
                mock_ensure.assert_not_called()

    def test_prepare_builds_when_image_missing(self, eugr_builder):
        """prepare_image() triggers a build when image is missing locally."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "container": "my-image",
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")) as mock_build:
                    with mock.patch.object(builder, "_verify_image_imports"):
                        with mock.patch.object(builder, "_save_build_metadata"):
                            builder.prepare_image("my-image", recipe, ["10.0.0.1"])
                    mock_build.assert_called_once()
                    cmd = mock_build.call_args[0][0]
                    assert str(repo_dir / "build-and-copy.sh") in cmd[0]
                    assert "-t" in cmd
                    assert "my-image" in cmd

    def test_prepare_dry_run(self, eugr_builder):
        """prepare_image() in dry-run does not execute the build."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing") as mock_build:
                    builder.prepare_image("vllm-node", recipe, ["10.0.0.1"], dry_run=True)
                    mock_build.assert_not_called()

    # Note: mod -> pre_exec injection moved out of the eugr builder into the
    # generic core/mods.py resolver. See tests/test_mods.py for that coverage.

    def test_prepare_build_failure_raises(self, eugr_builder):
        """prepare_image() raises RuntimeError on build failure."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch(
                    "sparkrun.builders.eugr._run_build_capturing",
                    return_value=(1, "FlashInfer build failed — restoring previous wheels...\n"),
                ):
                    with pytest.raises(RuntimeError, match="eugr container build failed"):
                        builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])


class TestEugrPreServe:
    """Test base RuntimePlugin._pre_serve() with pre_exec from recipe."""

    def test_pre_serve_with_pre_exec(self):
        """_pre_serve() runs pre_exec commands from recipe."""
        from sparkrun.runtimes.base import RuntimePlugin

        runtime = RuntimePlugin()
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "vllm",
                "pre_exec": ["echo hello"],
            }
        )
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=False,
                recipe=recipe,
                config_chain=None,
            )
            mock_hook.assert_called_once()

    def test_pre_serve_without_pre_exec(self):
        """_pre_serve() is a no-op when recipe has no pre_exec."""
        runtime = EugrVllmRayRuntime()
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
            }
        )
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=False,
                recipe=recipe,
            )
            mock_hook.assert_not_called()

    def test_pre_serve_dry_run(self):
        """_pre_serve() passes dry_run through to hooks."""
        from sparkrun.runtimes.base import RuntimePlugin

        runtime = RuntimePlugin()
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "pre_exec": ["echo hello"],
            }
        )
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=True,
                recipe=recipe,
                config_chain=None,
            )
            mock_hook.assert_called_once()
            # Verify dry_run was passed through
            assert mock_hook.call_args[1]["dry_run"] is True

    def test_pre_serve_no_recipe(self):
        """_pre_serve() is a no-op when no recipe provided (backward compat)."""
        runtime = EugrVllmRayRuntime()
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=False,
            )
            mock_hook.assert_not_called()


class TestEugrFollowLogs:
    """Test EugrVllmRuntime.follow_logs() — inherited from VllmRuntime."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_tails_serve_log(self, mock_stream):
        """Single-host eugr tails serve log in solo container (inherited)."""
        runtime = EugrVllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == "10.0.0.1"
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_tails_head(self, mock_stream):
        """Multi-host eugr tails serve log on head container (inherited from vllm)."""
        runtime = EugrVllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_head"
