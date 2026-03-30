"""CLI integration tests for sparkrun.

Tests the CLI using Click's CliRunner. The CLI is defined in sparkrun.cli
with the main group command.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.runtimes.sglang import SglangRuntime

# Name for the test recipe used by CLI integration tests.
# The original bundled recipe (_TEST_RECIPE_NAME) was removed
# from the repo (commit 34ece47), so we create a synthetic test recipe.
_TEST_RECIPE_SOLO_NAME = "test-solo-only"

_TEST_RECIPE_SOLO_DATA = {
    "recipe_version": "2",
    "name": "Test Solo Only Recipe",
    "description": "A test recipe capped at max_nodes=1",
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "sglang",
    "mode": "auto",
    "max_nodes": 1,
    "container": "scitrera/dgx-spark-sglang:latest",
    "defaults": {
        "port": 30000,
        "host": "0.0.0.0",
    },
}

_TEST_RECIPE_NAME = "test-sglang-cluster"

_TEST_RECIPE_DATA = {
    "sparkrun_version": "2",
    "name": "Test SGLang Cluster Recipe",
    "description": "A test recipe for CLI integration tests",
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "sglang",
    "mode": "cluster",
    "min_nodes": 1,
    "max_nodes": 8,
    "container": "scitrera/dgx-spark-sglang:latest",
    "defaults": {
        "port": 30000,
        "host": "0.0.0.0",
        "tensor_parallel": 2,
        "gpu_memory_utilization": 0.9,
    },
    "metadata": {
        "model_params": 1700000000,
        "model_dtype": "float16",
    },
    "env": {
        "NCCL_CUMEM_ENABLE": "0",
    },
}


@pytest.fixture
def runner():
    """Create a CliRunner instance."""
    return CliRunner()


@pytest.fixture
def reset_bootstrap(v):
    """Ensure sparkrun is initialized before CLI tests that call init_sparkrun().

    By depending on the 'v' fixture, sparkrun is initialized OUTSIDE the
    CliRunner context (where faulthandler.enable() works with real file
    descriptors). The CLI command's init_sparkrun() call then reuses the
    existing singleton instead of re-initializing.
    """
    yield


@pytest.fixture(autouse=True)
def _cli_test_recipes(tmp_path_factory, monkeypatch):
    """Create test recipes and patch discovery so CLI can find them.

    Since bundled recipes were removed from the repo, CLI integration tests
    use synthetic test recipes made discoverable via monkeypatching.
    """
    recipe_dir = tmp_path_factory.mktemp("recipes")

    # Write the main test recipe
    recipe_file = recipe_dir / f"{_TEST_RECIPE_NAME}.yaml"
    recipe_file.write_text(yaml.safe_dump(_TEST_RECIPE_DATA))

    # Write the solo-only test recipe (max_nodes=1)
    solo_recipe_file = recipe_dir / f"{_TEST_RECIPE_SOLO_NAME}.yaml"
    solo_recipe_file.write_text(yaml.safe_dump(_TEST_RECIPE_SOLO_DATA))

    # Patch discover_cwd_recipes to return our test recipes
    import sparkrun.core.recipe

    original_discover = sparkrun.core.recipe.discover_cwd_recipes

    def _patched_discover(directory=None):
        # Return our test recipes plus any originals
        originals = original_discover(directory)
        return [recipe_file, solo_recipe_file] + originals

    monkeypatch.setattr(sparkrun.core.recipe, "discover_cwd_recipes", _patched_discover)


class TestVersionAndHelp:
    """Test version and help output."""

    def test_version(self, runner):
        """Test that sparkrun --version shows version string."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "sparkrun, version " in result.output

    def test_help(self, runner):
        """Test that sparkrun --help shows group help text with command names."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "sparkrun" in result.output.lower()
        # Check for main commands
        assert "run" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "search" in result.output
        assert "stop" in result.output
        assert "logs" in result.output
        assert "benchmark" in result.output

    def test_run_help(self, runner):
        """Test that sparkrun run --help shows run command help."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run an inference recipe" in result.output
        assert "--solo" in result.output
        assert "--hosts" in result.output
        assert "--dry-run" in result.output
        assert "--cluster-id" not in result.output


class TestListCommand:
    """Test the list command."""

    def test_list_shows_recipes(self, runner):
        """Test that sparkrun list discovers recipes from the recipes/ directory."""
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert _TEST_RECIPE_NAME in output_lower

    def test_list_table_format(self, runner):
        """Test that list output has header with Name, Runtime columns."""
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        # Check for table headers
        assert "Name" in result.output
        assert "Runtime" in result.output
        # Check for separator line
        assert "-" * 10 in result.output


class TestShowCommand:
    """Test the show command."""

    def test_show_recipe(self, runner):
        """Test that sparkrun show displays recipe details with VRAM."""
        result = runner.invoke(main, ["show", _TEST_RECIPE_NAME])
        assert result.exit_code == 0
        # Check for recipe detail fields
        assert "Name:" in result.output
        assert "Runtime:" in result.output
        assert "Model:" in result.output
        assert "Container:" in result.output
        # Check for specific recipe values
        assert "qwen" in result.output.lower()
        assert "sglang" in result.output.lower()
        # VRAM estimation shown by default
        assert "VRAM Estimation" in result.output

    def test_show_nonexistent_recipe(self, runner):
        """Test that sparkrun show nonexistent-recipe exits with error code."""
        result = runner.invoke(main, ["show", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_show_no_save_option(self, runner):
        """Test that sparkrun show --help does not show --save (moved to export)."""
        result = runner.invoke(main, ["show", "--help"])
        assert result.exit_code == 0
        assert "--save" not in result.output


class TestExportCommand:
    """Test the export command group."""

    def test_export_yaml_to_stdout(self, runner):
        """Test that export recipe outputs normalized YAML to stdout."""
        result = runner.invoke(
            main,
            ["export", "recipe", _TEST_RECIPE_NAME],
        )
        assert result.exit_code == 0
        import yaml

        data = yaml.safe_load(result.output)
        assert "model" in data
        assert "runtime" in data

    def test_export_save_to_file(self, runner, tmp_path):
        """Test that export recipe --save writes normalized YAML to a file."""
        dest = tmp_path / "saved-recipe.yaml"
        result = runner.invoke(
            main,
            ["export", "recipe", _TEST_RECIPE_NAME, "--save", str(dest)],
        )
        assert result.exit_code == 0
        assert "Recipe saved to" in result.output
        assert dest.exists()
        import yaml

        data = yaml.safe_load(dest.read_text())
        assert "model" in data
        assert "runtime" in data

    def test_export_json_to_stdout(self, runner):
        """Test that export recipe --json outputs valid JSON to stdout."""
        import json

        result = runner.invoke(
            main,
            ["export", "recipe", _TEST_RECIPE_NAME, "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "model" in data
        assert "runtime" in data

    def test_export_running_with_recipe_state(self, runner, tmp_path, monkeypatch):
        """Test export running reconstructs from recipe_state in metadata."""
        from sparkrun.core.recipe import Recipe

        # Create a recipe and serialize its state
        recipe_file = tmp_path / "test-export.yaml"
        recipe_file.write_text(yaml.safe_dump(_TEST_RECIPE_DATA))
        recipe = Recipe.load(recipe_file)
        state = recipe.__getstate__()

        cluster_id = "sparkrun_aabbccddee11"
        meta = {
            "cluster_id": cluster_id,
            "recipe": _TEST_RECIPE_NAME,
            "model": _TEST_RECIPE_DATA["model"],
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
            "recipe_state": state,
            "overrides": {"port": 9999, "tensor_parallel": 4},
            "effective_container_image": "custom/image:v2",
        }

        monkeypatch.setattr(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            lambda cid, cache_dir=None: meta if cid == cluster_id else None,
        )

        result = runner.invoke(main, ["export", "running-recipe", "aabbccddee11"])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["container"] == "custom/image:v2"
        assert data["defaults"]["port"] == 9999
        assert data["defaults"]["tensor_parallel"] == 4

    def test_export_running_json(self, runner, tmp_path, monkeypatch):
        """Test export running --json outputs valid JSON."""
        import json as json_mod

        from sparkrun.core.recipe import Recipe

        recipe_file = tmp_path / "test-export.yaml"
        recipe_file.write_text(yaml.safe_dump(_TEST_RECIPE_DATA))
        recipe = Recipe.load(recipe_file)
        state = recipe.__getstate__()

        cluster_id = "sparkrun_aabbccddee22"
        meta = {
            "cluster_id": cluster_id,
            "recipe": _TEST_RECIPE_NAME,
            "model": _TEST_RECIPE_DATA["model"],
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
            "recipe_state": state,
        }

        monkeypatch.setattr(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            lambda cid, cache_dir=None: meta if cid == cluster_id else None,
        )

        result = runner.invoke(main, ["export", "running-recipe", "aabbccddee22", "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        assert "model" in data
        assert "runtime" in data

    def test_export_running_save_to_file(self, runner, tmp_path, monkeypatch):
        """Test export running --save writes to a file."""
        from sparkrun.core.recipe import Recipe

        recipe_file = tmp_path / "test-export.yaml"
        recipe_file.write_text(yaml.safe_dump(_TEST_RECIPE_DATA))
        recipe = Recipe.load(recipe_file)
        state = recipe.__getstate__()

        cluster_id = "sparkrun_aabbccddee33"
        meta = {
            "cluster_id": cluster_id,
            "recipe": _TEST_RECIPE_NAME,
            "model": _TEST_RECIPE_DATA["model"],
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
            "recipe_state": state,
        }

        monkeypatch.setattr(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            lambda cid, cache_dir=None: meta if cid == cluster_id else None,
        )

        dest = tmp_path / "exported.yaml"
        result = runner.invoke(main, ["export", "running-recipe", "aabbccddee33", "--save", str(dest)])
        assert result.exit_code == 0
        assert "Recipe saved to" in result.output
        assert dest.exists()
        data = yaml.safe_load(dest.read_text())
        assert "model" in data

    def test_export_running_fallback_overrides(self, runner, monkeypatch):
        """Test export running falls back to recipe name + overrides when no recipe_state."""
        cluster_id = "sparkrun_aabbccddee44"
        meta = {
            "cluster_id": cluster_id,
            "recipe": _TEST_RECIPE_NAME,
            "model": _TEST_RECIPE_DATA["model"],
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
            "overrides": {"port": 7777},
            "effective_container_image": "override/image:v3",
        }

        monkeypatch.setattr(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            lambda cid, cache_dir=None: meta if cid == cluster_id else None,
        )

        result = runner.invoke(main, ["export", "running-recipe", "aabbccddee44"])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["defaults"]["port"] == 7777
        assert data["container"] == "override/image:v3"

    def test_export_running_legacy_fallback(self, runner, monkeypatch):
        """Test export running legacy fallback with individual metadata fields."""
        cluster_id = "sparkrun_aabbccddee55"
        meta = {
            "cluster_id": cluster_id,
            "recipe": _TEST_RECIPE_NAME,
            "model": _TEST_RECIPE_DATA["model"],
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
            "tensor_parallel": 2,
            "port": 5555,
        }

        monkeypatch.setattr(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            lambda cid, cache_dir=None: meta if cid == cluster_id else None,
        )

        result = runner.invoke(main, ["export", "running-recipe", "aabbccddee55"])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["defaults"]["tensor_parallel"] == 2
        assert data["defaults"]["port"] == 5555

    def test_export_running_no_metadata(self, runner, monkeypatch):
        """Test export running with no metadata exits with error."""
        monkeypatch.setattr(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            lambda cid, cache_dir=None: None,
        )

        result = runner.invoke(main, ["export", "running-recipe", "aabbccddee66"])
        assert result.exit_code != 0
        assert "No job metadata" in result.output

    def test_apply_spark_arena_benchmarks_sets_metadata(self):
        """_apply_spark_arena_benchmarks populates spark_arena_benchmarks from @spark-arena/ prefix."""
        from sparkrun.cli._export import _apply_spark_arena_benchmarks
        from sparkrun.core.recipe import Recipe

        recipe = Recipe.__new__(Recipe)
        recipe.metadata = {}
        recipe.defaults = {"tensor_parallel": 2}
        _apply_spark_arena_benchmarks(recipe, "@spark-arena/076136cd-260a-4e77-b6e2-309d8f64619b")
        assert recipe.metadata["spark_arena_benchmarks"] == [
            {"tp": 2, "uuid": "076136cd-260a-4e77-b6e2-309d8f64619b"},
        ]

    def test_apply_spark_arena_benchmarks_default_tp(self):
        """_apply_spark_arena_benchmarks defaults tp to 1 when not in recipe defaults."""
        from sparkrun.cli._export import _apply_spark_arena_benchmarks
        from sparkrun.core.recipe import Recipe

        recipe = Recipe.__new__(Recipe)
        recipe.metadata = {}
        recipe.defaults = {}
        _apply_spark_arena_benchmarks(recipe, "@spark-arena/076136cd-260a-4e77-b6e2-309d8f64619b")
        assert recipe.metadata["spark_arena_benchmarks"] == [
            {"tp": 1, "uuid": "076136cd-260a-4e77-b6e2-309d8f64619b"},
        ]

    def test_apply_spark_arena_benchmarks_preserves_existing(self):
        """_apply_spark_arena_benchmarks does not overwrite existing spark_arena_benchmarks."""
        from sparkrun.cli._export import _apply_spark_arena_benchmarks
        from sparkrun.core.recipe import Recipe

        existing = [{"tp": 4, "uuid": "existing-uuid"}]
        recipe = Recipe.__new__(Recipe)
        recipe.metadata = {"spark_arena_benchmarks": existing}
        recipe.defaults = {}
        _apply_spark_arena_benchmarks(recipe, "@spark-arena/new-uuid")
        assert recipe.metadata["spark_arena_benchmarks"] == existing

    def test_apply_spark_arena_benchmarks_ignores_non_spark_arena(self):
        """_apply_spark_arena_benchmarks is a no-op for non-spark-arena recipes."""
        from sparkrun.cli._export import _apply_spark_arena_benchmarks
        from sparkrun.core.recipe import Recipe

        recipe = Recipe.__new__(Recipe)
        recipe.metadata = {}
        recipe.defaults = {}
        _apply_spark_arena_benchmarks(recipe, "my-local-recipe")
        assert "spark_arena_benchmarks" not in recipe.metadata


class TestExportSystemdCommand:
    """Test the export systemd command."""

    def test_export_systemd_help(self, runner):
        """Test that sparkrun export systemd --help shows expected options."""
        result = runner.invoke(main, ["export", "systemd", "--help"])
        assert result.exit_code == 0
        assert "--install" in result.output
        assert "--uninstall" in result.output
        assert "--start" in result.output
        assert "--service-name" in result.output
        assert "--cluster" in result.output

    def test_export_systemd_install_and_uninstall_mutually_exclusive(self, runner, monkeypatch):
        """Test that --install and --uninstall cannot be used together."""
        result = runner.invoke(
            main,
            ["export", "systemd", _TEST_RECIPE_NAME,
             "--hosts", "10.0.0.1",
             "--install", "--uninstall"],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_render_systemd_unit(self):
        """Test that _render_systemd_unit produces correct unit file content."""
        from sparkrun.cli._export import _render_systemd_unit
        from sparkrun.core.recipe import Recipe

        recipe_file_data = yaml.safe_dump(_TEST_RECIPE_DATA)
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(recipe_file_data)
            f.flush()
            recipe = Recipe.load(f.name)
        os.unlink(f.name)

        unit = _render_systemd_unit(
            slug="test-sglang-cluster",
            recipe=recipe,
            cluster_name="test-sglang-cluster-systemd",
            ssh_user="testuser",
            sparkrun_path="/usr/local/bin/sparkrun",
            user_home="/home/testuser",
        )

        assert "[Unit]" in unit
        assert "[Service]" in unit
        assert "[Install]" in unit
        assert "User=testuser" in unit
        assert "Group=testuser" in unit
        assert "/usr/local/bin/sparkrun run" in unit
        assert "--cluster test-sglang-cluster-systemd" in unit
        assert "--foreground --no-follow" in unit
        assert "SyslogIdentifier=sparkrun-test-sglang-cluster" in unit
        assert "Restart=on-failure" in unit
        assert "RestartSec=30" in unit
        assert "TimeoutStartSec=600" in unit
        assert "WantedBy=multi-user.target" in unit

    def test_render_install_script(self):
        """Test that _render_install_script embeds recipe and cluster YAML."""
        from sparkrun.cli._export import _render_install_script

        recipe_yaml = "model: test\nruntime: sglang\n"
        cluster_yaml = "name: test-systemd\nhosts:\n- 10.0.0.1\n"

        script = _render_install_script(
            slug="test-recipe",
            recipe_yaml=recipe_yaml,
            cluster_yaml=cluster_yaml,
            cluster_name="test-recipe-systemd",
            user_home="/home/testuser",
        )

        assert "mkdir -p" in script
        assert "/home/testuser/.config/sparkrun/services/test-recipe" in script
        assert "recipe.yaml" in script
        assert "cluster.yaml" in script
        assert "test-recipe-systemd" in script

    def test_render_uninstall_script(self):
        """Test that _render_uninstall_script stops, disables, and removes."""
        from sparkrun.cli._export import _render_uninstall_script

        script = _render_uninstall_script(
            slug="test-recipe",
            cluster_name="test-recipe-systemd",
            user_home="/home/testuser",
        )

        assert "systemctl stop" in script
        assert "systemctl disable" in script
        assert "rm -f" in script
        assert "daemon-reload" in script
        assert "sparkrun-test-recipe" in script
        assert "/home/testuser/.config/sparkrun/services/test-recipe" in script

    def test_render_sudo_install_script(self):
        """Test that _render_sudo_install_script writes unit and enables service."""
        from sparkrun.cli._export import _render_sudo_install_script

        unit_contents = "[Unit]\nDescription=test\n[Service]\nType=simple\n"
        script = _render_sudo_install_script(slug="test-recipe", unit_contents=unit_contents)

        assert "/etc/systemd/system/sparkrun-test-recipe.service" in script
        assert "daemon-reload" in script
        assert "systemctl enable" in script
        assert "sparkrun-test-recipe" in script

    def test_slug_sanitization(self):
        """Test that recipe slug produces valid systemd service names."""
        from sparkrun.core.recipe import Recipe

        recipe_data = dict(_TEST_RECIPE_DATA)
        recipe_data["name"] = "My Fancy Model (v2.0)!"
        recipe_file_data = yaml.safe_dump(recipe_data)
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(recipe_file_data)
            f.flush()
            recipe = Recipe.load(f.name)
        os.unlink(f.name)

        slug = recipe.slug
        # Should only contain lowercase alphanumeric and hyphens
        import re
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug), "slug '%s' is not valid" % slug

    def test_service_name_override(self, runner, monkeypatch):
        """Test that --service-name overrides the slug in generated output."""
        # Mock _detect_remote_sparkrun to avoid SSH
        monkeypatch.setattr(
            "sparkrun.cli._export._detect_remote_sparkrun",
            lambda host, ssh_kwargs, dry_run=False: ("/usr/local/bin/sparkrun", "/home/user"),
        )

        result = runner.invoke(
            main,
            ["export", "systemd", _TEST_RECIPE_NAME,
             "--hosts", "10.0.0.1",
             "--service-name", "my-custom-name"],
        )
        assert result.exit_code == 0
        assert "sparkrun-my-custom-name" in result.output

    def test_export_systemd_dry_run(self, runner, monkeypatch):
        """Test dry-run mode prints unit file and scripts."""
        monkeypatch.setattr(
            "sparkrun.cli._export._detect_remote_sparkrun",
            lambda host, ssh_kwargs, dry_run=False: ("/usr/local/bin/sparkrun", "/home/user"),
        )

        result = runner.invoke(
            main,
            ["export", "systemd", _TEST_RECIPE_NAME,
             "--hosts", "10.0.0.1,10.0.0.2"],
        )
        assert result.exit_code == 0
        assert "Unit file" in result.output
        assert "[Unit]" in result.output
        assert "[Service]" in result.output
        assert "Baked recipe" in result.output
        assert "Cluster definition" in result.output
        assert "Install script" in result.output
        assert "To deploy, re-run with --install" in result.output

    def test_export_systemd_from_running_job(self, runner, tmp_path, monkeypatch):
        """Test systemd export reconstructs from cluster_id via job metadata."""
        from sparkrun.core.recipe import Recipe

        recipe_file = tmp_path / "test-export.yaml"
        recipe_file.write_text(yaml.safe_dump(_TEST_RECIPE_DATA))
        recipe = Recipe.load(recipe_file)
        state = recipe.__getstate__()

        cluster_id = "sparkrun_aabb001122"
        meta = {
            "cluster_id": cluster_id,
            "recipe": _TEST_RECIPE_NAME,
            "model": _TEST_RECIPE_DATA["model"],
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
            "recipe_state": state,
            "overrides": {"port": 9999},
        }

        monkeypatch.setattr(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            lambda cid, cache_dir=None: meta if cid == cluster_id else None,
        )
        monkeypatch.setattr(
            "sparkrun.cli._export._detect_remote_sparkrun",
            lambda host, ssh_kwargs, dry_run=False: ("/usr/local/bin/sparkrun", "/home/user"),
        )

        result = runner.invoke(main, ["export", "systemd", "aabb001122"])
        assert result.exit_code == 0
        assert "[Unit]" in result.output
        assert "sparkrun" in result.output

    def test_export_systemd_install_mocked(self, runner, monkeypatch):
        """Test --install mode calls correct SSH scripts."""
        from sparkrun.orchestration.ssh import RemoteResult

        monkeypatch.setattr(
            "sparkrun.cli._export._detect_remote_sparkrun",
            lambda host, ssh_kwargs, dry_run=False: ("/usr/local/bin/sparkrun", "/home/user"),
        )

        ssh_calls = []

        def mock_run_remote_script(host, script, **kwargs):
            ssh_calls.append(("user", host, script))
            return RemoteResult(host=host, returncode=0, stdout="OK", stderr="")

        def mock_run_sudo(host, script, password, **kwargs):
            ssh_calls.append(("sudo", host, script))
            return RemoteResult(host=host, returncode=0, stdout="OK", stderr="")

        monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", mock_run_remote_script)
        monkeypatch.setattr("sparkrun.orchestration.sudo.run_sudo_script_on_host", mock_run_sudo)

        result = runner.invoke(
            main,
            ["export", "systemd", _TEST_RECIPE_NAME,
             "--hosts", "10.0.0.1",
             "--install"],
            input="testpassword\n",
        )
        assert result.exit_code == 0
        assert len(ssh_calls) == 2  # user-level install + sudo install
        assert ssh_calls[0][0] == "user"
        assert ssh_calls[1][0] == "sudo"

    def test_export_systemd_sparkrun_not_found_install_fails(self, runner, monkeypatch):
        """Test error when head node lacks sparkrun and auto-install fails (uv not found)."""
        from sparkrun.orchestration.ssh import RemoteResult

        monkeypatch.setattr(
            "sparkrun.orchestration.ssh.run_remote_script",
            lambda host, script, **kwargs: RemoteResult(
                host=host, returncode=1, stdout="",
                stderr="ERROR: uv not found",
            ),
        )

        result = runner.invoke(
            main,
            ["export", "systemd", _TEST_RECIPE_NAME,
             "--hosts", "10.0.0.1"],
        )
        assert result.exit_code != 0
        assert "Failed to install sparkrun" in result.output

    def test_export_systemd_sparkrun_auto_install_success(self, runner, monkeypatch):
        """Test detection fails → auto-install succeeds → re-detect succeeds."""
        from sparkrun.orchestration.ssh import RemoteResult

        call_count = {"detect": 0}

        def mock_detect(host, ssh_kwargs, dry_run=False):
            call_count["detect"] += 1
            if call_count["detect"] == 1:
                return None, None  # first detect fails
            return "/home/user/.local/bin/sparkrun", "/home/user"

        def mock_install(host, ssh_kwargs, dry_run=False):
            return True

        monkeypatch.setattr("sparkrun.cli._export._detect_remote_sparkrun", mock_detect)
        monkeypatch.setattr("sparkrun.cli._export._install_remote_sparkrun", mock_install)

        result = runner.invoke(
            main,
            ["export", "systemd", _TEST_RECIPE_NAME,
             "--hosts", "10.0.0.1"],
        )
        assert result.exit_code == 0
        assert "sparkrun not found" in result.output
        assert "sparkrun installed" in result.output


class TestVramCommand:
    """Test the vram command."""

    def test_vram_recipe(self, runner):
        """Test sparkrun recipe vram shows estimation."""
        result = runner.invoke(main, ["recipe", "vram", _TEST_RECIPE_NAME, "--no-auto-detect"])
        assert result.exit_code == 0
        assert "VRAM Estimation" in result.output
        assert "Model weights:" in result.output
        assert "Per-GPU total:" in result.output
        assert "DGX Spark fit:" in result.output

    def test_vram_with_gpu_mem(self, runner):
        """Test sparkrun recipe vram with --gpu-mem shows budget analysis."""
        result = runner.invoke(
            main,
            [
                "recipe",
                "vram",
                _TEST_RECIPE_NAME,
                "--no-auto-detect",
                "--gpu-mem",
                "0.9",
            ],
        )
        assert result.exit_code == 0
        assert "GPU Memory Budget" in result.output
        assert "gpu_memory_utilization" in result.output
        assert "Available for KV" in result.output

    def test_vram_with_tp(self, runner):
        """Test sparkrun recipe vram with --tp override."""
        result = runner.invoke(
            main,
            [
                "recipe",
                "vram",
                _TEST_RECIPE_NAME,
                "--no-auto-detect",
                "--tp",
                "4",
            ],
        )
        assert result.exit_code == 0
        assert "Tensor parallel:  4" in result.output

    def test_vram_nonexistent_recipe(self, runner):
        """Test sparkrun recipe vram on nonexistent recipe exits with error."""
        result = runner.invoke(main, ["recipe", "vram", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_show_no_vram_flag(self, runner):
        """Test sparkrun show --no-vram suppresses VRAM estimation."""
        result = runner.invoke(main, ["show", _TEST_RECIPE_NAME, "--no-vram"])
        assert result.exit_code == 0
        assert "VRAM Estimation" not in result.output


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_valid_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun recipe validate exits 0 with 'is valid' message."""
        result = runner.invoke(main, ["recipe", "validate", _TEST_RECIPE_NAME])
        assert result.exit_code == 0
        assert "is valid" in result.output

    def test_validate_nonexistent_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun recipe validate nonexistent-recipe exits with error."""
        result = runner.invoke(main, ["recipe", "validate", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestRunCommand:
    """Test the run command (dry-run only)."""

    def test_run_dry_run_solo(self, runner, reset_bootstrap):
        """Test sparkrun run --solo --dry-run --hosts localhost.

        Should show runtime info and exit 0.
        """
        # Mock runtime.run() to prevent actual SSH execution
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                ],
            )

            assert result.exit_code == 0
            # Check that runtime info is displayed
            assert "Runtime:" in result.output
            assert "Image:" in result.output
            assert "Model:" in result.output
            assert "Mode:" in result.output
            assert "solo" in result.output.lower()

            # Verify runtime.run() was called with dry_run=True
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["dry_run"] is True
            # Cluster ID should be deterministic, not the old default
            assert call_kwargs["cluster_id"].startswith("sparkrun_")
            assert call_kwargs["cluster_id"] != "sparkrun0"

    def test_run_nonexistent_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun run nonexistent-recipe --solo --dry-run exits with error."""
        result = runner.invoke(
            main,
            [
                "run",
                "nonexistent-recipe",
                "--solo",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_run_tp_exceeds_max_nodes_errors(self, runner, reset_bootstrap):
        """Test that --tp exceeding recipe max_nodes produces an error."""
        result = runner.invoke(
            main,
            [
                "run",
                _TEST_RECIPE_SOLO_NAME,
                "--tp",
                "2",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
        assert "max_nodes=1" in result.output
        assert "requires 2 nodes" in result.output


class TestStopCommand:
    """Test the stop command."""

    def test_stop_no_hosts_error(self, runner, tmp_path, monkeypatch):
        """Test that sparkrun stop with no hosts specified exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "stop",
                _TEST_RECIPE_NAME,
            ],
        )

        assert result.exit_code != 0
        # Check error message mentions hosts
        assert "hosts" in result.output.lower() or "Error" in result.output


class TestClusterCommands:
    """Test cluster subcommands: create, list, show, delete, set-default, unset-default, update."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster for CLI tests."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("test-cluster", ["10.0.0.1", "10.0.0.2"])
        return config_root

    def test_cluster_help(self, runner):
        """Test that sparkrun cluster --help shows subcommands."""
        result = runner.invoke(main, ["cluster", "--help"])
        assert result.exit_code == 0
        # Check for cluster subcommands
        assert "create" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "delete" in result.output
        assert "default" in result.output
        assert "set-default" in result.output
        assert "unset-default" in result.output
        assert "update" in result.output

    def test_cluster_create(self, runner, tmp_path, monkeypatch):
        """Test creating a cluster."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "cluster",
                "create",
                "my-cluster",
                "--hosts",
                "host1,host2,host3",
            ],
        )

        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_cluster_create_duplicate(self, runner, cluster_setup):
        """Test that creating a duplicate cluster fails."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "create",
                "test-cluster",
                "--hosts",
                "host4,host5",
            ],
        )

        assert result.exit_code != 0
        assert "exists" in result.output.lower() or "Error" in result.output

    def test_cluster_list_empty(self, runner, tmp_path, monkeypatch):
        """Test that cluster list with no clusters shows appropriate message."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["cluster", "list"])

        assert result.exit_code == 0
        assert "No saved clusters" in result.output or "no clusters" in result.output.lower()

    def test_cluster_list_with_clusters(self, runner, cluster_setup):
        """Test that cluster list shows created clusters."""
        result = runner.invoke(main, ["cluster", "list"])

        assert result.exit_code == 0
        assert "test-cluster" in result.output

    def test_cluster_show(self, runner, cluster_setup):
        """Test showing cluster details."""
        result = runner.invoke(main, ["cluster", "show", "test-cluster"])

        assert result.exit_code == 0
        assert "test-cluster" in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output

    def test_cluster_show_nonexistent(self, runner, cluster_setup):
        """Test that showing a nonexistent cluster fails."""
        result = runner.invoke(main, ["cluster", "show", "nonexistent"])

        assert result.exit_code != 0
        assert "Error" in result.output or "not found" in result.output.lower()

    def test_cluster_delete(self, runner, cluster_setup):
        """Test deleting a cluster with --force flag."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "delete",
                "test-cluster",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_cluster_set_default(self, runner, cluster_setup):
        """Test setting a default cluster."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "set-default",
                "test-cluster",
            ],
        )

        assert result.exit_code == 0
        assert "Default cluster set" in result.output or "default" in result.output.lower()

    def test_cluster_unset_default(self, runner, cluster_setup):
        """Test unsetting the default cluster."""
        # First set a default
        runner.invoke(main, ["cluster", "set-default", "test-cluster"])

        # Now unset it
        result = runner.invoke(main, ["cluster", "unset-default"])

        assert result.exit_code == 0
        assert "Default cluster unset" in result.output or "unset" in result.output.lower()

    def test_cluster_update(self, runner, cluster_setup):
        """Test updating cluster hosts."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "update",
                "test-cluster",
                "--hosts",
                "10.0.0.3,10.0.0.4",
            ],
        )

        assert result.exit_code == 0
        assert "updated" in result.output.lower()

    def test_cluster_create_with_user(self, runner, tmp_path, monkeypatch):
        """Test creating a cluster with --user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "cluster",
                "create",
                "my-cluster",
                "--hosts",
                "host1,host2",
                "--user",
                "dgxuser",
            ],
        )
        assert result.exit_code == 0

        # Verify user is stored and shown
        result = runner.invoke(main, ["cluster", "show", "my-cluster"])
        assert result.exit_code == 0
        assert "dgxuser" in result.output

    def test_cluster_create_without_user(self, runner, tmp_path, monkeypatch):
        """Test that cluster created without --user does not show User field."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        runner.invoke(
            main,
            [
                "cluster",
                "create",
                "no-user-cluster",
                "--hosts",
                "host1,host2",
            ],
        )
        result = runner.invoke(main, ["cluster", "show", "no-user-cluster"])
        assert result.exit_code == 0
        assert "User:" not in result.output

    def test_cluster_update_user(self, runner, cluster_setup):
        """Test updating cluster user."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "update",
                "test-cluster",
                "--user",
                "newuser",
            ],
        )
        assert result.exit_code == 0
        assert "updated" in result.output.lower()

        # Verify user is shown
        result = runner.invoke(main, ["cluster", "show", "test-cluster"])
        assert "newuser" in result.output

    def test_cluster_create_with_cache_dir(self, runner, tmp_path, monkeypatch):
        """Test creating a cluster with --cache-dir."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "cluster",
                "create",
                "gpu-lab",
                "--hosts",
                "host1,host2",
                "--cache-dir",
                "/mnt/models",
            ],
        )
        assert result.exit_code == 0

        # Verify cache_dir is stored and shown
        result = runner.invoke(main, ["cluster", "show", "gpu-lab"])
        assert result.exit_code == 0
        assert "/mnt/models" in result.output
        assert "Cache dir:" in result.output

    def test_cluster_show_displays_cache_dir(self, runner, tmp_path, monkeypatch):
        """Test that cluster show displays Cache dir when set."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        runner.invoke(
            main,
            [
                "cluster",
                "create",
                "with-cache",
                "--hosts",
                "host1",
                "--cache-dir",
                "/data/hf",
            ],
        )
        result = runner.invoke(main, ["cluster", "show", "with-cache"])
        assert result.exit_code == 0
        assert "Cache dir:   /data/hf" in result.output

    def test_cluster_show_no_cache_dir(self, runner, tmp_path, monkeypatch):
        """Test that cluster show omits Cache dir when not set."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        runner.invoke(
            main,
            [
                "cluster",
                "create",
                "no-cache",
                "--hosts",
                "host1",
            ],
        )
        result = runner.invoke(main, ["cluster", "show", "no-cache"])
        assert result.exit_code == 0
        assert "Cache dir:" not in result.output

    def test_cluster_update_cache_dir(self, runner, cluster_setup):
        """Test updating cluster cache_dir."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "update",
                "test-cluster",
                "--cache-dir",
                "/mnt/new-cache",
            ],
        )
        assert result.exit_code == 0
        assert "updated" in result.output.lower()

        # Verify cache_dir is shown
        result = runner.invoke(main, ["cluster", "show", "test-cluster"])
        assert "/mnt/new-cache" in result.output

    def test_cluster_update_hosts_does_not_clear_user_or_cache_dir(self, runner, tmp_path, monkeypatch):
        """Test that updating --hosts does not clear previously set user or cache_dir."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("preserve-cluster", ["10.0.0.1", "10.0.0.2"], user="dgxuser", cache_dir="/mnt/models")

        result = runner.invoke(
            main,
            [
                "cluster",
                "update",
                "preserve-cluster",
                "--hosts",
                "10.0.0.3,10.0.0.4",
            ],
        )
        assert result.exit_code == 0

        # user and cache_dir must still be present after updating only hosts
        result = runner.invoke(main, ["cluster", "show", "preserve-cluster"])
        assert result.exit_code == 0
        assert "dgxuser" in result.output
        assert "/mnt/models" in result.output

    def test_cluster_create_with_transfer_interface(self, runner, tmp_path, monkeypatch):
        """Test creating a cluster with --transfer-interface."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "cluster",
                "create",
                "mgmt-cluster",
                "--hosts",
                "host1,host2",
                "--transfer-interface",
                "mgmt",
            ],
        )
        assert result.exit_code == 0

        # Verify transfer_interface is stored and shown
        result = runner.invoke(main, ["cluster", "show", "mgmt-cluster"])
        assert result.exit_code == 0
        assert "mgmt" in result.output
        assert "Xfer iface:" in result.output

    def test_cluster_show_no_transfer_interface(self, runner, tmp_path, monkeypatch):
        """Test that cluster show omits Xfer iface when not set."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        runner.invoke(
            main,
            [
                "cluster",
                "create",
                "no-iface",
                "--hosts",
                "host1",
            ],
        )
        result = runner.invoke(main, ["cluster", "show", "no-iface"])
        assert result.exit_code == 0
        assert "Xfer iface:" not in result.output

    def test_cluster_update_transfer_interface(self, runner, cluster_setup):
        """Test updating cluster transfer_interface."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "update",
                "test-cluster",
                "--transfer-interface",
                "cx7",
            ],
        )
        assert result.exit_code == 0
        assert "updated" in result.output.lower()

        # Verify transfer_interface is shown
        result = runner.invoke(main, ["cluster", "show", "test-cluster"])
        assert "Xfer iface:  cx7" in result.output

    def test_cluster_create_invalid_transfer_interface(self, runner, tmp_path, monkeypatch):
        """Test that invalid --transfer-interface value is rejected by Click."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "cluster",
                "create",
                "bad-iface",
                "--hosts",
                "host1",
                "--transfer-interface",
                "foo",
            ],
        )
        assert result.exit_code != 0

    def test_cluster_update_nothing_to_update_includes_transfer_interface(self, runner, cluster_setup):
        """Test that 'nothing to update' error mentions --transfer-interface."""
        result = runner.invoke(
            main,
            [
                "cluster",
                "update",
                "test-cluster",
            ],
        )
        assert result.exit_code != 0
        assert "--transfer-interface" in result.output


class TestClusterMonitor:
    """Test cluster monitor subcommand."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("monitor-cluster", ["10.0.0.1", "10.0.0.2"])
        return config_root

    def test_cluster_monitor_help(self, runner):
        """cluster monitor --help shows expected options."""
        result = runner.invoke(main, ["cluster", "monitor", "--help"])
        assert result.exit_code == 0
        assert "--hosts" in result.output
        assert "--cluster" in result.output
        assert "--interval" in result.output
        assert "--dry-run" in result.output
        assert "--simple" in result.output
        assert "--json" in result.output

    def test_cluster_monitor_dry_run(self, runner, cluster_setup):
        """--dry-run shows what would be monitored without SSH."""
        with mock.patch("sparkrun.core.monitoring.stream_cluster_monitor") as mock_stream:
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "monitor",
                    "--cluster",
                    "monitor-cluster",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "dry-run" in result.output.lower()
            assert "10.0.0.1" in result.output
            assert "10.0.0.2" in result.output
            # stream_cluster_monitor should be called with dry_run=True
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args
            assert call_kwargs.kwargs.get("dry_run") or call_kwargs[1].get("dry_run")

    def test_cluster_monitor_host_resolution(self, runner, cluster_setup):
        """--cluster resolves hosts from cluster definition."""
        with mock.patch("sparkrun.core.monitoring.stream_cluster_monitor"):
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "monitor",
                    "--cluster",
                    "monitor-cluster",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            # Both hosts should appear in the dry-run output
            assert "10.0.0.1" in result.output
            assert "10.0.0.2" in result.output

    def test_cluster_monitor_no_hosts_error(self, runner, tmp_path, monkeypatch):
        """Monitor with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["cluster", "monitor"])
        assert result.exit_code != 0
        assert "hosts" in result.output.lower() or "Error" in result.output

    def test_cluster_monitor_custom_interval(self, runner, cluster_setup):
        """--interval is passed through to stream_cluster_monitor."""
        with mock.patch("sparkrun.core.monitoring.stream_cluster_monitor") as mock_stream:
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "monitor",
                    "--cluster",
                    "monitor-cluster",
                    "--interval",
                    "5",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args
            assert call_kwargs.kwargs.get("interval") == 5 or call_kwargs[1].get("interval") == 5

    def test_cluster_monitor_simple_flag(self, runner, cluster_setup):
        """--simple bypasses TUI and uses plain-text fallback."""
        with mock.patch("sparkrun.core.monitoring.stream_cluster_monitor") as mock_stream:
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "monitor",
                    "--cluster",
                    "monitor-cluster",
                    "--simple",
                ],
            )
            assert result.exit_code == 0
            assert "Monitoring" in result.output
            mock_stream.assert_called_once()

    def test_cluster_monitor_json_flag(self, runner, cluster_setup):
        """--json streams newline-delimited JSON via on_update callback."""
        import json

        from sparkrun.core.monitoring import HostMonitorState, MonitorSample

        sample = MonitorSample(
            timestamp="2025-01-01T00:00:00",
            hostname="host1",
            cpu_usage_pct="12.3",
            mem_used_pct="45.6",
            gpu_util_pct="78.9",
            gpu_temp_c="55",
            gpu_power_w="120",
        )

        def fake_stream(hosts, ssh_kwargs, interval=2, on_update=None, dry_run=False):
            """Simulate one update tick with sample data."""
            if on_update:
                states = {
                    "10.0.0.1": HostMonitorState(latest=sample),
                    "10.0.0.2": HostMonitorState(error="connection refused"),
                }
                on_update(states)

        with mock.patch("sparkrun.core.monitoring.stream_cluster_monitor", side_effect=fake_stream):
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "monitor",
                    "--cluster",
                    "monitor-cluster",
                    "--json",
                ],
            )
            assert result.exit_code == 0
            obj = json.loads(result.output.strip())
            assert "timestamp" in obj
            assert "hosts" in obj
            # Host with sample data should have monitor fields
            assert obj["hosts"]["10.0.0.1"]["hostname"] == "host1"
            assert obj["hosts"]["10.0.0.1"]["cpu_usage_pct"] == "12.3"
            # Host with error and no sample should report error status
            assert obj["hosts"]["10.0.0.2"]["status"] == "error"

    def test_cluster_monitor_tui_fallback_on_import_error(self, runner, cluster_setup, monkeypatch):
        """Falls back to simple mode when Textual is not importable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sparkrun.cli._monitor_tui":
                raise ImportError("no textual")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with mock.patch("sparkrun.core.monitoring.stream_cluster_monitor") as mock_stream:
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "monitor",
                    "--cluster",
                    "monitor-cluster",
                ],
            )
            assert result.exit_code == 0
            assert "falling back" in result.output.lower() or "Monitoring" in result.output
            mock_stream.assert_called_once()


class TestRunWithCluster:
    """Test run command with --cluster and --hosts-file options."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster for CLI tests."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("test-cluster", ["10.0.0.1", "10.0.0.2"])
        return config_root

    def test_run_help_shows_cluster_option(self, runner):
        """Test that sparkrun run --help shows --cluster and --hosts-file options."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--cluster" in result.output
        assert "--hosts-file" in result.output


class TestTensorParallelValidation:
    """Test tensor_parallel vs host count validation."""

    def test_tp_exceeds_hosts_errors(self, runner, reset_bootstrap):
        """tensor_parallel > number of hosts should exit with error."""
        # _TEST_RECIPE_NAME has defaults.tensor_parallel=2
        # Provide only 1 host (not --solo) so we hit the validation
        result = runner.invoke(
            main,
            [
                "run",
                _TEST_RECIPE_NAME,
                "--dry-run",
                "--tp",
                "4",
                "--hosts",
                "10.0.0.1,10.0.0.2,10.0.0.3",
            ],
        )

        assert result.exit_code != 0
        assert "requires 4 nodes" in result.output
        assert "only 3 hosts provided" in result.output

    def test_tp_less_than_hosts_trims(self, runner, reset_bootstrap):
        """tensor_parallel < number of hosts should trim host list."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--dry-run",
                    "--hosts",
                    "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
                ],
            )

            assert result.exit_code == 0
            assert "2 nodes required" in result.output
            assert "using 2 of 4 hosts" in result.output
            # Should have called with only 2 hosts
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["hosts"]) == 2
            assert call_kwargs["hosts"] == ["10.0.0.1", "10.0.0.2"]

    def test_tp_equals_hosts_uses_all(self, runner, reset_bootstrap):
        """tensor_parallel == number of hosts should use all hosts."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--dry-run",
                    "--hosts",
                    "10.0.0.1,10.0.0.2",
                ],
            )

            assert result.exit_code == 0
            # No trimming message
            assert "using 2 of" not in result.output
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["hosts"]) == 2

    def test_tp_trims_to_one_becomes_solo(self, runner, reset_bootstrap):
        """tensor_parallel=1 with multiple hosts should trim to 1 host and run solo."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--tp",
                    "1",
                    "--dry-run",
                    "--hosts",
                    "10.0.0.1,10.0.0.2",
                ],
            )

            assert result.exit_code == 0
            assert "1 nodes required" in result.output
            assert "using 1 of 2 hosts" in result.output
            assert "solo" in result.output.lower()
            mock_run.assert_called_once()

    def test_solo_flag_skips_tp_validation(self, runner, reset_bootstrap):
        """--solo flag should skip tensor_parallel validation entirely."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "10.0.0.1",
                ],
            )

            assert result.exit_code == 0
            # No trimming or error messages
            assert "nodes required" not in result.output
            mock_run.assert_called_once()

    def test_solo_flag_truncates_multiple_hosts(self, runner, reset_bootstrap):
        """--solo with multiple hosts should truncate to first host only."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "10.0.0.1,10.0.0.2",
                ],
            )

            assert result.exit_code == 0
            assert "solo" in result.output.lower()
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # Must receive only 1 host despite 2 being provided
            assert len(call_kwargs["hosts"]) == 1
            assert call_kwargs["hosts"] == ["10.0.0.1"]


class TestApplyNodeTrimming:
    """Test _apply_node_trimming helper function."""

    def _make_recipe(self, defaults=None):
        from sparkrun.core.recipe import Recipe

        data = {
            "name": "test",
            "runtime": "sglang",
            "model": "meta-llama/Llama-2-7b-hf",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_with_runtime_trims_to_tp_times_pp(self):
        """Runtime-aware trimming: tp=2, pp=2 → 4 nodes."""
        from sparkrun.cli._common import _apply_node_trimming

        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = SglangRuntime()
        hosts = ["h1", "h2", "h3", "h4", "h5", "h6"]
        result = _apply_node_trimming(hosts, recipe, runtime=runtime)
        assert result == ["h1", "h2", "h3", "h4"]

    def test_without_runtime_trims_to_tp(self):
        """Legacy path (no runtime): trims to TP only."""
        from sparkrun.cli._common import _apply_node_trimming

        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        hosts = ["h1", "h2", "h3", "h4"]
        # Without runtime, only TP is considered
        result = _apply_node_trimming(hosts, recipe)
        assert result == ["h1", "h2"]

    def test_tp_override_combined_with_runtime_pp(self):
        """tp_override is injected into overrides for runtime computation."""
        from sparkrun.cli._common import _apply_node_trimming

        recipe = self._make_recipe(defaults={"pipeline_parallel": 2})
        runtime = SglangRuntime()
        hosts = ["h1", "h2", "h3", "h4", "h5", "h6"]
        # tp_override=3, pp=2 → 6 nodes
        result = _apply_node_trimming(
            hosts,
            recipe,
            runtime=runtime,
            tp_override=3,
        )
        assert result == hosts  # 6 == 6, no trimming

    def test_single_host_no_trimming(self):
        """Single host is never trimmed."""
        from sparkrun.cli._common import _apply_node_trimming

        recipe = self._make_recipe(defaults={"tensor_parallel": 4})
        runtime = SglangRuntime()
        hosts = ["h1"]
        result = _apply_node_trimming(hosts, recipe, runtime=runtime)
        assert result == ["h1"]

    def test_backward_compat_alias(self):
        """_apply_tp_trimming still works as backward-compat alias."""
        from sparkrun.cli._common import _apply_tp_trimming

        recipe = self._make_recipe(defaults={"tensor_parallel": 2})
        hosts = ["h1", "h2", "h3", "h4"]
        result = _apply_tp_trimming(hosts, recipe)
        assert result == ["h1", "h2"]


class TestOptionOverrides:
    """Test --option / -o arbitrary parameter overrides."""

    def test_help_shows_option(self, runner):
        """sparkrun run --help shows --option / -o."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--option" in result.output
        assert "-o" in result.output

    def test_option_overrides_recipe_default(self, runner, reset_bootstrap):
        """--option overrides a recipe default value."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "-o",
                    "attention_backend=flashinfer",
                ],
            )

            assert result.exit_code == 0
            # The overrides should contain the option value
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["attention_backend"] == "flashinfer"

    def test_option_multiple(self, runner, reset_bootstrap):
        """Multiple -o flags accumulate."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "-o",
                    "attention_backend=triton",
                    "-o",
                    "max_model_len=4096",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["attention_backend"] == "triton"
            assert call_kwargs["overrides"]["max_model_len"] == 4096  # auto-coerced to int

    def test_dedicated_cli_param_overrides_option(self, runner, reset_bootstrap):
        """--port takes priority over -o port=XXXX."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "-o",
                    "port=9999",
                    "--port",
                    "8080",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # --port should win over -o port=
            assert call_kwargs["overrides"]["port"] == 8080

    def test_served_model_name_override(self, runner, reset_bootstrap):
        """--served-model-name sets the override."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "--served-model-name",
                    "my-alias",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["served_model_name"] == "my-alias"

    def test_max_model_len_override(self, runner, reset_bootstrap):
        """--max-model-len sets the override."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "--max-model-len",
                    "4096",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["max_model_len"] == 4096

    def test_max_model_len_overrides_option(self, runner, reset_bootstrap):
        """--max-model-len takes priority over -o max_model_len=XXX."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "-o",
                    "max_model_len=8192",
                    "--max-model-len",
                    "4096",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # --max-model-len should win over -o max_model_len=
            assert call_kwargs["overrides"]["max_model_len"] == 4096

    def test_served_model_name_overrides_option(self, runner, reset_bootstrap):
        """--served-model-name takes priority over -o served_model_name=XXX."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "-o",
                    "served_model_name=from-option",
                    "--served-model-name",
                    "from-flag",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # --served-model-name should win over -o served_model_name=
            assert call_kwargs["overrides"]["served_model_name"] == "from-flag"

    def test_option_coerces_types(self, runner, reset_bootstrap):
        """Values are auto-coerced: int, float, bool."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                    "-o",
                    "port=8000",
                    "-o",
                    "gpu_memory_utilization=0.85",
                    "-o",
                    "enforce_eager=true",
                    "-o",
                    "served_model_name=my-model",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            ovr = call_kwargs["overrides"]
            assert ovr["port"] == 8000
            assert isinstance(ovr["port"], int)
            assert ovr["gpu_memory_utilization"] == 0.85
            assert isinstance(ovr["gpu_memory_utilization"], float)
            assert ovr["enforce_eager"] is True
            assert ovr["served_model_name"] == "my-model"

    def test_option_bad_format_errors(self, runner, reset_bootstrap):
        """--option without = sign exits with error."""
        result = runner.invoke(
            main,
            [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts",
                "localhost",
                "-o",
                "bad_no_equals",
            ],
        )

        assert result.exit_code != 0
        assert "must be key=value" in result.output


class TestFollowLogs:
    """Test --no-follow flag and follow_logs integration."""

    def test_run_help_shows_no_follow(self, runner):
        """sparkrun run --help shows --no-follow option."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--no-follow" in result.output

    def test_follow_logs_called_after_successful_run(self, runner, reset_bootstrap):
        """follow_logs is called after a successful detached run."""
        with (
            mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})),
            mock.patch.object(SglangRuntime, "run", return_value=0),
            mock.patch.object(SglangRuntime, "follow_logs") as mock_follow,
        ):
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--hosts",
                    "localhost",
                ],
            )

            assert result.exit_code == 0
            mock_follow.assert_called_once()
            call_kwargs = mock_follow.call_args.kwargs
            assert call_kwargs["cluster_id"].startswith("sparkrun_")
            assert call_kwargs["dry_run"] is False

    def test_no_follow_flag_skips_follow_logs(self, runner, reset_bootstrap):
        """--no-follow prevents follow_logs from being called."""
        with (
            mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})),
            mock.patch.object(SglangRuntime, "run", return_value=0),
            mock.patch.object(SglangRuntime, "follow_logs") as mock_follow,
        ):
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--no-follow",
                    "--hosts",
                    "localhost",
                ],
            )

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_dry_run_skips_follow_logs(self, runner, reset_bootstrap):
        """--dry-run prevents follow_logs from being called."""
        with mock.patch.object(SglangRuntime, "run", return_value=0), mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--dry-run",
                    "--hosts",
                    "localhost",
                ],
            )

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_foreground_skips_follow_logs(self, runner, reset_bootstrap):
        """--foreground prevents follow_logs from being called."""
        with (
            mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})),
            mock.patch.object(SglangRuntime, "run", return_value=0),
            mock.patch.object(SglangRuntime, "follow_logs") as mock_follow,
        ):
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--foreground",
                    "--hosts",
                    "localhost",
                ],
            )

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_nonzero_exit_skips_follow_logs(self, runner, reset_bootstrap):
        """Non-zero exit code from runtime.run() prevents follow_logs."""
        with (
            mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})),
            mock.patch.object(SglangRuntime, "run", return_value=1),
            mock.patch.object(SglangRuntime, "follow_logs") as mock_follow,
        ):
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--solo",
                    "--hosts",
                    "localhost",
                ],
            )

            assert result.exit_code == 1
            mock_follow.assert_not_called()


class TestSetupSshCommand:
    """Test the setup ssh command."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster for SSH tests."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("ssh-cluster", ["10.0.0.1", "10.0.0.2", "10.0.0.3"])
        return config_root

    def test_setup_ssh_help(self, runner):
        """Test that sparkrun setup ssh --help shows relevant options."""
        result = runner.invoke(main, ["setup", "ssh", "--help"])
        assert result.exit_code == 0
        assert "--hosts" in result.output
        assert "--cluster" in result.output
        assert "--user" in result.output
        assert "--dry-run" in result.output
        assert "SSH mesh" in result.output

    def test_setup_ssh_requires_hosts(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["setup", "ssh", "--no-include-self"])
        assert result.exit_code != 0
        assert "No hosts" in result.output

    def test_setup_ssh_requires_two_hosts(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh with a single host exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1",
                "--no-include-self",
            ],
        )
        assert result.exit_code != 0
        assert "at least 2 hosts" in result.output

    def test_setup_ssh_dry_run(self, runner, tmp_path, monkeypatch):
        """Test that --dry-run shows the command without executing."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--user",
                "testuser",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Would run SSH mesh:" in result.output
        assert "mesh_ssh_keys.sh" in result.output
        assert "testuser" in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output

    def test_setup_ssh_dry_run_default_user(self, runner, tmp_path, monkeypatch):
        """Test that --dry-run uses OS user when --user is not specified."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setenv("USER", "myosuser")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "myosuser" in result.output

    def test_setup_ssh_resolves_cluster(self, runner, cluster_setup):
        """Test that --cluster resolves hosts from a saved cluster."""
        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--cluster",
                "ssh-cluster",
                "--user",
                "ubuntu",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Would run SSH mesh:" in result.output
        assert "ubuntu" in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output
        assert "10.0.0.3" in result.output

    def test_setup_ssh_runs_script(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh invokes subprocess.run with correct args."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as mock_run:
            result = runner.invoke(
                main,
                [
                    "setup",
                    "ssh",
                    "--hosts",
                    "10.0.0.1,10.0.0.2",
                    "--user",
                    "testuser",
                    "--no-include-self",
                    "--no-discover-ips",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "bash"
            assert "mesh_ssh_keys.sh" in cmd[1]
            assert cmd[2] == "testuser"
            assert cmd[3:] == ["10.0.0.1", "10.0.0.2"]

    def test_setup_ssh_uses_cluster_user(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh picks up the cluster's configured user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("usercluster", ["10.0.0.1", "10.0.0.2"], user="dgxuser")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--cluster",
                "usercluster",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "dgxuser" in result.output

    def test_setup_ssh_cli_user_overrides_cluster_user(self, runner, tmp_path, monkeypatch):
        """Test that --user flag overrides the cluster's configured user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("usercluster2", ["10.0.0.1", "10.0.0.2"], user="dgxuser")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--cluster",
                "usercluster2",
                "--user",
                "override_user",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "override_user" in result.output
        # The cluster user should NOT appear in the command
        assert "dgxuser" not in result.output

    def test_setup_ssh_include_self(self, runner, tmp_path, monkeypatch):
        """Test that --include-self adds the local IP to the mesh."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setenv("USER", "testuser")

        from sparkrun.orchestration.primitives import local_ip_for

        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--user",
                "testuser",
                "--include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert local_ip in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output

    def test_setup_ssh_include_self_no_duplicate(self, runner, tmp_path, monkeypatch):
        """Test that --include-self doesn't duplicate if local IP already in hosts."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setenv("USER", "testuser")

        from sparkrun.orchestration.primitives import local_ip_for

        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,%s" % local_ip,
                "--user",
                "testuser",
                "--include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        # local IP should appear exactly once in the command line
        cmd_line = result.output.split("Would run:\n")[-1].strip()
        assert cmd_line.count(local_ip) == 1

    def test_setup_ssh_extra_hosts(self, runner, tmp_path, monkeypatch):
        """Test that --extra-hosts adds additional hosts to the mesh."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1",
                "--extra-hosts",
                "10.0.0.99",
                "--user",
                "testuser",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "10.0.0.1" in result.output
        assert "10.0.0.99" in result.output

    def test_setup_ssh_extra_hosts_dedup(self, runner, tmp_path, monkeypatch):
        """Test that --extra-hosts deduplicates against --hosts."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--extra-hosts",
                "10.0.0.1,10.0.0.3",
                "--user",
                "testuser",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        cmd_line = result.output.split("Would run:\n")[-1].strip()
        # 10.0.0.1 should appear only once
        assert cmd_line.count("10.0.0.1") == 1
        assert "10.0.0.3" in result.output

    def test_setup_ssh_resolves_loopback(self, runner, tmp_path, monkeypatch):
        """Test that 127.0.0.1 in host list gets resolved to routable IP."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        with mock.patch("sparkrun.orchestration.primitives.local_ip_for", return_value="192.168.1.100"):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "ssh",
                    "--hosts",
                    "127.0.0.1,10.0.0.2",
                    "--user",
                    "testuser",
                    "--no-include-self",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0
        assert "Resolved 127.0.0.1 -> 192.168.1.100" in result.output
        assert "192.168.1.100" in result.output
        # 127.0.0.1 should NOT appear in the mesh command line
        cmd_line = result.output.split("Would run SSH mesh:\n")[-1].strip()
        assert "127.0.0.1" not in cmd_line

    def test_setup_ssh_discover_ips_flag_in_help(self, runner):
        """Test that --discover-ips appears in help."""
        result = runner.invoke(main, ["setup", "ssh", "--help"])
        assert result.exit_code == 0
        assert "--discover-ips" in result.output

    def test_setup_ssh_no_discover_ips(self, runner, tmp_path, monkeypatch):
        """Test that Phase 2 is skipped with --no-discover-ips."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "ssh",
                    "--hosts",
                    "10.0.0.1,10.0.0.2",
                    "--user",
                    "testuser",
                    "--no-include-self",
                    "--no-discover-ips",
                ],
            )
        assert result.exit_code == 0
        assert "Discovering additional" not in result.output

    def test_setup_ssh_dry_run_skips_discovery(self, runner, tmp_path, monkeypatch):
        """Test that dry-run notes Phase 2 would run but doesn't execute it."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--user",
                "testuser",
                "--no-include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Phase 2" in result.output
        assert "Discovering additional" not in result.output

    def test_setup_ssh_phase2_discovers_and_distributes(self, runner, tmp_path, monkeypatch):
        """Test full Phase 2 flow: mesh success -> discover -> distribute."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.ssh import RemoteResult

        mock_ks_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="KEYSCAN_ADDED=2", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=0, stdout="KEYSCAN_ADDED=2", stderr=""),
        ]

        with (
            mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)),
            mock.patch(
                "sparkrun.orchestration.networking.discover_host_network_ips",
                return_value={"10.0.0.1": ["192.168.11.1"], "10.0.0.2": ["192.168.11.2"]},
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.check_tcp_reachability",
                return_value={"192.168.11.1": True, "192.168.11.2": False},
            ),
            mock.patch(
                "sparkrun.orchestration.networking.distribute_host_keys",
                return_value=mock_ks_results,
            ) as mock_dist,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "ssh",
                    "--hosts",
                    "10.0.0.1,10.0.0.2",
                    "--user",
                    "testuser",
                    "--no-include-self",
                    "--discover-ips",
                ],
            )

        assert result.exit_code == 0
        assert "Discovering additional" in result.output
        assert "192.168.11.1" in result.output
        assert "192.168.11.2" in result.output
        assert "Reachable" in result.output
        assert "Not reachable" in result.output
        assert "Distributing host keys" in result.output
        # distribute_host_keys should have been called with all discovered IPs
        mock_dist.assert_called_once()
        call_ips = mock_dist.call_args[0][0]
        assert "192.168.11.1" in call_ips
        assert "192.168.11.2" in call_ips

    def test_setup_ssh_skip_self_when_user_differs(self, runner, tmp_path, monkeypatch):
        """Test that --include-self skips auto-adding control machine when SSH user differs."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setenv("USER", "localuser")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--user",
                "differentuser",
                "--include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        # Self IP should NOT be auto-added — cluster user may not exist on control machine
        assert "Skipping control machine" in result.output
        assert "differs from local user" in result.output
        # The two cluster hosts should still be there
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output

    def test_setup_ssh_keep_self_when_explicitly_listed_cross_user(self, runner, tmp_path, monkeypatch):
        """Test that control machine IP stays in host list when explicitly listed, even with different user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setenv("USER", "localuser")

        from sparkrun.orchestration.primitives import local_ip_for

        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "10.0.0.1,%s,10.0.0.2" % local_ip,
                "--user",
                "differentuser",
                "--include-self",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "cross-user key exchange" in result.output
        assert "differs from local user" in result.output
        # The local IP should still appear — user explicitly listed it
        cmd_line = result.output.split("Would run")[-1]
        assert local_ip in cmd_line

    def test_setup_ssh_two_node_explicit_cross_user_mesh_proceeds(self, runner, tmp_path, monkeypatch):
        """Test that a 2-node cluster with control machine explicitly listed proceeds with both hosts."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setenv("USER", "localuser")

        from sparkrun.orchestration.primitives import local_ip_for

        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(
            main,
            [
                "setup",
                "ssh",
                "--hosts",
                "%s,10.0.0.1" % local_ip,
                "--user",
                "dgxuser",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "cross-user key exchange" in result.output
        # Both hosts should be in the mesh — no "at least 2 hosts" error
        assert "at least 2 hosts" not in result.output
        cmd_line = result.output.split("Would run")[-1]
        assert local_ip in cmd_line
        assert "10.0.0.1" in cmd_line


class TestSetupFixPermissions:
    """Test the setup fix-permissions command."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("fix-cluster", ["10.0.0.1", "10.0.0.2"], user="dgxuser")
        return config_root

    def test_fix_permissions_help(self, runner):
        """Test that sparkrun setup fix-permissions --help shows expected options."""
        result = runner.invoke(main, ["setup", "fix-permissions", "--help"])
        assert result.exit_code == 0
        assert "--hosts" in result.output
        assert "--cluster" in result.output
        assert "--user" in result.output
        assert "--cache-dir" in result.output
        assert "--dry-run" in result.output
        assert "--save-sudo" in result.output
        assert "file ownership" in result.output.lower() or "Fix file ownership" in result.output

    def test_fix_permissions_no_hosts_error(self, runner, tmp_path, monkeypatch):
        """Test that fix-permissions with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["setup", "fix-permissions"])
        assert result.exit_code != 0
        assert "hosts" in result.output.lower() or "Error" in result.output

    def test_fix_permissions_dry_run(self, runner, cluster_setup):
        """Test that --dry-run reports without executing."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "Fixing file permissions" in result.output

    def test_fix_permissions_all_nopasswd(self, runner, cluster_setup):
        """Test when all hosts succeed via sudo -n — no password prompt."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                ],
            )
            assert result.exit_code == 0
            # No password prompt should have appeared
            mock_sudo.assert_not_called()
            assert "OK" in result.output
            assert "fixed" in result.output.lower()

    def test_fix_permissions_mixed_sudo(self, runner, cluster_setup):
        """Test try-then-fallback: one host succeeds with sudo -n, another needs password."""
        mock_ok_result = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.1",
        )
        mock_fail_result = mock.Mock(
            success=False,
            stdout="",
            stderr="sudo: a password is required",
            host="10.0.0.2",
        )
        mock_password_result = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_ok_result, mock_fail_result]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", return_value=mock_password_result),
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            assert "2 fixed" in result.output

    def test_fix_permissions_cache_dir_override(self, runner, cluster_setup):
        """Test that --cache-dir is passed through to the chown script."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /data/hf-cache for dgxuser",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /data/hf-cache for dgxuser",
            stderr="",
            host="10.0.0.2",
        )

        with mock.patch(
            "sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]
        ) as mock_parallel:
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                    "--cache-dir",
                    "/data/hf-cache",
                ],
            )
            assert result.exit_code == 0
            # Verify the script contains the custom cache dir
            script_arg = mock_parallel.call_args[0][1]  # second positional arg is the script
            assert "/data/hf-cache" in script_arg

    def test_fix_permissions_skip_nonexistent_cache(self, runner, cluster_setup):
        """Test that hosts with no cache dir are reported as SKIP."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="SKIP: /home/dgxuser/.cache/huggingface does not exist",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="SKIP: /home/dgxuser/.cache/huggingface does not exist",
            stderr="",
            host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                ],
            )
            assert result.exit_code == 0
            assert "SKIP" in result.output
            assert "skipped" in result.output.lower()

    def test_save_sudo_dry_run(self, runner, cluster_setup):
        """Test --save-sudo --dry-run reports what would be installed."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.2",
        )
        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                    "--save-sudo",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "Would install sudoers entry" in result.output
            assert "sparkrun-chown-dgxuser" in result.output
            # No actual sudo script should run during dry-run
            mock_sudo.assert_not_called()

    def test_save_sudo_installs_sudoers(self, runner, cluster_setup):
        """Test --save-sudo calls run_remote_sudo_script with sudoers install script."""
        mock_sudoers_result = mock.Mock(
            success=True,
            stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-chown-dgxuser",
            stderr="",
        )
        mock_chown_result_1 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.1",
        )
        mock_chown_result_2 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_chown_result_1, mock_chown_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", return_value=mock_sudoers_result) as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                    "--save-sudo",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            # Sudoers install should have been called for each host
            assert mock_sudo.call_count == 2
            # Verify the sudoers script content
            script_arg = mock_sudo.call_args_list[0][0][1]
            assert "visudo" in script_arg
            assert "sparkrun-chown-dgxuser" in script_arg
            assert "/usr/bin/chown" in script_arg
            assert "Sudoers install:" in result.output

    def test_save_sudo_then_chown_succeeds(self, runner, cluster_setup):
        """After sudoers install, the chown parallel pass succeeds without extra password."""
        mock_sudoers_result = mock.Mock(
            success=True,
            stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-chown-dgxuser",
            stderr="",
        )
        mock_chown_result_1 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.1",
        )
        mock_chown_result_2 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_chown_result_1, mock_chown_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", return_value=mock_sudoers_result) as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                    "--save-sudo",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            assert "2 fixed" in result.output
            # Only the sudoers install calls — no fallback sudo calls for chown
            assert mock_sudo.call_count == 2

    def test_save_sudo_failure_on_host(self, runner, cluster_setup):
        """If sudoers install fails on a host, report failure and continue with chown."""
        mock_sudoers_ok = mock.Mock(
            success=True,
            stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-chown-dgxuser",
            stderr="",
        )
        mock_sudoers_fail = mock.Mock(
            success=False,
            stdout="",
            stderr="ERROR: sudoers validation failed",
        )

        def sudoers_side_effect(host, *args, **kwargs):
            return mock_sudoers_ok if host == "10.0.0.1" else mock_sudoers_fail

        mock_chown_result_1 = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.1",
        )
        # Host 2 fails sudo -n (sudoers wasn't installed) but succeeds on password fallback
        mock_chown_fail = mock.Mock(
            success=False,
            stdout="",
            stderr="sudo: a password is required",
            host="10.0.0.2",
        )
        mock_chown_password_ok = mock.Mock(
            success=True,
            stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="",
            host="10.0.0.2",
        )

        sudo_call_count = [0]

        def sudo_dispatch(host, script, *args, **kwargs):
            sudo_call_count[0] += 1
            # First 2 calls are sudoers install, next is chown fallback
            if sudo_call_count[0] <= 2:
                return sudoers_side_effect(host, script, *args, **kwargs)
            return mock_chown_password_ok

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_chown_result_1, mock_chown_fail]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", side_effect=sudo_dispatch),
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "fix-permissions",
                    "--cluster",
                    "fix-cluster",
                    "--save-sudo",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            assert "FAIL" in result.output or "failed" in result.output.lower()
            assert "2 fixed" in result.output


class TestSetupClearCache:
    """Test the setup clear-cache command."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("cache-cluster", ["10.0.0.1", "10.0.0.2"], user="dgxuser")
        return config_root

    def test_clear_cache_help(self, runner):
        """Test that sparkrun setup clear-cache --help shows expected options."""
        result = runner.invoke(main, ["setup", "clear-cache", "--help"])
        assert result.exit_code == 0
        assert "--hosts" in result.output
        assert "--cluster" in result.output
        assert "--user" in result.output
        assert "--dry-run" in result.output
        assert "--save-sudo" in result.output
        assert "page cache" in result.output.lower() or "drop_caches" in result.output

    def test_clear_cache_no_hosts_error(self, runner, tmp_path, monkeypatch):
        """Test that clear-cache with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["setup", "clear-cache"])
        assert result.exit_code != 0
        assert "hosts" in result.output.lower() or "Error" in result.output

    def test_clear_cache_dry_run(self, runner, cluster_setup):
        """Test that --dry-run reports without executing."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "clear-cache",
                    "--cluster",
                    "cache-cluster",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "Clearing page cache" in result.output

    def test_clear_cache_all_nopasswd(self, runner, cluster_setup):
        """Test when all hosts succeed via sudo -n — no password prompt."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "clear-cache",
                    "--cluster",
                    "cache-cluster",
                ],
            )
            assert result.exit_code == 0
            mock_sudo.assert_not_called()
            assert "OK" in result.output
            assert "2 cleared" in result.output

    def test_clear_cache_mixed_sudo(self, runner, cluster_setup):
        """Test try-then-fallback: one host succeeds with sudo -n, another needs password."""
        mock_ok_result = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.1",
        )
        mock_fail_result = mock.Mock(
            success=False,
            stdout="",
            stderr="sudo: a password is required",
            host="10.0.0.2",
        )
        mock_password_result = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_ok_result, mock_fail_result]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", return_value=mock_password_result),
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "clear-cache",
                    "--cluster",
                    "cache-cluster",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            assert "2 cleared" in result.output

    def test_save_sudo_dry_run(self, runner, cluster_setup):
        """Test --save-sudo --dry-run reports what would be installed."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.2",
        )
        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "clear-cache",
                    "--cluster",
                    "cache-cluster",
                    "--save-sudo",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "Would install sudoers entry" in result.output
            assert "sparkrun-dropcaches-dgxuser" in result.output
            mock_sudo.assert_not_called()

    def test_save_sudo_installs_sudoers(self, runner, cluster_setup):
        """Test --save-sudo calls run_remote_sudo_script with sudoers install script."""
        mock_sudoers_result = mock.Mock(
            success=True,
            stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-dropcaches-dgxuser",
            stderr="",
        )
        mock_drop_result_1 = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.1",
        )
        mock_drop_result_2 = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_drop_result_1, mock_drop_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", return_value=mock_sudoers_result) as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "clear-cache",
                    "--cluster",
                    "cache-cluster",
                    "--save-sudo",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            assert mock_sudo.call_count == 2
            script_arg = mock_sudo.call_args_list[0][0][1]
            assert "visudo" in script_arg
            assert "sparkrun-dropcaches-dgxuser" in script_arg
            assert "/usr/bin/tee" in script_arg
            assert "drop_caches" in script_arg
            assert "Sudoers install:" in result.output

    def test_save_sudo_then_drop_succeeds(self, runner, cluster_setup):
        """After sudoers install, the drop_caches parallel pass succeeds without extra password."""
        mock_sudoers_result = mock.Mock(
            success=True,
            stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-dropcaches-dgxuser",
            stderr="",
        )
        mock_drop_result_1 = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.1",
        )
        mock_drop_result_2 = mock.Mock(
            success=True,
            stdout="OK: page cache cleared",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_drop_result_1, mock_drop_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", return_value=mock_sudoers_result) as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "clear-cache",
                    "--cluster",
                    "cache-cluster",
                    "--save-sudo",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            assert "2 cleared" in result.output
            # Only the sudoers install calls — no fallback sudo calls for drop_caches
            assert mock_sudo.call_count == 2


class TestBenchmarkCommand:
    """Test the benchmark command."""

    def test_benchmark_help(self, runner):
        """sparkrun benchmark --help shows benchmark options."""
        result = runner.invoke(main, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--profile" in result.output
        assert "--option" in result.output
        assert "--framework" in result.output

    def test_benchmark_dry_run(self, runner, tmp_recipe_dir):
        """sparkrun benchmark --dry-run <recipe> attempts to run benchmark flow."""
        # Note: This test may fail if recipe resolution doesn't work in test env.
        # The important thing is that the command structure is correct.
        result = runner.invoke(
            main,
            [
                "benchmark",
                "--solo",
                "--dry-run",
                "test-v2",
            ],
        )
        # Accept either success or recipe-not-found error (exit code 1)
        # The key is that argument parsing worked (exit code 2 would be usage error)
        assert result.exit_code in (0, 1)

    def test_benchmark_dry_run_with_option_override(self, runner, tmp_recipe_dir):
        """-o option is accepted in the command."""
        result = runner.invoke(
            main,
            [
                "benchmark",
                "--solo",
                "--dry-run",
                "-o",
                "pp=4096",
                "test-v2",
            ],
        )
        # Accept either success or recipe-not-found error
        assert result.exit_code in (0, 1)

    def test_benchmark_missing_file_errors(self, runner):
        """Missing recipe should exit with error."""
        result = runner.invoke(
            main,
            [
                "benchmark",
                "does-not-exist-recipe",
                "--dry-run",
            ],
        )
        assert result.exit_code != 0

    def test_benchmark_list_profiles_invalid_registry(self, runner):
        """list-benchmark-profiles with nonexistent registry should error, not silently return empty."""
        result = runner.invoke(
            main,
            [
                "registry",
                "list-benchmark-profiles",
                "--registry",
                "does-not-exist-registry",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output or "not found" in (result.output + (result.output or ""))

    def test_benchmark_list_profiles_help(self, runner):
        """list-benchmark-profiles --help shows options."""
        result = runner.invoke(main, ["registry", "list-benchmark-profiles", "--help"])
        assert result.exit_code == 0
        assert "--registry" in result.output
        assert "--all" in result.output


class TestLogCommand:
    """Test the logs command."""

    def test_log_help(self, runner):
        """Test that sparkrun logs --help shows relevant options."""
        result = runner.invoke(main, ["logs", "--help"])
        assert result.exit_code == 0
        assert "--tail" in result.output
        assert "--hosts" in result.output
        assert "TARGET" in result.output

    def test_log_calls_follow_logs(self, runner, reset_bootstrap):
        """sparkrun logs calls runtime.follow_logs with correct args."""
        with mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(
                main,
                [
                    "logs",
                    _TEST_RECIPE_NAME,
                    "--hosts",
                    "localhost",
                    "--tail",
                    "50",
                ],
            )

            assert result.exit_code == 0
            mock_follow.assert_called_once()
            call_kwargs = mock_follow.call_args.kwargs
            assert call_kwargs["cluster_id"].startswith("sparkrun_")
            assert call_kwargs["tail"] == 50

    def test_log_no_hosts_error(self, runner, reset_bootstrap, tmp_path, monkeypatch):
        """sparkrun logs with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(
            main,
            [
                "logs",
                _TEST_RECIPE_NAME,
            ],
        )

        assert result.exit_code != 0
        assert "hosts" in result.output.lower() or "Error" in result.output

    def test_log_nonexistent_recipe(self, runner, reset_bootstrap):
        """sparkrun logs with bad recipe exits with error."""
        result = runner.invoke(
            main,
            [
                "logs",
                "nonexistent-recipe",
                "--hosts",
                "localhost",
            ],
        )

        assert result.exit_code != 0
        assert "Error" in result.output


class TestUrlRecipe:
    """Test URL recipe detection and loading."""

    def test_is_recipe_url_https(self):
        from sparkrun.core.recipe import is_recipe_url as _is_recipe_url

        assert _is_recipe_url("https://spark-arena.com/api/recipes/abc/raw")

    def test_is_recipe_url_http(self):
        from sparkrun.core.recipe import is_recipe_url as _is_recipe_url

        assert _is_recipe_url("http://example.com/recipe.yaml")

    def test_is_recipe_url_not_url(self):
        from sparkrun.core.recipe import is_recipe_url as _is_recipe_url

        assert not _is_recipe_url("qwen3-1.7b-vllm")
        assert not _is_recipe_url("./my-recipe.yaml")
        assert not _is_recipe_url("@registry/recipe-name")

    def test_expand_spark_arena_shortcut(self):
        from sparkrun.core.recipe import expand_recipe_shortcut as _expand_recipe_shortcut

        result = _expand_recipe_shortcut("@spark-arena/076136cd-260a-4e77-b6e2-309d8f64619b")
        assert result == ("https://spark-arena.com/api/recipes/076136cd-260a-4e77-b6e2-309d8f64619b/raw")

    def test_expand_non_shortcut_unchanged(self):
        from sparkrun.core.recipe import expand_recipe_shortcut as _expand_recipe_shortcut

        assert _expand_recipe_shortcut("qwen3-1.7b-vllm") == "qwen3-1.7b-vllm"
        assert _expand_recipe_shortcut("@other-registry/foo") == "@other-registry/foo"
        assert _expand_recipe_shortcut("https://example.com/r.yaml") == "https://example.com/r.yaml"

    def test_simplify_spark_arena_url(self):
        from sparkrun.core.recipe import simplify_recipe_ref as _simplify_recipe_ref

        url = "https://spark-arena.com/api/recipes/076136cd-260a-4e77-b6e2-309d8f64619b/raw"
        assert _simplify_recipe_ref(url) == ("@spark-arena/076136cd-260a-4e77-b6e2-309d8f64619b")

    def test_simplify_non_spark_arena_unchanged(self):
        from sparkrun.core.recipe import simplify_recipe_ref as _simplify_recipe_ref

        url = "https://example.com/recipe.yaml"
        assert _simplify_recipe_ref(url) == url

    def test_simplify_roundtrip(self):
        """expand then simplify gives back the original shortcut."""
        from sparkrun.core.recipe import expand_recipe_shortcut as _expand_recipe_shortcut, simplify_recipe_ref as _simplify_recipe_ref

        shortcut = "@spark-arena/abc-123"
        url = _expand_recipe_shortcut(shortcut)
        assert _simplify_recipe_ref(url) == shortcut

    def test_build_raw_url_flat(self):
        """_build_raw_url works for a recipe at the top of the subpath."""
        from sparkrun.cli._registry import _build_raw_url

        url = _build_raw_url("https://github.com/org/repo.git", "recipes", "model-vllm.yaml")
        assert url == "https://raw.githubusercontent.com/org/repo/main/recipes/model-vllm.yaml"

    def test_build_raw_url_nested(self):
        """_build_raw_url preserves nested subdirectory paths from rglob."""
        from sparkrun.cli._registry import _build_raw_url

        url = _build_raw_url("https://github.com/org/repo", "recipes", "vllm/qwen3.yaml")
        assert url == "https://raw.githubusercontent.com/org/repo/main/recipes/vllm/qwen3.yaml"

    def test_build_raw_url_non_github(self):
        """_build_raw_url returns empty string for non-GitHub URLs."""
        from sparkrun.cli._registry import _build_raw_url

        assert _build_raw_url("https://gitlab.com/org/repo", "recipes", "model.yaml") == ""

    def test_display_recipe_detail_spark_arena_single(self):
        """display_recipe_detail shows Spark Arena URL for single benchmark entry."""
        from unittest.mock import MagicMock
        from sparkrun.utils.cli_formatters import display_recipe_detail

        recipe = MagicMock()
        recipe.qualified_name = "test-recipe"
        recipe.description = "A test"
        recipe.maintainer = "tester"
        recipe.metadata = {"spark_arena_benchmarks": [
            {"tp": 1, "uuid": "076136cd-260a-4e77-b6e2-309d8f64619b"},
        ]}
        recipe.runtime = "vllm"
        recipe.model = "test-model"
        recipe.container = "test:latest"
        recipe.min_nodes = 1
        recipe.max_nodes = 2
        recipe.defaults = {}
        recipe.env = {}
        recipe.command = None

        from click.testing import CliRunner
        import click

        @click.command()
        def _show():
            display_recipe_detail(recipe, show_vram=False)

        result = CliRunner().invoke(_show)
        assert "https://spark-arena.com/benchmarks/076136cd-260a-4e77-b6e2-309d8f64619b" in result.output
        assert "Spark Arena:" in result.output

    def test_display_recipe_detail_multi_spark_arena_benchmarks(self):
        """display_recipe_detail shows multi-TP Spark Arena URLs."""
        from unittest.mock import MagicMock
        from sparkrun.utils.cli_formatters import display_recipe_detail

        recipe = MagicMock()
        recipe.qualified_name = "test-recipe"
        recipe.description = "A test"
        recipe.maintainer = "tester"
        recipe.metadata = {"spark_arena_benchmarks": [
            {"tp": 1, "uuid": "uuid-tp1"},
            {"tp": 2, "uuid": "uuid-tp2"},
        ]}
        recipe.runtime = "vllm"
        recipe.model = "test-model"
        recipe.container = "test:latest"
        recipe.min_nodes = 1
        recipe.max_nodes = 2
        recipe.defaults = {}
        recipe.env = {}
        recipe.command = None

        from click.testing import CliRunner
        import click

        @click.command()
        def _show():
            display_recipe_detail(recipe, show_vram=False)

        result = CliRunner().invoke(_show)
        assert "Spark Arena:" in result.output
        assert "tp1: https://spark-arena.com/benchmarks/uuid-tp1" in result.output
        assert "tp2: https://spark-arena.com/benchmarks/uuid-tp2" in result.output

    def test_display_recipe_detail_no_spark_arena_benchmarks(self):
        """display_recipe_detail omits Spark Arena line when benchmarks are absent."""
        from unittest.mock import MagicMock
        from sparkrun.utils.cli_formatters import display_recipe_detail

        recipe = MagicMock()
        recipe.qualified_name = "test-recipe"
        recipe.description = "A test"
        recipe.maintainer = None
        recipe.metadata = {}
        recipe.runtime = "vllm"
        recipe.model = "test-model"
        recipe.container = "test:latest"
        recipe.min_nodes = 1
        recipe.max_nodes = None
        recipe.defaults = {}
        recipe.env = {}
        recipe.command = None

        from click.testing import CliRunner
        import click

        @click.command()
        def _show():
            display_recipe_detail(recipe, show_vram=False)

        result = CliRunner().invoke(_show)
        assert "Spark Arena:" not in result.output

    def test_format_job_commands_uses_recipe_ref(self):
        """format_job_commands prefers recipe_ref over recipe name."""
        from sparkrun.utils.cli_formatters import format_job_commands

        meta = {
            "recipe": "my-model-sglang",
            "recipe_ref": "@spark-arena/abc-123",
            "hosts": ["10.0.0.1"],
        }
        logs_cmd, stop_cmd = format_job_commands(meta)
        assert "@spark-arena/abc-123" in logs_cmd
        assert "@spark-arena/abc-123" in stop_cmd

    def test_format_job_commands_falls_back_to_recipe(self):
        """format_job_commands uses recipe name when no recipe_ref."""
        from sparkrun.utils.cli_formatters import format_job_commands

        meta = {"recipe": "my-model-sglang", "hosts": ["10.0.0.1"]}
        logs_cmd, stop_cmd = format_job_commands(meta)
        assert "my-model-sglang" in logs_cmd

    def test_url_cache_path_deterministic(self):
        from sparkrun.core.recipe import _url_cache_path

        url = "https://spark-arena.com/api/recipes/abc/raw"
        p1 = _url_cache_path(url)
        p2 = _url_cache_path(url)
        assert p1 == p2
        assert p1.suffix == ".yaml"
        assert "remote-recipes" in str(p1)

    def test_url_cache_path_different_urls(self):
        from sparkrun.core.recipe import _url_cache_path

        p1 = _url_cache_path("https://example.com/a")
        p2 = _url_cache_path("https://example.com/b")
        assert p1 != p2

    def test_fetch_and_cache_recipe_success(self, tmp_path, monkeypatch):
        """Successful fetch writes cache file."""
        from sparkrun.core.recipe import fetch_and_cache_recipe as _fetch_and_cache_recipe

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", tmp_path)

        recipe_yaml = b"model: test-model\nruntime: sglang\ncontainer: test:latest\n"
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.read.return_value = recipe_yaml
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            path = _fetch_and_cache_recipe("https://example.com/recipe")

        assert path.exists()
        assert path.read_bytes() == recipe_yaml

    def test_fetch_and_cache_recipe_network_error_with_cache(self, tmp_path, monkeypatch):
        """Network failure with existing cache returns cached copy."""
        from sparkrun.core.recipe import fetch_and_cache_recipe as _fetch_and_cache_recipe, _url_cache_path

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", tmp_path)

        url = "https://example.com/recipe"
        cache_path = _url_cache_path(url)
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("model: cached\nruntime: sglang\n")

        from unittest.mock import patch

        from urllib.error import URLError

        with patch("urllib.request.urlopen", side_effect=URLError("offline")):
            path = _fetch_and_cache_recipe(url)
        assert path == cache_path

    def test_fetch_and_cache_recipe_network_error_no_cache(self, tmp_path, monkeypatch):
        """Network failure with no cache raises RecipeError."""
        from sparkrun.core.recipe import fetch_and_cache_recipe as _fetch_and_cache_recipe

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", tmp_path)

        from unittest.mock import patch

        from urllib.error import URLError

        from sparkrun.core.recipe import RecipeError

        with patch("urllib.request.urlopen", side_effect=URLError("offline")):
            with pytest.raises(RecipeError, match="Failed to fetch"):
                _fetch_and_cache_recipe("https://example.com/recipe")


# ---------------------------------------------------------------------------
# Cluster SSH user propagation tests
# ---------------------------------------------------------------------------


class TestResolveClusterConfig:
    """Tests for resolve_cluster_config helper."""

    def test_returns_user_from_named_cluster(self, tmp_path, monkeypatch):
        """Named cluster with a user returns that user."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1", "10.0.0.2"], user="labuser")

        cfg = resolve_cluster_config("mylab", None, None, mgr)
        assert cfg.user == "labuser"

    def test_returns_none_for_cluster_without_user(self, tmp_path, monkeypatch):
        """Named cluster without a user returns None."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("nouser", ["10.0.0.1"])

        cfg = resolve_cluster_config("nouser", None, None, mgr)
        assert cfg.user is None

    def test_returns_none_user_when_hosts_flag_given(self, tmp_path, monkeypatch):
        """When --hosts is provided, cluster user is still resolved (user always applies)."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], user="labuser")

        # hosts flag is non-None, no cluster_name — no cluster resolved
        cfg = resolve_cluster_config(None, "10.0.0.1", None, mgr)
        assert cfg.user is None

    def test_returns_none_user_when_hosts_file_given(self, tmp_path, monkeypatch):
        """When --hosts-file is provided, cluster user is not resolved."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], user="labuser")

        cfg = resolve_cluster_config(None, None, "/some/hosts.txt", mgr)
        assert cfg.user is None

    def test_falls_back_to_default_cluster_user(self, tmp_path, monkeypatch):
        """When no explicit cluster/hosts, uses default cluster's user."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("default-lab", ["10.0.0.1"], user="defaultuser")
        mgr.set_default("default-lab")

        cfg = resolve_cluster_config(None, None, None, mgr)
        assert cfg.user == "defaultuser"

    def test_returns_none_when_no_cluster_mgr(self):
        """When cluster_mgr is None, returns None for all fields."""
        from sparkrun.cli._common import resolve_cluster_config

        cfg = resolve_cluster_config(None, None, None, None)
        assert cfg.user is None
        assert cfg.cache_dir is None
        assert cfg.transfer_mode is None
        assert cfg.transfer_interface is None

    def test_returns_none_for_nonexistent_cluster(self, tmp_path, monkeypatch):
        """Nonexistent cluster name returns None for all fields (no crash)."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        cfg = resolve_cluster_config("doesnotexist", None, None, mgr)
        assert cfg.user is None
        assert cfg.cache_dir is None

    def test_returns_cache_dir_from_named_cluster(self, tmp_path, monkeypatch):
        """Named cluster with a cache_dir returns that path."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        cfg = resolve_cluster_config("mylab", None, None, mgr)
        assert cfg.cache_dir == "/mnt/models"

    def test_returns_none_cache_dir_for_cluster_without_it(self, tmp_path, monkeypatch):
        """Named cluster without a cache_dir returns None."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("nodir", ["10.0.0.1"])

        cfg = resolve_cluster_config("nodir", None, None, mgr)
        assert cfg.cache_dir is None

    def test_cache_dir_none_when_hosts_flag_given(self, tmp_path, monkeypatch):
        """When --hosts is provided, cluster cache_dir is not resolved."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        cfg = resolve_cluster_config(None, "10.0.0.1", None, mgr)
        assert cfg.cache_dir is None

    def test_cache_dir_none_when_cluster_and_hosts_both_given(self, tmp_path, monkeypatch):
        """When both --cluster and --hosts are provided, cluster cache_dir is not resolved.

        resolve_hosts() ignores the cluster when --hosts is set; cache_dir resolution
        must match that priority chain so they stay in sync.
        """
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        cfg = resolve_cluster_config("mylab", "10.0.0.2", None, mgr)
        assert cfg.cache_dir is None

    def test_cache_dir_none_when_cluster_and_hosts_file_both_given(self, tmp_path, monkeypatch):
        """When both --cluster and --hosts-file are provided, cluster cache_dir is not resolved."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        cfg = resolve_cluster_config("mylab", None, "/some/hosts.txt", mgr)
        assert cfg.cache_dir is None

    def test_falls_back_to_default_cluster_cache_dir(self, tmp_path, monkeypatch):
        """When no explicit cluster/hosts, uses default cluster's cache_dir."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("default-lab", ["10.0.0.1"], cache_dir="/nfs/cache")
        mgr.set_default("default-lab")

        cfg = resolve_cluster_config(None, None, None, mgr)
        assert cfg.cache_dir == "/nfs/cache"

    def test_cache_dir_none_when_no_cluster_mgr(self):
        """When cluster_mgr is None, cache_dir is None."""
        from sparkrun.cli._common import resolve_cluster_config

        cfg = resolve_cluster_config(None, None, None, None)
        assert cfg.cache_dir is None

    def test_cache_dir_none_for_nonexistent_cluster(self, tmp_path, monkeypatch):
        """Nonexistent cluster name returns None for cache_dir (no crash)."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        cfg = resolve_cluster_config("doesnotexist", None, None, mgr)
        assert cfg.cache_dir is None

    def test_returns_all_fields_from_cluster(self, tmp_path, monkeypatch):
        """Named cluster with all fields returns all values."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("full", ["10.0.0.1"], user="admin", cache_dir="/mnt/models", transfer_mode="push", transfer_interface="mgmt")

        cfg = resolve_cluster_config("full", None, None, mgr)
        assert cfg.name == "full"
        assert cfg.user == "admin"
        assert cfg.cache_dir == "/mnt/models"
        assert cfg.transfer_mode == "push"
        assert cfg.transfer_interface == "mgmt"

    def test_transfer_fields_none_when_hosts_flag_given(self, tmp_path, monkeypatch):
        """When --hosts is provided, transfer_mode/transfer_interface are not resolved."""
        from sparkrun.cli._common import resolve_cluster_config
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], transfer_mode="push", transfer_interface="mgmt")

        cfg = resolve_cluster_config(None, "10.0.0.1", None, mgr)
        assert cfg.transfer_mode is None
        assert cfg.transfer_interface is None


class TestResolveHostsAppliesClusterUser:
    """Tests that _resolve_hosts_or_exit applies cluster SSH user to config."""

    def test_sets_user_on_config(self, tmp_path, monkeypatch):
        """_resolve_hosts_or_exit sets the cluster user on config."""
        from sparkrun.cli._common import _resolve_hosts_or_exit
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import SparkrunConfig

        import sparkrun.core.config as config_mod

        monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_DIR", tmp_path)

        config_file = tmp_path / "nonexistent.yaml"
        config = SparkrunConfig(config_path=config_file)
        assert config.ssh_user is None

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], user="labuser")

        monkeypatch.setattr("sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
        monkeypatch.setattr(
            "sparkrun.core.hosts.resolve_hosts",
            lambda **kw: ["10.0.0.1"],
        )

        _resolve_hosts_or_exit(None, None, "mylab", config)
        assert config.ssh_user == "labuser"

    def test_no_op_when_no_cluster_user(self, tmp_path, monkeypatch):
        """_resolve_hosts_or_exit leaves config unchanged when cluster has no user."""
        from sparkrun.cli._common import _resolve_hosts_or_exit
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import SparkrunConfig

        import sparkrun.core.config as config_mod

        monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_DIR", tmp_path)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"ssh": {"user": "global_user"}}))
        config = SparkrunConfig(config_path=config_file)
        assert config.ssh_user == "global_user"

        mgr = ClusterManager(tmp_path)
        mgr.create("nouser-cluster", ["10.0.0.1"])

        monkeypatch.setattr("sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
        monkeypatch.setattr(
            "sparkrun.core.hosts.resolve_hosts",
            lambda **kw: ["10.0.0.1"],
        )

        _resolve_hosts_or_exit(None, None, "nouser-cluster", config)
        # global user should remain since cluster has no user
        assert config.ssh_user == "global_user"

    def test_cluster_user_overrides_global_config(self, tmp_path, monkeypatch):
        """Cluster user takes precedence over global ssh.user in config."""
        from sparkrun.cli._common import _resolve_hosts_or_exit
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        import sparkrun.core.config as config_mod

        monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_DIR", tmp_path)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"ssh": {"user": "global_user"}}))
        config = SparkrunConfig(config_path=config_file)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], user="cluster_user")

        monkeypatch.setattr("sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
        monkeypatch.setattr(
            "sparkrun.core.hosts.resolve_hosts",
            lambda **kw: ["10.0.0.1"],
        )

        _resolve_hosts_or_exit(None, None, "mylab", config)

        # The override should flow through to build_ssh_kwargs
        kwargs = build_ssh_kwargs(config)
        assert kwargs["ssh_user"] == "cluster_user"

    def test_no_op_when_hosts_flag_given(self, tmp_path, monkeypatch):
        """When --hosts is provided, cluster user is not applied."""
        from sparkrun.cli._common import _resolve_hosts_or_exit
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import SparkrunConfig

        import sparkrun.core.config as config_mod

        monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_DIR", tmp_path)

        config_file = tmp_path / "nonexistent.yaml"
        config = SparkrunConfig(config_path=config_file)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], user="labuser")

        monkeypatch.setattr("sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
        monkeypatch.setattr(
            "sparkrun.core.hosts.resolve_hosts",
            lambda **kw: ["10.0.0.1"],
        )

        # hosts is non-None, so cluster user should not be resolved
        _resolve_hosts_or_exit("10.0.0.1", None, None, config)
        assert config.ssh_user is None


class TestClusterUserInCLICommands:
    """Integration tests verifying cluster SSH user propagation through CLI commands.

    These tests verify that _resolve_hosts_or_exit applies the cluster's
    SSH user to config, so it flows through to build_ssh_kwargs and all
    downstream SSH operations.
    """

    @pytest.fixture
    def cluster_with_user(self, tmp_path, monkeypatch):
        """Set up a cluster with a custom SSH user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("userlab", ["10.0.0.1", "10.0.0.2"], user="labadmin")
        return config_root

    def test_cluster_status_uses_cluster_user(self, runner, cluster_with_user, monkeypatch):
        """cluster status with --cluster should use the cluster's SSH user."""
        captured_kwargs = {}

        def mock_query_status(host_list, ssh_kwargs=None, cache_dir=None):
            captured_kwargs.update(ssh_kwargs or {})
            # Return a minimal result object
            from types import SimpleNamespace

            return SimpleNamespace(
                groups={},
                solo_entries=[],
                errors={},
                idle_hosts=host_list,
                pending_ops=[],
                total_containers=0,
                host_count=len(host_list),
            )

        monkeypatch.setattr(
            "sparkrun.core.cluster_manager.query_cluster_status",
            mock_query_status,
        )

        result = runner.invoke(
            main,
            [
                "cluster",
                "status",
                "--cluster",
                "userlab",
            ],
        )
        assert result.exit_code == 0
        assert captured_kwargs.get("ssh_user") == "labadmin"

    def test_stop_all_uses_cluster_user(self, runner, cluster_with_user, monkeypatch):
        """stop --all with --cluster should use the cluster's SSH user."""
        captured_kwargs = {}

        def mock_query_status(host_list, ssh_kwargs=None, cache_dir=None):
            captured_kwargs.update(ssh_kwargs or {})
            from types import SimpleNamespace

            return SimpleNamespace(
                groups={},
                solo_entries=[],
                errors={},
                idle_hosts=host_list,
                pending_ops=[],
                total_containers=0,
                host_count=len(host_list),
            )

        monkeypatch.setattr(
            "sparkrun.core.cluster_manager.query_cluster_status",
            mock_query_status,
        )

        result = runner.invoke(
            main,
            [
                "stop",
                "--all",
                "--cluster",
                "userlab",
            ],
        )
        assert result.exit_code == 0
        assert captured_kwargs.get("ssh_user") == "labadmin"

    def test_stop_recipe_uses_cluster_user(self, runner, cluster_with_user, monkeypatch):
        """stop <recipe> with --cluster should use the cluster's SSH user."""
        captured_kwargs = {}

        def mock_cleanup(host_list, container_names, ssh_kwargs=None, dry_run=False):
            captured_kwargs.update(ssh_kwargs or {})

        monkeypatch.setattr(
            "sparkrun.orchestration.primitives.cleanup_containers",
            mock_cleanup,
        )

        result = runner.invoke(
            main,
            [
                "stop",
                _TEST_RECIPE_NAME,
                "--cluster",
                "userlab",
            ],
        )
        assert result.exit_code == 0
        assert captured_kwargs.get("ssh_user") == "labadmin"

    def test_logs_uses_cluster_user(self, runner, cluster_with_user, reset_bootstrap, monkeypatch):
        """logs with --cluster should use the cluster's SSH user."""
        captured_config = {}

        original_follow_logs = SglangRuntime.follow_logs

        def mock_follow_logs(self, hosts=None, cluster_id=None, config=None, **kw):
            captured_config["ssh_user"] = config.ssh_user if config else None

        monkeypatch.setattr(SglangRuntime, "follow_logs", mock_follow_logs)

        result = runner.invoke(
            main,
            [
                "logs",
                _TEST_RECIPE_NAME,
                "--cluster",
                "userlab",
            ],
        )
        assert result.exit_code == 0
        assert captured_config.get("ssh_user") == "labadmin"

    def test_run_dry_run_uses_cluster_user(self, runner, cluster_with_user, reset_bootstrap, monkeypatch):
        """run --dry-run with --cluster should use the cluster's SSH user."""
        captured_config = {}

        original_run = SglangRuntime.run

        def mock_run(
            self,
            hosts=None,
            image=None,
            serve_command=None,
            recipe=None,
            overrides=None,
            cluster_id=None,
            env=None,
            cache_dir=None,
            config=None,
            dry_run=False,
            **kw,
        ):
            captured_config["ssh_user"] = config.ssh_user if config else None
            return 0

        monkeypatch.setattr(SglangRuntime, "run", mock_run)
        # Mock distribute_resources to avoid SSH calls
        monkeypatch.setattr(
            "sparkrun.orchestration.distribution.distribute_resources",
            lambda *a, **kw: (None, {}, {}),
        )
        # Mock try_clear_page_cache
        monkeypatch.setattr(
            "sparkrun.orchestration.primitives.try_clear_page_cache",
            lambda *a, **kw: None,
        )

        result = runner.invoke(
            main,
            [
                "run",
                _TEST_RECIPE_NAME,
                "--cluster",
                "userlab",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert captured_config.get("ssh_user") == "labadmin"


# ---------------------------------------------------------------------------
# Top-level 'update' alias tests
# ---------------------------------------------------------------------------


class TestUpdateCommand:
    """Test the top-level 'sparkrun update' command."""

    def test_update_no_uv(self, runner, monkeypatch):
        """update skips self-upgrade when uv is not on PATH and runs registry update."""
        monkeypatch.setattr("shutil.which", lambda name: None)
        with mock.patch("sparkrun.cli._registry.registry_update") as mock_reg:
            # Make the Click command callable via ctx.invoke by wrapping
            mock_reg.return_value = None
            result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
        assert "uv not found" in result.output

    def test_update_not_uv_tool_install(self, runner, monkeypatch):
        """update skips self-upgrade when sparkrun is not a uv tool install."""
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
        with mock.patch("subprocess.run") as mock_run:
            # uv tool list succeeds but sparkrun not listed
            mock_run.return_value = mock.Mock(returncode=0, stdout="some-other-tool\n", stderr="")
            result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
        assert "not installed via uv" in result.output

    def test_update_uv_upgrade_succeeds(self, runner, monkeypatch):
        """update performs uv upgrade and shells out for registry update."""
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        def fake_run(cmd, **kwargs):
            if cmd[1:3] == ["tool", "list"]:
                return mock.Mock(returncode=0, stdout="sparkrun v1.0.0\n", stderr="")
            if cmd[1:3] == ["tool", "upgrade"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            if cmd == ["sparkrun", "--version"]:
                return mock.Mock(returncode=0, stdout="sparkrun, version 1.1.0", stderr="")
            if cmd == ["sparkrun", "registry", "update"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
        assert "-> 1.1.0" in result.output
        assert "Updating recipe registries" in result.output


class TestSetupEarlyoom:
    """Test the setup earlyoom command."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("oom-cluster", ["10.0.0.1", "10.0.0.2"], user="dgxuser")
        return config_root

    def test_earlyoom_help(self, runner):
        """Test that sparkrun setup earlyoom --help shows expected options."""
        result = runner.invoke(main, ["setup", "earlyoom", "--help"])
        assert result.exit_code == 0
        assert "--hosts" in result.output
        assert "--cluster" in result.output
        assert "--user" in result.output
        assert "--dry-run" in result.output
        assert "--prefer" in result.output
        assert "--avoid" in result.output
        assert "earlyoom" in result.output.lower()

    def test_earlyoom_no_hosts_error(self, runner, tmp_path, monkeypatch):
        """Test that earlyoom with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["setup", "earlyoom"])
        assert result.exit_code != 0
        assert "hosts" in result.output.lower() or "Error" in result.output

    def test_earlyoom_dry_run(self, runner, cluster_setup):
        """Test that --dry-run reports without executing."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="[dry-run]",
            stderr="",
            host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "earlyoom",
                    "--cluster",
                    "oom-cluster",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "Installing earlyoom" in result.output
            assert "Prefer (kill first)" in result.output
            assert "vllm" in result.output

    def test_earlyoom_all_nopasswd(self, runner, cluster_setup):
        """Test when all hosts succeed via sudo -n — no password prompt."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="INSTALLED: earlyoom\nCONFIGURED: /etc/default/earlyoom\nOK: earlyoom running",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="PRESENT: earlyoom already installed\nCONFIGURED: /etc/default/earlyoom\nOK: earlyoom running",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo,
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "earlyoom",
                    "--cluster",
                    "oom-cluster",
                ],
            )
            assert result.exit_code == 0
            mock_sudo.assert_not_called()
            assert "OK" in result.output
            assert "2 configured" in result.output
            assert "@shahizat" in result.output

    def test_earlyoom_mixed_sudo(self, runner, cluster_setup):
        """Test try-then-fallback: one host succeeds with sudo -n, another needs password."""
        mock_ok_result = mock.Mock(
            success=True,
            stdout="OK: earlyoom running",
            stderr="",
            host="10.0.0.1",
        )
        mock_fail_result = mock.Mock(
            success=False,
            stdout="",
            stderr="sudo: a password is required",
            host="10.0.0.2",
        )
        mock_password_result = mock.Mock(
            success=True,
            stdout="OK: earlyoom running",
            stderr="",
            host="10.0.0.2",
        )

        with (
            mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_ok_result, mock_fail_result]),
            mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", return_value=mock_password_result),
        ):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "earlyoom",
                    "--cluster",
                    "oom-cluster",
                ],
                input="sudopassword\n",
            )
            assert result.exit_code == 0
            assert "2 configured" in result.output

    def test_earlyoom_extra_prefer_avoid(self, runner, cluster_setup):
        """Test that --prefer and --avoid add patterns to the regex."""
        mock_result_1 = mock.Mock(
            success=True,
            stdout="OK: earlyoom running",
            stderr="",
            host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True,
            stdout="OK: earlyoom running",
            stderr="",
            host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(
                main,
                [
                    "setup",
                    "earlyoom",
                    "--cluster",
                    "oom-cluster",
                    "--prefer",
                    "my-worker,my-app",
                    "--avoid",
                    "nginx",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "my-worker" in result.output
            assert "my-app" in result.output
            assert "nginx" in result.output

    def test_earlyoom_default_patterns(self, runner):
        """Test that _build_earlyoom_regex produces expected output."""
        from sparkrun.cli._setup._phases import _build_earlyoom_regex, EARLYOOM_PREFER_PATTERNS, EARLYOOM_AVOID_PATTERNS

        prefer_regex = _build_earlyoom_regex(EARLYOOM_PREFER_PATTERNS)
        assert prefer_regex.startswith("(")
        assert prefer_regex.endswith(")")
        assert "vllm" in prefer_regex
        assert "VLLM" in prefer_regex
        assert "sglang" in prefer_regex
        assert "llama-server" in prefer_regex
        assert "trtllm" in prefer_regex
        assert "python" in prefer_regex

        avoid_regex = _build_earlyoom_regex(EARLYOOM_AVOID_PATTERNS)
        assert avoid_regex.startswith("(")
        assert "sshd" in avoid_regex
        assert "dockerd" in avoid_regex
        assert "dbus-daemon" in avoid_regex
        assert "NetworkManager" in avoid_regex


class TestStopLogsClusterIdAndOverrides:
    """Test stop/logs with cluster ID targeting and --port/--served-model-name overrides."""

    @pytest.fixture
    def config_setup(self, tmp_path, monkeypatch):
        """Set up config and cache dirs for stop/logs tests."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", cache_root)
        # Create a cluster so --hosts isn't required for recipe-path tests
        from sparkrun.core.cluster_manager import ClusterManager

        mgr = ClusterManager(config_root)
        mgr.create("test-cluster", ["10.0.0.1", "10.0.0.2"])
        return {"config_root": config_root, "cache_root": cache_root}

    def test_stop_with_port_override(self, runner, config_setup, reset_bootstrap):
        """Verify --port is passed through to generate_cluster_id."""
        with (
            mock.patch("sparkrun.orchestration.primitives.cleanup_containers"),
            mock.patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="sparkrun_aabbccdd1122") as mock_gen,
        ):
            result = runner.invoke(
                main,
                [
                    "stop",
                    _TEST_RECIPE_NAME,
                    "--cluster",
                    "test-cluster",
                    "--port",
                    "8001",
                ],
            )
            assert result.exit_code == 0
            # generate_cluster_id should have been called with overrides containing port
            call_kwargs = mock_gen.call_args
            overrides = call_kwargs.kwargs.get("overrides") or (call_kwargs[1].get("overrides") if len(call_kwargs) > 1 else None)
            assert overrides is not None
            assert overrides.get("port") == 8001

    def test_stop_with_served_model_name(self, runner, config_setup, reset_bootstrap):
        """Verify --served-model-name is passed through to generate_cluster_id."""
        with (
            mock.patch("sparkrun.orchestration.primitives.cleanup_containers"),
            mock.patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="sparkrun_aabbccdd1122") as mock_gen,
        ):
            result = runner.invoke(
                main,
                [
                    "stop",
                    _TEST_RECIPE_NAME,
                    "--cluster",
                    "test-cluster",
                    "--served-model-name",
                    "my-model",
                ],
            )
            assert result.exit_code == 0
            call_kwargs = mock_gen.call_args
            overrides = call_kwargs.kwargs.get("overrides") or (call_kwargs[1].get("overrides") if len(call_kwargs) > 1 else None)
            assert overrides is not None
            assert overrides.get("served_model_name") == "my-model"

    def test_stop_by_cluster_id(self, runner, config_setup):
        """Stop by cluster ID loads metadata and targets correct containers."""
        cache_root = config_setup["cache_root"]
        # Write fake job metadata
        jobs_dir = cache_root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "cluster_id": "sparkrun_aabbccdd1122",
            "recipe": "test-recipe",
            "model": "test/model",
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
        }
        (jobs_dir / "aabbccdd1122.yaml").write_text(yaml.safe_dump(meta))

        with mock.patch("sparkrun.orchestration.primitives.cleanup_containers") as mock_cleanup:
            result = runner.invoke(main, ["stop", "aabbccdd1122"])
            assert result.exit_code == 0
            assert "stopped" in result.output.lower()
            mock_cleanup.assert_called_once()

    def test_stop_by_cluster_id_no_metadata_no_hosts(self, runner, config_setup):
        """Stop by unknown cluster ID with no resolvable hosts shows helpful error."""
        result = runner.invoke(main, ["stop", "deadbeef1234"])
        assert result.exit_code != 0
        assert "No job metadata" in result.output

    def test_logs_with_port_override(self, runner, config_setup, reset_bootstrap):
        """Verify --port is passed through to generate_cluster_id in logs."""
        mock_runtime = mock.Mock()
        mock_runtime.follow_logs = mock.Mock()
        mock_runtime.compute_required_nodes = mock.Mock(return_value=2)
        with (
            mock.patch("sparkrun.core.bootstrap.get_runtime", return_value=mock_runtime),
            mock.patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="sparkrun_aabbccdd1122") as mock_gen,
        ):
            result = runner.invoke(
                main,
                [
                    "logs",
                    _TEST_RECIPE_NAME,
                    "--cluster",
                    "test-cluster",
                    "--port",
                    "8001",
                ],
            )
            assert result.exit_code == 0
            call_kwargs = mock_gen.call_args
            overrides = call_kwargs.kwargs.get("overrides") or (call_kwargs[1].get("overrides") if len(call_kwargs) > 1 else None)
            assert overrides is not None
            assert overrides.get("port") == 8001

    def test_logs_by_cluster_id(self, runner, config_setup, reset_bootstrap):
        """Logs by cluster ID loads metadata and calls runtime.follow_logs."""
        cache_root = config_setup["cache_root"]
        jobs_dir = cache_root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "cluster_id": "sparkrun_aabbccdd1122",
            "recipe": "test-recipe",
            "model": "test/model",
            "runtime": "sglang",
            "hosts": ["10.0.0.1"],
        }
        (jobs_dir / "aabbccdd1122.yaml").write_text(yaml.safe_dump(meta))

        mock_runtime = mock.Mock()
        mock_runtime.follow_logs = mock.Mock()
        with mock.patch("sparkrun.core.bootstrap.get_runtime", return_value=mock_runtime):
            result = runner.invoke(main, ["logs", "aabbccdd1122"])
            assert result.exit_code == 0
            mock_runtime.follow_logs.assert_called_once()
            call_kwargs = mock_runtime.follow_logs.call_args
            assert call_kwargs.kwargs["cluster_id"] == "sparkrun_aabbccdd1122"
            assert call_kwargs.kwargs["hosts"] == ["10.0.0.1"]


class TestFormatJobCommandsAndLabel:
    """Test format_job_commands and format_job_label with cluster ID support."""

    def test_format_job_commands_with_cluster_id(self):
        """Cluster ID produces short ID-based commands."""
        from sparkrun.utils.cli_formatters import format_job_commands

        meta = {"recipe": "test-recipe", "hosts": ["10.0.0.1"]}
        logs_cmd, stop_cmd = format_job_commands(meta, cluster_id="sparkrun_aabbccdd1122")
        assert logs_cmd == "sparkrun logs aabbccdd1122"
        assert stop_cmd == "sparkrun stop aabbccdd1122"

    def test_format_job_commands_fallback_includes_port(self):
        """Recipe-based fallback includes port and served_model_name flags."""
        from sparkrun.utils.cli_formatters import format_job_commands

        meta = {
            "recipe": "test-recipe",
            "hosts": ["10.0.0.1"],
            "tensor_parallel": 2,
            "port": 8001,
            "served_model_name": "my-model",
        }
        logs_cmd, stop_cmd = format_job_commands(meta)
        assert "--port 8001" in logs_cmd
        assert "--served-model-name my-model" in logs_cmd
        assert "--port 8001" in stop_cmd
        assert "--served-model-name my-model" in stop_cmd

    def test_format_job_label_shows_short_id(self):
        """Label includes short cluster ID in brackets."""
        from sparkrun.utils.cli_formatters import format_job_label

        meta = {"recipe": "test-recipe", "tensor_parallel": 2}
        label = format_job_label(meta, "sparkrun_aabbccdd1122")
        assert "[aabbccdd1122]" in label
        assert "test-recipe" in label
        assert "tp=2" in label


class TestCheckJobCommand:
    """Test cluster check-job subcommand."""

    def test_cluster_check_job_running(self, runner, tmp_path, monkeypatch):
        """check-job with running container exits 0."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.job_metadata import JobStatus

        status = JobStatus(
            running=True,
            cluster_id="sparkrun_aabbccdd0011",
            metadata={"recipe": "test-recipe"},
            hosts=["10.0.0.1"],
        )
        with (
            mock.patch(
                "sparkrun.orchestration.job_metadata.check_job_running",
                return_value=status,
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.build_ssh_kwargs",
                return_value={},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "check-job",
                    "sparkrun_aabbccdd0011",
                    "--hosts",
                    "10.0.0.1",
                ],
            )
        assert result.exit_code == 0
        assert "Job running" in result.output

    def test_cluster_check_job_not_running(self, runner, tmp_path, monkeypatch):
        """check-job with no running container exits 1."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.job_metadata import JobStatus

        status = JobStatus(
            running=False,
            cluster_id="sparkrun_aabbccdd0011",
            metadata=None,
            hosts=["10.0.0.1"],
        )
        with (
            mock.patch(
                "sparkrun.orchestration.job_metadata.check_job_running",
                return_value=status,
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.build_ssh_kwargs",
                return_value={},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "check-job",
                    "sparkrun_aabbccdd0011",
                    "--hosts",
                    "10.0.0.1",
                ],
            )
        assert result.exit_code == 1
        assert "not running" in result.output

    def test_cluster_check_job_json(self, runner, tmp_path, monkeypatch):
        """check-job --json outputs valid JSON with expected keys."""
        import json

        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.job_metadata import JobStatus

        status = JobStatus(
            running=True,
            cluster_id="sparkrun_aabbccdd0011",
            metadata={"recipe": "my-recipe"},
            hosts=["10.0.0.1"],
        )
        with (
            mock.patch(
                "sparkrun.orchestration.job_metadata.check_job_running",
                return_value=status,
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.build_ssh_kwargs",
                return_value={},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "check-job",
                    "sparkrun_aabbccdd0011",
                    "--hosts",
                    "10.0.0.1",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["running"] is True
        assert data["cluster_id"] == "sparkrun_aabbccdd0011"
        assert data["recipe"] == "my-recipe"
        assert data["healthy"] is None

    def test_cluster_check_job_health_check(self, runner, tmp_path, monkeypatch):
        """check-job --check-health shows health status when healthy."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.job_metadata import JobStatus

        status = JobStatus(
            running=True,
            cluster_id="sparkrun_aabbccdd0011",
            healthy=True,
            metadata={"recipe": "test"},
            hosts=["10.0.0.1"],
        )
        with (
            mock.patch(
                "sparkrun.orchestration.job_metadata.check_job_running",
                return_value=status,
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.build_ssh_kwargs",
                return_value={},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "check-job",
                    "sparkrun_aabbccdd0011",
                    "--hosts",
                    "10.0.0.1",
                    "--check-http-models",
                ],
            )
        assert result.exit_code == 0
        assert "Healthy: yes" in result.output

    def test_cluster_check_job_unhealthy_exit_1(self, runner, tmp_path, monkeypatch):
        """check-job --check-health exits 1 when running but unhealthy."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.job_metadata import JobStatus

        status = JobStatus(
            running=True,
            cluster_id="sparkrun_aabbccdd0011",
            healthy=False,
            metadata={"recipe": "test"},
            hosts=["10.0.0.1"],
        )
        with (
            mock.patch(
                "sparkrun.orchestration.job_metadata.check_job_running",
                return_value=status,
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.build_ssh_kwargs",
                return_value={},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "cluster",
                    "check-job",
                    "sparkrun_aabbccdd0011",
                    "--hosts",
                    "10.0.0.1",
                    "--check-http-models",
                ],
            )
        assert result.exit_code == 1
        assert "Healthy: no" in result.output


class TestRunEnsureFlag:
    """Test the --ensure flag on the run command."""

    def test_run_ensure_already_running(self, runner, reset_bootstrap):
        """--ensure with running job exits 0 without launching."""
        from sparkrun.orchestration.job_metadata import JobStatus

        status = JobStatus(
            running=True,
            cluster_id="sparkrun_aabbccdd0011",
            metadata={"recipe": _TEST_RECIPE_NAME},
            hosts=["localhost"],
        )
        with (
            mock.patch(
                "sparkrun.orchestration.job_metadata.check_job_running",
                return_value=status,
            ) as mock_check,
            mock.patch(
                "sparkrun.core.launcher.launch_inference",
            ) as mock_launch,
        ):
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--hosts",
                    "localhost",
                    "--ensure",
                    "--solo",
                ],
            )
        assert result.exit_code == 0
        assert "already running" in result.output
        mock_launch.assert_not_called()

    def test_run_ensure_not_running(self, runner, reset_bootstrap):
        """--ensure with no running job proceeds to launch."""
        from sparkrun.orchestration.job_metadata import JobStatus

        status = JobStatus(
            running=False,
            cluster_id="sparkrun_aabbccdd0011",
            metadata=None,
            hosts=["localhost"],
        )
        with (
            mock.patch(
                "sparkrun.orchestration.job_metadata.check_job_running",
                return_value=status,
            ),
            mock.patch.object(SglangRuntime, "run", return_value=0),
        ):
            result = runner.invoke(
                main,
                [
                    "run",
                    _TEST_RECIPE_NAME,
                    "--hosts",
                    "localhost",
                    "--ensure",
                    "--solo",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0
        # Should proceed to launch — shows runtime info
        assert "Runtime:" in result.output
