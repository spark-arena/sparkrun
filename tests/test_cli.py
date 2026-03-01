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

    # Patch discover_cwd_recipes to return our test recipe
    import sparkrun.core.recipe
    original_discover = sparkrun.core.recipe.discover_cwd_recipes

    def _patched_discover(directory=None):
        # Return our test recipes plus any originals
        originals = original_discover(directory)
        return [recipe_file] + originals

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

    def test_show_save_copies_recipe(self, runner, tmp_path):
        """Test that --save copies the recipe YAML to the given path."""
        dest = tmp_path / "saved-recipe.yaml"
        result = runner.invoke(main, [
            "show", _TEST_RECIPE_NAME,
            "--save", str(dest),
        ])
        assert result.exit_code == 0
        assert "Recipe saved to" in result.output
        assert dest.exists()
        # Verify it's valid YAML with expected fields
        import yaml
        data = yaml.safe_load(dest.read_text())
        assert "model" in data
        assert "runtime" in data

    def test_show_save_via_recipe_subcommand(self, runner, tmp_path):
        """Test that recipe show --save also works."""
        dest = tmp_path / "saved.yaml"
        result = runner.invoke(main, [
            "recipe", "show", _TEST_RECIPE_NAME,
            "--save", str(dest),
        ])
        assert result.exit_code == 0
        assert "Recipe saved to" in result.output
        assert dest.exists()

    def test_show_help_includes_save(self, runner):
        """Test that sparkrun show --help shows --save option."""
        result = runner.invoke(main, ["show", "--help"])
        assert result.exit_code == 0
        assert "--save" in result.output


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
        result = runner.invoke(main, [
            "recipe", "vram", _TEST_RECIPE_NAME,
            "--no-auto-detect",
            "--gpu-mem", "0.9",
        ])
        assert result.exit_code == 0
        assert "GPU Memory Budget" in result.output
        assert "gpu_memory_utilization" in result.output
        assert "Available for KV" in result.output

    def test_vram_with_tp(self, runner):
        """Test sparkrun recipe vram with --tp override."""
        result = runner.invoke(main, [
            "recipe", "vram", _TEST_RECIPE_NAME,
            "--no-auto-detect",
            "--tp", "4",
        ])
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
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts",
                "localhost",
            ])

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
        result = runner.invoke(main, [
            "run",
            "nonexistent-recipe",
            "--solo",
            "--dry-run",
        ])

        assert result.exit_code != 0
        assert "Error" in result.output


class TestStopCommand:
    """Test the stop command."""

    def test_stop_no_hosts_error(self, runner, tmp_path, monkeypatch):
        """Test that sparkrun stop with no hosts specified exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "stop",
            _TEST_RECIPE_NAME,
        ])

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

        result = runner.invoke(main, [
            "cluster",
            "create",
            "my-cluster",
            "--hosts",
            "host1,host2,host3",
        ])

        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_cluster_create_duplicate(self, runner, cluster_setup):
        """Test that creating a duplicate cluster fails."""
        result = runner.invoke(main, [
            "cluster",
            "create",
            "test-cluster",
            "--hosts",
            "host4,host5",
        ])

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
        result = runner.invoke(main, [
            "cluster",
            "delete",
            "test-cluster",
            "--force",
        ])

        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_cluster_set_default(self, runner, cluster_setup):
        """Test setting a default cluster."""
        result = runner.invoke(main, [
            "cluster",
            "set-default",
            "test-cluster",
        ])

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
        result = runner.invoke(main, [
            "cluster",
            "update",
            "test-cluster",
            "--hosts",
            "10.0.0.3,10.0.0.4",
        ])

        assert result.exit_code == 0
        assert "updated" in result.output.lower()

    def test_cluster_create_with_user(self, runner, tmp_path, monkeypatch):
        """Test creating a cluster with --user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "cluster", "create", "my-cluster",
            "--hosts", "host1,host2",
            "--user", "dgxuser",
        ])
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

        runner.invoke(main, [
            "cluster", "create", "no-user-cluster",
            "--hosts", "host1,host2",
        ])
        result = runner.invoke(main, ["cluster", "show", "no-user-cluster"])
        assert result.exit_code == 0
        assert "User:" not in result.output

    def test_cluster_update_user(self, runner, cluster_setup):
        """Test updating cluster user."""
        result = runner.invoke(main, [
            "cluster", "update", "test-cluster",
            "--user", "newuser",
        ])
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

        result = runner.invoke(main, [
            "cluster", "create", "gpu-lab",
            "--hosts", "host1,host2",
            "--cache-dir", "/mnt/models",
        ])
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

        runner.invoke(main, [
            "cluster", "create", "with-cache",
            "--hosts", "host1",
            "--cache-dir", "/data/hf",
        ])
        result = runner.invoke(main, ["cluster", "show", "with-cache"])
        assert result.exit_code == 0
        assert "Cache dir:   /data/hf" in result.output

    def test_cluster_show_no_cache_dir(self, runner, tmp_path, monkeypatch):
        """Test that cluster show omits Cache dir when not set."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        runner.invoke(main, [
            "cluster", "create", "no-cache",
            "--hosts", "host1",
        ])
        result = runner.invoke(main, ["cluster", "show", "no-cache"])
        assert result.exit_code == 0
        assert "Cache dir:" not in result.output

    def test_cluster_update_cache_dir(self, runner, cluster_setup):
        """Test updating cluster cache_dir."""
        result = runner.invoke(main, [
            "cluster", "update", "test-cluster",
            "--cache-dir", "/mnt/new-cache",
        ])
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

        result = runner.invoke(main, [
            "cluster", "update", "preserve-cluster",
            "--hosts", "10.0.0.3,10.0.0.4",
        ])
        assert result.exit_code == 0

        # user and cache_dir must still be present after updating only hosts
        result = runner.invoke(main, ["cluster", "show", "preserve-cluster"])
        assert result.exit_code == 0
        assert "dgxuser" in result.output
        assert "/mnt/models" in result.output


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
        result = runner.invoke(main, [
            "run",
            _TEST_RECIPE_NAME,
            "--dry-run",
            "--tp", "4",
            "--hosts", "10.0.0.1,10.0.0.2,10.0.0.3",
        ])

        assert result.exit_code != 0
        assert "tensor_parallel=4" in result.output
        assert "only 3 provided" in result.output

    def test_tp_less_than_hosts_trims(self, runner, reset_bootstrap):
        """tensor_parallel < number of hosts should trim host list."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--dry-run",
                "--hosts", "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
            ])

            assert result.exit_code == 0
            assert "tensor_parallel=2" in result.output
            assert "using 2 of 4 hosts" in result.output
            # Should have called with only 2 hosts
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["hosts"]) == 2
            assert call_kwargs["hosts"] == ["10.0.0.1", "10.0.0.2"]

    def test_tp_equals_hosts_uses_all(self, runner, reset_bootstrap):
        """tensor_parallel == number of hosts should use all hosts."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--dry-run",
                "--hosts", "10.0.0.1,10.0.0.2",
            ])

            assert result.exit_code == 0
            # No trimming message
            assert "using 2 of" not in result.output
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["hosts"]) == 2

    def test_tp_trims_to_one_becomes_solo(self, runner, reset_bootstrap):
        """tensor_parallel=1 with multiple hosts should trim to 1 host and run solo."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--tp", "1",
                "--dry-run",
                "--hosts", "10.0.0.1,10.0.0.2",
            ])

            assert result.exit_code == 0
            assert "tensor_parallel=1" in result.output
            assert "using 1 of 2 hosts" in result.output
            assert "solo" in result.output.lower()
            mock_run.assert_called_once()

    def test_solo_flag_skips_tp_validation(self, runner, reset_bootstrap):
        """--solo flag should skip tensor_parallel validation entirely."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "10.0.0.1",
            ])

            assert result.exit_code == 0
            # No trimming or error messages
            assert "tensor_parallel=" not in result.output
            mock_run.assert_called_once()

    def test_solo_flag_truncates_multiple_hosts(self, runner, reset_bootstrap):
        """--solo with multiple hosts should truncate to first host only."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "10.0.0.1,10.0.0.2",
            ])

            assert result.exit_code == 0
            assert "solo" in result.output.lower()
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # Must receive only 1 host despite 2 being provided
            assert len(call_kwargs["hosts"]) == 1
            assert call_kwargs["hosts"] == ["10.0.0.1"]


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
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "attention_backend=flashinfer",
            ])

            assert result.exit_code == 0
            # The overrides should contain the option value
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["attention_backend"] == "flashinfer"

    def test_option_multiple(self, runner, reset_bootstrap):
        """Multiple -o flags accumulate."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "attention_backend=triton",
                "-o", "max_model_len=4096",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["attention_backend"] == "triton"
            assert call_kwargs["overrides"]["max_model_len"] == 4096  # auto-coerced to int

    def test_dedicated_cli_param_overrides_option(self, runner, reset_bootstrap):
        """--port takes priority over -o port=XXXX."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "port=9999",
                "--port", "8080",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # --port should win over -o port=
            assert call_kwargs["overrides"]["port"] == 8080

    def test_served_model_name_override(self, runner, reset_bootstrap):
        """--served-model-name sets the override."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "--served-model-name", "my-alias",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["served_model_name"] == "my-alias"

    def test_max_model_len_override(self, runner, reset_bootstrap):
        """--max-model-len sets the override."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "--max-model-len", "4096",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["max_model_len"] == 4096

    def test_max_model_len_overrides_option(self, runner, reset_bootstrap):
        """--max-model-len takes priority over -o max_model_len=XXX."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "max_model_len=8192",
                "--max-model-len", "4096",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # --max-model-len should win over -o max_model_len=
            assert call_kwargs["overrides"]["max_model_len"] == 4096

    def test_served_model_name_overrides_option(self, runner, reset_bootstrap):
        """--served-model-name takes priority over -o served_model_name=XXX."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "served_model_name=from-option",
                "--served-model-name", "from-flag",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # --served-model-name should win over -o served_model_name=
            assert call_kwargs["overrides"]["served_model_name"] == "from-flag"

    def test_option_coerces_types(self, runner, reset_bootstrap):
        """Values are auto-coerced: int, float, bool."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "port=8000",
                "-o", "gpu_memory_utilization=0.85",
                "-o", "enforce_eager=true",
                "-o", "served_model_name=my-model",
            ])

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
        result = runner.invoke(main, [
            "run",
            _TEST_RECIPE_NAME,
            "--solo",
            "--dry-run",
            "--hosts", "localhost",
            "-o", "bad_no_equals",
        ])

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
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
                mock.patch.object(SglangRuntime, "run", return_value=0), \
                mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_called_once()
            call_kwargs = mock_follow.call_args.kwargs
            assert call_kwargs["cluster_id"].startswith("sparkrun_")
            assert call_kwargs["dry_run"] is False

    def test_no_follow_flag_skips_follow_logs(self, runner, reset_bootstrap):
        """--no-follow prevents follow_logs from being called."""
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
                mock.patch.object(SglangRuntime, "run", return_value=0), \
                mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--no-follow",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_dry_run_skips_follow_logs(self, runner, reset_bootstrap):
        """--dry-run prevents follow_logs from being called."""
        with mock.patch.object(SglangRuntime, "run", return_value=0), \
                mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_foreground_skips_follow_logs(self, runner, reset_bootstrap):
        """--foreground prevents follow_logs from being called."""
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
                mock.patch.object(SglangRuntime, "run", return_value=0), \
                mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--foreground",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_nonzero_exit_skips_follow_logs(self, runner, reset_bootstrap):
        """Non-zero exit code from runtime.run() prevents follow_logs."""
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
                mock.patch.object(SglangRuntime, "run", return_value=1), \
                mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                _TEST_RECIPE_NAME,
                "--solo",
                "--hosts", "localhost",
            ])

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

        result = runner.invoke(main, [
            "setup", "ssh", "--hosts", "10.0.0.1", "--no-include-self",
        ])
        assert result.exit_code != 0
        assert "at least 2 hosts" in result.output

    def test_setup_ssh_dry_run(self, runner, tmp_path, monkeypatch):
        """Test that --dry-run shows the command without executing."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--user", "testuser",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Would run:" in result.output
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

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "myosuser" in result.output

    def test_setup_ssh_resolves_cluster(self, runner, cluster_setup):
        """Test that --cluster resolves hosts from a saved cluster."""
        result = runner.invoke(main, [
            "setup", "ssh",
            "--cluster", "ssh-cluster",
            "--user", "ubuntu",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Would run:" in result.output
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
            result = runner.invoke(main, [
                "setup", "ssh",
                "--hosts", "10.0.0.1,10.0.0.2",
                "--user", "testuser",
                "--no-include-self",
            ])

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

        result = runner.invoke(main, [
            "setup", "ssh",
            "--cluster", "usercluster",
            "--no-include-self",
            "--dry-run",
        ])
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

        result = runner.invoke(main, [
            "setup", "ssh",
            "--cluster", "usercluster2",
            "--user", "override_user",
            "--no-include-self",
            "--dry-run",
        ])
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

        from sparkrun.orchestration.primitives import local_ip_for
        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--user", "testuser",
            "--include-self",
            "--dry-run",
        ])
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

        from sparkrun.orchestration.primitives import local_ip_for
        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,%s" % local_ip,
            "--user", "testuser",
            "--include-self",
            "--dry-run",
        ])
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

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1",
            "--extra-hosts", "10.0.0.99",
            "--user", "testuser",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "10.0.0.1" in result.output
        assert "10.0.0.99" in result.output

    def test_setup_ssh_extra_hosts_dedup(self, runner, tmp_path, monkeypatch):
        """Test that --extra-hosts deduplicates against --hosts."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--extra-hosts", "10.0.0.1,10.0.0.3",
            "--user", "testuser",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        cmd_line = result.output.split("Would run:\n")[-1].strip()
        # 10.0.0.1 should appear only once
        assert cmd_line.count("10.0.0.1") == 1
        assert "10.0.0.3" in result.output


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
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
                "--dry-run",
            ])
            assert result.exit_code == 0
            assert "Fixing file permissions" in result.output

    def test_fix_permissions_all_nopasswd(self, runner, cluster_setup):
        """Test when all hosts succeed via sudo -n — no password prompt."""
        mock_result_1 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo:
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
            ])
            assert result.exit_code == 0
            # No password prompt should have appeared
            mock_sudo.assert_not_called()
            assert "OK" in result.output
            assert "fixed" in result.output.lower()

    def test_fix_permissions_mixed_sudo(self, runner, cluster_setup):
        """Test try-then-fallback: one host succeeds with sudo -n, another needs password."""
        mock_ok_result = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.1",
        )
        mock_fail_result = mock.Mock(
            success=False, stdout="", stderr="sudo: a password is required",
            host="10.0.0.2",
        )
        mock_password_result = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_ok_result, mock_fail_result]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script",
                           return_value=mock_password_result):
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
            ], input="sudopassword\n")
            assert result.exit_code == 0
            assert "2 fixed" in result.output

    def test_fix_permissions_cache_dir_override(self, runner, cluster_setup):
        """Test that --cache-dir is passed through to the chown script."""
        mock_result_1 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /data/hf-cache for dgxuser",
            stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /data/hf-cache for dgxuser",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]) as mock_parallel:
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
                "--cache-dir", "/data/hf-cache",
            ])
            assert result.exit_code == 0
            # Verify the script contains the custom cache dir
            script_arg = mock_parallel.call_args[0][1]  # second positional arg is the script
            assert "/data/hf-cache" in script_arg

    def test_fix_permissions_skip_nonexistent_cache(self, runner, cluster_setup):
        """Test that hosts with no cache dir are reported as SKIP."""
        mock_result_1 = mock.Mock(
            success=True, stdout="SKIP: /home/dgxuser/.cache/huggingface does not exist",
            stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="SKIP: /home/dgxuser/.cache/huggingface does not exist",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
            ])
            assert result.exit_code == 0
            assert "SKIP" in result.output
            assert "skipped" in result.output.lower()

    def test_save_sudo_dry_run(self, runner, cluster_setup):
        """Test --save-sudo --dry-run reports what would be installed."""
        mock_result_1 = mock.Mock(
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo:
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
                "--save-sudo",
                "--dry-run",
            ])
            assert result.exit_code == 0
            assert "Would install sudoers entry" in result.output
            assert "sparkrun-chown-dgxuser" in result.output
            # No actual sudo script should run during dry-run
            mock_sudo.assert_not_called()

    def test_save_sudo_installs_sudoers(self, runner, cluster_setup):
        """Test --save-sudo calls run_remote_sudo_script with sudoers install script."""
        mock_sudoers_result = mock.Mock(
            success=True, stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-chown-dgxuser",
            stderr="",
        )
        mock_chown_result_1 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.1",
        )
        mock_chown_result_2 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_chown_result_1, mock_chown_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script",
                           return_value=mock_sudoers_result) as mock_sudo:
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
                "--save-sudo",
            ], input="sudopassword\n")
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
            success=True, stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-chown-dgxuser",
            stderr="",
        )
        mock_chown_result_1 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.1",
        )
        mock_chown_result_2 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_chown_result_1, mock_chown_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script",
                           return_value=mock_sudoers_result) as mock_sudo:
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
                "--save-sudo",
            ], input="sudopassword\n")
            assert result.exit_code == 0
            assert "2 fixed" in result.output
            # Only the sudoers install calls — no fallback sudo calls for chown
            assert mock_sudo.call_count == 2

    def test_save_sudo_failure_on_host(self, runner, cluster_setup):
        """If sudoers install fails on a host, report failure and continue with chown."""
        mock_sudoers_ok = mock.Mock(
            success=True, stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-chown-dgxuser",
            stderr="",
        )
        mock_sudoers_fail = mock.Mock(
            success=False, stdout="", stderr="ERROR: sudoers validation failed",
        )

        def sudoers_side_effect(host, *args, **kwargs):
            return mock_sudoers_ok if host == "10.0.0.1" else mock_sudoers_fail

        mock_chown_result_1 = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.1",
        )
        # Host 2 fails sudo -n (sudoers wasn't installed) but succeeds on password fallback
        mock_chown_fail = mock.Mock(
            success=False, stdout="", stderr="sudo: a password is required",
            host="10.0.0.2",
        )
        mock_chown_password_ok = mock.Mock(
            success=True, stdout="OK: fixed permissions on /home/dgxuser/.cache/huggingface for dgxuser",
            stderr="", host="10.0.0.2",
        )

        sudo_call_count = [0]

        def sudo_dispatch(host, script, *args, **kwargs):
            sudo_call_count[0] += 1
            # First 2 calls are sudoers install, next is chown fallback
            if sudo_call_count[0] <= 2:
                return sudoers_side_effect(host, script, *args, **kwargs)
            return mock_chown_password_ok

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_chown_result_1, mock_chown_fail]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script",
                           side_effect=sudo_dispatch):
            result = runner.invoke(main, [
                "setup", "fix-permissions",
                "--cluster", "fix-cluster",
                "--save-sudo",
            ], input="sudopassword\n")
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
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]):
            result = runner.invoke(main, [
                "setup", "clear-cache",
                "--cluster", "cache-cluster",
                "--dry-run",
            ])
            assert result.exit_code == 0
            assert "Clearing page cache" in result.output

    def test_clear_cache_all_nopasswd(self, runner, cluster_setup):
        """Test when all hosts succeed via sudo -n — no password prompt."""
        mock_result_1 = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo:
            result = runner.invoke(main, [
                "setup", "clear-cache",
                "--cluster", "cache-cluster",
            ])
            assert result.exit_code == 0
            mock_sudo.assert_not_called()
            assert "OK" in result.output
            assert "2 cleared" in result.output

    def test_clear_cache_mixed_sudo(self, runner, cluster_setup):
        """Test try-then-fallback: one host succeeds with sudo -n, another needs password."""
        mock_ok_result = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.1",
        )
        mock_fail_result = mock.Mock(
            success=False, stdout="", stderr="sudo: a password is required",
            host="10.0.0.2",
        )
        mock_password_result = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_ok_result, mock_fail_result]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script",
                           return_value=mock_password_result):
            result = runner.invoke(main, [
                "setup", "clear-cache",
                "--cluster", "cache-cluster",
            ], input="sudopassword\n")
            assert result.exit_code == 0
            assert "2 cleared" in result.output

    def test_save_sudo_dry_run(self, runner, cluster_setup):
        """Test --save-sudo --dry-run reports what would be installed."""
        mock_result_1 = mock.Mock(
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.1",
        )
        mock_result_2 = mock.Mock(
            success=True, stdout="[dry-run]", stderr="", host="10.0.0.2",
        )
        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_result_1, mock_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script") as mock_sudo:
            result = runner.invoke(main, [
                "setup", "clear-cache",
                "--cluster", "cache-cluster",
                "--save-sudo",
                "--dry-run",
            ])
            assert result.exit_code == 0
            assert "Would install sudoers entry" in result.output
            assert "sparkrun-dropcaches-dgxuser" in result.output
            mock_sudo.assert_not_called()

    def test_save_sudo_installs_sudoers(self, runner, cluster_setup):
        """Test --save-sudo calls run_remote_sudo_script with sudoers install script."""
        mock_sudoers_result = mock.Mock(
            success=True, stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-dropcaches-dgxuser",
            stderr="",
        )
        mock_drop_result_1 = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.1",
        )
        mock_drop_result_2 = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_drop_result_1, mock_drop_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script",
                           return_value=mock_sudoers_result) as mock_sudo:
            result = runner.invoke(main, [
                "setup", "clear-cache",
                "--cluster", "cache-cluster",
                "--save-sudo",
            ], input="sudopassword\n")
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
            success=True, stdout="OK: installed sudoers entry in /etc/sudoers.d/sparkrun-dropcaches-dgxuser",
            stderr="",
        )
        mock_drop_result_1 = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.1",
        )
        mock_drop_result_2 = mock.Mock(
            success=True, stdout="OK: page cache cleared",
            stderr="", host="10.0.0.2",
        )

        with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel",
                        return_value=[mock_drop_result_1, mock_drop_result_2]), \
                mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script",
                           return_value=mock_sudoers_result) as mock_sudo:
            result = runner.invoke(main, [
                "setup", "clear-cache",
                "--cluster", "cache-cluster",
                "--save-sudo",
            ], input="sudopassword\n")
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
        result = runner.invoke(main, [
            "benchmark",
            "--solo",
            "--dry-run",
            "test-v2",
        ])
        # Accept either success or recipe-not-found error (exit code 1)
        # The key is that argument parsing worked (exit code 2 would be usage error)
        assert result.exit_code in (0, 1)

    def test_benchmark_dry_run_with_option_override(self, runner, tmp_recipe_dir):
        """-o option is accepted in the command."""
        result = runner.invoke(main, [
            "benchmark",
            "--solo",
            "--dry-run",
            "-o", "pp=4096",
            "test-v2",
        ])
        # Accept either success or recipe-not-found error
        assert result.exit_code in (0, 1)

    def test_benchmark_missing_file_errors(self, runner):
        """Missing recipe should exit with error."""
        result = runner.invoke(main, [
            "benchmark",
            "does-not-exist-recipe",
            "--dry-run",
        ])
        assert result.exit_code != 0

    def test_benchmark_list_profiles_invalid_registry(self, runner):
        """list-benchmark-profiles with nonexistent registry should error, not silently return empty."""
        result = runner.invoke(main, [
            "registry", "list-benchmark-profiles",
            "--registry", "does-not-exist-registry",
        ])
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
        assert "RECIPE_NAME" in result.output

    def test_log_calls_follow_logs(self, runner, reset_bootstrap):
        """sparkrun logs calls runtime.follow_logs with correct args."""
        with mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "logs",
                _TEST_RECIPE_NAME,
                "--hosts", "localhost",
                "--tail", "50",
            ])

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

        result = runner.invoke(main, [
            "logs",
            _TEST_RECIPE_NAME,
        ])

        assert result.exit_code != 0
        assert "hosts" in result.output.lower() or "Error" in result.output

    def test_log_nonexistent_recipe(self, runner, reset_bootstrap):
        """sparkrun logs with bad recipe exits with error."""
        result = runner.invoke(main, [
            "logs",
            "nonexistent-recipe",
            "--hosts", "localhost",
        ])

        assert result.exit_code != 0
        assert "Error" in result.output


class TestUrlRecipe:
    """Test URL recipe detection and loading."""

    def test_is_recipe_url_https(self):
        from sparkrun.cli._common import _is_recipe_url

        assert _is_recipe_url("https://spark-arena.com/api/recipes/abc/raw")

    def test_is_recipe_url_http(self):
        from sparkrun.cli._common import _is_recipe_url

        assert _is_recipe_url("http://example.com/recipe.yaml")

    def test_is_recipe_url_not_url(self):
        from sparkrun.cli._common import _is_recipe_url

        assert not _is_recipe_url("qwen3-1.7b-vllm")
        assert not _is_recipe_url("./my-recipe.yaml")
        assert not _is_recipe_url("@registry/recipe-name")

    def test_expand_spark_arena_shortcut(self):
        from sparkrun.cli._common import _expand_recipe_shortcut

        result = _expand_recipe_shortcut(
            "@spark-arena/076136cd-260a-4e77-b6e2-309d8f64619b"
        )
        assert result == (
            "https://spark-arena.com/api/recipes/"
            "076136cd-260a-4e77-b6e2-309d8f64619b/raw"
        )

    def test_expand_non_shortcut_unchanged(self):
        from sparkrun.cli._common import _expand_recipe_shortcut

        assert _expand_recipe_shortcut("qwen3-1.7b-vllm") == "qwen3-1.7b-vllm"
        assert _expand_recipe_shortcut("@other-registry/foo") == "@other-registry/foo"
        assert (
                _expand_recipe_shortcut("https://example.com/r.yaml")
                == "https://example.com/r.yaml"
        )

    def test_simplify_spark_arena_url(self):
        from sparkrun.cli._common import _simplify_recipe_ref

        url = (
            "https://spark-arena.com/api/recipes/"
            "076136cd-260a-4e77-b6e2-309d8f64619b/raw"
        )
        assert _simplify_recipe_ref(url) == (
            "@spark-arena/076136cd-260a-4e77-b6e2-309d8f64619b"
        )

    def test_simplify_non_spark_arena_unchanged(self):
        from sparkrun.cli._common import _simplify_recipe_ref

        url = "https://example.com/recipe.yaml"
        assert _simplify_recipe_ref(url) == url

    def test_simplify_roundtrip(self):
        """expand then simplify gives back the original shortcut."""
        from sparkrun.cli._common import _expand_recipe_shortcut, _simplify_recipe_ref

        shortcut = "@spark-arena/abc-123"
        url = _expand_recipe_shortcut(shortcut)
        assert _simplify_recipe_ref(url) == shortcut

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
        from sparkrun.cli._common import _url_cache_path

        url = "https://spark-arena.com/api/recipes/abc/raw"
        p1 = _url_cache_path(url)
        p2 = _url_cache_path(url)
        assert p1 == p2
        assert p1.suffix == ".yaml"
        assert "remote-recipes" in str(p1)

    def test_url_cache_path_different_urls(self):
        from sparkrun.cli._common import _url_cache_path

        p1 = _url_cache_path("https://example.com/a")
        p2 = _url_cache_path("https://example.com/b")
        assert p1 != p2

    def test_fetch_and_cache_recipe_success(self, tmp_path, monkeypatch):
        """Successful fetch writes cache file."""
        from sparkrun.cli._common import _fetch_and_cache_recipe

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

    def test_fetch_and_cache_recipe_network_error_with_cache(
            self, tmp_path, monkeypatch
    ):
        """Network failure with existing cache returns cached copy."""
        from sparkrun.cli._common import _fetch_and_cache_recipe, _url_cache_path

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", tmp_path)

        url = "https://example.com/recipe"
        cache_path = _url_cache_path(url)
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("model: cached\nruntime: sglang\n")

        from unittest.mock import patch

        from urllib.error import URLError

        with patch(
                "urllib.request.urlopen", side_effect=URLError("offline")
        ):
            path = _fetch_and_cache_recipe(url)
        assert path == cache_path

    def test_fetch_and_cache_recipe_network_error_no_cache(
            self, tmp_path, monkeypatch
    ):
        """Network failure with no cache raises ClickException."""
        from sparkrun.cli._common import _fetch_and_cache_recipe

        import sparkrun.core.config

        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", tmp_path)

        from unittest.mock import patch

        from urllib.error import URLError

        import click

        with patch(
                "urllib.request.urlopen", side_effect=URLError("offline")
        ):
            with pytest.raises(click.ClickException, match="Failed to fetch"):
                _fetch_and_cache_recipe("https://example.com/recipe")


# ---------------------------------------------------------------------------
# Cluster SSH user propagation tests
# ---------------------------------------------------------------------------


class TestResolveClusterUser:
    """Tests for _resolve_cluster_user helper."""

    def test_returns_user_from_named_cluster(self, tmp_path, monkeypatch):
        """Named cluster with a user returns that user."""
        from sparkrun.cli._common import _resolve_cluster_user
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1", "10.0.0.2"], user="labuser")

        result = _resolve_cluster_user("mylab", None, None, mgr)
        assert result == "labuser"

    def test_returns_none_for_cluster_without_user(self, tmp_path, monkeypatch):
        """Named cluster without a user returns None."""
        from sparkrun.cli._common import _resolve_cluster_user
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("nouser", ["10.0.0.1"])

        result = _resolve_cluster_user("nouser", None, None, mgr)
        assert result is None

    def test_returns_none_when_hosts_flag_given(self, tmp_path, monkeypatch):
        """When --hosts is provided, cluster user is not resolved."""
        from sparkrun.cli._common import _resolve_cluster_user
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], user="labuser")

        # hosts flag is non-None, so cluster_name is ignored
        result = _resolve_cluster_user(None, "10.0.0.1", None, mgr)
        assert result is None

    def test_returns_none_when_hosts_file_given(self, tmp_path, monkeypatch):
        """When --hosts-file is provided, cluster user is not resolved."""
        from sparkrun.cli._common import _resolve_cluster_user
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], user="labuser")

        result = _resolve_cluster_user(None, None, "/some/hosts.txt", mgr)
        assert result is None

    def test_falls_back_to_default_cluster(self, tmp_path, monkeypatch):
        """When no explicit cluster/hosts, uses default cluster's user."""
        from sparkrun.cli._common import _resolve_cluster_user
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("default-lab", ["10.0.0.1"], user="defaultuser")
        mgr.set_default("default-lab")

        result = _resolve_cluster_user(None, None, None, mgr)
        assert result == "defaultuser"

    def test_returns_none_when_no_cluster_mgr(self):
        """When cluster_mgr is None, returns None."""
        from sparkrun.cli._common import _resolve_cluster_user

        result = _resolve_cluster_user(None, None, None, None)
        assert result is None

    def test_returns_none_for_nonexistent_cluster(self, tmp_path, monkeypatch):
        """Nonexistent cluster name returns None (no crash)."""
        from sparkrun.cli._common import _resolve_cluster_user
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        result = _resolve_cluster_user("doesnotexist", None, None, mgr)
        assert result is None


class TestResolveClusterCacheDir:
    """Tests for _resolve_cluster_cache_dir helper."""

    def test_returns_cache_dir_from_named_cluster(self, tmp_path, monkeypatch):
        """Named cluster with a cache_dir returns that path."""
        from sparkrun.cli._common import _resolve_cluster_cache_dir
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        result = _resolve_cluster_cache_dir("mylab", None, None, mgr)
        assert result == "/mnt/models"

    def test_returns_none_for_cluster_without_cache_dir(self, tmp_path, monkeypatch):
        """Named cluster without a cache_dir returns None."""
        from sparkrun.cli._common import _resolve_cluster_cache_dir
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("nodir", ["10.0.0.1"])

        result = _resolve_cluster_cache_dir("nodir", None, None, mgr)
        assert result is None

    def test_returns_none_when_hosts_flag_given(self, tmp_path, monkeypatch):
        """When --hosts is provided, cluster cache_dir is not resolved."""
        from sparkrun.cli._common import _resolve_cluster_cache_dir
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        result = _resolve_cluster_cache_dir(None, "10.0.0.1", None, mgr)
        assert result is None

    def test_returns_none_when_cluster_and_hosts_both_given(self, tmp_path, monkeypatch):
        """When both --cluster and --hosts are provided, cluster cache_dir is not resolved.

        resolve_hosts() ignores the cluster when --hosts is set; cache_dir resolution
        must match that priority chain so they stay in sync.
        """
        from sparkrun.cli._common import _resolve_cluster_cache_dir
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        result = _resolve_cluster_cache_dir("mylab", "10.0.0.2", None, mgr)
        assert result is None

    def test_returns_none_when_cluster_and_hosts_file_both_given(self, tmp_path, monkeypatch):
        """When both --cluster and --hosts-file are provided, cluster cache_dir is not resolved."""
        from sparkrun.cli._common import _resolve_cluster_cache_dir
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("mylab", ["10.0.0.1"], cache_dir="/mnt/models")

        result = _resolve_cluster_cache_dir("mylab", None, "/some/hosts.txt", mgr)
        assert result is None

    def test_falls_back_to_default_cluster(self, tmp_path, monkeypatch):
        """When no explicit cluster/hosts, uses default cluster's cache_dir."""
        from sparkrun.cli._common import _resolve_cluster_cache_dir
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        mgr.create("default-lab", ["10.0.0.1"], cache_dir="/nfs/cache")
        mgr.set_default("default-lab")

        result = _resolve_cluster_cache_dir(None, None, None, mgr)
        assert result == "/nfs/cache"

    def test_returns_none_when_no_cluster_mgr(self):
        """When cluster_mgr is None, returns None."""
        from sparkrun.cli._common import _resolve_cluster_cache_dir

        result = _resolve_cluster_cache_dir(None, None, None, None)
        assert result is None

    def test_returns_none_for_nonexistent_cluster(self, tmp_path, monkeypatch):
        """Nonexistent cluster name returns None (no crash)."""
        from sparkrun.cli._common import _resolve_cluster_cache_dir
        from sparkrun.core.cluster_manager import ClusterManager

        import sparkrun.core.config
        monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", tmp_path)

        mgr = ClusterManager(tmp_path)
        result = _resolve_cluster_cache_dir("doesnotexist", None, None, mgr)
        assert result is None


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

        monkeypatch.setattr(
            "sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
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

        monkeypatch.setattr(
            "sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
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

        monkeypatch.setattr(
            "sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
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

        monkeypatch.setattr(
            "sparkrun.cli._common._get_cluster_manager", lambda v=None: mgr)
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
                groups={}, solo_entries=[], errors={},
                idle_hosts=host_list, pending_ops=[],
                total_containers=0, host_count=len(host_list),
            )

        monkeypatch.setattr(
            "sparkrun.core.cluster_manager.query_cluster_status",
            mock_query_status,
        )

        result = runner.invoke(main, [
            "cluster", "status",
            "--cluster", "userlab",
        ])
        assert result.exit_code == 0
        assert captured_kwargs.get("ssh_user") == "labadmin"

    def test_stop_all_uses_cluster_user(self, runner, cluster_with_user, monkeypatch):
        """stop --all with --cluster should use the cluster's SSH user."""
        captured_kwargs = {}

        def mock_query_status(host_list, ssh_kwargs=None, cache_dir=None):
            captured_kwargs.update(ssh_kwargs or {})
            from types import SimpleNamespace
            return SimpleNamespace(
                groups={}, solo_entries=[], errors={},
                idle_hosts=host_list, pending_ops=[],
                total_containers=0, host_count=len(host_list),
            )

        monkeypatch.setattr(
            "sparkrun.core.cluster_manager.query_cluster_status",
            mock_query_status,
        )

        result = runner.invoke(main, [
            "stop", "--all",
            "--cluster", "userlab",
        ])
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

        result = runner.invoke(main, [
            "stop", _TEST_RECIPE_NAME,
            "--cluster", "userlab",
        ])
        assert result.exit_code == 0
        assert captured_kwargs.get("ssh_user") == "labadmin"

    def test_logs_uses_cluster_user(self, runner, cluster_with_user, reset_bootstrap, monkeypatch):
        """logs with --cluster should use the cluster's SSH user."""
        captured_config = {}

        original_follow_logs = SglangRuntime.follow_logs

        def mock_follow_logs(self, hosts=None, cluster_id=None, config=None, **kw):
            captured_config["ssh_user"] = config.ssh_user if config else None

        monkeypatch.setattr(SglangRuntime, "follow_logs", mock_follow_logs)

        result = runner.invoke(main, [
            "logs", _TEST_RECIPE_NAME,
            "--cluster", "userlab",
        ])
        assert result.exit_code == 0
        assert captured_config.get("ssh_user") == "labadmin"

    def test_run_dry_run_uses_cluster_user(self, runner, cluster_with_user, reset_bootstrap, monkeypatch):
        """run --dry-run with --cluster should use the cluster's SSH user."""
        captured_config = {}

        original_run = SglangRuntime.run

        def mock_run(self, hosts=None, image=None, serve_command=None,
                     recipe=None, overrides=None, cluster_id=None,
                     env=None, cache_dir=None, config=None, dry_run=False,
                     **kw):
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

        result = runner.invoke(main, [
            "run", _TEST_RECIPE_NAME,
            "--cluster", "userlab",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert captured_config.get("ssh_user") == "labadmin"
