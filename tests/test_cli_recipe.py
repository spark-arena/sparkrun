"""Tests for sparkrun recipe CLI subcommands."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.cli import main


@pytest.fixture
def runner():
    """Create a CliRunner instance."""
    return CliRunner()


@pytest.fixture
def registry_setup(tmp_path: Path, monkeypatch):
    """Set up config with a fake cached registry containing recipe files."""
    config_root = tmp_path / "config"
    cache_root = tmp_path / "cache" / "registries"
    config_root.mkdir(parents=True)
    cache_root.mkdir(parents=True)

    # Point SparkrunConfig to our temp dirs
    import sparkrun.core.config

    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", tmp_path / "cache")

    # Write registries.yaml
    registry_data = {
        "registries": [
            {
                "name": "test-registry",
                "url": "https://github.com/example/repo",
                "subpath": "recipes",
                "description": "Test recipes",
                "enabled": True,
            }
        ]
    }
    with open(config_root / "registries.yaml", "w") as f:
        yaml.dump(registry_data, f)

    # Create fake cached repo with recipes
    recipe_dir = cache_root / "test-registry" / "recipes"
    recipe_dir.mkdir(parents=True)
    (cache_root / "test-registry" / ".git").mkdir()

    recipe1 = {
        "sparkrun_version": "2",
        "name": "Test vLLM Recipe",
        "description": "A test recipe for vLLM inference",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "container": "scitrera/dgx-spark-vllm:latest",
    }
    recipe2 = {
        "sparkrun_version": "2",
        "name": "Test SGLang Recipe",
        "description": "A test recipe for SGLang inference",
        "model": "mistralai/Mistral-7B-v0.1",
        "runtime": "sglang",
        "container": "scitrera/dgx-spark-sglang:latest",
    }
    with open(recipe_dir / "test-vllm.yaml", "w") as f:
        yaml.dump(recipe1, f)
    with open(recipe_dir / "test-sglang.yaml", "w") as f:
        yaml.dump(recipe2, f)

    return config_root, cache_root


class TestRecipeHelp:
    """Test recipe subcommand help."""

    def test_recipe_help(self, runner):
        """Test that sparkrun recipe --help shows subcommands."""
        result = runner.invoke(main, ["recipe", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "search" in result.output
        assert "show" in result.output

    def test_main_help_includes_recipe(self, runner):
        """Test that sparkrun --help includes recipe subgroup."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "recipe" in result.output


class TestRecipeList:
    """Test recipe list command."""

    def test_recipe_list_with_registry(self, runner, registry_setup):
        """Test listing recipes from a registry."""
        result = runner.invoke(main, ["recipe", "list"])
        assert result.exit_code == 0
        assert "Name" in result.output
        assert "Runtime" in result.output
        assert "test-vllm" in result.output.lower() or "vllm" in result.output.lower()

    def test_recipe_list_with_query(self, runner, registry_setup):
        """Test listing recipes with a search query."""
        result = runner.invoke(main, ["recipe", "list", "llama"])
        assert result.exit_code == 0
        assert "vllm" in result.output.lower()
        # SGLang recipe uses Mistral, should not match llama
        assert "sglang" not in result.output.lower()

    def test_recipe_list_no_results(self, runner, registry_setup):
        """Test listing recipes with no matches."""
        result = runner.invoke(main, ["recipe", "list", "nonexistent-xyz-123"])
        assert result.exit_code == 0
        assert "No recipes found" in result.output


class TestRecipeSearch:
    """Test recipe search command."""

    def test_search_by_name(self, runner, registry_setup):
        """Test searching recipes by name."""
        result = runner.invoke(main, ["recipe", "search", "vllm"])
        assert result.exit_code == 0
        assert "vLLM" in result.output or "vllm" in result.output.lower()

    def test_search_by_model(self, runner, registry_setup):
        """Test searching recipes by model name."""
        result = runner.invoke(main, ["recipe", "search", "mistral"])
        assert result.exit_code == 0
        assert "sglang" in result.output.lower()

    def test_search_no_results(self, runner, registry_setup):
        """Test search with no results."""
        result = runner.invoke(main, ["recipe", "search", "nonexistent-model-xyz"])
        assert result.exit_code == 0
        assert "No recipes found" in result.output


class TestRecipeShow:
    """Test recipe show command."""

    def test_show_recipe_from_registry(self, runner, registry_setup):
        """Test showing a recipe from a registry."""
        result = runner.invoke(main, ["recipe", "show", "test-vllm"])
        assert result.exit_code == 0
        assert "Name:" in result.output
        assert "Runtime:" in result.output
        assert "Model:" in result.output

    def test_show_nonexistent(self, runner, registry_setup):
        """Test showing a nonexistent recipe."""
        result = runner.invoke(main, ["recipe", "show", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestRecipeUpdate:
    """Test recipe update command."""

    def test_update_all(self, runner, registry_setup):
        """Test updating all registries."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            result = runner.invoke(main, ["recipe", "update"])
            assert result.exit_code == 0
            assert "updated" in result.output.lower()

    def test_update_specific_registry(self, runner, registry_setup):
        """Test updating a specific registry."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            result = runner.invoke(main, ["recipe", "update", "--registry", "test-registry"])
            assert result.exit_code == 0
            assert "test-registry" in result.output

    def test_update_nonexistent_registry(self, runner, registry_setup):
        """Test updating a nonexistent registry."""
        result = runner.invoke(main, ["recipe", "update", "--registry", "nonexistent"])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestRegistryList:
    """Test registry list command."""

    def test_list_registries(self, runner, registry_setup):
        """Test listing configured registries."""
        result = runner.invoke(main, ["registry", "list"])
        assert result.exit_code == 0
        assert "test-registry" in result.output
        assert "https://github.com/example/repo" in result.output
        assert "yes" in result.output.lower()  # enabled


class TestRegistryAddRemove:
    """Test registry add and remove commands."""

    def test_add_registry_from_url(self, runner, registry_setup):
        """Test adding registries from a URL (manifest-based discovery)."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="clone fail")
            result = runner.invoke(
                main,
                [
                    "registry",
                    "add",
                    "https://github.com/test/repo",
                ],
            )
            # Clone failure → error exit
            assert result.exit_code != 0
            assert "Error" in result.output

    def test_add_registry_auto_updates_added_entries(self, runner, registry_setup):
        """After adding registries from a URL, update() is invoked to fetch contents."""
        from sparkrun.core.registry import RegistryEntry

        added = [
            RegistryEntry(
                name="freshly-added",
                url="https://github.com/test/repo",
                subpath="recipes",
                description="freshly added",
                enabled=True,
            )
        ]
        with (
            mock.patch(
                "sparkrun.core.registry.RegistryManager.add_registry_from_url",
                return_value=added,
            ),
            mock.patch("sparkrun.core.registry.RegistryManager.update") as mock_update,
        ):
            mock_update.return_value = {"freshly-added": True}
            result = runner.invoke(main, ["registry", "add", "https://github.com/test/repo"])
            assert result.exit_code == 0, result.output
            assert "Fetching registry contents" in result.output
            # update() should have been called for the freshly added registry
            assert mock_update.called
            call_args = mock_update.call_args
            assert call_args.args[0] == "freshly-added" or call_args.kwargs.get("name") == "freshly-added"

    def test_add_registry_no_update_flag_skips_fetch(self, runner, registry_setup):
        """--no-update skips the post-add fetch."""
        from sparkrun.core.registry import RegistryEntry

        added = [
            RegistryEntry(
                name="freshly-added",
                url="https://github.com/test/repo",
                subpath="recipes",
                enabled=True,
            )
        ]
        with (
            mock.patch(
                "sparkrun.core.registry.RegistryManager.add_registry_from_url",
                return_value=added,
            ),
            mock.patch("sparkrun.core.registry.RegistryManager.update") as mock_update,
        ):
            result = runner.invoke(main, ["registry", "add", "--no-update", "https://github.com/test/repo"])
            assert result.exit_code == 0, result.output
            assert "Fetching registry contents" not in result.output
            assert not mock_update.called

    def test_remove_registry(self, runner, registry_setup):
        """Test removing a registry."""
        result = runner.invoke(
            main,
            [
                "registry",
                "remove",
                "test-registry",
            ],
        )
        assert result.exit_code == 0
        assert "removed" in result.output.lower()

    def test_remove_nonexistent_registry(self, runner, registry_setup):
        """Test removing a nonexistent registry fails."""
        result = runner.invoke(
            main,
            [
                "registry",
                "remove",
                "nonexistent",
            ],
        )
        assert result.exit_code != 0
        assert "Error" in result.output


class TestLoadRecipeRetryAfterUpdate:
    """Test _load_recipe's optional registry-refresh retry on recipe-not-found."""

    def test_retry_after_update_refreshes_and_succeeds(self, registry_setup):
        """When retry_after_update=True, a missing recipe triggers update() and retries."""
        from sparkrun.cli._common import _load_recipe
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.recipe import RecipeError
        from sparkrun.core.registry import RegistryManager

        config = SparkrunConfig()

        # registry_setup returns (config_root, cache_root); the cached recipe
        # at test-registry/recipes/test-vllm.yaml already exists on disk.
        _config_root, cache_root = registry_setup
        recipe_path = Path(cache_root) / "test-registry" / "recipes" / "test-vllm.yaml"

        calls = {"find": 0, "update": 0}

        def _fake_find(*args, **kwargs):
            calls["find"] += 1
            if calls["find"] == 1:
                raise RecipeError("Recipe 'new-recipe' not found")
            return recipe_path

        def _fake_update(self, *a, **kw):
            calls["update"] += 1
            return {}

        with (
            mock.patch("sparkrun.core.recipe.find_recipe", side_effect=_fake_find),
            mock.patch.object(RegistryManager, "update", _fake_update),
        ):
            recipe, _path, _reg = _load_recipe(config, "new-recipe", resolve=False, retry_after_update=True)

        assert calls["find"] == 2
        assert calls["update"] == 1
        assert recipe is not None

    def test_retry_disabled_by_default(self, registry_setup):
        """Without retry_after_update, a missing recipe exits with an error."""
        from sparkrun.cli._common import _load_recipe
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.recipe import RecipeError
        from sparkrun.core.registry import RegistryManager

        config = SparkrunConfig()

        update_called = {"n": 0}

        def _fake_find(*args, **kwargs):
            raise RecipeError("Recipe 'new-recipe' not found")

        def _fake_update(self, *a, **kw):
            update_called["n"] += 1
            return {}

        with (
            mock.patch("sparkrun.core.recipe.find_recipe", side_effect=_fake_find),
            mock.patch.object(RegistryManager, "update", _fake_update),
            pytest.raises(SystemExit),
        ):
            _load_recipe(config, "new-recipe", resolve=False)

        assert update_called["n"] == 0

    def test_retry_skipped_for_path_like_names(self, registry_setup):
        """Path-like names don't trigger a registry update even with retry_after_update."""
        from sparkrun.cli._common import _load_recipe
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.recipe import RecipeError
        from sparkrun.core.registry import RegistryManager

        config = SparkrunConfig()

        update_called = {"n": 0}

        def _fake_find(*args, **kwargs):
            raise RecipeError("Recipe './missing.yaml' not found")

        def _fake_update(self, *a, **kw):
            update_called["n"] += 1
            return {}

        with (
            mock.patch("sparkrun.core.recipe.find_recipe", side_effect=_fake_find),
            mock.patch.object(RegistryManager, "update", _fake_update),
            pytest.raises(SystemExit),
        ):
            _load_recipe(config, "./missing.yaml", resolve=False, retry_after_update=True)

        assert update_called["n"] == 0


class TestRegistryRevertToDefault:
    """Test registry revert-to-default command."""

    def test_revert_to_defaults(self, runner, registry_setup):
        """Test reverting registries to defaults."""
        # Mock subprocess.run to prevent real git clones during manifest discovery
        # (reset_to_defaults resets _manifest_discovery_attempted and re-runs discovery)
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr="mocked")):
            result = runner.invoke(main, ["registry", "revert-to-defaults"])
        assert result.exit_code == 0
        assert "reset to defaults" in result.output.lower()

    def test_revert_to_defaults_hidden(self, runner):
        """Test that revert-to-defaults is hidden from registry help."""
        result = runner.invoke(main, ["registry", "--help"])
        assert result.exit_code == 0
        assert "revert-to-defaults" not in result.output
