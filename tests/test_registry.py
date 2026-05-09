"""Tests for sparkrun.registry module."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml

from sparkrun.core.registry import (
    EXTERNAL_RESERVED_NAMES,
    FALLBACK_DEFAULT_REGISTRIES,
    RESERVED_NAME_PREFIXES,
    RegistryEntry,
    RegistryError,
    RegistryManager,
    _get_git_org,
    validate_registry_name,
)


@pytest.fixture
def reg_dirs(tmp_path: Path):
    """Create config and cache directories for RegistryManager."""
    config = tmp_path / "config"
    cache = tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    return config, cache


@pytest.fixture
def mgr(reg_dirs):
    """Create a RegistryManager with temp dirs.

    Manifest discovery is disabled to avoid real git clones during tests.
    """
    config, cache = reg_dirs
    m = RegistryManager(config, cache)
    m._manifest_discovery_attempted = True  # skip network calls in tests
    return m


@pytest.fixture
def sample_entry() -> RegistryEntry:
    """A sample registry entry for testing."""
    return RegistryEntry(
        name="test-registry",
        url="https://github.com/example/repo",
        subpath="recipes",
        description="Test recipes",
    )


@pytest.fixture
def populated_cache(reg_dirs, sample_entry) -> tuple[RegistryManager, Path]:
    """Create a RegistryManager with a fake cached registry containing recipe files."""
    config, cache = reg_dirs
    mgr = RegistryManager(config, cache)
    mgr._manifest_discovery_attempted = True  # skip network calls in tests

    # Save the sample registry to config
    mgr._save_registries([sample_entry])

    # Create fake cached repo with recipes
    recipe_dir = cache / sample_entry.name / sample_entry.subpath
    recipe_dir.mkdir(parents=True)
    # Also create a .git dir to simulate a cloned repo
    (cache / sample_entry.name / ".git").mkdir()

    # Create test recipe files
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
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "container": "scitrera/dgx-spark-sglang:latest",
    }
    with open(recipe_dir / "test-vllm.yaml", "w") as f:
        yaml.dump(recipe1, f)
    with open(recipe_dir / "test-sglang.yaml", "w") as f:
        yaml.dump(recipe2, f)

    return mgr, recipe_dir


class TestRegistryEntry:
    """Test RegistryEntry dataclass."""

    def test_default_values(self):
        """Test that RegistryEntry has sensible defaults."""
        entry = RegistryEntry(name="test", url="https://example.com", subpath="recipes")
        assert entry.description == ""
        assert entry.enabled is True

    def test_all_fields(self):
        """Test creating entry with all fields."""
        entry = RegistryEntry(
            name="test",
            url="https://example.com",
            subpath="recipes/sub",
            description="Test registry",
            enabled=False,
        )
        assert entry.name == "test"
        assert entry.url == "https://example.com"
        assert entry.subpath == "recipes/sub"
        assert entry.description == "Test registry"
        assert entry.enabled is False


class TestDefaultRegistries:
    """Test FALLBACK_DEFAULT_REGISTRIES."""

    def test_has_testing_registry(self):
        """Test that FALLBACK_DEFAULT_REGISTRIES includes the sparkrun-testing registry."""
        assert len(FALLBACK_DEFAULT_REGISTRIES) >= 1
        testing = FALLBACK_DEFAULT_REGISTRIES[0]
        assert testing.name == "sparkrun-testing"
        assert "github.com/dbotwinick/sparkrun-recipe-registry" in testing.url
        assert testing.enabled is True

    def test_has_official_registry(self):
        """Test that FALLBACK_DEFAULT_REGISTRIES includes the official registry."""
        assert len(FALLBACK_DEFAULT_REGISTRIES) >= 2
        official = FALLBACK_DEFAULT_REGISTRIES[1]
        assert official.name == "official"
        assert "github.com/spark-arena/recipe-registry" in official.url
        assert official.subpath == "official-recipes"
        assert official.enabled is True
        assert official.visible is True

    def test_has_transitional_registry(self):
        """Test that FALLBACK_DEFAULT_REGISTRIES includes the transitional registry."""
        assert len(FALLBACK_DEFAULT_REGISTRIES) >= 4
        transitional = FALLBACK_DEFAULT_REGISTRIES[3]
        assert transitional.name == "sparkrun-transitional"
        assert "github.com/dbotwinick/sparkrun-recipe-registry" in transitional.url
        assert transitional.subpath == "transitional/recipes"
        assert transitional.enabled is True
        assert transitional.visible is True

    def test_has_experimental_registry(self):
        """Test that FALLBACK_DEFAULT_REGISTRIES includes the experimental registry."""
        assert len(FALLBACK_DEFAULT_REGISTRIES) >= 5
        experimental = FALLBACK_DEFAULT_REGISTRIES[4]
        assert experimental.name == "experimental"
        assert "github.com/spark-arena/recipe-registry" in experimental.url
        assert experimental.subpath == "experimental-recipes"
        assert experimental.enabled is True
        assert experimental.visible is False

    def test_has_eugr_registry(self):
        """Test that FALLBACK_DEFAULT_REGISTRIES includes the eugr registry."""
        assert len(FALLBACK_DEFAULT_REGISTRIES) >= 3
        eugr = FALLBACK_DEFAULT_REGISTRIES[2]
        assert eugr.name == "eugr"
        assert "eugr/spark-vllm-docker" in eugr.url
        assert eugr.enabled is True
        assert eugr.visible is True

    def test_seven_default_registries(self):
        """Test that there are exactly seven default registries."""
        assert len(FALLBACK_DEFAULT_REGISTRIES) == 7

    def test_testing_registry_subpath(self):
        """Test the sparkrun-testing registry subpath."""
        testing = FALLBACK_DEFAULT_REGISTRIES[0]
        assert testing.subpath == "testing/recipes"


class TestRegistryManagerInit:
    """Test RegistryManager initialization."""

    def test_creates_directories(self, tmp_path: Path):
        """Test that init creates config and cache directories."""
        config = tmp_path / "new_config"
        cache = tmp_path / "new_cache"
        RegistryManager(config, cache)  # side effect: creates dirs
        assert config.exists()
        assert cache.exists()

    def test_default_cache_root(self, tmp_path: Path):
        """Test that cache_root defaults when not provided."""
        mgr = RegistryManager(tmp_path)
        assert mgr.cache_root == Path.home() / ".cache/sparkrun/registries"

    def test_registries_path(self, mgr):
        """Test the registries.yaml path property."""
        assert mgr._registries_path.name == "registries.yaml"


class TestRegistryCRUD:
    """Test registry add/remove/list/get operations."""

    def test_list_defaults_when_no_config(self, mgr):
        """Test that list_registries returns defaults when no config file exists."""
        registries = mgr.list_registries()
        assert len(registries) == len(FALLBACK_DEFAULT_REGISTRIES)
        assert registries[0].name == FALLBACK_DEFAULT_REGISTRIES[0].name

    def test_add_registry(self, mgr, sample_entry):
        """Test adding a new registry."""
        mgr.add_registry(sample_entry)
        registries = mgr.list_registries()
        names = [r.name for r in registries]
        assert sample_entry.name in names

    def test_add_duplicate_raises(self, mgr, sample_entry):
        """Test that adding a duplicate registry raises RegistryError."""
        mgr.add_registry(sample_entry)
        with pytest.raises(RegistryError, match="already exists"):
            mgr.add_registry(sample_entry)

    def test_remove_registry(self, mgr, sample_entry):
        """Test removing a registry."""
        mgr.add_registry(sample_entry)
        mgr.remove_registry(sample_entry.name)
        registries = mgr.list_registries()
        names = [r.name for r in registries]
        assert sample_entry.name not in names

    def test_remove_nonexistent_raises(self, mgr):
        """Test that removing a nonexistent registry raises RegistryError."""
        with pytest.raises(RegistryError, match="not found"):
            mgr.remove_registry("nonexistent")

    def test_get_registry(self, mgr, sample_entry):
        """Test getting a registry by name."""
        mgr.add_registry(sample_entry)
        retrieved = mgr.get_registry(sample_entry.name)
        assert retrieved.name == sample_entry.name
        assert retrieved.url == sample_entry.url

    def test_get_nonexistent_raises(self, mgr):
        """Test that getting a nonexistent registry raises RegistryError."""
        with pytest.raises(RegistryError, match="not found"):
            mgr.get_registry("nonexistent")


class TestRegistrySaveLoad:
    """Test registry persistence via YAML."""

    def test_save_and_load_roundtrip(self, mgr, sample_entry):
        """Test that registries survive save/load cycle."""
        entries = [sample_entry]
        mgr._save_registries(entries)
        loaded = mgr._load_registries()
        assert len(loaded) == 1
        assert loaded[0].name == sample_entry.name
        assert loaded[0].url == sample_entry.url
        assert loaded[0].subpath == sample_entry.subpath

    def test_save_creates_yaml_file(self, mgr, sample_entry):
        """Test that _save_registries creates the YAML file."""
        mgr._save_registries([sample_entry])
        assert mgr._registries_path.exists()

    def test_load_with_disabled_registry(self, mgr):
        """Test loading a registry with enabled=false."""
        entry = RegistryEntry(
            name="disabled",
            url="https://example.com",
            subpath="recipes",
            enabled=False,
        )
        mgr._save_registries([entry])
        loaded = mgr._load_registries()
        assert loaded[0].enabled is False


class TestRegistryCache:
    """Test cache directory management."""

    def test_cache_dir_path(self, mgr):
        """Test that _cache_dir returns correct path."""
        path = mgr._cache_dir("test-registry")
        assert path == mgr.cache_root / "test-registry"

    def test_recipe_dir_returns_none_when_not_cached(self, mgr, sample_entry):
        """Test that _recipe_dir returns None when cache doesn't exist."""
        assert mgr._recipe_dir(sample_entry) is None

    def test_recipe_dir_returns_path_when_cached(self, mgr, sample_entry):
        """Test that _recipe_dir returns path when cache exists."""
        # Create fake cache
        recipe_dir = mgr._cache_dir(sample_entry.name) / sample_entry.subpath
        recipe_dir.mkdir(parents=True)
        result = mgr._recipe_dir(sample_entry)
        assert result == recipe_dir


class TestRegistryUpdate:
    """Test registry update (git clone/pull) operations."""

    def test_update_calls_clone_for_new(self, mgr, sample_entry):
        """Test that update clones a new registry."""
        mgr._save_registries([sample_entry])
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            mgr.update(sample_entry.name)
            # Should call git clone
            calls = mock_run.call_args_list
            assert any("clone" in str(c) for c in calls)

    def test_update_calls_pull_for_existing(self, mgr, sample_entry):
        """Test that update pulls an existing registry."""
        mgr._save_registries([sample_entry])
        # Create fake .git dir to simulate existing clone
        cache_dir = mgr._cache_dir(sample_entry.name)
        (cache_dir / ".git").mkdir(parents=True)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            mgr.update(sample_entry.name)
            # Should call git pull
            calls = mock_run.call_args_list
            assert any("pull" in str(c) for c in calls)

    def test_update_all_registries(self, mgr, sample_entry):
        """Test that update() with no name updates all enabled registries."""
        second = RegistryEntry(
            name="second",
            url="https://example.com/2",
            subpath="recipes",
        )
        mgr._save_registries([sample_entry, second])
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            mgr.update()
            # Should have clone calls for both
            assert mock_run.call_count >= 2

    def test_update_skips_disabled(self, mgr):
        """Test that update skips disabled registries."""
        disabled = RegistryEntry(
            name="disabled",
            url="https://example.com",
            subpath="recipes",
            enabled=False,
        )
        mgr._save_registries([disabled])
        with mock.patch("subprocess.run") as mock_run:
            mgr.update()
            mock_run.assert_not_called()

    def test_clone_failure_is_logged_not_raised(self, mgr, sample_entry):
        """Test that git clone failure is logged but doesn't raise."""
        mgr._save_registries([sample_entry])
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="fatal: error")
            # Should not raise
            mgr.update(sample_entry.name)

    def test_clone_timeout_is_logged_not_raised(self, mgr, sample_entry):
        """Test that git timeout is logged but doesn't raise."""
        import subprocess as sp

        mgr._save_registries([sample_entry])
        with mock.patch("subprocess.run", side_effect=sp.TimeoutExpired("git", 60)):
            # Should not raise
            mgr._clone_or_pull(sample_entry)


class TestEnsureInitialized:
    """Test ensure_initialized auto-download behavior."""

    def test_calls_update_when_no_cache(self, mgr, sample_entry):
        """Test that ensure_initialized calls update when no cache exists."""
        mgr._save_registries([sample_entry])
        with mock.patch.object(mgr, "update") as mock_update:
            mgr.ensure_initialized()
            mock_update.assert_called_once()

    def test_skips_when_cache_exists(self, mgr, sample_entry):
        """Test that ensure_initialized skips when cache already exists."""
        mgr._save_registries([sample_entry])
        # Create fake .git dir
        (mgr._cache_dir(sample_entry.name) / ".git").mkdir(parents=True)
        with mock.patch.object(mgr, "update") as mock_update:
            mgr.ensure_initialized()
            mock_update.assert_not_called()


class TestRecipeDiscovery:
    """Test recipe path and search functionality."""

    def test_get_recipe_paths_empty(self, mgr):
        """Test get_recipe_paths with no cached registries."""
        paths = mgr.get_recipe_paths()
        assert paths == []

    def test_get_recipe_paths_with_cache(self, populated_cache):
        """Test get_recipe_paths returns cached recipe directories."""
        mgr, recipe_dir = populated_cache
        paths = mgr.get_recipe_paths()
        assert len(paths) == 1
        assert paths[0] == recipe_dir

    def test_get_recipe_paths_skips_disabled(self, reg_dirs):
        """Test that get_recipe_paths skips disabled registries."""
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        entry = RegistryEntry(
            name="disabled",
            url="https://example.com",
            subpath="recipes",
            enabled=False,
        )
        mgr._save_registries([entry])
        # Create the cache anyway
        recipe_dir = cache / "disabled" / "recipes"
        recipe_dir.mkdir(parents=True)
        paths = mgr.get_recipe_paths()
        assert paths == []

    def test_search_recipes_by_name(self, populated_cache):
        """Test searching recipes by name (filename stem)."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("vllm")
        assert len(results) >= 1
        assert any("vllm" in r["name"] for r in results)

    def test_search_recipes_by_model(self, populated_cache):
        """Test searching recipes by model name."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("llama")
        assert len(results) >= 2  # Both recipes use llama

    def test_search_recipes_by_file_stem(self, populated_cache):
        """Test searching recipes by file stem."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("test-vllm")
        assert len(results) >= 1

    def test_search_recipes_case_insensitive(self, populated_cache):
        """Test that recipe search is case-insensitive."""
        mgr, _ = populated_cache
        upper = mgr.search_recipes("VLLM")
        lower = mgr.search_recipes("vllm")
        assert len(upper) == len(lower)

    def test_search_recipes_no_results(self, populated_cache):
        """Test that search returns empty list for no matches."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("nonexistent-model-xyz")
        assert results == []

    def test_search_results_have_registry_field(self, populated_cache):
        """Test that search results include registry name."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("llama")
        for r in results:
            assert "registry" in r
            assert r["registry"] == "test-registry"

    def test_find_recipe_in_registries(self, populated_cache):
        """Test finding a recipe by file stem."""
        mgr, _ = populated_cache
        matches = mgr.find_recipe_in_registries("test-vllm")
        assert len(matches) == 1
        registry_name, path = matches[0]
        assert registry_name == "test-registry"
        assert path.name == "test-vllm.yaml"

    def test_find_recipe_not_found(self, populated_cache):
        """Test that find returns empty for nonexistent recipe."""
        mgr, _ = populated_cache
        matches = mgr.find_recipe_in_registries("nonexistent-recipe")
        assert matches == []

    def test_find_recipe_multiple_registries(self, reg_dirs):
        """Test finding a recipe that exists in multiple registries."""
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)

        entries = [
            RegistryEntry(name="reg1", url="https://example.com/1", subpath="recipes"),
            RegistryEntry(name="reg2", url="https://example.com/2", subpath="recipes"),
        ]
        mgr._save_registries(entries)

        # Create same recipe in both registries
        for entry in entries:
            recipe_dir = cache / entry.name / entry.subpath
            recipe_dir.mkdir(parents=True)
            (cache / entry.name / ".git").mkdir(exist_ok=True)
            with open(recipe_dir / "shared-recipe.yaml", "w") as f:
                yaml.dump({"name": "Shared Recipe", "model": "test"}, f)

        matches = mgr.find_recipe_in_registries("shared-recipe")
        assert len(matches) == 2
        registry_names = {m[0] for m in matches}
        assert registry_names == {"reg1", "reg2"}

    def test_find_recipe_in_subdirectory_fallback(self, reg_dirs):
        """Test finding a recipe in a nested subdirectory when no flat match exists."""
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)

        entry = RegistryEntry(
            name="nested-registry",
            url="https://example.com/nested",
            subpath="recipes",
        )
        mgr._save_registries([entry])

        # Create registry with recipes ONLY in a subdirectory (no flat recipes)
        recipe_dir = cache / entry.name / entry.subpath
        nested_dir = recipe_dir / "qwen3"
        nested_dir.mkdir(parents=True)
        (cache / entry.name / ".git").mkdir(exist_ok=True)

        # Create recipe in subdirectory
        nested_recipe = {
            "sparkrun_version": "2",
            "name": "Qwen3 vLLM Recipe",
            "description": "A nested test recipe",
            "model": "Qwen/Qwen3-1.7b",
            "runtime": "vllm",
            "container": "scitrera/dgx-spark-vllm:latest",
        }
        with open(nested_dir / "qwen3-1.7b-vllm.yaml", "w") as f:
            yaml.dump(nested_recipe, f)

        # Should find the recipe by stem even though it's in a subdirectory
        matches = mgr.find_recipe_in_registries("qwen3-1.7b-vllm")
        assert len(matches) == 1
        registry_name, path = matches[0]
        assert registry_name == "nested-registry"
        assert path.name == "qwen3-1.7b-vllm.yaml"
        assert "qwen3" in str(path)  # Verify it's in the subdirectory


class TestRegistryEntryNewFields:
    """Test new RegistryEntry fields."""

    def test_visible_default_true(self):
        entry = RegistryEntry(name="test", url="https://example.com", subpath="recipes")
        assert entry.visible is True

    def test_visible_can_be_false(self):
        entry = RegistryEntry(name="test", url="https://example.com", subpath="recipes", visible=False)
        assert entry.visible is False

    def test_tuning_subpath_default_empty(self):
        entry = RegistryEntry(name="test", url="https://example.com", subpath="recipes")
        assert entry.tuning_subpath == ""

    def test_benchmark_subpath_default_empty(self):
        entry = RegistryEntry(name="test", url="https://example.com", subpath="recipes")
        assert entry.benchmark_subpath == ""

    def test_all_new_fields(self):
        entry = RegistryEntry(
            name="test",
            url="https://example.com",
            subpath="recipes",
            visible=False,
            tuning_subpath="tuning",
            benchmark_subpath="benchmarks",
        )
        assert entry.visible is False
        assert entry.tuning_subpath == "tuning"
        assert entry.benchmark_subpath == "benchmarks"


class TestRegistrySaveLoadNewFields:
    """Test serialization of new fields."""

    def test_save_omits_default_visible(self, mgr):
        """visible=True should be omitted from YAML."""
        entry = RegistryEntry(name="test", url="https://example.com", subpath="r")
        mgr._save_registries([entry])
        import yaml

        data = yaml.safe_load(mgr._registries_path.read_text())
        assert "visible" not in data["registries"][0]

    def test_save_includes_visible_false(self, mgr):
        """visible=False should be saved."""
        entry = RegistryEntry(name="test", url="https://example.com", subpath="r", visible=False)
        mgr._save_registries([entry])
        import yaml

        data = yaml.safe_load(mgr._registries_path.read_text())
        assert data["registries"][0]["visible"] is False

    def test_save_omits_empty_tuning_subpath(self, mgr):
        entry = RegistryEntry(name="test", url="https://example.com", subpath="r")
        mgr._save_registries([entry])
        import yaml

        data = yaml.safe_load(mgr._registries_path.read_text())
        assert "tuning_subpath" not in data["registries"][0]

    def test_save_includes_tuning_subpath(self, mgr):
        entry = RegistryEntry(name="test", url="https://example.com", subpath="r", tuning_subpath="tuning")
        mgr._save_registries([entry])
        import yaml

        data = yaml.safe_load(mgr._registries_path.read_text())
        assert data["registries"][0]["tuning_subpath"] == "tuning"

    def test_roundtrip_new_fields(self, mgr):
        entry = RegistryEntry(
            name="test",
            url="https://example.com",
            subpath="r",
            visible=False,
            tuning_subpath="t",
            benchmark_subpath="b",
        )
        mgr._save_registries([entry])
        loaded = mgr._load_registries()
        assert loaded[0].visible is False
        assert loaded[0].tuning_subpath == "t"
        assert loaded[0].benchmark_subpath == "b"

    def test_load_missing_visible_defaults_true(self, mgr):
        """Old YAML without visible field should default to True."""
        import yaml

        data = {"registries": [{"name": "old", "url": "https://example.com", "subpath": "r"}]}
        mgr._registries_path.write_text(yaml.dump(data))
        loaded = mgr._load_registries()
        assert loaded[0].visible is True


class TestEnableDisableRegistry:
    """Test enable/disable methods."""

    def test_disable_registry(self, mgr, sample_entry):
        mgr.add_registry(sample_entry)
        mgr.disable_registry(sample_entry.name)
        reg = mgr.get_registry(sample_entry.name)
        assert reg.enabled is False

    def test_enable_registry(self, mgr, sample_entry):
        mgr.add_registry(sample_entry)
        mgr.disable_registry(sample_entry.name)
        mgr.enable_registry(sample_entry.name)
        reg = mgr.get_registry(sample_entry.name)
        assert reg.enabled is True

    def test_enable_nonexistent_raises(self, mgr):
        with pytest.raises(RegistryError, match="not found"):
            mgr.enable_registry("nonexistent")

    def test_disable_nonexistent_raises(self, mgr):
        with pytest.raises(RegistryError, match="not found"):
            mgr.disable_registry("nonexistent")


class TestVisibilityFiltering:
    """Test visibility filtering in recipe discovery."""

    def test_get_recipe_paths_excludes_hidden(self, reg_dirs):
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        hidden = RegistryEntry(
            name="hidden",
            url="https://example.com",
            subpath="recipes",
            visible=False,
        )
        visible = RegistryEntry(
            name="visible",
            url="https://example.com/2",
            subpath="recipes",
            visible=True,
        )
        mgr._save_registries([hidden, visible])

        # Create cache dirs for both
        for entry in [hidden, visible]:
            recipe_dir = cache / entry.name / entry.subpath
            recipe_dir.mkdir(parents=True)
            (cache / entry.name / ".git").mkdir(exist_ok=True)

        paths = mgr.get_recipe_paths(include_hidden=False)
        path_strs = [str(p) for p in paths]
        assert any("visible" in s for s in path_strs)
        assert not any("hidden" in s and "visible" not in s for s in path_strs)

    def test_get_recipe_paths_includes_hidden_when_requested(self, reg_dirs):
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        hidden = RegistryEntry(
            name="hidden-reg",
            url="https://example.com",
            subpath="recipes",
            visible=False,
        )
        mgr._save_registries([hidden])
        recipe_dir = cache / "hidden-reg" / "recipes"
        recipe_dir.mkdir(parents=True)
        (cache / "hidden-reg" / ".git").mkdir(exist_ok=True)

        paths = mgr.get_recipe_paths(include_hidden=True)
        assert len(paths) == 1

    def test_search_recipes_excludes_hidden(self, reg_dirs):
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        hidden = RegistryEntry(
            name="hidden-search",
            url="https://example.com",
            subpath="recipes",
            visible=False,
        )
        mgr._save_registries([hidden])

        recipe_dir = cache / "hidden-search" / "recipes"
        recipe_dir.mkdir(parents=True)
        (cache / "hidden-search" / ".git").mkdir(exist_ok=True)
        import yaml

        with open(recipe_dir / "my-recipe.yaml", "w") as f:
            yaml.dump({"name": "Hidden Recipe", "model": "test-model", "runtime": "vllm"}, f)

        results = mgr.search_recipes("test-model", include_hidden=False)
        assert len(results) == 0

    def test_search_recipes_includes_hidden_when_requested(self, reg_dirs):
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        hidden = RegistryEntry(
            name="hidden-search2",
            url="https://example.com",
            subpath="recipes",
            visible=False,
        )
        mgr._save_registries([hidden])

        recipe_dir = cache / "hidden-search2" / "recipes"
        recipe_dir.mkdir(parents=True)
        (cache / "hidden-search2" / ".git").mkdir(exist_ok=True)
        import yaml

        with open(recipe_dir / "my-recipe2.yaml", "w") as f:
            yaml.dump({"name": "Hidden Recipe 2", "model": "test-model-2", "runtime": "vllm"}, f)

        results = mgr.search_recipes("test-model-2", include_hidden=True)
        assert len(results) == 1

    def test_find_recipe_excludes_hidden(self, reg_dirs):
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        hidden = RegistryEntry(
            name="hidden-find",
            url="https://example.com",
            subpath="recipes",
            visible=False,
        )
        mgr._save_registries([hidden])

        recipe_dir = cache / "hidden-find" / "recipes"
        recipe_dir.mkdir(parents=True)
        (cache / "hidden-find" / ".git").mkdir(exist_ok=True)
        import yaml

        with open(recipe_dir / "find-me.yaml", "w") as f:
            yaml.dump({"name": "Find Me", "model": "test"}, f)

        matches = mgr.find_recipe_in_registries("find-me", include_hidden=False)
        assert len(matches) == 0

    def test_find_recipe_includes_hidden_when_requested(self, reg_dirs):
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        hidden = RegistryEntry(
            name="hidden-find2",
            url="https://example.com",
            subpath="recipes",
            visible=False,
        )
        mgr._save_registries([hidden])

        recipe_dir = cache / "hidden-find2" / "recipes"
        recipe_dir.mkdir(parents=True)
        (cache / "hidden-find2" / ".git").mkdir(exist_ok=True)
        import yaml

        with open(recipe_dir / "find-me2.yaml", "w") as f:
            yaml.dump({"name": "Find Me 2", "model": "test"}, f)

        matches = mgr.find_recipe_in_registries("find-me2", include_hidden=True)
        assert len(matches) == 1


class TestDefaultRegistriesNewFields:
    """Test new fields on FALLBACK_DEFAULT_REGISTRIES."""

    def test_testing_has_tuning_subpath(self):
        testing = FALLBACK_DEFAULT_REGISTRIES[0]
        assert testing.tuning_subpath == "testing/tuning"

    def test_testing_has_benchmark_subpath(self):
        testing = FALLBACK_DEFAULT_REGISTRIES[0]
        assert testing.benchmark_subpath == "testing/benchmarking"

    def test_official_is_visible(self):
        official = FALLBACK_DEFAULT_REGISTRIES[1]
        assert official.visible is True

    def test_official_name(self):
        official = FALLBACK_DEFAULT_REGISTRIES[1]
        assert official.name == "official"

    def test_transitional_is_visible(self):
        transitional = FALLBACK_DEFAULT_REGISTRIES[3]
        assert transitional.visible is True

    def test_experimental_is_hidden(self):
        experimental = FALLBACK_DEFAULT_REGISTRIES[4]
        assert experimental.visible is False


class TestSharedClone:
    """Test shared clone optimization."""

    def test_clone_dir_for_url_deterministic(self, mgr):
        url = "https://github.com/example/repo"
        d1 = mgr._clone_dir_for_url(url)
        d2 = mgr._clone_dir_for_url(url)
        assert d1 == d2

    def test_clone_dir_for_url_different_urls(self, mgr):
        d1 = mgr._clone_dir_for_url("https://github.com/a/b")
        d2 = mgr._clone_dir_for_url("https://github.com/c/d")
        assert d1 != d2

    def test_sparse_checkout_paths_for_url(self, mgr):
        entries = [
            RegistryEntry(name="r1", url="https://example.com", subpath="recipes", tuning_subpath="tuning"),
            RegistryEntry(name="r2", url="https://example.com", subpath="experimental", benchmark_subpath="bench"),
        ]
        mgr._save_registries(entries)
        paths = mgr._sparse_checkout_paths_for_url("https://example.com")
        assert ".sparkrun" in paths
        assert "recipes" in paths
        assert "experimental" in paths
        assert "tuning" in paths
        assert "bench" in paths

    def test_sparse_checkout_paths_ignores_other_urls(self, mgr):
        entries = [
            RegistryEntry(name="r1", url="https://example.com", subpath="recipes"),
            RegistryEntry(name="r2", url="https://other.com", subpath="other"),
        ]
        mgr._save_registries(entries)
        paths = mgr._sparse_checkout_paths_for_url("https://example.com")
        assert "recipes" in paths
        assert "other" not in paths

    def test_link_registry_to_shared(self, mgr, sample_entry):
        """Test that _link_registry_to_shared creates a symlink."""
        # Create the shared dir
        shared = mgr._clone_dir_for_url(sample_entry.url)
        shared.mkdir(parents=True)
        (shared / sample_entry.subpath).mkdir(parents=True)

        mgr._link_registry_to_shared(sample_entry)

        per_reg = mgr._cache_dir(sample_entry.name)
        assert per_reg.is_symlink()
        assert per_reg.resolve() == shared.resolve()


class TestManifestParsing:
    """Test add_registry_from_url manifest discovery."""

    def test_add_from_url_no_manifest_raises(self, mgr):
        """Test that missing manifest raises RegistryError."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            # The temp dir won't have a manifest
            with pytest.raises(RegistryError, match="No .sparkrun/registry.yaml"):
                mgr.add_registry_from_url("https://example.com/repo")

    def test_add_from_url_clone_failure_raises(self, mgr):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="fatal: error")
            with pytest.raises(RegistryError, match="Failed to clone"):
                mgr.add_registry_from_url("https://example.com/repo")


class TestDeprecatedRegistries:
    """Test deprecated registry cleanup."""

    def test_cleanup_no_deprecated(self, mgr, sample_entry):
        # Save only the sample entry (no defaults whose URL might match DEPRECATED_REGISTRIES)
        mgr._save_registries([sample_entry])
        cleaned = mgr.cleanup_deprecated()
        assert cleaned == []
        # Entry should still exist
        assert mgr.get_registry(sample_entry.name)

    def test_cleanup_with_deprecated(self, mgr):
        """Test cleanup matches on URL, not name."""
        entry = RegistryEntry(name="old-reg", url="https://example.com/old/repo", subpath="r")
        mgr.add_registry(entry)

        # Temporarily patch DEPRECATED_REGISTRIES with a URL
        from sparkrun.core import registry as reg_module

        original = reg_module.DEPRECATED_REGISTRIES
        try:
            reg_module.DEPRECATED_REGISTRIES = ["https://example.com/old/repo"]
            cleaned = mgr.cleanup_deprecated()
            assert "old-reg" in cleaned
            with pytest.raises(RegistryError):
                mgr.get_registry("old-reg")
        finally:
            reg_module.DEPRECATED_REGISTRIES = original

    def test_cleanup_matches_url_with_git_suffix(self, mgr):
        """Test that cleanup matches URLs regardless of .git suffix."""
        entry = RegistryEntry(name="dotgit-reg", url="https://example.com/org/repo.git", subpath="r")
        mgr.add_registry(entry)

        from sparkrun.core import registry as reg_module

        original = reg_module.DEPRECATED_REGISTRIES
        try:
            # Deprecated list has URL without .git, entry has .git
            reg_module.DEPRECATED_REGISTRIES = ["https://example.com/org/repo"]
            cleaned = mgr.cleanup_deprecated()
            assert "dotgit-reg" in cleaned
        finally:
            reg_module.DEPRECATED_REGISTRIES = original

    def test_cleanup_does_not_match_by_name(self, mgr):
        """Test that cleanup does NOT match by registry name."""
        entry = RegistryEntry(name="some-name", url="https://example.com/safe/repo", subpath="r")
        mgr.add_registry(entry)

        from sparkrun.core import registry as reg_module

        original = reg_module.DEPRECATED_REGISTRIES
        try:
            # Put the name in DEPRECATED_REGISTRIES — should NOT match
            reg_module.DEPRECATED_REGISTRIES = ["some-name"]
            cleaned = mgr.cleanup_deprecated()
            assert cleaned == []
            # Entry should still exist
            assert mgr.get_registry("some-name")
        finally:
            reg_module.DEPRECATED_REGISTRIES = original


class TestRestoreMissingDefaults:
    """Test restore_missing_defaults functionality."""

    def test_restore_no_missing(self, mgr):
        """When all fallback defaults are present, nothing is added."""
        mgr._save_registries(list(FALLBACK_DEFAULT_REGISTRIES))
        restored = mgr.restore_missing_defaults()
        assert restored == []
        assert len(mgr.list_registries()) == len(FALLBACK_DEFAULT_REGISTRIES)

    def test_restore_adds_missing_entry(self, mgr):
        """Missing fallback entries are appended to the config."""
        # Save only the first fallback entry — the rest should be restored
        mgr._save_registries([FALLBACK_DEFAULT_REGISTRIES[0]])
        restored = mgr.restore_missing_defaults()
        expected_missing = [e.name for e in FALLBACK_DEFAULT_REGISTRIES[1:]]
        assert sorted(restored) == sorted(expected_missing)
        # All fallback entries should now be present
        names = {e.name for e in mgr.list_registries()}
        for fb in FALLBACK_DEFAULT_REGISTRIES:
            assert fb.name in names

    def test_restore_preserves_existing_entries(self, mgr):
        """Existing non-default entries are preserved after restore."""
        custom = RegistryEntry(name="my-custom", url="https://example.com/repo", subpath="r")
        mgr._save_registries([custom])
        mgr.restore_missing_defaults()
        entries = mgr.list_registries()
        names = {e.name for e in entries}
        assert "my-custom" in names
        for fb in FALLBACK_DEFAULT_REGISTRIES:
            assert fb.name in names

    def test_restore_does_not_duplicate(self, mgr):
        """Calling restore twice does not create duplicate entries."""
        mgr._save_registries([FALLBACK_DEFAULT_REGISTRIES[0]])
        mgr.restore_missing_defaults()
        restored_again = mgr.restore_missing_defaults()
        assert restored_again == []
        names = [e.name for e in mgr.list_registries()]
        assert len(names) == len(set(names)), "Duplicate registry names found"

    def test_restore_empty_config(self, mgr):
        """All fallback defaults are added when config is empty."""
        mgr._save_registries([])
        restored = mgr.restore_missing_defaults()
        assert sorted(restored) == sorted(e.name for e in FALLBACK_DEFAULT_REGISTRIES)

    def test_restore_returns_empty_when_no_file(self, mgr):
        """When no registries.yaml exists, falls back to _load_registries which already has defaults."""
        # No file saved — _load_registries returns defaults, so all names are present
        restored = mgr.restore_missing_defaults()
        assert restored == []


class TestReservedNamePrefixes:
    """Test reserved registry name prefix enforcement."""

    def test_non_reserved_name_allowed_from_any_url(self):
        """Non-reserved names should pass regardless of URL."""
        validate_registry_name("my-custom-recipes", "https://github.com/random-user/repo")

    def test_reserved_prefix_allowed_org_scitrera(self):
        """Reserved prefix from allowed org (scitrera) should pass."""
        validate_registry_name("sparkrun-foo", "https://github.com/scitrera/some-repo")

    def test_reserved_prefix_allowed_org_eugr(self):
        """Reserved prefix from allowed org (eugr) should pass."""
        validate_registry_name("sparkrun-custom", "https://github.com/eugr/some-repo")

    def test_reserved_prefix_allowed_org_spark_arena(self):
        """Reserved prefix from allowed org (spark-arena) should pass."""
        validate_registry_name("arena-benchmarks", "https://github.com/spark-arena/bench")

    def test_reserved_prefix_disallowed_org_raises(self):
        """Reserved prefix from a non-allowed org should raise RegistryError."""
        with pytest.raises(RegistryError, match="reserved prefix"):
            validate_registry_name("sparkrun-custom", "https://github.com/random-user/repo")

    @pytest.mark.parametrize("prefix", RESERVED_NAME_PREFIXES)
    def test_each_reserved_prefix_blocked(self, prefix):
        """Each reserved prefix should be blocked for non-allowed orgs."""
        name = prefix + "-something"
        with pytest.raises(RegistryError, match="reserved prefix"):
            validate_registry_name(name, "https://github.com/random-user/repo")

    @pytest.mark.parametrize("prefix", RESERVED_NAME_PREFIXES)
    def test_each_reserved_prefix_allowed_for_scitrera(self, prefix):
        """Each reserved prefix should be allowed for scitrera org."""
        name = prefix + "-something"
        validate_registry_name(name, "https://github.com/scitrera/repo")

    def test_case_insensitive(self):
        """Name matching should be case-insensitive."""
        with pytest.raises(RegistryError, match="reserved prefix"):
            validate_registry_name("SparkRun-foo", "https://github.com/random-user/repo")

    def test_case_insensitive_mixed(self):
        """Mixed case names should still be caught."""
        with pytest.raises(RegistryError, match="reserved prefix"):
            validate_registry_name("ARENA-benchmarks", "https://github.com/random-user/repo")

    def test_non_github_url_with_reserved_prefix_rejected(self):
        """Non-GitHub URLs with reserved prefixes should be rejected."""
        with pytest.raises(RegistryError, match="reserved prefix"):
            validate_registry_name("sparkrun-evil", "https://gitlab.com/evil/repo")

    def test_non_github_url_with_non_reserved_name_allowed(self):
        """Non-GitHub URLs with non-reserved names should pass."""
        validate_registry_name("my-recipes", "https://gitlab.com/someone/repo")

    def test_github_url_with_git_suffix(self):
        """GitHub URLs with .git suffix should still extract the org correctly."""
        validate_registry_name("sparkrun-official", "https://github.com/scitrera/repo.git")

    def test_exact_prefix_match_not_substring(self):
        """Names that don't start with a prefix should not be blocked."""
        # "my-sparkrun" does not start with "sparkrun"
        validate_registry_name("my-sparkrun-recipes", "https://github.com/random-user/repo")

    def test_add_registry_enforces_reserved_names(self, mgr):
        """add_registry() should reject reserved names from non-allowed orgs."""
        entry = RegistryEntry(
            name="sparkrun-impersonator",
            url="https://github.com/malicious-user/repo",
            subpath="recipes",
        )
        with pytest.raises(RegistryError, match="reserved prefix"):
            mgr.add_registry(entry)

    def test_add_registry_allows_reserved_names_from_allowed_org(self, mgr):
        """add_registry() should allow reserved names from allowed orgs."""
        entry = RegistryEntry(
            name="sparkrun-contrib",
            url="https://github.com/scitrera/contrib-recipes",
            subpath="recipes",
        )
        mgr.add_registry(entry)
        assert mgr.get_registry("sparkrun-contrib").name == "sparkrun-contrib"


class TestGetGitOrg:
    """Tests for `_get_git_org` URL parsing."""

    def test_https_github_url(self):
        assert _get_git_org("https://github.com/scitrera/repo") == "scitrera"

    def test_https_github_url_with_git_suffix(self):
        assert _get_git_org("https://github.com/scitrera/repo.git") == "scitrera"

    def test_www_github_url(self):
        assert _get_git_org("https://www.github.com/spark-arena/repo") == "spark-arena"

    def test_uppercase_url_lowercased(self):
        """Hostname comparison and returned org should both be lowercased."""
        assert _get_git_org("HTTPS://GITHUB.COM/Avarok-Cybersecurity/atlas-recipes.git") == "avarok-cybersecurity"

    def test_mixed_case_org_lowercased(self):
        """Mixed-case org names should be normalized to lowercase."""
        assert _get_git_org("https://github.com/Avarok-Cybersecurity/atlas-recipes") == "avarok-cybersecurity"

    def test_non_github_url_returns_none(self):
        assert _get_git_org("https://gitlab.com/someone/repo") is None

    def test_invalid_url_returns_none(self):
        assert _get_git_org("not-a-url") is None

    def test_scp_style_url_returns_none(self):
        """scp-style git URLs (git@github.com:org/repo.git) lack a hostname under urlparse."""
        assert _get_git_org("git@github.com:scitrera/repo.git") is None

    def test_empty_url_returns_none(self):
        assert _get_git_org("") is None

    def test_github_url_with_trailing_slash(self):
        assert _get_git_org("https://github.com/scitrera/repo/") == "scitrera"


class TestExternalReservedNames:
    """Tests for `EXTERNAL_RESERVED_NAMES` exact-match validation."""

    def test_atlas_allowed_for_avarok(self):
        """The reserved 'atlas' name must be allowed for the Avarok-Cybersecurity org."""
        validate_registry_name("atlas", "https://github.com/Avarok-Cybersecurity/atlas-recipes.git")

    def test_atlas_allowed_case_insensitive_url(self):
        """URL case shouldn't matter — `_get_git_org` lowercases."""
        validate_registry_name("atlas", "https://github.com/avarok-cybersecurity/atlas-recipes")

    def test_atlas_allowed_case_insensitive_name(self):
        """Name case shouldn't matter."""
        validate_registry_name("ATLAS", "https://github.com/Avarok-Cybersecurity/atlas-recipes.git")

    def test_atlas_rejected_from_other_org(self):
        """The 'atlas' name from a non-allowed org should raise."""
        with pytest.raises(RegistryError, match="reserved"):
            validate_registry_name("atlas", "https://github.com/random-user/atlas-fork")

    def test_atlas_rejected_from_non_github(self):
        """The 'atlas' name from a non-GitHub URL should raise (org cannot be verified)."""
        with pytest.raises(RegistryError, match="reserved"):
            validate_registry_name("atlas", "https://gitlab.com/avarok-cybersecurity/atlas-recipes")

    def test_atlas_prefix_falls_through_to_prefix_check(self):
        """`atlas-foo` is not an exact match — should bypass EXTERNAL_RESERVED_NAMES.

        Since 'atlas' is not in RESERVED_NAME_PREFIXES, 'atlas-foo' from any org should pass.
        """
        validate_registry_name("atlas-foo", "https://github.com/random-user/repo")

    def test_atlas_default_fallback_entry_validates(self):
        """The hardcoded atlas FALLBACK entry must satisfy validate_registry_name.

        Regression: a case-mismatch between `_get_git_org`'s lowercase output and
        the EXTERNAL_RESERVED_NAMES values would silently drop the entry from
        manifest discovery (the RegistryError is caught and logged).
        """
        atlas = next((e for e in FALLBACK_DEFAULT_REGISTRIES if e.name == "atlas"), None)
        assert atlas is not None, "atlas missing from FALLBACK_DEFAULT_REGISTRIES"
        validate_registry_name(atlas.name, atlas.url)

    def test_external_reserved_org_values_lowercase(self):
        """Org values in EXTERNAL_RESERVED_NAMES must be lowercase to match `_get_git_org`."""
        for name, orgs in EXTERNAL_RESERVED_NAMES.items():
            for org in orgs:
                assert org == org.lower(), (
                    "EXTERNAL_RESERVED_NAMES[%r] contains non-lowercase org %r — "
                    "_get_git_org returns lowercase so this entry would never match." % (name, org)
                )


class TestReservedNamePrefixesIntegrity:
    """Regression tests for RESERVED_NAME_PREFIXES tuple integrity."""

    def test_sparkrun_is_separate_entry(self):
        """Guard against implicit string concatenation merging 'sparkrun' with the next entry."""
        assert "sparkrun" in RESERVED_NAME_PREFIXES

    def test_official_is_separate_entry(self):
        """Guard against implicit string concatenation merging 'official' with the previous entry."""
        assert "official" in RESERVED_NAME_PREFIXES

    def test_no_concatenated_entries(self):
        """No entry should contain a comma (sign of implicit string concatenation bug)."""
        for prefix in RESERVED_NAME_PREFIXES:
            assert "," not in prefix, "Found comma in prefix %r — likely implicit string concatenation" % prefix


class TestLoadRegistriesFiltersDeprecated:
    """Test that _load_registries filters deprecated entries from config."""

    def test_deprecated_entries_filtered_from_config(self, mgr):
        """Entries whose URL matches DEPRECATED_REGISTRIES should be filtered out."""
        entries = [
            RegistryEntry(name="good-reg", url="https://example.com/good/repo", subpath="r"),
            RegistryEntry(name="deprecated-reg", url="https://example.com/old/repo", subpath="r"),
        ]
        mgr._save_registries(entries)

        from sparkrun.core import registry as reg_module

        original = reg_module.DEPRECATED_REGISTRIES
        try:
            reg_module.DEPRECATED_REGISTRIES = ["https://example.com/old/repo"]
            loaded = mgr._load_registries()
            names = [e.name for e in loaded]
            assert "good-reg" in names
            assert "deprecated-reg" not in names
        finally:
            reg_module.DEPRECATED_REGISTRIES = original

    def test_non_deprecated_entries_preserved(self, mgr):
        """Entries not in DEPRECATED_REGISTRIES should be loaded normally."""
        entries = [
            RegistryEntry(name="safe-reg", url="https://example.com/safe/repo", subpath="r"),
        ]
        mgr._save_registries(entries)

        from sparkrun.core import registry as reg_module

        original = reg_module.DEPRECATED_REGISTRIES
        try:
            reg_module.DEPRECATED_REGISTRIES = ["https://example.com/other/repo"]
            loaded = mgr._load_registries()
            assert len(loaded) == 1
            assert loaded[0].name == "safe-reg"
        finally:
            reg_module.DEPRECATED_REGISTRIES = original


class TestDefaultRegistriesFallback:
    """Test _default_registries fallback behavior."""

    def test_falls_back_to_hardcoded_on_manifest_failure(self, mgr):
        """When manifest discovery fails, _default_registries returns hardcoded defaults."""
        with mock.patch.object(mgr, "_init_defaults_from_manifests", return_value=[]):
            result = mgr._default_registries()
        assert len(result) == len(FALLBACK_DEFAULT_REGISTRIES)
        assert result[0].name == FALLBACK_DEFAULT_REGISTRIES[0].name

    def test_returns_manifest_entries_plus_fallbacks_on_success(self, mgr):
        """When manifest discovery succeeds, _default_registries returns manifest entries plus non-conflicting fallbacks."""
        manifest_entries = [
            RegistryEntry(name="from-manifest", url="https://example.com/m", subpath="r"),
        ]
        mgr._manifest_discovery_attempted = False  # allow discovery for this test
        with mock.patch.object(mgr, "_init_defaults_from_manifests", return_value=manifest_entries):
            result = mgr._default_registries()
        # Manifest entry comes first, then all fallback entries (none conflict by name)
        assert result[0].name == "from-manifest"
        assert len(result) == 1 + len(FALLBACK_DEFAULT_REGISTRIES)
        fallback_names = {e.name for e in FALLBACK_DEFAULT_REGISTRIES}
        result_names = {e.name for e in result[1:]}
        assert result_names == fallback_names

    def test_manifest_entries_override_fallbacks_by_name(self, mgr):
        """When a manifest entry shares a name with a fallback, the manifest entry wins."""
        # Use a name that matches one of the FALLBACK entries
        fallback_name = FALLBACK_DEFAULT_REGISTRIES[0].name
        manifest_entries = [
            RegistryEntry(name=fallback_name, url="https://example.com/new", subpath="new-recipes"),
        ]
        mgr._manifest_discovery_attempted = False
        with mock.patch.object(mgr, "_init_defaults_from_manifests", return_value=manifest_entries):
            result = mgr._default_registries()
        # Should have manifest version of the conflicting entry, plus remaining fallbacks
        assert len(result) == len(FALLBACK_DEFAULT_REGISTRIES)
        first = result[0]
        assert first.name == fallback_name
        assert first.url == "https://example.com/new"  # manifest version, not fallback
        assert first.subpath == "new-recipes"

    def test_init_manifests_returns_empty_on_all_urls_fail(self, mgr):
        """_init_defaults_from_manifests returns [] when all URLs fail."""
        with mock.patch.object(mgr, "_discover_manifest_entries", side_effect=RegistryError("clone fail")):
            result = mgr._init_defaults_from_manifests()
        assert result == []


class TestDiscoverManifestEntries:
    """Test _discover_manifest_entries method."""

    def test_returns_entries_from_manifest_canonical_keys(self, mgr):
        """Successful clone with canonical keys (subpath, tuning_subpath) returns parsed entries."""
        manifest_yaml = yaml.dump(
            {
                "registries": [
                    {"name": "reg-a", "subpath": "recipes-a", "description": "Registry A"},
                    {"name": "reg-b", "subpath": "recipes-b", "tuning_subpath": "tuning"},
                ]
            }
        )

        def fake_run(cmd, **kwargs):
            if "clone" in cmd:
                # Create the manifest in the temp dir
                tmp_path = Path(cmd[-1])
                tmp_path.mkdir(parents=True, exist_ok=True)
                manifest_dir = tmp_path / ".sparkrun"
                manifest_dir.mkdir(parents=True)
                (manifest_dir / "registry.yaml").write_text(manifest_yaml)
                return mock.Mock(returncode=0, stderr="")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            entries = mgr._discover_manifest_entries("https://example.com/repo")

        assert len(entries) == 2
        assert entries[0].name == "reg-a"
        assert entries[0].url == "https://example.com/repo"
        assert entries[0].subpath == "recipes-a"
        assert entries[0].description == "Registry A"
        assert entries[1].name == "reg-b"
        assert entries[1].tuning_subpath == "tuning"

    def test_returns_entries_from_manifest_short_keys(self, mgr):
        """Manifest using short keys (recipes, tuning, benchmarks) is parsed correctly."""
        manifest_yaml = yaml.dump(
            {
                "registries": [
                    {
                        "name": "testing-reg",
                        "description": "Testing registry",
                        "recipes": "testing/recipes",
                        "tuning": "testing/tuning",
                        "benchmarks": "testing/benchmarking",
                        "visible": False,
                    },
                    {
                        "name": "transitional-reg",
                        "description": "Transitional registry",
                        "recipes": "transitional/recipes",
                        "visible": True,
                    },
                ]
            }
        )

        def fake_run(cmd, **kwargs):
            if "clone" in cmd:
                tmp_path = Path(cmd[-1])
                tmp_path.mkdir(parents=True, exist_ok=True)
                manifest_dir = tmp_path / ".sparkrun"
                manifest_dir.mkdir(parents=True)
                (manifest_dir / "registry.yaml").write_text(manifest_yaml)
                return mock.Mock(returncode=0, stderr="")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            entries = mgr._discover_manifest_entries("https://example.com/repo")

        assert len(entries) == 2
        # First entry: short keys mapped correctly
        assert entries[0].name == "testing-reg"
        assert entries[0].subpath == "testing/recipes"
        assert entries[0].tuning_subpath == "testing/tuning"
        assert entries[0].benchmark_subpath == "testing/benchmarking"
        assert entries[0].visible is False
        # Second entry: no tuning/benchmark
        assert entries[1].name == "transitional-reg"
        assert entries[1].subpath == "transitional/recipes"
        assert entries[1].tuning_subpath == ""
        assert entries[1].benchmark_subpath == ""

    def test_canonical_keys_take_precedence_over_short_keys(self, mgr):
        """When both canonical and short keys are present, canonical wins."""
        manifest_yaml = yaml.dump(
            {
                "registries": [
                    {
                        "name": "both-keys",
                        "subpath": "canonical-path",
                        "recipes": "short-path",
                        "tuning_subpath": "canonical-tuning",
                        "tuning": "short-tuning",
                    }
                ]
            }
        )

        def fake_run(cmd, **kwargs):
            if "clone" in cmd:
                tmp_path = Path(cmd[-1])
                tmp_path.mkdir(parents=True, exist_ok=True)
                manifest_dir = tmp_path / ".sparkrun"
                manifest_dir.mkdir(parents=True)
                (manifest_dir / "registry.yaml").write_text(manifest_yaml)
                return mock.Mock(returncode=0, stderr="")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            entries = mgr._discover_manifest_entries("https://example.com/repo")

        assert entries[0].subpath == "canonical-path"
        assert entries[0].tuning_subpath == "canonical-tuning"

    def test_clone_failure_raises(self, mgr):
        """Failed clone raises RegistryError."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="fatal: error")
            with pytest.raises(RegistryError, match="Failed to clone"):
                mgr._discover_manifest_entries("https://example.com/repo")

    def test_no_manifest_raises(self, mgr):
        """Missing manifest file raises RegistryError."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            with pytest.raises(RegistryError, match="No .sparkrun/registry.yaml"):
                mgr._discover_manifest_entries("https://example.com/repo")

    def test_empty_manifest_raises(self, mgr):
        """Manifest with no registries raises RegistryError."""
        manifest_yaml = yaml.dump({"registries": []})

        def fake_run(cmd, **kwargs):
            if "clone" in cmd:
                tmp_path = Path(cmd[-1])
                tmp_path.mkdir(parents=True, exist_ok=True)
                manifest_dir = tmp_path / ".sparkrun"
                manifest_dir.mkdir(parents=True)
                (manifest_dir / "registry.yaml").write_text(manifest_yaml)
                return mock.Mock(returncode=0, stderr="")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RegistryError, match="declares no registries"):
                mgr._discover_manifest_entries("https://example.com/repo")


class TestInitDefaultsFromManifests:
    """Test _init_defaults_from_manifests bulk-save flow."""

    def test_returns_entries_without_saving_or_calling_add_registry(self, mgr):
        """Entries are returned without saving or calling add_registry (no re-entrancy)."""
        entries_url1 = [
            RegistryEntry(name="m1", url="https://example.com/r1", subpath="recipes"),
        ]
        entries_url2 = [
            RegistryEntry(name="m2", url="https://example.com/r2", subpath="recipes"),
        ]

        call_count = [0]

        def fake_discover(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return entries_url1
            return entries_url2

        with mock.patch.object(mgr, "_discover_manifest_entries", side_effect=fake_discover):
            with mock.patch.object(mgr, "add_registry") as mock_add:
                result = mgr._init_defaults_from_manifests()

        # add_registry should NOT be called (that's the whole point of the fix)
        mock_add.assert_not_called()
        assert len(result) == 2
        assert result[0].name == "m1"
        assert result[1].name == "m2"
        # _init_defaults_from_manifests does NOT save; _default_registries handles that
        assert not mgr._registries_path.exists()

    def test_partial_failure_still_returns_successful_entries(self, mgr):
        """If one URL fails, entries from the other URL are still returned."""
        good_entries = [
            RegistryEntry(name="good-reg", url="https://example.com/good", subpath="recipes"),
        ]

        def fake_discover(url):
            if "bad" in url:
                raise RegistryError("clone failed")
            return good_entries

        from sparkrun.core import registry as reg_module

        original = reg_module.DEFAULT_REGISTRIES_GIT
        try:
            reg_module.DEFAULT_REGISTRIES_GIT = [
                "https://example.com/bad-repo",
                "https://example.com/good-repo",
            ]
            with mock.patch.object(mgr, "_discover_manifest_entries", side_effect=fake_discover):
                result = mgr._init_defaults_from_manifests()
        finally:
            reg_module.DEFAULT_REGISTRIES_GIT = original

        assert len(result) == 1
        assert result[0].name == "good-reg"

    def test_deduplicates_by_name(self, mgr):
        """Duplicate names across URLs are deduplicated (first wins)."""
        entries_url1 = [
            RegistryEntry(name="shared-name", url="https://example.com/r1", subpath="from-r1"),
        ]
        entries_url2 = [
            RegistryEntry(name="shared-name", url="https://example.com/r2", subpath="from-r2"),
            RegistryEntry(name="unique", url="https://example.com/r2", subpath="recipes"),
        ]

        call_count = [0]

        def fake_discover(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return entries_url1
            return entries_url2

        with mock.patch.object(mgr, "_discover_manifest_entries", side_effect=fake_discover):
            result = mgr._init_defaults_from_manifests()

        names = [e.name for e in result]
        assert names == ["shared-name", "unique"]
        # First one wins — subpath should be from url1
        assert result[0].subpath == "from-r1"

    def test_all_urls_fail_returns_empty(self, mgr):
        """When every URL fails, returns [] for fallback."""
        with mock.patch.object(mgr, "_discover_manifest_entries", side_effect=RegistryError("fail")):
            result = mgr._init_defaults_from_manifests()
        assert result == []
        # No file should have been saved
        assert not mgr._registries_path.exists()

    def test_no_re_entrancy_on_first_run(self, reg_dirs):
        """Full integration: first-run path does not re-enter _load_registries via add_registry."""
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        # Do NOT set _manifest_discovery_attempted — simulate real first run

        manifest_entries = [
            RegistryEntry(name="from-manifest", url="https://example.com/m", subpath="r"),
        ]
        with mock.patch.object(mgr, "_discover_manifest_entries", return_value=manifest_entries):
            # This calls _load_registries → _default_registries → _init_defaults_from_manifests
            registries = mgr.list_registries()

        # Manifest entry first, then non-conflicting fallbacks
        assert registries[0].name == "from-manifest"
        assert len(registries) == 1 + len(FALLBACK_DEFAULT_REGISTRIES)
        # Verify the file was saved (persisted by _default_registries)
        assert mgr._registries_path.exists()
        # Verify saved file matches what was returned
        loaded = mgr._load_registries_from_file()
        assert len(loaded) == len(registries)
        assert loaded[0].name == "from-manifest"


class TestResetToDefaults:
    """Test reset_to_defaults method.

    reset_to_defaults() clears _manifest_discovery_attempted and re-runs
    _default_registries(), which would attempt real git clones via
    _init_defaults_from_manifests().  Mock that method to avoid network calls.
    """

    def test_deletes_config_and_returns_defaults(self, mgr, sample_entry):
        """reset_to_defaults removes registries.yaml and returns fresh defaults."""
        mgr._save_registries([sample_entry])
        assert mgr._registries_path.exists()

        with mock.patch.object(mgr, "_init_defaults_from_manifests", return_value=[]):
            entries = mgr.reset_to_defaults()
        # Config file should be recreated with defaults
        assert mgr._registries_path.exists()
        names = [e.name for e in entries]
        assert sample_entry.name not in names
        assert len(entries) > 0

    def test_works_when_no_config_exists(self, mgr):
        """reset_to_defaults works even if registries.yaml doesn't exist."""
        assert not mgr._registries_path.exists()
        with mock.patch.object(mgr, "_init_defaults_from_manifests", return_value=[]):
            entries = mgr.reset_to_defaults()
        assert len(entries) > 0
        assert mgr._registries_path.exists()

    def test_saves_defaults_to_file(self, mgr, sample_entry):
        """After reset, the saved file contains the default entries."""
        mgr._save_registries([sample_entry])
        with mock.patch.object(mgr, "_init_defaults_from_manifests", return_value=[]):
            entries = mgr.reset_to_defaults()
        # Load from file directly to verify persistence
        loaded = mgr._load_registries_from_file()
        assert len(loaded) == len(entries)
        assert loaded[0].name == entries[0].name

    def test_clears_cache_on_reset(self, mgr):
        """reset_to_defaults removes cached registry clones."""
        # Create fake cache directories and a symlink
        (mgr.cache_root / "fake-registry" / ".git").mkdir(parents=True)
        (mgr.cache_root / "_url_abc123def456").mkdir(parents=True)
        shared = mgr.cache_root / "_url_abc123def456"
        link = mgr.cache_root / "linked-registry"
        link.symlink_to(shared)

        assert (mgr.cache_root / "fake-registry").exists()
        assert link.is_symlink()

        with mock.patch.object(mgr, "_init_defaults_from_manifests", return_value=[]):
            mgr.reset_to_defaults()

        # All cache entries should be gone
        remaining = [p.name for p in mgr.cache_root.iterdir()]
        assert "fake-registry" not in remaining
        assert "_url_abc123def456" not in remaining
        assert "linked-registry" not in remaining


class TestClearCache:
    """Test clear_cache method."""

    def test_clears_directories_and_symlinks(self, mgr):
        """clear_cache removes all dirs and symlinks from cache_root."""
        (mgr.cache_root / "reg1").mkdir()
        (mgr.cache_root / "reg2").mkdir()
        shared = mgr.cache_root / "_url_shared"
        shared.mkdir()
        link = mgr.cache_root / "reg3"
        link.symlink_to(shared)

        count = mgr.clear_cache()
        assert count == 4  # reg1, reg2, _url_shared, reg3
        assert list(mgr.cache_root.iterdir()) == []

    def test_returns_zero_on_empty_cache(self, mgr):
        """clear_cache returns 0 when cache is already empty."""
        assert mgr.clear_cache() == 0
