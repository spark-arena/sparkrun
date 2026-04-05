"""Tests for sparkrun.config module."""

from __future__ import annotations

from pathlib import Path

import yaml

from sparkrun.core.config import SparkrunConfig, DEFAULT_CACHE_DIR, DEFAULT_HF_CACHE_DIR


def test_config_defaults_no_file(tmp_path: Path):
    """Test that config with no file uses sensible defaults.

    Verifies default behavior when no configuration file exists.
    """
    nonexistent = tmp_path / "nonexistent" / "config.yaml"
    config = SparkrunConfig(config_path=nonexistent)

    # Should use defaults when file doesn't exist
    assert config.cache_dir == DEFAULT_CACHE_DIR
    assert config.hf_cache_dir == DEFAULT_HF_CACHE_DIR
    assert config.default_hosts == []
    assert config.ssh_user is None
    assert config.ssh_key is None
    assert config.ssh_options == []


def test_config_loads_yaml(tmp_path: Path):
    """Test that config properly loads values from a YAML file.

    Verifies that configuration values are correctly read from a config file.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "cache_dir": "/custom/cache",
        "hf_cache_dir": "/custom/hf",
        "cluster": {
            "hosts": ["host1", "host2", "host3"],
        },
        "ssh": {
            "user": "testuser",
            "key": "~/.ssh/test_key",
            "options": ["-o StrictHostKeyChecking=no", "-o ConnectTimeout=10"],
        },
        "defaults": {
            "image_prefix": "custom/prefix",
            "transformers": "t5",
        },
        "recipe_paths": [
            "/custom/recipes",
        ],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)

    assert str(config.cache_dir) == "/custom/cache"
    assert str(config.hf_cache_dir) == "/custom/hf"
    assert config.default_hosts == ["host1", "host2", "host3"]
    assert config.ssh_user == "testuser"
    assert config.ssh_key is not None  # Will be expanded
    assert config.ssh_options == ["-o StrictHostKeyChecking=no", "-o ConnectTimeout=10"]
    assert config.default_image_prefix == "custom/prefix"
    assert config.default_transformers_tag == "t5"


def test_config_cache_dir(tmp_path: Path):
    """Verify cache_dir property returns the configured cache directory.

    Tests that the cache_dir configuration value is properly accessible.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {"cache_dir": "/test/cache"}

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.cache_dir == Path("/test/cache")


def test_config_hf_cache_dir(tmp_path: Path):
    """Verify hf_cache_dir property returns the HuggingFace cache directory.

    Tests that the HuggingFace cache directory configuration is accessible.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {"hf_cache_dir": "/test/hf/cache"}

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.hf_cache_dir == Path("/test/hf/cache")


def test_config_default_hosts(tmp_path: Path):
    """Verify default_hosts from cluster.hosts configuration.

    Tests that cluster host configuration is properly read.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "cluster": {
            "hosts": ["192.168.1.10", "192.168.1.11", "192.168.1.12"],
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.default_hosts == ["192.168.1.10", "192.168.1.11", "192.168.1.12"]


def test_config_ssh_settings(tmp_path: Path):
    """Verify SSH configuration settings (user, key, options).

    Tests that SSH-related configuration is properly accessible.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "ssh": {
            "user": "admin",
            "key": "~/.ssh/custom_key",
            "options": ["-v", "-o ConnectTimeout=30"],
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.ssh_user == "admin"
    assert config.ssh_key is not None  # Should be expanded from ~
    assert "/.ssh/custom_key" in config.ssh_key  # Tilde should be expanded
    assert config.ssh_options == ["-v", "-o ConnectTimeout=30"]


def test_config_get_dotted_path(tmp_path: Path):
    """Test the get() method with dot-separated key paths.

    Verifies that nested configuration values can be accessed using dot notation.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "level1": {
            "level2": {
                "level3": "deep_value",
            },
            "simple": "value",
        },
        "top": "top_value",
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)

    assert config.get("top") == "top_value"
    assert config.get("level1.simple") == "value"
    assert config.get("level1.level2.level3") == "deep_value"
    assert config.get("nonexistent") is None
    assert config.get("nonexistent", "default") == "default"
    assert config.get("level1.nonexistent") is None


def test_config_recipe_search_paths(tmp_path: Path):
    """Test get_recipe_search_paths with various configured paths.

    Verifies that recipe search paths are correctly resolved and ordered.
    """
    # Create directories
    cwd_recipes = tmp_path / "cwd" / "recipes"
    cwd_recipes.mkdir(parents=True)

    user_recipes = tmp_path / "user" / "recipes"
    user_recipes.mkdir(parents=True)

    extra_recipes = tmp_path / "extra" / "recipes"
    extra_recipes.mkdir(parents=True)

    config_file = tmp_path / "config.yaml"
    config_data = {
        "recipe_paths": [str(extra_recipes)],
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Temporarily change to the cwd directory
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path / "cwd")

        config = SparkrunConfig(config_path=config_file)
        paths = config.get_recipe_search_paths()

        # Should include cwd/recipes and extra path
        # (User config dir won't be included since we're using a custom config path)
        path_strs = [str(p) for p in paths]

        # At least cwd/recipes should be present
        assert any("cwd" in p and "recipes" in p for p in path_strs)

        # Extra path should be present
        assert str(extra_recipes) in path_strs

    finally:
        os.chdir(original_cwd)


def test_config_default_image_prefix(tmp_path: Path):
    """Test default_image_prefix property.

    Verifies that the default container image prefix is properly accessible.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "defaults": {
            "image_prefix": "myregistry/myprefix",
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.default_image_prefix == "myregistry/myprefix"


def test_config_default_transformers_tag(tmp_path: Path):
    """Test default_transformers_tag property.

    Verifies that the default transformers tag is properly accessible.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "defaults": {
            "transformers": "t6",
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.default_transformers_tag == "t6"


def test_config_ssh_key_expansion(tmp_path: Path):
    """Test that SSH key paths are properly expanded from tilde notation.

    Verifies that ~ in SSH key paths is expanded to the user's home directory.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "ssh": {
            "key": "~/.ssh/id_rsa",
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)

    # Should expand ~ to home directory
    assert config.ssh_key is not None
    assert not config.ssh_key.startswith("~")
    assert "/.ssh/id_rsa" in config.ssh_key


def test_config_empty_cluster_section(tmp_path: Path):
    """Test config with empty or missing cluster section.

    Verifies proper defaults when cluster configuration is absent.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "cache_dir": "/test",
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.default_hosts == []


def test_config_empty_ssh_section(tmp_path: Path):
    """Test config with empty or missing SSH section.

    Verifies proper defaults when SSH configuration is absent.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "cache_dir": "/test",
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.ssh_user is None
    assert config.ssh_key is None
    assert config.ssh_options == []


def test_config_empty_defaults_section(tmp_path: Path):
    """Test config with empty or missing defaults section.

    Verifies proper defaults when defaults configuration is absent.
    """
    config_file = tmp_path / "config.yaml"
    config_data = {
        "cache_dir": "/test",
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = SparkrunConfig(config_path=config_file)
    assert config.default_image_prefix == ""
    assert config.default_transformers_tag == "t4"  # Default value


def test_get_config_root_without_variables():
    """get_config_root without Variables returns DEFAULT_CONFIG_DIR."""
    from sparkrun.core.config import get_config_root, DEFAULT_CONFIG_DIR

    result = get_config_root()
    assert result == DEFAULT_CONFIG_DIR


def test_get_config_root_with_variables(v):
    """get_config_root with initialized Variables returns SAF stateful root."""
    from sparkrun.core.config import get_config_root

    result = get_config_root(v)
    # Should be a Path (either SAF root or default)
    assert isinstance(result, Path)


def test_get_config_root_none_variables():
    """get_config_root(None) returns DEFAULT_CONFIG_DIR."""
    from sparkrun.core.config import get_config_root, DEFAULT_CONFIG_DIR

    result = get_config_root(None)
    assert result == DEFAULT_CONFIG_DIR


# ---------------------------------------------------------------------------
# ssh_user override (setter) tests
# ---------------------------------------------------------------------------


class TestSshUserOverride:
    """Tests for the ssh_user property setter added to fix cluster SSH user propagation."""

    def test_setter_overrides_config_value(self, tmp_path: Path):
        """Setting ssh_user overrides the value from config YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"ssh": {"user": "yaml_user"}}))
        config = SparkrunConfig(config_path=config_file)

        assert config.ssh_user == "yaml_user"
        config.ssh_user = "override_user"
        assert config.ssh_user == "override_user"

    def test_setter_overrides_none_default(self, tmp_path: Path):
        """Setting ssh_user works when no user was configured in YAML."""
        config_file = tmp_path / "nonexistent.yaml"
        config = SparkrunConfig(config_path=config_file)

        assert config.ssh_user is None
        config.ssh_user = "cluster_user"
        assert config.ssh_user == "cluster_user"

    def test_setter_to_none_clears_override(self, tmp_path: Path):
        """Setting ssh_user to None clears a previous override."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"ssh": {"user": "yaml_user"}}))
        config = SparkrunConfig(config_path=config_file)

        config.ssh_user = "override_user"
        assert config.ssh_user == "override_user"

        # Setting to None should still return None (the override takes effect)
        config.ssh_user = None
        assert config.ssh_user is None

    def test_override_flows_to_build_ssh_kwargs(self, tmp_path: Path):
        """Verify the override propagates through build_ssh_kwargs."""
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"ssh": {"user": "yaml_user"}}))
        config = SparkrunConfig(config_path=config_file)

        # Before override
        kwargs = build_ssh_kwargs(config)
        assert kwargs["ssh_user"] == "yaml_user"

        # After override
        config.ssh_user = "cluster_user"
        kwargs = build_ssh_kwargs(config)
        assert kwargs["ssh_user"] == "cluster_user"

    def test_override_does_not_affect_other_ssh_fields(self, tmp_path: Path):
        """Setting ssh_user does not change ssh_key or ssh_options."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "ssh": {
                        "user": "yaml_user",
                        "key": "~/.ssh/test_key",
                        "options": ["-o StrictHostKeyChecking=no"],
                    },
                }
            )
        )
        config = SparkrunConfig(config_path=config_file)
        config.ssh_user = "cluster_user"

        assert config.ssh_user == "cluster_user"
        assert config.ssh_key is not None
        assert "test_key" in config.ssh_key
        assert config.ssh_options == ["-o StrictHostKeyChecking=no"]
