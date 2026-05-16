"""Tests for sparkrun.recipe module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sparkrun.core.recipe import (
    Recipe,
    RecipeError,
    find_recipe,
    list_recipes,
    recipe_summary,
    resolve_runtime,
    resolve_builder,
    is_recipe_file,
    discover_cwd_recipes,
)


def test_load_v2_recipe(tmp_recipe_dir: Path):
    """Load a v2 YAML recipe file and verify all fields are correctly parsed.

    Tests that a v2 recipe file is loaded with all expected fields:
    name, model, runtime, container, mode, defaults, env, and command.
    """
    recipe_path = tmp_recipe_dir / "test-vllm.yaml"
    recipe = Recipe.load(recipe_path)

    assert recipe.name == "test-vllm"  # name is always the filename stem
    assert recipe.description == "A test recipe for vLLM"
    assert recipe.model == "meta-llama/Llama-2-7b-hf"
    assert recipe.runtime == "vllm-distributed"
    assert recipe.mode == "auto"
    assert recipe.container == "scitrera/dgx-spark-vllm:latest"
    assert recipe.recipe_version == "2"

    # Verify defaults
    assert recipe.defaults["port"] == 8000
    assert recipe.defaults["host"] == "0.0.0.0"
    assert recipe.defaults["tensor_parallel"] == 1
    assert recipe.defaults["gpu_memory_utilization"] == 0.9

    # Verify env
    assert recipe.env["VLLM_BATCH_INVARIANT"] == "1"

    # Verify command
    assert recipe.command == "vllm serve {model} --port {port} --host {host}"


def test_load_v1_recipe_migrates_to_eugr(tmp_recipe_dir: Path):
    """Load a v1 recipe with mods/build_args and verify it auto-sets eugr builder.

    Tests the v1->v2 migration path for eugr-style recipes that require
    custom build arguments and patches.
    """
    recipe_path = tmp_recipe_dir / "test-eugr.yaml"
    recipe = Recipe.load(recipe_path)

    assert recipe.name == "test-eugr"  # name is always the filename stem
    assert recipe.model == "meta-llama/Llama-2-7b-hf"
    assert recipe.recipe_version == "1"

    # Should auto-set eugr builder and resolve to vllm-distributed
    assert recipe.runtime == "vllm-distributed"
    assert recipe.builder == "eugr"

    # build_args stays in runtime_config; mods is now a top-level field
    # (auto-migrated from v1 runtime_config to recipe.mods so the generic
    # core/mods.py resolver can handle any builder)
    assert recipe.runtime_config["build_args"] == ["ARG1=value1"]
    assert recipe.mods == ["mod1.patch"]
    assert "mods" not in recipe.runtime_config


def test_load_v1_recipe_no_mods_still_eugr(tmp_recipe_dir: Path):
    """Load a v1 recipe without mods and verify eugr builder is auto-set.

    The v1 format is the eugr native format, so all v1 vllm recipes
    should auto-set eugr builder regardless of whether build_args or
    mods are present.
    """
    recipe_path = tmp_recipe_dir / "test-plain-v1.yaml"
    recipe = Recipe.load(recipe_path)

    assert recipe.name == "test-plain-v1"  # name is always the filename stem
    assert recipe.model == "meta-llama/Llama-2-7b-hf"
    assert recipe.recipe_version == "1"

    # v1 format auto-sets eugr builder with standard vllm variant resolution
    assert recipe.runtime == "vllm-distributed"
    assert recipe.builder == "eugr"

    # No runtime_config should be set for these keys
    assert recipe.runtime_config.get("build_args", []) == []
    assert recipe.runtime_config.get("mods", []) == []


def test_recipe_from_dict(sample_v2_recipe_data: dict[str, Any]):
    """Create a recipe from a dict and verify all fields are correctly set.

    Tests the Recipe.from_dict() factory method.
    """
    recipe = Recipe.from_dict(sample_v2_recipe_data)

    assert recipe.name == "unnamed"  # from_dict has no source_path, so name defaults
    assert recipe.model == "meta-llama/Llama-2-7b-hf"
    assert recipe.runtime == "vllm-distributed"
    assert recipe.mode == "auto"
    assert recipe.min_nodes == 1
    assert recipe.max_nodes == 4
    assert recipe.container == "scitrera/dgx-spark-vllm:0.16.0"
    assert recipe.defaults["port"] == 8000
    assert recipe.env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_recipe_model_revision():
    """model_revision is parsed from recipe data."""
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "org/model",
            "model_revision": "abc123def",
        }
    )
    assert recipe.model_revision == "abc123def"


def test_recipe_model_revision_default_none():
    """model_revision defaults to None when not specified."""
    recipe = Recipe.from_dict({"name": "Test", "model": "org/model"})
    assert recipe.model_revision is None


def test_recipe_model_revision_not_in_runtime_config():
    """model_revision should not leak into runtime_config."""
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "org/model",
            "model_revision": "v2.1",
        }
    )
    assert "model_revision" not in recipe.runtime_config


def test_recipe_name_defaults_to_unnamed():
    """Recipe without name and no source_path should default to 'unnamed'."""
    recipe = Recipe.from_dict({"model": "test-model"})
    assert recipe.name == "unnamed"


def test_recipe_name_defaults_to_filename(tmp_path):
    """Recipe without name should derive name from source filename."""
    recipe_file = tmp_path / "my-cool-recipe.yaml"
    recipe_file.write_text("model: test-model\nruntime: vllm\n")
    recipe = Recipe.load(recipe_file)
    assert recipe.name == "my-cool-recipe"


def test_recipe_name_explicit_overrides_filename(tmp_path):
    """Recipe name is always the filename stem, ignoring the YAML name field."""
    recipe_file = tmp_path / "some-file.yaml"
    recipe_file.write_text("name: My Custom Name\nmodel: test-model\nruntime: vllm\n")
    recipe = Recipe.load(recipe_file)
    assert recipe.name == "some-file"


def test_qualified_name_bare():
    """qualified_name returns bare name when no source info."""
    recipe = Recipe.from_dict({"model": "test-model"})
    assert recipe.qualified_name == "unnamed"


def test_qualified_name_with_registry():
    """qualified_name returns @registry/name when source_registry is set."""
    recipe = Recipe.from_dict({"model": "test-model"})
    recipe.source_registry = "my-registry"
    assert recipe.qualified_name == "@my-registry/unnamed"


def test_qualified_name_with_path(tmp_path):
    """qualified_name returns filesystem path for path-loaded recipes."""
    recipe_file = tmp_path / "my-recipe.yaml"
    recipe_file.write_text("model: test-model\nruntime: vllm\n")
    recipe = Recipe.load(recipe_file)
    assert recipe.qualified_name == str(recipe_file)


def test_qualified_name_registry_takes_priority(tmp_path):
    """source_registry takes priority over source_path."""
    recipe_file = tmp_path / "my-recipe.yaml"
    recipe_file.write_text("model: test-model\nruntime: vllm\n")
    recipe = Recipe.load(recipe_file)
    recipe.source_registry = "official"
    assert recipe.qualified_name == "@official/my-recipe"


def test_qualified_name_url():
    """qualified_name returns URL for URL-sourced recipes."""
    recipe = Recipe.from_dict({"model": "test-model"})
    recipe.source_path = "https://example.com/recipe.yaml"
    assert recipe.qualified_name == "https://example.com/recipe.yaml"


def test_recipe_summary_qualified_name(tmp_path):
    """recipe_summary produces @registry/name when registry_name is given."""
    recipe_file = tmp_path / "my-recipe.yaml"
    recipe_file.write_text("model: test-model\nruntime: vllm\n")
    summary = recipe_summary(recipe_file, registry_name="spark-arena")
    assert summary["name"] == "@spark-arena/my-recipe"
    assert summary["file"] == "my-recipe"


def test_recipe_summary_bare_name(tmp_path):
    """recipe_summary produces bare name when no registry_name."""
    recipe_file = tmp_path / "my-recipe.yaml"
    recipe_file.write_text("model: test-model\nruntime: vllm\n")
    summary = recipe_summary(recipe_file)
    assert summary["name"] == "my-recipe"


def test_recipe_slug(tmp_path):
    """Test slug generation from recipe names (filename stems).

    Recipe.name is always the filename stem, so slugs derive from filenames.
    """
    f1 = tmp_path / "My Test Recipe.yaml"
    f1.write_text("model: test\nruntime: vllm\n")
    recipe1 = Recipe.load(f1)
    assert recipe1.slug == "my-test-recipe"

    f2 = tmp_path / "Recipe!!!With@Special#Chars.yaml"
    f2.write_text("model: test\nruntime: vllm\n")
    recipe2 = Recipe.load(f2)
    assert recipe2.slug == "recipe-with-special-chars"

    f3 = tmp_path / "CamelCaseRecipe.yaml"
    f3.write_text("model: test\nruntime: vllm\n")
    recipe3 = Recipe.load(f3)
    assert recipe3.slug == "camelcaserecipe"

    # from_dict with no source_path defaults to "unnamed"
    recipe4 = Recipe.from_dict({"model": "test"})
    assert recipe4.slug == "unnamed"


def test_recipe_validate_valid(sample_v2_recipe_data: dict[str, Any]):
    """Validate a valid recipe and verify it returns an empty error list.

    Tests that a well-formed recipe passes validation without issues.
    """
    recipe = Recipe.from_dict(sample_v2_recipe_data)
    issues = recipe.validate()
    assert issues == []


def test_recipe_validate_missing_name():
    """Validate a recipe missing model/runtime and verify errors are returned.

    Recipe.name is always populated from the filename stem (or 'unnamed'
    for from_dict), so we test validation of other required fields instead.
    """
    recipe = Recipe.from_dict({"name": "ignored"})
    issues = recipe.validate()
    # Should flag missing model (runtime may be resolved by resolvers)
    assert any("model" in i for i in issues)
    # Should not find name issue
    # assert any("name" in issue.lower() for issue in issues)


def test_recipe_validate_missing_model():
    """Validate a recipe with missing model field and verify error is returned.

    Tests validation error detection for missing required field.
    """
    recipe = Recipe.from_dict({"name": "Test", "runtime": "vllm"})
    issues = recipe.validate()
    assert len(issues) > 0
    assert any("model" in issue.lower() for issue in issues)


def test_recipe_validate_invalid_mode():
    """Validate a recipe with invalid mode and verify error is generated.

    Tests that invalid mode values are caught during validation.
    """
    recipe = Recipe.from_dict({"name": "Test", "model": "test-model", "runtime": "vllm", "mode": "invalid_mode"})
    issues = recipe.validate()
    assert len(issues) > 0
    assert any("mode" in issue.lower() for issue in issues)


def test_recipe_validate_min_max_nodes():
    """Validate that max_nodes < min_nodes generates an error.

    Tests validation of node count constraints.
    """
    recipe = Recipe.from_dict({"name": "Test", "model": "test-model", "runtime": "vllm", "min_nodes": 4, "max_nodes": 2})
    issues = recipe.validate()
    assert len(issues) > 0
    assert any("max_nodes" in issue and "min_nodes" in issue for issue in issues)


def test_recipe_build_config_chain(sample_v2_recipe_data: dict[str, Any]):
    """Build a config chain with CLI overrides and verify override precedence.

    Tests that CLI overrides take precedence over recipe defaults in the
    configuration chain.
    """
    recipe = Recipe.from_dict(sample_v2_recipe_data)

    # CLI overrides should take precedence
    cli_overrides = {"port": 9000, "tensor_parallel": 4}
    config = recipe.build_config_chain(cli_overrides)

    # Override values should be used
    assert config.get("port") == 9000
    assert config.get("tensor_parallel") == 4

    # Defaults should still be present for non-overridden values
    assert config.get("host") == "0.0.0.0"
    assert config.get("gpu_memory_utilization") == 0.9

    # Model should be injected
    assert config.get("model") == "meta-llama/Llama-2-7b-hf"


def test_recipe_render_command(sample_v2_recipe_data: dict[str, Any]):
    """Render a recipe command template and verify placeholders are substituted.

    Tests that command templates correctly substitute values from the config chain.
    """
    recipe = Recipe.from_dict(sample_v2_recipe_data)
    config = recipe.build_config_chain({"port": 9000})

    rendered = recipe.render_command(config)

    assert rendered is not None
    assert "{model}" not in rendered
    assert "{port}" not in rendered
    assert "{tensor_parallel}" not in rendered
    assert "meta-llama/Llama-2-7b-hf" in rendered
    assert "9000" in rendered  # Overridden port


def test_recipe_render_command_no_template():
    """Test that a recipe without a command template returns None.

    Tests behavior when no command template is defined in the recipe.
    """
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "test-model",
            "runtime": "vllm",
        }
    )
    config = recipe.build_config_chain()
    rendered = recipe.render_command(config)

    assert rendered is None


def test_render_command_fixes_trailing_space_continuations():
    """Trailing spaces after backslash line-continuations are stripped.

    In bash ``\\<space><newline>`` is an escaped space, not a continuation.
    YAML editors often introduce these accidentally; render_command should
    silently clean them up.
    """
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "test-model",
            "runtime": "vllm",
            "defaults": {"port": 8000, "host": "0.0.0.0"},
            "command": (
                "vllm serve {model} \\\n"
                "    --host {host} \\ \n"  # trailing space after backslash
                "    --port {port} \\  \n"  # two trailing spaces
                "    --trust-remote-code"
            ),
        }
    )
    config = recipe.build_config_chain()
    rendered = recipe.render_command(config)

    assert rendered is not None
    # Every backslash-newline should be a clean continuation (no trailing spaces)
    assert "\\ \n" not in rendered
    # The args should all be present (nothing dropped by broken continuation)
    assert "--host 0.0.0.0" in rendered
    assert "--port 8000" in rendered
    assert "--trust-remote-code" in rendered


def test_render_command_preserves_escaped_spaces_mid_line():
    """Backslash-space in the middle of a line is NOT a continuation — preserve it."""
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "test-model",
            "runtime": "vllm",
            "command": "echo hello\\ world",
        }
    )
    config = recipe.build_config_chain()
    rendered = recipe.render_command(config)

    assert rendered is not None
    assert "hello\\ world" in rendered


def test_find_recipe_direct_path(tmp_recipe_dir: Path):
    """Find a recipe by direct file path and verify it's located correctly.

    Tests that find_recipe can locate recipes by direct file path.
    """
    recipe_path = tmp_recipe_dir / "test-vllm.yaml"
    found = find_recipe(str(recipe_path))

    assert found == recipe_path
    assert found.exists()


def test_find_recipe_by_name(tmp_recipe_dir: Path):
    """Find a recipe by name in search paths and verify it's located.

    Tests recipe discovery by name across configured search paths.
    """
    found = find_recipe("test-vllm", search_paths=[tmp_recipe_dir])

    assert found.name == "test-vllm.yaml"
    assert found.exists()


def test_find_recipe_not_found(tmp_recipe_dir: Path):
    """Verify that find_recipe raises RecipeError when recipe is not found.

    Tests error handling for non-existent recipes.
    """
    with pytest.raises(RecipeError) as exc_info:
        find_recipe("nonexistent-recipe", search_paths=[tmp_recipe_dir])

    assert "not found" in str(exc_info.value).lower()


def test_list_recipes(tmp_recipe_dir: Path):
    """List recipes from a directory with multiple YAML files.

    Tests recipe listing functionality across a directory of recipe files.
    """
    recipes = list_recipes(search_paths=[tmp_recipe_dir])

    # Should find all recipes in the directory
    assert len(recipes) >= 4  # At least the 4 we created

    recipe_names = {r["file"] for r in recipes}
    assert "test-vllm" in recipe_names
    assert "test-sglang" in recipe_names
    assert "test-eugr" in recipe_names
    assert "test-plain-v1" in recipe_names

    # Verify recipe metadata
    vllm_recipe = next(r for r in recipes if r["file"] == "test-vllm")
    assert vllm_recipe["name"] == "test-vllm"  # name is always the filename stem
    assert vllm_recipe["runtime"] == "vllm-distributed"
    assert "path" in vllm_recipe


def test_recipe_get_default():
    """Test Recipe.get_default() method for retrieving default values.

    Verifies that default values can be retrieved with optional fallback.
    """
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "test-model",
            "defaults": {
                "port": 8000,
                "host": "0.0.0.0",
            },
        }
    )

    assert recipe.get_default("port") == 8000
    assert recipe.get_default("host") == "0.0.0.0"
    assert recipe.get_default("nonexistent") is None
    assert recipe.get_default("nonexistent", "fallback") == "fallback"


def test_recipe_load_nonexistent_file():
    """Verify that Recipe.load() raises RecipeError for non-existent files.

    Tests error handling when attempting to load a recipe file that doesn't exist.
    """
    with pytest.raises(RecipeError) as exc_info:
        Recipe.load("/nonexistent/path/recipe.yaml")

    assert "not found" in str(exc_info.value).lower()


def test_recipe_v1_cluster_only_migration(tmp_path: Path):
    """Test that v1 cluster_only flag correctly sets min_nodes and mode.

    Verifies v1->v2 migration of cluster topology constraints.
    """
    recipe_data = {
        "recipe_version": "1",
        "name": "Cluster Only Test",
        "model": "test-model",
        "cluster_only": True,
    }

    recipe_file = tmp_path / "cluster-only.yaml"
    with open(recipe_file, "w") as f:
        yaml.dump(recipe_data, f)

    recipe = Recipe.load(recipe_file)

    assert recipe.min_nodes == 2
    assert recipe.mode == "cluster"


def test_recipe_v1_solo_only_migration(tmp_path: Path):
    """Test that v1 solo_only flag correctly sets max_nodes and mode.

    Verifies v1->v2 migration of solo-only topology constraints.
    """
    recipe_data = {
        "recipe_version": "1",
        "name": "Solo Only Test",
        "model": "test-model",
        "solo_only": True,
    }

    recipe_file = tmp_path / "solo-only.yaml"
    with open(recipe_file, "w") as f:
        yaml.dump(recipe_data, f)

    recipe = Recipe.load(recipe_file)

    assert recipe.max_nodes == 1
    assert recipe.mode == "solo"


def test_recipe_runtime_config_catchall():
    """Test that unknown top-level keys are swept into runtime_config."""
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "test-model",
            "runtime": "vllm",
            "build_args": ["--pre-tf"],
            "mods": ["mods/fix-something"],
            "custom_field": "custom_value",
        }
    )
    assert recipe.runtime_config["build_args"] == ["--pre-tf"]
    # mods is a known top-level field, not swept into runtime_config
    assert recipe.mods == ["mods/fix-something"]
    assert "mods" not in recipe.runtime_config
    assert recipe.runtime_config["custom_field"] == "custom_value"
    # Known keys should NOT be in runtime_config
    assert "name" not in recipe.runtime_config
    assert "model" not in recipe.runtime_config


def test_recipe_runtime_config_explicit_key():
    """Test that explicit runtime_config YAML key is loaded."""
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "test-model",
            "runtime_config": {"build_args": ["arg1"], "custom": "val"},
        }
    )
    assert recipe.runtime_config["build_args"] == ["arg1"]
    assert recipe.runtime_config["custom"] == "val"


def test_recipe_runtime_config_explicit_over_sweep():
    """Test that explicit runtime_config takes precedence over auto-swept keys."""
    recipe = Recipe.from_dict(
        {
            "name": "Test",
            "model": "test-model",
            "runtime_config": {"custom_field": "from_config"},
            "custom_field": "from_toplevel",
        }
    )
    assert recipe.runtime_config["custom_field"] == "from_config"


def test_recipe_solo_only_v2():
    """Test solo_only as a v2 field (not just v1 migration)."""
    recipe = Recipe.from_dict(
        {
            "sparkrun_version": "2",
            "name": "Test",
            "model": "test-model",
            "solo_only": True,
        }
    )
    assert recipe.max_nodes == 1
    assert recipe.mode == "solo"


def test_recipe_cluster_only_v2():
    """Test cluster_only as a v2 field (not just v1 migration)."""
    recipe = Recipe.from_dict(
        {
            "sparkrun_version": "2",
            "name": "Test",
            "model": "test-model",
            "cluster_only": True,
        }
    )
    assert recipe.min_nodes == 2
    assert recipe.mode == "cluster"


class TestRecipeMetadata:
    """Test recipe metadata section parsing and validation."""

    def test_metadata_parsing(self):
        """Test that metadata section is parsed correctly."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "model_dtype": "float16",
                    "model_params": "7B",
                    "kv_dtype": "bfloat16",
                    "num_layers": 32,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                },
            }
        )
        assert recipe.metadata["model_dtype"] == "float16"
        assert recipe.metadata["model_params"] == "7B"
        assert recipe.metadata["kv_dtype"] == "bfloat16"
        assert recipe.metadata["num_layers"] == 32
        assert recipe.metadata["num_kv_heads"] == 8
        assert recipe.metadata["head_dim"] == 128

    def test_no_metadata(self):
        """Test that recipes without metadata have empty metadata dict."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.metadata == {}

    def test_metadata_not_in_runtime_config(self):
        """Test that metadata does NOT leak into runtime_config."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"model_dtype": "float16"},
            }
        )
        assert "metadata" not in recipe.runtime_config

    def test_validate_bad_metadata_dtype(self):
        """Test validation catches invalid metadata dtype."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"model_dtype": "invalid_dtype_xyz"},
            }
        )
        issues = recipe.validate()
        assert any("model_dtype" in i for i in issues)

    def test_validate_bad_metadata_params(self):
        """Test validation catches invalid model_params."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"model_params": "not_a_number"},
            }
        )
        issues = recipe.validate()
        assert any("model_params" in i for i in issues)

    def test_validate_bad_kv_dtype(self):
        """Test validation catches invalid kv_dtype."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"kv_dtype": "bogus_type"},
            }
        )
        issues = recipe.validate()
        assert any("kv_dtype" in i for i in issues)

    def test_validate_good_metadata(self):
        """Test that valid metadata passes validation without issues."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "model_dtype": "float16",
                    "model_params": "7B",
                    "kv_dtype": "bfloat16",
                },
            }
        )
        issues = recipe.validate()
        assert issues == []

    def test_metadata_with_vram_overrides(self):
        """Test that model_vram and kv_vram_per_token are stored in metadata."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "model_vram": 5.2,
                    "kv_vram_per_token": 0.00004,
                },
            }
        )
        assert recipe.metadata["model_vram"] == 5.2
        assert recipe.metadata["kv_vram_per_token"] == 0.00004

    def test_spark_arena_benchmarks_roundtrip(self):
        """Test that spark_arena_benchmarks is preserved in recipe metadata."""
        benchmarks = [
            {"tp": 1, "uuid": "076136cd-260a-4e77-b6e2-309d8f64619b"},
            {"tp": 2, "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
        ]
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "spark_arena_benchmarks": benchmarks,
                },
            }
        )
        assert recipe.spark_arena_benchmarks == benchmarks

    def test_spark_arena_benchmarks_absent(self):
        """Test that spark_arena_benchmarks returns empty list when not provided."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.spark_arena_benchmarks == []

    def test_estimate_vram_with_metadata(self):
        """Test estimate_vram() with full metadata (no HF detection)."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "model_dtype": "float16",
                    "model_params": "7B",
                    "kv_dtype": "bfloat16",
                    "num_layers": 32,
                    "num_kv_heads": 32,
                    "head_dim": 128,
                },
                "defaults": {
                    "max_model_len": 4096,
                    "tensor_parallel": 1,
                },
            }
        )
        est = recipe.estimate_vram(auto_detect=False)
        assert est.model_weights_gb > 0
        assert est.kv_cache_total_gb is not None
        assert est.kv_cache_total_gb > 0
        assert est.total_per_gpu_gb > 0
        assert est.tensor_parallel == 1

    def test_estimate_vram_with_overrides(self):
        """Test estimate_vram() with model_vram and kv_vram_per_token overrides."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "model_vram": 5.0,
                    "kv_vram_per_token": 0.0001,
                },
                "defaults": {
                    "max_model_len": 10000,
                    "tensor_parallel": 1,
                },
            }
        )
        est = recipe.estimate_vram(auto_detect=False)
        assert est.model_weights_gb == 5.0
        assert est.kv_cache_total_gb is not None
        assert abs(est.kv_cache_total_gb - 1.0) < 0.001
        assert abs(est.total_per_gpu_gb - 6.0) < 0.01

    def test_estimate_vram_cli_override_tp(self):
        """Test that CLI tensor_parallel override is respected."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"model_vram": 10.0},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(cli_overrides={"tensor_parallel": 2}, auto_detect=False)
        assert est.tensor_parallel == 2
        assert abs(est.total_per_gpu_gb - 5.0) < 0.01

    def test_estimate_vram_cli_override_pp(self):
        """Test that CLI pipeline_parallel override is respected."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"model_vram": 12.0},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(cli_overrides={"pipeline_parallel": 2}, auto_detect=False)
        assert est.pipeline_parallel == 2
        assert abs(est.total_per_gpu_gb - 6.0) < 0.01

    def test_estimate_vram_tp_and_pp(self):
        """Test VRAM estimation with both TP and PP from defaults."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"model_vram": 12.0},
                "defaults": {"tensor_parallel": 2, "pipeline_parallel": 3},
            }
        )
        est = recipe.estimate_vram(auto_detect=False)
        assert est.tensor_parallel == 2
        assert est.pipeline_parallel == 3
        # 12 / (2 * 3) = 2.0
        assert abs(est.total_per_gpu_gb - 2.0) < 0.01

    def test_estimate_vram_cli_override_max_model_len(self):
        """Test that CLI max_model_len override is respected."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"kv_vram_per_token": 0.001},
                "defaults": {"max_model_len": 1000},
            }
        )
        est1 = recipe.estimate_vram(auto_detect=False)
        est2 = recipe.estimate_vram(cli_overrides={"max_model_len": 2000}, auto_detect=False)
        assert est2.kv_cache_total_gb > est1.kv_cache_total_gb

    def test_estimate_vram_kv_cache_dtype_from_defaults(self):
        """Test that kv_cache_dtype from defaults is used for estimation."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "model_params": "7B",
                    "model_dtype": "float16",
                    "num_layers": 32,
                    "num_kv_heads": 32,
                    "head_dim": 128,
                },
                "defaults": {
                    "max_model_len": 4096,
                    "kv_cache_dtype": "fp8",
                },
            }
        )
        est = recipe.estimate_vram(auto_detect=False)
        assert est.kv_dtype == "fp8"

    def test_estimate_vram_gpu_memory_utilization(self):
        """Test that gpu_memory_utilization from defaults flows through."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {
                    "model_vram": 10.0,
                    "num_layers": 32,
                    "num_kv_heads": 32,
                    "head_dim": 128,
                },
                "defaults": {
                    "max_model_len": 4096,
                    "gpu_memory_utilization": 0.9,
                },
            }
        )
        est = recipe.estimate_vram(auto_detect=False)
        assert est.gpu_memory_utilization == 0.9
        assert est.usable_gpu_memory_gb is not None
        assert est.available_kv_gb is not None
        assert est.max_context_tokens is not None
        assert est.context_multiplier is not None

    def test_estimate_vram_gpu_mem_cli_override(self):
        """Test that CLI gpu_memory_utilization override works."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "metadata": {"model_vram": 10.0},
                "defaults": {"gpu_memory_utilization": 0.9},
            }
        )
        est = recipe.estimate_vram(
            cli_overrides={"gpu_memory_utilization": 0.5},
            auto_detect=False,
        )
        assert est.gpu_memory_utilization == 0.5

    @pytest.fixture()
    def _mock_hf_config(self, monkeypatch):
        """Mock fetch_model_config to return a nested multimodal config."""

        def _fake_fetch(model_id, revision=None, cache_dir=None):
            return {
                "architectures": ["SomeVLModel"],
                "text_config": {
                    "dtype": "bfloat16",
                    "num_hidden_layers": 64,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 24,
                    "head_dim": 128,
                },
            }

        monkeypatch.setattr("sparkrun.models.vram.fetch_model_config", _fake_fetch)

    @pytest.fixture()
    def _mock_safetensors_params_none(self, monkeypatch):
        """Mock fetch_safetensors_params to return None so size fallback is tested."""
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_params",
            lambda model_id, revision=None: None,
        )

    @pytest.fixture()
    def _mock_safetensors(self, monkeypatch):
        """Mock fetch_safetensors_size to return a known total_size."""
        # 35B params * 2 bytes (bfloat16) = 70_000_000_000 bytes
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_size",
            lambda model_id, revision=None, cache_dir=None: 70_000_000_000,
        )

    @pytest.mark.usefixtures("_mock_hf_config", "_mock_safetensors")
    def test_estimate_vram_safetensors_fallback(self):
        """model_vram set directly from safetensors total bytes for non-runtime-quant."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-35b",
                "metadata": {},
                "defaults": {
                    "max_model_len": 4096,
                    "tensor_parallel": 1,
                },
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        # 70_000_000_000 bytes / 1024^3 ≈ 65.19 GiB used directly as model_vram
        assert est.model_weights_gb > 60
        assert est.model_dtype == "bfloat16"
        assert est.kv_cache_total_gb is not None
        assert est.kv_cache_total_gb > 0

    @pytest.mark.usefixtures("_mock_hf_config")
    def test_estimate_vram_safetensors_not_called_when_params_provided(self, monkeypatch):
        """Safetensors fallback should NOT fire when metadata provides model_params."""
        called = []
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_params",
            lambda model_id, revision=None: called.append("params") or 99_999_999_999,
        )
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_size",
            lambda model_id, revision=None, cache_dir=None: called.append("size") or 99_999_999_999,
        )
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-35b",
                "metadata": {"model_params": "7B"},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert called == []  # safetensors never queried
        assert est.model_params == 7_000_000_000

    @pytest.mark.usefixtures("_mock_hf_config")
    def test_estimate_vram_safetensors_not_called_when_model_vram(self, monkeypatch):
        """Safetensors fallback should NOT fire when model_vram override is set."""
        called = []
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_params",
            lambda model_id, revision=None: called.append("params") or 99_999_999_999,
        )
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_size",
            lambda model_id, revision=None, cache_dir=None: called.append("size") or 99_999_999_999,
        )
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-35b",
                "metadata": {"model_vram": 50.0},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert called == []
        assert est.model_weights_gb == 50.0

    @pytest.mark.usefixtures("_mock_hf_config")
    def test_estimate_vram_safetensors_returns_none(self, monkeypatch):
        """When both safetensors API and index unavailable, model_params stays None."""
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_params",
            lambda model_id, revision=None: None,
        )
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_size",
            lambda model_id, revision=None, cache_dir=None: None,
        )
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-35b",
                "metadata": {},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert est.model_params is None
        assert est.model_weights_gb == 0.0

    def test_estimate_vram_quant_dtype_from_hf_config(self, monkeypatch):
        """HF quantization_config.quant_method should override torch_dtype for model weights."""

        def _fake_fetch(model_id, revision=None, cache_dir=None):
            return {
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 64,
                "num_key_value_heads": 4,
                "num_attention_heads": 24,
                "head_dim": 128,
                "quantization_config": {"quant_method": "fp8"},
            }

        monkeypatch.setattr("sparkrun.models.vram.fetch_model_config", _fake_fetch)
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_size",
            lambda model_id, revision=None, cache_dir=None: 35_000_000_000,
        )

        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-35b-fp8",
                "metadata": {"model_params": "35B"},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert est.model_dtype == "fp8"
        # 35B * 1 byte (fp8) / 1024^3 ≈ 32.6 GiB
        assert est.model_weights_gb < 35
        assert est.model_weights_gb > 30

    @pytest.mark.usefixtures("_mock_hf_config")
    def test_estimate_vram_safetensors_size_sets_model_vram(self, monkeypatch):
        """fetch_safetensors_size total bytes used directly as model_vram for non-runtime-quant."""
        params_called = []
        # 18 GB model on disk
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_size",
            lambda model_id, revision=None, cache_dir=None: 18_000_000_000,
        )
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_params",
            lambda model_id, revision=None: params_called.append(1) or 9_400_000_000,
        )
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-9b",
                "metadata": {},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert params_called == []  # fetch_safetensors_params not called when size succeeds
        # 18_000_000_000 / 1024^3 ≈ 16.76 GiB used directly as model_vram
        assert est.model_weights_gb > 16
        assert est.model_weights_gb < 18

    @pytest.mark.usefixtures("_mock_hf_config")
    def test_estimate_vram_params_fallback_when_size_unavailable(self, monkeypatch):
        """fetch_safetensors_params used as fallback when size returns None."""
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_size",
            lambda model_id, revision=None, cache_dir=None: None,
        )
        monkeypatch.setattr(
            "sparkrun.models.vram.fetch_safetensors_params",
            lambda model_id, revision=None: 9_400_000_000,
        )
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-9b",
                "metadata": {},
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert est.model_params == 9_400_000_000
        assert est.model_weights_gb > 0

    def test_estimate_vram_recipe_quantization_default_overrides_hf(self, monkeypatch):
        """Recipe defaults quantization key should take priority over HF torch_dtype."""

        def _fake_fetch(model_id, revision=None, cache_dir=None):
            return {
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "head_dim": 128,
            }

        monkeypatch.setattr("sparkrun.models.vram.fetch_model_config", _fake_fetch)

        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-7b",
                "metadata": {"model_params": "7B"},
                "defaults": {
                    "tensor_parallel": 1,
                    "quantization": "fp8",
                },
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert est.model_dtype == "fp8"

    def test_estimate_vram_metadata_dtype_takes_precedence_over_quant(self, monkeypatch):
        """Explicit metadata.model_dtype should win over any quant detection."""

        def _fake_fetch(model_id, revision=None, cache_dir=None):
            return {
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "head_dim": 128,
                "quantization_config": {"quant_method": "fp8"},
            }

        monkeypatch.setattr("sparkrun.models.vram.fetch_model_config", _fake_fetch)

        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "org/model-7b",
                "metadata": {
                    "model_params": "7B",
                    "model_dtype": "bfloat16",
                },
                "defaults": {"tensor_parallel": 1},
            }
        )
        est = recipe.estimate_vram(auto_detect=True)
        # metadata.model_dtype is explicit — should not be overridden
        assert est.model_dtype == "bfloat16"

    def test_estimate_vram_quantization_none_falls_through(self, monkeypatch):
        """quantization: none/auto in defaults should not override HF dtype."""

        def _fake_fetch(model_id, revision=None, cache_dir=None):
            return {
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "head_dim": 128,
            }

        monkeypatch.setattr("sparkrun.models.vram.fetch_model_config", _fake_fetch)

        for quant_val in ("none", "auto", "None", ""):
            recipe = Recipe.from_dict(
                {
                    "name": "Test",
                    "model": "org/model-7b",
                    "metadata": {"model_params": "7B"},
                    "defaults": {
                        "tensor_parallel": 1,
                        "quantization": quant_val,
                    },
                }
            )
            est = recipe.estimate_vram(auto_detect=True)
            assert est.model_dtype == "bfloat16", f"quantization={quant_val!r} should not override dtype"


class TestResolverChain:
    """Test the resolver chain and resolve_runtime() function."""

    def test_resolve_v1_sets_eugr(self):
        """v1 recipe auto-sets eugr builder and resolves to vllm-distributed."""
        recipe = Recipe.from_dict(
            {
                "recipe_version": "1",
                "name": "Test",
                "model": "test-model",
            }
        )
        assert recipe.runtime == "vllm-distributed"
        assert recipe.builder == "eugr"

    def test_resolve_eugr_signals_build_args(self):
        """build_args triggers eugr builder, runtime resolves to vllm-distributed."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "build_args": ["--pre-tf"],
            }
        )
        assert recipe.runtime == "vllm-distributed"
        assert recipe.builder == "eugr"

    def test_resolve_eugr_signals_mods(self):
        """mods triggers eugr builder, runtime resolves to vllm-distributed."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "mods": ["mods/fix.patch"],
            }
        )
        assert recipe.runtime == "vllm-distributed"
        assert recipe.builder == "eugr"

    def test_resolve_vllm_defaults_to_distributed(self):
        """Bare vllm -> vllm-distributed."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
            }
        )
        assert recipe.runtime == "vllm-distributed"

    def test_resolve_vllm_ray_hint_in_defaults(self):
        """distributed_executor_backend: ray -> vllm-ray."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "defaults": {"distributed_executor_backend": "ray"},
            }
        )
        assert recipe.runtime == "vllm-ray"

    def test_resolve_vllm_ray_hint_in_command(self):
        """Command template with ray -> vllm-ray."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "command": "vllm serve {model} --distributed-executor-backend ray",
            }
        )
        assert recipe.runtime == "vllm-ray"

    @pytest.mark.parametrize("runtime", ["sglang", "vllm-ray", "vllm-distributed", "llama-cpp"])
    def test_resolve_explicit_runtime_unchanged(self, runtime: str):
        """Explicit non-vllm runtimes pass through unchanged."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": runtime,
            }
        )
        assert recipe.runtime == runtime

    def test_resolve_eugr_with_ray_hints_respects_variant(self):
        """v1 with ray hints gets vllm-ray + eugr builder (ray hints respected)."""
        recipe = Recipe.from_dict(
            {
                "recipe_version": "1",
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "defaults": {"distributed_executor_backend": "ray"},
            }
        )
        assert recipe.runtime == "vllm-ray"
        assert recipe.builder == "eugr"

    @pytest.mark.parametrize(
        "data,expected",
        [
            ({"runtime": "vllm"}, "vllm-distributed"),
            ({"runtime": "sglang"}, "sglang"),
            ({"recipe_version": "1"}, "vllm-distributed"),
            ({"runtime": "vllm", "build_args": ["a"]}, "vllm-distributed"),
            ({"runtime": "vllm", "mods": ["m"]}, "vllm-distributed"),
            ({"runtime": "vllm", "defaults": {"distributed_executor_backend": "ray"}}, "vllm-ray"),
            ({"runtime": "vllm", "command": "vllm serve --distributed-executor-backend ray"}, "vllm-ray"),
            ({"runtime": "llama-cpp"}, "llama-cpp"),
            ({"runtime": "vllm-distributed"}, "vllm-distributed"),
            ({"runtime": "vllm-ray"}, "vllm-ray"),
        ],
    )
    def test_resolve_runtime_matches_recipe(self, data: dict[str, Any], expected: str):
        """resolve_runtime(data) matches Recipe.from_dict(data).runtime."""
        full_data = {"name": "Test", "model": "test-model", **data}
        assert resolve_runtime(full_data) == expected
        assert Recipe.from_dict(full_data).runtime == expected

    def test_resolve_runtime_standalone(self):
        """resolve_runtime() works on raw dicts without Recipe construction."""
        assert resolve_runtime({"runtime": "vllm"}) == "vllm-distributed"
        assert resolve_runtime({"runtime": "sglang"}) == "sglang"
        assert resolve_runtime({"recipe_version": "1"}) == "vllm-distributed"
        assert resolve_runtime({}) == "vllm-distributed"

    @pytest.mark.parametrize(
        "cmd",
        [
            "vllm serve {model} --distributed-executor-backend  ray",
            "vllm serve {model} --distributed-executor-backend\tray",
            "vllm serve {model} --distributed-executor-backend\n  ray",
        ],
    )
    def test_resolve_vllm_ray_hint_whitespace_variants(self, cmd: str):
        """Ray hint in command is detected despite varied whitespace."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "command": cmd,
            }
        )
        assert recipe.runtime == "vllm-ray"
        assert resolve_runtime({"runtime": "vllm", "command": cmd}) == "vllm-ray"

    def test_resolve_runtime_empty_runtime_string(self):
        """Empty runtime string is treated like 'vllm'."""
        assert resolve_runtime({"runtime": ""}) == "vllm-distributed"
        recipe = Recipe.from_dict({"name": "T", "model": "m", "runtime": ""})
        assert recipe.runtime == "vllm-distributed"

    @pytest.mark.parametrize(
        "bad_defaults",
        [
            "some_string",
            ["item1", "item2"],
            42,
        ],
    )
    def test_resolve_runtime_non_dict_defaults_raises(self, bad_defaults):
        """Non-mapping 'defaults' field is treated as a malformed recipe."""
        with pytest.raises(RecipeError, match="defaults.*mapping"):
            resolve_runtime({"runtime": "vllm", "defaults": bad_defaults})

    # --- Command-hint resolver tests ---

    @pytest.mark.parametrize(
        "cmd,expected",
        [
            ("vllm serve {model} -tp 4", "vllm-distributed"),
            ("vllm serve {model} --distributed-executor-backend ray", "vllm-ray"),
            ("sglang serve {model} --tp-size 4", "sglang"),
            ("sglang serve --model {model}", "sglang"),
            ("python -m sglang.launch_server --model {model}", "sglang"),
            ("python3 -m sglang.launch_server --model {model}", "sglang"),
            ("llama-server --model {model} -ngl 999", "llama-cpp"),
        ],
    )
    def test_resolve_command_hint(self, cmd: str, expected: str):
        """Command prefix infers runtime when no explicit runtime is set."""
        data = {"name": "Test", "model": "test-model", "command": cmd}
        assert Recipe.from_dict(data).runtime == expected
        assert resolve_runtime(data) == expected

    def test_resolve_command_hint_no_runtime_field(self):
        """Command hint works even when runtime field is entirely absent."""
        data = {"name": "Test", "model": "m", "command": "sglang serve m"}
        assert Recipe.from_dict(data).runtime == "sglang"
        assert resolve_runtime(data) == "sglang"

    def test_resolve_command_hint_empty_runtime(self):
        """Command hint works when runtime is explicitly empty string."""
        data = {"name": "T", "model": "m", "runtime": "", "command": "llama-server --model m"}
        assert Recipe.from_dict(data).runtime == "llama-cpp"
        assert resolve_runtime(data) == "llama-cpp"

    def test_resolve_command_hint_explicit_runtime_wins(self):
        """Explicit non-default runtime is not overridden by command hint."""
        data = {
            "name": "T",
            "model": "m",
            "runtime": "sglang",
            "command": "vllm serve {model}",
        }
        assert Recipe.from_dict(data).runtime == "sglang"

    def test_resolve_command_hint_vllm_serve_stays_vllm(self):
        """vllm serve command leaves runtime as vllm for variant resolution."""
        data = {"name": "T", "model": "m", "command": "vllm serve {model}"}
        assert Recipe.from_dict(data).runtime == "vllm-distributed"
        assert resolve_runtime(data) == "vllm-distributed"

    def test_resolve_command_hint_no_command(self):
        """No command field → normal fallback to vllm-distributed."""
        data = {"name": "T", "model": "m"}
        assert Recipe.from_dict(data).runtime == "vllm-distributed"

    def test_resolve_command_hint_does_not_override_v1(self):
        """v1 migration still wins for v1 recipes with a command."""
        data = {
            "recipe_version": "1",
            "name": "T",
            "model": "m",
            "command": "sglang serve {model}",
        }
        # v1 with default runtime → eugr-vllm takes priority
        # (command hint sets sglang, but v1 migration doesn't touch non-vllm)
        recipe = Recipe.from_dict(data)
        assert recipe.runtime == "sglang"


class TestFindRecipePrefix:
    """Test @registry/name scoped recipe lookups."""

    def test_parse_at_prefix(self, tmp_recipe_dir):
        """Test that @registry/name is parsed correctly."""
        from sparkrun.core.recipe import find_recipe, RecipeError

        # Should raise because there's no registry manager for scoped lookup
        with pytest.raises(RecipeError, match="not found"):
            find_recipe("@fake-reg/some-recipe", [tmp_recipe_dir])

    def test_find_recipe_ambiguous_raises(self, tmp_path):
        """Test that ambiguous matches raise RecipeAmbiguousError."""
        from sparkrun.core.recipe import find_recipe, RecipeAmbiguousError
        from sparkrun.core.registry import RegistryManager, RegistryEntry

        config = tmp_path / "config"
        cache = tmp_path / "cache"
        config.mkdir()
        cache.mkdir()
        mgr = RegistryManager(config, cache)

        # Create same recipe in two registries
        entries = [
            RegistryEntry(name="reg1", url="https://example.com/1", subpath="recipes"),
            RegistryEntry(name="reg2", url="https://example.com/2", subpath="recipes"),
        ]
        mgr._save_registries(entries)

        for entry in entries:
            recipe_dir = cache / entry.name / entry.subpath
            recipe_dir.mkdir(parents=True)
            (cache / entry.name / ".git").mkdir(exist_ok=True)
            with open(recipe_dir / "ambiguous-recipe.yaml", "w") as f:
                yaml.dump({"name": "Ambiguous", "model": "test", "runtime": "vllm"}, f)

        with pytest.raises(RecipeAmbiguousError) as exc_info:
            find_recipe("ambiguous-recipe", registry_manager=mgr)
        assert len(exc_info.value.matches) == 2

    def test_scoped_find_resolves_ambiguity(self, tmp_path):
        """Test that @registry/name resolves ambiguity."""
        from sparkrun.core.recipe import find_recipe
        from sparkrun.core.registry import RegistryManager, RegistryEntry

        config = tmp_path / "config"
        cache = tmp_path / "cache"
        config.mkdir()
        cache.mkdir()
        mgr = RegistryManager(config, cache)

        entries = [
            RegistryEntry(name="reg1", url="https://example.com/1", subpath="recipes"),
            RegistryEntry(name="reg2", url="https://example.com/2", subpath="recipes"),
        ]
        mgr._save_registries(entries)

        for entry in entries:
            recipe_dir = cache / entry.name / entry.subpath
            recipe_dir.mkdir(parents=True)
            (cache / entry.name / ".git").mkdir(exist_ok=True)
            with open(recipe_dir / "scoped-recipe.yaml", "w") as f:
                yaml.dump({"name": "Scoped", "model": "test", "runtime": "vllm"}, f)

        # Scoped to reg1 should work
        path = find_recipe("@reg1/scoped-recipe", registry_manager=mgr)
        assert "reg1" in str(path)

    def test_scoped_find_not_found(self, tmp_path):
        """Test that scoped find raises RecipeError when not found."""
        from sparkrun.core.recipe import find_recipe, RecipeError
        from sparkrun.core.registry import RegistryManager, RegistryEntry

        config = tmp_path / "config"
        cache = tmp_path / "cache"
        config.mkdir()
        cache.mkdir()
        mgr = RegistryManager(config, cache)

        entry = RegistryEntry(name="reg1", url="https://example.com", subpath="recipes")
        mgr._save_registries([entry])
        recipe_dir = cache / "reg1" / "recipes"
        recipe_dir.mkdir(parents=True)
        (cache / "reg1" / ".git").mkdir(exist_ok=True)

        with pytest.raises(RecipeError, match="not found in registry"):
            find_recipe("@reg1/nonexistent", registry_manager=mgr)


class TestIsRecipeFile:
    """Test is_recipe_file() validation."""

    def test_valid(self, tmp_path):
        """YAML with runtime, model, container returns True."""
        f = tmp_path / "good.yaml"
        f.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        assert is_recipe_file(f) is True

    def test_missing_model(self, tmp_path):
        """Returns False when model is missing."""
        f = tmp_path / "no-model.yaml"
        f.write_text(
            yaml.dump(
                {
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        assert is_recipe_file(f) is False

    def test_missing_container(self, tmp_path):
        """Returns False when container is missing."""
        f = tmp_path / "no-container.yaml"
        f.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "runtime": "vllm",
                }
            )
        )
        assert is_recipe_file(f) is False

    def test_missing_runtime(self, tmp_path):
        """Returns False when resolve_runtime returns unknown (non-dict doesn't reach it)."""
        f = tmp_path / "no-runtime.yaml"
        # A YAML list is not a dict, so is_recipe_file returns False
        f.write_text("- item1\n- item2\n")
        assert is_recipe_file(f) is False

    def test_not_yaml_dict(self, tmp_path):
        """Returns False for a YAML list."""
        f = tmp_path / "list.yaml"
        f.write_text("[1, 2, 3]\n")
        assert is_recipe_file(f) is False

    def test_nonexistent(self, tmp_path):
        """Returns False for a file that doesn't exist."""
        assert is_recipe_file(tmp_path / "nope.yaml") is False

    def test_invalid_yaml(self, tmp_path):
        """Returns False for unparseable YAML."""
        f = tmp_path / "bad.yaml"
        f.write_text(": :\n  - [invalid\n")
        assert is_recipe_file(f) is False

    def test_empty_model_string(self, tmp_path):
        """Returns False when model is an empty string."""
        f = tmp_path / "empty-model.yaml"
        f.write_text(
            yaml.dump(
                {
                    "model": "",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        assert is_recipe_file(f) is False


class TestDiscoverCwdRecipes:
    """Test discover_cwd_recipes() directory scanning."""

    def test_finds_valid(self, tmp_path):
        """Directory with valid + invalid YAML, only valid returned."""
        valid = tmp_path / "good.yaml"
        valid.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "sglang",
                }
            )
        )
        invalid = tmp_path / "not-recipe.yaml"
        invalid.write_text(yaml.dump({"key": "value"}))

        result = discover_cwd_recipes(tmp_path)
        assert len(result) == 1
        assert result[0] == valid

    def test_flat_only(self, tmp_path):
        """Recipe in subdirectory is NOT found."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        nested = sub / "nested.yaml"
        nested.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )

        result = discover_cwd_recipes(tmp_path)
        assert result == []

    def test_empty_dir(self, tmp_path):
        """Returns empty list for a directory with no YAML files."""
        result = discover_cwd_recipes(tmp_path)
        assert result == []

    def test_yml_extension(self, tmp_path):
        """Files with .yml extension are also discovered."""
        f = tmp_path / "recipe.yml"
        f.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        result = discover_cwd_recipes(tmp_path)
        assert len(result) == 1
        assert result[0] == f

    def test_nonexistent_dir(self, tmp_path):
        """Returns empty list for a nonexistent directory."""
        result = discover_cwd_recipes(tmp_path / "does-not-exist")
        assert result == []

    def test_sorted_output(self, tmp_path):
        """Results are sorted by path."""
        for name in ("z-recipe.yaml", "a-recipe.yaml", "m-recipe.yaml"):
            (tmp_path / name).write_text(
                yaml.dump(
                    {
                        "model": "org/model",
                        "container": "img:latest",
                        "runtime": "vllm",
                    }
                )
            )
        result = discover_cwd_recipes(tmp_path)
        assert len(result) == 3
        assert result == sorted(result)


class TestListRecipesLocalFiles:
    """Test list_recipes() with local_files parameter."""

    def test_local_files_appear_without_registry(self, tmp_path):
        """Local files appear with no registry key."""
        f = tmp_path / "local-recipe.yaml"
        f.write_text(
            yaml.dump(
                {
                    "name": "My Local Recipe",
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "sglang",
                    "defaults": {"tensor_parallel": 2},
                }
            )
        )
        recipes = list_recipes(local_files=[f])
        assert len(recipes) == 1
        assert recipes[0]["name"] == "local-recipe"  # name is the filename stem
        assert recipes[0]["file"] == "local-recipe"
        assert recipes[0]["runtime"] == "sglang"
        assert "registry" not in recipes[0]

    def test_local_files_dedup_with_registry(self, tmp_path):
        """Local file with same stem as a registry recipe wins (listed first)."""
        f = tmp_path / "dupe.yaml"
        f.write_text(
            yaml.dump(
                {
                    "name": "Local Dupe",
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        # Create a search_path with same-stem recipe
        search = tmp_path / "search"
        search.mkdir()
        (search / "dupe.yaml").write_text(
            yaml.dump(
                {
                    "name": "Registry Dupe",
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        recipes = list_recipes(search_paths=[search], local_files=[f])
        # Only one recipe with stem "dupe" should appear (the local one)
        dupe_recipes = [r for r in recipes if r["file"] == "dupe"]
        assert len(dupe_recipes) == 1
        assert dupe_recipes[0]["name"] == "dupe"  # name is the filename stem


class TestFindRecipeLocalFiles:
    """Test find_recipe() with local_files parameter."""

    def test_matches_local_file_by_stem(self, tmp_path):
        """find_recipe resolves from local_files by stem."""
        f = tmp_path / "my-local.yaml"
        f.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        result = find_recipe("my-local", local_files=[f])
        assert result == f

    def test_matches_local_file_with_extension(self, tmp_path):
        """find_recipe matches local_files when name includes .yaml extension."""
        f = tmp_path / "my-local.yaml"
        f.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        result = find_recipe("my-local.yaml", local_files=[f])
        assert result == f

    def test_direct_path_takes_priority(self, tmp_path):
        """Direct file path still wins over local_files."""
        direct = tmp_path / "direct.yaml"
        direct.write_text(
            yaml.dump(
                {
                    "model": "org/model",
                    "container": "img:latest",
                    "runtime": "vllm",
                }
            )
        )
        other = tmp_path / "other" / "direct.yaml"
        other.parent.mkdir()
        other.write_text(
            yaml.dump(
                {
                    "model": "org/other",
                    "container": "img:latest",
                    "runtime": "sglang",
                }
            )
        )
        result = find_recipe(str(direct), local_files=[other])
        assert result == direct


class TestLifecycleHookFields:
    """Test pre_exec, post_exec, post_commands, stop_after_post recipe fields."""

    def test_pre_exec_string_list(self):
        """pre_exec parses as list of strings."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "pre_exec": ["pip install transformers", "echo hello"],
            }
        )
        assert recipe.pre_exec == ["pip install transformers", "echo hello"]

    def test_pre_exec_dict_entries(self):
        """pre_exec supports dict entries for copy commands."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "pre_exec": [
                    {"copy": "/path/to/mod", "dest": "/workspace/mods/mod"},
                    "cd /workspace/mods/mod && ./run.sh",
                ],
            }
        )
        assert len(recipe.pre_exec) == 2
        assert isinstance(recipe.pre_exec[0], dict)
        assert recipe.pre_exec[0]["copy"] == "/path/to/mod"
        assert isinstance(recipe.pre_exec[1], str)

    def test_pre_exec_default_empty(self):
        """pre_exec defaults to empty list."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.pre_exec == []

    def test_post_exec_string_list(self):
        """post_exec parses as list of strings."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "post_exec": ["echo 'Server ready'"],
            }
        )
        assert recipe.post_exec == ["echo 'Server ready'"]

    def test_post_exec_default_empty(self):
        """post_exec defaults to empty list."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.post_exec == []

    def test_post_commands_string_list(self):
        """post_commands parses as list of strings."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "post_commands": [
                    "curl http://{head_host}:{port}/v1/models",
                    "echo done",
                ],
            }
        )
        assert len(recipe.post_commands) == 2

    def test_post_commands_default_empty(self):
        """post_commands defaults to empty list."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.post_commands == []

    def test_stop_after_post_true(self):
        """stop_after_post parses as bool True."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "stop_after_post": True,
            }
        )
        assert recipe.stop_after_post is True

    def test_stop_after_post_default_false(self):
        """stop_after_post defaults to False."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.stop_after_post is False

    def test_hook_fields_not_in_runtime_config(self):
        """Hook fields should not leak into runtime_config."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "pre_exec": ["echo hello"],
                "post_exec": ["echo done"],
                "post_commands": ["curl localhost"],
                "stop_after_post": True,
            }
        )
        assert "pre_exec" not in recipe.runtime_config
        assert "post_exec" not in recipe.runtime_config
        assert "post_commands" not in recipe.runtime_config
        assert "stop_after_post" not in recipe.runtime_config

    def test_pre_exec_in_export(self):
        """pre_exec appears in export dict when set."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "pre_exec": ["echo hello"],
            }
        )
        export = recipe._build_export_dict()
        assert "pre_exec" in export
        assert export["pre_exec"] == ["echo hello"]

    def test_post_commands_in_export(self):
        """post_commands appears in export dict when set."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "post_commands": ["curl localhost"],
            }
        )
        export = recipe._build_export_dict()
        assert "post_commands" in export

    def test_stop_after_post_in_export(self):
        """stop_after_post appears in export dict when True."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "stop_after_post": True,
            }
        )
        export = recipe._build_export_dict()
        assert "stop_after_post" in export
        assert export["stop_after_post"] is True

    def test_empty_hooks_not_in_export(self):
        """Empty hook lists and False stop_after_post don't appear in export."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
            }
        )
        export = recipe._build_export_dict()
        assert "pre_exec" not in export
        assert "post_exec" not in export
        assert "post_commands" not in export
        assert "stop_after_post" not in export


class TestBuilderFields:
    """Test builder and builder_config recipe fields."""

    def test_builder_field(self):
        """builder field is parsed from recipe data."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "builder": "eugr",
            }
        )
        assert recipe.builder == "eugr"

    def test_builder_default_empty(self):
        """builder defaults to empty string."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.builder == ""

    def test_builder_config_field(self):
        """builder_config field is parsed from recipe data."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "builder": "eugr",
                "builder_config": {"repo": "https://example.com/repo.git"},
            }
        )
        assert recipe.builder_config == {"repo": "https://example.com/repo.git"}

    def test_builder_config_default_empty(self):
        """builder_config defaults to empty dict."""
        recipe = Recipe.from_dict({"name": "Test", "model": "test-model"})
        assert recipe.builder_config == {}

    def test_builder_not_in_runtime_config(self):
        """builder field should not leak into runtime_config."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "builder": "eugr",
                "builder_config": {"repo": "url"},
            }
        )
        assert "builder" not in recipe.runtime_config
        assert "builder_config" not in recipe.runtime_config

    def test_builder_in_export(self):
        """builder appears in export dict when set."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "builder": "eugr",
            }
        )
        export = recipe._build_export_dict()
        assert "builder" in export
        assert export["builder"] == "eugr"

    def test_builder_config_in_export(self):
        """builder_config appears in export dict when set."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "builder": "eugr",
                "builder_config": {"repo": "url"},
            }
        )
        export = recipe._build_export_dict()
        assert "builder_config" in export

    def test_empty_builder_not_in_export(self):
        """Empty builder string does NOT appear in export dict."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
            }
        )
        export = recipe._build_export_dict()
        assert "builder" not in export
        assert "builder_config" not in export


class TestExportWithOverrides:
    """Tests for Recipe.export() with overrides and container_image params."""

    def test_export_bakes_overrides_into_defaults(self):
        """Overrides are merged into the exported defaults dict."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "defaults": {"port": 8000, "host": "0.0.0.0"},
            }
        )
        text = recipe.export(overrides={"port": 9999, "tensor_parallel": 4})
        import yaml

        data = yaml.safe_load(text)
        assert data["defaults"]["port"] == 9999
        assert data["defaults"]["tensor_parallel"] == 4
        assert data["defaults"]["host"] == "0.0.0.0"

    def test_export_container_image_override(self):
        """container_image param overrides container in export."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "original:latest",
            }
        )
        text = recipe.export(container_image="custom/image:v2")
        import yaml

        data = yaml.safe_load(text)
        assert data["container"] == "custom/image:v2"

    def test_export_no_overrides_unchanged(self):
        """Without overrides, export matches base behavior."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "defaults": {"port": 8000},
            }
        )
        text_base = recipe.export()
        text_override = recipe.export(overrides=None, container_image=None)
        import yaml

        assert yaml.safe_load(text_base) == yaml.safe_load(text_override)


class TestResolveWithOverrides:
    """Tests for Recipe.resolve() with CLI overrides influencing runtime."""

    def test_resolve_distributed_executor_override_ray(self):
        """Override distributed_executor_backend=ray switches vllm-distributed to vllm-ray."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
            }
        )
        # Default: vllm-distributed
        assert recipe.runtime == "vllm-distributed"

        # Re-resolve with override
        recipe.resolve({"distributed_executor_backend": "ray"})
        assert recipe.runtime == "vllm-ray"

    def test_resolve_override_ray_to_distributed(self):
        """Override distributed_executor_backend=mp overrides recipe default of ray."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
                "defaults": {"distributed_executor_backend": "ray"},
            }
        )
        assert recipe.runtime == "vllm-ray"

        # Override to mp → should resolve to vllm-distributed
        recipe.resolve({"distributed_executor_backend": "mp"})
        assert recipe.runtime == "vllm-distributed"

    def test_resolve_non_affecting_overrides(self):
        """Overrides without runtime-affecting keys don't change runtime."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "runtime": "sglang",
                "container": "img:latest",
            }
        )
        assert recipe.runtime == "sglang"

        recipe.resolve({"tensor_parallel": 4, "max_model_len": 8192})
        assert recipe.runtime == "sglang"

    def test_resolve_idempotent(self):
        """Calling resolve() twice with same overrides produces same result."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
            }
        )
        recipe.resolve({"distributed_executor_backend": "ray"})
        assert recipe.runtime == "vllm-ray"

        recipe.resolve({"distributed_executor_backend": "ray"})
        assert recipe.runtime == "vllm-ray"

    def test_resolve_resets_before_rerunning(self):
        """resolve() resets runtime to raw YAML value before re-running chain."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
            }
        )
        recipe.resolve({"distributed_executor_backend": "ray"})
        assert recipe.runtime == "vllm-ray"

        # Re-resolve without override — should go back to default
        recipe.resolve()
        assert recipe.runtime == "vllm-distributed"

    def test_from_dict_backward_compat(self):
        """from_dict() auto-resolves (existing tests rely on this)."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
                "defaults": {"distributed_executor_backend": "ray"},
            }
        )
        assert recipe.runtime == "vllm-ray"

    def test_from_dict_with_overrides(self):
        """from_dict(overrides=...) passes overrides to resolve."""
        recipe = Recipe.from_dict(
            {"model": "test-model", "container": "img:latest"},
            overrides={"distributed_executor_backend": "ray"},
        )
        assert recipe.runtime == "vllm-ray"

    def test_load_resolve_false(self, tmp_recipe_dir: Path):
        """Recipe.load(path, resolve=False) leaves runtime unresolved."""
        recipe_path = tmp_recipe_dir / "test-vllm.yaml"
        recipe = Recipe.load(recipe_path, resolve=False)
        # Runtime should be the raw value from YAML (resolvers haven't run)
        assert recipe.runtime == "vllm"

        # After explicit resolve, runtime is fully resolved
        recipe.resolve()
        assert recipe.runtime == "vllm-distributed"

        # Re-resolve with override
        recipe.resolve({"distributed_executor_backend": "ray"})
        assert recipe.runtime == "vllm-ray"

    def test_effective_default_override_wins(self):
        """_effective_default returns override value over recipe default."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
                "defaults": {"tensor_parallel": 2},
            }
        )
        recipe._applied_overrides = {"tensor_parallel": 4}
        assert recipe._effective_default("tensor_parallel") == 4

    def test_effective_default_falls_back_to_defaults(self):
        """_effective_default returns recipe default when no override."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
                "defaults": {"tensor_parallel": 2},
            }
        )
        assert recipe._effective_default("tensor_parallel") == 2

    def test_effective_default_fallback(self):
        """_effective_default returns fallback when key absent everywhere."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "container": "img:latest",
            }
        )
        assert recipe._effective_default("nonexistent", "default_val") == "default_val"

    def test_resolve_runtime_standalone_with_overrides(self):
        """resolve_runtime() standalone function respects overrides."""
        data = {"model": "test-model", "container": "img:latest"}
        assert resolve_runtime(data) == "vllm-distributed"
        assert resolve_runtime(data, overrides={"distributed_executor_backend": "ray"}) == "vllm-ray"

    def test_v1_recipe_with_ray_hints_gets_vllm_ray(self):
        """v1 recipe with Ray hints resolves to vllm-ray + eugr builder."""
        recipe = Recipe.from_dict(
            {
                "recipe_version": "1",
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "defaults": {"distributed_executor_backend": "ray"},
            }
        )
        assert recipe.runtime == "vllm-ray"
        assert recipe.builder == "eugr"

    def test_explicit_builder_not_overwritten_by_v1(self):
        """Explicit builder: custom is not overwritten by v1 migration."""
        recipe = Recipe.from_dict(
            {
                "recipe_version": "1",
                "name": "Test",
                "model": "test-model",
                "builder": "custom",
            }
        )
        assert recipe.builder == "custom"

    def test_explicit_builder_not_overwritten_by_eugr_signals(self):
        """Explicit builder: custom is not overwritten by build_args/mods."""
        recipe = Recipe.from_dict(
            {
                "name": "Test",
                "model": "test-model",
                "runtime": "vllm",
                "builder": "custom",
                "build_args": ["--pre-tf"],
            }
        )
        assert recipe.builder == "custom"
        assert recipe.runtime == "vllm-distributed"


class TestResolveBuilder:
    """Test the resolve_builder() standalone function."""

    def test_explicit_builder(self):
        """Explicit builder field is returned as-is."""
        assert resolve_builder({"builder": "custom"}) == "custom"

    def test_v1_recipe(self):
        """v1 recipe returns eugr."""
        assert resolve_builder({"recipe_version": "1"}) == "eugr"

    def test_v1_recipe_non_vllm(self):
        """v1 recipe with non-vllm runtime returns empty."""
        assert resolve_builder({"recipe_version": "1", "runtime": "sglang"}) == ""

    def test_build_args(self):
        """build_args returns eugr."""
        assert resolve_builder({"runtime": "vllm", "build_args": ["a"]}) == "eugr"

    def test_mods(self):
        """mods returns eugr."""
        assert resolve_builder({"runtime": "vllm", "mods": ["m"]}) == "eugr"

    def test_runtime_config_build_args(self):
        """build_args in runtime_config returns eugr."""
        assert resolve_builder({"runtime": "vllm", "runtime_config": {"build_args": ["a"]}}) == "eugr"

    def test_plain_vllm(self):
        """Plain vllm recipe returns empty."""
        assert resolve_builder({"runtime": "vllm"}) == ""

    def test_empty_data(self):
        """Empty data returns empty."""
        assert resolve_builder({}) == ""

    def test_explicit_builder_takes_priority(self):
        """Explicit builder takes priority over v1 signals."""
        assert resolve_builder({"recipe_version": "1", "builder": "custom"}) == "custom"


class TestRecipeSerialization:
    """Tests for Recipe.__getstate__/__setstate__ and YAML round-trip."""

    def test_getstate_has_serialization_version(self, sample_v2_recipe_data: dict):
        """__getstate__ includes _serialization_version marker."""
        recipe = Recipe.from_dict(sample_v2_recipe_data)
        state = recipe.__getstate__()
        assert state["_serialization_version"] == 1

    def test_round_trip_v2_recipe(self, sample_v2_recipe_data: dict):
        """Round-trip a v2 recipe through __getstate__/__setstate__."""
        original = Recipe.from_dict(sample_v2_recipe_data)
        state = original.__getstate__()
        restored = Recipe._deserialize(state)

        assert restored.name == original.name
        assert restored.model == original.model
        assert restored.runtime == original.runtime
        assert restored.mode == original.mode
        assert restored.container == original.container
        assert restored.defaults == original.defaults
        assert restored.env == original.env
        assert restored.command == original.command
        assert restored.min_nodes == original.min_nodes
        assert restored.max_nodes == original.max_nodes
        assert restored.metadata == original.metadata
        assert restored.recipe_version == original.recipe_version
        assert restored.description == original.description
        assert restored._raw == original._raw

    def test_round_trip_yaml(self, sample_v2_recipe_data: dict):
        """Round-trip through _serialize_yaml / _deserialize_yaml."""
        original = Recipe.from_dict(sample_v2_recipe_data)
        yaml_text = original._serialize_yaml()
        restored = Recipe._deserialize_yaml(yaml_text)

        assert restored.name == original.name
        assert restored.model == original.model
        assert restored.runtime == original.runtime
        assert restored.defaults == original.defaults
        assert restored.env == original.env
        assert restored.command == original.command
        assert restored._applied_overrides == original._applied_overrides

    def test_round_trip_with_overrides(self, sample_v2_recipe_data: dict):
        """Overrides survive serialization round-trip."""
        original = Recipe.from_dict(
            sample_v2_recipe_data,
            overrides={"distributed_executor_backend": "ray"},
        )
        assert original.runtime == "vllm-ray"

        yaml_text = original._serialize_yaml()
        restored = Recipe._deserialize_yaml(yaml_text)

        assert restored.runtime == "vllm-ray"
        assert restored._applied_overrides == {"distributed_executor_backend": "ray"}

    def test_round_trip_with_registry_info(self, sample_v2_recipe_data: dict):
        """Registry metadata survives round-trip."""
        original = Recipe.from_dict(sample_v2_recipe_data)
        original.source_registry = "my-registry"
        original.source_registry_url = "https://github.com/example/recipes.git"
        original.source_path = "/some/path/recipe.yaml"

        restored = Recipe._deserialize(original.__getstate__())

        assert restored.source_registry == "my-registry"
        assert restored.source_registry_url == "https://github.com/example/recipes.git"
        assert restored.source_path == "/some/path/recipe.yaml"
        assert restored.qualified_name == "@my-registry/%s" % original.name

    def test_round_trip_lifecycle_hooks(self):
        """Lifecycle hooks (pre_exec, post_exec, etc.) survive round-trip."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "runtime": "sglang",
                "container": "img:latest",
                "pre_exec": ["echo pre"],
                "post_exec": ["echo post"],
                "post_commands": ["curl localhost"],
                "stop_after_post": True,
            }
        )
        restored = Recipe._deserialize(recipe.__getstate__())

        assert restored.pre_exec == ["echo pre"]
        assert restored.post_exec == ["echo post"]
        assert restored.post_commands == ["curl localhost"]
        assert restored.stop_after_post is True

    def test_round_trip_builder_fields(self):
        """Builder and builder_config survive round-trip."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "runtime": "vllm",
                "container": "img:latest",
                "builder": "eugr",
                "builder_config": {"repo": "https://example.com"},
            }
        )
        restored = Recipe._deserialize(recipe.__getstate__())

        assert restored.builder == "eugr"
        assert restored.builder_config == {"repo": "https://example.com"}

    def test_round_trip_executor_config(self):
        """executor_config survives round-trip."""
        recipe = Recipe.from_dict(
            {
                "model": "test-model",
                "runtime": "sglang",
                "container": "img:latest",
                "executor_config": {"auto_remove": True},
            }
        )
        restored = Recipe._deserialize(recipe.__getstate__())
        assert restored.executor_config == {"auto_remove": True}

    def test_round_trip_qualified_name_override(self, sample_v2_recipe_data: dict):
        """_qualified_name_override survives round-trip."""
        original = Recipe.from_dict(sample_v2_recipe_data)
        original._qualified_name_override = "custom-qualified-name"

        restored = Recipe._deserialize(original.__getstate__())
        assert restored.qualified_name == "custom-qualified-name"

    def test_deserialize_yaml_invalid_input(self):
        """_deserialize_yaml raises RecipeError on non-dict YAML."""
        with pytest.raises(RecipeError, match="YAML mapping"):
            Recipe._deserialize_yaml("- just a list")

    def test_re_resolve_after_deserialize(self, sample_v2_recipe_data: dict):
        """A deserialized recipe can be re-resolved with new overrides."""
        original = Recipe.from_dict(sample_v2_recipe_data)
        assert original.runtime == "vllm-distributed"

        restored = Recipe._deserialize(original.__getstate__())
        restored.resolve({"distributed_executor_backend": "ray"})
        assert restored.runtime == "vllm-ray"
