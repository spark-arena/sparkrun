"""Tests for sparkrun.tuning module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.tuning.sglang import (
    build_tuning_command,
    get_sglang_tuning_dir,
    get_sglang_tuning_env,
    get_sglang_tuning_volumes,
    SglangTuner,
    SGLANG_CLONE_DIR,
    TUNE_CONTAINER_NAME,
    TUNING_CONTAINER_OUTPUT_PATH,
    TUNING_CONTAINER_PATH,
    TUNING_ENV_PATH,
    DEFAULT_TP_SIZES,
    _format_duration,
)
from sparkrun.tuning.vllm import (
    build_vllm_tuning_command,
    get_vllm_tuning_dir,
    get_vllm_tuning_env,
    get_vllm_tuning_volumes,
    VllmTuner,
    TUNE_VLLM_CONTAINER_NAME,
    VLLM_CLONE_DIR,
    VLLM_TUNING_CONTAINER_OUTPUT_PATH,
    VLLM_TUNING_CONTAINER_PATH,
    DEFAULT_TP_SIZES as VLLM_DEFAULT_TP_SIZES,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

class TestGetSglangTuningDir:
    def test_returns_path_under_cache(self):
        d = get_sglang_tuning_dir()
        assert isinstance(d, Path)
        assert str(d).endswith("sparkrun/tuning/sglang")

    def test_is_under_home(self):
        d = get_sglang_tuning_dir()
        assert str(d).startswith(str(Path.home()))


class TestGetSglangTuningVolumes:
    def test_returns_none_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning._common.DEFAULT_CACHE_DIR",
            tmp_path / "nonexistent_cache",
        )
        assert get_sglang_tuning_volumes() is None

    def test_returns_none_when_dir_empty(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "sparkrun" / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_dir",
            lambda: tuning_dir,
        )
        assert get_sglang_tuning_volumes() is None

    def test_returns_mapping_when_json_exists(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_sglang_tuning_volumes()
        assert result is not None
        assert result[str(tuning_dir)] == TUNING_CONTAINER_PATH

    def test_returns_mapping_for_nested_json(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        nested = tuning_dir / "configs" / "triton_3_2_0"
        nested.mkdir(parents=True)
        (nested / "E=128_N=256.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_sglang_tuning_volumes()
        assert result is not None


class TestGetSglangTuningEnv:
    def test_returns_none_when_no_configs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: None,
        )
        assert get_sglang_tuning_env() is None

    def test_returns_env_when_configs_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: {"/some/path": TUNING_CONTAINER_PATH},
        )
        result = get_sglang_tuning_env()
        assert result is not None
        assert "SGLANG_MOE_CONFIG_DIR" in result
        assert result["SGLANG_MOE_CONFIG_DIR"] == TUNING_ENV_PATH


# ---------------------------------------------------------------------------
# build_tuning_command
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_under_60_seconds(self):
        assert _format_duration(5.3) == "5.3s"
        assert _format_duration(0.0) == "0.0s"
        assert _format_duration(59.9) == "59.9s"

    def test_minutes(self):
        assert _format_duration(60) == "1m 0s"
        assert _format_duration(90) == "1m 30s"
        assert _format_duration(754) == "12m 34s"

    def test_hours(self):
        assert _format_duration(3600) == "1h 0m 0s"
        assert _format_duration(3661) == "1h 1m 1s"
        assert _format_duration(7384) == "2h 3m 4s"


class TestBuildTuningCommand:
    def test_contains_model(self):
        cmd = build_tuning_command("Qwen/Qwen3-MoE", 4)
        assert "Qwen/Qwen3-MoE" in cmd

    def test_contains_tp_size(self):
        cmd = build_tuning_command("test-model", 8)
        assert "--tp-size 8" in cmd

    def test_contains_tune_flag(self):
        cmd = build_tuning_command("test-model", 1)
        assert "--tune" in cmd

    def test_contains_config_dir(self):
        cmd = build_tuning_command("test-model", 2)
        assert "SGLANG_MOE_CONFIG_DIR=" in cmd
        assert TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_config_dir_is_base_path(self):
        """SGLANG_MOE_CONFIG_DIR should be the base output path — SGLang
        internally appends configs/triton_<ver>/ via get_config_file_name()."""
        cmd = build_tuning_command("test-model", 2, triton_version="3.6.0")
        assert "SGLANG_MOE_CONFIG_DIR=%s " % TUNING_CONTAINER_OUTPUT_PATH in cmd
        # Should NOT double-nest the triton version in the env var
        assert "SGLANG_MOE_CONFIG_DIR=%s/triton_" % TUNING_CONTAINER_OUTPUT_PATH not in cmd

    def test_versioned_mkdir(self):
        """When triton_version is provided, pre-create the versioned subdir."""
        cmd = build_tuning_command("test-model", 2, triton_version="3.6.0")
        assert "mkdir -p %s/triton_3_6_0" % TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_no_version_uses_base_dir(self):
        """Without triton_version, output_subdir is the base path itself."""
        cmd = build_tuning_command("test-model", 2)
        assert "mkdir -p %s " % TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_unknown_version_uses_base_dir(self):
        cmd = build_tuning_command("test-model", 2, triton_version="unknown")
        assert "mkdir -p %s " % TUNING_CONTAINER_OUTPUT_PATH in cmd
        assert "triton_unknown" not in cmd

    def test_cwd_is_output_dir(self):
        """CWD must be the output subdir so save_configs() writes there."""
        cmd = build_tuning_command("test-model", 1)
        assert "cd %s " % TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_script_uses_full_clone_path(self):
        """Script must be invoked via full path since CWD is the output dir."""
        cmd = build_tuning_command("test-model", 1)
        assert "%s/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py" % SGLANG_CLONE_DIR in cmd

    def test_various_tp_sizes(self):
        for tp in DEFAULT_TP_SIZES:
            cmd = build_tuning_command("model", tp)
            assert "--tp-size %d" % tp in cmd

    def test_model_with_single_quote_is_shell_safe(self):
        """Model names with single quotes must not break shell quoting."""
        import shlex
        cmd = build_tuning_command("org/model'name", 1)
        assert shlex.quote("org/model'name") in cmd

    def test_model_with_spaces_is_shell_safe(self):
        """Model names with spaces must be shell-quoted."""
        import shlex
        cmd = build_tuning_command("org/model name", 1)
        assert shlex.quote("org/model name") in cmd


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestTuneSglangCLI:
    def test_help(self):
        from sparkrun.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["tune", "sglang", "--help"])
        assert result.exit_code == 0
        assert "Tune SGLang fused MoE Triton kernels" in result.output
        assert "--tp" in result.output
        assert "--skip-clone" in result.output

    def test_tune_group_help(self):
        from sparkrun.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["tune", "--help"])
        assert result.exit_code == 0
        assert "sglang" in result.output

    def test_rejects_non_sglang_recipe(self, tmp_path, monkeypatch):
        """tune sglang should reject recipes with non-sglang runtime."""
        from sparkrun.cli import main

        # Create a vllm recipe
        recipe_file = tmp_path / "test-vllm.yaml"
        recipe_file.write_text(yaml.dump({
            "sparkrun_version": "2",
            "name": "Test vLLM",
            "model": "test/model",
            "runtime": "vllm",
            "container": "test:latest",
        }))

        runner = CliRunner()
        result = runner.invoke(main, [
            "tune", "sglang", str(recipe_file), "-H", "10.0.0.1", "-n",
        ])
        assert result.exit_code != 0
        assert "requires an SGLang recipe" in result.output


# ---------------------------------------------------------------------------
# SglangTuner dry-run
# ---------------------------------------------------------------------------

class TestSglangTunerDryRun:
    @pytest.fixture
    def sglang_recipe_file(self, tmp_path):
        recipe_file = tmp_path / "test-sglang.yaml"
        recipe_file.write_text(yaml.dump({
            "sparkrun_version": "2",
            "name": "Test SGLang",
            "model": "Qwen/Qwen3-MoE",
            "runtime": "sglang",
            "container": "scitrera/dgx-spark-sglang:latest",
            "defaults": {"tensor_parallel": 2},
        }))
        return recipe_file

    def test_tuner_dry_run_returns_zero(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1,))
        assert rc == 0

    def test_tuner_dry_run_all_default_tp_sizes(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning()
        assert rc == 0

    def test_tuner_custom_output_dir(self, tmp_path):
        custom_dir = str(tmp_path / "custom_tuning")
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            output_dir=custom_dir,
            dry_run=True,
        )
        assert tuner.output_dir == custom_dir

    def test_tuner_skip_clone(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            skip_clone=True,
            dry_run=True,
        )
        assert tuner.skip_clone is True
        rc = tuner.run_tuning(tp_sizes=(1,))
        assert rc == 0

    def test_default_tp_sizes_constant(self):
        assert DEFAULT_TP_SIZES == (1, 2, 4, 8)

    def test_tuner_dry_run_parallel(self):
        tuner = SglangTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1, 2, 4), parallel=2)
        assert rc == 0

    def test_container_name_constant(self):
        assert TUNE_CONTAINER_NAME == "sparkrun_tune"


# ---------------------------------------------------------------------------
# Auto-mount integration
# ---------------------------------------------------------------------------

class TestSglangRuntimeAutoMount:
    def test_get_extra_volumes_empty_when_no_configs(self, v, monkeypatch):
        """SglangRuntime.get_extra_volumes returns {} when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: None,
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_volumes() == {}

    def test_get_extra_env_empty_when_no_configs(self, v, monkeypatch):
        """SglangRuntime.get_extra_env returns {} when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_env",
            lambda: None,
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface"}

    def test_get_extra_volumes_returns_mapping(self, v, monkeypatch):
        """SglangRuntime.get_extra_volumes returns mapping when configs exist."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"/cache/tuning/sglang": TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_volumes() == expected

    def test_get_extra_env_returns_env(self, v, monkeypatch):
        """SglangRuntime.get_extra_env returns env when configs exist."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"HF_HOME": "/cache/huggingface", "SGLANG_MOE_CONFIG_DIR": TUNING_ENV_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.sglang.get_sglang_tuning_env",
            lambda: {"SGLANG_MOE_CONFIG_DIR": TUNING_ENV_PATH},
        )
        runtime = get_runtime("sglang", v)
        assert runtime.get_extra_env() == expected


# ---------------------------------------------------------------------------
# Base runtime hooks
# ---------------------------------------------------------------------------

class TestBaseRuntimeHooks:
    def test_default_get_extra_volumes(self, v):
        """Base RuntimePlugin.get_extra_volumes returns empty dict."""
        from sparkrun.core.bootstrap import get_runtime
        # llama-cpp doesn't override get_extra_volumes
        runtime = get_runtime("llama-cpp", v)
        assert runtime.get_extra_volumes() == {}

    def test_default_get_extra_env(self, v):
        """Base RuntimePlugin.get_extra_env returns HF_HOME."""
        from sparkrun.core.bootstrap import get_runtime
        runtime = get_runtime("llama-cpp", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface"}


# ===========================================================================
# vLLM tuning tests
# ===========================================================================

# ---------------------------------------------------------------------------
# vLLM path helpers
# ---------------------------------------------------------------------------

class TestGetVllmTuningDir:
    def test_returns_path_under_cache(self):
        d = get_vllm_tuning_dir()
        assert isinstance(d, Path)
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_is_under_home(self):
        d = get_vllm_tuning_dir()
        assert str(d).startswith(str(Path.home()))


class TestGetVllmTuningVolumes:
    def test_returns_none_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning._common.DEFAULT_CACHE_DIR",
            tmp_path / "nonexistent_cache",
        )
        assert get_vllm_tuning_volumes() is None

    def test_returns_none_when_dir_empty(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "sparkrun" / "tuning" / "vllm"
        tuning_dir.mkdir(parents=True)
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_dir",
            lambda: tuning_dir,
        )
        assert get_vllm_tuning_volumes() is None

    def test_returns_mapping_when_json_exists(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "vllm"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_vllm_tuning_volumes()
        assert result is not None
        assert result[str(tuning_dir)] == VLLM_TUNING_CONTAINER_PATH

    def test_returns_mapping_for_nested_json(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "vllm"
        nested = tuning_dir / "configs" / "triton_3_2_0"
        nested.mkdir(parents=True)
        (nested / "E=128_N=256.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_dir",
            lambda: tuning_dir,
        )
        result = get_vllm_tuning_volumes()
        assert result is not None


class TestGetVllmTuningEnv:
    def test_returns_none_when_no_configs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: None,
        )
        assert get_vllm_tuning_env() is None

    def test_returns_env_when_configs_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: {"/some/path": VLLM_TUNING_CONTAINER_PATH},
        )
        result = get_vllm_tuning_env()
        assert result is not None
        assert "VLLM_TUNED_CONFIG_FOLDER" in result
        assert result["VLLM_TUNED_CONFIG_FOLDER"] == VLLM_TUNING_CONTAINER_PATH


# ---------------------------------------------------------------------------
# build_vllm_tuning_command
# ---------------------------------------------------------------------------

class TestBuildVllmTuningCommand:
    def test_contains_model(self):
        cmd = build_vllm_tuning_command("Qwen/Qwen3-MoE", 4)
        assert "Qwen/Qwen3-MoE" in cmd

    def test_contains_tp_size(self):
        cmd = build_vllm_tuning_command("test-model", 8)
        assert "--tp-size 8" in cmd

    def test_contains_tune_flag(self):
        cmd = build_vllm_tuning_command("test-model", 1)
        assert "--tune" in cmd

    def test_contains_config_dir(self):
        cmd = build_vllm_tuning_command("test-model", 2)
        assert "VLLM_TUNED_CONFIG_FOLDER=" in cmd
        assert VLLM_TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_creates_output_dir(self):
        cmd = build_vllm_tuning_command("test-model", 2)
        assert "mkdir -p %s" % VLLM_TUNING_CONTAINER_OUTPUT_PATH in cmd

    def test_runs_from_clone_dir(self):
        cmd = build_vllm_tuning_command("test-model", 1)
        assert cmd.startswith("cd %s" % VLLM_CLONE_DIR)

    def test_uses_benchmark_moe_script(self):
        cmd = build_vllm_tuning_command("test-model", 1)
        assert "benchmarks/kernels/benchmark_moe.py" in cmd

    def test_various_tp_sizes(self):
        for tp in VLLM_DEFAULT_TP_SIZES:
            cmd = build_vllm_tuning_command("model", tp)
            assert "--tp-size %d" % tp in cmd

    def test_model_with_single_quote_is_shell_safe(self):
        """Model names with single quotes must not break shell quoting."""
        import shlex
        cmd = build_vllm_tuning_command("org/model'name", 1)
        assert shlex.quote("org/model'name") in cmd

    def test_model_with_spaces_is_shell_safe(self):
        """Model names with spaces must be shell-quoted."""
        import shlex
        cmd = build_vllm_tuning_command("org/model name", 1)
        assert shlex.quote("org/model name") in cmd


# ---------------------------------------------------------------------------
# vLLM CLI smoke tests
# ---------------------------------------------------------------------------

class TestTuneVllmCLI:
    def test_help(self):
        from sparkrun.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["tune", "vllm", "--help"])
        assert result.exit_code == 0
        assert "Tune vLLM fused MoE Triton kernels" in result.output
        assert "--tp" in result.output
        assert "--skip-clone" in result.output

    def test_tune_group_lists_vllm(self):
        from sparkrun.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["tune", "--help"])
        assert result.exit_code == 0
        assert "vllm" in result.output

    def test_rejects_non_vllm_recipe(self, tmp_path, monkeypatch):
        """tune vllm should reject recipes with non-vllm runtime."""
        from sparkrun.cli import main

        # Create an sglang recipe
        recipe_file = tmp_path / "test-sglang.yaml"
        recipe_file.write_text(yaml.dump({
            "sparkrun_version": "2",
            "name": "Test SGLang",
            "model": "test/model",
            "runtime": "sglang",
            "container": "test:latest",
        }))

        runner = CliRunner()
        result = runner.invoke(main, [
            "tune", "vllm", str(recipe_file), "-H", "10.0.0.1", "-n",
        ])
        assert result.exit_code != 0
        assert "requires a vLLM recipe" in result.output

    def test_accepts_vllm_ray_recipe(self, tmp_path, monkeypatch):
        """tune vllm should accept vllm-ray runtime recipes in dry-run."""
        from sparkrun.cli import main

        recipe_file = tmp_path / "test-vllm-ray.yaml"
        recipe_file.write_text(yaml.dump({
            "sparkrun_version": "2",
            "name": "Test vLLM Ray",
            "model": "test/model",
            "runtime": "vllm-ray",
            "container": "test:latest",
        }))

        runner = CliRunner()
        result = runner.invoke(main, [
            "tune", "vllm", str(recipe_file), "-H", "10.0.0.1", "-n",
        ])
        # Should not fail with runtime validation error
        assert "requires a vLLM recipe" not in (result.output or "")


# ---------------------------------------------------------------------------
# VllmTuner dry-run
# ---------------------------------------------------------------------------

class TestVllmTunerDryRun:
    def test_tuner_dry_run_returns_zero(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1,))
        assert rc == 0

    def test_tuner_dry_run_all_default_tp_sizes(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning()
        assert rc == 0

    def test_tuner_custom_output_dir(self, tmp_path):
        custom_dir = str(tmp_path / "custom_tuning")
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            output_dir=custom_dir,
            dry_run=True,
        )
        assert tuner.output_dir == custom_dir

    def test_tuner_skip_clone(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="test-model",
            skip_clone=True,
            dry_run=True,
        )
        assert tuner.skip_clone is True
        rc = tuner.run_tuning(tp_sizes=(1,))
        assert rc == 0

    def test_default_tp_sizes_constant(self):
        assert VLLM_DEFAULT_TP_SIZES == (1, 2, 4, 8)

    def test_tuner_dry_run_parallel(self):
        tuner = VllmTuner(
            host="10.0.0.1",
            image="test:latest",
            model="Qwen/Qwen3-MoE",
            dry_run=True,
        )
        rc = tuner.run_tuning(tp_sizes=(1, 2, 4), parallel=2)
        assert rc == 0

    def test_container_name_constant(self):
        assert TUNE_VLLM_CONTAINER_NAME == "sparkrun_tune_vllm"


# ---------------------------------------------------------------------------
# vLLM runtime auto-mount integration
# ---------------------------------------------------------------------------

class TestVllmRayRuntimeAutoMount:
    def test_get_extra_volumes_empty_when_no_configs(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_volumes returns {} when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: None,
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_volumes() == {}

    def test_get_extra_env_empty_when_no_configs(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_env returns base env when no tuning configs."""
        from sparkrun.core.bootstrap import get_runtime
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: None,
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface"}

    def test_get_extra_volumes_returns_mapping(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_volumes returns mapping when configs exist."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"/cache/tuning/vllm": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_volumes() == expected

    def test_get_extra_env_returns_env(self, v, monkeypatch):
        """VllmRayRuntime.get_extra_env returns env when configs exist."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"HF_HOME": "/cache/huggingface", "VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: {"VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH},
        )
        runtime = get_runtime("vllm-ray", v)
        assert runtime.get_extra_env() == expected


class TestVllmDistributedAutoMount:
    def test_get_extra_volumes_empty_when_no_configs(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_volumes returns {} when no configs."""
        from sparkrun.core.bootstrap import get_runtime
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: None,
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_volumes() == {}

    def test_get_extra_env_empty_when_no_configs(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_env returns base env when no configs."""
        from sparkrun.core.bootstrap import get_runtime
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: None,
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_env() == {"HF_HOME": "/cache/huggingface"}

    def test_get_extra_volumes_returns_mapping(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_volumes returns mapping when configs exist."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"/cache/tuning/vllm": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_volumes() == expected

    def test_get_extra_env_returns_env(self, v, monkeypatch):
        """VllmDistributedRuntime.get_extra_env returns env when configs exist."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"HF_HOME": "/cache/huggingface", "VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: {"VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH},
        )
        runtime = get_runtime("vllm-distributed", v)
        assert runtime.get_extra_env() == expected


class TestEugrVllmAutoMount:
    def test_inherits_vllm_ray_auto_mount(self, v, monkeypatch):
        """EugrVllmRayRuntime inherits get_extra_volumes from VllmRayRuntime."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"/cache/tuning/vllm": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_volumes",
            lambda: expected,
        )
        runtime = get_runtime("eugr-vllm", v)
        assert runtime.get_extra_volumes() == expected

    def test_inherits_vllm_ray_auto_env(self, v, monkeypatch):
        """EugrVllmRayRuntime inherits get_extra_env from VllmRayRuntime."""
        from sparkrun.core.bootstrap import get_runtime
        expected = {"HF_HOME": "/cache/huggingface", "VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH}
        monkeypatch.setattr(
            "sparkrun.tuning.vllm.get_vllm_tuning_env",
            lambda: {"VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH},
        )
        runtime = get_runtime("eugr-vllm", v)
        assert runtime.get_extra_env() == expected


# ===========================================================================
# Pre-check tests
# ===========================================================================

class TestPreCheckTp:
    """Test that _pre_check_tp is called and can skip tuning."""

    def test_base_pre_check_returns_false(self):
        """BaseTuner._pre_check_tp always returns False (tune anyway)."""
        from sparkrun.tuning._common import BaseTuner
        tuner = BaseTuner.__new__(BaseTuner)
        assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_sglang_pre_check_returns_false_in_dry_run(self):
        """SglangTuner._pre_check_tp returns False in dry-run mode."""
        tuner = SglangTuner(
            host="10.0.0.1", image="test:latest",
            model="test-model", dry_run=True,
        )
        assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_vllm_pre_check_returns_false_in_dry_run(self):
        """VllmTuner._pre_check_tp returns False in dry-run mode."""
        tuner = VllmTuner(
            host="10.0.0.1", image="test:latest",
            model="test-model", dry_run=True,
        )
        assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_sglang_pre_check_returns_false_on_error(self):
        """SglangTuner._pre_check_tp returns False on any exception."""
        from unittest.mock import patch
        tuner = SglangTuner(
            host="10.0.0.1", image="test:latest",
            model="test-model", dry_run=False,
        )
        with patch("sparkrun.orchestration.primitives.run_command_on_host",
                    side_effect=RuntimeError("connection refused")):
            assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_vllm_pre_check_returns_false_on_error(self):
        """VllmTuner._pre_check_tp returns False on any exception."""
        from unittest.mock import patch
        tuner = VllmTuner(
            host="10.0.0.1", image="test:latest",
            model="test-model", dry_run=False,
        )
        with patch("sparkrun.orchestration.primitives.run_command_on_host",
                    side_effect=RuntimeError("connection refused")):
            assert tuner._pre_check_tp(1, "3.6.0") is False

    def test_sglang_pre_check_returns_true_on_success(self):
        """SglangTuner._pre_check_tp returns True when command succeeds."""
        from unittest.mock import patch
        from sparkrun.orchestration.ssh import RemoteResult
        tuner = SglangTuner(
            host="10.0.0.1", image="test:latest",
            model="test-model", dry_run=False,
        )
        mock_result = RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="")
        with patch("sparkrun.orchestration.primitives.run_command_on_host",
                    return_value=mock_result):
            assert tuner._pre_check_tp(1, "3.6.0") is True

    def test_vllm_pre_check_returns_true_on_success(self):
        """VllmTuner._pre_check_tp returns True when command succeeds."""
        from unittest.mock import patch
        from sparkrun.orchestration.ssh import RemoteResult
        tuner = VllmTuner(
            host="10.0.0.1", image="test:latest",
            model="test-model", dry_run=False,
        )
        mock_result = RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="")
        with patch("sparkrun.orchestration.primitives.run_command_on_host",
                    return_value=mock_result):
            assert tuner._pre_check_tp(1, "3.6.0") is True

    def test_sglang_pre_check_returns_false_on_failure(self):
        """SglangTuner._pre_check_tp returns False when command fails."""
        from unittest.mock import patch
        from sparkrun.orchestration.ssh import RemoteResult
        tuner = SglangTuner(
            host="10.0.0.1", image="test:latest",
            model="test-model", dry_run=False,
        )
        mock_result = RemoteResult(host="10.0.0.1", returncode=1, stdout="", stderr="")
        with patch("sparkrun.orchestration.primitives.run_command_on_host",
                    return_value=mock_result):
            assert tuner._pre_check_tp(1, "3.6.0") is False


# ===========================================================================
# Sync path tests
# ===========================================================================

class TestSyncGetLocalTuningDir:
    """Test that _get_local_tuning_dir returns correct paths."""

    def test_sglang_path(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        d = _get_local_tuning_dir("sglang")
        assert str(d).endswith("sparkrun/tuning/sglang")

    def test_vllm_path(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        d = _get_local_tuning_dir("vllm")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_vllm_ray_maps_to_vllm(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        d = _get_local_tuning_dir("vllm-ray")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_vllm_distributed_maps_to_vllm(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        d = _get_local_tuning_dir("vllm-distributed")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_eugr_vllm_maps_to_vllm(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        d = _get_local_tuning_dir("eugr-vllm")
        assert str(d).endswith("sparkrun/tuning/vllm")

    def test_other_runtime_uses_tuning_prefix(self):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        d = _get_local_tuning_dir("llama-cpp")
        assert str(d).endswith("sparkrun/tuning/llama-cpp")


# ===========================================================================
# Mount-point constant tests
# ===========================================================================

class TestMountPointConstants:
    """Verify mount-point layout is correct for SGLang config lookup."""

    def test_sglang_container_path_ends_with_configs(self):
        """TUNING_CONTAINER_PATH must end with /configs so SGLang's internal
        $SGLANG_MOE_CONFIG_DIR/configs/triton_X_Y_Z/ resolves correctly."""
        assert TUNING_CONTAINER_PATH.endswith("/configs")

    def test_sglang_env_path_is_parent_of_container_path(self):
        """TUNING_ENV_PATH must be the parent of TUNING_CONTAINER_PATH."""
        assert TUNING_CONTAINER_PATH.startswith(TUNING_ENV_PATH)
        assert TUNING_CONTAINER_PATH == TUNING_ENV_PATH + "/configs"
