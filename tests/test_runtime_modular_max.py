"""Unit tests for sparkrun.runtimes.modular_max (ModularMaxRuntime)."""

from types import SimpleNamespace
from unittest import mock

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.modular_max import ModularMaxRuntime


def _recipe(defaults=None, **extra):
    data = {
        "name": "test-recipe",
        "model": "google/gemma-3-27b-it",
        "runtime": "modular-max",
    }
    if defaults is not None:
        data["defaults"] = defaults
    data.update(extra)
    return Recipe.from_dict(data)


# --- identity / container ---


def test_modular_max_runtime_name():
    assert ModularMaxRuntime().runtime_name == "modular-max"


def test_modular_max_default_executor_is_none():
    # None -> falls through to the global default (docker).
    assert ModularMaxRuntime().default_executor() is None


def test_modular_max_resolve_container_default():
    runtime = ModularMaxRuntime()
    assert runtime.resolve_container(_recipe()) == "modular/max-nvidia-full:latest"


def test_modular_max_resolve_container_override():
    runtime = ModularMaxRuntime()
    recipe = _recipe(container="modular/max-nvidia-base:25.6")
    assert runtime.resolve_container(recipe) == "modular/max-nvidia-base:25.6"


# --- placement: single node always ---


def test_modular_max_world_size_is_one_even_with_tp():
    runtime = ModularMaxRuntime()
    parallelism = SimpleNamespace(tensor_parallel=4, total_gpus=4)
    ws = runtime.world_size(parallelism, recipe=_recipe(defaults={"tensor_parallel": 4}), cluster=None)
    assert ws == 1


# --- command construction ---


def test_modular_max_generate_command_solo_basics():
    runtime = ModularMaxRuntime()
    recipe = _recipe(defaults={"port": 8000, "max_model_len": 8192, "quantization": "bfloat16"})

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)

    assert cmd.startswith("max serve --model google/gemma-3-27b-it")
    assert "--port 8000" in cmd
    assert "--max-length 8192" in cmd
    assert "--quantization-encoding bfloat16" in cmd
    # vLLM/SGLang-only flags must NOT leak in.
    assert "--gpu-memory-utilization" not in cmd
    assert "--tensor-parallel" not in cmd


def test_modular_max_default_port_when_unset():
    runtime = ModularMaxRuntime()
    cmd = runtime.generate_command(_recipe(), {}, is_cluster=False)
    assert "--port 8000" in cmd


def test_modular_max_tp1_omits_devices():
    runtime = ModularMaxRuntime()
    cmd = runtime.generate_command(_recipe(defaults={"tensor_parallel": 1}), {}, is_cluster=False)
    assert "--devices" not in cmd


def test_modular_max_tp_expands_to_local_devices():
    runtime = ModularMaxRuntime()
    cmd = runtime.generate_command(_recipe(defaults={"tensor_parallel": 4}), {}, is_cluster=False)
    assert "--devices gpu:0,1,2,3" in cmd


def test_modular_max_explicit_devices_wins_over_tp():
    runtime = ModularMaxRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 4, "devices": "gpu:all"})
    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--devices gpu:all" in cmd
    assert "gpu:0,1,2,3" not in cmd


def test_modular_max_command_template_honored_and_served_name_augmented():
    runtime = ModularMaxRuntime()
    recipe = _recipe(
        defaults={"port": 9001, "served_model_name": "gemma"},
        command="max serve --model {model} --port {port}",
    )
    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "max serve --model google/gemma-3-27b-it --port 9001" in cmd
    # served_model_name not in the template -> appended by the base helper.
    assert "--served-model-name gemma" in cmd


def test_modular_max_skip_keys_suppresses_served_model_name():
    runtime = ModularMaxRuntime()
    recipe = _recipe(defaults={"served_model_name": "gemma"})
    cmd = runtime.generate_command(recipe, {}, is_cluster=False, skip_keys={"served_model_name"})
    assert "--served-model-name" not in cmd


# --- validation ---


def test_modular_max_validate_rejects_multinode():
    runtime = ModularMaxRuntime()
    recipe = _recipe(min_nodes=2)
    issues = runtime.validate_recipe(recipe)
    assert any("single-node" in i for i in issues)


def test_modular_max_validate_solo_ok():
    runtime = ModularMaxRuntime()
    assert runtime.validate_recipe(_recipe()) == []


# --- prepare() GPU-count guard ---


def test_modular_max_prepare_dry_run_skips_probe():
    runtime = ModularMaxRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2})
    with mock.patch("sparkrun.core.hardware_probe.probe_hosts") as probe:
        runtime.prepare(recipe, ["10.0.0.1"], dry_run=True)
    probe.assert_not_called()


def test_modular_max_prepare_tp1_skips_probe():
    runtime = ModularMaxRuntime()
    with mock.patch("sparkrun.core.hardware_probe.probe_hosts") as probe:
        runtime.prepare(_recipe(defaults={"tensor_parallel": 1}), ["10.0.0.1"])
    probe.assert_not_called()


def test_modular_max_prepare_rejects_tp_on_single_gpu_host():
    runtime = ModularMaxRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2})
    fake = {"10.0.0.1": SimpleNamespace(total_gpus=1)}
    with mock.patch("sparkrun.core.hardware_probe.probe_hosts", return_value=fake):
        with pytest.raises(RuntimeError, match="single-node"):
            runtime.prepare(recipe, ["10.0.0.1"])


def test_modular_max_prepare_allows_tp_on_multi_gpu_host():
    runtime = ModularMaxRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 4})
    fake = {"10.0.0.1": SimpleNamespace(total_gpus=8)}
    with mock.patch("sparkrun.core.hardware_probe.probe_hosts", return_value=fake):
        # Should not raise.
        runtime.prepare(recipe, ["10.0.0.1"])
