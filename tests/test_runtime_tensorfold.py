"""Unit tests for sparkrun.runtimes.tensorfold (upstream TensorFold runtime)."""

from types import SimpleNamespace

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.tensorfold import TensorFoldRuntime


def _recipe(defaults=None, **extra):
    data = {
        "name": "test-recipe",
        "model": "TensorFold/Qwen3.8-27B-MLX-4bit",
        "runtime": "tensorfold",
    }
    if defaults is not None:
        data["defaults"] = defaults
    data.update(extra)
    return Recipe.from_dict(data)


# --- identity / capabilities / image ---------------------------------------


def test_tensorfold_runtime_name_and_capability():
    runtime = TensorFoldRuntime()
    assert runtime.runtime_name == "tensorfold"
    assert runtime.requires_capability == frozenset({"gb10"})
    assert runtime.required_feature_flag == "runtime.tensorfold"
    assert runtime.default_executor() is None


def test_tensorfold_default_image_comes_from_platform():
    runtime = TensorFoldRuntime()
    assert runtime.resolve_container(_recipe()) == "ghcr.io/spark-arena/tensorfold:1.1.3"


def test_tensorfold_explicit_container_wins():
    runtime = TensorFoldRuntime()
    recipe = _recipe(container="example/tensorfold:custom")
    assert runtime.resolve_container(recipe) == "example/tensorfold:custom"


def test_tensorfold_native_apis_and_readiness():
    runtime = TensorFoldRuntime()
    assert runtime.native_api_options() == ["chat_completions", "responses", "messages"]
    assert runtime.readiness_health_path == "/health"
    assert runtime.readiness_styles


# --- solo command -----------------------------------------------------------


def test_solo_command_binds_network_and_emits_flags():
    runtime = TensorFoldRuntime()
    recipe = _recipe(
        defaults={
            "port": 8001,
            "max_model_len": 8192,
            "max_num_seqs": 4,
            "thinking": False,
            "vision": True,
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 20,
        }
    )
    command = runtime.generate_command(recipe, {}, is_cluster=False)

    assert command.startswith("tensorfold-native serve TensorFold/Qwen3.8-27B-MLX-4bit")
    assert "--host 0.0.0.0" in command
    assert "--port 8001" in command
    assert "--context 8192" in command
    assert "--parallel 4" in command
    assert "--no-thinking" in command
    assert "--thinking " not in command
    assert "--vision" in command
    assert "--temperature 0.0" in command
    assert "--top-p 0.9" in command
    assert "--top-k 20" in command
    assert "--no-update-check" in command
    # Solo has no rank placement.
    assert "--tp" not in command and "--rank" not in command and "--master" not in command


def test_thinking_true_emits_positive_flag():
    runtime = TensorFoldRuntime()
    command = runtime.generate_command(_recipe(defaults={"thinking": True}), {}, is_cluster=False)
    assert "--thinking" in command
    assert "--no-thinking" not in command


def test_served_name_alias_and_api_key():
    runtime = TensorFoldRuntime()
    recipe = _recipe(
        defaults={
            "served_model_name": "local-model",
            "alias": "qwen",
            "api_key": "sk-test",
        }
    )
    command = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--name local-model" in command
    assert "--alias qwen" in command
    assert "--api-key sk-test" in command
    assert runtime.resolve_api_key(recipe) == "sk-test"


def test_api_key_can_come_from_command_template():
    runtime = TensorFoldRuntime()
    recipe = _recipe(container="example/tf:1", command="tensorfold-native serve {model} --api-key sk-literal")
    assert runtime.resolve_api_key(recipe) == "sk-literal"


def test_command_template_is_honoured_and_name_is_augmented():
    runtime = TensorFoldRuntime()
    recipe = _recipe(
        command="tensorfold-native serve {model} --port {port}",
        defaults={"port": 8123, "served_model_name": "templated"},
    )
    command = runtime.generate_command(recipe, {}, is_cluster=False)
    assert command.startswith("tensorfold-native serve TensorFold/Qwen3.8-27B-MLX-4bit")
    assert "--port 8123" in command
    assert "--name templated" in command


def test_prepared_model_path_wins_over_recipe_model():
    runtime = TensorFoldRuntime()
    recipe = _recipe()
    command = runtime.generate_command(
        recipe,
        {"_prepared_model_path": "/cache/huggingface/hub/models--a--b/snapshots/abc"},
        is_cluster=False,
    )
    assert "/cache/huggingface/hub/models--a--b/snapshots/abc" in command


def test_nfs_absolute_model_path_is_served_directly():
    runtime = TensorFoldRuntime()
    recipe = _recipe(model="/mnt/nfs/models/qwen3.8-27b-mlx4")
    command = runtime.generate_command(recipe, {}, is_cluster=False)
    assert command.startswith("tensorfold-native serve /mnt/nfs/models/qwen3.8-27b-mlx4")
    assert "--host 0.0.0.0" in command


def test_command_binary_can_be_blank_for_entrypoint_images():
    runtime = TensorFoldRuntime()
    command = runtime.generate_command(_recipe(defaults={"command_binary": ""}), {}, is_cluster=False)
    assert command.startswith("serve TensorFold/Qwen3.8-27B-MLX-4bit")
    assert "command_binary" not in command


# --- placement / TP ---------------------------------------------------------


def test_world_size_is_one_solo():
    runtime = TensorFoldRuntime()
    parallelism = SimpleNamespace(tensor_parallel=2, pipeline_parallel=1, data_parallel=1)
    assert runtime.world_size(parallelism, recipe=_recipe(), cluster=None) == 1


def test_world_size_follows_tensor_parallel():
    runtime = TensorFoldRuntime()
    parallelism = SimpleNamespace(tensor_parallel=2, pipeline_parallel=1, data_parallel=1)
    recipe = _recipe(defaults={"tensor_parallel": 2})
    assert runtime.world_size(parallelism, recipe=recipe, cluster=None) == 2


def test_start_order_and_port_span_for_tp2():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2})
    assert runtime.native_start_order(recipe) == "workers_first"
    assert runtime.native_rendezvous_port(recipe, init_port=25000) == 25000
    assert runtime.native_rendezvous_port_span(recipe) == 1
    assert runtime.native_start_order(_recipe()) == "head_first"


def test_tp_rank_commands_use_upstream_flags_and_rank0_only_http():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "port": 8100, "max_model_len": 8192})
    head = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0, init_port=25000)
    worker = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1, init_port=25000)

    for command in (head, worker):
        assert "tensorfold-native serve TensorFold/Qwen3.8-27B-MLX-4bit" in command
        assert "--tp 2" in command
        assert "--master 10.0.0.1" in command
        assert "--master-port 25000" in command
        assert "--no-update-check" in command
    assert "--rank 0" in head and "--host 0.0.0.0" in head and "--port 8100" in head
    assert "--rank 1" in worker
    assert "--host" not in worker
    assert "--port" not in worker
    assert "--name" not in worker


def test_cluster_preview_is_a_rank0_shape():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "port": 8100})
    command = runtime.generate_command(recipe, {}, is_cluster=True)
    assert "tensorfold-native serve TensorFold/Qwen3.8-27B-MLX-4bit" in command
    assert "--host 0.0.0.0" in command
    assert "--port 8100" in command
    # Preview only: no rank placement is executed from generate_command.
    assert "--rank" not in command


# --- validation / prepare ---------------------------------------------------


def test_validation_accepts_tp2():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "min_nodes": 2}, max_nodes=2)
    issues = [issue for issue in runtime.validate_recipe(recipe) if getattr(issue, "severity", "") == "error"]
    assert issues == []


def test_validation_rejects_pp_dp_ep():
    runtime = TensorFoldRuntime()
    for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
        recipe = _recipe(defaults={"tensor_parallel": 2, key: 2})
        messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
        assert any(key in message for message in messages), key


def test_validation_rejects_command_template_for_tp():
    runtime = TensorFoldRuntime()
    recipe = _recipe(
        command="tensorfold-native serve {model} --port {port}",
        defaults={"tensor_parallel": 2},
    )
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("command: template" in message for message in messages)


def test_validation_rejects_removed_tf_family():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "tf_family": "glm"})
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("tf_family was removed" in message for message in messages)


def test_validation_rejects_min_nodes_above_tp():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2}, min_nodes=4)
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("below min_nodes" in message for message in messages)


def test_prepare_accepts_tp2():
    runtime = TensorFoldRuntime()
    runtime.prepare(_recipe(defaults={"tensor_parallel": 2}), ["h1", "h2"])


def test_prepare_requires_one_host_per_rank():
    runtime = TensorFoldRuntime()
    with pytest.raises(RuntimeError, match="one host per tensor_parallel"):
        runtime.prepare(_recipe(defaults={"tensor_parallel": 2}), ["h1"])


def test_prepare_rejects_removed_family_and_command_template():
    runtime = TensorFoldRuntime()
    with pytest.raises(RuntimeError, match="tf_family was removed"):
        runtime.prepare(_recipe(defaults={"tf_family": "dsv41"}), ["h1", "h2"])
    with pytest.raises(RuntimeError, match="command: template"):
        runtime.prepare(_recipe(defaults={"tensor_parallel": 2}, command="tensorfold-native serve {model}"), ["h1", "h2"])


# --- misc hooks -------------------------------------------------------------


def test_known_config_keys_cover_flag_map_and_owned_keys():
    keys = TensorFoldRuntime().known_config_keys()
    assert {"max_model_len", "command_binary", "served_model_name", "port"} <= keys
    assert "tf_family" not in keys
    assert "follower_binary" not in keys
    assert TensorFoldRuntime().serve_flag_map() == {
        "max_model_len": "--context",
        "max_num_seqs": "--parallel",
        "max_tokens": "--max-tokens",
        "temperature": "--temperature",
        "top_p": "--top-p",
        "top_k": "--top-k",
        "min_p": "--min-p",
        "thinking": "--thinking",
        "vision": "--vision",
        "no_drafts": "--no-drafts",
        "drafter": "--drafter",
        "drafter_bits": "--drafter-bits",
        "backend": "--backend",
        "prompt_cache_gib": "--prompt-cache-gib",
        "reasoning_effort": "--reasoning-effort",
        "loop_guard": "--loop-guard",
        "metrics_open": "--metrics-open",
        "dashboard": "--dashboard",
        "alias": "--alias",
        "chat_template": "--chat-template",
    }


def test_version_commands_report_engine_and_nccl():
    commands = TensorFoldRuntime().version_commands()
    assert "tensorfold-native --version" in commands["tensorfold"]
    assert "libnccl" in commands["nccl_lib"]
    assert "tensorfold_dsv41" not in commands


def test_extra_docker_opts_carry_memlock():
    opts = TensorFoldRuntime().get_extra_docker_opts()
    assert "--cap-add=IPC_LOCK" in opts
    assert "memlock=-1" in opts


def test_common_env_is_hf_offline_and_no_update_check():
    env = TensorFoldRuntime().get_common_env()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TENSORFOLD_NO_UPDATE_CHECK"] == "1"


def test_dashboard_and_metrics_are_rank0_only():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "dashboard": True, "metrics_open": True})
    head = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0)
    worker = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1)
    assert "--dashboard" in head and "--metrics-open" in head
    assert "--dashboard" not in worker and "--metrics-open" not in worker


def test_tensorfold_uses_the_native_cluster_strategy():
    assert TensorFoldRuntime().cluster_strategy() == "native"
