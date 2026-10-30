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
    # Without this an unreachable update check can stall a host with no internet.
    assert "--no-update-check" in command


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


def test_world_size_is_one_without_a_tp_family():
    runtime = TensorFoldRuntime()
    parallelism = SimpleNamespace(tensor_parallel=2, pipeline_parallel=1, data_parallel=1)
    assert runtime.world_size(parallelism, recipe=_recipe(defaults={"tensor_parallel": 2}), cluster=None) == 1


def test_world_size_follows_tensor_parallel_for_a_tp_family():
    runtime = TensorFoldRuntime()
    parallelism = SimpleNamespace(tensor_parallel=2, pipeline_parallel=1, data_parallel=1)
    recipe = _recipe(defaults={"tensor_parallel": 2, "tf_family": "glm"})
    assert runtime.world_size(parallelism, recipe=recipe, cluster=None) == 2


def test_start_order_and_port_span_for_glm_tp2():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "tf_family": "glm"})
    assert runtime.native_start_order(recipe) == "workers_first"
    assert runtime.native_rendezvous_port_span(recipe) == 2
    # A solo launch keeps the historical head-first order.
    assert runtime.native_start_order(_recipe()) == "head_first"
    assert runtime.native_rendezvous_port_span(_recipe()) == 1


def test_glm_rank_commands_use_family_env_and_rank0_only_http():
    runtime = TensorFoldRuntime()
    recipe = _recipe(
        defaults={
            "tensor_parallel": 2,
            "tf_family": "glm",
            "port": 8100,
            "max_model_len": 8192,
            "model_name": "ignored",
        },
    )
    head = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0, init_port=25000)
    worker = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1, init_port=25000)

    for command in (head, worker):
        assert "TF_GLM_RANK=" in command
        assert "TF_GLM_WORLD=2" in command
        assert "NCCL_COMM_ID=10.0.0.1:25000" in command
        assert "tensorfold-native serve TensorFold/Qwen3.8-27B-MLX-4bit" in command
        assert "--no-update-check" in command
    assert "TF_GLM_RANK=0" in head and "--host 0.0.0.0" in head and "--port 8100" in head
    assert "TF_GLM_RANK=1" in worker
    assert "--host" not in worker
    assert "--port" not in worker
    assert "--name" not in worker


def test_nemotron_rank_commands_use_its_own_env_vars():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "tf_family": "nemotron", "max_model_len": 16384})
    command = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.2", num_nodes=2, node_rank=1, init_port=25001)
    assert "TF_NEMOTRON_RANK=1" in command
    assert "TF_NEMOTRON_WORLD=2" in command
    assert "NCCL_COMM_ID=10.0.0.2:25001" in command


def test_generate_node_command_refuses_missing_family():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2})
    with pytest.raises(ValueError, match="tf_family"):
        runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0)


# --- validation / prepare ---------------------------------------------------


def test_validation_accepts_glm_tp2():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "tf_family": "glm", "min_nodes": 2}, max_nodes=2)
    issues = [issue for issue in runtime.validate_recipe(recipe) if getattr(issue, "severity", "") == "error"]
    assert issues == []


def test_validation_rejects_tp_without_family():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2})
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("tf_family" in message for message in messages)


def test_validation_rejects_pp_and_dp():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "tf_family": "glm", "pipeline_parallel": 2})
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("pipeline_parallel" in message for message in messages)


def test_validation_rejects_command_template_for_tp():
    runtime = TensorFoldRuntime()
    recipe = _recipe(
        command="tensorfold-native serve {model} --port {port}",
        defaults={"tensor_parallel": 2, "tf_family": "glm"},
    )
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("command: template" in message for message in messages)


def test_validation_warns_family_solo():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tf_family": "glm"})
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("ignored solo" in message for message in messages)


def test_prepare_rejects_cli_only_tp_without_family():
    runtime = TensorFoldRuntime()
    recipe = _recipe()
    with pytest.raises(RuntimeError, match="tf_family"):
        runtime.prepare(recipe, ["h1", "h2"], overrides={"tensor_parallel": 2})


def test_prepare_accepts_cli_only_tp_with_family():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tf_family": "glm"})
    runtime.prepare(recipe, ["h1", "h2"], overrides={"tensor_parallel": 2})


# --- misc hooks -------------------------------------------------------------


def test_known_config_keys_cover_flag_map_and_owned_keys():
    keys = TensorFoldRuntime().known_config_keys()
    assert {"max_model_len", "tf_family", "command_binary", "served_model_name", "port"} <= keys
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


def test_extra_docker_opts_carry_memlock():
    opts = TensorFoldRuntime().get_extra_docker_opts()
    assert "--cap-add=IPC_LOCK" in opts
    assert "memlock=-1" in opts


def test_common_env_is_hf_offline_and_no_update_check():
    env = TensorFoldRuntime().get_common_env()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TENSORFOLD_NO_UPDATE_CHECK"] == "1"


# --- DeepSeek-V4.1 fork family (same runtime, dsv41 family) ------------------

_DSV41_BUNDLE = "/mnt/nfs/sparkrun/tensorfold-dsv41-flash"
_DSV41_IMAGE = "ghcr.io/example/tensorfold-dsv41:3ff3b2a"


def _dsv41_recipe(defaults=None, **extra):
    data = {
        "name": "dsv41-tp2",
        "model": _DSV41_BUNDLE,
        "runtime": "tensorfold",
        "container": _DSV41_IMAGE,
        "min_nodes": 2,
        "max_nodes": 2,
    }
    base_defaults = {
        "tf_family": "dsv41",
        "tensor_parallel": 2,
        "port": 8000,
        "max_model_len": 1048576,
        "max_num_seqs": 4,
        "streams": 4,
        "pool": 1048576,
        "served_model_name": "deepseek-v4.1-flash",
        "thinking": False,
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "rdma_hca": "rocep1s0f0,roceP2p1s0f0",
        "net_if": "enp1s0f0np0",
    }
    if defaults:
        base_defaults.update(defaults)
    data["defaults"] = base_defaults
    data.update(extra)
    return Recipe.from_dict(data)


def test_dsv41_is_a_family_of_the_single_runtime():
    runtime = TensorFoldRuntime()
    assert runtime.runtime_name == "tensorfold"
    assert runtime.requires_capability == frozenset({"gb10"})
    # The explicit container is required for this family; no platform default.
    assert runtime.resolve_container(_dsv41_recipe()) == _DSV41_IMAGE


def test_dsv41_world_size_and_start_order():
    runtime = TensorFoldRuntime()
    parallelism = SimpleNamespace(tensor_parallel=2, pipeline_parallel=1, data_parallel=1)
    recipe = _dsv41_recipe()
    assert runtime.world_size(parallelism, recipe=recipe, cluster=None) == 2
    assert runtime.native_start_order(recipe) == "workers_first"
    assert runtime.native_rendezvous_port_span(recipe) == 2
    tp4 = _dsv41_recipe(defaults={"tensor_parallel": 4}, min_nodes=4, max_nodes=4)
    assert runtime.native_start_order(tp4) == "workers_first"


def test_dsv41_rank0_command_shape():
    runtime = TensorFoldRuntime()
    command = runtime.generate_node_command(_dsv41_recipe(), {}, head_ip="192.168.192.2", num_nodes=2, node_rank=0, init_port=29620)
    assert "/opt/tensorfold-dsv41/bin/tensorfold-native serve" in command
    assert "%s/model" % _DSV41_BUNDLE in command
    assert "--host 0.0.0.0" in command
    assert "--port 8000" in command
    assert "--name deepseek-v4.1-flash" in command
    assert "--context 1048576" in command
    assert "--parallel 4" in command
    assert "--no-thinking" in command
    assert "--no-update-check" in command
    assert "--backend cuda" in command
    assert "TF_TP_RANK=0" in command
    assert "TF_TP_WORLD=2" in command
    assert "TF_TP_MASTER=192.168.192.2" in command
    assert "TF_TP_PORT=29620" in command
    assert "TF_DS_RANK_CACHE=%s/cache/rank-cache" % _DSV41_BUNDLE in command
    assert "TF_DS_KIT=%s/kit" % _DSV41_BUNDLE in command
    assert "TF_DS_ENGRAM=%s/engram" % _DSV41_BUNDLE in command
    assert "TF_DS_TOKEN_MAP=%s/cache/dsv41_token_map.json" % _DSV41_BUNDLE in command
    assert "TF_RDMA_DEVICES=rocep1s0f0,roceP2p1s0f0" in command
    assert "NCCL_SOCKET_IFNAME=enp1s0f0np0" in command


def test_dsv41_follower_command_is_a_different_binary():
    runtime = TensorFoldRuntime()
    command = runtime.generate_node_command(_dsv41_recipe(), {}, head_ip="192.168.192.2", num_nodes=2, node_rank=1, init_port=29620)
    assert "/opt/tensorfold-dsv41/bin/tf-dsv41-lanes" in command
    assert "/opt/tensorfold-dsv41/bin/tensorfold-native serve" not in command
    assert "%s/model" % _DSV41_BUNDLE in command
    assert "%s/cache/rank-cache" % _DSV41_BUNDLE in command
    assert " 1 2 192.168.192.2 29620 " in command
    assert "--engram %s/engram" % _DSV41_BUNDLE in command
    assert "--token-map %s/cache/dsv41_token_map.json" % _DSV41_BUNDLE in command
    assert "--pool 1048576" in command
    assert "--streams 4" in command
    assert "--graphs 1" in command
    assert "--rdma rocep1s0f0,roceP2p1s0f0" in command
    assert "--host" not in command
    assert "--port 8000" not in command


def test_dsv41_tp4_adds_2d_split_env():
    runtime = TensorFoldRuntime()
    tp4 = _dsv41_recipe(defaults={"tensor_parallel": 4}, min_nodes=4, max_nodes=4)
    command = runtime.generate_node_command(tp4, {}, head_ip="192.168.192.2", num_nodes=4, node_rank=2, init_port=29620)
    assert "TF_DS_2D_GU=parity" in command
    assert "TF_DS_WINDOW=2097152" in command
    assert "TF_DS_ROUND_ROWS=48" in command
    assert "TF_DS_SERVED_K=solo" in command
    assert " 2 4 192.168.192.2 29620 " in command


def test_dsv41_validation_accepts_prepared_bundle_and_container():
    runtime = TensorFoldRuntime()
    messages = [str(issue) for issue in runtime.validate_recipe(_dsv41_recipe())]
    assert not any("prepared bundle" in message or "TP2/TP4" in message or "explicit container" in message for message in messages)


def test_dsv41_validation_rejects_single_rank():
    runtime = TensorFoldRuntime()
    recipe = _dsv41_recipe(defaults={"tensor_parallel": 1})
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("TP2/TP4" in message for message in messages)


def test_dsv41_validation_rejects_relative_model_path():
    runtime = TensorFoldRuntime()
    recipe = _dsv41_recipe(model="LibertAIDAI/DeepSeek-V4.1-Flash")
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("absolute model" in message for message in messages)


def test_dsv41_validation_rejects_missing_explicit_container():
    runtime = TensorFoldRuntime()
    data = {
        "name": "dsv41",
        "model": _DSV41_BUNDLE,
        "runtime": "tensorfold",
        "min_nodes": 2,
        "max_nodes": 2,
        "defaults": {"tf_family": "dsv41", "tensor_parallel": 2},
    }
    messages = [str(issue) for issue in runtime.validate_recipe(Recipe.from_dict(data))]
    assert any("explicit container" in message for message in messages)


def test_dsv41_validation_rejects_command_template():
    runtime = TensorFoldRuntime()
    recipe = _dsv41_recipe(command="tensorfold-native serve {model}")
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("command: template" in message for message in messages)


def test_dsv41_prepare_rejects_wrong_host_count():
    runtime = TensorFoldRuntime()
    with pytest.raises(RuntimeError, match="exactly tensor_parallel hosts"):
        runtime.prepare(_dsv41_recipe(), ["h1"])


def test_dsv41_prepare_accepts_two_hosts():
    runtime = TensorFoldRuntime()
    runtime.prepare(_dsv41_recipe(), ["h1", "h2"])


def test_dsv41_solo_generate_command_is_refused():
    runtime = TensorFoldRuntime()
    with pytest.raises(ValueError, match="multi-node only"):
        runtime.generate_command(_dsv41_recipe(), {}, is_cluster=False)


def test_dsv41_custom_bundle_subdirs():
    runtime = TensorFoldRuntime()
    recipe = _dsv41_recipe(
        defaults={
            "cache_dir": "/shared/rank-cache",
            "kit_dir": "/shared/kit",
            "engram_dir": "/shared/engram",
            "token_map": "/shared/token_map.json",
        }
    )
    command = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1)
    assert "/shared/rank-cache" in command
    assert "/shared/kit" in command
    assert "/shared/engram" in command
    assert "/shared/token_map.json" in command


def test_dsv41_version_commands_report_both_binaries():
    commands = TensorFoldRuntime().version_commands()
    assert "tensorfold-dsv41/bin/tensorfold-native" in commands["tensorfold_dsv41"]
    assert "tf-dsv41-lanes" in commands["tensorfold_dsv41_lanes"]


def test_dsv41_generate_command_for_cluster_is_a_rank0_preview():
    runtime = TensorFoldRuntime()
    command = runtime.generate_command(_dsv41_recipe(), {}, is_cluster=True)
    assert "/opt/tensorfold-dsv41/bin/tensorfold-native serve" in command
    assert "%s/model" % _DSV41_BUNDLE in command


def test_tensorfold_uses_the_native_cluster_strategy():
    assert TensorFoldRuntime().cluster_strategy() == "native"


def test_dashboard_and_metrics_are_rank0_only():
    runtime = TensorFoldRuntime()
    recipe = _recipe(defaults={"tf_family": "glm", "tensor_parallel": 2, "dashboard": True, "metrics_open": True})
    head = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0)
    worker = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1)
    assert "--dashboard" in head
    assert "--metrics-open" in head
    assert "--dashboard" not in worker
    assert "--metrics-open" not in worker


def test_dsv41_dashboard_is_rank0_only():
    runtime = TensorFoldRuntime()
    recipe = _dsv41_recipe(defaults={"dashboard": True, "metrics_open": True})
    head = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0)
    follower = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1)
    assert "--dashboard" in head
    assert "--metrics-open" in head
    assert "--dashboard" not in follower
    assert "--metrics-open" not in follower


# --- Jayleaton Zig engine family -------------------------------------------

_JAY_MODEL = "marker/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw-Abliterated"
_JAY_PATH = "/home/marker/.cache/huggingface/jayleaton-flash17-ablit"
_JAY_IMAGE = "tensorfold-dsv41-zig:local"


def _jayleaton_recipe(defaults=None, **extra):
    data = {
        "name": "jayleaton-tp2",
        "model": _JAY_MODEL,
        "runtime": "tensorfold",
        "container": _JAY_IMAGE,
        "min_nodes": 2,
        "max_nodes": 2,
        "cluster_config": {"resolved_model_path": _JAY_PATH},
    }
    base_defaults = {
        "tf_family": "jayleaton",
        "tensor_parallel": 2,
        "port": 8000,
        "served_model_name": "deepseek-v4.1-flash",
        "max_model_len": 1048576,
        "max_num_seqs": 4,
        "max_tokens": 32768,
        "assets_dir": "/assets",
        "engram_dir": "/engram",
    }
    if defaults:
        base_defaults.update(defaults)
    data["defaults"] = base_defaults
    data.update(extra)
    return Recipe.from_dict(data)


def test_jayleaton_is_a_multi_node_family():
    runtime = TensorFoldRuntime()
    recipe = _jayleaton_recipe()
    parallelism = SimpleNamespace(tensor_parallel=2, pipeline_parallel=1, data_parallel=1)
    assert runtime.world_size(parallelism, recipe=recipe, cluster=None) == 2
    assert runtime.native_start_order(recipe) == "workers_first"
    assert runtime.native_rendezvous_port_span(recipe) == 1
    assert runtime.native_rendezvous_port(recipe, None) == 29571


def test_jayleaton_rank_commands():
    runtime = TensorFoldRuntime()
    head = runtime.generate_node_command(_jayleaton_recipe(), {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0, init_port=29571)
    worker = runtime.generate_node_command(_jayleaton_recipe(), {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1, init_port=29571)
    assert "/opt/tensorfold/bin/tensorfold-dsv41 serve %s" % _JAY_PATH in head
    assert "--host 0.0.0.0" in head
    assert "--port 8000" in head
    assert "--name deepseek-v4.1-flash" in head
    assert "TF_TP_RANK=0" in head
    assert "TF_TP_WORLD=2" in head
    assert "TF_TP_MASTER=10.0.0.1" in head
    assert "TF_DSV41_ASSETS=/assets" in head
    assert "TF_DSV41_ENGRAM_DIR=/engram" in head
    assert "/opt/tensorfold/bin/tf-dsv41-m1 follow %s /assets" % _JAY_PATH in worker
    assert "TF_TP_RANK=1" in worker
    assert "--port 8000" not in worker


def test_jayleaton_cluster_preview_and_solo_refusal():
    runtime = TensorFoldRuntime()
    preview = runtime.generate_command(_jayleaton_recipe(), {}, is_cluster=True)
    assert "/opt/tensorfold/bin/tensorfold-dsv41 serve %s" % _JAY_PATH in preview
    with pytest.raises(ValueError, match="multi-node only"):
        runtime.generate_command(_jayleaton_recipe(), {}, is_cluster=False)


def test_jayleaton_validation_and_prepare():
    runtime = TensorFoldRuntime()
    recipe = _jayleaton_recipe()
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert not any("TP=2" in m or "flash17" in m or "explicit container" in m for m in messages)
    runtime.prepare(recipe, ["h1", "h2"])
    with pytest.raises(RuntimeError, match="exactly two hosts"):
        runtime.prepare(recipe, ["h1"])


def test_jayleaton_requires_explicit_image_and_two_ranks():
    runtime = TensorFoldRuntime()
    data = {
        "name": "jay-bad",
        "model": _JAY_MODEL,
        "runtime": "tensorfold",
        "min_nodes": 2,
        "max_nodes": 2,
        "cluster_config": {"resolved_model_path": _JAY_PATH},
        "defaults": {"tf_family": "jayleaton", "tensor_parallel": 4},
    }
    messages = [str(issue) for issue in runtime.validate_recipe(Recipe.from_dict(data))]
    assert any("TP=2" in message for message in messages)
    assert any("explicit container" in message for message in messages)
