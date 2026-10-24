"""Unit tests for sparkrun.runtimes.tensorfold_dsv41 (DeepSeek-V4.1 TP2/TP4 lane)."""

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.tensorfold_dsv41 import TensorFoldDSV41Runtime

BUNDLE = "/mnt/nfs/sparkrun/tensorfold-dsv41-flash"


def _recipe(defaults=None, **extra):
    data = {
        "name": "dsv41-tp2",
        "model": BUNDLE,
        "runtime": "tensorfold-dsv41",
        "min_nodes": 2,
        "max_nodes": 2,
    }
    base_defaults = {
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


def test_dsv41_identity_and_image():
    runtime = TensorFoldDSV41Runtime()
    assert runtime.runtime_name == "tensorfold-dsv41"
    assert runtime.requires_capability == frozenset({"gb10"})
    assert runtime.resolve_container(_recipe()) == "ghcr.io/spark-arena/tensorfold-dsv41:3ff3b2a"


def test_dsv41_world_size_and_start_order():
    runtime = TensorFoldDSV41Runtime()
    import types

    parallelism = types.SimpleNamespace(tensor_parallel=2)
    assert runtime.world_size(parallelism, recipe=_recipe(), cluster=None) == 2
    assert runtime.native_start_order(_recipe()) == "workers_first"
    assert runtime.native_rendezvous_port_span(_recipe()) == 2

    tp4 = _recipe(defaults={"tensor_parallel": 4}, min_nodes=4, max_nodes=4)
    assert runtime.native_start_order(tp4) == "workers_first"


def test_dsv41_rank0_command_shape():
    runtime = TensorFoldDSV41Runtime()
    command = runtime.generate_node_command(_recipe(), {}, head_ip="192.168.192.2", num_nodes=2, node_rank=0, init_port=29620)

    # rank 0 serve process
    assert "/opt/tensorfold-dsv41/bin/tensorfold-native serve" in command
    assert "%s/model" % BUNDLE in command
    assert "--host 0.0.0.0" in command
    assert "--port 8000" in command
    assert "--name deepseek-v4.1-flash" in command
    assert "--context 1048576" in command
    assert "--parallel 4" in command
    assert "--no-thinking" in command
    assert "--max-tokens 32768" in command
    assert "--no-update-check" in command
    assert "--backend cuda" in command
    # rank environment
    assert "TF_TP_RANK=0" in command
    assert "TF_TP_WORLD=2" in command
    assert "TF_TP_MASTER=192.168.192.2" in command
    assert "TF_TP_PORT=29620" in command
    assert "TF_DS_RANK_CACHE=%s/cache/rank-cache" % BUNDLE in command
    assert "TF_DS_KIT=%s/kit" % BUNDLE in command
    assert "TF_DS_ENGRAM=%s/engram" % BUNDLE in command
    assert "TF_DS_TOKEN_MAP=%s/cache/dsv41_token_map.json" % BUNDLE in command
    assert "TF_RDMA_DEVICES=rocep1s0f0,roceP2p1s0f0" in command
    assert "NCCL_SOCKET_IFNAME=enp1s0f0np0" in command


def test_dsv41_follower_command_is_a_different_binary():
    runtime = TensorFoldDSV41Runtime()
    command = runtime.generate_node_command(_recipe(), {}, head_ip="192.168.192.2", num_nodes=2, node_rank=1, init_port=29620)
    assert "/opt/tensorfold-dsv41/bin/tf-dsv41-lanes" in command
    assert "/opt/tensorfold-dsv41/bin/tensorfold-native serve" not in command
    assert "%s/model" % BUNDLE in command
    assert "%s/cache/rank-cache" % BUNDLE in command
    # positional rank/world/master/port/kit
    assert " 1 2 192.168.192.2 29620 " in command
    assert "--engram %s/engram" % BUNDLE in command
    assert "--token-map %s/cache/dsv41_token_map.json" % BUNDLE in command
    assert "--pool 1048576" in command
    assert "--streams 4" in command
    assert "--graphs 1" in command
    assert "--rdma rocep1s0f0,roceP2p1s0f0" in command
    # No HTTP binds on the follower.
    assert "--host" not in command
    assert "--port 8000" not in command


def test_dsv41_tp4_adds_2d_split_env():
    runtime = TensorFoldDSV41Runtime()
    tp4 = _recipe(defaults={"tensor_parallel": 4}, min_nodes=4, max_nodes=4)
    command = runtime.generate_node_command(tp4, {}, head_ip="192.168.192.2", num_nodes=4, node_rank=2, init_port=29620)
    assert "TF_DS_2D_GU=parity" in command
    assert "TF_DS_WINDOW=2097152" in command
    assert "TF_DS_ROUND_ROWS=48" in command
    assert " 2 4 192.168.192.2 29620 " in command


def test_dsv41_served_name_and_sampling_flags():
    runtime = TensorFoldDSV41Runtime()
    command = runtime.generate_node_command(_recipe(), {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0)
    assert "--temperature 1.0" in command
    assert "--top-p 1.0" in command
    assert "--top-k 20" in command


def test_dsv41_validation_accepts_prepared_bundle_tp2():
    runtime = TensorFoldDSV41Runtime()
    messages = [str(issue) for issue in runtime.validate_recipe(_recipe())]
    # No blocking error about the bundle or the world.
    assert not any("prepared bundle" in message or "TP2/TP4" in message for message in messages)


def test_dsv41_validation_rejects_single_rank():
    runtime = TensorFoldDSV41Runtime()
    recipe = _recipe(defaults={"tensor_parallel": 1})
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("TP2/TP4" in message for message in messages)


def test_dsv41_validation_rejects_relative_model_path():
    runtime = TensorFoldDSV41Runtime()
    recipe = _recipe(model="LibertAIDAI/DeepSeek-V4.1-Flash")
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("absolute model" in message for message in messages)


def test_dsv41_validation_rejects_command_template():
    runtime = TensorFoldDSV41Runtime()
    recipe = _recipe(command="tensorfold-native serve {model}")
    messages = [str(issue) for issue in runtime.validate_recipe(recipe)]
    assert any("command: template" in message for message in messages)


def test_dsv41_prepare_rejects_wrong_host_count():
    runtime = TensorFoldDSV41Runtime()
    with pytest.raises(RuntimeError, match="exactly tensor_parallel hosts"):
        runtime.prepare(_recipe(), ["h1"])


def test_dsv41_prepare_accepts_two_hosts():
    runtime = TensorFoldDSV41Runtime()
    runtime.prepare(_recipe(), ["h1", "h2"])


def test_dsv41_solo_generate_command_is_refused():
    runtime = TensorFoldDSV41Runtime()
    with pytest.raises(ValueError, match="multi-node"):
        runtime.generate_command(_recipe(), {}, is_cluster=False)


def test_dsv41_custom_bundle_subdirs():
    runtime = TensorFoldDSV41Runtime()
    recipe = _recipe(
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
    commands = TensorFoldDSV41Runtime().version_commands()
    assert "tensorfold-dsv41/bin/tensorfold-native" in commands["tensorfold_dsv41"]
    assert "tf-dsv41-lanes" in commands["tensorfold_dsv41_lanes"]
