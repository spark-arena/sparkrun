"""Unit tests for sparkrun.runtimes.trtllm module."""

from __future__ import annotations

from unittest import mock

import yaml

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.trtllm import TrtllmRuntime


# --- Basic properties ---

def test_runtime_name():
    """TrtllmRuntime.runtime_name == 'trtllm'."""
    runtime = TrtllmRuntime()
    assert runtime.runtime_name == "trtllm"


def test_cluster_strategy():
    """TRT-LLM uses native clustering."""
    runtime = TrtllmRuntime()
    assert runtime.cluster_strategy() == "native"


def test_resolve_container_from_recipe():
    """Recipe with container field uses it directly."""
    recipe = Recipe.from_dict({
        "name": "test", "model": "nvidia/model",
        "runtime": "trtllm",
        "container": "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6",
    })
    runtime = TrtllmRuntime()
    assert runtime.resolve_container(recipe) == "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6"


def test_resolve_container_default():
    """Recipe without container falls back to default prefix."""
    recipe = Recipe.from_dict({
        "name": "test", "model": "nvidia/model", "runtime": "trtllm",
    })
    runtime = TrtllmRuntime()
    assert runtime.resolve_container(recipe) == "nvcr.io/nvidia/tensorrt-llm/release:latest"


# --- Command generation ---

def test_generate_command_structured():
    """Generates trtllm-serve command with flags from defaults."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/Qwen3-235B-A22B-FP4",
        "runtime": "trtllm",
        "defaults": {
            "port": 8355,
            "tensor_parallel": 2,
            "backend": "pytorch",
            "max_num_tokens": 32768,
        },
    })
    runtime = TrtllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("trtllm-serve nvidia/Qwen3-235B-A22B-FP4")
    assert "--tp_size 2" in cmd
    assert "--port 8355" in cmd
    assert "--backend pytorch" in cmd
    assert "--max_num_tokens 32768" in cmd


def test_generate_command_default_backend():
    """When no backend specified, defaults to pytorch."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "defaults": {"port": 8000},
    })
    runtime = TrtllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--backend pytorch" in cmd


def test_generate_command_template():
    """Recipe with command template renders it."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "command": "trtllm-serve {model} --port {port} --tp_size {tensor_parallel}",
        "defaults": {"port": 8355, "tensor_parallel": 2},
    })
    runtime = TrtllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "trtllm-serve nvidia/model --port 8355 --tp_size 2"


def test_generate_command_overrides():
    """CLI overrides take precedence over recipe defaults."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "defaults": {"port": 8355, "tensor_parallel": 2},
    })
    runtime = TrtllmRuntime()

    cmd = runtime.generate_command(recipe, {"port": 9000}, is_cluster=False)
    assert "--port 9000" in cmd
    assert "--port 8355" not in cmd


def test_generate_command_bool_flags():
    """Boolean flags are handled correctly."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "defaults": {"trust_remote_code": True},
    })
    runtime = TrtllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--trust_remote_code" in cmd


def test_generate_command_skip_keys():
    """skip_keys omits specified flags from generated command."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "defaults": {"port": 8355, "tensor_parallel": 2},
    })
    runtime = TrtllmRuntime()

    cmd = runtime.generate_command(
        recipe, {}, is_cluster=False, skip_keys={"port"},
    )
    assert "--port" not in cmd
    assert "--tp_size 2" in cmd


# --- mpirun command ---

def test_build_mpirun_command():
    """Verify mpirun wrapping with rsh agent, host list, env passthrough."""
    runtime = TrtllmRuntime()

    cmd = runtime._build_mpirun_command(
        "trtllm-serve nvidia/model --tp_size 2 --backend pytorch",
        host_ips=["192.168.1.10", "192.168.1.11"],
        nccl_env={"NCCL_SOCKET_IFNAME": "ibp65s0"},
    )

    assert cmd.startswith("mpirun --allow-run-as-root")
    assert "--mca plm_rsh_agent /tmp/sparkrun-rsh-wrapper.sh" in cmd
    assert "--mca rmaps_ppr_n_pernode 1" in cmd
    assert "-H 192.168.1.10,192.168.1.11" in cmd
    assert "-x NCCL_SOCKET_IFNAME" in cmd
    assert "-x HF_TOKEN" in cmd
    assert "trtllm-llmapi-launch trtllm-serve nvidia/model" in cmd


def test_build_mpirun_command_no_nccl():
    """mpirun command works without NCCL env vars."""
    runtime = TrtllmRuntime()

    cmd = runtime._build_mpirun_command(
        "trtllm-serve nvidia/model",
        host_ips=["10.0.0.1"],
        nccl_env=None,
    )

    assert "mpirun" in cmd
    assert "-H 10.0.0.1" in cmd
    # Should still include default propagated keys
    assert "-x HF_TOKEN" in cmd
    assert "trtllm-llmapi-launch" in cmd


# --- rsh wrapper ---

def test_generate_rsh_wrapper():
    """Verify case mapping, SSH key path, docker exec invocation."""
    host_ip_map = {
        "192.168.1.10": "sparkrun0_node_0",
        "192.168.1.11": "sparkrun0_node_1",
    }
    wrapper = TrtllmRuntime._generate_rsh_wrapper(
        host_ip_map, "sparkrun0",
        ssh_key_path="/tmp/.ssh/id_ed25519",
    )

    assert "#!/bin/bash" in wrapper
    assert "HOST=$1; shift" in wrapper
    assert 'case $HOST in' in wrapper
    assert '192.168.1.10) CONTAINER="sparkrun0_node_0"' in wrapper
    assert '192.168.1.11) CONTAINER="sparkrun0_node_1"' in wrapper
    assert '-i /tmp/.ssh/id_ed25519' in wrapper
    assert 'docker exec "$CONTAINER" "$@"' in wrapper


def test_generate_rsh_wrapper_custom_key():
    """Wrapper uses custom SSH key path."""
    wrapper = TrtllmRuntime._generate_rsh_wrapper(
        {"10.0.0.1": "c0_node_0"}, "c0",
        ssh_key_path="/tmp/.ssh/id_rsa",
    )
    assert "-i /tmp/.ssh/id_rsa" in wrapper


# --- Extra config YAML ---

def test_build_extra_config():
    """Verify YAML generation from recipe defaults."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "defaults": {
            "free_gpu_memory_fraction": 0.9,
            "kv_cache_dtype": "auto",
            "cuda_graph_padding": True,
            "print_iter_log": False,
        },
    })
    runtime = TrtllmRuntime()
    config_yaml = runtime._build_extra_config(recipe)

    assert config_yaml is not None
    parsed = yaml.safe_load(config_yaml)
    assert parsed["print_iter_log"] is False
    assert parsed["kv_cache_config"]["free_gpu_memory_fraction"] == 0.9
    assert parsed["kv_cache_config"]["dtype"] == "auto"
    assert parsed["cuda_graph_config"]["enable_padding"] is True


def test_build_extra_config_empty():
    """Returns None when no extra config keys present."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "defaults": {"port": 8355},
    })
    runtime = TrtllmRuntime()
    assert runtime._build_extra_config(recipe) is None


def test_build_extra_config_partial():
    """Only present keys are included."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "defaults": {"free_gpu_memory_fraction": 0.85},
    })
    runtime = TrtllmRuntime()
    config_yaml = runtime._build_extra_config(recipe)

    assert config_yaml is not None
    parsed = yaml.safe_load(config_yaml)
    assert "kv_cache_config" in parsed
    assert parsed["kv_cache_config"]["free_gpu_memory_fraction"] == 0.85
    assert "cuda_graph_config" not in parsed
    assert "print_iter_log" not in parsed


# --- Environment and Docker opts ---

def test_cluster_env():
    """Returns OMPI and NCCL vars."""
    runtime = TrtllmRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["OMPI_ALLOW_RUN_AS_ROOT"] == "1"
    assert env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] == "1"
    assert env["NCCL_CUMEM_ENABLE"] == "0"
    assert env["OMPI_MCA_rmaps_ppr_n_pernode"] == "1"


def test_extra_docker_opts():
    """Returns ulimit flags."""
    runtime = TrtllmRuntime()
    opts = runtime.get_extra_docker_opts()

    assert "--ulimit" in opts
    assert "memlock=-1" in opts
    assert "stack=67108864" in opts


def test_extra_docker_opts_returns_copy():
    """get_extra_docker_opts returns a new list each time (not mutable ref)."""
    runtime = TrtllmRuntime()
    opts1 = runtime.get_extra_docker_opts()
    opts2 = runtime.get_extra_docker_opts()
    assert opts1 == opts2
    assert opts1 is not opts2


# --- Volumes ---

def test_extra_volumes_with_ssh_dir(tmp_path):
    """Returns SSH key mount when ~/.ssh exists."""
    runtime = TrtllmRuntime()
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()

    with mock.patch("sparkrun.runtimes.trtllm.Path.home", return_value=tmp_path):
        vols = runtime.get_extra_volumes()

    assert str(ssh_dir) in vols
    assert vols[str(ssh_dir)] == "/tmp/.ssh:ro"


def test_extra_volumes_no_ssh_dir(tmp_path):
    """Returns empty when ~/.ssh does not exist."""
    runtime = TrtllmRuntime()

    with mock.patch("sparkrun.runtimes.trtllm.Path.home", return_value=tmp_path):
        vols = runtime.get_extra_volumes()

    assert vols == {}


# --- Validation ---

def test_validate_recipe_valid():
    """Valid recipe returns no issues."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
    })
    runtime = TrtllmRuntime()
    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe = Recipe.from_dict({
        "name": "test",
        "runtime": "trtllm",
    })
    runtime = TrtllmRuntime()
    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_validate_recipe_multinode_no_ssh(tmp_path):
    """Multi-node recipe warns about missing SSH keys."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "min_nodes": 2,
    })
    runtime = TrtllmRuntime()

    with mock.patch("sparkrun.runtimes.trtllm.Path.home", return_value=tmp_path):
        issues = runtime.validate_recipe(recipe)

    assert any("SSH keys" in i for i in issues)


def test_validate_recipe_multinode_with_ssh(tmp_path):
    """Multi-node recipe with SSH keys returns no SSH warning."""
    (tmp_path / ".ssh").mkdir()
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "min_nodes": 2,
    })
    runtime = TrtllmRuntime()

    with mock.patch("sparkrun.runtimes.trtllm.Path.home", return_value=tmp_path):
        issues = runtime.validate_recipe(recipe)

    assert not any("SSH keys" in i for i in issues)


# --- Container naming ---

def test_head_container_name():
    """Head container uses {cluster_id}_node_0 pattern."""
    runtime = TrtllmRuntime()
    assert runtime._head_container_name("sparkrun0") == "sparkrun0_node_0"
    assert runtime._head_container_name("mytest") == "mytest_node_0"


def test_cluster_log_mode():
    """TRT-LLM cluster uses docker log mode."""
    runtime = TrtllmRuntime()
    assert runtime._cluster_log_mode() == "docker"


# --- Follow logs ---

class TestTrtllmFollowLogs:
    """Test TrtllmRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo(self, mock_stream):
        """Single-host trtllm tails serve log in solo container."""
        runtime = TrtllmRuntime()
        runtime.follow_logs(hosts=["10.0.0.1"], cluster_id="test0")

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_cluster(self, mock_stream):
        """Multi-host trtllm follows docker logs on _node_0 container."""
        runtime = TrtllmRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_node_0"


# --- Command-hint detection ---

def test_command_hint_trtllm_serve():
    """Recipe with 'trtllm-serve' command is detected as trtllm runtime."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "container": "nvcr.io/nvidia/tensorrt-llm/release:latest",
        "command": "trtllm-serve nvidia/model --tp_size 2",
    })
    assert recipe.runtime == "trtllm"


def test_command_hint_mpirun_trtllm():
    """Recipe with 'mpirun ... trtllm' command is detected as trtllm runtime."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "container": "nvcr.io/nvidia/tensorrt-llm/release:latest",
        "command": "mpirun --allow-run-as-root trtllm-llmapi-launch trtllm-serve nvidia/model",
    })
    assert recipe.runtime == "trtllm"


def test_command_hint_explicit_runtime_not_overridden():
    """Explicit runtime is not overridden by command hints."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "nvidia/model",
        "runtime": "trtllm",
        "container": "custom:latest",
        "command": "custom-launcher --model nvidia/model",
    })
    assert recipe.runtime == "trtllm"


# --- Stop cluster ---

def test_stop_cluster_delegates_to_native():
    """_stop_cluster delegates to _stop_native_cluster."""
    runtime = TrtllmRuntime()
    with mock.patch.object(runtime, "_stop_native_cluster", return_value=0) as m:
        rc = runtime._stop_cluster(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="spark0",
            config=None,
            dry_run=True,
        )
        assert rc == 0
        m.assert_called_once_with(
            ["10.0.0.1", "10.0.0.2"], "spark0",
            config=None, dry_run=True,
        )


# --- Base class extra_docker_opts integration ---

def test_base_get_extra_docker_opts_default():
    """Base RuntimePlugin.get_extra_docker_opts() returns empty list."""
    from sparkrun.runtimes.base import RuntimePlugin
    runtime = RuntimePlugin()
    assert runtime.get_extra_docker_opts() == []


def test_generate_node_script_extra_docker_opts():
    """_generate_node_script passes extra_docker_opts to docker_run_cmd."""
    from sparkrun.runtimes.base import RuntimePlugin

    script = RuntimePlugin._generate_node_script(
        image="test:latest",
        container_name="test_node_0",
        serve_command="echo hello",
        extra_docker_opts=["--ulimit", "memlock=-1"],
    )
    assert "--ulimit" in script
    assert "memlock=-1" in script


def test_generate_node_script_no_extra_opts():
    """_generate_node_script works without extra_docker_opts."""
    from sparkrun.runtimes.base import RuntimePlugin

    script = RuntimePlugin._generate_node_script(
        image="test:latest",
        container_name="test_node_0",
        serve_command="echo hello",
    )
    assert "docker run" in script
    # Should NOT contain ulimit (no extra opts passed)
    assert "memlock" not in script
