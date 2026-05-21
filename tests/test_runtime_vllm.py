"""Unit tests for sparkrun.runtimes.vllm_ray (VllmRayRuntime)."""

from unittest import mock

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.vllm_ray import VllmRayRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime


# --- VllmRuntime Tests ---


def test_vllm_runtime_name():
    """VllmRayRuntime.runtime_name == 'vllm-ray'."""
    runtime = VllmRayRuntime()
    assert runtime.runtime_name == "vllm-ray"


def test_vllm_resolve_container_from_recipe():
    """Recipe with container field."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "container": "custom-vllm:v1.0",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-vllm:v1.0"


def test_vllm_resolve_container_default():
    """Recipe without container uses default prefix."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest"


def test_vllm_generate_command_from_template():
    """Recipe with command template renders correctly."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "command": "vllm serve {model} --port {port}",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "vllm serve meta-llama/Llama-2-7b-hf --port 8000"


def test_vllm_generate_command_structured():
    """Recipe without template generates vllm serve command from defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "port": 8000,
            "tensor_parallel": 2,
            "gpu_memory_utilization": 0.9,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "-tp 2" in cmd
    assert "--port 8000" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd


def test_vllm_generate_command_cluster():
    """Cluster mode adds --distributed-executor-backend ray."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "vllm",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert "--distributed-executor-backend ray" in cmd
    assert "-tp 4" in cmd


def test_vllm_generate_command_bool_flags():
    """Boolean flags like enforce_eager are handled."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "enforce_eager": True,
            "enable_prefix_caching": False,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--enforce-eager" in cmd
    # enable_prefix_caching is False, should not appear
    assert "--enable-prefix-caching" not in cmd


def test_vllm_validate_recipe_valid():
    """Valid recipe returns no issues."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_vllm_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_vllm_cluster_env():
    """get_cluster_env returns RAY_memory_monitor_refresh_ms."""
    runtime = VllmRayRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["RAY_memory_monitor_refresh_ms"] == "0"


# --- resolve_api_key Tests ---


def test_vllm_resolve_api_key_from_defaults():
    """defaults.api_key is the recommended source."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "defaults": {"api_key": "sk-default"},
        }
    )
    assert VllmRayRuntime().resolve_api_key(recipe) == "sk-default"
    assert VllmDistributedRuntime().resolve_api_key(recipe) == "sk-default"


def test_vllm_resolve_api_key_from_env():
    """env.VLLM_API_KEY is honored when defaults.api_key is absent."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "env": {"VLLM_API_KEY": "sk-env"},
        }
    )
    assert VllmRayRuntime().resolve_api_key(recipe) == "sk-env"


def test_vllm_resolve_api_key_overrides_take_priority():
    """CLI override beats defaults and env."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "defaults": {"api_key": "sk-default"},
            "env": {"VLLM_API_KEY": "sk-env"},
        }
    )
    assert VllmRayRuntime().resolve_api_key(recipe, {"api_key": "sk-cli"}) == "sk-cli"


def test_vllm_resolve_api_key_defaults_beat_env():
    """defaults.api_key takes precedence over env.VLLM_API_KEY."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "defaults": {"api_key": "sk-default"},
            "env": {"VLLM_API_KEY": "sk-env"},
        }
    )
    assert VllmRayRuntime().resolve_api_key(recipe) == "sk-default"


def test_vllm_resolve_api_key_none_when_unset():
    """Returns None when no api_key is configured anywhere."""
    recipe = Recipe.from_dict({"name": "r", "model": "m", "runtime": "vllm"})
    assert VllmRayRuntime().resolve_api_key(recipe) is None


def test_vllm_resolve_api_key_parses_inline_command_flag():
    """Literal --api-key in a fixed command string is extracted."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "command": "vllm serve m --api-key sk-inline --port 8000",
        }
    )
    assert VllmRayRuntime().resolve_api_key(recipe) == "sk-inline"


def test_vllm_resolve_api_key_parses_equals_form():
    """`--api-key=value` form is also extracted."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "command": "vllm serve m --api-key=sk-eq --port 8000",
        }
    )
    assert VllmRayRuntime().resolve_api_key(recipe) == "sk-eq"


def test_vllm_resolve_api_key_ignores_placeholder_in_command():
    """`--api-key {api_key}` placeholder is ignored — defaults path handles it."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "command": "vllm serve m --api-key {api_key} --port 8000",
            "defaults": {"api_key": "sk-default"},
        }
    )
    # defaults.api_key wins over the (rejected) placeholder
    assert VllmRayRuntime().resolve_api_key(recipe) == "sk-default"


def test_vllm_resolve_api_key_defaults_beat_inline_command():
    """defaults.api_key takes precedence over inline --api-key in command."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "command": "vllm serve m --api-key sk-inline",
            "defaults": {"api_key": "sk-default"},
        }
    )
    assert VllmRayRuntime().resolve_api_key(recipe) == "sk-default"


def test_vllm_api_key_emitted_as_flag_for_structured_command():
    """defaults.api_key auto-emits as --api-key on structured (no-template) commands."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm",
            "defaults": {"port": 8000, "api_key": "sk-flag"},
        }
    )
    cmd = VllmRayRuntime().generate_command(recipe, {}, is_cluster=False)
    assert "--api-key sk-flag" in cmd


def test_vllm_cluster_injects_ray_backend_into_template():
    """Cluster mode injects --distributed-executor-backend ray into command templates."""
    recipe_data = {
        "name": "test-recipe",
        "model": "nvidia/some-model",
        "runtime": "vllm",
        "command": "vllm serve {model} -tp {tensor_parallel} --port {port}",
        "defaults": {"tensor_parallel": 2, "port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    # Solo mode: no injection
    cmd_solo = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--distributed-executor-backend" not in cmd_solo

    # Cluster mode: auto-injected
    cmd_cluster = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert "--distributed-executor-backend ray" in cmd_cluster


def test_vllm_cluster_preserves_existing_backend_in_template():
    """Cluster mode does not double-add if template already has the flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "nvidia/some-model",
        "runtime": "vllm",
        "command": "vllm serve {model} --distributed-executor-backend ray",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert cmd.count("--distributed-executor-backend") == 1


def test_vllm_overrides_in_command():
    """Test that CLI overrides properly override defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    # Override port
    cmd = runtime.generate_command(recipe, {"port": 9000}, is_cluster=False)
    assert "--port 9000" in cmd
    assert "--port 8000" not in cmd


class TestVllmFollowLogs:
    """Test VllmRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_tails_serve_log(self, mock_stream):
        """Single-host vllm tails serve log in solo container."""
        runtime = VllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == "10.0.0.1"
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_tails_serve_log_on_head(self, mock_stream):
        """Multi-host vllm tails serve log in _head container on hosts[0]."""
        runtime = VllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_head"
