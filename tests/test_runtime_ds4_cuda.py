"""Unit tests for the ds4 CUDA runtime plugin."""

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.ds4_cuda import Ds4CudaRuntime


def _recipe(**overrides) -> Recipe:
    base = {
        "name": "test-recipe",
        "model": "/models/DeepSeek-V4-Flash-FP8.gguf",
        "runtime": "ds4-cuda",
    }
    base.update(overrides)
    return Recipe.from_dict(base)


# --- Identity / container ---


def test_ds4_runtime_name():
    runtime = Ds4CudaRuntime()
    assert runtime.runtime_name == "ds4-cuda"
    assert runtime.cluster_strategy() == "native"


def test_ds4_default_executor():
    """ds4-cuda defaults to the LocalExecutor (native binary, no Docker)."""
    runtime = Ds4CudaRuntime()
    assert runtime.default_executor() == "local"


def test_ds4_get_family():
    runtime = Ds4CudaRuntime()
    assert runtime.get_family() == "ds4-cuda"


def test_ds4_resolve_container_default():
    """No container field → ':latest' (empty prefix, native binary).

    ds4 has no Docker image, so ``default_image_prefix`` is empty and
    ``resolve_container`` returns ``":latest"``.  ``default_image_for``
    (the preferred override point) returns ``None`` — tested below.
    """
    runtime = Ds4CudaRuntime()
    assert runtime.resolve_container(_recipe()) == ":latest"


def test_ds4_default_image_for_returns_none():
    """default_image_for returns None (no default image for native runtime)."""
    runtime = Ds4CudaRuntime()
    assert runtime.default_image_for() is None


def test_ds4_resolve_container_from_recipe():
    """Recipe container field wins when set."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(container="custom/ds4:v1.0")
    assert runtime.resolve_container(recipe) == "custom/ds4:v1.0"


# --- Solo command generation ---


def test_ds4_generate_command_structured():
    """Generates `ds4-serve -m <model> -c <ctx> --port <port> --host <host>`."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(
        defaults={
            "port": 8000,
            "context": 65536,
            "host": "0.0.0.0",
        },
    )

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("ds4-serve -m /models/DeepSeek-V4-Flash-FP8.gguf")
    assert "-c 65536" in cmd
    assert "--port 8000" in cmd
    assert "--host 0.0.0.0" in cmd


def test_ds4_generate_command_defaults_applied():
    """Built-in defaults (context=131072, host=0.0.0.0, port=8000) are applied when unset."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("ds4-serve -m /models/DeepSeek-V4-Flash-FP8.gguf")
    assert "-c 131072" in cmd
    assert "--port 8000" in cmd
    assert "--host 0.0.0.0" in cmd


def test_ds4_generate_command_from_template():
    """Recipe with explicit command template renders it verbatim."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(
        command="ds4-serve -m {model} --port {port}",
        defaults={"port": 9000},
    )

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "ds4-serve -m /models/DeepSeek-V4-Flash-FP8.gguf --port 9000"


def test_ds4_cli_overrides_defaults():
    """CLI overrides take priority over recipe defaults."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(defaults={"port": 8000})
    cmd = runtime.generate_command(recipe, {"port": 9000}, is_cluster=False)
    assert "--port 9000" in cmd
    assert "--port 8000" not in cmd


def test_ds4_max_model_len_translated_to_context():
    """max_model_len (cross-runtime key) is translated to context (-c)."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(defaults={"max_model_len": 32768})

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "-c 32768" in cmd


def test_ds4_context_takes_precedence_over_max_model_len():
    """When both context and max_model_len are set, context wins."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(defaults={"context": 65536, "max_model_len": 32768})

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "-c 65536" in cmd
    assert "-c 32768" not in cmd


# --- skip_keys / flag stripping ---


def test_ds4_skip_keys_strips_port():
    """skip_keys suppresses --port from structured commands."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(defaults={"port": 8000})
    cmd = runtime.generate_command(recipe, {}, is_cluster=False, skip_keys={"port"})
    assert "--port" not in cmd


def test_ds4_skip_keys_strips_from_template():
    """skip_keys also strips flags from rendered command templates."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe(
        command="ds4-serve -m {model} --port {port} --host {host}",
        defaults={"port": 8000, "host": "0.0.0.0"},
    )
    cmd = runtime.generate_command(recipe, {}, is_cluster=False, skip_keys={"port"})
    assert "--port" not in cmd
    assert "--host 0.0.0.0" in cmd


def test_ds4_skip_keys_strips_context():
    """skip_keys suppresses -c (context) from structured commands."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe()
    cmd = runtime.generate_command(recipe, {}, is_cluster=False, skip_keys={"context"})
    assert " -c " not in cmd


# --- Environment ---


def test_ds4_get_common_env_hf_offline():
    """Common env sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE."""
    runtime = Ds4CudaRuntime()
    env = runtime.get_common_env()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_ds4_get_extra_env_defaults():
    """Extra env includes ds4 default tuning knobs."""
    runtime = Ds4CudaRuntime()
    env = runtime.get_extra_env()
    assert env["DS4_BATCH_FIT_HEADROOM_MB"] == "8192"
    assert env["DS4_SERVER_SERIAL_MAX_TOKENS"] == "131072"
    assert env["DS4_SERVER_COALESCE_MAX"] == "2"


# --- resolve_api_key ---


def test_ds4_resolve_api_key_from_defaults():
    """defaults.api_key is honored."""
    recipe = _recipe(defaults={"api_key": "sk-default"})
    assert Ds4CudaRuntime().resolve_api_key(recipe) == "sk-default"


def test_ds4_resolve_api_key_from_env():
    """env.DS4_API_KEY is honored when defaults.api_key is absent."""
    recipe = _recipe(env={"DS4_API_KEY": "sk-env"})
    assert Ds4CudaRuntime().resolve_api_key(recipe) == "sk-env"


def test_ds4_resolve_api_key_overrides_take_priority():
    """CLI override beats defaults and env."""
    recipe = _recipe(defaults={"api_key": "sk-default"}, env={"DS4_API_KEY": "sk-env"})
    assert Ds4CudaRuntime().resolve_api_key(recipe, {"api_key": "sk-cli"}) == "sk-cli"


def test_ds4_resolve_api_key_defaults_beat_env():
    """defaults.api_key takes precedence over env.DS4_API_KEY."""
    recipe = _recipe(defaults={"api_key": "sk-default"}, env={"DS4_API_KEY": "sk-env"})
    assert Ds4CudaRuntime().resolve_api_key(recipe) == "sk-default"


def test_ds4_resolve_api_key_none_when_unset():
    """Returns None when no api_key is configured anywhere."""
    assert Ds4CudaRuntime().resolve_api_key(_recipe()) is None


def test_ds4_resolve_api_key_parses_inline_command_flag():
    """Literal --api-key in a fixed command string is extracted."""
    recipe = _recipe(
        command="ds4-serve -m /models/m.gguf --api-key sk-inline --port 8080",
    )
    assert Ds4CudaRuntime().resolve_api_key(recipe) == "sk-inline"


def test_ds4_resolve_api_key_ignores_placeholder_in_command():
    """`--api-key {api_key}` placeholder is ignored — defaults path handles it."""
    recipe = _recipe(
        command="ds4-serve -m /models/m.gguf --api-key {api_key} --port 8080",
        defaults={"api_key": "sk-default"},
    )
    assert Ds4CudaRuntime().resolve_api_key(recipe) == "sk-default"


# --- validate_recipe ---


def test_ds4_validate_recipe_valid():
    """Valid recipe returns no issues."""
    runtime = Ds4CudaRuntime()
    assert runtime.validate_recipe(_recipe()) == []


def test_ds4_validate_recipe_no_model():
    """Missing model returns issue."""
    runtime = Ds4CudaRuntime()
    recipe = Recipe.from_dict({"name": "test", "runtime": "ds4-cuda"})
    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_ds4_validate_recipe_warns_on_docker_executor():
    """Non-local executor surfaces a warning (ds4 is native-only)."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe()
    recipe.executor = "docker"
    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "executor='docker'" in issues[0]
    assert "executor=local is supported" in issues[0]


def test_ds4_validate_recipe_warns_on_multi_node():
    """max_nodes > 1 surfaces a warning (ds4 has no distributed path)."""
    runtime = Ds4CudaRuntime()
    recipe = _recipe()
    recipe.max_nodes = 2
    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "multi-node inference" in issues[0]


# --- version_commands ---


def test_ds4_version_commands():
    """Version commands include ds4-server --version."""
    runtime = Ds4CudaRuntime()
    cmds = runtime.version_commands()
    assert "ds4" in cmds
    assert "ds4-server --version" in cmds["ds4"]
