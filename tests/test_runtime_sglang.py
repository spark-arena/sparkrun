"""Unit tests for sparkrun.runtimes.sglang (SglangRuntime)."""

from unittest import mock

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.sglang import SglangRuntime


# --- SglangRuntime Tests ---


def test_sglang_runtime_name():
    """SglangRuntime.runtime_name == 'sglang'."""
    runtime = SglangRuntime()
    assert runtime.runtime_name == "sglang"


def test_sglang_resolve_container():
    """Container resolution."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "scitrera/dgx-spark-sglang:latest"


def test_sglang_generate_command_structured():
    """Generates python3 -m sglang.launch_server with --tp-size, etc."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "defaults": {
            "port": 30000,
            "tensor_parallel": 2,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("python3 -m sglang.launch_server")
    assert "--model-path meta-llama/Llama-2-7b-hf" in cmd
    assert "--tp-size 2" in cmd
    assert "--port 30000" in cmd


def test_sglang_generate_command_cluster():
    """Cluster mode adds --nnodes and --dist-init-addr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2, head_ip="192.168.1.100")
    assert "--dist-init-addr 192.168.1.100:25000" in cmd
    assert "--nnodes 2" in cmd
    assert "--tp-size 4" in cmd


def test_sglang_cluster_env():
    """Returns NCCL_CUMEM_ENABLE."""
    runtime = SglangRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["NCCL_CUMEM_ENABLE"] == "0"


# --- SGLang resolve_api_key Tests ---


def test_sglang_resolve_api_key_from_defaults():
    """defaults.api_key is the recommended source for sglang too."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "sglang",
            "defaults": {"api_key": "sk-default"},
        }
    )
    assert SglangRuntime().resolve_api_key(recipe) == "sk-default"


def test_sglang_resolve_api_key_from_env():
    """env.SGLANG_API_KEY is honored when defaults.api_key is absent."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "sglang",
            "env": {"SGLANG_API_KEY": "sk-env"},
        }
    )
    assert SglangRuntime().resolve_api_key(recipe) == "sk-env"


def test_sglang_resolve_api_key_overrides_take_priority():
    """CLI override beats defaults and env."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "sglang",
            "defaults": {"api_key": "sk-default"},
            "env": {"SGLANG_API_KEY": "sk-env"},
        }
    )
    assert SglangRuntime().resolve_api_key(recipe, {"api_key": "sk-cli"}) == "sk-cli"


def test_sglang_resolve_api_key_defaults_beat_env():
    """defaults.api_key takes precedence over env.SGLANG_API_KEY."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "sglang",
            "defaults": {"api_key": "sk-default"},
            "env": {"SGLANG_API_KEY": "sk-env"},
        }
    )
    assert SglangRuntime().resolve_api_key(recipe) == "sk-default"


def test_sglang_resolve_api_key_none_when_unset():
    """Returns None when no api_key is configured anywhere."""
    recipe = Recipe.from_dict({"name": "r", "model": "m", "runtime": "sglang"})
    assert SglangRuntime().resolve_api_key(recipe) is None


def test_sglang_resolve_api_key_parses_inline_command_flag():
    """Literal --api-key in a fixed command string is extracted."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "sglang",
            "command": "python -m sglang.launch_server --model-path m --api-key sk-inline --port 30000",
        }
    )
    assert SglangRuntime().resolve_api_key(recipe) == "sk-inline"


def test_sglang_resolve_api_key_ignores_placeholder_in_command():
    """`--api-key {api_key}` placeholder is ignored — defaults path handles it."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "sglang",
            "command": "python -m sglang.launch_server --api-key {api_key} --port 30000",
            "defaults": {"api_key": "sk-default"},
        }
    )
    assert SglangRuntime().resolve_api_key(recipe) == "sk-default"


def test_sglang_api_key_emitted_as_flag_for_structured_command():
    """defaults.api_key auto-emits as --api-key on structured (no-template) commands."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "sglang",
            "defaults": {"port": 30000, "api_key": "sk-flag"},
        }
    )
    cmd = SglangRuntime().generate_command(recipe, {}, is_cluster=False)
    assert "--api-key sk-flag" in cmd


def test_sglang_validate_recipe():
    """Validate recipe."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_sglang_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


# --- SGLang prepare(): speculative draft-model pre-sync ---


def _sglang_recipe(**overrides) -> Recipe:
    data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
    }
    data.update(overrides)
    return Recipe.from_dict(data)


def _model_names(recipe) -> list[str]:
    return [e.name for e in recipe.distribution_config.models.entries]


def test_sglang_prepare_canonical_key_adds_draft_model():
    """speculative_draft_model_path → distribution_config.add_model."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(defaults={"speculative_draft_model_path": "draft/repo"})
    runtime.prepare(recipe, hosts=["10.0.0.1"])
    assert "draft/repo" in _model_names(recipe)


def test_sglang_prepare_alias_key_adds_draft_model():
    """speculative_draft_model alias also triggers add_model."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(defaults={"speculative_draft_model": "alias/draft"})
    runtime.prepare(recipe, hosts=["10.0.0.1"])
    assert "alias/draft" in _model_names(recipe)


def test_sglang_prepare_canonical_wins_when_both_set():
    """When both keys are set, canonical key wins; add_model called once."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(
        defaults={
            "speculative_draft_model_path": "canonical/draft",
            "speculative_draft_model": "alias/draft",
        },
    )
    runtime.prepare(recipe, hosts=["10.0.0.1"])
    names = _model_names(recipe)
    assert "canonical/draft" in names
    assert "alias/draft" not in names


def test_sglang_prepare_no_speculative_is_noop():
    """prepare() does nothing when neither key is set."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe()
    before = list(_model_names(recipe))
    runtime.prepare(recipe, hosts=["10.0.0.1"])
    assert _model_names(recipe) == before


def test_sglang_speculative_canonical_emits_flag():
    """Generated command includes --speculative-draft-model-path."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(defaults={"speculative_draft_model_path": "draft/repo"})
    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--speculative-draft-model-path draft/repo" in cmd


def test_sglang_speculative_alias_emits_flag():
    """Alias key normalizes to canonical so the flag still emits."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(defaults={"speculative_draft_model": "alias/draft"})
    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--speculative-draft-model-path alias/draft" in cmd


def test_sglang_speculative_alias_emits_flag_in_node_command():
    """Alias normalization also applies to per-node command generation."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(defaults={"speculative_draft_model": "alias/draft"})
    cmd = runtime.generate_node_command(
        recipe,
        {},
        head_ip="10.0.0.1",
        num_nodes=2,
        node_rank=0,
    )
    assert "--speculative-draft-model-path alias/draft" in cmd


def test_sglang_speculative_skip_key_strips_flag():
    """skip_keys suppresses --speculative-draft-model-path."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(defaults={"speculative_draft_model_path": "draft/repo"})
    cmd = runtime.generate_command(
        recipe,
        {},
        is_cluster=False,
        skip_keys={"speculative_draft_model_path"},
    )
    assert "--speculative-draft-model-path" not in cmd


def test_sglang_overrides_in_command():
    """Test that CLI overrides work for sglang."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "defaults": {"port": 30000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    # Override port
    cmd = runtime.generate_command(recipe, {"port": 31000}, is_cluster=False)
    assert "--port 31000" in cmd
    assert "--port 30000" not in cmd


class TestSglangFollowLogs:
    """Test SglangRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_uses_file_logs(self, mock_stream):
        """Single-host sglang tails serve log file inside solo container."""
        runtime = SglangRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_uses_node_0(self, mock_stream):
        """Multi-host sglang follows the _node_0 container (file mode, sleep-infinity + exec)."""
        runtime = SglangRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_node_0"


class TestSglangComputeRequiredNodes:
    """Test SglangRuntime inherits base tp*pp (no override needed)."""

    def _make_recipe(self, defaults=None):
        data = {
            "name": "test",
            "runtime": "sglang",
            "model": "meta-llama/Llama-2-7b-hf",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_tp_only(self):
        """TP=4, no PP → requires 4 nodes."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 4})
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_tp_times_pp(self):
        """TP=2, PP=2 → requires 4 nodes."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_pp_only(self):
        """PP=3 with no explicit TP → 1*3 = 3 nodes."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 3})
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) == 3

    def test_no_parallelism_returns_none(self):
        """Neither TP nor PP set → None."""
        recipe = self._make_recipe()
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) is None

    def test_overrides_pp(self):
        """CLI overrides PP value."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2})
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe, {"pipeline_parallel": 3}) == 6

    def test_overrides_both(self):
        """CLI overrides both TP and PP."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe, {"tensor_parallel": 4, "pipeline_parallel": 3}) == 12


def test_sglang_pp_size_in_generated_command():
    """SGLang --pp-size flag appears in generated command."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {
            "tensor_parallel": 2,
            "pipeline_parallel": 2,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--pp-size 2" in cmd
    assert "--tp-size 2" in cmd


def test_sglang_pp_size_override_in_command():
    """SGLang --pp-size from overrides appears in generated command."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {"tensor_parallel": 2},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {"pipeline_parallel": 3}, is_cluster=False)
    assert "--pp-size 3" in cmd
    assert "--tp-size 2" in cmd
