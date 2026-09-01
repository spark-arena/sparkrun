"""Unit tests for sparkrun.runtimes.sglang (SglangRuntime)."""

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.core.log_source import MODE_FILE, SCOPE_ALL


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
    """Generates `sglang serve` with --tp-size, etc."""
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
    assert cmd.startswith("sglang serve")
    assert "sglang.launch_server" not in cmd
    assert "--model-path meta-llama/Llama-2-7b-hf" in cmd
    assert "--tp-size 2" in cmd
    assert "--port 30000" in cmd


def test_sglang_legacy_launch_server_command_still_honored():
    """A recipe pinning the legacy entrypoint keeps it verbatim.

    Switching the *generated* entrypoint to ``sglang serve`` must not rewrite
    recipes that spell ``python3 -m sglang.launch_server`` in their own
    ``command:`` template — older pinned images may only have that form.
    """
    recipe = Recipe.from_dict(
        {
            "name": "legacy-recipe",
            "model": "meta-llama/Llama-2-7b-hf",
            "runtime": "sglang",
            "defaults": {"port": 30000},
            "command": "python3 -m sglang.launch_server --model-path {model} --port {port}",
        }
    )

    cmd = SglangRuntime().generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("python3 -m sglang.launch_server")


def test_sglang_serve_flags_emitted_without_command_template():
    """Serving-behaviour keys reach the generated command (no `command:` block).

    Regression guard for the flag-map gaps: these were previously absent from
    ``_SGLANG_FLAG_MAP``, so a command-less recipe silently served a
    differently-configured server — no parsers, default attention backend and
    no speculative decoding — with nothing reported.
    """
    recipe = Recipe.from_dict(
        {
            "name": "flags-recipe",
            "model": "Qwen/Qwen3.8-27B-FP8",
            "runtime": "sglang",
            "defaults": {
                "port": 8000,
                "attention_backend": "flashinfer",
                "load_format": "instanttensor",
                "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen3_coder",
                "speculative_algorithm": "NEXTN",
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 4,
                "enable_torch_compile": True,
                "disable_prefill_cuda_graph": True,
            },
        }
    )

    cmd = SglangRuntime().generate_command(recipe, {}, is_cluster=False)
    for expected in (
        "--attention-backend flashinfer",
        "--load-format instanttensor",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-torch-compile",
        "--disable-prefill-cuda-graph",
    ):
        assert expected in cmd, "missing %r in: %s" % (expected, cmd)


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


def test_sglang_node_command_rendezvous_at_head_with_hosts():
    """Regression: every worker must rendezvous at the head, not its own IP.

    With a hosts list (1 GPU per node on the Spark cluster), each node's
    --dist-init-addr must still point at the head node. Previously
    generate_node_command forwarded hosts/placement to _make_node_command_args
    with the default replica_size=1, so _resolve_master_addr returned
    hosts[node_rank] (each node's own IP) and only rank 0 bound the store,
    producing "1/N clients joined" rendezvous timeouts.
    """
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]

    for rank in range(4):
        cmd = runtime.generate_node_command(
            recipe,
            {},
            head_ip="10.0.0.1",
            num_nodes=4,
            node_rank=rank,
            hosts=hosts,
        )
        assert "--dist-init-addr 10.0.0.1:25000" in cmd, "rank %d -> %s" % (rank, cmd)
        assert "--nnodes 4" in cmd
        assert "--node-rank %d" % rank in cmd


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
    assert "model is required" in str(issues[0])


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


def test_sglang_prepare_pins_draft_model_revision_and_emits_flag():
    runtime = SglangRuntime()
    recipe = _sglang_recipe(
        defaults={
            "speculative_draft_model_path": "draft/repo",
            "speculative_draft_model_revision": "b" * 40,
        }
    )
    runtime.prepare(recipe, hosts=["10.0.0.1"])

    draft = next(entry for entry in recipe.distribution_config.models.entries if entry.name == "draft/repo")
    assert draft.revision == "b" * 40
    assert "--speculative-draft-model-revision %s" % ("b" * 40) in runtime.generate_command(recipe, {}, is_cluster=False)


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


def _draft_entry(recipe, name):
    return next(e for e in recipe.distribution_config.models.entries if e.name == name)


def test_sglang_prepare_draft_model_is_unpinned_by_default():
    """The draft repo must not inherit the served model's revision pin."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(model_revision="deadbeef", defaults={"speculative_draft_model_path": "draft/repo"})
    runtime.prepare(recipe, hosts=["10.0.0.1"])
    assert _draft_entry(recipe, "draft/repo").revision is None


def test_sglang_prepare_pins_draft_model_revision_when_declared():
    """speculative_draft_model_revision pins the draft repo, not the served one."""
    runtime = SglangRuntime()
    recipe = _sglang_recipe(
        model_revision="deadbeef",
        defaults={
            "speculative_draft_model_path": "draft/repo",
            "speculative_draft_model_revision": "cafe1234",
        },
    )
    runtime.prepare(recipe, hosts=["10.0.0.1"])
    assert _draft_entry(recipe, "draft/repo").revision == "cafe1234"
    assert _draft_entry(recipe, "{model}").revision == "deadbeef"


def test_sglang_draft_revision_key_is_declared_known():
    """Otherwise report_unmapped_config_keys warns the key was dropped."""
    assert "speculative_draft_model_revision" in SglangRuntime().known_config_keys()


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

    def test_follow_logs_solo_uses_file_logs(self, log_sources_spy):
        """Single-host sglang tails serve log file inside solo container."""
        SglangRuntime().follow_logs(hosts=["10.0.0.1"], cluster_id="test0")

        (source,) = log_sources_spy[0].sources
        assert (source.container, source.mode) == ("test0_solo", MODE_FILE)

    def test_follow_logs_cluster_uses_node_0(self, log_sources_spy):
        """Multi-host sglang follows the _node_0 container (file mode, sleep-infinity + exec)."""
        SglangRuntime().follow_logs(hosts=["10.0.0.1", "10.0.0.2"], cluster_id="mycluster")

        (source,) = log_sources_spy[0].sources
        assert (source.host, source.container, source.mode) == ("10.0.0.1", "mycluster_node_0", MODE_FILE)

    def test_scope_all_names_each_rank(self, log_sources_spy):
        """Native runtimes name one ranked container per host, rank i on hosts[i]."""
        SglangRuntime().follow_logs(hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"], cluster_id="mycluster", scope=SCOPE_ALL)

        sources = log_sources_spy[0].sources
        assert [(s.host, s.container, s.rank) for s in sources] == [
            ("10.0.0.1", "mycluster_node_0", 0),
            ("10.0.0.2", "mycluster_node_1", 1),
            ("10.0.0.3", "mycluster_node_2", 2),
        ]


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


def test_sglang_bool_flags_are_all_reachable():
    """Every boolean key must also appear in the flag map (see vLLM twin)."""
    from sparkrun.runtimes.sglang import _SGLANG_BOOL_FLAGS, _SGLANG_FLAG_MAP

    unreachable = sorted(k for k in _SGLANG_BOOL_FLAGS if k not in _SGLANG_FLAG_MAP)
    assert not unreachable, "bool keys missing from _SGLANG_FLAG_MAP: %s" % unreachable


# ---------------------------------------------------------------------------
# Data parallelism (issue #284)
#
# SGLang spells DP two incompatible ways and refuses one of them across nodes:
#   assert (tp_size * pp_size) % nnodes == 0
#   assert not (dp_size > 1 and nnodes != 1 and not enable_dp_attention)
# so the launch topology, not the recipe alone, decides which flags are legal.
# ---------------------------------------------------------------------------


def _dp_recipe(defaults, command=None):
    data = {
        "name": "dp-recipe",
        "model": "Qwen/Qwen3-8B",
        "runtime": "sglang",
        "defaults": defaults,
    }
    if command:
        data["command"] = command
    return Recipe.from_dict(data)


def test_sglang_pure_dp_node_command_has_no_rendezvous():
    """dp>1, tp*pp==1: each node is a standalone replica.

    Injecting --nnodes/--node-rank here trips "tp_size must be divisible by
    number of nodes" *before* the server binds a port, which is the whole of
    issue #284.
    """
    recipe = _dp_recipe({"data_parallel": 2})
    runtime = SglangRuntime()
    hosts = ["10.0.0.1", "10.0.0.2"]

    for rank in range(2):
        cmd = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=rank, hosts=hosts)
        assert "--nnodes" not in cmd, cmd
        assert "--node-rank" not in cmd, cmd
        assert "--dist-init-addr" not in cmd, cmd
        # --dp-size describes replicas inside ONE launch; each of these
        # launches owns exactly one.
        assert "--dp-size" not in cmd, cmd


def test_sglang_pure_dp_cluster_command_has_no_rendezvous():
    """The non-node-specific cluster command follows the same rule."""
    recipe = _dp_recipe({"data_parallel": 2})
    cmd = SglangRuntime().generate_command(recipe, {}, is_cluster=True, num_nodes=2, head_ip="10.0.0.1")
    assert "--nnodes" not in cmd
    assert "--dist-init-addr" not in cmd
    assert "--dp-size" not in cmd


def test_sglang_hybrid_dp_tp_rendezvous_per_replica():
    """dp=2 x tp=2: two independent 2-node worlds, each rendezvousing at its own head."""
    recipe = _dp_recipe({"data_parallel": 2, "tensor_parallel": 2})
    runtime = SglangRuntime()
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]

    expected = {0: ("10.0.0.1", 0), 1: ("10.0.0.1", 1), 2: ("10.0.0.3", 0), 3: ("10.0.0.3", 1)}
    for rank, (master, intra_rank) in expected.items():
        cmd = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=4, node_rank=rank, hosts=hosts)
        assert "--nnodes 2" in cmd, "rank %d -> %s" % (rank, cmd)
        assert "--node-rank %d" % intra_rank in cmd, "rank %d -> %s" % (rank, cmd)
        assert "--dist-init-addr %s:25000" % master in cmd, "rank %d -> %s" % (rank, cmd)
        # Replicas are joined by a router, not by --dp-size.
        assert "--dp-size" not in cmd, cmd


def test_sglang_dp_attention_keeps_single_global_world():
    """DP attention is the one multi-node DP shape upstream allows."""
    recipe = _dp_recipe({"data_parallel": 2, "tensor_parallel": 2, "enable_dp_attention": True})
    runtime = SglangRuntime()
    hosts = ["10.0.0.1", "10.0.0.2"]

    for rank in range(2):
        cmd = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=rank, hosts=hosts)
        assert "--nnodes 2" in cmd
        assert "--node-rank %d" % rank in cmd
        assert "--dist-init-addr 10.0.0.1:25000" in cmd
        assert "--dp-size 2" in cmd
        assert "--enable-dp-attention" in cmd


def test_sglang_dp_attention_hardcoded_in_command_template_is_seen():
    """A template that spells --enable-dp-attention must keep its rendezvous.

    The config chain cannot see a literal in ``command:``; missing it would
    classify a working DeepSeek/Qwen-MoE recipe as independent replicas and
    strip the flags that make it work.
    """
    recipe = _dp_recipe(
        {"data_parallel": 2, "tensor_parallel": 2},
        command="sglang serve --model-path {model} --tp-size 2 --dp-size 2 --enable-dp-attention",
    )
    cmd = SglangRuntime().generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1, hosts=["10.0.0.1", "10.0.0.2"])
    assert "--nnodes 2" in cmd
    assert "--node-rank 1" in cmd
    assert "--dist-init-addr 10.0.0.1:25000" in cmd
    # Already present in the template — not duplicated.
    assert cmd.count("--dp-size") == 1


def test_sglang_solo_dp_emits_dp_size():
    """Single launch unit: --dp-size is exactly right and was silently dropped."""
    recipe = _dp_recipe({"data_parallel": 4})
    cmd = SglangRuntime().generate_command(recipe, {}, is_cluster=False, num_nodes=1)
    assert "--dp-size 4" in cmd
    assert "--nnodes" not in cmd


def test_sglang_dp_size_override_reaches_solo_command():
    """`-o data_parallel=2` is honoured, not dropped (same class as #276)."""
    recipe = _dp_recipe({})
    cmd = SglangRuntime().generate_command(recipe, {"data_parallel": 2}, is_cluster=False, num_nodes=1)
    assert "--dp-size 2" in cmd


def test_sglang_dp1_cluster_command_unchanged():
    """dp==1 keeps the pre-#284 behaviour byte for byte."""
    recipe = _dp_recipe({"tensor_parallel": 4})
    runtime = SglangRuntime()
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]

    for rank in range(4):
        cmd = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=4, node_rank=rank, hosts=hosts)
        assert "--dist-init-addr 10.0.0.1:25000" in cmd
        assert "--nnodes 4" in cmd
        assert "--node-rank %d" % rank in cmd
        assert "--dp-size" not in cmd


def test_sglang_world_size_dp_attention_does_not_multiply():
    """dp partitions the tp world under DP attention; the default formula over-schedules."""
    from sparkrun.core.parallelism import ParallelismConfig

    runtime = SglangRuntime()
    p = ParallelismConfig(tensor_parallel=16, data_parallel=16)

    attention = _dp_recipe({"tensor_parallel": 16, "data_parallel": 16, "enable_dp_attention": True})
    assert runtime.world_size(p, recipe=attention, cluster=None) == 16

    replicas = _dp_recipe({"tensor_parallel": 2, "data_parallel": 3})
    assert runtime.world_size(ParallelismConfig(tensor_parallel=2, data_parallel=3), recipe=replicas, cluster=None) == 6


def test_sglang_native_rendezvous_port():
    """No rendezvous under pure DP; unchanged everywhere else."""
    runtime = SglangRuntime()

    assert runtime.native_rendezvous_port(_dp_recipe({"data_parallel": 2}), {}, num_nodes=2, init_port=25000) is None
    assert runtime.native_rendezvous_port(_dp_recipe({"tensor_parallel": 2}), {}, num_nodes=2, init_port=25000) == 25000
    hybrid = _dp_recipe({"data_parallel": 2, "tensor_parallel": 2})
    assert runtime.native_rendezvous_port(hybrid, {}, num_nodes=4, init_port=25000) == 25000
    attention = _dp_recipe({"data_parallel": 2, "tensor_parallel": 2, "enable_dp_attention": True})
    assert runtime.native_rendezvous_port(attention, {}, num_nodes=2, init_port=25000) == 25000


def test_sglang_validate_dp_attention_requires_dp_equals_tp():
    recipe = _dp_recipe({"tensor_parallel": 4, "data_parallel": 2, "enable_dp_attention": True})
    issues = SglangRuntime().validate_recipe(recipe)
    assert any("enable_dp_attention" in str(i) and i.severity == "error" for i in issues), issues


def test_sglang_validate_dp_replicas_warns_about_routing():
    """N replicas means N endpoints — say so rather than implying one address."""
    recipe = _dp_recipe({"data_parallel": 2})
    issues = SglangRuntime().validate_recipe(recipe)
    routing = [i for i in issues if "independent" in str(i)]
    assert routing and routing[0].severity == "warning", issues
    assert "router" in str(routing[0]) or "proxy" in str(routing[0])


def test_sglang_validate_dp1_is_quiet():
    assert not SglangRuntime()._validate_parallelism(_dp_recipe({"tensor_parallel": 2}))
