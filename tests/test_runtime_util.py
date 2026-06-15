"""Unit tests for sparkrun.runtimes._util helpers."""

from __future__ import annotations

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes._util import resolve_api_key


def _recipe(*, defaults=None, env=None, command=None) -> Recipe:
    """Build a minimal Recipe with the optional fields."""
    data: dict = {"name": "r", "model": "m", "runtime": "vllm"}
    if defaults is not None:
        data["defaults"] = defaults
    if env is not None:
        data["env"] = env
    if command is not None:
        data["command"] = command
    return Recipe.from_dict(data)


# --- resolve_api_key priority chain ---


def test_resolve_api_key_override_beats_defaults_env_and_command():
    """CLI override is highest priority."""
    recipe = _recipe(
        defaults={"api_key": "sk-default"},
        env={"VLLM_API_KEY": "sk-env"},
        command="vllm serve foo --api-key sk-inline",
    )
    assert resolve_api_key(recipe, {"api_key": "sk-cli"}, "VLLM_API_KEY") == "sk-cli"


def test_resolve_api_key_defaults_beat_env_and_command():
    """defaults.api_key wins over env and inline command flags."""
    recipe = _recipe(
        defaults={"api_key": "sk-default"},
        env={"VLLM_API_KEY": "sk-env"},
        command="vllm serve foo --api-key sk-inline",
    )
    assert resolve_api_key(recipe, None, "VLLM_API_KEY") == "sk-default"


def test_resolve_api_key_env_beats_inline_command():
    """env var wins over inline command flag when defaults are absent."""
    recipe = _recipe(
        env={"VLLM_API_KEY": "sk-env"},
        command="vllm serve foo --api-key sk-inline",
    )
    assert resolve_api_key(recipe, None, "VLLM_API_KEY") == "sk-env"


def test_resolve_api_key_falls_back_to_inline_command_flag():
    """Final fallback: parse the literal flag from command string."""
    recipe = _recipe(command="vllm serve foo --api-key sk-inline")
    assert resolve_api_key(recipe, None, "VLLM_API_KEY") == "sk-inline"


def test_resolve_api_key_returns_none_when_unset():
    """No source set anywhere -> None."""
    recipe = _recipe()
    assert resolve_api_key(recipe, None, "VLLM_API_KEY") is None


def test_resolve_api_key_custom_flag_name():
    """flag_name parameter targets non-standard CLI flags (e.g. --auth-token)."""
    recipe = _recipe(command="spark serve foo --auth-token sk-atlas")
    # The default --api-key probe would not match.
    assert resolve_api_key(recipe, None, "ATLAS_API_KEY") is None
    # Targeting --auth-token explicitly recovers the value.
    assert resolve_api_key(recipe, None, "ATLAS_API_KEY", flag_name="--auth-token") == "sk-atlas"


def test_resolve_api_key_env_var_name_matters():
    """env_var parameter selects which key in recipe.env is consulted."""
    recipe = _recipe(env={"SGLANG_API_KEY": "sk-sglang", "LLAMA_API_KEY": "sk-llama"})
    assert resolve_api_key(recipe, None, "SGLANG_API_KEY") == "sk-sglang"
    assert resolve_api_key(recipe, None, "LLAMA_API_KEY") == "sk-llama"
    # Unrelated env var name returns None.
    assert resolve_api_key(recipe, None, "MISSING_API_KEY") is None


def test_resolve_api_key_overrides_empty_dict_skips_to_defaults():
    """An empty overrides dict should not short-circuit; falls through."""
    recipe = _recipe(defaults={"api_key": "sk-default"})
    assert resolve_api_key(recipe, {}, "VLLM_API_KEY") == "sk-default"


def test_resolve_api_key_overrides_none_value_falls_through():
    """overrides with api_key=None should not short-circuit."""
    recipe = _recipe(defaults={"api_key": "sk-default"})
    assert resolve_api_key(recipe, {"api_key": None}, "VLLM_API_KEY") == "sk-default"


# --- _make_node_command_args helper ---


def test_make_node_command_args_canonical_shape():
    """Returns a flat dict with stringified ints for the four key knobs."""
    from sparkrun.runtimes.base import RuntimePlugin

    class _Stub(RuntimePlugin):
        runtime_name = "stub"

        def generate_command(self, *args, **kwargs):
            return ""

    # Rank 0 with replica_size=1 — every fallback path returns head_ip
    # (dp_rank=0, hosts[0] = head_ip already).
    args = _Stub()._make_node_command_args(
        head_ip="10.0.0.1",
        num_nodes=2,
        node_rank=0,
        init_port=25000,
        hosts=["10.0.0.1", "10.0.0.2"],
        replica_size=1,
    )
    assert args == {
        "num_nodes": "2",
        "node_rank": "0",
        "master_addr": "10.0.0.1",
        "master_port": "25000",
    }


def test_make_node_command_args_hybrid_picks_replica_head():
    """Hybrid tp+dp: replica 1's nodes get host[2] as master_addr, not head."""
    from sparkrun.runtimes.base import RuntimePlugin

    class _Stub(RuntimePlugin):
        runtime_name = "stub"

        def generate_command(self, *args, **kwargs):
            return ""

    # 4 hosts, replica_size=2 (tp=2), so replica 0 = [A,B], replica 1 = [C,D]
    args = _Stub()._make_node_command_args(
        head_ip="A",
        num_nodes=2,
        node_rank=3,  # global rank 3 -> dp_rank=1, intra=1
        init_port=25000,
        hosts=["A", "B", "C", "D"],
        replica_size=2,
    )
    # Replica head for global rank 3 is hosts[1 * 2] = "C"
    assert args["master_addr"] == "C"


def test_make_node_command_args_no_hosts_fallback_to_head_ip():
    """No hosts and no placement -> master_addr falls back to head_ip."""
    from sparkrun.runtimes.base import RuntimePlugin

    class _Stub(RuntimePlugin):
        runtime_name = "stub"

        def generate_command(self, *args, **kwargs):
            return ""

    args = _Stub()._make_node_command_args(
        head_ip="10.0.0.99",
        num_nodes=2,
        node_rank=0,
        init_port=29500,
        hosts=None,
        replica_size=1,
    )
    assert args["master_addr"] == "10.0.0.99"
    assert args["master_port"] == "29500"


# --- VllmMixin._build_command ---


def test_vllm_mixin_build_command_solo():
    """Solo path: no cluster flags appended."""
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    runtime = VllmDistributedRuntime()
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "meta-llama/Llama-2-7b-hf",
            "runtime": "vllm-distributed",
            "defaults": {"port": 8000},
        }
    )
    config = recipe.build_config_chain({})
    cmd = runtime._build_command(recipe, config, is_cluster=False, num_nodes=1)
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "--port 8000" in cmd
    assert "--nnodes" not in cmd
    assert "--distributed-executor-backend" not in cmd


def test_vllm_mixin_build_command_native_cluster_emits_nnodes_master():
    """Cluster + head_ip + no backend -> emits --nnodes/--master-addr/--master-port."""
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    runtime = VllmDistributedRuntime()
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "meta-llama/Llama-2-7b-hf",
            "runtime": "vllm-distributed",
            "defaults": {"tensor_parallel": 2},
        }
    )
    config = recipe.build_config_chain({})
    cmd = runtime._build_command(
        recipe,
        config,
        is_cluster=True,
        num_nodes=2,
        head_ip="10.0.0.1",
    )
    assert "-tp 2" in cmd
    assert "--nnodes 2" in cmd
    assert "--master-addr 10.0.0.1" in cmd
    assert "--master-port 25000" in cmd
    assert "--distributed-executor-backend" not in cmd


def test_vllm_mixin_build_command_ray_backend():
    """cluster_backend set -> emits --distributed-executor-backend ray."""
    from sparkrun.runtimes.vllm_ray import VllmRayRuntime

    runtime = VllmRayRuntime()
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "meta-llama/Llama-2-7b-hf",
            "runtime": "vllm",
            "defaults": {"tensor_parallel": 2},
        }
    )
    config = recipe.build_config_chain({})
    cmd = runtime._build_command(
        recipe,
        config,
        is_cluster=True,
        num_nodes=2,
        cluster_backend="ray",
    )
    assert "-tp 2" in cmd
    assert "--distributed-executor-backend ray" in cmd
    # When cluster_backend is set, native --nnodes/--master-addr should not appear.
    assert "--nnodes" not in cmd
    assert "--master-addr" not in cmd


def test_vllm_mixin_build_command_skip_keys_strips_from_flag_map():
    """skip_keys propagated to flag emission."""
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    runtime = VllmDistributedRuntime()
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "m",
            "runtime": "vllm-distributed",
            "defaults": {"port": 8000, "served_model_name": "alias"},
        }
    )
    config = recipe.build_config_chain({})
    cmd = runtime._build_command(
        recipe,
        config,
        is_cluster=False,
        num_nodes=1,
        skip_keys={"served_model_name"},
    )
    assert "--served-model-name" not in cmd
    assert "--port 8000" in cmd


def test_reconcile_flag_fill_mode():
    """override=False (fill): append when absent, never touch an existing value."""
    from sparkrun.runtimes.base import RuntimePlugin

    rec = RuntimePlugin.reconcile_flag_in_command
    # Absent -> appended.
    assert rec("vllm serve m", "--served-model-name", "alias") == "vllm serve m --served-model-name alias"
    # Present -> left exactly as-is (template author wins).
    assert rec("vllm serve m --served-model-name keep", "--served-model-name", "alias", override=False) == (
        "vllm serve m --served-model-name keep"
    )


def test_reconcile_flag_override_mode():
    """override=True: replace an existing value, or append when absent. Idempotent."""
    from sparkrun.runtimes.base import RuntimePlugin

    rec = RuntimePlugin.reconcile_flag_in_command
    flag = "--distributed-executor-backend"
    # Replace existing value.
    assert rec("vllm serve m %s ray" % flag, flag, "mp", override=True) == "vllm serve m %s mp" % flag
    # Append when absent.
    assert rec("vllm serve m", flag, "mp", override=True) == "vllm serve m %s mp" % flag
    # Only the value changes; surrounding flags survive and no duplication.
    out = rec("vllm serve m %s ray -tp 2" % flag, flag, "mp", override=True)
    assert out == "vllm serve m %s mp -tp 2" % flag
    assert out.count(flag) == 1
    # Idempotent under the same target value.
    assert rec(out, flag, "mp", override=True) == out
