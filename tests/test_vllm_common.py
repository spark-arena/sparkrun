"""Behavior tests for ``max_model_len: auto`` in the vLLM runtimes.

``--max-model-len auto`` is a valid vLLM flag (vLLM auto-calculates the
length), so ``"auto"`` flows straight through the config chain to every
command path.  The only place that must special-case it is VRAM estimation,
which otherwise does ``int("auto")``.
"""

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.runtimes.vllm_ray import VllmRayRuntime


def _recipe(max_model_len="auto"):
    return Recipe.from_dict(
        {
            "name": "test",
            "model": "meta-llama/Llama-2-7b-hf",
            "runtime": "vllm",
            "defaults": {"max_model_len": max_model_len},
        }
    )


def test_auto_flows_to_ray_command():
    """vllm-ray emits ``--max-model-len auto`` from recipe defaults."""
    cmd = VllmRayRuntime().generate_command(_recipe(), {}, is_cluster=False)
    assert "--max-model-len auto" in cmd


def test_auto_flows_to_distributed_command():
    """vllm-distributed emits ``--max-model-len auto`` from recipe defaults."""
    cmd = VllmDistributedRuntime().generate_command(_recipe(), {}, is_cluster=False)
    assert "--max-model-len auto" in cmd


def test_auto_flows_to_distributed_node_command():
    """The per-node command path also carries ``--max-model-len auto``.

    This path (``generate_node_command``) rebuilds the config independently,
    so it must surface ``auto`` without any per-method injection.
    """
    cmd = VllmDistributedRuntime().generate_node_command(_recipe(), {}, head_ip="10.0.0.1", num_nodes=1, node_rank=0)
    assert "--max-model-len auto" in cmd


def test_auto_from_cli_override_flows_to_command():
    """``auto`` supplied as a CLI override (not a default) also flows through."""
    recipe = Recipe.from_dict({"name": "test", "model": "m", "runtime": "vllm"})
    cmd = VllmRayRuntime().generate_command(recipe, {"max_model_len": "auto"}, is_cluster=False)
    assert "--max-model-len auto" in cmd


def test_vllm_defaults_served_model_name_to_logical_model_id():
    """Offline snapshot-path rewrites must not change the public model name."""
    recipe = _recipe()
    for runtime in (VllmRayRuntime(), VllmDistributedRuntime()):
        command = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--served-model-name meta-llama/Llama-2-7b-hf" in command


def test_vllm_distributed_node_defaults_served_model_name():
    """The per-node TP path carries the same stable public model name."""
    command = VllmDistributedRuntime().generate_node_command(
        _recipe(),
        {},
        head_ip="10.0.0.1",
        num_nodes=1,
        node_rank=0,
    )
    assert "--served-model-name meta-llama/Llama-2-7b-hf" in command


def test_estimate_vram_handles_auto_max_model_len():
    """VRAM estimation must not choke on ``max_model_len: auto`` (no int('auto'))."""
    # auto_detect=False keeps this offline; the max_model_len guard still runs.
    est = _recipe().estimate_vram(auto_detect=False)
    assert est is not None


# ---------------------------------------------------------------------------
# finalize_host_comm_env: vLLM mirrors NODE_IP into VLLM_HOST_IP
# ---------------------------------------------------------------------------


def test_finalize_host_comm_env_mirrors_node_ip_into_vllm_host_ip():
    """Both vLLM runtimes advertise VLLM_HOST_IP equal to the host's NODE_IP."""
    for runtime in (VllmDistributedRuntime(), VllmRayRuntime()):
        host_env = {"NODE_IP": "192.168.0.155", "GLOO_SOCKET_IFNAME": "cx0"}
        out = runtime.finalize_host_comm_env(host_env)
        assert out["VLLM_HOST_IP"] == "192.168.0.155"
        # NODE_IP and the rest are preserved; input is not mutated.
        assert out["NODE_IP"] == "192.168.0.155"
        assert out["GLOO_SOCKET_IFNAME"] == "cx0"
        assert "VLLM_HOST_IP" not in host_env


def test_finalize_host_comm_env_no_node_ip_leaves_env_alone():
    """Without a NODE_IP there is nothing to advertise, so VLLM_HOST_IP is unset."""
    out = VllmDistributedRuntime().finalize_host_comm_env({"NCCL_NET": "IB"})
    assert "VLLM_HOST_IP" not in out


def test_finalize_host_comm_env_base_runtime_is_noop():
    """Non-vLLM runtimes do not invent VLLM_HOST_IP from NODE_IP."""
    from sparkrun.runtimes.sglang import SglangRuntime

    host_env = {"NODE_IP": "192.168.0.155"}
    out = SglangRuntime().finalize_host_comm_env(host_env)
    assert "VLLM_HOST_IP" not in out
    assert out == host_env


def test_serve_flags_emitted_without_command_template():
    """Serving-behaviour keys reach the generated command (no ``command:`` block).

    Regression guard for the flag-map gaps: these were previously absent from
    ``VLLM_FLAG_MAP``, so a command-less recipe silently served a
    differently-configured server — no tool/reasoning parsing, default
    attention backend, default weight loader — with nothing reported.

    ``enable_auto_tool_choice`` is the sharpest case: it was listed in
    ``VLLM_BOOL_FLAGS`` but *not* in ``VLLM_FLAG_MAP``, and
    ``build_flags_from_map`` iterates the map, so it was unreachable.
    """
    recipe = Recipe.from_dict(
        {
            "name": "flags",
            "model": "Qwen/Qwen3.8-27B-FP8",
            "runtime": "vllm",
            "defaults": {
                "port": 8000,
                "attention_backend": "flashinfer",
                "load_format": "instanttensor",
                "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen3_coder",
                "enable_auto_tool_choice": True,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "async_scheduling": True,
                "speculative_config": '{"method": "mtp", "num_speculative_tokens": 3}',
                "tokenizer_mode": "auto",
                "seed": 0,
            },
        }
    )

    cmd = VllmDistributedRuntime().generate_command(recipe, {}, is_cluster=False)
    for expected in (
        "--attention-backend flashinfer",
        "--load-format instanttensor",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--enable-auto-tool-choice",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--async-scheduling",
        "--tokenizer-mode auto",
        "--seed 0",
    ):
        assert expected in cmd, "missing %r in: %s" % (expected, cmd)


def test_bool_flags_are_all_reachable():
    """Every boolean key must also appear in the flag map.

    ``build_flags_from_map`` iterates ``VLLM_FLAG_MAP`` and consults
    ``bool_keys`` only to decide how to *render* a key it already found, so a
    key listed solely in ``VLLM_BOOL_FLAGS`` is dead code that silently emits
    nothing.  This guards the two lists staying in sync.
    """
    from sparkrun.runtimes._vllm_common import VLLM_BOOL_FLAGS, VLLM_FLAG_MAP

    unreachable = sorted(k for k in VLLM_BOOL_FLAGS if k not in VLLM_FLAG_MAP)
    assert not unreachable, "bool keys missing from VLLM_FLAG_MAP: %s" % unreachable
