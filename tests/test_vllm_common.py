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


def test_estimate_vram_handles_auto_max_model_len():
    """VRAM estimation must not choke on ``max_model_len: auto`` (no int('auto'))."""
    # auto_detect=False keeps this offline; the max_model_len guard still runs.
    est = _recipe().estimate_vram(auto_detect=False)
    assert est is not None
