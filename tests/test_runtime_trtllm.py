"""Unit tests for sparkrun.runtimes.trtllm (TrtllmRuntime)."""

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.trtllm import TrtllmRuntime


class TestTrtllmComputeRequiredNodes:
    """Test TrtllmRuntime inherits base tp*pp behavior."""

    def _make_recipe(self, defaults=None):
        data = {
            "name": "test",
            "runtime": "trtllm",
            "model": "meta-llama/Llama-2-7b-hf",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_tp_times_pp(self):
        """TRT-LLM inherits base class tp*pp."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = TrtllmRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_pp_only(self):
        """PP=2 with no TP → 2 nodes."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 2})
        runtime = TrtllmRuntime()
        assert runtime.compute_required_nodes(recipe) == 2

    def test_returns_none_when_neither(self):
        """No TP or PP → None."""
        recipe = self._make_recipe()
        runtime = TrtllmRuntime()
        assert runtime.compute_required_nodes(recipe) is None
