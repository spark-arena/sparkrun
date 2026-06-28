from __future__ import annotations

import json
import time

import sparkrun.api as api
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.recipe import Recipe
from sparkrun.telemetry.benchmark import build_benchmark_event
from sparkrun.telemetry.events import build_run_event


def test_run_event_reads_quantization_after_recipe_estimate_vram(monkeypatch):
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "org/model"})
    calls = []

    def _estimate_vram(cli_overrides=None, auto_detect=True, cache_dir=None):
        calls.append((cli_overrides, auto_detect, cache_dir))
        recipe.metadata.update({"quantization": "nvfp4", "quant_bits": 4, "model_dtype": "nvfp4", "kv_dtype": "fp8"})

    monkeypatch.setattr(recipe, "estimate_vram", _estimate_vram)
    result = api.RunResult(
        cluster_id="sparkrun_deadbeef_deadbeef",
        host_list=("h1",),
        placement=None,
        scheduler="greedy",
        runtime="vllm",
        executor="docker",
        started_at=time.time(),
        dry_run=True,
        is_solo=True,
        rc=0,
        metadata={},
    )
    options = api.RunOptions(recipe=recipe, hosts=("h1",), overrides={"tensor_parallel": 2})

    event = build_run_event(
        result=result,
        recipe=recipe,
        cluster=ClusterDefinition(name="lab", hosts=["h1"]),
        options=options,
    )

    assert calls == [({"tensor_parallel": 2}, True, None)]
    assert event["model_quantization"] == {"quantization": "nvfp4", "quant_bits": 4, "model_dtype": "nvfp4", "kv_dtype": "fp8"}


def test_benchmark_event_uses_result_quantization_when_recipe_is_unresolved():
    result = api.BenchmarkResult(
        success=True,
        benchmark_id="bench_secret",
        category="performance",
        framework="llama-benchy",
        profile=None,
        metadata={
            "model_quantization": {
                "quantization": "gguf",
                "model_dtype": "q4_k",
                "kv_dtype": "q8_0",
            }
        },
    )
    options = api.BenchmarkOptions(recipe="private-recipe-ref")

    event = build_benchmark_event(result=result, options=options)

    assert event["model_quantization"] == {"quantization": "gguf", "model_dtype": "q4_k", "kv_dtype": "q8_0"}
    assert "private-recipe-ref" not in json.dumps(event)
