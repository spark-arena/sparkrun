"""One runtime/recipe declaration feeds routing and inference readiness."""

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.core.readiness import (
    ANTHROPIC_MESSAGES_STREAM,
    OPENAI_CHAT_STREAM,
    OPENAI_RESPONSES_STREAM,
    ReadinessSettings,
    resolve_inference_style,
    validate_readiness_policy,
)
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.runtimes.vllm_ray import VllmRayRuntime


def recipe(container="ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest", **metadata):
    return Recipe({"recipe_version": "2", "model": "test/model", "runtime": "vllm", "container": container, "metadata": metadata})


@pytest.mark.parametrize("runtime", [VllmDistributedRuntime(), VllmRayRuntime()])
@pytest.mark.parametrize("image", ["nightly:latest", "custom@sha256:abc", "vllm:v0.12.0", "vllm:v0.20.2"])
def test_current_vllm_family_defaults_include_all_three_apis(runtime, image):
    value = recipe(image)
    assert runtime.native_apis(value) == ["chat_completions", "responses", "messages"]
    assert runtime.native_protocols(value) == ["openai", "anthropic"]
    assert runtime.native_capabilities(value) == ["responses"]
    assert resolve_inference_style(ReadinessSettings(), runtime, recipe=value) == OPENAI_CHAT_STREAM


def test_known_old_image_keeps_chat_floor_and_explicit_recipe_wins():
    runtime = VllmDistributedRuntime()
    assert runtime.native_apis(recipe("vllm:v0.11.0")) == ["chat_completions"]
    assert runtime.native_apis(recipe("vllm:v0.11.0", native_apis=["chat_completions", "responses"])) == ["chat_completions", "responses"]
    narrow = recipe(native_apis=["chat_completions"])
    assert runtime.native_protocols(narrow) == ["openai"]
    assert runtime.native_capabilities(narrow) == []
    for style in [OPENAI_RESPONSES_STREAM, ANTHROPIC_MESSAGES_STREAM]:
        with pytest.raises(ValueError, match="does not support"):
            resolve_inference_style(ReadinessSettings(inference_style=style), runtime, recipe=narrow)


def test_auto_selects_only_an_api_declared_by_the_recipe():
    value = recipe(native_apis=["messages"])
    runtime = VllmDistributedRuntime()
    assert runtime.native_protocols(value) == ["anthropic"]
    assert resolve_inference_style(ReadinessSettings(), runtime, recipe=value) == ANTHROPIC_MESSAGES_STREAM
    assert (
        resolve_inference_style(ReadinessSettings(inference_style=OPENAI_RESPONSES_STREAM), runtime, recipe=recipe())
        == OPENAI_RESPONSES_STREAM
    )


@pytest.mark.parametrize("apis", [[], "responses", ["typo"], ["responses"], [None], [42]])
def test_invalid_declarations_fail_policy_validation_even_with_inference_disabled(apis):
    value = recipe(native_apis=apis)
    value.readiness = {"inference": False}
    with pytest.raises(ValueError, match="native_apis|responses requires"):
        validate_readiness_policy(recipe=value, runtime=VllmDistributedRuntime())
