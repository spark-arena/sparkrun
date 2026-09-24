"""Structured serve-flag rendering: JSON values, negatable booleans, and the B12X/lil vLLM surface."""

from __future__ import annotations

import shlex

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.base import RuntimePlugin, render_flag_value
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime


@pytest.mark.parametrize(
    "value,want",
    [
        ({"method": "mtp", "num_speculative_tokens": 3}, '\'{"method":"mtp","num_speculative_tokens":3}\''),
        (["all"], "'[\"all\"]'"),
        ('{"reasoning_effort": "medium"}', '\'{"reasoning_effort": "medium"}\''),  # JSON string: quoted, kept verbatim
        ("0.9", "0.9"),
        (8192, "8192"),
        ("$HOME/templates/chat.jinja", "$HOME/templates/chat.jinja"),  # plain values keep host-side expansion
    ],
)
def test_render_flag_value(value, want):
    assert render_flag_value(value) == want


def test_rendered_json_survives_the_shell():
    """What the engine receives after bash word-splitting is exactly one valid JSON argument."""
    import json

    value = {"method": "dflash", "model": "org/draft's-name", "nested": {"a": [1, 2]}}
    [arg] = shlex.split(render_flag_value(value))
    assert json.loads(arg) == value


def test_negatable_booleans_render_no_form_only_when_listed():
    config = {"on": True, "off": False, "plain_off": False}
    flags = RuntimePlugin.build_flags_from_map(
        config,
        {"on": "--on", "off": "--off", "plain_off": "--plain-off"},
        bool_keys={"on", "off", "plain_off"},
        negatable_keys={"on", "off"},
    )
    assert flags == ["--on", "--no-off"]  # default-off flags stay omitted on False


def _vllm_command(defaults: dict) -> str:
    recipe = Recipe.from_dict({"model": "org/model", "runtime": "vllm", "defaults": {"port": 8000, **defaults}})
    return VllmDistributedRuntime().generate_command(recipe, {}, is_cluster=False)


def test_speculative_config_mapping_renders_as_json():
    cmd = _vllm_command({"speculative_config": {"method": "mtp", "num_speculative_tokens": 3}})
    assert '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\'' in cmd
    assert "{'method'" not in cmd


@pytest.mark.parametrize(
    "key,flag",
    [
        ("enable_prefix_caching", "--no-enable-prefix-caching"),
        ("enable_chunked_prefill", "--no-enable-chunked-prefill"),
        ("async_scheduling", "--no-async-scheduling"),
        ("enable_flashinfer_autotune", "--no-enable-flashinfer-autotune"),
        ("scheduler_reserve_full_isl", "--no-scheduler-reserve-full-isl"),
    ],
)
def test_default_on_vllm_flags_can_be_turned_off(key, flag):
    assert flag in _vllm_command({key: False}).split()


@pytest.mark.parametrize("key", ["trust_remote_code", "enforce_eager", "enable_auto_tool_choice", "enable_request_id_headers"])
def test_default_off_vllm_flags_are_omitted_when_false(key):
    cmd = _vllm_command({key: False}).split()
    assert not any(part.endswith(key.replace("_", "-")) for part in cmd)


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("linear_backend", "b12x", "--linear-backend b12x"),
        ("moe_backend", "b12x", "--moe-backend b12x"),
        ("gdn_decode_kernel", "b12x", "--gdn-decode-kernel b12x"),
        ("gdn_prefill_backend", "b12x", "--gdn-prefill-backend b12x"),
        ("mamba_cache_mode", "align", "--mamba-cache-mode align"),
        ("prefill_policy", "decode-aware", "--prefill-policy decode-aware"),
        ("prefill_compute_share", "auto", "--prefill-compute-share auto"),
        ("decode_refill_target", "auto", "--decode-refill-target auto"),
        ("max_parallel_prefills", "auto", "--max-parallel-prefills auto"),
        ("generation_config", "vllm", "--generation-config vllm"),
        ("kv_cache_memory_bytes", "2G", "--kv-cache-memory-bytes 2G"),
        ("mm_processor_cache_gb", 0, "--mm-processor-cache-gb 0"),
        (
            "compilation_config",
            {"cudagraph_mode": "FULL_AND_PIECEWISE"},
            '--compilation-config \'{"cudagraph_mode":"FULL_AND_PIECEWISE"}\'',
        ),
        ("limit_mm_per_prompt", {"image": 1}, "--limit-mm-per-prompt '{\"image\":1}'"),
        ("default_chat_template_kwargs", {"thinking": True}, "--default-chat-template-kwargs '{\"thinking\":true}'"),
        ("enable_prompt_tokens_details", True, "--enable-prompt-tokens-details"),
    ],
)
def test_b12x_and_lil_flags_render(key, value, expected):
    assert expected in _vllm_command({key: value})


def test_new_flags_are_emitted_only_when_set():
    """A recipe that never names them renders as before (stock images reject fork flags)."""
    cmd = _vllm_command({})
    for flag in ("--linear-backend", "--prefill-policy", "--compilation-config", "--no-"):
        assert flag not in cmd


def test_fork_flags_are_no_longer_reported_as_dropped():
    from sparkrun.core.launcher import report_unmapped_config_keys

    recipe = Recipe.from_dict({"model": "org/model", "runtime": "vllm", "defaults": {"linear_backend": "b12x", "prefill_policy": "auto"}})
    assert report_unmapped_config_keys(recipe, VllmDistributedRuntime(), None, log=False) == []


def test_negatable_flags_are_reachable_booleans():
    from sparkrun.runtimes._vllm_common import VLLM_BOOL_FLAGS, VLLM_FLAG_MAP, VLLM_NEGATABLE_BOOL_FLAGS

    assert VLLM_NEGATABLE_BOOL_FLAGS <= VLLM_BOOL_FLAGS
    assert all(key in VLLM_FLAG_MAP for key in VLLM_NEGATABLE_BOOL_FLAGS)


def test_structured_commands_pass_a_pinned_revision():
    from sparkrun.runtimes.vllm_ray import VllmRayRuntime

    recipe = Recipe.from_dict({"model": "org/model", "model_revision": "a" * 40, "runtime": "vllm-distributed", "defaults": {"port": 8000}})
    assert ("--revision " + "a" * 40) in VllmDistributedRuntime().generate_command(recipe, {}, is_cluster=False)
    ray = Recipe.from_dict({"model": "org/model", "model_revision": "a" * 40, "runtime": "vllm-ray", "defaults": {"port": 8000}})
    assert ("--revision " + "a" * 40) in VllmRayRuntime().generate_command(ray, {}, is_cluster=False)


def test_no_revision_for_gguf_or_pre_placed_weights():
    gguf = Recipe.from_dict({"model": "org/model-GGUF:Q4_K_M", "model_revision": "a" * 40, "runtime": "vllm", "defaults": {"port": 8000}})
    assert "--revision" not in VllmDistributedRuntime().generate_command(gguf, {}, is_cluster=False)
    local = Recipe.from_dict({"model": "/models/qwen", "model_revision": "a" * 40, "runtime": "vllm", "defaults": {"port": 8000}})
    assert "--revision" not in VllmDistributedRuntime().generate_command(local, {}, is_cluster=False)


def test_unpinned_revision_warning_now_only_for_command_templates():
    from sparkrun.core.validation import check_unpinned_model_revision

    structured = Recipe.from_dict({"model": "org/model", "model_revision": "a" * 40, "runtime": "vllm", "defaults": {"port": 8000}})
    assert check_unpinned_model_revision(structured, VllmDistributedRuntime()) == []
    templated = Recipe.from_dict(
        {
            "model": "org/model",
            "model_revision": "a" * 40,
            "runtime": "vllm",
            "command": "vllm serve {model} --port {port}",
            "defaults": {"port": 8000},
        }
    )
    assert [i.code for i in check_unpinned_model_revision(templated, VllmDistributedRuntime())] == ["unpinned-model-revision"]
