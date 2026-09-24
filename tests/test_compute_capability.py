"""Compute capability: platform declaration first, inventory as fallback and cross-check."""

from __future__ import annotations

import pytest

from sparkrun.core.fingerprint import build_host_hardware, compute_fingerprint_hash, parse_fingerprint_output
from sparkrun.core.hardware import (
    AcceleratorSpec,
    HostHardware,
    compute_capability_to_arch,
    default_dgx_spark_hardware,
    normalize_arch,
    normalize_compute_capability,
)
from sparkrun.platforms import DgxSparkPlatform, GenericNvidiaPlatform, resolve_accelerator_arch, resolve_compute_capability


def _host(model: str, cc: str | None = None, *, vendor: str = "nvidia", source: str = "detected") -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor=vendor, model=model, compute_capability=cc)], source=source)


@pytest.mark.parametrize(
    "raw,want",
    [("12.1", "12.1"), (12.1, "12.1"), (12.0, "12.0"), (" 9.0 ", "9.0"), ("10.3", "10.3")],
)
def test_normalize_compute_capability_accepts_probe_and_yaml_spellings(raw, want):
    assert normalize_compute_capability(raw) == want


@pytest.mark.parametrize("raw", [None, True, "", "[N/A]", "12", "sm_121", "12.10", "abc"])
def test_normalize_compute_capability_rejects_non_capabilities(raw):
    assert normalize_compute_capability(raw) is None


def test_compute_capability_to_arch():
    assert compute_capability_to_arch("12.1") == "sm_121"
    assert compute_capability_to_arch("9.0") == "sm_90"
    assert compute_capability_to_arch("10.3") == "sm_103"
    assert compute_capability_to_arch(None) is None


@pytest.mark.parametrize(
    "raw,want",
    [
        ("sm_120a", "sm_120"),  # arch-specific feature suffix (lil / CuTe spelling)
        ("sm_100f", "sm_100"),  # family suffix
        ("SM_121", "sm_121"),
        ("sm121", "sm_121"),
        ("sm_90", "sm_90"),
        ("12.0", "sm_120"),  # a compute capability is also accepted
        ("gb10", None),
        (None, None),
    ],
)
def test_normalize_arch(raw, want):
    assert normalize_arch(raw) == want


def test_accelerator_spec_round_trips_compute_capability():
    spec = AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000", compute_capability="12.0")
    assert spec.to_dict()["compute_capability"] == "12.0"
    assert AcceleratorSpec.from_dict(spec.to_dict()) == spec


def test_accelerator_spec_omits_unknown_compute_capability():
    assert "compute_capability" not in AcceleratorSpec(vendor="nvidia", model="gb10").to_dict()


def test_hand_written_numeric_inventory_value_is_normalized():
    """YAML reads ``compute_capability: 12.0`` as a float."""
    assert AcceleratorSpec.from_dict({"vendor": "nvidia", "model": "x", "compute_capability": 12.0}).compute_capability == "12.0"


# --- platform declarations -------------------------------------------------


def test_dgx_spark_declares_gb10():
    assert DgxSparkPlatform().default_compute_capability(AcceleratorSpec(vendor="nvidia", model="gb10")) == "12.1"
    assert DgxSparkPlatform().default_compute_capability(AcceleratorSpec(vendor="nvidia", model="h100")) is None


@pytest.mark.parametrize(
    "model,want",
    [
        ("h100", "9.0"),
        ("h100-80gb-hbm3", "9.0"),  # suffixes match through the prefix rule
        ("h200", "9.0"),
        ("b200", "10.0"),
        ("gb300", "10.3"),
        ("rtx-pro-6000", "12.0"),
        ("rtx-pro-6000-blackwell-workstation-edition", "12.0"),
        ("rtx-pro-6000-blackwell-max-q-workstation-edition", "12.0"),
        ("geforce-rtx-5090", "12.0"),
        ("l40s", "8.9"),
        ("l4", "8.9"),
    ],
)
def test_generic_nvidia_declares_known_models(model, want):
    assert GenericNvidiaPlatform().default_compute_capability(AcceleratorSpec(vendor="nvidia", model=model)) == want


@pytest.mark.parametrize(
    "model",
    [
        "rtx-6000-ada-generation",  # Ada, not "RTX PRO"
        "rtx-a6000",
        "l40",  # "l4" must not claim "l40" (whole-token prefix only)
        "h1000",  # "h100" must not claim a longer token
        "unknown-gpu",
    ],
)
def test_generic_nvidia_does_not_guess(model):
    assert GenericNvidiaPlatform().default_compute_capability(AcceleratorSpec(vendor="nvidia", model=model)) is None


def test_generic_nvidia_ignores_other_vendors():
    assert GenericNvidiaPlatform().default_compute_capability(AcceleratorSpec(vendor="amd", model="h100")) is None


# --- resolution --------------------------------------------------------------


def test_resolution_prefers_platform_declaration_over_inventory():
    hw = _host("gb10", "12.0")
    assert resolve_compute_capability(hw.accelerators[0], hw) == ("12.1", "platform dgx-spark")


def test_resolution_falls_back_to_inventory_for_undeclared_models():
    hw = _host("rtx-a6000", "8.6")
    assert resolve_compute_capability(hw.accelerators[0], hw) == ("8.6", "detected")
    assert resolve_accelerator_arch(hw.accelerators[0], hw) == "sm_86"


def test_resolution_unknown_when_neither_source_knows():
    hw = _host("rtx-a6000")
    assert resolve_compute_capability(hw.accelerators[0], hw) == (None, "unknown")
    assert resolve_accelerator_arch(hw.accelerators[0], hw) is None


def test_assumed_dgx_spark_hardware_resolves_without_a_probe():
    """Arch must be known for --dry-run / pre-placement, where nothing was probed."""
    hw = default_dgx_spark_hardware()
    assert resolve_accelerator_arch(hw.accelerators[0], hw) == "sm_121"


# --- cross-check ---------------------------------------------------------------


def test_dgx_spark_warns_when_probe_disagrees_with_declaration():
    warnings = DgxSparkPlatform().validate_host(_host("gb10", "12.0"))
    assert any("declares compute capability 12.1" in w and "reports 12.0" in w for w in warnings)


def test_dgx_spark_silent_when_probe_agrees_or_is_absent():
    for cc in ("12.1", None):
        assert not [w for w in DgxSparkPlatform().validate_host(_host("gb10", cc)) if "compute capability" in w]


def test_generic_nvidia_warns_on_disagreement():
    warnings = GenericNvidiaPlatform().validate_host(_host("h100-80gb-hbm3", "8.0"))
    assert any("declares compute capability 9.0" in w for w in warnings)


def test_generic_nvidia_undeclared_model_never_warns():
    assert GenericNvidiaPlatform().validate_host(_host("rtx-a6000", "8.6")) == []


# --- probe ---------------------------------------------------------------------


def _probe_output(*gpus: tuple[str, str, str | None]) -> str:
    lines = ["NVIDIA_GPU_COUNT=%d" % len(gpus), "NVIDIA_PRESENT=1"]
    for i, (name, mib, cc) in enumerate(gpus):
        lines += ["NVIDIA_GPU_%d_NAME=%s" % (i, name), "NVIDIA_GPU_%d_MEMORY_MIB=%s" % (i, mib)]
        if cc is not None:
            lines.append("NVIDIA_GPU_%d_COMPUTE_CAP=%s" % (i, cc))
    return "\n".join(lines) + "\n"


def test_probe_records_compute_capability():
    hw = build_host_hardware(parse_fingerprint_output(_probe_output(("NVIDIA GB10", "[N/A]", "12.1"))))
    assert hw.accelerators[0].compute_capability == "12.1"


def test_probe_without_compute_cap_still_detects_gpu():
    """Older drivers reject the compute_cap query; detection must not depend on it."""
    hw = build_host_hardware(parse_fingerprint_output(_probe_output(("NVIDIA H100 80GB HBM3", "81559", None))))
    assert hw.accelerators[0].model == "h100-80gb-hbm3"
    assert hw.accelerators[0].compute_capability is None


def test_probe_groups_identical_gpus_and_splits_on_capability():
    same = build_host_hardware(parse_fingerprint_output(_probe_output(*[("NVIDIA H100", "81559", "9.0")] * 2)))
    assert [(a.count, a.compute_capability) for a in same.accelerators] == [(2, "9.0")]
    mixed = build_host_hardware(parse_fingerprint_output(_probe_output(("NVIDIA X", "1000", "8.6"), ("NVIDIA X", "1000", "8.9"))))
    assert [a.compute_capability for a in mixed.accelerators] == ["8.6", "8.9"]


def test_fingerprint_hash_ignores_compute_capability():
    """The capability is implied by the model; adding the probe field must not re-key every host."""
    base = [AcceleratorSpec(vendor="nvidia", model="gb10")]
    probed = [AcceleratorSpec(vendor="nvidia", model="gb10", compute_capability="12.1")]
    assert compute_fingerprint_hash(base) == compute_fingerprint_hash(probed)


def test_both_probe_scripts_query_compute_cap_separately():
    """The combined probe keeps its own brace-escaped copy of the GPU query; both must carry it."""
    from sparkrun.core.fingerprint import generate_fingerprint_script
    from sparkrun.core.hardware_probe import generate_combined_probe_script

    for script in (generate_fingerprint_script(), generate_combined_probe_script()):
        assert "--query-gpu=compute_cap " in script
        assert "NVIDIA_GPU_${CC_INDEX}_COMPUTE_CAP" in script
        assert "--query-gpu=name,memory.total " in script  # untouched: no compute_cap in the main query


# --- platform env ----------------------------------------------------------------


def test_dgx_spark_publishes_cute_dsl_arch_for_vllm_and_sglang():
    gb10 = AcceleratorSpec(vendor="nvidia", model="gb10")
    for runtime, family in (("vllm-distributed", "vllm"), ("sglang", "sglang")):
        assert DgxSparkPlatform().default_env(runtime, gb10, runtime_family=family)["CUTE_DSL_ARCH"] == "sm_121a"
    assert "CUTE_DSL_ARCH" not in DgxSparkPlatform().default_env("llama-cpp", gb10, runtime_family="llama-cpp")


@pytest.mark.parametrize(
    "model,cc,want",
    [
        ("rtx-pro-6000-blackwell-workstation-edition", None, "sm_120a"),  # declared
        ("h100-80gb-hbm3", None, "sm_90a"),
        ("b200", None, "sm_100a"),
        ("some-new-gpu", "10.3", "sm_103a"),  # undeclared: inventory fills in
    ],
)
def test_generic_nvidia_publishes_cute_dsl_arch(model, cc, want):
    accel = AcceleratorSpec(vendor="nvidia", model=model, compute_capability=cc)
    assert GenericNvidiaPlatform().default_env("vllm-distributed", accel, runtime_family="vllm") == {"CUTE_DSL_ARCH": want}
    assert GenericNvidiaPlatform().default_env("sglang", accel, runtime_family="sglang") == {"CUTE_DSL_ARCH": want}


def test_generic_nvidia_no_cute_dsl_arch_below_sm90_or_for_other_runtimes():
    """``sm_80a`` does not exist; llama.cpp / TRT-LLM never JIT CuTe DSL kernels."""
    a100 = AcceleratorSpec(vendor="nvidia", model="a100")
    assert GenericNvidiaPlatform().default_env("vllm-distributed", a100, runtime_family="vllm") == {}
    h100 = AcceleratorSpec(vendor="nvidia", model="h100")
    assert GenericNvidiaPlatform().default_env("llama-cpp", h100, runtime_family="llama-cpp") == {}


# --- restated platform env (validation) --------------------------------------------------


def _vllm_recipe(env=None, overrides=None):
    from sparkrun.core.recipe import Recipe

    data = {"model": "org/model", "runtime": "vllm-distributed", "env": env or {}}
    if overrides:
        data["overrides"] = overrides
    return Recipe.from_dict(data)


def _restated(recipe):
    from sparkrun.core.validation import check_restated_platform_env
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    return check_restated_platform_env(recipe, VllmDistributedRuntime())


def test_restating_a_platform_env_value_is_a_suggestion():
    from sparkrun.core.validation import SUGGESTION

    [issue] = _restated(_vllm_recipe(env={"CUTE_DSL_ARCH": "sm_121a"}))
    assert (issue.severity, issue.code) == (SUGGESTION, "restated-platform-env")
    assert "DGX Spark (gb10)" in issue.summary


def test_generic_platform_values_are_recognized_too():
    [issue] = _restated(_vllm_recipe(env={"CUTE_DSL_ARCH": "sm_120a"}))
    assert "rtx-pro-6000" in issue.summary


def test_a_deliberately_different_value_is_never_flagged():
    assert _restated(_vllm_recipe(env={"CUTE_DSL_ARCH": "sm_100f", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"})) == []


def test_override_env_layers_are_checked():
    [issue] = _restated(_vllm_recipe(overrides=[{"when": {"arch": "sm_120"}, "env": {"CUTE_DSL_ARCH": "sm_120a"}}]))
    assert issue.summary.startswith("overrides[0].env:") and "conditional layer" in issue.summary


def test_platform_env_for_other_runtimes_is_not_matched():
    """GB10 publishes nothing for llama.cpp, so the same key there is not a restatement."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.validation import check_restated_platform_env
    from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

    recipe = Recipe.from_dict({"model": "org/m-GGUF:Q4_K_M", "runtime": "llama-cpp", "env": {"CUTE_DSL_ARCH": "sm_121a"}})
    assert check_restated_platform_env(recipe, LlamaCppRuntime()) == []
