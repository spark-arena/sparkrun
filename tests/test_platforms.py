"""Tests for HardwarePlatformPlugin + concrete platforms (Phase 8)."""

from __future__ import annotations

import pytest

from sparkrun.core.hardware import AcceleratorSpec, HostHardware, default_dgx_spark_hardware
from sparkrun.orchestration.collectives import NcclBackend
from sparkrun.platforms import (
    DgxSparkPlatform,
    GenericNvidiaPlatform,
    HardwarePlatformPlugin,
    get_platform_by_name,
    iter_platforms,
    register_platform,
    resolve_platform,
)


def _nvidia(model: str, *, memory_gb: float | None = 80.0) -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model=model, memory_gb=memory_gb, capabilities=frozenset({"cuda"}))])


def _amd(model: str = "mi300x") -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model=model, capabilities=frozenset({"rocm"}))])


# --------------------------------------------------------------------------
# Plugin contract
# --------------------------------------------------------------------------


def test_dgx_spark_platform_metadata():
    p = DgxSparkPlatform()
    assert p.platform_name == "dgx-spark"
    assert p.vendors == frozenset({"nvidia"})
    assert p.name() == "sparkrun.platform.dgx-spark"


def test_generic_nvidia_platform_metadata():
    p = GenericNvidiaPlatform()
    assert p.platform_name == "nvidia-generic"
    assert p.vendors == frozenset({"nvidia"})


def test_platform_is_multi_extension():
    """Mirrors RuntimePlugin: multi-extension marker + extension-point name."""
    p = DgxSparkPlatform()
    assert p.is_multi_extension(None) is True
    assert p.extension_point_name(None) == "sparkrun.platform"
    # SAF requires is_enabled False for multi-extension plugins
    assert p.is_enabled(None) is False


# --------------------------------------------------------------------------
# DgxSparkPlatform
# --------------------------------------------------------------------------


def test_dgx_spark_matches_gb10_default_hardware():
    """The library's DGX Spark default fingerprint is recognised."""
    assert DgxSparkPlatform().matches(default_dgx_spark_hardware()) is True


def test_dgx_spark_matches_gb10_explicit():
    assert DgxSparkPlatform().matches(_nvidia("gb10")) is True


def test_dgx_spark_does_not_match_h100():
    assert DgxSparkPlatform().matches(_nvidia("h100")) is False


def test_dgx_spark_does_not_match_amd():
    assert DgxSparkPlatform().matches(_amd()) is False


def test_dgx_spark_collective_is_nccl():
    assert isinstance(DgxSparkPlatform().collective_backend(), NcclBackend)


def test_dgx_spark_accelerator_vendor():
    assert DgxSparkPlatform().accelerator_vendor() == "nvidia"


def test_dgx_spark_default_images():
    p = DgxSparkPlatform()
    assert p.default_image("atlas") == "avarok/atlas-gb10:latest"
    assert p.default_image("vllm-distributed") == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"
    assert p.default_image("sglang") == "scitrera/dgx-spark-sglang:latest"
    assert p.default_image("nonexistent-runtime") is None


# --------------------------------------------------------------------------
# GenericNvidiaPlatform
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["h100", "h200", "b200", "gb10", "rtx-pro-6000", "a100"])
def test_generic_nvidia_matches_any_nvidia_host(model: str):
    assert GenericNvidiaPlatform().matches(_nvidia(model)) is True


def test_generic_nvidia_does_not_match_amd():
    assert GenericNvidiaPlatform().matches(_amd()) is False


def test_generic_nvidia_does_not_match_empty_host():
    assert GenericNvidiaPlatform().matches(HostHardware()) is False


def test_generic_nvidia_default_images():
    p = GenericNvidiaPlatform()
    assert p.default_image("vllm-distributed") == "vllm/vllm-openai:latest"
    assert p.default_image("sglang") == "lmsysorg/sglang:latest"
    # Atlas/eugr deliberately omitted from generic defaults — Phase 7 gates them.
    assert p.default_image("atlas") is None


# --------------------------------------------------------------------------
# Registry / resolve_platform
# --------------------------------------------------------------------------


def test_resolve_platform_prefers_dgx_for_gb10():
    """Both DgxSpark + GenericNvidia match GB10; resolution returns DgxSpark."""
    p = resolve_platform(_nvidia("gb10"))
    assert isinstance(p, DgxSparkPlatform)


def test_resolve_platform_falls_back_to_generic_for_h100():
    p = resolve_platform(_nvidia("h100"))
    assert isinstance(p, GenericNvidiaPlatform)


def test_resolve_platform_returns_none_for_amd():
    """No AMD platform shipped yet — resolver returns None instead of guessing."""
    assert resolve_platform(_amd()) is None


def test_resolve_platform_returns_none_for_empty_host():
    assert resolve_platform(HostHardware()) is None


def test_iter_platforms_returns_fresh_list():
    """Caller can mutate the returned list without polluting the registry."""
    a = iter_platforms()
    a.clear()
    b = iter_platforms()
    assert len(b) >= 2  # at least dgx-spark + nvidia-generic


def test_get_platform_by_name():
    assert isinstance(get_platform_by_name("dgx-spark"), DgxSparkPlatform)
    assert isinstance(get_platform_by_name("nvidia-generic"), GenericNvidiaPlatform)
    assert get_platform_by_name("nonexistent") is None


# --------------------------------------------------------------------------
# register_platform extensibility
# --------------------------------------------------------------------------


class _FakePlatform(HardwarePlatformPlugin):
    platform_name = "fake"
    vendors = frozenset({"amd"})

    def matches(self, host_hardware):
        return any(a.vendor == "amd" for a in host_hardware.accelerators)

    def accelerator_vendor(self):
        return "amd"

    def collective_backend(self):
        from sparkrun.orchestration.collectives import RcclBackend

        return RcclBackend()


@pytest.fixture
def _isolate_registry():
    """Snapshot the registry around tests that mutate it."""
    from sparkrun import platforms as plat_mod

    snapshot = list(plat_mod._REGISTRY)
    yield
    plat_mod._REGISTRY[:] = snapshot


def test_register_platform_append_default(_isolate_registry):
    instance = _FakePlatform()
    register_platform(instance)
    assert iter_platforms()[-1] is instance


def test_register_platform_prepend_wins_resolution(_isolate_registry):
    instance = _FakePlatform()
    register_platform(instance, prepend=True)
    # AMD host now resolves to the registered fake platform.
    p = resolve_platform(_amd())
    assert p is instance


def test_register_platform_does_not_disturb_existing_resolution(_isolate_registry):
    """Appending leaves NVIDIA resolution unchanged."""
    register_platform(_FakePlatform())
    assert isinstance(resolve_platform(_nvidia("gb10")), DgxSparkPlatform)


# --------------------------------------------------------------------------
# validate_host — base default
# --------------------------------------------------------------------------


class _MinimalPlatform(HardwarePlatformPlugin):
    """Stub that uses only the base-class validate_host (no override)."""

    platform_name = "minimal-stub"
    vendors = frozenset({"nvidia"})

    def matches(self, host_hardware):
        return True

    def accelerator_vendor(self):
        return "nvidia"

    def collective_backend(self):
        from sparkrun.orchestration.collectives import NcclBackend

        return NcclBackend()


def test_validate_host_default_returns_empty():
    """Base-class default implementation always returns an empty list."""
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100")])
    assert _MinimalPlatform().validate_host(hw) == []


# --------------------------------------------------------------------------
# validate_host — DgxSparkPlatform
# --------------------------------------------------------------------------


def _dgx_spark_hw(*, with_roce: bool = True) -> HostHardware:
    caps = frozenset({"cuda", "unified-memory", "rdma:roce-v2"}) if with_roce else frozenset({"cuda", "unified-memory"})
    return HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0, capabilities=caps)])


def test_dgx_spark_validate_host_happy_path():
    """GB10 + RoCEv2 → no warnings."""
    assert DgxSparkPlatform().validate_host(_dgx_spark_hw(with_roce=True)) == []


def test_dgx_spark_validate_host_missing_roce():
    """GB10 without RoCEv2 capability → exactly one warning mentioning ConnectX-7."""
    warnings = DgxSparkPlatform().validate_host(_dgx_spark_hw(with_roce=False))
    assert len(warnings) == 1
    assert "rdma:roce-v2" in warnings[0]
    assert "ConnectX-7" in warnings[0]


def test_dgx_spark_validate_host_wrong_model():
    """Non-GB10 NVIDIA host matched (edge case) → warning about unexpected model."""
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", capabilities=frozenset({"cuda"}))])
    warnings = DgxSparkPlatform().validate_host(hw)
    assert len(warnings) >= 1
    assert "h100" in warnings[0]


def test_dgx_spark_validate_host_default_hardware_is_clean():
    """default_dgx_spark_hardware() should validate cleanly — it has RoCEv2."""
    assert DgxSparkPlatform().validate_host(default_dgx_spark_hardware()) == []


# --------------------------------------------------------------------------
# validate_host — GenericNvidiaPlatform
# --------------------------------------------------------------------------


def test_nvidia_generic_validate_host_happy_path():
    """Any NVIDIA host → no warnings."""
    assert GenericNvidiaPlatform().validate_host(_nvidia("h100")) == []


def test_nvidia_generic_validate_host_amd_vendor():
    """Non-NVIDIA host (e.g. AMD) passed to GenericNvidiaPlatform → warning."""
    hw = _amd("mi300x")
    warnings = GenericNvidiaPlatform().validate_host(hw)
    assert len(warnings) == 1
    assert "amd" in warnings[0]
    assert "NVIDIA" in warnings[0]


def test_nvidia_generic_validate_host_no_accelerators():
    """Host with no accelerators → warning about missing vendor."""
    hw = HostHardware()
    warnings = GenericNvidiaPlatform().validate_host(hw)
    assert len(warnings) == 1
    assert "none" in warnings[0]


# --------------------------------------------------------------------------
# default_max_gpu_memory_utilization (usable-memory cap, platform tier)
# --------------------------------------------------------------------------


def test_dgx_default_max_gpu_memory_utilization_for_gb10():
    """DGX Spark caps GB10 usable memory at 0.85 (unified memory headroom)."""
    accel = AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0)
    assert DgxSparkPlatform().default_max_gpu_memory_utilization(accel) == 0.85


def test_dgx_default_max_gpu_memory_utilization_other_model_is_none():
    """A non-GB10 accelerator gets no DGX default."""
    accel = AcceleratorSpec(vendor="nvidia", model="h200", memory_gb=141.0)
    assert DgxSparkPlatform().default_max_gpu_memory_utilization(accel) is None


def test_generic_nvidia_default_max_gpu_memory_utilization_is_none():
    """Generic NVIDIA publishes no cap → resolution falls through to 1.0."""
    accel = AcceleratorSpec(vendor="nvidia", model="h100", memory_gb=80.0)
    assert GenericNvidiaPlatform().default_max_gpu_memory_utilization(accel) is None
