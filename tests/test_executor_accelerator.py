"""Tests for vendor-branched executor device flags (Phase 4)."""

from __future__ import annotations

import pytest

from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.orchestration.executor import ExecutorConfig, accelerator_vendor_for
from sparkrun.orchestration.executors.docker import DockerExecutor


def _build_opts(vendor: str | None = None, gpus: str = "all") -> list[str]:
    cfg = ExecutorConfig(accelerator_vendor=vendor, gpus=gpus, privileged=False, network="")
    return DockerExecutor(cfg)._accelerator_opts()


# --------------------------------------------------------------------------
# Default (None) and explicit nvidia preserve legacy behavior
# --------------------------------------------------------------------------


def test_default_vendor_emits_gpus_all():
    """No vendor declared -> legacy --gpus all (byte-identical to pre-Phase-4)."""
    assert _build_opts(vendor=None, gpus="all") == ["--gpus", "all"]


def test_explicit_nvidia_emits_gpus_all():
    assert _build_opts(vendor="nvidia", gpus="all") == ["--gpus", "all"]


def test_nvidia_with_custom_gpu_spec():
    assert _build_opts(vendor="nvidia", gpus="device=0,1") == ["--gpus", "device=0,1"]


def test_default_with_empty_gpus_emits_nothing():
    """Empty gpus + no vendor -> no flag (matches legacy behavior)."""
    assert _build_opts(vendor=None, gpus="") == []


# --------------------------------------------------------------------------
# AMD / Intel / Apple / CPU
# --------------------------------------------------------------------------


def test_amd_emits_rocm_device_flags():
    opts = _build_opts(vendor="amd")
    assert opts == [
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "video",
    ]


def test_amd_ignores_gpus_field():
    """Setting --gpus is meaningless under ROCm; AMD path emits its own device flags only."""
    opts = _build_opts(vendor="amd", gpus="all")
    assert "--gpus" not in opts


def test_intel_gaudi_emits_accel_device():
    assert _build_opts(vendor="intel") == ["--device", "/dev/accel"]


@pytest.mark.parametrize("vendor", ["apple", "cpu"])
def test_apple_cpu_emit_no_device_flag(vendor: str):
    assert _build_opts(vendor=vendor) == []


def test_unknown_vendor_emits_nothing_with_warning(caplog):
    """Unknown vendors degrade gracefully: no flag + a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        opts = _build_opts(vendor="weird-fpga")
    assert opts == []
    assert any("unknown accelerator_vendor" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------
# accelerator_vendor_for(HostHardware) helper
# --------------------------------------------------------------------------


def test_accelerator_vendor_for_single_vendor():
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10")])
    assert accelerator_vendor_for(hw) == "nvidia"


def test_accelerator_vendor_for_multi_count_single_vendor():
    """count > 1 still reports the single vendor."""
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x", count=8)])
    assert accelerator_vendor_for(hw) == "amd"


def test_accelerator_vendor_for_mixed_returns_none():
    hw = HostHardware(
        accelerators=[
            AcceleratorSpec(vendor="apple", model="m5"),
            AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000"),
        ]
    )
    assert accelerator_vendor_for(hw) is None


def test_accelerator_vendor_for_empty_returns_none():
    assert accelerator_vendor_for(HostHardware()) is None
    assert accelerator_vendor_for(None) is None


# --------------------------------------------------------------------------
# Full run_cmd integration — confirms the flag lands in the docker run line
# --------------------------------------------------------------------------


def test_run_cmd_amd_contains_rocm_flags():
    cfg = ExecutorConfig(accelerator_vendor="amd", privileged=False, gpus="all", network="")
    cmd = DockerExecutor(cfg).run_cmd(image="rocm/vllm:latest", container_name="test")
    assert "--device /dev/kfd" in cmd
    assert "--device /dev/dri" in cmd
    assert "--group-add video" in cmd
    assert "--gpus" not in cmd


def test_run_cmd_default_byte_for_byte_with_legacy_gpus_flag():
    """Default (no accelerator_vendor) keeps the legacy --gpus all flag in run_cmd output."""
    cfg = ExecutorConfig(privileged=False, gpus="all", network="")
    cmd = DockerExecutor(cfg).run_cmd(image="img", container_name="test")
    assert "--gpus all" in cmd
