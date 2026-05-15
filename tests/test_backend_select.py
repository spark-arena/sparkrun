"""Tests for sparkrun.core.backend_select (Phase 6)."""

from __future__ import annotations

import pytest

from sparkrun.core.backend_select import (
    BackendBundle,
    NoMatchingBackendError,
    known_vendors,
    select_backends,
)
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.orchestration.collectives import HcclBackend, NcclBackend, RcclBackend


def _hw(vendor: str, model: str = "x") -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor=vendor, model=model)])


def test_known_vendors_includes_nvidia_amd_intel():
    assert set(known_vendors()) >= {"nvidia", "amd", "intel"}


def test_select_backends_nvidia_returns_nccl_bundle():
    bundle = select_backends(_hw("nvidia", "gb10"))
    assert isinstance(bundle, BackendBundle)
    assert bundle.accelerator_vendor == "nvidia"
    assert isinstance(bundle.collective, NcclBackend)


def test_select_backends_amd_returns_rccl_bundle():
    bundle = select_backends(_hw("amd", "mi300x"))
    assert bundle.accelerator_vendor == "amd"
    assert isinstance(bundle.collective, RcclBackend)


def test_select_backends_intel_returns_hccl_bundle():
    bundle = select_backends(_hw("intel", "gaudi3"))
    assert bundle.accelerator_vendor == "intel"
    assert isinstance(bundle.collective, HcclBackend)


def test_select_backends_apple_raises_no_matching_backend():
    """Apple has no backend yet — error must be explicit, not a silent NCCL fallback."""
    hw = _hw("apple", "m5")
    with pytest.raises(NoMatchingBackendError) as ei:
        select_backends(hw)
    msg = str(ei.value)
    assert "apple/m5" in msg
    assert "nvidia" in msg  # known vendor list surfaced for actionability


def test_select_backends_empty_hardware_raises():
    with pytest.raises(NoMatchingBackendError, match="<none>"):
        select_backends(HostHardware())


def test_select_backends_multi_vendor_host_raises():
    """A single host with multiple vendors is ambiguous — refuse to pick."""
    hw = HostHardware(
        accelerators=[
            AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000"),
            AcceleratorSpec(vendor="apple", model="m5"),
        ]
    )
    with pytest.raises(NoMatchingBackendError):
        select_backends(hw)


def test_no_matching_backend_error_carries_known_vendors_and_hw():
    hw = _hw("apple", "m5")
    try:
        select_backends(hw)
    except NoMatchingBackendError as e:
        assert e.host_hardware is hw
        assert "nvidia" in e.known_vendors
        assert "amd" in e.known_vendors
        assert "intel" in e.known_vendors
    else:
        pytest.fail("expected NoMatchingBackendError")
