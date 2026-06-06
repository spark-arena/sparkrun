"""Tests for sparkrun.core.limits — usable-memory cap resolution."""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware, default_dgx_spark_hardware
from sparkrun.core.limits import (
    resolve_max_gpu_memory_utilization,
    resolved_hardware_for_scheduling,
    usable_memory_gb,
)


def _gb10_host(max_util: float | None = None) -> tuple[AcceleratorSpec, HostHardware]:
    accel = AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0, max_gpu_memory_utilization=max_util)
    return accel, HostHardware(accelerators=[accel])


# --------------------------------------------------------------------------
# Precedence chain
# --------------------------------------------------------------------------


def test_hard_fallback_is_one_for_unknown_accelerator():
    accel = AcceleratorSpec(vendor="nvidia", model="h200", memory_gb=141.0)
    hw = HostHardware(accelerators=[accel])
    assert resolve_max_gpu_memory_utilization(accel, hw, None) == 1.0


def test_platform_default_applies_for_gb10():
    accel, hw = _gb10_host()
    assert resolve_max_gpu_memory_utilization(accel, hw, None) == 0.85


def test_cluster_wide_overrides_platform():
    accel, hw = _gb10_host()
    cluster = ClusterDefinition(name="c", hosts=["h"], max_gpu_memory_utilization=0.7)
    assert resolve_max_gpu_memory_utilization(accel, hw, cluster) == 0.7


def test_per_type_overrides_cluster_wide():
    accel, hw = _gb10_host()
    cluster = ClusterDefinition(
        name="c",
        hosts=["h"],
        max_gpu_memory_utilization=0.7,
        accelerator_memory_limits={"gb10": 0.6},
    )
    assert resolve_max_gpu_memory_utilization(accel, hw, cluster) == 0.6


def test_per_accelerator_field_wins_over_everything():
    accel, hw = _gb10_host(max_util=0.5)
    cluster = ClusterDefinition(
        name="c",
        hosts=["h"],
        max_gpu_memory_utilization=0.7,
        accelerator_memory_limits={"gb10": 0.6},
    )
    assert resolve_max_gpu_memory_utilization(accel, hw, cluster) == 0.5


def test_per_type_only_matches_its_model():
    accel, hw = _gb10_host()
    cluster = ClusterDefinition(name="c", hosts=["h"], accelerator_memory_limits={"h200": 0.95})
    # No gb10 entry → falls through to the platform default.
    assert resolve_max_gpu_memory_utilization(accel, hw, cluster) == 0.85


# --------------------------------------------------------------------------
# Range validation — out-of-range falls through
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
def test_out_of_range_per_accel_falls_through_to_platform(bad):
    accel, hw = _gb10_host(max_util=bad)
    assert resolve_max_gpu_memory_utilization(accel, hw, None) == 0.85


def test_out_of_range_cluster_wide_falls_through():
    accel, hw = _gb10_host()
    cluster = ClusterDefinition(name="c", hosts=["h"], max_gpu_memory_utilization=1.5)
    assert resolve_max_gpu_memory_utilization(accel, hw, cluster) == 0.85


def test_boundary_value_one_is_accepted():
    accel, hw = _gb10_host(max_util=1.0)
    assert resolve_max_gpu_memory_utilization(accel, hw, None) == 1.0


# --------------------------------------------------------------------------
# usable_memory_gb
# --------------------------------------------------------------------------


def test_usable_memory_applies_cap():
    accel, hw = _gb10_host()
    assert usable_memory_gb(accel, hw, None) == pytest.approx(102.85)


def test_usable_memory_none_when_capacity_unknown():
    accel = AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=None)
    hw = HostHardware(accelerators=[accel])
    assert usable_memory_gb(accel, hw, None) is None


# --------------------------------------------------------------------------
# resolved_hardware_for_scheduling
# --------------------------------------------------------------------------


def test_resolved_hardware_folds_cap_into_field():
    resolved = resolved_hardware_for_scheduling(None, ["h1", "h2"])
    assert set(resolved) == {"h1", "h2"}
    for hw in resolved.values():
        accel = hw.accelerators[0]
        # Platform default folded in; nominal memory untouched.
        assert accel.max_gpu_memory_utilization == 0.85
        assert accel.memory_gb == 121.0


def test_resolved_hardware_honors_cluster_override():
    cluster = ClusterDefinition(name="c", hosts=["h1"], max_gpu_memory_utilization=0.7)
    resolved = resolved_hardware_for_scheduling(cluster, ["h1"])
    assert resolved["h1"].accelerators[0].max_gpu_memory_utilization == 0.7


def test_resolved_hardware_uses_dgx_default_for_unlisted_hosts():
    cluster = ClusterDefinition(name="c", hosts=["h1"])
    resolved = resolved_hardware_for_scheduling(cluster, ["h1"])
    accel = resolved["h1"].accelerators[0]
    assert accel.model == default_dgx_spark_hardware().accelerators[0].model
    assert accel.max_gpu_memory_utilization == 0.85
