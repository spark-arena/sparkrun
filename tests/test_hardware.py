"""Tests for sparkrun.core.hardware (Phase 1 of hardware abstraction)."""

from __future__ import annotations

from sparkrun.core.hardware import (
    AcceleratorSpec,
    HostHardware,
    default_dgx_spark_hardware,
)


def test_accelerator_spec_round_trip_minimal():
    """A minimal AcceleratorSpec omits defaults from to_dict and parses back."""
    spec = AcceleratorSpec(vendor="nvidia", model="gb10")
    d = spec.to_dict()
    assert d == {"vendor": "nvidia", "model": "gb10"}
    assert AcceleratorSpec.from_dict(d) == spec
    # max_gpu_memory_utilization defaults to None and is omitted from YAML.
    assert spec.max_gpu_memory_utilization is None
    assert "max_gpu_memory_utilization" not in d


def test_accelerator_spec_round_trip_max_gpu_memory_utilization():
    """The usable-memory cap survives to_dict/from_dict when set."""
    spec = AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0, max_gpu_memory_utilization=0.8)
    d = spec.to_dict()
    assert d["max_gpu_memory_utilization"] == 0.8
    assert AcceleratorSpec.from_dict(d) == spec


def test_accelerator_spec_round_trip_full():
    """All fields round-trip including capabilities (sorted on emit)."""
    spec = AcceleratorSpec(
        vendor="nvidia",
        model="h200",
        count=8,
        memory_gb=141.0,
        capabilities=frozenset({"cuda", "nvlink"}),
    )
    d = spec.to_dict()
    assert d == {
        "vendor": "nvidia",
        "model": "h200",
        "count": 8,
        "memory_gb": 141.0,
        "capabilities": ["cuda", "nvlink"],
    }
    assert AcceleratorSpec.from_dict(d) == spec


def test_host_hardware_round_trip():
    """HostHardware with multiple accelerators round-trips."""
    hw = HostHardware(
        accelerators=[
            AcceleratorSpec(vendor="apple", model="m5", memory_gb=64.0, capabilities=frozenset({"mlx"})),
            AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000", count=2, memory_gb=96.0, capabilities=frozenset({"cuda"})),
        ],
        fingerprint="abc123",
        notes="laptop dev box",
    )
    d = hw.to_dict()
    restored = HostHardware.from_dict(d)
    assert restored == hw


def test_host_hardware_omits_empty_optional_fields():
    """fingerprint/notes are omitted when empty."""
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x")])
    d = hw.to_dict()
    assert "fingerprint" not in d
    assert "notes" not in d


def test_host_hardware_total_gpus_sums_counts():
    """total_gpus aggregates count across heterogeneous accelerators."""
    hw = HostHardware(
        accelerators=[
            AcceleratorSpec(vendor="apple", model="m5"),
            AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000", count=2),
        ]
    )
    assert hw.total_gpus == 3


def test_host_hardware_vendors_distinct():
    """vendors returns the distinct vendor set."""
    hw = HostHardware(
        accelerators=[
            AcceleratorSpec(vendor="nvidia", model="h200", count=8),
            AcceleratorSpec(vendor="nvidia", model="h200", count=8),
        ]
    )
    assert hw.vendors == frozenset({"nvidia"})


def test_host_hardware_has_capability():
    """has_capability checks any accelerator on the host."""
    hw = HostHardware(
        accelerators=[
            AcceleratorSpec(vendor="apple", model="m5"),
            AcceleratorSpec(vendor="nvidia", model="rtx-pro-6000", capabilities=frozenset({"cuda"})),
        ]
    )
    assert hw.has_capability("cuda") is True
    assert hw.has_capability("rocm") is False


def test_default_dgx_spark_hardware_shape():
    """The DGX Spark fallback is 1× GB10 with 121 GB unified memory."""
    hw = default_dgx_spark_hardware()
    assert len(hw.accelerators) == 1
    accel = hw.accelerators[0]
    assert accel.vendor == "nvidia"
    assert accel.model == "gb10"
    assert accel.count == 1
    assert accel.memory_gb == 121.0
    assert "cuda" in accel.capabilities
