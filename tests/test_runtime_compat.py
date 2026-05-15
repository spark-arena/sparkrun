"""Tests for runtime ↔ host compatibility (Phase 7)."""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.layout import Placement, RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.placement import compute_placement
from sparkrun.runtimes.atlas import AtlasRuntime
from sparkrun.runtimes.base import RuntimePlugin
from sparkrun.runtimes.compatibility import (
    IncompatibleHardwareError,
    assert_runtime_cluster_compatibility,
    check_runtime_cluster_compatibility,
    check_runtime_host_compatibility,
)
from sparkrun.runtimes.eugr_vllm_ray import EugrVllmRayRuntime


# --------------------------------------------------------------------------
# Fixtures: minimal runtime subclasses
# --------------------------------------------------------------------------


class _UnrestrictedRuntime(RuntimePlugin):
    runtime_name = "unrestricted"

    def generate_command(self, recipe, overrides, is_cluster, num_nodes=1, head_ip=None, skip_keys=frozenset()):
        return ""


class _CudaRequiredRuntime(RuntimePlugin):
    runtime_name = "cuda-required"
    requires_capability = frozenset({"cuda"})

    def generate_command(self, recipe, overrides, is_cluster, num_nodes=1, head_ip=None, skip_keys=frozenset()):
        return ""


def _hw(vendor: str, model: str, *, memory_gb: float | None = None, caps: frozenset[str] = frozenset()) -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor=vendor, model=model, memory_gb=memory_gb, capabilities=caps)])


# --------------------------------------------------------------------------
# requires_capability defaults
# --------------------------------------------------------------------------


def test_default_runtime_has_no_capability_constraint():
    """A vanilla runtime accepts any host."""
    assert RuntimePlugin.requires_capability == frozenset()
    assert _UnrestrictedRuntime().requires_capability == frozenset()


def test_atlas_runtime_requires_gb10():
    assert AtlasRuntime.requires_capability == frozenset({"gb10"})


def test_eugr_runtime_requires_gb10():
    assert EugrVllmRayRuntime.requires_capability == frozenset({"gb10"})


# --------------------------------------------------------------------------
# Default RuntimePlugin.default_image_for
# --------------------------------------------------------------------------


def test_default_image_for_returns_legacy_prefix():
    class _R(RuntimePlugin):
        runtime_name = "x"
        default_image_prefix = "ghcr.io/example/img"

        def generate_command(self, *a, **k):
            return ""

    assert _R().default_image_for() == "ghcr.io/example/img:latest"


def test_default_image_for_returns_none_when_no_prefix():
    """Runtimes with no default prefix (e.g. eugr) surface None so callers can prompt."""

    class _R(RuntimePlugin):
        runtime_name = "x"
        default_image_prefix = ""

        def generate_command(self, *a, **k):
            return ""

    assert _R().default_image_for() is None


# --------------------------------------------------------------------------
# Single-host compatibility
# --------------------------------------------------------------------------


def test_unrestricted_runtime_accepts_any_host():
    rt = _UnrestrictedRuntime()
    assert check_runtime_host_compatibility(rt, "h", _hw("apple", "m5")) == []
    assert check_runtime_host_compatibility(rt, "h", _hw("amd", "mi300x")) == []


def test_cuda_required_accepts_host_with_cuda_capability():
    rt = _CudaRequiredRuntime()
    nvidia_hw = _hw("nvidia", "gb10", caps=frozenset({"cuda"}))
    assert check_runtime_host_compatibility(rt, "h", nvidia_hw) == []


def test_cuda_required_rejects_amd_host():
    rt = _CudaRequiredRuntime()
    amd_hw = _hw("amd", "mi300x", caps=frozenset({"rocm"}))
    errors = check_runtime_host_compatibility(rt, "amd-box", amd_hw)
    assert len(errors) == 1
    assert "cuda" in errors[0]
    assert "amd-box" in errors[0]


def test_gb10_required_accepts_gb10_host_by_model_name():
    """``requires_capability={"gb10"}`` matches an accelerator with model="gb10"."""
    rt = AtlasRuntime()
    gb10_hw = _hw("nvidia", "gb10", caps=frozenset({"cuda"}))
    assert check_runtime_host_compatibility(rt, "spark-01", gb10_hw) == []


def test_gb10_required_rejects_h100_host():
    rt = AtlasRuntime()
    h100_hw = _hw("nvidia", "h100", caps=frozenset({"cuda"}))
    errors = check_runtime_host_compatibility(rt, "h100-box", h100_hw)
    assert len(errors) == 1
    assert "gb10" in errors[0]


def test_gb10_required_accepts_capability_tag_alias():
    """Manually tagging a host with capability 'gb10' also satisfies the requirement."""
    rt = AtlasRuntime()
    tagged_hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="custom", capabilities=frozenset({"gb10", "cuda"}))])
    assert check_runtime_host_compatibility(rt, "h", tagged_hw) == []


# --------------------------------------------------------------------------
# Cluster-level compatibility
# --------------------------------------------------------------------------


def test_cluster_compat_passes_when_all_hosts_satisfy():
    cluster = ClusterDefinition(name="dgx", hosts=["s1", "s2"])  # defaults to DGX Spark
    assert check_runtime_cluster_compatibility(AtlasRuntime(), cluster) == []


def test_cluster_compat_reports_each_failing_host():
    cluster = ClusterDefinition(
        name="mixed",
        hosts=["spark", "h100"],
        hosts_hardware={
            "spark": _hw("nvidia", "gb10", caps=frozenset({"cuda"})),
            "h100": _hw("nvidia", "h100", caps=frozenset({"cuda"})),
        },
    )
    errors = check_runtime_cluster_compatibility(AtlasRuntime(), cluster)
    assert len(errors) == 1
    assert "h100" in errors[0]


def test_cluster_compat_walks_only_placed_hosts():
    """Heterogeneous cluster with layout excluding bad host -> no error."""
    cluster = ClusterDefinition(
        name="mixed",
        hosts=["spark", "h100"],
        hosts_hardware={
            "spark": _hw("nvidia", "gb10", caps=frozenset({"cuda"})),
            "h100": _hw("nvidia", "h100", caps=frozenset({"cuda"})),
        },
    )
    # Layout places only the spark host
    placement = compute_placement(
        ParallelismConfig(),
        cluster.hosts,
        host_hardware=cluster.hosts_hardware,
        layout=RecipeLayout(placements=[Placement(host="spark", ranks=(0,))]),
    )
    assert check_runtime_cluster_compatibility(AtlasRuntime(), cluster, placement) == []


def test_assert_runtime_cluster_compatibility_raises():
    cluster = ClusterDefinition(
        name="amd",
        hosts=["box"],
        hosts_hardware={"box": _hw("amd", "mi300x", caps=frozenset({"rocm"}))},
    )
    with pytest.raises(IncompatibleHardwareError) as ei:
        assert_runtime_cluster_compatibility(AtlasRuntime(), cluster)
    err = ei.value
    assert err.runtime_name == "atlas"
    assert len(err.errors) == 1
    assert "gb10" in err.errors[0]


def test_assert_runtime_cluster_compatibility_no_op_when_compatible():
    cluster = ClusterDefinition(name="dgx", hosts=["s1"])
    # Should not raise.
    assert_runtime_cluster_compatibility(_UnrestrictedRuntime(), cluster)
