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


# --------------------------------------------------------------------------
# Launcher-level compatibility gate (A4)
# --------------------------------------------------------------------------


def _make_launch_monkeypatches(monkeypatch, tmp_path):
    """Apply the standard set of no-op monkeypatches used for launch_inference tests."""
    from sparkrun.core import launcher

    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.distribute_from_config",
        lambda *a, **kw: (None, {}, {}),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.save_job_metadata",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.derive_cluster_id",
        lambda *a, **kw: "sparkrun_testabc12345",
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.build_ssh_kwargs",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(
        launcher,
        "resolve_effective_cache_dir",
        lambda *a, **kw: str(tmp_path),
    )
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr(
        "sparkrun.orchestration.executor.resolve_executor",
        lambda **kw: type("Ex", (), {})(),
    )


class _StubRuntime:
    """Minimal RuntimePlugin-ish stub for launcher-level compatibility tests."""

    runtime_name = "stub"
    requires_capability: frozenset = frozenset()
    run_called: bool = False

    def is_delegating_runtime(self):
        return False

    def resolve_container(self, recipe, overrides=None):
        return "stub:latest"

    def prepare(self, *args, **kwargs):
        return None

    def get_head_container_name(self, cluster_id, is_solo=False):
        return "%s_solo" % cluster_id

    def generate_command(self, **kwargs):
        return "echo serve"

    def resolve_api_key(self, recipe, overrides=None):
        return None

    def _collect_runtime_info(self, *args, **kwargs):
        return {}

    def run(self, **kwargs):
        type(self).run_called = True
        return 0


class _CudaOnlyStubRuntime(_StubRuntime):
    runtime_name = "cuda-only-stub"
    requires_capability = frozenset({"cuda"})


class _FakeConfig:
    def __init__(self, tmp_path):
        self.hf_cache_dir = tmp_path / "hf"
        self.cache_dir = tmp_path / "cache"

    def get_registry_manager(self):
        return None


class _FakeRecipe:
    runtime = "stub"
    model = "stub-model"
    env = {}
    builder = None
    mods = []
    source_registry = None
    source_registry_url = None
    defaults = {"port": 8000}
    pre_exec = []
    post_exec = []
    post_commands = []
    layout = None
    stop_after_post = False
    executor = ""
    executor_config = None
    qualified_name = "stub-recipe"
    name = "stub-recipe"
    container = "stub:latest"
    model_revision = None

    def build_config_chain(self, overrides=None):
        class _CC:
            def get(self, k, default=None):
                return (overrides or {}).get(k, {"port": 8000}.get(k, default))

        return _CC()


def test_launcher_blocks_incompatible_host_before_run(monkeypatch, tmp_path):
    """launch_inference raises IncompatibleHardwareError before runtime.run() for incompatible hardware."""
    from sparkrun.core.launcher import launch_inference
    from sparkrun.runtimes.compatibility import IncompatibleHardwareError

    _make_launch_monkeypatches(monkeypatch, tmp_path)

    runtime = _CudaOnlyStubRuntime()
    _CudaOnlyStubRuntime.run_called = False

    # AMD host without "cuda" capability
    amd_hw = _hw("amd", "mi300x", caps=frozenset({"rocm"}))
    cluster = ClusterDefinition(
        name="amd-cluster",
        hosts=["amd-box"],
        hosts_hardware={"amd-box": amd_hw},
    )

    with pytest.raises(IncompatibleHardwareError) as ei:
        launch_inference(
            recipe=_FakeRecipe(),
            runtime=runtime,
            host_list=["amd-box"],
            overrides={},
            config=_FakeConfig(tmp_path),
            cluster=cluster,
            is_solo=True,
            dry_run=True,
            sync_tuning=False,
        )

    err = ei.value
    assert err.runtime_name == "cuda-only-stub"
    assert any("cuda" in e for e in err.errors)
    # runtime.run() must NOT have been called
    assert not _CudaOnlyStubRuntime.run_called


def test_launcher_allows_compatible_hosts(monkeypatch, tmp_path):
    """launch_inference succeeds (no exception) when all hosts satisfy runtime requirements."""
    from sparkrun.core.launcher import launch_inference

    _make_launch_monkeypatches(monkeypatch, tmp_path)

    runtime = _CudaOnlyStubRuntime()
    _CudaOnlyStubRuntime.run_called = False

    nvidia_hw = _hw("nvidia", "gb10", caps=frozenset({"cuda"}))
    cluster = ClusterDefinition(
        name="nvidia-cluster",
        hosts=["nv-box"],
        hosts_hardware={"nv-box": nvidia_hw},
    )

    result = launch_inference(
        recipe=_FakeRecipe(),
        runtime=runtime,
        host_list=["nv-box"],
        overrides={},
        config=_FakeConfig(tmp_path),
        cluster=cluster,
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
    )

    assert result.rc == 0
    assert _CudaOnlyStubRuntime.run_called


def test_launcher_skips_check_when_hardware_unknown(monkeypatch, tmp_path):
    """When cluster is None (--hosts / --hosts-file bypass), the compatibility gate is skipped."""
    from sparkrun.core.launcher import launch_inference

    _make_launch_monkeypatches(monkeypatch, tmp_path)

    runtime = _CudaOnlyStubRuntime()
    _CudaOnlyStubRuntime.run_called = False

    # No cluster supplied — fingerprint data unavailable, gate must not crash.
    result = launch_inference(
        recipe=_FakeRecipe(),
        runtime=runtime,
        host_list=["unknown-host"],
        overrides={},
        config=_FakeConfig(tmp_path),
        cluster=None,
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
    )

    assert result.rc == 0
    assert _CudaOnlyStubRuntime.run_called
