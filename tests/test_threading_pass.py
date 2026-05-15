"""Consolidated threading-pass tests (Phase X).

Verifies that cluster + placement now flow from the CLI down to
ClusterContext, that DGX behaviour stays byte-equivalent on the legacy
path (no cluster threaded), and that multi-GPU hosts opt into the new
placement-aware rank math.
"""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.placement import compute_placement


# --------------------------------------------------------------------------
# ClusterContext.build computes placement when cluster + recipe present
# --------------------------------------------------------------------------


def _stub_recipe(**defaults):
    """Minimal recipe stub that satisfies ClusterContext.build's recipe usage."""

    class _Recipe:
        def __init__(self):
            self.env = {}
            self.layout = None

        def build_config_chain(self, *args, **kwargs):
            class _Cfg:
                def __init__(self, d):
                    self._d = d

                def get(self, k, default=None):
                    return self._d.get(k, default)

            return _Cfg(defaults)

    return _Recipe()


def _stub_runtime():
    class _R:
        runtime_name = "stub"

        def get_extra_volumes(self):
            return {}

        def get_common_env(self):
            return {}

        def get_extra_env(self):
            return {}

        def get_cluster_env(self, head_ip, num_nodes):
            return {}

    return _R()


def test_cluster_context_legacy_path_has_no_placement():
    """No cluster/recipe -> placement stays None (DGX byte-for-byte path)."""
    from sparkrun.runtimes._cluster_ops import ClusterContext

    ctx = ClusterContext.build(
        runtime=_stub_runtime(),
        hosts=["a", "b"],
        image="img",
        cluster_id="cid",
        env={},
        cache_dir=None,
        config=None,
        dry_run=True,
    )
    assert ctx.placement is None
    assert ctx.cluster is None


def test_cluster_context_computes_placement_when_threaded():
    from sparkrun.runtimes._cluster_ops import ClusterContext

    cluster = ClusterDefinition(name="c", hosts=["a", "b"])
    recipe = _stub_recipe(tensor_parallel=2)
    ctx = ClusterContext.build(
        runtime=_stub_runtime(),
        hosts=["a", "b"],
        image="img",
        cluster_id="cid",
        env={},
        cache_dir=None,
        config=None,
        dry_run=True,
        cluster=cluster,
        recipe=recipe,
    )
    assert ctx.cluster is cluster
    assert ctx.placement is not None
    assert ctx.placement.hosts_used == ("a", "b")


def test_cluster_context_hardware_for_falls_back_to_dgx():
    """Unknown host returns DGX Spark default — preserves legacy assumption."""
    from sparkrun.runtimes._cluster_ops import ClusterContext

    ctx = ClusterContext.build(
        runtime=_stub_runtime(),
        hosts=["a"],
        image="img",
        cluster_id="cid",
        env={},
        cache_dir=None,
        config=None,
        dry_run=True,
    )
    hw = ctx.hardware_for("a")
    assert hw.accelerators[0].vendor == "nvidia"
    assert hw.accelerators[0].model == "gb10"


# --------------------------------------------------------------------------
# vllm_distributed rank math: placement-aware for multi-rank-per-host
# --------------------------------------------------------------------------


def test_vllm_distributed_legacy_path_uses_hosts_indexing():
    """No placement -> falls back to hosts[i] for tp_master_addr (DGX behaviour)."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    recipe = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img", "defaults": {"tensor_parallel": 2}})
    cmd = VllmDistributedRuntime().generate_node_command(
        recipe=recipe,
        overrides={},
        head_ip="10.0.0.1",
        num_nodes=2,
        node_rank=1,
        hosts=["10.0.0.1", "10.0.0.2"],
    )
    assert "--master-addr 10.0.0.1" in cmd


def test_vllm_distributed_with_placement_resolves_through_host_for_rank():
    """Placement on a multi-GPU host correctly routes tp_master_addr."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    placement = compute_placement(
        ParallelismConfig(tensor_parallel=4),
        ["dgx-h200"],
        host_hardware={"dgx-h200": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200", count=4, memory_gb=141.0)])},
    )
    recipe = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img", "defaults": {"tensor_parallel": 4}})
    cmd = VllmDistributedRuntime().generate_node_command(
        recipe=recipe,
        overrides={},
        head_ip="dgx-h200",
        num_nodes=4,
        node_rank=1,
        hosts=["dgx-h200"],  # legacy fallback would fail
        placement=placement,
    )
    # Placement returns the same single host for all 4 ranks.
    assert "--master-addr dgx-h200" in cmd


# --------------------------------------------------------------------------
# trtllm slot count from placement
# --------------------------------------------------------------------------


def test_trtllm_mpirun_default_slot_count_is_one():
    """No placement -> rmaps_ppr_n_pernode=1 (DGX byte-for-byte)."""
    from sparkrun.runtimes.trtllm import TrtllmRuntime

    cmd = TrtllmRuntime()._build_mpirun_command(
        serve_cmd="trtllm-serve x",
        host_ips=["10.0.0.1", "10.0.0.2"],
    )
    assert "rmaps_ppr_n_pernode 1" in cmd


def test_trtllm_mpirun_slot_count_scales_to_placement():
    """rmaps_ppr_n_pernode reflects max_ranks_per_host on multi-GPU placements."""
    from sparkrun.runtimes.trtllm import TrtllmRuntime

    cmd = TrtllmRuntime()._build_mpirun_command(
        serve_cmd="trtllm-serve x",
        host_ips=["10.0.0.1"],
        ranks_per_node=8,
    )
    assert "rmaps_ppr_n_pernode 8" in cmd


# --------------------------------------------------------------------------
# CLI: validate_and_prepare_hosts forwards cluster to compute_required_nodes
# --------------------------------------------------------------------------


def test_validate_and_prepare_hosts_threads_cluster(monkeypatch):
    from sparkrun.cli import _common

    captured = {}

    class _Runtime:
        runtime_name = "stub"

        def compute_required_nodes(self, recipe, overrides, *, cluster=None):
            captured["cluster"] = cluster
            return 2

    class _Recipe:
        max_nodes = None
        mode = "auto"

    cluster = ClusterDefinition(name="c", hosts=["a", "b", "c"])
    host_list, is_solo = _common.validate_and_prepare_hosts(["a", "b", "c"], _Recipe(), {}, _Runtime(), solo=False, cluster=cluster)
    assert captured["cluster"] is cluster
    assert len(host_list) == 2
    assert is_solo is False


# --------------------------------------------------------------------------
# CLI: cluster update --infer-hardware (mocked SSH)
# --------------------------------------------------------------------------


def test_cluster_update_infer_hardware_persists_fingerprints(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli._cluster import cluster
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.hardware import default_dgx_spark_hardware

    mgr = ClusterManager(tmp_path)
    mgr.create("test-cluster", ["host1", "host2"])

    # Mock fingerprint_host to return a known HostHardware
    def _fake_fingerprint(host, ssh_kwargs):
        hw = default_dgx_spark_hardware()
        # Replace notes with host name so we can tell each host apart in storage
        hw.notes = "fingerprinted-%s" % host
        return hw

    # Mock the cluster manager factory so the CLI uses our tmp_path manager
    def _fake_get_mgr():
        return mgr

    monkeypatch.setattr("sparkrun.core.fingerprint.fingerprint_host", _fake_fingerprint)
    monkeypatch.setattr("sparkrun.cli._cluster._get_cluster_manager", _fake_get_mgr)

    runner = CliRunner()
    result = runner.invoke(cluster, ["update", "test-cluster", "--infer-hardware"])
    assert result.exit_code == 0, result.output
    assert "Fingerprinting 2 host" in result.output

    restored = mgr.get("test-cluster")
    assert set(restored.hosts_hardware.keys()) == {"host1", "host2"}
    assert restored.hosts_hardware["host1"].notes == "fingerprinted-host1"
    assert restored.hosts_hardware["host1"].accelerators[0].model == "gb10"


# --------------------------------------------------------------------------
# Pre-flight compatibility check fires in cli/_run.py path
# --------------------------------------------------------------------------


def test_display_vram_estimate_renders_per_host_fit(capsys):
    """When cluster+placement are threaded, the formatter renders per-host fit."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.utils.cli_formatters import display_vram_estimate

    cluster = ClusterDefinition(
        name="dgx",
        hosts=["s1", "s2"],
        hosts_hardware={
            "s1": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0)]),
            "s2": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0)]),
        },
    )
    placement = compute_placement(ParallelismConfig(tensor_parallel=2), cluster.hosts)
    recipe = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img"})

    # Auto-detect off keeps test hermetic; estimate returns model_weights=0 etc.
    display_vram_estimate(recipe, auto_detect=False, cluster=cluster, placement=placement)
    out = capsys.readouterr().out
    assert "Per-host fit" in out
    assert "s1:" in out
    assert "s2:" in out


def test_display_vram_estimate_skips_per_host_fit_without_cluster(capsys):
    """Legacy path (no cluster) doesn't add the per-host block."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.utils.cli_formatters import display_vram_estimate

    recipe = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img"})
    display_vram_estimate(recipe, auto_detect=False)
    out = capsys.readouterr().out
    assert "Per-host fit" not in out
    # Legacy DGX line is still present for back-compat.
    assert "DGX Spark fit" in out


def test_default_image_for_consults_platform_registry():
    """Passing host_hardware routes default_image_for through the matched platform."""
    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun.runtimes.atlas import AtlasRuntime

    rt = AtlasRuntime()
    # With no host_hardware -> legacy default_image_prefix:latest
    assert rt.default_image_for() == "avarok/atlas-gb10:latest"
    # With DGX Spark host hardware -> DgxSparkPlatform's curated default for atlas
    assert rt.default_image_for(default_dgx_spark_hardware()) == "avarok/atlas-gb10:latest"


def test_default_image_for_falls_back_when_no_platform_match():
    """An AMD host with no matching platform falls back to legacy default_image_prefix."""
    from sparkrun.runtimes.base import RuntimePlugin

    class _R(RuntimePlugin):
        runtime_name = "fake-runtime"
        default_image_prefix = "fallback/img"

        def generate_command(self, *a, **k):
            return ""

    amd_hw = HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x")])
    # No platform claims AMD -> fall back to legacy prefix.
    assert _R().default_image_for(amd_hw) == "fallback/img:latest"


def test_refuse_unsupported_collectives_passes_for_nvidia():
    """Single-vendor NVIDIA placed cluster: legacy NCCL path stays available."""
    from sparkrun.runtimes._cluster_ops import ClusterContext, _refuse_unsupported_collectives

    ctx = ClusterContext.build(
        runtime=_stub_runtime(),
        hosts=["a"],
        image="img",
        cluster_id="cid",
        env={},
        cache_dir=None,
        config=None,
        dry_run=True,
        cluster=ClusterDefinition(name="dgx", hosts=["a"]),
        recipe=_stub_recipe(),
    )
    _refuse_unsupported_collectives(ctx)  # no raise


def test_refuse_unsupported_collectives_raises_for_amd_scaffold():
    """RCCL scaffold raises NotImplementedError -> launch path surfaces a clean error."""
    from sparkrun.runtimes._cluster_ops import ClusterContext, _refuse_unsupported_collectives

    cluster = ClusterDefinition(
        name="amd",
        hosts=["box"],
        hosts_hardware={
            "box": HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x")]),
        },
    )
    ctx = ClusterContext.build(
        runtime=_stub_runtime(),
        hosts=["box"],
        image="img",
        cluster_id="cid",
        env={},
        cache_dir=None,
        config=None,
        dry_run=True,
        cluster=cluster,
        recipe=_stub_recipe(),
    )
    with pytest.raises(RuntimeError, match="RCCL backend not yet implemented"):
        _refuse_unsupported_collectives(ctx)


def test_assert_runtime_cluster_compatibility_raises_for_atlas_on_h100():
    from sparkrun.runtimes.atlas import AtlasRuntime
    from sparkrun.runtimes.compatibility import (
        IncompatibleHardwareError,
        assert_runtime_cluster_compatibility,
    )

    cluster = ClusterDefinition(
        name="h100",
        hosts=["box"],
        hosts_hardware={
            "box": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", capabilities=frozenset({"cuda"}))])
        },
    )
    with pytest.raises(IncompatibleHardwareError):
        assert_runtime_cluster_compatibility(AtlasRuntime(), cluster)
