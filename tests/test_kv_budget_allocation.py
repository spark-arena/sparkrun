"""Explicit runtime KV allocation is independent of estimated context demand (#297)."""

import pytest
from sparkrun.core.recipe import Recipe
from sparkrun.models.vram import estimate_vram, extract_model_info
from sparkrun.models.kv import arch_marker_names
from sparkrun.utils.data import byte_size_setting


@pytest.mark.parametrize(
    "value,wanted",
    [
        (8 * 1024**3, 8 * 1024**3),
        ("8G", 8 * 1024**3),
        ("8g", 8_000_000_000),
        ("25.6k", 25600),
        ("1024", 1024),
        ("1M", 1024**2),
        ("2t", 2_000_000_000_000),
    ],
)
def test_byte_size_matches_runtime(value, wanted):
    assert byte_size_setting(value, key="kv_cache_memory_bytes") == wanted


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "0", "-8G", "1.5G", "8GB", "nan", float("inf"), "", {}, "0.0001k"])
def test_invalid_byte_budget_is_rejected(value):
    with pytest.raises(ValueError, match="kv_cache_memory_bytes"):
        byte_size_setting(value, key="kv_cache_memory_bytes")


def recipe(command=None, **defaults):
    return Recipe(
        {
            "model": "example/model",
            "runtime": "vllm-distributed",
            "container": "image",
            "metadata": {
                "model_vram": 174.75,
                "model_type": "glm5_next_text",
                "num_layers": 45,
                "num_kv_heads": 64,
                "kv_lora_rank": 512,
                "kv_dtype": "fp8",
                "layer_types": ["linear_attention"] * 34 + ["deepseek_sparse_attention"] * 11,
            },
            "defaults": {"tensor_parallel": 2, "max_model_len": 1048576, "gpu_memory_utilization": 0.87, **defaults},
            "command": command,
        }
    )


def test_issue_297_hybrid_budget_and_fresh_override():
    r = recipe(kv_cache_memory_bytes="8G")
    est = r.estimate_vram(auto_detect=False)
    assert est.total_per_gpu_gb == pytest.approx(95.375)
    assert est.kv_cache_per_gpu_gb == 8
    assert est.kv_cache_total_gb is None
    assert est.max_context_tokens is None
    assert est.available_kv_gb == 8
    assert est.usable_gpu_memory_gb is None  # No utilization-derived KV budget
    assert any("Hybrid" in warning for warning in est.warnings)
    assert r.estimate_vram(cli_overrides={"kv_cache_memory_bytes": "1G"}, auto_detect=False).total_per_gpu_gb == pytest.approx(88.375)


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (4, 2)])
def test_explicit_kv_is_never_sharded(tp, pp):
    est = estimate_vram(model_vram=100, kv_cache_memory_bytes=8 * 1024**3, tensor_parallel=tp, pipeline_parallel=pp)
    assert est.total_per_gpu_gb == 100 / (tp * pp) + 8


@pytest.mark.parametrize(
    "command,defaults",
    [
        ("vllm serve model --kv-cache-memory-bytes 8G", {}),
        ("vllm serve model --kv-cache-memory-bytes='8G'", {}),
        ("vllm serve model --kv-cache-memory 8G", {}),
        ("vllm serve model --kv-cache-memory-bytes {budget}", {"budget": "8G"}),
        ("vllm serve model --kv-cache-memory-bytes 1G --kv-cache-memory-bytes 8G", {}),
    ],
)
def test_command_budget_fallback(command, defaults):
    assert recipe(command, **defaults).resolve_kv_cache_memory_bytes() == 8 * 1024**3


def test_model_config_preserves_hybrid_markers():
    cfg = {
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "kv_lora_rank": 512,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        }
    }
    info = extract_model_info(cfg)
    assert info["layer_types"] == cfg["text_config"]["layer_types"]
    est = estimate_vram(
        model_vram=174.75, kv_dtype="fp8", num_layers=45, max_model_len=1048576, arch={k: info[k] for k in arch_marker_names() if k in info}
    )
    assert est.kv_cache_total_gb is None


def test_explicit_budget_controls_context_estimate():
    est = estimate_vram(
        model_vram=20,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        kv_dtype="fp8",
        tensor_parallel=2,
        max_model_len=4096,
        kv_cache_memory_bytes=1024**3,
        gpu_memory_utilization=0.1,
    )
    assert est.max_context_tokens == 2 * 1024**3 // 1024
    assert est.available_kv_gb == 1
    assert est.fit_budget_gb(121, 0.9) == pytest.approx(108.9)


def test_invalid_budget_fails_before_occupancy_or_metadata(monkeypatch):
    from sparkrun.api._hosts import resolve_effective_hosts
    from sparkrun.api import SparkrunError

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid budget reached external lookup")

    monkeypatch.setattr("sparkrun.api.status", forbidden)
    r = recipe(kv_cache_memory_bytes="garbage")
    monkeypatch.setattr(r, "estimate_vram", forbidden)
    with pytest.raises(SparkrunError, match="kv_cache_memory_bytes"):
        resolve_effective_hosts(["h1", "h2"], r, {})


def test_cluster_override_and_cli_fit_use_the_same_budget(capsys):
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.scheduler import RankAssignment, RankSlot
    from sparkrun.models.fit import check_fit
    from sparkrun.utils.cli_formatters import display_memory_plan

    r = recipe(kv_cache_memory_bytes="8G", gpu_memory_utilization=0.1)
    cluster = ClusterDefinition(name="sparks", hosts=["h1", "h2"], max_gpu_memory_utilization=0.75)
    placement = RankAssignment((RankSlot("h1", 0), RankSlot("h2", 0)), ("h1", "h2"))
    est = r.estimate_vram(auto_detect=False)
    fit = check_fit(est, cluster, placement)
    assert not fit.ok
    assert fit.per_host["h1"].accelerator_memory_gb == pytest.approx(90.75)
    assert fit.per_host["h1"].memory_limit_source == "cluster override"
    display_memory_plan(r, auto_detect=False, cluster=cluster, placement=placement)
    output = capsys.readouterr().out
    assert "Fit            EXCEEDS    on h1, h2" in output
    assert "DGX Spark memory fit:" not in output
    assert "KV cache       8.0 GiB    explicit kv_cache_memory_bytes" in output
    assert "Context     unverified    runtime sizes the cache" in output
    assert "x 75% usable cap (cluster override)" in output
    assert est.to_dict()["memory_estimate_complete"] is True
    assert "dgx_spark_fit_budget_gb" not in est.to_dict()
    assert fit.status == "exceeds"


def test_fit_reports_assigned_gpu_only():
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import HostHardware, AcceleratorSpec
    from sparkrun.core.scheduler import RankAssignment, RankSlot
    from sparkrun.models.fit import check_fit
    from sparkrun.utils.cli_formatters import _resolve_target_accelerator

    hardware = HostHardware(accelerators=[AcceleratorSpec("nvidia", "h100", memory_gb=40), AcceleratorSpec("nvidia", "h100", memory_gb=80)])
    cluster = ClusterDefinition(name="c", hosts=["h"], hosts_hardware={"h": hardware})
    placement = RankAssignment((RankSlot("h", 1),), ("h",))
    est = estimate_vram(model_vram=50, kv_cache_memory_bytes=8 * 1024**3, gpu_memory_utilization=0.1)
    assert check_fit(est, cluster, placement).ok
    assert check_fit(est, cluster, placement).per_host["h"].accelerator_memory_gb == 80
    assert _resolve_target_accelerator(cluster, placement) == (80, "h100")
