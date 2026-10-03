"""The `sparkrun run` pre-launch summary: hosts table, then the memory plan sized to it."""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.hardware_assessment import assess_launch_hardware
from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.models.fit import check_fit
from sparkrun.models.vram import derive_model_max_len, estimate_vram
from sparkrun.utils.cli_formatters import format_host_table, format_memory_plan

_H = ["10.24.11.13", "10.24.11.14"]


class _Runtime:
    runtime_name = "stub"
    requires_capability: frozenset = frozenset()


def _hw(driver="610.43.02", memory=121.7, source="detected", ports="enp1s0f0np0,enP2p1s0f0np0", count=1) -> HostHardware:
    return HostHardware(
        accelerators=[AcceleratorSpec("nvidia", "gb10", count=count, memory_gb=memory, capabilities=frozenset({"cuda", "rdma:roce-v2"}))],
        source=source,
        driver_versions={"nvidia": driver} if driver else {},
        ib_info={"IB_DETECTED": "1", "DETECTED_NET_LIST": ports} if ports is not None else None,
    )


def _placement(hosts=_H, gpus_per_host=1):
    slots = tuple(RankSlot(h, g) for h in hosts for g in range(gpus_per_host))
    return RankAssignment(slots, tuple(hosts))


def _table(inventory, observed=None, hosts=_H, placement=None, **kw):
    cluster = ClusterDefinition(name="c", hosts=list(hosts), hosts_hardware=inventory)
    placement = placement or _placement(hosts)
    assessment = assess_launch_hardware(_Runtime(), hosts, cluster, placement, inventory if observed is None else observed)
    return "\n".join(format_host_table(assessment, list(hosts), placement, **kw))


def _row(text: str, host: str) -> list[str]:
    return next(line.split() for line in text.splitlines() if host in line.split())


# --------------------------------------------------------------------------
# format_host_table
# --------------------------------------------------------------------------


def test_fully_probed_cluster_reads_clean():
    out = _table({h: _hw() for h in _H})
    assert out.splitlines()[0].split() == ["ROLE", "HOST", "GPU", "MEMORY", "DRIVER", "RDMA"]
    assert _row(out, _H[0]) == ["head", _H[0], "gb10", "121.7", "GiB", "610.43.02", "2", "ports"]
    assert _row(out, _H[1])[0] == "worker"
    assert "All facts probed this run. RDMA links not tested." in out
    for mark in ("*", "~", "!"):
        assert mark not in out


def test_driver_that_differs_from_head_is_marked():
    out = _table({_H[0]: _hw("610.43.02"), _H[1]: _hw("580.173.02")})
    assert _row(out, _H[1])[5:7] == ["580.173.02", "!"]
    assert "! differs from head" in out
    assert "All facts probed" not in out


def test_inventory_and_assumed_facts_are_marked_with_one_legend_each():
    inventory = {_H[0]: _hw(source="inventory"), _H[1]: _hw()}
    out = _table(inventory, observed={_H[1]: inventory[_H[1]]})
    assert _row(out, _H[0])[2:4] == ["gb10", "*"]
    assert "* from saved cluster inventory, not probed this run" in out

    out = _table({}, observed={}, dry_run=True)
    assert _row(out, _H[0])[2:6] == ["gb10", "~", "121.0", "GiB"]
    assert out.count("~ assumed by platform policy, not probed (dry run doesn't probe)") == 1


def test_ranks_column_only_when_a_host_runs_several():
    inventory = {h: _hw(count=2) for h in _H}
    out = _table(inventory, placement=_placement(gpus_per_host=2))
    assert "RANKS" in out.splitlines()[0]
    assert _row(out, _H[0])[2:4] == ["2", "gb10"]
    assert "x2" in out


def test_verbose_names_interfaces_and_appends_evidence():
    out = _table({h: _hw() for h in _H}, verbose=True)
    assert "enp1s0f0np0, enP2p1s0f0np0" in out
    assert "%s: GPU 0 nvidia/gb10 detected" % _H[0] in out


def test_missing_rdma_and_driver_render_as_dashes():
    out = _table({h: _hw(driver="", ports=None) for h in _H})
    assert _row(out, _H[0])[-2:] == ["—", "—"]
    assert "RDMA links not tested" not in out


# --------------------------------------------------------------------------
# format_memory_plan
# --------------------------------------------------------------------------


def _plan(memories=(121.7, 121.7), **estimate_kw) -> str:
    inventory = {h: _hw(memory=m) for h, m in zip(_H, memories, strict=True)}
    cluster = ClusterDefinition(name="c", hosts=_H, hosts_hardware=inventory)
    placement = _placement()
    base = dict(model_vram=155.43, model_dtype="fp8", kv_dtype="fp8", tensor_parallel=2, total_gpu_memory_gb=min(memories))
    est = estimate_vram(**{**base, **estimate_kw})
    return "\n".join(format_memory_plan(est, check_fit(est, cluster, placement), hosts=2))


def test_each_number_appears_once_with_its_arithmetic():
    out = _plan(gpu_memory_utilization=0.85, num_layers=43, num_kv_heads=8, head_dim=128, model_max_len=131072)
    lines = out.splitlines()
    assert lines[0] == "Memory per GPU (tp=2, sized to the smallest host above):"
    assert lines[1].split() == ["Weights", "77.7", "GiB", "155.4", "GiB", "fp8", "/", "2"]
    assert lines[2].split()[:3] == ["Budget", "103.4", "GiB"] and "121.7 GiB x 85% gpu_memory_utilization" in lines[2]
    assert lines[3].split()[:4] == ["KV", "space", "25.7", "GiB"]
    assert out.count("25.7") == 1  # no per-host rows repeating it when hosts agree
    assert "Fit         UNVERIFIED    KV cache not sized (no max_model_len)" in out


def test_budget_derived_context_is_bounded_by_the_model_limit():
    out = _plan(gpu_memory_utilization=0.85, num_layers=4, num_kv_heads=1, head_dim=64, model_max_len=131072)
    assert "M tokens of fp8 cache; model limit 131,072" in out


def test_context_within_the_model_limit_reports_the_multiplier():
    out = _plan(gpu_memory_utilization=0.85, num_layers=43, num_kv_heads=8, head_dim=128, max_model_len=8192, model_max_len=10**9)
    assert "x max_model_len)" in out
    assert "model limit" not in out


def test_hosts_that_differ_get_rows_and_the_failing_one_is_named():
    out = _plan(memories=(121.7, 62.0), gpu_memory_utilization=0.85)
    assert "Budget        52.7 GiB    62.0 GiB x 85% gpu_memory_utilization" in out
    assert "KV space       0.0 GiB    none: the weights alone exceed the budget" in out
    assert "Fit            EXCEEDS    on %s" % _H[1] in out
    rows = [line.split() for line in out.splitlines() if line.startswith("    ")]
    assert rows[0][:5] == [_H[0], "103.4", "GiB", "budget", "25.7"]
    assert rows[1][-1] == "EXCEEDS"


def test_without_a_runtime_budget_the_spare_comes_from_the_fit():
    out = _plan(num_layers=43, num_kv_heads=8, head_dim=128, max_model_len=8192)
    assert "KV demand" in out
    assert "  Spare  " in out and "after weights and KV demand" in out
    assert "KV space" not in out


# --------------------------------------------------------------------------
# derive_model_max_len (mirrors the serving engine's derivation)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({"max_position_embeddings": 32768}, 32768),
        ({"max_position_embeddings": 32768, "seq_length": 8192}, 8192),
        ({"max_position_embeddings": 32768, "rope_scaling": {"rope_type": "linear", "factor": 4}}, 131072),
        (
            {"max_position_embeddings": 32768, "rope_scaling": {"type": "yarn", "factor": 4, "original_max_position_embeddings": 8192}},
            32768,
        ),
        ({"max_position_embeddings": 131072, "rope_scaling": {"rope_type": "llama3", "factor": 8}}, 131072),
        ({"max_position_embeddings": True}, None),
        ({}, None),
    ],
    ids=["plain", "smallest-key", "linear-rope", "yarn-original", "llama3-already-extended", "bool-ignored", "absent"],
)
def test_derive_model_max_len(cfg, expected):
    assert derive_model_max_len(cfg) == expected


def test_detected_model_limit_survives_later_estimates_via_metadata(monkeypatch):
    from sparkrun.core.recipe import Recipe

    calls = []

    def fake_config(*a, **kw):
        calls.append(1)
        return {
            "num_hidden_layers": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 40960,
        }

    monkeypatch.setattr("sparkrun.models.vram.fetch_model_config", fake_config)
    monkeypatch.setattr("sparkrun.models.quantization.fetch_hf_quant_config", lambda *a, **kw: None)
    recipe = Recipe.from_dict({"model": "org/m", "runtime": "vllm-distributed", "metadata": {"model_vram": 2}})
    assert recipe.estimate_vram().model_max_len == 40960
    assert recipe.metadata["model_max_len"] == 40960
    # Detection is satisfied now; the limit must come back from metadata.
    assert recipe.estimate_vram().model_max_len == 40960
    assert len(calls) == 1
