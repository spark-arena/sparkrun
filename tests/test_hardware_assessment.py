"""Hardware checks decided at plan time, and fit verdicts that name their cause."""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest
import yaml

import sparkrun.api as api
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.hardware_assessment import assess_launch_hardware, driver_mismatch_warnings
from sparkrun.models.vram import VRAMEstimate

_HOSTS = ["h1", "h2"]


def _gb10(driver: str = "610.43.02", source: str = "detected", model: str = "gb10") -> HostHardware:
    return HostHardware(
        accelerators=[AcceleratorSpec("nvidia", model, capabilities=frozenset({"cuda", "rdma:roce-v2"}))],
        source=source,
        driver_versions={"nvidia": driver} if driver else {},
        ib_info={"IB_DETECTED": "1", "DETECTED_NET_LIST": "roce0"},
    )


class _Runtime:
    runtime_name = "stub"
    requires_capability: frozenset = frozenset()


# --------------------------------------------------------------------------
# driver_mismatch_warnings
# --------------------------------------------------------------------------


def test_driver_branches_differing_across_hosts_warns():
    (warning,) = driver_mismatch_warnings({"h1": _gb10("610.43.02"), "h2": _gb10("580.173.02")})
    assert "nvidia driver branches differ" in warning
    assert "580.173.02 (h2)" in warning and "610.43.02 (h1)" in warning


def test_driver_versions_within_a_branch_are_named_as_versions():
    (warning,) = driver_mismatch_warnings({"h1": _gb10("580.95.05"), "h2": _gb10("580.173.02")})
    assert "nvidia driver versions differ" in warning


@pytest.mark.parametrize(
    "hardware",
    [
        {"h1": _gb10("610.43.02"), "h2": _gb10("610.43.02")},
        {"h1": _gb10("610.43.02")},
        {"h1": _gb10("610.43.02"), "h2": _gb10("")},
    ],
    ids=["matching", "single-host", "unknown-version"],
)
def test_no_driver_warning_without_a_disagreement(hardware):
    assert driver_mismatch_warnings(hardware) == []


# --------------------------------------------------------------------------
# assess_launch_hardware
# --------------------------------------------------------------------------


def test_evidence_and_driver_check_cover_only_detected_hosts():
    # h2's record is inventory: history, not a fact about the host now.
    cluster = ClusterDefinition(name="c", hosts=_HOSTS, hosts_hardware={"h1": _gb10("610.43.02"), "h2": _gb10("580.173.02", "inventory")})
    assessment = assess_launch_hardware(_Runtime(), _HOSTS, cluster, None, {"h1": cluster.hosts_hardware["h1"]})
    assert list(assessment.evidence) == ["h1"]
    assert "nvidia/gb10" in assessment.evidence["h1"]
    assert not any("driver" in w for w in assessment.warnings)


def test_incompatible_runtime_is_an_error_not_a_warning():
    runtime = _Runtime()
    runtime.requires_capability = frozenset({"gb10"})
    hw = _gb10(model="h100-80gb-hbm3")
    cluster = ClusterDefinition(name="c", hosts=["h1"], hosts_hardware={"h1": hw})
    assessment = assess_launch_hardware(runtime, ["h1"], cluster, None, {"h1": hw})
    assert assessment.errors and "h1" in assessment.errors[0]


def test_assumed_hardware_is_warned():
    assessment = assess_launch_hardware(_Runtime(), ["h1"], None, None)
    assert any("Host h1 hardware is assumed" in w for w in assessment.warnings)
    assert assessment.evidence == {}


# --------------------------------------------------------------------------
# api.plan carries it; api.run hands it on
# --------------------------------------------------------------------------


@pytest.fixture
def recipe(tmp_path):
    from sparkrun.core.recipe import Recipe

    path = tmp_path / "assess.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "sparkrun_version": "2",
                "model": "Qwen/Qwen3-1.7B",
                "runtime": "sglang",
                "mode": "cluster",
                "container": "scitrera/dgx-spark-sglang:latest",
                "defaults": {"port": 30000, "tensor_parallel": 2},
            }
        )
    )
    return Recipe.load(str(path), resolve=False)


def _live_plan(recipe, monkeypatch, observed):
    monkeypatch.setattr("sparkrun.core.hardware_probe.probe_hosts", lambda hosts, **kw: {h: observed[h] for h in hosts})
    monkeypatch.setattr(
        api, "status", lambda hosts, **kw: ClusterStatus(hosts=tuple(HostOccupancy(host=h, free_slots=1) for h in hosts), executor="docker")
    )
    monkeypatch.setattr(
        recipe,
        "estimate_vram",
        lambda **kw: VRAMEstimate(
            model_weights_gb=4.0,
            total_per_gpu_gb=2.0,
            tensor_parallel=2,
            kv_cache_per_token_bytes=None,
            kv_cache_total_gb=None,
            max_model_len=None,
        ),
    )
    options = api.RunOptions(
        recipe=recipe, cluster=ClusterDefinition(name="c", hosts=list(_HOSTS), scheduler="occupancy-sparse"), hosts=tuple(_HOSTS)
    )
    return options, api.plan(options)


def test_plan_reports_driver_mismatch_and_evidence_for_placed_hosts(recipe, monkeypatch, v):
    observed = {"h1": _gb10("610.43.02"), "h2": _gb10("580.173.02")}
    _, plan = _live_plan(recipe, monkeypatch, observed)
    assert set(plan.hardware_assessment.evidence) == set(plan.host_list) == set(_HOSTS)
    assert any("nvidia driver branches differ" in w for w in plan.hardware_assessment.warnings)
    assert plan.hardware_reported is False


def test_plan_refuses_incompatible_hardware_before_any_launch_phase(recipe, monkeypatch, v):
    from sparkrun.runtimes.sglang import SglangRuntime

    monkeypatch.setattr(SglangRuntime, "requires_capability", frozenset({"gb10"}))
    observed = {h: _gb10(model="h100-80gb-hbm3") for h in _HOSTS}
    with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
        with pytest.raises(api.SparkrunError, match="incompatible"):
            _live_plan(recipe, monkeypatch, observed)
    launch.assert_not_called()


@pytest.mark.parametrize("reported", [False, True])
def test_run_hands_the_plan_assessment_to_the_launch(recipe, monkeypatch, v, reported):
    options, plan = _live_plan(recipe, monkeypatch, {h: _gb10() for h in _HOSTS})
    if reported:
        plan = replace(plan, hardware_reported=True)
    with mock.patch("sparkrun.core.launcher.launch_inference") as launch:
        launch.return_value = mock.MagicMock(
            cluster_id=plan.cluster_id,
            host_list=list(plan.host_list),
            rc=0,
            is_solo=False,
            runtime_info={},
            recipe_ref=None,
            serve_command="",
            container_image="",
            serve_port=0,
            effective_cache_dir="",
        )
        api.run(options, plan=plan)
    assert launch.call_args.kwargs["hardware_assessment"] is plan.hardware_assessment
    assert launch.call_args.kwargs["hardware_reported"] is reported


# --------------------------------------------------------------------------
# UNVERIFIED names its cause
# --------------------------------------------------------------------------


def _estimate(**kw) -> VRAMEstimate:
    base = dict(
        model_weights_gb=155.4,
        total_per_gpu_gb=77.7,
        tensor_parallel=2,
        kv_cache_per_token_bytes=None,
        kv_cache_total_gb=None,
        max_model_len=None,
    )
    return VRAMEstimate(**{**base, **kw})


@pytest.mark.parametrize(
    "estimate, gap",
    [
        (_estimate(), "KV cache not sized (no max_model_len)"),
        (_estimate(max_model_len=8192), "KV cache could not be sized"),
        (
            _estimate(kv_cache_total_gb=4.0, kv_estimate_is_floor=True, kv_arch="mla"),
            "MLA KV estimate is a lower bound (auxiliary caches not counted)",
        ),
        (_estimate(model_weights_gb=0.0, kv_cache_total_gb=1.0), "model weight size unknown"),
    ],
)
def test_estimate_names_its_gap(estimate, gap):
    assert not estimate.memory_estimate_complete
    assert estimate.memory_estimate_gaps == (gap,)


def test_complete_estimate_has_no_gaps():
    estimate = _estimate(kv_cache_total_gb=4.0, max_model_len=8192)
    assert estimate.memory_estimate_complete and estimate.memory_estimate_gaps == ()


def test_detected_capacity_with_partial_estimate_blames_the_estimate_only(capsys):
    from sparkrun.models.fit import check_fit
    from sparkrun.utils.cli_formatters import display_vram_estimate

    hw = replace(_gb10(), accelerators=[replace(_gb10().accelerators[0], memory_gb=121.7)])
    cluster = ClusterDefinition(name="c", hosts=["h1"], hosts_hardware={"h1": hw})
    fit = check_fit(_estimate(), cluster)
    assert fit.status == "unknown"
    assert fit.unverified_reasons == ("KV cache not sized (no max_model_len)",)

    class _Recipe:
        def estimate_vram(self, **kw):
            return _estimate()

    display_vram_estimate(_Recipe(), cluster=cluster)
    out = capsys.readouterr().out
    assert "memory fit: UNVERIFIED (KV cache not sized (no max_model_len))" in out


def test_unknown_capacity_and_assumed_hardware_are_named():
    from sparkrun.models.fit import check_fit

    cluster = ClusterDefinition(
        name="c",
        hosts=["h1"],
        hosts_hardware={"h1": HostHardware(accelerators=[AcceleratorSpec("acme", "x1")], source="assumed")},
    )
    reasons = check_fit(_estimate(kv_cache_total_gb=4.0, max_model_len=8192), cluster).unverified_reasons
    # The estimate itself is complete; unknown capacity must not be reported as an estimate gap.
    assert reasons == ("accelerator capacity unknown", "hardware assumed, not probed")
