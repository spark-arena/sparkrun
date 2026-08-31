"""Launch-stage timing in the benchmark provenance artifact.

The numbers land nested under ``metadata["timing"]``, beside the existing
benchmark ``start``/``end``/``duration`` — additively, because that mapping
is the Spark Arena submission and consumers key off the existing shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from sparkrun.benchmarking.base import BenchmarkResult
from sparkrun.core.launcher import ServeReadiness
from sparkrun.core.timing import Timeline


def _recipe() -> MagicMock:
    recipe = MagicMock()
    recipe.name = "r"
    recipe.qualified_name = "@reg/r"
    recipe.container = "img:latest"
    recipe.model = "org/model"
    recipe.runtime = "vllm-distributed"
    recipe.metadata = {}
    recipe.model_revision = None
    recipe.source_registry = "reg"
    recipe.source_registry_url = ""
    recipe.export.return_value = "yaml"
    return recipe


def _result(*, timeline=None, readiness=None, resumed=False) -> BenchmarkResult:
    br = BenchmarkResult()
    now = datetime.now(timezone.utc)
    br.start_time = now
    br.end_time = now
    br.recipe = _recipe()
    br.overrides = {}
    br.cluster_id = "cid"
    br.host_list = ["h1"]
    br.container_image = "img:latest"
    br.framework = MagicMock()
    br.framework.framework_name = "llama-benchy"
    br.profile = "default"
    br.benchmark_args = {}
    br.resumed = resumed
    br.readiness = readiness
    if timeline is not None:
        launch_result = MagicMock()
        launch_result.builder = None
        launch_result.timeline = timeline
        # generate_metadata prefers launch_result fields when present.
        launch_result.recipe = br.recipe
        launch_result.overrides = {}
        launch_result.cluster_id = "cid"
        launch_result.host_list = ["h1"]
        launch_result.container_image = "img:latest"
        launch_result.runtime_info = {}
        br.launch_result = launch_result
    return br


def _readiness(port_s: float, health_s: float) -> ServeReadiness:
    return ServeReadiness(
        ready=True,
        head_host="h1",
        head_ip="10.0.0.1",
        port=8000,
        container="cid_node_0",
        port_wait_s=port_s,
        health_wait_s=health_s,
    )


def test_timing_carries_the_full_span_list_and_serve_ready_breakdown():
    tl = Timeline()
    with tl.span("launch"):
        with tl.span("launch.distribute", mode="push"):
            pass

    meta = _result(timeline=tl, readiness=_readiness(41.5, 88.25)).generate_metadata()
    timing = meta["timing"]

    # Existing keys are untouched — this is an additive change to a
    # published artifact.
    assert {"start", "end", "duration"} <= set(timing)

    assert timing["serve_ready"] == {"port_open_s": 41.5, "health_ok_s": 88.25, "total_s": 129.75}

    names = [s["name"] for s in timing["launch"]["spans"]]
    assert names == ["launch", "launch.distribute"]
    assert timing["launch"]["spans"][1]["attrs"]["mode"] == "push"
    assert "wall_origin" in timing["launch"]


def test_nothing_launched_omits_the_keys_rather_than_zeroing_them():
    """``--skip-run`` measured a workload it did not start.

    A recorded ``total_s: 0.0`` would read as an instantaneous launch.
    """
    timing = _result().generate_metadata()["timing"]
    assert "serve_ready" not in timing
    assert "launch" not in timing


def test_resumed_run_does_not_report_launch_timings():
    """A resumed run re-emits recorded results; its launch was a different one.

    Same reasoning as ``measured_at`` for the benchmark numbers themselves
    (issue #267) — provenance must not attribute a stale launch to this
    result.
    """
    tl = Timeline()
    with tl.span("launch"):
        pass

    timing = _result(timeline=tl, readiness=_readiness(1.0, 2.0), resumed=True).generate_metadata()["timing"]
    assert "serve_ready" not in timing
    assert "launch" not in timing
    # The benchmark's own timing keys still apply to this invocation.
    assert "duration" in timing
