"""Tests for the sparkrun.api.benchmark public surface (step 3 thin wrapper)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sparkrun.api import (
    BenchmarkOptions,
    BenchmarkResult,
    ResumeMode,
    SparkrunError,
    benchmark,
)
from sparkrun.api._errors import BenchmarkFailed


def _fake_bench_result(**overrides):
    """Build a stand-in for the internal sparkrun.benchmarking.base.BenchmarkResult."""
    obj = MagicMock()
    obj.success = True
    obj.benchmark_id = "bench_test123"
    # framework is a BenchmarkingPlugin object on the internal type; simulate that
    fw = MagicMock()
    fw.framework_name = "llama-benchy"
    fw.primary_category = "performance"
    obj.framework = fw
    obj.profile = None
    obj.results = {"throughput": 100}
    obj.outputs = {"yaml": "/tmp/x.yaml"}
    obj.cluster_id = "sparkrun_intent_abc"
    obj.host_list = ["host-1"]
    obj.container_image = "img:tag"
    obj.container_image_sha = "sha256:deadbeef"
    obj.container_image_sha_pinned = True
    obj.longterm_image_ref = "ghcr.io/x/y@sha256:eee"
    obj.longterm_image_pinned = True
    obj.benchmark_args = {"pp": [2048]}
    obj.state_dir = "/tmp/state"
    obj.resumed = False
    obj.submission_id = None
    for k, v in overrides.items():
        setattr(obj, k, v)
    return obj


def test_surface_exports():
    """Public API exposes all the expected names."""
    from sparkrun.api import (
        AmbiguousCategoryError,
        BenchmarkFailed,
        CategoryNotFound,
        FrameworkCategoryMismatch,
        NoResumableState,
        ProgressEvent,
        ResumeMode,
        benchmark,
    )

    assert callable(benchmark)
    assert ResumeMode.AUTO.value == "auto"
    assert ResumeMode.IF_EXISTS.value == "if_exists"
    assert ResumeMode.FRESH.value == "fresh"
    assert ResumeMode.REQUIRED.value == "required"
    # error classes are SparkrunError subclasses
    assert issubclass(BenchmarkFailed, SparkrunError)
    assert issubclass(NoResumableState, SparkrunError)
    assert issubclass(CategoryNotFound, SparkrunError)
    assert issubclass(AmbiguousCategoryError, SparkrunError)
    assert issubclass(FrameworkCategoryMismatch, SparkrunError)
    # ProgressEvent is a frozen dataclass
    ev = ProgressEvent(kind="test")
    assert ev.kind == "test"
    assert ev.data == {}


def test_options_dataclass_is_frozen():
    opts = BenchmarkOptions(recipe="my-recipe")
    with pytest.raises(Exception):
        opts.recipe = "other"  # type: ignore[misc]


def test_result_dataclass_is_frozen():
    r = BenchmarkResult(
        success=True,
        benchmark_id="x",
        category="performance",
        framework="llama-benchy",
        profile=None,
    )
    with pytest.raises(Exception):
        r.success = False  # type: ignore[misc]


def test_benchmark_translates_systemexit_to_benchmarkfailed():
    """A sys.exit(N) inside _run_benchmark surfaces as BenchmarkFailed."""
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=SystemExit(2)):
        with pytest.raises(BenchmarkFailed) as excinfo:
            benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert excinfo.value.exit_code == 2


def test_benchmark_translates_unexpected_exception_to_sparkrunerror():
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=RuntimeError("boom")):
        with pytest.raises(SparkrunError):
            benchmark(BenchmarkOptions(recipe="my-recipe"))


def test_benchmark_propagates_keyboardinterrupt():
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            benchmark(BenchmarkOptions(recipe="my-recipe"))


def test_benchmark_translates_internal_result_to_api_result():
    fake = _fake_bench_result()
    with patch("sparkrun.cli._benchmark._run_benchmark", return_value=fake):
        result = benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert isinstance(result, BenchmarkResult)
    assert result.success is True
    assert result.benchmark_id == "bench_test123"
    assert result.framework == "llama-benchy"
    assert result.cluster_id == "sparkrun_intent_abc"
    assert result.host_list == ("host-1",)
    assert result.container_image_sha == "sha256:deadbeef"
    assert result.container_image_sha_pinned is True
    assert result.container_image_longterm_ref == "ghcr.io/x/y@sha256:eee"
    assert result.outputs == {"yaml": "/tmp/x.yaml"}
    assert result.results == {"throughput": 100}


def test_benchmark_resolves_category_from_framework_primary():
    """primary_category is extracted from the plugin object on the internal result."""
    fake = _fake_bench_result()
    with patch("sparkrun.cli._benchmark._run_benchmark", return_value=fake):
        result = benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert result.category == "performance"


def test_benchmark_honors_explicit_category():
    fake = _fake_bench_result()
    with patch("sparkrun.cli._benchmark._run_benchmark", return_value=fake):
        result = benchmark(BenchmarkOptions(recipe="my-recipe", category="evals"))
    assert result.category == "evals"


def test_benchmark_passes_bench_args_through():
    """bench_args dict is rendered into key=value tuples for _run_benchmark."""
    captured: dict = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", bench_args={"pp": "[2048]", "depth": 4}))

    bench_options = captured.get("bench_options")
    assert bench_options is not None
    assert any("pp=" in s for s in bench_options)
    assert any("depth=" in s for s in bench_options)


def test_benchmark_fresh_resume_mode_sets_fresh_kwarg():
    captured: dict = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", resume=ResumeMode.FRESH))
    assert captured.get("fresh") is True


def test_benchmark_default_resume_mode_does_not_force_fresh():
    captured: dict = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert captured.get("fresh") is False


def test_benchmark_if_exists_resume_mode_does_not_force_fresh():
    captured: dict = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", resume=ResumeMode.IF_EXISTS))
    assert captured.get("fresh") is False


def test_benchmark_threads_submission_id_through_state_extras():
    captured: dict = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture):
        benchmark(
            BenchmarkOptions(
                recipe="my-recipe",
                arena=True,
                state_extras={"submission_id": "sub-abc-123"},
            )
        )
    assert captured.get("submission_id_for_extras") == "sub-abc-123"


def test_benchmark_null_state_extras_gives_none_submission_id():
    captured: dict = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert captured.get("submission_id_for_extras") is None


def test_benchmark_framework_string_in_result_when_internal_has_none():
    """When bench_result.framework is None, fall back to options.framework."""
    fake = _fake_bench_result()
    fake.framework = None
    with patch("sparkrun.cli._benchmark._run_benchmark", return_value=fake):
        result = benchmark(BenchmarkOptions(recipe="my-recipe", framework="my-fw"))
    assert result.framework == "my-fw"


def test_benchmark_passed_kwargs_include_scheduler_name():
    captured: dict = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", scheduler="greedy"))
    assert captured.get("scheduler_name") == "greedy"


def test_benchmark_exit_code_none_on_non_int_systemexit():
    """SystemExit with non-int code still produces BenchmarkFailed with code=1."""
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=SystemExit("oops")):
        with pytest.raises(BenchmarkFailed) as excinfo:
            benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert excinfo.value.exit_code == 1


def test_benchmark_sparkrunerror_propagates_unchanged():
    """A SparkrunError raised by _run_benchmark is not re-wrapped."""
    original = SparkrunError("direct typed error")
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=original):
        with pytest.raises(SparkrunError) as excinfo:
            benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert excinfo.value is original
