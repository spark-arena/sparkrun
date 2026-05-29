"""Tests for the sparkrun.api.benchmark public surface (step 7 — orchestration lifted)."""

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


def _fake_internal_result(**overrides):
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


def test_benchmark_translates_benchmarkfailed_raised_directly():
    """A BenchmarkFailed raised inside _execute_benchmark surfaces as BenchmarkFailed."""
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=BenchmarkFailed("boom", exit_code=2)):
        with pytest.raises(BenchmarkFailed) as excinfo:
            benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert excinfo.value.exit_code == 2


def test_benchmark_translates_unexpected_exception_to_sparkrunerror():
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=RuntimeError("boom")):
        with pytest.raises(SparkrunError):
            benchmark(BenchmarkOptions(recipe="my-recipe"))


def test_benchmark_propagates_keyboardinterrupt():
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            benchmark(BenchmarkOptions(recipe="my-recipe"))


def test_benchmark_translates_internal_result_to_api_result():
    fake = _fake_internal_result()
    with patch("sparkrun.api._benchmark._execute_benchmark", return_value=fake):
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
    fake = _fake_internal_result()
    with patch("sparkrun.api._benchmark._execute_benchmark", return_value=fake):
        result = benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert result.category == "performance"


def test_benchmark_honors_explicit_category():
    fake = _fake_internal_result()
    with patch("sparkrun.api._benchmark._execute_benchmark", return_value=fake):
        result = benchmark(BenchmarkOptions(recipe="my-recipe", category="evals"))
    assert result.category == "evals"


def test_benchmark_passes_bench_args_through():
    """bench_args dict is passed as BenchmarkOptions.bench_args to _execute_benchmark."""
    captured: list = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_internal_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", bench_args={"pp": "[2048]", "depth": 4}))

    assert captured
    opts = captured[0]
    assert opts.bench_args.get("pp") == "[2048]"
    assert opts.bench_args.get("depth") == 4


def test_benchmark_fresh_resume_mode_sets_fresh():
    """FRESH resume mode is passed correctly via BenchmarkOptions."""
    captured: list = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_internal_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", resume=ResumeMode.FRESH))
    assert captured[0].resume == ResumeMode.FRESH


def test_benchmark_default_resume_mode_is_if_exists():
    """Default BenchmarkOptions.resume is IF_EXISTS."""
    captured: list = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_internal_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert captured[0].resume == ResumeMode.IF_EXISTS


def test_benchmark_if_exists_resume_mode_passed():
    captured: list = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_internal_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", resume=ResumeMode.IF_EXISTS))
    assert captured[0].resume == ResumeMode.IF_EXISTS


def test_benchmark_threads_submission_id_through_state_extras():
    captured: list = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_internal_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        benchmark(
            BenchmarkOptions(
                recipe="my-recipe",
                arena=True,
                state_extras={"submission_id": "sub-abc-123"},
            )
        )
    opts = captured[0]
    assert opts.state_extras.get("submission_id") == "sub-abc-123"


def test_benchmark_null_state_extras_gives_no_submission_id():
    captured: list = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_internal_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe"))
    opts = captured[0]
    assert opts.state_extras.get("submission_id") is None


def test_benchmark_framework_string_in_result_when_internal_has_none():
    """When bench_result.framework is None, fall back to options.framework."""
    fake = _fake_internal_result()
    fake.framework = None
    with patch("sparkrun.api._benchmark._execute_benchmark", return_value=fake):
        result = benchmark(BenchmarkOptions(recipe="my-recipe", framework="my-fw"))
    assert result.framework == "my-fw"


def test_benchmark_passed_scheduler_in_options():
    captured: list = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_internal_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        benchmark(BenchmarkOptions(recipe="my-recipe", scheduler="greedy"))
    assert captured[0].scheduler == "greedy"


def test_benchmark_benchmarkfailed_exit_code_preserved():
    """BenchmarkFailed with specific exit code is re-raised unchanged."""
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=BenchmarkFailed("fail", exit_code=42)):
        with pytest.raises(BenchmarkFailed) as excinfo:
            benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert excinfo.value.exit_code == 42


def test_benchmark_sparkrunerror_propagates_unchanged():
    """A SparkrunError raised by _execute_benchmark is not re-wrapped."""
    original = SparkrunError("direct typed error")
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=original):
        with pytest.raises(SparkrunError) as excinfo:
            benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert excinfo.value is original


# ---------------------------------------------------------------------------
# Category threading: benchmark() forwards options.category to the
# orchestration body (_execute_benchmark) unchanged.  These replace the
# retired _build_run_benchmark_kwargs shim by asserting the real entry point.
# ---------------------------------------------------------------------------


def test_benchmark_threads_category_to_execute():
    """``benchmark()`` forwards ``options.category`` into ``_execute_benchmark``."""
    from unittest.mock import patch

    received = []

    def _capture(options, *, sctx, emitter):
        received.append(options)
        return MagicMock(success=True)

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        from sparkrun.api import benchmark

        benchmark(BenchmarkOptions(recipe="my-recipe", category="performance"))
    assert received[0].category == "performance"


def test_benchmark_no_category_threads_none():
    from unittest.mock import patch

    received = []

    def _capture(options, *, sctx, emitter):
        received.append(options)
        return MagicMock(success=True)

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        from sparkrun.api import benchmark

        benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert received[0].category is None
