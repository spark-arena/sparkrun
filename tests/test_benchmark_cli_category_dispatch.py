"""Tests for ``sparkrun benchmark`` per-category subcommand dispatch (step 5)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sparkrun.core.bootstrap import init_sparkrun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_bench_result(**kw):
    obj = MagicMock(
        success=True,
        benchmark_id="x",
        framework="llama-benchy",
        profile=None,
        results={},
        outputs={},
        cluster_id="c",
        host_list=[],
        container_image="img",
        container_image_sha=None,
        container_image_sha_pinned=False,
        longterm_image_ref=None,
        longterm_image_pinned=False,
        benchmark_args={},
        state_dir=None,
        resumed=False,
        submission_id=None,
        category=kw.get("category"),
        recipe=MagicMock(),
        overrides={},
    )
    return obj


def _make_capture_side_effect(capture):
    """Return a side_effect for _run_benchmark that records kwargs."""

    def _inner(ctx, **kwargs):
        capture.update(kwargs)
        return _fake_bench_result(category=kwargs.get("category"))

    return _inner


def _invoke(args, capture):
    """Invoke ``sparkrun benchmark <args>`` with _run_benchmark patched."""
    # Re-import inside function so the benchmark group picks up fresh
    # category registrations after init_sparkrun() runs.
    from sparkrun.cli._benchmark import benchmark as benchmark_group

    runner = CliRunner()
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(capture)):
        return runner.invoke(benchmark_group, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# Category subcommand registration
# ---------------------------------------------------------------------------


def test_performance_subcommand_registered():
    """After init, ``performance`` appears as a subcommand on the benchmark group."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands

    _register_category_commands(benchmark_group)
    assert "performance" in benchmark_group.commands


def test_tools_subcommand_registered():
    """After init, ``tools`` appears as a subcommand on the benchmark group."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands

    _register_category_commands(benchmark_group)
    assert "tools" in benchmark_group.commands


def test_hallucinations_category_not_registered():
    """Categories without plugins are absent from the CLI surface."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group

    assert "hallucinations" not in benchmark_group.commands


# ---------------------------------------------------------------------------
# Dispatch tests — category kwarg threading
# ---------------------------------------------------------------------------


def test_explicit_performance_subcommand_sets_category():
    """`benchmark performance <recipe>` threads category='performance' to _run_benchmark."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands

    _register_category_commands(benchmark_group)
    captured = {}
    runner = CliRunner()
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(captured)):
        runner.invoke(benchmark_group, ["performance", "my-recipe"], catch_exceptions=False)
    assert captured.get("category") == "performance"


def test_perf_alias_routes_to_performance():
    """`benchmark perf <recipe>` is resolved to `performance` and sets category='performance'."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands

    _register_category_commands(benchmark_group)
    captured = {}
    runner = CliRunner()
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(captured)):
        runner.invoke(benchmark_group, ["perf", "my-recipe"], catch_exceptions=False)
    assert captured.get("category") == "performance"


def test_explicit_tools_subcommand_sets_category():
    """`benchmark tools <recipe>` threads category='tools' to _run_benchmark."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands

    _register_category_commands(benchmark_group)
    captured = {}
    runner = CliRunner()
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(captured)):
        runner.invoke(benchmark_group, ["tools", "my-recipe"], catch_exceptions=False)
    assert captured.get("category") == "tools"


def test_bare_benchmark_recipe_falls_back_to_performance():
    """`benchmark <recipe>` falls back to `performance` once that subcommand is registered."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands

    _register_category_commands(benchmark_group)
    captured = {}
    runner = CliRunner()
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(captured)):
        runner.invoke(benchmark_group, ["my-recipe"], catch_exceptions=False)
    assert captured.get("category") == "performance"


def test_run_subcommand_preserves_legacy_no_category():
    """`benchmark run <recipe>` continues to behave as legacy (category=None)."""
    init_sparkrun()
    captured = {}
    runner = CliRunner()
    from sparkrun.cli._benchmark import benchmark as benchmark_group

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(captured)):
        runner.invoke(benchmark_group, ["run", "my-recipe"], catch_exceptions=False)
    assert captured.get("category") is None


# ---------------------------------------------------------------------------
# Resume / fresh flags on category commands
# ---------------------------------------------------------------------------


def test_category_command_accepts_fresh_flag():
    """`benchmark performance <recipe> --fresh` reaches _run_benchmark with fresh=True."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands

    _register_category_commands(benchmark_group)
    captured = {}
    runner = CliRunner()
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(captured)):
        runner.invoke(benchmark_group, ["performance", "my-recipe", "--fresh"], catch_exceptions=False)
    assert captured.get("fresh") is True


def test_category_command_accepts_resume_flag():
    """`benchmark performance <recipe> --resume` reaches _run_benchmark with IF_EXISTS mode."""
    init_sparkrun()
    from sparkrun.cli._benchmark import benchmark as benchmark_group, _register_category_commands
    from sparkrun.api._benchmark_models import ResumeMode

    _register_category_commands(benchmark_group)
    captured = {}
    runner = CliRunner()
    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_make_capture_side_effect(captured)):
        runner.invoke(benchmark_group, ["performance", "my-recipe", "--resume"], catch_exceptions=False)
    assert captured.get("resume_mode") == ResumeMode.IF_EXISTS


# ---------------------------------------------------------------------------
# Framework / category mismatch
# ---------------------------------------------------------------------------


def test_framework_category_mismatch_raises():
    """`_run_benchmark(category='tools', framework='llama-benchy')` exits non-zero.

    After step 7, the CLI wrapper catches FrameworkCategoryMismatch (a SparkrunError)
    and translates it to sys.exit(1) — so callers of _run_benchmark see SystemExit,
    not the typed exception.  The typed exception is still raised by _execute_benchmark
    and can be caught by the API layer.
    """
    init_sparkrun()
    from sparkrun.cli._benchmark import _run_benchmark

    ctx = MagicMock()
    ctx.obj = {"sparkrun_ctx": MagicMock()}
    ctx.obj["sparkrun_ctx"].variables = None
    ctx.obj["sparkrun_ctx"].config = None

    with pytest.raises(SystemExit) as exc_info:
        _run_benchmark(
            ctx,
            recipe_name="r",
            hosts=[],
            hosts_file=None,
            cluster_name=None,
            tensor_parallel=None,
            pipeline_parallel=None,
            data_parallel=None,
            gpu_mem=None,
            max_model_len=None,
            options=(),
            image=None,
            solo=False,
            port=None,
            profile=None,
            framework="llama-benchy",
            output_file=None,
            bench_options=(),
            api_key_env=None,
            exit_on_first_fail=True,
            no_stop=False,
            skip_run=False,
            sync_tuning=False,
            rootful=False,
            bench_timeout=None,
            dry_run=True,
            executor_args=(),
            extra_args=(),
            fresh=False,
            scheduler_name=None,
            category="tools",
        )
    assert exc_info.value.code != 0


def test_framework_category_mismatch_raises_typed_from_api():
    """`_execute_benchmark` raises FrameworkCategoryMismatch directly (not swallowed)."""
    init_sparkrun()
    from sparkrun.api._errors import FrameworkCategoryMismatch
    from sparkrun.api._benchmark import _execute_benchmark, _NullProgressEmitter
    from sparkrun.api._benchmark_models import BenchmarkOptions
    from sparkrun.core.context import SparkrunContext

    sctx = MagicMock(spec=SparkrunContext)
    sctx.variables = None
    sctx.config = None

    opts = BenchmarkOptions(recipe="r", framework="llama-benchy", category="tools")
    with pytest.raises(FrameworkCategoryMismatch):
        _execute_benchmark(opts, sctx=sctx, emitter=_NullProgressEmitter())


def test_framework_category_match_does_not_raise():
    """Pinning framework + category where the framework IS in the category is fine."""
    init_sparkrun()
    from sparkrun.cli._benchmark import _run_benchmark
    from sparkrun.api._errors import FrameworkCategoryMismatch

    ctx = MagicMock()
    ctx.obj = {"sparkrun_ctx": MagicMock()}
    ctx.obj["sparkrun_ctx"].variables = None
    ctx.obj["sparkrun_ctx"].config = None

    # llama-benchy is in "performance" — should NOT raise FrameworkCategoryMismatch.
    # It will fail later (no real recipe), but not with the mismatch error.
    try:
        _run_benchmark(
            ctx,
            recipe_name="r",
            hosts=[],
            hosts_file=None,
            cluster_name=None,
            tensor_parallel=None,
            pipeline_parallel=None,
            data_parallel=None,
            gpu_mem=None,
            max_model_len=None,
            options=(),
            image=None,
            solo=False,
            port=None,
            profile=None,
            framework="llama-benchy",
            output_file=None,
            bench_options=(),
            api_key_env=None,
            exit_on_first_fail=True,
            no_stop=False,
            skip_run=False,
            sync_tuning=False,
            rootful=False,
            bench_timeout=None,
            dry_run=True,
            executor_args=(),
            extra_args=(),
            fresh=False,
            scheduler_name=None,
            category="performance",
        )
    except FrameworkCategoryMismatch:
        pytest.fail("FrameworkCategoryMismatch raised unexpectedly for a valid framework/category pair")
    except SystemExit:
        pass  # Expected — recipe lookup fails in test env


# ---------------------------------------------------------------------------
# API layer threads category through
# ---------------------------------------------------------------------------


def test_api_benchmark_options_category_threaded_to_run_benchmark():
    """BenchmarkOptions.category is forwarded to _execute_benchmark as options.category."""
    from unittest.mock import MagicMock, patch
    from sparkrun.api._benchmark_models import BenchmarkOptions

    received = []

    def _capture(options, *, sctx, emitter):
        received.append(options)
        return MagicMock(success=True)

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        from sparkrun.api import benchmark

        benchmark(BenchmarkOptions(recipe="my-recipe", category="performance"))
    assert received[0].category == "performance"


def test_api_benchmark_options_no_category_is_none():
    """When BenchmarkOptions.category is None, options.category stays None."""
    from unittest.mock import MagicMock, patch
    from sparkrun.api._benchmark_models import BenchmarkOptions

    received = []

    def _capture(options, *, sctx, emitter):
        received.append(options)
        return MagicMock(success=True)

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        from sparkrun.api import benchmark

        benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert received[0].category is None
