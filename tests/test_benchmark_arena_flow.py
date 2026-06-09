"""Tests for the --arena flag dispatch and arena_flow helper extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sparkrun.cli._benchmark import benchmark as benchmark_group


def _fake_bench_result(**kw):
    obj = MagicMock()
    obj.success = True
    obj.benchmark_id = "x"
    fw = MagicMock()
    fw.framework_name = "llama-benchy"
    fw.primary_category = "performance"
    obj.framework = fw
    obj.profile = None
    obj.results = {"csv": "col1,col2\n1,2\n"}
    obj.outputs = {}
    obj.cluster_id = "c"
    obj.host_list = []
    obj.container_image = "img"
    obj.container_image_sha = None
    obj.container_image_sha_pinned = False
    obj.longterm_image_ref = None
    obj.longterm_image_pinned = False
    obj.benchmark_args = {}
    obj.state_dir = None
    obj.resumed = False
    obj.submission_id = None
    obj.launch_result = None
    obj.recipe = MagicMock()
    obj.overrides = {}
    for k, v in kw.items():
        setattr(obj, k, v)
    return obj


def test_arena_flag_triggers_preflight_and_finalize():
    """`benchmark perf --arena --local-test` runs preflight + finalize in order."""
    captured = {"order": [], "kwargs": {}}

    def _capture(ctx, **kwargs):
        captured["order"].append("run_benchmark")
        captured["kwargs"].update(kwargs)
        return _fake_bench_result()

    def _fake_preflight(*, local_test, ctx):
        captured["order"].append("preflight")
        return ("sub-test-123", "@official/spark-arena-v2")

    def _fake_finalize(**kw):
        captured["order"].append("finalize")
        captured["finalize_kwargs"] = kw

    runner = CliRunner()
    with (
        patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture),
        patch("sparkrun.cli._arena_flow.preflight_arena", side_effect=_fake_preflight),
        patch("sparkrun.cli._arena_flow.finalize_arena", side_effect=_fake_finalize),
    ):
        runner.invoke(
            benchmark_group,
            ["perf", "my-recipe", "--arena", "--local-test"],
            catch_exceptions=False,
        )

    assert captured["order"] == ["preflight", "run_benchmark", "finalize"]
    assert captured["kwargs"].get("submission_id_for_extras") == "sub-test-123"
    assert captured["kwargs"].get("profile") == "@official/spark-arena-v2"
    assert captured["finalize_kwargs"]["submission_id"] == "sub-test-123"
    assert captured["finalize_kwargs"]["local_test"] is True


def test_arena_flag_does_not_override_explicit_profile():
    """If user passes --profile explicitly with --arena, explicit profile wins."""
    captured = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    runner = CliRunner()
    with (
        patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture),
        patch("sparkrun.cli._arena_flow.preflight_arena", return_value=("s", "@official/spark-arena-v2")),
        patch("sparkrun.cli._arena_flow.finalize_arena"),
    ):
        runner.invoke(
            benchmark_group,
            ["perf", "my-recipe", "--arena", "--local-test", "--profile", "custom"],
            catch_exceptions=False,
        )
    assert captured.get("profile") == "custom"


def test_arena_flag_absent_no_preflight():
    """Without --arena, neither preflight nor finalize should be called."""
    captured = {"preflight_called": False, "finalize_called": False}

    def _capture(ctx, **kwargs):
        return _fake_bench_result()

    def _fake_preflight(**kw):
        captured["preflight_called"] = True
        return ("s", "p")

    def _fake_finalize(**kw):
        captured["finalize_called"] = True

    runner = CliRunner()
    with (
        patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture),
        patch("sparkrun.cli._arena_flow.preflight_arena", side_effect=_fake_preflight),
        patch("sparkrun.cli._arena_flow.finalize_arena", side_effect=_fake_finalize),
    ):
        runner.invoke(
            benchmark_group,
            ["perf", "my-recipe"],
            catch_exceptions=False,
        )

    assert not captured["preflight_called"]
    assert not captured["finalize_called"]


def test_arena_benchmark_run_uses_same_arena_flow_helpers():
    """`sparkrun arena benchmark run <r>` delegates to preflight_arena + finalize_arena."""
    from sparkrun.cli._arena import arena_benchmark

    order = []

    def _capture(ctx, *args, **kwargs):
        order.append("run_benchmark")
        return _fake_bench_result()

    def _fake_preflight(*, local_test, ctx):
        order.append("preflight")
        return ("s", "@official/spark-arena-v2")

    def _fake_finalize(**kw):
        order.append("finalize")

    runner = CliRunner()
    with (
        patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture),
        patch("sparkrun.cli._arena_flow.preflight_arena", side_effect=_fake_preflight),
        patch("sparkrun.cli._arena_flow.finalize_arena", side_effect=_fake_finalize),
    ):
        runner.invoke(arena_benchmark, ["run", "my-recipe", "--local-test"], catch_exceptions=False)

    assert order == ["preflight", "run_benchmark", "finalize"]


def test_arena_benchmark_run_threads_dry_run_to_benchmark():
    """``arena benchmark run`` must pass dry_run through the benchmark wrapper."""
    from sparkrun.api._benchmark_models import ResumeMode
    from sparkrun.cli._arena import arena_benchmark

    captured = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return _fake_bench_result()

    runner = CliRunner()
    with (
        patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture),
        patch("sparkrun.cli._arena_flow.preflight_arena", return_value=("s", "@official/spark-arena-v2")),
        patch("sparkrun.cli._arena_flow.finalize_arena"),
    ):
        runner.invoke(
            arena_benchmark,
            ["my-recipe", "--local-test", "--dry-run", "--tp", "2"],
            catch_exceptions=False,
        )

    assert captured["recipe_name"] == "my-recipe"
    assert captured["tensor_parallel"] == 2
    assert captured["api_key_env"] is None
    assert captured["dry_run"] is True
    assert captured["profile"] == "@official/spark-arena-v2"
    assert captured["export_results_files"] is False
    assert captured["resume_mode"] == ResumeMode.AUTO


def test_arena_benchmark_run_no_finalize_on_failure():
    """``arena benchmark run`` must NOT call finalize_arena when the benchmark fails."""
    from sparkrun.cli._arena import arena_benchmark

    finalize_called = []

    def _capture(ctx, *args, **kwargs):
        result = _fake_bench_result()
        result.success = False
        return result

    def _fake_finalize(**kw):
        finalize_called.append(True)

    runner = CliRunner()
    with (
        patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_capture),
        patch("sparkrun.cli._arena_flow.preflight_arena", return_value=("s", None)),
        patch("sparkrun.cli._arena_flow.finalize_arena", side_effect=_fake_finalize),
    ):
        runner.invoke(arena_benchmark, ["run", "my-recipe", "--local-test"], catch_exceptions=False)

    assert not finalize_called


def test_api_arena_defaults_profile_and_category():
    """api.benchmark(BenchmarkOptions(arena=True)) defaults profile and category."""
    from sparkrun.api import benchmark, BenchmarkOptions

    captured = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_bench_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        try:
            benchmark(BenchmarkOptions(recipe="my-recipe", arena=True))
        except SystemExit:
            pass

    assert captured
    assert captured[0].profile == "@official/spark-arena-v2"
    assert captured[0].category == "performance"


def test_api_arena_respects_explicit_profile():
    """When profile is explicit with arena=True, explicit profile is preserved."""
    from sparkrun.api import benchmark, BenchmarkOptions

    captured = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_bench_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        try:
            benchmark(BenchmarkOptions(recipe="my-recipe", arena=True, profile="@local/test"))
        except SystemExit:
            pass

    assert captured
    assert captured[0].profile == "@local/test"


def test_api_arena_respects_explicit_category():
    """When category is explicit with arena=True, explicit category is preserved."""
    from sparkrun.api import benchmark, BenchmarkOptions

    captured = []

    def _capture(options, *, sctx, emitter):
        captured.append(options)
        return _fake_bench_result()

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_capture):
        try:
            benchmark(BenchmarkOptions(recipe="my-recipe", arena=True, category="evals"))
        except SystemExit:
            pass

    assert captured
    assert captured[0].category == "evals"


def test_finalize_arena_dry_run_skips_metadata_generation(capsys):
    """``finalize_arena`` must not inspect benchmark artifacts in dry-run mode."""
    from sparkrun.cli._arena_flow import finalize_arena

    bench_result = MagicMock()
    bench_result.generate_metadata.side_effect = AssertionError("dry-run metadata is unavailable")

    finalize_arena(
        ctx=MagicMock(),
        bench_result=bench_result,
        submission_id="sub-test-123",
        local_test=False,
        dry_run=True,
    )

    assert "[dry-run] Would upload results to Spark Arena" in capsys.readouterr().out
    bench_result.generate_metadata.assert_not_called()


def test_arena_flow_module_constants():
    """ARENA_BENCHMARK_PROFILE constant matches the expected value."""
    from sparkrun.cli._arena_flow import ARENA_BENCHMARK_PROFILE

    assert ARENA_BENCHMARK_PROFILE == "@official/spark-arena-v2"


def test_arena_flow_exports():
    """_arena_flow __all__ exposes expected names."""
    import sparkrun.cli._arena_flow as m

    assert hasattr(m, "preflight_arena")
    assert hasattr(m, "finalize_arena")
    assert hasattr(m, "persist_arena_extras")
    assert hasattr(m, "ARENA_BENCHMARK_PROFILE")
