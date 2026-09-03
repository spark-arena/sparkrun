"""Tests for the --arena flag dispatch and arena_flow helper extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
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

    def _fake_preflight(*, local_test, ctx, recipe_name=None, dry_run=False):
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
        result = runner.invoke(
            benchmark_group,
            ["perf", "my-recipe", "--arena", "--local-test", "--hosts", "h1"],
            catch_exceptions=False,
        )

    # Asserted before the order check: without it, an early exit shows up as an
    # empty capture list rather than as the error that caused it.
    assert result.exit_code == 0, "CLI exited %s: %s" % (result.exit_code, result.output)
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
        result = runner.invoke(
            benchmark_group,
            ["perf", "my-recipe", "--arena", "--local-test", "--profile", "custom", "--hosts", "h1"],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, "CLI exited %s: %s" % (result.exit_code, result.output)
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
        result = runner.invoke(
            benchmark_group,
            ["perf", "my-recipe", "--hosts", "h1"],
            catch_exceptions=False,
        )

    # This test asserts only negatives, so it passes vacuously if the command
    # exits before dispatch. Pin the exit code so it proves the *absence* of a
    # preflight on a run that actually happened.
    assert result.exit_code == 0, "CLI exited %s: %s" % (result.exit_code, result.output)
    assert not captured["preflight_called"]
    assert not captured["finalize_called"]


def test_arena_benchmark_run_uses_same_arena_flow_helpers():
    """`sparkrun arena benchmark run <r>` delegates to preflight_arena + finalize_arena."""
    from sparkrun.cli._arena import arena_benchmark

    order = []

    def _capture(ctx, *args, **kwargs):
        order.append("run_benchmark")
        return _fake_bench_result()

    def _fake_preflight(*, local_test, ctx, recipe_name=None, dry_run=False):
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
        runner.invoke(arena_benchmark, ["run", "my-recipe", "--local-test", "--hosts", "h1"], catch_exceptions=False)

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
            ["my-recipe", "--local-test", "--dry-run", "--tp", "2", "--hosts", "h1"],
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
        runner.invoke(arena_benchmark, ["run", "my-recipe", "--local-test", "--hosts", "h1"], catch_exceptions=False)

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


# --------------------------------------------------------------------------
# Pre-submission validation gate
# --------------------------------------------------------------------------
#
# The arena paths publish the recipe, so they show the *full* `recipe validate`
# report — suggestions included — and ask before proceeding. Ordinary
# `sparkrun benchmark` publishes nothing and keeps the launch-path contract
# (`validate_for_launch`: suggestions withheld, deprecations collapsed).

_CLEAN_RECIPE = {
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "vllm-distributed",
    "container": "vllm/vllm-openai:latest",
    "defaults": {"port": 8000},
    "command": "vllm serve {model} --port {port}",
}

# `name:` -> deprecated-recipe-name (warning);
# literal model id -> restated-model-arg (suggestion).
_MESSY_RECIPE = {
    **_CLEAN_RECIPE,
    "name": "Arena Demo",
    "command": "vllm serve Qwen/Qwen3-1.7B --port {port}",
}

_ERROR_RECIPE = {"runtime": "vllm-distributed", "container": "x:latest"}  # no model:


def _write(tmp_path: Path, data: dict) -> str:
    path = tmp_path / "arena-demo.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _gate(recipe_path: str, *, tty: bool, answer: str | None = None, dry_run: bool = False):
    """Run the gate under a Click context, returning (exit_code, output)."""
    from sparkrun.cli._arena_flow import validate_recipe_for_submission

    runner = CliRunner()

    @click.command()
    def _cmd():
        validate_recipe_for_submission(recipe_path, ctx=click.get_current_context(), dry_run=dry_run)
        click.echo("PROCEEDED")

    with patch("sparkrun.cli._arena_flow._is_interactive", lambda: tty):
        result = runner.invoke(_cmd, [], input=answer)
    return result


def test_clean_recipe_says_nothing_and_does_not_prompt(tmp_path):
    """A report with no findings is noise between the user and their benchmark."""
    result = _gate(_write(tmp_path, _CLEAN_RECIPE), tty=True)
    assert result.exit_code == 0
    assert "PROCEEDED" in result.output
    assert "We suggest" not in result.output


def test_findings_are_shown_in_full_including_suggestions(tmp_path):
    """The whole point of the gate: `validate_for_launch` withholds suggestions,
    and a recipe about to be published is exactly when its author needs them."""
    result = _gate(_write(tmp_path, _MESSY_RECIPE), tty=True, answer="y\n")
    assert "warning  deprecated-recipe-name" in result.output
    assert "suggestion  restated-model-arg" in result.output
    assert "The recipe contains 1 warning and 1 suggestion." in result.output
    assert "may not be published to Spark Arena" in result.output
    assert "PROCEEDED" in result.output


def test_declining_the_prompt_aborts(tmp_path):
    result = _gate(_write(tmp_path, _MESSY_RECIPE), tty=True, answer="n\n")
    assert result.exit_code == 1
    assert "PROCEEDED" not in result.output


def test_prompt_defaults_to_no(tmp_path):
    """Bare Enter must not submit a recipe the user was just warned about."""
    result = _gate(_write(tmp_path, _MESSY_RECIPE), tty=True, answer="\n")
    assert result.exit_code == 1
    assert "PROCEEDED" not in result.output


def test_non_interactive_proceeds_with_a_notice(tmp_path):
    """Quality advice, not a security gate — the hook trust prompt is the one
    that refuses without a TTY. Aborting here would break a scripted
    submission that worked yesterday."""
    result = _gate(_write(tmp_path, _MESSY_RECIPE), tty=False)
    assert result.exit_code == 0
    assert "Not running interactively" in result.output
    assert "PROCEEDED" in result.output


def test_errors_abort_without_prompting(tmp_path):
    """The launch would refuse anyway; "continue?" is not a real question when
    the answer cannot be yes."""
    result = _gate(_write(tmp_path, _ERROR_RECIPE), tty=True)
    assert result.exit_code == 1
    assert "cannot be submitted" in result.output
    assert "Continue with the benchmark" not in result.output


def test_dry_run_reports_but_does_not_prompt(tmp_path):
    result = _gate(_write(tmp_path, _MESSY_RECIPE), tty=True, dry_run=True)
    assert result.exit_code == 0
    assert "The recipe contains" in result.output
    assert "[dry-run] Not prompting" in result.output
    assert "PROCEEDED" in result.output


@pytest.mark.parametrize(
    "counts,expected",
    [
        ({"warning": 3, "suggestion": 4}, "3 warnings and 4 suggestions"),
        ({"warning": 1, "suggestion": 0}, "1 warning"),
        ({"warning": 0, "suggestion": 1}, "1 suggestion"),
        ({"warning": 0, "suggestion": 2}, "2 suggestions"),
    ],
)
def test_count_phrase_omits_absent_levels(counts, expected):
    from sparkrun.cli._arena_flow import _count_phrase

    assert _count_phrase({"error": 0, **counts}) == expected


def test_local_test_still_validates(tmp_path):
    """`--local-test` rehearses a submission; rehearsing without the checks
    would defeat the rehearsal."""
    from sparkrun.cli._arena_flow import preflight_arena

    seen = {}

    def _fake(recipe_name, *, ctx, dry_run=False):
        seen["recipe"] = recipe_name

    runner = CliRunner()

    @click.command()
    def _cmd():
        preflight_arena(local_test=True, ctx=click.get_current_context(), recipe_name="r", dry_run=False)

    with patch("sparkrun.cli._arena_flow.validate_recipe_for_submission", _fake):
        runner.invoke(_cmd, [])
    assert seen["recipe"] == "r"
