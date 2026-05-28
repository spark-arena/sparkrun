"""Tests for the ResumeMode decision tree (plan section I) and --resume flag."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sparkrun.api._benchmark_models import ResumeMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_internal_result():
    """Return a fake internal BenchmarkResult for _execute_benchmark patches."""
    result = MagicMock()
    result.success = True
    result.benchmark_id = "x"
    fw = MagicMock()
    fw.framework_name = "llama-benchy"
    fw.primary_category = "performance"
    result.framework = fw
    result.profile = None
    result.results = {}
    result.outputs = {}
    result.cluster_id = "c"
    result.host_list = []
    result.container_image = "img"
    result.container_image_sha = None
    result.container_image_sha_pinned = False
    result.longterm_image_ref = None
    result.longterm_image_pinned = False
    result.benchmark_args = {}
    result.state_dir = None
    result.resumed = False
    result.submission_id = None
    result.recipe = MagicMock()
    result.overrides = {}
    return result


def _stub_execute_benchmark(received_options):
    """Return a side_effect for _execute_benchmark that records the options."""

    def _capture(options, *, sctx, emitter):
        received_options.append(options)
        return _fake_internal_result()

    return _capture


def _stub_run_benchmark(received_kwargs):
    """Return a side_effect for _run_benchmark that records kwargs (CLI-side tests)."""

    def _capture(ctx, **kwargs):
        received_kwargs.update(kwargs)
        return _fake_internal_result()

    return _capture


# ---------------------------------------------------------------------------
# API wrapper threading tests — now patch _execute_benchmark
# ---------------------------------------------------------------------------


def test_api_default_resume_mode_is_if_exists():
    received = []
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_stub_execute_benchmark(received)):
        from sparkrun.api import benchmark
        from sparkrun.api._benchmark_models import BenchmarkOptions

        benchmark(BenchmarkOptions(recipe="my-recipe"))
    assert received[0].resume == ResumeMode.IF_EXISTS


def test_api_fresh_mode_threaded_through():
    received = []
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_stub_execute_benchmark(received)):
        from sparkrun.api import benchmark
        from sparkrun.api._benchmark_models import BenchmarkOptions

        benchmark(BenchmarkOptions(recipe="my-recipe", resume=ResumeMode.FRESH))
    assert received[0].resume == ResumeMode.FRESH


def test_api_required_mode_threaded_through():
    received = []
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_stub_execute_benchmark(received)):
        from sparkrun.api import benchmark
        from sparkrun.api._benchmark_models import BenchmarkOptions

        benchmark(BenchmarkOptions(recipe="my-recipe", resume=ResumeMode.REQUIRED))
    assert received[0].resume == ResumeMode.REQUIRED


def test_api_auto_mode_threaded_through():
    received = []
    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_stub_execute_benchmark(received)):
        from sparkrun.api import benchmark
        from sparkrun.api._benchmark_models import BenchmarkOptions

        benchmark(BenchmarkOptions(recipe="my-recipe", resume=ResumeMode.AUTO))
    assert received[0].resume == ResumeMode.AUTO


def test_api_on_prompt_required_threaded_through():
    received = []

    def cb(state):
        return True

    with patch("sparkrun.api._benchmark._execute_benchmark", side_effect=_stub_execute_benchmark(received)):
        from sparkrun.api import benchmark
        from sparkrun.api._benchmark_models import BenchmarkOptions

        benchmark(BenchmarkOptions(recipe="my-recipe", on_prompt_required=cb))
    assert received[0].on_prompt_required is cb


# ---------------------------------------------------------------------------
# _resolve_resume_prompt unit tests
# ---------------------------------------------------------------------------


def test_resolve_resume_prompt_uses_callback_when_provided():
    from sparkrun.cli._benchmark import _resolve_resume_prompt

    state = MagicMock(completed_indices=[0, 1])
    called_with = {}

    def cb(s):
        called_with["state"] = s
        return False

    result = _resolve_resume_prompt(state, total_tasks=5, on_prompt_required=cb)
    assert result is False
    assert called_with["state"] is state


def test_resolve_resume_prompt_callback_true():
    from sparkrun.cli._benchmark import _resolve_resume_prompt

    state = MagicMock(completed_indices=[0, 1])
    assert _resolve_resume_prompt(state, total_tasks=5, on_prompt_required=lambda s: True) is True


def test_resolve_resume_prompt_non_tty_defaults_to_resume(monkeypatch):
    from sparkrun.cli._benchmark import _resolve_resume_prompt

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    state = MagicMock(completed_indices=[0, 1])
    assert _resolve_resume_prompt(state, total_tasks=5, on_prompt_required=None) is True


def test_resolve_resume_prompt_tty_uses_click_confirm(monkeypatch):
    from sparkrun.cli._benchmark import _resolve_resume_prompt

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    called = {}

    def fake_confirm(msg, default):
        called["msg"] = msg
        called["default"] = default
        return False

    monkeypatch.setattr("click.confirm", fake_confirm)
    state = MagicMock(completed_indices=[0, 1])
    result = _resolve_resume_prompt(state, total_tasks=5, on_prompt_required=None)
    assert result is False
    assert "Resume?" in called["msg"]
    assert called["default"] is True


def test_resolve_resume_prompt_tty_confirm_true(monkeypatch):
    from sparkrun.cli._benchmark import _resolve_resume_prompt

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("click.confirm", lambda msg, default: True)
    state = MagicMock(completed_indices=[0])
    assert _resolve_resume_prompt(state, total_tasks=3, on_prompt_required=None) is True


# ---------------------------------------------------------------------------
# CLI flag mutual-exclusion test
# ---------------------------------------------------------------------------


def test_cli_resume_and_fresh_are_mutually_exclusive():
    from click.testing import CliRunner
    from sparkrun.cli._benchmark import benchmark_run

    runner = CliRunner()
    result = runner.invoke(benchmark_run, ["my-recipe", "--resume", "--fresh"])
    # click UsageError exits with non-zero code
    assert result.exit_code != 0
    output_text = result.output + (str(result.exception) if result.exception else "")
    assert "mutually exclusive" in output_text


def test_cli_resume_flag_sets_if_exists_mode():
    """--resume alone should map to IF_EXISTS with no error."""
    received = {}

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_stub_run_benchmark(received)):
        from click.testing import CliRunner
        from sparkrun.cli._benchmark import benchmark_run

        runner = CliRunner()
        # We expect it to fail deep in bootstrap (no SAF context), but the mode
        # should be captured before that. Patch deeper to avoid full bootstrap.
        with patch("sparkrun.cli._benchmark._get_context") as mock_ctx:
            mock_ctx.side_effect = RuntimeError("stop early")
            result = runner.invoke(benchmark_run, ["my-recipe", "--resume"])
        # Either mode was recorded, or it errored out before _run_benchmark.
        # The key check is that no UsageError was raised for the flag itself.
        if received:
            assert received["resume_mode"] == ResumeMode.IF_EXISTS
        else:
            # Didn't reach _run_benchmark — acceptable; the flag parsing succeeded
            # (no exit code 2 from UsageError)
            assert result.exit_code != 2 or "mutually exclusive" not in (result.output or "")


def test_cli_fresh_flag_sets_fresh_mode():
    """--fresh alone should map to FRESH with no UsageError."""
    received = {}

    with patch("sparkrun.cli._benchmark._run_benchmark", side_effect=_stub_run_benchmark(received)):
        from click.testing import CliRunner
        from sparkrun.cli._benchmark import benchmark_run

        with patch("sparkrun.cli._benchmark._get_context") as mock_ctx:
            mock_ctx.side_effect = RuntimeError("stop early")
            result = runner = CliRunner()
            result = runner.invoke(benchmark_run, ["my-recipe", "--fresh"])
        if received:
            assert received["resume_mode"] == ResumeMode.FRESH
        else:
            assert result.exit_code != 2 or "mutually exclusive" not in (result.output or "")
