"""Tests verifying the benchmark CLI uses ``sparkrun.api.run`` /
``sparkrun.api.stop`` (rather than ``launch_inference`` / ``runtime.stop``
directly) and that the ``--scheduler`` flag is plumbed through to
``RunOptions``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sparkrun.cli import main as cli_main
from sparkrun.cli._benchmark import _stop_inference


# ---------------------------------------------------------------------------
# benchmark module no longer imports launch_inference
# ---------------------------------------------------------------------------


def test_benchmark_module_does_not_import_launch_inference():
    """Sanity: ``sparkrun.cli._benchmark`` must no longer pull in
    ``launch_inference`` at import-or-runtime — its launch path is now
    ``api.run``.
    """
    source = Path("src/sparkrun/cli/_benchmark.py").read_text()
    assert (
        "from sparkrun.core.launcher import" not in source
        or "launch_inference" not in source.split("from sparkrun.core.launcher import")[-1].split("\n")[0]
    )


# ---------------------------------------------------------------------------
# _stop_inference dispatches via api.stop, not runtime.stop
# ---------------------------------------------------------------------------


def test_stop_inference_calls_api_stop_with_cluster_id_and_hosts():
    """Verify ``_stop_inference`` delegates to ``api.stop`` with the
    expected keyword arguments and does NOT touch ``runtime.stop``.
    """
    runtime = MagicMock()
    runtime.stop = MagicMock()

    fake_sctx = object()

    with patch("sparkrun.api.stop") as mock_stop:
        _stop_inference(
            runtime=runtime,
            host_list=["h1", "h2"],
            cluster_id="sparkrun_0123456789abcdef_aabbccddeeff",
            config=None,
            dry_run=False,
            sctx=fake_sctx,
        )

    mock_stop.assert_called_once()
    _, kwargs = mock_stop.call_args
    assert kwargs["cluster_id"] == "sparkrun_0123456789abcdef_aabbccddeeff"
    assert kwargs["hosts"] == ("h1", "h2")
    assert kwargs["sctx"] is fake_sctx

    # Direct runtime.stop must not be called from the benchmark stop path.
    runtime.stop.assert_not_called()


def test_stop_inference_dry_run_skips_api_stop():
    """Dry-run mode prints the would-stop line and skips ``api.stop``."""
    runtime = MagicMock()
    with patch("sparkrun.api.stop") as mock_stop:
        _stop_inference(
            runtime=runtime,
            host_list=["h1"],
            cluster_id="sparkrun_0123456789abcdef_aabbccddeeff",
            config=None,
            dry_run=True,
            sctx=None,
        )
    mock_stop.assert_not_called()
    runtime.stop.assert_not_called()


# ---------------------------------------------------------------------------
# benchmark run wires --scheduler into RunOptions and invokes api.run
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_recipe_env(tmp_path: Path, monkeypatch):
    """Patch ``_load_recipe`` in the benchmark CLI module so the CLI can run
    without a configured recipe-search path.  Returns the in-memory Recipe.
    """
    from sparkrun.core.recipe import Recipe

    recipe = Recipe(
        {
            "name": "test-recipe",
            "sparkrun_version": "2",
            "runtime": "vllm-distributed",
            "model": "test/model",
            "container": "test/container:latest",
            "command": "serve",
            "defaults": {"port": 8000},
        }
    )

    monkeypatch.setattr(
        "sparkrun.cli._benchmark._load_recipe",
        lambda config, recipe_name, resolve=False: (recipe, Path("/dev/null"), None),
    )
    monkeypatch.chdir(tmp_path)
    return recipe


def _make_fake_run_result(cluster_id: str = "sparkrun_0123456789abcdef_aabbccddeeff"):
    """Build a fake RunResult with the minimal fields _benchmark.py reads."""
    rr = MagicMock()
    rr.cluster_id = cluster_id
    rr.intent_id = "0123456789abcdef"
    rr.placement_token = "aabbccddeeff"
    rr.host_list = ("h1",)
    rr.placement = None
    rr.scheduler = "greedy"
    rr.runtime = "vllm-distributed"
    rr.started_at = 0.0
    rr.dry_run = True
    rr.serve_command = ""
    rr.serve_port = 8000
    rr.metadata = {}
    rr.launch_result = None
    return rr


def test_benchmark_run_uses_api_run_and_threads_scheduler_flag(fake_recipe_env, monkeypatch):
    """``sparkrun benchmark run --scheduler greedy <recipe> --solo --dry-run``
    must call ``api.run`` exactly once with ``RunOptions.scheduler == 'greedy'``.
    """
    captured_options: list[Any] = []

    def _fake_run(options, *, sctx=None):
        captured_options.append(options)
        return _make_fake_run_result()

    # Patch the api.run / api.stop entry points the CLI now uses.
    # Also patch wait_for_port / wait_for_healthy so the benchmark flow
    # doesn't try to SSH anywhere when not dry-run.  We *do* use dry_run
    # so the bench command body is skipped.
    with (
        patch("sparkrun.api.run", side_effect=_fake_run) as mock_run,
        patch("sparkrun.api.stop") as mock_stop,
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "benchmark",
                "run",
                "--solo",
                "--dry-run",
                "--scheduler",
                "greedy",
                "--hosts",
                "h1",
                "test-recipe",
            ],
            catch_exceptions=False,
        )

    # Diagnostic on failure: surface CLI stdout to aid debugging.
    assert result.exit_code == 0, "CLI exited non-zero: %s\n%s" % (result.exit_code, result.output)
    assert mock_run.call_count == 1, "api.run should be invoked exactly once for a non-skip-run benchmark"

    opts = captured_options[0]
    assert opts.scheduler == "greedy"
    assert opts.solo is True
    assert opts.dry_run is True
    # The CLI translates host strings to a tuple
    assert opts.hosts == ("h1",)

    # api.stop is also called (post-benchmark cleanup) — dry-run skips it
    # at the _stop_inference shim.
    mock_stop.assert_not_called()


def test_benchmark_run_scheduler_flag_defaults_to_none(fake_recipe_env):
    """Without --scheduler, ``RunOptions.scheduler`` must be ``None`` (let
    the registry default kick in).
    """
    captured_options: list[Any] = []

    def _fake_run(options, *, sctx=None):
        captured_options.append(options)
        return _make_fake_run_result()

    with patch("sparkrun.api.run", side_effect=_fake_run), patch("sparkrun.api.stop"):
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "benchmark",
                "run",
                "--solo",
                "--dry-run",
                "--hosts",
                "h1",
                "test-recipe",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert captured_options[0].scheduler is None


def test_benchmark_run_skip_run_does_not_call_api_run(fake_recipe_env):
    """``--skip-run`` short-circuits the launch — ``api.run`` must not be called."""
    with patch("sparkrun.api.run") as mock_run, patch("sparkrun.api.stop"):
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "benchmark",
                "run",
                "--solo",
                "--skip-run",
                "--dry-run",
                "--hosts",
                "h1",
                "test-recipe",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    mock_run.assert_not_called()
