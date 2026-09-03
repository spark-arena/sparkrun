"""Regression tests for spark-arena/sparkrun#257.

Two independent defects on the benchmark CLI paths:

1. ``sparkrun arena benchmark run --hosts a,b`` iterated the comma-separated
   string *character by character*, so ``spark1,spark2`` reached the launch as
   twelve single-letter hostnames.  ``sparkrun benchmark run`` was unaffected —
   it carries ``@with_host_context``, which the arena command did not.
2. ``BenchmarkFailed`` was caught and exited without rendering its message, so
   every benchmark failure — including the full "inference launch failed:
   <reason>" chain — produced rc=1 with no output on stdout, stderr or the log.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from sparkrun.api._errors import BenchmarkFailed
from sparkrun.core.hosts import parse_host_list


# ---------------------------------------------------------------------------
# parse_host_list — the shared comma splitter
# ---------------------------------------------------------------------------


def test_parse_host_list_splits_comma_string():
    assert parse_host_list("spark1,spark2") == ["spark1", "spark2"]


def test_parse_host_list_strips_and_drops_blanks():
    assert parse_host_list(" a , , b ") == ["a", "b"]


def test_parse_host_list_passes_through_iterable():
    assert parse_host_list(["spark1", "spark2"]) == ["spark1", "spark2"]


def test_parse_host_list_splits_commas_inside_iterable_entries():
    assert parse_host_list(["spark1,spark2", "spark3"]) == ["spark1", "spark2", "spark3"]


def test_parse_host_list_empty_inputs():
    assert parse_host_list(None) == []
    assert parse_host_list("") == []
    assert parse_host_list([]) == []


# ---------------------------------------------------------------------------
# --hosts reaches BenchmarkOptions unsplit on both benchmark entry points
# ---------------------------------------------------------------------------


def _capture_hosts(argv, *, arena: bool):
    """Invoke a benchmark command and return the ``BenchmarkOptions.hosts``
    that would have reached the orchestration."""
    from sparkrun.cli import main

    captured: dict = {}

    def _fake_execute(opts, sctx=None, emitter=None):
        captured["hosts"] = opts.hosts
        raise SystemExit(99)

    patches = [patch("sparkrun.api._benchmark._execute_benchmark", _fake_execute)]
    if arena:
        # Skip the arena auth/preflight round-trip; irrelevant to host parsing.
        patches.append(
            patch(
                "sparkrun.cli._arena_flow.preflight_arena",
                lambda local_test=False, ctx=None, recipe_name=None, dry_run=False: ("sub-1", None),
            )
        )

    with patches[0]:
        if arena:
            with patches[1]:
                CliRunner().invoke(main, argv)
        else:
            CliRunner().invoke(main, argv)
    return captured.get("hosts")


def test_benchmark_run_hosts_are_not_char_split():
    hosts = _capture_hosts(
        ["benchmark", "run", "dummy-recipe", "--hosts", "spark1,spark2", "--dry-run"],
        arena=False,
    )
    assert hosts == ("spark1", "spark2")


def test_arena_benchmark_run_hosts_are_not_char_split():
    """The reported bug: this path lacked ``@with_host_context``."""
    hosts = _capture_hosts(
        ["arena", "benchmark", "run", "dummy-recipe", "--hosts", "spark1,spark2", "--dry-run"],
        arena=True,
    )
    assert hosts == ("spark1", "spark2")
    # The precise failure shape from the issue, asserted directly so a
    # regression can't pass by merely being "some tuple".
    assert "s" not in hosts


# ---------------------------------------------------------------------------
# BenchmarkFailed messages are rendered, not swallowed
# ---------------------------------------------------------------------------

_LAUNCH_ERROR = (
    "Error: inference launch failed: launch_inference failed: pre_exec require "
    "confirmation but stdin is not a TTY. Use --trust to allow pre_exec from "
    "third-party registries."
)


def _invoke_with_failure(exc):
    from sparkrun.cli import main

    def _fake_execute(opts, sctx=None, emitter=None):
        raise exc

    with patch("sparkrun.api._benchmark._execute_benchmark", _fake_execute):
        return CliRunner().invoke(main, ["benchmark", "run", "dummy", "--hosts", "h1", "--dry-run"])


def test_benchmark_failure_message_is_printed():
    result = _invoke_with_failure(BenchmarkFailed(_LAUNCH_ERROR, exit_code=1))
    assert result.exit_code == 1
    assert "inference launch failed" in result.output
    assert "not a TTY" in result.output
    # The embedded "Error: " prefix must not be doubled.
    assert "Error: Error:" not in result.output


def test_benchmark_failure_without_exit_code_still_prints_and_exits_1():
    result = _invoke_with_failure(BenchmarkFailed("something broke"))
    assert result.exit_code == 1
    assert "something broke" in result.output


def test_benchmark_failure_exit_code_zero_is_not_an_error():
    """``exit_code=0`` is the "already complete" case: stdout, rc 0."""
    result = _invoke_with_failure(BenchmarkFailed("Benchmark abc is already complete.", exit_code=0))
    assert result.exit_code == 0
    assert "already complete" in result.output
    assert "Error:" not in result.output


# ---------------------------------------------------------------------------
# --trust reaches BenchmarkOptions on both benchmark entry points
#
# Without this flag a recipe with hooks from an untrusted registry could not be
# benchmarked non-interactively at all: the trust gate refuses on a non-TTY and
# there is no prompt to answer.  ``sparkrun run`` has always had --trust.
# ---------------------------------------------------------------------------


def _capture_trust(argv, *, arena: bool):
    from sparkrun.cli import main

    captured: dict = {}

    def _fake_execute(opts, sctx=None, emitter=None):
        captured["trust"] = opts.trust
        raise SystemExit(99)

    with patch("sparkrun.api._benchmark._execute_benchmark", _fake_execute):
        if arena:
            with patch(
                "sparkrun.cli._arena_flow.preflight_arena",
                lambda local_test=False, ctx=None, recipe_name=None, dry_run=False: ("sub-1", None),
            ):
                CliRunner().invoke(main, argv)
        else:
            CliRunner().invoke(main, argv)
    return captured.get("trust")


def test_benchmark_run_defaults_to_untrusted():
    assert _capture_trust(["benchmark", "run", "r", "--hosts", "h1", "--dry-run"], arena=False) is False


def test_benchmark_run_accepts_trust():
    assert _capture_trust(["benchmark", "run", "r", "--hosts", "h1", "--dry-run", "--trust"], arena=False) is True


def test_arena_benchmark_run_accepts_trust():
    argv = ["arena", "benchmark", "run", "r", "--hosts", "h1", "--dry-run", "--trust"]
    assert _capture_trust(argv, arena=True) is True


def test_benchmark_options_trust_defaults_to_false():
    from sparkrun.api._benchmark_models import BenchmarkOptions

    assert BenchmarkOptions(recipe="r").trust is False
    assert BenchmarkOptions(recipe="r", trust=True).trust is True


def test_execute_benchmark_forwards_trust_to_run_options():
    """The ``BenchmarkOptions.trust`` -> ``RunOptions.trust`` hop.

    ``RunOptions.trust`` is the field ``resolve_recipe_trust`` consults, and it
    was previously hardcoded to ``None`` — so no flag on any benchmark command
    could ever have reached it.  Asserted against the parsed AST rather than a
    live call: every test harness for this module mocks ``_execute_benchmark``
    out wholesale, and standing up enough cluster/recipe state to reach the
    ``RunOptions`` construction would test the fixtures, not the wiring.
    """
    import ast
    import inspect

    import sparkrun.api._benchmark as mod

    tree = ast.parse(inspect.getsource(mod))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "RunOptions"
    ]
    assert calls, "no api.RunOptions(...) construction found in api._benchmark"
    for call in calls:
        trust_kw = next((kw for kw in call.keywords if kw.arg == "trust"), None)
        assert trust_kw is not None, "RunOptions built without an explicit trust= argument"
        assert isinstance(trust_kw.value, ast.Attribute) and trust_kw.value.attr == "trust", (
            "RunOptions.trust must come from the caller's options, not a literal"
        )
