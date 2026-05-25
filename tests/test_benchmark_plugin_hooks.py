"""Tests for plugin-level hooks on BenchmarkingPlugin.

Covers:
- ``apply_session_warmup_state`` default no-op behavior
- llama-benchy override (no_warmup + skip_coherence on non-first task)
- tool-eval-bench override (same)
- ``framework_pinned_version`` sentinel consumed by
  ``LlamaBenchyFramework.build_benchmark_command``
"""

from __future__ import annotations

from typing import Any

from sparkrun.benchmarking.base import BenchmarkingPlugin
from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
from sparkrun.benchmarking.tool_eval_bench import ToolEvalBenchFramework


class _BareFW(BenchmarkingPlugin):
    """Minimal BenchmarkingPlugin used to exercise default hook behavior."""

    framework_name = "bare"

    def check_prerequisites(self) -> list[str]:
        return []

    def build_benchmark_command(
        self,
        target_url: str,
        model: str,
        args: dict[str, Any],
        result_file: str | None = None,
    ) -> list[str]:
        return ["/usr/bin/true"]

    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Default hook
# ---------------------------------------------------------------------------


def test_apply_session_warmup_state_default_is_noop_first_task():
    """Default implementation must not mutate args on the first task."""
    fw = _BareFW()
    original = {"some_key": "some_value", "another": 1}
    out = fw.apply_session_warmup_state(original, is_first_task=True)

    assert out == original
    assert out is not original, "Default should return a copy, not the input dict"


def test_apply_session_warmup_state_default_is_noop_subsequent_task():
    """Default implementation must not mutate args on non-first tasks either."""
    fw = _BareFW()
    original = {"some_key": "some_value"}
    out = fw.apply_session_warmup_state(original, is_first_task=False)

    assert out == original
    assert out is not original


def test_apply_session_warmup_state_default_does_not_add_warmup_flags():
    """Default implementation must NOT inject framework-specific keys."""
    fw = _BareFW()
    out = fw.apply_session_warmup_state({}, is_first_task=False)
    assert "no_warmup" not in out
    assert "skip_coherence" not in out


# ---------------------------------------------------------------------------
# llama-benchy override
# ---------------------------------------------------------------------------


def test_llama_benchy_apply_session_warmup_state_first_task_noop():
    """First task: warmup/coherence flags must NOT be injected."""
    fw = LlamaBenchyFramework()
    out = fw.apply_session_warmup_state({"depth": [0]}, is_first_task=True)
    assert out.get("no_warmup") is not True
    assert out.get("skip_coherence") is not True


def test_llama_benchy_apply_session_warmup_state_subsequent_task_sets_flags():
    """Non-first task: no_warmup and skip_coherence default to True."""
    fw = LlamaBenchyFramework()
    out = fw.apply_session_warmup_state({"depth": [0]}, is_first_task=False)
    assert out.get("no_warmup") is True
    assert out.get("skip_coherence") is True


def test_llama_benchy_apply_session_warmup_state_respects_user_override():
    """If args already set no_warmup explicitly, do not stomp on it."""
    fw = LlamaBenchyFramework()
    out = fw.apply_session_warmup_state({"no_warmup": False}, is_first_task=False)
    # setdefault honors existing key
    assert out.get("no_warmup") is False


def test_llama_benchy_apply_session_warmup_state_returns_copy():
    """Function must not mutate the input dict."""
    fw = LlamaBenchyFramework()
    original = {"depth": [0]}
    out = fw.apply_session_warmup_state(original, is_first_task=False)
    assert "no_warmup" not in original
    assert out is not original


# ---------------------------------------------------------------------------
# tool-eval-bench override
# ---------------------------------------------------------------------------


def test_tool_eval_bench_apply_session_warmup_state_first_task_noop():
    """First task: warmup/coherence flags must NOT be injected."""
    fw = ToolEvalBenchFramework()
    out = fw.apply_session_warmup_state({"backend": "vllm"}, is_first_task=True)
    assert out.get("no_warmup") is not True
    assert out.get("skip_coherence") is not True


def test_tool_eval_bench_apply_session_warmup_state_subsequent_task_sets_flags():
    """Non-first task: no_warmup and skip_coherence default to True."""
    fw = ToolEvalBenchFramework()
    out = fw.apply_session_warmup_state({"backend": "vllm"}, is_first_task=False)
    assert out.get("no_warmup") is True
    assert out.get("skip_coherence") is True


def test_tool_eval_bench_apply_session_warmup_state_returns_copy():
    """Function must not mutate the input dict."""
    fw = ToolEvalBenchFramework()
    original = {"backend": "vllm"}
    out = fw.apply_session_warmup_state(original, is_first_task=False)
    assert "no_warmup" not in original
    assert out is not original


# ---------------------------------------------------------------------------
# framework_pinned_version sentinel — consumed by build_benchmark_command
# ---------------------------------------------------------------------------


def test_llama_benchy_pinned_version_threads_into_package_spec():
    """When ``framework_pinned_version`` is in args, the uvx package spec
    becomes ``llama-benchy@<version>``.
    """
    fw = LlamaBenchyFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args={"framework_pinned_version": "0.5.7"},
    )
    # Ensure the pinned spec appears in argv (second token after "uvx")
    assert "llama-benchy@0.5.7" in cmd


def test_llama_benchy_no_pinned_version_uses_floating_package_spec():
    """Without the sentinel, the package spec is plain ``llama-benchy``."""
    fw = LlamaBenchyFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args={},
    )
    assert "llama-benchy" in cmd
    # No version-pinned variant should appear
    assert not any(token.startswith("llama-benchy@") for token in cmd)


def test_llama_benchy_pinned_version_is_not_forwarded_as_flag():
    """The sentinel key must be stripped — never emitted as ``--framework-pinned-version <v>``."""
    fw = LlamaBenchyFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args={"framework_pinned_version": "0.5.7"},
    )
    assert "--framework-pinned-version" not in cmd
    # Old sentinel name must also not leak in
    assert "--pinned-version" not in cmd
    assert "--_pinned-version" not in cmd
