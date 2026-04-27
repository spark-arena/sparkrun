"""Tests for the tool-eval-bench benchmarking framework plugin."""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

from sparkrun.benchmarking.tool_eval_bench import (
    ToolEvalBenchFramework,
    scenario_results_to_csv,
)


def test_framework_name():
    fw = ToolEvalBenchFramework()
    assert fw.framework_name == "tool-eval-bench"


def test_default_args():
    fw = ToolEvalBenchFramework()
    defaults = fw.get_default_args()
    assert defaults["backend"] == "vllm"
    assert defaults["parallel"] == 1
    assert defaults["timeout"] == 60
    assert defaults["max_turns"] == 8
    assert defaults["temperature"] == 0.0


def test_build_command_basic():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args={"backend": "vllm", "parallel": 2},
    )
    assert cmd[0] == "uvx"
    assert "--from" in cmd
    assert "tool-eval-bench" in cmd
    assert "--base-url" in cmd
    assert "http://localhost:8000/v1" in cmd
    assert "--model" in cmd
    assert "org/model" in cmd
    assert "--json" in cmd
    assert "--backend" in cmd
    assert "vllm" in cmd
    assert "--parallel" in cmd
    assert "2" in cmd


def test_build_command_default_ref_pinned():
    """The git ref pin appears in the --from spec."""
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert "git+https://github.com/SeraphimSerapis/tool-eval-bench@" in spec


def test_build_command_ref_override():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"ref": "main"},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert spec.endswith("@main")
    # `ref` is consumed by the plugin and must not be forwarded as a flag.
    assert "--ref" not in cmd


def test_build_command_perf_extra_activated():
    """Setting perf=True switches the --from spec to the [perf] extra."""
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"perf": True},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert "tool-eval-bench[perf]" in spec
    assert "--perf" in cmd


def test_build_command_perf_extra_not_activated_by_default():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert "[perf]" not in spec


def test_build_command_bool_flag():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"no_think": True, "short": True, "hardmode": False},
    )
    assert "--no-think" in cmd
    assert "--short" in cmd
    assert "--hardmode" not in cmd


def test_build_command_list_args():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"categories": ["A", "B", "K"], "scenarios": ["TC-01", "TC-07"]},
    )
    cat_idx = cmd.index("--categories")
    assert cmd[cat_idx + 1] == "A"
    assert cmd[cat_idx + 2] == "B"
    assert cmd[cat_idx + 3] == "K"

    sc_idx = cmd.index("--scenarios")
    assert cmd[sc_idx + 1] == "TC-01"
    assert cmd[sc_idx + 2] == "TC-07"


def test_build_command_kebab_case_keys():
    """snake_case arg keys render as --kebab-case flags."""
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"max_turns": 16, "reference_date": "2026-04-25"},
    )
    assert "--max-turns" in cmd
    assert "16" in cmd
    assert "--reference-date" in cmd
    assert "2026-04-25" in cmd


def test_interpret_arg_list():
    fw = ToolEvalBenchFramework()
    assert fw.interpret_arg("categories", "A,B,K") == ["A", "B", "K"]
    # Scenarios aren't all numeric; coerce_value falls back to str when needed.
    assert fw.interpret_arg("scenarios", "TC-01,TC-07") == ["TC-01", "TC-07"]


def test_interpret_arg_bool():
    fw = ToolEvalBenchFramework()
    assert fw.interpret_arg("no_think", "true") is True
    assert fw.interpret_arg("short", "false") is False


def test_interpret_arg_scalar():
    fw = ToolEvalBenchFramework()
    assert fw.interpret_arg("parallel", "4") == 4
    assert fw.interpret_arg("temperature", "0.7") == 0.7
    assert fw.interpret_arg("backend", "sglang") == "sglang"


def test_estimate_test_count_default():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({}) == 69


def test_estimate_test_count_short():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({"short": True}) == 15


def test_estimate_test_count_hardmode():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({"hardmode": True}) == 74


def test_estimate_test_count_explicit_scenarios():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({"scenarios": ["TC-01", "TC-02", "TC-03"]}) == 3


def test_check_prerequisites_with_uvx():
    fw = ToolEvalBenchFramework()
    with patch.object(shutil, "which", return_value="/usr/bin/uvx"):
        assert fw.check_prerequisites() == []


def test_check_prerequisites_without_uvx():
    fw = ToolEvalBenchFramework()
    with patch.object(shutil, "which", return_value=None):
        missing = fw.check_prerequisites()
    assert len(missing) == 1
    assert "uvx" in missing[0].lower()


def test_parse_results_with_scenarios():
    fw = ToolEvalBenchFramework()
    payload = {
        "run_id": "2026-04-25T12-00-00Z_abc123",
        "status": "completed",
        "config": {"model": "org/m", "backend": "vllm"},
        "scores": {
            "final_score": 87,
            "total_points": 120,
            "max_points": 138,
            "rating": "★★★★",
            "category_scores": [
                {
                    "category": "A",
                    "label": "Basic Tool Selection",
                    "earned": 10,
                    "max": 10,
                    "percent": 100,
                    "pass_count": 5,
                    "partial_count": 0,
                    "fail_count": 0,
                },
            ],
            "scenario_results": [
                {
                    "scenario_id": "TC-01",
                    "status": "pass",
                    "points": 2,
                    "summary": "Selected get_weather correctly",
                    "duration_seconds": 1.23,
                    "ttft_ms": 95.1,
                    "turn_count": 1,
                    "prompt_tokens": 200,
                    "completion_tokens": 50,
                    "total_tokens": 250,
                },
                {
                    "scenario_id": "TC-02",
                    "status": "fail",
                    "points": 0,
                    "summary": "Hallucinated tool name",
                    "duration_seconds": 0.8,
                    "turn_count": 1,
                },
            ],
        },
        "metadata": {},
    }
    results = fw.parse_results(json.dumps(payload), "")

    assert results["json"]["scores"]["final_score"] == 87
    assert results["stdout"] == json.dumps(payload)

    csv_lines = results["csv"].strip().splitlines()
    assert len(csv_lines) == 3  # header + 2 scenarios
    assert csv_lines[0].startswith("scenario_id,status,points,")
    assert "TC-01" in csv_lines[1]
    assert "pass" in csv_lines[1]
    assert "TC-02" in csv_lines[2]
    assert "fail" in csv_lines[2]


def test_parse_results_empty_stdout():
    fw = ToolEvalBenchFramework()
    results = fw.parse_results("", "")
    assert results["json"] == {}
    assert results["csv"] == ""


def test_parse_results_invalid_json():
    fw = ToolEvalBenchFramework()
    results = fw.parse_results("not json at all", "")
    assert results["json"] == {}
    assert results["csv"] == ""
    assert results["stdout"] == "not json at all"


def test_scenario_results_to_csv_strips_newlines_in_summary():
    rows = [
        {
            "scenario_id": "TC-01",
            "status": "pass",
            "points": 2,
            "summary": "line one\nline two",
            "duration_seconds": 1.0,
            "turn_count": 1,
        },
    ]
    csv_text = scenario_results_to_csv(rows)
    # Newline-stripped summary should not introduce extra CSV rows.
    assert csv_text.count("\n") == 2  # one for header, one for row, no embedded newline
    assert "line one line two" in csv_text


def test_scenario_results_to_csv_missing_optional_fields():
    rows = [
        {
            "scenario_id": "TC-05",
            "status": "partial",
            "points": 1,
            "summary": "ok-ish",
            "duration_seconds": 0.5,
            "turn_count": 2,
            # ttft_ms and token counts intentionally omitted
        },
    ]
    csv_text = scenario_results_to_csv(rows)
    lines = csv_text.strip().splitlines()
    assert len(lines) == 2
    # ttft_ms column exists but value is empty
    header = lines[0].split(",")
    row = lines[1].split(",")
    ttft_idx = header.index("ttft_ms")
    assert row[ttft_idx] == ""
