"""tool-eval-bench benchmarking framework plugin for sparkrun.

Runs `SeraphimSerapis/tool-eval-bench`_ against an OpenAI-compatible
endpoint to measure tool-call correctness across a deterministic suite of
scenarios (69 by default, plus 5 optional Hard Mode scenarios).

.. _SeraphimSerapis/tool-eval-bench: https://github.com/SeraphimSerapis/tool-eval-bench
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
from logging import Logger
from typing import Any

from scitrera_app_framework import Variables

from sparkrun.benchmarking.base import BenchmarkingPlugin
from sparkrun.utils.shell import quote_list

logger = logging.getLogger(__name__)

# Pinned upstream ref. Bump as needed; users can override per-call with
# ``-b ref=<tag-or-branch>`` (the value is consumed by the plugin and not
# forwarded to the subprocess).
_DEFAULT_REF = "v1.4.3"
_GIT_URL = "https://github.com/SeraphimSerapis/tool-eval-bench"

# tool-eval-bench args that take multiple space-separated values.
_LIST_ARGS = {
    "scenarios",
    "categories",
    "pp",
    "tg",
    "depth",
    "concurrency",
    "spec_prompts",
}

# Boolean flags (present when truthy, omitted otherwise).
_BOOL_ARGS = {
    "no_think",
    "short",
    "hardmode",
    "no_warmup",
    "no_live",
    "skip_coherence",
    "redact_url",
    "no_probe_engine",
    "experimental_async",
    "llm_judge",
    "perf",
    "perf_only",
    "perf_legacy",
    "perf_legacy_only",
    "spec_bench",
    "spec_live",
}

# Args consumed by the plugin and NOT forwarded to the subprocess.
_HIDDEN_ARGS = {"ref"}

# Setting any of these to true requires the [perf] extra (pulls llama-benchy).
_PERF_TRIGGER_ARGS = {"perf", "perf_only", "perf_legacy", "perf_legacy_only"}


class ToolEvalBenchFramework(BenchmarkingPlugin):
    """tool-eval-bench framework — deterministic tool-call correctness.

    Invokes ``uvx tool-eval-bench --json`` against an OpenAI-compatible
    endpoint and parses the JSON object emitted on stdout. Synthesizes a
    per-scenario CSV for the standard sparkrun result triple.
    """

    framework_name = "tool-eval-bench"
    default_args: dict[str, Any] = {
        "backend": "vllm",
        "parallel": 1,
        "timeout": 60,
        "max_turns": 8,
        "temperature": 0.0,
    }
    passthrough_args: set[str] = set()

    def initialize(self, v: Variables, logger_arg: Logger) -> ToolEvalBenchFramework:
        return self

    def check_prerequisites(self) -> list[str]:
        missing: list[str] = []
        if shutil.which("uvx") is None:
            missing.append("uvx not found on PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return missing

    def build_benchmark_command(
        self,
        target_url: str,
        model: str,
        args: dict[str, Any],
        result_file: str | None = None,
    ) -> list[str]:
        """Assemble the ``uvx tool-eval-bench`` argv.

        ``result_file`` is unused: tool-eval-bench has no per-run JSON output
        flag, so we capture stdout in :meth:`parse_results` instead. The
        SQLite/markdown side-effects under uvx land in uvx's ephemeral cache.
        """
        ref = args.get("ref") or _DEFAULT_REF
        wants_perf = any(args.get(k) for k in _PERF_TRIGGER_ARGS)
        if wants_perf:
            from_spec = "tool-eval-bench[perf] @ git+%s@%s" % (_GIT_URL, ref)
        else:
            from_spec = "git+%s@%s" % (_GIT_URL, ref)

        cmd = [
            "uvx",
            "--from",
            from_spec,
            "tool-eval-bench",
            "--base-url",
            target_url,
            "--model",
            model,
            "--json",
        ]

        for key, value in args.items():
            if key in _HIDDEN_ARGS:
                continue
            if key in ("base_url", "model", "json"):
                continue

            flag = "--" + key.replace("_", "-")

            if key in _BOOL_ARGS or isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue

            if isinstance(value, list):
                cmd.append(flag)
                for item in value:
                    cmd.append(str(item))
                continue

            cmd.extend([flag, str(value)])

        return quote_list(cmd)

    def interpret_arg(self, key: str, value: str) -> Any:
        from sparkrun.utils import coerce_value

        if key in _BOOL_ARGS:
            return coerce_value(value)
        if key in _LIST_ARGS or "," in value:
            return [coerce_value(v.strip()) for v in value.split(",")]
        return coerce_value(value)

    def estimate_test_count(self, args: dict[str, Any]) -> int | None:
        scenarios = args.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            return len(scenarios)
        if args.get("short"):
            return 15
        if args.get("hardmode"):
            return 74
        return 69

    def parse_results(
        self,
        stdout: str,
        stderr: str,
        result_file: str | None = None,
    ) -> dict[str, Any]:
        json_data: dict[str, Any] = {}
        text = (stdout or "").strip()
        if text:
            try:
                json_data = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse tool-eval-bench JSON output")

        scores = json_data.get("scores", {}) if isinstance(json_data, dict) else {}
        scenario_results = scores.get("scenario_results", []) if isinstance(scores, dict) else []
        csv_text = scenario_results_to_csv(scenario_results) if scenario_results else ""

        return {
            "json": json_data,
            "csv": csv_text,
            "stdout": stdout,
        }


_CSV_HEADERS = [
    "scenario_id",
    "status",
    "points",
    "duration_seconds",
    "ttft_ms",
    "turn_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "summary",
]


def scenario_results_to_csv(scenario_results: list[dict[str, Any]]) -> str:
    """Render tool-eval-bench scenario results as CSV.

    Optional fields (``ttft_ms``, token counts) emit as empty strings when
    absent, mirroring the convention used by ``llama_benchy.json_to_csv``.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS, extrasaction="ignore")
    writer.writeheader()
    for r in scenario_results:
        row = {h: r.get(h, "") for h in _CSV_HEADERS}
        # Strip newlines so each scenario is one CSV row in grep-style consumers.
        if isinstance(row.get("summary"), str):
            row["summary"] = row["summary"].replace("\n", " ").strip()
        writer.writerow(row)
    return buf.getvalue()
