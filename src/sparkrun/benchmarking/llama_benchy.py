"""llama-benchy benchmarking framework plugin for sparkrun."""

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

logger = logging.getLogger(__name__)

# llama-benchy args that accept multiple space-separated values on CLI,
# stored as comma-separated strings in profiles.
_LIST_ARGS = {"pp", "tg", "depth", "concurrency"}

# llama-benchy boolean flags (present = enabled, absent = disabled).
_BOOL_ARGS = {
    "no_cache",
    "enable_prefix_caching",
    "no_warmup",
    "skip_coherence",
    "adapt_prompt",
    "no_adapt_prompt",
    "save_total_throughput_timeseries",
    "save_all_throughput_timeseries",
    "exit_on_first_fail",
}

# Common shorthand aliases → canonical llama-benchy arg names.
# Profiles may use shorter key names for convenience.
_ARG_ALIASES: dict[str, str] = {
    "prefix_caching": "enable_prefix_caching",
}

_PASSTHROUGH_ARGS = {
    "tokenizer",  # tokenizer value configured in a recipe can pass-thru to benchmark profile definitions
}


class LlamaBenchyFramework(BenchmarkingPlugin):
    """llama-benchy benchmarking framework.

    Uses ``uvx llama-benchy`` to run OpenAI-compatible inference benchmarks.
    Produces JSON output for machine-parseable results.
    """

    framework_name = "llama-benchy"
    default_args: dict[str, Any] = {
        "pp": [2048],
        "depth": [0],
        "prefix_caching": True,
    }
    passthrough_args = _PASSTHROUGH_ARGS

    def initialize(self, v: Variables, logger_arg: Logger) -> LlamaBenchyFramework:
        return self

    def check_prerequisites(self) -> list[str]:
        """Check that uvx is available on PATH."""
        missing = []
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
        """Build the uvx llama-benchy command.

        Always uses ``--format json`` for machine-parseable output and
        ``--save-result`` to capture results to a file.
        """
        # Strip GGUF quant suffix (e.g. "repo/model-GGUF:Q4_K_M" → "repo/model-GGUF")
        # The colon syntax is a sparkrun convention; the served model uses the repo ID.
        from sparkrun.models.download import parse_gguf_model_spec

        model_id, _ = parse_gguf_model_spec(model)

        cmd = [
            "uvx",
            "llama-benchy",
            "--base-url",
            target_url,
            "--model",
            model_id,
            "--format",
            "json",
        ]

        if result_file:
            cmd.extend(["--save-result", result_file])

        # Render args as CLI flags
        for key, value in args.items():
            # Skip args we handle explicitly above
            if key in ("base_url", "model", "format", "save_result"):
                continue

            # Resolve shorthand aliases to canonical names
            key = _ARG_ALIASES.get(key, key)

            flag = "--" + key.replace("_", "-")

            if key in _BOOL_ARGS or isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue

            if isinstance(value, list):
                # llama-benchy takes space-separated values after the flag
                cmd.append(flag)
                for item in value:
                    cmd.append(str(item))
                continue

            cmd.extend([flag, str(value)])

        return cmd

    def interpret_arg(self, key: str, value: str) -> Any:
        """Interpret a CLI string arg into the correct type.

        Known list args (pp, tg, depth, concurrency) become lists when
        comma-separated. Known booleans become bool. Others use coerce_value().
        """
        from sparkrun.utils import coerce_value

        # Resolve shorthand aliases
        key = _ARG_ALIASES.get(key, key)

        if key in _BOOL_ARGS:
            return coerce_value(value)

        if key in _LIST_ARGS or "," in value:
            return [coerce_value(v.strip()) for v in value.split(",")]

        return coerce_value(value)

    def estimate_test_count(self, args: dict[str, Any]) -> int | None:
        """Estimate total test combinations from benchmark args.

        llama-benchy runs the cartesian product of pp x tg x depth x concurrency,
        each repeated ``runs`` times.
        """
        pp = args.get("pp", [2048])
        tg = args.get("tg", [32])
        depth = args.get("depth", [0])
        concurrency = args.get("concurrency", [1])
        runs = args.get("runs", 3)

        def _len(v: Any) -> int:
            return len(v) if isinstance(v, list) else 1

        # noinspection PyBroadException
        try:
            combos = _len(pp) * _len(tg) * _len(depth) * _len(concurrency)
            return combos * int(runs) if combos > 0 else None
        except:
            return None

    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        """Parse llama-benchy JSON output into structured results.

        Reads from the ``--save-result`` file if available (expected to be available)
        """
        json_data: dict[str, Any] = {}

        # Try reading from saved result file first
        json_text = None
        if result_file:
            try:
                from pathlib import Path

                json_text = Path(result_file).read_text()
            except (OSError, FileNotFoundError):
                logger.debug("Could not read result file %s, falling back to stdout", result_file)

        # Parse JSON content
        if json_text is not None and json_text.strip():
            try:
                json_data = json.loads(json_text)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse JSON output from llama-benchy")

        # # Extract benchmark rows for summary display
        # rows = json_data.get("benchmarks", [])

        # Build CSV from JSON when benchmark data is available
        csv_text = ""
        if json_data.get("benchmarks"):
            csv_text = json_to_csv(json_data)

        return {
            # "rows": rows,
            "json": json_data,
            "csv": csv_text,
            "stdout": stdout,
        }


# -- CSV headers matching llama-benchy's save_report(format="csv") --
_CSV_HEADERS = [
    "model",
    "test_name",
    "t_s_mean",
    "t_s_std",
    "t_s_req_mean",
    "t_s_req_std",
    "peak_ts_mean",
    "peak_ts_std",
    "peak_ts_req_mean",
    "peak_ts_req_std",
    "ttfr_mean",
    "ttfr_std",
    "est_ppt_mean",
    "est_ppt_std",
    "e2e_ttft_mean",
    "e2e_ttft_std",
]


def json_to_csv(json_data: dict[str, Any]) -> str:
    """Convert llama-benchy JSON results to a CSV string.

    Replicates the exact row-generation and column layout that llama-benchy
    uses when invoked with ``--format csv``.

    Args:
        json_data: Parsed JSON dict from llama-benchy (the top-level object
            containing ``benchmarks``, ``model``, ``max_concurrency``, etc.).

    Returns:
        CSV string with header row and one data row per metric per benchmark run.
    """
    benchmarks = json_data.get("benchmarks", [])
    model_name = json_data.get("model", "Unknown")
    max_concurrency = json_data.get("max_concurrency", 1)

    rows = _generate_csv_rows(benchmarks, model_name, max_concurrency)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _generate_csv_rows(
    benchmarks: list[dict],
    model_name: str,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    """Replicate llama-benchy's ``_generate_rows()`` → CSV row mapping.

    Each BenchmarkRun in the JSON produces up to two display rows:
    one for prompt-processing (PP) metrics and one for token-generation (TG).
    """
    rows: list[dict[str, Any]] = []

    def _mean(metric: dict | None) -> float | None:
        return metric["mean"] if metric else None

    def _std(metric: dict | None) -> float | None:
        return metric["std"] if metric else None

    def _csv_row(
        model: str,
        test_name: str,
        t_s: dict | None,
        t_s_req: dict | None,
        peak_ts: dict | None,
        peak_ts_req: dict | None,
        ttfr: dict | None,
        est_ppt: dict | None,
        e2e_ttft: dict | None,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "test_name": test_name,
            "t_s_mean": _mean(t_s),
            "t_s_std": _std(t_s),
            "t_s_req_mean": _mean(t_s_req),
            "t_s_req_std": _std(t_s_req),
            "peak_ts_mean": _mean(peak_ts),
            "peak_ts_std": _std(peak_ts),
            "peak_ts_req_mean": _mean(peak_ts_req),
            "peak_ts_req_std": _std(peak_ts_req),
            "ttfr_mean": _mean(ttfr),
            "ttfr_std": _std(ttfr),
            "est_ppt_mean": _mean(est_ppt),
            "est_ppt_std": _std(est_ppt),
            "e2e_ttft_mean": _mean(e2e_ttft),
            "e2e_ttft_std": _std(e2e_ttft),
        }

    for run in benchmarks:
        concurrency = run.get("concurrency", 1)
        context_size = run.get("context_size", 0)
        prompt_size = run.get("prompt_size", 0)
        response_size = run.get("response_size", 0)
        is_context_phase = run.get("is_context_prefill_phase", False)

        c_suffix = " (c%d)" % concurrency if max_concurrency > 1 else ""

        pp_tp = run.get("pp_throughput")
        pp_req_tp = run.get("pp_req_throughput")
        tg_tp = run.get("tg_throughput")
        tg_req_tp = run.get("tg_req_throughput")
        peak_tp = run.get("peak_throughput")
        peak_req_tp = run.get("peak_req_throughput")
        ttfr = run.get("ttfr")
        est_ppt = run.get("est_ppt")
        e2e_ttft = run.get("e2e_ttft")

        if is_context_phase:
            # Context prefill — PP row
            if pp_tp:
                rows.append(
                    _csv_row(
                        model_name,
                        "ctx_pp @ d%d%s" % (context_size, c_suffix),
                        t_s=pp_tp,
                        t_s_req=pp_req_tp,
                        peak_ts=None,
                        peak_ts_req=None,
                        ttfr=ttfr,
                        est_ppt=est_ppt,
                        e2e_ttft=e2e_ttft,
                    )
                )
            # Context prefill — TG row
            if tg_tp:
                rows.append(
                    _csv_row(
                        model_name,
                        "ctx_tg @ d%d%s" % (context_size, c_suffix),
                        t_s=tg_tp,
                        t_s_req=tg_req_tp,
                        peak_ts=peak_tp,
                        peak_ts_req=peak_req_tp,
                        ttfr=None,
                        est_ppt=None,
                        e2e_ttft=None,
                    )
                )
        else:
            d_suffix = " @ d%d" % context_size if context_size > 0 else ""

            # Standard — PP row
            if pp_tp:
                rows.append(
                    _csv_row(
                        model_name,
                        "pp%d%s%s" % (prompt_size, d_suffix, c_suffix),
                        t_s=pp_tp,
                        t_s_req=pp_req_tp,
                        peak_ts=None,
                        peak_ts_req=None,
                        ttfr=ttfr,
                        est_ppt=est_ppt,
                        e2e_ttft=e2e_ttft,
                    )
                )
            # Standard — TG row
            if tg_tp:
                rows.append(
                    _csv_row(
                        model_name,
                        "tg%d%s%s" % (response_size, d_suffix, c_suffix),
                        t_s=tg_tp,
                        t_s_req=tg_req_tp,
                        peak_ts=peak_tp,
                        peak_ts_req=peak_req_tp,
                        ttfr=None,
                        est_ppt=None,
                        e2e_ttft=None,
                    )
                )

    return rows
