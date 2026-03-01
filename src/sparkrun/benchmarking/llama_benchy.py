"""llama-benchy benchmarking framework plugin for sparkrun."""

from __future__ import annotations

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
    "no_cache", "enable_prefix_caching", "no_warmup", "skip_coherence",
    "adapt_prompt", "no_adapt_prompt",
    "save_total_throughput_timeseries", "save_all_throughput_timeseries",
}

# Common shorthand aliases → canonical llama-benchy arg names.
# Profiles may use shorter key names for convenience.
_ARG_ALIASES: dict[str, str] = {
    "prefix_caching": "enable_prefix_caching",
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
        "enable_prefix_caching": True,
    }

    def initialize(self, v: Variables, logger_arg: Logger) -> LlamaBenchyFramework:
        return self

    def check_prerequisites(self) -> list[str]:
        """Check that uvx is available on PATH."""
        missing = []
        if shutil.which("uvx") is None:
            missing.append(
                "uvx not found on PATH. Install uv: "
                "https://docs.astral.sh/uv/getting-started/installation/"
            )
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
        cmd = [
            "uvx", "llama-benchy",
            "--base-url", target_url,
            "--model", model,
            "--format", "json",
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

        return {
            # "rows": rows,
            "json": json_data,
            "stdout": stdout,
        }
