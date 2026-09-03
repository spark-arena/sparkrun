"""Sparkrun field diagnostics — host and run diagnostic collection.

Public API:
    - :func:`collect_spark_diagnostics` — collect hardware/firmware/network/Docker
      info from one or more hosts.
    - :func:`summarize_host_diagnostics` — condense one host's raw probe output
      into a display summary (platform, OS, CPU/RAM, GPU + driver, CUDA, Docker).
    - :class:`NDJSONWriter` — append-only, immediate-flush NDJSON writer.
    - :class:`RunDiagnosticsCollector` — wraps a ``sparkrun run`` lifecycle with
      phase timing, error capture, and log collection.
"""

from __future__ import annotations

from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.diagnostics.spark_collector import (
    collect_config_diagnostics,
    collect_spark_diagnostics,
    collect_sudo_diagnostics,
    summarize_host_diagnostics,
)
from sparkrun.diagnostics.run_collector import RunDiagnosticsCollector

__all__ = [
    "NDJSONWriter",
    "RunDiagnosticsCollector",
    "collect_config_diagnostics",
    "collect_spark_diagnostics",
    "collect_sudo_diagnostics",
    "summarize_host_diagnostics",
]
