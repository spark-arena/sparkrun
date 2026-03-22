"""Sparkrun field diagnostics — host and run diagnostic collection.

Public API:
    - :func:`collect_spark_diagnostics` — collect hardware/firmware/network/Docker
      info from one or more hosts.
    - :class:`NDJSONWriter` — append-only, immediate-flush NDJSON writer.
    - :class:`RunDiagnosticsCollector` — wraps a ``sparkrun run`` lifecycle with
      phase timing, error capture, and log collection.
"""

from __future__ import annotations

from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.diagnostics.spark_collector import collect_spark_diagnostics, collect_sudo_diagnostics
from sparkrun.diagnostics.run_collector import RunDiagnosticsCollector

__all__ = [
    "NDJSONWriter",
    "RunDiagnosticsCollector",
    "collect_spark_diagnostics",
    "collect_sudo_diagnostics",
]
