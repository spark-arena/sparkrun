"""Run diagnostics collector — wraps a sparkrun run lifecycle.

Captures recipe info, phase timing, errors, container logs, and host
diagnostics into an NDJSON file for post-mortem analysis.
"""

from __future__ import annotations

import logging
import time
import traceback

from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.diagnostics.spark_collector import collect_spark_diagnostics

logger = logging.getLogger(__name__)


class _DiagnosticsLogHandler(logging.Handler):
    """Logging handler that captures records into an NDJSON writer.

    Attached to the root logger at DEBUG level so that *all* log
    records are captured regardless of the console verbosity setting.
    Records are emitted as ``log`` events in the diagnostics file.
    """

    def __init__(self, writer: NDJSONWriter):
        super().__init__(level=logging.DEBUG)
        self._writer = writer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._writer.emit(
                "log",
                {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": self.format(record),
                    "timestamp": record.created,
                },
            )
        except Exception:
            pass  # never let diagnostics logging break the run


class RunDiagnosticsCollector:
    """Collects diagnostics around a ``sparkrun run`` lifecycle.

    Usage::

        with RunDiagnosticsCollector(path, hosts, ssh_kwargs) as diag:
            diag.collect_spark_diagnostics()
            diag.emit_recipe(recipe, overrides)
            diag.phase_start("launch")
            result = launch_inference(...)
            diag.phase_end("launch")
            diag.emit_summary()

    When opened, attaches a logging handler to the root logger that
    captures **all** log records (DEBUG and above) into the NDJSON
    file.  This provides full diagnostic logging in the output file
    even when the console is at default (quiet) verbosity.
    """

    def __init__(
        self,
        output_path: str,
        hosts: list[str],
        ssh_kwargs: dict,
        dry_run: bool = False,
    ):
        self._writer = NDJSONWriter(output_path)
        self._hosts = hosts
        self._ssh_kwargs = ssh_kwargs
        self._dry_run = dry_run
        self._start_time = time.monotonic()
        self._phases: dict[str, dict] = {}
        self._current_phase: str | None = None
        self._success = True
        self._log_handler: _DiagnosticsLogHandler | None = None
        self._prev_root_level: int | None = None

    @property
    def writer(self) -> NDJSONWriter:
        return self._writer

    def open(self) -> RunDiagnosticsCollector:
        self._writer.open()
        self._start_time = time.monotonic()
        # Attach a log handler that captures all records at DEBUG level
        self._log_handler = _DiagnosticsLogHandler(self._writer)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        # Ensure root logger accepts DEBUG even if console is at PROGRESS
        if root.level > logging.DEBUG:
            self._prev_root_level = root.level
            root.setLevel(logging.DEBUG)
        else:
            self._prev_root_level = None
        return self

    def close(self) -> None:
        # Remove the diagnostics handler and restore root level
        if self._log_handler is not None:
            root = logging.getLogger()
            root.removeHandler(self._log_handler)
            if self._prev_root_level is not None:
                root.setLevel(self._prev_root_level)
            self._log_handler = None
        self._writer.close()

    def collect_spark_diagnostics(self) -> dict[str, dict]:
        """Run host diagnostics (Tool #1) and emit to the writer."""
        return collect_spark_diagnostics(
            hosts=self._hosts,
            ssh_kwargs=self._ssh_kwargs,
            writer=self._writer,
            dry_run=self._dry_run,
        )

    def emit_header(self, **kwargs) -> None:
        """Emit a ``diag_header`` record with sparkrun version and hosts."""
        try:
            from sparkrun import __version__
        except Exception:
            __version__ = "unknown"
        data = {
            "sparkrun_version": __version__,
            "hosts": self._hosts,
        }
        data.update(kwargs)
        self._writer.emit("diag_header", data)

    def emit_recipe(self, recipe, overrides: dict | None = None) -> None:
        """Emit ``run_recipe`` with recipe details and CLI overrides."""
        data = {
            "name": getattr(recipe, "qualified_name", str(recipe)),
            "model": getattr(recipe, "model", ""),
            "runtime": getattr(recipe, "runtime", ""),
            "container": getattr(recipe, "container", ""),
            "defaults": getattr(recipe, "defaults", {}),
            "overrides": overrides or {},
        }
        self._writer.emit("run_recipe", data)

    def emit_config(self, **kwargs) -> None:
        """Emit ``run_config`` with launch configuration."""
        self._writer.emit("run_config", kwargs)

    def emit_serve_command(self, command: str, container_image: str) -> None:
        """Emit ``run_serve_command`` with the generated command and image."""
        self._writer.emit(
            "run_serve_command",
            {
                "command": command,
                "container_image": container_image,
            },
        )

    def phase_start(self, name: str) -> None:
        """Record the start of a named phase."""
        self._current_phase = name
        self._phases[name] = {"start": time.monotonic(), "end": None, "error": None}
        self._writer.emit("run_phase", {"phase": name, "status": "start"})

    def phase_end(self, name: str, error: str | None = None) -> None:
        """Record the end of a named phase."""
        phase = self._phases.get(name, {})
        phase["end"] = time.monotonic()
        phase["error"] = error
        duration = phase["end"] - phase.get("start", phase["end"])

        status = "error" if error else "end"
        data = {"phase": name, "status": status, "duration_seconds": round(duration, 2)}
        if error:
            data["error"] = error
            self._success = False
        self._writer.emit("run_phase", data)

    def emit_launch_result(self, result) -> None:
        """Emit ``run_launch_result`` from a LaunchResult."""
        data = {
            "rc": getattr(result, "rc", -1),
            "cluster_id": getattr(result, "cluster_id", ""),
            "runtime_info": getattr(result, "runtime_info", {}),
            "nccl_env": getattr(result, "nccl_env", None),
        }
        if data["rc"] != 0:
            self._success = False
        self._writer.emit("run_launch_result", data)

    def emit_health_check(self, url: str, attempt: int, status_code: int | None, success: bool) -> None:
        """Emit ``run_health_check`` for one health-check attempt."""
        self._writer.emit(
            "run_health_check",
            {
                "url": url,
                "attempt": attempt,
                "status_code": status_code,
                "success": success,
            },
        )

    def capture_container_logs(self, host: str, container: str, ssh_kwargs: dict, tail: int = 200) -> None:
        """Capture tail of container logs and emit as ``run_container_logs``."""
        from sparkrun.orchestration.ssh import run_remote_script

        script = "docker logs --tail %d %s 2>&1 || echo '[no logs]'" % (tail, container)
        result = run_remote_script(host, script, dry_run=self._dry_run, **ssh_kwargs)
        lines = result.stdout.strip().splitlines() if result.success else []
        self._writer.emit(
            "run_container_logs",
            {
                "host": host,
                "container": container,
                "lines": lines,
                "tail": tail,
            },
        )

    def emit_error(self, phase: str, error: str | Exception, tb: str | None = None) -> None:
        """Emit ``run_error`` with phase context and optional traceback."""
        self._success = False
        if tb is None and isinstance(error, Exception):
            tb_list = traceback.format_exception(type(error), error, error.__traceback__)
            tb = "".join(tb_list)
        self._writer.emit(
            "run_error",
            {
                "phase": phase,
                "error": str(error),
                "traceback": tb,
            },
        )

    def emit_summary(self) -> None:
        """Emit ``run_summary`` with total duration and phase timings."""
        total = time.monotonic() - self._start_time
        phases = {}
        for name, info in self._phases.items():
            start = info.get("start", 0)
            end = info.get("end") or time.monotonic()
            phases[name] = {
                "duration_seconds": round(end - start, 2),
                "error": info.get("error"),
            }
        self._writer.emit(
            "run_summary",
            {
                "total_duration_seconds": round(total, 2),
                "phases": phases,
                "success": self._success,
            },
        )

    def __enter__(self) -> RunDiagnosticsCollector:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val is not None:
            phase = self._current_phase or "unknown"
            self.emit_error(phase, exc_val)
            self.emit_summary()
        self.close()
