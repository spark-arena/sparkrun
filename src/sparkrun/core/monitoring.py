"""Cluster monitoring — parse host_monitor.sh CSV output and manage parallel SSH streams."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# Column names matching host_monitor.sh CSV field order (single source of truth).
MONITOR_COLUMNS = (
    "timestamp",
    "hostname",
    "uptime_sec",
    "cpu_load_1m",
    "cpu_load_5m",
    "cpu_load_15m",
    "cpu_usage_pct",
    "cpu_freq_mhz",
    "cpu_temp_c",
    "mem_total_mb",
    "mem_used_mb",
    "mem_available_mb",
    "mem_used_pct",
    "swap_total_mb",
    "swap_used_mb",
    "gpu_name",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_mem_used_pct",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_power_limit_w",
    "gpu_clock_mhz",
    "gpu_mem_clock_mhz",
    "sparkrun_jobs",
    "sparkrun_job_names",
)


@dataclass
class MonitorSample:
    """Parsed fields from one CSV row of host_monitor.sh output."""

    timestamp: str = ""
    hostname: str = ""
    uptime_sec: str = ""
    cpu_load_1m: str = ""
    cpu_load_5m: str = ""
    cpu_load_15m: str = ""
    cpu_usage_pct: str = ""
    cpu_freq_mhz: str = ""
    cpu_temp_c: str = ""
    mem_total_mb: str = ""
    mem_used_mb: str = ""
    mem_available_mb: str = ""
    mem_used_pct: str = ""
    swap_total_mb: str = ""
    swap_used_mb: str = ""
    gpu_name: str = ""
    gpu_util_pct: str = ""
    gpu_mem_used_mb: str = ""
    gpu_mem_total_mb: str = ""
    gpu_mem_used_pct: str = ""
    gpu_temp_c: str = ""
    gpu_power_w: str = ""
    gpu_power_limit_w: str = ""
    gpu_clock_mhz: str = ""
    gpu_mem_clock_mhz: str = ""
    sparkrun_jobs: str = ""
    sparkrun_job_names: str = ""


@dataclass
class HostMonitorState:
    """Per-host state for the monitoring stream."""

    latest: MonitorSample | None = None
    error: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    last_updated: float | None = field(default=None, repr=False)


def parse_monitor_line(line: str) -> MonitorSample | None:
    """Parse a CSV line from host_monitor.sh into a MonitorSample.

    Returns None if the line is malformed (wrong number of fields, empty, etc.).
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split(",")
    if len(parts) != len(MONITOR_COLUMNS):
        return None

    kwargs = {}
    for col_name, value in zip(MONITOR_COLUMNS, parts):
        kwargs[col_name] = value.strip()

    return MonitorSample(**kwargs)


class ClusterMonitor:
    """Manage parallel SSH monitor streams across cluster hosts.

    Provides a start/stop lifecycle suitable for both the simple fallback
    (via :func:`stream_cluster_monitor`) and the Textual TUI.
    """

    def __init__(self, hosts: list[str], ssh_kwargs: dict, interval: int = 2):
        self.hosts = list(hosts)
        self.ssh_kwargs = ssh_kwargs
        self.interval = interval
        self.states: dict[str, HostMonitorState] = {h: HostMonitorState() for h in hosts}
        self._started = False
        self._script: str = ""

    def start(self) -> None:
        """Launch SSH subprocesses and reader threads for every host."""
        if self._started:
            return

        from sparkrun.scripts import read_script

        self._script = read_script("host_monitor.sh")

        for host in self.hosts:
            self._start_host(host)

        self._started = True

        # Background watchdog detects stale connections and reconnects.
        watchdog = threading.Thread(target=self._watchdog, daemon=True)
        watchdog.start()

    def _start_host(self, host: str) -> None:
        """Launch an SSH subprocess and reader thread for a single *host*."""
        from sparkrun.orchestration.ssh import build_ssh_cmd

        cmd = build_ssh_cmd(
            host,
            ssh_user=self.ssh_kwargs.get("ssh_user"),
            ssh_key=self.ssh_kwargs.get("ssh_key"),
            ssh_options=self.ssh_kwargs.get("ssh_options"),
        )
        cmd.extend(["bash", "-s", "--", str(self.interval)])

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc.stdin.write(self._script)
            proc.stdin.close()

            self.states[host].process = proc

            thread = threading.Thread(
                target=self._reader, args=(host, proc), daemon=True,
            )
            thread.start()
        except OSError as e:
            self.states[host].error = "Failed to start SSH: %s" % e
            logger.warning("Failed to start monitor on %s: %s", host, e)

    def stop(self) -> None:
        """Terminate all SSH subprocesses."""
        for host, state in self.states.items():
            if state.process is not None:
                try:
                    state.process.terminate()
                except OSError:
                    pass
                try:
                    state.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    state.process.kill()
                    try:
                        state.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
        self._started = False

    def _reader(self, host: str, proc: subprocess.Popen) -> None:
        """Read stdout from an SSH process line by line, updating state."""
        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                sample = parse_monitor_line(line)
                if sample is not None:
                    self.states[host].latest = sample
                    self.states[host].last_updated = time.monotonic()
                    self.states[host].error = None
        except Exception as e:
            logger.debug("Reader thread for %s exited: %s", host, e)
        finally:
            rc = proc.poll()
            if rc is not None and rc != 0:
                stderr_text = ""
                try:
                    stderr_text = proc.stderr.read().strip() if proc.stderr else ""
                except Exception:
                    pass
                if self.states[host].latest is None:
                    self.states[host].error = stderr_text or "SSH connection failed (rc=%d)" % rc

    # -- staleness detection --------------------------------------------------

    def _watchdog(self) -> None:
        """Periodically check for stale host data and reconnect."""
        stale_threshold = self.interval * 5
        while self._started:
            time.sleep(self.interval)
            if not self._started:
                break
            now = time.monotonic()
            for host in self.hosts:
                state = self.states[host]
                if state.last_updated is None:
                    continue  # still connecting or already in error
                age = now - state.last_updated
                if age > stale_threshold:
                    logger.warning(
                        "Host %s data is %.1fs stale (threshold %.1fs), reconnecting",
                        host, age, stale_threshold,
                    )
                    self._reconnect_host(host)

    def _reconnect_host(self, host: str) -> None:
        """Kill the stale SSH process for *host* and start a fresh one."""
        state = self.states[host]

        # Terminate the old process so its reader thread exits.
        if state.process is not None:
            try:
                state.process.terminate()
            except OSError:
                pass
            try:
                state.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                state.process.kill()
                try:
                    state.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

        state.error = "stale data — reconnecting"
        # Reset timestamp so the watchdog doesn't re-trigger immediately.
        state.last_updated = time.monotonic()

        self._start_host(host)


def stream_cluster_monitor(
    hosts: list[str],
    ssh_kwargs: dict,
    interval: int = 2,
    on_update: Callable[[dict[str, HostMonitorState]], None] | None = None,
    dry_run: bool = False,
) -> None:
    """Stream host_monitor.sh on all cluster hosts and call on_update with latest data.

    Simple blocking interface — launches SSH subprocesses, loops calling
    *on_update* with the current snapshot until KeyboardInterrupt.  Used
    by the plain-text fallback when Textual is unavailable.

    Args:
        hosts: List of hostnames/IPs to monitor.
        ssh_kwargs: SSH connection kwargs (ssh_user, ssh_key, ssh_options).
        interval: Sampling interval in seconds passed to host_monitor.sh.
        on_update: Callback receiving ``dict[host, HostMonitorState]`` each tick.
        dry_run: If True, show what would be run and return immediately.
    """
    if dry_run:
        from sparkrun.orchestration.ssh import build_ssh_cmd

        for host in hosts:
            cmd = build_ssh_cmd(
                host,
                ssh_user=ssh_kwargs.get("ssh_user"),
                ssh_key=ssh_kwargs.get("ssh_key"),
                ssh_options=ssh_kwargs.get("ssh_options"),
            )
            cmd.extend(["bash", "-s", "--", str(interval)])
            logger.info("[dry-run] Would run on %s: %s", host, " ".join(cmd))
        return

    monitor = ClusterMonitor(hosts, ssh_kwargs, interval)
    monitor.start()

    try:
        while True:
            time.sleep(1)
            if on_update:
                on_update(monitor.states)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
