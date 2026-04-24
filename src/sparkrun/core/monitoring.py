"""Cluster monitoring — parse host_monitor.sh CSV output and manage parallel SSH streams."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable
import contextlib

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

    # Extended fields (populated by nv-monitor backend)
    gpu_encoder_pct: str = ""
    gpu_decoder_pct: str = ""
    gpu_fan_pct: str = ""
    mem_bufcache_mb: str = ""


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
    for col_name, value in zip(MONITOR_COLUMNS, parts, strict=False):
        kwargs[col_name] = value.strip()

    return MonitorSample(**kwargs)


def prometheus_to_sample(metrics: dict[str, float], hostname: str) -> MonitorSample:
    """Convert parsed Prometheus metrics from nv-monitor to a MonitorSample.

    Maps nv-monitor metric names to MonitorSample fields.

    Args:
        metrics: Flat dict from :func:`~sparkrun.core.prometheus.parse_prometheus_text`.
        hostname: Hostname to set on the sample.

    Returns:
        Populated MonitorSample.
    """
    import time

    from sparkrun.core.prometheus import extract_label

    def _get(key: str, default: str = "") -> str:
        """Get a metric value as a formatted string."""
        val = metrics.get(key)
        if val is None:
            return default
        return str(int(val)) if val == int(val) else "%.1f" % val

    def _get_mb(key: str) -> str:
        """Get a bytes metric as MB string."""
        val = metrics.get(key)
        if val is None:
            return ""
        return str(int(val / (1024 * 1024)))

    # Extract GPU name from info metric label
    gpu_name = ""
    for key in metrics:
        if key.startswith("nv_gpu_info{"):
            gpu_name = extract_label(key, "name") or ""
            break

    # Compute mem_used_pct
    mem_total = metrics.get("nv_memory_total_bytes", 0)
    mem_used = metrics.get("nv_memory_used_bytes", 0)
    mem_available = mem_total - mem_used if mem_total else 0
    mem_used_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

    # Compute gpu_mem_used_pct
    gpu_mem_total = metrics.get('nv_gpu_memory_total_bytes{gpu="0"}', 0)
    gpu_mem_used = metrics.get('nv_gpu_memory_used_bytes{gpu="0"}', 0)
    gpu_mem_used_pct = (gpu_mem_used / gpu_mem_total * 100) if gpu_mem_total > 0 else 0

    return MonitorSample(
        timestamp=str(int(time.time())),
        hostname=hostname,
        uptime_sec=_get("nv_system_uptime_seconds"),
        cpu_load_1m=_get('nv_load_average{interval="1m"}'),
        cpu_load_5m=_get('nv_load_average{interval="5m"}'),
        cpu_load_15m=_get('nv_load_average{interval="15m"}'),
        cpu_usage_pct=_get('nv_cpu_usage_percent{cpu="overall"}'),
        cpu_freq_mhz=_get("nv_cpu_frequency_mhz"),
        cpu_temp_c=_get("nv_cpu_temperature_celsius"),
        mem_total_mb=_get_mb("nv_memory_total_bytes"),
        mem_used_mb=_get_mb("nv_memory_used_bytes"),
        mem_available_mb=str(int(mem_available / (1024 * 1024))) if mem_total else "",
        mem_used_pct="%.1f" % mem_used_pct if mem_total else "",
        swap_total_mb=_get_mb("nv_swap_total_bytes"),
        swap_used_mb=_get_mb("nv_swap_used_bytes"),
        gpu_name=gpu_name,
        gpu_util_pct=_get('nv_gpu_utilization_percent{gpu="0"}'),
        gpu_mem_used_mb=_get_mb('nv_gpu_memory_used_bytes{gpu="0"}'),
        gpu_mem_total_mb=_get_mb('nv_gpu_memory_total_bytes{gpu="0"}'),
        gpu_mem_used_pct="%.1f" % gpu_mem_used_pct if gpu_mem_total else "",
        gpu_temp_c=_get('nv_gpu_temperature_celsius{gpu="0"}'),
        gpu_power_w=_get('nv_gpu_power_watts{gpu="0"}'),
        gpu_power_limit_w=_get('nv_gpu_power_limit_watts{gpu="0"}'),
        gpu_clock_mhz=_get('nv_gpu_clock_mhz{gpu="0",type="graphics"}'),
        gpu_mem_clock_mhz=_get('nv_gpu_clock_mhz{gpu="0",type="memory"}'),
        # Extended fields from nv-monitor
        gpu_encoder_pct=_get('nv_gpu_encoder_utilization_percent{gpu="0"}'),
        gpu_decoder_pct=_get('nv_gpu_decoder_utilization_percent{gpu="0"}'),
        gpu_fan_pct=_get('nv_gpu_fan_speed_percent{gpu="0"}'),
        mem_bufcache_mb=_get_mb("nv_memory_bufcache_bytes"),
    )


def prom2json_to_sample(metrics_list: list[dict], hostname: str) -> MonitorSample:
    """Convert prom2json structured JSON output to a MonitorSample.

    prom2json outputs a list of metric families, each with ``name``,
    ``type``, and ``metrics`` (list of ``{labels, value}`` dicts).
    This function flattens them into the same key format used by
    :func:`prometheus_to_sample` and delegates to it.

    Args:
        metrics_list: Parsed JSON array from prom2json.
        hostname: Hostname to set on the sample.

    Returns:
        Populated MonitorSample.
    """
    flat: dict[str, float] = {}
    for family in metrics_list:
        name = family.get("name", "")
        for metric in family.get("metrics", []):
            labels = metric.get("labels")
            try:
                value = float(metric.get("value", 0))
            except (ValueError, TypeError):
                continue

            if labels:
                label_str = ",".join('%s="%s"' % (k, v) for k, v in sorted(labels.items()))
                key = "%s{%s}" % (name, label_str)
            else:
                key = name
            flat[key] = value

    return prometheus_to_sample(flat, hostname)


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
            assert proc.stdin is not None
            proc.stdin.write(self._script)
            proc.stdin.close()

            self.states[host].process = proc

            thread = threading.Thread(
                target=self._reader,
                args=(host, proc),
                daemon=True,
            )
            thread.start()
        except OSError as e:
            self.states[host].error = "Failed to start SSH: %s" % e
            logger.warning("Failed to start monitor on %s: %s", host, e)

    def stop(self) -> None:
        """Terminate all SSH subprocesses."""
        for _host, state in self.states.items():
            if state.process is not None:
                with contextlib.suppress(OSError):
                    state.process.terminate()
                try:
                    state.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    state.process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        state.process.wait(timeout=2)
        self._started = False

    def _reader(self, host: str, proc: subprocess.Popen) -> None:
        """Read stdout from an SSH process line by line, updating state."""
        try:
            assert proc.stdout is not None
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
                with contextlib.suppress(Exception):
                    stderr_text = proc.stderr.read().strip() if proc.stderr else ""
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
                        host,
                        age,
                        stale_threshold,
                    )
                    self._reconnect_host(host)

    def _reconnect_host(self, host: str) -> None:
        """Kill the stale SSH process for *host* and start a fresh one."""
        state = self.states[host]

        # Terminate the old process so its reader thread exits.
        if state.process is not None:
            with contextlib.suppress(OSError):
                state.process.terminate()
            try:
                state.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                state.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    state.process.wait(timeout=2)

        state.error = "stale data — reconnecting"
        # Reset timestamp so the watchdog doesn't re-trigger immediately.
        state.last_updated = time.monotonic()

        self._start_host(host)


class NvMonitorClusterMonitor:
    """Monitor cluster hosts using nv-monitor + prom2json over SSH.

    Uses the same single-SSH-process-per-host architecture as
    :class:`ClusterMonitor` to avoid GIL contention.  Each host runs a
    wrapper script that starts nv-monitor, polls its Prometheus endpoint
    via curl + prom2json, and streams JSON lines to stdout.  Reader
    threads parse JSON and update state — identical to how the bash
    backend reads CSV lines.
    """

    def __init__(
        self,
        hosts: list[str],
        ssh_kwargs: dict,
        interval: int = 2,
        port: int | None = None,
    ):
        from sparkrun.orchestration.nv_monitor import NV_MONITOR_DEFAULT_PORT

        self.hosts = list(hosts)
        self.ssh_kwargs = ssh_kwargs
        self.interval = interval
        self.port = port or NV_MONITOR_DEFAULT_PORT
        self.states: dict[str, HostMonitorState] = {h: HostMonitorState() for h in hosts}
        self._started = False
        self._script: str = ""
        self._saved_log_levels: dict[str, int] = {}

    def start(self) -> None:
        """Deploy binaries and launch SSH processes for every host.

        Binary deployment runs in a background thread so the TUI renders
        immediately.  Once deployed, each host gets a single SSH process
        streaming JSON — identical pattern to :class:`ClusterMonitor`.
        """
        if self._started:
            return

        from sparkrun.scripts import read_script

        self._script = read_script("nv_monitor_wrapper.sh")
        self._started = True

        # Suppress background loggers — errors go to state, not terminal
        self._saved_log_levels = self._suppress_background_loggers()

        # Mark all hosts as connecting
        for host in self.hosts:
            self.states[host].error = "deploying nv-monitor..."

        # Run deployment + host start in background thread
        setup_thread = threading.Thread(target=self._setup, daemon=True)
        setup_thread.start()

        # Watchdog for stale connections (same as bash backend)
        watchdog = threading.Thread(target=self._watchdog, daemon=True)
        watchdog.start()

    def _suppress_background_loggers(self) -> dict[str, int]:
        """Suppress loggers that would corrupt TUI output."""
        suppressed = {}
        for name in (
            "sparkrun.orchestration.nv_monitor",
            "sparkrun.orchestration.ssh",
        ):
            lg = logging.getLogger(name)
            suppressed[name] = lg.level
            lg.setLevel(logging.CRITICAL)
        return suppressed

    @staticmethod
    def _restore_loggers(saved: dict[str, int]) -> None:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)

    def _setup(self) -> None:
        """Background: deploy binaries then start SSH processes."""
        from sparkrun.orchestration.nv_monitor import ensure_nv_monitor

        deploy_status = ensure_nv_monitor(self.hosts, self.ssh_kwargs)
        for h, ok in deploy_status.items():
            if not ok:
                self.states[h].error = "nv-monitor deploy failed"

        # Start SSH processes for deployed hosts
        for host in self.hosts:
            if deploy_status.get(host, False):
                self._start_host(host)

    def _start_host(self, host: str) -> None:
        """Launch an SSH process running the nv-monitor wrapper script."""
        from sparkrun.orchestration.ssh import build_ssh_cmd

        cmd = build_ssh_cmd(
            host,
            ssh_user=self.ssh_kwargs.get("ssh_user"),
            ssh_key=self.ssh_kwargs.get("ssh_key"),
            ssh_options=self.ssh_kwargs.get("ssh_options"),
        )
        cmd.extend(["bash", "-s", "--", str(self.port), str(self.interval)])

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert proc.stdin is not None
            proc.stdin.write(self._script)
            proc.stdin.close()

            self.states[host].process = proc
            self.states[host].error = "waiting for metrics..."

            thread = threading.Thread(target=self._reader, args=(host, proc), daemon=True)
            thread.start()
        except OSError as e:
            self.states[host].error = "SSH failed: %s" % e

    def stop(self) -> None:
        """Terminate all SSH processes."""
        for _host, state in self.states.items():
            if state.process is not None:
                with contextlib.suppress(OSError):
                    state.process.terminate()
                try:
                    state.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    state.process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        state.process.wait(timeout=2)
        self._started = False
        self._restore_loggers(self._saved_log_levels)

    def _reader(self, host: str, proc: subprocess.Popen) -> None:
        """Read JSON lines from the wrapper script, updating state."""
        import json

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                if "error" in data and "metrics" not in data:
                    self.states[host].error = data["error"]
                    continue

                metrics_list = data.get("metrics", [])
                if not metrics_list:
                    continue

                sample = prom2json_to_sample(metrics_list, host)
                sample.sparkrun_jobs = data.get("sparkrun_jobs", "0")
                sample.sparkrun_job_names = data.get("sparkrun_job_names", "")

                self.states[host].latest = sample
                self.states[host].last_updated = time.monotonic()
                self.states[host].error = None
        except Exception as e:
            logger.debug("Reader thread for %s exited: %s", host, e)
        finally:
            rc = proc.poll()
            if rc is not None and rc != 0:
                stderr_text = ""
                with contextlib.suppress(Exception):
                    stderr_text = proc.stderr.read().strip() if proc.stderr else ""
                if self.states[host].latest is None:
                    short = stderr_text.splitlines()[-1] if stderr_text else "rc=%d" % rc
                    self.states[host].error = short

    # -- staleness detection (same pattern as ClusterMonitor) ----------------

    def _watchdog(self) -> None:
        """Periodically check for stale data and reconnect."""
        stale_threshold = self.interval * 5
        while self._started:
            time.sleep(self.interval)
            if not self._started:
                break
            now = time.monotonic()
            for host in self.hosts:
                state = self.states[host]
                if state.last_updated is None:
                    continue
                age = now - state.last_updated
                if age > stale_threshold:
                    self._reconnect_host(host)

    def _reconnect_host(self, host: str) -> None:
        """Kill stale SSH process and start fresh."""
        state = self.states[host]
        if state.process is not None:
            with contextlib.suppress(OSError):
                state.process.terminate()
            try:
                state.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                state.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    state.process.wait(timeout=2)

        state.error = "stale data — reconnecting"
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
