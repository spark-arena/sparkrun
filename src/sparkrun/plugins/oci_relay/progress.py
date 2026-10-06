# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see LICENSE_EXCEPTION.

"""Bounded, default-visible status; relay observations never decide success."""

from __future__ import annotations

import logging
import threading
import time

# Sparkrun's public progress level (between INFO and WARNING). Keep rendering
# importable without the host so the dependency-isolated plugin CI can test it.
PROGRESS = 25

logger = logging.getLogger(__name__)


def size(value):
    value = max(0, value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024


def layers(value):
    return f"{value} layer{'s' if value != 1 else ''} reused"


class Progress:
    def __init__(self, image, *, interval=30, heartbeat=30):
        self.started = time.monotonic()
        self.interval = interval
        self.heartbeat = heartbeat
        self.label = "starting"
        self.last_output = self.started
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.hosts = {}
        self.seen = {}
        self.last_host = {}
        self.finished = set()
        self.pending = set()
        self.registry_sample = None
        self.registry_bytes = None
        self.wire_bytes = 0
        self.wire_known = True
        self.reused_layers = 0
        self.cleanup_incomplete = False
        self.phase(f"preparing {image}")
        self.thread = threading.Thread(target=self._heartbeat, name="oci-relay-progress", daemon=True)
        self.thread.start()

    def _log(self, message):
        logger.log(PROGRESS, "OCI Relay: %s", message)
        self.last_output = time.monotonic()

    def _heartbeat(self):
        while not self.stopped.wait(min(1, self.heartbeat)):
            with self.lock:
                if time.monotonic() - self.last_output >= self.heartbeat:
                    self._log(f"{self.label}; running {time.monotonic() - self.started:.0f}s")

    def phase(self, label):
        with self.lock:
            self.label = label
            self._log(label)

    def bind(self, hosts):
        with self.lock:
            self.hosts = dict(hosts)

    def event(self, event):
        with self.lock:
            now = time.monotonic()
            for identity, result in event.get("receivers", {}).items():
                if identity not in self.hosts or identity in self.pending or identity in self.finished:
                    continue
                self.pending.add(identity)
                label = "import finished" if result.get("state") == "COMPLETE" else "receiver stopped"
                self._log(f"{self.hosts[identity]}: {label}; awaiting final checks")
            for identity, observation in event.get("receiver_progress", {}).items():
                if identity not in self.hosts or identity in self.finished or identity in self.pending:
                    continue
                sequence = observation.get("sequence", 0)
                previous = self.seen.get(identity, {})
                if sequence <= previous.get("sequence", 0):
                    continue
                phase = observation.get("phase", "checking")
                # Phase changes are immediate; byte updates are throttled per host.
                if phase == previous.get("phase") and now - self.last_host.get(identity, 0) < self.interval:
                    continue
                self.seen[identity] = observation
                self.last_host[identity] = now
                label = {
                    "checking": "checking image",
                    "discovering": "checking cached layers",
                    "transferring": "transferring / importing",
                    "importing": "importing into Docker",
                    "verifying": "verifying image",
                    "cleanup": "cleaning up",
                }.get(phase, "working")
                if phase == "importing" and observation.get("active_streams", 0):
                    label = "transferring / importing"
                detail = ""
                if observation.get("inventory_ready"):
                    received = observation.get("received_bytes", 0)
                    expected = observation.get("expected_bytes", 0)
                    detail = f"; {size(received)} received (up to {size(expected)} needed)"
                    rate = observation.get("bytes_per_second", 0)
                    if rate > 0:
                        detail += f", {size(rate)}/s"
                    detail += f"; {layers(observation.get('reused_layers', 0))}"
                self._log(f"{self.hosts[identity]}: {label}{detail}; {observation.get('elapsed_seconds', 0):.0f}s")
            metrics = event.get("registry_metrics")
            if metrics is not None:
                current = metrics.get("upstream_blob_bytes", 0)
                self.registry_bytes = current
                if self.registry_sample is None or now - self.registry_sample[0] >= self.interval:
                    detail = ""
                    if self.registry_sample is not None:
                        dt = now - self.registry_sample[0]
                        detail = f", {size(max(0, current - self.registry_sample[1]) / dt)}/s"
                    self._log(f"registry download: {size(current)}{detail}")
                    self.registry_sample = (now, current)

    def outcome(self, identity, observation, valid):
        with self.lock:
            self.finished.add(identity)
            p = observation.get("progress", {})
            self.wire_bytes += p.get("wire_bytes", 0)
            self.wire_known = self.wire_known and bool(p)
            reused = observation.get("reused_layers", 0) + observation.get("cached_blob_layers", 0)
            self.reused_layers += reused
            if not valid:
                label = "failed"
            elif observation.get("already_present"):
                label = "image already present — verified"
            else:
                label = "image verified"
            if observation.get("cleanup_error"):
                self.cleanup_incomplete = True
                label += "; cleanup incomplete"
            detail = f"; {size(p['wire_bytes'])} transferred" if p else ""
            self._log(f"{self.hosts[identity]}: {label}{detail}; {layers(reused)}")

    def close(self, completed, cleanup):
        self.stopped.set()
        self.thread.join(timeout=1)
        with self.lock:
            label = "complete" if completed else "failed"
            if cleanup or self.cleanup_incomplete:
                label += "; cleanup incomplete"
            upstream = f"; registry downloaded {size(self.registry_bytes)}" if self.registry_bytes is not None else ""
            transfer = (f"{size(self.wire_bytes)} transferred to receivers"
                        if self.wire_known and self.hosts and len(self.finished) == len(self.hosts)
                        else "receiver byte total unavailable")
            self._log(f"{label} in {time.monotonic() - self.started:.1f}s; "
                      f"{transfer}; {layers(self.reused_layers)}{upstream}")
