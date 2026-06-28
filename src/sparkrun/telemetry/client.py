"""Best-effort telemetry HTTP client."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from sparkrun import __version__
from sparkrun.core.config import SparkrunConfig

from .config import (
    TELEMETRY_AUTH_HEADER,
    ensure_installation_id,
    telemetry_enabled,
    telemetry_endpoint,
    telemetry_key,
    telemetry_timeout,
)
from .types import TelemetryEvent

logger = logging.getLogger(__name__)


def prepare_event(config: SparkrunConfig, event: TelemetryEvent) -> TelemetryEvent | None:
    """Build the telemetry envelope, or None when telemetry is disabled."""
    if not telemetry_enabled(config):
        return None

    envelope: TelemetryEvent = {
        "schema_version": 1,
        "event_id": str(uuid4()),
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "installation_id": ensure_installation_id(config),
        "sparkrun_version": __version__,
    }
    envelope.update(event)
    return envelope


def send_event(config: SparkrunConfig, event: TelemetryEvent) -> None:
    """Send one telemetry event without allowing telemetry failures to escape."""
    payload = prepare_event(config, event)
    if payload is None:
        return

    try:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError):
        logger.debug("Telemetry payload was not JSON serializable", exc_info=True)
        return

    request = Request(
        telemetry_endpoint(),
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "user-agent": "sparkrun/%s" % __version__,
            TELEMETRY_AUTH_HEADER: telemetry_key(),
        },
    )
    try:
        with urlopen(request, timeout=telemetry_timeout()) as response:
            response.read(0)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        logger.debug("Telemetry delivery failed", exc_info=True)
