"""Telemetry runtime configuration."""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)

TELEMETRY_ENV = "SPARKRUN_TELEMETRY"
NO_TELEMETRY_ENV = "SPARKRUN_NO_TELEMETRY"
TELEMETRY_ENDPOINT_ENV = "SPARKRUN_TELEMETRY_ENDPOINT"
TELEMETRY_KEY_ENV = "SPARKRUN_TELEMETRY_KEY"
TELEMETRY_TIMEOUT_ENV = "SPARKRUN_TELEMETRY_TIMEOUT"

TELEMETRY_ENABLED_KEY = "telemetry.enabled"
TELEMETRY_INSTALLATION_ID_KEY = "telemetry.installation_id"

DEFAULT_TELEMETRY_ENDPOINT = "https://telemetry.sparkrun.dev/"
DEFAULT_TELEMETRY_KEY = "sparkrun-telemetry-v1"
TELEMETRY_AUTH_HEADER = "x-sparkrun-telemetry-key"
DEFAULT_TIMEOUT_SECONDS = 0.75

_TRUE_VALUES = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}


def parse_bool(value: bool | str | int | float | None) -> bool | None:
    """Parse common boolean spellings, returning None for unset/unknown."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def env_telemetry_override() -> bool | None:
    """Return the environment-level telemetry decision, if one is set."""
    no_telemetry = parse_bool(os.environ.get(NO_TELEMETRY_ENV))
    if no_telemetry is True:
        return False
    return parse_bool(os.environ.get(TELEMETRY_ENV))


def telemetry_enabled(config: SparkrunConfig) -> bool:
    """Return whether telemetry should be emitted for this process."""
    env_override = env_telemetry_override()
    if env_override is not None:
        return env_override
    configured = parse_bool(config.get(TELEMETRY_ENABLED_KEY))
    if configured is not None:
        return configured
    return True


def persistent_telemetry_setting(config: SparkrunConfig) -> bool | None:
    """Return the persisted telemetry setting, ignoring environment overrides."""
    return parse_bool(config.get(TELEMETRY_ENABLED_KEY))


def set_persistent_telemetry(config: SparkrunConfig, enabled: bool) -> None:
    """Persist the telemetry enablement preference."""
    config.set(TELEMETRY_ENABLED_KEY, bool(enabled))
    config.save()


def telemetry_endpoint() -> str:
    return os.environ.get(TELEMETRY_ENDPOINT_ENV, DEFAULT_TELEMETRY_ENDPOINT).strip() or DEFAULT_TELEMETRY_ENDPOINT


def telemetry_key() -> str:
    return os.environ.get(TELEMETRY_KEY_ENV, DEFAULT_TELEMETRY_KEY).strip() or DEFAULT_TELEMETRY_KEY


def telemetry_timeout() -> float:
    raw = os.environ.get(TELEMETRY_TIMEOUT_ENV)
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(0.05, min(timeout, 5.0))


def ensure_installation_id(config: SparkrunConfig) -> str:
    """Return a persistent anonymous installation id, creating one if needed."""
    raw = config.get(TELEMETRY_INSTALLATION_ID_KEY)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    installation_id = str(uuid4())
    try:
        config.set(TELEMETRY_INSTALLATION_ID_KEY, installation_id)
        config.save()
    except OSError:
        logger.debug("Failed to persist telemetry installation id", exc_info=True)
    return installation_id
