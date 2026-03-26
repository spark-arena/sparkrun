"""Proxy configuration management.

Reads/writes ``~/.config/sparkrun/proxy.yaml`` for proxy settings,
model aliases, and (future) default recipe mappings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from sparkrun.proxy import (
    DEFAULT_DISCOVER_INTERVAL,
    DEFAULT_MASTER_KEY,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
)

logger = logging.getLogger(__name__)


class ProxyConfig:
    """Manages proxy configuration stored in ``proxy.yaml``."""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            from sparkrun.core.config import DEFAULT_CONFIG_DIR

            config_path = DEFAULT_CONFIG_DIR / "proxy.yaml"
        self.config_path = config_path
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = yaml.safe_load(f)
                self._data = data if isinstance(data, dict) else {}
            except Exception:
                logger.debug("Failed to load proxy config: %s", self.config_path, exc_info=True)
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        """Write current config to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False, sort_keys=False)
        logger.debug("Saved proxy config to %s", self.config_path)

    # -- Proxy settings --

    @property
    def port(self) -> int:
        return int(self._data.get("proxy", {}).get("port", DEFAULT_PROXY_PORT))

    @property
    def host(self) -> str:
        return str(self._data.get("proxy", {}).get("host", DEFAULT_PROXY_HOST))

    @property
    def master_key(self) -> str | None:
        val = self._data.get("proxy", {}).get("master_key", DEFAULT_MASTER_KEY)
        return str(val) if val is not None else None

    @property
    def auto_discover(self) -> bool:
        return bool(self._data.get("proxy", {}).get("auto_discover", True))

    @property
    def discover_interval(self) -> int:
        return int(self._data.get("proxy", {}).get("discover_interval", DEFAULT_DISCOVER_INTERVAL))

    def set_proxy(self, **kwargs: Any) -> None:
        """Update proxy settings (port, host, master_key, etc.)."""
        proxy = self._data.setdefault("proxy", {})
        proxy.update(kwargs)

    # -- Alias management --

    @property
    def aliases(self) -> dict[str, str]:
        """Return alias -> model group mapping."""
        return dict(self._data.get("aliases", {}))

    def add_alias(self, alias: str, target: str) -> None:
        """Add or update an alias mapping."""
        aliases = self._data.setdefault("aliases", {})
        aliases[alias] = target

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if it existed."""
        aliases = self._data.get("aliases", {})
        if alias in aliases:
            del aliases[alias]
            return True
        return False

    def list_aliases(self) -> list[tuple[str, str]]:
        """Return list of (alias, target) pairs."""
        return list(self._data.get("aliases", {}).items())

    # -- Default recipes (schema only, not wired up) --

    @property
    def default_recipes(self) -> dict[str, dict]:
        """Return default recipe mappings (future use)."""
        return dict(self._data.get("default_recipes", {}))
