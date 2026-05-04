"""User configuration management for sparkrun."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

from vpd.next.util import read_yaml

if TYPE_CHECKING:
    from scitrera_app_framework import Variables
    from sparkrun.core.registry import RegistryManager

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "sparkrun"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sparkrun"

# Defer to huggingface_hub's own resolution of the cache root, which
# respects HF_HOME, HF_HUB_CACHE, and HUGGINGFACE_HUB_CACHE env vars.
try:
    from huggingface_hub.constants import HF_HOME as _HF_HOME

    DEFAULT_HF_CACHE_DIR = Path(_HF_HOME)
except ImportError:  # pragma: no cover — huggingface_hub is a required dep
    DEFAULT_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"


def resolve_sparkrun_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve sparkrun's own cache directory (~/.cache/sparkrun/).

    For HuggingFace model cache, use ``resolve_cache_dir()`` instead.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR


def resolve_hf_cache_home(cache_dir: str | None) -> str:
    """Resolve an optional cache directory override to a concrete path.

    Returns *cache_dir* if provided, otherwise the HuggingFace cache
    directory as resolved by ``huggingface_hub`` (respecting ``HF_HOME``
    and related env vars).
    """
    return cache_dir or str(DEFAULT_HF_CACHE_DIR)


def resolve_hf_token() -> Optional[str]:
    try:
        # noinspection PyUnusedImports
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        pass

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_config_root(v: Variables | None = None) -> Path:
    """Config root from SAF stateful root, falling back to DEFAULT_CONFIG_DIR."""
    if v is not None:
        from scitrera_app_framework.core import is_stateful_ready

        stateful_root = is_stateful_ready(v)
        if stateful_root:
            return Path(stateful_root)
    return DEFAULT_CONFIG_DIR


class SparkrunConfig:
    """Manages sparkrun user configuration."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or (DEFAULT_CONFIG_DIR / "config.yaml")
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.config_path.exists():
            self._data = read_yaml(str(self.config_path)) or {}
        else:
            self._data = {}

    @property
    def cache_dir(self) -> Path:
        return Path(self._data.get("cache_dir", str(DEFAULT_CACHE_DIR)))

    @property
    def hf_cache_dir(self) -> Path:
        return Path(self._data.get("hf_cache_dir", str(DEFAULT_HF_CACHE_DIR)))

    @property
    def default_benchmark_output_dir(self) -> Path:
        defaults = self._data.get("defaults", {})
        dir_val = defaults.get("benchmark_output_dir")
        return Path(os.path.expanduser(str(dir_val))) if dir_val else Path.cwd()

    @property
    def default_hosts(self) -> list[str]:
        cluster = self._data.get("cluster", {})
        return cluster.get("hosts", [])

    @property
    def default_image_prefix(self) -> str:
        defaults = self._data.get("defaults", {})
        return defaults.get("image_prefix", "")

    @property
    def default_transformers_tag(self) -> str:
        defaults = self._data.get("defaults", {})
        return defaults.get("transformers", "t4")

    @property
    def ssh_user(self) -> str | None:
        if hasattr(self, "_ssh_user_override"):
            return self._ssh_user_override
        ssh = self._data.get("ssh", {})
        return ssh.get("user")

    @ssh_user.setter
    def ssh_user(self, value: str | None) -> None:
        self._ssh_user_override = value

    @property
    def ssh_key(self) -> str | None:
        ssh = self._data.get("ssh", {})
        key = ssh.get("key")
        return os.path.expanduser(key) if key else None

    @property
    def ssh_options(self) -> list[str]:
        ssh = self._data.get("ssh", {})
        return ssh.get("options", [])

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path."""
        parts = key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    @property
    def monitor_backend(self) -> str | None:
        """Monitoring backend preference: ``"bash"`` or ``"nv-monitor"``."""
        return self._data.get("monitor_backend")

    def get_recipe_search_paths(self) -> list[Path]:
        """Return ordered list of paths to search for recipes."""
        paths = []
        # 1. Current directory recipes/
        cwd_recipes = Path.cwd() / "recipes"
        if cwd_recipes.is_dir():
            paths.append(cwd_recipes)
        # 2. User config recipes/
        user_recipes = DEFAULT_CONFIG_DIR / "recipes"
        if user_recipes.is_dir():
            paths.append(user_recipes)
        # 3. Extra search paths from config
        for extra in self._data.get("recipe_paths", []):
            p = Path(os.path.expanduser(extra))
            if p.is_dir():
                paths.append(p)
        return paths

    def get_registry_manager(self) -> "RegistryManager":
        """Create a RegistryManager using the config root and cache dir."""
        from sparkrun.core.registry import RegistryManager

        return RegistryManager(
            config_root=self.config_path.parent if self.config_path else DEFAULT_CONFIG_DIR,
            cache_root=self.cache_dir / "registries",
        )
