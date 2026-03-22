"""Base class for sparkrun builders."""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING

from scitrera_app_framework import Plugin, Variables

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

EXT_BUILDER = "sparkrun.builder"


def _flatten_dict(d: dict, prefix: str = "", sep: str = "_", normalize: bool = False) -> dict[str, str]:
    """Recursively flatten a nested dict, joining keys with *sep*.

    When *normalize* is True, dots, slashes, and dashes in keys are
    replaced with *sep* — useful for OCI labels that use dotted names
    like ``org.opencontainers.image.version``.

    >>> _flatten_dict({"version": "1.0", "git": {"commit": "abc"}}, prefix="build")
    {'build_version': '1.0', 'build_git_commit': 'abc'}
    """
    result: dict[str, str] = {}
    for k, v in d.items():
        if normalize:
            k = k.replace(".", sep).replace("/", sep).replace("-", sep)
        full_key = "%s%s%s" % (prefix, sep, k) if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=full_key, sep=sep, normalize=normalize))
        else:
            result[full_key] = str(v)
    return result


class BuilderPlugin(Plugin):
    """Abstract base class for sparkrun image builders.

    Each builder is an SAF Plugin that registers as a multi-extension
    under the 'sparkrun.builder' extension point.

    Subclasses must define:
        - builder_name: str identifier (e.g. "docker-pull", "eugr")
    """

    eager = False
    builder_name: str = ""

    def name(self) -> str:
        return "sparkrun.builder.%s" % self.builder_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_BUILDER

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: Logger) -> BuilderPlugin:
        return self

    def prepare_image(
            self,
            image: str,
            recipe: Recipe,
            hosts: list[str],
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            transfer_mode: str = "local",
            ssh_kwargs: dict | None = None,
    ) -> str:
        """Ensure image is available. Returns final image name.

        Called before the distribution phase. After this returns,
        the image should exist locally (or on the head node when
        *transfer_mode* is ``"delegated"``) so distribution can sync
        it to remote hosts.

        Args:
            image: Target image name.
            recipe: The loaded recipe.
            hosts: Target host list (first element is head).
            config: SparkrunConfig for cache dir resolution.
            dry_run: Show what would be done without executing.
            transfer_mode: ``"local"`` (build locally) or
                ``"delegated"`` (build on head node via SSH).
            ssh_kwargs: SSH connection kwargs (needed for delegated mode).
        """
        return image

    def version_info_commands(self) -> dict[str, str]:
        """Return label→shell command pairs for raw data capture from container.

        Unlike runtime version_commands (single value per command), these
        commands can produce multi-line output. Raw stdout is passed to
        process_version_info() for Python-side processing.
        """
        return {}

    def process_version_info(self, raw: dict[str, str]) -> dict[str, str]:
        """Process raw command outputs into flat key-value pairs.

        Args:
            raw: {label: raw_stdout} from version_info_commands().
        Returns:
            Flat dict to merge into runtime_info.
        """
        return {}

    def collect_container_labels(
            self,
            container_name: str,
            host: str,
            ssh_kwargs: dict,
    ) -> dict[str, str]:
        """Inspect container labels and return as flat dict with 'container_' prefix.

        Default implementation uses ``docker inspect``. Subclasses may
        override for alternative container engines (podman, etc.).
        Fails silently — label collection never blocks a launch.
        """
        try:
            import json
            from sparkrun.orchestration.primitives import run_script_on_host

            script = "docker inspect --format '{{json .Config.Labels}}' %s 2>/dev/null || echo '{}'" % container_name
            result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=15)
            if result.returncode != 0 or not result.stdout.strip():
                return {}
            raw = result.stdout.strip()
            data = json.loads(raw)
            if not isinstance(data, dict) or not data:
                return {}
            return _flatten_dict(data, prefix="container", normalize=True)
        except Exception:
            logger.debug("Container label collection failed", exc_info=True)
            return {}

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate builder-specific recipe fields."""
        return []

    def __repr__(self) -> str:
        return "%s(builder_name=%r)" % (self.__class__.__name__, self.builder_name)
