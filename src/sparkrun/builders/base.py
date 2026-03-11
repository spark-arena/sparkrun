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

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate builder-specific recipe fields."""
        return []

    def __repr__(self) -> str:
        return "%s(builder_name=%r)" % (self.__class__.__name__, self.builder_name)
