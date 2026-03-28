"""Unified session context for sparkrun CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.registry import RegistryManager


@dataclass
class SparkrunContext:
    """Single session context bundling SAF Variables and SparkrunConfig.

    Created lazily by ``_get_context()`` in CLI commands, replacing the
    repeated ``init_sparkrun()`` + ``SparkrunConfig()`` boilerplate.
    """

    variables: Variables
    config: SparkrunConfig
    verbose: bool = False

    @cached_property
    def registry_manager(self) -> RegistryManager:
        return self.config.get_registry_manager()

    @cached_property
    def cluster_manager(self) -> ClusterManager:
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import get_config_root

        return ClusterManager(get_config_root(self.variables))
