"""eugr-vllm runtime: extends VllmRayRuntime with eugr container builds and mods.

.. deprecated::
    Use ``vllm-ray`` runtime with ``builder: eugr`` instead.  The eugr-vllm
    runtime is retained for backward compatibility with v1 recipes that
    auto-resolve to ``eugr-vllm``.  It now delegates all build and mod
    logic to :class:`~sparkrun.builders.eugr.EugrBuilder`.
"""

from __future__ import annotations

import logging
import warnings
from logging import Logger
from typing import Any, TYPE_CHECKING

from scitrera_app_framework import Variables

from sparkrun.runtimes.vllm_ray import VllmRayRuntime

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


class EugrVllmRayRuntime(VllmRayRuntime):
    """eugr-vllm runtime extending native vLLM with eugr build and mod support.

    .. deprecated::
        Use ``vllm-ray`` runtime with ``builder: eugr`` instead.

    This runtime is now a thin wrapper around VllmRayRuntime that delegates
    image building and mod application to :class:`~sparkrun.builders.eugr.EugrBuilder`.
    It exists for backward compatibility with v1 recipes.
    """

    _v: Variables = None

    runtime_name = "eugr-vllm"
    default_image_prefix = ""  # eugr uses local builds

    def initialize(self, v: Variables, logger_arg: Logger) -> EugrVllmRayRuntime:
        """Initialize the eugr-vllm runtime plugin."""
        self._v = v
        return self

    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve container -- eugr images use plain names, not prefix:tag."""
        return recipe.container or "vllm-node"

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate eugr-vllm-specific recipe fields."""
        issues = super().validate_recipe(recipe)
        if not recipe.command:
            issues.append("[eugr-vllm] command template is recommended for eugr recipes")
        return issues

    def prepare(
            self,
            recipe: Recipe,
            hosts: list[str],
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            transfer_mode: str = "local",
    ) -> None:
        """Delegate to EugrBuilder for image building and mod injection.

        Issues a deprecation warning since users should migrate to
        ``vllm-ray`` runtime with ``builder: eugr``.
        """
        warnings.warn(
            "eugr-vllm runtime is deprecated. Use 'vllm-ray' runtime with 'builder: eugr' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from sparkrun.core.bootstrap import get_builder
        from sparkrun.orchestration.primitives import build_ssh_kwargs
        try:
            builder = get_builder("eugr", v=self._v)
        except ValueError:
            logger.warning("eugr builder not found; skipping prepare")
            return

        image = self.resolve_container(recipe)
        builder.prepare_image(
            image, recipe, hosts, config=config, dry_run=dry_run,
            transfer_mode=transfer_mode,
            ssh_kwargs=build_ssh_kwargs(config),
        )
