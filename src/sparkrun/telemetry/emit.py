"""Convenience emitters that keep telemetry failures isolated."""

from __future__ import annotations

from collections.abc import Mapping
import logging

from sparkrun.core.config import SparkrunConfig

from .benchmark import build_benchmark_event
from .client import send_event
from .events import build_run_event, build_setup_wizard_event, build_update_event

logger = logging.getLogger(__name__)


def emit_update_event(
    config: SparkrunConfig,
    *,
    command: str,
    old_version: str,
    new_version: str | None,
    upgraded: bool,
    self_upgrade_attempted: bool = True,
) -> None:
    """Emit a best-effort update event."""
    try:
        registries = config.get_registry_manager().list_registries()
        send_event(
            config,
            build_update_event(
                command=command,
                old_version=old_version,
                new_version=new_version,
                upgraded=upgraded,
                registries=registries,
                self_upgrade_attempted=self_upgrade_attempted,
            ),
        )
    except Exception:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        logger.debug("Failed to emit update telemetry", exc_info=True)


def emit_setup_wizard_event(
    config: SparkrunConfig,
    *,
    wizard_run_kind: str,
    results: Mapping[str, str | int | float | bool | None],
    cluster_node_count: int,
    dry_run: bool,
    cx7_detected: bool,
) -> None:
    """Emit a best-effort setup wizard event."""
    try:
        send_event(
            config,
            build_setup_wizard_event(
                wizard_run_kind=wizard_run_kind,
                results=results,
                cluster_node_count=cluster_node_count,
                dry_run=dry_run,
                cx7_detected=cx7_detected,
            ),
        )
    except Exception:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        logger.debug("Failed to emit setup wizard telemetry", exc_info=True)


def emit_run_telemetry(
    config: SparkrunConfig,
    *,
    result,
    recipe,
    cluster,
    options,
) -> None:
    try:
        send_event(config, build_run_event(result=result, recipe=recipe, cluster=cluster, options=options))
    except Exception:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        logger.debug("Failed to emit run telemetry", exc_info=True)


def emit_benchmark_telemetry(
    config: SparkrunConfig,
    *,
    result,
    options,
    recipe=None,
) -> None:
    try:
        send_event(config, build_benchmark_event(result=result, options=options, recipe=recipe))
    except Exception:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        logger.debug("Failed to emit benchmark telemetry", exc_info=True)
