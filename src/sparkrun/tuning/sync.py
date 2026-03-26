"""Sync tuning configs from registries to local cache."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.registry import RegistryManager

logger = logging.getLogger(__name__)


def _resolve_tuning_runtime(runtime: str) -> str:
    """Normalize runtime name for tuning directory lookup.

    Maps runtime variants to their tuning directory names:
    - vllm-ray, vllm-distributed, eugr-vllm → vllm
    - sglang → sglang
    - llama-cpp → llama-cpp
    """
    if runtime in ("vllm-ray", "vllm-distributed", "eugr-vllm"):
        return "vllm"
    return runtime


def _get_local_tuning_dir(runtime: str) -> Path:
    """Return the local tuning config directory for a runtime.

    Uses the same paths as the existing tuning module.
    """
    from sparkrun.tuning._common import _get_tuning_dir
    from sparkrun.tuning.sglang import TUNING_CACHE_SUBDIR
    from sparkrun.tuning.vllm import VLLM_TUNING_CACHE_SUBDIR

    tuning_runtime = _resolve_tuning_runtime(runtime)
    if tuning_runtime == "sglang":
        return _get_tuning_dir(TUNING_CACHE_SUBDIR)
    elif tuning_runtime == "vllm":
        return _get_tuning_dir(VLLM_TUNING_CACHE_SUBDIR)
    else:
        return _get_tuning_dir("tuning/%s" % tuning_runtime)


def sync_registry_tuning(
    registry_manager: RegistryManager,
    runtime: str,
    dry_run: bool = False,
    registry_name: str | None = None,
) -> int:
    """Sync tuning configs from a registry to local cache.

    Copies registry-hosted tuning configs to the local cache directory
    so they can be auto-mounted in inference runs.  Both registry and
    local use the same flat layout (``tuning/<runtime>/...``), so the
    relative path within the runtime directory is preserved as-is.

    Only copies files that don't already exist locally (preserves local
    tuning results which take precedence).

    Args:
        registry_manager: Registry manager instance.
        runtime: Runtime name (e.g. "sglang", "vllm-ray").
        dry_run: If True, log what would be synced without copying.
        registry_name: If provided, only sync from this registry.
            When None, no tuning configs are synced (local recipes
            have no associated registry).

    Returns:
        Number of config files synced.
    """
    if registry_name is None:
        logger.debug("No source registry for recipe; skipping tuning sync")
        return 0

    tuning_runtime = _resolve_tuning_runtime(runtime)
    local_dir = _get_local_tuning_dir(runtime)

    configs = registry_manager.find_tuning_configs(tuning_runtime, registry_name=registry_name)
    if not configs:
        logger.debug("No registry tuning configs found for %s", tuning_runtime)
        return 0

    synced = 0
    for reg_name, config_path in configs:
        # Determine relative path within the runtime tuning dir.
        # Registry structure: tuning/<runtime>/triton_X_Y_Z/E=..json (flat)
        # Local structure:    ~/.cache/sparkrun/tuning/<runtime>/triton_X_Y_Z/E=..json
        parts = config_path.parts
        try:
            runtime_idx = parts.index(tuning_runtime)
            rel_parts = parts[runtime_idx + 1 :]
            rel_path = Path(*rel_parts) if rel_parts else Path(config_path.name)
        except (ValueError, TypeError):
            rel_path = Path(config_path.name)

        local_path = local_dir / rel_path

        # Skip if local file already exists (local tuning takes precedence)
        if local_path.exists():
            logger.debug("Skipping %s (local exists): %s", reg_name, rel_path)
            continue

        if dry_run:
            logger.info("[dry-run] Would sync %s from %s", rel_path, reg_name)
            synced += 1
            continue

        # Copy the config file
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, local_path)
        logger.info("Synced tuning config from %s: %s", reg_name, rel_path)
        synced += 1

    return synced
