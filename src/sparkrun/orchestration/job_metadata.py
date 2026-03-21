"""Job and cluster metadata storage.

Persists cluster_id → recipe mapping in ``~/.cache/sparkrun/jobs/`` so
``cluster status`` and other commands can display recipe info for
running clusters.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


def generate_cluster_id(recipe: "Recipe", hosts: list[str], overrides: dict | None = None) -> str:
    """Deterministic cluster identifier from recipe, host set, and overrides.

    Hashes: runtime + model + sorted hosts + port + served_model_name.
    Port and served_model_name are resolved from overrides -> recipe defaults
    so that two instances of the same model on different ports get distinct IDs.
    """
    # Resolve effective port
    port = None
    if overrides:
        port = overrides.get("port")
    if port is None and recipe.defaults:
        port = recipe.defaults.get("port")

    # Resolve effective served_model_name
    served_name = None
    if overrides:
        served_name = overrides.get("served_model_name")
    if served_name is None and recipe.defaults:
        served_name = recipe.defaults.get("served_model_name")

    parts = [recipe.runtime, recipe.model] + sorted(hosts)
    if port is not None:
        parts.append("port=%s" % port)
    if served_name is not None:
        parts.append("name=%s" % served_name)
    key = "\0".join(parts)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return "sparkrun_%s" % digest


def save_job_metadata(
    cluster_id: str,
    recipe: "Recipe",
    hosts: list[str],
    overrides: dict | None = None,
    cache_dir: str | None = None,
    ib_ip_map: dict[str, str] | None = None,
    mgmt_ip_map: dict[str, str] | None = None,
    recipe_ref: str | None = None,
    runtime_info: dict[str, str] | None = None,
) -> None:
    """Persist job metadata so ``cluster status`` can display recipe info.

    Writes a small YAML file to ``{cache_dir}/jobs/{hash}.yaml`` where
    *hash* is the 12-char hex portion of *cluster_id*.
    """
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    jobs_dir = Path(cache_dir) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    tp = None
    if overrides:
        tp = overrides.get("tensor_parallel")
    if tp is None and recipe.defaults:
        tp = recipe.defaults.get("tensor_parallel")

    meta = {
        "cluster_id": cluster_id,
        "recipe": recipe.qualified_name,
        "model": recipe.model,
        "runtime": recipe.runtime,
        "hosts": hosts,
    }
    if recipe_ref:
        meta["recipe_ref"] = recipe_ref
    if tp is not None:
        meta["tensor_parallel"] = int(tp)
    # Persist port for proxy discovery
    port = None
    if overrides:
        port = overrides.get("port")
    if port is None and recipe.defaults:
        port = recipe.defaults.get("port")
    if port is not None:
        meta["port"] = int(port)

    # Persist served_model_name for proxy discovery
    served_name = None
    if overrides:
        served_name = overrides.get("served_model_name")
    if served_name is None and recipe.defaults:
        served_name = recipe.defaults.get("served_model_name")
    if served_name is not None:
        meta["served_model_name"] = str(served_name)

    if ib_ip_map:
        meta["ib_ip_map"] = ib_ip_map
    if mgmt_ip_map:
        meta["mgmt_ip_map"] = mgmt_ip_map
    if runtime_info:
        meta["runtime_info"] = runtime_info

    meta_path = jobs_dir / f"{digest}.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
    logger.debug("Saved job metadata to %s", meta_path)


def remove_job_metadata(cluster_id: str, cache_dir: str | None = None) -> None:
    """Delete the cached job metadata file for a cluster_id.

    No-op if the file does not exist.
    """
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    meta_path.unlink(missing_ok=True)
    logger.debug("Removed job metadata %s", meta_path)


def load_job_metadata(cluster_id: str, cache_dir: str | None = None) -> dict | None:
    """Load job metadata for a cluster_id.  Returns ``None`` if not found."""
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    if not meta_path.exists():
        return None
    try:
        from sparkrun.utils import load_yaml
        data = load_yaml(meta_path)
        return data or None
    except Exception:
        logger.debug("Failed to load job metadata for %s", cluster_id, exc_info=True)
        return None
