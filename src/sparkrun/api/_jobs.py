"""``sparkrun.api.list_jobs`` — enumerate persisted job metadata.

Walks ``~/.cache/sparkrun/jobs/*.yaml`` (or a caller-supplied
``cache_dir``) and surfaces each as a :class:`~sparkrun.api.JobInfo`.
Stale entries whose YAML fails to parse are logged at debug and
skipped — the function never raises on a single bad file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.api._models import JobInfo

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def list_jobs(
    *,
    cache_dir: str | Path | None = None,
    sctx: "SparkrunContext | None" = None,
) -> list[JobInfo]:
    """Return a list of :class:`JobInfo` for every persisted job metadata file.

    Args:
        cache_dir: Override for the sparkrun cache root.  Takes
            precedence when set.  Otherwise falls back to
            ``sctx.config.cache_dir`` (when *sctx* is provided), then
            to :data:`sparkrun.core.config.DEFAULT_CACHE_DIR`.
        sctx: Optional shared :class:`SparkrunContext`.

    Returns:
        :class:`JobInfo` entries sorted by ``started_at`` descending
        (most recent first); entries without a timestamp come last,
        ordered by ``cluster_id``.
    """
    if cache_dir is None and sctx is not None:
        try:
            cache_dir = sctx.config.cache_dir
        except Exception:
            cache_dir = None
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR

        cache_dir = DEFAULT_CACHE_DIR

    jobs_dir = Path(cache_dir) / "jobs"
    if not jobs_dir.is_dir():
        return []

    entries: list[JobInfo] = []
    for meta_path in jobs_dir.glob("*.yaml"):
        info = _job_info_from_file(meta_path)
        if info is not None:
            entries.append(info)

    # Stable order: most-recent first, untimed jobs last.
    def _sort_key(j: JobInfo):
        return (j.started_at is None, -(j.started_at or 0.0), j.cluster_id)

    entries.sort(key=_sort_key)
    return entries


def _job_info_from_file(meta_path: Path) -> JobInfo | None:
    """Load one job metadata YAML and return a :class:`JobInfo`, or ``None`` on failure."""
    try:
        from sparkrun.utils import load_yaml

        data = load_yaml(meta_path) or {}
    except Exception:
        logger.debug("list_jobs: failed to load %s", meta_path, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None

    cluster_id = data.get("cluster_id") or _cluster_id_from_filename(meta_path)
    if not cluster_id:
        return None

    hosts_raw = data.get("hosts") or ()
    hosts = tuple(str(h) for h in hosts_raw) if isinstance(hosts_raw, (list, tuple)) else ()

    started_raw = data.get("started_at")
    started_at: float | None
    try:
        started_at = float(started_raw) if started_raw is not None else None
    except (TypeError, ValueError):
        started_at = None

    # Decompose the cluster_id when the metadata didn't already record
    # intent_id / placement_token (e.g. a job metadata file written by
    # an older tool that doesn't persist these keys).  Non-canonical
    # cluster_ids surface as ``None`` on both fields — a data-quality
    # signal that callers can detect via :class:`JobInfo`.
    intent_id = data.get("intent_id")
    placement_token = data.get("placement_token")
    if intent_id is None or placement_token is None:
        try:
            from sparkrun.orchestration.job_metadata import parse_cluster_id

            parsed_intent, parsed_token = parse_cluster_id(str(cluster_id))
            if intent_id is None:
                intent_id = parsed_intent
            if placement_token is None:
                placement_token = parsed_token
        except ValueError:
            pass

    return JobInfo(
        cluster_id=str(cluster_id),
        intent_id=intent_id,
        placement_token=placement_token,
        recipe=data.get("recipe"),
        runtime=data.get("runtime"),
        hosts=hosts,
        started_at=started_at,
        metadata=dict(data),
    )


def _cluster_id_from_filename(meta_path: Path) -> str:
    """Recover ``sparkrun_<digest>`` from a metadata filename (back-compat)."""
    stem = meta_path.stem
    return ("sparkrun_%s" % stem) if not stem.startswith("sparkrun_") else stem


__all__ = ["list_jobs"]
