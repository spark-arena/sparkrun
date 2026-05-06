"""Resumable benchmark run state.

Persists benchmark progress in ``~/.cache/sparkrun/benchmarks/<benchmark_id>/``
so that interrupted runs can be resumed with ``sparkrun benchmark resume <id>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sparkrun.core.config import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_cache_dir(cache_dir: str | None) -> str:
    if cache_dir is None:
        return str(DEFAULT_CACHE_DIR)
    return cache_dir


def derive_benchmark_id(
    cluster_id: str,
    framework: str,
    profile: str | None,
    base_args: dict[str, Any],
    schedule: list[dict[str, Any]] | None,
) -> str:
    """Stable ID derived from canonical-JSON of inputs. Returns ``'bench_<12hex>'``."""
    payload = {
        "cluster_id": cluster_id,
        "framework": framework,
        "profile": profile,
        "base_args": base_args,
        "schedule": schedule,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return "bench_%s" % digest


@dataclass
class BenchmarkRunState:
    """Persistent progress state for a scheduled benchmark run."""

    benchmark_id: str
    cluster_id: str
    recipe_qualified_name: str
    framework: str
    profile: str | None
    base_args: dict[str, Any]
    schedule: list[dict[str, Any]]  # raw schedule_entry dicts in order
    completed_indices: list[int] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)
    crash_count: int = 0
    session_count: int = 0
    sessions: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)  # arena uses for submission_id, etc.
    created_at: str = ""  # ISO-8601 UTC
    updated_at: str = ""  # ISO-8601 UTC

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def state_dir(self, cache_dir: str | None = None) -> Path:
        """Return ``~/.cache/sparkrun/benchmarks/<benchmark_id>/``."""
        return Path(_resolve_cache_dir(cache_dir)) / "benchmarks" / self.benchmark_id

    def runs_dir(self, cache_dir: str | None = None) -> Path:
        """Return the per-run artefact directory (``state_dir / "runs"``)."""
        return self.state_dir(cache_dir) / "runs"

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, cache_dir: str | None = None) -> Path:
        """Atomically persist state to ``state_dir/state.yaml``.

        Sets ``created_at`` on first save; always updates ``updated_at``.
        Returns the path to the written file.
        """
        now = _now_iso()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

        sdir = self.state_dir(cache_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        self.runs_dir(cache_dir).mkdir(parents=True, exist_ok=True)

        state_path = sdir / "state.yaml"
        tmp_path = sdir / "state.yaml.tmp"

        data = asdict(self)
        with open(tmp_path, "w") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)

        tmp_path.replace(state_path)
        logger.debug("Saved benchmark run state to %s", state_path)
        return state_path

    @classmethod
    def load(cls, benchmark_id: str, cache_dir: str | None = None) -> "BenchmarkRunState | None":
        """Load state from disk. Returns ``None`` if no state file exists."""
        sdir = Path(_resolve_cache_dir(cache_dir)) / "benchmarks" / benchmark_id
        state_path = sdir / "state.yaml"
        if not state_path.exists():
            return None
        try:
            with open(state_path) as fh:
                data = yaml.safe_load(fh)
            if not data:
                return None
            return cls(**data)
        except Exception:
            logger.debug("Failed to load benchmark run state for %s", benchmark_id, exc_info=True)
            return None

    # -------------------------------------------------------------------------
    # Progress tracking
    # -------------------------------------------------------------------------

    def mark_started(self, idx: int, pid: int | None = None) -> None:
        """Record that task *idx* has started (optionally with process *pid*)."""
        logger.debug("Benchmark %s: task %d started (pid=%s)", self.benchmark_id, idx, pid)

    def mark_completed(self, idx: int) -> None:
        """Record task *idx* as completed.

        Deduplicates — safe to call more than once.  Also removes *idx* from
        ``failed_indices`` if present.
        """
        if idx not in self.completed_indices:
            self.completed_indices.append(idx)
        if idx in self.failed_indices:
            self.failed_indices.remove(idx)

    def mark_failed(self, idx: int, error: str | None = None) -> None:
        """Record task *idx* as failed for this session."""
        if idx not in self.failed_indices:
            self.failed_indices.append(idx)
        logger.debug("Benchmark %s: task %d failed — %s", self.benchmark_id, idx, error or "no detail")

    def mark_session_started(self) -> None:
        """Increment ``session_count`` and append a new sessions entry."""
        self.session_count += 1
        self.sessions.append({"session": self.session_count, "started_at": _now_iso(), "status": "running"})

    def mark_session_ended(self, status: str) -> None:
        """Update the last sessions entry with *status* and end timestamp."""
        if self.sessions:
            self.sessions[-1]["ended_at"] = _now_iso()
            self.sessions[-1]["status"] = status

    def mark_crash(self) -> None:
        """Increment ``crash_count``."""
        self.crash_count += 1
        logger.debug("Benchmark %s: crash #%d recorded", self.benchmark_id, self.crash_count)

    # -------------------------------------------------------------------------
    # Scheduling helpers
    # -------------------------------------------------------------------------

    def next_pending(self, total_tasks: int) -> int | None:
        """Return the smallest index in ``[0, total_tasks)`` not yet completed.

        Failed indices from previous sessions are retried; failed indices in the
        *current* session are skipped to avoid tight crash loops.  Retry
        semantics can be refined in a later iteration — for now this returns
        the smallest idx not in ``completed_indices``.
        """
        for idx in range(total_tasks):
            if idx not in self.completed_indices:
                return idx
        return None

    def is_complete(self, total_tasks: int) -> bool:
        """Return ``True`` when every task in ``[0, total_tasks)`` is completed."""
        return all(idx in self.completed_indices for idx in range(total_tasks))
