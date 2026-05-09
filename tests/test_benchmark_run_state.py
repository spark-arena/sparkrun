"""Tests for BenchmarkRunState and derive_benchmark_id in run_state.py."""

from __future__ import annotations

import re
from pathlib import Path

from sparkrun.benchmarking.run_state import BenchmarkRunState, derive_benchmark_id


# ---------------------------------------------------------------------------
# derive_benchmark_id
# ---------------------------------------------------------------------------


def test_derive_benchmark_id_stable(tmp_path: Path):
    """Same inputs always produce the same benchmark id."""
    id1 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {"pp": [2048]}, None)
    id2 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {"pp": [2048]}, None)
    assert id1 == id2


def test_derive_benchmark_id_different_cluster(tmp_path: Path):
    """Different cluster_id produces a different benchmark id."""
    id1 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {"pp": [2048]}, None)
    id2 = derive_benchmark_id("cluster-xyz", "llama-benchy", "default", {"pp": [2048]}, None)
    assert id1 != id2


def test_derive_benchmark_id_different_framework(tmp_path: Path):
    """Different framework produces a different benchmark id."""
    id1 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {}, None)
    id2 = derive_benchmark_id("cluster-abc", "other-fw", "default", {}, None)
    assert id1 != id2


def test_derive_benchmark_id_different_profile(tmp_path: Path):
    """Different profile produces a different benchmark id."""
    id1 = derive_benchmark_id("cluster-abc", "llama-benchy", "profile-a", {}, None)
    id2 = derive_benchmark_id("cluster-abc", "llama-benchy", "profile-b", {}, None)
    assert id1 != id2


def test_derive_benchmark_id_different_base_args(tmp_path: Path):
    """Different base_args produces a different benchmark id."""
    id1 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {"pp": [2048]}, None)
    id2 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {"pp": [4096]}, None)
    assert id1 != id2


def test_derive_benchmark_id_different_schedule(tmp_path: Path):
    """Different schedule produces a different benchmark id."""
    id1 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {}, None)
    id2 = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {}, [{"depth": 0, "concurrency": 1}])
    assert id1 != id2


def test_derive_benchmark_id_format(tmp_path: Path):
    """ID must start with 'bench_' followed by exactly 12 hex characters."""
    bid = derive_benchmark_id("cluster-abc", "llama-benchy", "default", {}, None)
    assert re.fullmatch(r"bench_[0-9a-f]{12}", bid), f"Bad format: {bid!r}"


# ---------------------------------------------------------------------------
# BenchmarkRunState.save / load
# ---------------------------------------------------------------------------


def _make_state(bid: str = "bench_aabbccddeeff") -> BenchmarkRunState:
    return BenchmarkRunState(
        benchmark_id=bid,
        cluster_id="cluster-abc",
        recipe_qualified_name="@registry/my-recipe",
        framework="llama-benchy",
        profile="default",
        base_args={"pp": [2048], "depth": [0]},
        schedule=[{"depth": 0, "concurrency": 1}],
    )


def test_save_creates_state_yaml_and_runs_dir(tmp_path: Path):
    """save() creates <cache_dir>/benchmarks/<id>/state.yaml and runs/ directory."""
    state = _make_state()
    state.save(str(tmp_path))

    state_yaml = tmp_path / "benchmarks" / state.benchmark_id / "state.yaml"
    runs_dir = tmp_path / "benchmarks" / state.benchmark_id / "runs"

    assert state_yaml.exists(), "state.yaml not created"
    assert runs_dir.is_dir(), "runs/ directory not created"


def test_load_round_trips_state(tmp_path: Path):
    """load() restores every field from a previously saved state."""
    state = _make_state()
    state.completed_indices = [0]
    state.failed_indices = []
    state.crash_count = 2
    state.session_count = 1
    state.sessions = [{"session": 1, "started_at": "2024-01-01T00:00:00+00:00", "status": "completed"}]
    state.extras = {"submission_id": "sub-123"}
    state.save(str(tmp_path))

    loaded = BenchmarkRunState.load(state.benchmark_id, str(tmp_path))
    assert loaded is not None
    assert loaded.benchmark_id == state.benchmark_id
    assert loaded.cluster_id == state.cluster_id
    assert loaded.recipe_qualified_name == state.recipe_qualified_name
    assert loaded.framework == state.framework
    assert loaded.profile == state.profile
    assert loaded.base_args == state.base_args
    assert loaded.schedule == state.schedule
    assert loaded.completed_indices == [0]
    assert loaded.crash_count == 2
    assert loaded.session_count == 1
    assert loaded.extras == {"submission_id": "sub-123"}


def test_load_returns_none_when_missing(tmp_path: Path):
    """load() returns None if no state file exists for the given id."""
    result = BenchmarkRunState.load("bench_nonexistent00", str(tmp_path))
    assert result is None


def test_save_sets_created_at_once(tmp_path: Path):
    """created_at is set on first save and not overwritten on subsequent saves."""
    state = _make_state()
    state.save(str(tmp_path))
    first_created = state.created_at

    state.save(str(tmp_path))
    assert state.created_at == first_created


# ---------------------------------------------------------------------------
# mark_completed
# ---------------------------------------------------------------------------


def test_mark_completed_idempotent(tmp_path: Path):
    """mark_completed() called twice does not duplicate the index."""
    state = _make_state()
    state.mark_completed(0)
    state.mark_completed(0)
    assert state.completed_indices.count(0) == 1


def test_mark_completed_removes_from_failed(tmp_path: Path):
    """mark_completed() removes the index from failed_indices if present."""
    state = _make_state()
    state.failed_indices = [1]
    state.mark_completed(1)
    assert 1 not in state.failed_indices
    assert 1 in state.completed_indices


# ---------------------------------------------------------------------------
# mark_failed
# ---------------------------------------------------------------------------


def test_mark_failed_idempotent(tmp_path: Path):
    """mark_failed() called twice does not duplicate the index."""
    state = _make_state()
    state.mark_failed(2, "some error")
    state.mark_failed(2, "some error again")
    assert state.failed_indices.count(2) == 1


# ---------------------------------------------------------------------------
# mark_session_started / mark_session_ended
# ---------------------------------------------------------------------------


def test_mark_session_started_increments_count(tmp_path: Path):
    """mark_session_started() increments session_count and appends a sessions entry."""
    state = _make_state()
    assert state.session_count == 0
    state.mark_session_started()
    assert state.session_count == 1
    assert len(state.sessions) == 1
    assert state.sessions[0]["session"] == 1
    assert state.sessions[0]["status"] == "running"

    state.mark_session_started()
    assert state.session_count == 2
    assert len(state.sessions) == 2


def test_mark_session_ended_updates_last_entry(tmp_path: Path):
    """mark_session_ended() updates the last sessions entry status and ended_at."""
    state = _make_state()
    state.mark_session_started()
    state.mark_session_ended("completed")

    entry = state.sessions[-1]
    assert entry["status"] == "completed"
    assert "ended_at" in entry


# ---------------------------------------------------------------------------
# mark_crash
# ---------------------------------------------------------------------------


def test_mark_crash_increments_count(tmp_path: Path):
    """mark_crash() increments crash_count each time it is called."""
    state = _make_state()
    assert state.crash_count == 0
    state.mark_crash()
    assert state.crash_count == 1
    state.mark_crash()
    assert state.crash_count == 2


# ---------------------------------------------------------------------------
# next_pending / is_complete
# ---------------------------------------------------------------------------


def test_next_pending_returns_smallest_incomplete(tmp_path: Path):
    """next_pending() returns the smallest index not yet in completed_indices."""
    state = _make_state()
    state.completed_indices = [0, 2]
    assert state.next_pending(4) == 1


def test_next_pending_returns_none_when_all_done(tmp_path: Path):
    """next_pending() returns None when all indices are completed."""
    state = _make_state()
    state.completed_indices = [0, 1, 2]
    assert state.next_pending(3) is None


def test_is_complete_true_when_all_done(tmp_path: Path):
    """is_complete() returns True when all indices in [0, total) are completed."""
    state = _make_state()
    state.completed_indices = [0, 1, 2]
    assert state.is_complete(3) is True


def test_is_complete_false_when_partial(tmp_path: Path):
    """is_complete() returns False when some indices are not completed."""
    state = _make_state()
    state.completed_indices = [0, 2]
    assert state.is_complete(3) is False


def test_is_complete_false_when_none_done(tmp_path: Path):
    """is_complete() returns False when no tasks have been completed."""
    state = _make_state()
    assert state.is_complete(3) is False
