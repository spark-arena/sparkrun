"""Tests for intent-id-based benchmark resume.

Validates that benchmark identity is keyed on the recipe *intent* (the
deterministic half of a sparkrun cluster_id), not the per-launch placement
token, so that relaunches of the same logical workload can resume in-place.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sparkrun.benchmarking.run_state import BenchmarkRunState, derive_benchmark_id
from sparkrun.orchestration.job_metadata import (
    generate_cluster_id,
    generate_intent_id,
    generate_placement_token,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent_hex(seed: int = 0) -> str:
    """Deterministic 16-hex intent_id for tests."""
    return ("%016x" % (0xDEADBEEFCAFEBABE ^ seed))[:16]


def _placement_hex(seed: int = 0) -> str:
    """Deterministic 12-hex placement_token for tests."""
    return ("%012x" % (0xABCDEF012345 ^ seed))[:12]


# ---------------------------------------------------------------------------
# derive_benchmark_id is intent-id-based
# ---------------------------------------------------------------------------


def test_derive_benchmark_id_same_intent_different_placement_yields_same_id():
    """Two cluster_ids sharing an intent but differing in placement token must
    produce the same benchmark_id (so resume works across relaunches).
    """
    intent = _intent_hex(1)
    cluster_id_a = generate_cluster_id(intent, _placement_hex(1))
    cluster_id_b = generate_cluster_id(intent, _placement_hex(2))
    assert cluster_id_a != cluster_id_b

    id_a = derive_benchmark_id(cluster_id_a, "llama-benchy", "default", {"pp": [2048]}, None)
    id_b = derive_benchmark_id(cluster_id_b, "llama-benchy", "default", {"pp": [2048]}, None)
    assert id_a == id_b, "benchmark_id should be stable across placement-token changes"


def test_derive_benchmark_id_different_intents_yield_different_ids():
    """Two cluster_ids with different intents (different recipes / parallelism /
    port / served_model_name / runtime) must produce different benchmark_ids.
    """
    cluster_id_a = generate_cluster_id(_intent_hex(1), _placement_hex(1))
    cluster_id_b = generate_cluster_id(_intent_hex(2), _placement_hex(1))

    id_a = derive_benchmark_id(cluster_id_a, "llama-benchy", "default", {}, None)
    id_b = derive_benchmark_id(cluster_id_b, "llama-benchy", "default", {}, None)
    assert id_a != id_b


def test_derive_benchmark_id_malformed_cluster_id_does_not_crash(caplog):
    """Old/malformed cluster_ids must be handled gracefully — hashing the
    verbatim string, and emitting a debug log so the operator can diagnose
    'why is my benchmark not resuming?'.
    """
    with caplog.at_level(logging.DEBUG, logger="sparkrun.benchmarking.run_state"):
        bid = derive_benchmark_id("legacy-style-cluster-id", "llama-benchy", "default", {}, None)

    assert bid.startswith("bench_")
    # The benchmark_id should differ from one keyed on a real intent, since
    # the malformed value gets hashed verbatim instead of via parse_cluster_id.
    cluster_id = generate_cluster_id(_intent_hex(7), _placement_hex(7))
    canonical_bid = derive_benchmark_id(cluster_id, "llama-benchy", "default", {}, None)
    assert bid != canonical_bid


# ---------------------------------------------------------------------------
# BenchmarkRunState.intent_id derivation
# ---------------------------------------------------------------------------


def test_benchmark_run_state_derives_intent_id_from_cluster_id():
    """``intent_id`` is auto-populated by ``__post_init__`` when cluster_id parses."""
    intent = _intent_hex(3)
    cluster_id = generate_cluster_id(intent, _placement_hex(3))

    state = BenchmarkRunState(
        benchmark_id="bench_aabbccddeeff",
        cluster_id=cluster_id,
        recipe_qualified_name="@registry/my-recipe",
        framework="llama-benchy",
        profile=None,
        base_args={},
        schedule=[],
    )
    assert state.intent_id == intent


def test_benchmark_run_state_handles_legacy_unparseable_cluster_id():
    """Legacy or malformed cluster_ids must not crash construction; intent_id
    is left empty so the operator can see 'this state cannot be resumed'.
    """
    state = BenchmarkRunState(
        benchmark_id="bench_aabbccddeeff",
        cluster_id="legacy-cluster-id",
        recipe_qualified_name="@registry/my-recipe",
        framework="llama-benchy",
        profile=None,
        base_args={},
        schedule=[],
    )
    assert state.intent_id == ""


def test_benchmark_run_state_round_trips_intent_id(tmp_path: Path):
    """Saved state round-trips intent_id through YAML."""
    intent = _intent_hex(5)
    cluster_id = generate_cluster_id(intent, _placement_hex(5))

    state = BenchmarkRunState(
        benchmark_id="bench_aabbccddeeff",
        cluster_id=cluster_id,
        recipe_qualified_name="@registry/my-recipe",
        framework="llama-benchy",
        profile=None,
        base_args={"pp": [2048]},
        schedule=[{"depth": 0, "concurrency": 1}],
    )
    state.save(str(tmp_path))

    loaded = BenchmarkRunState.load(state.benchmark_id, str(tmp_path))
    assert loaded is not None
    assert loaded.intent_id == intent
    assert loaded.cluster_id == cluster_id


def test_benchmark_run_state_load_derives_intent_id_for_old_state_files(tmp_path: Path):
    """An old state file written before ``intent_id`` existed loads cleanly and
    backfills the field from ``cluster_id``.
    """
    intent = _intent_hex(9)
    cluster_id = generate_cluster_id(intent, _placement_hex(9))

    # Simulate an old saved state by writing YAML *without* an intent_id field.
    import yaml

    legacy_data = {
        "benchmark_id": "bench_legacy00bead",
        "cluster_id": cluster_id,
        "recipe_qualified_name": "@registry/my-recipe",
        "framework": "llama-benchy",
        "profile": None,
        "base_args": {},
        "schedule": [],
        "completed_indices": [],
        "failed_indices": [],
        "crash_count": 0,
        "session_count": 0,
        "sessions": [],
        "extras": {},
        "created_at": "",
        "updated_at": "",
    }
    sdir = tmp_path / "benchmarks" / legacy_data["benchmark_id"]
    sdir.mkdir(parents=True)
    (sdir / "state.yaml").write_text(yaml.safe_dump(legacy_data))

    loaded = BenchmarkRunState.load(legacy_data["benchmark_id"], str(tmp_path))
    assert loaded is not None
    # intent_id derived from cluster_id by __post_init__
    assert loaded.intent_id == intent


# ---------------------------------------------------------------------------
# Resume semantics: identity follows intent, not placement
# ---------------------------------------------------------------------------


def test_resume_matches_when_only_placement_token_changes():
    """The two passes share an intent_id → derive_benchmark_id matches → the
    state loaded from disk under the OLD cluster_id is a valid resume target
    for the NEW launch's cluster_id.
    """
    intent = _intent_hex(11)
    old_cluster = generate_cluster_id(intent, _placement_hex(11))
    new_cluster = generate_cluster_id(intent, _placement_hex(12))

    bench_id_old = derive_benchmark_id(old_cluster, "llama-benchy", "default", {"pp": [2048]}, None)
    bench_id_new = derive_benchmark_id(new_cluster, "llama-benchy", "default", {"pp": [2048]}, None)
    assert bench_id_old == bench_id_new


def test_resume_does_not_match_when_intents_differ():
    """A relaunch with a different recipe / port / parallelism produces a
    different intent_id, so the benchmark_id differs and we do NOT resume.
    """
    cluster_a = generate_cluster_id(_intent_hex(20), _placement_hex(20))
    cluster_b = generate_cluster_id(_intent_hex(21), _placement_hex(20))

    id_a = derive_benchmark_id(cluster_a, "llama-benchy", "default", {"pp": [2048]}, None)
    id_b = derive_benchmark_id(cluster_b, "llama-benchy", "default", {"pp": [2048]}, None)
    assert id_a != id_b


# ---------------------------------------------------------------------------
# Real intent generation (sanity-check the canonical path)
# ---------------------------------------------------------------------------


class _StubRecipe:
    """Minimal recipe object that satisfies ``generate_intent_id`` inputs."""

    runtime = "vllm-distributed"
    model = "org/model"
    defaults = {"port": 8000}


def test_generated_intent_id_is_stable_across_invocations():
    """Belt-and-suspenders: ``generate_intent_id`` is deterministic given
    identical (recipe, overrides) — so two launches of the same workload
    really do produce the same intent_id."""
    recipe = _StubRecipe()
    intent_a = generate_intent_id(recipe, overrides={"port": 8000})
    intent_b = generate_intent_id(recipe, overrides={"port": 8000})
    assert intent_a == intent_b
    # And differs from a different port
    intent_c = generate_intent_id(recipe, overrides={"port": 8001})
    assert intent_a != intent_c


def test_placement_tokens_are_random_per_launch():
    """``generate_placement_token`` returns a fresh token per call — without
    this, the placement-token-stability concern would not arise.
    """
    seen = {generate_placement_token() for _ in range(20)}
    assert len(seen) == 20
