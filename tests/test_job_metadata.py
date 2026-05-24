"""Tests for ``sparkrun.orchestration.job_metadata`` — backends persistence (A1)
plus identifier-model coverage (intent_id / placement_token split)."""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import pytest

from sparkrun.core.backend_select import BackendBundle
from sparkrun.orchestration.collectives import NcclBackend, RcclBackend
from sparkrun.orchestration.job_metadata import (
    INTENT_ID_LEN,
    PLACEMENT_TOKEN_LEN,
    derive_cluster_id,
    generate_cluster_id,
    generate_intent_id,
    generate_placement_token,
    is_cluster_id,
    load_job_metadata,
    parse_cluster_id,
    save_job_metadata,
)


def _make_cluster_id(intent_hex: str = "a", token_hex: str = "0") -> str:
    """Build a canonical cluster_id with single-char-repeated hex segments.

    Convenience for tests that need a syntactically-valid cluster_id
    but don't care about specific bytes.
    """
    return "sparkrun_%s_%s" % (intent_hex * INTENT_ID_LEN, token_hex * PLACEMENT_TOKEN_LEN)


@pytest.fixture
def mock_recipe():
    """Recipe stub with the attributes save_job_metadata reads."""
    r = mock.MagicMock()
    r.runtime = "vllm"
    r.model = "Qwen/Qwen3-1.7B"
    r.defaults = {"port": 8000}
    r.qualified_name = "test-recipe"
    r.executor = ""
    r.executor_config = None
    r.__getstate__ = mock.MagicMock(return_value={})
    return r


def test_save_job_metadata_persists_backends(tmp_path: Path, mock_recipe):
    """``backends`` kwarg is serialized to ``meta['backends']`` with
    ``{host: {vendor, backend}}`` shape."""
    cluster_id = _make_cluster_id("a", "0")
    hosts = ["nv-host", "amd-host"]
    backends = {
        "nv-host": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend()),
        "amd-host": BackendBundle(accelerator_vendor="amd", collective=RcclBackend()),
    }

    save_job_metadata(
        cluster_id,
        mock_recipe,
        hosts,
        cache_dir=str(tmp_path),
        backends=backends,
    )

    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    assert "backends" in meta
    assert meta["backends"] == {
        "nv-host": {"vendor": "nvidia", "backend": "nccl"},
        "amd-host": {"vendor": "amd", "backend": "rccl"},
    }


def test_save_job_metadata_omits_backends_when_empty(tmp_path: Path, mock_recipe):
    """Empty backends dict is omitted from persisted metadata."""
    cluster_id = _make_cluster_id("b", "1")
    save_job_metadata(
        cluster_id,
        mock_recipe,
        ["h1"],
        cache_dir=str(tmp_path),
        backends={},
    )
    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    assert "backends" not in meta


def test_save_job_metadata_backends_none_omitted(tmp_path: Path, mock_recipe):
    """backends=None (default) is omitted from persisted metadata."""
    cluster_id = _make_cluster_id("c", "2")
    save_job_metadata(
        cluster_id,
        mock_recipe,
        ["h1"],
        cache_dir=str(tmp_path),
    )
    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    assert "backends" not in meta


def test_save_job_metadata_backends_roundtrip(tmp_path: Path, mock_recipe):
    """Single-host NVIDIA backend roundtrips through YAML serialization."""
    cluster_id = _make_cluster_id("d", "3")
    backends = {
        "10.0.0.1": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend()),
    }
    save_job_metadata(
        cluster_id,
        mock_recipe,
        ["10.0.0.1"],
        cache_dir=str(tmp_path),
        backends=backends,
    )
    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    persisted = meta["backends"]["10.0.0.1"]
    # Schema: {vendor, backend} — names that survive readback unchanged.
    assert persisted["vendor"] == "nvidia"
    assert persisted["backend"] == "nccl"


# ---------------------------------------------------------------------------
# Identifier-model: generate_intent_id / generate_placement_token / parse
# ---------------------------------------------------------------------------


@pytest.fixture
def intent_recipe():
    """Bare-bones recipe stub for intent_id derivation tests."""
    r = mock.MagicMock()
    r.runtime = "vllm"
    r.model = "meta-llama/Llama-2-7b-hf"
    r.defaults = {}
    return r


def test_intent_id_is_deterministic(intent_recipe):
    """Same inputs produce the same intent_id."""
    assert generate_intent_id(intent_recipe) == generate_intent_id(intent_recipe)


def test_intent_id_format(intent_recipe):
    """intent_id is INTENT_ID_LEN lowercase-hex characters with no prefix."""
    intent_id = generate_intent_id(intent_recipe)
    assert re.fullmatch(r"[0-9a-f]{%d}" % INTENT_ID_LEN, intent_id)


def test_intent_id_ignores_hosts(intent_recipe):
    """Hosts are explicitly NOT hashed into intent_id — that's the whole point.

    Same recipe + parallelism + port → same intent_id, regardless of
    which hosts the scheduler picked at launch time.  This is what
    lets stop / logs find the workload after a load-aware scheduler
    placed it on a different host set than the user supplied.
    """
    # ``generate_intent_id`` does not accept a hosts arg; calling it
    # twice with the same recipe yields identical IDs by construction.
    assert generate_intent_id(intent_recipe) == generate_intent_id(intent_recipe)


def test_intent_id_changes_with_port(intent_recipe):
    a = generate_intent_id(intent_recipe, overrides={"port": 8000})
    b = generate_intent_id(intent_recipe, overrides={"port": 9000})
    assert a != b


def test_intent_id_changes_with_tp(intent_recipe):
    """Non-default tensor_parallel changes the intent."""
    base = generate_intent_id(intent_recipe)
    tp2 = generate_intent_id(intent_recipe, overrides={"tensor_parallel": 2})
    assert base != tp2


def test_intent_id_ignores_default_parallelism(intent_recipe):
    """tp=1, pp=1, etc. are equivalent to "not set" — match no-override case."""
    base = generate_intent_id(intent_recipe)
    tp1 = generate_intent_id(intent_recipe, overrides={"tensor_parallel": 1})
    assert base == tp1


def test_intent_id_hashes_all_parallelism_dimensions(intent_recipe):
    """Every PARALLELISM_KEYS dim distinguishes intent (not just tp)."""
    base = generate_intent_id(intent_recipe)
    ids = {base}
    for dim in ("tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel", "context_parallel"):
        ids.add(generate_intent_id(intent_recipe, overrides={dim: 2}))
    # Every parallelism dim produces a distinct id (5 dims + base = 6).
    assert len(ids) == 6


def test_placement_token_format():
    token = generate_placement_token()
    assert re.fullmatch(r"[0-9a-f]{%d}" % PLACEMENT_TOKEN_LEN, token)


def test_placement_token_is_unique():
    """Two calls produce different tokens (collision is astronomically unlikely)."""
    tokens = {generate_placement_token() for _ in range(8)}
    assert len(tokens) == 8


def test_generate_cluster_id_new_form_composes():
    intent = "a" * INTENT_ID_LEN
    token = "0" * PLACEMENT_TOKEN_LEN
    cid = generate_cluster_id(intent, token)
    assert cid == "sparkrun_%s_%s" % (intent, token)


def test_generate_cluster_id_rejects_bad_intent():
    with pytest.raises(ValueError):
        generate_cluster_id("notHex", "0" * PLACEMENT_TOKEN_LEN)


def test_generate_cluster_id_rejects_bad_token():
    with pytest.raises(ValueError):
        generate_cluster_id("a" * INTENT_ID_LEN, "tooshort")


def test_derive_cluster_id_is_deterministic(intent_recipe):
    """``derive_cluster_id(recipe, hosts)`` produces a deterministic
    ``(intent, host-derived token)`` cluster_id."""
    cid = derive_cluster_id(intent_recipe, ["10.0.0.1"])
    assert re.fullmatch(r"sparkrun_[0-9a-f]{%d}_[0-9a-f]{%d}" % (INTENT_ID_LEN, PLACEMENT_TOKEN_LEN), cid)
    # Same hosts → same cluster_id (deterministic).
    assert derive_cluster_id(intent_recipe, ["10.0.0.1"]) == cid


def test_derive_cluster_id_is_host_order_independent(intent_recipe):
    """Host ordering does not affect the derived cluster_id."""
    cid_a = derive_cluster_id(intent_recipe, ["10.0.0.1", "10.0.0.2"])
    cid_b = derive_cluster_id(intent_recipe, ["10.0.0.2", "10.0.0.1"])
    assert cid_a == cid_b


# ---------------------------------------------------------------------------
# parse_cluster_id / is_cluster_id
# ---------------------------------------------------------------------------


def test_parse_cluster_id_canonical_format():
    intent = "a" * INTENT_ID_LEN
    token = "0" * PLACEMENT_TOKEN_LEN
    cid = "sparkrun_%s_%s" % (intent, token)
    parsed_intent, parsed_token = parse_cluster_id(cid)
    assert parsed_intent == intent
    assert parsed_token == token


def test_parse_cluster_id_rejects_legacy_format():
    """Pre-0.3 single-segment ``sparkrun_<hex>`` IDs are no longer
    accepted — they raise :class:`ValueError`."""
    with pytest.raises(ValueError):
        parse_cluster_id("sparkrun_%s" % ("a" * INTENT_ID_LEN))


def test_parse_cluster_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_cluster_id("not-a-cluster-id")


def test_is_cluster_id_accepts_only_canonical_format():
    canonical = "sparkrun_%s_%s" % ("a" * INTENT_ID_LEN, "0" * PLACEMENT_TOKEN_LEN)
    legacy = "sparkrun_%s" % ("a" * INTENT_ID_LEN)
    assert is_cluster_id(canonical) is True
    assert is_cluster_id(legacy) is False
    assert is_cluster_id("nope") is False


# ---------------------------------------------------------------------------
# save_job_metadata persists intent_id + placement_token
# ---------------------------------------------------------------------------


def test_save_job_metadata_persists_intent_and_token(tmp_path: Path, mock_recipe):
    """The identifier components are written as separate metadata fields."""
    intent = "a" * INTENT_ID_LEN
    token = "0" * PLACEMENT_TOKEN_LEN
    cid = "sparkrun_%s_%s" % (intent, token)
    save_job_metadata(cid, mock_recipe, ["h1"], cache_dir=str(tmp_path))
    meta = load_job_metadata(cid, cache_dir=str(tmp_path))
    assert meta is not None
    assert meta["intent_id"] == intent
    assert meta["placement_token"] == token


def test_save_job_metadata_rejects_non_canonical_cluster_id(tmp_path: Path, mock_recipe):
    """Non-canonical cluster_ids raise :class:`ValueError` from parse_cluster_id."""
    with pytest.raises(ValueError):
        save_job_metadata("sparkrun_abc123abc123", mock_recipe, ["h1"], cache_dir=str(tmp_path))


def test_load_job_metadata_filename_roundtrip(tmp_path: Path, mock_recipe):
    """Canonical cluster_ids written as ``sparkrun_<intent>_<token>.yaml`` roundtrip."""
    intent = "a" * INTENT_ID_LEN
    token = "b" * PLACEMENT_TOKEN_LEN
    cid = "sparkrun_%s_%s" % (intent, token)
    save_job_metadata(cid, mock_recipe, ["h1"], cache_dir=str(tmp_path))
    # Filename is the digest (the part after ``sparkrun_``), unchanged.
    assert (tmp_path / "jobs" / ("%s_%s.yaml" % (intent, token))).exists()
    meta = load_job_metadata(cid, cache_dir=str(tmp_path))
    assert meta is not None
    assert meta["cluster_id"] == cid


# ---------------------------------------------------------------------------
# api.stop recipe path: status-driven discovery
# ---------------------------------------------------------------------------


def test_api_stop_recipe_path_raises_job_not_found_on_zero_matches(tmp_path, intent_recipe, monkeypatch):
    """No workloads running matching the intent → JobNotFound (not Ambiguous)."""
    import sparkrun.api as api
    from sparkrun.core.cluster_status import ClusterStatus

    # Stub executor.query_status to return an empty snapshot.
    def fake_query_status(self, hosts, **kw):
        return ClusterStatus(hosts=(), executor="docker")

    from sparkrun.orchestration.executors.docker import DockerExecutor

    monkeypatch.setattr(DockerExecutor, "query_status", fake_query_status)

    with pytest.raises(api.JobNotFound):
        api.stop(recipe=intent_recipe, hosts=("h1",), cache_dir=str(tmp_path))


def test_api_stop_recipe_path_raises_ambiguous_on_multiple_matches(tmp_path, intent_recipe, monkeypatch):
    """Two workloads with the same intent on different host sets → AmbiguousWorkload."""
    import sparkrun.api as api
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    intent_id = generate_intent_id(intent_recipe)

    cid_a = "sparkrun_%s_%s" % (intent_id, "a" * PLACEMENT_TOKEN_LEN)
    cid_b = "sparkrun_%s_%s" % (intent_id, "b" * PLACEMENT_TOKEN_LEN)

    def fake_query_status(self, hosts, **kw):
        return ClusterStatus(
            hosts=tuple(
                HostOccupancy(
                    host=h,
                    workloads=(
                        RunningWorkload(cluster_id=cid_a),
                        RunningWorkload(cluster_id=cid_b),
                    ),
                )
                for h in hosts
            ),
            executor="docker",
        )

    from sparkrun.orchestration.executors.docker import DockerExecutor

    monkeypatch.setattr(DockerExecutor, "query_status", fake_query_status)

    with pytest.raises(api.AmbiguousWorkload) as exc_info:
        api.stop(recipe=intent_recipe, hosts=("h1",), cache_dir=str(tmp_path))
    assert set(exc_info.value.cluster_ids) == {cid_a, cid_b}


def test_api_stop_recipe_path_succeeds_on_single_match(tmp_path, intent_recipe, monkeypatch):
    """Exactly one matching workload → status-driven discovery resolves it."""
    import sparkrun.api as api
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    intent_id = generate_intent_id(intent_recipe)
    cid = "sparkrun_%s_%s" % (intent_id, "0" * PLACEMENT_TOKEN_LEN)

    def fake_query_status(self, hosts, **kw):
        return ClusterStatus(
            hosts=tuple(HostOccupancy(host=h, workloads=(RunningWorkload(cluster_id=cid),)) for h in hosts),
            executor="docker",
        )

    from sparkrun.orchestration.executors.docker import DockerExecutor

    monkeypatch.setattr(DockerExecutor, "query_status", fake_query_status)
    # Stub cleanup to short-circuit the actual SSH dispatch.
    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers", lambda *a, **kw: None)

    result = api.stop(recipe=intent_recipe, hosts=("h1",), cache_dir=str(tmp_path))
    assert result.cluster_id == cid


# ---------------------------------------------------------------------------
# AmbiguousWorkload carries the cluster_ids attribute
# ---------------------------------------------------------------------------


def test_ambiguous_workload_carries_cluster_ids():
    import sparkrun.api as api

    err = api.AmbiguousWorkload("multiple matches", cluster_ids=["a", "b"])
    assert err.cluster_ids == ("a", "b")
