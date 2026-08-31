"""Tests for ``sparkrun.orchestration.job_metadata`` — backends persistence (A1)
plus identifier-model coverage (intent_id / placement_token split)."""

from __future__ import annotations

import yaml

import time

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


def test_save_job_metadata_is_owner_only(tmp_path: Path, mock_recipe):
    """S2: metadata may carry the upstream api_key, so the dir is 0700 and the
    file 0600 — never a umask-default world/group-readable window."""
    import stat

    intent = "a" * INTENT_ID_LEN
    token = "d" * PLACEMENT_TOKEN_LEN
    cid = "sparkrun_%s_%s" % (intent, token)
    save_job_metadata(cid, mock_recipe, ["h1"], cache_dir=str(tmp_path))
    meta_path = tmp_path / "jobs" / ("%s_%s.yaml" % (intent, token))
    assert stat.S_IMODE(meta_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "jobs").stat().st_mode) == 0o700


def test_save_job_metadata_refuses_symlinked_target(tmp_path: Path, mock_recipe):
    """S2: a pre-planted symlink at the metadata path is refused (O_NOFOLLOW),
    so a secret-bearing write is never redirected through another user's link."""
    intent = "a" * INTENT_ID_LEN
    token = "e" * PLACEMENT_TOKEN_LEN
    cid = "sparkrun_%s_%s" % (intent, token)
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")
    (jobs / ("%s_%s.yaml" % (intent, token))).symlink_to(victim)

    with pytest.raises(OSError):
        save_job_metadata(cid, mock_recipe, ["h1"], cache_dir=str(tmp_path))
    assert victim.read_text() == "untouched"  # write was not followed through


def test_save_job_metadata_writes_where_o_nofollow_is_unavailable(tmp_path: Path, mock_recipe, monkeypatch):
    """A Windows control node has no ``os.O_NOFOLLOW``.

    Naming it directly raised AttributeError before the launcher's broad
    ``except`` swallowed it, so a Windows machine wrote *no* job metadata for
    the jobs it launched — and then `logs` and `stop` could not resolve their
    hosts from the cluster id.  The symlink hardening is POSIX-only by nature;
    losing it must not cost us the write itself.
    """
    import os

    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    intent = "b" * INTENT_ID_LEN
    token = "f" * PLACEMENT_TOKEN_LEN
    cid = "sparkrun_%s_%s" % (intent, token)
    save_job_metadata(cid, mock_recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    meta = load_job_metadata(cid, cache_dir=str(tmp_path))
    assert meta is not None
    assert meta["hosts"] == ["h1", "h2"]


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
    def fake_cleanup(host_containers, ssh_kwargs=None, dry_run=False, max_workers=None):
        from sparkrun.orchestration.ssh import RemoteResult

        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=1", stderr="") for h in host_containers}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", fake_cleanup)

    result = api.stop(recipe=intent_recipe, hosts=("h1",), cache_dir=str(tmp_path))
    assert result.cluster_id == cid


# ---------------------------------------------------------------------------
# AmbiguousWorkload carries the cluster_ids attribute
# ---------------------------------------------------------------------------


def test_ambiguous_workload_carries_cluster_ids():
    import sparkrun.api as api

    err = api.AmbiguousWorkload("multiple matches", cluster_ids=["a", "b"])
    assert err.cluster_ids == ("a", "b")


# --------------------------------------------------------------------------
# parse_container_name — canonical container-name decomposition
# --------------------------------------------------------------------------


class TestParseContainerName:
    """The single source of truth for ``sparkrun_<intent>_<placement>[_<role>]``
    splitting.  Both ``query_cluster_status`` and the monitor TUI depend on
    this returning the full cluster_id (not the intent prefix) for distinct
    placements of the same recipe."""

    INTENT = "221f3a3a45d7fa4d"
    PLACE = "0123456789ab"

    def test_head_role(self):
        from sparkrun.orchestration.job_metadata import parse_container_name

        result = parse_container_name("sparkrun_%s_%s_head" % (self.INTENT, self.PLACE))
        assert result == ("sparkrun_%s_%s" % (self.INTENT, self.PLACE), "head")

    def test_node_role_with_index(self):
        from sparkrun.orchestration.job_metadata import parse_container_name

        result = parse_container_name("sparkrun_%s_%s_node_3" % (self.INTENT, self.PLACE))
        assert result == ("sparkrun_%s_%s" % (self.INTENT, self.PLACE), "node_3")

    def test_solo_shorthand(self):
        from sparkrun.orchestration.job_metadata import parse_container_name

        result = parse_container_name("sparkrun_%s_%s_solo" % (self.INTENT, self.PLACE))
        assert result == ("sparkrun_%s_%s" % (self.INTENT, self.PLACE), "solo")

    def test_no_role_suffix(self):
        from sparkrun.orchestration.job_metadata import parse_container_name

        # Cluster_id by itself (rare but legal) returns role="?".
        result = parse_container_name("sparkrun_%s_%s" % (self.INTENT, self.PLACE))
        assert result == ("sparkrun_%s_%s" % (self.INTENT, self.PLACE), "?")

    def test_two_workloads_same_intent_distinct_placements(self):
        """Critical regression: same intent + different placement → different cluster_ids."""
        from sparkrun.orchestration.job_metadata import parse_container_name

        place_a = "aabbccddeeff"
        place_b = "112233445566"
        cid_a, _ = parse_container_name("sparkrun_%s_%s_head" % (self.INTENT, place_a))
        cid_b, _ = parse_container_name("sparkrun_%s_%s_head" % (self.INTENT, place_b))
        assert cid_a != cid_b
        assert cid_a == "sparkrun_%s_%s" % (self.INTENT, place_a)
        assert cid_b == "sparkrun_%s_%s" % (self.INTENT, place_b)

    def test_unparseable_returns_none(self):
        from sparkrun.orchestration.job_metadata import parse_container_name

        assert parse_container_name("not-a-sparkrun-container") is None
        assert parse_container_name("sparkrun_short_head") is None
        # Wrong intent length (15 instead of 16).
        assert parse_container_name("sparkrun_%s_%s_head" % ("a" * 15, self.PLACE)) is None
        # Wrong placement length (11 instead of 12).
        assert parse_container_name("sparkrun_%s_%s_head" % (self.INTENT, "a" * 11)) is None


# --------------------------------------------------------------------------
# started_at — launch time, and the ordering that depends on it
# --------------------------------------------------------------------------


class TestStartedAt:
    """``list_jobs`` documents "most recent first"; nothing recorded a time.

    ``_job_info_from_file`` has always read ``started_at``, but
    ``save_job_metadata`` never wrote it — so every job resolved to ``None``
    and the sort key silently degraded to alphabetical by cluster_id.
    """

    def _recipe(self):
        from sparkrun.core.recipe import Recipe

        return Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})

    def test_save_records_launch_time(self, tmp_path):
        import time

        from sparkrun.orchestration.job_metadata import load_job_metadata, save_job_metadata

        before = time.time()
        cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
        save_job_metadata(cluster_id, self._recipe(), ["h1"], cache_dir=str(tmp_path))
        after = time.time()

        meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
        assert before <= meta["started_at"] <= after

    def test_list_jobs_reads_it(self, tmp_path):
        from sparkrun.api import list_jobs
        from sparkrun.orchestration.job_metadata import save_job_metadata

        save_job_metadata("sparkrun_aaaaaaaaaaaaaaaa_111111111111", self._recipe(), ["h1"], cache_dir=str(tmp_path))
        (job,) = list_jobs(cache_dir=str(tmp_path))
        assert job.started_at is not None

    def test_orders_most_recent_first(self, tmp_path, monkeypatch):
        from sparkrun.api import list_jobs
        from sparkrun.orchestration.job_metadata import save_job_metadata

        # Launch order is deliberately the *reverse* of alphabetical order, so
        # the assertion fails under the old (alphabetical) behaviour rather
        # than passing by coincidence.  cluster_ids are hex — `parse_cluster_id`
        # rejects anything else — so the two differ only in a/f.
        clock = iter([1_700_000_000.0, 1_700_000_500.0])
        monkeypatch.setattr("sparkrun.orchestration.job_metadata.time.time", lambda: next(clock))

        for cid in ("sparkrun_aaaaaaaaaaaaaaaa_111111111111", "sparkrun_ffffffffffffffff_222222222222"):
            save_job_metadata(cid, self._recipe(), ["h1"], cache_dir=str(tmp_path))

        ordered = [j.cluster_id for j in list_jobs(cache_dir=str(tmp_path))]
        assert ordered[0].startswith("sparkrun_ffff"), "newest launch must sort first, not the alphabetical winner"


class TestStartedAtBackfill:
    """Jobs written before the field existed must still order correctly.

    On a long-lived cache that is most of them, and leaving those as "no
    timestamp" reproduces the exact ordering bug the field was added to fix.
    """

    def _write(self, tmp_path, name: str, body: str, mtime: float | None = None):
        import os

        jobs = tmp_path / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        path = jobs / name
        path.write_text(body)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_missing_started_at_falls_back_to_mtime(self, tmp_path):
        from sparkrun.api import list_jobs

        self._write(
            tmp_path,
            "aaaaaaaaaaaaaaaa_111111111111.yaml",
            "cluster_id: sparkrun_aaaaaaaaaaaaaaaa_111111111111\nhosts: [h1]\n",
            mtime=1_700_000_000.0,
        )
        (job,) = list_jobs(cache_dir=str(tmp_path))
        assert job.started_at == 1_700_000_000.0

    def test_recorded_value_beats_mtime(self, tmp_path):
        """A rewrite (backup restore, cache rsync) moves mtime; the record doesn't."""
        from sparkrun.api import list_jobs

        self._write(
            tmp_path,
            "aaaaaaaaaaaaaaaa_111111111111.yaml",
            "cluster_id: sparkrun_aaaaaaaaaaaaaaaa_111111111111\nstarted_at: 1600000000.0\nhosts: [h1]\n",
            mtime=1_700_000_000.0,
        )
        (job,) = list_jobs(cache_dir=str(tmp_path))
        assert job.started_at == 1_600_000_000.0

    def test_unparseable_value_falls_back_to_mtime(self, tmp_path):
        from sparkrun.api import list_jobs

        self._write(
            tmp_path,
            "aaaaaaaaaaaaaaaa_111111111111.yaml",
            "cluster_id: sparkrun_aaaaaaaaaaaaaaaa_111111111111\nstarted_at: not-a-number\nhosts: [h1]\n",
            mtime=1_700_000_000.0,
        )
        (job,) = list_jobs(cache_dir=str(tmp_path))
        assert job.started_at == 1_700_000_000.0

    def test_mixed_recorded_and_backfilled_sort_together(self, tmp_path):
        """A cache mid-migration holds both shapes; one ordering must cover both."""
        from sparkrun.api import list_jobs

        self._write(
            tmp_path,
            "aaaaaaaaaaaaaaaa_111111111111.yaml",
            "cluster_id: sparkrun_aaaaaaaaaaaaaaaa_111111111111\nhosts: [h1]\n",
            mtime=1_600_000_000.0,  # old, backfilled from mtime
        )
        self._write(
            tmp_path,
            "bbbbbbbbbbbbbbbb_222222222222.yaml",
            "cluster_id: sparkrun_bbbbbbbbbbbbbbbb_222222222222\nstarted_at: 1700000000.0\nhosts: [h1]\n",
            mtime=1_500_000_000.0,  # mtime is older, but the record is newer
        )
        ordered = [j.cluster_id for j in list_jobs(cache_dir=str(tmp_path))]
        assert ordered == [
            "sparkrun_bbbbbbbbbbbbbbbb_222222222222",
            "sparkrun_aaaaaaaaaaaaaaaa_111111111111",
        ]


# --------------------------------------------------------------------------
# prune_job_metadata
# --------------------------------------------------------------------------


class TestPruneJobMetadata:
    """The cache is append-only, so without a prune it grows without bound.

    Keep = among the newest ``keep_per_intent`` for its intent AND younger
    than ``max_age_days``.  Everything else goes.
    """

    def _write(self, tmp_path, intent: str, token: str, age_days: float, recipe="r"):
        import os

        from sparkrun.core.recipe import Recipe
        from sparkrun.orchestration.job_metadata import save_job_metadata

        cid = "sparkrun_%s_%s" % (intent, token)
        recipe_obj = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": recipe})
        save_job_metadata(cid, recipe_obj, ["h1"], cache_dir=str(tmp_path))
        # Rewrite started_at so age is deterministic, and match mtime to it.
        path = tmp_path / "jobs" / ("%s_%s.yaml" % (intent, token))
        data = yaml.safe_load(path.read_text())
        stamp = time.time() - age_days * 86400
        data["started_at"] = stamp
        path.write_text(yaml.safe_dump(data))
        os.utime(path, (stamp, stamp))
        return cid

    def test_age_rule_prunes_even_the_only_job_of_an_intent(self, tmp_path):
        """Both conditions must hold to keep, so age alone is enough to prune.

        An "or" here would keep the newest entry of every intent ever launched,
        leaving a cache that never shrinks.
        """
        from sparkrun.orchestration.job_metadata import prune_job_metadata

        fresh = self._write(tmp_path, "a" * 16, "1" * 12, age_days=1)
        old = self._write(tmp_path, "b" * 16, "2" * 12, age_days=90)

        removed = set(prune_job_metadata(cache_dir=str(tmp_path)))
        assert old in removed
        assert fresh not in removed

    def test_per_intent_rule_keeps_a_short_history(self, tmp_path):
        """Among *recent* jobs, each intent keeps its newest few relaunches."""
        from sparkrun.orchestration.job_metadata import prune_job_metadata

        jobs = [self._write(tmp_path, "c" * 16, "%012d" % i, age_days=i) for i in range(5)]

        removed = set(prune_job_metadata(cache_dir=str(tmp_path), keep_per_intent=3))
        assert removed == {jobs[3], jobs[4]}

    def test_per_intent_window_is_not_a_global_cap(self, tmp_path):
        """A rarely-run intent must not be erased by a busy one's relaunches."""
        from sparkrun.orchestration.job_metadata import prune_job_metadata

        busy = [self._write(tmp_path, "a" * 16, "%012d" % i, age_days=i) for i in range(5)]
        rare = self._write(tmp_path, "b" * 16, "9" * 12, age_days=6)

        removed = set(prune_job_metadata(cache_dir=str(tmp_path), keep_per_intent=2))
        assert rare not in removed
        assert set(busy[2:]) <= removed

    def test_never_deletes_a_running_workload(self, tmp_path):
        """Age is not a sufficient guard — a server can outlive the cutoff.

        Its metadata is load-bearing for stop/logs/proxy discovery, so deleting
        it would strand the deployment.
        """
        from sparkrun.orchestration.job_metadata import load_job_metadata, prune_job_metadata

        ancient = self._write(tmp_path, "d" * 16, "3" * 12, age_days=400)
        removed = prune_job_metadata(cache_dir=str(tmp_path), protected_cluster_ids={ancient})
        assert removed == []
        assert load_job_metadata(ancient, cache_dir=str(tmp_path)) is not None

    def test_dry_run_deletes_nothing(self, tmp_path):
        from sparkrun.orchestration.job_metadata import load_job_metadata, prune_job_metadata

        old = self._write(tmp_path, "e" * 16, "4" * 12, age_days=90)
        removed = prune_job_metadata(cache_dir=str(tmp_path), dry_run=True)
        assert removed == [old]
        assert load_job_metadata(old, cache_dir=str(tmp_path)) is not None

    def test_age_zero_disables_the_age_test(self, tmp_path):
        """Then keep_per_intent is the only rule — useful for a hard compaction."""
        from sparkrun.orchestration.job_metadata import prune_job_metadata

        recent = [self._write(tmp_path, "f" * 16, "%012d" % i, age_days=i) for i in range(3)]
        removed = prune_job_metadata(cache_dir=str(tmp_path), max_age_days=0, keep_per_intent=1)
        assert set(removed) == {recent[1], recent[2]}

    def test_empty_cache_is_a_noop(self, tmp_path):
        from sparkrun.orchestration.job_metadata import prune_job_metadata

        assert prune_job_metadata(cache_dir=str(tmp_path)) == []


class TestRunningSnapshot:
    def test_round_trip(self, tmp_path):
        from sparkrun.orchestration.job_metadata import load_running_snapshot, save_running_snapshot

        save_running_snapshot({"sparkrun_a_b"}, ["h1", "h2"], cache_dir=str(tmp_path))
        running, covered = load_running_snapshot(cache_dir=str(tmp_path))
        assert running == {"sparkrun_a_b"}
        assert covered == {"h1", "h2"}

    def test_absent_snapshot_is_none(self, tmp_path):
        from sparkrun.orchestration.job_metadata import load_running_snapshot

        assert load_running_snapshot(cache_dir=str(tmp_path)) is None

    def test_stale_snapshot_is_none(self, tmp_path):
        """Used to *hide* things, so it must expire rather than mislead."""
        from sparkrun.orchestration.job_metadata import load_running_snapshot, save_running_snapshot

        save_running_snapshot({"sparkrun_a_b"}, ["h1"], cache_dir=str(tmp_path))
        assert load_running_snapshot(cache_dir=str(tmp_path), max_age_s=-1) is None

    def test_corrupt_snapshot_is_none(self, tmp_path):
        from sparkrun.orchestration.job_metadata import RUNNING_SNAPSHOT_FILE, load_running_snapshot

        (tmp_path / RUNNING_SNAPSHOT_FILE).write_text("{not json")
        assert load_running_snapshot(cache_dir=str(tmp_path)) is None
