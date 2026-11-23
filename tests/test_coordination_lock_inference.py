"""Lock-host inference in ``core.coordination.operation_boundary`` (B13).

The decorator infers which hosts a mutating API call is about to touch, so it
can lock exactly those machines. Before this fix the inference skipped the
default-cluster step that ``api._resolve.resolve_cluster`` applies between
explicit hosts and ``config.default_hosts`` -- so with both a default cluster
and ``config.yaml``'s ``cluster.hosts`` configured, a flagless mutating call
(e.g. ``stop(recipe=...)``) locked ``default_hosts`` while the call itself
(via ``resolve_cluster``) actually acted on the default cluster's hosts: two
concurrent mutations holding disjoint locks over the same machines.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import yaml

from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.coordination import operation_boundary, reservation_keys


def _dummy_mutation(
    recipe=None,
    sctx=None,
    hosts=None,
    cluster=None,
    workload_id=None,
    cluster_id=None,
    options=None,
    plan=None,
    dry_run=False,
):
    return "ok"


def _capture_reserved_keys(monkeypatch):
    captured: dict = {}

    def fake_reserve_resources(root, keys):
        captured["keys"] = keys
        return nullcontext()

    monkeypatch.setattr("sparkrun.core.coordination.reserve_resources", fake_reserve_resources)
    return captured


def test_flagless_mutation_locks_the_default_cluster_not_default_hosts(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"cluster": {"hosts": ["confighost1", "confighost2"]}}))
    config = SparkrunConfig(config_path=config_path)

    mgr = ClusterManager(tmp_path)
    mgr.create("default-cluster", ["clusterhost1", "clusterhost2"])
    mgr.set_default("default-cluster")

    captured = _capture_reserved_keys(monkeypatch)
    decorated = operation_boundary(mutation=True)(_dummy_mutation)

    sctx = SimpleNamespace(config=config)
    assert decorated(recipe="some-recipe", sctx=sctx) == "ok"

    # Must match resolve_cluster's precedence: default cluster before
    # config.default_hosts, not the reverse.
    assert captured["keys"] == reservation_keys(["clusterhost1", "clusterhost2"])
    assert captured["keys"] != reservation_keys(["confighost1", "confighost2"])


def test_flagless_mutation_falls_back_to_default_hosts_without_a_default_cluster(tmp_path: Path, monkeypatch):
    """No default cluster set: config.default_hosts is still the fallback."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"cluster": {"hosts": ["confighost1", "confighost2"]}}))
    config = SparkrunConfig(config_path=config_path)

    captured = _capture_reserved_keys(monkeypatch)
    decorated = operation_boundary(mutation=True)(_dummy_mutation)

    sctx = SimpleNamespace(config=config)
    assert decorated(recipe="some-recipe", sctx=sctx) == "ok"

    assert captured["keys"] == reservation_keys(["confighost1", "confighost2"])


def test_plan_locks_candidate_hosts_not_just_the_chosen_host_list(tmp_path: Path, monkeypatch):
    """Eviction of a superseded deployment sweeps candidate_hosts, so the lock
    must cover that wider set, not only the narrower placement (host_list)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({}))
    config = SparkrunConfig(config_path=config_path)

    captured = _capture_reserved_keys(monkeypatch)
    decorated = operation_boundary(mutation=True)(_dummy_mutation)

    plan = SimpleNamespace(
        candidate_hosts=("cand1", "cand2", "cand3"),
        host_list=("cand1",),
        cluster=None,
    )
    sctx = SimpleNamespace(config=config)
    assert decorated(plan=plan, sctx=sctx) == "ok"

    assert captured["keys"] == reservation_keys(("cand1", "cand2", "cand3"))


def test_release_reservations_frees_hosts_while_the_operation_continues(tmp_path: Path):
    """A long non-mutating tail (foreground proxy supervision) must not keep
    the hosts its startup reserved: another launch on them has to succeed."""
    from sparkrun.core.coordination import release_reservations, reservations, reserve_resources

    keys = reservation_keys(["h1", "h2"])
    with reserve_resources(tmp_path, keys):
        assert {row["resource"] for row in reservations(tmp_path)} == keys
        release_reservations()
        assert reservations(tmp_path) == []
        # Released for this context too: a nested reservation really claims.
        with reserve_resources(tmp_path, reservation_keys(["h1"])):
            assert [row["resource"] for row in reservations(tmp_path)] == ["host:h1"]
        assert reservations(tmp_path) == []
    assert reservations(tmp_path) == []


def _hold_in_thread(root, keys, seconds):
    import threading

    from sparkrun.core.coordination import reserve_resources

    held = threading.Event()

    def holder():
        with reserve_resources(root, keys):
            held.set()
            threading.Event().wait(seconds)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(10)
    return thread


def test_busy_error_names_a_live_holder_and_its_command(tmp_path: Path):
    import pytest

    from sparkrun.core.coordination import ResourceBusy, reserve_resources

    thread = _hold_in_thread(tmp_path, reservation_keys(["h1", "h2"]), 1.0)
    try:
        with pytest.raises(ResourceBusy) as caught:
            with reserve_resources(tmp_path, reservation_keys(["h2"])):
                pass
    finally:
        thread.join()
    message = str(caught.value)
    assert "host:h2" in message and "host:h1" not in message  # only what conflicts
    assert "running pid %d" % __import__("os").getpid() in message
    assert "python" in message.lower()  # the holder's command line
    assert "sparkrun daemon resources list" in message
    assert [h["resource"] for h in caught.value.holders] == ["host:h2"]
    assert caught.value.holders[0]["alive"] is True


def test_busy_error_for_a_dead_owner_gives_the_reconcile_command(tmp_path: Path):
    import json
    import sqlite3

    import pytest

    from sparkrun.core.coordination import ResourceBusy, reserve_resources

    with reserve_resources(tmp_path, {"host:other"}):
        pass  # creates the table
    db = sqlite3.connect(tmp_path / "operations.sqlite3", isolation_level=None)
    db.execute(
        "INSERT INTO reservations VALUES(?,?,?)", ("deadbeef", "host:h1", json.dumps({"pid": 2**22 + 7, "boot": "x", "created": "1"}))
    )
    db.close()
    with pytest.raises(ResourceBusy) as caught:
        with reserve_resources(tmp_path, reservation_keys(["h1"])):
            pass
    assert "sparkrun daemon resources reconcile deadbeef --acknowledge-effects" in str(caught.value)
    assert caught.value.holders[0]["alive"] is False


def test_brief_controller_hold_is_waited_out(tmp_path: Path):
    from sparkrun.core.coordination import reservations, reserve_resources

    thread = _hold_in_thread(tmp_path, {"*"}, 0.5)
    try:
        with reserve_resources(tmp_path, reservation_keys(["h1"]), wait=5):
            assert [row["resource"] for row in reservations(tmp_path)] == ["host:h1"]
    finally:
        thread.join()


def test_controller_hold_outlasting_the_wait_still_refuses(tmp_path: Path):
    import time

    import pytest

    from sparkrun.core.coordination import ResourceBusy, reserve_resources

    thread = _hold_in_thread(tmp_path, {"*"}, 2.0)
    try:
        started = time.monotonic()
        with pytest.raises(ResourceBusy, match=r"\*: held by running pid"):
            with reserve_resources(tmp_path, reservation_keys(["h1"]), wait=0.3):
                pass
        assert time.monotonic() - started < 1.5
    finally:
        thread.join()


def test_host_conflicts_are_not_retried(tmp_path: Path):
    """Only `*` holds are brief; a host held by a launch is refused at once."""
    import time

    import pytest

    from sparkrun.core.coordination import ResourceBusy, reserve_resources

    thread = _hold_in_thread(tmp_path, reservation_keys(["h1"]), 1.0)
    try:
        started = time.monotonic()
        with pytest.raises(ResourceBusy):
            with reserve_resources(tmp_path, reservation_keys(["h1"]), wait=5):
                pass
        assert time.monotonic() - started < 0.5
    finally:
        thread.join()


def test_infer_reservation_hosts_resolves_named_then_default_cluster(tmp_path: Path):
    """The daemon worker shares this with operation_boundary, so a flagless
    daemon job locks the default cluster rather than every host ("*")."""
    from sparkrun.core.coordination import infer_reservation_hosts

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({}))
    config = SparkrunConfig(config_path=config_path)
    mgr = ClusterManager(tmp_path)
    mgr.create("named", ["n1"])
    mgr.create("dflt", ["d1", "d2"])

    assert not infer_reservation_hosts(config)[0]  # nothing configured: unknown, reserved as "*"
    mgr.set_default("dflt")
    hosts, cluster = infer_reservation_hosts(config)
    assert list(hosts) == ["d1", "d2"] and cluster.name == "dflt"
    hosts, cluster = infer_reservation_hosts(config, cluster="named")
    assert list(hosts) == ["n1"]


def test_resume_locks_the_benchmarks_recorded_hosts_not_the_default_cluster(tmp_path: Path, monkeypatch):
    """A resume acts on the hosts its benchmark ran on, which need not be the
    default cluster's: locking the default cluster left those hosts open."""
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({}))
    config = SparkrunConfig(config_path=config_path)
    mgr = ClusterManager(tmp_path)
    mgr.create("dflt", ["d1", "d2"])
    mgr.set_default("dflt")

    cluster_id = generate_cluster_id("a" * 16, "b" * 12)
    BenchmarkRunState(
        benchmark_id="bench_resume_hosts",
        cluster_id=cluster_id,
        recipe_qualified_name="r",
        framework="llama-benchy",
        profile=None,
        base_args={},
        schedule=[],
        host_list=["b1", "b2"],
    ).save(str(config.cache_dir))
    # The live launch (what resume reaches) was relaunched onto b3 as well.
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.load_job_metadata",
        lambda cid, cache_dir=None: {"hosts": ["b1", "b3"]} if cid == cluster_id else None,
    )

    def resume(benchmark_id, *, dry_run=False, sctx=None):
        return "ok"

    captured = _capture_reserved_keys(monkeypatch)
    assert operation_boundary(mutation=True)(resume)("bench_resume_hosts", sctx=SimpleNamespace(config=config)) == "ok"
    assert captured["keys"] == reservation_keys(["b1", "b3", "b2"])

    # Unknown state: nothing to go on, so everything is reserved.
    operation_boundary(mutation=True)(resume)("bench_missing", sctx=SimpleNamespace(config=config))
    assert captured["keys"] == {"*"}
