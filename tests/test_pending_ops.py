"""Tests for sparkrun.core.pending_ops lock liveness and ownership (C5).

Covers staleness via dead PID, PID-reuse guard via max-age pruning,
ownership-checked removal, and concurrent same-key locks not clobbering
each other.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sparkrun.core import pending_ops
from sparkrun.core.pending_ops import (
    LOCK_MAX_AGE_SECONDS,
    create_pending_op,
    list_pending_ops,
    pending_op,
    remove_pending_op,
    _lock_path,
    _pending_dir,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_create_records_host_token_and_pid(tmp_path):
    path = create_pending_op("clusterA", "model_download", cache_dir=str(tmp_path))
    info = _read(path)
    assert info["cluster_id"] == "clusterA"
    assert info["operation"] == "model_download"
    assert info["pid"] > 0
    assert info["token"]  # non-empty unique token
    assert "host" in info
    assert isinstance(info["started_at"], float)


def test_list_pending_ops_returns_live_lock(tmp_path):
    create_pending_op("c1", "image_pull", cache_dir=str(tmp_path))
    ops = list_pending_ops(cache_dir=str(tmp_path))
    assert len(ops) == 1
    assert ops[0]["operation"] == "image_pull"
    assert "elapsed_seconds" in ops[0]


def test_stale_dead_pid_is_pruned(tmp_path):
    """A lock owned by a dead PID (same host) is pruned on read."""
    path = create_pending_op("c1", "model_download", cache_dir=str(tmp_path))
    info = _read(path)
    # PID 0/negative or an impossible PID is treated as dead by os.kill.
    info["pid"] = 2**31 - 1  # almost certainly not a running PID
    path.write_text(json.dumps(info))

    ops = list_pending_ops(cache_dir=str(tmp_path))
    assert ops == []
    assert not path.exists()


def test_pid_reuse_guarded_by_max_age(tmp_path):
    """A very old lock is pruned even when its PID appears alive (PID reuse)."""
    path = create_pending_op("c1", "image_distribute", cache_dir=str(tmp_path))
    info = _read(path)
    # PID is *this* live process — os.kill(pid, 0) would say "alive" — but the
    # lock is older than the max age, so it must still be pruned.
    info["started_at"] = time.time() - (LOCK_MAX_AGE_SECONDS + 60)
    path.write_text(json.dumps(info))

    ops = list_pending_ops(cache_dir=str(tmp_path))
    assert ops == []
    assert not path.exists()


def test_other_host_lock_not_pruned_by_pid(tmp_path):
    """A fresh lock from another host is NOT pruned just because its PID isn't local."""
    path = create_pending_op("c1", "model_download", cache_dir=str(tmp_path))
    info = _read(path)
    info["host"] = "some-other-host-not-me"
    info["pid"] = 2**31 - 1  # dead locally, but irrelevant for a remote host
    info["started_at"] = time.time()  # fresh
    path.write_text(json.dumps(info))

    ops = list_pending_ops(cache_dir=str(tmp_path))
    assert len(ops) == 1  # survives: max-age is the only signal for remote hosts
    assert path.exists()


def test_other_host_lock_pruned_when_old(tmp_path):
    """A remote-host lock is still pruned once it exceeds max age."""
    path = create_pending_op("c1", "model_download", cache_dir=str(tmp_path))
    info = _read(path)
    info["host"] = "some-other-host-not-me"
    info["started_at"] = time.time() - (LOCK_MAX_AGE_SECONDS + 60)
    path.write_text(json.dumps(info))

    ops = list_pending_ops(cache_dir=str(tmp_path))
    assert ops == []


def test_corrupt_lock_is_removed(tmp_path):
    d = _pending_dir(str(tmp_path))
    d.mkdir(parents=True, exist_ok=True)
    bad = d / "garbage_op.json"
    bad.write_text("{not valid json")
    ops = list_pending_ops(cache_dir=str(tmp_path))
    assert ops == []
    assert not bad.exists()


def test_remove_with_matching_token_unlinks(tmp_path):
    path = create_pending_op("c1", "image_pull", cache_dir=str(tmp_path))
    token = _read(path)["token"]
    remove_pending_op("c1", "image_pull", cache_dir=str(tmp_path), token=token)
    assert not path.exists()


def test_remove_with_wrong_token_preserves_lock(tmp_path):
    """A token mismatch must NOT delete another run's lock."""
    path = create_pending_op("c1", "image_pull", cache_dir=str(tmp_path))
    remove_pending_op("c1", "image_pull", cache_dir=str(tmp_path), token="not-the-token")
    assert path.exists()


def test_remove_without_token_unlinks_unconditionally(tmp_path):
    """Legacy callers (no token) still remove the lock."""
    path = create_pending_op("c1", "image_pull", cache_dir=str(tmp_path))
    remove_pending_op("c1", "image_pull", cache_dir=str(tmp_path))
    assert not path.exists()


def test_concurrent_same_key_does_not_clobber(tmp_path):
    """A second create for a held key must not overwrite the first run's lock."""
    path1 = create_pending_op("c1", "model_download", cache_dir=str(tmp_path))
    token1 = _read(path1)["token"]

    # Simulate a second, concurrent run with the same key.  It should detect
    # the live lock and leave it intact (returning the same path).
    path2 = create_pending_op("c1", "model_download", cache_dir=str(tmp_path))
    assert path2 == path1
    assert _read(path1)["token"] == token1  # original token preserved


def test_concurrent_run_exit_does_not_remove_others_lock(tmp_path, monkeypatch):
    """A second run's context-exit must not delete the first run's lock.

    Reproduces the original clobber bug: run A holds the lock; run B enters
    the same key (finds it held, does not write its own), then exits — B's
    removal must be suppressed by the ownership check, leaving A's lock.
    """
    # Run A creates the lock.
    path_a = create_pending_op("c1", "model_download", cache_dir=str(tmp_path))
    token_a = _read(path_a)["token"]

    # Run B enters the same key via the context manager. Because A's lock is
    # live, B does not own it and must not remove it on exit.
    with pending_op("c1", "model_download", cache_dir=str(tmp_path)):
        # Inside B's block, A's lock (and token) is still the one on disk.
        assert _read(path_a)["token"] == token_a

    # After B exits, A's lock must still be present and unchanged.
    assert path_a.exists()
    assert _read(path_a)["token"] == token_a


def test_pending_op_context_manager_owner_removes_on_exit(tmp_path):
    """The owning run's context manager removes its own lock on exit."""
    lock = _lock_path(_pending_dir(str(tmp_path)), "c1", "image_pull")
    with pending_op("c1", "image_pull", cache_dir=str(tmp_path)):
        assert lock.exists()
    assert not lock.exists()


def test_pending_op_reclaims_stale_lock(tmp_path):
    """A new run reclaims a stale lock (dead pid) instead of refusing it."""
    # Plant a stale lock (dead pid).
    d = _pending_dir(str(tmp_path))
    d.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(d, "c1", "model_download")
    lock.write_text(
        json.dumps(
            {
                "cluster_id": "c1",
                "operation": "model_download",
                "pid": 2**31 - 1,
                "host": pending_ops._hostname(),
                "token": "old-token",
                "started_at": time.time(),
            }
        )
    )
    # New run should overwrite the stale lock with its own token.
    path = create_pending_op("c1", "model_download", cache_dir=str(tmp_path))
    assert _read(path)["token"] != "old-token"
    assert _read(path)["pid"] > 0
