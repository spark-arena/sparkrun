"""A local cache must never be rsynced onto itself through a host/path alias."""

import hashlib
import os
from unittest.mock import Mock

import pytest

from sparkrun.models.distribute import _model_rsync_options
from sparkrun.orchestration.ssh import run_rsync


@pytest.mark.parametrize("preserve_perms", [False, True])
def test_shared_blob_cache_self_copy_preserves_links(tmp_path, monkeypatch, preserve_perms):
    repo = tmp_path / "hub/models--org--model"
    digest = hashlib.sha256(b"weights").hexdigest()
    backing = tmp_path / "hub/blobs" / digest[:2] / digest
    backing.parent.mkdir(parents=True)
    backing.write_bytes(b"weights")
    blob = repo / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.symlink_to(os.path.relpath(backing, blob.parent))
    snapshot = repo / "snapshots/rev/model.safetensors"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    monkeypatch.setattr("sparkrun.orchestration.ssh.should_run_locally", lambda *a: True)
    execute = Mock(side_effect=AssertionError("self-copy must not reach rsync"))
    monkeypatch.setattr("sparkrun.orchestration.ssh._run_subprocess", execute)

    result = run_rsync(str(repo), "local-alias", str(alias), rsync_options=_model_rsync_options(preserve_perms))

    assert result.success
    assert blob.is_symlink()
    assert snapshot.read_bytes() == b"weights"
    execute.assert_not_called()


def test_distinct_cache_on_same_host_is_copied(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    monkeypatch.setattr("sparkrun.orchestration.ssh.should_run_locally", lambda *a: True)
    from sparkrun.orchestration.ssh import RemoteResult

    execute = Mock(return_value=RemoteResult("localhost", 0, "", ""))
    monkeypatch.setattr("sparkrun.orchestration.ssh._run_subprocess", execute)
    assert run_rsync(str(source), "localhost", str(target)).success
    execute.assert_called_once()
