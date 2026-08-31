"""Tests for tuning-distribution prefs parity and the --delete safety net."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.core.cluster_manager import ClusterDistributionConfig, ResourceDistributionPrefs
from sparkrun.orchestration.ssh import RemoteResult, guard_rsync_delete


# ---------------------------------------------------------------------------
# distribution.tuning prefs (inherit distribution.model unless spelled out)
# ---------------------------------------------------------------------------


def test_tuning_prefs_inherit_model_when_absent():
    """An NFS cluster's existing model block must cover tuning too.

    Both caches normally live on the SSH user's $HOME, so whatever made the
    model cache need a relaxation is true of the tuning cache for the same
    reason — requiring a second knob means hitting the same failure twice.
    """
    cfg = ClusterDistributionConfig.from_dict({"model": {"preserve_perms": False, "skip_fan_out": True}})
    assert cfg.tuning is None
    assert cfg.tuning_prefs.preserve_perms is False
    assert cfg.tuning_prefs.skip_fan_out is True


def test_explicit_tuning_block_overrides_model():
    """The split case: shared model mount, node-local $HOME."""
    cfg = ClusterDistributionConfig.from_dict({"model": {"skip_fan_out": True}, "tuning": {"skip_fan_out": False}})
    assert cfg.model.skip_fan_out is True
    assert cfg.tuning_prefs.skip_fan_out is False


def test_empty_tuning_block_opts_out_of_inheritance():
    """`tuning: {}` means "defaults for tuning", not "inherit the model block"."""
    cfg = ClusterDistributionConfig.from_dict({"model": {"preserve_perms": False}, "tuning": {}})
    assert cfg.tuning is not None
    assert cfg.tuning_prefs.preserve_perms is True


def test_tuning_block_round_trips_when_set():
    cfg = ClusterDistributionConfig(
        model=ResourceDistributionPrefs(preserve_perms=False),
        tuning=ResourceDistributionPrefs(),
    )
    out = cfg.to_dict()
    # Present-but-empty must survive: its presence is the instruction.
    assert "tuning" in out
    assert ClusterDistributionConfig.from_dict(out).tuning is not None


def test_default_config_serializes_nothing():
    assert ClusterDistributionConfig().to_dict() == {}
    assert ClusterDistributionConfig().is_default() is True


def test_model_distribution_prefs_alias_preserved():
    from sparkrun.core.cluster_manager import ModelDistributionPrefs

    assert ModelDistributionPrefs is ResourceDistributionPrefs


# ---------------------------------------------------------------------------
# guard_rsync_delete — an empty source must never clear a destination
# ---------------------------------------------------------------------------


def test_delete_kept_for_populated_source(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    opts = ["-az", "--delete", "--partial"]
    assert guard_rsync_delete(opts, str(tmp_path)) == opts


def test_delete_stripped_for_empty_source(tmp_path):
    """--delete from an empty source is 'erase the destination', never intended."""
    out = guard_rsync_delete(["-az", "--delete", "--partial"], str(tmp_path))
    assert out == ["-az", "--partial"]


def test_delete_stripped_for_missing_source(tmp_path):
    out = guard_rsync_delete(["-az", "--delete-during"], str(tmp_path / "nope"))
    assert out == ["-az"]


def test_guard_is_a_noop_without_delete(tmp_path):
    opts = ["-az", "--partial"]
    assert guard_rsync_delete(opts, str(tmp_path)) is opts


def test_run_rsync_applies_the_guard(tmp_path):
    """The guard is wired into the push path, not merely available."""
    captured = []

    class P:
        returncode, stdout, stderr = 0, b"", b""

    with mock.patch("sparkrun.orchestration.ssh.subprocess.run", lambda cmd, **kw: (captured.append(cmd), P())[1]):
        from sparkrun.orchestration.ssh import run_rsync

        run_rsync(str(tmp_path), "h1", "/remote", rsync_options=["-az", "--delete"])
    assert "--delete" not in captured[0]


# ---------------------------------------------------------------------------
# tuning distribution wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def tuning_dir(tmp_path):
    d = tmp_path / "tuning" / "sglang"
    (d / "configs").mkdir(parents=True)
    (d / "configs" / "E=256.json").write_text("{}")
    return d


def _run_tuning(tuning_dir, monkeypatch, remote_dest, **kwargs):
    import sparkrun.tuning.distribute as td

    calls = {}

    def fake_parallel(source, hosts, dest, **kw):
        calls["source"], calls["hosts"], calls["dest"] = source, hosts, dest
        calls["options"] = kw.get("rsync_options")
        return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    monkeypatch.setattr(td, "_get_local_tuning_dir", lambda r: tuning_dir)
    monkeypatch.setattr(td, "_get_remote_tuning_dir", lambda r, ssh_user=None: remote_dest)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_rsync_parallel", fake_parallel)
    failed = td.distribute_tuning_to_hosts("sglang", ["h1", "h2"], **kwargs)
    return failed, calls


def test_tuning_skip_fan_out_skips_transfer(tuning_dir, monkeypatch):
    failed, calls = _run_tuning(tuning_dir, monkeypatch, "/remote/tuning", skip_fan_out=True)
    assert failed == []
    assert calls == {}


def test_tuning_delete_dropped_when_paths_identical(tuning_dir, monkeypatch):
    """Same path on both sides is the shared-$HOME signature: source == dest."""
    _, calls = _run_tuning(tuning_dir, monkeypatch, str(tuning_dir))
    assert "--delete" not in calls["options"]


def test_tuning_delete_kept_when_paths_differ(tuning_dir, monkeypatch):
    _, calls = _run_tuning(tuning_dir, monkeypatch, "/home/other/.cache/sparkrun/tuning/sglang")
    assert "--delete" in calls["options"]


def test_tuning_preserve_perms_false_drops_archive(tuning_dir, monkeypatch):
    _, calls = _run_tuning(tuning_dir, monkeypatch, "/remote/tuning", preserve_perms=False)
    assert "-az" not in calls["options"]
    assert "-rz" in calls["options"]


def test_tuning_default_carries_nfs_safe_options(tuning_dir, monkeypatch):
    _, calls = _run_tuning(tuning_dir, monkeypatch, "/remote/tuning")
    assert {"--no-perms", "--no-group", "--omit-dir-times"} <= set(calls["options"])


def test_tuning_head_to_worker_hop_never_deletes(tuning_dir, monkeypatch):
    """That hop uses $SOURCE as both sides — --delete would prune against itself."""
    import sparkrun.tuning.distribute as td

    scripts = []
    monkeypatch.setattr(td, "_get_local_tuning_dir", lambda r: tuning_dir)
    monkeypatch.setattr(td, "_get_remote_tuning_dir", lambda r, ssh_user=None: "/remote/tuning")
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_rsync_parallel",
        lambda s, hosts, d, **kw: [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts],
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_script",
        lambda host, script, **kw: (scripts.append(script), RemoteResult(host=host, returncode=0, stdout="", stderr=""))[1],
    )
    td.distribute_tuning_to_hosts("sglang", ["h1", "h2", "h3"], transfer_mode="push")
    rsync_line = [ln for ln in scripts[0].splitlines() if "rsync" in ln][0]
    assert "--delete" not in rsync_line
    assert "--no-perms" in rsync_line
