"""Tests for the remote tuning-directory preflight.

Regression cover for the root cause behind "Tuning config distribution failed":
the tuning directory is bind-mounted into the inference container, the decision
to mount is made from the *control node's* copy, and Docker creates a missing
bind-mount source **root-owned** — locking the SSH user out of its own cache.
"""

from __future__ import annotations

import pytest

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.tuning._common import tuning_configs_present


@pytest.fixture
def tuning_dir(tmp_path):
    d = tmp_path / "tuning" / "sglang"
    (d / "configs").mkdir(parents=True)
    (d / "configs" / "E=256.json").write_text("{}")
    return d


# ---------------------------------------------------------------------------
# the shared predicate
# ---------------------------------------------------------------------------


def test_predicate_true_for_nested_configs(tuning_dir):
    assert tuning_configs_present(tuning_dir) is True


def test_predicate_false_for_empty_and_missing(tmp_path):
    (tmp_path / "empty").mkdir()
    assert tuning_configs_present(tmp_path / "empty") is False
    assert tuning_configs_present(tmp_path / "nope") is False


def test_mount_and_preflight_share_the_predicate(tuning_dir):
    """They must not drift: the mount is decided centrally, applied everywhere.

    If the mount fires where the preflight does not, the daemon creates the
    directory root-owned and the host is locked out permanently.
    """
    from sparkrun.tuning._common import _get_tuning_volumes

    assert _get_tuning_volumes(lambda: tuning_dir, "/c") == {str(tuning_dir): "/c"}
    assert _get_tuning_volumes(lambda: tuning_dir.parent / "absent", "/c") is None


# ---------------------------------------------------------------------------
# ensure_remote_tuning_dirs
# ---------------------------------------------------------------------------


def _wire(monkeypatch, mkdir_results, chown_calls=None):
    """Patch the SSH fan-out; return the list of scripts it was handed."""
    import sparkrun.tuning.distribute as td

    scripts = []
    seq = list(mkdir_results)

    def fake_parallel(hosts, script, **kw):
        scripts.append((list(hosts), script))
        rcs = seq.pop(0)
        return [RemoteResult(host=h, returncode=rcs.get(h, 0), stdout="", stderr="") for h in hosts]

    monkeypatch.setattr(td, "_get_remote_tuning_dir", lambda r, ssh_user=None: "/home/u/.cache/sparkrun/tuning/sglang")
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", fake_parallel)
    monkeypatch.setattr(
        "sparkrun.orchestration.sudo.ensure_remote_dir_ownership",
        lambda path, hosts, **kw: (chown_calls.append((path, list(hosts))) if chown_calls is not None else None) or [],
    )
    return scripts


def test_preflight_creates_the_directory(monkeypatch):
    from sparkrun.tuning.distribute import ensure_remote_tuning_dirs

    scripts = _wire(monkeypatch, [{}])
    failed = ensure_remote_tuning_dirs("sglang", ["h1", "h2"])
    assert failed == []
    hosts, script = scripts[0]
    assert hosts == ["h1", "h2"]
    assert 'mkdir -p "/home/u/.cache/sparkrun/tuning/sglang"' in script


def test_preflight_skips_localhost(monkeypatch):
    """The control node's own copy is the one we already checked."""
    import sparkrun.tuning.distribute as td
    from sparkrun.tuning.distribute import ensure_remote_tuning_dirs

    monkeypatch.setattr(td, "is_local_host", lambda h: h == "localhost")
    scripts = _wire(monkeypatch, [{}])
    ensure_remote_tuning_dirs("sglang", ["localhost"])
    assert scripts == []


def test_failure_triggers_ownership_repair_scoped_to_failing_hosts(monkeypatch):
    """A root-owned ancestor is the expected cause and is repairable."""
    from sparkrun.tuning.distribute import ensure_remote_tuning_dirs

    chown = []
    # h2 fails the first mkdir, then succeeds after the chown.
    scripts = _wire(monkeypatch, [{"h2": 1}, {}], chown_calls=chown)
    failed = ensure_remote_tuning_dirs("sglang", ["h1", "h2"])

    assert failed == []
    assert chown == [("/home/u/.cache/sparkrun/tuning/sglang", ["h2"])]
    # The retry is scoped to the broken host — a healthy cluster pays nothing.
    assert scripts[1][0] == ["h2"]


def test_unrepairable_host_is_reported_not_raised(monkeypatch):
    """Best-effort: a launch without tuning configs is slower, not broken."""
    from sparkrun.tuning.distribute import ensure_remote_tuning_dirs

    scripts = _wire(monkeypatch, [{"h1": 1}, {"h1": 1}], chown_calls=[])
    failed = ensure_remote_tuning_dirs("sglang", ["h1"])
    assert failed == ["h1"]
    assert len(scripts) == 2


def test_no_hosts_is_a_noop(monkeypatch):
    from sparkrun.tuning.distribute import ensure_remote_tuning_dirs

    scripts = _wire(monkeypatch, [])
    assert ensure_remote_tuning_dirs("sglang", []) == []
    assert scripts == []
