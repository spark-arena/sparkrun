"""known_hosts refresh used by ``setup ssh`` / the wizard (distribute_host_keys).

The script is run under real bash against a sandboxed ``$HOME``, with a stub
``ssh-keyscan`` on ``PATH`` (no network) and the real ``ssh-keygen`` doing the
known_hosts lookups, removals and hashing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from sparkrun.orchestration.networking import (
    HostKeyDistribution,
    KeyscanOutcome,
    build_known_hosts_refresh_script,
    distribute_host_keys,
    parse_keyscan_output,
)
from sparkrun.orchestration.ssh import RemoteResult

needs_ssh_keygen = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen not installed")


# ---------------------------------------------------------------------------
# Script rendering / parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["a;b", "a b", "$(id)", "-oProxyCommand=x", "host*", ""])
def test_rejects_unsafe_targets(bad):
    with pytest.raises(ValueError):
        build_known_hosts_refresh_script(["10.0.0.1", bad])


def test_accepts_ips_hostnames_ipv6():
    script = build_known_hosts_refresh_script(["10.0.0.1", "spark-01.lan", "fe80::1", "localhost"], timeout=7)
    assert "KEYSCAN_TARGETS='10.0.0.1 spark-01.lan fe80::1 localhost'" in script
    assert "KEYSCAN_TIMEOUT=7" in script


def test_parse_success():
    out = "KEYSCAN_ADDED=1\nKEYSCAN_UNCHANGED=2\nKEYSCAN_REPLACED=10.0.0.1 10.0.0.2\nKEYSCAN_UNANSWERED=10.0.1.1\n"
    o = parse_keyscan_output("h1", 0, out, "")
    assert o.success
    assert (o.added, o.unchanged) == (1, 2)
    assert o.replaced == ["10.0.0.1", "10.0.0.2"]
    assert o.unanswered == ["10.0.1.1"]


def test_parse_script_error():
    o = parse_keyscan_output("h1", 1, "KEYSCAN_ERROR=ssh-keyscan/ssh-keygen not found\n", "")
    assert not o.success
    assert "not found" in o.error


def test_parse_nonzero_exit_is_failure():
    """A timeout (rc 124) or killed script must not read as success."""
    o = parse_keyscan_output("local", 124, "", "local execution timed out after 60s")
    assert not o.success
    assert "timed out" in o.error


def test_parse_no_result_is_failure():
    o = parse_keyscan_output("h1", 0, "", "")
    assert not o.success


# ---------------------------------------------------------------------------
# Script behaviour under real bash
# ---------------------------------------------------------------------------


def _keypair(tmp_path: Path, name: str, key_type: str = "ed25519") -> str:
    path = tmp_path / name
    subprocess.run(["ssh-keygen", "-q", "-t", key_type, "-N", "", "-f", str(path)], check=True)
    fields = (tmp_path / (name + ".pub")).read_text().split()
    return "%s %s" % (fields[0], fields[1])


class _Env:
    """Sandboxed HOME + stub ssh-keyscan answering from a host -> keys table."""

    def __init__(self, tmp_path: Path):
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.table = tmp_path / "keyscan_table"
        self.table.write_text("")
        stub = self.bin / "ssh-keyscan"
        # Prints table lines for each host argument (non-option words).
        stub.write_text(
            '#!/bin/bash\nfor a in "$@"; do case "$a" in -*) ;; [0-9]) ;; *) awk -v h="$a" \'$1 == h\' "%s";; esac; done\n' % self.table
        )
        stub.chmod(0o755)

    @property
    def known_hosts(self) -> Path:
        return self.home / ".ssh" / "known_hosts"

    def serve(self, host_keys: dict[str, list[str]]) -> None:
        self.table.write_text("".join("%s %s\n" % (h, k) for h, keys in host_keys.items() for k in keys))

    def run(self, targets: list[str]) -> KeyscanOutcome:
        env = dict(os.environ, HOME=str(self.home), PATH="%s:%s" % (self.bin, os.environ.get("PATH", "")))
        proc = subprocess.run(
            ["bash", "-s"],
            input=build_known_hosts_refresh_script(targets, timeout=1),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        return parse_keyscan_output("t", proc.returncode, proc.stdout, proc.stderr)

    def recorded(self, host: str) -> set[str]:
        proc = subprocess.run(
            ["ssh-keygen", "-F", host, "-f", str(self.known_hosts)],
            capture_output=True,
            text=True,
        )
        return {" ".join(line.split()[1:3]) for line in proc.stdout.splitlines() if line and not line.startswith("#")}


@needs_ssh_keygen
def test_fresh_then_idempotent(tmp_path):
    env = _Env(tmp_path)
    ed, ec = _keypair(tmp_path, "a"), _keypair(tmp_path, "b", "ecdsa")
    env.serve({"10.0.0.1": [ed, ec], "spark-02": [ed]})

    o = env.run(["10.0.0.1", "spark-02", "10.0.1.9"])
    assert o.success, o.error
    assert (o.added, o.unchanged, o.replaced) == (2, 0, [])
    assert o.unanswered == ["10.0.1.9"]
    assert env.recorded("10.0.0.1") == {ed, ec}
    # Written hashed, like ssh-keyscan -H.
    assert all(line.startswith("|1|") for line in env.known_hosts.read_text().splitlines())

    before = env.known_hosts.read_text()
    o = env.run(["10.0.0.1", "spark-02", "10.0.1.9"])
    assert (o.added, o.unchanged, o.replaced) == (0, 2, [])
    # Hashes are salted, so appending again would grow the file every run.
    assert env.known_hosts.read_text() == before


@needs_ssh_keygen
def test_changed_key_is_replaced_and_reported(tmp_path):
    env = _Env(tmp_path)
    old, new, other = _keypair(tmp_path, "old"), _keypair(tmp_path, "new"), _keypair(tmp_path, "other")
    env.known_hosts.parent.mkdir()
    env.known_hosts.write_text("10.0.0.1 %s\n10.0.0.2 %s\n" % (old, other))
    env.serve({"10.0.0.1": [new], "10.0.0.2": [other]})

    o = env.run(["10.0.0.1", "10.0.0.2"])
    assert o.success, o.error
    assert o.replaced == ["10.0.0.1"]
    assert o.unchanged == 1
    assert env.recorded("10.0.0.1") == {new}
    # The stale key no longer trusted, and the untouched host's entry kept.
    assert old.split()[1] not in env.known_hosts.read_text()
    assert env.recorded("10.0.0.2") == {other}


@needs_ssh_keygen
def test_new_key_type_is_added_not_replaced(tmp_path):
    env = _Env(tmp_path)
    ed, ec = _keypair(tmp_path, "a"), _keypair(tmp_path, "b", "ecdsa")
    env.known_hosts.parent.mkdir()
    env.known_hosts.write_text("10.0.0.1 %s\n" % ed)
    env.serve({"10.0.0.1": [ed, ec]})

    o = env.run(["10.0.0.1"])
    assert (o.added, o.replaced) == (1, [])
    assert env.recorded("10.0.0.1") == {ed, ec}


@needs_ssh_keygen
def test_unrelated_entries_survive(tmp_path):
    env = _Env(tmp_path)
    old, new = _keypair(tmp_path, "old"), _keypair(tmp_path, "new")
    env.known_hosts.parent.mkdir()
    env.known_hosts.write_text("# my comment\ngithub.com %s\n10.0.0.1 %s\n" % (new, old))
    env.serve({"10.0.0.1": [new]})

    env.run(["10.0.0.1"])
    text = env.known_hosts.read_text()
    assert "# my comment" in text
    assert "github.com %s" % new in text


# ---------------------------------------------------------------------------
# distribute_host_keys dispatch
# ---------------------------------------------------------------------------

_OK = "KEYSCAN_ADDED=1\nKEYSCAN_UNCHANGED=0\nKEYSCAN_REPLACED=\nKEYSCAN_UNANSWERED=\n"


def test_local_scan_limited_to_local_ips():
    local_scripts = []

    def fake_local(script, dry_run=False, timeout=None):
        local_scripts.append(script)
        return RemoteResult(host="localhost", returncode=0, stdout=_OK, stderr="")

    remote = mock.Mock(return_value=[RemoteResult(host="h1", returncode=0, stdout=_OK, stderr="")])
    with (
        mock.patch("sparkrun.orchestration.ssh.run_local_script", fake_local),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote),
        mock.patch("sparkrun.utils.is_local_host", return_value=False),
    ):
        dist = distribute_host_keys(["192.168.1.1", "10.0.0.1"], ["h1"], local_ips=["192.168.1.1"])

    assert len(local_scripts) == 1
    assert "KEYSCAN_TARGETS=192.168.1.1\n" in local_scripts[0]
    # Hosts scan everything.
    assert "KEYSCAN_TARGETS='192.168.1.1 10.0.0.1'" in remote.call_args[0][1]
    assert dist.local is not None and dist.local.success
    assert [o.target for o in dist.hosts] == ["h1"]


def test_no_local_scan_when_nothing_reachable():
    remote = mock.Mock(return_value=[RemoteResult(host="h1", returncode=0, stdout=_OK, stderr="")])
    with (
        mock.patch("sparkrun.orchestration.ssh.run_local_script") as local,
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote),
        mock.patch("sparkrun.utils.is_local_host", return_value=False),
    ):
        dist = distribute_host_keys(["10.0.0.1"], ["h1"], local_ips=[])
    local.assert_not_called()
    assert dist.local is None


def test_failures_and_replacements_surface():
    replaced = "KEYSCAN_ADDED=0\nKEYSCAN_UNCHANGED=0\nKEYSCAN_REPLACED=10.0.0.1\nKEYSCAN_UNANSWERED=\n"
    remote = mock.Mock(
        return_value=[
            RemoteResult(host="h1", returncode=0, stdout=replaced, stderr=""),
            RemoteResult(host="h2", returncode=255, stdout="", stderr="Permission denied"),
        ]
    )
    with (
        mock.patch(
            "sparkrun.orchestration.ssh.run_local_script",
            return_value=RemoteResult(host="localhost", returncode=124, stdout="", stderr="timed out"),
        ),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote),
        mock.patch("sparkrun.utils.is_local_host", return_value=False),
    ):
        dist = distribute_host_keys(["10.0.0.1"], ["h1", "h2"])

    assert not dist.local.success
    assert [o.success for o in dist.hosts] == [True, False]
    assert dist.replaced == {"h1": ["10.0.0.1"]}


def test_cli_report_renders_replacements_and_local_failure(capsys):
    from sparkrun.cli._setup._ssh import _report_host_key_distribution

    dist = HostKeyDistribution(
        local=KeyscanOutcome(target="local", success=False, error="timed out"),
        hosts=[
            KeyscanOutcome(target="h1", success=True, replaced=["10.0.0.1"]),
            KeyscanOutcome(target="h2", success=True),
        ],
    )
    _report_host_key_distribution(dist, 3)
    out = capsys.readouterr()
    assert "failed on this machine: timed out" in out.err
    assert "CHANGED host key(s) replaced in h1's known_hosts: 10.0.0.1" in out.err
    # No "+ local" claim when the local refresh failed.
    assert "distributed to 2/2 host(s)." in out.out
    assert "this machine." not in out.out


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


@needs_ssh_keygen
def test_stale_key_beside_current_is_removed(tmp_path):
    """The append-only refresh left hosts holding both keys: ssh accepted them
    (any match wins), so nothing was missing — the stale one must still go."""
    env = _Env(tmp_path)
    old, new = _keypair(tmp_path, "old"), _keypair(tmp_path, "new")
    env.known_hosts.parent.mkdir()
    env.known_hosts.write_text("10.0.0.1 %s\n10.0.0.1 %s\n" % (old, new))
    env.serve({"10.0.0.1": [new]})

    o = env.run(["10.0.0.1"])
    assert o.replaced == ["10.0.0.1"]
    assert env.recorded("10.0.0.1") == {new}


@needs_ssh_keygen
def test_mixed_case_hostname_matches_keyscan_output(tmp_path):
    """ssh-keyscan prints hostnames lowercased."""
    env = _Env(tmp_path)
    ed = _keypair(tmp_path, "a")
    env.serve({"spark-01": [ed]})

    o = env.run(["Spark-01"])
    assert (o.added, o.unanswered) == (1, [])
    assert env.recorded("spark-01") == {ed}


def test_parse_failed_removal_is_failure():
    out = "KEYSCAN_ADDED=0\nKEYSCAN_UNCHANGED=0\nKEYSCAN_REPLACED=\nKEYSCAN_UNANSWERED=\nKEYSCAN_FAILED=10.0.0.1\n"
    o = parse_keyscan_output("h1", 0, out, "")
    assert not o.success
    assert "10.0.0.1" in o.error


def test_all_unanswered_is_not_registered():
    out = "KEYSCAN_ADDED=0\nKEYSCAN_UNCHANGED=0\nKEYSCAN_REPLACED=\nKEYSCAN_UNANSWERED=10.0.0.1\nKEYSCAN_FAILED=\n"
    o = parse_keyscan_output("local", 0, out, "")
    assert o.success
    assert not o.registered


def test_targets_normalized_and_invalid_skipped():
    remote = mock.Mock(return_value=[RemoteResult(host="h1", returncode=0, stdout=_OK, stderr="")])
    with (
        mock.patch("sparkrun.orchestration.ssh.run_local_script") as local,
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote),
        mock.patch("sparkrun.utils.is_local_host", return_value=False),
    ):
        distribute_host_keys(["ubuntu@Spark-01", "spark-01", "bad;host", "10.0.0.1"], ["h1"], local_ips=[])
    local.assert_not_called()
    assert "KEYSCAN_TARGETS='spark-01 10.0.0.1'" in remote.call_args[0][1]
