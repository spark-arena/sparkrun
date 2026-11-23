"""Changed SSH host keys: detection, approved replacement, and the setup prompt.

``ssh -G`` and ``ssh-keygen`` run for real against a scratch known_hosts (named
through ``UserKnownHostsFile``); only the network scan is supplied by the test.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import sparkrun.api.setup._host_keys as host_keys
from sparkrun.api.setup import SshAccessError, find_changed_host_keys, key_fingerprint, replace_host_key


def _keypair(tmp_path: Path, name: str, kind: str = "ed25519") -> tuple[str, str]:
    """(type, base64 blob) of a fresh public key."""
    path = tmp_path / name
    subprocess.run(["ssh-keygen", "-q", "-t", kind, "-N", "", "-f", str(path)], check=True)
    key_type, blob = path.with_suffix(".pub").read_text().split()[:2]
    return key_type, blob


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """A known_hosts recording *old* keys and a host that now presents *new* ones."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))  # the writer's lock lives in ~/.ssh
    old_ed = _keypair(tmp_path, "old_ed")
    new_ed = _keypair(tmp_path, "new_ed")
    rsa = _keypair(tmp_path, "rsa", "rsa")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("10.0.0.1 %s %s\n10.0.0.1 %s %s\n" % (*old_ed, *rsa))
    presented = {"10.0.0.1": {new_ed[0]: new_ed[1], rsa[0]: rsa[1]}}
    scans = []

    def scan(target, timeout):
        scans.append((target.hostname, target.port, target.lookup))
        return dict(presented.get(target.hostname, {}))

    monkeypatch.setattr(host_keys, "_scan", scan)
    # Pinned: Ubuntu's /etc/ssh/ssh_config sets HashKnownHosts yes, and the
    # tests must not depend on the developer's ssh config.
    options = ["-o", "UserKnownHostsFile=%s" % known_hosts, "-o", "HashKnownHosts=no"]
    return {"kh": known_hosts, "opts": options, "old": old_ed, "new": new_ed, "rsa": rsa, "presented": presented, "scans": scans}


def test_fingerprint_matches_ssh_keygen(tmp_path):
    key_type, blob = _keypair(tmp_path, "k")
    listed = subprocess.run(["ssh-keygen", "-lf", str(tmp_path / "k.pub")], capture_output=True, text=True, check=True).stdout.split()[1]
    assert key_fingerprint(blob) == listed


def test_changed_type_is_reported_with_both_fingerprints(scene):
    changes = find_changed_host_keys(["10.0.0.1"], options=scene["opts"])

    assert len(changes) == 1  # the RSA key still matches; only ed25519 changed
    change = changes[0]
    assert (change.host, change.lookup, change.key_type) == ("10.0.0.1", "10.0.0.1", scene["new"][0])
    assert change.recorded == (key_fingerprint(scene["old"][1]),)
    assert change.presented == key_fingerprint(scene["new"][1])
    assert change.known_hosts == str(scene["kh"])


def test_new_types_unknown_hosts_and_silent_hosts_are_not_changes(scene, tmp_path):
    ecdsa = _keypair(tmp_path, "ecdsa", "ecdsa")
    scene["presented"]["10.0.0.1"] = {scene["rsa"][0]: scene["rsa"][1], ecdsa[0]: ecdsa[1]}
    scene["presented"]["10.0.0.9"] = {ecdsa[0]: ecdsa[1]}  # never recorded at all

    assert find_changed_host_keys(["10.0.0.1", "10.0.0.9", "10.0.0.2"], options=scene["opts"]) == []


def test_ssh_config_decides_which_entry_is_checked(scene):
    scene["kh"].write_text("[10.0.0.1]:2222 %s %s\n" % scene["old"])

    changes = find_changed_host_keys(["10.0.0.1"], options=scene["opts"] + ["-o", "Port=2222"])

    assert [c.lookup for c in changes] == ["[10.0.0.1]:2222"]
    assert scene["scans"][-1] == ("10.0.0.1", "2222", "[10.0.0.1]:2222")


def test_replace_requires_the_approved_fingerprint(scene):
    before = scene["kh"].read_text()

    with pytest.raises(SshAccessError, match="no longer presents the approved key"):
        replace_host_key("10.0.0.1", key_fingerprint(scene["old"][1]), options=scene["opts"])

    assert scene["kh"].read_text() == before


def test_replace_records_the_presented_keys_and_keeps_a_backup(scene):
    approved = key_fingerprint(scene["new"][1])

    result = replace_host_key("10.0.0.1", approved, options=scene["opts"])

    assert (result.lookup, result.fingerprint) == ("10.0.0.1", approved)
    assert find_changed_host_keys(["10.0.0.1"], options=scene["opts"]) == []
    text = scene["kh"].read_text()
    assert scene["new"][1] in text and scene["old"][1] not in text
    assert scene["old"][1] in Path(str(scene["kh"]) + ".old").read_text()


def test_replace_hashes_when_known_hosts_is_hashed(scene):
    hashed = [o if o != "HashKnownHosts=no" else "HashKnownHosts=yes" for o in scene["opts"]]  # ssh keeps the first value
    replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=hashed)

    new_line = scene["kh"].read_text().splitlines()[-1]
    assert new_line.startswith("|1|") and "10.0.0.1" not in new_line
    found = subprocess.run(["ssh-keygen", "-F", "10.0.0.1", "-f", str(scene["kh"])], capture_output=True, text=True).stdout
    assert scene["new"][1] in found  # ssh's own matcher accepts our hashed entry
    assert find_changed_host_keys(["10.0.0.1"], options=scene["opts"]) == []


def test_hashed_entries_are_matched_and_replaced(scene):
    subprocess.run(["ssh-keygen", "-q", "-H", "-f", str(scene["kh"])], check=True, capture_output=True)
    assert "10.0.0.1" not in scene["kh"].read_text()

    [change] = find_changed_host_keys(["10.0.0.1"], options=scene["opts"])
    replace_host_key("10.0.0.1", change.presented, options=scene["opts"])

    assert scene["old"][1] not in scene["kh"].read_text()
    assert scene["rsa"][1] in scene["kh"].read_text()  # the other type's hashed entry survives
    assert find_changed_host_keys(["10.0.0.1"], options=scene["opts"]) == []


def test_only_the_approved_type_is_written(scene, tmp_path):
    """Two changed types, one approved: the other is neither replaced nor trusted."""
    new_rsa = _keypair(tmp_path, "new_rsa", "rsa")
    ecdsa = _keypair(tmp_path, "ecdsa", "ecdsa")  # never recorded
    scene["presented"]["10.0.0.1"] = {scene["new"][0]: scene["new"][1], new_rsa[0]: new_rsa[1], ecdsa[0]: ecdsa[1]}
    assert len(find_changed_host_keys(["10.0.0.1"], options=scene["opts"])) == 2

    replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])

    text = scene["kh"].read_text()
    assert scene["new"][1] in text
    assert new_rsa[1] not in text and scene["rsa"][1] in text  # rsa untouched: still the recorded one
    assert ecdsa[1] not in text  # an unapproved new type is left for ssh's own accept-new
    [remaining] = find_changed_host_keys(["10.0.0.1"], options=scene["opts"])
    assert remaining.key_type == new_rsa[0]


def test_a_shared_line_keeps_its_other_names(scene):
    scene["kh"].write_text("spark1,10.0.0.1 %s %s\n" % scene["old"])

    replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])

    lines = scene["kh"].read_text().splitlines()
    assert lines == ["spark1 %s %s" % scene["old"], "10.0.0.1 %s %s" % scene["new"]]


def test_a_wildcard_entry_is_never_edited(scene):
    scene["kh"].write_text("10.0.0.* %s %s\n" % scene["old"])
    before = scene["kh"].read_text()

    [change] = find_changed_host_keys(["10.0.0.1"], options=scene["opts"])
    with pytest.raises(SshAccessError, match="wildcard entry"):
        replace_host_key("10.0.0.1", change.presented, options=scene["opts"])

    assert scene["kh"].read_text() == before


def test_a_revoked_key_is_never_trusted(scene):
    scene["kh"].write_text(scene["kh"].read_text() + "@revoked 10.0.0.1 %s %s\n" % scene["new"])

    with pytest.raises(SshAccessError, match="revoked"):
        replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])


def test_a_symlinked_known_hosts_is_written_through(scene, tmp_path):
    real = tmp_path / "real_known_hosts"
    scene["kh"].rename(real)
    scene["kh"].symlink_to(real)

    replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])

    assert scene["kh"].is_symlink()
    assert scene["new"][1] in real.read_text()


def test_writes_lf_and_keeps_the_file_mode(scene):
    scene["kh"].chmod(0o644)

    replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])

    assert b"\r" not in scene["kh"].read_bytes()
    assert scene["kh"].stat().st_mode & 0o777 == 0o644


def test_keyscan_lines_for_other_hosts_are_ignored():
    output = "# 10.0.0.1:22 SSH-2.0-OpenSSH\n10.0.0.1 ssh-ed25519 AAAA1\nevil ssh-rsa AAAA2\n[10.0.0.1]:22 ssh-rsa AAAA3\n10.0.0.1 ssh-ed25519 AAAA4\n"

    assert host_keys.parse_keyscan(output, "10.0.0.1") == {"ssh-ed25519": "AAAA1", "ssh-rsa": "AAAA3"}


@pytest.mark.parametrize("option", ["HostKeyAlias=*", "HostKeyAlias=a,b", "HostKeyAlias=!x"])
def test_pattern_shaped_names_are_never_written(scene, option):
    with pytest.raises(SshAccessError, match="not a host name"):
        replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"] + ["-o", option])


def test_scan_failures_become_ssh_access_errors(scene, monkeypatch):
    def hang(target, timeout):
        raise subprocess.TimeoutExpired("ssh-keyscan", timeout)

    monkeypatch.setattr(host_keys, "_scan", hang)
    with pytest.raises(SshAccessError, match="Could not scan"):
        replace_host_key("10.0.0.1", "SHA256:x", options=scene["opts"])


def test_replace_refuses_a_silent_host(scene):
    with pytest.raises(SshAccessError, match="did not answer"):
        replace_host_key("10.0.0.2", "SHA256:x", options=scene["opts"])


def test_a_host_that_cannot_be_checked_never_fails_the_check(scene, monkeypatch):
    def broken(*a, **k):
        raise TypeError("unexpected")

    monkeypatch.setattr(host_keys, "_resolve_target", broken)
    assert find_changed_host_keys(["10.0.0.1"], options=scene["opts"]) == []


# ---------------------------------------------------------------------------
# The prompt (cli/_setup/_ssh.py)
# ---------------------------------------------------------------------------


def _prompt(scene, monkeypatch, *, answer, tty=True, yes=False):
    from types import SimpleNamespace

    from sparkrun.cli._setup._ssh import _resolve_changed_host_keys

    config = SimpleNamespace(ssh_options=scene["opts"], ssh_key=None)
    asked = []

    monkeypatch.setattr("sparkrun.cli._setup._ssh._stdin_is_terminal", lambda: tty)
    monkeypatch.setattr(click, "confirm", lambda text, default: asked.append((text, default)) or answer)

    @click.command()
    def cmd():
        click.echo("BLOCKED=%s" % ",".join(_resolve_changed_host_keys(["10.0.0.1"], None, config, yes=yes)))

    return CliRunner().invoke(cmd, []), asked


def test_prompt_defaults_to_no_and_leaves_the_key(scene, monkeypatch):
    before = scene["kh"].read_text()

    result, asked = _prompt(scene, monkeypatch, answer=False)

    assert result.exit_code == 0, result.output
    assert asked and asked[0][1] is False  # default No
    assert "BLOCKED=10.0.0.1" in result.output
    assert key_fingerprint(scene["old"][1]) in result.output and key_fingerprint(scene["new"][1]) in result.output
    assert "ssh-keygen -R 10.0.0.1" in result.output
    assert scene["kh"].read_text() == before


def test_prompt_is_asked_once_per_host_per_process(scene, monkeypatch, tmp_path):
    new_rsa = _keypair(tmp_path, "new_rsa2", "rsa")
    scene["presented"]["10.0.0.1"][new_rsa[0]] = new_rsa[1]  # two changed types, one question

    _result, asked = _prompt(scene, monkeypatch, answer=False)
    again, asked_again = _prompt(scene, monkeypatch, answer=True)

    assert len(asked) == 1
    assert not asked_again  # the wizard's mesh step does not ask about a declined host again
    assert "BLOCKED=10.0.0.1" in again.output


def test_prompt_yes_replaces(scene, monkeypatch):
    result, _asked = _prompt(scene, monkeypatch, answer=True)

    assert "BLOCKED=\n" in result.output
    assert "Trusted the new ssh-ed25519 key in" in result.output
    assert find_changed_host_keys(["10.0.0.1"], options=scene["opts"]) == []


@pytest.mark.parametrize(("tty", "yes"), [(False, False), (True, True)])
def test_never_accepted_without_a_person(scene, monkeypatch, tty, yes):
    before = scene["kh"].read_text()

    result, asked = _prompt(scene, monkeypatch, answer=True, tty=tty, yes=yes)

    assert not asked
    assert "BLOCKED=10.0.0.1" in result.output
    assert scene["kh"].read_text() == before


# ---------------------------------------------------------------------------
# setup ssh: asked before the mesh script, which never runs on a declined key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer", [False, True])
def test_setup_ssh_asks_before_the_mesh_script(tmp_path, monkeypatch, answer):
    from unittest import mock

    import yaml

    import sparkrun.core.config
    from sparkrun.cli import main
    from sparkrun.orchestration.ssh import RemoteResult

    root = tmp_path / "config"
    root.mkdir()
    key = tmp_path / "keys" / "cluster_ed25519"
    key.parent.mkdir()
    key.write_text("PRIVATE")
    key.with_name(key.name + ".pub").write_text("ssh-ed25519 AAAACONFIGURED cluster\n")
    old, new = _keypair(tmp_path, "old"), _keypair(tmp_path, "new")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("10.0.0.1 %s %s\n" % old)
    (root / "config.yaml").write_text(yaml.safe_dump({"ssh": {"key": str(key), "options": ["-o", "UserKnownHostsFile=%s" % known_hosts]}}))
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", root)
    monkeypatch.setattr("sparkrun.cli._setup._ssh._mesh_uses_native", lambda: False)
    monkeypatch.setattr("sparkrun.orchestration.ssh.offered_identities", mock.Mock(return_value=[]))
    monkeypatch.setattr(host_keys, "_scan", lambda target, timeout: {new[0]: new[1]} if target.hostname == "10.0.0.1" else {})
    monkeypatch.setattr("sparkrun.cli._setup._ssh._stdin_is_terminal", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda text, default: answer)
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda hosts, script, **k: [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts],
    )
    real_run = subprocess.run
    scripts = []

    def run(cmd, *args, **kwargs):
        if cmd[0] == "bash":
            scripts.append(cmd)
            return mock.Mock(returncode=0)
        return real_run(cmd, *args, **kwargs)

    with mock.patch("subprocess.run", side_effect=run):
        result = CliRunner().invoke(
            main, ["setup", "ssh", "--hosts", "10.0.0.1,10.0.0.2", "--user", "u", "--no-include-self", "--no-discover-ips"]
        )

    assert "SSH host key CHANGED for 10.0.0.1" in result.output
    if answer:
        assert result.exit_code == 0, result.output
        assert scripts, "the mesh runs once the key is replaced"
        assert new[1] in known_hosts.read_text()
    else:
        assert result.exit_code == 1
        assert not scripts, "the mesh script must not run against a declined key"
        assert "Resolve the changed host key(s) for 10.0.0.1" in result.output
        assert known_hosts.read_text() == "10.0.0.1 %s %s\n" % old


# ---------------------------------------------------------------------------
# Second review: matcher and rewrite edge cases
# ---------------------------------------------------------------------------


def test_a_tab_separated_line_loses_its_stale_key(scene):
    scene["kh"].write_text("10.0.0.1\t%s\t%s\n10.0.0.1 %s %s\n" % (*scene["old"], *scene["rsa"]))

    replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])

    assert scene["old"][1] not in scene["kh"].read_text()
    assert find_changed_host_keys(["10.0.0.1"], options=scene["opts"]) == []


@pytest.mark.parametrize(
    ("pattern", "lookup", "matches"),
    [
        ("[10.0.0.*]:2222", "[10.0.0.5]:2222", True),
        ("spark[12]*", "spark1x", False),  # [ is literal in OpenSSH patterns
        ("spark[12]*", "spark[12]-a", True),
        ("10.0.0.?", "10.0.0.7", True),
        ("10.0.0.?", "10.0.0.77", False),
    ],
)
def test_patterns_follow_openssh(pattern, lookup, matches):
    assert host_keys._pattern_matches(pattern, lookup) is matches


def test_negation_vetoes_the_line():
    assert host_keys._names_match("10.0.0.*,!10.0.0.1", "10.0.0.1") == (False, False)
    assert host_keys._names_match("10.0.0.*,!10.0.0.1", "10.0.0.2") == (False, True)


def test_a_pattern_beside_the_exact_name_is_refused(scene):
    scene["kh"].write_text("10.0.0.*,10.0.0.1 %s %s\n" % scene["old"])
    before = scene["kh"].read_text()

    with pytest.raises(SshAccessError, match="wildcard entry"):
        replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])

    assert scene["kh"].read_text() == before


def test_hashed_entries_on_a_non_default_port(scene):
    opts = scene["opts"] + ["-o", "Port=2222"]
    scene["kh"].write_text("[10.0.0.1]:2222 %s %s\n" % scene["old"])
    subprocess.run(["ssh-keygen", "-q", "-H", "-f", str(scene["kh"])], check=True, capture_output=True)

    [change] = find_changed_host_keys(["10.0.0.1"], options=opts)
    assert change.lookup == "[10.0.0.1]:2222"
    replace_host_key("10.0.0.1", change.presented, options=opts)

    found = subprocess.run(["ssh-keygen", "-F", "[10.0.0.1]:2222", "-f", str(scene["kh"])], capture_output=True, text=True).stdout
    assert scene["new"][1] in found and scene["old"][1] not in found


def test_a_corrupt_recorded_line_does_not_hide_a_change(scene):
    scene["kh"].write_text(scene["kh"].read_text() + "10.0.0.1 %s not!base64\n" % scene["old"][0])

    [change] = find_changed_host_keys(["10.0.0.1"], options=scene["opts"])

    assert change.recorded == (key_fingerprint(scene["old"][1]),)


def test_undecodable_bytes_elsewhere_survive_a_rewrite(scene):
    scene["kh"].write_bytes(scene["kh"].read_bytes() + b"# caf\xe9 latin-1 comment\n")

    replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=scene["opts"])

    assert b"# caf\xe9 latin-1 comment\n" in scene["kh"].read_bytes()


def test_a_proxied_host_is_not_scanned(scene):
    opts = scene["opts"] + ["-o", "ProxyJump=bastion"]

    assert find_changed_host_keys(["10.0.0.1"], options=opts) == []
    with pytest.raises(SshAccessError, match="reached through proxyjump"):
        replace_host_key("10.0.0.1", key_fingerprint(scene["new"][1]), options=opts)
    assert scene["scans"] == []


def test_a_known_hosts_path_with_a_space(tmp_path):
    spaced = tmp_path / "First Last" / "known_hosts"
    spaced.parent.mkdir()
    spaced.write_text("")

    assert host_keys._first_known_hosts("%s %s/known_hosts2" % (spaced, tmp_path)) == spaced
    assert host_keys._first_known_hosts("/nope/known_hosts /nope/known_hosts2") == Path("/nope/known_hosts")


def test_a_host_with_two_changed_types_recovers_with_one_yes_and_one_backup(scene, monkeypatch, tmp_path):
    new_rsa = _keypair(tmp_path, "new_rsa3", "rsa")
    scene["presented"]["10.0.0.1"][new_rsa[0]] = new_rsa[1]
    original = scene["kh"].read_text()

    result, asked = _prompt(scene, monkeypatch, answer=True)

    assert len(asked) == 1
    assert "BLOCKED=\n" in result.output
    assert find_changed_host_keys(["10.0.0.1"], options=scene["opts"]) == []
    assert Path(str(scene["kh"]) + ".old").read_text() == original  # the file before this host, not in between


def test_a_host_accepted_in_the_gate_is_not_asked_again(scene, monkeypatch):
    _first, asked = _prompt(scene, monkeypatch, answer=True)
    scene["presented"]["10.0.0.1"] = {scene["old"][0]: scene["old"][1]}  # would look changed again
    again, asked_again = _prompt(scene, monkeypatch, answer=True)

    assert len(asked) == 1 and not asked_again
    assert "BLOCKED=\n" in again.output


def test_the_gate_does_not_repeat_the_hint_for_a_host_just_asked_about(scene, monkeypatch):
    from types import SimpleNamespace

    from sparkrun.api.setup import SshProbe
    from sparkrun.cli._setup import _ssh

    monkeypatch.setattr(_ssh, "_stdin_is_terminal", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda text, default: False)
    probes = [SshProbe(host="10.0.0.1", ok=False, host_key_failed=True), SshProbe(host="10.0.0.2", ok=False, host_key_failed=True)]
    monkeypatch.setattr("sparkrun.api.setup.probe_ssh_access", lambda *a, **k: probes)
    config = SimpleNamespace(ssh_options=scene["opts"], ssh_key=None)

    @click.command()
    def cmd():
        _ssh._ensure_ssh_access(["10.0.0.1", "10.0.0.2"], None, config)

    output = CliRunner().invoke(cmd, []).output
    assert output.count("ssh-keygen -R 10.0.0.1") == 1  # from the prompt only
    assert "ssh-keygen -R 10.0.0.2" in output  # never asked about, so the gate explains it
