"""Per-host sudo credentials and the shared sudoers probe/installer.

Scenarios for mixed per-host sudo passwords are adapted from PR #310: a
cluster's hosts need not share a password, so the one typed at the shared
prompt can be rejected on some of them.
"""

from __future__ import annotations

import os
import subprocess
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.features import FeatureFlag, register_feature
from sparkrun.core.setup_actions import SetupActionContext, SetupActionResult
from sparkrun.core.setup_models import CheckItem, OK, WARN
from sparkrun.core.setup_steps import SetupStep, apply_setup_step, register_setup_step
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.orchestration.sudo import SudoPasswords, is_sudo_auth_failure
from test_setup_steps import approve_test_steps, state_context

REJECTED = "Sorry, try again.\nsudo: 1 incorrect password attempt"


def _ok(host, out="OK"):
    return RemoteResult(host, 0, out, "")


def _rejected(host):
    return RemoteResult(host, 1, "", REJECTED)


# ---------------------------------------------------------------------------
# SudoPasswords
# ---------------------------------------------------------------------------


def test_auth_failure_detection_ignores_script_failures():
    assert is_sudo_auth_failure(_rejected("h"))
    assert is_sudo_auth_failure(RemoteResult("h", 1, "", "sudo: a password is required"))
    assert not is_sudo_auth_failure(RemoteResult("h", 1, "ERROR: sudoers validation failed", ""))
    assert not is_sudo_auth_failure(_ok("h"))


def test_rejected_host_is_asked_once_and_its_password_reused():
    prompt = mock.Mock(return_value="host-pw")
    passwords = SudoPasswords(shared="shared-pw", prompt_host=prompt)
    seen = []

    def attempt(password):
        seen.append(password)
        return _ok("h2") if password == "host-pw" else _rejected("h2")

    assert passwords.run("h2", attempt).success
    assert passwords.run("h2", attempt).success  # later action: no new prompt
    assert seen == ["shared-pw", "host-pw", "host-pw"]
    prompt.assert_called_once_with("h2")


def test_script_failure_is_not_a_password_problem():
    prompt = mock.Mock()
    passwords = SudoPasswords(shared="pw", prompt_host=prompt)
    result = passwords.run("h", lambda pw: RemoteResult("h", 1, "ERROR: sudoers validation failed", ""))
    assert not result.success
    prompt.assert_not_called()


def test_headless_rejection_fails_without_prompting():
    result = SudoPasswords(shared="pw").run("h", lambda pw: _rejected("h"))
    assert is_sudo_auth_failure(result)


def test_shared_password_is_collected_lazily_once():
    shared = mock.Mock(return_value="pw")
    passwords = SudoPasswords(prompt_shared=shared)
    assert passwords.password_for("h1") == "pw"
    assert passwords.password_for("h2") == "pw"
    shared.assert_called_once_with()
    assert "pw" not in repr(passwords)


def test_action_context_retries_rejected_host():
    dispatch = mock.Mock(side_effect=lambda host, script, pw, timeout: _ok(host) if pw == "own" else _rejected(host))
    action = SetupActionContext("tester", sudo_password="shared", dispatch=dispatch, passwords=SudoPasswords(prompt_host=lambda h: "own"))
    assert action.run("h1", "true").success
    assert [c.args[2] for c in dispatch.call_args_list] == ["shared", "own"]


# ---------------------------------------------------------------------------
# Shared runner: host_credentials is the API/wizard seam
# ---------------------------------------------------------------------------


def test_runner_reuses_a_hosts_password_across_steps(monkeypatch):
    from sparkrun.api.setup import run_setup_steps

    state, context = state_context()
    approve_test_steps(monkeypatch, "first", "second")
    for key in ("first", "second"):
        register_feature(FeatureFlag("setup.steps." + key, key, default=True))
        register_setup_step(
            SetupStep(
                key,
                key,
                checks=(lambda *_, key=key: CheckItem(key, key, WARN),),
                apply=lambda s, ctx, action: SetupActionResult(s.host, OK if action.run(s.host, "true").success else "fail", "ran"),
                feature_flag="setup.steps." + key,
            )
        )
    dispatch = mock.Mock(side_effect=lambda host, script, pw, timeout: _ok(host) if pw == "own" else _rejected(host))
    host_credentials = mock.Mock(return_value="own")
    result = run_setup_steps(
        {state.host: state},
        context,
        SetupActionContext("tester", dispatch=dispatch),
        credentials=lambda: "shared",
        host_credentials=host_credentials,
        only_steps={"first", "second"},
    )
    assert result.steps == {"first": OK, "second": OK}
    host_credentials.assert_called_once_with(state.host)
    assert [c.args[2] for c in dispatch.call_args_list] == ["shared", "own", "own"]


def test_sudoers_step_installs_only_missing_entries():
    state, ctx = state_context(CHECK_SUDOERS_CHOWN="1", CHECK_SUDOERS_DROPCACHES="0")
    dispatch = mock.Mock(return_value=_ok("h1", "OK: installed"))
    result = apply_setup_step("sudoers", state, ctx, SetupActionContext("tester", dispatch=dispatch))
    assert result.status == OK
    assert result.extra["files"] == ["/etc/sudoers.d/sparkrun-dropcaches-tester"]
    assert dispatch.call_count == 1


# ---------------------------------------------------------------------------
# The sudoers probe fragment, under real bash with a fake sudo
# ---------------------------------------------------------------------------

_FAKE_SUDO = """#!/bin/bash
case "$*" in
    "-n true") exit "$FAKE_SUDO_ALL" ;;
    "-n -l")
        if [ -n "$FAKE_LISTING" ]; then printf '%s\\n' "$FAKE_LISTING"; exit 0; fi
        echo "sudo: a password is required" >&2; exit 1 ;;
    "-n test -e "*) case " $FAKE_FILES " in *" ${*: -1} "*) exit 0 ;; esac; exit 1 ;;
esac
exit 1
"""


def _probe(tmp_path, **env):
    from sparkrun.scripts import read_script

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "sudo").write_text(_FAKE_SUDO)
    (bin_dir / "getent").write_text("#!/bin/bash\necho 'tester:x:1000:1000::/home/tester:/bin/bash'\n")
    for name in ("sudo", "getent"):
        (bin_dir / name).chmod(0o755)
    script = "WHO=tester\n" + read_script("_sudoers_probe.sh")
    full_env = {**os.environ, "PATH": "%s:%s" % (bin_dir, os.environ["PATH"]), "FAKE_SUDO_ALL": "1", "FAKE_LISTING": "", "FAKE_FILES": ""}
    full_env.update(env)
    out = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env=full_env, check=True).stdout
    return dict(line.split("=", 1) for line in out.split())


def test_probe_reads_scoped_entries_on_password_sudo_hosts(tmp_path):
    """The case the old probe called "unknown": password sudo plus our scoped entry."""
    listing = "    (ALL : ALL) ALL\n    (root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches"
    facts = _probe(tmp_path, FAKE_LISTING=listing)
    assert facts == {"CHECK_SUDOERS_CHOWN": "0", "CHECK_SUDOERS_DROPCACHES": "1"}


def test_probe_matches_the_exact_nopasswd_command(tmp_path):
    """Our entry for this user and cache dir, on a NOPASSWD line — nothing looser."""
    listing = "    (ALL : ALL) ALL\n    (root) NOPASSWD: /usr/bin/chown -R tester /home/tester/.cache/huggingface"
    assert _probe(tmp_path, FAKE_LISTING=listing)["CHECK_SUDOERS_CHOWN"] == "1"
    assert _probe(tmp_path, FAKE_LISTING=listing, SPARKRUN_SUDOERS_CACHE_DIR="/data/hf")["CHECK_SUDOERS_CHOWN"] == "0"
    assert _probe(tmp_path, FAKE_LISTING="    (root) NOPASSWD: /usr/bin/true")["CHECK_SUDOERS_DROPCACHES"] == "0"
    # The same command under a password-requiring rule is not passwordless.
    listing = "    (root) NOPASSWD: /usr/bin/true\n    (root) /usr/bin/tee /proc/sys/vm/drop_caches"
    assert _probe(tmp_path, FAKE_LISTING=listing)["CHECK_SUDOERS_DROPCACHES"] == "0"


def test_probe_without_any_nopasswd_rule_reports_missing(tmp_path):
    assert _probe(tmp_path) == {"CHECK_SUDOERS_CHOWN": "0", "CHECK_SUDOERS_DROPCACHES": "0"}


def test_probe_with_unrestricted_sudo_checks_the_files(tmp_path):
    facts = _probe(tmp_path, FAKE_SUDO_ALL="0", FAKE_FILES="/etc/sudoers.d/sparkrun-chown-tester")
    assert facts == {"CHECK_SUDOERS_CHOWN": "1", "CHECK_SUDOERS_DROPCACHES": "0"}


# ---------------------------------------------------------------------------
# CLI: --save-sudo and the action, on a mixed-password cluster
# ---------------------------------------------------------------------------

HOSTS = ["10.0.0.1", "10.0.0.2"]


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
    ClusterManager(config_root).create("lab", HOSTS, user="dgxuser")
    return "lab"


def _fake_hosts(entries, passwords, nopasswd_action=False):
    """SSH-layer fakes: per-host sudo passwords and installed sudoers entries.

    *entries* maps host -> set of installed labels; installs mutate it, so a
    later ``sudo -n`` action succeeds where the entry now exists.
    """

    def parallel(hosts, script, **kwargs):
        results = []
        for host in hosts:
            if "CHECK_SUDOERS" in script:
                have = entries[host]
                out = "CHECK_SUDOERS_CHOWN=%d\nCHECK_SUDOERS_DROPCACHES=%d\n" % ("chown" in have, "dropcaches" in have)
                results.append(_ok(host, out))
            else:
                label = "dropcaches" if "drop_caches" in script else "chown"
                works = nopasswd_action or label in entries[host]
                results.append(_ok(host, "OK: done") if works else RemoteResult(host, 1, "", "sudo: a password is required"))
        return results

    def sudo(host, script, password, **kwargs):
        if password != passwords[host]:
            return _rejected(host)
        if "visudo" in script:
            label = "dropcaches" if "dropcaches" in script else "chown"
            entries[host].add(label)
            return _ok(host, "OK: installed sudoers entry for %s" % label)
        return _ok(host, "OK: done")

    return (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=parallel),
        mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", side_effect=sudo),
    )


def _invoke(args, patches, input_text):
    with patches[0], patches[1] as sudo:
        result = CliRunner().invoke(main, ["setup", *args, "--cluster", "lab"], input=input_text)
    return result, sudo


def test_clear_cache_save_sudo_converges_on_mixed_passwords(cluster):
    """Shared prompt, one per-host prompt, both entries installed, then the clear runs passwordless."""
    entries = {h: set() for h in HOSTS}
    result, sudo = _invoke(
        ["clear-cache", "--save-sudo"], _fake_hosts(entries, {"10.0.0.1": "pw-one", "10.0.0.2": "pw-two"}), "pw-one\npw-two\n"
    )
    assert result.exit_code == 0, result.output
    assert "[sudo] password for dgxuser @ 10.0.0.2" in result.output
    assert "Sudoers install: 2 OK, 0 failed." in result.output
    assert "2 cleared" in result.output
    # Only the command's own entry — never the chown one.
    assert entries == {h: {"dropcaches"} for h in HOSTS}
    assert all("dropcaches" in c.args[1] for c in sudo.call_args_list)


def test_save_sudo_skips_hosts_that_already_have_the_entry(cluster):
    entries = {"10.0.0.1": {"dropcaches"}, "10.0.0.2": set()}
    result, sudo = _invoke(["clear-cache", "--save-sudo"], _fake_hosts(entries, dict.fromkeys(HOSTS, "pw")), "pw\n")
    assert result.exit_code == 0, result.output
    assert "on 1 host(s)" in result.output
    assert [c.args[0] for c in sudo.call_args_list] == ["10.0.0.2"]


def test_save_sudo_never_prompts_when_every_host_has_the_entry(cluster):
    entries = {h: {"dropcaches"} for h in HOSTS}
    result, sudo = _invoke(["clear-cache", "--save-sudo"], _fake_hosts(entries, dict.fromkeys(HOSTS, "pw")), "")
    assert result.exit_code == 0, result.output
    assert "already in effect on every host" in result.output
    assert "[sudo] password" not in result.output
    sudo.assert_not_called()


def test_fix_permissions_asks_a_rejecting_host_for_its_own_password(cluster):
    """No --save-sudo: the action itself retries the rejecting host individually."""
    entries = {h: set() for h in HOSTS}
    result, sudo = _invoke(["fix-permissions"], _fake_hosts(entries, {"10.0.0.1": "pw-one", "10.0.0.2": "pw-two"}), "pw-one\npw-two\n")
    assert result.exit_code == 0, result.output
    assert "2 fixed" in result.output
    assert result.output.count("[sudo] password for dgxuser @") == 1
    assert entries == {h: set() for h in HOSTS}  # nothing installed without --save-sudo


def test_fix_permissions_save_sudo_installs_only_chown_for_the_cache_dir(cluster):
    entries = {h: set() for h in HOSTS}
    result, sudo = _invoke(
        ["fix-permissions", "--save-sudo", "--cache-dir", "/data/hf"], _fake_hosts(entries, dict.fromkeys(HOSTS, "pw")), "pw\n"
    )
    assert result.exit_code == 0, result.output
    assert entries == {h: {"chown"} for h in HOSTS}
    assert all("/data/hf" in c.args[1] for c in sudo.call_args_list if "visudo" in c.args[1])


def test_install_failure_that_is_not_auth_does_not_reprompt(cluster):
    entries = {h: set() for h in HOSTS}
    patches = _fake_hosts(entries, dict.fromkeys(HOSTS, "pw"), nopasswd_action=True)
    broken = RemoteResult("10.0.0.2", 1, "ERROR: sudoers validation failed", "")
    original = patches[1].kwargs["side_effect"]
    patches[1].kwargs["side_effect"] = lambda host, script, pw, **kw: broken if host == "10.0.0.2" else original(host, script, pw, **kw)
    result, _sudo = _invoke(["clear-cache", "--save-sudo"], patches, "pw\n")
    assert "@ 10.0.0.2" not in result.output
    assert "Sudoers install: 1 OK, 1 failed." in result.output
    assert "2 cleared" in result.output
