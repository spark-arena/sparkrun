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
    script = "WHO=tester\n" + read_script("_sudo_nopasswd.sh") + read_script("_sudoers_probe.sh")
    full_env = {**os.environ, "PATH": "%s:%s" % (bin_dir, os.environ["PATH"]), "FAKE_SUDO_ALL": "1", "FAKE_LISTING": "", "FAKE_FILES": ""}
    full_env.update(env)
    out = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env=full_env, check=True).stdout
    facts = dict(line.split("=", 1) for line in out.split())
    assert facts.pop("CHECK_SUDO_NOPASSWD") == ("1" if full_env["FAKE_SUDO_ALL"] == "0" else "0")
    return facts


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


def create_discoverable_cluster(monkeypatch, manager, name, hosts, hardware=None):
    """A saved cluster whose setup discovery reports *hardware* (default DGX Spark).

    ``--save-sudo`` is held to the ``sudoers`` step's hardware plan, so tests
    that install entries need targets whose platform selects that step.
    """
    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun.core.setup_models import HostState
    from sparkrun.core.setup_probe import resolve_setup_context

    inventory = {host: hardware or default_dgx_spark_hardware() for host in hosts}
    manager.create(name, hosts, user="dgxuser", hosts_hardware=inventory)

    def discover(targets, **kwargs):
        assert kwargs["discovery_only"]
        states = {host: HostState(host, facts={"CHECK_OS": "Linux"}, hardware=inventory[host]) for host in targets}
        return states, resolve_setup_context(states, config=kwargs["config"], cluster=kwargs.get("cluster"), cluster_name=name)

    monkeypatch.setattr("sparkrun.core.setup_probe.probe_setup_hosts", discover)


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
    create_discoverable_cluster(monkeypatch, ClusterManager(config_root), "lab", HOSTS)
    return "lab"


def _fake_hosts(entries, passwords, nopasswd_action=False, nopasswd_hosts=()):
    """SSH-layer fakes: per-host sudo passwords and installed sudoers entries.

    *entries* maps host -> set of installed labels; installs mutate it, so a
    later ``sudo -n`` action succeeds where the entry now exists.
    *nopasswd_hosts* grant unrestricted passwordless sudo and must never be
    sent a password.
    """

    def parallel(hosts, script, **kwargs):
        results = []
        for host in hosts:
            if "CHECK_SUDOERS" in script:
                have = entries[host]
                out = "CHECK_SUDO_NOPASSWD=%d\nCHECK_SUDOERS_CHOWN=%d\nCHECK_SUDOERS_DROPCACHES=%d\n" % (
                    host in nopasswd_hosts,
                    "chown" in have,
                    "dropcaches" in have,
                )
                results.append(_ok(host, out))
            elif script == "sudo -n true":
                results.append(_ok(host) if host in nopasswd_hosts else RemoteResult(host, 1, "", "sudo: a password is required"))
            else:
                label = "dropcaches" if "drop_caches" in script else "chown"
                works = nopasswd_action or label in entries[host]
                results.append(_ok(host, "OK: done") if works else RemoteResult(host, 1, "", "sudo: a password is required"))
        return results

    def sudo(host, script, password, **kwargs):
        if host in nopasswd_hosts:
            assert password is None, "a password was sent to a NOPASSWD host"
        elif password != passwords[host]:
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


def test_cx7_apply_asks_a_rejecting_host_for_its_own_password():
    """setup cx7 and the wizard's CX7 phase share apply_cx7_plan's per-host retry."""
    from sparkrun.orchestration.networking import CX7ClusterPlan, CX7HostPlan, apply_cx7_plan

    host_plans = [CX7HostPlan(host=h, assignments=[mock.Mock(), mock.Mock()], needs_change=True) for h in HOSTS]
    plan = CX7ClusterPlan(host_plans=host_plans)
    host_pw = {"10.0.0.1": "pw-one", "10.0.0.2": "pw-two"}

    def configure(hp, mtu, prefix_len, ssh_kwargs=None, dry_run=False, sudo_password=None):
        return _ok(hp.host) if sudo_password == host_pw[hp.host] else _rejected(hp.host)

    prompt = mock.Mock(return_value="pw-two")
    with mock.patch("sparkrun.orchestration.networking.configure_cx7_host", side_effect=configure):
        results = apply_cx7_plan(plan, sudo_password="pw-one", sudo_hosts=set(HOSTS), passwords=SudoPasswords(prompt_host=prompt))
    assert all(r.success for r in results)
    prompt.assert_called_once_with("10.0.0.2")


# ---------------------------------------------------------------------------
# Review follow-ups: indirect su, NOPASSWD hosts, shared-password verification
# ---------------------------------------------------------------------------


def test_script_output_mentioning_passwords_is_not_an_auth_failure():
    """stdout belongs to the script; only sudo/su's stderr decides."""
    assert not is_sudo_auth_failure(RemoteResult("h", 1, "pam: Authentication failure while reading x", ""))


def test_nopasswd_hosts_get_sudo_n_never_a_password():
    passwords = SudoPasswords(shared="pw", nopasswd={"h1"})
    seen = []
    passwords.run("h1", lambda pw: seen.append(pw) or _ok("h1"))
    assert seen == [None]


def test_runner_never_sends_a_password_to_a_probed_nopasswd_host(monkeypatch):
    from sparkrun.api.setup import run_setup_steps

    state, context = state_context(CHECK_SUDO_NOPASSWD="1")
    approve_test_steps(monkeypatch, "needs_root")
    register_feature(FeatureFlag("setup.steps.needs_root", "needs_root", default=True))
    register_setup_step(
        SetupStep(
            "needs_root",
            "needs_root",
            checks=(lambda *_: CheckItem("needs_root", "needs_root", WARN),),
            apply=lambda s, ctx, action: SetupActionResult(s.host, OK if action.run(s.host, "true").success else "fail", "ran"),
            feature_flag="setup.steps.needs_root",
        )
    )
    dispatch = mock.Mock(return_value=_ok(state.host))
    run_setup_steps(
        {state.host: state}, context, SetupActionContext("tester", dispatch=dispatch), credentials=lambda: "pw", only_steps={"needs_root"}
    )
    assert [c.args[2] for c in dispatch.call_args_list] == [None]


def test_save_sudo_installs_on_a_nopasswd_host_without_a_password(cluster):
    """Mixed cluster: the NOPASSWD host must not have a password piped to sudo -S."""
    entries = {h: set() for h in HOSTS}
    patches = _fake_hosts(entries, dict.fromkeys(HOSTS, "pw"), nopasswd_hosts={"10.0.0.1"})
    result, sudo = _invoke(["clear-cache", "--save-sudo"], patches, "pw\n")
    assert result.exit_code == 0, result.output
    assert entries == {h: {"dropcaches"} for h in HOSTS}
    assert {c.args[0]: c.args[2] for c in sudo.call_args_list if "visudo" in c.args[1]} == {"10.0.0.1": None, "10.0.0.2": "pw"}


def test_action_failing_on_a_nopasswd_host_retries_as_root_without_asking(cluster):
    """A NOPASSWD host whose sudo -n action failed is retried via sudo -n, not with a prompt."""
    entries = {h: set() for h in HOSTS}
    result, sudo = _invoke(["clear-cache"], _fake_hosts(entries, {}, nopasswd_hosts=set(HOSTS)), "")
    assert result.exit_code == 0, result.output
    assert "[sudo] password" not in result.output
    assert {c.args[2] for c in sudo.call_args_list} == {None}


def _verify_patches(accepting, rejection="Sorry, try again.\nsudo: 1 incorrect password attempt"):
    """sudo -n fails everywhere; *accepting* maps host -> the password its sudo -S takes."""

    def parallel(hosts, script, **kwargs):
        return [RemoteResult(h, 1, "", "sudo: a password is required") for h in hosts]

    def sudo(host, script, password, **kwargs):
        return _ok(host) if accepting.get(host) == password else RemoteResult(host, 1, "", rejection)

    return (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=parallel),
        mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", side_effect=sudo),
    )


def _ensure(accepting, answers, **kwargs):
    from sparkrun.cli._setup._sudo import ensure_sudo_password

    patches = _verify_patches(accepting, **kwargs.pop("verify", {}))
    prompts = iter(answers)
    with patches[0], patches[1], mock.patch("click.prompt", side_effect=lambda *a, **k: next(prompts)) as prompt:
        result = ensure_sudo_password(HOSTS, "dgxuser", {"ssh_user": "dgxuser"}, allow_indirect=True, default_user="admin")
    return result, prompt


def test_shared_password_accepted_by_some_host_is_kept():
    """Host 0 differs — not a reason to abandon direct sudo for an indirect user."""
    (password, alt_user), prompt = _ensure({"10.0.0.2": "pw"}, ["pw"])
    assert (password, alt_user) == ("pw", None)
    assert prompt.call_count == 1


def test_password_every_host_rejects_is_asked_again():
    (password, alt_user), prompt = _ensure(dict.fromkeys(HOSTS, "right"), ["typo", "right"])
    assert (password, alt_user) == ("right", None)
    assert prompt.call_count == 2


def test_user_without_sudo_rights_is_offered_indirect_sudo():
    not_sudoer = {"rejection": "dgxuser is not in the sudoers file."}
    (password, alt_user), _prompt = _ensure({}, ["pw", "admin", "admin-pw"], verify=not_sudoer)
    assert (password, alt_user) == ("admin-pw", "admin")


def test_probe_default_accepts_a_chown_entry_for_any_cache_dir(tmp_path):
    """Without a requested dir, a --cache-dir entry is in effect: the wizard must not overwrite it."""
    listing = "    (root) NOPASSWD: /usr/bin/chown -R tester /data/hf"
    assert _probe(tmp_path, FAKE_LISTING=listing)["CHECK_SUDOERS_CHOWN"] == "1"
    # A requested dir must match exactly, not as a prefix.
    listing = "    (root) NOPASSWD: /usr/bin/chown -R tester /data/hf-old"
    assert _probe(tmp_path, FAKE_LISTING=listing, SPARKRUN_SUDOERS_CACHE_DIR="/data/hf")["CHECK_SUDOERS_CHOWN"] == "0"


# ---------------------------------------------------------------------------
# --save-sudo honors the sudoers step's platform plan and feature flag
# ---------------------------------------------------------------------------


def _generic_nvidia_cluster(tmp_path, monkeypatch):
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
    h100 = HostHardware(accelerators=[AcceleratorSpec("nvidia", "h100", count=8, memory_gb=80, capabilities=frozenset({"cuda"}))])
    create_discoverable_cluster(monkeypatch, ClusterManager(config_root), "lab", HOSTS, hardware=h100)


def test_save_sudo_refused_where_the_platform_plan_omits_sudoers(tmp_path, monkeypatch):
    """Generic NVIDIA recognition does not qualify OS changes such as sudoers entries."""
    _generic_nvidia_cluster(tmp_path, monkeypatch)
    entries = {h: set() for h in HOSTS}
    result, sudo = _invoke(["clear-cache", "--save-sudo"], _fake_hosts(entries, dict.fromkeys(HOSTS, "pw")), "pw\n")
    assert result.exit_code != 0
    assert "Setup step sudoers is unavailable" in result.output
    assert "[sudo] password" not in result.output
    sudo.assert_not_called()


def test_clear_cache_itself_is_not_a_setup_step(tmp_path, monkeypatch):
    """Without --save-sudo the command runs on any platform, as before."""
    _generic_nvidia_cluster(tmp_path, monkeypatch)
    entries = {h: {"dropcaches"} for h in HOSTS}
    result, _sudo = _invoke(["clear-cache"], _fake_hosts(entries, dict.fromkeys(HOSTS, "pw")), "")
    assert result.exit_code == 0, result.output
    assert "2 cleared" in result.output


def test_save_sudo_refused_when_the_sudoers_step_is_disabled(cluster, monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_SETUP_STEPS_SUDOERS", "0")
    entries = {h: set() for h in HOSTS}
    result, sudo = _invoke(["clear-cache", "--save-sudo"], _fake_hosts(entries, dict.fromkeys(HOSTS, "pw")), "pw\n")
    assert result.exit_code != 0
    assert "disabled by application or user policy" in result.output
    sudo.assert_not_called()


@pytest.mark.parametrize("exclude_sudo_steps", [False, True])
def test_nopasswd_probe_runs_only_for_selected_sudo_steps(v, exclude_sudo_steps):
    """Probes follow the host's plan: no step acting with sudo, no sudo probe."""
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun.core.setup_probe import probe_setup_hosts
    from sparkrun.core.setup_steps import all_setup_steps, register_setup_constraint
    from test_setup_steps import FACTS

    if exclude_sudo_steps:
        sudo_steps = {step.key for step in all_setup_steps() if step.requires_sudo and step.apply is not None}
        register_setup_constraint("no-sudo", lambda key, state, context: "test" if key in sudo_steps else "")
    scripts = []
    stdout = "SPARKRUN_PROBE_ACCEL_END\n" + "\n".join(key + "=" + value for key, value in FACTS.items())

    def run(host, script, *a, **kw):
        scripts.append(script)
        return RemoteResult(host, 0, stdout, "")

    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_script", side_effect=run),
        mock.patch("sparkrun.core.hardware_probe._parse_probe_result", return_value=default_dgx_spark_hardware()),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={}),
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value={}),
    ):
        probe_setup_hosts(["h1"], ssh_kwargs={}, config=SparkrunConfig())
    readiness = next(s for s in scripts if "SETUP_STEPS=" in s)
    assert ("if [ 0 = 1 ]" if exclude_sudo_steps else "if [ 1 = 1 ]") in readiness


def test_indirect_wrapper_reports_a_rejected_su_password(tmp_path, monkeypatch):
    """Run the real pty wrapper against a fake su: the rejection must be detectable."""
    from sparkrun.orchestration import ssh as ssh_mod
    from sparkrun.orchestration.sudo import run_indirect_sudo_script

    fake_su = tmp_path / "su"
    # Echoes what it read, as a pty would if the password arrived before echo was off.
    fake_su.write_text('#!/bin/bash\nprintf "Password: "\nread -r pw\necho "$pw"\necho "su: Authentication failure"\nexit 1\n')
    fake_su.chmod(0o755)
    monkeypatch.setenv("PATH", "%s:%s" % (tmp_path, os.environ["PATH"]))
    # Run the wrapper here instead of over SSH: the "ssh command" is a local shell.
    monkeypatch.setattr(ssh_mod, "build_ssh_cmd", lambda host, **kw: ["bash", "-c"])

    result = run_indirect_sudo_script("h1", "true", sudo_user="admin", sudo_password="wrong-pw", timeout=30)
    assert not result.success
    assert is_sudo_auth_failure(result), (result.stdout, result.stderr)
    assert "wrong-pw" not in result.stdout + result.stderr


def test_unreachable_host_does_not_turn_a_typo_into_indirect_sudo():
    """Only hosts that asked for a password vote on whether the password was wrong."""
    from sparkrun.cli._setup._sudo import ensure_sudo_password

    def parallel(hosts, script, **kwargs):
        return [
            RemoteResult(h, 255, "", "ssh: connect to host %s: No route to host" % h)
            if h == "10.0.0.2"
            else RemoteResult(h, 1, "", "sudo: a password is required")
            for h in hosts
        ]

    verified = []

    def sudo(host, script, password, **kwargs):
        verified.append(host)
        return _ok(host) if password == "right" else _rejected(host)

    answers = iter(["typo", "right"])
    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=parallel),
        mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", side_effect=sudo),
        mock.patch("click.prompt", side_effect=lambda *a, **k: next(answers)),
    ):
        result = ensure_sudo_password(HOSTS, "dgxuser", {"ssh_user": "dgxuser"}, allow_indirect=True, default_user="admin")
    assert result == ("right", None)
    assert set(verified) == {"10.0.0.1"}


def test_no_prompt_when_every_host_is_passwordless_or_unreachable():
    from sparkrun.cli._setup._sudo import ensure_sudo_password

    def parallel(hosts, script, **kwargs):
        return [_ok(h) if h == "10.0.0.1" else RemoteResult(h, 255, "", "ssh: No route to host") for h in hosts]

    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=parallel),
        mock.patch("click.prompt") as prompt,
    ):
        assert ensure_sudo_password(HOSTS, "dgxuser", {"ssh_user": "dgxuser"}) == (None, None)
    prompt.assert_not_called()


def test_cx7_apply_leaves_passwordless_hosts_to_their_own_sudo_and_routes_indirect():
    """Only sudo_hosts get a password; with a dispatcher (indirect sudo) they go through it."""
    from sparkrun.orchestration.networking import CX7ClusterPlan, CX7HostPlan, CX7InterfaceAssignment, apply_cx7_plan

    def plan_for(host):
        assignments = [CX7InterfaceAssignment("if%d" % n, "192.168.1%d.%s" % (n, host[-1]), "192.168.1%d.0/24" % n) for n in range(2)]
        return CX7HostPlan(host=host, assignments=assignments, needs_change=True)

    plan = CX7ClusterPlan(host_plans=[plan_for(h) for h in HOSTS])
    configured = {}
    dispatched = {}

    def configure(hp, mtu, prefix_len, ssh_kwargs=None, dry_run=False, sudo_password=None):
        configured[hp.host] = sudo_password
        return _ok(hp.host)

    def dispatch(host, script, password, timeout=60):
        dispatched[host] = password
        return _ok(host)

    with mock.patch("sparkrun.orchestration.networking.configure_cx7_host", side_effect=configure):
        apply_cx7_plan(plan, sudo_password="pw", sudo_hosts={"10.0.0.2"}, passwords=SudoPasswords(), dispatch=dispatch)
    assert configured == {"10.0.0.1": None}  # passwordless host: its own sudo, no password
    assert dispatched == {"10.0.0.2": "pw"}


def test_a_localized_sudo_message_still_means_a_password_is_needed():
    """sudo's text follows the host's locale (ssh forwards LANG); the exit code does not."""
    from sparkrun.cli._setup._sudo import ensure_sudo_password

    def parallel(hosts, script, **kwargs):
        return [RemoteResult(h, 1, "", "sudo: Ein Passwort ist notwendig") for h in hosts]

    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=parallel),
        mock.patch("sparkrun.orchestration.ssh.run_remote_sudo_script", side_effect=lambda host, *a, **k: _ok(host)),
        mock.patch("click.prompt", return_value="pw") as prompt,
    ):
        assert ensure_sudo_password(HOSTS, "dgxuser", {"ssh_user": "dgxuser"}) == ("pw", None)
    prompt.assert_called_once()
