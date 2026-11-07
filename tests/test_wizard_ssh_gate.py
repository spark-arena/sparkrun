"""Regression tests for the wizard's SSH access gate and user defaults.

These pin the two failures a Windows control machine hit on a fresh cluster:
an SSH username of ``root`` invented from an unset POSIX ``$USER``, and probes
firing over SSH before the wizard had asked who to connect as.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.api.setup import LocalSshKey, SshProbe
from sparkrun.cli import main
from sparkrun.cli._setup._ssh import _default_ssh_user, _ensure_ssh_access
from sparkrun.core.cluster_manager import ClusterManager


@pytest.fixture(autouse=True)
def _controller_without_ssh_server(monkeypatch):
    """Wizard fixtures describe the cluster, not the developer's SSH server."""
    monkeypatch.setattr("sparkrun.utils.net.local_ip_for", lambda host: "192.0.2.254")
    monkeypatch.setattr("sparkrun.utils.net.accepts_tcp", lambda *a, **kw: False)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def patched_cluster_mgr(tmp_path):
    config_root = tmp_path / "wizard_config"
    config_root.mkdir(parents=True, exist_ok=True)
    mgr = ClusterManager(config_root)
    with (
        mock.patch("sparkrun.cli._common._get_cluster_manager", return_value=mgr),
        mock.patch("sparkrun.cli._setup._get_cluster_manager", return_value=mgr),
        mock.patch("sparkrun.cli._setup._sudo._get_cluster_manager", return_value=mgr),
    ):
        yield mgr


# ---------------------------------------------------------------------------
# _default_ssh_user
# ---------------------------------------------------------------------------


def test_default_ssh_user_never_invents_root(monkeypatch):
    """On Windows $USER is unset; the old code fell back to 'root'.

    Every DGX Spark has a root account, so the guess produced a clean
    "Permission denied" rather than an obvious misconfiguration.
    """
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.delenv("LNAME", raising=False)
    monkeypatch.setenv("USERNAME", "DrewOnWindows")

    assert _default_ssh_user() == "DrewOnWindows"


def test_default_ssh_user_prefers_posix_user(monkeypatch):
    monkeypatch.setenv("LOGNAME", "ubuntu")
    assert _default_ssh_user() == "ubuntu"


def test_default_ssh_user_returns_empty_when_undeterminable():
    """Empty means "prompt without a default" — better than a wrong default."""
    with mock.patch("getpass.getuser", side_effect=OSError("no such user")):
        assert _default_ssh_user() == ""


# ---------------------------------------------------------------------------
# _ensure_ssh_access
# ---------------------------------------------------------------------------


def _probes(**by_host):
    return [SshProbe(host=h, **kw) for h, kw in by_host.items()]


def test_gate_passes_through_when_all_hosts_reachable(capsys):
    with mock.patch(
        "sparkrun.api.setup.probe_ssh_access",
        return_value=_probes(h1={"ok": True}, h2={"ok": True}),
    ):
        outcome = _ensure_ssh_access(["h1", "h2"], "ubuntu", None)

    assert outcome.all_ok
    assert outcome.ok_hosts == ["h1", "h2"]
    assert not outcome.bootstrapped


def test_gate_dry_run_makes_no_connection():
    with mock.patch("sparkrun.api.setup.probe_ssh_access") as m:
        outcome = _ensure_ssh_access(["h1"], "ubuntu", None, dry_run=True)
    m.assert_not_called()
    assert outcome.ok_hosts == ["h1"]


def test_gate_under_yes_advises_instead_of_prompting(capsys):
    """--yes must not hang waiting for a password."""
    with (
        mock.patch(
            "sparkrun.api.setup.probe_ssh_access",
            return_value=_probes(h1={"ok": False, "auth_failed": True, "error": "Permission denied"}),
        ),
        mock.patch("sparkrun.api.setup.install_public_key_interactive") as install,
    ):
        outcome = _ensure_ssh_access(["h1"], "ubuntu", None, yes=True)

    install.assert_not_called()
    assert outcome.blocked == ["h1"]
    assert "Re-run without --yes" in capsys.readouterr().err


def test_gate_does_not_offer_bootstrap_for_changed_host_key(capsys):
    """A changed host key is a security event, not a credential to install."""
    with (
        mock.patch(
            "sparkrun.api.setup.probe_ssh_access",
            return_value=_probes(h1={"ok": False, "host_key_failed": True, "error": "IDENTIFICATION HAS CHANGED"}),
        ),
        mock.patch("sparkrun.api.setup.install_public_key_interactive") as install,
    ):
        outcome = _ensure_ssh_access(["h1"], "ubuntu", None)

    install.assert_not_called()
    assert outcome.blocked == ["h1"]
    assert "ssh-keygen -R h1" in capsys.readouterr().err


def test_gate_verifies_with_the_key_it_just_generated():
    """A fresh controller has no ``ssh.key``, so the gate generates one and installs it.

    The re-probe must offer that key. It used the key read from config before the
    key existed (``None``), so every host stayed "Still failing" and the wizard
    skipped every later setup step as unreachable.
    """
    key = LocalSshKey(
        path=Path("/cfg/ssh/sparkrun_ed25519"),
        public_path=Path("/cfg/ssh/sparkrun_ed25519.pub"),
        public_key="ssh-ed25519 AAAA sparkrun@test",
        generated=True,
    )
    config = mock.Mock(ssh_key=None, ssh_options=[], config_path="/cfg/config.yaml")

    def probe(hosts, user, *, key=None, options=None):
        accepted = key == "/cfg/ssh/sparkrun_ed25519"
        return _probes(**{h: {"ok": accepted, "auth_failed": not accepted} for h in hosts})

    with (
        mock.patch("sparkrun.api.setup.probe_ssh_access", side_effect=probe),
        mock.patch("sparkrun.api.setup._ssh_access.ensure_local_key", return_value=key),
        mock.patch("sparkrun.orchestration.ssh.offered_identities", return_value=[]),
        mock.patch("sparkrun.api.setup.install_public_key_interactive") as install,
        # "Try a different SSH username first?" -> no; "Set up key-based SSH access?" -> yes
        mock.patch("click.confirm", side_effect=[False, True]),
    ):
        outcome = _ensure_ssh_access(["h1"], "ubuntu", config)

    install.assert_called_once()
    assert outcome.ok_hosts == ["h1"]
    assert not outcome.blocked


def test_gate_does_not_offer_bootstrap_for_unreachable_hosts(capsys):
    with (
        mock.patch(
            "sparkrun.api.setup.probe_ssh_access",
            return_value=_probes(h1={"ok": False, "error": "No route to host"}),
        ),
        mock.patch("sparkrun.api.setup.install_public_key_interactive") as install,
    ):
        outcome = _ensure_ssh_access(["h1"], "ubuntu", None)

    install.assert_not_called()
    assert outcome.blocked == ["h1"]
    assert "Not reachable" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Wizard ordering
# ---------------------------------------------------------------------------


def test_wizard_asks_for_user_before_any_ssh(runner, v, patched_cluster_mgr):
    """The SSH username prompt must precede the first outbound connection.

    Previously remote CX7 detection ran first, so a fresh cluster produced four
    "Permission denied" failures as a guessed user before the wizard ever asked
    which user to use.
    """
    order = []

    def record_probe(hosts, user, **kw):
        order.append("probe:%s" % user)
        return [SshProbe(host=h, ok=True) for h in hosts]

    def record_cx7(hosts, **kw):
        order.append("cx7")
        return {h: mock.Mock(detected=False) for h in hosts}

    with (
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")),
        mock.patch("sparkrun.api.setup.probe_ssh_access", side_effect=record_probe),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", side_effect=record_cx7),
        mock.patch("sparkrun.models.distribute.detect_shared_cache", return_value=False),
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
    ):
        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", "10.0.0.1,10.0.0.2", "--cluster", "ordered"],
            # --cluster is given, so the first prompt is the SSH username;
            # then decline every later phase
            input="ubuntu\nn\nn\nn\nn\nn\nn\nn\n",
        )

    assert result.exit_code == 0, result.output
    # The access probe ran first, and it ran as the user we typed.
    assert order[0] == "probe:ubuntu", order
    assert "cx7" not in order[:1]
    # Prompt ordering in the transcript itself.
    assert result.output.index("SSH username") < result.output.index("Checking SSH access")


def test_wizard_skips_probes_when_ssh_access_fails(runner, v, patched_cluster_mgr):
    """No host authenticated → don't fan out CX7 / shared-cache probes at all."""
    with (
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")),
        mock.patch(
            "sparkrun.api.setup.probe_ssh_access",
            return_value=[SshProbe(host="10.0.0.1", ok=False, auth_failed=True, error="Permission denied")],
        ),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as cx7,
        mock.patch("sparkrun.models.distribute.detect_shared_cache") as shared,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
    ):
        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", "10.0.0.1", "--cluster", "blocked", "--yes"],
        )

    assert result.exit_code == 0, result.output
    cx7.assert_not_called()
    shared.assert_not_called()


def test_wizard_uses_corrected_user_for_the_created_cluster(runner, v, patched_cluster_mgr):
    """Correcting the username at the gate must stick, not just unblock SSH."""
    seen_users = []

    def record_probe(hosts, user, **kw):
        seen_users.append(user)
        ok = user == "spark"
        return [SshProbe(host=h, ok=ok, auth_failed=not ok, error="Permission denied") for h in hosts]

    with (
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")),
        mock.patch("sparkrun.api.setup.probe_ssh_access", side_effect=record_probe),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={}),
        mock.patch("sparkrun.models.distribute.detect_shared_cache", return_value=False),
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
    ):
        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", "10.0.0.1", "--cluster", "fixeduser"],
            # ssh user (wrong), try a different user? y, spark,
            # then decline the remaining phases
            input="notme\ny\nspark\nn\nn\nn\nn\nn\nn\n",
        )

    assert result.exit_code == 0, result.output
    assert seen_users == ["notme", "spark"]
    assert patched_cluster_mgr.get("fixeduser").user == "spark"


def test_wizard_audits_identities_after_access_even_when_mesh_is_declined(runner, v, patched_cluster_mgr):
    from sparkrun.api.setup import DuplicateHostIdentity, SshHostIdentity, SshIdentityReport
    from test_setup_check import _facts_kv, _FACTS_ALL_GOOD
    from sparkrun.orchestration.ssh import RemoteResult

    hosts = ["10.0.0.1", "10.0.0.2"]
    report = SshIdentityReport(
        identities=tuple(SshHostIdentity(h) for h in hosts),
        duplicates=(DuplicateHostIdentity("machine-id", "a" * 64, tuple(hosts)),),
    )
    order = []

    def access(hosts, user, **kw):
        order.append("access")
        return [SshProbe(host=h, ok=True) for h in hosts]

    def audit(hosts, **kw):
        order.append("identity")
        assert kw["ssh_kwargs"]["ssh_user"] == "ubuntu"
        return report

    with (
        mock.patch(
            "sparkrun.orchestration.ssh.run_remote_script",
            side_effect=lambda h, *a, **kw: RemoteResult(h, 0, _facts_kv(_FACTS_ALL_GOOD), ""),
        ),
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value={}),
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")),
        mock.patch("sparkrun.api.setup.probe_ssh_access", side_effect=access),
        mock.patch("sparkrun.api.setup.check_ssh_host_identities", side_effect=audit) as check,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={}),
        mock.patch("sparkrun.models.distribute.detect_shared_cache", return_value=False),
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
    ):
        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", ",".join(hosts), "--cluster", "identity-test"],
            input="ubuntu\nn\nn\nn\nn\nn\nn\nn\n",
        )
    assert result.exit_code == 0, result.output
    assert order[:2] == ["access", "identity"]
    check.assert_called_once()
    assert "duplicate machine-id" in result.output
    assert "ssh-rotate-keys" in result.output and "--machine-id" in result.output
