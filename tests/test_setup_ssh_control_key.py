"""``setup ssh`` authorizes the key sparkrun authenticates with.

Regression for: ``setup ssh`` installed a stock ``~/.ssh`` identity while later
commands connected with ``-i <ssh.key>``.  ssh drops its default identities when
``-i`` names an existing key, so ``ssh user@host`` worked and every sparkrun
command (and setup's own later phases) failed with "Permission denied (publickey)"
until the wizard's access gate installed the right key.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.api.setup import LocalSshKey, SshIdentityReport
from sparkrun.cli import main
from sparkrun.cli._setup._ssh import SshAccessOutcome
from sparkrun.orchestration.ssh import RemoteResult

HOSTS = ["10.0.0.1", "10.0.0.2"]


@pytest.fixture
def config_root(tmp_path, monkeypatch):
    import sparkrun.core.config

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", root)
    monkeypatch.setattr("sparkrun.cli._setup._ssh._mesh_uses_native", lambda: False)
    # No ssh-config IdentityFile unless a test says so (never the developer's ~/.ssh/config).
    monkeypatch.setattr("sparkrun.orchestration.ssh.offered_identities", mock.Mock(return_value=[]))
    return root


@pytest.fixture
def configured_key(tmp_path, config_root):
    """An ``ssh.key`` that is not one of the stock ``~/.ssh`` identities."""
    key = tmp_path / "keys" / "cluster_ed25519"
    key.parent.mkdir()
    key.write_text("PRIVATE")
    key.with_name(key.name + ".pub").write_text("ssh-ed25519 AAAACONFIGURED cluster\n")
    (config_root / "config.yaml").write_text(yaml.safe_dump({"ssh": {"key": str(key)}}))
    return key


def _invoke(*extra):
    return CliRunner().invoke(main, ["setup", "ssh", "--hosts", ",".join(HOSTS), "--no-include-self", "--no-discover-ips", *extra])


def _ok(hosts, script, **kwargs):
    return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]


def test_mesh_script_installs_the_configured_ssh_key(monkeypatch, configured_key):
    """Same user on both ends: the case where the script previously installed nothing of its own."""
    monkeypatch.setenv("USER", "testuser")
    verify = mock.Mock(side_effect=_ok)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", verify)
    monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", mock.Mock(return_value=SshIdentityReport()))

    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as run:
        result = _invoke("--user", "testuser")

    assert result.exit_code == 0, result.output
    env = run.call_args.kwargs["env"]
    assert env["SPARKRUN_CONTROL_PUBKEY"] == "ssh-ed25519 AAAACONFIGURED cluster"
    assert env["SPARKRUN_CONTROL_KEY"] == str(configured_key)
    # The post-mesh check connects exactly as later commands will.
    assert verify.call_args.kwargs["ssh_key"] == str(configured_key)
    assert verify.call_args.kwargs["ssh_user"] == "testuser"


def test_generated_control_key_is_recorded_and_used(monkeypatch, config_root):
    key = LocalSshKey(
        path=config_root / "ssh" / "sparkrun_ed25519",
        public_path=config_root / "ssh" / "sparkrun_ed25519.pub",
        public_key="ssh-ed25519 AAAAGENERATED sparkrun@test",
        generated=True,
    )
    monkeypatch.setattr("sparkrun.api.setup._ssh_access.ensure_local_key", mock.Mock(return_value=key))
    verify = mock.Mock(side_effect=_ok)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", verify)
    monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", mock.Mock(return_value=SshIdentityReport()))

    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as run:
        result = _invoke("--user", "testuser")

    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["env"]["SPARKRUN_CONTROL_PUBKEY"] == key.public_key
    saved = yaml.safe_load((config_root / "config.yaml").read_text())
    assert saved["ssh"]["key"] == str(key.path)
    assert verify.call_args.kwargs["ssh_key"] == str(key.path)


def test_failed_post_mesh_check_fails_setup_and_names_the_key(monkeypatch, configured_key):
    def refused(hosts, script, **kwargs):
        return [RemoteResult(host=h, returncode=255, stdout="", stderr="Permission denied (publickey).") for h in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", refused)
    audit = mock.Mock()
    monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", audit)

    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)):
        result = _invoke("--user", "testuser")

    assert result.exit_code == 1
    assert "Permission denied (publickey)" in result.output
    assert str(configured_key) in result.output
    audit.assert_not_called()


def _keypair(directory, name="id_cluster"):
    key = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)], check=True)
    return key


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")
def test_ssh_key_without_pub_is_derived_not_replaced(monkeypatch, tmp_path, config_root):
    """A ``.pem``-style ``ssh.key`` (no ``.pub``) is still the key installed, and stays ``ssh.key``."""
    key = _keypair(tmp_path / "keys")
    expected = key.with_name(key.name + ".pub").read_text().strip()
    key.with_name(key.name + ".pub").unlink()
    (config_root / "config.yaml").write_text(yaml.safe_dump({"ssh": {"key": str(key)}}))
    ensure = mock.Mock()
    monkeypatch.setattr("sparkrun.api.setup._ssh_access.ensure_local_key", ensure)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", mock.Mock(side_effect=_ok))
    monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", mock.Mock(return_value=SshIdentityReport()))

    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd[0] == "bash":
            return mock.Mock(returncode=0)
        return real_run(cmd, *args, **kwargs)

    with mock.patch("subprocess.run", side_effect=run) as spy:
        result = _invoke("--user", "testuser")

    assert result.exit_code == 0, result.output
    script = next(c for c in spy.call_args_list if c.args[0][0] == "bash")
    assert script.kwargs["env"]["SPARKRUN_CONTROL_PUBKEY"] == expected
    ensure.assert_not_called()
    assert yaml.safe_load((config_root / "config.yaml").read_text())["ssh"]["key"] == str(key)


def test_missing_ssh_key_fails_instead_of_substituting(monkeypatch, tmp_path, config_root):
    missing = tmp_path / "gone.pem"
    (config_root / "config.yaml").write_text(yaml.safe_dump({"ssh": {"key": str(missing)}}))
    ensure = mock.Mock()
    monkeypatch.setattr("sparkrun.api.setup._ssh_access.ensure_local_key", ensure)

    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as run:
        result = _invoke("--user", "testuser")

    assert result.exit_code == 1
    assert "does not exist" in result.output
    run.assert_not_called()
    ensure.assert_not_called()
    assert yaml.safe_load((config_root / "config.yaml").read_text())["ssh"]["key"] == str(missing)


def test_ssh_config_identity_file_is_the_key_installed(monkeypatch, tmp_path, config_root):
    """Without ``ssh.key``, the key ssh would offer (an ``IdentityFile``) wins over the stock list."""
    key = tmp_path / "keys" / "cluster_key"
    key.parent.mkdir()
    key.write_text("PRIVATE")
    key.with_name("cluster_key.pub").write_text("ssh-ed25519 AAAAFROMCONFIG c\n")
    identities = mock.Mock(return_value=[key])
    monkeypatch.setattr("sparkrun.orchestration.ssh.offered_identities", identities)
    ensure = mock.Mock()
    monkeypatch.setattr("sparkrun.api.setup._ssh_access.ensure_local_key", ensure)
    verify = mock.Mock(side_effect=_ok)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", verify)
    monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", mock.Mock(return_value=SshIdentityReport()))

    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as run:
        result = _invoke("--user", "otheruser")

    assert result.exit_code == 0, result.output
    identities.assert_called_once_with("10.0.0.1", ssh_user="otheruser", ssh_options=[])
    assert run.call_args.kwargs["env"]["SPARKRUN_CONTROL_PUBKEY"] == "ssh-ed25519 AAAAFROMCONFIG c"
    ensure.assert_not_called()
    # Nothing recorded: ssh offers that key on its own, exactly as the check just did.
    assert verify.call_args.kwargs["ssh_key"] is None
    assert not (config_root / "config.yaml").exists()


def test_unrecordable_generated_key_is_fatal(tmp_path):
    from sparkrun.api.setup import SshKeyError
    from sparkrun.cli._setup._ssh import _resolve_control_key

    key = LocalSshKey(path=tmp_path / "k", public_path=tmp_path / "k.pub", public_key="ssh-ed25519 AAAA x", generated=True)
    config = mock.Mock(ssh_key=None, config_path=str(tmp_path / "config.yaml"))
    config.save.side_effect = OSError("read-only")
    with (
        mock.patch("sparkrun.api.setup._ssh_access.ensure_local_key", return_value=key),
        pytest.raises(SshKeyError, match="could not record it as ssh.key"),
    ):
        _resolve_control_key(config)


def test_dry_run_does_not_generate_a_key(monkeypatch, config_root):
    from sparkrun.api.setup import SshKeyError

    ensure = mock.Mock(side_effect=SshKeyError("none"))
    monkeypatch.setattr("sparkrun.api.setup._ssh_access.ensure_local_key", ensure)

    result = _invoke("--user", "testuser", "--dry-run")

    assert result.exit_code == 0, result.output
    assert ensure.call_args.kwargs["generate"] is False
    assert "would be generated" in result.output
    assert not (config_root / "config.yaml").exists()


class TestNativeMeshGate:
    """Without a local bash the mesh logs in with keys, so access is bootstrapped first."""

    @pytest.fixture(autouse=True)
    def native(self, monkeypatch, config_root):
        monkeypatch.setattr("sparkrun.cli._setup._ssh._mesh_uses_native", lambda: True)
        self.mesh = mock.Mock(return_value=True)
        monkeypatch.setattr("sparkrun.cli._setup._ssh._run_ssh_mesh_native", self.mesh)
        monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", mock.Mock(side_effect=_ok))
        monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", mock.Mock(return_value=SshIdentityReport()))

    def test_blocked_hosts_stop_before_the_mesh(self, monkeypatch):
        gate = mock.Mock(return_value=SshAccessOutcome(ok_hosts=HOSTS[:1], user="testuser", blocked=HOSTS[1:]))
        monkeypatch.setattr("sparkrun.cli._setup._ssh._ensure_ssh_access", gate)

        result = _invoke("--user", "testuser")

        assert result.exit_code == 1
        gate.assert_called_once()
        assert gate.call_args.args[:2] == (HOSTS, "testuser")
        self.mesh.assert_not_called()

    def test_gate_may_not_change_the_user(self, monkeypatch):
        """``setup ssh`` does not persist a user, so a corrected one would mesh as a user later commands never use."""
        gate = mock.Mock(return_value=SshAccessOutcome(ok_hosts=list(HOSTS), user="testuser"))
        monkeypatch.setattr("sparkrun.cli._setup._ssh._ensure_ssh_access", gate)

        result = _invoke("--user", "testuser")

        assert result.exit_code == 0, result.output
        assert gate.call_args.kwargs["offer_user_change"] is False
        assert self.mesh.call_args.args[1] == "testuser"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestMeshScript:
    """The real ``mesh_ssh_keys.sh`` against a recording fake ``ssh``."""

    def _run(self, tmp_path, user, env_extra, hosts=("h1",)):
        from sparkrun.scripts import get_script_path

        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "ssh.log"
        fake = bindir / "ssh"
        # Answers the Phase 3 key read with a per-host key; records argv and stdin.
        fake.write_text(
            "#!/bin/bash\n"
            'printf "ARGS %s\\n" "$*" >> "$LOG"\n'
            'case "$*" in *"cat ~/.ssh/id_ed25519.pub"*) for a in "$@"; do case "$a" in *@*) echo "ssh-ed25519 AAAAHOST ${a#*@}";; esac; done; exit 0;; esac\n'
            'if [ ! -t 0 ]; then sed "s/^/STDIN /" >> "$LOG"; fi\n'
            "exit 0\n"
        )
        fake.chmod(0o755)
        env = {"PATH": "%s:/usr/bin:/bin" % bindir, "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "LOG": str(log), **env_extra}
        with get_script_path("mesh_ssh_keys.sh") as script:
            proc = subprocess.run(["bash", str(script), user, *hosts], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        return proc, log.read_text()

    def test_control_key_is_installed_and_verified_for_the_same_user(self, tmp_path):
        user = subprocess.check_output(["whoami"], text=True).strip()
        proc, log = self._run(
            tmp_path,
            user,
            {"SPARKRUN_CONTROL_PUBKEY": "ssh-ed25519 AAAACTL sparkrun", "SPARKRUN_CONTROL_KEY": "/cfg/ssh/sparkrun_ed25519"},
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "STDIN ssh-ed25519 AAAACTL sparkrun" in log
        verify = [line for line in log.splitlines() if "BatchMode=yes" in line]
        assert verify and "IdentitiesOnly=yes -i /cfg/ssh/sparkrun_ed25519" in verify[0]

    def test_standalone_same_user_installs_no_control_key(self, tmp_path):
        user = subprocess.check_output(["whoami"], text=True).strip()
        proc, log = self._run(tmp_path, user, {})

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "STDIN" not in log
        assert "Installing control machine key" not in proc.stdout

    def test_multi_host_mesh_exchanges_every_host_key(self, tmp_path):
        user = subprocess.check_output(["whoami"], text=True).strip()
        proc, log = self._run(tmp_path, user, {"SPARKRUN_CONTROL_PUBKEY": "ssh-ed25519 AAAACTL sparkrun"}, hosts=("h1", "h2"))

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "STDIN ssh-ed25519 AAAAHOST h1" in log
        assert "STDIN ssh-ed25519 AAAAHOST h2" in log
        assert "- h1 -> h2" in proc.stdout and "- h2 -> h1" in proc.stdout

    def test_standalone_cross_user_still_installs_a_stock_key(self, tmp_path):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAASTOCK me\n")
        proc, log = self._run(tmp_path, "someone-else", {})

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "STDIN ssh-ed25519 AAAASTOCK me" in log
        assert "IdentitiesOnly" not in log


def test_validate_public_key_refuses_multiline():
    from sparkrun.api.setup import validate_public_key

    assert validate_public_key("  ssh-ed25519 AAAA x \n") == "ssh-ed25519 AAAA x"
    with pytest.raises(ValueError):
        validate_public_key("ssh-ed25519 AAAA x\nssh-rsa BBBB y")
    with pytest.raises(ValueError):
        validate_public_key("ssh-ed25519 'AAAA'")


def test_resolve_control_key_prefers_ssh_key(configured_key):
    from sparkrun.cli._setup._ssh import _resolve_control_key
    from sparkrun.core.config import SparkrunConfig

    assert _resolve_control_key(SparkrunConfig()).path == Path(configured_key)


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")
def test_load_identity_refuses_an_encrypted_key_without_pub(tmp_path):
    from sparkrun.api.setup import SshKeyError, load_identity

    key = tmp_path / "locked"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "secret", "-f", str(key)], check=True)
    key.with_name("locked.pub").unlink()
    with pytest.raises(SshKeyError, match="ssh-keygen -y -f"):
        load_identity(key)


def test_offered_identities_asks_ssh_with_sparkrun_connection_args(tmp_path):
    """``ssh -G`` must see ``-i`` and ``ssh.options``; a bare ``user@host`` answers a different question."""
    from sparkrun.orchestration.ssh import offered_identities

    present = tmp_path / "cluster_key"
    present.write_text("PRIVATE")
    output = "user bob\nidentityfile %s\nidentityfile %s\nidentityfile %s\n" % (present, tmp_path / "absent", present)
    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout=output, stderr="")) as run:
        assert offered_identities("h1", ssh_user="bob", ssh_options=["-F", "/cfg/ssh_config"]) == [present]
    argv = run.call_args.args[0]
    assert argv[:2] == ["ssh", "-G"]
    assert argv[-1] == "bob@h1"
    assert argv[argv.index("-F") + 1] == "/cfg/ssh_config"


def test_effective_ssh_config_raises_on_failure():
    from sparkrun.orchestration.ssh import effective_ssh_config

    with (
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=255, stdout="", stderr="bad option")),
        pytest.raises(ValueError, match="bad option"),
    ):
        effective_ssh_config("h1")


def test_resolve_control_key_uses_the_context_config(configured_key):
    """With no ``config=``, the api resolves the operation context's (the daemon path)."""
    from sparkrun.api.setup import resolve_control_key

    resolved = resolve_control_key(host="h1", user="bob")
    assert (resolved.source, resolved.key.path, resolved.recorded) == ("ssh.key", Path(configured_key), False)


def test_resolve_control_key_wire_shape(configured_key):
    from sparkrun.api._wire import to_wire
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS
    from sparkrun.api.setup import resolve_control_key

    wire = to_wire(resolve_control_key())
    schema = RESULT_SCHEMAS["setup.resolve_control_key"]
    assert set(schema["required"]) <= set(wire)
    assert wire["key"]["public_key"] == "ssh-ed25519 AAAACONFIGURED cluster"
    assert wire["source"] == "ssh.key" and wire["skipped"] == []


def test_daemon_op_resolves_with_the_operation_config(configured_key):
    """The desktop/daemon path gets the same answer as ``setup ssh``, in the declared wire shape."""
    from sparkrun.api._context import default_sctx
    from sparkrun.core.daemon_registry import get_daemon_op
    from sparkrun.daemon.ops import check_request, invoke, register_core_ops, validate

    register_core_ops()
    op = get_daemon_op("setup.resolve_control_key")
    request = check_request(op, {"host": "h1", "user": "bob", "generate": False})
    result = invoke("setup.resolve_control_key", request, default_sctx())
    validate(result, op.result_schema)
    assert (result["source"], result["key"]["path"]) == ("ssh.key", str(configured_key))
