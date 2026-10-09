"""``proxy systemd``: unit naming, rendering, privileged scripts, and delegation.

No real systemd is touched: unit directories are sandboxed by conftest, the
privileged scripts run under real bash against sandbox paths with ``visudo``
and ``systemctl`` shimmed on PATH, and ``systemctl`` / ``sudo`` calls from
Python are mocked at the subprocess boundary.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from sparkrun.api.proxy import _ops, _service
from sparkrun.api.proxy import ProxyAlreadyRunning, ProxyServiceError, ProxyServiceOptions, ProxyStartOptions, SudoPasswordRequired
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.proxy import _systemd
from sparkrun.proxy._supervisor import SUPERVISOR_ENV

MARKER = "# sparkrun.distribution=sparkrun"


def _unit_file(path: Path, *, user: str | None = "alice", profile: str = "sparkrun") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# sparkrun.distribution=%s" % profile, "[Service]"]
    if user:
        lines.append("User=%s" % user)
    path.write_text("\n".join(lines) + "\n")
    return path


def _inputs(**overrides) -> _systemd.UnitInputs:
    values = dict(sparkrun_path="/opt/sr/bin/sparkrun", home="/home/alice", group="alice", path_env="/opt/sr/bin:/usr/bin:/bin")
    values.update(overrides)
    return _systemd.UnitInputs(**values)


@pytest.fixture
def system_dir():
    return _systemd.SYSTEM_UNIT_DIR


# -- Naming and lookup -----------------------------------------------------------


def test_plain_name_when_free(system_dir):
    spec = _systemd.choose_install_target("system", "alice")
    assert spec.name == "sparkrun-proxy.service"
    assert spec.path == system_dir / "sparkrun-proxy.service"


def test_plain_name_reused_when_ours(system_dir):
    _unit_file(system_dir / "sparkrun-proxy.service")
    assert _systemd.choose_install_target("system", "alice").name == "sparkrun-proxy.service"
    assert _systemd.find_installed("system", "alice").name == "sparkrun-proxy.service"


def test_suffixed_name_when_plain_belongs_to_another_user(system_dir):
    _unit_file(system_dir / "sparkrun-proxy.service", user="bob")
    spec = _systemd.choose_install_target("system", "alice")
    assert spec.name == "sparkrun-proxy-alice.service"
    assert spec.sudoers_path.name == "sparkrun-proxy-alice"
    # Bob's unit is never "ours", whatever its name.
    assert _systemd.find_installed("system", "alice") is None


def test_suffixed_name_found_by_lookup(system_dir):
    _unit_file(system_dir / "sparkrun-proxy.service", user="bob")
    _unit_file(system_dir / "sparkrun-proxy-alice.service")
    assert _systemd.find_installed("system", "alice").name == "sparkrun-proxy-alice.service"
    assert _systemd.find_installed("system", "bob").name == "sparkrun-proxy.service"


def test_other_profiles_unit_is_not_ours(system_dir):
    _unit_file(system_dir / "sparkrun-proxy.service", profile="jetsonrun")
    assert _systemd.find_installed("system", "alice") is None
    assert _systemd.choose_install_target("system", "alice").name == "sparkrun-proxy-alice.service"


def test_unit_named_for_us_but_owned_by_another_is_refused(system_dir):
    _unit_file(system_dir / "sparkrun-proxy.service", user="bob")
    _unit_file(system_dir / "sparkrun-proxy-alice.service", user="mallory")
    with pytest.raises(_systemd.SystemdError, match="taken"):
        _systemd.choose_install_target("system", "alice")


def test_user_scope_has_one_candidate():
    spec = _systemd.choose_install_target("user", "alice")
    assert spec.path == _systemd.user_unit_dir() / "sparkrun-proxy.service"
    assert spec.sudoers_path is None
    _unit_file(spec.path, user=None)
    assert _systemd.find_installed("user", "alice") == spec


# -- Rendering ---------------------------------------------------------------------


def test_render_system_unit():
    spec = _systemd.candidates("system", "alice")[0]
    text = _systemd.render_unit(spec, _inputs())
    lines = text.splitlines()
    assert lines[0] == MARKER
    for expected in (
        "User=alice",
        "Group=alice",
        "Environment=HOME=/home/alice",
        "ExecStart=/opt/sr/bin/sparkrun proxy start --foreground",
        "Environment=%s=systemd:system:sparkrun-proxy.service" % SUPERVISOR_ENV,
        "Environment=PATH=/opt/sr/bin:/usr/bin:/bin",
        "Restart=on-failure",
        "KillMode=mixed",
        "SuccessExitStatus=130",
        "WantedBy=multi-user.target",
    ):
        assert expected in lines, expected
    assert any(line.startswith("Environment=SPARKRUN_APPLICATION_PROFILE=") for line in lines)
    assert not any(line.startswith("Environment=SPARKRUN_APPLICATION_CONFIG=") for line in lines)


def test_render_user_unit_omits_identity():
    spec = _systemd.candidates("user", "alice")[0]
    lines = _systemd.render_unit(spec, _inputs(config_path="/srv/cfg/config.yaml")).splitlines()
    assert not any(line.startswith(("User=", "Group=", "Environment=HOME=")) for line in lines)
    assert "WantedBy=default.target" in lines
    assert "Environment=SPARKRUN_APPLICATION_CONFIG=/srv/cfg/config.yaml" in lines
    assert "Environment=%s=systemd:user:sparkrun-proxy.service" % SUPERVISOR_ENV in lines


def test_render_sudoers_grants_exactly_three_commands():
    spec = _systemd.candidates("system", "alice")[1]
    text = _systemd.render_sudoers(spec, "/usr/bin/systemctl")
    rule = [line for line in text.splitlines() if not line.startswith("#")]
    assert rule == [
        "alice ALL=(root) NOPASSWD: /usr/bin/systemctl start sparkrun-proxy-alice.service, "
        "/usr/bin/systemctl stop sparkrun-proxy-alice.service, /usr/bin/systemctl restart sparkrun-proxy-alice.service"
    ]
    assert "*" not in text


@pytest.mark.parametrize("bad", ["/a b", "/a%h", "/a$HOME", '/a"b', "relative", "/a\nb", "/a;b"])
def test_validate_path_refuses_unwritable(bad):
    with pytest.raises(_systemd.SystemdError):
        _systemd.validate_path(bad, "path")


@pytest.mark.parametrize("bad", ["Alice", "a.b", "host$", "", "a b"])
def test_validate_user_refuses_unsafe(bad):
    with pytest.raises(_systemd.SystemdError):
        _systemd.validate_user(bad)


def test_control_argv_matches_the_grant(monkeypatch):
    monkeypatch.setattr(_systemd, "systemctl_path", lambda: "/usr/bin/systemctl")
    system, user = _systemd.candidates("system", "alice")[0], _systemd.candidates("user", "alice")[0]
    assert _systemd.control_argv(system, "stop") == ["sudo", "-n", "/usr/bin/systemctl", "stop", "sparkrun-proxy.service"]
    assert _systemd.control_argv(user, "restart") == ["systemctl", "--user", "restart", "sparkrun-proxy.service"]
    with pytest.raises(ValueError):
        _systemd.control_argv(system, "enable")


def test_unavailable_off_linux(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    assert "Linux" in _systemd.systemd_unavailable_reason()


# -- Privileged scripts, run for real against sandbox paths ------------------------


@pytest.fixture
def shims(tmp_path, monkeypatch):
    """``visudo`` / ``systemctl`` shims that log their argv."""
    bindir = tmp_path / "shims"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    for name in ("systemctl", "visudo"):
        script = bindir / name
        script.write_text('#!/bin/bash\necho "%s $*" >> %s\nexit ${%s_RC:-0}\n' % (name, log, name.upper()))
        script.chmod(0o755)
    monkeypatch.setattr(_systemd, "SUDOERS_DIR", tmp_path / "sudoers.d")
    (tmp_path / "sudoers.d").mkdir()
    _systemd.SYSTEM_UNIT_DIR.mkdir(parents=True, exist_ok=True)

    def run(script: str, **env) -> subprocess.CompletedProcess:
        full_env = dict(os.environ, PATH="%s:%s" % (bindir, os.environ["PATH"]), **env)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=full_env)

    run.log = log
    return run


def _install_script(user="alice"):
    spec = _systemd.choose_install_target("system", user)
    unit = _systemd.render_unit(spec, _inputs(home="/home/%s" % user, group=user))
    sudoers = _systemd.render_sudoers(spec, "/usr/bin/systemctl")
    return spec, unit, sudoers, _systemd.render_system_install(spec, unit, sudoers)


def test_install_script_writes_both_files_and_enables(shims):
    spec, unit, sudoers, script = _install_script()
    result = shims(script)
    assert result.returncode == 0, result.stderr
    assert spec.path.read_text() == unit
    assert spec.sudoers_path.read_text() == sudoers
    assert stat.S_IMODE(spec.path.stat().st_mode) == 0o644
    assert stat.S_IMODE(spec.sudoers_path.stat().st_mode) == 0o440
    calls = shims.log.read_text()
    assert "visudo -cf" in calls
    assert "systemctl daemon-reload" in calls
    assert "systemctl enable sparkrun-proxy.service" in calls
    assert not list(spec.path.parent.glob("*.XXXXXX")) and not list(spec.path.parent.glob("sparkrun-proxy.service.*"))


def test_install_script_refuses_another_users_unit(shims):
    spec, _unit, _sudoers, script = _install_script()
    _unit_file(spec.path, user="bob")
    result = shims(script)
    assert result.returncode != 0
    assert "another user" in result.stderr
    assert "User=bob" in spec.path.read_text()


def test_install_script_refuses_foreign_sudoers_file(shims):
    spec, _unit, _sudoers, script = _install_script()
    spec.sudoers_path.write_text("bob ALL=(ALL) ALL\n")
    result = shims(script)
    assert result.returncode != 0
    assert not spec.path.exists()


def test_install_script_aborts_when_visudo_rejects(shims):
    spec, _unit, _sudoers, script = _install_script()
    result = shims(script, VISUDO_RC="1")
    assert result.returncode != 0
    assert not spec.path.exists()
    assert not spec.sudoers_path.exists()
    assert not list(spec.sudoers_path.parent.iterdir())


def test_uninstall_script_deletes_unit_and_grant(shims):
    spec, _unit, _sudoers, script = _install_script()
    assert shims(script).returncode == 0
    result = shims(_systemd.render_system_uninstall(spec))
    assert result.returncode == 0, result.stderr
    assert not spec.path.exists() and not spec.sudoers_path.exists()
    assert "systemctl disable --now sparkrun-proxy.service" in shims.log.read_text()


def test_uninstall_script_refuses_another_users_unit(shims):
    spec = _systemd.candidates("system", "alice")[0]
    _unit_file(spec.path, user="bob")
    result = shims(_systemd.render_system_uninstall(spec))
    assert result.returncode != 0
    assert spec.path.exists()


# -- API: install / remove / status ------------------------------------------------------


@pytest.fixture
def host(monkeypatch):
    """Pretend to be a systemd Linux host, as user alice.

    Returns a per-test namespace: set ``sudo_outcome`` / ``control_rc`` /
    ``control_stderr`` / ``active`` to steer it; read ``sudo_calls`` and
    ``controls`` to see what was asked.
    """
    host = SimpleNamespace(
        sudo_calls=[],
        controls=[],
        control_rc=0,
        control_stderr="",
        active="inactive",
        sudo_outcome=lambda password: RemoteResult(host="localhost", returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(_systemd, "systemd_unavailable_reason", lambda: None)
    monkeypatch.setattr(_systemd, "current_user", lambda: "alice")
    monkeypatch.setattr(_systemd, "systemctl_path", lambda: "/usr/bin/systemctl")
    monkeypatch.setattr(_service, "_unit_inputs", lambda sctx, warnings: _inputs())
    monkeypatch.setattr(os, "geteuid", lambda: 1000)

    def sudo(host_name, script, password, timeout=300, **_):
        host.sudo_calls.append((script, password))
        return host.sudo_outcome(password)

    def control(spec, action):
        host.controls.append((spec.name, action))
        return subprocess.CompletedProcess([], host.control_rc, "", host.control_stderr)

    monkeypatch.setattr("sparkrun.orchestration.sudo.run_sudo_script_on_host", sudo)
    monkeypatch.setattr(_systemd, "control", control)
    monkeypatch.setattr(_systemd, "query", lambda spec, verb: host.active if verb == "is-active" else "enabled")
    return host


@pytest.fixture
def context(tmp_path):
    from sparkrun.application import initialize

    return initialize(config_path=tmp_path / "config.yaml")


def test_install_system_runs_one_privileged_script(host, context):
    result = _service.install_service(ProxyServiceOptions(), sctx=context)
    assert result.unit == "sparkrun-proxy.service"
    assert result.sudoers_text and "NOPASSWD" in result.sudoers_text
    ((script, password),) = host.sudo_calls
    assert password is None
    assert "visudo -cf" in script and "systemctl enable sparkrun-proxy.service" in script
    assert host.controls == []


def test_install_dry_run_changes_nothing(host, context):
    result = _service.install_service(ProxyServiceOptions(dry_run=True, cluster=None), sctx=context)
    assert result.dry_run and result.unit_text.startswith(MARKER)
    assert host.sudo_calls == []


def test_install_refuses_root(host, context, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(ProxyServiceError, match="not as root"):
        _service.install_service(ProxyServiceOptions(), sctx=context)


def test_install_without_systemd(context, monkeypatch):
    monkeypatch.setattr(_systemd, "systemd_unavailable_reason", lambda: "this machine is not running systemd.")
    with pytest.raises(ProxyServiceError, match="not running systemd"):
        _service.install_service(ProxyServiceOptions(), sctx=context)


def test_install_needs_password_then_retries(host, context):
    host.sudo_outcome = lambda password: (
        RemoteResult(host="localhost", returncode=0, stdout="ok", stderr="")
        if password
        else RemoteResult(host="localhost", returncode=1, stdout="", stderr="sudo: a password is required\n")
    )
    with pytest.raises(SudoPasswordRequired):
        _service.install_service(ProxyServiceOptions(), sctx=context)
    _service.install_service(ProxyServiceOptions(sudo_password="pw"), sctx=context)
    assert host.sudo_calls[-1][1] == "pw"


def test_install_script_failure_is_reported(host, context):
    host.sudo_outcome = lambda password: RemoteResult(
        host="localhost", returncode=1, stdout="", stderr="Refusing to touch x: owned by another user"
    )
    with pytest.raises(ProxyServiceError, match="owned by another user"):
        _service.install_service(ProxyServiceOptions(), sctx=context)


def test_install_saves_explicit_cluster(host, context):
    context.cluster_manager.create("lab", ["h1"])
    result = _service.install_service(ProxyServiceOptions(cluster="lab"), sctx=context)
    assert result.persisted == ("cluster",)
    assert context.proxy_config.cluster == "lab"


def test_install_rejects_unknown_cluster(host, context):
    with pytest.raises(ProxyServiceError, match="Unknown cluster"):
        _service.install_service(ProxyServiceOptions(cluster="typo"), sctx=context)
    assert host.sudo_calls == []


def test_install_now_replaces_adhoc_proxy(host, context, monkeypatch):
    engine = Mock()
    engine.is_running.return_value = True
    engine.get_state.return_value = {"pid": 4242}
    engine.current_pid.return_value = 4242
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx: engine)
    stopped = []
    monkeypatch.setattr(_ops, "_stop_and_wait", lambda e: stopped.append(e) or True)
    monkeypatch.setattr(_service, "_wait_listening", lambda sctx: True)
    result = _service.install_service(ProxyServiceOptions(now=True), sctx=context)
    assert stopped == [engine]
    assert result.stopped_adhoc_pid == 4242
    assert host.controls == [("sparkrun-proxy.service", "start")]
    assert result.started and result.listening


def test_install_now_restarts_an_active_unit(host, context, monkeypatch):
    host.active = "active"
    engine = Mock()
    engine.is_running.return_value = False
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx: engine)
    monkeypatch.setattr(_service, "_wait_listening", lambda sctx: True)
    _service.install_service(ProxyServiceOptions(now=True), sctx=context)
    assert host.controls == [("sparkrun-proxy.service", "restart")]


def test_install_user_scope_needs_no_sudo(host, context, monkeypatch):
    calls = []
    monkeypatch.setattr(_systemd, "user_command", lambda spec, *args: calls.append(args) or subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(_systemd, "linger_enabled", lambda user: False)
    monkeypatch.setattr(_systemd, "enable_linger", lambda user: True)
    result = _service.install_service(ProxyServiceOptions(scope="user"), sctx=context)
    assert host.sudo_calls == []
    assert calls == [("daemon-reload",), ("enable", "sparkrun-proxy.service")]
    assert Path(result.unit_path).read_text() == result.unit_text
    assert result.linger is True


def test_install_user_scope_reports_linger_failure(host, context, monkeypatch):
    monkeypatch.setattr(_systemd, "user_command", lambda spec, *args: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(_systemd, "linger_enabled", lambda user: False)
    monkeypatch.setattr(_systemd, "enable_linger", lambda user: False)
    result = _service.install_service(ProxyServiceOptions(scope="user"), sctx=context)
    assert result.linger is False
    assert any("enable-linger" in w for w in result.warnings)


def test_find_service_refuses_two_scopes(host):
    _unit_file(_systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service")
    _unit_file(_systemd.user_unit_dir() / "sparkrun-proxy.service", user=None)
    with pytest.raises(ProxyServiceError, match="proxy systemd uninstall --system"):
        _service.find_service()
    assert _service.find_service("user").scope == "user"


def test_uninstall_system_unit(host, context):
    _unit_file(_systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service")
    result = _service.uninstall_service(sctx=context)
    assert result.uninstalled
    ((script, _pw),) = host.sudo_calls
    assert "systemctl disable --now sparkrun-proxy.service" in script


def test_uninstall_without_unit(host, context):
    with pytest.raises(ProxyServiceError, match="No proxy unit"):
        _service.uninstall_service(sctx=context)


def test_status_reports_unit(host, context, monkeypatch):
    _unit_file(_systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service")
    monkeypatch.setattr(_service, "_grant_works", lambda spec: True)
    monkeypatch.setattr(_systemd, "journal_tail", lambda spec, lines: "log line")
    status = _service.service_status(sctx=context)
    assert (status.installed, status.unit, status.active, status.grant_ok, status.journal) == (
        True,
        "sparkrun-proxy.service",
        "inactive",
        True,
        "log line",
    )
    assert status.to_dict()["discovery"]["source"] == "none"


# -- Delegation from proxy start / stop / status --------------------------------------------


@pytest.fixture
def installed(host):
    _unit_file(_systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service")
    return host


@pytest.fixture
def no_discovery(monkeypatch):
    calls = []
    monkeypatch.setattr(_ops, "_discover", lambda **kw: calls.append(kw) or [])
    monkeypatch.setattr(_ops, "resolve_gateway", lambda *_a, **_k: "litellm")
    return calls


def test_start_goes_through_installed_unit(installed, context, no_discovery):
    result = _ops.start(ProxyStartOptions(port=4100), sctx=context)
    assert result.unit == "sparkrun-proxy.service" and result.started
    assert installed.controls == [("sparkrun-proxy.service", "start")]
    assert no_discovery == []
    # The setting is saved for the unit to read.
    assert "port" in result.persisted and context.proxy_config.port == 4100


def test_start_refuses_when_unit_active(installed, context, no_discovery):
    installed.active = "active"
    with pytest.raises(ProxyAlreadyRunning, match="systemd unit sparkrun-proxy.service"):
        _ops.start(ProxyStartOptions(), sctx=context)
    assert installed.controls == []


def test_start_restart_restarts_unit(installed, context, no_discovery):
    installed.active = "active"
    result = _ops.start(ProxyStartOptions(restart=True), sctx=context)
    assert result.restarted
    assert installed.controls == [("sparkrun-proxy.service", "restart")]


def test_start_dry_run_through_unit_does_nothing(installed, context, no_discovery):
    result = _ops.start(ProxyStartOptions(dry_run=True), sctx=context)
    assert result.dry_run and result.unit and not result.started
    assert installed.controls == []


def test_start_reports_missing_grant(installed, context, no_discovery):
    installed.control_rc = 1
    installed.control_stderr = "sudo: a password is required"
    with pytest.raises(_ops.ProxyStartFailed, match="sudo systemctl start sparkrun-proxy.service"):
        _ops.start(ProxyStartOptions(), sctx=context)


def test_foreground_start_is_never_delegated(installed, context, monkeypatch):
    assert _ops._unit_to_delegate(ProxyStartOptions(foreground=True)) is None
    monkeypatch.setenv(SUPERVISOR_ENV, "systemd:system:sparkrun-proxy.service")
    assert _ops._unit_to_delegate(ProxyStartOptions()) is None


def _running_engine_with(monkeypatch, state):
    engine = Mock()
    engine.is_running.return_value = True
    engine.current_pid.return_value = state.get("pid")
    engine.get_state.return_value = state
    engine.stop.return_value = True
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx: engine)
    return engine


SUPERVISED = {"pid": 77, "supervisor": {"kind": "systemd", "scope": "system", "unit": "sparkrun-proxy.service"}}


def test_stop_goes_through_unit(installed, context, monkeypatch):
    engine = _running_engine_with(monkeypatch, SUPERVISED)
    result = _ops.stop(sctx=context)
    assert result.unit == "sparkrun-proxy.service" and result.stopped
    assert installed.controls == [("sparkrun-proxy.service", "stop")]
    engine.stop.assert_not_called()


def test_stop_falls_back_to_signal_without_grant(installed, context, monkeypatch):
    installed.control_rc = 1
    installed.control_stderr = "sudo: a password is required"
    engine = _running_engine_with(monkeypatch, SUPERVISED)
    result = _ops.stop(sctx=context)
    assert result.stopped and result.unit is None
    engine.stop.assert_called_once()


def test_stop_ignores_a_record_naming_someone_elses_unit(host, context, monkeypatch):
    _unit_file(_systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service", user="bob")
    engine = _running_engine_with(monkeypatch, SUPERVISED)
    _ops.stop(sctx=context)
    assert host.controls == []
    engine.stop.assert_called_once()


def test_status_names_the_unit(monkeypatch, context):
    engine = _running_engine_with(monkeypatch, dict(SUPERVISED, gateway="litellm"))
    engine.query_models.return_value = ()
    status = _ops.status(sctx=context)
    assert status.managed_by == "sparkrun-proxy.service"
    assert status.to_dict()["managed_by"] == "sparkrun-proxy.service"


# -- CLI ---------------------------------------------------------------------------------------


def test_cli_install_dry_run(host, context):
    from sparkrun.cli._proxy_systemd import proxy_systemd

    result = CliRunner().invoke(proxy_systemd, ["install", "--dry-run"], obj={"sparkrun_ctx": context})
    assert result.exit_code == 0, result.output
    assert "[dry-run] Unit:" in result.output and "NOPASSWD" in result.output
    assert host.sudo_calls == []


def test_cli_remove_prompts_for_sudo_password(host, context):
    from sparkrun.cli._proxy_systemd import proxy_systemd

    _unit_file(_systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service")
    host.sudo_outcome = lambda password: (
        RemoteResult(host="localhost", returncode=0, stdout="", stderr="")
        if password == "secret"
        else RemoteResult(host="localhost", returncode=1, stdout="", stderr="sudo: a password is required\n")
    )
    result = CliRunner().invoke(proxy_systemd, ["uninstall"], input="secret\n", obj={"sparkrun_ctx": context})
    assert result.exit_code == 0, result.output
    assert "Uninstalled sparkrun-proxy.service." in result.output
    assert [pw for _s, pw in host.sudo_calls] == [None, "secret"]


def test_cli_uninstall_is_the_opposite_of_install(host, context):
    from sparkrun.cli._proxy_systemd import proxy_systemd

    help_text = CliRunner().invoke(proxy_systemd, ["--help"]).output
    assert "install" in help_text and "uninstall" in help_text and "remove" not in help_text
    _unit_file(_systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service")
    result = CliRunner().invoke(proxy_systemd, ["uninstall", "--dry-run"], obj={"sparkrun_ctx": context})
    assert result.exit_code == 0 and "Would stop, disable and uninstall" in result.output


def test_cli_status_without_unit(host, context):
    from sparkrun.cli._proxy_systemd import proxy_systemd

    result = CliRunner().invoke(proxy_systemd, ["status", "--lines", "0"], obj={"sparkrun_ctx": context})
    assert result.exit_code == 0
    assert "No proxy unit installed" in result.output


def test_cli_scope_flags_conflict(host, context):
    from sparkrun.cli._proxy_systemd import proxy_systemd

    result = CliRunner().invoke(proxy_systemd, ["status", "--system", "--user"], obj={"sparkrun_ctx": context})
    assert result.exit_code != 0 and "mutually exclusive" in result.output


# -- Review follow-ups ------------------------------------------------------------


ADHOC = {"pid": 55}


def test_start_refuses_while_an_adhoc_proxy_holds_the_port(installed, context, no_discovery, monkeypatch):
    """Starting the unit then would fail every RestartSec while the CLI said 'started'."""
    _running_engine_with(monkeypatch, ADHOC)
    with pytest.raises(ProxyAlreadyRunning, match="outside systemd unit"):
        _ops.start(ProxyStartOptions(), sctx=context)
    assert installed.controls == []


def test_start_restart_replaces_an_adhoc_proxy_with_the_unit(installed, context, no_discovery, monkeypatch):
    engine = _running_engine_with(monkeypatch, ADHOC)
    stopped = []
    monkeypatch.setattr(_ops, "_stop_and_wait", lambda e: stopped.append(e) or True)
    result = _ops.start(ProxyStartOptions(restart=True), sctx=context)
    assert stopped == [engine]
    assert installed.controls == [("sparkrun-proxy.service", "start")]
    assert result.restarted


def test_start_with_the_unit_already_running_it_is_not_adhoc(installed, context, no_discovery, monkeypatch):
    installed.active = "active"
    _running_engine_with(monkeypatch, SUPERVISED)
    with pytest.raises(ProxyAlreadyRunning, match="running as systemd unit"):
        _ops.start(ProxyStartOptions(), sctx=context)


def test_install_refuses_a_second_scope(host, context):
    _unit_file(_systemd.user_unit_dir() / "sparkrun-proxy.service", user=None)
    with pytest.raises(ProxyServiceError, match="proxy systemd uninstall --user"):
        _service.install_service(ProxyServiceOptions(), sctx=context)
    assert host.sudo_calls == []
