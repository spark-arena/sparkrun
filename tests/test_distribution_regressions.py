"""Regression coverage for the distribution worktree review findings."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import os
import subprocess
import sys

import pytest
import yaml
from click.testing import CliRunner


def test_implicit_bootstrap_cannot_be_rebound_to_another_config(tmp_path):
    from sparkrun.application import initialize
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import resolve_config_path

    variables = init_sparkrun()
    original = resolve_config_path()
    with pytest.raises(RuntimeError, match="another configuration path"):
        initialize(config_path=tmp_path / "other.yaml")
    assert resolve_config_path() == original
    assert initialize(config_path=original).variables is variables


def test_config_path_aliases_reuse_the_same_initialization(tmp_path):
    from sparkrun.application import initialize

    config = tmp_path / "site.yaml"
    config.write_text("{}")
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(config)
    first = initialize(config_path=config)
    second = initialize(config_path=alias)
    assert first.variables is second.variables
    assert first.config.config_path == second.config.config_path


def test_profile_source_plugins_load_without_user_config(tmp_path, monkeypatch):
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.external_plugins import load_external_plugins
    from scitrera_app_framework import Variables

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "review_default_plugin.py").write_text(
        'SPARKRUN_PLUGIN_API_VERSION = 1\ndef register(v):\n    v.set("review.plugin.loaded", True)\n'
    )
    select_application_profile(
        ApplicationProfile(
            id="review-app",
            display_name="Review",
            command="review-app",
            package="review-app",
            defaults={"plugins": {"paths": [str(root)]}},
            feature_defaults={"core.external_plugins": True},
        )
    )
    config = SparkrunConfig()
    assert not config.config_path.exists()
    variables = Variables()
    assert "review_default_plugin" in load_external_plugins(variables)
    assert variables.get("review.plugin.loaded") is True
    assert not config.config_path.exists()


@pytest.mark.parametrize("operation", ["prepare", "start", "save", "sidecar"])
def test_foreign_gateway_state_is_preserved_before_any_mutation(tmp_path, monkeypatch, operation):
    from sparkrun.proxy._supervisor import GatewayOperationError
    from sparkrun.proxy.engine import ProxyEngine

    state = tmp_path / "state.yaml"
    state.write_text(yaml.safe_dump({"distribution": "jetsonrun", "pid": os.getpid(), "master_key": "foreign-key"}))
    config = tmp_path / "litellm_config.yaml"
    config.write_text("foreign config")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    engine = ProxyEngine(state_dir=tmp_path)
    spawn = Mock(return_value=99999)
    monkeypatch.setattr(engine, "_build_command", lambda *_: ["mock-gateway"])
    monkeypatch.setattr(engine, "_launch_background", spawn)
    monkeypatch.setattr("sparkrun.proxy._supervisor.subprocess.Popen", spawn)
    assert engine.get_state() is None
    with pytest.raises(GatewayOperationError, match="another application"):
        if operation == "prepare":
            engine.prepare_config([], {})
        elif operation == "start":
            engine.start()
        elif operation == "save":
            engine._save_state(99999)
        else:
            engine.start_autodiscover(99999)
    spawn.assert_not_called()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_gateway_claim_survives_stop_and_rejects_other_application(tmp_path):
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine(state_dir=tmp_path)
    engine._save_state(99999)
    engine._clear_state()
    code = """
import sys
from pathlib import Path
from sparkrun.core.application_profile import ApplicationProfile, select_application_profile
from sparkrun.proxy._supervisor import GatewayOperationError, GatewayState
select_application_profile(ApplicationProfile(id="jetsonrun", display_name="Jetson", command="jetsonrun", package="jetsonrun"))
try:
    GatewayState(Path(sys.argv[1])).claim_state_directory()
except GatewayOperationError:
    sys.exit(0)
sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".distribution").read_text().strip() == "sparkrun"
    assert not (tmp_path / "state.yaml").exists()


def test_gateway_prepare_respects_explicit_state_directory_and_dry_run(tmp_path):
    from sparkrun.proxy.engine import ProxyEngine

    root = tmp_path / "custom"
    engine = ProxyEngine(state_dir=root)
    engine.prepare_config([], {}, write=False)
    assert not root.exists()
    path, _, _ = engine.prepare_config([], {})
    assert path == root / "litellm_config.yaml"
    assert yaml.safe_load(path.read_text())["model_list"] == []


@pytest.mark.parametrize("explicit", [False, True])
def test_autodiscover_child_retains_custom_config(tmp_path, monkeypatch, explicit):
    from sparkrun.application import initialize
    from sparkrun.proxy._supervisor import GatewaySupervisor

    config_path = tmp_path / "site.yaml"
    config_path.write_text("integrations: {}\n")
    initialize(config_path=tmp_path / "parent.yaml" if explicit else config_path)
    options = {"application_config_path": config_path} if explicit else {}
    popen = Mock(return_value=Mock(pid=99999))
    with monkeypatch.context() as patch:
        patch.setattr("sparkrun.proxy._supervisor.subprocess.Popen", popen)
        GatewaySupervisor(tmp_path / "proxy").start_autodiscover(99999, **options)
    environment = popen.call_args.kwargs["env"]
    code = "from sparkrun.application import initialize; print(initialize().config.config_path)"
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == config_path


@pytest.mark.parametrize("host_order", [("bad", "good"), ("good", "bad")])
@pytest.mark.parametrize("failure", ["action", "exception", "unreachable", "prerequisite"])
def test_setup_command_preserves_failure_across_hosts(tmp_path, monkeypatch, host_order, failure):
    from sparkrun.cli._setup._commands import setup_docker_group
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun.core.setup_models import CheckContext, HostState
    from sparkrun.orchestration.ssh import RemoteResult

    facts = {"CHECK_OS": "Linux", "CHECK_DOCKER_INSTALLED": "1", "CHECK_DOCKER_GROUP": "0", "CHECK_DOCKER_USABLE": "0"}
    states = {h: HostState(h, facts=dict(facts), hardware=default_dgx_spark_hardware()) for h in host_order}
    if failure == "unreachable":
        states["bad"].reachable = False
    elif failure == "prerequisite":
        states["bad"].facts["CHECK_DOCKER_INSTALLED"] = "0"
    context = CheckContext(None, True, config=SparkrunConfig(), executor_names={h: "docker" for h in states}, strict=True)
    manager = SimpleNamespace(get_default=lambda: None, clusters_dir=tmp_path / "clusters")
    monkeypatch.setattr("sparkrun.cli._common._resolve_setup_context", lambda *args: (list(states), "tester", {}))
    monkeypatch.setattr("sparkrun.cli._common._get_cluster_manager", lambda: manager)
    monkeypatch.setattr("sparkrun.core.setup_probe.probe_setup_hosts", lambda hosts, **kw: ({h: states[h] for h in hosts}, context))
    monkeypatch.setattr("sparkrun.cli._setup._sudo.ensure_sudo_password", lambda *args, **kw: (None, []))

    def run(self, host, *args, **kwargs):
        if host == "bad" and failure == "exception":
            raise RuntimeError("action exploded")
        if host == "bad":
            return RemoteResult(host, 1, "", "usermod failed")
        states[host].facts.update(CHECK_DOCKER_GROUP="1", CHECK_DOCKER_USABLE="1")
        return RemoteResult(host, 0, "", "")

    monkeypatch.setattr("sparkrun.core.setup_actions.SetupActionContext.run", run)
    result = CliRunner().invoke(setup_docker_group, ["--hosts", ",".join(host_order), "--user", "tester"])
    assert result.exit_code == 1, result.output
    assert "Setup step docker_group failed" in result.output


def test_concurrent_applications_cannot_claim_the_same_empty_gateway_directory(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    code = """
import sys
from pathlib import Path
from sparkrun.core.application_profile import ApplicationProfile, select_application_profile
from sparkrun.proxy._supervisor import GatewayState, GatewayOperationError
name = sys.argv[1]
select_application_profile(ApplicationProfile(id=name, command=name, package=name, display_name=name))
try:
    GatewayState(Path(sys.argv[2])).claim_state_directory()
except GatewayOperationError:
    print("conflict")
else:
    print("claimed")
"""

    def claim(name):
        result = subprocess.run([sys.executable, "-c", code, name, str(tmp_path / "shared")], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("sparkrun", "jetsonrun")))
    assert sorted(results) == ["claimed", "conflict"]


def test_legacy_gateway_state_can_be_adopted_by_sparkrun(tmp_path):
    from sparkrun.proxy.engine import ProxyEngine

    (tmp_path / "state.yaml").write_text(yaml.safe_dump({"pid": 99999, "master_key": "legacy-key"}))
    engine = ProxyEngine(state_dir=tmp_path)
    engine.claim_state_directory()
    assert engine.get_state()["master_key"] == "legacy-key"


@pytest.mark.parametrize("state", ["[invalid", "- list", "null"])
def test_invalid_gateway_state_cannot_be_overwritten(tmp_path, state):
    from sparkrun.proxy.engine import ProxyEngine
    from sparkrun.proxy._supervisor import GatewayOperationError

    path = tmp_path / "state.yaml"
    path.write_text(state)
    with pytest.raises(GatewayOperationError):
        ProxyEngine(state_dir=tmp_path).prepare_config([], {})
    assert path.read_text() == state
    assert not (tmp_path / "litellm_config.yaml").exists()


def test_gateway_api_checks_ownership_before_plugin_config_writes(tmp_path, monkeypatch):
    from sparkrun.api.proxy import _ops
    from sparkrun.api.proxy import ProxyStartFailed, ProxyStartOptions
    from sparkrun.application import initialize
    from sparkrun.proxy._supervisor import GatewayState

    state_path = tmp_path / "state.yaml"
    foreign = yaml.safe_dump({"distribution": "jetsonrun", "pid": os.getpid()})
    state_path.write_text(foreign)
    context = initialize(config_path=tmp_path / "config.yaml")
    engine = Mock()
    engine.claim_state_directory.side_effect = GatewayState(tmp_path).claim_state_directory
    monkeypatch.setattr(_ops, "_engine_class", lambda *_: Mock(return_value=engine))
    monkeypatch.setattr(_ops, "_discovery_args", lambda *_: (_ops.DiscoveryScope("none"), None, None, None, None, []))
    monkeypatch.setattr(_ops, "_discover", lambda **_: [])
    with pytest.raises(ProxyStartFailed, match="another application"):
        _ops.start(ProxyStartOptions(persist=False, auto_discover=False), sctx=context)
    engine.prepare_config.assert_not_called()
    engine.start.assert_not_called()
    assert state_path.read_text() == foreign


@pytest.mark.parametrize("contents", ["distribution: jetsonrun\n", "[invalid", "null", "- list"])
def test_uninstall_rejects_foreign_or_unreadable_manifest_before_host_actions(monkeypatch, contents):
    from sparkrun.cli._setup._uninstall import setup_uninstall
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root

    manager = ClusterManager(get_config_root())
    manager.create("lab", ["fake-host"], user="tester")
    path = manager.clusters_dir / "lab.manifest.yaml"
    path.write_text(contents)
    teardown = Mock()
    monkeypatch.setattr("sparkrun.core.setup_undo_actions.undo_docker_group", teardown)
    check = Mock(return_value=[])
    monkeypatch.setattr("sparkrun.cli._setup._uninstall._check_running_containers", check)
    result = CliRunner().invoke(setup_uninstall, ["lab", "--phase", "docker_group", "--yes", "--force", "--keep-cluster"])
    assert result.exit_code == 1, result.output
    assert "Cannot use setup manifest" in result.output
    teardown.assert_not_called()
    check.assert_not_called()
    assert path.read_text() == contents


def test_failed_initialization_cannot_return_partial_context(tmp_path, monkeypatch):
    from sparkrun.application import initialize
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.installed_plugins import load_installed_plugins
    from scitrera_app_framework import Variables

    path = tmp_path / "config.yaml"
    path.write_text("integrations: []\n")
    with pytest.raises(ValueError, match="integrations"):
        initialize(config_path=path)
    path.write_text("integrations: {}\n")
    with pytest.raises(RuntimeError, match="previously failed; restart"):
        initialize(config_path=path)
    with pytest.raises(RuntimeError, match="previously failed; restart"):
        get_variables()

    # Failed discovery also must not consume the loader's one-time attempt.
    from sparkrun.core.config import SparkrunConfig

    variables = Variables()
    config = SparkrunConfig()
    config.set("integrations", [])
    entries = Mock(return_value=[])
    monkeypatch.setattr("sparkrun.core.installed_plugins.entry_points", entries)
    with pytest.raises(ValueError, match="integrations"):
        load_installed_plugins(variables, config=config)
    entries.assert_not_called()
    config.set("integrations", {})
    load_installed_plugins(variables, config=config)
    entries.assert_called_once()


@pytest.mark.parametrize("owner", ["jetsonrun", "sparkrun", "legacy", "unrelated"])
def test_systemd_install_checks_owner_before_replacement(tmp_path, owner):
    from sparkrun.cli._export import _render_sudo_install_script

    unit = tmp_path / "sparkrun-demo.service"
    original = {
        "legacy": "[Unit]\nDescription=sparkrun inference: previous\n",
        "unrelated": "[Unit]\nDescription=Other service\n",
    }.get(owner, "# sparkrun.distribution=%s\n[Unit]\nDescription=previous\n" % owner)
    unit.write_text(original)
    replacement = "# sparkrun.distribution=sparkrun\n[Unit]\nDescription=replacement\n"
    from sparkrun.cli._export import _service_artifacts

    for artifact in _service_artifacts("demo", "demo", str(tmp_path)):
        path = Path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# sparkrun.distribution=sparkrun\n")
    script = _render_sudo_install_script("demo", replacement, user_home=str(tmp_path), cluster_name="demo").replace(
        "/etc/systemd/system/", str(tmp_path) + "/"
    )
    result = subprocess.run(["bash", "-c", "systemctl() { :; };\n" + script], capture_output=True, text=True, timeout=10)
    if owner in {"sparkrun", "legacy"}:
        assert result.returncode == 0, result.stderr
        assert "Description=replacement" in unit.read_text()
    else:
        assert result.returncode != 0
        assert "Refusing to replace" in result.stderr
        assert unit.read_text() == original


@pytest.mark.parametrize("foreign_target", ["unit", "recipe", "cluster"])
def test_service_user_install_preflights_all_artifacts(tmp_path, foreign_target):
    from sparkrun.cli._export import _render_install_script

    service_dir = tmp_path / ".config/sparkrun/services/demo"
    service_dir.mkdir(parents=True)
    target = {
        "unit": tmp_path / "sparkrun-demo.service",
        "recipe": service_dir / "recipe.yaml",
        "cluster": tmp_path / ".config/sparkrun/clusters/demo.yaml",
    }[foreign_target]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# sparkrun.distribution=jetsonrun\n")
    script = _render_install_script("demo", "name: new", "hosts: []", "demo", str(tmp_path))
    script = script.replace("/etc/systemd/system/", str(tmp_path) + "/")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p.name != ".install.lock"}
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Refusing to replace" in result.stderr
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p.name != ".install.lock"} == before


def test_native_serve_rechecks_owner_after_preflight(tmp_path):
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.executors._base import ExecutorConfig

    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    pid = tmp_path / (name + ".pid")
    owner = tmp_path / (name + ".pid.owner")
    log = tmp_path / "shared.log"
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path), log_file=str(log)))
    preflight = executor.generate_launch_script(image="", container_name=name, command="true")
    subprocess.run(["bash", "-c", preflight], check=True, capture_output=True, timeout=10)
    # A competing application claims the same PID path after preflight.
    pid.write_text(str(os.getpid()))
    owner.write_text("jetsonrun")
    script = executor.generate_exec_serve_script(container_name=name, serve_command="true")
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "another application" in result.stderr
    assert pid.read_text() == str(os.getpid())
    assert owner.read_text() == "jetsonrun"
    assert not log.exists()


def test_competing_native_launches_cannot_both_claim_pid_path(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.executors._base import ExecutorConfig

    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path), log_file=str(tmp_path / "shared.log")))
    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    scripts = []
    for owner in ("sparkrun", "jetsonrun"):
        with monkeypatch.context() as patch:
            patch.setattr("sparkrun.orchestration.executors.local.get_application_profile", lambda owner=owner: SimpleNamespace(id=owner))
            scripts.append("setsid() { :; };\n" + executor.generate_exec_serve_script(container_name=name, serve_command="true"))

    def launch(script):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(launch, scripts))
    assert sorted(result.returncode for result in results) == [0, 1]
    winner = "sparkrun" if results[0].returncode == 0 else "jetsonrun"
    assert (tmp_path / (name + ".pid.owner")).read_text() == winner
