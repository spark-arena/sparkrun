"""Opt-in offline wheel composition proof: SPARKRUN_TEST_WHEELS=1 pytest ... ."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SPARKRUN_TEST_WHEELS") != "1", reason="set SPARKRUN_TEST_WHEELS=1 for offline wheel installation tests"
)
ROOT = Path(__file__).resolve().parents[1]
CORE_VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


@pytest.fixture(scope="module")
def wheels(tmp_path_factory):
    uv = shutil.which("uv")
    if not uv:
        pytest.fail("uv is required for wheel composition tests")
    root = tmp_path_factory.mktemp("application-profile-wheels")
    output = root / "wheels"

    def run(args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=120, cwd=root)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    for source in (
        ROOT,
        ROOT / "tests/fixtures/application_profiles/plugin",
        ROOT / "tests/fixtures/application_profiles/application",
    ):
        # Building through an sdist excludes stale local build/ output.
        run([uv, "build", "--offline", "--out-dir", str(output), str(source)])
    environment = root / "environment"
    run([uv, "venv", "--python", sys.executable, str(environment)])
    python = environment / "bin/python"
    core_wheel = next(output.glob("sparkrun-*.whl"))
    run([uv, "pip", "install", "--offline", "--python", str(python), str(core_wheel)])
    core = next(environment.glob("lib/python*/site-packages/sparkrun"))

    def digest():
        return {
            str(p.relative_to(core)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in core.rglob("*")
            if p.is_file() and p.suffix != ".pyc"
        }

    before = digest()
    plugin_wheel = next(output.glob("profile_test_plugin-*.whl"))
    run([uv, "pip", "install", "--offline", "--python", str(python), str(plugin_wheel)])
    run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.util; import profile_test_plugin; assert importlib.util.find_spec('profile_test_app') is None; "
            "from sparkrun.core.features import register_feature; "
            "[register_feature(flag) for flag in profile_test_plugin.FEATURE_DEFINITIONS]; "
            "profile_test_plugin.register(None); from sparkrun.core.features import get_feature; "
            "assert get_feature('test.installed_plugin') is not None",
        ]
    )
    app_wheel = next(output.glob("profile_test_app-*.whl"))
    run([uv, "pip", "install", "--offline", "--python", str(python), str(app_wheel)])
    assert digest() == before, "Downstream installation changed the upstream wheel's files"
    return root, environment, python


def test_distributions_are_apache_and_keep_only_oci_relay_vendor_snapshot(wheels):
    from email.parser import BytesParser
    import tarfile
    import zipfile

    root, _, _ = wheels
    with zipfile.ZipFile(next((root / "wheels").glob("sparkrun-*.whl"))) as archive:
        names = archive.namelist()
        metadata = BytesParser().parsebytes(archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))))
        assert metadata["License-Expression"] == "Apache-2.0"
        assert "sparkrun/plugins/oci_relay/__init__.py" in names
        assert "sparkrun/plugins/oci_relay/LICENSE" in names
        assert "sparkrun/plugins/oci_relay/VENDORED.toml" in names
    with tarfile.open(next((root / "wheels").glob("sparkrun-*.tar.gz"))) as archive:
        source_names = archive.getnames()
        assert any(n.endswith("/vendor/oci-relay.lock") for n in source_names)
        assert any(n.endswith("/scripts/vendor-oci-relay.py") for n in source_names)
        metadata = BytesParser().parsebytes(archive.extractfile(next(n for n in source_names if n.endswith("/PKG-INFO"))).read())
        assert metadata["License-Expression"] == "Apache-2.0"
    for name in [*names, *source_names]:
        assert "plugins/sparkroute/" not in name
        assert "plugins/coldsnap/" not in name
        assert "vendor/sparkroute" not in name and "vendor/coldsnap" not in name
        assert not name.endswith("LICENSE_EXCEPTION")


def invoke(wheels, tmp_path, *args, env_extra=None):
    _, environment, _ = wheels
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SPARKRUN_", "PROFILE_TEST_APP_")) and k not in {"PYTHONPATH", "STATEFUL_ROOT", "VIRTUAL_ENV"}
    }
    env.update(HOME=str(tmp_path), SPARKRUN_NO_TELEMETRY="1", SPARKRUN_NO_EXTERNAL_PLUGINS="1")
    env.update(env_extra or {})
    result = subprocess.run(
        [str(environment / "bin" / args[0]), *args[1:]], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    return result


def test_wheel_launchers_help_version_completion_and_isolation(wheels, tmp_path):
    for command in ("sparkrun", "profile-test-app"):
        result = invoke(wheels, tmp_path, command, "--help")
        assert result.returncode == 0, result.stderr
        assert "Usage: " + command in result.stdout
        result = invoke(wheels, tmp_path, command, "--version")
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith(command + ", version ")
        result = invoke(wheels, tmp_path, command, "setup", "version")
        assert result.returncode == 0, result.stderr
        assert "Application: " + command + " " in result.stdout
        assert "Channel:     stable" in result.stdout
        assert f"Core:        sparkrun {CORE_VERSION}" in result.stdout
        assert "Loaded plugins:" in result.stdout
        if command == "profile-test-app":
            assert "Profile:     profile-test-app" in result.stdout
            assert "profile-test-plugin: 0.1.0 (profile-test-plugin; installed; required)" in result.stdout
        completion = "_" + command.upper().replace("-", "_") + "_COMPLETE"
        result = invoke(wheels, tmp_path, command, env_extra={completion: "bash_source"})
        assert result.returncode == 0, result.stderr
        assert command in result.stdout
    code = """
import json
from sparkrun.application import initialize
c = initialize()
print(json.dumps({'config': str(c.config.config_path.parent), 'cache': str(c.config.cache_dir)}))
"""
    result = invoke(
        wheels,
        tmp_path,
        "python",
        "-c",
        code,
        env_extra={
            "SPARKRUN_APPLICATION_PROFILE": "profile_test_app.profile:PROFILE_TEST_APP",
            "SPARKRUN_CACHE_DIR": str(tmp_path / "forbidden"),
        },
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["config"] == str(tmp_path / ".config/profile-test-app")
    assert data["cache"] == str(tmp_path / ".cache/profile-test-app")
    assert not (tmp_path / "forbidden").exists()
    result = invoke(wheels, tmp_path, "profile-test-app", "setup", "version", "--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["version"] == "1.0.0"
    assert data["distribution"]["id"] == "profile-test-app"
    assert data["core"]["version"] == CORE_VERSION
    plugin = next(p for p in data["plugins"] if p["name"] == "profile-test-plugin")
    assert plugin["package"] == "profile-test-plugin" and plugin["version"] == "0.1.0"
    assert plugin["module"] == "profile_test_plugin" and plugin["required"] and plugin["loaded"]


def test_same_wheel_plugin_explicitly_selected_in_sparkrun(wheels, tmp_path):
    config = tmp_path / ".config/sparkrun/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("integrations:\n  profile-test-plugin: true\n")
    result = invoke(wheels, tmp_path, "sparkrun", "setup", "plugins", "list", "--json")
    assert result.returncode == 0, result.stderr
    plugin = next(p for p in json.loads(result.stdout) if p["name"] == "profile-test-plugin")
    assert plugin["selected"] and plugin["loaded"] and plugin["failure"] is None
    config.write_text("integrations:\n  profile-test-plugin: false\n")
    code = "from sparkrun.application import initialize; import sys; initialize(); assert 'profile_test_plugin' not in sys.modules"
    result = invoke(wheels, tmp_path, "python", "-c", code)
    assert result.returncode == 0, result.stderr


def test_child_api_profile_is_reconstructed_from_installed_package(wheels, tmp_path):
    code = "from sparkrun.application import initialize; import sys; c=initialize(); print(c.application_profile.id); assert 'sparkrun.cli' not in sys.modules"
    result = invoke(
        wheels, tmp_path, "python", "-c", code, env_extra={"SPARKRUN_APPLICATION_PROFILE": "profile_test_app.profile:PROFILE_TEST_APP"}
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "profile-test-app"
    result = invoke(wheels, tmp_path, "python", "-m", "sparkrun", "--version")
    assert result.returncode == 0 and result.stdout.startswith("sparkrun, version"), result.stderr


def test_inventory_keeps_oci_relay_without_removed_plugins(wheels, tmp_path):
    result = invoke(
        wheels,
        tmp_path,
        "profile-test-app",
        "setup",
        "plugins",
        "list",
        "--json",
        env_extra={
            "PROFILE_TEST_APP_FEATURE_GATEWAY_SPARKROUTE": "1",
            "PROFILE_TEST_APP_FEATURE_PLUGINS_COLDSNAP": "1",
            "PROFILE_TEST_APP_FEATURE_PLUGINS_OCI_RELAY": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    rows = data if isinstance(data, list) else data["plugins"]
    assert {"sparkroute", "coldsnap"}.isdisjoint(item["name"] for item in rows)
    row = next(item for item in rows if item["name"] == "oci_relay")
    assert row["loaded"] and row["enabled"] and row["failure"] is None


def test_alternate_tune_default_can_be_overridden_without_affecting_sparkrun(wheels, tmp_path):
    disabled = invoke(wheels, tmp_path, "profile-test-app", "tune", "vllm", "--help")
    assert disabled.returncode == 1, disabled.stderr
    assert "profile-test-app setup features enable cli.tune" in disabled.stderr
    for command in ("sparkrun", "profile-test-app"):
        enabled = invoke(wheels, tmp_path, command, "tune", "vllm", "--help", env_extra={"PROFILE_TEST_APP_FEATURE_CLI_TUNE": "1"})
        assert enabled.returncode == 0, enabled.stderr
        assert "--tp" in enabled.stdout


def test_arena_plugin_profile_defaults_and_override(wheels, tmp_path):
    for command, enabled in (("sparkrun", True), ("profile-test-app", False)):
        result = invoke(wheels, tmp_path, command, "setup", "plugins", "list", "--json")
        assert result.returncode == 0, result.stderr
        plugins = {plugin["name"]: plugin for plugin in json.loads(result.stdout)}
        assert "arena" not in plugins
        plugin = plugins["sparkarena"]
        assert plugin["module"] == "sparkrun.plugins.sparkarena"
        assert plugin["loaded"] is enabled
        assert plugin["version"] == (CORE_VERSION if enabled else None)
        result = invoke(wheels, tmp_path, command, "--help")
        assert result.returncode == 0, result.stderr
        assert ("  arena " in result.stdout) is enabled
        result = invoke(wheels, tmp_path, command, "benchmark", "perf", "--help")
        assert result.returncode == 0, result.stderr
        assert ("--arena" in result.stdout) is enabled
        result = invoke(
            wheels,
            tmp_path,
            command,
            env_extra={
                "_" + command.upper().replace("-", "_") + "_COMPLETE": "bash_complete",
                "COMP_WORDS": command + " benchmark perf --ar",
                "COMP_CWORD": "3",
            },
        )
        assert result.returncode == 0, result.stderr
        assert ("plain,--arena" in result.stdout) is enabled
    result = invoke(
        wheels, tmp_path, "profile-test-app", "benchmark", "perf", "--help", env_extra={"PROFILE_TEST_APP_FEATURE_INTEGRATION_ARENA": "1"}
    )
    assert result.returncode == 0, result.stderr
    assert "--arena" in result.stdout
    assert "profile-test-app arena login" in result.stdout


def test_alternate_registry_commands_never_restore_spark_defaults(wheels, tmp_path):
    # A populated Sparkrun catalog beside the fresh Profile test application home must not leak.
    spark_config = tmp_path / ".config/sparkrun"
    spark_config.mkdir(parents=True)
    (spark_config / "registries.yaml").write_text(
        "config_version: 1\nregistries:\n- name: atlas\n  url: https://github.com/Atlas-Inf/sparkrun-recipes.git\n"
        "  subpath: recipes\n  trusted: true\n"
    )
    original = (spark_config / "registries.yaml").read_bytes()
    for args in (
        ("registry", "list", "--json"),
        ("registry", "revert-to-defaults", "--no-update"),
        ("registry", "update"),
        ("registry", "list", "--json"),
    ):
        result = invoke(wheels, tmp_path, "profile-test-app", *args)
        assert result.returncode == 0, result.stdout + result.stderr
        if args[1] == "list":
            assert json.loads(result.stdout) == []
        elif args[1] == "update":
            assert "No enabled registries to update" in result.stdout
        else:
            assert "0 entries" in result.stdout
    result = invoke(wheels, tmp_path, "profile-test-app", "registry", "list-benchmark-profiles", "--all")
    assert result.returncode == 0, result.stderr
    assert "No benchmark profiles found" in result.stdout
    assert (spark_config / "registries.yaml").read_bytes() == original


@pytest.mark.parametrize("command", ["sparkrun", "profile-test-app"])
@pytest.mark.parametrize("channel", ["stable", "beta", "alpha"])
def test_installed_k8s_plugin_defaults_and_explicit_opt_in(wheels, tmp_path, command, channel):
    config = tmp_path / ".config" / command / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"features": {"channel": channel}}))
    enabled = command == "sparkrun" and channel == "alpha"
    result = invoke(wheels, tmp_path, command, "setup", "plugins", "list", "--json")
    assert result.returncode == 0, result.stderr
    row = next(row for row in json.loads(result.stdout) if row["name"] == "k8s")
    assert row["loaded"] is enabled and row["enabled"] is enabled
    assert row["version"] == (CORE_VERSION if enabled else None)
    result = invoke(wheels, tmp_path, command, "setup", "features", "list", "--json")
    assert result.returncode == 0, result.stderr
    flags = {row["name"] for row in json.loads(result.stdout)}
    children = {"executor.k8s", "cli.setup.k8s", "api.run.k8s"}
    assert children & flags == (children if enabled else set())
    result = invoke(wheels, tmp_path, command, "setup", "k8s", "kubectl", "--list")
    assert result.returncode == (0 if enabled else 2), result.stdout + result.stderr
    result = invoke(
        wheels,
        tmp_path,
        command,
        "setup",
        "k8s",
        "kubectl",
        "--list",
        env_extra={command.upper().replace("-", "_") + "_FEATURE_INTEGRATION_K8S": "1"},
    )
    assert result.returncode == 0 and "No cached kubectl binaries" in result.stdout, result.stdout + result.stderr


def test_wheel_alternate_wizard_preview_and_visibility(wheels, tmp_path):
    result = invoke(
        wheels, tmp_path, "profile-test-app", "setup", "wizard", "--hosts", "192.0.2.1", "--cluster", "preview", "--dry-run", "--yes"
    )
    assert result.returncode == 0, result.stderr
    assert "no changes made" in result.stdout
    assert not list(tmp_path.rglob("preview.yaml"))
    result = invoke(wheels, tmp_path, "profile-test-app", "setup", "--help", env_extra={"PROFILE_TEST_APP_FEATURE_CLI_SETUP_WIZARD": "0"})
    assert result.returncode == 0, result.stderr
    assert "wizard" not in result.stdout
    result = invoke(wheels, tmp_path, "profile-test-app", "setup", env_extra={"PROFILE_TEST_APP_FEATURE_CLI_SETUP_WIZARD": "0"})
    assert result.returncode == 0, result.stderr
    assert "Setup and configuration commands" in result.stdout


def test_installed_plugin_observes_application_and_controller_identity(wheels, tmp_path):
    config = tmp_path / "site.yaml"
    config.write_text("integrations:\n  profile-test-plugin: true\n")
    other_config = tmp_path / "other.yaml"
    other_config.write_text(config.read_text())
    code = """
import json, sys
from sparkrun.application import initialize, get_controller_identity
context = initialize(config_path=sys.argv[1])
assert context.variables.get("test.plugin.application_identity") == context.application_identity
assert get_controller_identity() == context.controller_identity
print(json.dumps(context.controller_identity.to_dict()))
"""
    identities = []
    for profile in ("sparkrun.core.application_profile:SPARKRUN", "profile_test_app.profile:PROFILE_TEST_APP"):
        profile_identities = []
        for config_path in (config, other_config):
            result = invoke(
                wheels, tmp_path, "python", "-I", "-c", code, str(config_path), env_extra={"SPARKRUN_APPLICATION_PROFILE": profile}
            )
            assert result.returncode == 0, result.stderr
            profile_identities.append(json.loads(result.stdout))
        assert profile_identities[0] == profile_identities[1]
        identities.append(profile_identities[0])
    assert {item["application"]["id"] for item in identities} == {"sparkrun", "profile-test-app"}
    assert len({item["controller_id"] for item in identities}) == 2
    assert len(list(tmp_path.glob(".controllers/*.id"))) == 2


def test_wheel_includes_default_seccomp_policy_and_provenance(wheels, tmp_path):
    _, _, python = wheels
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
import json
from importlib.resources import files
from sparkrun.orchestration.executors._seccomp import io_uring_profile
root = files("sparkrun.orchestration.executors").joinpath("seccomp")
assert "Apache License" in root.joinpath("LICENSE").read_text()
assert "61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31" in root.joinpath("README.md").read_text()
upstream = json.loads(root.joinpath("default.json").read_text())
profile = json.loads(io_uring_profile())
assert profile["syscalls"].pop()["names"] == ["io_uring_enter", "io_uring_register", "io_uring_setup"]
assert profile == upstream
""",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_native_state_helpers_fail_closed(wheels, tmp_path):
    _, _, python = wheels
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
import os
import signal
import subprocess
from pathlib import Path
from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.orchestration.executors.local import LocalExecutor
root = Path.cwd()
name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
executor = LocalExecutor(ExecutorConfig(pid_dir=str(root / "pids"), log_dir=str(root / "logs")))
script = executor.teardown_script([name])
assert subprocess.run(["bash", "-c", script], capture_output=True).returncode == 0
pid = root / "pids" / (name + ".pid")
child = subprocess.Popen(["sleep", "30"], start_new_session=True)
try:
    pid.write_text(str(child.pid))
    stopped = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert stopped.returncode == 0 and "sparkrun_removed=1" in stopped.stdout
    assert child.wait(timeout=5) != 0 and not pid.exists()
finally:
    child.kill()
    child.wait(timeout=5)
launched_pid = None
try:
    launched = subprocess.run(["bash", "-c", executor.run_cmd("", "exec sleep 30", name)], capture_output=True, text=True)
    assert launched.returncode == 0, launched.stderr
    launched_pid = int(pid.read_text())
    assert not list(pid.parent.glob("*.pending.*"))
    stopped = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert stopped.returncode == 0 and "sparkrun_removed=1" in stopped.stdout
    assert not pid.exists()
finally:
    if launched_pid is not None:
        try:
            os.killpg(launched_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
try:
    LocalExecutor(ExecutorConfig(pid_dir="pids")).resolve_target()
except ValueError:
    pass
else:
    raise AssertionError("relative managed destination was accepted")
pid.write_text("invalid PID")
result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
assert result.returncode != 0 and "invalid" in result.stderr
assert pid.read_text() == "invalid PID"
""",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("application", ["sparkrun", "profile-test-app"])
def test_installed_typed_gateway_public_api_and_cli(wheels, tmp_path, application):
    config = tmp_path / "config.yaml"
    config.write_text("integrations:\n  profile-test-plugin: true\nfeatures:\n  gateway.profile-test: true\n")
    code = """
import json, os, sys
from sparkrun.application import initialize
from sparkrun import api
from sparkrun.proxy.contracts import ProxyModel
context = initialize(config_path=sys.argv[1])
assert 'sparkrun.cli' not in sys.modules and 'click' not in sys.modules
assert api.proxy.ProxyModel is ProxyModel
assert 'profile-test' in api.proxy.list_gateways(sctx=context)
assert api.proxy.resolve_gateway('profile-test', sctx=context) == 'profile-test'
# Represent an already running fixture gateway using this test process. No
# gateway binary, listener, remote host, or workload is created.
state = context.config.cache_dir / 'proxy' / 'state.yaml'
state.parent.mkdir(parents=True, exist_ok=True)
state.write_text(json.dumps({'gateway': 'profile-test', 'pid': os.getpid(), 'distribution': context.application_identity.id}))
models = api.proxy.models(sctx=context)
assert models == (ProxyModel('fixture-model', 'http://fixture/v1', 8192),)
assert api.proxy.status(sctx=context).require_models() == models
assert not api.proxy.ui(sctx=context).auth_required
credential = api.proxy.ui(issue_token=True, sctx=context).token
assert credential and api.proxy.admin_token(sctx=context) == credential
replacement = api.proxy.admin_token(rotate=True, sctx=context)
assert replacement != credential
assert api.proxy.admin_token(clear=True, sctx=context) is None
assert 'sparkrun.cli' not in sys.modules and 'click' not in sys.modules
from click.testing import CliRunner
from sparkrun.cli import main
result = CliRunner().invoke(main, ['proxy', 'models', '--json'])
assert result.exit_code == 0, result.output
assert json.loads(result.stdout) == [model.to_dict() for model in models]
os.environ['PROFILE_TEST_GATEWAY_FAIL'] = '1'
assert api.proxy.status(sctx=context).model_query_error == 'fixture control plane unavailable'
try:
    api.proxy.models(sctx=context)
except api.proxy.ProxyQueryFailed:
    pass
else:
    raise AssertionError('unavailable model query was treated as empty')
result = CliRunner().invoke(main, ['proxy', 'models', '--json'])
assert result.exit_code == 1 and 'unavailable' in result.stderr and not result.stdout, result.output
print('typed gateway API and CLI: OK')
"""
    env = {}
    if application != "sparkrun":
        env["SPARKRUN_APPLICATION_PROFILE"] = "profile_test_app.profile:PROFILE_TEST_APP"
    result = invoke(wheels, tmp_path, "python", "-c", code, str(config), env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "typed gateway API and CLI: OK" in result.stdout


@pytest.mark.parametrize("application", ["sparkrun", "profile-test-app"])
@pytest.mark.parametrize("first_call", ["list_gateways", "resolve_gateway"])
def test_installed_gateway_selection_initializes_implicitly(wheels, tmp_path, application, first_call):
    root = tmp_path / ".config" / application
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("integrations:\n  profile-test-plugin: true\nfeatures:\n  gateway.profile-test: true\n")
    (root / "proxy.yaml").write_text("proxy:\n  gateway: profile-test\n")
    code = """
import sys
from sparkrun import api
from sparkrun.application import initialize
from sparkrun.proxy.supervisor import GatewaySupervisor
from sparkrun.proxy.contracts import GatewayOperationError
first = getattr(api.proxy, sys.argv[1])()
context = initialize()
assert api.proxy.resolve_gateway() == api.proxy.resolve_gateway(sctx=context) == 'profile-test'
assert api.proxy.list_gateways() == api.proxy.list_gateways(sctx=context)
assert 'profile-test' in first if sys.argv[1] == 'list_gateways' else first == 'profile-test'
from profile_test_plugin.gateway import ProfileTestGateway
assert issubclass(ProfileTestGateway, GatewaySupervisor)
from sparkrun.proxy._supervisor import GatewayOperationError as LegacyError
assert LegacyError is GatewayOperationError
assert 'click' not in sys.modules and 'sparkrun.cli' not in sys.modules
"""
    env = {"SPARKRUN_APPLICATION_PROFILE": "profile_test_app.profile:PROFILE_TEST_APP"} if application != "sparkrun" else {}
    result = invoke(wheels, tmp_path, "python", "-c", code, first_call, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("alternate", [False, True])
def test_installed_scheduler_first_call_and_catalog_types(wheels, tmp_path, alternate):
    config = tmp_path / "config.yaml"
    config.write_text("integrations:\n  profile-test-plugin: true\n")
    # An explicit empty registry inventory makes browsing offline even on the
    # built-in profile's first run (which otherwise discovers default manifests).
    (tmp_path / "registries.yaml").write_text("registries: []\n")
    code = """
import json, sys
from importlib.resources import files
from sparkrun import api
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import SchedulingRequest
request = SchedulingRequest(parallelism=ParallelismConfig(), hosts=('localhost',))
first = api.schedule(request, scheduler='profile-test')
context = api.default_sctx()
assert first == api.schedule(request, scheduler='profile-test', sctx=context)
assert first.assignment.hosts_used == ('localhost',)
from profile_test_plugin.catalog import browse, preview
uploaded: api.CatalogRecipeDetails = api.import_recipe(
    'model: test/model\\nruntime: sglang\\ncontainer: test/image\\ndefaults: {tensor_parallel: 1}\\n', sctx=context)
page: api.CatalogPage = browse(context)
assert page['total'] == 1 and page['next_offset'] is None
assert page['recipes'][0]['reference'] == uploaded['reference']
details, resolved = preview(uploaded['reference'], context)
assert isinstance(details, dict) and type(resolved) is tuple
assert resolved[1]['tensor_parallel'] == 1
assert details['trusted'] is False and resolved[0].is_url_sourced
assert json.loads(json.dumps(page)) == page
assert 'pp' in api.CatalogRecipe.__optional_keys__
assert 'reference' in api.CatalogRecipe.__required_keys__
assert files('sparkrun').joinpath('py.typed').is_file()
assert 'sparkrun.cli' not in sys.modules
print(context.application_profile.id)
"""
    env = {
        "SPARKRUN_APPLICATION_CONFIG": str(config),
        "SPARKRUN_APPLICATION_PROFILE": (
            "profile_test_app.profile:PROFILE_TEST_APP" if alternate else "sparkrun.core.application_profile:SPARKRUN"
        ),
    }
    result = invoke(wheels, tmp_path, "python", "-c", code, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ("profile-test-app" if alternate else "sparkrun")


def test_installed_catalog_consumer_types(wheels, tmp_path):
    from _api_typecheck import check_api_consumer

    _root, _environment, python = wheels
    check_api_consumer(tmp_path, python=python)


@pytest.mark.parametrize("application", ["sparkrun", "profile-test-app"])
@pytest.mark.parametrize("first_call", ["status", "models"])
def test_installed_gateway_management_initializes_implicitly(wheels, tmp_path, application, first_call):
    config = tmp_path / "custom-config" / "config.yaml"
    config.parent.mkdir()
    config.write_text("integrations:\n  profile-test-plugin: true\nfeatures:\n  gateway.profile-test: true\n")
    (config.parent / "proxy.yaml").write_text("proxy:\n  gateway: litellm\n")
    code = """
import json, os, sys
from pathlib import Path
from sparkrun import api
from sparkrun.core import bootstrap
from sparkrun.proxy import gateway
from sparkrun.proxy.contracts import ProxyModel
assert bootstrap._variables is None
assert 'profile-test' not in gateway._GATEWAY_LOADERS
# The fixture provider queries locally; this process supplies a live PID only.
state = Path.home() / '.cache' / sys.argv[3] / 'proxy' / 'state.yaml'
state.parent.mkdir(parents=True)
state.write_text(json.dumps({'gateway': 'profile-test', 'pid': os.getpid(), 'distribution': sys.argv[3]}))
first = getattr(api.proxy, sys.argv[1])()
models = first.require_models() if sys.argv[1] == 'status' else first
assert models == (ProxyModel('fixture-model', 'http://fixture/v1', 8192),)
context = api.default_sctx()
assert context.config.config_path == Path(sys.argv[2])
assert context.application_identity.id == sys.argv[3]
assert context.proxy_config.gateway == 'litellm'
assert api.proxy.models(sctx=context) == models
assert 'click' not in sys.modules and 'sparkrun.cli' not in sys.modules
"""
    profile_ref = "sparkrun.core.application_profile:SPARKRUN" if application == "sparkrun" else "profile_test_app.profile:PROFILE_TEST_APP"
    result = invoke(
        wheels,
        tmp_path,
        "python",
        "-c",
        code,
        first_call,
        str(config),
        application,
        env_extra={"SPARKRUN_APPLICATION_PROFILE": profile_ref, "SPARKRUN_APPLICATION_CONFIG": str(config)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("application", ["sparkrun", "profile-test-app"])
@pytest.mark.parametrize("command", ["status", "stop", "models", "sync", "refresh"])
def test_installed_cli_gateway_recovery_after_bootstrap_failure(wheels, tmp_path, application, command):
    config = tmp_path / "custom-config" / "config.yaml"
    config.parent.mkdir()
    config.write_text("integrations: []\n")  # Invalid installed-plugin configuration poisons bootstrap.
    code = """
import json, os, subprocess, sys
from pathlib import Path
application, command = sys.argv[1:]
state = Path.home() / '.cache' / application / 'proxy' / 'state.yaml'
state.parent.mkdir(parents=True)
state.write_text(json.dumps({'gateway': 'removed-provider', 'pid': os.getpid(), 'distribution': application}))
before = state.read_bytes()
# The parent supplies a live PID. Stop is always a dry run; no signal is sent.
flags = ['--dry-run'] if command == 'stop' else ['--json']
subcommand = command
if command == 'refresh':
    subcommand = 'models'
    flags.append('--refresh')
result = subprocess.run([str(Path(sys.executable).with_name(application)), 'proxy', subcommand, *flags], capture_output=True, text=True, timeout=20)
assert 'integrations' in result.stderr, result.stderr
if command in {'sync', 'refresh'}:
    assert result.returncode == 1, result.stderr
    assert 'Error:' in result.stderr and 'Restore its plugin' in result.stderr
    assert 'removed-provider' in result.stderr and not result.stdout
elif command == 'models':
    assert result.returncode == 1, result.stderr
    assert 'Model list unavailable' in result.stderr and not result.stdout
else:
    assert result.returncode == 0, result.stderr
    if command == 'status':
        snapshot = json.loads(result.stdout)
        assert snapshot['running'] and snapshot['pid'] == os.getpid()
        assert snapshot['gateway'] == 'removed-provider' and snapshot['model_query_error']
    else:
        assert 'Proxy stopped.' in result.stdout
assert state.read_bytes() == before
"""
    profile_ref = "sparkrun.core.application_profile:SPARKRUN" if application == "sparkrun" else "profile_test_app.profile:PROFILE_TEST_APP"
    result = invoke(
        wheels,
        tmp_path,
        "python",
        "-c",
        code,
        application,
        command,
        env_extra={"SPARKRUN_APPLICATION_PROFILE": profile_ref, "SPARKRUN_APPLICATION_CONFIG": str(config)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("application", ["sparkrun", "profile-test-app"])
def test_installed_configuration_binding_and_metadata_cache(wheels, tmp_path, application):
    config = tmp_path / "custom-config" / "config.yaml"
    cache = tmp_path / "configured-job-cache"
    config.parent.mkdir()
    config.write_text("cache_dir: %s\n" % cache)
    code = """
import sys
from pathlib import Path
from unittest.mock import patch
import yaml
from sparkrun import api
from sparkrun.application import initialize, run_cli
from sparkrun.core import bootstrap
from sparkrun.core.application_profile import get_application_profile
root = Path(sys.argv[1])
cluster_id = get_application_profile().resource_namespace + '_fixture'
metadata = root / 'jobs' / 'fixture.yaml'
metadata.parent.mkdir(parents=True)
metadata.write_text(yaml.safe_dump({'cluster_id': cluster_id, 'hosts': ['worker'], 'distribution': get_application_profile().id}))
assert bootstrap._variables is None
assert [j.cluster_id for j in api.list_jobs()] == [cluster_id]
assert bootstrap._variables is None
assert 'click' not in sys.modules and 'sparkrun.cli' not in sys.modules
context = initialize()
assert api.list_jobs() == api.list_jobs(sctx=context)
for operation in (api.stop, api.logs):
    with patch('sparkrun.api._resolve.resolve_cluster_for_job', side_effect=RuntimeError('metadata resolved')) as resolve:
        try:
            operation(cluster_id=cluster_id)
        except RuntimeError as error:
            assert str(error) == 'metadata resolved'
        else:
            raise AssertionError('metadata was not selected')
        assert resolve.call_args.kwargs['meta']['cluster_id'] == cluster_id
import click
from sparkrun.cli import main
main._cli_ext_loaded = True
try:
    run_cli(args=['proxy', 'alias', 'add', 'friendly', 'model'], obj={'config_path': root / 'wrong.yaml'}, standalone_mode=False)
except click.UsageError as error:
    assert 'another configuration path' in str(error)
else:
    raise AssertionError('CLI silently ignored the requested config')
assert not (root / 'proxy.yaml').exists()
assert not (context.config.config_path.parent / 'proxy.yaml').exists()
"""
    profile_ref = "sparkrun.core.application_profile:SPARKRUN" if application == "sparkrun" else "profile_test_app.profile:PROFILE_TEST_APP"
    result = invoke(
        wheels,
        tmp_path,
        "python",
        "-c",
        code,
        str(cache),
        env_extra={"SPARKRUN_APPLICATION_PROFILE": profile_ref, "SPARKRUN_APPLICATION_CONFIG": str(config)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
