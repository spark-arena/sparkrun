"""Application profile, composition and coexistence contracts."""

from dataclasses import FrozenInstanceError

from _status_fixtures import host_snapshot
import json
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from sparkrun.core.application_profile import (
    ApplicationProfile,
    SPARKRUN,
    UpdateSource,
    select_application_profile,
)


def alternate(**kwargs):
    values = dict(id="alternate-test", display_name="Alternate test", command="alternate-test", package="alternate-test")
    values.update(kwargs)
    return select_application_profile(ApplicationProfile(**values))


def test_profile_freezes_nested_data_and_rejects_switch():
    original = {"plugins": {"test": {"list": [1, 2]}}}
    profile = alternate(defaults=original)
    original["plugins"]["test"]["list"].append(3)
    assert profile.defaults["plugins"]["test"]["list"] == (1, 2)
    with pytest.raises(TypeError):
        profile.defaults["plugins"]["test"]["other"] = True
    with pytest.raises(FrozenInstanceError):
        profile.id = "other"
    assert select_application_profile(profile) is profile
    with pytest.raises(RuntimeError, match="already selected"):
        select_application_profile(SPARKRUN)


@pytest.mark.parametrize(
    "field,value", [("id", "../other"), ("command", "alternate;id"), ("resource_namespace", "unsafe/x"), ("env_prefix", "X;Y")]
)
def test_profile_namespace_validation(field, value):
    with pytest.raises(ValueError):
        alternate(**{field: value})


def test_config_roots_environment_and_effective_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SPARKRUN_CONFIG_DIR", str(tmp_path / "foreign-config"))
    monkeypatch.setenv("SPARKRUN_CACHE_DIR", str(tmp_path / "foreign-cache"))
    alternate(
        defaults={
            "defaults": {"executor": "local"},
            "plugins": {"example": {"enabled": True, "options": [1]}},
            "recipe_paths": ["/inherited"],
        }
    )
    from sparkrun.core.config import SparkrunConfig, get_config_root, resolve_sparkrun_cache_dir

    config = SparkrunConfig()
    assert config.config_path == tmp_path / ".config/alternate-test/config.yaml"
    assert get_config_root() == config.config_path.parent
    assert resolve_sparkrun_cache_dir() == tmp_path / ".cache/alternate-test"
    assert config.default_executor == "local"
    config.set("plugins.example.enabled", False)
    config.set("recipe_paths", [])
    config.set("unrelated", 1)
    config.save()
    assert config.plugin_settings("example") == {"enabled": False, "options": [1]}
    persisted = yaml.safe_load(config.config_path.read_text())
    assert persisted == {"plugins": {"example": {"enabled": False}}, "recipe_paths": [], "unrelated": 1}
    assert config.setting_source("defaults.executor") == "distribution"
    assert config.setting_source("plugins.example.enabled") == "config"
    config.set("plugins.example", {})
    assert config.plugin_settings("example") == {}
    monkeypatch.setenv("ALTERNATE_TEST_CACHE_DIR", str(tmp_path / "custom"))
    assert SparkrunConfig().cache_dir == tmp_path / "custom"


def test_core_feature_channel_is_separate(monkeypatch):
    alternate(default_channel="beta", update_sources={"beta": UpdateSource("alternate-test")}, feature_defaults={"executor.local": False})
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import feature_source, is_feature_enabled

    config = SparkrunConfig()
    assert config.self_update_channel == "beta"
    assert config.feature_channel == "stable"
    # The fixture's SPARKRUN_FEATURE_EXECUTOR_LOCAL=1 must not bleed through.
    assert not is_feature_enabled("executor.local", config=config)
    assert feature_source("executor.local", config=config) == "distribution"
    monkeypatch.setenv("ALTERNATE_TEST_FEATURE_EXECUTOR_LOCAL", "1")
    assert is_feature_enabled("executor.local", config=config)
    config.set("features.channel", "alpha")
    assert config.feature_channel == "alpha"


def test_remote_cache_expression_is_target_owned(monkeypatch):
    alternate()
    from sparkrun.orchestration import primitives

    probe = Mock(return_value="/remote/custom/cache")
    monkeypatch.setattr(primitives, "probe_remote_path", probe)
    assert primitives.probe_remote_sparkrun_cache("target") == "/remote/custom/cache"
    assert probe.call_args.args == ("target", "${ALTERNATE_TEST_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/alternate-test}")


def test_registry_overlay_and_user_decisions_survive_profile_update(tmp_path):
    profile = alternate(registries=({"name": "alternate-recipes", "url": "https://example.test/recipes.git", "trusted": True},))
    from sparkrun.core.registry import RegistryManager

    manager = RegistryManager(tmp_path / "config", tmp_path / "cache")
    assert [e.name for e in manager.list_registries()] == ["alternate-recipes"]
    manager.untrust_registry("alternate-recipes")
    assert not RegistryManager(tmp_path / "config", tmp_path / "cache").get_registry("alternate-recipes").trusted
    manager.remove_registry("alternate-recipes")
    assert manager.list_registries() == []
    assert manager.restore_missing_defaults() == []
    assert profile.registries[0]["trusted"] is True


def test_application_profile_selection_does_not_trust_plugin_registries(tmp_path):
    alternate()
    from sparkrun.core.registry import RegistryEntry, RegistryManager
    from sparkrun.core.registry_defaults import register_default_registry

    register_default_registry(
        RegistryEntry(name="external", url="https://example.test/external", subpath="", trusted=True), owner="example"
    )
    entry = RegistryManager(tmp_path / "config", tmp_path / "cache").get_registry("external")
    assert not entry.trusted


def test_reserved_registry_names_stay_reserved(tmp_path):
    alternate(registries=({"name": "official", "url": "https://example.test/foreign"},))
    from sparkrun.core.registry import RegistryError, RegistryManager

    with pytest.raises(RegistryError):
        RegistryManager(tmp_path / "config", tmp_path / "cache").list_registries()


def test_channel_strategy_and_update_ownership(monkeypatch, tmp_path):
    alternate(update_sources={"stable": UpdateSource("alternate-test"), "beta": UpdateSource("alternate-test==2.0b1")})
    from sparkrun.cli import _self_update as update
    from sparkrun.core.channels import channel_requirement, is_git_channel

    assert update.update_argv("uv", "stable") == ["uv", "tool", "upgrade", "alternate-test"]
    assert "alternate-test==2.0b1" in update.update_argv("uv", "beta")
    assert not is_git_channel("beta")
    with pytest.raises(ValueError, match="does not support"):
        channel_requirement("alpha")
    monkeypatch.setattr(update.sys, "prefix", str(tmp_path / "tool"))
    runner = Mock(return_value=SimpleNamespace(returncode=0, stdout="alternate-test-plugin v1 (%s)\n" % (tmp_path / "tool")))
    monkeypatch.setattr(update.subprocess, "run", runner)
    assert not update.is_uv_tool_install("uv")
    runner.return_value.stdout = "alternate-test v1 (/other/tool)\n"
    assert not update.is_uv_tool_install("uv")
    runner.return_value.stdout = "alternate-test v1 (%s)\n" % (tmp_path / "tool")
    assert update.is_uv_tool_install("uv")
    runner.return_value.stdout = '{"version":"1.0", "commit":null, "distribution":{"id":"alternate-test"}}'
    assert update.new_binary_identity() == ("1.0", None)
    assert runner.call_args.args[0][0] == str(tmp_path / "tool/bin/alternate-test")


def test_recipe_identity_and_resource_coexistence():
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import generate_intent_id, generate_cluster_id, parse_cluster_id, parse_container_name
    from sparkrun.orchestration.executors.docker import _parse_docker_ps_output
    from sparkrun.core.ownership import OWNER_LABEL

    recipe = Recipe({"model": "example/model", "runtime": "vllm-ray", "container": "example/image"})
    original = generate_intent_id(recipe)
    # Building a recipe does not resolve application identity.
    alternate()
    assert generate_intent_id(recipe) == original
    own = generate_cluster_id(original, "123456abcdef")
    foreign = "sparkrun_" + original + "_123456abcdef"
    assert own.startswith("alternate-test_")
    assert parse_cluster_id(own) == parse_cluster_id(foreign)
    assert parse_container_name(own + "_head") == (own, "head")
    lines = [
        {"Names": own + "_head", "Labels": OWNER_LABEL + "=alternate-test"},
        {"Names": foreign + "_head", "Labels": ""},
        {"Names": own + "_worker", "Labels": OWNER_LABEL + "=foreign"},
    ]
    workloads, used = _parse_docker_ps_output("\n".join(json.dumps(line) for line in lines), "host")
    assert [w.cluster_id for w in workloads] == [own]
    assert used == 3


def test_foreign_metadata_and_teardown_are_rejected(tmp_path):
    alternate()
    from sparkrun.core.ownership import owns_metadata, docker_owner_guard
    from sparkrun.orchestration.job_metadata import load_job_metadata, remove_job_metadata

    name = "alternate-test_1234567890abcdef_123456abcdef"
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    path = jobs / "1234567890abcdef_123456abcdef.yaml"
    path.write_text(yaml.safe_dump({"cluster_id": name, "distribution": "foreign"}))
    assert load_job_metadata(name, cache_dir=str(tmp_path)) is None
    remove_job_metadata(name, cache_dir=str(tmp_path))
    assert path.exists()
    assert not owns_metadata({"cluster_id": "sparkrun_abc", "sparkrun_version": "0.3.9"})
    with pytest.raises(ValueError, match="another distribution"):
        docker_owner_guard("sparkrun_1234567890abcdef_123456abcdef_head")
    assert "sparkrun.distribution" in docker_owner_guard(name + "_head")


def test_alternate_fallback_and_telemetry_are_explicit():
    alternate()
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.hardware import resolve_fallback_hardware
    from sparkrun.telemetry.config import telemetry_enabled, telemetry_endpoint

    assert not telemetry_enabled(SparkrunConfig())
    assert telemetry_endpoint() == ""
    with pytest.raises(ValueError, match="metadata is required"):
        resolve_fallback_hardware()


def fake_entry(name, module, package="example-provider"):
    return SimpleNamespace(
        name=name, value=module.__name__, dist=SimpleNamespace(name=package, version="1.0"), load=Mock(return_value=module)
    )


def test_disabled_installed_plugin_never_imported(monkeypatch):
    from sparkrun.core import installed_plugins as plugins

    alternate()
    ep = fake_entry("example", ModuleType("unselected"))
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [ep])
    from scitrera_app_framework import Variables

    plugins.load_installed_plugins(Variables())
    ep.load.assert_not_called()
    assert plugins.installed_plugin_inventory()[0].package == "example-provider"


@pytest.mark.parametrize("failure", ["missing", "incompatible", "hook", "disabled", "duplicate"])
def test_required_integration_failures_keep_diagnostics_and_block_launch(monkeypatch, failure):
    from sparkrun.core import installed_plugins as plugins
    from sparkrun.core.config import SparkrunConfig
    from scitrera_app_framework import Variables

    alternate(required_integrations=("example",))
    monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS")
    module = ModuleType("required_example")
    module.SPARKRUN_PLUGIN_API_VERSION = 999 if failure == "incompatible" else 1
    module.register = Mock(side_effect=RuntimeError("hook failed") if failure == "hook" else None)
    ep = fake_entry("example", module)
    entries = [] if failure == "missing" else [ep]
    if failure == "duplicate":
        entries.append(fake_entry("example", module, "other-provider"))
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: entries)
    config = SparkrunConfig()
    if failure == "disabled":
        config.set("integrations.example", False)
    plugins.load_installed_plugins(Variables(), config=config)
    assert any(p.failure for p in plugins.installed_plugin_inventory())
    with pytest.raises(plugins.RequiredIntegrationError, match="example"):
        plugins.require_integrations()


def test_failed_optional_plugin_rolls_back_registrations(monkeypatch):
    from sparkrun.core import installed_plugins as plugins, cli_registry
    from sparkrun.core.features import FeatureFlag, get_feature
    from scitrera_app_framework import Variables

    alternate(integrations=("example",))
    monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS")
    module = ModuleType("broken_optional")
    module.SPARKRUN_PLUGIN_API_VERSION = 1
    module.FEATURE_DEFINITIONS = [FeatureFlag("example.transaction", "transaction test")]

    def broken(v):
        from sparkrun.proxy.gateway import register_gateway

        register_gateway("broken-example", feature_flag="example.transaction", loader=lambda: None)
        cli_registry.register_cli_command(lambda: None, name="broken-example")
        raise RuntimeError("rollback me")

    module.register = broken
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [fake_entry("example", module)])
    plugins.load_installed_plugins(Variables())
    assert get_feature("example.transaction") is None
    assert not any(c.name == "broken-example" for c in cli_registry.registered_cli_commands())
    from sparkrun.proxy.gateway import _GATEWAY_LOADERS

    assert "broken-example" not in _GATEWAY_LOADERS
    assert not plugins.installed_plugin_inventory()[0].loaded
    plugins.require_integrations()  # optional failure does not prevent independent launches


def test_implementation_conflicts_name_both_providers():
    from sparkrun.core.installed_plugins import claim_implementation, PluginConflictError

    v = object()
    a = type("First", (), {"runtime_name": "example"})
    b = type("Second", (), {"runtime_name": "example"})
    claim_implementation(a, v)
    claim_implementation(a, v)
    with pytest.raises(PluginConflictError, match="First.*Second"):
        claim_implementation(b, v)


def test_launcher_manifest_propagates_profile_and_owner():
    profile = alternate(profile_ref="example.profile:PROFILE")
    from sparkrun.plugins.k8s.orchestration.job import LauncherJobSpec, job_manifest

    manifest = job_manifest(LauncherJobSpec(name="example", image="local/test", command=["python", "-m", "sparkrun"]))
    assert manifest["metadata"]["namespace"] == "alternate-test"
    assert manifest["metadata"]["labels"]["sparkrun.distribution"] == "alternate-test"
    env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    assert {"name": "SPARKRUN_APPLICATION_PROFILE", "value": profile.profile_ref} in env


@pytest.mark.parametrize("owner,allowed", [("alternate-test", True), ("sparkrun", False), ("", False)])
def test_docker_stop_checks_persisted_owner_in_real_shell(owner, allowed):
    import subprocess
    from sparkrun.orchestration.executors.docker import DockerExecutor

    alternate()
    name = "alternate-test_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    stub = 'docker() { if [ "$1" = inspect ]; then printf %s "' + owner + '"; else echo MUTATED; fi; }\n'
    result = subprocess.run(["bash", "-c", stub + DockerExecutor().stop_cmd(name)], capture_output=True, text=True)
    assert ("MUTATED" in result.stdout) == allowed
    assert (result.returncode == 0) == allowed


def test_docker_stop_missing_container_is_idempotent():
    import subprocess
    from sparkrun.orchestration.executors.docker import DockerExecutor

    alternate()
    stub = 'docker() { if [ "$1" = inspect ]; then return 1; fi; return 0; }\n'
    result = subprocess.run(["bash", "-c", stub + DockerExecutor().stop_cmd("missing")], capture_output=True, text=True)
    assert result.returncode == 0


def test_native_stop_preserves_foreign_pidfile(tmp_path):
    import subprocess
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.executors._base import ExecutorConfig

    alternate()
    name = "alternate-test_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    pidfile = tmp_path / (name + ".pid")
    pidfile.write_text("12345")
    pidfile.with_suffix(".pid.owner").write_text("sparkrun")
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    result = subprocess.run(["bash", "-c", "kill() { echo MUTATED; }; " + executor.stop_cmd(name)], capture_output=True, text=True)
    assert result.returncode != 0
    assert pidfile.read_text() == "12345"
    assert "MUTATED" not in result.stdout


def test_foreign_metadata_and_setup_survive_prune_and_delete(tmp_path):
    from sparkrun.orchestration.job_metadata import prune_job_metadata
    from sparkrun.core.setup_manifest import ManifestManager

    alternate()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    foreign = jobs / "foreign.yaml"
    foreign.write_text(yaml.safe_dump({"distribution": "sparkrun", "cluster_id": "sparkrun_legacy", "started_at": 1}))
    assert prune_job_metadata(cache_dir=str(tmp_path), keep_per_intent=0) == []
    assert foreign.exists()
    manifests = ManifestManager(tmp_path / "clusters")
    path = manifests._manifest_path("shared")
    path.write_text(yaml.safe_dump({"distribution": "sparkrun", "cluster": "shared", "version": 1}))
    manifests.delete("shared")
    assert path.exists()


def test_selected_plugin_defines_feature_before_extension_gate_and_registers_once(monkeypatch):
    from scitrera_app_framework import Variables
    from sparkrun.core import installed_plugins as plugins
    from sparkrun.core.features import FeatureFlag
    from sparkrun.orchestration.executors.docker import DockerExecutor

    alternate(integrations=("example",))
    monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS")
    module = ModuleType("feature_example")
    module.SPARKRUN_PLUGIN_API_VERSION = 1
    module.FEATURE_DEFINITIONS = [FeatureFlag("example.executor", "Example executor", default=True)]

    class ExampleExecutor(DockerExecutor):
        executor_name = "example"
        required_feature_flag = "example.executor"

    module.ExampleExecutor = ExampleExecutor
    module.register = Mock()
    ep = fake_entry("example", module)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [ep])
    v = Variables()
    plugins.load_installed_plugins(v)
    plugins.load_installed_plugins(v)
    assert plugins.installed_plugin_inventory()[0].loaded
    assert ExampleExecutor().is_multi_extension(v)
    ep.load.assert_called_once()
    module.register.assert_called_once_with(v)


def test_config_pin_does_not_persist_inherited_subtree(tmp_path):
    alternate(defaults={"k8s": {"kubectl": {"pinned": {"inherited": "1.31"}}, "other": True}})
    from sparkrun.core.config import SparkrunConfig

    config = SparkrunConfig(tmp_path / "config.yaml")
    from sparkrun.plugins.k8s.config import K8sSettings

    K8sSettings(config).pin_kubectl_version("mine", "1.32")
    config.save()
    assert yaml.safe_load(config.config_path.read_text()) == {"k8s": {"kubectl": {"pinned": {"mine": "1.32"}}}}
    assert config.get("k8s.kubectl.pinned") == {"inherited": "1.31", "mine": "1.32"}


def test_profile_cannot_update_a_different_package():
    with pytest.raises(ValueError, match="owning distribution package"):
        alternate(update_sources={"stable": UpdateSource("sparkrun")})


def test_version_diagnostics_separate_distribution_and_core_commits(monkeypatch):
    alternate()
    from sparkrun.core import version
    from sparkrun.core.config import SparkrunConfig

    monkeypatch.setattr(version, "version", lambda package: "1.0" if package == "alternate-test" else "0.3.9")
    monkeypatch.setattr(
        version,
        "distribution",
        lambda package: SimpleNamespace(read_text=lambda path: json.dumps({"vcs_info": {"commit_id": package + "-commit"}})),
    )
    data = version.version_diagnostics(SparkrunConfig())
    assert data["version"] == "1.0" and data["core"]["version"] == "0.3.9"
    assert data["distribution"]["commit"] == "alternate-test-commit"
    assert data["core"]["commit"] == "sparkrun-commit"

    from click.testing import CliRunner
    from sparkrun.cli._setup import _commands

    monkeypatch.setattr(_commands, "_get_context", lambda ctx: SimpleNamespace(config=SparkrunConfig()))
    output = CliRunner().invoke(_commands.setup_version)
    assert output.exit_code == 0, output.output
    assert "Application: alternate-test 1.0 (commit alternate-test-commit)" in output.output
    assert "Core:        sparkrun 0.3.9 (commit sparkrun-commit)" in output.output
    if not data["plugins"]:
        assert "Loaded plugins:\n  (none)" in output.output
    output = CliRunner().invoke(_commands.setup_version, ["--json"])
    assert output.exit_code == 0, output.output
    assert json.loads(output.output) == data


def test_pinned_legacy_gateway_cannot_run_under_an_alternate_profile(monkeypatch):
    alternate(feature_defaults={"gateway.sparkroute": True})
    from sparkrun.core.in_tree_plugins import load_in_tree_plugins
    from sparkrun.core.plugin_inventory import list_plugins
    from sparkrun.proxy.gateway import GatewayUnavailableError, gateway_class, require_gateway_enabled
    from scitrera_app_framework import Variables
    import sparkrun.core.in_tree_plugins as loader

    monkeypatch.setattr(loader, "plugin_application_profile_api", lambda name: None)
    imports = Mock(side_effect=AssertionError("unsupported plugin must not import"))
    with monkeypatch.context() as patch:
        patch.setattr(loader, "iter_in_tree_plugin_names", lambda package=None: ["sparkroute"])
        patch.setattr(loader.importlib, "import_module", imports)
        assert load_in_tree_plugins(Variables()) == []
        imports.assert_not_called()
        row = next(p for p in list_plugins() if p.name == "sparkroute")
        assert row.enabled and row.failure and not row.loaded
    with pytest.raises(GatewayUnavailableError, match="pinned SparkRoute"):
        gateway_class("sparkroute")
    with pytest.raises(GatewayUnavailableError, match="pinned SparkRoute"):
        require_gateway_enabled("sparkroute")


def test_post_install_tool_resolution_is_independent_of_running_uvx_environment(monkeypatch, tmp_path):
    alternate()
    from sparkrun.cli import _self_update as update

    monkeypatch.setattr(update.sys, "prefix", str(tmp_path / "uvx"))
    monkeypatch.setattr(
        update.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=0, stdout="alternate-test v1 (%s)\n" % (tmp_path / "installed"))),
    )
    assert update.installed_tool_executable("uv") == tmp_path / "installed/bin/alternate-test"
    assert not update.is_uv_tool_install("uv")


def test_explicit_feature_environment_alias():
    alternate(env_aliases={"FEATURE_EXECUTOR_LOCAL": ("COMPAT_LOCAL",)})
    from sparkrun.core.features import is_feature_enabled

    assert is_feature_enabled("executor.local", env={"COMPAT_LOCAL": "1"})
    assert not is_feature_enabled("executor.local", env={"COMPAT_LOCAL": "1", "ALTERNATE_TEST_FEATURE_EXECUTOR_LOCAL": "0"})


def test_running_snapshot_preserves_other_distribution_in_shared_cache(tmp_path):
    from sparkrun.orchestration.job_metadata import save_running_snapshot, load_running_snapshot

    legacy = tmp_path / "running.json"
    legacy.write_text("preserve this other product's snapshot")
    alternate()
    save_running_snapshot(host_snapshot(["alternate-test_example"], ["host"]), cache_dir=str(tmp_path))
    assert legacy.read_text() == "preserve this other product's snapshot"
    assert load_running_snapshot(cache_dir=str(tmp_path)) == host_snapshot({"alternate-test_example"}, ["host"])


def test_application_feature_channel_defaults_are_frozen_and_validated():
    original = {"stable": {"cli.tune": False}}
    profile = alternate(feature_channel_defaults=original)
    original["stable"]["cli.tune"] = True
    assert profile.feature_channel_defaults["stable"]["cli.tune"] is False
    with pytest.raises(TypeError):
        profile.feature_channel_defaults["stable"]["cli.tune"] = True


@pytest.mark.parametrize(
    "defaults,error",
    [
        ({"stable": True}, TypeError),
        ({"stable": {"cli.tune": "false"}}, TypeError),
        ({"stable": {"cli.tune": 0}}, TypeError),
        ({"../next": {}}, ValueError),
        ({"next": {"cli.tune": True}}, ValueError),
    ],
)
def test_invalid_application_feature_channel_defaults(defaults, error):
    with pytest.raises(error):
        alternate(feature_channel_defaults=defaults)


@pytest.mark.parametrize("application_channel", ["stable", "next"])
@pytest.mark.parametrize("core_channel", ["stable", "alpha"])
def test_application_feature_channels_are_independent(application_channel, core_channel, tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import feature_source, is_feature_enabled

    alternate(
        update_sources={"stable": UpdateSource("alternate-test"), "next": UpdateSource("alternate-test==2.0b1")},
        feature_defaults={"cli.tune": False},
        feature_channel_defaults={"next": {"cli.tune": True, "executor.docker": False}},
    )
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set("self_update.channel", application_channel)
    config.set("features.channel", core_channel)
    assert config.feature_channel == core_channel
    assert is_feature_enabled("cli.tune", config=config, env={}) == (application_channel == "next")
    assert feature_source("cli.tune", config=config, env={}) == (
        "distribution-channel" if application_channel == "next" else "distribution"
    )
    assert is_feature_enabled("executor.docker", config=config, env={}) == (application_channel != "next")
    # Unspecified flags retain upstream maturity behavior, even under `next`.
    assert is_feature_enabled("gateway.sparkroute", config=config, env={}) == (core_channel == "alpha")
    # The explicit channel argument continues to mean core maturity only.
    assert is_feature_enabled("cli.tune", config=config, channel="alpha", env={}) == (application_channel == "next")


def test_application_channel_precedence_and_default_without_config(tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import feature_source, is_feature_enabled

    alternate(
        default_channel="next",
        update_sources={"next": UpdateSource("alternate-test")},
        feature_defaults={"cli.tune": True},
        feature_channel_defaults={"next": {"cli.tune": False}},
    )
    assert not is_feature_enabled("cli.tune", env={})
    assert feature_source("cli.tune", env={}) == "distribution-channel"
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set("features", {"cli.tune": True})
    assert is_feature_enabled("cli.tune", config=config, env={})
    assert feature_source("cli.tune", config=config, env={}) == "config"
    env = {"ALTERNATE_TEST_FEATURE_CLI_TUNE": "0"}
    assert not is_feature_enabled("cli.tune", config=config, env=env)
    assert feature_source("cli.tune", config=config, env=env) == "env"
    config.save()
    assert yaml.safe_load(config.config_path.read_text()) == {"features": {"cli.tune": True}}


def test_custom_channel_updates_target_application_sources(tmp_path):
    from sparkrun.core.channels import channel_requirement, is_git_channel
    from sparkrun.core.config import SparkrunConfig

    requirement = "alternate-test @ git+https://example.test/alternate.git@next"
    alternate(update_sources={"stable": UpdateSource("alternate-test"), "next": UpdateSource(requirement, "git")})
    from sparkrun.cli._self_update import update_argv

    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set_self_update_channel("next")
    assert config.self_update_channel == "next"
    assert config.feature_channel == "stable"
    assert is_git_channel("next")
    assert update_argv("uv", config.self_update_channel) == ["uv", "tool", "install", requirement, "--force"]
    assert update_argv("uv", "stable") == ["uv", "tool", "upgrade", "alternate-test"]
    with pytest.raises(ValueError, match="does not support"):
        channel_requirement("beta")


def test_feature_diagnostics_identify_application_channel(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from sparkrun.core.config import SparkrunConfig

    alternate(
        update_sources={"stable": UpdateSource("alternate-test"), "next": UpdateSource("alternate-test==2.0b1")},
        feature_channel_defaults={"next": {"cli.tune": False}},
    )
    from sparkrun.cli._setup import _commands

    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set("self_update.channel", "next")
    monkeypatch.setattr(_commands, "_get_context", lambda ctx: SimpleNamespace(config=config))
    result = CliRunner().invoke(_commands.setup_features_list, ["--json"])
    assert result.exit_code == 0, result.output
    row = next(row for row in json.loads(result.output) if row["name"] == "cli.tune")
    assert row["enabled"] is False
    assert row["source"] == "distribution-channel"
    assert row["channel"] == "stable"
    assert row["application_channel"] == "next"
    result = CliRunner().invoke(_commands.setup_features_list)
    assert result.exit_code == 0, result.output
    assert "Feature channel: stable" in result.output
    assert "Application channel: next" in result.output


def test_feature_reset_restores_application_channel_policy(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from sparkrun.core.config import SparkrunConfig

    alternate(
        default_channel="next",
        update_sources={"next": UpdateSource("alternate-test")},
        feature_channel_defaults={"next": {"cli.tune": False}},
    )
    from sparkrun.cli._setup import _commands

    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set("features", {"cli.tune": True, "executor.docker": False, "channel": "alpha"})
    config.save()
    monkeypatch.setattr(_commands, "_get_context", lambda ctx: SimpleNamespace(config=config))
    result = CliRunner().invoke(_commands.setup_features_reset, ["cli.tune"])
    assert result.exit_code == 0, result.output
    reloaded = SparkrunConfig(config.config_path)
    assert reloaded.feature_override("cli.tune") is None
    assert not reloaded.is_feature_enabled("cli.tune")
    assert reloaded.feature_override("executor.docker") is False
    assert reloaded.feature_channel == "alpha"
    assert yaml.safe_load(config.config_path.read_text()) == {"features": {"executor.docker": False, "channel": "alpha"}}


@pytest.mark.parametrize("field", ["integrations", "required_integrations", "bootstrap_registry_urls"])
@pytest.mark.parametrize("bad", ["arena", {"arena": True}, 1, None, [None], [""]])
def test_profile_collection_fields_reject_invalid_shapes(field, bad):
    from sparkrun.core.application_profile import ApplicationProfile

    with pytest.raises((TypeError, ValueError), match=field):
        ApplicationProfile(id="example", display_name="Example", command="example", package="example", **{field: bad})


@pytest.mark.parametrize("field", ["integrations", "required_integrations", "bootstrap_registry_urls"])
def test_profile_collection_fields_copy_lists_to_tuples(field):
    from sparkrun.core.application_profile import ApplicationProfile

    values = ["example"]
    profile = ApplicationProfile(id="example", display_name="Example", command="example", package="example", **{field: values})
    values.append("later")
    assert getattr(profile, field) == ("example",)


@pytest.mark.parametrize("bad", ["COMPAT_KEY", {"COMPAT_KEY": True}, 1, None, [None]])
def test_profile_env_aliases_require_sequences_of_names(bad):
    from sparkrun.core.application_profile import ApplicationProfile

    with pytest.raises(TypeError, match="env_aliases"):
        ApplicationProfile(id="example", display_name="Example", command="example", package="example", env_aliases={"SETTING": bad})


def test_inventory_is_immutable_and_cannot_change_required_enforcement(monkeypatch):
    from dataclasses import FrozenInstanceError
    from scitrera_app_framework import Variables
    from sparkrun.core import installed_plugins as plugins

    alternate(required_integrations=("example",))
    monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS")
    module = ModuleType("required_example")
    module.SPARKRUN_PLUGIN_API_VERSION = 1
    module.register = Mock(side_effect=RuntimeError("provider failed"))
    entry = fake_entry("example", module)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [entry])
    discovered = plugins.discover_installed_plugins()[0]
    assert not hasattr(discovered, "entry_point")
    entry.load.assert_not_called()
    plugins.load_installed_plugins(Variables())
    inventory = plugins.installed_plugin_inventory()
    with pytest.raises(FrozenInstanceError):
        inventory[0].loaded = True
    inventory.clear()
    assert discovered.failure is None
    assert plugins.installed_plugin_inventory()[0].failure
    with pytest.raises(plugins.RequiredIntegrationError, match="provider failed"):
        plugins.require_integrations()


def test_profile_namespaces_are_materialized_strings():
    from sparkrun.core.application_profile import ApplicationProfile

    profile = ApplicationProfile(id="example", display_name="Example", command="example", package="example")
    for name in ("config_namespace", "cache_namespace", "state_namespace", "resource_namespace"):
        assert getattr(profile, name) == "example"
    assert profile.env_prefix == "EXAMPLE"


@pytest.mark.parametrize("name", ["config_namespace", "cache_namespace", "state_namespace", "resource_namespace", "env_prefix"])
def test_profile_namespaces_reject_none(name):
    from sparkrun.core.application_profile import ApplicationProfile

    with pytest.raises(ValueError):
        ApplicationProfile(id="example", display_name="Example", command="example", package="example", **{name: None})
