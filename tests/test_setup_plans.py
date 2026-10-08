"""Hardware ownership, positive selection, and discovery-before-probe contracts."""

from types import ModuleType
from unittest.mock import Mock
import subprocess

import pytest

from sparkrun.core.features import FeatureFlag, register_feature
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.setup_actions import SetupActionContext
from sparkrun.core.setup_models import CheckItem, WARN, SKIP
from sparkrun.core.setup_plans import SetupPlan
from sparkrun.core.setup_steps import (
    SetupStep,
    apply_setup_step,
    build_setup_plan,
    register_setup_step,
    setup_probe_script,
    setup_selection,
    validate_setup_plans,
)
from sparkrun.platforms.nvidia_generic import GenericNvidiaPlatform
from sparkrun.orchestration.ssh import RemoteResult
from test_setup_steps import state_context, approve_test_steps, FACTS


@pytest.fixture
def owned_platform(monkeypatch):
    from sparkrun import platforms

    class OwnedPlatform(GenericNvidiaPlatform):
        platform_name = "reviewed-device"
        setup_plans = (SetupPlan("docker", ("docker", "docker_group")), SetupPlan("local", ()))

        def matches(self, hardware):
            return any(a.vendor == "nvidia" and a.model == "reviewed" for a in hardware.accelerators)

    platform = OwnedPlatform()
    monkeypatch.setattr(platforms, "_REGISTRY", [platform, *platforms.iter_platforms()])
    return platform


def owned_state():
    return state_context(HostHardware([AcceleratorSpec("nvidia", "reviewed")]))


def test_registered_enabled_plugin_step_is_not_implicitly_selected(owned_platform, monkeypatch):
    callback, check = Mock(), Mock(return_value=CheckItem("service", "service", WARN))
    register_feature(FeatureFlag("plugin.service", "Service", default=True))
    register_setup_step(
        SetupStep(
            "service", "Service", checks=(check,), apply=callback, feature_flag="plugin.service", probe_script="echo SERVICE_PROBED=1"
        )
    )
    state, context = owned_state()
    monkeypatch.setenv("SPARKRUN_FEATURE_PLUGIN_SERVICE", "1")
    assert "SERVICE_PROBED" not in setup_probe_script(state, context)
    entry = next(row for row in build_setup_plan(state, context) if row.step.key == "service")
    assert not entry.selected and entry.platform == "reviewed-device"
    assert apply_setup_step("service", state, context, SetupActionContext("tester")).status == SKIP
    callback.assert_not_called()
    check.assert_not_called()
    owned_platform.setup_plans = (SetupPlan("docker", ("docker", "docker_group", "service")),)
    assert "SERVICE_PROBED" in setup_probe_script(state, context)
    assert next(row for row in build_setup_plan(state, context) if row.step.key == "service").selected


def test_new_core_step_does_not_join_existing_plan(owned_platform, monkeypatch):
    from sparkrun.core import setup_steps

    check = Mock()
    original = setup_steps.builtin_steps()
    register_feature(FeatureFlag("setup.steps.future", "Future", default=True))
    monkeypatch.setattr(setup_steps, "builtin_steps", lambda: (*original, SetupStep("future", "Future", checks=(check,))))
    for state, context in (owned_state(), state_context()):
        row = next(row for row in build_setup_plan(state, context) if row.step.key == "future")
        assert not row.selected and "not included" in row.reason
    check.assert_not_called()


@pytest.mark.parametrize("executor", [None, "local", "k8s", "unregistered"])
def test_executor_selection_has_no_docker_fallback(owned_platform, executor):
    state, context = owned_state()
    context.strict = False  # Even legacy/non-strict callers must supply an executor.
    context.executor_names = {state.host: executor} if executor else {}
    assert {row.step.key for row in build_setup_plan(state, context) if row.selected} == {"hardware"}


@pytest.mark.parametrize("operating_system", [None, "Darwin", "Windows"])
def test_plan_requires_qualified_os(owned_platform, operating_system):
    state, context = owned_state()
    state.facts["CHECK_OS"] = operating_system
    assert {row.step.key for row in build_setup_plan(state, context) if row.selected} == {"hardware"}


def test_broken_specific_matcher_does_not_fall_through_to_generic(owned_platform, monkeypatch):
    monkeypatch.setattr(owned_platform, "matches", Mock(side_effect=RuntimeError("broken matcher")))
    state, context = owned_state()
    assert setup_selection(state, context)[1]["docker"] == "hardware platform could not be resolved"
    assert apply_setup_step("docker_group", state, context, SetupActionContext("tester")).status == SKIP


@pytest.mark.parametrize("key", ["docker_group", "nvidia_cdi", "earlyoom", "sudoers"])
def test_direct_core_actions_honor_plan(key, owned_platform):
    from sparkrun.core import setup_actions

    owned_platform.setup_plans = ()
    state, context = owned_state()
    dispatch = Mock()
    assert getattr(setup_actions, key)(state, context, SetupActionContext("tester", dispatch=dispatch)).status == SKIP
    dispatch.assert_not_called()


def test_dependency_policy_blocks_probe_without_using_uncollected_readiness(monkeypatch):
    register_feature(FeatureFlag("plugin.service", "Service", default=True))
    register_setup_step(
        SetupStep("service", "Service", requires=("docker",), feature_flag="plugin.service", probe_script="echo SERVICE_PROBED=1")
    )
    approve_test_steps(monkeypatch, "service")
    state, context = state_context()
    state.facts = {"CHECK_OS": "Linux", "CHECK_APT": "1", "CHECK_SYSTEMD": "1"}
    assert "SERVICE_PROBED" in setup_probe_script(state, context)
    monkeypatch.setenv("SPARKRUN_FEATURE_SETUP_STEPS_DOCKER", "0")
    assert "SERVICE_PROBED" not in setup_probe_script(state, context)


@pytest.mark.parametrize("steps,error", [(("missing",), "Unknown setup step"), (("docker_group",), "omits prerequisites")])
def test_incomplete_plans_fail_without_expanding_dependencies(owned_platform, steps, error):
    owned_platform.setup_plans = (SetupPlan("docker", steps),)
    with pytest.raises(ValueError, match=error):
        validate_setup_plans()
    with pytest.raises(ValueError, match=error):
        build_setup_plan(*owned_state())


def test_invalid_plugin_plan_rolls_back_steps_features_and_platform(monkeypatch, v):
    from sparkrun import platforms
    from sparkrun.core.external_plugins import load_plugin_module
    from sparkrun.core.features import get_feature
    from sparkrun.core.setup_steps import all_setup_steps

    monkeypatch.setattr(platforms, "_REGISTRY", platforms.iter_platforms())
    before = platforms.iter_platforms()
    module = ModuleType("invalid_setup_owner")
    module.SPARKRUN_PLUGIN_API_VERSION = 1

    class InvalidPlatform(GenericNvidiaPlatform):
        platform_name = "invalid-plan"
        setup_plans = (SetupPlan("docker", ("owned_service", "missing_provider")),)

    def register(v):
        register_feature(FeatureFlag("plugin.owned_service", "Service", default=True))
        register_setup_step(SetupStep("owned_service", "Service", feature_flag="plugin.owned_service"))
        platforms.register_platform(InvalidPlatform(), prepend=True)

    module.register = register
    with pytest.raises(ValueError, match="Unknown setup step"):
        load_plugin_module(module, v, strict=True)
    assert platforms.iter_platforms() == before
    assert get_feature("plugin.owned_service") is None
    assert "owned_service" not in {step.key for step in all_setup_steps()}


def test_plan_declarations_are_immutable_and_reject_duplicate_executors(owned_platform):
    keys = ["docker"]
    plan = SetupPlan("docker", keys)
    keys.append("earlyoom")
    assert plan.steps == ("docker",)
    with pytest.raises(ValueError, match="Duplicate setup plan step"):
        SetupPlan("docker", ("docker", "docker"))
    owned_platform.setup_plans = (plan, plan)
    with pytest.raises(ValueError, match="Duplicate setup executor"):
        validate_setup_plans()


def test_probes_discover_before_selection_and_keep_mixed_hosts_separate(owned_platform, monkeypatch):
    from sparkrun.core.setup_probe import probe_setup_hosts
    from sparkrun.core.hardware import default_dgx_spark_hardware

    register_feature(FeatureFlag("plugin.service", "Service", default=True))
    register_setup_step(SetupStep("service", "Service", feature_flag="plugin.service", probe_script="echo SERVICE_PROBED=1"))
    owned_platform.setup_plans = (SetupPlan("docker", ("docker", "docker_group", "service")),)
    calls = {host: [] for host in ("owned", "spark", "unknown")}
    parsed = []

    def parse(stdout):
        host = stdout.splitlines()[0]
        parsed.append(host)
        return owned_state()[0].hardware if host == "owned" else default_dgx_spark_hardware()

    def remote(host, script, **kwargs):
        calls[host].append(script)
        syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert syntax.returncode == 0, syntax.stderr
        if "CHECK_DISCOVERY_COMPLETE" in script:
            assert "SERVICE_PROBED" not in script and "CHECK_EARLYOOM" not in script
            sentinel = "SPARKRUN_PROBE_ACCEL_END\n" if host != "unknown" else ""
            return RemoteResult(
                host, 0, host + "\n" + sentinel + "CHECK_DISCOVERY_COMPLETE=1\nCHECK_OS=Linux\nCHECK_APT=1\nCHECK_SYSTEMD=1", ""
            )
        assert set(parsed) == {"owned", "spark"}, "All targets must be identified before optional probes"
        assert ("SERVICE_PROBED" in script) is (host == "owned")
        return RemoteResult(host, 0, "\n".join(key + "=" + value for key, value in FACTS.items()), "")

    monkeypatch.setattr("sparkrun.core.hardware_probe._parse_probe_result", parse)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    cx7, rdma = Mock(return_value={}), Mock(return_value={})
    monkeypatch.setattr("sparkrun.orchestration.networking.detect_cx7_for_hosts", cx7)
    monkeypatch.setattr("sparkrun.api.setup._rdma._run_probe", rdma)
    states, context = probe_setup_hosts(list(calls), ssh_kwargs={}, config=state_context()[1].config)
    assert all(state.reachable for state in states.values())
    assert [len(scripts) for scripts in calls.values()] == [2, 2, 1]
    cx7.assert_called_once_with(["spark"], ssh_kwargs={})
    rdma.assert_called_once_with(["spark"], {}, dry_run=False)
    assert "SETUP_STEPS='docker docker_group service'" in calls["owned"][1]
    assert not any(row.selected and row.step.key == "earlyoom" for row in build_setup_plan(states["owned"], context))


def test_disabled_shell_blocks_execute_no_optional_commands(tmp_path):
    from sparkrun.scripts import read_script

    # Replace command discovery with a failing recorder. Baseline user identity
    # is allowed; no Docker/toolkit/systemd/sudo/peer command may be consulted.
    log = tmp_path / "optional-commands"
    prelude = f'command() {{ echo "$*" >> {str(log)!r}; return 1; }}\nsudo() {{ echo "sudo $*" >> {str(log)!r}; return 1; }}\n'
    script = read_script("setup_check.sh").format(steps="''", peers="''", needs_sudo=0)
    result = subprocess.run(["bash", "-s"], input=prelude + script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "CHECK_COMPLETE=1" in result.stdout
    assert "CHECK_DOCKER" not in result.stdout and "CHECK_EARLYOOM" not in result.stdout
    assert "CHECK_SUDO_NOPASSWD" not in result.stdout
    assert not log.exists()


def test_local_peer_discovery_rejects_unqualified_hardware(owned_platform, monkeypatch):
    from sparkrun.core.setup_probe import local_setup_step_supported

    state, context = owned_state()
    monkeypatch.setattr("sparkrun.core.setup_probe._discovered_state", lambda *args: state)
    run = Mock(return_value=type("Result", (), dict(returncode=0, stdout="", stderr=""))())
    monkeypatch.setattr(subprocess, "run", run)
    assert not local_setup_step_supported("cx7", context.config)
    assert run.call_count == 1  # hardware discovery only


def test_readiness_reports_missing_plan_instead_of_claiming_full_coverage(owned_platform):
    from sparkrun.core.setup_steps import evaluate_host

    owned_platform.setup_plans = ()
    state, context = owned_state()
    result = next(item for item in evaluate_host(state, context) if item.key == "setup_plan")
    assert result.status == WARN and "discovery only" in result.detail


@pytest.mark.parametrize("command", ["earlyoom", "cx7"])
@pytest.mark.parametrize("alternate_app", [False, True])
def test_standalone_hardware_commands_reject_unapproved_targets(command, alternate_app, owned_platform, monkeypatch):
    from click.testing import CliRunner
    from sparkrun.cli import main
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile

    if alternate_app:
        select_application_profile(ApplicationProfile(id="custom", display_name="Custom", package="custom", command="custom"))
    monkeypatch.setattr("sparkrun.cli._setup._commands._resolve_setup_context", lambda *args: (["h1", "h2"], "tester", {}))
    state, _ = owned_state()
    monkeypatch.setattr("sparkrun.core.hardware_probe._parse_probe_result", lambda *args: state.hardware)
    remote = Mock(
        side_effect=lambda host, *args, **kwargs: RemoteResult(
            host, 0, "SPARKRUN_PROBE_ACCEL_END\nCHECK_OS=Linux\nCHECK_APT=1\nCHECK_SYSTEMD=1\nCHECK_DISCOVERY_COMPLETE=1", ""
        )
    )
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    detection, sudo = Mock(), Mock()
    monkeypatch.setattr("sparkrun.orchestration.networking.detect_cx7_for_hosts", detection)
    monkeypatch.setattr("sparkrun.orchestration.sudo.run_with_sudo_fallback", sudo)
    result = CliRunner().invoke(main, ["setup", command, "--hosts", "h1,h2"])
    assert result.exit_code != 0 and "not included in reviewed-device/docker setup plan" in result.output
    assert remote.call_count == 2  # Discovery only, with no readiness/topology probes.
    detection.assert_not_called()
    sudo.assert_not_called()


def test_standalone_preflight_uses_hardware_owner_independent_of_application(monkeypatch):
    from sparkrun.cli._setup._step_runner import require_setup_step_targets
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile
    from sparkrun.core.hardware import default_dgx_spark_hardware

    select_application_profile(ApplicationProfile(id="custom", display_name="Custom", package="custom", command="custom"))
    monkeypatch.setattr("sparkrun.core.hardware_probe._parse_probe_result", lambda *args: default_dgx_spark_hardware())
    remote = Mock(
        side_effect=lambda host, *args, **kwargs: RemoteResult(
            host, 0, "SPARKRUN_PROBE_ACCEL_END\nCHECK_OS=Linux\nCHECK_APT=1\nCHECK_SYSTEMD=1\nCHECK_DISCOVERY_COMPLETE=1", ""
        )
    )
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    require_setup_step_targets("earlyoom", ["h1"], {}, state_context()[1].config, cluster_name=None, explicit_hosts=True, dry_run=False)
    remote.assert_called_once()


def test_standalone_preview_requires_inventory_and_never_probes(monkeypatch):
    import click
    from sparkrun.cli._setup._step_runner import require_setup_step_targets

    remote = Mock()
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    with pytest.raises(click.ClickException, match="hardware has not been identified"):
        require_setup_step_targets(
            "earlyoom", ["unknown"], {}, state_context()[1].config, cluster_name=None, explicit_hosts=True, dry_run=True
        )
    remote.assert_not_called()
