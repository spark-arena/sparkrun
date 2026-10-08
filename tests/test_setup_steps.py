"""Hardware-aware setup selection and action boundary regressions."""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.features import FeatureFlag, register_feature
from sparkrun.core.hardware import default_dgx_spark_hardware
from sparkrun.core.setup_actions import SetupActionContext, SetupActionResult
from sparkrun.core.setup_models import CheckContext, CheckItem, HostState, FAIL, OK, WARN, SKIP
from sparkrun.core.setup_steps import SetupStep, register_setup_step, build_setup_plan, apply_setup_step, evaluate_host
from sparkrun.orchestration.ssh import RemoteResult


FACTS = {
    "CHECK_COMPLETE": "1",
    "CHECK_DISCOVERY_COMPLETE": "1",
    "CHECK_OS": "Linux",
    "CHECK_APT": "1",
    "CHECK_SYSTEMD": "1",
    "CHECK_NETPLAN": "1",
    "CHECK_USER": "tester",
    "CHECK_DOCKER_INSTALLED": "1",
    "CHECK_DOCKER_GROUP": "1",
    "CHECK_DOCKER_USABLE": "1",
    "CHECK_NVIDIA_CTK": "1",
    "CHECK_CDI_SPEC": "1",
    "CHECK_GPU_PRESENT": "1",
    "CHECK_EARLYOOM_ACTIVE": "1",
    "CHECK_SUDOERS_CHOWN": "1",
    "CHECK_SUDOERS_DROPCACHES": "1",
}


def state_context(hardware=None, **facts):
    state = HostState("h1", facts={**FACTS, **facts}, hardware=hardware or default_dgx_spark_hardware())
    ctx = CheckContext("lab", False, config=SparkrunConfig(), executor_names={"h1": "docker"}, gpu_access_modes={"h1": "cdi"}, strict=True)
    return state, ctx


def approve_test_steps(monkeypatch, *keys):
    """Explicitly extend only the test Spark plan, without mutating core plans."""
    from copy import copy
    from dataclasses import replace
    from sparkrun import platforms

    registry = [copy(platform) for platform in platforms.iter_platforms()]
    spark = next(p for p in registry if p.platform_name == "dgx-spark")
    spark.setup_plans = tuple(
        replace(plan, steps=(*plan.steps, *keys)) if plan.executor == "docker" else plan for plan in spark.setup_plans
    )
    monkeypatch.setattr(platforms, "_REGISTRY", registry)


def test_application_policy_disables_checks_and_actions(monkeypatch):
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile

    select_application_profile(
        ApplicationProfile(
            id="custom", display_name="Custom", package="custom", command="custom", feature_defaults={"setup.steps.earlyoom": False}
        )
    )
    state, ctx = state_context(CHECK_EARLYOOM_ACTIVE="0")
    dispatch = mock.Mock()
    assert "earlyoom" not in {item.key for item in evaluate_host(state, ctx)}
    result = apply_setup_step("earlyoom", state, ctx, SetupActionContext("tester", dispatch=dispatch))
    assert result.status == SKIP
    dispatch.assert_not_called()
    monkeypatch.setenv("CUSTOM_FEATURE_SETUP_STEPS_EARLYOOM", "1")
    # Use the profile's actual prefix (default is derived from its identity).
    from sparkrun.core.application_profile import env_name

    monkeypatch.setenv(env_name("FEATURE_SETUP_STEPS_EARLYOOM"), "1")
    assert "earlyoom" in {item.key for item in evaluate_host(state, ctx)}


@pytest.mark.parametrize("executor", ["local", "k8s", "modal"])
def test_docker_changes_require_docker_executor(executor):
    state, ctx = state_context(CHECK_DOCKER_GROUP="0")
    ctx.executor_names["h1"] = executor
    assert not next(p for p in build_setup_plan(state, ctx) if p.step.key == "docker_group").selected


def test_missing_docker_blocks_group_changes():
    state, ctx = state_context(CHECK_DOCKER_INSTALLED="0", CHECK_DOCKER_GROUP="0")
    entry = next(p for p in build_setup_plan(state, ctx) if p.step.key == "docker_group")
    assert entry.blocked_by == ("docker",)
    action = mock.Mock()
    assert apply_setup_step("docker_group", state, ctx, SetupActionContext("tester", dispatch=action)).status == SKIP
    action.assert_not_called()


def test_unknown_hardware_and_unknown_executor_fail_readiness():
    state, ctx = state_context(CHECK_DOCKER_GROUP="0")
    state.hardware = None
    assert any(i.key == "hardware" and i.status == FAIL for i in evaluate_host(state, ctx))
    state.hardware = default_dgx_spark_hardware()
    ctx.executor_names.clear()
    assert any(i.key == "executor" and i.status == FAIL for i in evaluate_host(state, ctx))


def test_dry_run_never_invokes_plugin_action(monkeypatch):
    register_feature(FeatureFlag("setup.steps.test_action", "test", default=True))
    callback = mock.Mock()
    register_setup_step(
        SetupStep(
            "test_action",
            "test",
            checks=(lambda s, c: CheckItem("test", "test", WARN),),
            apply=callback,
            feature_flag="setup.steps.test_action",
        )
    )
    state, ctx = state_context()
    approve_test_steps(monkeypatch, "test_action")
    result = apply_setup_step("test_action", state, ctx, SetupActionContext("tester", dry_run=True))
    assert result.status == SKIP and "would apply" in result.detail
    callback.assert_not_called()


def test_plugin_action_dependency_and_registration_rollback(v, monkeypatch):
    from sparkrun.core.registration import registry_transaction
    from sparkrun.core.setup_steps import _STEPS

    register_feature(FeatureFlag("setup.steps.test_action", "test", default=True))
    with pytest.raises(RuntimeError), registry_transaction(v):
        register_setup_step(SetupStep("test_action", "test", feature_flag="setup.steps.test_action"))
        raise RuntimeError("broken plugin")
    assert "test_action" not in _STEPS
    callback = mock.Mock(return_value=SetupActionResult("h1", OK, "done", True))
    register_setup_step(
        SetupStep(
            "test_action",
            "test",
            checks=(lambda s, c: CheckItem("test", "test", WARN),),
            apply=callback,
            requires=("docker",),
            feature_flag="setup.steps.test_action",
        )
    )
    state, ctx = state_context()
    approve_test_steps(monkeypatch, "test_action")
    assert apply_setup_step("test_action", state, ctx, SetupActionContext("tester")).changed
    callback.assert_called_once()


def test_plugin_cycle_and_conflicts_fail():
    flag = "setup.steps.cycle"
    register_feature(FeatureFlag(flag, "test", default=True))
    register_setup_step(SetupStep("cycle_a", "A", requires=("cycle_b",), feature_flag=flag))
    register_setup_step(SetupStep("cycle_b", "B", requires=("cycle_a",), feature_flag=flag))
    state, ctx = state_context()
    with pytest.raises(ValueError, match="cycle"):
        build_setup_plan(state, ctx)
    with pytest.raises(ValueError, match="reserved"):
        register_setup_step(SetupStep("docker", "replacement", feature_flag=flag))


def test_cdi_action_only_when_selected_gpu_mode_requires_it():
    state, ctx = state_context(CHECK_CDI_SPEC="0")
    callback = mock.Mock(return_value=RemoteResult("h1", 0, "GENERATED: /etc/cdi/nvidia.yaml", ""))
    ctx.gpu_access_modes["h1"] = "gpus"
    assert apply_setup_step("nvidia_cdi", state, ctx, SetupActionContext("tester", dispatch=callback)).status == SKIP
    callback.assert_not_called()
    ctx.gpu_access_modes["h1"] = "cdi"
    result = apply_setup_step("nvidia_cdi", state, ctx, SetupActionContext("tester", dispatch=callback))
    assert result.status == OK and result.changed
    callback.assert_called_once()


def test_manifest_retains_per_host_changes(tmp_path):
    from sparkrun.core.setup_manifest import ManifestManager

    mgr = ManifestManager(tmp_path)
    mgr.record_phase("lab", "tester", ["h1"], "earlyoom", host_details={"h1": {"installed_package": False}})
    mgr.record_phase("lab", "tester", ["h2"], "earlyoom", host_details={"h2": {"installed_package": True}})
    record = mgr.load("lab").phases["earlyoom"]
    assert record.extra["host_details"] == {"h1": {"installed_package": False}, "h2": {"installed_package": True}}


@pytest.mark.parametrize("existing", [False, True])
def test_wizard_dry_run_has_no_install_probe_or_cluster_writes(tmp_path, v, existing):
    from sparkrun.cli import main
    from sparkrun.core.cluster_manager import ClusterManager

    manager = ClusterManager(tmp_path / "clusters")
    if existing:
        manager.create("lab", ["h1"], hosts_hardware={"h1": default_dgx_spark_hardware()})
        manager.set_default("lab")
    before = {str(p): p.read_bytes() for p in manager.clusters_dir.rglob("*") if p.is_file()}
    with (
        mock.patch("sparkrun.cli._common._get_cluster_manager", return_value=manager),
        mock.patch("sparkrun.cli._setup._get_cluster_manager", return_value=manager),
        mock.patch("sparkrun.core.setup_probe.probe_setup_hosts", side_effect=AssertionError("must not probe")),
        mock.patch("subprocess.run", side_effect=AssertionError("must not launch subprocess")),
        mock.patch("sparkrun.orchestration.ssh.run_remote_script", side_effect=AssertionError("must not SSH")),
    ):
        result = CliRunner().invoke(main, ["setup", "wizard", "--cluster", "lab", "--hosts", "h1", "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    assert "no changes made" in result.output
    assert before == {str(p): p.read_bytes() for p in manager.clusters_dir.rglob("*") if p.is_file()}


def test_wizard_disabled_routes_bare_setup_to_help(monkeypatch, v, tmp_path):
    from sparkrun.cli import main
    from sparkrun.core.cluster_manager import ClusterManager

    manager = ClusterManager(tmp_path)
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_WIZARD", "0")
    with mock.patch("sparkrun.cli._setup._get_cluster_manager", return_value=manager):
        result = CliRunner().invoke(main, ["setup"])
    assert result.exit_code == 0
    assert "Setup and configuration commands" in result.output
    result = CliRunner().invoke(main, ["setup", "wizard", "--yes"])
    assert result.exit_code != 0 and "disabled" in result.output


def test_disabled_prerequisite_does_not_create_dependent_gaps(monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_SETUP_STEPS_DOCKER", "0")
    state, context = state_context(CHECK_DOCKER_GROUP="0", CHECK_DOCKER_USABLE="0")
    assert not any(i.key in {"docker_group", "docker_usable"} for i in evaluate_host(state, context))


def test_sudoers_partial_changes_are_tracked_per_host():
    state, ctx = state_context(CHECK_SUDOERS_CHOWN="0", CHECK_SUDOERS_DROPCACHES="0")
    dispatch = mock.Mock(side_effect=[RemoteResult("h1", 0, "installed", ""), RemoteResult("h1", 1, "", "failed")])
    result = apply_setup_step("sudoers", state, ctx, SetupActionContext("tester", dispatch=dispatch))
    assert result.status == FAIL and result.changed
    assert result.extra["files"] == ["/etc/sudoers.d/sparkrun-chown-tester"]


def test_plugin_teardown_uses_recorded_host_details(tmp_path, v, monkeypatch):
    from sparkrun.cli import main
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.setup_manifest import ManifestManager

    config_root = tmp_path / "config"
    manager = ClusterManager(config_root)
    manager.create("lab", ["h1", "h2"], user="tester")
    manifest = ManifestManager(manager.clusters_dir)
    manifest.record_phase("lab", "tester", ["h2"], "test_action", host_details={"h2": {"created": "owned-file"}})
    undo = mock.Mock(return_value=SetupActionResult("h2", OK, "removed"))
    register_feature(FeatureFlag("setup.steps.test_action", "test", default=True))
    register_setup_step(SetupStep("test_action", "test action", feature_flag="setup.steps.test_action", undo=undo, requires_sudo=False))
    monkeypatch.setattr("sparkrun.core.config.get_config_root", lambda *args, **kwargs: config_root)
    with mock.patch("sparkrun.cli._setup._uninstall._check_running_containers", return_value=[]):
        result = CliRunner().invoke(main, ["setup", "uninstall", "lab", "--yes", "--keep-cluster", "--phase", "test_action"])
    assert result.exit_code == 0, result.output
    undo.assert_called_once()
    assert undo.call_args.args[:2] == ("h2", {"created": "owned-file"})


def test_working_docker_access_does_not_add_group_privileges():
    state, ctx = state_context(CHECK_DOCKER_GROUP="0", CHECK_DOCKER_USABLE="1")
    dispatch = mock.Mock()
    result = apply_setup_step("docker_group", state, ctx, SetupActionContext("tester", dispatch=dispatch))
    assert result.status == SKIP
    dispatch.assert_not_called()
    assert not any(i.status in {FAIL, WARN} for i in evaluate_host(state, ctx) if i.key.startswith("docker"))


def test_unavailable_plugin_teardown_preserves_manifest(tmp_path, v, monkeypatch):
    from sparkrun.cli import main
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.setup_manifest import ManifestManager

    manager = ClusterManager(tmp_path)
    manager.create("lab", ["h1"], user="tester")
    manifests = ManifestManager(manager.clusters_dir)
    manifests.record_phase("lab", "tester", ["h1"], "missing_plugin")
    monkeypatch.setattr("sparkrun.core.config.get_config_root", lambda *a, **kw: tmp_path)
    with mock.patch("sparkrun.cli._setup._uninstall._check_running_containers", return_value=[]):
        result = CliRunner().invoke(main, ["setup", "uninstall", "lab", "--yes"])
        assert result.exit_code == 0, result.output
        assert "Keeping cluster and manifest" in result.output
        result = CliRunner().invoke(main, ["setup", "uninstall", "lab", "--yes", "--phase", "missing_plugin"])
    assert result.exit_code != 0
    assert manifests.load("lab") is not None
    assert manager.get("lab").hosts == ["h1"]


def test_plugin_undo_never_guesses_changes_without_a_manifest(tmp_path, v, monkeypatch):
    from sparkrun.cli import main
    from sparkrun.core.cluster_manager import ClusterManager

    manager = ClusterManager(tmp_path)
    manager.create("lab", ["h1"], user="tester")
    register_feature(FeatureFlag("setup.steps.test_action", "test", default=True))
    undo = mock.Mock()
    register_setup_step(SetupStep("test_action", "test", feature_flag="setup.steps.test_action", undo=undo))
    monkeypatch.setattr("sparkrun.core.config.get_config_root", lambda *a, **kw: tmp_path)
    with mock.patch("sparkrun.cli._setup._uninstall._check_running_containers", return_value=[]):
        result = CliRunner().invoke(main, ["setup", "uninstall", "lab", "--yes", "--keep-cluster", "--phase", "test_action"])
    assert result.exit_code == 0, result.output
    undo.assert_not_called()


@pytest.mark.parametrize("key", ["earlyoom", "nvidia_cdi", "cx7"])
def test_plugin_constraints_exclude_only_affected_hosts_even_when_enabled(key, monkeypatch):
    from sparkrun.core.setup_steps import register_setup_constraint

    def constraint(step, state, context):
        return "Managed by hardware plugin" if step == key and state.host == "managed" else ""

    register_setup_constraint("managed-hosts", constraint)
    monkeypatch.setenv("SPARKRUN_FEATURE_SETUP_STEPS_" + key.upper(), "1")
    state, context = state_context(CHECK_EARLYOOM_ACTIVE="0", CHECK_CDI_SPEC="0")
    context.multi_host = True
    assert next(p for p in build_setup_plan(state, context) if p.step.key == key).selected
    state.host = "managed"
    context.executor_names["managed"] = "docker"
    dispatch = mock.Mock()
    entry = next(p for p in build_setup_plan(state, context) if p.step.key == key)
    assert not entry.selected and entry.reason == "Managed by hardware plugin" and not entry.checks
    assert apply_setup_step(key, state, context, SetupActionContext("tester", dispatch=dispatch)).status == SKIP
    dispatch.assert_not_called()


def test_constraints_cannot_disable_hardware_or_enable_disabled_steps(monkeypatch):
    from sparkrun.core.setup_steps import register_setup_constraint

    register_setup_constraint("exclude-all", lambda key, state, context: "unavailable")
    state, context = state_context()
    state.hardware = None
    hardware = next(p for p in build_setup_plan(state, context) if p.step.key == "hardware")
    assert hardware.selected and any(check.status == FAIL for check in hardware.checks)
    monkeypatch.setenv("SPARKRUN_FEATURE_SETUP_STEPS_EARLYOOM", "0")
    assert not next(p for p in build_setup_plan(state, context) if p.step.key == "earlyoom").selected


def test_constraint_registration_validation_and_rollback(v):
    from sparkrun.core.installed_plugins import PluginConflictError
    from sparkrun.core.registration import registry_transaction
    from sparkrun.core.setup_steps import register_setup_constraint, setup_constraint_reason, _CONSTRAINTS

    def callback(key, state, context):
        return ""

    with pytest.raises(RuntimeError), registry_transaction(v):
        register_setup_constraint("rollback", callback)
        raise RuntimeError("registration failed")
    assert "rollback" not in _CONSTRAINTS
    register_setup_constraint("example", callback)
    register_setup_constraint("example", callback)
    with pytest.raises(PluginConflictError):
        register_setup_constraint("example", lambda key, state, context: "exclude")
    with pytest.raises(ValueError):
        register_setup_constraint("invalid.name", callback)
    with pytest.raises(TypeError):
        register_setup_constraint("invalid", None)
    register_setup_constraint("invalid-result", lambda key, state, context: None)
    state, context = state_context()
    with pytest.raises(TypeError, match="return a string"):
        setup_constraint_reason("earlyoom", state, context)


def test_followup_probes_honor_per_host_constraints(v):
    from sparkrun.core.setup_steps import register_setup_constraint
    from sparkrun.core.setup_probe import probe_setup_hosts

    register_setup_constraint("managed", lambda key, state, context: "unsupported" if state.host == "managed" else "")
    stdout = "SPARKRUN_PROBE_ACCEL_END\n" + "\n".join(key + "=" + value for key, value in FACTS.items())
    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_script", side_effect=lambda host, *a, **kw: RemoteResult(host, 0, stdout, "")),
        mock.patch("sparkrun.core.hardware_probe._parse_probe_result", return_value=default_dgx_spark_hardware()),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={}) as cx7,
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value={}) as rdma,
    ):
        probe_setup_hosts(["managed", "ordinary"], ssh_kwargs={}, config=SparkrunConfig())
    cx7.assert_called_once_with(["ordinary"], ssh_kwargs={})
    rdma.assert_called_once_with(["ordinary"], {}, dry_run=False)


def test_extra_mesh_peers_are_probed_and_kept_on_context(v):
    """A control machine joining the mesh is dialed by every host's mesh probe."""
    from sparkrun.core.setup_probe import probe_setup_hosts

    scripts = {}
    stdout = "SPARKRUN_PROBE_ACCEL_END\n" + "\n".join(key + "=" + value for key, value in FACTS.items())

    def run(host, script, *a, **kw):
        scripts.setdefault(host, []).append(script)
        return RemoteResult(host, 0, stdout, "")

    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_script", side_effect=run),
        mock.patch("sparkrun.core.hardware_probe._parse_probe_result", return_value=default_dgx_spark_hardware()),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={}),
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value={}),
    ):
        _states, context = probe_setup_hosts(["h1", "h2"], ssh_kwargs={}, config=SparkrunConfig(), extra_mesh_peers=["ctl", "h2", "ctl"])
    assert context.extra_mesh_peers == ("ctl",)
    readiness = [s for s in scripts["h1"] if "PEERS=" in s]
    assert readiness and "PEERS='h2 ctl'" in readiness[0]


def test_mesh_also_dials_peers_cx7_addresses_on_shared_subnets(v):
    """Transfers go to CX7 addresses, so those are mesh legs too — but only reachable ones."""
    from sparkrun.core.setup_probe import probe_setup_hosts
    from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface

    def cx7(host, subnets):
        last = host[-1]
        ifaces = [
            CX7Interface(
                name="if%d" % n, ip="192.168.%d.%s" % (n, last), prefix=24, subnet="192.168.%d.0/24" % n, mtu=9000, state="up", hca=""
            )
            for n in subnets
        ]
        return CX7HostDetection(host=host, interfaces=ifaces, detected=True)

    readiness = "SPARKRUN_PROBE_ACCEL_END\n" + "\n".join(key + "=" + value for key, value in FACTS.items())
    followups = {}

    def run(host, script, *a, **kw):
        if "MESH_HOSTKEY" in script and "SETUP_STEPS" not in script:
            followups[host] = next(line for line in script.splitlines() if line.startswith("PEERS="))
            return RemoteResult(host, 0, "CHECK_MESH_TOTAL=1\nCHECK_MESH_OK=0\nCHECK_MESH_HOSTKEY=1\n", "")
        return RemoteResult(host, 0, readiness, "")

    detections = {"h1": cx7("h1", [10, 11]), "h2": cx7("h2", [10]), "h3": cx7("h3", [12])}
    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_script", side_effect=run),
        mock.patch("sparkrun.core.hardware_probe._parse_probe_result", return_value=default_dgx_spark_hardware()),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=detections),
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value={}),
    ):
        states, _context = probe_setup_hosts(["h1", "h2", "h3"], ssh_kwargs={}, config=SparkrunConfig())
    # h1 and h2 share 192.168.10.0/24; h3 shares nothing, so it dials nobody.
    assert followups == {"h1": "PEERS=192.168.10.2", "h2": "PEERS=192.168.10.1"}
    assert states["h1"].facts["CHECK_MESH_FABRIC_HOSTKEY"] == "1"
    assert "CHECK_MESH_FABRIC_TOTAL" not in states["h3"].facts


def test_fabric_mesh_legs_follow_each_hosts_plan(v):
    """Per-host exclusions hold for the CX7 follow-up: no cx7 → no addresses, no ssh_mesh → no legs."""
    from sparkrun.core.setup_probe import probe_setup_hosts
    from sparkrun.core.setup_steps import register_setup_constraint
    from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface

    excluded = {("cx7", "h2"), ("ssh_mesh", "h3")}
    register_setup_constraint("partial", lambda key, state, context: "excluded" if (key, state.host) in excluded else "")

    def cx7(host):
        iface = CX7Interface(name="if0", ip="192.168.10.%s" % host[-1], prefix=24, subnet="192.168.10.0/24", mtu=9000, state="up", hca="")
        return CX7HostDetection(host=host, interfaces=[iface], detected=True)

    readiness = "SPARKRUN_PROBE_ACCEL_END\n" + "\n".join(key + "=" + value for key, value in FACTS.items())
    followups = {}

    def run(host, script, *a, **kw):
        if "MESH_HOSTKEY" in script and "SETUP_STEPS" not in script:
            followups[host] = next(line for line in script.splitlines() if line.startswith("PEERS="))
        return RemoteResult(host, 0, readiness, "")

    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_script", side_effect=run),
        mock.patch("sparkrun.core.hardware_probe._parse_probe_result", return_value=default_dgx_spark_hardware()),
        # Detection only ever sees hosts whose plan selects cx7.
        mock.patch(
            "sparkrun.orchestration.networking.detect_cx7_for_hosts", side_effect=lambda targets, **kw: {h: cx7(h) for h in targets}
        ),
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value={}),
    ):
        probe_setup_hosts(["h1", "h2", "h3", "h4"], ssh_kwargs={}, config=SparkrunConfig())
    # h2 has no cx7 (no address to dial), h3 is outside the mesh (neither dials nor is dialed).
    assert followups == {"h1": "PEERS=192.168.10.4", "h4": "PEERS=192.168.10.1"}


def test_failed_fabric_probe_reads_as_unknown_never_as_reachable(v):
    """A follow-up that times out must not let ssh_mesh pass on management names alone."""
    from sparkrun.core.setup_probe import _probe_fabric_mesh
    from sparkrun.core.setup_steps import evaluate_host
    from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface

    def cx7(host):
        iface = CX7Interface(name="if0", ip="192.168.10.%s" % host[-1], prefix=24, subnet="192.168.10.0/24", mtu=9000, state="up", hca="")
        return CX7HostDetection(host=host, interfaces=[iface], detected=True)

    facts = dict(FACTS, CHECK_MESH_TOTAL="1", CHECK_MESH_OK="1")
    states = {h: HostState(h, facts=dict(facts), hardware=default_dgx_spark_hardware(), cx7=cx7(h)) for h in ("h1", "h2")}
    timeouts = []

    def timed_out(host, script, *a, timeout=None, **kw):
        timeouts.append(timeout)
        return RemoteResult(host, 124, "", "Command timed out")

    with mock.patch("sparkrun.orchestration.ssh.run_remote_script", side_effect=timed_out):
        _probe_fabric_mesh(states, ["h1", "h2"], {})
    assert states["h1"].facts["CHECK_MESH_FABRIC_UNKNOWN"] == "1"
    assert timeouts == [21, 21]  # scales with the legs dialed (one each here)
    ctx = CheckContext("lab", True, config=SparkrunConfig(), executor_names={"h1": "docker"}, strict=True)
    item = next(i for i in evaluate_host(states["h1"], ctx) if i.key == "ssh_mesh")
    assert item.status == WARN and "1 CX7 address(es) could not be checked" in item.detail
