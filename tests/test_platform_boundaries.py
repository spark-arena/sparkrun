"""Platform boundary regressions with measured edge-device inventories.

Reference: jetsonrun 6110191, jetson-plugin/test_jetson_hardware.py. Orin boards
span 3.6–61.3 GiB, Thor reports 125 GiB, and the plugin reserves 20% of shared
RAM. These are test data and a minimal adapter, not shipped Jetson support.
"""

from dataclasses import replace
from pathlib import Path
import ast

import pytest

from sparkrun.core.hardware import AcceleratorSpec, HostHardware, resolve_hardware, resolve_host_hardware
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.core.limits import resolved_hardware_for_scheduling, resolve_accelerator_memory_gb
from sparkrun.core.recipe import Recipe
from sparkrun.models.vram import estimate_vram
from sparkrun.models.fit import check_fit
from sparkrun.platforms.base import HardwarePlatformPlugin
from sparkrun.orchestration.collectives import NcclBackend


class ReferencePlatform(HardwarePlatformPlugin):
    """Test adapter exercising existing extension hooks with edge-device facts."""

    platform_name = "reference-edge"
    display_name = "Reference shared-memory edge device"
    vendors = frozenset({"nvidia"})

    def matches(self, hw):
        return any(a.model in {"jetson-orin", "jetson-thor"} for a in hw.accelerators)

    def accelerator_vendor(self):
        return "nvidia"

    def collective_backend(self):
        return NcclBackend()

    def default_max_gpu_memory_utilization(self, accelerator):
        return 0.8

    def default_env(self, runtime_name, accelerator, *, runtime_family=None):
        return {"EDGE_PLATFORM": accelerator.model}

    def default_executor_config(self, executor_name):
        return {"gpu_access_mode": "gpus"} if executor_name == "docker" else {}


@pytest.fixture
def edge_platform(monkeypatch):
    from sparkrun import platforms

    platform = ReferencePlatform()
    monkeypatch.setattr(platforms, "_REGISTRY", [platform, *platforms.iter_platforms()])
    return platform


def hardware(model, capacity=None):
    return HostHardware(
        [AcceleratorSpec("nvidia", model, memory_gb=capacity, capabilities=frozenset({"cuda", "unified-memory"}))], source="detected"
    )


def placed(hw, index=0):
    cluster = ClusterDefinition("test", ["edge"], hosts_hardware={"edge": hw})
    return cluster, RankAssignment((RankSlot("edge", index),), ("edge",))


def test_metadata_required_profile_does_not_affect_targetless_estimation(monkeypatch):
    from sparkrun.core import application_profile

    monkeypatch.setattr(application_profile, "_active", replace(application_profile.SPARKRUN, hardware_fallback="require-metadata"))
    with pytest.raises(ValueError, match="metadata is required"):
        resolve_hardware()
    est = estimate_vram(model_vram=3, gpu_memory_utilization=0.8)
    assert est.total_gpu_memory_gb is None
    assert est.available_kv_gb is None
    assert est.max_context_tokens is None
    assert est.total_per_gpu_gb == 3
    assert "fits_dgx_spark" not in est.to_dict()
    explicit = estimate_vram(model_vram=3, kv_cache_memory_bytes=1024**3, kv_vram_per_token=1 / 1024**2, max_model_len=4096)
    assert explicit.total_gpu_memory_gb is None
    assert explicit.available_kv_gb == 1
    assert explicit.max_context_tokens == 1024**2


@pytest.mark.parametrize("capacity", [0, -1, float("nan"), float("inf"), True])
def test_invalid_target_is_not_replaced_with_platform_default(capacity):
    with pytest.raises(ValueError, match="finite and positive"):
        estimate_vram(model_vram=3, total_gpu_memory_gb=capacity)


@pytest.mark.parametrize(
    "model,capacity", [("jetson-orin", 3.6), ("jetson-orin", 7.5), ("jetson-orin", 15.3), ("jetson-orin", 61.3), ("jetson-thor", 125.0)]
)
def test_measured_shared_memory_and_explicit_kv_budget(edge_platform, model, capacity):
    hw = hardware(model, capacity)
    cluster, placement = placed(hw)
    original = hw.to_dict()
    planned = resolved_hardware_for_scheduling(cluster, cluster.hosts)["edge"]
    assert planned.accelerators[0].memory_gb == capacity
    assert planned.accelerators[0].max_gpu_memory_utilization == 0.8
    assert planned.accelerators[0].memory_capacity_source == "detected"
    est = estimate_vram(model_vram=5, kv_cache_memory_bytes=2 * 1024**3, gpu_memory_utilization=0.1)
    fit = check_fit(est, cluster, placement)
    detail = fit.per_host["edge"]
    assert detail.accelerator_memory_gb == pytest.approx(capacity * 0.8)
    assert detail.status == ("fits" if capacity * 0.8 >= 7 else "exceeds")
    assert fit.to_dict()["status"] == detail.status
    assert detail.memory_capacity_source == "detected"
    assert hw.to_dict() == original


def test_unknown_edge_capacity_remains_unknown(edge_platform, capsys):
    from sparkrun.utils.cli_formatters import display_memory_plan

    hw = hardware("jetson-orin")
    cluster, placement = placed(hw)
    assert resolve_accelerator_memory_gb(hw.accelerators[0], hw) is None
    planned = resolved_hardware_for_scheduling(cluster, cluster.hosts)["edge"]
    assert planned.accelerators[0].memory_gb is None
    assert planned.accelerators[0].max_gpu_memory_utilization == 0.8
    est = estimate_vram(model_vram=3, kv_cache_memory_bytes=1024**3)
    fit = check_fit(est, cluster, placement)
    assert fit.ok  # Whole-device admission remains independent of verified fit.
    assert fit.status == "unknown"
    recipe = Recipe.from_dict(
        {"model": "test", "runtime": "vllm-distributed", "metadata": {"model_vram": 3}, "defaults": {"gpu_memory_utilization": 0.8}}
    )
    display_memory_plan(recipe, auto_detect=False, cluster=cluster, placement=placement)
    output = capsys.readouterr().out
    assert "UNVERIFIED    accelerator capacity unknown" in output
    assert "121" not in output
    assert "DGX Spark" not in output


def test_mixed_device_host_resolves_each_assigned_device(edge_platform):
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
    from sparkrun.core.launcher import resolve_platform_env_defaults

    hw = hardware("jetson-orin", 61.3)
    hw.accelerators.insert(0, AcceleratorSpec("nvidia", "rtx-3090", memory_gb=24))
    cluster, placement = placed(hw, 1)
    original = hw.to_dict()
    planned = resolved_hardware_for_scheduling(cluster, cluster.hosts)["edge"]
    assert [a.max_gpu_memory_utilization for a in planned.accelerators] == [1.0, 0.8]
    target = resolve_host_hardware(cluster.hosts, cluster, placement)["edge"]
    assert [a.model for a in target.accelerators] == ["jetson-orin"]
    assert resolve_platform_env_defaults(VllmDistributedRuntime(), target) == {"EDGE_PLATFORM": "jetson-orin"}
    assert VllmDistributedRuntime().default_image_for(target) is None
    discrete = hw.selected([0])
    assert VllmDistributedRuntime().default_image_for(discrete) == "vllm/vllm-openai:latest"
    assert hw.to_dict() == original


def test_edge_image_requires_explicit_target_qualified_image(edge_platform):
    from sparkrun.core.images import resolve_runtime_image_plan, ImagePlanError
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    hw = hardware("jetson-orin", 61.3)
    cluster, _ = placed(hw)
    recipe = Recipe.from_dict({"model": "test", "runtime": "vllm-distributed"})
    with pytest.raises(ImagePlanError):
        resolve_runtime_image_plan(recipe, VllmDistributedRuntime(), cluster.hosts, cluster=cluster)
    recipe.container = "example/jetpack-qualified:v1"
    assert resolve_runtime_image_plan(recipe, VllmDistributedRuntime(), cluster.hosts, cluster=cluster).head_image() == recipe.container


def test_platform_collective_hook_is_authoritative(edge_platform, monkeypatch):
    from sparkrun.core.backend_select import select_backends

    class QualifiedBackend(NcclBackend):
        name = "qualified-edge"

    monkeypatch.setattr(edge_platform, "collective_backend", QualifiedBackend)
    assert isinstance(select_backends(hardware("jetson-thor", 125)).collective, QualifiedBackend)


def test_collective_selection_ignores_unassigned_vendor(edge_platform):
    from sparkrun.core.launcher import resolve_per_host_backends

    hw = hardware("jetson-orin", 61.3)
    hw.accelerators.insert(0, AcceleratorSpec("amd", "mi300x", memory_gb=192))
    cluster, placement = placed(hw, 1)
    bundles = resolve_per_host_backends(cluster.hosts, cluster, placement=placement)
    assert bundles["edge"].accelerator_vendor == "nvidia"


def test_unknown_vendor_never_silently_gets_nccl():
    from sparkrun.core.launcher import resolve_per_host_backends
    from sparkrun.core.backend_select import NoMatchingBackendError

    cluster, _ = placed(HostHardware([AcceleratorSpec("apple", "m5", memory_gb=24)]))
    with pytest.raises(NoMatchingBackendError):
        resolve_per_host_backends(cluster.hosts, cluster)
    single = resolve_per_host_backends(cluster.hosts, cluster, require_collectives=False)
    assert single["edge"].collective.name == "none"
    assert single["edge"].collective.env_for_host({}) == {}


def test_conflicting_device_defaults_require_explicit_setting(edge_platform, monkeypatch):
    from sparkrun.platforms import accelerator_defaults

    hw = hardware("jetson-orin", 61.3)
    hw.accelerators += hardware("jetson-thor", 125).accelerators

    def getter(p, a):
        return {"max_model_len": 8192 if a.model == "jetson-orin" else 32768}

    with pytest.raises(ValueError, match="Conflicting platform default"):
        accelerator_defaults(hw, getter)
    assert accelerator_defaults(hw, getter, overrides={"max_model_len": 4096}) == {}


def test_selected_executor_policy_uses_platform_and_preserves_overrides(edge_platform):
    from sparkrun.orchestration.executor import resolve_executor

    hw = hardware("jetson-orin", 61.3)
    recipe = Recipe.from_dict({"model": "test", "runtime": "vllm-distributed"})
    edge = resolve_executor(recipe=recipe, host_hardware=hw)
    discrete = resolve_executor(recipe=recipe, host_hardware=HostHardware([AcceleratorSpec("nvidia", "h100", memory_gb=80)]))
    assert edge.config.gpu_access_mode == "gpus"
    assert discrete.config.gpu_access_mode == "cdi"
    assert edge.config.accelerator_vendor == "nvidia"
    override = resolve_executor(recipe=recipe, host_hardware=hw, cli_overrides={"gpu_access_mode": "cdi"})
    assert override.config.gpu_access_mode == "cdi"
    edge.bind_host_executors({"edge": edge, "discrete": discrete})
    assert edge.for_host("discrete") is discrete
    assert edge.for_host("edge") is edge


def test_generic_decision_modules_do_not_import_platform_constants():
    root = Path(__file__).parents[1] / "src" / "sparkrun"
    for folder in ("api", "cli", "models", "schedulers", "utils"):
        for source in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("sparkrun.platforms.dgx_spark"), source
                    assert all(not alias.name.startswith("DGX_SPARK_") and alias.name != "DEFAULT_VRAM_GB" for alias in node.names), source


def test_materialize_uses_assigned_device_env_mounts_and_common_flags(edge_platform, monkeypatch):
    import sparkrun.api as api
    from test_api_materialize import _fixture

    options, plan, sctx = _fixture()
    edge = hardware("jetson-orin", 61.3)
    edge.accelerators.insert(0, AcceleratorSpec("nvidia", "rtx-3090", memory_gb=24))
    plan.cluster.hosts_hardware = {"h1": edge, "h2": hardware("h100", 80)}
    plan = replace(plan, placement=RankAssignment((RankSlot("h1", 1), RankSlot("h2", 0)), ("h1", "h2")))
    monkeypatch.setattr(edge_platform, "default_executor_config", lambda name: {"volumes": ["/edge-assets:/assets:ro"]})
    monkeypatch.setattr(edge_platform, "default_runtime_flags", lambda name, accel: {"max_num_batched_tokens": 512})
    spec = api.materialize(options, plan=plan, sctx=sctx)
    edge_unit, discrete_unit = spec.units
    assert edge_unit.devices == ("1",)
    assert edge_unit.environment["EDGE_PLATFORM"] == "jetson-orin"
    assert "EDGE_PLATFORM" not in discrete_unit.environment
    assert any(m.source == "/edge-assets" for m in edge_unit.mounts)
    assert all(m.source != "/edge-assets" for m in discrete_unit.mounts)
    assert all("--max-num-batched-tokens 512" in unit.command[4] for unit in spec.units)
    assert "max_num_batched_tokens" not in plan.recipe.defaults
    plan.recipe.env["EDGE_PLATFORM"] = "user-choice"
    assert all(unit.environment["EDGE_PLATFORM"] == "user-choice" for unit in api.materialize(options, plan=plan, sctx=sctx).units)


@pytest.mark.parametrize("explicit", [False, True])
def test_cluster_submission_uses_host_executor_and_env_precedence(monkeypatch, explicit):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
    from sparkrun.runtimes._cluster_ops import ClusterContext, launch_containers_parallel
    from sparkrun.orchestration.executors.docker import DockerExecutor

    runtime = VllmDistributedRuntime()
    runtime.platform_env_by_host = {"edge": {"TUNING": "edge"}, "server": {"TUNING": "server"}}
    monkeypatch.setattr(runtime, "get_common_env", lambda: {"TUNING": "runtime"})
    ctx = ClusterContext.build(
        runtime, ["edge", "server"], "image", "sparkrun_fixture", {"TUNING": "user"} if explicit else {}, None, None, True
    )
    executors = {h: DockerExecutor() for h in ctx.hosts}
    root = executors["edge"]
    root.bind_host_executors(executors)
    captured = {}
    for host, executor in executors.items():

        def generate(*, env, _host=host, **kwargs):
            captured[_host] = env
            return "true"

        monkeypatch.setattr(executor, "generate_launch_script", generate)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", Mock(return_value=SimpleNamespace(success=True)))
    assert launch_containers_parallel(ctx, [(h, "sparkrun_fixture_" + h) for h in ctx.hosts], root, None) == 0
    assert {h: e["TUNING"] for h, e in captured.items()} == (
        {"edge": "user", "server": "user"} if explicit else {"edge": "edge", "server": "server"}
    )


@pytest.mark.parametrize(
    "vendor,variable",
    [
        ("nvidia", "CUDA_VISIBLE_DEVICES"),
        ("amd", "ROCR_VISIBLE_DEVICES"),
        ("intel", "HABANA_VISIBLE_MODULES"),
        ("cpu", None),
        ("apple", None),
    ],
)
def test_native_device_visibility_follows_resolved_vendor(vendor, variable):
    from sparkrun.core.allocations import ALLOCATION_LABEL, GpuAllocation, encode_allocations
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    executor = LocalExecutor(
        ExecutorConfig(executor_type="local", accelerator_vendor=vendor, gpus="device=1", pid_dir="/tmp", log_dir="/tmp")
    )
    script = executor.run_cmd(
        image="",
        command="true",
        container_name="sparkrun_visibility_solo",
        sparkrun_labels={ALLOCATION_LABEL: encode_allocations((GpuAllocation(0, 1),))},
    )
    if variable is not None:
        assert "export %s=1" % variable in script
    if vendor != "nvidia":
        assert "export CUDA_VISIBLE_DEVICES=" not in script


def test_assumed_policy_is_marked_and_does_not_invent_network():
    from sparkrun.core.hardware import default_dgx_spark_hardware

    assumed = default_dgx_spark_hardware()
    assert assumed.source == "assumed"
    assert assumed.fingerprint is None
    assert not any(c.startswith("rdma:") for a in assumed.accelerators for c in a.capabilities)
    cluster, placement = placed(assumed)
    estimate = estimate_vram(model_vram=3, kv_cache_memory_bytes=1024**3)
    fit = check_fit(estimate, cluster, placement)
    assert fit.status == "unknown"
    assert fit.per_host["edge"].memory_capacity_source == "assumed platform default"


def test_application_fallback_accepts_registered_platform_policy(edge_platform, monkeypatch):
    from sparkrun.core import application_profile

    # No edge capacity is guessed by the platform's default implementation.
    monkeypatch.setattr(
        application_profile, "_active", replace(application_profile.SPARKRUN, hardware_fallback=edge_platform.platform_name)
    )
    with pytest.raises(ValueError, match="metadata is required"):
        resolve_hardware()
    # An application may deliberately supply its own qualified compatibility policy.
    monkeypatch.setattr(edge_platform, "assumed_hardware", lambda: hardware("jetson-orin", 61.3))
    assert resolve_hardware().source == "assumed"


def test_direct_cluster_runtime_uses_platform_provider_when_map_is_omitted(edge_platform, monkeypatch):
    from types import SimpleNamespace
    from sparkrun.runtimes._cluster_ops import resolve_comm_env
    from sparkrun.orchestration.comm_env import ClusterCommEnv

    class Provider(NcclBackend):
        def env_for_host(self, ib_info, *, topology=None):
            return {"QUALIFIED_PROVIDER": "edge"}

    monkeypatch.setattr(edge_platform, "collective_backend", Provider)
    ctx = SimpleNamespace(
        hosts=["edge"],
        hardware_for=lambda h: hardware("jetson-orin", 61.3),
        ssh_kwargs={},
        dry_run=True,
        topology=None,
        mgmt_interface=None,
    )

    def detect(hosts, *, backends, **kwargs):
        return SimpleNamespace(comm_env=ClusterCommEnv.from_per_host({h: backends[h].collective.env_for_host({}) for h in hosts}))

    monkeypatch.setattr("sparkrun.orchestration.infiniband.detect_ib_for_hosts", detect)
    assert resolve_comm_env(ctx, None).get_env("edge")["QUALIFIED_PROVIDER"] == "edge"


def test_broken_platform_match_cannot_fall_back_to_generic_nvidia(edge_platform, monkeypatch):
    def broken(hw):
        raise RuntimeError("platform detection failed")

    monkeypatch.setattr(edge_platform, "matches", broken)
    cluster, _ = placed(hardware("jetson-orin", 61.3))
    with pytest.raises(RuntimeError, match="platform detection failed"):
        resolved_hardware_for_scheduling(cluster, cluster.hosts)


@pytest.mark.parametrize("cap", [0, -1, True, float("nan"), float("inf")])
def test_invalid_platform_cap_cannot_expand_the_budget(edge_platform, monkeypatch, cap):
    monkeypatch.setattr(edge_platform, "default_max_gpu_memory_utilization", lambda a: cap)
    cluster, _ = placed(hardware("jetson-orin", 61.3))
    with pytest.raises(ValueError, match="invalid scheduling memory fraction"):
        resolved_hardware_for_scheduling(cluster, cluster.hosts)


def test_broken_platform_cap_cannot_expand_the_budget(edge_platform, monkeypatch):
    def broken(accel):
        raise RuntimeError("platform memory policy failed")

    monkeypatch.setattr(edge_platform, "default_max_gpu_memory_utilization", broken)
    cluster, _ = placed(hardware("jetson-orin", 61.3))
    with pytest.raises(RuntimeError, match="platform memory policy failed"):
        resolved_hardware_for_scheduling(cluster, cluster.hosts)
