from __future__ import annotations

from types import SimpleNamespace

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.recipe import Recipe
from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime


_DIGEST = "a" * 64


def _fixture():
    recipe = Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "Qwen/Qwen3.5-0.8B",
            "model_revision": "model-commit",
            "runtime": "vllm-distributed",
            "container": "org/capsule@sha256:%s" % _DIGEST,
            "defaults": {"tensor_parallel": 2, "port": 8000},
            "env": {"RECIPE_FLAG": "yes"},
        }
    )
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"], cache_dir="/cache/hf")
    placement = RankAssignment(
        by_rank=(RankSlot("h1", 0), RankSlot("h2", 0)),
        hosts_used=("h1", "h2"),
    )
    plan = api.RunPlan(
        recipe=recipe,
        runtime=VllmDistributedRuntime(),
        cluster=cluster,
        candidate_hosts=("h1", "h2"),
        host_list=("h1", "h2"),
        is_solo=False,
        placement=placement,
        cluster_id="sparkrun_test_test",
    )
    options = api.RunOptions(recipe=recipe, hosts=("h1", "h2"), init_port=29731)
    sctx = SimpleNamespace(config=SimpleNamespace(hf_cache_dir="/fallback"))
    return options, plan, sctx


def test_materialize_renders_pinned_per_unit_vllm_commands():
    options, plan, sctx = _fixture()
    spec = api.materialize(options, plan=plan, sctx=sctx)

    assert spec.engine == "vllm"
    assert spec.world_size == 2
    assert spec.tensor_parallel == 2
    assert spec.model_revision == "model-commit"
    assert [unit.host for unit in spec.units] == ["h1", "h2"]
    assert all(unit.image_digest == "sha256:%s" % _DIGEST for unit in spec.units)
    assert [worker.id for worker in spec.execution.workers] == ["worker-0", "worker-1"]
    assert [worker.unit for worker in spec.execution.workers] == ["unit-0", "unit-1"]
    assert spec.units[1].command[:4] == (
        "bash",
        "--noprofile",
        "--norc",
        "-c",
    )
    assert "--node-rank" in spec.units[1].command[4]
    assert "--headless" in spec.units[1].command[4]
    assert spec.units[0].environment["RECIPE_FLAG"] == "yes"
    hf_mount = next(mount for mount in spec.units[0].mounts if mount.target == "/cache/huggingface")
    assert hf_mount.read_only is True


def test_materialize_leaves_mutable_image_unpinned():
    options, plan, sctx = _fixture()
    plan.recipe.container = "org/capsule:latest"
    spec = api.materialize(options, plan=plan, sctx=sctx)
    assert all(unit.image_digest == "" for unit in spec.units)


def test_materialize_mounts_prepared_local_model_read_only():
    options, plan, sctx = _fixture()
    plan.recipe.model = "/models/qwen"

    spec = api.materialize(options, plan=plan, sctx=sctx)

    model_mount = next(mount for mount in spec.units[0].mounts if mount.target == "/models/qwen")
    assert model_mount.source == "/models/qwen"
    assert model_mount.read_only is True


def test_materialize_accepts_prepared_per_node_image_identities():
    options, plan, sctx = _fixture()
    images = ("sha256:" + "b" * 64, "sha256:" + "c" * 64)

    spec = api.materialize(options, plan=plan, sctx=sctx, images_by_node=images)

    assert tuple(unit.image for unit in spec.units) == images
    assert tuple(unit.image_digest for unit in spec.units) == images


def test_materialize_applies_sparkrun_owned_per_host_comm_env():
    options, plan, sctx = _fixture()
    comm_env = ClusterCommEnv.from_per_host(
        {
            "h1": {
                "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
                "NCCL_IB_HCA": "rocep1s0f0",
                "NODE_IP": "10.0.0.1",
            },
            "h2": {
                "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
                "NCCL_IB_HCA": "rocep1s0f1",
                "NODE_IP": "10.0.0.2",
            },
        }
    )

    spec = api.materialize(options, plan=plan, comm_env=comm_env, sctx=sctx)

    assert spec.units[0].environment["GLOO_SOCKET_IFNAME"] == "enp1s0f0np0"
    assert spec.units[1].environment["GLOO_SOCKET_IFNAME"] == "enp1s0f1np1"
    assert spec.units[0].environment["NCCL_IB_HCA"] == "rocep1s0f0"
    assert spec.units[1].environment["NCCL_IB_HCA"] == "rocep1s0f1"
    assert spec.units[0].environment["VLLM_HOST_IP"] == "10.0.0.1"
    assert spec.units[1].environment["VLLM_HOST_IP"] == "10.0.0.2"


def test_materialize_includes_recipe_executor_volumes():
    options, plan, sctx = _fixture()
    plan.recipe.executor_config["volumes"] = [
        "/opt/shared",
        "/host/runtime.py:/usr/local/bin/runtime.py:ro",
    ]

    spec = api.materialize(options, plan=plan, sctx=sctx)

    mounts = {mount.target: mount for mount in spec.units[0].mounts}
    assert mounts["/opt/shared"] == api.ResolvedMount(
        source="/opt/shared",
        target="/opt/shared",
        read_only=False,
    )
    assert mounts["/usr/local/bin/runtime.py"] == api.ResolvedMount(
        source="/host/runtime.py",
        target="/usr/local/bin/runtime.py",
        read_only=True,
    )
    assert spec.units[1].mounts == spec.units[0].mounts


def test_materialize_groups_four_workers_into_two_multi_gpu_units():
    options, plan, sctx = _fixture()
    plan.recipe.defaults["tensor_parallel"] = 4
    placement = RankAssignment(
        by_rank=(RankSlot("h1", 0), RankSlot("h1", 1), RankSlot("h2", 0), RankSlot("h2", 1)),
        hosts_used=("h1", "h2"),
    )
    plan = api.RunPlan(
        **{**plan.__dict__, "placement": placement},
    )

    spec = api.materialize(options, plan=plan, sctx=sctx)

    assert [unit.devices for unit in spec.units] == [("0", "1"), ("0", "1")]
    assert [(worker.unit, worker.process_slot, worker.device_slots) for worker in spec.execution.workers] == [
        ("unit-0", 0, (0,)),
        ("unit-0", 1, (1,)),
        ("unit-1", 0, (0,)),
        ("unit-1", 1, (1,)),
    ]
    assert "--nnodes 2" in spec.units[0].command[4]
    assert "--node-rank 1" in spec.units[1].command[4]
    assert "--headless" in spec.units[1].command[4]


def test_materialize_rejects_uneven_vllm_workers_across_launch_units():
    options, plan, sctx = _fixture()
    plan.recipe.defaults["tensor_parallel"] = 3
    placement = RankAssignment(
        by_rank=(RankSlot("h1", 0), RankSlot("h1", 1), RankSlot("h2", 0)),
        hosts_used=("h1", "h2"),
    )
    plan = api.RunPlan(
        **{**plan.__dict__, "placement": placement},
    )

    with pytest.raises(ValueError, match="same number of local workers"):
        api.materialize(options, plan=plan, sctx=sctx)


def test_materialize_keeps_parallel_axes_opaque_in_adapter_identity():
    options, plan, sctx = _fixture()
    plan.recipe.defaults.update(
        {"tensor_parallel": 2, "pipeline_parallel": 1, "data_parallel": 1, "expert_parallel": 2, "context_parallel": 2}
    )

    spec = api.materialize(options, plan=plan, sctx=sctx)

    assert spec.execution.adapter.schema == "vllm:sparkrun-v1"
    assert spec.execution.adapter.payload["dimensions"] == {
        "tensor": 2,
        "pipeline": 1,
        "data": 1,
        "expert": 2,
        "context": 2,
    }
    assert spec.execution.groups[0].kind == "vllm:world"


def test_materialize_single_host_tp8_as_one_unit_with_eight_workers():
    options, plan, sctx = _fixture()
    plan.recipe.defaults["tensor_parallel"] = 8
    cluster = ClusterDefinition(
        name="c",
        hosts=["h1"],
        cache_dir="/cache/hf",
        hosts_hardware={"h1": HostHardware(accelerators=[AcceleratorSpec("nvidia", "h200", count=8, capabilities=frozenset({"cuda"}))])},
    )
    plan = api.RunPlan(
        **{
            **plan.__dict__,
            "cluster": cluster,
            "candidate_hosts": ("h1",),
            "host_list": ("h1",),
            "is_solo": True,
            "placement": None,
        }
    )

    spec = api.materialize(options, plan=plan, sctx=sctx)

    assert len(spec.units) == 1
    assert spec.units[0].devices == tuple(str(index) for index in range(8))
    assert len(spec.execution.workers) == 8
    assert all(worker.unit == "unit-0" for worker in spec.execution.workers)
    assert "--nnodes" not in spec.units[0].command[4]
