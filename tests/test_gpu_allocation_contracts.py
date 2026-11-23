"""Admission, worker recovery and cross-executor ownership of actual GPUs."""

from dataclasses import replace

import json
import subprocess

import pytest
from sparkrun.core.allocations import ALLOCATION_LABEL, GpuAllocation, encode_allocations, decode_allocations
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, ContainerDetail, RunningWorkload, with_gpu_allocations
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import ResourceRequest, SchedulingRequest, RankAssignment, RankSlot, InfeasibleScheduleError
from sparkrun.schedulers.sparse_pack import SparsePackScheduler
from sparkrun.schedulers.dense_pack import DensePackScheduler
from sparkrun.orchestration.executors._base import Executor, ExecutorConfig
from sparkrun.orchestration.executors.docker import DockerExecutor, _parse_docker_ps_output
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.ssh import RemoteResult

CID = "sparkrun_" + "a" * 16 + "_" + "b" * 12


def hardware(count=1):
    return HostHardware(accelerators=[AcceleratorSpec("nvidia", "gb10", count=count, memory_gb=121, max_gpu_memory_utilization=0.9)])


def request(fraction=1.0, memory=None, status=None, count=1):
    return SchedulingRequest(
        ParallelismConfig(),
        ("h",),
        host_hardware={"h": hardware(count)},
        status=status,
        resources=ResourceRequest(util_fraction=fraction, memory_gb=memory),
    )


def observed(allocation, *, cid=CID, executor="docker", capacity=1):
    detail = ContainerDetail(cid + "_solo", "solo", "Up", "image", allocations=(allocation,))
    work = RunningWorkload(cid, containers=(detail,))
    occ = with_gpu_allocations(HostOccupancy("h", (work,), used_slots=1, free_slots=capacity - 1), capacity)
    return ClusterStatus((occ,), executor=executor)


@pytest.mark.parametrize("factory", [SparsePackScheduler, DensePackScheduler])
@pytest.mark.parametrize("fraction,expected", [(0.799, 0.799), (0.8, 1.0), (0.95, 1.0), (1.0, 1.0)])
def test_threshold_promotes_to_exclusive(factory, fraction, expected):
    result = factory().schedule(request(fraction))
    slot = result.assignment.by_rank[0]
    assert slot.util_fraction == expected
    assert slot.memory_gb == (pytest.approx(121 * fraction) if expected < 1 else None)


@pytest.mark.parametrize("factory", [SparsePackScheduler, DensePackScheduler])
def test_exclusive_allocation_ignores_estimated_footprint(factory):
    assert factory().schedule(request(memory=1000)).assignment.by_rank[0].util_fraction == 1


@pytest.mark.parametrize("existing,incoming", [(0.8, 0.1), (1.0, 0.1), (0.1, 0.8)])
@pytest.mark.parametrize("factory", [SparsePackScheduler, DensePackScheduler])
def test_exclusive_owner_excludes_both_arrival_orders(factory, existing, incoming):
    status = observed(GpuAllocation(0, 0, existing, 10 if existing < 0.8 else None))
    with pytest.raises(InfeasibleScheduleError) as error:
        factory().schedule(request(incoming, memory=1, status=status))
    assert error.value.rejections[0].reason in ("exclusive_owner", "occupied")


def test_shared_memory_budget_is_mandatory_and_explained():
    status = observed(GpuAllocation(0, 0, 0.4, 80))
    with pytest.raises(InfeasibleScheduleError) as error:
        SparsePackScheduler().schedule(request(0.1, 30, status))
    rejection = error.value.rejections[0]
    assert rejection.reason == "memory_budget"
    assert rejection.available_memory_gb == pytest.approx(28.9)
    assert "90%" in rejection.detail


def test_shared_reservations_from_both_executors_are_counted():
    docker = observed(GpuAllocation(0, 0, 0.4, 40))
    native = observed(GpuAllocation(0, 0, 0.3, 30), cid="sparkrun_" + "c" * 16 + "_" + "d" * 12, executor="local")
    merged = docker.merged_with(native)
    gpu = merged.hosts[0].gpus[0]
    assert gpu.used_util_fraction == pytest.approx(0.7)
    assert gpu.used_memory_gb == 70
    assert SparsePackScheduler().schedule(request(0.1, 10, merged))
    with pytest.raises(InfeasibleScheduleError):
        SparsePackScheduler().schedule(request(0.4, 10, merged))


def test_duplicate_observations_do_not_double_reserve():
    status = observed(GpuAllocation(0, 0, 0.4, 40))
    merged = status.merged_with(status)
    assert merged.hosts[0].gpus[0].used_util_fraction == 0.4
    assert merged.hosts[0].gpus[0].used_memory_gb == 40


def test_unknown_legacy_workload_is_not_shareable():
    legacy = ClusterStatus((HostOccupancy("h", (RunningWorkload(CID),), used_slots=1),))
    with pytest.raises(InfeasibleScheduleError):
        SparsePackScheduler().schedule(request(0.1, 1, legacy))


def test_failed_observation_has_specific_reason():
    with pytest.raises(InfeasibleScheduleError) as error:
        SparsePackScheduler().schedule(request(status=ClusterStatus(errors={"h": "Docker status failed"})))
    assert error.value.rejections[0].reason == "observation_failed"


def test_owner_on_gpu_one_does_not_reserve_gpu_zero():
    status = observed(GpuAllocation(4, 1), capacity=2)
    assert SparsePackScheduler().schedule(request(status=status, count=2)).assignment.by_rank[0].local_gpu == 0
    assert with_gpu_allocations(status.hosts[0], 2).gpus[0].exclusive is False


def test_worker_labels_recover_partial_survivor_and_physical_gpu(monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.executors.docker._load_metadata_safely", lambda _: None)
    placement = RankAssignment((RankSlot("gone", 0), RankSlot("h", 1)), ("gone", "h"))
    labels = Executor.workload_labels_for_cluster(CID, placement=placement, host="h", rank=1)
    payload = json.dumps({"Names": CID + "_node_1", "ID": "worker", "Labels": ",".join(k + "=" + v for k, v in labels.items())})
    workloads, used = _parse_docker_ps_output(payload, "h")
    recovered = with_gpu_allocations(HostOccupancy("h", tuple(workloads), used_slots=used), 2)
    assert not recovered.gpus[0].exclusive
    assert recovered.gpus[1].exclusive
    assert recovered.gpus[1].workloads[0].containers[0].name.endswith("node_1")
    command = DockerExecutor(ExecutorConfig(gpu_access_mode="gpus")).run_cmd(
        "image", container_name=CID + "_node_1", sparkrun_labels=labels
    )
    assert "device=1" in command


def test_real_native_launch_persists_and_cleans_allocation(tmp_path, monkeypatch):
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path / "pids"), log_dir=str(tmp_path / "logs")))
    allocation = GpuAllocation(0, 0, 0.4, 40)
    labels = {ALLOCATION_LABEL: encode_allocations((allocation,))}
    name = CID + "_solo"

    def execute(hosts, script, **kwargs):
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
        return [RemoteResult(host, result.returncode, result.stdout, result.stderr) for host in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", execute)
    monkeypatch.setattr("sparkrun.orchestration.executors.local._load_metadata_safely", lambda _: None)
    command = executor.generate_exec_serve_script(name, "sleep 60", sparkrun_labels=labels)
    try:
        subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True, timeout=10)
        state = executor.query_status(["h"], host_hardware={"h": hardware()})
        assert not state.errors
        gpu = state.hosts[0].gpus[0]
        assert gpu.used_util_fraction == 0.4 and gpu.used_memory_gb == 40
        assert not gpu.exclusive
        assert gpu.workloads[0].containers[0].allocations == (allocation,)
    finally:
        subprocess.run(["bash", "-c", executor.stop_cmd(name)], capture_output=True, text=True, check=True, timeout=10)
    assert not list((tmp_path / "pids").glob("*.allocations"))


@pytest.mark.parametrize("value", [None, "bad", "e30=", encode_allocations(())])
def test_invalid_or_missing_allocation_record_never_establishes_sharing(value):
    assert decode_allocations(value) == ()


@pytest.mark.parametrize("factory", [SparsePackScheduler, DensePackScheduler])
def test_solo_all_ranks_require_one_host(factory):
    req = SchedulingRequest(
        ParallelismConfig(tensor_parallel=2),
        ("small", "large"),
        host_hardware={"small": hardware(), "large": hardware(2)},
        single_host=True,
    )
    placement = factory().schedule(req).assignment
    assert placement.hosts_used == ("large",)
    assert [slot.local_gpu for slot in placement.by_rank] == [0, 1]
    with pytest.raises(InfeasibleScheduleError):
        factory().schedule(replace(req, host_hardware={"small": hardware(), "large": hardware()}))


@pytest.mark.parametrize("indices,reason", [([0, 0], "same_request"), ([0, 2], "invalid_gpu")])
def test_explicit_layout_cannot_duplicate_or_invent_exclusive_gpus(indices, reason):
    from sparkrun.core.layout import RecipeLayout, Placement

    req = replace(
        request(count=2),
        parallelism=ParallelismConfig(tensor_parallel=2),
        layout=RecipeLayout(placements=(Placement(host="h", ranks=(0, 1), local_gpus=tuple(indices)),)),
    )
    with pytest.raises(InfeasibleScheduleError) as error:
        SparsePackScheduler().schedule(req)
    assert error.value.rejections[0].reason == reason


def test_same_job_in_distinct_executors_reserves_both_processes():
    from sparkrun.core.cluster_status import attribute_executor

    status = observed(GpuAllocation(0, 0, 0.3, 35))
    merged = attribute_executor(status, "docker").merged_with(attribute_executor(status, "local"))
    gpu = merged.hosts[0].gpus[0]
    assert len(merged.hosts[0].workloads[0].containers) == 2
    assert gpu.used_util_fraction == pytest.approx(0.6)
    assert gpu.used_memory_gb == 70
    with pytest.raises(InfeasibleScheduleError):
        SparsePackScheduler().schedule(request(0.2, 40, merged))


def test_conflicting_records_cannot_establish_sharing():
    status = observed(GpuAllocation(0, 0, 0.3, 30))
    work = status.hosts[0].workloads[0]
    conflict = replace(work.containers[0], name="conflicting", allocations=(GpuAllocation(0, 0, 0.1, 10),))
    occupancy = replace(status.hosts[0], workloads=(replace(work, containers=work.containers + (conflict,)),))
    recovered = with_gpu_allocations(occupancy, 1)
    assert recovered.gpus[0].exclusive


def test_idle_legacy_process_with_gpu_indices_does_not_allow_sharing():
    from sparkrun.core.cluster_status import GpuOccupancy

    legacy = RunningWorkload(CID)
    status = ClusterStatus((HostOccupancy("h", workloads=(legacy,), used_slots=1, gpus=(GpuOccupancy(0, workloads=(legacy,)),)),))
    with pytest.raises(InfeasibleScheduleError):
        SparsePackScheduler().schedule(request(0.1, 1, status))


def test_native_allocation_write_failure_does_not_spawn(tmp_path):
    import os
    from sparkrun.core.allocations import GpuAllocation

    pids = tmp_path / "pids"
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(pids), log_dir=str(tmp_path / "logs")))
    ready = tmp_path / "started"
    shims = tmp_path / "bin"
    shims.mkdir()
    shim = shims / "mv"
    shim.write_text('#!/bin/bash\ncase "${!#}" in *.allocations) exit 73 ;; esac\nexec /usr/bin/mv "$@"\n')
    shim.chmod(0o700)
    labels = {ALLOCATION_LABEL: encode_allocations((GpuAllocation(0, 0),))}
    script = executor.run_cmd("", "touch %s" % ready, CID + "_solo", sparkrun_labels=labels)
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PATH": str(shims) + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert not ready.exists()
    assert not list(pids.glob("*.pid"))


@pytest.mark.parametrize("factory", [SparsePackScheduler, DensePackScheduler])
def test_solo_explicit_layout_can_select_later_host(factory):
    from sparkrun.core.layout import RecipeLayout, Placement

    req = SchedulingRequest(
        ParallelismConfig(),
        ("first", "second"),
        single_host=True,
        host_hardware={"first": hardware(), "second": hardware()},
        layout=RecipeLayout(placements=[Placement("second", [0], [0])]),
    )
    assert factory().schedule(req).assignment.hosts_used == ("second",)
    conflicting = replace(
        req,
        parallelism=ParallelismConfig(tensor_parallel=2),
        layout=RecipeLayout(placements=[Placement("first", [0], [0]), Placement("second", [1], [0])]),
    )
    with pytest.raises(InfeasibleScheduleError, match="one host"):
        factory().schedule(conflicting)


def test_multi_gpu_docker_device_list_survives_shell_and_csv_parsing():
    import csv
    import shlex

    labels = {ALLOCATION_LABEL: encode_allocations((GpuAllocation(0, 1), GpuAllocation(1, 0)))}
    command = DockerExecutor(ExecutorConfig(gpu_access_mode="gpus")).run_cmd("image", sparkrun_labels=labels)
    args = shlex.split(command)
    device_arg = args[args.index("--gpus") + 1]
    assert next(csv.reader([device_arg])) == ["device=1,0"]
    cdi = DockerExecutor(ExecutorConfig(gpu_access_mode="cdi")).run_cmd("image", sparkrun_labels=labels)
    assert "nvidia.com/gpu=1" in cdi and "nvidia.com/gpu=0" in cdi
