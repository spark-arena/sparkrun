"""Tests for ``sparkrun.api.schedule``, ``sparkrun.api.status``, and
``sparkrun.api.list_jobs`` (Task 6).

These three API entries are the read-side / planning surfaces — they
don't launch anything.  Coverage:

- ``schedule``: invokes a named scheduler; translates scheduler-level
  exceptions into the API hierarchy.
- ``status``: resolves the executor via the layered chain and calls
  its ``query_status``.  Verified with mocked SSH.
- ``list_jobs``: walks the on-disk job metadata directory and surfaces
  :class:`JobInfo` entries; gracefully skips unparseable files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import sparkrun.api as api
from sparkrun.api._jobs import _job_info_from_file
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.cluster_status import ClusterStatus, empty_status
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.core.scheduler import (
    RankAssignment,
    ResourceRequest,
    Scheduler,
    SchedulingRequest,
    SchedulingResult,
)
from sparkrun.orchestration.ssh import RemoteResult


# --------------------------------------------------------------------------
# api.schedule
# --------------------------------------------------------------------------


def test_schedule_default_uses_greedy():
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
    )
    result = api.schedule(req)
    assert isinstance(result, SchedulingResult)
    assert result.scheduler_name == "greedy"
    assert result.assignment.hosts_used == ("h1", "h2")


def test_schedule_explicit_greedy_matches_default():
    req = SchedulingRequest(parallelism=ParallelismConfig(tensor_parallel=1), hosts=("h1",))
    by_name = api.schedule(req, scheduler="greedy")
    by_default = api.schedule(req)
    assert by_name.assignment == by_default.assignment


def test_schedule_infeasible_raises_insufficient_capacity():
    """Scheduler-level InfeasibleScheduleError → api.InsufficientCapacity."""
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=10),
        hosts=("h1", "h2"),
    )
    with pytest.raises(api.InsufficientCapacity) as exc_info:
        api.schedule(req)
    assert "10" in str(exc_info.value)


def test_schedule_layout_conflict_raises_layout_required():
    """Scheduler-level LayoutConflictError → api.LayoutRequired."""
    hw = {
        "h1": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", count=1)]),
        "h2": HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300", count=1)]),
    }
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        host_hardware=hw,
    )
    with pytest.raises(api.LayoutRequired):
        api.schedule(req)


def test_schedule_unknown_scheduler_raises_sparkrun_error():
    req = SchedulingRequest(parallelism=ParallelismConfig(), hosts=("h1",))
    with pytest.raises(api.SparkrunError) as exc_info:
        api.schedule(req, scheduler="nonexistent")
    assert "nonexistent" in str(exc_info.value).lower() or "Unknown scheduler" in str(exc_info.value)


def test_schedule_greedy_rejects_fractional_request_via_api():
    """GreedyScheduler's fractional rejection bubbles up as SparkrunError."""
    req = SchedulingRequest(
        parallelism=ParallelismConfig(tensor_parallel=2),
        hosts=("h1", "h2"),
        resources=ResourceRequest(util_fraction=0.5),
    )
    with pytest.raises(api.SparkrunError) as exc_info:
        api.schedule(req)
    assert "fractional" in str(exc_info.value).lower()


def test_schedule_custom_scheduler_selectable_by_name():
    """Register a no-op scheduler and select it through the API."""
    from scitrera_app_framework import register_plugin

    from sparkrun.core.bootstrap import init_sparkrun

    class _Noop(Scheduler):
        scheduler_name = "noop-api-test"

        def schedule(self, request: SchedulingRequest) -> SchedulingResult:
            return SchedulingResult(
                assignment=RankAssignment(by_rank=(), hosts_used=()),
                scheduler_name=self.scheduler_name,
                diagnostics=("noop",),
            )

    v = init_sparkrun()
    register_plugin(_Noop, v=v)

    req = SchedulingRequest(parallelism=ParallelismConfig(), hosts=("h1",))
    result = api.schedule(req, scheduler="noop-api-test")
    assert result.scheduler_name == "noop-api-test"


# --------------------------------------------------------------------------
# api.status
# --------------------------------------------------------------------------


def test_status_default_executor_is_docker():
    """Without cluster/executor override, the default Docker executor's
    query_status runs (mocked here so we just verify the resolved
    executor name surfaces on the returned ClusterStatus)."""
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[RemoteResult(host="host-a", returncode=0, stdout="", stderr="")],
    ):
        snapshot = api.status(["host-a"])
    assert isinstance(snapshot, ClusterStatus)
    assert snapshot.executor == "docker"


def test_status_explicit_executor_override_wins():
    """``executor='local'`` selects LocalExecutor; status comes back with executor='local'."""
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[RemoteResult(host="host-a", returncode=0, stdout="", stderr="")],
    ):
        snapshot = api.status(["host-a"], executor="local")
    assert snapshot.executor == "local"


def test_status_cluster_executor_is_honored():
    """Cluster's ``executor`` selector flows through resolve_executor."""
    cluster = ClusterDefinition(name="c", hosts=["host-a"], executor="local")
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[RemoteResult(host="host-a", returncode=0, stdout="", stderr="")],
    ):
        snapshot = api.status(["host-a"], cluster=cluster)
    assert snapshot.executor == "local"


def test_status_cli_override_beats_cluster():
    """``executor=`` (CLI-level) wins over cluster.executor."""
    cluster = ClusterDefinition(name="c", hosts=["host-a"], executor="local")
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[RemoteResult(host="host-a", returncode=0, stdout="", stderr="")],
    ):
        snapshot = api.status(["host-a"], executor="docker", cluster=cluster)
    assert snapshot.executor == "docker"


def test_status_cluster_hardware_propagates_to_query():
    """Cluster's ``hosts_hardware`` reaches the executor's query_status."""
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200", count=4, memory_gb=141.0)])
    cluster = ClusterDefinition(
        name="c",
        hosts=["big-host"],
        executor="docker",
        hosts_hardware={"big-host": hw},
    )
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[RemoteResult(host="big-host", returncode=0, stdout="", stderr="")],
    ):
        snapshot = api.status(["big-host"], cluster=cluster)
    occ = snapshot.for_host("big-host")
    # 4-GPU host, nothing running → 4 free slots (proves host_hardware passed through).
    assert occ is not None
    assert occ.free_slots == 4


def test_status_empty_host_list_returns_empty():
    snapshot = api.status([])
    assert snapshot.hosts == ()


def test_status_k8s_executor_returns_safe_default():
    """K8sExecutor inherits the empty-status default in Phase 1."""
    snapshot = api.status(["host-a", "host-b"], executor="k8s")
    # No SSH was called — default impl returns empty_status directly.
    assert snapshot.executor == "k8s"
    assert len(snapshot.hosts) == 2
    assert all(h.workloads == () for h in snapshot.hosts)


# --------------------------------------------------------------------------
# api.list_jobs
# --------------------------------------------------------------------------


def _write_job_meta(jobs_dir: Path, digest: str, **fields) -> Path:
    """Helper: write a YAML job metadata file with sane defaults."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "cluster_id": "sparkrun_%s" % digest,
        "recipe": "test-recipe",
        "runtime": "vllm",
        "hosts": ["h1"],
        **fields,
    }
    path = jobs_dir / ("%s.yaml" % digest)
    path.write_text(yaml.safe_dump(data))
    return path


def test_list_jobs_empty_when_no_jobs_dir(tmp_path: Path):
    """No ``jobs/`` directory → empty list, never raises."""
    assert api.list_jobs(cache_dir=tmp_path) == []


def test_list_jobs_returns_job_info_entries(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    _write_job_meta(jobs_dir, "aaaaaaaaaaaa", started_at=100.0)
    _write_job_meta(jobs_dir, "bbbbbbbbbbbb", started_at=200.0)

    jobs = api.list_jobs(cache_dir=tmp_path)
    assert len(jobs) == 2
    cluster_ids = {j.cluster_id for j in jobs}
    assert cluster_ids == {"sparkrun_aaaaaaaaaaaa", "sparkrun_bbbbbbbbbbbb"}


def test_list_jobs_sorted_most_recent_first(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    _write_job_meta(jobs_dir, "111111111111", started_at=100.0)
    _write_job_meta(jobs_dir, "222222222222", started_at=300.0)
    _write_job_meta(jobs_dir, "333333333333", started_at=200.0)

    jobs = api.list_jobs(cache_dir=tmp_path)
    assert [j.started_at for j in jobs] == [300.0, 200.0, 100.0]


def test_list_jobs_untimed_entries_come_last(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    _write_job_meta(jobs_dir, "withtime0000", started_at=100.0)
    _write_job_meta(jobs_dir, "notime000000")  # no started_at field

    jobs = api.list_jobs(cache_dir=tmp_path)
    assert len(jobs) == 2
    assert jobs[0].cluster_id == "sparkrun_withtime0000"
    assert jobs[1].cluster_id == "sparkrun_notime000000"
    assert jobs[1].started_at is None


def test_list_jobs_surfaces_recipe_runtime_hosts(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    _write_job_meta(
        jobs_dir,
        "abcdef012345",
        recipe="@arena/qwen3-vllm",
        runtime="vllm",
        hosts=["spark-01", "spark-02"],
    )
    jobs = api.list_jobs(cache_dir=tmp_path)
    assert jobs[0].recipe == "@arena/qwen3-vllm"
    assert jobs[0].runtime == "vllm"
    assert jobs[0].hosts == ("spark-01", "spark-02")


def test_list_jobs_skips_unparseable_files(tmp_path: Path):
    """A corrupt YAML file is skipped, not raised."""
    jobs_dir = tmp_path / "jobs"
    _write_job_meta(jobs_dir, "validvalidvali", started_at=100.0)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "corrupt.yaml").write_text("this is :: not valid: yaml: [[[")

    jobs = api.list_jobs(cache_dir=tmp_path)
    cluster_ids = {j.cluster_id for j in jobs}
    assert "sparkrun_validvalidvali" in cluster_ids
    # The corrupt file's entry is absent rather than crashing the call.
    assert all(j.cluster_id != "sparkrun_corrupt" for j in jobs)


def test_list_jobs_metadata_field_exposes_raw_yaml(tmp_path: Path):
    """Beyond canonical fields, the raw YAML is available under .metadata."""
    jobs_dir = tmp_path / "jobs"
    _write_job_meta(
        jobs_dir,
        "abcdef012345",
        port=8000,
        served_model_name="qwen3",
        custom_extension={"foo": "bar"},
    )
    job = api.list_jobs(cache_dir=tmp_path)[0]
    assert job.metadata["port"] == 8000
    assert job.metadata["served_model_name"] == "qwen3"
    assert job.metadata["custom_extension"] == {"foo": "bar"}


def test_job_info_from_file_recovers_cluster_id_from_filename(tmp_path: Path):
    """Files without ``cluster_id`` field fall back to the filename digest."""
    p = tmp_path / "recoveredfromname.yaml"
    p.write_text(yaml.safe_dump({"recipe": "x", "runtime": "y"}))
    info = _job_info_from_file(p)
    assert info is not None
    assert info.cluster_id == "sparkrun_recoveredfromname"


# --------------------------------------------------------------------------
# Click independence — full API surface still must not import click
# --------------------------------------------------------------------------


def test_full_api_imports_without_click():
    """Importing the full ``sparkrun.api`` (now with run-side stubs and
    read-side functions) must not bring click into sys.modules."""
    import subprocess
    import sys

    code = (
        "import sys, importlib;"
        "m = importlib.import_module('sparkrun.api');"
        "assert all(hasattr(m, n) for n in ('schedule', 'status', 'list_jobs'));"
        "assert 'click' not in sys.modules, 'click should not be pulled in'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, "stdout=%s\nstderr=%s" % (result.stdout, result.stderr)


# Cross-check: empty_status is the right shape for the executor default.
def test_empty_status_shape_matches_query_status_default():
    """Sanity: empty_status returns the same shape Executor.query_status's
    default uses, so any consumer of api.status sees a uniform structure."""
    s = empty_status(["a", "b"], executor="docker")
    assert isinstance(s, ClusterStatus)
    assert len(s.hosts) == 2
    assert s.executor == "docker"
