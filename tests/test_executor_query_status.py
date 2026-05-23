"""Tests for Executor.query_status and the label-schema vocabulary.

Covers:

- :meth:`Executor.workload_labels` produces the canonical key set.
- :meth:`Executor.query_status` default returns a safe zero-occupancy
  snapshot (K8sExecutor inherits this in Phase 1).
- :func:`sparkrun.orchestration.executors.docker._parse_docker_ps_output`
  groups containers by ``cluster_id``, recovers rank from name,
  honors labels when present, and ignores non-sparkrun names.
- :func:`sparkrun.orchestration.executors.local._parse_local_pidfile_output`
  groups pidfile lines by cluster_id.
- End-to-end ``DockerExecutor.query_status`` via mocked
  ``run_remote_scripts_parallel``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from sparkrun.core.cluster_status import ClusterStatus, RunningWorkload
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.orchestration.executors._base import (
    LABEL_CLUSTER_ID,
    LABEL_RANK,
    LABEL_RECIPE,
    LABEL_RUNTIME,
    Executor,
)
from sparkrun.orchestration.executors.docker import (
    DockerExecutor,
    _parse_docker_labels,
    _parse_docker_ps_output,
)
from sparkrun.orchestration.executors.k8s import K8sExecutor
from sparkrun.orchestration.executors.local import (
    LocalExecutor,
    _parse_local_pidfile_output,
)
from sparkrun.orchestration.ssh import RemoteResult


# --------------------------------------------------------------------------
# workload_labels
# --------------------------------------------------------------------------


def test_workload_labels_minimal_has_cluster_id_only():
    labels = Executor.workload_labels("sparkrun_abc123abc123")
    assert labels == {LABEL_CLUSTER_ID: "sparkrun_abc123abc123"}


def test_workload_labels_full_set():
    labels = Executor.workload_labels(
        "sparkrun_abc123abc123",
        recipe_name="@arena/qwen3-1.7b-vllm",
        runtime_name="vllm",
        rank=3,
    )
    assert labels[LABEL_CLUSTER_ID] == "sparkrun_abc123abc123"
    assert labels[LABEL_RECIPE] == "@arena/qwen3-1.7b-vllm"
    assert labels[LABEL_RUNTIME] == "vllm"
    assert labels[LABEL_RANK] == "3"


def test_workload_labels_rank_zero_emitted():
    """Rank=0 is meaningful (the head rank) and should be emitted."""
    labels = Executor.workload_labels("sparkrun_x", rank=0)
    assert labels[LABEL_RANK] == "0"


def test_workload_labels_skips_empty_strings():
    """Empty recipe/runtime aren't emitted (falsy treatment)."""
    labels = Executor.workload_labels("sparkrun_x", recipe_name="", runtime_name="")
    assert LABEL_RECIPE not in labels
    assert LABEL_RUNTIME not in labels


# --------------------------------------------------------------------------
# Default Executor.query_status — safe zero-occupancy
# --------------------------------------------------------------------------


def test_k8s_query_status_default_returns_empty_with_executor_name():
    """K8sExecutor inherits the default empty-status implementation in Phase 1."""
    status = K8sExecutor().query_status(["host-a", "host-b"])
    assert isinstance(status, ClusterStatus)
    assert status.executor == "k8s"
    assert len(status.hosts) == 2
    assert all(h.used_slots == 0 and h.free_slots == 0 for h in status.hosts)
    assert all(h.workloads == () for h in status.hosts)


def test_default_query_status_empty_hosts_returns_empty():
    status = K8sExecutor().query_status([])
    assert status.hosts == ()
    assert status.executor == "k8s"


# --------------------------------------------------------------------------
# Docker label parsing
# --------------------------------------------------------------------------


def test_parse_docker_labels_basic():
    assert _parse_docker_labels("a=1,b=2,c=3") == {"a": "1", "b": "2", "c": "3"}


def test_parse_docker_labels_empty():
    assert _parse_docker_labels("") == {}
    assert _parse_docker_labels(None) == {}  # type: ignore[arg-type]


def test_parse_docker_labels_ignores_malformed_tokens():
    assert _parse_docker_labels("a=1,malformed,b=2") == {"a": "1", "b": "2"}


# --------------------------------------------------------------------------
# Docker ps output parsing
# --------------------------------------------------------------------------


def _docker_ps_line(name: str, container_id: str = "abc123", labels: str = "") -> str:
    return json.dumps(
        {
            "Names": name,
            "ID": container_id,
            "Image": "vllm:latest",
            "Labels": labels,
            "State": "running",
            "Status": "Up 5 minutes",
        }
    )


def test_parse_docker_ps_empty_output():
    workloads, used = _parse_docker_ps_output("", "host-a")
    assert workloads == []
    assert used == 0


def test_parse_docker_ps_solo_container():
    stdout = _docker_ps_line("sparkrun_abc123abc123_solo", container_id="cid-1")
    workloads, used = _parse_docker_ps_output(stdout, "host-a")
    assert used == 1
    assert len(workloads) == 1
    w = workloads[0]
    assert w.cluster_id == "sparkrun_abc123abc123"
    assert w.ranks_on_host == 1
    assert "cid-1" in w.container_ids


def test_parse_docker_ps_multi_rank_aggregates():
    """Two ranks of the same cluster on one host = one workload with ranks_on_host=2."""
    stdout = "\n".join(
        [
            _docker_ps_line("sparkrun_abc123abc123_node_0", container_id="c0"),
            _docker_ps_line("sparkrun_abc123abc123_node_1", container_id="c1"),
        ]
    )
    workloads, used = _parse_docker_ps_output(stdout, "host-a")
    assert used == 2
    assert len(workloads) == 1
    assert workloads[0].ranks_on_host == 2
    assert set(workloads[0].container_ids) == {"c0", "c1"}


def test_parse_docker_ps_distinct_clusters_kept_separate():
    stdout = "\n".join(
        [
            _docker_ps_line("sparkrun_aaa111aaa111_solo"),
            _docker_ps_line("sparkrun_bbb222bbb222_solo"),
        ]
    )
    workloads, used = _parse_docker_ps_output(stdout, "host-a")
    assert used == 2
    cluster_ids = {w.cluster_id for w in workloads}
    assert cluster_ids == {"sparkrun_aaa111aaa111", "sparkrun_bbb222bbb222"}


def test_parse_docker_ps_ignores_non_sparkrun_names():
    stdout = "\n".join(
        [
            _docker_ps_line("my-postgres", container_id="pg"),
            _docker_ps_line("sparkrun_abc123abc123_solo", container_id="sr"),
            _docker_ps_line("some-other-container"),
        ]
    )
    workloads, used = _parse_docker_ps_output(stdout, "host-a")
    assert used == 1
    assert workloads[0].cluster_id == "sparkrun_abc123abc123"


def test_parse_docker_ps_labels_populate_recipe_and_runtime():
    stdout = _docker_ps_line(
        "sparkrun_abc123abc123_solo",
        labels="%s=@arena/qwen3-vllm,%s=vllm" % (LABEL_RECIPE, LABEL_RUNTIME),
    )
    workloads, _ = _parse_docker_ps_output(stdout, "host-a")
    assert workloads[0].recipe_name == "@arena/qwen3-vllm"
    assert workloads[0].runtime_name == "vllm"


def test_parse_docker_ps_labels_rank_override():
    """When sparkrun.rank label is present, it overrides the name-derived rank."""
    stdout = _docker_ps_line(
        "sparkrun_abc123abc123_node_5",
        labels="%s=7" % LABEL_RANK,
    )
    workloads, _ = _parse_docker_ps_output(stdout, "host-a")
    # Single sighting → one rank on this host regardless of the rank index.
    assert workloads[0].ranks_on_host == 1


def test_parse_docker_ps_ignores_non_json_lines():
    stdout = "\n".join(
        [
            "this is not json",
            _docker_ps_line("sparkrun_abc123abc123_solo"),
            "",
        ]
    )
    workloads, used = _parse_docker_ps_output(stdout, "host-a")
    assert used == 1
    assert len(workloads) == 1


# --------------------------------------------------------------------------
# DockerExecutor.query_status — integration with mocked SSH
# --------------------------------------------------------------------------


def _mock_remote_results(per_host_stdout: dict[str, str], returncode: int = 0) -> list[RemoteResult]:
    return [RemoteResult(host=host, returncode=returncode, stdout=stdout, stderr="") for host, stdout in per_host_stdout.items()]


def test_docker_query_status_two_hosts_one_solo_each():
    executor = DockerExecutor()
    per_host = {
        "host-a": _docker_ps_line("sparkrun_aaa111aaa111_solo"),
        "host-b": _docker_ps_line("sparkrun_bbb222bbb222_solo"),
    }
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=_mock_remote_results(per_host),
    ):
        status = executor.query_status(["host-a", "host-b"])
    assert status.executor == "docker"
    assert len(status.hosts) == 2
    occ_a = status.for_host("host-a")
    occ_b = status.for_host("host-b")
    assert occ_a is not None and occ_a.used_slots == 1
    assert occ_b is not None and occ_b.used_slots == 1
    # DGX Spark default = 1 GPU/host → no free slots when one workload is on it.
    assert occ_a.free_slots == 0
    assert occ_b.free_slots == 0


def test_docker_query_status_unreachable_host_is_skipped():
    executor = DockerExecutor()
    results = [
        RemoteResult(host="host-a", returncode=0, stdout=_docker_ps_line("sparkrun_aaaaaaaaaaaa_solo"), stderr=""),
        RemoteResult(host="host-down", returncode=-1, stdout="", stderr="ssh: unreachable"),
    ]
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=results,
    ):
        status = executor.query_status(["host-a", "host-down"])
    # host-down is omitted; caller can detect via for_host()
    assert {h.host for h in status.hosts} == {"host-a"}
    assert status.for_host("host-down") is None


def test_docker_query_status_respects_host_hardware_capacity():
    """A 4-GPU host with 2 ranks running shows 2 used, 2 free."""
    executor = DockerExecutor()
    h200 = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200", count=4, memory_gb=141.0)])
    stdout = "\n".join(
        [
            _docker_ps_line("sparkrun_aaaaaaaaaaaa_node_0"),
            _docker_ps_line("sparkrun_aaaaaaaaaaaa_node_1"),
        ]
    )
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=_mock_remote_results({"big-host": stdout}),
    ):
        status = executor.query_status(["big-host"], host_hardware={"big-host": h200})
    occ = status.for_host("big-host")
    assert occ is not None
    assert occ.used_slots == 2
    assert occ.free_slots == 2


def test_docker_query_status_empty_host_list():
    """Empty input → empty status, no SSH attempted."""
    with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_ssh:
        status = DockerExecutor().query_status([])
        assert status.hosts == ()
        mock_ssh.assert_not_called()


# --------------------------------------------------------------------------
# LocalExecutor pidfile parsing + query_status
# --------------------------------------------------------------------------


def test_parse_local_pidfile_output_empty():
    workloads, used = _parse_local_pidfile_output("")
    assert workloads == []
    assert used == 0


def test_parse_local_pidfile_output_single():
    stdout = "sparkrun_abc123abc123_solo\t12345\n"
    workloads, used = _parse_local_pidfile_output(stdout)
    assert used == 1
    assert workloads[0].cluster_id == "sparkrun_abc123abc123"
    assert workloads[0].ranks_on_host == 1


def test_parse_local_pidfile_output_multi_rank():
    stdout = "sparkrun_abc123abc123_node_0\t100\nsparkrun_abc123abc123_node_1\t200\n"
    workloads, used = _parse_local_pidfile_output(stdout)
    assert used == 2
    assert len(workloads) == 1
    assert workloads[0].ranks_on_host == 2


def test_parse_local_pidfile_output_ignores_non_sparkrun():
    stdout = "redis-server\t1\nsparkrun_abcabcabcabc_solo\t2\n"
    workloads, used = _parse_local_pidfile_output(stdout)
    assert used == 1
    assert workloads[0].cluster_id == "sparkrun_abcabcabcabc"


def test_local_query_status_runs_ssh_script():
    executor = LocalExecutor()
    with patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=_mock_remote_results({"host-a": "sparkrun_abcdef012345_solo\t999\n"}),
    ) as mock_ssh:
        status = executor.query_status(["host-a"])
    assert mock_ssh.called
    assert status.executor == "local"
    assert status.for_host("host-a").used_slots == 1


def test_local_query_status_empty_host_list():
    with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_ssh:
        status = LocalExecutor().query_status([])
        assert status.hosts == ()
        mock_ssh.assert_not_called()


# --------------------------------------------------------------------------
# Running workload data shape (round-trip with new runtime_name field)
# --------------------------------------------------------------------------


def test_running_workload_runtime_name_optional():
    w = RunningWorkload(cluster_id="x")
    assert w.runtime_name is None

    w2 = RunningWorkload(cluster_id="x", runtime_name="vllm")
    assert w2.runtime_name == "vllm"
