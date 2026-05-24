"""Tests for sparkrun label emission at container/Pod launch time.

These tests cover the second half of the workload-identity label
contract: the parser side is in ``test_executor_query_status.py``
(``_parse_docker_ps_output`` reads labels off ``docker ps`` output).
Here we verify that the emission side actually attaches the labels
in the first place — so ``docker ps --filter
"label=sparkrun.intent_id=<x>"`` becomes a real discovery mechanism.

Covers:

- :meth:`Executor.workload_labels_for_cluster` derives the canonical
  label set from cluster_id + Recipe + RuntimePlugin (+ optional rank).
- :meth:`DockerExecutor.run_cmd` emits ``--label key=value`` flags
  alongside user-supplied ``cfg.labels``.
- :meth:`K8sExecutor.run_cmd` emits ``--labels=key=value`` flags.
- :meth:`LocalExecutor.run_cmd` accepts and ignores the kwarg (no
  container to tag).
- Round-trip: emit via ``run_cmd``, parse via
  ``_parse_docker_ps_output`` — recipe / runtime / rank survive
  the trip cleanly.
"""

from __future__ import annotations

import json

from sparkrun.orchestration.executors._base import (
    LABEL_CLUSTER_ID,
    LABEL_INTENT_ID,
    LABEL_RANK,
    LABEL_RECIPE,
    LABEL_RUNTIME,
    Executor,
    ExecutorConfig,
)
from sparkrun.orchestration.executors.docker import (
    DockerExecutor,
    _parse_docker_ps_output,
)
from sparkrun.orchestration.executors.k8s import K8sExecutor
from sparkrun.orchestration.executors.local import LocalExecutor


# --------------------------------------------------------------------------
# Helpers — minimal stand-ins for Recipe / RuntimePlugin
# --------------------------------------------------------------------------


class _StubRecipe:
    def __init__(self, qualified_name: str):
        self.qualified_name = qualified_name


class _StubRuntime:
    def __init__(self, runtime_name: str):
        self.runtime_name = runtime_name


# --------------------------------------------------------------------------
# workload_labels_for_cluster
# --------------------------------------------------------------------------


def test_workload_labels_for_cluster_full_set():
    labels = Executor.workload_labels_for_cluster(
        cluster_id="sparkrun_abc123abc123abc1_def456abcdef",
        recipe=_StubRecipe("@arena/qwen3-1.7b-vllm"),
        runtime=_StubRuntime("vllm"),
        rank=2,
    )
    assert labels[LABEL_CLUSTER_ID] == "sparkrun_abc123abc123abc1_def456abcdef"
    assert labels[LABEL_INTENT_ID] == "abc123abc123abc1"
    assert labels[LABEL_RECIPE] == "@arena/qwen3-1.7b-vllm"
    assert labels[LABEL_RUNTIME] == "vllm"
    assert labels[LABEL_RANK] == "2"


def test_workload_labels_for_cluster_no_rank_omits_rank_label():
    labels = Executor.workload_labels_for_cluster(
        cluster_id="sparkrun_abc123abc123abc1_def456abcdef",
        recipe=_StubRecipe("@arena/qwen3-vllm"),
        runtime=_StubRuntime("vllm"),
    )
    assert LABEL_RANK not in labels


def test_workload_labels_for_cluster_rank_zero_is_emitted():
    labels = Executor.workload_labels_for_cluster(
        cluster_id="sparkrun_abc123abc123abc1_def456abcdef",
        recipe=None,
        runtime=None,
        rank=0,
    )
    assert labels[LABEL_RANK] == "0"


def test_workload_labels_for_cluster_none_recipe_runtime_still_emits_identity():
    labels = Executor.workload_labels_for_cluster(
        cluster_id="sparkrun_abc123abc123abc1_def456abcdef",
    )
    assert labels[LABEL_CLUSTER_ID] == "sparkrun_abc123abc123abc1_def456abcdef"
    assert labels[LABEL_INTENT_ID] == "abc123abc123abc1"
    assert LABEL_RECIPE not in labels
    assert LABEL_RUNTIME not in labels


def test_workload_labels_for_cluster_empty_cluster_id_returns_empty():
    assert Executor.workload_labels_for_cluster(cluster_id="") == {}


# --------------------------------------------------------------------------
# DockerExecutor.run_cmd label emission
# --------------------------------------------------------------------------


def test_docker_run_cmd_emits_sparkrun_label_flags():
    executor = DockerExecutor()
    cmd = executor.run_cmd(
        image="img",
        command="echo hi",
        container_name="test-container",
        sparkrun_labels={
            LABEL_CLUSTER_ID: "sparkrun_abc123abc123abc1_def456abcdef",
            LABEL_INTENT_ID: "abc123abc123abc1",
            LABEL_RECIPE: "@arena/qwen3-vllm",
            LABEL_RUNTIME: "vllm",
            LABEL_RANK: "0",
        },
    )
    assert "--label sparkrun.cluster_id=sparkrun_abc123abc123abc1_def456abcdef" in cmd
    assert "--label sparkrun.intent_id=abc123abc123abc1" in cmd
    assert "--label %s=@arena/qwen3-vllm" % LABEL_RECIPE in cmd
    assert "--label sparkrun.runtime=vllm" in cmd
    assert "--label sparkrun.rank=0" in cmd


def test_docker_run_cmd_no_sparkrun_labels_emits_nothing_new():
    """Backwards compat: callers that don't pass sparkrun_labels keep working."""
    executor = DockerExecutor()
    cmd = executor.run_cmd(
        image="img",
        command="echo hi",
        container_name="test-container",
    )
    assert "sparkrun.cluster_id" not in cmd
    assert "sparkrun.intent_id" not in cmd


def test_docker_run_cmd_user_labels_coexist_with_sparkrun_labels():
    """cfg.labels (user-supplied) AND sparkrun_labels both emit."""
    cfg = ExecutorConfig(labels=["env=staging", "team=ml"])
    executor = DockerExecutor(config=cfg)
    cmd = executor.run_cmd(
        image="img",
        command="echo hi",
        container_name="test-container",
        sparkrun_labels={LABEL_CLUSTER_ID: "sparkrun_xyz"},
    )
    assert "--label env=staging" in cmd
    assert "--label team=ml" in cmd
    assert "--label sparkrun.cluster_id=sparkrun_xyz" in cmd


# --------------------------------------------------------------------------
# K8sExecutor.run_cmd label emission
# --------------------------------------------------------------------------


def test_k8s_run_cmd_emits_sparkrun_labels_as_kubectl_flags():
    executor = K8sExecutor()
    cmd = executor.run_cmd(
        image="img",
        command="echo hi",
        container_name="pod-name",
        sparkrun_labels={
            LABEL_CLUSTER_ID: "sparkrun_xyz",
            LABEL_INTENT_ID: "xyz",
        },
    )
    assert "--labels=sparkrun.cluster_id=sparkrun_xyz" in cmd
    assert "--labels=sparkrun.intent_id=xyz" in cmd


def test_k8s_run_cmd_no_sparkrun_labels_kwarg_does_not_break():
    executor = K8sExecutor()
    cmd = executor.run_cmd(
        image="img",
        command="echo hi",
        container_name="pod-name",
    )
    assert "sparkrun.cluster_id" not in cmd


# --------------------------------------------------------------------------
# LocalExecutor.run_cmd no-op for labels
# --------------------------------------------------------------------------


def test_local_run_cmd_accepts_sparkrun_labels_kwarg_no_op():
    """LocalExecutor has no container — kwarg should be accepted and ignored."""
    executor = LocalExecutor()
    cmd = executor.run_cmd(
        image="ignored",
        command="echo hi",
        container_name="sparkrun_abc123abc123abc1_def456abcdef_solo",
        sparkrun_labels={LABEL_CLUSTER_ID: "sparkrun_abc"},
    )
    # No --label flags should appear (it's a setsid native launcher).
    assert "--label" not in cmd
    assert "sparkrun.cluster_id" not in cmd


# --------------------------------------------------------------------------
# Round-trip — emit via run_cmd, parse via _parse_docker_ps_output
# --------------------------------------------------------------------------


def _docker_ps_line(name: str, container_id: str = "abc123", labels: str = "") -> str:
    return json.dumps({"Names": name, "ID": container_id, "Labels": labels})


def test_label_emission_round_trip_through_parser():
    """Emit labels via DockerExecutor → simulate docker ps output → parse them back."""
    labels = Executor.workload_labels_for_cluster(
        cluster_id="sparkrun_abc123abc123abc1_def456abcdef",
        recipe=_StubRecipe("@arena/qwen3-vllm"),
        runtime=_StubRuntime("vllm"),
        rank=0,
    )
    # Emit (sanity check that the labels are well-formed for docker ps)
    executor = DockerExecutor()
    cmd = executor.run_cmd(
        image="img",
        command="echo hi",
        container_name="sparkrun_abc123abc123abc1_def456abcdef_solo",
        sparkrun_labels=labels,
    )
    assert "--label sparkrun.cluster_id=" in cmd

    # Simulate the docker ps output line that the daemon would produce
    # for a container created with these labels.  Docker emits them as
    # comma-separated ``k1=v1,k2=v2`` in the ``Labels`` field.
    labels_str = ",".join("%s=%s" % (k, v) for k, v in sorted(labels.items()))
    stdout = _docker_ps_line(
        "sparkrun_abc123abc123abc1_def456abcdef_solo",
        labels=labels_str,
    )

    workloads, used = _parse_docker_ps_output(stdout, "host-a")
    assert used == 1
    w = workloads[0]
    assert w.cluster_id == "sparkrun_abc123abc123abc1_def456abcdef"
    assert w.intent_id == "abc123abc123abc1"
    assert w.recipe_name == "@arena/qwen3-vllm"
    assert w.runtime_name == "vllm"


# --------------------------------------------------------------------------
# Script-generator forwarding — high-level generators pass labels through
# --------------------------------------------------------------------------


def test_generate_node_script_forwards_labels_to_run_cmd():
    """DockerExecutor.generate_node_script must thread sparkrun_labels through."""
    executor = DockerExecutor()
    script = executor.generate_node_script(
        image="img",
        container_name="sparkrun_abc_def_node_0",
        serve_command="serve",
        sparkrun_labels={LABEL_CLUSTER_ID: "sparkrun_abc"},
    )
    assert "--label sparkrun.cluster_id=sparkrun_abc" in script


def test_generate_launch_script_forwards_labels_to_run_cmd():
    executor = DockerExecutor()
    script = executor.generate_launch_script(
        image="img",
        container_name="sparkrun_abc_def_solo",
        command="sleep infinity",
        sparkrun_labels={LABEL_INTENT_ID: "abc"},
    )
    assert "--label sparkrun.intent_id=abc" in script


def test_generate_ray_head_script_forwards_labels_to_run_cmd():
    executor = DockerExecutor()
    script = executor.generate_ray_head_script(
        image="img",
        container_name="sparkrun_abc_def_head",
        sparkrun_labels={LABEL_RUNTIME: "vllm-ray"},
    )
    assert "--label sparkrun.runtime=vllm-ray" in script


def test_generate_ray_worker_script_forwards_labels_to_run_cmd():
    executor = DockerExecutor()
    script = executor.generate_ray_worker_script(
        image="img",
        container_name="sparkrun_abc_def_worker",
        head_ip="10.0.0.1",
        sparkrun_labels={LABEL_RUNTIME: "vllm-ray"},
    )
    assert "--label sparkrun.runtime=vllm-ray" in script


def test_generate_exec_serve_script_accepts_labels_but_does_not_emit():
    """exec attaches to existing container — labels live on the parent run_cmd."""
    executor = DockerExecutor()
    script = executor.generate_exec_serve_script(
        container_name="sparkrun_abc_def_solo",
        serve_command="serve",
        sparkrun_labels={LABEL_CLUSTER_ID: "sparkrun_abc"},
    )
    # No --label flags appear in the docker exec command line.
    assert "--label" not in script
