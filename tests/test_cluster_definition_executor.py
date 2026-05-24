"""Tests for ClusterDefinition.executor and ClusterDefinition.executor_config.

Covers:

- Dataclass defaults (both fields default to None — back-compat).
- YAML round-trip: write then read returns identical fields.
- Back-compat: old cluster YAMLs without the new fields still load.
- ``ClusterManager.create`` and ``ClusterManager.update`` accept the
  new fields.
- ``resolve_cluster_config`` propagates them into
  :class:`ResolvedClusterConfig` regardless of whether hosts come
  from the cluster or from explicit --hosts.
"""

from __future__ import annotations

from pathlib import Path

from sparkrun.core.cluster_manager import (
    ClusterDefinition,
    ClusterManager,
    ResolvedClusterConfig,
    resolve_cluster_config,
)


# --------------------------------------------------------------------------
# Dataclass defaults — back-compat
# --------------------------------------------------------------------------


def test_cluster_definition_executor_defaults_none():
    cluster = ClusterDefinition(name="c", hosts=["h1"])
    assert cluster.executor is None
    assert cluster.executor_config is None


def test_cluster_definition_carries_executor_fields():
    cluster = ClusterDefinition(
        name="c",
        hosts=["h1"],
        executor="docker",
        executor_config={"privileged": False, "shm_size": "16g"},
    )
    assert cluster.executor == "docker"
    assert cluster.executor_config == {"privileged": False, "shm_size": "16g"}


def test_to_dict_omits_executor_fields_when_unset():
    cluster = ClusterDefinition(name="c", hosts=["h1"])
    d = cluster.to_dict()
    assert "executor" not in d
    assert "executor_config" not in d


def test_to_dict_emits_executor_fields_when_set():
    cluster = ClusterDefinition(
        name="c",
        hosts=["h1"],
        executor="k8s",
        executor_config={"k8s_namespace": "inference"},
    )
    d = cluster.to_dict()
    assert d["executor"] == "k8s"
    assert d["executor_config"] == {"k8s_namespace": "inference"}


# --------------------------------------------------------------------------
# YAML round-trip
# --------------------------------------------------------------------------


def test_cluster_manager_yaml_round_trip_with_executor_fields(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(
        name="prod",
        hosts=["h1", "h2"],
        executor="docker",
        executor_config={"privileged": True, "memory_limit": "64g"},
    )
    loaded = mgr.get("prod")
    assert loaded.executor == "docker"
    assert loaded.executor_config == {"privileged": True, "memory_limit": "64g"}


def test_cluster_manager_yaml_round_trip_without_executor_fields(tmp_path: Path):
    """Cluster without executor fields loads with None defaults."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="basic", hosts=["h1"])
    loaded = mgr.get("basic")
    assert loaded.executor is None
    assert loaded.executor_config is None


def test_cluster_manager_back_compat_with_legacy_yaml(tmp_path: Path):
    """Pre-existing cluster YAML without the new fields still loads cleanly."""
    yaml_text = """
name: legacy
hosts:
- spark-01
- spark-02
description: pre-Task-3 format
user: drew
transfer_mode: push
""".strip()
    (tmp_path / "clusters").mkdir()
    (tmp_path / "clusters" / "legacy.yaml").write_text(yaml_text)

    mgr = ClusterManager(tmp_path)
    loaded = mgr.get("legacy")
    assert loaded.name == "legacy"
    assert loaded.hosts == ["spark-01", "spark-02"]
    assert loaded.user == "drew"
    assert loaded.transfer_mode == "push"
    assert loaded.executor is None
    assert loaded.executor_config is None


def test_cluster_manager_back_compat_treats_empty_executor_config_as_none(tmp_path: Path):
    """An explicit empty dict in YAML should normalize to None on load."""
    yaml_text = """
name: c
hosts: [h1]
executor_config: {}
""".strip()
    (tmp_path / "clusters").mkdir()
    (tmp_path / "clusters" / "c.yaml").write_text(yaml_text)
    mgr = ClusterManager(tmp_path)
    assert mgr.get("c").executor_config is None


# --------------------------------------------------------------------------
# ClusterManager.update accepts the new fields
# --------------------------------------------------------------------------


def test_cluster_manager_update_sets_executor(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"])
    assert mgr.get("c").executor is None

    mgr.update("c", executor="local")
    assert mgr.get("c").executor == "local"


def test_cluster_manager_update_clears_executor(tmp_path: Path):
    """Passing ``executor=None`` explicitly clears the field."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], executor="docker")
    assert mgr.get("c").executor == "docker"

    mgr.update("c", executor=None)
    assert mgr.get("c").executor is None


def test_cluster_manager_update_preserves_when_unset(tmp_path: Path):
    """Omitting executor in update keeps the prior value (UNSET sentinel)."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], executor="k8s")
    mgr.update("c", description="changed only the description")
    assert mgr.get("c").executor == "k8s"


def test_cluster_manager_update_sets_executor_config(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"])
    mgr.update("c", executor_config={"privileged": False})
    assert mgr.get("c").executor_config == {"privileged": False}


def test_cluster_manager_update_clears_executor_config_with_none(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], executor_config={"shm_size": "8g"})
    mgr.update("c", executor_config=None)
    assert mgr.get("c").executor_config is None


def test_cluster_manager_update_empty_dict_clears_executor_config(tmp_path: Path):
    """An empty dict normalizes to None — matches how YAML round-tripping works."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], executor_config={"shm_size": "8g"})
    mgr.update("c", executor_config={})
    assert mgr.get("c").executor_config is None


# --------------------------------------------------------------------------
# ResolvedClusterConfig propagation
# --------------------------------------------------------------------------


def test_resolved_cluster_config_defaults_none():
    cfg = ResolvedClusterConfig()
    assert cfg.executor is None
    assert cfg.executor_config is None


def test_resolve_cluster_config_propagates_executor_from_named_cluster(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(
        name="prod",
        hosts=["h1"],
        executor="docker",
        executor_config={"privileged": False},
    )
    resolved = resolve_cluster_config(
        cluster_name="prod",
        hosts=None,
        hosts_file=None,
        cluster_mgr=mgr,
    )
    assert resolved.executor == "docker"
    assert resolved.executor_config == {"privileged": False}


def test_resolve_cluster_config_executor_applies_even_with_explicit_hosts(tmp_path: Path):
    """Executor selection is a cluster-deployment property; it applies even
    when --hosts is passed alongside --cluster (unlike transfer_mode etc.)."""
    mgr = ClusterManager(tmp_path)
    mgr.create(
        name="prod",
        hosts=["h1"],
        executor="k8s",
        executor_config={"k8s_namespace": "inference"},
    )
    resolved = resolve_cluster_config(
        cluster_name="prod",
        hosts="other-host",  # explicit --hosts
        hosts_file=None,
        cluster_mgr=mgr,
    )
    # Transfer-related fields should NOT be set (explicit --hosts wins),
    # but the executor selector should propagate.
    assert resolved.transfer_mode is None
    assert resolved.executor == "k8s"
    assert resolved.executor_config == {"k8s_namespace": "inference"}


def test_resolve_cluster_config_no_cluster_yields_none(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    resolved = resolve_cluster_config(
        cluster_name=None,
        hosts="some-host",
        hosts_file=None,
        cluster_mgr=mgr,
    )
    assert resolved.executor is None
    assert resolved.executor_config is None


def test_resolve_cluster_config_executor_config_is_a_copy(tmp_path: Path):
    """Mutations to the resolved dict shouldn't bleed back into the on-disk cluster."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], executor_config={"shm_size": "32g"})
    resolved = resolve_cluster_config("c", None, None, mgr)
    assert resolved.executor_config is not None
    resolved.executor_config["shm_size"] = "MUTATED"
    # Re-load — should still be the original.
    assert mgr.get("c").executor_config == {"shm_size": "32g"}


# --------------------------------------------------------------------------
# ClusterDefinition.scheduler — mirrors the executor pattern above
# --------------------------------------------------------------------------


def test_cluster_definition_scheduler_defaults_none():
    cluster = ClusterDefinition(name="c", hosts=["h1"])
    assert cluster.scheduler is None


def test_cluster_definition_carries_scheduler_field():
    cluster = ClusterDefinition(name="c", hosts=["h1"], scheduler="occupancy-aware")
    assert cluster.scheduler == "occupancy-aware"


def test_to_dict_omits_scheduler_when_unset():
    cluster = ClusterDefinition(name="c", hosts=["h1"])
    assert "scheduler" not in cluster.to_dict()


def test_to_dict_emits_scheduler_when_set():
    cluster = ClusterDefinition(name="c", hosts=["h1"], scheduler="greedy")
    assert cluster.to_dict()["scheduler"] == "greedy"


def test_cluster_manager_yaml_round_trip_with_scheduler(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(name="prod", hosts=["h1", "h2"], scheduler="occupancy-aware")
    loaded = mgr.get("prod")
    assert loaded.scheduler == "occupancy-aware"


def test_cluster_manager_yaml_round_trip_without_scheduler(tmp_path: Path):
    """Cluster without scheduler field loads with None default."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="basic", hosts=["h1"])
    assert mgr.get("basic").scheduler is None


def test_cluster_manager_back_compat_with_legacy_yaml_no_scheduler(tmp_path: Path):
    """Pre-existing cluster YAML without scheduler still loads cleanly."""
    yaml_text = """
name: legacy
hosts:
- spark-01
description: pre-scheduler-field format
""".strip()
    (tmp_path / "clusters").mkdir()
    (tmp_path / "clusters" / "legacy.yaml").write_text(yaml_text)

    mgr = ClusterManager(tmp_path)
    loaded = mgr.get("legacy")
    assert loaded.name == "legacy"
    assert loaded.scheduler is None


def test_cluster_manager_create_persists_scheduler(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], scheduler="greedy")
    assert mgr.get("c").scheduler == "greedy"


def test_cluster_manager_update_sets_scheduler(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"])
    assert mgr.get("c").scheduler is None

    mgr.update("c", scheduler="occupancy-aware")
    assert mgr.get("c").scheduler == "occupancy-aware"


def test_cluster_manager_update_clears_scheduler(tmp_path: Path):
    """Passing ``scheduler=None`` explicitly clears the field."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], scheduler="greedy")
    assert mgr.get("c").scheduler == "greedy"

    mgr.update("c", scheduler=None)
    assert mgr.get("c").scheduler is None


def test_cluster_manager_update_preserves_scheduler_when_unset(tmp_path: Path):
    """Omitting scheduler in update keeps the prior value (UNSET sentinel)."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="c", hosts=["h1"], scheduler="occupancy-aware")
    mgr.update("c", description="changed only the description")
    assert mgr.get("c").scheduler == "occupancy-aware"


def test_resolved_cluster_config_scheduler_defaults_none():
    cfg = ResolvedClusterConfig()
    assert cfg.scheduler is None


def test_resolve_cluster_config_propagates_scheduler_from_named_cluster(tmp_path: Path):
    mgr = ClusterManager(tmp_path)
    mgr.create(name="prod", hosts=["h1"], scheduler="occupancy-aware")
    resolved = resolve_cluster_config(
        cluster_name="prod",
        hosts=None,
        hosts_file=None,
        cluster_mgr=mgr,
    )
    assert resolved.scheduler == "occupancy-aware"


def test_resolve_cluster_config_scheduler_applies_even_with_explicit_hosts(tmp_path: Path):
    """Scheduler selection is a cluster-deployment property; it applies even
    when --hosts is passed alongside --cluster (mirrors executor semantics)."""
    mgr = ClusterManager(tmp_path)
    mgr.create(name="prod", hosts=["h1"], scheduler="occupancy-aware")
    resolved = resolve_cluster_config(
        cluster_name="prod",
        hosts="other-host",  # explicit --hosts
        hosts_file=None,
        cluster_mgr=mgr,
    )
    # Transfer-related fields should NOT be set (explicit --hosts wins),
    # but the scheduler selector should propagate.
    assert resolved.transfer_mode is None
    assert resolved.scheduler == "occupancy-aware"
