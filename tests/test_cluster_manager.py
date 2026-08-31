"""Tests for sparkrun.cluster_manager and sparkrun.core.monitoring modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sparkrun.core.cluster_manager import ClusterManager, ClusterError, ResolvedClusterConfig
from sparkrun.core.hardware import AcceleratorSpec, HostHardware


def test_create_cluster(tmp_path: Path):
    """Create a cluster and verify all fields persist correctly."""
    manager = ClusterManager(tmp_path)
    manager.create("test-cluster", ["host1", "host2"], description="Test description")

    cluster = manager.get("test-cluster")
    assert cluster.name == "test-cluster"
    assert cluster.hosts == ["host1", "host2"]
    assert cluster.description == "Test description"


def test_create_cluster_already_exists(tmp_path: Path):
    """Creating a cluster with an existing name raises ClusterError."""
    manager = ClusterManager(tmp_path)
    manager.create("duplicate", ["host1"])

    with pytest.raises(ClusterError):
        manager.create("duplicate", ["host2"])


def test_max_gpu_memory_utilization_round_trips(tmp_path: Path):
    """Cluster-wide cap persists via create → write → read; omitted when unset."""
    from sparkrun.core.cluster_manager import ClusterDefinition

    manager = ClusterManager(tmp_path)

    # Unset by default — neither field appears in the serialized dict.
    manager.create("plain", ["host1"])
    plain = manager.get("plain")
    assert plain.max_gpu_memory_utilization is None
    assert plain.accelerator_memory_limits == {}
    assert "max_gpu_memory_utilization" not in plain.to_dict()
    assert "accelerator_memory_limits" not in plain.to_dict()

    # Cluster-wide cap via create() survives a round-trip.
    manager.create("capped", ["host1"], max_gpu_memory_utilization=0.8)
    assert manager.get("capped").max_gpu_memory_utilization == 0.8

    # Per-accelerator-type map (set via the dataclass / YAML) survives too.
    manager._write_cluster(ClusterDefinition(name="typed", hosts=["host1"], accelerator_memory_limits={"gb10": 0.85, "h200": 0.95}))
    assert manager.get("typed").accelerator_memory_limits == {"gb10": 0.85, "h200": 0.95}


def test_distribution_model_enabled_round_trips(tmp_path: Path):
    """``distribution.model.enabled: false`` survives write → read; True is the omitted default."""
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterDistributionConfig, ModelDistributionPrefs

    manager = ClusterManager(tmp_path)

    # Default (enabled=True) is omitted from the serialized dict.
    prefs = ModelDistributionPrefs()
    assert prefs.enabled is True
    assert prefs.is_default() is True
    assert "enabled" not in prefs.to_dict()

    manager._write_cluster(
        ClusterDefinition(
            name="nodist",
            hosts=["host1"],
            distribution=ClusterDistributionConfig(model=ModelDistributionPrefs(enabled=False)),
        )
    )
    reloaded = manager.get("nodist")
    assert reloaded.distribution.model.enabled is False
    assert reloaded.distribution.model.to_dict() == {"enabled": False}


def test_update_max_gpu_memory_utilization(tmp_path: Path):
    """update() sets and clears the cluster-wide cap."""
    manager = ClusterManager(tmp_path)
    manager.create("c", ["host1"], max_gpu_memory_utilization=0.8)

    manager.update("c", max_gpu_memory_utilization=0.7)
    assert manager.get("c").max_gpu_memory_utilization == 0.7

    manager.update("c", max_gpu_memory_utilization=None)
    assert manager.get("c").max_gpu_memory_utilization is None


def test_mgmt_interface_round_trips(tmp_path: Path):
    """mgmt_interface (issue #275) survives create/update/load; unset is omitted."""
    manager = ClusterManager(tmp_path)

    manager.create("plain", ["host1"])
    plain = manager.get("plain")
    assert plain.mgmt_interface is None
    assert "mgmt_interface" not in plain.to_dict()

    manager.create("pinned", ["a", "b"], mgmt_interface="enP7s7")
    assert manager.get("pinned").mgmt_interface == "enP7s7"
    assert manager.get("pinned").to_dict()["mgmt_interface"] == "enP7s7"

    # An unrelated update leaves the pin intact (_UNSET sentinel).
    manager.update("pinned", topology="switch")
    assert manager.get("pinned").mgmt_interface == "enP7s7"

    manager.update("pinned", mgmt_interface="enP7s8")
    assert manager.get("pinned").mgmt_interface == "enP7s8"

    # Explicit clear restores per-host detection.
    manager.update("pinned", mgmt_interface=None)
    assert manager.get("pinned").mgmt_interface is None
    assert "mgmt_interface" not in manager.get("pinned").to_dict()


def test_mgmt_interface_reaches_resolved_config(tmp_path: Path):
    """It applies whenever the cluster is named — even with explicit --hosts.

    The pinned NIC is a property of the machines, not of how their addresses
    were supplied, so it must not be gated like transfer_mode/topology are.
    """
    from sparkrun.core.cluster_manager import resolve_cluster_config

    manager = ClusterManager(tmp_path)
    manager.create("c", ["a", "b"], mgmt_interface="enP7s7", topology="switch")

    from_cluster = resolve_cluster_config("c", None, None, manager)
    assert from_cluster.mgmt_interface == "enP7s7"
    assert from_cluster.topology == "switch"

    explicit_hosts = resolve_cluster_config("c", "a,b", None, manager)
    assert explicit_hosts.mgmt_interface == "enP7s7"
    assert explicit_hosts.topology is None  # host-list-gated, unlike mgmt_interface


def test_fabric_interfaces_round_trips(tmp_path: Path):
    """fabric_interfaces (issue #203) survives create/update/load; empty is omitted."""
    manager = ClusterManager(tmp_path)

    # Default is an empty list, omitted from the serialized dict.
    manager.create("plain", ["host1"])
    plain = manager.get("plain")
    assert plain.fabric_interfaces == []
    assert "fabric_interfaces" not in plain.to_dict()

    # Set via create() and survive a round-trip.
    manager.create("two", ["a", "b"], fabric_interfaces=["*np1"])
    assert manager.get("two").fabric_interfaces == ["*np1"]
    assert manager.get("two").to_dict()["fabric_interfaces"] == ["*np1"]

    # An unrelated update leaves the saved selection intact (_UNSET sentinel).
    manager.update("two", topology="switch")
    assert manager.get("two").fabric_interfaces == ["*np1"]

    # update() replaces and clears the selection.
    manager.update("two", fabric_interfaces=["*np0"])
    assert manager.get("two").fabric_interfaces == ["*np0"]
    manager.update("two", fabric_interfaces=[])
    assert manager.get("two").fabric_interfaces == []
    assert "fabric_interfaces" not in manager.get("two").to_dict()


def test_env_envfile_syncsource_round_trip(tmp_path: Path):
    """env / env_file / sync_source survive create/update/load; empty omitted."""
    manager = ClusterManager(tmp_path)

    manager.create("plain", ["h1"])
    plain = manager.get("plain")
    assert plain.env == {} and plain.env_file is None and plain.sync_source is None
    for key in ("env", "env_file", "sync_source"):
        assert key not in plain.to_dict()

    manager.create(
        "svd",
        ["a", "b"],
        env={"HF_TOKEN": "${CONTAINER_HF_TOKEN}"},
        env_file="/path/.env",
        sync_source="spark_vllm_docker:/path/.env",
    )
    c = manager.get("svd")
    assert c.env == {"HF_TOKEN": "${CONTAINER_HF_TOKEN}"}
    assert c.env_file == "/path/.env"
    assert c.sync_source == "spark_vllm_docker:/path/.env"

    # Unrelated update preserves the import-owned fields (_UNSET sentinel).
    manager.update("svd", description="hi")
    assert manager.get("svd").env == {"HF_TOKEN": "${CONTAINER_HF_TOKEN}"}

    # update replaces / clears.
    manager.update("svd", env={}, env_file=None, sync_source=None)
    cleared = manager.get("svd")
    assert cleared.env == {} and cleared.env_file is None and cleared.sync_source is None


def test_resolve_env_substitutes_from_env_file(tmp_path: Path):
    """${VAR} resolves from env_file; literals pass through."""
    envf = tmp_path / "svd.env"
    envf.write_text('# c\nCONTAINER_HF_TOKEN="secret123"\nPLAINSRC=nope\n')
    manager = ClusterManager(tmp_path)
    manager.create(
        "svd",
        ["a"],
        env={"HF_TOKEN": "${CONTAINER_HF_TOKEN}", "LIT": "literal"},
        env_file=str(envf),
    )
    assert manager.get("svd").resolve_env() == {"HF_TOKEN": "secret123", "LIT": "literal"}


def test_resolve_env_missing_var_is_hard_error(tmp_path: Path):
    envf = tmp_path / "svd.env"
    envf.write_text("OTHER=1\n")
    manager = ClusterManager(tmp_path)
    manager.create("svd", ["a"], env={"X": "${NOPE}"}, env_file=str(envf))
    with pytest.raises(ClusterError, match=r"\$\{NOPE\}"):
        manager.get("svd").resolve_env()


def test_resolve_env_reference_without_env_file_errors(tmp_path: Path):
    manager = ClusterManager(tmp_path)
    manager.create("svd", ["a"], env={"X": "${Y}"})
    with pytest.raises(ClusterError, match="no env_file"):
        manager.get("svd").resolve_env()


def test_resolve_env_no_refs_needs_no_file(tmp_path: Path):
    """All-literal env resolves without reading any file."""
    manager = ClusterManager(tmp_path)
    manager.create("svd", ["a"], env={"A": "1", "B": "two"})
    assert manager.get("svd").resolve_env() == {"A": "1", "B": "two"}


def test_get_cluster_not_found(tmp_path: Path):
    """Getting a non-existent cluster raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.get("nonexistent")


def test_update_cluster_hosts(tmp_path: Path):
    """Update only hosts, description remains unchanged."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"], description="Original description")

    manager.update("test", hosts=["host2", "host3"])

    cluster = manager.get("test")
    assert cluster.hosts == ["host2", "host3"]
    assert cluster.description == "Original description"


def test_update_cluster_description(tmp_path: Path):
    """Update only description, hosts remain unchanged."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1", "host2"], description="Old description")

    manager.update("test", description="New description")

    cluster = manager.get("test")
    assert cluster.hosts == ["host1", "host2"]
    assert cluster.description == "New description"


def test_update_nonexistent_cluster(tmp_path: Path):
    """Updating a non-existent cluster raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.update("nonexistent", hosts=["host1"])


def test_delete_cluster(tmp_path: Path):
    """Delete a cluster, then getting it raises ClusterError."""
    manager = ClusterManager(tmp_path)
    manager.create("to-delete", ["host1"])

    manager.delete("to-delete")

    with pytest.raises(ClusterError):
        manager.get("to-delete")


def test_delete_nonexistent_cluster(tmp_path: Path):
    """Deleting a non-existent cluster raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.delete("nonexistent")


def test_list_clusters_empty(tmp_path: Path):
    """List clusters when none exist returns empty list."""
    manager = ClusterManager(tmp_path)

    clusters = manager.list_clusters()
    assert clusters == []


def test_list_clusters_multiple(tmp_path: Path):
    """List multiple clusters returns all of them."""
    manager = ClusterManager(tmp_path)
    manager.create("cluster-a", ["host1"])
    manager.create("cluster-b", ["host2"])
    manager.create("cluster-c", ["host3"])

    clusters = manager.list_clusters()
    assert len(clusters) == 3

    names = {c.name for c in clusters}
    assert names == {"cluster-a", "cluster-b", "cluster-c"}


def test_list_clusters_sorted(tmp_path: Path):
    """List clusters returns them sorted alphabetically by name."""
    manager = ClusterManager(tmp_path)
    manager.create("zebra", ["host1"])
    manager.create("alpha", ["host2"])
    manager.create("middle", ["host3"])

    clusters = manager.list_clusters()
    names = [c.name for c in clusters]
    assert names == ["alpha", "middle", "zebra"]


def test_set_default(tmp_path: Path):
    """Set default cluster and verify get_default returns it."""
    manager = ClusterManager(tmp_path)
    manager.create("my-cluster", ["host1"])

    manager.set_default("my-cluster")

    assert manager.get_default() == "my-cluster"


def test_set_default_nonexistent(tmp_path: Path):
    """Setting a non-existent cluster as default raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.set_default("nonexistent")


def test_unset_default(tmp_path: Path):
    """Unsetting default returns None after unset."""
    manager = ClusterManager(tmp_path)
    manager.create("cluster", ["host1"])
    manager.set_default("cluster")

    manager.unset_default()

    assert manager.get_default() is None


def test_unset_default_when_not_set(tmp_path: Path):
    """Unsetting default when not set raises no error."""
    manager = ClusterManager(tmp_path)

    # Should not raise
    manager.unset_default()
    assert manager.get_default() is None


def test_get_default_stale(tmp_path: Path):
    """Delete cluster that was default, get_default returns None."""
    manager = ClusterManager(tmp_path)
    manager.create("cluster", ["host1"])
    manager.set_default("cluster")

    manager.delete("cluster")

    assert manager.get_default() is None


def test_delete_cluster_clears_default(tmp_path: Path):
    """Deleting the default cluster clears the default pointer."""
    manager = ClusterManager(tmp_path)
    manager.create("default-cluster", ["host1"])
    manager.create("other-cluster", ["host2"])
    manager.set_default("default-cluster")

    manager.delete("default-cluster")

    assert manager.get_default() is None


def test_invalid_name_starts_with_dash(tmp_path: Path):
    """Cluster name starting with dash raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.create("-invalid", ["host1"])


def test_invalid_name_special_chars(tmp_path: Path):
    """Cluster name with special characters raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.create("invalid@name", ["host1"])

    with pytest.raises(ClusterError):
        manager.create("invalid.name", ["host1"])

    with pytest.raises(ClusterError):
        manager.create("invalid name", ["host1"])


def test_valid_name_with_underscores_hyphens(tmp_path: Path):
    """Cluster name with valid underscores and hyphens succeeds."""
    manager = ClusterManager(tmp_path)

    # These should all succeed
    manager.create("valid_name", ["host1"])
    manager.create("valid-name", ["host2"])
    manager.create("valid_name-123", ["host3"])
    manager.create("a1_b2-c3", ["host4"])

    clusters = manager.list_clusters()
    assert len(clusters) == 4


# ---------------------------------------------------------------------------
# cache_dir field tests
# ---------------------------------------------------------------------------


def test_create_cluster_with_cache_dir(tmp_path: Path):
    """Create a cluster with cache_dir and verify persistence."""
    manager = ClusterManager(tmp_path)
    manager.create("gpu-lab", ["host1", "host2"], cache_dir="/mnt/models")

    cluster = manager.get("gpu-lab")
    assert cluster.cache_dir == "/mnt/models"
    assert cluster.hosts == ["host1", "host2"]


def test_create_cluster_without_cache_dir(tmp_path: Path):
    """Cluster created without cache_dir defaults to None."""
    manager = ClusterManager(tmp_path)
    manager.create("basic", ["host1"])

    cluster = manager.get("basic")
    assert cluster.cache_dir is None


def test_update_cluster_cache_dir(tmp_path: Path):
    """Update cache_dir without affecting other fields."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"], description="My cluster", user="admin")

    manager.update("test", cache_dir="/data/hf-cache")

    cluster = manager.get("test")
    assert cluster.cache_dir == "/data/hf-cache"
    assert cluster.hosts == ["host1"]
    assert cluster.description == "My cluster"
    assert cluster.user == "admin"


def test_update_cluster_clear_cache_dir(tmp_path: Path):
    """Pass cache_dir=None explicitly to clear it."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"], cache_dir="/mnt/models")

    manager.update("test", cache_dir=None)

    cluster = manager.get("test")
    assert cluster.cache_dir is None


# ---------------------------------------------------------------------------
# transfer_mode field tests
# ---------------------------------------------------------------------------


def test_create_cluster_with_transfer_mode(tmp_path: Path):
    """Create a cluster with transfer_mode and verify persistence."""
    manager = ClusterManager(tmp_path)
    manager.create("remote-lab", ["host1", "host2"], transfer_mode="push")

    cluster = manager.get("remote-lab")
    assert cluster.transfer_mode == "push"
    assert cluster.hosts == ["host1", "host2"]


def test_create_cluster_with_delegated_mode(tmp_path: Path):
    """Create a cluster with delegated transfer_mode."""
    manager = ClusterManager(tmp_path)
    manager.create("delegated-lab", ["host1"], transfer_mode="delegated")

    cluster = manager.get("delegated-lab")
    assert cluster.transfer_mode == "delegated"


def test_create_cluster_without_transfer_mode(tmp_path: Path):
    """Cluster created without transfer_mode defaults to None."""
    manager = ClusterManager(tmp_path)
    manager.create("basic", ["host1"])

    cluster = manager.get("basic")
    assert cluster.transfer_mode is None


def test_create_cluster_invalid_transfer_mode(tmp_path: Path):
    """Invalid transfer_mode raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.create("bad", ["host1"], transfer_mode="invalid")


def test_update_cluster_transfer_mode(tmp_path: Path):
    """Update transfer_mode without affecting other fields."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"], description="My cluster", user="admin")

    manager.update("test", transfer_mode="push")

    cluster = manager.get("test")
    assert cluster.transfer_mode == "push"
    assert cluster.hosts == ["host1"]
    assert cluster.description == "My cluster"
    assert cluster.user == "admin"


def test_update_cluster_clear_transfer_mode(tmp_path: Path):
    """Pass transfer_mode=None explicitly to clear it."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"], transfer_mode="push")

    manager.update("test", transfer_mode=None)

    cluster = manager.get("test")
    assert cluster.transfer_mode is None


def test_update_cluster_invalid_transfer_mode(tmp_path: Path):
    """Updating with invalid transfer_mode raises ClusterError."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"])

    with pytest.raises(ClusterError):
        manager.update("test", transfer_mode="bogus")


def test_create_cluster_with_all_fields(tmp_path: Path):
    """Create a cluster with all optional fields including transfer_mode."""
    manager = ClusterManager(tmp_path)
    manager.create(
        "full",
        ["host1", "host2"],
        description="Full cluster",
        user="admin",
        cache_dir="/mnt/models",
        transfer_mode="delegated",
    )

    cluster = manager.get("full")
    assert cluster.name == "full"
    assert cluster.hosts == ["host1", "host2"]
    assert cluster.description == "Full cluster"
    assert cluster.user == "admin"
    assert cluster.cache_dir == "/mnt/models"
    assert cluster.transfer_mode == "delegated"


def test_valid_transfer_modes_constant():
    """VALID_TRANSFER_MODES contains the expected values."""
    from sparkrun.core.cluster_manager import VALID_TRANSFER_MODES

    assert VALID_TRANSFER_MODES == ("auto", "local", "push", "delegated", "pull")


# ---------------------------------------------------------------------------
# transfer_interface field tests
# ---------------------------------------------------------------------------


def test_create_cluster_with_transfer_interface(tmp_path: Path):
    """Create a cluster with transfer_interface and verify persistence."""
    manager = ClusterManager(tmp_path)
    manager.create("ib-lab", ["host1", "host2"], transfer_interface="cx7")

    cluster = manager.get("ib-lab")
    assert cluster.transfer_interface == "cx7"
    assert cluster.hosts == ["host1", "host2"]


def test_create_cluster_with_mgmt_interface(tmp_path: Path):
    """Create a cluster with mgmt transfer_interface."""
    manager = ClusterManager(tmp_path)
    manager.create("mgmt-lab", ["host1"], transfer_interface="mgmt")

    cluster = manager.get("mgmt-lab")
    assert cluster.transfer_interface == "mgmt"


def test_create_cluster_without_transfer_interface(tmp_path: Path):
    """Cluster created without transfer_interface defaults to None."""
    manager = ClusterManager(tmp_path)
    manager.create("basic", ["host1"])

    cluster = manager.get("basic")
    assert cluster.transfer_interface is None


def test_create_cluster_invalid_transfer_interface(tmp_path: Path):
    """Invalid transfer_interface raises ClusterError."""
    manager = ClusterManager(tmp_path)

    with pytest.raises(ClusterError):
        manager.create("bad", ["host1"], transfer_interface="invalid")


def test_update_cluster_transfer_interface(tmp_path: Path):
    """Update transfer_interface without affecting other fields."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"], description="My cluster", user="admin")

    manager.update("test", transfer_interface="mgmt")

    cluster = manager.get("test")
    assert cluster.transfer_interface == "mgmt"
    assert cluster.hosts == ["host1"]
    assert cluster.description == "My cluster"
    assert cluster.user == "admin"


def test_update_cluster_clear_transfer_interface(tmp_path: Path):
    """Pass transfer_interface=None explicitly to clear it."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"], transfer_interface="cx7")

    manager.update("test", transfer_interface=None)

    cluster = manager.get("test")
    assert cluster.transfer_interface is None


def test_update_cluster_invalid_transfer_interface(tmp_path: Path):
    """Updating with invalid transfer_interface raises ClusterError."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1"])

    with pytest.raises(ClusterError):
        manager.update("test", transfer_interface="bogus")


def test_create_cluster_with_all_fields_including_interface(tmp_path: Path):
    """Create a cluster with all optional fields including transfer_interface."""
    manager = ClusterManager(tmp_path)
    manager.create(
        "full",
        ["host1", "host2"],
        description="Full cluster",
        user="admin",
        cache_dir="/mnt/models",
        transfer_mode="delegated",
        transfer_interface="mgmt",
    )

    cluster = manager.get("full")
    assert cluster.name == "full"
    assert cluster.hosts == ["host1", "host2"]
    assert cluster.description == "Full cluster"
    assert cluster.user == "admin"
    assert cluster.cache_dir == "/mnt/models"
    assert cluster.transfer_mode == "delegated"
    assert cluster.transfer_interface == "mgmt"


def test_valid_transfer_interfaces_constant():
    """VALID_TRANSFER_INTERFACES contains the expected values."""
    from sparkrun.core.cluster_manager import VALID_TRANSFER_INTERFACES

    assert VALID_TRANSFER_INTERFACES == ("cx7", "mgmt")


# ---------------------------------------------------------------------------
# topology field tests
# ---------------------------------------------------------------------------


def test_create_cluster_with_topology(tmp_path: Path):
    """Create a cluster with topology and verify persistence."""
    manager = ClusterManager(tmp_path)
    manager.create("ring-lab", ["host1", "host2", "host3"], topology="ring")

    cluster = manager.get("ring-lab")
    assert cluster.topology == "ring"
    assert cluster.hosts == ["host1", "host2", "host3"]


def test_create_cluster_without_topology(tmp_path: Path):
    """Cluster created without topology defaults to None."""
    manager = ClusterManager(tmp_path)
    manager.create("basic", ["host1"])

    cluster = manager.get("basic")
    assert cluster.topology is None


def test_update_cluster_topology(tmp_path: Path):
    """Update topology without affecting other fields."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1", "host2"], description="My cluster", user="admin")

    manager.update("test", topology="direct")

    cluster = manager.get("test")
    assert cluster.topology == "direct"
    assert cluster.hosts == ["host1", "host2"]
    assert cluster.description == "My cluster"
    assert cluster.user == "admin"


def test_update_cluster_clear_topology(tmp_path: Path):
    """Pass topology=None explicitly to clear it."""
    manager = ClusterManager(tmp_path)
    manager.create("test", ["host1", "host2", "host3"], topology="ring")

    manager.update("test", topology=None)

    cluster = manager.get("test")
    assert cluster.topology is None


def test_create_cluster_with_all_fields_including_topology(tmp_path: Path):
    """Create a cluster with all optional fields including topology."""
    manager = ClusterManager(tmp_path)
    manager.create(
        "full",
        ["host1", "host2", "host3"],
        description="Full ring cluster",
        user="admin",
        cache_dir="/mnt/models",
        transfer_mode="delegated",
        transfer_interface="cx7",
        topology="ring",
    )

    cluster = manager.get("full")
    assert cluster.name == "full"
    assert cluster.hosts == ["host1", "host2", "host3"]
    assert cluster.description == "Full ring cluster"
    assert cluster.user == "admin"
    assert cluster.cache_dir == "/mnt/models"
    assert cluster.transfer_mode == "delegated"
    assert cluster.transfer_interface == "cx7"
    assert cluster.topology == "ring"


def test_existing_yaml_without_topology_loads(tmp_path: Path):
    """YAML without topology field loads with topology=None (backward compat)."""
    import yaml

    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir()
    yaml_file = clusters_dir / "old-cluster.yaml"
    yaml_file.write_text(
        yaml.dump(
            {
                "name": "old-cluster",
                "hosts": ["host1", "host2"],
                "description": "Old cluster without topology",
            }
        )
    )

    manager = ClusterManager(tmp_path)
    cluster = manager.get("old-cluster")
    assert cluster.topology is None
    assert cluster.hosts == ["host1", "host2"]


# ---------------------------------------------------------------------------
# resolve_transfer_config tests
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in for SparkrunConfig with hf_cache_dir."""

    def __init__(self, hf_cache_dir: str = "/home/testuser/.cache/huggingface"):
        self.hf_cache_dir = Path(hf_cache_dir)


class TestResolveTransferConfig:
    """Tests for ResolvedClusterConfig.resolve_transfer_config."""

    def test_explicit_cache_dir_always_used(self):
        """When the cluster defines cache_dir, it's always used as remote_cache_dir."""
        cfg = ResolvedClusterConfig(cache_dir="/mnt/shared/models")
        config = _FakeConfig()
        _, remote, _, _ = cfg.resolve_transfer_config(config)
        assert remote == "/mnt/shared/models"

    def test_no_cluster_cache_dir_returns_none(self):
        """Without an explicit cluster cache_dir, remote resolution is deferred.

        Returning ``None`` signals the caller (typically ``launch_inference``
        via ``resolve_effective_cache_dir``) to probe the target so the path
        reflects the SSH login user's ``$HOME`` / ``HF_HOME`` rather than the
        control machine's.
        """
        cfg = ResolvedClusterConfig()
        config = _FakeConfig()
        _, remote, _, _ = cfg.resolve_transfer_config(config)
        assert remote is None

    def test_cluster_user_alone_does_not_synthesize_remote_path(self):
        """Cluster-defined user is no longer used to synthesize a /home/<user>/...
        remote cache path; resolution defers to the probe."""
        cfg = ResolvedClusterConfig(user="gpuadmin")
        config = _FakeConfig()
        with patch.dict("os.environ", {"USER": "localuser"}):
            _, remote, _, _ = cfg.resolve_transfer_config(config)
        assert remote is None

    def test_local_cache_dir_always_returned(self):
        """Local cache dir is always populated from the config (never deferred)."""
        cfg = ResolvedClusterConfig()
        config = _FakeConfig("/home/drew/.cache/huggingface")
        local, _, _, _ = cfg.resolve_transfer_config(config)
        assert local == "/home/drew/.cache/huggingface"

    def test_transfer_mode_override(self):
        """CLI transfer_mode_override takes precedence over cluster setting."""
        cfg = ResolvedClusterConfig(transfer_mode="push")
        config = _FakeConfig()
        _, _, mode, _ = cfg.resolve_transfer_config(config, transfer_mode_override="delegated")
        assert mode == "delegated"

    def test_transfer_mode_from_cluster(self):
        """Cluster transfer_mode used when no CLI override."""
        cfg = ResolvedClusterConfig(transfer_mode="push")
        config = _FakeConfig()
        _, _, mode, _ = cfg.resolve_transfer_config(config)
        assert mode == "push"

    def test_transfer_mode_defaults_to_auto(self):
        """No override and no cluster setting → defaults to 'auto'."""
        cfg = ResolvedClusterConfig()
        config = _FakeConfig()
        _, _, mode, _ = cfg.resolve_transfer_config(config)
        assert mode == "auto"


# ---------------------------------------------------------------------------
# Monitoring parse tests
# ---------------------------------------------------------------------------


class TestParseMonitorLine:
    """Tests for sparkrun.core.monitoring.parse_monitor_line."""

    def test_parse_valid_line(self):
        """Valid CSV line with all 27 fields parses to MonitorSample."""
        from sparkrun.core.monitoring import parse_monitor_line, MONITOR_COLUMNS

        fields = [
            "2026-03-02T10:00:00Z",
            "spark-01",
            "12345",
            "1.2",
            "0.8",
            "0.5",
            "23.4",
            "2400",
            "55.0",
            "128000",
            "45120",
            "82880",
            "35.2",
            "8192",
            "100",
            "NVIDIA GH200",
            "85.0",
            "60000",
            "131072",
            "45.8",
            "62",
            "180.5",
            "300.0",
            "1500",
            "5001",
            "3",
            "sparkrun_abc|sparkrun_def|sparkrun_ghi",
        ]
        assert len(fields) == len(MONITOR_COLUMNS)
        line = ",".join(fields)
        sample = parse_monitor_line(line)

        assert sample is not None
        assert sample.timestamp == "2026-03-02T10:00:00Z"
        assert sample.hostname == "spark-01"
        assert sample.cpu_usage_pct == "23.4"
        assert sample.mem_total_mb == "128000"
        assert sample.mem_used_mb == "45120"
        assert sample.gpu_name == "NVIDIA GH200"
        assert sample.gpu_util_pct == "85.0"
        assert sample.gpu_temp_c == "62"
        assert sample.gpu_power_w == "180.5"
        assert sample.sparkrun_jobs == "3"
        assert sample.sparkrun_job_names == "sparkrun_abc|sparkrun_def|sparkrun_ghi"

    def test_parse_missing_gpu_fields(self):
        """CSV line with empty GPU fields parses correctly."""
        from sparkrun.core.monitoring import parse_monitor_line, MONITOR_COLUMNS

        fields = [
            "2026-03-02T10:00:00Z",
            "spark-02",
            "12345",
            "0.5",
            "0.3",
            "0.2",
            "5.1",
            "2400",
            "45.0",
            "64000",
            "12000",
            "52000",
            "18.7",
            "4096",
            "0",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "0",
            "",
        ]
        assert len(fields) == len(MONITOR_COLUMNS)
        line = ",".join(fields)
        sample = parse_monitor_line(line)

        assert sample is not None
        assert sample.hostname == "spark-02"
        assert sample.cpu_usage_pct == "5.1"
        assert sample.gpu_name == ""
        assert sample.gpu_util_pct == ""
        assert sample.gpu_temp_c == ""
        assert sample.gpu_power_w == ""

    def test_parse_malformed_too_few_fields(self):
        """CSV line with too few fields returns None."""
        from sparkrun.core.monitoring import parse_monitor_line

        assert parse_monitor_line("a,b,c") is None

    def test_parse_malformed_too_many_fields(self):
        """CSV line with too many fields returns None."""
        from sparkrun.core.monitoring import parse_monitor_line, MONITOR_COLUMNS

        line = ",".join(["x"] * (len(MONITOR_COLUMNS) + 1))
        assert parse_monitor_line(line) is None

    def test_parse_empty_line(self):
        """Empty string returns None."""
        from sparkrun.core.monitoring import parse_monitor_line

        assert parse_monitor_line("") is None
        assert parse_monitor_line("   ") is None

    def test_parse_strips_whitespace(self):
        """Leading/trailing whitespace in fields is stripped."""
        from sparkrun.core.monitoring import parse_monitor_line, MONITOR_COLUMNS

        fields = [" 2026-03-02T10:00:00Z "] + [" val "] * (len(MONITOR_COLUMNS) - 1)
        line = ",".join(fields)
        sample = parse_monitor_line(line)

        assert sample is not None
        assert sample.timestamp == "2026-03-02T10:00:00Z"
        assert sample.hostname == "val"


# ---------------------------------------------------------------------------
# Per-host hardware metadata (Phase 1)
# ---------------------------------------------------------------------------


def test_create_cluster_without_hosts_hardware_defaults_dgx_spark(tmp_path: Path):
    """A cluster created without hosts_hardware returns DGX Spark default per host."""
    manager = ClusterManager(tmp_path)
    manager.create("legacy", ["host1", "host2"])

    cluster = manager.get("legacy")
    assert cluster.hosts_hardware == {}

    hw = cluster.hardware_for("host1")
    assert len(hw.accelerators) == 1
    assert hw.accelerators[0].vendor == "nvidia"
    assert hw.accelerators[0].model == "gb10"
    assert hw.accelerators[0].memory_gb == 121.0


def test_create_cluster_with_hosts_hardware_round_trips(tmp_path: Path):
    """A cluster with explicit per-host hardware round-trips through YAML."""
    manager = ClusterManager(tmp_path)
    hw_spark = HostHardware(
        accelerators=[
            AcceleratorSpec(
                vendor="nvidia",
                model="gb10",
                memory_gb=121.0,
                capabilities=frozenset({"cuda"}),
            )
        ]
    )
    hw_rtx = HostHardware(
        accelerators=[
            AcceleratorSpec(
                vendor="nvidia",
                model="rtx-pro-6000",
                count=2,
                memory_gb=96.0,
                capabilities=frozenset({"cuda"}),
            )
        ],
        notes="workstation",
    )
    manager.create(
        "mixed",
        ["spark-01", "rtx-box"],
        hosts_hardware={"spark-01": hw_spark, "rtx-box": hw_rtx},
    )

    restored = manager.get("mixed")
    assert restored.hardware_for("spark-01") == hw_spark
    assert restored.hardware_for("rtx-box") == hw_rtx


def test_hardware_for_unknown_host_returns_default(tmp_path: Path):
    """Hosts absent from hosts_hardware still get a DGX Spark default."""
    manager = ClusterManager(tmp_path)
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x", memory_gb=192.0)])
    manager.create("partial", ["amd-box", "mystery-host"], hosts_hardware={"amd-box": hw})

    restored = manager.get("partial")
    assert restored.hardware_for("amd-box").accelerators[0].vendor == "amd"
    # Unknown host -> DGX Spark default
    default_hw = restored.hardware_for("mystery-host")
    assert default_hw.accelerators[0].model == "gb10"


def test_update_hosts_hardware(tmp_path: Path):
    """update() with hosts_hardware overwrites the field."""
    manager = ClusterManager(tmp_path)
    manager.create("c", ["h1"])
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="intel", model="gaudi3", memory_gb=128.0)])
    manager.update("c", hosts_hardware={"h1": hw})

    restored = manager.get("c")
    assert restored.hardware_for("h1") == hw


def test_to_dict_includes_hosts_hardware_only_when_set(tmp_path: Path):
    """to_dict omits hosts_hardware when empty (back-compat with old consumers)."""
    manager = ClusterManager(tmp_path)
    manager.create("legacy", ["h1"])
    assert "hosts_hardware" not in manager.get("legacy").to_dict()

    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200", count=8, memory_gb=141.0)])
    manager.update("legacy", hosts_hardware={"h1": hw})
    assert "hosts_hardware" in manager.get("legacy").to_dict()


# --------------------------------------------------------------------------
# classify_cluster_status — snapshot → ClusterStatusResult classification
# --------------------------------------------------------------------------


class TestQueryClusterStatusParsing:
    """Regression coverage for cluster_id extraction in classify_cluster_status.

    Container names follow ``sparkrun_<intent>_<placement>[_<role>]``.  The
    cluster_id is the full ``sparkrun_<intent>_<placement>`` — workloads
    with the same intent but different placement tokens (the same recipe
    replayed back to back) are distinct jobs and must not collapse on the
    intent prefix.
    """

    @staticmethod
    def _cd(name, role, status="Up 1 minute", image="img"):
        from sparkrun.core.cluster_status import ContainerDetail

        return ContainerDetail(name=name, role=role, status=status, image=image)

    @classmethod
    def _wl(cls, cluster_id, containers):
        from sparkrun.core.cluster_status import RunningWorkload

        return RunningWorkload(cluster_id=cluster_id, containers=tuple(containers))

    @staticmethod
    def _status(host_workloads, errors=None):
        """Build a canned merged :class:`ClusterStatus` snapshot.

        ``host_workloads`` maps host → list of RunningWorkloads (each host
        present is *reachable*).  ``errors`` maps unreachable host → message
        (those hosts are absent from ``hosts``).
        """
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

        hosts = tuple(HostOccupancy(host=h, workloads=tuple(ws)) for h, ws in host_workloads.items())
        return ClusterStatus(hosts=hosts, executor="docker", errors=dict(errors or {}))

    def test_two_workloads_same_intent_distinct_placements_are_separate_clusters(self, tmp_path):
        """Same recipe launched twice → same intent_id, different placement_tokens
        → must report as two distinct cluster_ids."""
        from sparkrun.core.cluster_manager import classify_cluster_status

        intent = "221f3a3a45d7fa4d"
        place_a = "aabbccddeeff"
        place_b = "112233445566"
        cid_a = "sparkrun_%s_%s" % (intent, place_a)
        cid_b = "sparkrun_%s_%s" % (intent, place_b)
        # Workload A on h1+h2, workload B on h3+h4 — both share the same intent prefix.
        status = self._status(
            {
                "h1": [self._wl(cid_a, [self._cd("%s_head" % cid_a, "head")])],
                "h2": [self._wl(cid_a, [self._cd("%s_node_1" % cid_a, "node_1")])],
                "h3": [self._wl(cid_b, [self._cd("%s_head" % cid_b, "head", "Up 30 seconds")])],
                "h4": [self._wl(cid_b, [self._cd("%s_node_1" % cid_b, "node_1", "Up 30 seconds")])],
            }
        )

        result = classify_cluster_status(status, cache_dir=str(tmp_path), host_list=["h1", "h2", "h3", "h4"])

        assert set(result.groups.keys()) == {cid_a, cid_b}
        # Each cluster has exactly two members (one per host).
        assert len(result.groups[cid_a].members) == 2
        assert len(result.groups[cid_b].members) == 2
        # Roles come straight from ContainerDetail.role (head + node_1).
        assert sorted(m[1] for m in result.groups[cid_a].members) == ["head", "node_1"]
        assert sorted(m[1] for m in result.groups[cid_b].members) == ["head", "node_1"]
        assert result.total_containers == 4

    def test_single_workload_parses_full_cluster_id(self, tmp_path):
        """A single 2-node workload must surface as exactly one cluster_id with full placement token."""
        from sparkrun.core.cluster_manager import classify_cluster_status

        intent = "deadbeefcafe1234"
        place = "0123456789ab"
        cid = "sparkrun_%s_%s" % (intent, place)
        status = self._status(
            {
                "h1": [self._wl(cid, [self._cd("%s_head" % cid, "head", "Up 5 minutes")])],
                "h2": [self._wl(cid, [self._cd("%s_node_1" % cid, "node_1", "Up 5 minutes")])],
            }
        )

        result = classify_cluster_status(status, cache_dir=str(tmp_path), host_list=["h1", "h2"])

        assert list(result.groups.keys()) == [cid]
        assert len(result.groups[cid].members) == 2

    def test_solo_container_uses_full_cluster_id(self, tmp_path):
        """Solo containers (`..._solo`) must yield the full cluster_id, not the intent prefix."""
        from sparkrun.core.cluster_manager import classify_cluster_status

        intent = "feedfacef00d4242"
        place = "abcdef012345"
        cid = "sparkrun_%s_%s" % (intent, place)
        status = self._status({"h1": [self._wl(cid, [self._cd("%s_solo" % cid, "solo", "Up 10 seconds")])]})

        result = classify_cluster_status(status, cache_dir=str(tmp_path), host_list=["h1"])

        assert result.groups == {}
        assert len(result.solo_entries) == 1
        assert result.solo_entries[0].cluster_id == cid

    def test_local_executor_pidfile_workload_is_surfaced(self, tmp_path):
        """A docker container + a native local process for DIFFERENT cluster_ids both
        surface as solo entries (the local one carries the ``(local process)`` image)."""
        from sparkrun.core.cluster_manager import classify_cluster_status

        intent, place = "0123456789abcdef", "fedcba987654"
        cid_docker = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
        cid_local = "sparkrun_%s_%s" % (intent, place)
        status = self._status(
            {
                "h1": [
                    self._wl(cid_docker, [self._cd("%s_solo" % cid_docker, "solo", "Up 5 seconds", "some/img")]),
                    self._wl(cid_local, [self._cd("%s_solo" % cid_local, "solo", "Up (pid 4242)", "(local process)")]),
                ]
            }
        )

        result = classify_cluster_status(status, cache_dir=str(tmp_path), host_list=["h1"])

        cids = {e.cluster_id for e in result.solo_entries}
        assert cid_docker in cids and cid_local in cids  # both distinct jobs visible
        local = next(e for e in result.solo_entries if e.cluster_id == cid_local)
        assert "pid 4242" in local.status and local.image == "(local process)"
        assert result.total_containers == 2

    def test_unreachable_host_is_errored_not_idle(self, tmp_path):
        """A host the executor could not reach (recorded in ``ClusterStatus.errors``,
        absent from ``hosts``) surfaces as an error and NOT as idle."""
        from sparkrun.core.cluster_manager import classify_cluster_status

        status = self._status({}, errors={"h1": "connection refused"})

        result = classify_cluster_status(status, cache_dir=str(tmp_path), host_list=["h1"])

        assert result.errors.get("h1") == "connection refused"  # surfaced as an error
        assert "h1" not in result.idle_hosts  # NOT misreported as idle
        assert result.total_containers == 0

    def test_reachable_host_with_no_workloads_is_idle(self, tmp_path):
        """A reachable host (present in the snapshot) with zero containers
        is idle — not an error."""
        from sparkrun.core.cluster_manager import classify_cluster_status

        status = self._status({"h1": []})

        result = classify_cluster_status(status, cache_dir=str(tmp_path), host_list=["h1"])

        assert result.idle_hosts == ["h1"]
        assert "h1" not in result.errors
        assert result.total_containers == 0


class TestPendingOpAttribution:
    """Pending operations are attributed to the hosts they will occupy.

    A host staging a multi-GB image or model download is minutes from taking
    its whole GPU; reporting it as "idle" is what sends the next launch at it.
    """

    @staticmethod
    def _status(hosts, errors=None):
        """Snapshot where every host in *hosts* is reachable and empty."""
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

        return ClusterStatus(
            hosts=tuple(HostOccupancy(host=h, workloads=()) for h in hosts),
            executor="docker",
            errors=dict(errors or {}),
        )

    @staticmethod
    def _lock(tmp_path, key, hosts, **kw):
        from sparkrun.core.pending_ops import create_pending_op

        create_pending_op(key, "image_distribute", hosts=hosts, cache_dir=str(tmp_path), **kw)

    def _classify(self, tmp_path, host_list, snapshot=None):
        from sparkrun.core.cluster_manager import classify_cluster_status

        return classify_cluster_status(
            snapshot if snapshot is not None else self._status(host_list),
            cache_dir=str(tmp_path),
            host_list=host_list,
        )

    def test_targeted_host_is_preparing_not_idle(self, tmp_path):
        """The op's target is 'preparing'; its peer stays idle."""
        self._lock(tmp_path, "k1", ["h1"], recipe="r1")

        result = self._classify(tmp_path, ["h1", "h2"])

        assert result.preparing_hosts == ["h1"]
        assert result.idle_hosts == ["h2"]  # h1 is NOT reported idle
        assert [op["recipe"] for op in result.pending_by_host["h1"]] == ["r1"]

    def test_op_without_hosts_is_shown_but_pins_nothing(self, tmp_path):
        """ "Unknown scope" must not read as "affects every host"."""
        self._lock(tmp_path, "k1", [], recipe="r1")

        result = self._classify(tmp_path, ["h1", "h2"])

        assert result.preparing_hosts == []
        assert result.idle_hosts == ["h1", "h2"]
        assert len(result.pending_ops) == 1  # still surfaced
        assert result.pending_ops[0]["matched_hosts"] == []

    def test_op_for_other_hosts_is_dropped(self, tmp_path):
        """An op targeting a different cluster entirely is not this report's."""
        self._lock(tmp_path, "k1", ["h9"], recipe="elsewhere")

        result = self._classify(tmp_path, ["h1", "h2"])

        assert result.pending_ops == []
        assert result.idle_hosts == ["h1", "h2"]

    def test_spanning_op_records_hosts_outside_the_cluster(self, tmp_path):
        """An op reaching beyond the queried hosts says so."""
        self._lock(tmp_path, "k1", ["h1", "h9"], recipe="r1")

        result = self._classify(tmp_path, ["h1", "h2"])

        (op,) = result.pending_ops
        assert op["matched_hosts"] == ["h1"]
        assert op["other_hosts"] == ["h9"]
        assert result.preparing_hosts == ["h1"]

    def test_host_spelling_is_normalized(self, tmp_path):
        """A lock recorded as ``user@Host`` still matches ``host``."""
        self._lock(tmp_path, "k1", ["drew@H1"], recipe="r1")

        result = self._classify(tmp_path, ["h1"])

        assert result.preparing_hosts == ["h1"]  # reported in the queried spelling
        assert result.pending_ops[0]["matched_hosts"] == ["h1"]

    def test_running_host_is_not_listed_as_preparing(self, tmp_path):
        """A host already running the workload is occupied, not preparing."""
        from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload

        self._lock(tmp_path, "k1", ["h1"], recipe="r1")
        cid = "sparkrun_221f3a3a45d7fa4d_111111111111"
        snapshot = ClusterStatus(
            hosts=(
                HostOccupancy(
                    host="h1",
                    workloads=(RunningWorkload(cluster_id=cid, containers=(ContainerDetail(cid + "_solo", "solo", "Up 1m", "img"),)),),
                ),
            ),
            executor="docker",
        )

        result = self._classify(tmp_path, ["h1"], snapshot=snapshot)

        assert result.preparing_hosts == []
        assert result.idle_hosts == []
        assert result.total_containers == 1
        assert len(result.pending_ops) == 1  # the op is still reported

    def test_unreachable_host_is_not_preparing(self, tmp_path):
        """An errored host is unknown, not "being prepared"."""
        self._lock(tmp_path, "k1", ["h1"], recipe="r1")

        result = self._classify(tmp_path, ["h1"], snapshot=self._status([], errors={"h1": "unreachable"}))

        assert result.preparing_hosts == []
        assert result.idle_hosts == []
        assert result.errors == {"h1": "unreachable"}

    def test_job_identity_is_recorded_on_the_lock(self, tmp_path):
        """The launch's cluster_id / cluster name travel with the op."""
        self._lock(tmp_path, "k1", ["h1"], recipe="r1", job_cluster_id="sparkrun_aaaa_bbbb", cluster="mylab")

        result = self._classify(tmp_path, ["h1"])

        (op,) = result.pending_ops
        assert op["job_cluster_id"] == "sparkrun_aaaa_bbbb"
        assert op["cluster"] == "mylab"

    def test_to_dict_exposes_preparing_hosts(self, tmp_path):
        """JSON consumers get the new split without losing ``idle_hosts``."""
        self._lock(tmp_path, "k1", ["h1"], recipe="r1")

        out = self._classify(tmp_path, ["h1", "h2"]).to_dict()

        assert out["preparing_hosts"] == ["h1"]
        assert out["idle_hosts"] == ["h2"]
        assert out["pending_ops"][0]["matched_hosts"] == ["h1"]


class TestClusterDistributionPrefs:
    """Round-trip + resolution for cluster ``distribution.model`` prefs."""

    def test_defaults_omitted_from_yaml(self, tmp_path: Path):
        """A cluster with default prefs writes no ``distribution`` block."""
        manager = ClusterManager(tmp_path)
        manager.create("c", ["h1"])
        text = (tmp_path / "clusters" / "c.yaml").read_text()
        assert "distribution" not in text
        # Defaults still materialize on read.
        cluster = manager.get("c")
        assert cluster.distribution.model.preserve_perms is True
        assert cluster.distribution.model.skip_fan_out is False

    def test_create_persists_non_default_prefs(self, tmp_path: Path):
        from sparkrun.core.cluster_manager import (
            ClusterDistributionConfig,
            ModelDistributionPrefs,
        )

        manager = ClusterManager(tmp_path)
        manager.create(
            "shared",
            ["h1", "h2"],
            distribution=ClusterDistributionConfig(model=ModelDistributionPrefs(preserve_perms=False, skip_fan_out=True)),
        )
        text = (tmp_path / "clusters" / "shared.yaml").read_text()
        assert "distribution" in text

        cluster = manager.get("shared")
        assert cluster.distribution.model.preserve_perms is False
        assert cluster.distribution.model.skip_fan_out is True

    def test_update_prefs(self, tmp_path: Path):
        from sparkrun.core.cluster_manager import (
            ClusterDistributionConfig,
            ModelDistributionPrefs,
        )

        manager = ClusterManager(tmp_path)
        manager.create("c", ["h1"])
        manager.update(
            "c",
            distribution=ClusterDistributionConfig(model=ModelDistributionPrefs(skip_fan_out=True)),
        )
        cluster = manager.get("c")
        assert cluster.distribution.model.skip_fan_out is True
        assert cluster.distribution.model.preserve_perms is True

    def test_legacy_yaml_without_distribution_loads(self, tmp_path: Path):
        """A cluster file with no ``distribution`` key loads with defaults."""
        clusters_dir = tmp_path / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)
        (clusters_dir / "old.yaml").write_text("name: old\nhosts:\n- h1\ndescription: legacy\n")
        manager = ClusterManager(tmp_path)
        cluster = manager.get("old")
        assert cluster.distribution.model.preserve_perms is True
        assert cluster.distribution.model.skip_fan_out is False

    def test_resolved_config_carries_prefs_from_cluster(self, tmp_path: Path):
        from sparkrun.core.cluster_manager import (
            ClusterDistributionConfig,
            ModelDistributionPrefs,
            resolve_cluster_config,
        )

        manager = ClusterManager(tmp_path)
        manager.create(
            "shared",
            ["h1", "h2"],
            distribution=ClusterDistributionConfig(model=ModelDistributionPrefs(preserve_perms=False, skip_fan_out=True)),
        )
        # Hosts sourced from the cluster → prefs apply.
        cfg = resolve_cluster_config("shared", None, None, manager)
        assert cfg.preserve_model_perms is False
        assert cfg.skip_model_fan_out is True

    def test_resolved_config_ignores_prefs_with_explicit_hosts(self, tmp_path: Path):
        """Explicit --hosts bypasses cluster transfer prefs (matches cache_dir)."""
        from sparkrun.core.cluster_manager import (
            ClusterDistributionConfig,
            ModelDistributionPrefs,
            resolve_cluster_config,
        )

        manager = ClusterManager(tmp_path)
        manager.create(
            "shared",
            ["h1", "h2"],
            distribution=ClusterDistributionConfig(model=ModelDistributionPrefs(preserve_perms=False, skip_fan_out=True)),
        )
        cfg = resolve_cluster_config("shared", "h9,h10", None, manager)
        assert cfg.preserve_model_perms is True
        assert cfg.skip_model_fan_out is False
