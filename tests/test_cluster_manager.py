"""Tests for sparkrun.cluster_manager module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sparkrun.core.cluster_manager import ClusterManager, ClusterError


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
