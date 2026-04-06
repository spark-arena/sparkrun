"""Tests for sparkrun.cluster_manager and sparkrun.core.monitoring modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sparkrun.core.cluster_manager import ClusterManager, ClusterError, ResolvedClusterConfig


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

    assert VALID_TRANSFER_MODES == ("auto", "local", "push", "delegated")


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

    def test_cross_user_derives_remote_from_ssh_user(self):
        """Different SSH user → remote cache derived from that user's home."""
        cfg = ResolvedClusterConfig(user="gpuadmin")
        config = _FakeConfig()
        with patch.dict("os.environ", {"USER": "localuser"}):
            _, remote, _, _ = cfg.resolve_transfer_config(config)
        assert remote == "/home/gpuadmin/.cache/huggingface"

    def test_non_linux_control_uses_linux_path(self):
        """macOS (or other non-Linux) control machine → remote path is Linux-safe."""
        cfg = ResolvedClusterConfig()
        config = _FakeConfig("/Users/drew/.cache/huggingface")
        with patch("sys.platform", "darwin"), patch.dict("os.environ", {"USER": "drew"}):
            _, remote, _, _ = cfg.resolve_transfer_config(config)
        assert remote == "/home/drew/.cache/huggingface"

    def test_non_linux_control_with_ssh_user(self):
        """macOS control + cluster SSH user → remote path uses SSH user."""
        cfg = ResolvedClusterConfig(user="drew")
        config = _FakeConfig("/Users/drew/.cache/huggingface")
        # Same username on both sides, so cross-user branch doesn't fire
        with patch("sys.platform", "darwin"), patch.dict("os.environ", {"USER": "drew"}):
            _, remote, _, _ = cfg.resolve_transfer_config(config)
        assert remote == "/home/drew/.cache/huggingface"

    def test_linux_control_uses_local_path(self):
        """Linux control machine → remote path matches local (same platform)."""
        cfg = ResolvedClusterConfig()
        config = _FakeConfig("/home/drew/.cache/huggingface")
        with patch("sys.platform", "linux"), patch.dict("os.environ", {"USER": "drew"}):
            _, remote, _, _ = cfg.resolve_transfer_config(config)
        assert remote == "/home/drew/.cache/huggingface"

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
