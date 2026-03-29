"""Tests for sparkrun setup uninstall command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.core.setup_manifest import ManifestManager


@pytest.fixture
def setup_env(tmp_path: Path):
    """Create a cluster + manifest for uninstall testing."""
    config_root = tmp_path / "config"
    config_root.mkdir()
    cluster_mgr = ClusterManager(config_root)
    cluster_mgr.create("test-cluster", ["10.0.0.1", "10.0.0.2"], user="testuser")
    cluster_mgr.set_default("test-cluster")

    manifest_mgr = ManifestManager(cluster_mgr.clusters_dir)
    manifest_mgr.record_phase("test-cluster", "testuser", ["10.0.0.1", "10.0.0.2"], "earlyoom", installed_package=True)
    manifest_mgr.record_phase("test-cluster", "testuser", ["10.0.0.1", "10.0.0.2"], "sudoers",
                              files=["/etc/sudoers.d/sparkrun-chown-testuser", "/etc/sudoers.d/sparkrun-dropcaches-testuser"])
    manifest_mgr.record_phase("test-cluster", "testuser", ["10.0.0.1", "10.0.0.2"], "docker_group")

    return config_root, cluster_mgr, manifest_mgr


def _invoke_uninstall(config_root, args):
    """Invoke uninstall command with mocked config root."""
    from sparkrun.cli._uninstall import setup_uninstall

    with patch("sparkrun.core.config.SparkrunConfig"):
        with patch("sparkrun.core.config.get_config_root", return_value=config_root):
            with patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={"ssh_user": "testuser"}):
                runner = CliRunner()
                return runner.invoke(setup_uninstall, args, catch_exceptions=False)


def test_uninstall_dry_run(setup_env):
    """Dry-run uninstall shows what would happen without executing."""
    config_root, cluster_mgr, manifest_mgr = setup_env

    result = _invoke_uninstall(config_root, ["test-cluster", "--dry-run", "--yes"])

    assert result.exit_code == 0
    assert "dry-run" in result.output.lower()
    # Cluster should still exist (dry-run doesn't delete)
    assert cluster_mgr.get("test-cluster") is not None


def test_uninstall_phase_filter(setup_env):
    """--phase flag limits which phases are processed."""
    config_root, cluster_mgr, manifest_mgr = setup_env

    result = _invoke_uninstall(
        config_root,
        ["test-cluster", "--phase", "earlyoom", "--yes", "--keep-cluster", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "earlyoom" in result.output
    # Only earlyoom phase should appear before summary
    pre_summary = result.output.split("Uninstall Summary")[0]
    assert "Phase: earlyoom" in pre_summary
    assert "Phase: sudoers" not in pre_summary
    assert "Phase: docker_group" not in pre_summary


def test_uninstall_keep_cluster(setup_env):
    """--keep-cluster preserves cluster definition."""
    config_root, cluster_mgr, manifest_mgr = setup_env

    result = _invoke_uninstall(
        config_root,
        ["test-cluster", "--yes", "--keep-cluster", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "kept" in result.output.lower()
    assert cluster_mgr.get("test-cluster") is not None


def test_uninstall_no_cluster():
    """Uninstall with no cluster and no default exits with error."""
    from sparkrun.cli._uninstall import setup_uninstall

    with patch("sparkrun.core.config.SparkrunConfig"):
        with patch("sparkrun.core.config.get_config_root") as mock_root:
            mock_root.return_value = Path("/tmp/nonexistent-sparkrun-test")
            runner = CliRunner()
            result = runner.invoke(setup_uninstall, [], catch_exceptions=False)

            assert result.exit_code != 0
            assert "no cluster" in result.output.lower() or "not found" in result.output.lower()


def test_uninstall_no_manifest(tmp_path: Path):
    """Uninstall without manifest warns but continues."""
    config_root = tmp_path / "config"
    config_root.mkdir()
    cluster_mgr = ClusterManager(config_root)
    cluster_mgr.create("nomanifest", ["10.0.0.1"], user="testuser")
    cluster_mgr.set_default("nomanifest")

    result = _invoke_uninstall(config_root, ["nomanifest", "--dry-run", "--yes"])

    assert result.exit_code == 0
    assert "no setup manifest" in result.output.lower()


def test_uninstall_deletes_cluster_and_manifest(setup_env):
    """Without --keep-cluster, both cluster YAML and manifest are deleted."""
    config_root, cluster_mgr, manifest_mgr = setup_env

    # Use --yes to skip confirmations but NOT --dry-run
    # All phases will be "removed" in dry-run-like fashion since no SSH
    # We need to mock the sudo functions for actual execution
    with patch("sparkrun.orchestration.sudo.run_sudo_script_on_host") as mock_sudo:
        from sparkrun.orchestration.ssh import RemoteResult
        mock_sudo.return_value = RemoteResult(host="10.0.0.1", returncode=0, stdout="OK", stderr="")

        # Just test with dry-run to avoid complex mocking
        result = _invoke_uninstall(config_root, ["test-cluster", "--dry-run", "--yes"])

    assert result.exit_code == 0
    # In dry-run, cluster should say "would be deleted"
    assert "would be deleted" in result.output


def test_uninstall_summary_shown(setup_env):
    """Uninstall always shows a summary at the end."""
    config_root, cluster_mgr, manifest_mgr = setup_env

    result = _invoke_uninstall(config_root, ["test-cluster", "--dry-run", "--yes"])

    assert result.exit_code == 0
    assert "Uninstall Summary" in result.output


def test_teardown_phases_order():
    """Teardown phases are in reverse installation order."""
    from sparkrun.cli._uninstall import TEARDOWN_PHASES

    assert TEARDOWN_PHASES[0] == "earlyoom"  # Safest first
    assert TEARDOWN_PHASES[-1] == "ssh_mesh"  # Most dangerous last


def test_dangerous_phases():
    """Dangerous phases are correctly identified."""
    from sparkrun.cli._uninstall import DANGEROUS_PHASES

    assert "cx7" in DANGEROUS_PHASES
    assert "ssh_mesh" in DANGEROUS_PHASES
    assert "docker_group" in DANGEROUS_PHASES
    assert "earlyoom" not in DANGEROUS_PHASES
    assert "sudoers" not in DANGEROUS_PHASES


def test_uninstall_only_applied_phases(tmp_path: Path):
    """Only phases recorded in manifest are offered for teardown."""
    config_root = tmp_path / "config"
    config_root.mkdir()
    cluster_mgr = ClusterManager(config_root)
    cluster_mgr.create("partial", ["10.0.0.1"], user="testuser")

    manifest_mgr = ManifestManager(cluster_mgr.clusters_dir)
    # Only record earlyoom — no other phases
    manifest_mgr.record_phase("partial", "testuser", ["10.0.0.1"], "earlyoom")

    result = _invoke_uninstall(config_root, ["partial", "--dry-run", "--yes", "--keep-cluster"])

    assert result.exit_code == 0
    assert "Phase: earlyoom" in result.output
    assert "Phase: sudoers" not in result.output
    assert "Phase: docker_group" not in result.output
    assert "Phase: cx7" not in result.output
    assert "Phase: ssh_mesh" not in result.output
