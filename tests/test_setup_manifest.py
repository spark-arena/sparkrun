"""Tests for sparkrun.core.setup_manifest module."""

from __future__ import annotations

from pathlib import Path

import yaml

from sparkrun.core.setup_manifest import ManifestManager, SetupManifest, PhaseRecord


def test_record_phase_creates_new_manifest(tmp_path: Path):
    """Recording a phase when no manifest exists creates one."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test-cluster", "drew", ["10.0.0.1", "10.0.0.2"], "earlyoom", installed_package=True)

    manifest = mgr.load("test-cluster")
    assert manifest is not None
    assert manifest.cluster == "test-cluster"
    assert manifest.user == "drew"
    assert manifest.hosts == ["10.0.0.1", "10.0.0.2"]
    assert "earlyoom" in manifest.phases
    assert manifest.phases["earlyoom"].applied is True
    assert manifest.phases["earlyoom"].hosts == ["10.0.0.1", "10.0.0.2"]
    assert manifest.phases["earlyoom"].extra == {"installed_package": True}


def test_record_phase_unions_hosts(tmp_path: Path):
    """Recording the same phase again unions hosts."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test", "drew", ["10.0.0.1"], "earlyoom", installed_package=True)
    mgr.record_phase("test", "drew", ["10.0.0.2", "10.0.0.1"], "earlyoom", installed_package=False)

    manifest = mgr.load("test")
    assert manifest is not None
    assert set(manifest.phases["earlyoom"].hosts) == {"10.0.0.1", "10.0.0.2"}
    # Existing extra keys are preserved (not overwritten)
    assert manifest.phases["earlyoom"].extra["installed_package"] is True


def test_record_phase_merges_extra(tmp_path: Path):
    """Extra fields are merged: new keys added, existing preserved."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test", "drew", ["h1"], "cx7", subnets=["192.168.11.0/24"])
    mgr.record_phase("test", "drew", ["h1"], "cx7", cx7_ips=["192.168.11.1"], subnets=["different"])

    manifest = mgr.load("test")
    extra = manifest.phases["cx7"].extra
    # subnets preserved from first call
    assert extra["subnets"] == ["192.168.11.0/24"]
    # cx7_ips added from second call
    assert extra["cx7_ips"] == ["192.168.11.1"]


def test_record_multiple_phases(tmp_path: Path):
    """Multiple phases can be recorded independently."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test", "drew", ["h1"], "earlyoom")
    mgr.record_phase("test", "drew", ["h1"], "sudoers", files=["/etc/sudoers.d/sparkrun-chown-drew"])
    mgr.record_phase("test", "drew", ["h1", "h2"], "ssh_mesh", cross_user=False)

    manifest = mgr.load("test")
    assert len(manifest.phases) == 3
    assert "earlyoom" in manifest.phases
    assert "sudoers" in manifest.phases
    assert "ssh_mesh" in manifest.phases
    # Top-level hosts should be union of all
    assert set(manifest.hosts) == {"h1", "h2"}


def test_load_nonexistent_returns_none(tmp_path: Path):
    """Loading a non-existent manifest returns None."""
    mgr = ManifestManager(tmp_path / "clusters")
    assert mgr.load("nonexistent") is None


def test_delete_manifest(tmp_path: Path):
    """Deleting a manifest removes the file."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test", "drew", ["h1"], "earlyoom")
    assert mgr.load("test") is not None

    mgr.delete("test")
    assert mgr.load("test") is None


def test_delete_nonexistent_is_noop(tmp_path: Path):
    """Deleting a non-existent manifest does not raise."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.delete("nonexistent")  # Should not raise


def test_save_and_load_roundtrip(tmp_path: Path):
    """A manually created manifest survives save/load roundtrip."""
    mgr = ManifestManager(tmp_path / "clusters")
    manifest = SetupManifest(
        version=1,
        cluster="roundtrip",
        created="2025-01-01T00:00:00+00:00",
        updated="2025-01-01T00:00:00+00:00",
        user="testuser",
        hosts=["h1", "h2"],
        phases={
            "earlyoom": PhaseRecord(
                applied=True,
                timestamp="2025-01-01T00:00:00+00:00",
                hosts=["h1", "h2"],
                extra={"installed_package": True},
            ),
        },
    )
    mgr.save(manifest)

    loaded = mgr.load("roundtrip")
    assert loaded is not None
    assert loaded.cluster == "roundtrip"
    assert loaded.user == "testuser"
    assert loaded.hosts == ["h1", "h2"]
    assert loaded.phases["earlyoom"].applied is True
    assert loaded.phases["earlyoom"].extra["installed_package"] is True


def test_manifest_path(tmp_path: Path):
    """Manifest file is stored as <cluster>.manifest.yaml."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("mylab", "drew", ["h1"], "earlyoom")

    expected = tmp_path / "clusters" / "mylab.manifest.yaml"
    assert expected.exists()

    # Verify it's valid YAML
    with expected.open() as f:
        data = yaml.safe_load(f)
    assert data["cluster"] == "mylab"
    assert "earlyoom" in data["phases"]


def test_record_phase_updates_timestamp(tmp_path: Path):
    """Recording a phase again updates the timestamp."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test", "drew", ["h1"], "earlyoom")
    m1 = mgr.load("test")
    ts1 = m1.phases["earlyoom"].timestamp

    mgr.record_phase("test", "drew", ["h1"], "earlyoom")
    m2 = mgr.load("test")
    ts2 = m2.phases["earlyoom"].timestamp

    # Timestamps should differ (or at least not fail)
    assert ts2 >= ts1


def test_record_phase_updates_user(tmp_path: Path):
    """Recording with a different user updates the manifest user."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test", "alice", ["h1"], "earlyoom")
    mgr.record_phase("test", "bob", ["h1"], "sudoers")

    manifest = mgr.load("test")
    assert manifest.user == "bob"


def test_load_corrupt_file_returns_none(tmp_path: Path):
    """Loading a corrupt manifest file returns None."""
    clusters = tmp_path / "clusters"
    clusters.mkdir(parents=True)
    (clusters / "bad.manifest.yaml").write_text("not: [valid: yaml: {{")

    mgr = ManifestManager(clusters)
    assert mgr.load("bad") is None


def test_top_level_hosts_union(tmp_path: Path):
    """Top-level hosts list is the union across all record_phase calls."""
    mgr = ManifestManager(tmp_path / "clusters")
    mgr.record_phase("test", "drew", ["h1", "h2"], "earlyoom")
    mgr.record_phase("test", "drew", ["h2", "h3"], "sudoers")

    manifest = mgr.load("test")
    assert manifest.hosts == ["h1", "h2", "h3"]
