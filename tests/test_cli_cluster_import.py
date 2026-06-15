"""CLI tests for `sparkrun cluster import --from-spark-vllm-docker-env`.

Relies on the autouse ``isolate_stateful`` fixture (conftest) so the cluster
manager writes under tmp, not the real ~/.config/sparkrun.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.cli._common import _get_cluster_manager


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Point cluster storage at a per-test temp dir (CLI falls back to
    DEFAULT_CONFIG_DIR), so tests never touch the real ~/.config/sparkrun."""
    import sparkrun.core.config

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", cfg)


def _env(tmp_path, name="cluster.env", nodes="10.0.0.1,10.0.0.2"):
    p = tmp_path / name
    p.write_text("CLUSTER_NODES=%s\nETH_IF=enp1s0f1np1\nCONTAINER_HF_TOKEN=x\n" % nodes)
    return p


def _runner():
    # click >= 8.2 keeps stdout/stderr separate by default.
    return CliRunner()


def test_import_creates_cluster_and_prints_name(tmp_path):
    envf = _env(tmp_path)
    r = _runner().invoke(main, ["cluster", "import", "--from-spark-vllm-docker-env", str(envf)])
    assert r.exit_code == 0, r.stderr
    # stdout is exactly the resolved cluster name.
    assert r.stdout.strip() == "cluster"
    # carried report on stderr (stdout stays clean for tooling).
    assert "carried" in r.stderr

    mgr: ClusterManager = _get_cluster_manager()
    c = mgr.get("cluster")
    assert c.hosts == ["10.0.0.1", "10.0.0.2"]
    assert c.fabric_interfaces == ["*np1"]
    assert c.env == {"HF_TOKEN": "${CONTAINER_HF_TOKEN}"}
    assert c.sync_source == "spark_vllm_docker:%s" % envf.resolve()


def test_reimport_is_rename_safe_idempotent_sync(tmp_path):
    envf = _env(tmp_path)
    runner = _runner()
    # First import under a custom name.
    r1 = runner.invoke(main, ["cluster", "import", "--from-spark-vllm-docker-env", str(envf), "--name", "prod"])
    assert r1.exit_code == 0, r1.stderr
    assert r1.stdout.strip() == "prod"

    # Re-import WITHOUT --name: must find by sync_source and sync 'prod' in
    # place (not create a second 'cluster'); hosts change is applied.
    envf.write_text("CLUSTER_NODES=10.0.0.9\nETH_IF=enp1s0f1np1\nCONTAINER_HF_TOKEN=x\n")
    r2 = runner.invoke(main, ["cluster", "import", "--from-spark-vllm-docker-env", str(envf)])
    assert r2.exit_code == 0, r2.stderr
    assert r2.stdout.strip() == "prod"

    mgr = _get_cluster_manager()
    names = {c.name for c in mgr.list_clusters()}
    assert "cluster" not in names  # no duplicate under the derived name
    assert mgr.get("prod").hosts == ["10.0.0.9"]  # synced


def test_import_name_collision_errors(tmp_path):
    # Pre-existing cluster named "cluster" from a different source.
    _get_cluster_manager().create("cluster", ["x"])
    envf = _env(tmp_path)  # derives name "cluster"
    r = _runner().invoke(main, ["cluster", "import", "--from-spark-vllm-docker-env", str(envf)])
    assert r.exit_code == 1
    assert "already exists from a different source" in r.stderr


def test_import_dry_run_writes_nothing(tmp_path):
    envf = _env(tmp_path)
    r = _runner().invoke(main, ["cluster", "import", "--from-spark-vllm-docker-env", str(envf), "--dry-run"])
    assert r.exit_code == 0, r.stderr
    assert r.stdout.strip() == "cluster"  # still emits resolved name
    names = {c.name for c in _get_cluster_manager().list_clusters()}
    assert "cluster" not in names
