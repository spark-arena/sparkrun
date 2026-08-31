"""``sparkrun cluster status`` rendering: what is targeted, and what is free.

Two things the report used to get wrong: it never said *which* cluster it was
reporting on (with no flags the host list comes from the default cluster), and
it listed a host as "idle" while a multi-GB image distribution was staging onto
it — which is exactly when the next launch gets aimed there.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from sparkrun.cli import main


@pytest.fixture
def cluster_mgr():
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root

    return ClusterManager(get_config_root(init_sparkrun()))


@pytest.fixture
def empty_sweep(monkeypatch):
    """Report every queried host as reachable with nothing running."""
    import sparkrun.orchestration.executor as executor_mod
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

    def _fake_query(cluster_def, hosts, **kwargs):
        return ClusterStatus(hosts=tuple(HostOccupancy(host=h, workloads=()) for h in hosts), executor="docker")

    monkeypatch.setattr(executor_mod, "query_status_for_cluster", _fake_query)


def _write_lock(hosts, **kw):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.pending_ops import create_pending_op

    create_pending_op(
        "lockkey",
        "image_distribute",
        hosts=hosts,
        cache_dir=str(SparkrunConfig().cache_dir),
        **kw,
    )


def test_status_names_the_default_cluster(cluster_mgr, empty_sweep):
    cluster_mgr.create("pinned-lab", ["10.0.0.1", "10.0.0.2"])
    cluster_mgr.set_default("pinned-lab")

    result = CliRunner().invoke(main, ["cluster", "status"])

    assert result.exit_code == 0, result.output
    assert "Cluster: pinned-lab (default) — 2 host(s)" in result.output


def test_status_names_explicit_hosts_as_such(cluster_mgr, empty_sweep):
    result = CliRunner().invoke(main, ["cluster", "status", "--hosts", "10.0.0.5"])

    assert result.exit_code == 0, result.output
    assert "Hosts: 1 host(s) (--hosts)" in result.output


def test_targeted_host_reported_as_preparing_not_idle(cluster_mgr, empty_sweep):
    """The host a distribution is staging onto is spoken for, not idle."""
    cluster_mgr.create("lab", ["10.0.0.1", "10.0.0.2"])
    cluster_mgr.set_default("lab")
    _write_lock(["10.0.0.1"], recipe="coldsnap-sglang", image="sparkrun/coldsnap:abc")

    result = CliRunner().invoke(main, ["cluster", "status"])
    out = result.output

    assert result.exit_code == 0, out
    preparing = out.index("Preparing (launch in progress")
    idle = out.index("Idle hosts")
    # 10.0.0.1 under Preparing, 10.0.0.2 under Idle — and only there.
    assert "10.0.0.1" in out[preparing:idle]
    assert "10.0.0.1" not in out[idle : out.index("Pending operations")]
    assert "10.0.0.2" in out[idle:]
    # The op names its targets and the image it is moving.
    assert "hosts: 10.0.0.1" in out
    assert "image=sparkrun/coldsnap:abc" in out


def test_json_reports_both_host_groups(cluster_mgr, empty_sweep):
    import json

    cluster_mgr.create("lab", ["10.0.0.1", "10.0.0.2"])
    cluster_mgr.set_default("lab")
    _write_lock(["10.0.0.1"], recipe="coldsnap-sglang")

    result = CliRunner().invoke(main, ["cluster", "status", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["preparing_hosts"] == ["10.0.0.1"]
    assert payload["idle_hosts"] == ["10.0.0.2"]
    assert payload["pending_ops"][0]["matched_hosts"] == ["10.0.0.1"]
