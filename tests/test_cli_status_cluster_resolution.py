"""The default cluster must reach the status sweep, not just the host list.

``resolve_hosts`` consults the default cluster (``sparkrun cluster set-default``)
when no ``--cluster``/``--hosts`` is given, so the CLI ends up with a concrete
host list.  Handing that list to ``api.status_report`` with ``cluster=None``
made ``api._resolve.resolve_cluster`` short-circuit to an *anonymous*
``ClusterDefinition`` — silently dropping the cluster's executor pin,
``executor_config`` and hardware from the query, with no error.  The CLI now
forwards the *effective* cluster name (``HostContext.cluster_name``).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from sparkrun.cli import main


@pytest.fixture
def cluster_mgr():
    """A ClusterManager rooted where the CLI's own context will look."""
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root

    return ClusterManager(get_config_root(init_sparkrun()))


@pytest.fixture
def captured_sweep(monkeypatch):
    """Intercept the one status choke point and record what it was asked."""
    import sparkrun.orchestration.executor as executor_mod
    from sparkrun.core.cluster_status import ClusterStatus

    seen: dict = {}

    def _fake_query(cluster_def, hosts, **kwargs):
        seen["cluster"] = cluster_def
        seen["hosts"] = list(hosts)
        return ClusterStatus()

    monkeypatch.setattr(executor_mod, "query_status_for_cluster", _fake_query)
    return seen


def _make_pinned_default(mgr, name="pinned-lab"):
    mgr.create(name, ["10.0.0.1"], executor="local", executor_config={"pid_dir": "/tmp/sparkrun-pids"})
    mgr.set_default(name)


def test_status_without_cluster_flag_uses_default_cluster(cluster_mgr, captured_sweep):
    """``sparkrun cluster status`` with no flags queries the *default cluster*,
    carrying its executor pin — not an anonymous host list."""
    _make_pinned_default(cluster_mgr)

    result = CliRunner().invoke(main, ["cluster", "status"])

    assert result.exit_code == 0, result.output
    assert captured_sweep["hosts"] == ["10.0.0.1"]
    cluster_def = captured_sweep["cluster"]
    assert cluster_def.name == "pinned-lab"
    assert cluster_def.executor == "local"
    assert cluster_def.executor_config == {"pid_dir": "/tmp/sparkrun-pids"}


def test_stop_all_without_cluster_flag_uses_default_cluster(cluster_mgr, captured_sweep):
    """Same for ``stop --all``: discovery must run on the substrate that
    teardown will use, or a workload is reported gone and left running."""
    _make_pinned_default(cluster_mgr)

    result = CliRunner().invoke(main, ["stop", "--all"])

    # No containers found (the fake sweep is empty) — what matters is *what*
    # was asked.
    assert captured_sweep["hosts"] == ["10.0.0.1"]
    assert captured_sweep["cluster"].executor == "local"
    assert "No sparkrun containers running." in result.output


def test_explicit_cluster_flag_still_wins(cluster_mgr, captured_sweep):
    """An explicit ``--cluster`` is used even when a different default exists."""
    _make_pinned_default(cluster_mgr)
    cluster_mgr.create("other-lab", ["10.0.0.9"], executor="docker")

    result = CliRunner().invoke(main, ["cluster", "status", "--cluster", "other-lab"])

    assert result.exit_code == 0, result.output
    assert captured_sweep["hosts"] == ["10.0.0.9"]
    assert captured_sweep["cluster"].name == "other-lab"
    assert captured_sweep["cluster"].executor == "docker"


def test_explicit_hosts_stay_unattached(cluster_mgr, captured_sweep):
    """``--hosts`` is not silently adopted by the default cluster.

    The fix attaches the default cluster only where the host list *came from*
    it; an explicit host list keeps resolving to an anonymous cluster, as it
    always has.
    """
    _make_pinned_default(cluster_mgr)

    result = CliRunner().invoke(main, ["cluster", "status", "--hosts", "10.0.0.2"])

    assert result.exit_code == 0, result.output
    assert captured_sweep["hosts"] == ["10.0.0.2"]
    assert captured_sweep["cluster"].name == ""
    assert captured_sweep["cluster"].executor is None
