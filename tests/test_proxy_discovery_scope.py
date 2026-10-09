"""Which hosts ``proxy start`` discovers on, and which layer decided it.

The default cluster (``sparkrun cluster set-default``) used to be skipped:
discovery read only ``--hosts`` / ``--cluster`` or ``config.default_hosts``, so
a user whose only host source was a default cluster got metadata-only
discovery as the control node's own login.  ``proxy.cluster`` is saved only
when ``--cluster`` is given; otherwise the default is resolved at each start.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.api.proxy import _ops
from sparkrun.api.proxy import ProxyStartFailed, ProxyStartOptions


@pytest.fixture
def context(tmp_path):
    from sparkrun.application import initialize

    return initialize(config_path=tmp_path / "config.yaml")


@pytest.fixture
def clusters(context):
    manager = context.cluster_manager
    manager.create("lab", ["lab-1", "lab-2"], user="alice")
    manager.create("edge", ["edge-1"], user="bob")
    return manager


def _scope(options, context):
    return _ops._discovery_args(options, context)


def _set_default_hosts(tmp_path, hosts):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"cluster": {"hosts": hosts}}))
    from sparkrun.application import initialize

    return initialize(config_path=path)


def test_nothing_configured_is_metadata_only(context):
    scope, host_filter, live, ssh, cluster_def, warnings = _scope(ProxyStartOptions(), context)
    assert scope.source == "none"
    assert (host_filter, live, ssh, cluster_def, warnings) == (None, None, None, None, [])
    assert "metadata only" in scope.describe()


def test_default_hosts_keep_their_liveness_only_role(tmp_path):
    context = _set_default_hosts(tmp_path, ["h1", "h2"])
    scope, host_filter, live, ssh, cluster_def, _ = _scope(ProxyStartOptions(), context)
    assert scope.source == "default hosts"
    assert live == ["h1", "h2"]
    assert host_filter is None
    assert cluster_def is None


def test_default_cluster_is_used_in_full(context, clusters):
    clusters.set_default("lab")
    scope, host_filter, live, ssh, cluster_def, _ = _scope(ProxyStartOptions(), context)
    assert (scope.source, scope.cluster) == ("default cluster", "lab")
    assert live == host_filter == ["lab-1", "lab-2"]
    assert ssh["ssh_user"] == "alice"
    assert cluster_def.name == "lab"
    assert scope.describe() == "cluster lab (from default cluster)"


def test_default_cluster_outranks_default_hosts(tmp_path):
    context = _set_default_hosts(tmp_path, ["h1"])
    context.cluster_manager.create("lab", ["lab-1"], user="alice")
    context.cluster_manager.set_default("lab")
    scope, *_ = _scope(ProxyStartOptions(), context)
    assert scope.source == "default cluster"


def test_saved_cluster_outranks_default(context, clusters):
    clusters.set_default("lab")
    context.proxy_config.set_proxy(cluster="edge")
    scope, _, live, ssh, _, _ = _scope(ProxyStartOptions(), context)
    assert (scope.source, scope.cluster) == ("proxy.yaml", "edge")
    assert live == ["edge-1"]
    assert ssh["ssh_user"] == "bob"


def test_explicit_cluster_outranks_saved(context, clusters):
    context.proxy_config.set_proxy(cluster="edge")
    scope, *_ = _scope(ProxyStartOptions(cluster="lab"), context)
    assert (scope.source, scope.cluster) == ("--cluster", "lab")


def test_explicit_hosts_win_but_keep_cluster_user(context, clusters):
    clusters.set_default("edge")
    scope, host_filter, live, ssh, _, _ = _scope(ProxyStartOptions(host_filter=["x1"], cluster="lab"), context)
    assert scope.source == "--hosts"
    assert live == host_filter == ["x1"]
    assert ssh["ssh_user"] == "alice"


def test_dangling_saved_cluster_warns_and_falls_through(context, clusters):
    clusters.set_default("lab")
    context.proxy_config.set_proxy(cluster="gone")
    scope, *_, warnings = _scope(ProxyStartOptions(), context)
    assert (scope.source, scope.cluster) == ("default cluster", "lab")
    assert len(warnings) == 1 and "'gone'" in warnings[0]


def test_clear_cluster_ignores_saved(context, clusters):
    clusters.set_default("lab")
    context.proxy_config.set_proxy(cluster="edge")
    scope, *_ = _scope(ProxyStartOptions(clear_cluster=True), context)
    assert scope.cluster == "lab"


def test_dangling_default_falls_through(tmp_path):
    context = _set_default_hosts(tmp_path, ["h1"])
    context.cluster_manager.create("lab", ["lab-1"])
    context.cluster_manager.set_default("lab")
    context.cluster_manager.default_file.write_text("deleted-cluster\n")
    scope, *_ = _scope(ProxyStartOptions(), context)
    assert scope.source == "default hosts"


# -- start(): persistence and validation ------------------------------------


@pytest.fixture
def stub_engine(monkeypatch):
    engine = Mock()
    engine.is_running.return_value = False
    engine.prepare_config.return_value = (None, set(), set())
    engine.start.return_value = 0
    engine.supports_autodiscover = True
    monkeypatch.setattr(_ops, "_engine_class", lambda *_: Mock(return_value=engine))
    monkeypatch.setattr(_ops, "resolve_gateway", lambda *_a, **_k: "litellm")
    seen = {}

    def discover(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(_ops, "_discover", discover)
    engine.seen = seen
    return engine


def _saved_cluster(context):
    path = context.proxy_config.config_path
    if not path.exists():
        return None
    return (yaml.safe_load(path.read_text()) or {}).get("proxy", {}).get("cluster")


def test_explicit_cluster_is_saved_and_used(context, clusters, stub_engine):
    result = _ops.start(ProxyStartOptions(cluster="lab", auto_discover=False), sctx=context)
    assert "cluster" in result.persisted
    assert _saved_cluster(context) == "lab"
    assert result.discovery.cluster == "lab"
    assert stub_engine.seen["cluster_def"].name == "lab"


def test_default_cluster_is_not_saved(context, clusters, stub_engine):
    clusters.set_default("lab")
    result = _ops.start(ProxyStartOptions(auto_discover=False), sctx=context)
    assert result.persisted == ()
    assert _saved_cluster(context) is None
    assert result.discovery.source == "default cluster"


def test_unknown_explicit_cluster_fails_before_saving(context, clusters, stub_engine):
    with pytest.raises(ProxyStartFailed, match="Unknown cluster 'typo'"):
        _ops.start(ProxyStartOptions(cluster="typo"), sctx=context)
    assert _saved_cluster(context) is None
    stub_engine.start.assert_not_called()


def test_clear_cluster_removes_saved(context, clusters, stub_engine):
    _ops.start(ProxyStartOptions(cluster="edge", auto_discover=False), sctx=context)
    clusters.set_default("lab")
    from sparkrun.application import initialize

    fresh = initialize(config_path=context.config.config_path)
    result = _ops.start(ProxyStartOptions(clear_cluster=True, auto_discover=False), sctx=fresh)
    assert "cluster (cleared)" in result.persisted
    assert _saved_cluster(fresh) is None
    assert result.discovery.cluster == "lab"


def test_cluster_and_clear_cluster_conflict(context, clusters, stub_engine):
    with pytest.raises(ProxyStartFailed, match="mutually exclusive"):
        _ops.start(ProxyStartOptions(cluster="lab", clear_cluster=True), sctx=context)


def test_autodiscover_daemon_receives_cluster(context, clusters, stub_engine):
    clusters.set_default("lab")
    _ops.start(ProxyStartOptions(auto_discover=True), sctx=context)
    kwargs = stub_engine.start.call_args.kwargs["autodiscover_kwargs"]
    assert kwargs["cluster"] == "lab"
    assert kwargs["host_list"] == ["lab-1", "lab-2"]
    assert kwargs["ssh_kwargs"]["ssh_user"] == "alice"


def test_unset_proxy_survives_a_concurrent_writer(context):
    """A removal is an explicit deletion; other writers' keys are kept."""
    cfg = context.proxy_config
    cfg.set_proxy(cluster="lab")
    cfg.save()
    from sparkrun.proxy.config import ProxyConfig

    other = ProxyConfig(config_path=cfg.config_path)
    other.set_proxy(port=4100)
    other.save()

    cfg.unset_proxy("cluster")
    cfg.save()
    saved = yaml.safe_load(cfg.config_path.read_text())["proxy"]
    assert "cluster" not in saved
    assert saved["port"] == 4100


def test_cli_prints_discovery_line(context, clusters, stub_engine, monkeypatch):
    from sparkrun.cli._proxy import proxy

    clusters.set_default("lab")
    monkeypatch.setattr("sparkrun.cli._proxy._get_context", lambda _ctx: context)
    result = CliRunner().invoke(proxy, ["start", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Discovery: cluster lab (from default cluster)" in result.output
