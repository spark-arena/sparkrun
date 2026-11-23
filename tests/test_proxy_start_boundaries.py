"""Generated gateway configuration belongs to the accepted start transition."""

from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest


from sparkrun import api
from sparkrun.api.proxy import _ops
from sparkrun.proxy.contracts import GatewayOperationError
from sparkrun.proxy.discovery import DiscoveredEndpoint


@pytest.fixture(params=["litellm", "fixture"])
def gateway(request, tmp_path, monkeypatch):
    from sparkrun.proxy import gateway as gateway_registry
    from sparkrun.proxy.gateway import register_gateway
    from sparkrun.plugins.litellm.engine import ProxyEngine

    name = request.param
    context = api.default_sctx()
    root = tmp_path / "gateway"
    events = []

    class LocalEngine(ProxyEngine):
        gateway_name = name

        def __init__(self, **kwargs):
            super().__init__(state_dir=root, **kwargs)

        def prepare_config(self, endpoints, aliases, *, write=True):
            events.append("write" if write else "preview")
            path, applied, pending = super().prepare_config(endpoints, aliases, write=write)
            # A provider may manage configuration without returning a path.
            return (path if name == "litellm" else None), applied, pending

        def start(self, **kwargs):
            events.append("start")
            assert kwargs["config_path"] == (self.config_path if name == "litellm" else None)
            return 0

    if name == "fixture":
        monkeypatch.setattr(gateway_registry, "_GATEWAY_LOADERS", dict(gateway_registry._GATEWAY_LOADERS))
        monkeypatch.setattr(gateway_registry, "GATEWAY_FEATURE_FLAGS", dict(gateway_registry.GATEWAY_FEATURE_FLAGS))
        register_gateway(name, feature_flag="gateway.litellm", loader=lambda: LocalEngine)
    engine = LocalEngine()
    engine.claim_state_directory()
    engine.prepare_config(
        [
            DiscoveredEndpoint(
                cluster_id="previous",
                model="previous-model",
                served_model_name=None,
                runtime="vllm",
                host="localhost",
                port=8000,
                healthy=True,
            )
        ],
        {},
    )
    engine._save_state(pid=12345)
    events.clear()
    monkeypatch.setattr(_ops, "_engine_class", lambda gateway: LocalEngine)
    monkeypatch.setattr(_ops, "_discover", lambda **kwargs: [])
    monkeypatch.setattr(LocalEngine, "is_running", lambda self: True)
    return name, context, engine, events


def generated_files(context, engine):
    roots = (engine.state_dir, context.proxy_config.config_path.parent)
    # Execution reservation bookkeeping is separate from generated gateway config.
    return {
        path: path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("operations.sqlite3")
    }


@pytest.mark.parametrize("restart", [False, True])
def test_refused_start_and_shutdown_timeout_preserve_generated_state(gateway, monkeypatch, restart):
    name, context, engine, events = gateway
    before = generated_files(context, engine)
    wait = Mock(return_value=False)
    monkeypatch.setattr(_ops, "_stop_and_wait", wait)
    error = api.proxy.ProxyStartFailed if restart else api.proxy.ProxyAlreadyRunning
    with pytest.raises(error):
        api.proxy.start(api.proxy.ProxyStartOptions(gateway=name, restart=restart, persist=False), sctx=context)
    assert generated_files(context, engine) == before
    assert events == ["preview"]
    assert wait.call_count == int(restart)


@pytest.mark.parametrize("foreground", [False, True])
@pytest.mark.parametrize("restart", [False, True])
def test_successful_start_writes_after_shutdown_and_normalizes_path(gateway, monkeypatch, foreground, restart):
    name, context, engine, events = gateway
    monkeypatch.setattr(type(engine), "is_running", lambda self: restart)

    def stop_and_wait(current):
        events.append("stop")
        return True

    monkeypatch.setattr(_ops, "_stop_and_wait", stop_and_wait)
    result = api.proxy.start(api.proxy.ProxyStartOptions(gateway=name, restart=restart, persist=False, foreground=foreground), sctx=context)
    assert events == ["preview", *(["stop"] if restart else []), "write", "start"]
    assert result.started and result.restarted == restart
    assert result.foreground_rc == (0 if foreground else None)
    expected_path = str(engine.config_path) if name == "litellm" else None
    assert result.config_path == expected_path
    assert json.loads(json.dumps(asdict(result)))["config_path"] == expected_path


def test_dry_run_preserves_generated_state_and_has_no_config_path(gateway):
    name, context, engine, events = gateway
    before = generated_files(context, engine)
    result = api.proxy.start(api.proxy.ProxyStartOptions(gateway=name, dry_run=True), sctx=context)
    assert not result.started and result.dry_run
    assert result.config_path is None
    assert json.loads(json.dumps(asdict(result)))["config_path"] is None
    assert generated_files(context, engine) == before
    assert events == ["preview"]


def test_invalid_preview_does_not_stop_a_working_gateway(gateway, monkeypatch):
    name, context, engine, _ = gateway
    before = generated_files(context, engine)
    monkeypatch.setattr(type(engine), "prepare_config", Mock(side_effect=GatewayOperationError("invalid configuration")))
    wait = Mock()
    monkeypatch.setattr(_ops, "_stop_and_wait", wait)
    with pytest.raises(api.proxy.ProxyStartFailed, match="invalid configuration") as caught:
        api.proxy.start(api.proxy.ProxyStartOptions(gateway=name, restart=True, persist=False), sctx=context)
    assert isinstance(caught.value.__cause__, GatewayOperationError)
    wait.assert_not_called()
    assert generated_files(context, engine) == before


def test_daemon_foreground_releases_start_reservation_before_wait(gateway, monkeypatch):
    from sparkrun.core.coordination import reservations
    from sparkrun.daemon import management, proxy as services

    name, context, engine, events = gateway
    monkeypatch.setattr(context.config, "is_feature_enabled", lambda name: True)
    monkeypatch.setattr(type(engine), "is_running", lambda self: False)
    monkeypatch.setattr(management, "ensure_daemon", lambda binding: {})

    def launch(*args, **kwargs):
        assert reservations(context.config.config_path.parent), "startup must stay guarded"
        events.append("daemon start")

    def wait(sctx):
        assert not reservations(sctx.config.config_path.parent), "foreground wait must not block workload activation"
        events.append("foreground wait")
        return 0

    monkeypatch.setattr(services, "launch_gateway", launch)
    monkeypatch.setattr(services, "wait_gateway", wait)
    result = api.proxy.start(
        api.proxy.ProxyStartOptions(gateway=name, discovery_backend="daemon", foreground=True, persist=False), sctx=context
    )
    assert result.started and result.foreground_rc == 0
    assert events == ["preview", "write", "daemon start", "foreground wait"]


def test_process_foreground_releases_start_reservation_before_supervision(gateway, monkeypatch):
    """`proxy start --foreground` (what the systemd unit runs) supervises the
    gateway for its whole life; holding the start reservation through that
    refused every `run` on the cluster's hosts with ResourceBusy."""
    import subprocess
    import sys

    from sparkrun.core.coordination import reservations

    from sparkrun.core.cluster_manager import ClusterManager

    name, context, engine, events = gateway
    root = context.config.config_path.parent
    # The reported shape: a flagless start reserves the default cluster's hosts.
    clusters = ClusterManager(root)
    clusters.create("lab", ["10.0.0.1", "10.0.0.2"])
    clusters.set_default("lab")
    monkeypatch.setattr(type(engine), "is_running", lambda self: False)

    def start(self, **kwargs):
        assert {row["resource"] for row in reservations(root)} == {"host:10.0.0.1", "host:10.0.0.2"}, "startup must stay guarded"
        events.append("start")
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self._save_state(proc.pid)
        self._mark_state(stop_requested=True)
        return self.supervise_foreground(proc)

    def finish(self, *args, **kwargs):
        assert not reservations(root), "foreground supervision must not hold host reservations"
        events.append("supervised")
        return original_finish(self, *args, **kwargs)

    original_finish = type(engine)._finish_foreground
    monkeypatch.setattr(type(engine), "start", start)
    monkeypatch.setattr(type(engine), "_finish_foreground", finish)
    result = api.proxy.start(api.proxy.ProxyStartOptions(gateway=name, foreground=True, persist=False, auto_discover=False), sctx=context)
    assert result.started
    assert events == ["preview", "write", "start", "supervised"]
    assert not reservations(root)


def test_stop_succeeds_while_another_owner_holds_the_cluster_hosts(gateway, monkeypatch):
    """`proxy stop` touches no cluster host. Reserving the default cluster
    made it raise ResourceBusy against the very proxy it was asked to stop."""
    import threading

    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.coordination import reservation_keys, reserve_resources

    name, context, engine, events = gateway
    root = context.config.config_path.parent
    clusters = ClusterManager(root)
    clusters.create("lab", ["10.0.0.1", "10.0.0.2"])
    clusters.set_default("lab")
    monkeypatch.setattr(type(engine), "is_running", lambda self: False)
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx: engine)

    held, release = threading.Event(), threading.Event()

    def holder():
        with reserve_resources(root, reservation_keys(["10.0.0.1", "10.0.0.2"])):
            held.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(10)
        result = api.proxy.stop(sctx=context)
    finally:
        release.set()
        thread.join()
    assert not result.was_running


@pytest.fixture(autouse=True)
def _gateway_provider(loaded_litellm_plugin):
    pass
