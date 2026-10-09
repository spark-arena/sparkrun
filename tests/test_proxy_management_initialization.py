"""First-call gateway management loads plugins without losing process recovery."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml

from sparkrun import api
from sparkrun.api.proxy import _ops
from sparkrun.core import bootstrap
from sparkrun.proxy._supervisor import GatewayState, GatewaySupervisor
from sparkrun.proxy.contracts import ProxyModel


def prepare_application(tmp_path, monkeypatch, *, alternate=False, enabled=True, provider="sparkroute"):
    from sparkrun.application import ApplicationProfile
    from sparkrun.proxy import gateway

    profile_ref = "sparkrun.core.application_profile:SPARKRUN"
    identity, prefix = "sparkrun", "SPARKRUN"
    if alternate:
        identity, prefix = "alternate-test", "ALTERNATE_TEST"
        profile_ref = "gateway_test_profile:PROFILE"
        module = ModuleType("gateway_test_profile")
        module.PROFILE = ApplicationProfile(
            id=identity, display_name="Alternate test", command=identity, package=identity, profile_ref=profile_ref
        )
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setenv("SPARKRUN_CACHE_DIR", str(tmp_path / "other-application-cache"))
    monkeypatch.setenv("SPARKRUN_APPLICATION_PROFILE", profile_ref)
    config = tmp_path / "custom-config" / "config.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({"features": {"gateway.sparkroute": enabled}}))
    (config.parent / "proxy.yaml").write_text("proxy:\n  gateway: litellm\n")
    monkeypatch.setenv("SPARKRUN_APPLICATION_CONFIG", str(config))
    monkeypatch.setenv(prefix + "_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(prefix + "_FEATURE_GATEWAY_SPARKROUTE", raising=False)
    monkeypatch.delitem(gateway._GATEWAY_LOADERS, provider, raising=False)
    state = tmp_path / "cache" / "proxy" / "state.yaml"
    state.parent.mkdir(parents=True)
    state.write_text(yaml.safe_dump({"gateway": provider, "distribution": identity, "pid": 12345, "port": 8000}))
    monkeypatch.setattr(GatewaySupervisor, "is_running", lambda self: True)
    return config, identity


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("first_call", ["status", "models"])
def test_first_management_call_initializes_profile_config_and_plugins(tmp_path, monkeypatch, alternate, first_call):
    from sparkrun.proxy import gateway
    from sparkrun.proxy.gateway import register_gateway
    from sparkrun.proxy.engine import ProxyEngine

    config, identity = prepare_application(tmp_path, monkeypatch, alternate=alternate, provider="fixture")
    observed = []
    expected = (ProxyModel("served-model", "http://fixture/v1"),)

    def query(engine):
        observed.append(engine.sctx)
        assert engine.sctx.config.config_path == config
        assert engine.sctx.application_identity.id == identity
        assert engine.proxy_config.gateway == "litellm"  # Recorded gateway wins.
        return expected

    class FixtureGateway(GatewaySupervisor):
        gateway_name = "fixture"
        wants_proxy_config = True
        query_models = query

        def __init__(self, *, proxy_config, sctx, **kwargs):
            super().__init__(**kwargs)
            self.proxy_config = proxy_config
            self.sctx = sctx

    register_plugins = bootstrap._register_plugins

    def register(v, *, config=None):
        register_plugins(v, config=config)
        register_gateway("fixture", feature_flag="gateway.sparkroute", loader=lambda: FixtureGateway)

    monkeypatch.setattr(gateway, "_GATEWAY_LOADERS", dict(gateway._GATEWAY_LOADERS))
    monkeypatch.setattr(gateway, "GATEWAY_FEATURE_FLAGS", dict(gateway.GATEWAY_FEATURE_FLAGS))
    monkeypatch.setattr(bootstrap, "_register_plugins", register)
    monkeypatch.setattr(ProxyEngine, "__init__", Mock(side_effect=AssertionError("management constructed LiteLLM")))
    assert bootstrap._variables is None
    result = getattr(api.proxy, first_call)()
    assert (result.require_models() if first_call == "status" else result) == expected
    assert bootstrap._variables is not None
    assert len(observed) == 1
    context = observed[0]
    # An explicit context must be reused without another default initialization.
    monkeypatch.setattr("sparkrun.api._context.default_sctx", Mock(side_effect=AssertionError("reinitialized")))
    assert api.proxy.models(sctx=context) == expected
    assert observed[-1] is context


@pytest.mark.parametrize("alternate", [False, True])
def test_disabled_plugin_keeps_status_and_stop_available(tmp_path, monkeypatch, alternate):
    prepare_application(tmp_path, monkeypatch, alternate=alternate, enabled=False)
    stop = Mock(return_value=True)
    monkeypatch.setattr(GatewaySupervisor, "stop", stop)
    result = api.proxy.status()
    assert result.running and result.gateway == "sparkroute"
    assert "sparkroute" in result.model_query_error
    assert api.proxy.stop().stopped
    stop.assert_called_once_with(dry_run=False)


@pytest.mark.parametrize("alternate", [False, True])
def test_failed_bootstrap_keeps_process_management_and_reports_failure(tmp_path, monkeypatch, alternate, caplog):
    prepare_application(tmp_path, monkeypatch, alternate=alternate)
    monkeypatch.setattr(bootstrap, "_register_plugins", Mock(side_effect=RuntimeError("broken plugin")))
    monkeypatch.setattr(_ops, "_engine_class", Mock(side_effect=AssertionError("used partial plugin registry")))
    stop = Mock(return_value=True)
    monkeypatch.setattr(GatewaySupervisor, "stop", stop)
    result = api.proxy.status()
    assert result.running and result.gateway == "sparkroute" and result.pid == 12345
    assert "sparkroute" in result.model_query_error
    assert "broken plugin" in caplog.text and "only process-level" in caplog.text
    with pytest.raises(api.proxy.ProxyQueryFailed, match="sparkroute"):
        api.proxy.models()
    assert api.proxy.stop().stopped
    stop.assert_called_once_with(dry_run=False)


def test_invalid_profile_cannot_fall_back_to_another_applications_state(monkeypatch):
    monkeypatch.setenv("SPARKRUN_APPLICATION_PROFILE", "missing_gateway_profile:PROFILE")
    monkeypatch.setattr(GatewayState, "__init__", Mock(side_effect=AssertionError("read wrong application's state")))
    with pytest.raises(api.SparkrunError, match="missing_gateway_profile"):
        api.proxy.status()


def test_initialization_interrupt_is_not_swallowed(monkeypatch):
    monkeypatch.setattr("sparkrun.application.initialize", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        api.proxy.status()
