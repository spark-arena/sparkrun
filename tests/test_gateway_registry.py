"""The gateway seam, exercised through a gateway that is not LiteLLM.

Asserting the registry against LiteLLM alone would prove nothing: a registry
with one entry that is also the default exercises no resolution. So these tests
register a synthetic second gateway at runtime and check it reaches selection,
construction, the management paths and the capability hooks with **no core
edit** — the executable form of the pluggability claim, following the pattern
``test_kv_strategies.py`` established for KV-cache strategies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparkrun.core.features import FEATURE_FLAGS, FeatureFlag, register_feature
from sparkrun.proxy._supervisor import GatewaySupervisor
from sparkrun.proxy.gateway import (
    DEFAULT_GATEWAY,
    GATEWAY_FEATURE_FLAGS,
    _GATEWAY_LOADERS,
    AmbiguousGatewayError,
    GatewayUnavailableError,
    gateway_class,
    is_gateway_enabled,
    list_gateways,
    register_gateway,
    resolve_gateway,
)

FAKE_GATEWAY = "fake"
FAKE_FLAG = "gateway.fake"


class FakeGateway(GatewaySupervisor):
    """A second gateway, defined entirely outside ``sparkrun.proxy``."""

    gateway_name = FAKE_GATEWAY
    required_feature_flag = FAKE_FLAG
    log_name = "fake.log"
    supports_autodiscover = False
    wants_proxy_config = True

    def __init__(self, host="127.0.0.1", port=4100, master_key=None, state_dir=None, host_configured=False, proxy_config=None, sctx=None):
        super().__init__(state_dir)
        self.host = host
        self.port = port
        self.master_key = master_key
        self.host_configured = host_configured
        self.proxy_config = proxy_config
        self.sctx = sctx
        self.prepared: list[tuple] = []

    def _state_payload(self):
        return {"port": self.port, "host": self.host}

    def prepare_config(self, endpoints, aliases, *, write=True):
        self.prepared.append((tuple(endpoints), dict(aliases), write))
        return (Path("/tmp/fake-gateway.conf") if write else None), set(aliases), set()


@pytest.fixture
def fake_gateway(monkeypatch):
    """Register :class:`FakeGateway`, enabled, and tear the registry down.

    ``_GATEWAY_LOADERS`` / ``GATEWAY_FEATURE_FLAGS`` / ``FEATURE_FLAGS`` are
    process-global, so a leaked registration would follow into every later
    test in the session.
    """
    saved_loaders = dict(_GATEWAY_LOADERS)
    saved_flags = dict(GATEWAY_FEATURE_FLAGS)
    saved_features = dict(FEATURE_FLAGS)

    register_feature(FeatureFlag(name=FAKE_FLAG, description="test gateway", default=False))
    register_gateway(FAKE_GATEWAY, feature_flag=FAKE_FLAG, loader=lambda: FakeGateway)
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_FAKE", "1")

    yield FakeGateway

    _GATEWAY_LOADERS.clear()
    _GATEWAY_LOADERS.update(saved_loaders)
    GATEWAY_FEATURE_FLAGS.clear()
    GATEWAY_FEATURE_FLAGS.update(saved_flags)
    FEATURE_FLAGS.clear()
    FEATURE_FLAGS.update(saved_features)


# --------------------------------------------------------------------------
# Registration and resolution
# --------------------------------------------------------------------------


def test_litellm_resolves_with_no_plugin_registered():
    """The built-in default is core, not a plugin: proxy must resolve to it."""
    from sparkrun.proxy.engine import ProxyEngine

    assert gateway_class(DEFAULT_GATEWAY) is ProxyEngine
    assert GATEWAY_FEATURE_FLAGS[DEFAULT_GATEWAY] == "gateway.litellm"


def test_a_registered_gateway_becomes_resolvable(fake_gateway):
    assert gateway_class(FAKE_GATEWAY) is FakeGateway
    assert is_gateway_enabled(FAKE_GATEWAY)
    assert FAKE_GATEWAY in list_gateways()


def test_unregistered_is_distinguishable_from_disabled(monkeypatch):
    """Two different problems must not share one message.

    A name can be known to the flag registry while its plugin failed to load;
    telling that user to enable a flag which is already on is a dead end.
    """
    with pytest.raises(GatewayUnavailableError) as unregistered:
        gateway_class("no-such-gateway")
    assert "No implementation registered" in str(unregistered.value)

    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_LITELLM", "0")
    with pytest.raises(GatewayUnavailableError) as disabled:
        resolve_gateway(DEFAULT_GATEWAY)
    assert "disabled" in str(disabled.value)
    assert "setup features enable" in str(disabled.value)


def test_enabling_a_second_gateway_does_not_switch_to_it(fake_gateway):
    """Availability is not selection: with the default enabled, it wins."""
    assert set(list_gateways()) >= {DEFAULT_GATEWAY, FAKE_GATEWAY}
    assert resolve_gateway() == DEFAULT_GATEWAY
    # Selecting the non-default one is an explicit act.
    assert resolve_gateway(FAKE_GATEWAY) == FAKE_GATEWAY


def test_ambiguous_when_default_disabled_and_several_remain(fake_gateway, monkeypatch):
    register_feature(FeatureFlag(name="gateway.fake2", description="test gateway 2", default=False))
    register_gateway("fake2", feature_flag="gateway.fake2", loader=lambda: FakeGateway)
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_FAKE2", "1")
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_LITELLM", "0")

    with pytest.raises(AmbiguousGatewayError) as exc:
        resolve_gateway()
    assert set(exc.value.available) >= {FAKE_GATEWAY, "fake2"}


def test_the_sole_enabled_gateway_wins_when_default_is_off(fake_gateway, monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_LITELLM", "0")
    assert resolve_gateway() == FAKE_GATEWAY


def test_registration_replaces_by_name(fake_gateway):
    """What lets an out-of-tree plugin substitute an in-tree implementation."""

    class Replacement(FakeGateway):
        pass

    register_gateway(FAKE_GATEWAY, feature_flag=FAKE_FLAG, loader=lambda: Replacement)
    assert gateway_class(FAKE_GATEWAY) is Replacement


# --------------------------------------------------------------------------
# The api.proxy management paths bind to the *running* gateway
# --------------------------------------------------------------------------


def _write_fake_state(tmp_path: Path, monkeypatch) -> Path:
    """Point the proxy state dir at *tmp_path* and record a fake-gateway run."""
    import sparkrun.core.config as config_mod

    monkeypatch.setattr(config_mod, "DEFAULT_CACHE_DIR", tmp_path)
    state_dir = tmp_path / "proxy"
    engine = FakeGateway(state_dir=state_dir)
    engine._save_state(pid=4242)
    return state_dir


def test_management_paths_resolve_the_engine_from_the_state_file(fake_gateway, tmp_path, monkeypatch):
    """``stop`` / ``status`` act on what is running, not what is configured."""
    from sparkrun.api.proxy import _ops

    _write_fake_state(tmp_path, monkeypatch)
    engine = _ops._running_engine(None)
    assert isinstance(engine, FakeGateway)
    assert engine.gateway_name == FAKE_GATEWAY


def test_wants_proxy_config_is_honored_on_management_paths(fake_gateway, tmp_path, monkeypatch):
    """Without the config a reconcile computes an *empty* desired state.

    That is the regression where ``proxy alias add`` silently deletes every
    deployment it was not explicitly told about.
    """
    from sparkrun.api.proxy import _ops

    _write_fake_state(tmp_path, monkeypatch)
    engine = _ops._running_engine(None)
    assert engine.proxy_config is not None
    assert engine.sctx is not None


def test_a_running_gateway_stays_manageable_when_its_plugin_is_gone(tmp_path, monkeypatch):
    """Deliberately *without* the fake_gateway fixture: nothing is registered.

    A state file can name a gateway whose plugin was removed, or whose flag was
    turned off since it started.  Raising would strand a live process — status
    could not describe it and stop could not kill it — which is precisely what
    the ungated management paths exist to prevent.  State reading and SIGTERM
    are gateway-independent, so the base supervisor covers both.
    """
    from sparkrun.api.proxy import _ops

    _write_fake_state(tmp_path, monkeypatch)

    engine = _ops._running_engine(None)
    assert engine.gateway_name == FAKE_GATEWAY
    assert engine.current_pid() == 4242
    assert hasattr(engine, "stop")

    status = _ops.status()
    assert status.gateway == FAKE_GATEWAY
    assert status.pid == 4242


def test_state_file_records_which_gateway_owns_the_process(fake_gateway, tmp_path, monkeypatch):
    state_dir = _write_fake_state(tmp_path, monkeypatch)
    from sparkrun.proxy._supervisor import GatewayState

    state = GatewayState(state_dir=state_dir)
    assert state.recorded_gateway() == FAKE_GATEWAY
    assert state.current_pid() == 4242


# --------------------------------------------------------------------------
# Capability declarations
# --------------------------------------------------------------------------


def test_unimplemented_model_management_names_the_gateway(fake_gateway):
    """A NotImplementedError naming itself, never a bare AttributeError.

    ``api.proxy`` resolves an engine from the state file, so a LiteLLM-only
    method reached against another gateway surfaces far from its cause.
    """
    engine = FakeGateway()
    for call in (
        lambda: engine.sync_models([], {}),
        lambda: engine.sync_aliases({}),
        engine.list_models_via_api,
    ):
        with pytest.raises(NotImplementedError) as exc:
            call()
        assert FAKE_GATEWAY in str(exc.value)


def test_discovery_driven_hooks_default_to_none(fake_gateway):
    """``None`` means "do the ordinary endpoint sync" — a true no-op seam."""
    engine = FakeGateway()
    assert engine.register_loaded_model("some-recipe") is None
    assert engine.unregister_loaded_model("some-recipe") is None


def test_status_reports_a_failed_model_query_rather_than_an_empty_list(fake_gateway, tmp_path, monkeypatch):
    """An empty list means "nothing registered", which would be a lie here."""
    from sparkrun.api.proxy import _ops

    _write_fake_state(tmp_path, monkeypatch)
    monkeypatch.setattr(FakeGateway, "is_running", lambda self: True)

    result = _ops.status()
    assert result.models == ()
    assert FAKE_GATEWAY in result.model_query_error
    assert result.to_dict()["model_query_error"] == result.model_query_error


def test_dry_run_prepares_the_config_without_writing(fake_gateway):
    """The preview is rendered by the same code that writes the real config."""
    engine = FakeGateway()
    path, applied, pending = engine.prepare_config([], {"fast": "some-model"}, write=False)
    assert path is None
    assert applied == {"fast"}
    assert engine.prepared[-1][2] is False


def test_data_plane_authentication_defaults_to_false(fake_gateway):
    """The safe assumption: a gateway that authenticates has to say so."""
    assert FakeGateway().data_plane_authenticated is False

    from sparkrun.proxy.engine import ProxyEngine

    assert ProxyEngine(master_key="sk-x").data_plane_authenticated is True
    assert ProxyEngine(master_key=None).data_plane_authenticated is False
