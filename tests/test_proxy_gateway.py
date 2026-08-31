"""Tests for the gateway seam behind ``sparkrun proxy``.

Covers three things:

- **Resolution** (:mod:`sparkrun.proxy.gateway`) — default-on ``litellm``,
  explicit pins, unknown/disabled names, and the ambiguity rule that stands in
  for "only one gateway at a time".
- **Gate placement** — ``ProxyEngine.start()`` refuses when the flag is off
  (including ``--dry-run``), while stop / status / sync stay usable so an
  already-running proxy remains manageable.
- **The api facade** (:mod:`sparkrun.api.proxy`) — the same rules surfaced as
  :class:`~sparkrun.api._errors.SparkrunError` subclasses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

FLAG_ENV = "SPARKRUN_FEATURE_GATEWAY_LITELLM"


@pytest.fixture
def gateway_off(monkeypatch):
    """Disable the litellm gateway for the duration of a test."""
    monkeypatch.setenv(FLAG_ENV, "0")


# =====================================================================
# Flag registration
# =====================================================================


class TestFlagRegistration:
    def test_flag_is_registered_and_on_every_channel(self):
        from sparkrun.core.channels import CHANNEL_ALPHA, CHANNEL_BETA, CHANNEL_STABLE
        from sparkrun.core.features import get_feature

        flag = get_feature("gateway.litellm")
        assert flag is not None
        for channel in (CHANNEL_STABLE, CHANNEL_BETA, CHANNEL_ALPHA):
            assert flag.default_for_channel(channel) is True

    def test_flag_name_matches_the_engine_declaration(self):
        """The engine's ``required_feature_flag`` is the registry key verbatim."""
        from sparkrun.proxy.engine import ProxyEngine
        from sparkrun.proxy.gateway import GATEWAY_FEATURE_FLAGS

        assert ProxyEngine.required_feature_flag == GATEWAY_FEATURE_FLAGS[ProxyEngine.gateway_name]


# =====================================================================
# Resolution
# =====================================================================


class TestResolveGateway:
    def test_default_is_litellm(self):
        from sparkrun.proxy.gateway import list_gateways, resolve_gateway

        assert resolve_gateway() == "litellm"
        assert list_gateways() == ["litellm"]

    def test_explicit_name_resolves(self):
        from sparkrun.proxy.gateway import resolve_gateway

        assert resolve_gateway("litellm") == "litellm"

    def test_unknown_name_raises_and_lists_known(self):
        from sparkrun.proxy.gateway import GatewayUnavailableError, resolve_gateway

        with pytest.raises(GatewayUnavailableError) as exc:
            resolve_gateway("nope")
        assert "Unknown gateway" in str(exc.value)
        assert "litellm" in str(exc.value)
        assert exc.value.gateway == "nope"

    def test_disabled_explicit_name_points_at_the_remedy(self, gateway_off):
        from sparkrun.proxy.gateway import GatewayUnavailableError, resolve_gateway

        with pytest.raises(GatewayUnavailableError) as exc:
            resolve_gateway("litellm")
        assert "setup features enable gateway.litellm" in str(exc.value)

    def test_nothing_enabled_raises(self, gateway_off):
        from sparkrun.proxy.gateway import GatewayUnavailableError, list_gateways, resolve_gateway

        assert list_gateways() == []
        with pytest.raises(GatewayUnavailableError) as exc:
            resolve_gateway()
        assert "No inference gateway is enabled" in str(exc.value)

    def test_sole_enabled_alternate_wins_when_default_is_off(self, monkeypatch, gateway_off):
        """Default disabled + exactly one other enabled -> that one, no guessing needed."""
        from sparkrun.proxy import gateway as gw

        monkeypatch.setitem(gw.GATEWAY_FEATURE_FLAGS, "other", "gateway.other")
        monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_OTHER", "1")

        assert gw.resolve_gateway() == "other"

    def test_several_enabled_without_default_is_ambiguous(self, monkeypatch, gateway_off):
        """Flags can't express mutual exclusion, so resolution refuses to guess."""
        from sparkrun.proxy import gateway as gw

        monkeypatch.setitem(gw.GATEWAY_FEATURE_FLAGS, "other", "gateway.other")
        monkeypatch.setitem(gw.GATEWAY_FEATURE_FLAGS, "third", "gateway.third")
        monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_OTHER", "1")
        monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_THIRD", "1")

        with pytest.raises(gw.AmbiguousGatewayError) as exc:
            gw.resolve_gateway()
        assert set(exc.value.available) == {"other", "third"}

    def test_default_still_wins_when_several_are_enabled(self, monkeypatch):
        """An enabled default is never ambiguous — it is the tie-break."""
        from sparkrun.proxy import gateway as gw

        monkeypatch.setitem(gw.GATEWAY_FEATURE_FLAGS, "other", "gateway.other")
        monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_OTHER", "1")

        assert gw.resolve_gateway() == "litellm"


# =====================================================================
# Gate placement on the engine
# =====================================================================


class TestEngineGate:
    def test_start_refuses_when_disabled(self, tmp_path: Path, gateway_off):
        from sparkrun.proxy.engine import ProxyEngine
        from sparkrun.proxy.gateway import GatewayUnavailableError

        engine = ProxyEngine(state_dir=tmp_path)
        with pytest.raises(GatewayUnavailableError):
            engine.start()

    def test_dry_run_start_also_refuses(self, tmp_path: Path, gateway_off):
        """A dry run must not advertise a start that would be refused."""
        from sparkrun.proxy.engine import ProxyEngine
        from sparkrun.proxy.gateway import GatewayUnavailableError

        engine = ProxyEngine(state_dir=tmp_path)
        with pytest.raises(GatewayUnavailableError):
            engine.start(dry_run=True)

    def test_stop_and_status_stay_usable_when_disabled(self, tmp_path: Path, monkeypatch):
        """Teardown is ungated: a proxy started while on must stay stoppable."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=tmp_path)
        engine._save_state(4321)

        monkeypatch.setenv(FLAG_ENV, "0")

        assert engine.get_state()["pid"] == 4321
        with patch("os.kill") as kill:
            assert engine.is_running() is True
            assert engine.stop() is True
        assert kill.called

    def test_state_records_the_gateway(self, tmp_path: Path):
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=tmp_path)
        engine._save_state(999)
        assert engine.get_state()["gateway"] == "litellm"


# =====================================================================
# api.proxy facade
# =====================================================================


class TestApiFacade:
    def test_resolve_and_list_are_exposed(self):
        from sparkrun import api

        assert api.proxy.resolve_gateway() == "litellm"
        assert api.proxy.list_gateways() == ["litellm"]

    def test_resolve_raises_typed_error_when_disabled(self, gateway_off):
        from sparkrun import api

        with pytest.raises(api.SparkrunError):
            api.proxy.resolve_gateway()

        with pytest.raises(api.proxy.GatewayUnavailable) as exc:
            api.proxy.resolve_gateway("litellm")
        assert exc.value.gateway == "litellm"

    def test_start_raises_gateway_unavailable_when_disabled(self, gateway_off):
        from sparkrun import api

        with pytest.raises(api.proxy.GatewayUnavailable):
            api.proxy.start()

    def test_start_honors_the_proxy_yaml_pin(self, tmp_path: Path):
        from sparkrun import api
        from sparkrun.proxy.config import ProxyConfig

        cfg = ProxyConfig(tmp_path / "proxy.yaml")
        cfg.set_proxy(gateway="nope")
        cfg.save()

        with patch("sparkrun.core.config.SparkrunConfig.get_proxy_config", lambda self: cfg):
            with pytest.raises(api.proxy.GatewayUnavailable) as exc:
                api.proxy.start()
        assert exc.value.gateway == "nope"

    def test_status_and_stop_work_when_disabled(self, gateway_off):
        """Management surfaces never consult the gate."""
        from sparkrun import api

        status = api.proxy.status()
        assert status.running is False

        result = api.proxy.stop()
        assert result.was_running is False

    def test_sync_requiring_a_running_proxy_is_a_noop_when_stopped(self):
        from sparkrun import api

        with patch("sparkrun.proxy.engine.ProxyEngine.is_running", return_value=False):
            result = api.proxy.sync(require_running=True)
        assert result.proxy_running is False
        assert result.changed is False

    def test_management_engine_follows_the_state_file(self):
        """Management paths bind to what is *running*, not what is configured."""
        from sparkrun.api.proxy._ops import _running_engine
        from sparkrun.proxy.engine import ProxyEngine

        with patch("sparkrun.proxy.engine.ProxyEngine.get_state", return_value={"pid": 1, "gateway": "litellm"}):
            assert isinstance(_running_engine(), ProxyEngine)

    def test_management_engine_degrades_for_an_unknown_recorded_gateway(self, caplog):
        """A live process must not be left undescribable and unkillable.

        This used to raise ``GatewayUnavailable``, which made ``proxy status``
        traceback and ``proxy stop`` impossible for a proxy whose plugin had
        been removed or whose flag was turned off after it started — the exact
        outcome the ungated management paths exist to prevent. State reading and
        SIGTERM are gateway-independent, so the base supervisor covers both;
        anything implementation-specific still fails, naming the gateway.
        """
        from sparkrun.api.proxy._ops import _running_engine
        from sparkrun.proxy._supervisor import GatewaySupervisor

        with patch("sparkrun.proxy.engine.ProxyEngine.get_state", return_value={"pid": 1, "gateway": "ghost"}):
            with caplog.at_level(logging.WARNING):
                engine = _running_engine()

        assert isinstance(engine, GatewaySupervisor)
        assert engine.gateway_name == "ghost"
        # Degrading is not the same as staying quiet about it.
        assert "ghost" in caplog.text

        with pytest.raises(NotImplementedError) as exc:
            engine.list_models_via_api()
        assert "ghost" in str(exc.value)
