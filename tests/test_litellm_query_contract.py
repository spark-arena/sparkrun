"""LiteLLM owns its wire conversion and uses the same query contract as plugins."""

import json
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from sparkrun.proxy.contracts import GatewayQueryError, ProxyModel
from sparkrun.proxy.engine import ProxyEngine
from sparkrun.proxy.supervisor import GatewaySupervisor


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"data": None},
        {"data": {}},
        {"data": [None]},
        {"data": [{}]},
        {"data": [{"model_name": ""}]},
        {"data": [{"model_name": "m", "model_info": []}]},
        {"data": [{"model_name": "m", "litellm_params": []}]},
        {"data": [{"model_name": "m", "litellm_params": {"api_base": 1}}]},
        {"data": [{"model_name": "m", "model_info": {"max_input_tokens": "8192"}}]},
        {"data": [{"model_name": "m", "model_info": {"max_input_tokens": True}}]},
    ],
)
def test_malformed_model_info_is_a_query_failure(tmp_path, monkeypatch, payload):
    engine = ProxyEngine(state_dir=tmp_path)
    monkeypatch.setattr(engine, "_api_request", lambda *args: payload)
    with pytest.raises(GatewayQueryError, match="invalid"):
        engine.query_models()


@pytest.mark.parametrize("key", ["max_input_tokens", "max_tokens", "max_model_len"])
def test_model_info_translates_to_immutable_records(tmp_path, monkeypatch, key):
    engine = ProxyEngine(state_dir=tmp_path)
    request = Mock(
        return_value={"data": [{"model_name": "m", "litellm_params": {"api_base": "http://worker/v1"}, "model_info": {key: 8192}}]}
    )
    monkeypatch.setattr(engine, "_api_request", request)
    assert engine.query_models() == (ProxyModel("m", "http://worker/v1", 8192),)
    request.assert_called_once_with("GET", "/model/info")


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("http://fixture", 401, "private request context", {}, None),
        URLError("private request context"),
        OSError("private request context"),
        json.JSONDecodeError("private request context", "", 0),
    ],
)
def test_operational_failure_preserves_cause_without_rendering_request_context(tmp_path, monkeypatch, error):
    engine = ProxyEngine(state_dir=tmp_path)
    monkeypatch.setattr(engine, "_api_request", Mock(side_effect=error))
    with pytest.raises(GatewayQueryError) as caught:
        engine.query_models()
    assert caught.value.__cause__ is error
    assert "private request context" not in str(caught.value)


@pytest.mark.parametrize("error", [RuntimeError("provider bug"), NotImplementedError("provider bug"), KeyboardInterrupt()])
def test_unexpected_failures_propagate(tmp_path, monkeypatch, error):
    engine = ProxyEngine(state_dir=tmp_path)
    monkeypatch.setattr(engine, "_api_request", Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        engine.query_models()
    assert caught.value is error


def test_gateway_contract_has_no_legacy_model_hooks():
    for cls in (GatewaySupervisor, ProxyEngine):
        assert not hasattr(cls, "list_models_via_api")
        assert not hasattr(cls, "model_query_error")
