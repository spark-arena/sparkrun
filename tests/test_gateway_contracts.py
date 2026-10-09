"""A typed gateway needs no LiteLLM dictionaries or optional console features."""

from __future__ import annotations

import json
import secrets

import pytest
from click.testing import CliRunner

from sparkrun import api
from sparkrun.api.proxy import _ops
from sparkrun.cli import main
from sparkrun.proxy.supervisor import GatewaySupervisor
from sparkrun.proxy.contracts import (
    GatewayAdminToken,
    GatewayConsole,
    GatewayConsoleCredentials,
    GatewayOperationError,
    GatewayQueryError,
    ProxyModel,
)


class TypedGateway(GatewaySupervisor):
    gateway_name = "typed"

    def get_state(self):
        return {"gateway": self.gateway_name, "pid": 123, "port": 4000}

    def is_running(self):
        return True

    def query_models(self):
        return (ProxyModel("example", "http://worker/v1", 32768),)


class ConsoleGateway(TypedGateway):
    ui_url = "http://127.0.0.1:9876/admin"
    admin_bind_host = "127.0.0.1"
    admin_exposed = False
    token = None

    @property
    def admin_auth_required(self):
        return self.token is not None

    def admin_token(self, *, rotate=False, clear=False):
        if clear:
            self.token = None
        elif rotate:
            self.token = secrets.token_urlsafe(32)
        return self.token

    def issue_ui_credential(self):
        return self.token or self.admin_token(rotate=True)


@pytest.fixture
def typed_gateway(tmp_path, monkeypatch):
    engine = TypedGateway(state_dir=tmp_path)
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: engine)
    return engine


def test_typed_models_share_the_public_result_class(typed_gateway):
    assert api.proxy.ProxyModel is ProxyModel
    result = api.proxy.models()
    assert result == typed_gateway.query_models()
    assert api.proxy.status().require_models() == result
    output = CliRunner().invoke(main, ["proxy", "models", "--json"])
    assert output.exit_code == 0, output.output
    assert json.loads(output.stdout) == [{"model_name": "example", "api_base": "http://worker/v1", "max_model_len": 32768}]


def test_typed_query_error_preserves_status(typed_gateway, monkeypatch):
    def unavailable():
        raise GatewayQueryError("control plane unavailable")

    monkeypatch.setattr(typed_gateway, "query_models", unavailable)
    assert api.proxy.status().model_query_error == "control plane unavailable"
    with pytest.raises(api.proxy.ProxyQueryFailed, match="control plane unavailable"):
        api.proxy.models()


def test_missing_optional_capabilities_are_explicit(typed_gateway):
    for protocol in (GatewayConsole, GatewayConsoleCredentials, GatewayAdminToken):
        assert not isinstance(typed_gateway, protocol)
    with pytest.raises(api.proxy.ProxyUnsupported, match="does not serve an admin console"):
        api.proxy.ui()
    with pytest.raises(api.proxy.ProxyUnsupported, match="no managed admin token"):
        api.proxy.admin_token()
    for command in (["ui"], ["admin-token", "get"]):
        result = CliRunner().invoke(main, ["proxy", *command])
        assert result.exit_code == 1, result.output


def test_console_and_tokens_follow_live_capabilities(tmp_path, monkeypatch):
    engine = ConsoleGateway(state_dir=tmp_path)
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: engine)
    for protocol in (GatewayConsole, GatewayConsoleCredentials, GatewayAdminToken):
        assert isinstance(engine, protocol)
    initial = api.proxy.ui()
    assert initial.url == engine.ui_url and not initial.auth_required and initial.token is None
    assert api.proxy.admin_token() is None
    issued = api.proxy.ui(issue_token=True)
    assert issued.token and issued.auth_required
    assert api.proxy.admin_token() == issued.token
    assert api.proxy.ui(issue_token=True).token == issued.token
    replacement = api.proxy.admin_token(rotate=True)
    assert replacement and replacement != issued.token
    assert api.proxy.admin_token(clear=True) is None
    assert not api.proxy.ui().auth_required
    with pytest.raises(ValueError, match="mutually exclusive"):
        api.proxy.admin_token(rotate=True, clear=True)
    assert api.proxy.admin_token() is None


def test_console_credential_issuance_is_independently_optional(tmp_path, monkeypatch):
    class ReadOnlyConsole(TypedGateway):
        ui_url = ConsoleGateway.ui_url
        admin_bind_host = "127.0.0.1"
        admin_exposed = False
        admin_auth_required = True

    monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: ReadOnlyConsole(state_dir=tmp_path))
    assert api.proxy.ui().auth_required
    with pytest.raises(api.proxy.ProxyUnsupported, match="cannot issue console credentials"):
        api.proxy.ui(issue_token=True)


def test_admin_failure_uses_public_error(tmp_path, monkeypatch):
    class RefusingConsole(ConsoleGateway):
        def admin_token(self, **kwargs):
            raise GatewayOperationError("credential storage is unavailable")

    monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: RefusingConsole(state_dir=tmp_path))
    with pytest.raises(api.proxy.ProxyUpdateFailed, match="credential storage"):
        api.proxy.admin_token(rotate=True)
    with pytest.raises(api.proxy.ProxyUpdateFailed, match="credential storage"):
        api.proxy.ui(issue_token=True)


def test_token_management_does_not_require_a_console(tmp_path, monkeypatch):
    class TokenOnly(TypedGateway):
        token = None
        admin_token = ConsoleGateway.admin_token

    engine = TokenOnly(state_dir=tmp_path)
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: engine)
    assert isinstance(engine, GatewayAdminToken) and not isinstance(engine, GatewayConsole)
    assert api.proxy.admin_token() is None
    assert api.proxy.admin_token(rotate=True)
    assert api.proxy.admin_token(clear=True) is None
    with pytest.raises(api.proxy.ProxyUnsupported):
        api.proxy.ui()


def test_public_gateway_contracts_preserve_legacy_class_identity():
    from sparkrun.proxy import _supervisor

    assert GatewaySupervisor is _supervisor.GatewaySupervisor
    assert GatewayOperationError is _supervisor.GatewayOperationError
    assert issubclass(GatewayQueryError, GatewayOperationError)


@pytest.mark.parametrize("operation", ["token", "credential", "console"])
@pytest.mark.parametrize("error_type", [GatewayOperationError, RuntimeError])
def test_optional_management_distinguishes_operational_and_programming_errors(tmp_path, monkeypatch, operation, error_type):
    cause = error_type("management failure")

    class BrokenConsole(ConsoleGateway):
        def admin_token(self, **kwargs):
            raise cause

        @property
        def ui_url(self):
            if operation == "console":
                raise cause
            return ConsoleGateway.ui_url

    monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: BrokenConsole(state_dir=tmp_path))
    error = api.proxy.ProxyUpdateFailed if error_type is GatewayOperationError else RuntimeError
    with pytest.raises(error, match="management failure") as caught:
        if operation == "token":
            api.proxy.admin_token(rotate=True)
        else:
            api.proxy.ui(issue_token=operation == "credential")
    assert (caught.value.__cause__ if error_type is GatewayOperationError else caught.value) is cause
