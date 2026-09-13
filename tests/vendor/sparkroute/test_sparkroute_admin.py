# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Tests for the gateway admin client and reconciler credential.

Two things matter beyond plumbing:

* the bearer secret must never reach argv, a log, or an error surface;
* a 409 is ordinary concurrent control-plane behaviour, so it must be
  distinguishable from a real failure and retried against a *fresh* token.
"""

from __future__ import annotations

import io
import json
import stat
import subprocess
import urllib.error
from unittest import mock

import pytest

from sparkrun.plugins.sparkroute import admin as admin_mod
from sparkrun.plugins.sparkroute import credentials as cred_mod
from sparkrun.plugins.sparkroute.admin import AdminClient, AdminError, RevisionConflict
from sparkrun.plugins.sparkroute.credentials import CredentialError, ReconcilerCredential

REV_A = "a" * 64
REV_B = "b" * 64


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _size=None):
        return self._body


def _http_error(status: int, code: str = "", message: str = ""):
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    return urllib.error.HTTPError("http://x", status, message or "error", {}, io.BytesIO(body))


def _client():
    return AdminClient("http://127.0.0.1:8081", "sk-reconciler-secret")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def test_reads_send_the_bearer_token_and_target_the_sparkrun_owner():
    with mock.patch.object(admin_mod.urllib.request, "urlopen", return_value=_Response({"active_revision": REV_A})) as urlopen:
        assert _client().active_revision() == REV_A
    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/v1/config/managed-sets")
    assert request.get_header("Authorization") == "Bearer sk-reconciler-secret"


def test_disabled_admin_client_sends_no_authorization_header():
    client = AdminClient("http://127.0.0.1:4000", None)
    with mock.patch.object(admin_mod.urllib.request, "urlopen", return_value=_Response({"active_revision": REV_A})) as urlopen:
        assert client.active_revision() == REV_A
    assert urlopen.call_args.args[0].get_header("Authorization") is None


def test_replace_puts_the_whole_set_with_the_cas_token():
    document = {"providers": [], "deployments": [], "virtual_models": []}
    with mock.patch.object(admin_mod.urllib.request, "urlopen", return_value=_Response({"changed": True})) as urlopen:
        _client().replace(document, REV_A, "because")
    request = urlopen.call_args.args[0]
    assert request.method == "PUT"
    assert request.full_url.endswith("/v1/config/managed-sets/sparkrun")
    body = json.loads(request.data)
    assert body == {"document": document, "expected_active_revision": REV_A, "reason": "because"}


def test_oversized_document_is_refused_before_the_request():
    huge = {"deployments": [{"name": "x" * 1000} for _ in range(20000)]}
    with mock.patch.object(admin_mod.urllib.request, "urlopen") as urlopen:
        with pytest.raises(AdminError, match="request limit"):
            _client().replace(huge, REV_A)
    urlopen.assert_not_called()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_revision_conflict_is_typed_and_retryable():
    with mock.patch.object(admin_mod.urllib.request, "urlopen", side_effect=_http_error(409, "revision_conflict", "moved")):
        with pytest.raises(RevisionConflict) as exc_info:
            _client().replace({}, REV_A)
    assert exc_info.value.retryable is True


def test_invalid_configuration_preserves_the_gateways_own_message():
    """That message names the conflicting entity and its owner — the only way
    sparkrun can explain a collision with a set it cannot read."""
    detail = 'alias "fast" owned by "sparkrun" conflicts with virtual model "fast" owned by "operator"'
    with mock.patch.object(admin_mod.urllib.request, "urlopen", side_effect=_http_error(400, "invalid_configuration", detail)):
        with pytest.raises(AdminError) as exc_info:
            _client().replace({}, REV_A)
    assert detail in str(exc_info.value)
    assert exc_info.value.retryable is False


def test_auth_failure_does_not_echo_the_response_body():
    with mock.patch.object(admin_mod.urllib.request, "urlopen", side_effect=_http_error(403, "forbidden", "sk-reconciler-secret")):
        with pytest.raises(AdminError) as exc_info:
            _client().active_revision()
    assert "sk-reconciler-secret" not in str(exc_info.value)
    assert exc_info.value.retryable is False


def test_transport_failure_is_retryable_and_hides_request_context():
    with mock.patch.object(admin_mod.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(AdminError) as exc_info:
            _client().active_revision()
    assert exc_info.value.retryable is True
    assert "sk-reconciler-secret" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Credential
# ---------------------------------------------------------------------------


def _created(secret="test-only-reconciler-key"):
    """The credential command's real payload shape (``IssuedCredential``)."""
    return subprocess.CompletedProcess(
        [],
        0,
        stdout=json.dumps({"credential": {"id": "cc_abc", "roles": ["config_reconcile:sparkrun", "status_read"]}, "api_key": secret}),
        stderr="",
    )


def test_credential_is_minted_offline_and_stored_owner_only(tmp_path):
    credential = ReconcilerCredential(tmp_path)
    with mock.patch.object(cred_mod.subprocess, "run", return_value=_created()) as run:
        assert credential.ensure(tmp_path / "llm-gateway") == "test-only-reconciler-key"

    cmd = run.call_args.args[0]
    assert cmd[1:3] == ["client-credentials", "create"]
    assert "config_reconcile:sparkrun,status_read" in cmd
    assert stat.S_IMODE(credential.secret_file.stat().st_mode) == 0o600
    assert credential.read_secret() == "test-only-reconciler-key"


def test_the_secret_never_appears_in_argv(tmp_path):
    credential = ReconcilerCredential(tmp_path)
    with mock.patch.object(cred_mod.subprocess, "run", return_value=_created("sk-topsecret")) as run:
        credential.ensure(tmp_path / "llm-gateway")
    assert "sk-topsecret" not in " ".join(run.call_args.args[0])


def test_existing_credential_is_reused_but_force_rotates(tmp_path):
    credential = ReconcilerCredential(tmp_path)
    with mock.patch.object(cred_mod.subprocess, "run", return_value=_created("first")):
        credential.ensure(tmp_path / "llm-gateway")

    with mock.patch.object(cred_mod.subprocess, "run") as run:
        assert credential.ensure(tmp_path / "llm-gateway") == "first"
    run.assert_not_called()

    with mock.patch.object(cred_mod.subprocess, "run", return_value=_created("second")):
        assert credential.ensure(tmp_path / "llm-gateway", force=True) == "second"
    assert credential.read_secret() == "second"


def test_live_admin_token_can_enable_rotate_and_disable_without_a_binary(tmp_path):
    credential = ReconcilerCredential(tmp_path)

    assert credential.read_admin_token() is None
    first = credential.ensure_admin_token()
    second = credential.ensure_admin_token(force=True)

    assert first.startswith("sr_admin_v1.")
    assert second != first
    assert credential.read_admin_token() == second
    assert stat.S_IMODE(credential.admin_token_file.stat().st_mode) == 0o600

    credential.clear_admin_token()
    assert credential.read_admin_token() is None


def test_master_key_caller_token_is_private_and_can_be_removed(tmp_path):
    credential = ReconcilerCredential(tmp_path)

    credential.set_caller_token("sk-master")
    assert credential.caller_token_file.read_text().strip() == "sk-master"
    assert stat.S_IMODE(credential.caller_token_file.stat().st_mode) == 0o600

    credential.set_caller_token(None)
    assert not credential.caller_token_file.exists()


def test_live_token_rejects_whitespace(tmp_path):
    with pytest.raises(CredentialError, match="without whitespace"):
        ReconcilerCredential(tmp_path).ensure_admin_token(preferred="not valid")


def test_admin_token_is_adopted_stored_and_reused(tmp_path):
    credential = ReconcilerCredential(tmp_path)
    with mock.patch.object(cred_mod.subprocess, "run", return_value=_created("llmgw_v1.cc_admin.first")) as run:
        assert credential.ensure_admin(tmp_path / "sparkroute") == "llmgw_v1.cc_admin.first"

    cmd = run.call_args.args[0]
    assert cmd[1:3] == ["client-credentials", "ensure"]
    assert "credentials_read" in cmd[-1] and "credentials_write" in cmd[-1]
    assert credential.read_admin_secret() == "llmgw_v1.cc_admin.first"
    assert stat.S_IMODE(credential.admin_secret_file.stat().st_mode) == 0o600

    with mock.patch.object(cred_mod.subprocess, "run") as run:
        assert credential.ensure_admin(tmp_path / "sparkroute") == "llmgw_v1.cc_admin.first"
    run.assert_not_called()


def test_admin_token_set_rotates_the_same_credential(tmp_path):
    credential = ReconcilerCredential(tmp_path)
    credential._store_secret("llmgw_v1.cc_admin.first", credential.admin_secret_file)
    with mock.patch.object(cred_mod.subprocess, "run", return_value=_created("llmgw_v1.cc_admin.second")) as run:
        assert credential.ensure_admin(tmp_path / "sparkroute", force=True) == "llmgw_v1.cc_admin.second"

    cmd = run.call_args.args[0]
    assert cmd[1:3] == ["client-credentials", "rotate"]
    assert cmd[-2:] == ["-credential-id", "cc_admin"]
    assert credential.read_admin_secret() == "llmgw_v1.cc_admin.second"


def test_a_failed_credential_command_is_reported(tmp_path):
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="database is locked")
    with mock.patch.object(cred_mod.subprocess, "run", return_value=failed):
        with pytest.raises(CredentialError, match="database is locked"):
            ReconcilerCredential(tmp_path).ensure(tmp_path / "llm-gateway")


def test_reset_removes_config_db_and_sidecars_but_not_credentials(tmp_path):
    credential = ReconcilerCredential(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    for name in ("config.db", "config.db-wal", "config.db-shm", "client-credentials.db"):
        (tmp_path / name).write_text("x")

    removed = credential.reset_configuration()

    assert {p.name for p in removed} == {"config.db", "config.db-wal", "config.db-shm"}
    # Removing this would strand the stored secret while leaving the gateway
    # unable to authenticate sparkrun.
    assert credential.credentials_db.exists()


def test_only_the_documented_secret_field_is_accepted(tmp_path):
    """A near-miss on a credential field must fail loudly, not silently pick
    something plausible — so extraction reads ``api_key`` and nothing else."""
    wrong = subprocess.CompletedProcess([], 0, stdout=json.dumps({"secret": "not-the-field"}), stderr="")
    with mock.patch.object(cred_mod.subprocess, "run", return_value=wrong):
        with pytest.raises(CredentialError, match="no secret"):
            ReconcilerCredential(tmp_path).ensure(tmp_path / "llm-gateway")
