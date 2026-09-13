# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Client for the gateway's managed-configuration admin API.

Sparkrun owns exactly one entity set — ``sparkrun`` — and replaces it whole,
under compare-and-swap on the merged active revision. It never reads or writes
the ``operator`` set; the credential's role makes that impossible rather than
merely discouraged.

Stdlib ``urllib`` on purpose, matching ``orchestration/tailscale/api.py``:
sparkrun must run on a bare control machine, and this is a handful of JSON
requests to loopback.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

#: The gateway's own request-body bound.
MAX_REQUEST_BYTES = 9 << 20

#: Cap on a response we will buffer.
MAX_RESPONSE_BYTES = 16 << 20

DEFAULT_TIMEOUT_SECONDS = 30.0

#: Owner set sparkrun may write.
OWNER = "sparkrun"


class AdminError(RuntimeError):
    """An admin API call failed.

    :attr:`code` carries the gateway's error code when it sent one, which is
    what callers branch on — notably ``revision_conflict``.
    """

    def __init__(self, message: str, *, status: int = 0, code: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.code = code

    @property
    def retryable(self) -> bool:
        """True when re-reading and rebuilding is the right response.

        A 409 is ordinary concurrent control-plane behaviour, not a failure —
        the operator changed something between our read and our write. 5xx and
        transport errors are transient. Everything else (validation, auth,
        role) needs the input or the deployment to change first, so retrying
        unchanged would just spin.
        """
        return self.code == "revision_conflict" or self.status >= 500 or self.status == 0


class RevisionConflict(AdminError):
    """The active revision moved between our read and our write."""


class AdminClient:
    """Authenticated client for one gateway's admin listener."""

    def __init__(self, base_url: str, token: str | None, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout

    # -- Reads --------------------------------------------------------------

    def active_revision(self) -> str:
        """Active merged revision — the CAS token for validate/replace.

        This is *storage* state. It is not a promise that the runtime
        generation for it has finished building; use :meth:`status` for what is
        actually serving.
        """
        return str(self._request("GET", "/v1/config/managed-sets").get("active_revision") or "")

    def get_set(self) -> dict[str, Any]:
        """Read the complete sparkrun set (metadata plus its document)."""
        return self._request("GET", "/v1/config/managed-sets/%s" % OWNER)

    def status(self) -> dict[str, Any]:
        """Serving revision and served model names (needs ``status_read``).

        ``served_model_names`` is a content-free sorted set of canonical names
        and aliases across *both* owner sets — the only way sparkrun can answer
        "what can I call?" completely, since its reconcile role cannot read the
        operator document.
        """
        return self._request("GET", "/v1/status")

    # -- Writes -------------------------------------------------------------

    def validate(self, document: dict[str, Any], expected_active_revision: str) -> dict[str, Any]:
        """Dry-run a replacement. Writes no revision, activation or audit event.

        Success here does not reserve anything — the following PUT can still
        conflict — so this buys a clearer error, not a guarantee.
        """
        return self._request(
            "POST",
            "/v1/config/managed-sets/%s/validate" % OWNER,
            {"document": document, "expected_active_revision": expected_active_revision},
        )

    def replace(self, document: dict[str, Any], expected_active_revision: str, reason: str = "") -> dict[str, Any]:
        """Atomically replace the whole sparkrun set.

        Idempotent by content: submitting what is already stored returns
        ``changed: false`` and creates no revision or activation.
        """
        payload: dict[str, Any] = {"document": document, "expected_active_revision": expected_active_revision}
        if reason:
            payload["reason"] = reason[:4096]
        return self._request("PUT", "/v1/config/managed-sets/%s" % OWNER, payload)

    # -- Transport ----------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            if len(body) > MAX_REQUEST_BYTES:
                raise AdminError("generated configuration exceeds the gateway's %d-byte request limit" % MAX_REQUEST_BYTES)

        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        if self._token:
            request.add_header("Authorization", "Bearer %s" % self._token)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - loopback admin API
                raw = response.read(MAX_RESPONSE_BYTES)
        except urllib.error.HTTPError as exc:
            raise self._error_from_response(exc) from None
        except (urllib.error.URLError, OSError) as exc:
            # Never render the exception's request context: it carries the
            # Authorization header.
            raise AdminError("gateway admin API is unreachable at %s" % self.base_url) from exc

        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdminError("gateway admin API returned a malformed response") from exc
        return decoded if isinstance(decoded, dict) else {}

    def _error_from_response(self, exc: "urllib.error.HTTPError") -> AdminError:
        """Translate an HTTP error, preserving the gateway's own diagnostic.

        The gateway's validation messages name the conflicting entity and its
        owner — the only way a user can be told that an alias collides with
        something in the operator set, which sparkrun cannot read.
        """
        code = ""
        message = ""
        try:
            payload = json.loads(exc.read(MAX_RESPONSE_BYTES))
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                code = str(error.get("code") or "")
                message = str(error.get("message") or "")
        except Exception:
            logger.debug("Could not decode gateway admin error body", exc_info=True)

        detail = message or "gateway admin API returned HTTP %d" % exc.code
        if code == "revision_conflict":
            return RevisionConflict(detail, status=exc.code, code=code)
        if exc.code in (401, 403):
            # Do not echo the body for auth failures; keep the secret out of
            # any surface that might be logged.
            return AdminError(
                "gateway admin API rejected sparkrun's credential (HTTP %d); replace it live with "
                "'sparkrun proxy admin-token set'" % exc.code,
                status=exc.code,
                code=code or "invalid_api_key",
            )
        return AdminError(detail, status=exc.code, code=code)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_REQUEST_BYTES",
    "OWNER",
    "AdminClient",
    "AdminError",
    "RevisionConflict",
]
