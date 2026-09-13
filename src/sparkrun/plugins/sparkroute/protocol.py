# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Strict, bounded protocol types for ``sparkrun gateway-bridge``."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, BinaryIO

#: Current bridge schema. This unreleased integration upgrades both sides together.
PROTOCOL_VERSION = 4
SUPPORTED_VERSIONS = (PROTOCOL_VERSION,)

SUPPORTED_OPERATIONS = (
    "capabilities",
    "resolve",
    "ensure_ready",
    "discover",
    "status",
    "stop",
    "sleep",
    "wake",
    "workload_status",
    "workloads",
    "catalog_registries",
    "catalog_clusters",
    "catalog_search",
    "catalog_resolve",
    "catalog_retain",
    "catalog_import",
    "catalog_refresh",
    "catalog_registry",
    "catalog_capacity",
    "catalog_plugins",
    "operation_status",
)

MAX_REQUEST_BYTES = 1 << 20
MAX_REQUEST_ID_BYTES = 128
MAX_RECIPE_BYTES = 4096
MAX_REVISION_BYTES = 1024
MAX_CLUSTER_CANDIDATES = 64
MAX_OVERRIDE_ENTRIES = 128
MAX_OVERRIDE_BYTES = 64 << 10
MAX_TIMEOUT_SECONDS = 3600.0

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")


class ProtocolError(Exception):
    """An expected request or operation failure safe to serialize."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, schema_version: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.request_id = ""
        #: Version the *caller* asked for, when it could be recovered.  The
        #: error response is emitted at this version so a strict client's
        #: correlation check passes and it reads the error rather than a
        #: version mismatch that would hide the diagnostic.
        self.schema_version = schema_version


@dataclass(frozen=True)
class Binding:
    recipe: str
    recipe_revision: str = ""
    cluster_candidates: tuple[str, ...] = ()
    overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    request_id: str
    operation: str
    schema_version: int = PROTOCOL_VERSION
    binding: Binding | None = None
    cluster_id: str = ""
    timeout_seconds: float = 900.0
    arguments: dict[str, Any] = field(default_factory=dict)
    wait: bool = True
    """False starts a recoverable background activation and returns its operation ID."""


def read_request(stream: BinaryIO) -> Request:
    raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ProtocolError("request_too_large", "request exceeds the bridge size limit")
    if not raw:
        raise ProtocolError("empty_request", "request body is required")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid_json", "request must be one JSON object") from exc
    try:
        return parse_request(value)
    except ProtocolError as exc:
        if isinstance(value, dict):
            request_id = value.get("request_id")
            if isinstance(request_id, str) and 0 < len(request_id.encode()) <= MAX_REQUEST_ID_BYTES:
                exc.request_id = request_id
            version = value.get("schema_version")
            if isinstance(version, int) and not isinstance(version, bool):
                exc.schema_version = version
        raise


def parse_request(value: Any) -> Request:
    if not isinstance(value, dict):
        raise ProtocolError("invalid_request", "request must be a JSON object")

    # Version first, unknown fields second.  A request from a newer schema will
    # carry fields this one has never heard of; reporting those as "unknown
    # fields" would point the caller at the wrong problem when the answer is
    # simply that we do not speak its version yet.
    requested = value.get("schema_version")
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise ProtocolError("invalid_request", "schema_version must be an integer")
    if requested not in SUPPORTED_VERSIONS:
        raise ProtocolError(
            "unsupported_version",
            "unsupported bridge schema version %d; this sparkrun serves %s" % (requested, ", ".join(str(v) for v in SUPPORTED_VERSIONS)),
            schema_version=requested,
        )

    unknown = set(value) - {
        "schema_version",
        "request_id",
        "operation",
        "binding",
        "cluster_id",
        "timeout_seconds",
        "wait",
        "arguments",
    }
    if unknown:
        raise ProtocolError("invalid_request", "request contains unknown fields", schema_version=requested)

    request_id = _required_string(value.get("request_id"), "request_id", MAX_REQUEST_ID_BYTES)
    operation = _required_string(value.get("operation"), "operation", 64)
    if operation not in SUPPORTED_OPERATIONS:
        raise ProtocolError("unsupported_operation", "unsupported bridge operation")

    binding_value = value.get("binding")
    binding = _parse_binding(binding_value) if binding_value is not None else None
    if operation in {"resolve", "ensure_ready", "status", "stop", "sleep", "wake", "workload_status"} and binding is None:
        raise ProtocolError("invalid_request", "operation requires a binding")

    cluster_id = _optional_string(value.get("cluster_id"), "cluster_id", 1024)
    # Same character class the cluster candidates are held to.  It is only ever
    # compared against known ids rather than interpolated anywhere, but an
    # identifier field on a machine boundary should not be the one place that
    # accepts arbitrary text.
    if cluster_id and not _SAFE_ID_RE.fullmatch(cluster_id):
        raise ProtocolError("invalid_request", "cluster_id is invalid")

    timeout_value = value.get("timeout_seconds", 900.0)
    if isinstance(timeout_value, bool) or not isinstance(timeout_value, (int, float)):
        raise ProtocolError("invalid_request", "timeout_seconds must be a number")
    timeout_seconds = float(timeout_value)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise ProtocolError("invalid_request", "timeout_seconds is outside the supported range")
    if cluster_id and operation not in {"status", "stop", "sleep", "wake", "workload_status"}:
        raise ProtocolError("invalid_request", "cluster_id is not valid for this operation")

    wait_value = value.get("wait", True)
    if not isinstance(wait_value, bool):
        raise ProtocolError("invalid_request", "wait must be a boolean")
    if not wait_value and operation != "ensure_ready":
        raise ProtocolError("invalid_request", "wait is not valid for this operation")

    arguments = value.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ProtocolError("invalid_request", "arguments must be an object")
    allowed = {
        "catalog_registries": set(),
        "catalog_clusters": set(),
        "catalog_search": {"query", "registry", "runtime", "local_only", "offset", "limit", "filters"},
        "catalog_resolve": {"reference", "overrides"},
        "catalog_import": {"content"},
        "catalog_retain": {"reference"},
        "catalog_refresh": set(),
        "catalog_registry": {"action", "name", "url", "subpath", "acknowledge_trust"},
        "catalog_capacity": {"cluster"},
        "catalog_plugins": set(),
        "operation_status": {"operation_id"},
    }.get(operation, set())
    if set(arguments) - allowed:
        raise ProtocolError("invalid_request", "operation arguments contain unknown fields")
    for key in ("query", "registry", "runtime", "reference", "operation_id", "action", "name", "url", "cluster"):
        if key in arguments:
            _required_string(arguments[key], key, 4096 if key == "reference" else 256)
    for key in ("offset", "limit"):
        if key in arguments and (isinstance(arguments[key], bool) or not isinstance(arguments[key], int)):
            raise ProtocolError("invalid_request", "catalog page must be an integer")
    if "filters" in arguments and (
        not isinstance(arguments["filters"], dict)
        or len(arguments["filters"]) > 8
        or any(not isinstance(k, str) or not isinstance(v, str) or len(v) > 128 for k, v in arguments["filters"].items())
    ):
        raise ProtocolError("invalid_request", "recipe filters are invalid")
    if "acknowledge_trust" in arguments and not isinstance(arguments["acknowledge_trust"], bool):
        raise ProtocolError("invalid_request", "trust acknowledgement must be a boolean")
    if "subpath" in arguments and (not isinstance(arguments["subpath"], str) or len(arguments["subpath"]) > 1024):
        raise ProtocolError("invalid_request", "registry subpath is invalid")
    if "local_only" in arguments and not isinstance(arguments["local_only"], bool):
        raise ProtocolError("invalid_request", "local_only must be a boolean")
    if "content" in arguments and (not isinstance(arguments["content"], str) or len(arguments["content"].encode()) > 256 * 1024):
        raise ProtocolError("invalid_request", "recipe import exceeds its size limit")
    if "overrides" in arguments:
        _parse_binding({"recipe": "validation", "overrides": arguments["overrides"]})

    return Request(
        arguments=arguments,
        schema_version=requested,
        request_id=request_id,
        operation=operation,
        binding=binding,
        cluster_id=cluster_id,
        timeout_seconds=timeout_seconds,
        wait=wait_value,
    )


def success_response(request_id: str, result: dict[str, Any], schema_version: int = PROTOCOL_VERSION) -> dict[str, Any]:
    """Build a success envelope at the accepted request's schema version."""
    return {
        "schema_version": schema_version,
        "request_id": request_id,
        "ok": True,
        "result": result,
    }


def error_response(request_id: str, error: ProtocolError, schema_version: int | None = None) -> dict[str, Any]:
    """Build an error envelope, preferring the version the caller asked for."""
    return {
        "schema_version": schema_version or error.schema_version or PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        },
    }


def _parse_binding(value: Any) -> Binding:
    if not isinstance(value, dict):
        raise ProtocolError("invalid_request", "binding must be a JSON object")
    unknown = set(value) - {"recipe", "recipe_revision", "cluster_candidates", "overrides"}
    if unknown:
        raise ProtocolError("invalid_request", "binding contains unknown fields")

    recipe = _required_string(value.get("recipe"), "binding.recipe", MAX_RECIPE_BYTES)
    revision = _optional_string(value.get("recipe_revision"), "binding.recipe_revision", MAX_REVISION_BYTES)

    raw_candidates = value.get("cluster_candidates", [])
    if not isinstance(raw_candidates, list) or len(raw_candidates) > MAX_CLUSTER_CANDIDATES:
        raise ProtocolError("invalid_request", "binding.cluster_candidates is invalid")
    candidates: list[str] = []
    seen: set[str] = set()
    for item in raw_candidates:
        candidate = _required_string(item, "binding.cluster_candidates[]", 1024)
        if not _SAFE_ID_RE.fullmatch(candidate) or candidate in seen:
            raise ProtocolError("invalid_request", "binding.cluster_candidates contains an invalid value")
        seen.add(candidate)
        candidates.append(candidate)

    raw_overrides = value.get("overrides", {})
    if not isinstance(raw_overrides, dict) or len(raw_overrides) > MAX_OVERRIDE_ENTRIES:
        raise ProtocolError("invalid_request", "binding.overrides is invalid")
    overrides: dict[str, str] = {}
    override_bytes = 0
    for raw_key, raw_value in raw_overrides.items():
        key = _required_string(raw_key, "binding.overrides key", 1024)
        override = _optional_string(raw_value, "binding.overrides value", 4096)
        if not _SAFE_ID_RE.fullmatch(key):
            raise ProtocolError("invalid_request", "binding.overrides contains an invalid key")
        override_bytes += len(key.encode()) + len(override.encode())
        if override_bytes > MAX_OVERRIDE_BYTES:
            raise ProtocolError("invalid_request", "binding.overrides exceeds the size limit")
        overrides[key] = override

    return Binding(
        recipe=recipe,
        recipe_revision=revision,
        cluster_candidates=tuple(candidates),
        overrides=overrides,
    )


def _required_string(value: Any, name: str, maximum: int) -> str:
    result = _optional_string(value, name, maximum)
    if not result:
        raise ProtocolError("invalid_request", "%s is required" % name)
    return result


def _optional_string(value: Any, name: str, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProtocolError("invalid_request", "%s must be a string" % name)
    if len(value.encode()) > maximum or any(ord(char) < 0x20 for char in value):
        raise ProtocolError("invalid_request", "%s is invalid" % name)
    return value
