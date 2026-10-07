# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Operation-scoped ColdSnap host-provider backed by sparkrun transports."""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import shutil
import socketserver
import tempfile
import time
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Self

from sparkrun.core.config import resolve_hf_token
from sparkrun.core.offline import resolve_offline
from sparkrun.orchestration.primitives import build_ssh_kwargs
from sparkrun.transports import open_cluster_host_session
from sparkrun.transports.session import HostCommandResult, HostSessionError

from .manager_runtime import DockerManagerRuntime, RuntimeOperationError
from .runtime_contract import validate_runtime_request

logger = logging.getLogger(__name__)

PROTOCOL_FORMAT = 1
MAXIMUM_MESSAGE_BYTES = 64 << 20
CAPABILITIES = (
    "exec",
    "huggingface-publish",
    "huggingface-resolve",
    "oci-pull",
    "oci-push",
    "runtime-v1",
    "upload",
)
REQUEST_FIELDS = frozenset(
    {
        "format",
        "id",
        "token",
        "session",
        "operation",
        "host",
        "arguments",
        "input",
        "combined",
        "sources",
        "destination",
        "recursive",
        "image",
        "repository",
        "revision",
        "source",
        "runtime",
    }
)

_HF_UPLOAD_PROGRAM = """import os, pathlib, shutil, subprocess, sys
token = sys.stdin.read().strip()
if not token:
    raise SystemExit("controller supplied no Hugging Face token")
executable = shutil.which("hf")
if not executable:
    raise SystemExit("Hugging Face CLI 'hf' is unavailable in the capsule")
environment = os.environ.copy()
environment["HF_TOKEN"] = token
xet_cache = pathlib.Path("/tmp/coldsnap-hf-xet")
xet_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
environment["HF_HUB_OFFLINE"] = "0"
environment["HF_XET_CACHE"] = str(xet_cache)
environment["HF_XET_HIGH_PERFORMANCE"] = "1"
subprocess.run(
    [executable, "upload", sys.argv[1], "/coldsnap-upload/native.pack", sys.argv[3],
     "--repo-type", "model", "--revision", sys.argv[2]],
    check=True,
    env=environment,
)
"""

_HF_REVISION_PROGRAM = """import sys
from huggingface_hub import HfApi
token = sys.stdin.read().strip()
if not token:
    raise SystemExit("controller supplied no Hugging Face token")
print(HfApi(token=token).repo_info(repo_id=sys.argv[1], revision=sys.argv[2], repo_type="model").sha)
"""


class _ThreadingUnixServer(socketserver.ThreadingUnixStreamServer):
    provider: ColdSnapHostProvider
    daemon_threads = True
    block_on_close = False


class _ProviderHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline(MAXIMUM_MESSAGE_BYTES + 1)
        request: Any = None
        try:
            if not line or len(line) > MAXIMUM_MESSAGE_BYTES:
                raise RuntimeError(f"host-provider request exceeds {MAXIMUM_MESSAGE_BYTES} bytes")
            if not line.endswith(b"\n"):
                raise RuntimeError("host-provider request is not newline-terminated")
            if self.rfile.read(1):
                raise RuntimeError("host-provider request has trailing data")
            request = json.loads(line)
            response = self.server.provider.dispatch(request)  # type: ignore[attr-defined]
        except Exception as error:  # noqa: BLE001 - protocol boundary
            request_id = request.get("id", "") if isinstance(request, Mapping) else ""
            response = {"format": PROTOCOL_FORMAT, "id": request_id, "ok": False, "error": str(error)}
            if isinstance(error, RuntimeOperationError):
                response["error_code"] = error.code
        try:
            payload = json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n"
            if len(payload) > MAXIMUM_MESSAGE_BYTES:
                payload = (
                    json.dumps(
                        {
                            "format": PROTOCOL_FORMAT,
                            "id": response.get("id", ""),
                            "ok": False,
                            "error": f"host-provider response exceeds {MAXIMUM_MESSAGE_BYTES} bytes",
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ColdSnapHostProvider:
    """Context manager exposing one request's authorized host session."""

    def __init__(
        self,
        request: Mapping[str, Any],
        *,
        sctx,
        cluster=None,
        session_factory=None,
        capabilities=None,
        runtime_factory=None,
        offline: bool | None = None,
    ):
        self.request_id = str(request.get("id") or "")
        units = request.get("launch", {}).get("units", [])
        self.hosts = frozenset(str(unit.get("host") or "") for unit in units if isinstance(unit, Mapping))
        if not self.request_id or not self.hosts or "" in self.hosts:
            raise RuntimeError("ColdSnap host-provider request identity is incomplete")
        config = sctx.config
        if cluster is not None and callable(getattr(config, "for_cluster", None)):
            config = config.for_cluster(cluster)
        elif cluster is not None and getattr(cluster, "user", None):
            config = copy(config)
            config.ssh_user = cluster.user
        ssh_kwargs = build_ssh_kwargs(config)
        factory = session_factory or open_cluster_host_session
        self.session = factory(cluster, ssh_kwargs=ssh_kwargs)
        try:
            self.runtime = (
                runtime_factory(self.session)
                if runtime_factory is not None
                else DockerManagerRuntime(
                    self.session,
                    config=config,
                    ssh_kwargs=ssh_kwargs,
                    offline=resolve_offline(offline, cluster).offline,
                )
            )
        except BaseException:
            self.session.close()
            raise
        self.workload = dict(request.get("workload") or {})
        selected_capabilities = tuple(CAPABILITIES if capabilities is None else capabilities)
        if not selected_capabilities or any(
            not isinstance(capability, str) or not capability or any(character in capability for character in "\r\n\x00")
            for capability in selected_capabilities
        ):
            self.session.close()
            raise RuntimeError("ColdSnap host-provider capabilities are invalid")
        self.capabilities = tuple(dict.fromkeys(selected_capabilities))
        self.token = secrets.token_urlsafe(32)
        # macOS TMPDIR can leave too little room for a Unix socket (104 bytes).
        # A short, randomly named 0700 directory preserves the private boundary.
        self._directory = Path(tempfile.mkdtemp(prefix="coldsnap-", dir="/tmp"))
        self.socket = self._directory / "provider.sock"
        self._closed = False
        self._metrics_lock = Lock()
        self._operation_counts: dict[str, int] = {}
        self._operation_seconds: dict[str, float] = {}
        try:
            self.server = _ThreadingUnixServer(str(self.socket), _ProviderHandler)
        except BaseException:
            self.session.close()
            shutil.rmtree(self._directory, ignore_errors=True)
            raise
        self.server.provider = self
        os.chmod(self.socket, 0o600)
        self.thread = Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.05),
            name="coldsnap-host-provider",
            daemon=True,
        )
        self.thread.start()
        logger.info(
            "ColdSnap host provider: %s session=%s hosts=%s",
            getattr(self.session, "provider_name", type(self.session).__name__),
            self.request_id,
            ",".join(sorted(self.hosts)),
        )

    @property
    def environment(self) -> dict[str, str]:
        return {
            "COLDSNAP_HOST_PROVIDER": "external",
            "COLDSNAP_HOST_PROVIDER_SOCKET": str(self.socket),
            "COLDSNAP_HOST_PROVIDER_TOKEN": self.token,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.shutdown()
        self.server.server_close()
        self.session.close()
        self.thread.join(timeout=2)
        shutil.rmtree(self._directory, ignore_errors=True)
        with self._metrics_lock:
            summary = ", ".join(
                f"{operation}={self._operation_counts[operation]}x/{self._operation_seconds[operation]:.3f}s"
                for operation in sorted(self._operation_counts)
            )
        if summary:
            logger.info("ColdSnap host provider completed: %s", summary)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("host-provider request must be an object")
        unknown = set(request) - REQUEST_FIELDS
        if unknown:
            raise RuntimeError("host-provider request has unknown fields: " + ", ".join(sorted(unknown)))
        request_id = request.get("id")
        if request.get("format") != PROTOCOL_FORMAT or not isinstance(request_id, str) or not request_id:
            raise RuntimeError("host-provider request identity is invalid")
        if request.get("token") != self.token or request.get("session") != self.request_id:
            raise RuntimeError("host-provider session is unauthorized")
        operation = request.get("operation")
        if not isinstance(operation, str) or not operation:
            raise RuntimeError("host-provider operation is invalid")
        started = time.monotonic()
        try:
            return self._dispatch_operation(request_id, operation, request)
        finally:
            seconds = time.monotonic() - started
            with self._metrics_lock:
                self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
                self._operation_seconds[operation] = self._operation_seconds.get(operation, 0.0) + seconds

    def _dispatch_operation(self, request_id: str, operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if operation == "capabilities":
            return self._response(
                request_id,
                provider=getattr(self.session, "provider_name", type(self.session).__name__),
                capabilities=list(self.capabilities),
            )
        if ("runtime-v1" if operation == "runtime" else operation) not in self.capabilities:
            raise RuntimeError(f"unsupported host-provider operation {operation!r}")
        host = request.get("host")
        if not isinstance(host, str) or host not in self.hosts:
            raise RuntimeError("host-provider target is outside this request")
        logger.debug("ColdSnap host provider: %s host=%s", operation, host)
        if operation == "exec":
            arguments = self._strings(request.get("arguments"), "arguments")
            input_data = self._bytes(request.get("input"))
            result = self.session.execute(
                host,
                arguments,
                input_data=input_data,
                combined=request.get("combined") is True,
            )
            return self._command_response(request_id, result)
        if operation == "upload":
            self.session.upload(
                host,
                self._strings(request.get("sources"), "sources"),
                self._string(request.get("destination"), "destination"),
                recursive=request.get("recursive") is True,
            )
            return self._response(request_id)
        if operation == "runtime":
            operation_spec = self._mapping(request.get("runtime"), "runtime")
            validate_runtime_request(operation_spec)
            registry_capability = {"image-pull": "oci-pull", "image-push": "oci-push"}.get(operation_spec["action"])
            if registry_capability and registry_capability not in self.capabilities:
                raise RuntimeError(f"unsupported host-provider capability {registry_capability!r}")
            if operation_spec["action"] == "workload-run":
                operation_spec = {**operation_spec, "workload": self._workload_metadata(operation_spec["workload"])}
            result = self.runtime.invoke(host, operation_spec)
            return self._response(request_id, runtime=result)
        if operation in {"oci-pull", "oci-push"}:
            self.runtime.invoke(host, {"action": operation.replace("oci-", "image-"), "image": self._string(request.get("image"), "image")})
            return self._response(request_id)
        if operation == "huggingface-publish":
            result = self._huggingface_publish(host, request)
            if result.returncode:
                raise HostSessionError(self._command_failure("Hugging Face publication", result))
            return self._response(request_id)
        if operation == "huggingface-resolve":
            result = self._huggingface_resolve(host, request)
            if result.returncode:
                raise HostSessionError(self._command_failure("Hugging Face revision", result))
            return self._response(request_id, value=result.stdout.decode("utf-8", errors="strict").strip())
        raise RuntimeError(f"unimplemented host-provider capability {operation!r}")

    def _workload_metadata(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        labels = dict(spec.get("labels") or {})
        cluster_id = self.workload.get("cluster_id")
        if cluster_id and labels.get("io.sparksq.coldsnap.workload") == cluster_id:
            for field in ("cluster_id", "intent_id", "recipe", "runtime", "model", "served_model_name"):
                labels["sparkrun." + field] = str(self.workload.get(field) or "")
            labels["sparkrun.rank"] = labels.get("io.sparksq.coldsnap.rank", "")
        return {**spec, "labels": labels}

    def _huggingface_publish(self, host: str, request: Mapping[str, Any]) -> HostCommandResult:
        source = self._string(request.get("source"), "source")
        return self._huggingface_helper(
            host,
            request,
            _HF_UPLOAD_PROGRAM,
            [
                self._string(request.get("repository"), "repository"),
                self._string(request.get("revision"), "revision"),
                self._string(request.get("destination"), "destination"),
            ],
            mounts=[{"source": source, "target": "/coldsnap-upload/native.pack", "read_only": True}],
        )

    def _huggingface_resolve(self, host: str, request: Mapping[str, Any]) -> HostCommandResult:
        return self._huggingface_helper(
            host,
            request,
            _HF_REVISION_PROGRAM,
            [
                self._string(request.get("repository"), "repository"),
                self._string(request.get("revision"), "revision"),
            ],
        )

    def _huggingface_helper(self, host, request, program, arguments, *, mounts=None):
        # Credentials travel on stdin; never store them in workload environment or labels.
        result = self.runtime.invoke(
            host,
            {
                "action": "workload-run",
                "workload": {
                    "image": self._string(request.get("image"), "image"),
                    "remove_after_exit": True,
                    "entrypoint": "python3",
                    "command": ["-c", program, *arguments],
                    "mounts": mounts or [],
                    "input": base64.b64encode(self._huggingface_token().encode("utf-8")).decode("ascii"),
                },
            },
        )
        return HostCommandResult(host, 0, base64.b64decode(result.get("output") or "", validate=True), b"")

    @staticmethod
    def _response(request_id: str, **values) -> dict[str, Any]:
        return {"format": PROTOCOL_FORMAT, "id": request_id, "ok": True, **values}

    def _command_response(self, request_id: str, result: HostCommandResult) -> dict[str, Any]:
        return self._response(
            request_id,
            exit_code=result.returncode,
            output=base64.b64encode(result.stdout).decode("ascii") if result.stdout else "",
            error_output=base64.b64encode(result.stderr).decode("ascii") if result.stderr else "",
        )

    @staticmethod
    def _bytes(value: Any) -> bytes | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise TypeError("host-provider input is invalid")
        try:
            return base64.b64decode(value, validate=True)
        except ValueError as error:
            raise RuntimeError("host-provider input is invalid") from error

    @staticmethod
    def _strings(value: Any, name: str) -> list[str]:
        if not isinstance(value, list) or not value or any(not isinstance(item, str) or "\x00" in item for item in value):
            raise RuntimeError(f"host-provider {name} are invalid")
        return value

    @staticmethod
    def _mapping(value: Any, name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{name} is invalid")
        return value

    @staticmethod
    def _string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise RuntimeError(f"host-provider {name} is invalid")
        return value

    @staticmethod
    def _huggingface_token() -> str:
        token = resolve_hf_token()
        if not token or any(character in token for character in "\r\n\x00"):
            raise RuntimeError("ColdSnap native publication requires Hugging Face authentication")
        return token

    @staticmethod
    def _command_failure(operation: str, result: HostCommandResult) -> str:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        suffix = ": " + detail[-2000:] if detail else ""
        return f"{operation} on {result.host} exited {result.returncode}{suffix}"


__all__ = ["CAPABILITIES", "PROTOCOL_FORMAT", "ColdSnapHostProvider"]
