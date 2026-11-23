"""ASGI transport over the framework-free operation and worker contracts."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import ipaddress
import json
import logging
import os
from pathlib import Path
import queue
import signal
import socket
import struct
import sys
import time
from typing import Any

import anyio
from anyio.to_thread import run_sync
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from sparkrun.api._wire import error_body, check_result
from sparkrun.core.daemon_registry import get_daemon_op, operation_catalog, resolve_route_operation
from sparkrun.core.operation import OperationState, operation_scope
from sparkrun.daemon.auth import Principal, TokenStore
from sparkrun.daemon.jobs import Conflict, JobStore, ReplayGap, TERMINAL
from sparkrun.daemon.ops import check_request, register_core_ops
from sparkrun.daemon.state import Binding
from sparkrun.daemon.supervisor import Coordinator

logger = logging.getLogger(__name__)


def public_job(job):
    return {key: value for key, value in job.items() if key not in {"nonce", "worker", "runtime", "arguments"}}


def sse(event):
    prefix = f"id: {event['id']}\n" if "id" in event else ""
    return prefix + "data: " + json.dumps(event, allow_nan=False) + "\n\n"


def status_code(exc):
    from sparkrun.api._errors import (
        JobNotFound,
        RecipeNotFound,
        TrustRejected,
        HostsUnreachable,
        InsufficientCapacity,
        LayoutRequired,
        AmbiguousWorkload,
    )
    from sparkrun.core.coordination import ResourceBusy

    if isinstance(exc, PermissionError):
        return 403
    if isinstance(exc, (TrustRejected,)):
        return 403
    if isinstance(exc, (Conflict, ResourceBusy, InsufficientCapacity, LayoutRequired, AmbiguousWorkload)):
        return 409
    if isinstance(exc, ReplayGap):
        return 410
    if isinstance(exc, (LookupError, JobNotFound, RecipeNotFound)):
        return 404
    if isinstance(exc, HostsUnreachable):
        return 502
    if isinstance(exc, (ValueError, TypeError)):
        return 422
    if isinstance(exc, TimeoutError):
        return 504
    return 500


class Boundary:
    """ASGI auth/body boundary; no client headers can impersonate Unix transport."""

    def __init__(self, app, *, tokens, body_limit=1024 * 1024):
        self.app, self.tokens, self.body_limit = app, tokens, body_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] != "/v1/health":
            try:
                uid = scope.get("sparkrun.peer_uid")
                authorization = dict(scope.get("headers", [])).get(b"authorization", b"").decode("ascii")
                if authorization:
                    if not authorization.startswith("Bearer "):
                        raise PermissionError("Bearer token required")
                    principal = self.tokens.authenticate(authorization[7:])
                elif uid is not None and hasattr(os, "getuid") and uid == os.getuid():
                    principal = Principal("uid:" + str(uid), "admin", True, True)
                else:
                    raise PermissionError("Bearer token required")
                scope["sparkrun.principal"] = principal
            except (PermissionError, UnicodeError):
                response = JSONResponse({"detail": "Authentication required"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
                return await response(scope, receive, send)
        if scope["method"] in {"POST", "PUT", "PATCH"}:
            body = bytearray()
            deadline = time.monotonic() + 30
            while True:
                try:
                    message = await asyncio.wait_for(receive(), timeout=max(0, deadline - time.monotonic()))
                except TimeoutError:
                    return await JSONResponse({"detail": "Request body timed out"}, status_code=408)(scope, receive, send)
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.body_limit:
                    return await JSONResponse({"detail": "Request body exceeds limit"}, status_code=413)(scope, receive, send)
                if not message.get("more_body", False):
                    break
            delivered = False
            original_receive = receive

            async def buffered():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await original_receive()

            receive = buffered
        return await self.app(scope, receive, send)


class Subscriptions:
    """Reserve before producer setup and release after the complete ASGI lifetime."""

    def __init__(self, maximum):
        self.maximum = maximum
        self.active = set()

    def response(self, factory, **kwargs):
        if len(self.active) >= self.maximum:
            raise Conflict("Subscriber limit reached")
        marker = object()
        self.active.add(marker)
        try:
            return SubscriptionResponse(factory(), release=lambda: self.active.discard(marker), **kwargs)
        except BaseException:
            self.active.discard(marker)
            raise


class SubscriptionResponse(StreamingResponse):
    def __init__(self, content, *, release, **kwargs):
        super().__init__(content, media_type="text/event-stream", **kwargs)
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                # A disconnect/send failure can precede the first iteration or
                # leave the generator suspended at a yield. Always close it.
                with anyio.CancelScope(shield=True):
                    close = getattr(self.body_iterator, "aclose", None)
                    if close is not None:
                        await close()
            finally:
                self.release()


def create_app(binding=None, *, worker_supervisor="process", start_workers=True, allow_request_trust=None):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.daemon.streams import MonitorHub

    binding = (binding or Binding.from_config()).prepare()
    config = SparkrunConfig(binding.config_path)
    import math

    from sparkrun.utils.data import positive_integer_setting, float_setting

    counts = {
        name: positive_integer_setting(config.get("daemon." + name, default), key="daemon." + name)
        for name, default in {
            "max_jobs": 4,
            "max_services": 2,
            "max_streams": 64,
            "max_query_threads": 8,
            "max_queued_queries": 32,
            "max_body_bytes": 1024 * 1024,
            "retained_jobs": 1000,
        }.items()
    }
    durations = {}
    for name, default in {"cancel_grace": 15, "worker_start_timeout": 60, "query_timeout": 30, "job_retention_days": 7}.items():
        value = float_setting(config.get("daemon." + name, default), key="daemon." + name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"daemon.{name} must be positive and finite")
        durations[name] = value
    tokens = TokenStore(binding)
    store = JobStore.from_config(binding.directory, config)
    coordinator = Coordinator(
        binding,
        store,
        systemd=worker_supervisor == "systemd",
        max_active=counts["max_jobs"],
        max_services=counts["max_services"],
        cancel_grace=durations["cancel_grace"],
        start_timeout=durations["worker_start_timeout"],
    )
    hub = MonitorHub()
    from sparkrun.daemon.services import Services, QueryCache

    services = Services(store)
    cache = QueryCache()
    stop = asyncio.Event()
    subscriptions = Subscriptions(counts["max_streams"])
    query_limit = anyio.CapacityLimiter(counts["max_query_threads"])
    query_admitted = 0

    @asynccontextmanager
    async def lifespan(app):
        from sparkrun.application import initialize
        from sparkrun.core.runtime_lease import runtime_lease

        with runtime_lease(config, kind="daemon") as runtime:
            initialize(config_path=binding.config_path)
            register_core_ops()
            app.state.runtime = runtime
            from sparkrun.daemon.http_routes import mount_operation_routes

            mount_operation_routes(app, operation_handler, envelope_handler)

            async def reconcile():
                last_prune = 0
                last_revision = None
                while not stop.is_set():
                    try:
                        revision = store.revision()
                        if revision != last_revision:
                            cache.clear()
                            last_revision = revision
                        if start_workers:
                            await asyncio.to_thread(coordinator.tick, runtime)
                        await asyncio.to_thread(
                            services.tick, SparkrunConfig(binding.config_path), runtime, maintenance=coordinator.maintenance
                        )
                        await asyncio.to_thread(hub.prune)
                        if time.monotonic() - last_prune > 60:
                            await asyncio.to_thread(
                                store.prune,
                                age=durations["job_retention_days"] * 86400,
                                count=counts["retained_jobs"],
                            )
                            last_prune = time.monotonic()
                    except Exception:
                        logger.exception("Daemon reconciliation failed")
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=0.5)
                    except TimeoutError:
                        pass

            task = asyncio.create_task(reconcile())
            try:
                yield
            finally:
                stop.set()
                await task
                await asyncio.to_thread(hub.prune, all_sessions=True)

    app = FastAPI(title="Sparkrun daemon", version="1", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store, app.state.coordinator, app.state.binding = store, coordinator, binding
    app.state.stop = stop
    app.state.operation_paths = {}
    app.add_middleware(Boundary, tokens=tokens, body_limit=counts["max_body_bytes"])

    @app.exception_handler(Exception)
    async def failure(request, exc):
        code = status_code(exc)
        if code == 500:
            logger.exception("Daemon request failed", exc_info=exc)
        body = {"detail": str(exc)} if isinstance(exc, PermissionError) else error_body(exc)
        return JSONResponse(body, status_code=code)

    from sparkrun.api._errors import SparkrunError
    from sparkrun.core.coordination import ResourceBusy

    for error_class in (PermissionError, ValueError, LookupError, TimeoutError, SparkrunError, ResourceBusy):
        app.add_exception_handler(error_class, failure)

    def principal(request, scope="read", *, secret=False):
        value = tokens.validate_principal(request.scope["sparkrun.principal"])
        if not value.allows(scope, secret=secret):
            raise PermissionError("Insufficient permission")
        return value

    def job_access(request, identifier):
        value = principal(request)
        job = store.get(identifier)
        if not value.owns(job):
            raise PermissionError("Job belongs to another principal")
        if not value.allows(job["scope"]):
            raise PermissionError("Job requires its operation permission")
        return job

    async def read_json(request):
        try:
            value = await request.json()
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Invalid JSON request") from exc
        if not isinstance(value, dict):
            raise ValueError("Request must be a JSON object")
        return value

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "protocol": 1}

    @app.get("/v1/info")
    async def info(request: Request):
        user = principal(request)
        from sparkrun.core.application_identity import get_controller_identity

        controller = get_controller_identity(SparkrunConfig(binding.config_path))
        return {
            **binding.info(),
            "controller": controller.to_dict(),
            "protocol": 1,
            "runtime": app.state.runtime,
            "restart_required": coordinator.maintenance,
            "manager": user.manager,
        }

    @app.get("/v1/ops")
    async def operations(request: Request):
        user = principal(request)
        return operation_catalog(
            SparkrunConfig(binding.config_path),
            allows=lambda op: user.allows(op.scope, secret=op.secret),
            http_paths=app.state.operation_paths,
        )

    @app.get("/docs")
    async def docs(request: Request):
        principal(request)
        from fastapi.openapi.docs import get_swagger_ui_html

        return get_swagger_ui_html(openapi_url="/openapi.json", title="Sparkrun API")

    @app.post("/v1/ops/{name}")
    async def operation(name: str, request: Request):
        return await dispatch(name, await read_json(request), request)

    def operation_handler(name):
        async def alias(request: Request):
            return await dispatch(name, await read_json(request), request)

        return alias

    async def dispatch(name, arguments, request):
        nonlocal query_admitted
        op = get_daemon_op(name)
        if op.visibility != "public":
            raise LookupError("Unknown public operation: " + name)
        user = principal(request, op.scope, secret=op.secret)
        arguments = check_request(op, arguments)
        current = SparkrunConfig(binding.config_path)
        if op.feature_flag and not current.is_feature_enabled(op.feature_flag):
            raise PermissionError("Operation feature is disabled")
        if arguments.get("trust") and not (
            user.allows("operate")
            and (allow_request_trust if allow_request_trust is not None else current.get("daemon.allow_request_trust", user.local))
        ):
            raise PermissionError("Per-request trust is disabled for this listener")
        from sparkrun.core.runtime_lease import runtime_identity

        if runtime_identity(current) != app.state.runtime:
            coordinator.maintenance = True
        if coordinator.maintenance:
            raise Conflict("Daemon is draining or needs restart before accepting operations")
        if op.kind == "stream":
            raise ValueError("Use a stream endpoint for this operation")
        if op.kind == "job":
            cache.clear()
            from sparkrun.core.runtime_lease import consistent_runtime

            try:
                with tokens.admission(user) as admitted, consistent_runtime(current, app.state.runtime):
                    identifier = store.submit(
                        op,
                        arguments,
                        admitted.id,
                        request.headers.get("X-Sparkrun-Client", ""),
                        app.state.runtime,
                        key=request.headers.get("Idempotency-Key"),
                        manager=admitted.manager,
                    )
            except RuntimeError as exc:
                coordinator.maintenance = True
                raise Conflict(str(exc)) from exc
            return JSONResponse({"job_id": identifier, "events_url": f"/v1/jobs/{identifier}/events"}, status_code=202)
        if query_admitted >= counts["max_queued_queries"]:
            raise Conflict("Query admission limit reached")
        cache_key = (
            op.name,
            json.dumps(arguments, sort_keys=True),
            user.id,
            user.manager,
            current.config_path.stat().st_mtime_ns if current.config_path.exists() else 0,
        )
        cached = cache.get(cache_key) if op.cacheable else None
        if cached is not None:
            return cached
        query_admitted += 1

        def query():
            from sparkrun.application import initialize

            sctx = initialize(config_path=binding.config_path)
            state = OperationState(deadline=time.monotonic() + durations["query_timeout"], via_daemon=True, manager=user.manager)
            with operation_scope(sctx, state=state):
                result = check_result(op, op.handler_loader()(arguments, sctx, None))
                state.check()
                if op.cacheable:
                    cache.put(cache_key, result)
                return result

        try:
            # Shield until completion so timed-out callers cannot free a live thread's slot.
            return await run_sync(query, limiter=query_limit)
        finally:
            query_admitted -= 1

    def envelope_handler(route):
        async def adapt(request: Request):
            from sparkrun.api._wire import validate

            principal(request)
            current = SparkrunConfig(binding.config_path)
            if route.feature_flag and not current.is_feature_enabled(route.feature_flag):
                raise PermissionError("Route feature is disabled")
            envelope = validate(await read_json(request), route.request_schema)
            name, arguments = route.decode_loader()(envelope)
            return await dispatch(resolve_route_operation(route, name), arguments, request)

        return adapt

    @app.get("/v1/jobs")
    async def jobs(request: Request, state: str | None = None, limit: int = 200):
        user = principal(request)
        return [
            public_job(j)
            for j in store.list(principal=None if user.local or user.scope == "admin" else user.id, state=state, limit=limit)
            if user.allows(j["scope"])
        ]

    @app.get("/v1/jobs/{identifier}")
    async def job(identifier: str, request: Request):
        return public_job(job_access(request, identifier))

    @app.get("/v1/jobs/{identifier}/artifacts/diagnostics", response_class=FileResponse)
    async def diagnostic_bundle(identifier: str, request: Request):
        """Download a finished job's diagnostic bundle under its job permissions."""
        from sparkrun.daemon.artifacts import diagnostics_path
        import stat

        job = job_access(request, identifier)
        if job["state"] not in TERMINAL:
            raise Conflict("Diagnostic bundle is available after the job finishes")
        path = diagnostics_path(binding.directory, identifier)
        try:
            if not stat.S_ISDIR(path.parent.lstat().st_mode) or not stat.S_ISREG(path.lstat().st_mode):
                raise FileNotFoundError(path)
        except FileNotFoundError:
            return JSONResponse({"detail": "Diagnostic bundle not found"}, status_code=404)
        return FileResponse(path, media_type="application/x-ndjson", filename=f"diagnostics-{identifier}.ndjson")

    @app.delete("/v1/jobs/{identifier}")
    async def cancel(identifier: str, request: Request):
        job_access(request, identifier)
        principal(request, "operate")
        store.cancel(identifier)
        return public_job(store.get(identifier))

    @app.post("/v1/jobs/{identifier}/input")
    async def answer(identifier: str, request: Request):
        job = job_access(request, identifier)
        data = await read_json(request)
        if set(data) != {"prompt_id", "value"} or not isinstance(data["prompt_id"], str):
            raise ValueError("Expected prompt_id and value")
        prompt = store.prompt_info(identifier, data["prompt_id"])
        principal(request, job["scope"], secret=prompt["kind"] == "secret")
        if not job["worker"] or job["state"] in TERMINAL:
            raise Conflict("Worker has finished; fetch the job result")
        from sparkrun.daemon.control import request as control

        return await asyncio.to_thread(control, job["worker"], job["nonce"], "input", **data)

    @app.get("/v1/reservations")
    async def reserved(request: Request):
        principal(request, "admin")
        from sparkrun.core.coordination import reservations

        return await asyncio.to_thread(reservations, binding.config_path.parent)

    @app.post("/v1/reservations/{owner}/reconcile")
    async def reconcile(owner: str, request: Request):
        principal(request, "admin")
        data = await read_json(request)
        if set(data) != {"acknowledge_effects"} or data["acknowledge_effects"] is not True:
            raise ValueError("Explicit acknowledgement of reconciled effects is required")
        from sparkrun.core.coordination import reconcile_reservation

        await asyncio.to_thread(reconcile_reservation, binding.config_path.parent, owner, acknowledge_effects=True)
        return {"released": owner}

    @app.get("/v1/snapshots/{name}")
    async def snapshot(name: str, request: Request):
        principal(request, "read")
        if name != "discovery.snapshot":
            raise LookupError("Unknown snapshot")
        return services.snapshot(name)

    @app.post("/v1/maintenance")
    async def maintenance(request: Request):
        principal(request, "admin")
        data = await read_json(request)
        if set(data) != {"drain"} or type(data["drain"]) is not bool:
            raise ValueError("Expected boolean drain")
        coordinator.maintenance = data["drain"]
        return {"draining": coordinator.maintenance}

    @app.post("/v1/shutdown")
    async def shutdown_api(request: Request):
        principal(request, "admin")
        if await read_json(request) != {}:
            raise ValueError("Expected an empty shutdown object")
        callback = getattr(app.state, "shutdown", None)
        if callback is None:
            raise Conflict("Server shutdown is unavailable")
        asyncio.get_running_loop().call_later(0.1, callback)
        return {"stopping": True, "workers_continue": True}

    @app.get("/v1/jobs/{identifier}/events")
    async def events(identifier: str, request: Request):
        job_access(request, identifier)
        after = int(request.headers.get("Last-Event-ID", "0"))
        if after < 0:
            raise ValueError("Invalid event cursor")
        store.events(identifier, after)  # Return 410 before stream headers on expired cursors.

        async def produce():
            nonlocal after
            while not stop.is_set() and not await request.is_disconnected():
                job_access(request, identifier)
                try:
                    batch = await asyncio.to_thread(store.events, identifier, after)
                except ReplayGap as exc:
                    yield sse({"type": "gap", "detail": str(exc)})
                    return
                for event in batch:
                    after = event["id"]
                    yield sse(event)
                job = store.get(identifier)
                if job["state"] in TERMINAL and after >= job["seq"]:
                    return
                if not batch:
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(1)

        return subscriptions.response(produce, headers={"Cache-Control": "no-cache"})

    @app.get("/v1/events")
    async def global_events(request: Request):
        user = principal(request)

        async def produce():
            while not stop.is_set() and not await request.is_disconnected():
                principal(request)
                rows = store.list(principal=None if user.local or user.scope == "admin" else user.id)
                yield sse({"type": "snapshot", "jobs": [public_job(j) for j in rows if user.allows(j["scope"])]})
                await asyncio.sleep(2)

        return subscriptions.response(produce)

    @app.get("/v1/streams/{kind}")
    async def stream(kind: str, request: Request):
        principal(request)
        if kind not in {"logs", "monitor"}:
            raise LookupError("Unknown stream")
        from sparkrun.daemon.ops import fields, B, INTEGER, S
        from sparkrun.api._wire import validate, object_schema

        arguments: dict[str, Any] = dict(request.query_params)
        for key in ("hosts",):
            if key in arguments:
                arguments[key] = arguments[key].split(",")
        for key in ("tail", "interval"):
            if key in arguments:
                arguments[key] = int(arguments[key])
        if "follow" in arguments:
            if arguments["follow"] not in {"true", "false"}:
                raise ValueError("follow must be true or false")
            arguments["follow"] = arguments["follow"] == "true"
        schema = fields("workload_id recipe hosts cluster scope tail" if kind == "logs" else "hosts cluster interval")
        if kind == "logs":
            schema["follow"] = B
            schema["scope"] = {"type": "string", "enum": ["head", "all"]}
            for selector in ("host", "container", "source_id"):
                schema[selector] = {**S, "minLength": 1}
                if selector in arguments and not arguments[selector].strip():
                    raise ValueError("%s must be a non-empty string" % selector)
        else:
            schema["interval"] = {**INTEGER, "minimum": 1, "maximum": 60}
        validate(arguments, object_schema(schema))

        def open_stream():
            from sparkrun.application import initialize

            sctx = initialize(config_path=binding.config_path)
            sctx.operation = OperationState(via_daemon=True)

            async def produce():
                resource: Any = None
                try:
                    if kind == "logs":
                        from sparkrun.daemon.streams import LogStream

                        resource = LogStream(arguments, sctx)
                        while not stop.is_set() and not await request.is_disconnected():
                            principal(request)
                            if resource.dropped:
                                count = resource.dropped
                                resource.dropped = 0
                                yield sse({"type": "gap", "dropped": count})
                            try:
                                yield sse(resource.queue.get_nowait())
                            except queue.Empty:
                                if resource.done.is_set():
                                    if resource.error:
                                        yield sse({"type": "error", **resource.error})
                                    return
                                yield ": heartbeat\n\n"
                                await asyncio.sleep(1)
                    else:
                        resource = hub.subscribe(arguments, sctx)
                        subscription = await run_sync(resource.__enter__, limiter=query_limit)
                        if not stop.is_set() and not await request.is_disconnected():
                            yield sse(subscription.history())
                        while not stop.is_set() and not await request.is_disconnected():
                            principal(request)
                            yield sse(subscription.frame())
                            await asyncio.sleep(arguments.get("interval", 2))
                finally:
                    if resource is not None:
                        if kind == "logs":
                            await asyncio.to_thread(resource.close)
                        else:
                            await asyncio.to_thread(resource.__exit__, None, None, None)

            return produce()

        return subscriptions.response(open_stream)

    original_openapi = app.openapi

    def openapi():
        from sparkrun.daemon.http_routes import add_operation_schemas

        return add_operation_schemas(original_openapi(), app.state.operation_paths)

    app.openapi = openapi
    return app


class PeerProtocol(H11Protocol):
    def connection_made(self, transport):
        if sys.platform != "linux":
            transport.close()
            return
        sock = transport.get_extra_info("socket")
        uid = None
        if sock is not None and sock.family == socket.AF_UNIX:
            if not hasattr(socket, "SO_PEERCRED"):
                transport.close()
                return
            _, uid, _ = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
            if uid != os.getuid():
                transport.close()
                return
        if uid is not None:
            original: Any = self.app

            async def trusted(scope, receive, send) -> None:
                scope["sparkrun.peer_uid"] = uid
                return await original(scope, receive, send)

            self.app = trusted
        super().connection_made(transport)


async def serve(
    binding,
    *,
    tcp=None,
    unix=None,
    cert=None,
    key=None,
    conn_file=None,
    parent_pid=None,
    worker_supervisor="process",
    allow_request_trust=None,
    allow_insecure: bool = False,
):
    from sparkrun.core.process_identity import process_identity, alive, user_identity
    from sparkrun.daemon.state import private_directory
    from sparkrun.utils.fs import atomic_private_write, file_lock

    started_at = time.monotonic()
    unix = sys.platform.startswith("linux") if unix is None else unix
    if unix and (not sys.platform.startswith("linux") or not hasattr(socket, "SO_PEERCRED")):
        raise ValueError("Unix peer authentication requires Linux; use --no-unix and TCP on this platform")
    if not unix and tcp is None:
        tcp = ("127.0.0.1", 0)
    if worker_supervisor == "systemd" and not sys.platform.startswith("linux"):
        raise ValueError("The systemd worker supervisor requires Linux")
    parent = process_identity(parent_pid) if parent_pid else None
    if parent_pid and parent is None:
        raise ValueError("Cannot identify parent process")
    identity = process_identity()
    if identity is None:
        raise RuntimeError("Cannot identify this server process")
    binding.prepare()
    lock = file_lock(binding.directory / "server.lock", blocking=False)
    try:
        lock.__enter__()
    except BlockingIOError:
        raise RuntimeError("A daemon is already running for this config binding") from None
    servers, sockets, credentials = [], [], []
    token_id = None
    tls_sockets = set()
    previous_signals = {}
    try:
        local_socket = None
        if unix and sys.platform == "linux":
            binding.socket.unlink(missing_ok=True)
            local_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sockets.append(local_socket)
            local_socket.bind(str(binding.socket))
            binding.socket.chmod(0o600)
            local_socket.listen(128)
            local_socket.setblocking(False)
        url = local_url = None
        local = True
        if bool(cert) != bool(key):
            raise ValueError("TLS requires both certificate and private key")
        if tcp:
            host, port = tcp
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                local = host == "localhost"
            if not local and not (cert and key) and not allow_insecure:
                raise ValueError("Remote TCP requires TLS certificate and key, or explicit --allow-insecure")
            network = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(network)
            if cert:
                tls_sockets.add(network)
            if os.name == "nt":
                network.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                network.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            network.bind((host, port))
            network.listen(128)
            network.setblocking(False)
            url = ("https" if cert else "http") + "://" + (f"[{host}]" if ":" in host else host) + ":" + str(network.getsockname()[1])
        if tcp:
            local_url = url
            if cert or not local:
                # Keep local lifecycle/recovery independent of the remote bind's
                # address, certificate hostname and trust store. Still bearer-only.
                control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(control_socket)
                if os.name == "nt":
                    control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                control_socket.bind(("127.0.0.1", 0))
                control_socket.listen(128)
                control_socket.setblocking(False)
                local_url = "http://127.0.0.1:" + str(control_socket.getsockname()[1])
        if conn_file and not tcp:
            raise ValueError("Connection-file mode requires token-authenticated TCP")
        app = create_app(binding, worker_supervisor=worker_supervisor, allow_request_trust=allow_request_trust)

        class Server(uvicorn.Server):
            @__import__("contextlib").contextmanager
            def capture_signals(self):
                yield

        for sock in sockets:
            is_unix = sock is local_socket
            cfg = uvicorn.Config(
                app,
                host="127.0.0.1",
                http=PeerProtocol if is_unix else H11Protocol,
                ws="none",
                lifespan="off",
                access_log=False,
                ssl_certfile=cert if sock in tls_sockets else None,
                ssl_keyfile=key if sock in tls_sockets else None,
                timeout_graceful_shutdown=15,
            )
            servers.append(Server(cfg))
        loop = asyncio.get_running_loop()

        def shutdown():
            app.state.stop.set()
            for server in servers:
                server.should_exit = True

        app.state.shutdown = shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_signals[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(shutdown))
        async with app.router.lifespan_context(app):
            state = {
                **binding.info(),
                "process": identity,
                "socket": str(binding.socket) if unix else None,
                "url": url,
                "protocol": 1,
                "runtime": app.state.runtime,
                "worker_supervisor": worker_supervisor,
                "service_unit": os.environ.get("SPARKRUN_DAEMON_UNIT"),
                "launch": {
                    "tcp": tcp,
                    "unix": unix,
                    "cert": cert,
                    "key": key,
                    "conn_file": str(conn_file) if conn_file else None,
                    "parent_pid": parent_pid,
                    "worker_supervisor": worker_supervisor,
                    "allow_request_trust": allow_request_trust,
                    "allow_insecure": allow_insecure,
                },
            }
            if tcp:
                token_id, token = TokenStore(binding).create(
                    "local-controller", scope="admin", secret=True, owner_process=identity, principal_id="uid:" + user_identity()
                )
                content = json.dumps({**state, "token": token})
                atomic_private_write(binding.connection, json.dumps({**state, "url": local_url, "token": token}))
                credentials.append(binding.connection)
                if conn_file and Path(conn_file).absolute() != binding.connection.absolute():
                    path = Path(conn_file)
                    private_directory(path.parent)
                    if not atomic_private_write(path, content, overwrite=False):
                        raise ValueError("Connection file already exists")
                    credentials.append(path)
            binding.write_state(state)

            async def watchdog():
                while not any(s.should_exit for s in servers):
                    if parent is not None and not alive(parent):
                        shutdown()
                        return
                    await asyncio.sleep(0.25)

            async def report_ready():
                # One event per serving process, across all listeners. Delivery
                # runs off the event loop and must never prevent serving/recovery.
                try:
                    while not all(server.started for server in servers):
                        if any(server.should_exit for server in servers):
                            return
                        await asyncio.sleep(0.05)
                    from sparkrun.core.config import SparkrunConfig
                    from sparkrun.telemetry.emit import emit_daemon_serve_telemetry

                    await asyncio.to_thread(
                        emit_daemon_serve_telemetry,
                        SparkrunConfig(binding.config_path),
                        unix_socket=bool(unix),
                        tcp=bool(tcp),
                        tls=bool(tls_sockets),
                        remote_tcp=bool(tcp and not local),
                        worker_supervisor=worker_supervisor,
                        parent_managed=parent is not None,
                        startup_ms=max(0, int((time.monotonic() - started_at) * 1000)),
                    )
                except Exception:
                    logger.debug("Failed to report daemon readiness telemetry", exc_info=True)

            task = asyncio.create_task(watchdog())
            telemetry_task = asyncio.create_task(report_ready())
            try:
                await asyncio.gather(*(server.serve(sockets=[sock]) for server, sock in zip(servers, sockets, strict=True)))
            finally:
                shutdown()
                await task
                await telemetry_task
    finally:
        for sig, previous in previous_signals.items():
            signal.signal(sig, previous)
        for sock in sockets:
            sock.close()
        if unix:
            binding.socket.unlink(missing_ok=True)
        if token_id:
            try:
                TokenStore(binding).revoke(token_id)
            except KeyError:
                pass  # Already revoked through the token-management CLI.
        for path in credentials:
            try:
                data = json.loads(path.read_text())
                if data.get("process") == identity:
                    path.unlink()
            except (FileNotFoundError, ValueError):
                pass
        (binding.directory / "server.json").unlink(missing_ok=True)
        lock.__exit__(None, None, None)
