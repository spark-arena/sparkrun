"""Short service controls and durable gateway/discovery worker adapters."""

from __future__ import annotations

from functools import partial
import json
import time
from pathlib import Path
import uuid

from sparkrun.daemon.jobs import JobStore, TERMINAL
from sparkrun.daemon.state import Binding
from sparkrun.proxy.discovery_service import session_for, run_session


def _store(sctx):
    binding = Binding.from_config(sctx.config)
    return JobStore.from_config(binding.directory, sctx.config)


def ensure_discovery(arguments, sctx, prompt=None):
    from sparkrun.core.daemon_registry import get_daemon_op
    from sparkrun.core.runtime_lease import consistent_runtime, runtime_identity

    session = session_for(sctx)
    with session.lock():
        record = session.matching(arguments["generation"])
        if record["descriptor"]["backend"] != "daemon":
            raise ValueError("Discovery session does not select the daemon backend")
        if Path(record["descriptor"]["application_config_path"]).resolve() != sctx.config.config_path.resolve():
            raise ValueError("Discovery session belongs to another controller binding")
        store = _store(sctx)
        if record["job_id"]:
            job = store.get(record["job_id"])
            if job["state"] in TERMINAL:
                raise ValueError("Discovery worker finished; explicitly stop/restart discovery to reconcile its owner")
            return {"job_id": job["id"]}
        runtime = runtime_identity(sctx.config)
        with consistent_runtime(sctx.config, runtime):
            identifier = store.submit(
                get_daemon_op("proxy.watch"),
                arguments,
                "service:daemon",
                "proxy discovery",
                runtime,
                key="proxy.watch:" + record["generation"],
            )
        record["job_id"] = identifier
        session.write(record)
        return {"job_id": identifier}


def discovery_status(arguments, sctx, prompt=None):
    result = session_for(sctx).status()
    if result.get("job_id"):
        try:
            job = _store(sctx).get(result["job_id"])
            if job["state"] == "lost":
                result["state"] = "recovery_required"
        except KeyError:
            result["state"] = "recovery_required"
    return result


def cancel_job(store, identifier):
    if not identifier:
        return
    try:
        job = store.get(identifier)
        if job["state"] in TERMINAL:
            return
        store.cancel(identifier)
        if job["worker"]:
            from sparkrun.daemon.control import request

            try:
                request(job["worker"], job["nonce"], "cancel")
            except (OSError, ValueError):
                pass  # Persisted cancellation is also observed on reconnect.
    except KeyError:
        pass


def stop_discovery(arguments, sctx, prompt=None):
    record = session_for(sctx).stop()
    if record:
        cancel_job(_store(sctx), record["job_id"])
    return {"stopped": bool(record)}


def watch_proxy(arguments, sctx, prompt=None):
    from sparkrun.core.operation import CURRENT

    operation = CURRENT.get()
    if operation is None:
        raise RuntimeError("Discovery service requires a daemon operation context")
    return run_session(
        session_for(sctx), arguments["generation"], sctx, cancelled=operation.cancel.is_set, wait=operation.cancel.wait, job_id=operation.id
    )


def serve_proxy(arguments, sctx, prompt=None):
    """Own the gateway lifetime in a separate durable service, never a watcher."""
    from sparkrun.api.proxy._ops import _engine_class, _gateway_context_kwargs
    from sparkrun.proxy._supervisor import GatewayState
    from sparkrun.utils.fs import file_lock

    directory = GatewayState().state_dir
    path = directory / "gateway-start.json"
    with file_lock(directory / ".gateway-start.lock"):
        saved = json.loads(path.read_text())
        if saved["generation"] != arguments["generation"] or saved.get("claimed"):
            raise ValueError("Gateway start request was superseded or already claimed")
        saved["claimed"] = True
        from sparkrun.utils.fs import atomic_private_write

        atomic_private_write(path, json.dumps(saved))
    engine_cls = _engine_class(saved["gateway"])
    engine = engine_cls(
        host=saved["host"],
        port=saved["port"],
        master_key=saved["master_key"],
        host_configured=True,
        **_gateway_context_kwargs(engine_cls, sctx),
    )
    # The engine's foreground supervisor checks operation cancellation and
    # handles restart requests itself, keeping replacements in its own unit.
    engine._discovery_control = True
    try:
        rc = engine.start(
            config_path=Path(saved["config_path"]) if saved["config_path"] else None,
            foreground=True,
            autodiscover_kwargs=saved["autodiscover"],
        )
        if rc:
            raise RuntimeError("Gateway service exited with status %s" % rc)
        return {"exit_code": rc}
    except BaseException:
        engine.stop()
        raise


def submit_service(sctx, name, arguments, *, key):
    """Host-local submission through the same durable registry and queue.

    Internal operations have no HTTP submission path. The local controller
    verifies the daemon runtime/admission state before publishing a service.
    """
    from sparkrun.core.daemon_registry import get_daemon_op
    from sparkrun.core.runtime_lease import consistent_runtime
    from sparkrun.daemon.client import Client

    info = Client(binding=Binding.from_config(sctx.config)).request("GET", "/v1/info")
    if info["restart_required"]:
        raise RuntimeError("Daemon is draining or needs restart before accepting services")
    from sparkrun.daemon.ops import register_core_ops

    register_core_ops()
    op = get_daemon_op(name)
    if op.visibility != "internal":
        raise ValueError("Service submission requires an internal operation")
    with consistent_runtime(sctx.config, info["runtime"]):
        identifier = _store(sctx).submit(op, arguments, "service:daemon", "proxy service", info["runtime"], key=key)
    return {"job_id": identifier, "events_url": "/v1/jobs/" + identifier + "/events"}


def launch_gateway(engine, config_path, ad_kwargs, sctx):
    """Submit a separate gateway service and wait only for its ready state."""
    from sparkrun.daemon.client import Client
    from sparkrun.daemon.management import ensure_daemon
    from sparkrun.utils.fs import atomic_private_write, file_lock

    binding = Binding.from_config(sctx.config)
    ensure_daemon(binding)
    client = Client(binding=binding)
    generation = uuid.uuid4().hex
    saved = {
        "generation": generation,
        "gateway": engine.gateway_name,
        "host": engine.host,
        "port": engine.port,
        "master_key": engine.master_key,
        "config_path": str(config_path) if config_path else None,
        "autodiscover": {k: str(v) if isinstance(v, Path) else v for k, v in (ad_kwargs or {}).items()},
    }
    if ad_kwargs and ad_kwargs.get("backend") == "daemon" and int(sctx.config.get("daemon.max_services", 2)) < 2:
        raise ValueError("Daemon gateway plus discovery requires daemon.max_services >= 2")
    with file_lock(engine.state_dir / ".gateway-start.lock"):
        pending_path = engine.state_dir / "gateway-start.json"
        if pending_path.exists():
            pending = json.loads(pending_path.read_text())
            try:
                pending_state = _store(sctx).get(pending["job_id"])["state"] if pending.get("job_id") else None
            except KeyError:
                pending_state = "expired"  # Only terminal job records are subject to retention.
            if pending_state and pending_state not in TERMINAL and pending_state != "expired":
                raise RuntimeError("Gateway startup is already pending; inspect daemon job " + pending["job_id"])
            if not pending.get("claimed") and not pending.get("job_id"):
                # Re-submit the original generation to resolve an interrupted job
                # submission before replacing its private launch descriptor.
                previous = submit_service(
                    sctx, "proxy.serve", {"generation": pending["generation"]}, key="proxy.serve:" + pending["generation"]
                )
                pending["job_id"] = previous["job_id"]
                atomic_private_write(pending_path, json.dumps(pending))
                raise RuntimeError("Previous gateway submission requires reconciliation; inspect daemon job " + previous["job_id"])
        atomic_private_write(engine.state_dir / "gateway-start.json", json.dumps(saved))
        result = submit_service(sctx, "proxy.serve", {"generation": generation}, key="proxy.serve:" + generation)
        saved["job_id"] = result["job_id"]
        atomic_private_write(engine.state_dir / "gateway-start.json", json.dumps(saved))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = engine.get_state() or {}
        if state.get("service_job") == result["job_id"] and state.get("service_ready") and engine.is_running():
            return result["job_id"]
        job = client.request("GET", "/v1/jobs/" + result["job_id"])
        if job["state"] in TERMINAL:
            raise RuntimeError("Gateway service exited before becoming ready; inspect its daemon job")
        time.sleep(0.1)
    # The accepted service may still start; preserve its identity for recovery.
    raise TimeoutError("Gateway service startup is pending; inspect daemon job " + result["job_id"])


def wait_gateway(sctx):
    """Foreground attachment does not retain the startup workload reservation."""
    from sparkrun.api.proxy._ops import _running_engine, stop
    from sparkrun.core.operation import CURRENT
    from sparkrun.daemon.client import Client

    engine = _running_engine(sctx)
    identifier = (engine.get_state() or {}).get("service_job")
    if not identifier:
        raise RuntimeError("Gateway did not publish its supervising daemon job")
    client = Client(binding=Binding.from_config(sctx.config))
    try:
        while True:
            operation = CURRENT.get()
            if operation:
                operation.check()
            job = client.request("GET", "/v1/jobs/" + identifier)
            if job["state"] in TERMINAL:
                return 0 if job["state"] == "succeeded" else 1
            time.sleep(0.2)
    except BaseException:
        stop(sctx=sctx)
        raise


def register_proxy_ops():
    from sparkrun.daemon.ops import DESCRIPTIONS

    from sparkrun.api._wire import object_schema
    from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op

    generation = object_schema({"generation": {"type": "string", "maxLength": 64}}, ["generation"])
    for name, handler, schema, kind, result in (
        ("proxy.discovery.ensure", ensure_discovery, generation, "query", object_schema({"job_id": {"type": "string"}}, ["job_id"])),
        ("proxy.discovery.status", discovery_status, object_schema(), "query", {"type": "object"}),
        ("proxy.discovery.stop", stop_discovery, object_schema(), "query", object_schema({"stopped": {"type": "boolean"}}, ["stopped"])),
        ("proxy.watch", watch_proxy, generation, "job", object_schema({"stopped": {"type": "string"}}, ["stopped"])),
        ("proxy.serve", serve_proxy, generation, "job", object_schema({"exit_code": {"type": "integer"}}, ["exit_code"])),
    ):
        register_daemon_op(
            DaemonOp(
                name,
                partial(_handler, handler),
                schema,
                result_schema=result,
                kind=kind,
                scope="admin",
                visibility="internal" if name in {"proxy.watch", "proxy.serve"} else "public",
                resources="none",
                feature_flag="gateway" if name in {"proxy.discovery.ensure", "proxy.watch", "proxy.serve"} else None,
                description=DESCRIPTIONS.get(name, "Manage the proxy's dedicated gateway or discovery service."),
            )
        )


def _handler(fn):
    return fn
