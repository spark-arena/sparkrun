"""One isolated workflow; reconnectable control and durable writes without ASGI."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hmac
import json
import logging
import os
import socketserver
import threading
import time
import uuid

from sparkrun.core.operation import OperationState, operation_scope
from sparkrun.core.process_identity import process_identity
from sparkrun.daemon.jobs import JobStore


@contextmanager
def cleanup_on_failure():
    try:
        yield
    except BaseException:
        from sparkrun.core.process_identity import terminate_tree

        terminate_tree(process_identity(), include_root=False)
        raise


class Worker:
    def __init__(self, binding, job):
        self.binding = binding
        from sparkrun.core.config import SparkrunConfig

        config = SparkrunConfig(binding.config_path)
        self.store = JobStore.from_config(binding.directory, config)
        self.prompt_timeout = float(config.get("daemon.prompt_timeout", 300))
        import math

        if not math.isfinite(self.prompt_timeout) or self.prompt_timeout <= 0:
            raise ValueError("daemon.prompt_timeout must be positive and finite")
        self.job = self.store.get(job)
        self.id = job
        self.nonce = self.job["nonce"]
        self.started = threading.Event()
        self.condition = threading.Condition()
        self.answer = None
        self.answered = {}
        self.secrets = set()
        self.operation = OperationState(
            id=job,
            events=lambda event: self.store.emit(job, self.redact(event)),
            via_daemon=True,
            manager=self.job["manager"],
            secrets=self.secrets,
        )

    def redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[redacted]")
        elif isinstance(value, dict):
            value = {key: self.redact(item) for key, item in value.items()}
        elif isinstance(value, (list, tuple)):
            value = [self.redact(item) for item in value]
        return value

    def control(self, data):
        if not isinstance(data.get("nonce"), str) or not hmac.compare_digest(self.nonce, data["nonce"]):
            raise PermissionError("Invalid worker credential")
        action = data.get("action")
        if action == "hello":
            return {"process": process_identity(), "job": self.id, "runtime": self.job["runtime"]}
        if action == "start":
            self.store.authorize(self.id, self.nonce)
            self.started.set()
            return {}
        if action == "cancel":
            self.operation.cancel.set()
            with self.condition:
                self.condition.notify_all()
            if not self.started.is_set():
                self.started.set()
            # Windows SIGTERM terminates without unwinding Python. Interrupt the
            # main thread cooperatively; the coordinator enforces the deadline.
            if os.name == "nt":
                import _thread

                _thread.interrupt_main()
            else:
                import signal

                os.kill(os.getpid(), signal.SIGTERM)
            return {}
        if action == "input":
            with self.condition:
                prompt_id = data.get("prompt_id")
                value = data.get("value")
                # Keep only a digest for duplicate delivery, never plaintext secrets.
                import hashlib

                digest = hashlib.sha256(json.dumps(value).encode()).hexdigest()
                if prompt_id in self.answered:
                    if self.answered[prompt_id] != digest:
                        raise ValueError("Prompt was already answered differently")
                    return {}
                pending = self.store.get(self.id)["prompt"]
                if not pending or pending["prompt_id"] != prompt_id or pending["deadline"] < time.time():
                    raise ValueError("Prompt expired or no longer pending")
                if pending["kind"] in {"approval", "decision"} and type(value) is not bool:
                    raise ValueError("Approval answers must be boolean")
                if pending["kind"] == "secret" and value is not None and not isinstance(value, str):
                    raise ValueError("Credential answer must be a string")
                if pending["kind"] == "secret" and value:
                    self.secrets.add(value)
                self.store.answer(self.id, prompt_id)
                self.answered[prompt_id] = digest
                self.answer = (prompt_id, value)
                self.condition.notify_all()
            return {}
        raise ValueError("Unknown control action")

    def prompt(self, kind, message, *, default=False, timeout=None):
        if len(self.answered) >= 128:
            raise ValueError("Job exceeded its input prompt limit")
        identifier = uuid.uuid4().hex
        deadline = time.time() + (self.prompt_timeout if timeout is None else timeout)
        with self.condition:
            self.answer = None
            self.store.prompt(
                self.id, {"prompt_id": identifier, "kind": kind, "message": message, "default": default, "deadline": deadline}
            )
            while self.answer is None:
                self.operation.check()
                left = deadline - time.time()
                if left <= 0:
                    self.store.answer(self.id, identifier)
                    return default
                self.condition.wait(min(left, 1))
            value = self.answer[1]
            self.answer = None
            return value

    def run(self):
        from sparkrun.api._wire import error_body
        from sparkrun.application import initialize
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.runtime_lease import runtime_lease
        from sparkrun.core.signals import install_termination_handlers
        from sparkrun.core.daemon_registry import get_daemon_op
        from sparkrun.daemon.ops import register_core_ops, check_request

        install_termination_handlers()
        worker = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(5)
                raw = self.rfile.readline(65537)
                try:
                    if len(raw) > 65536 or not raw.endswith(b"\n"):
                        raise ValueError("Invalid control message")
                    result = {"ok": True, **worker.control(json.loads(raw))}
                except Exception as exc:
                    result = {
                        "ok": False,
                        "detail": str(exc) if isinstance(exc, (ValueError, PermissionError)) else "Control request failed",
                    }
                try:
                    self.wfile.write(json.dumps(result).encode() + b"\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True

            def __init__(self, *args):
                self.slots = threading.BoundedSemaphore(8)
                super().__init__(*args)

            def server_bind(self):
                if os.name == "nt":
                    import socket

                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                super().server_bind()

            def process_request(self, request, client_address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self.slots.release()
                    raise

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self.slots.release()

        containment = None
        succeeded = False
        try:
            if os.name == "nt":
                from sparkrun.utils.windows import WorkerJob

                containment = WorkerJob(process_identity())
            with ExitStack() as stack:
                from sparkrun.core.feedback_logging import forward_operation_logs

                stack.enter_context(forward_operation_logs(worker=self.operation))
                config = SparkrunConfig(self.binding.config_path)
                stack.enter_context(runtime_lease(config, kind="worker", expected=self.job["runtime"]))
                sctx = initialize(config_path=self.binding.config_path)
                sctx.operation = self.operation
                from sparkrun.core.progress import LaunchProgress
                from sparkrun.core.timing import Timeline

                sctx.timing = Timeline()
                sctx.progress = LaunchProgress(timeline=sctx.timing)
                register_core_ops()
                op = get_daemon_op(self.job["operation"])
                if op.version != self.job["version"]:
                    raise ValueError("Operation version changed")
                if op.feature_flag and not sctx.config.is_feature_enabled(op.feature_flag):
                    raise ValueError("Operation feature is disabled")
                check_request(op, self.job["arguments"])
                server = stack.enter_context(Server(("127.0.0.1", 0), Handler))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                stack.callback(server.shutdown)
                self.store.register_worker(
                    self.id,
                    self.nonce,
                    {
                        "process": process_identity(),
                        "endpoint": {"transport": "tcp", "host": "127.0.0.1", "port": server.server_address[1]},
                    },
                )
                while not self.started.wait(0.2):
                    self.operation.check()
                    if self.store.get(self.id)["state"] == "cancelling":
                        raise InterruptedError("Cancelled before execution")
                self.operation.check()
                with operation_scope(sctx):
                    from sparkrun.core.coordination import infer_reservation_hosts, reserve_resources, reservation_keys

                    # Composite adapters use the same reservation mechanism as CLI/API.
                    # A launch, teardown or benchmark resume naming no hosts locks what
                    # the API call it wraps locks: the default cluster, or the workload's
                    # or benchmark's recorded hosts. Anything else keeps "*": catalog
                    # and cluster-definition jobs act on state no host list describes.
                    args = self.job["arguments"]
                    hosts = args.get("hosts")
                    cluster = args.get("cluster")
                    workload_id = args.get("workload_id") or args.get("cluster_id")
                    benchmark_id = args.get("benchmark_id")
                    if not hosts and (cluster or workload_id or benchmark_id or args.get("recipe")):
                        hosts, cluster = infer_reservation_hosts(
                            config, cluster=cluster, workload_id=workload_id, benchmark_id=benchmark_id
                        )
                    executor_config = args.get("executor_config") or getattr(cluster, "executor_config", None)
                    keys = reservation_keys(hosts, executor_config=executor_config, controller=op.resources == "controller")
                    if op.scope != "read" and op.resources != "none":
                        stack.enter_context(reserve_resources(config.config_path.parent, keys))
                    # Cleanup precedes releasing mutation reservations.
                    stack.enter_context(cleanup_on_failure())
                    from sparkrun.api._wire import check_result

                    result = check_result(op, op.handler_loader()(dict(args), sctx, self.prompt), redact=self.redact)
                    self.store.finish(self.id, "succeeded", result=self.redact(result))
            succeeded = True
        except (KeyboardInterrupt, InterruptedError):
            self.store.finish(self.id, "cancelled", error={"detail": "Execution cancelled; remote effects may require reconciliation"})
        except BaseException as exc:
            import traceback
            from sparkrun.utils.fs import atomic_private_write

            # Stack frames identify adapter bugs without persisting exception values/secrets.
            atomic_private_write(
                self.binding.directory / (self.id + ".error.log"),
                type(exc).__name__ + "\n" + "".join(traceback.format_tb(exc.__traceback__)),
            )
            logging.getLogger(__name__).error("Worker %s failed: %s", self.id, type(exc).__name__)
            self.store.finish(self.id, "failed", error=self.redact(error_body(exc)))
        finally:
            if containment and succeeded:
                containment.release_success()
            # On failure, keep the Job Object alive through terminal persistence;
            # process exit closes its handle and kills any surviving descendants.


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute one durable daemon job in an isolated process.")
    parser.add_argument("--config", required=True)
    parser.add_argument("job")
    args = parser.parse_args(argv)
    from sparkrun.core.application_profile import initialize_child_application_profile

    initialize_child_application_profile()
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.daemon.state import Binding

    binding = Binding.from_config(SparkrunConfig(args.config)).prepare()
    # Validate the identifier before constructing any paths.
    if len(args.job) != 32 or any(c not in "0123456789abcdef" for c in args.job):
        parser.error("Invalid job identifier")
    Worker(binding, args.job).run()


if __name__ == "__main__":
    main()
