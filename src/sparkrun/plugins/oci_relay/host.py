# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0

"""Operation-owned setup, managed processes and bounded output readers."""

from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
import queue
import re
import shlex
import threading
import time
import uuid

from .release import file_digest
from .source_policy import LOCAL_DOCKER, classic_store_reason, containerd_store_reason, docker_facts, native_mounts


logger = logging.getLogger(__name__)


class OperationError(RuntimeError):
    pass


class Runner:
    def __init__(self, session, settings):
        from sparkrun.transports.session import SshHostSession

        if not callable(getattr(session, "open_process", None)):
            raise OperationError("host transport does not support streaming processes")
        self.session = session
        self.local = SshHostSession()
        self.settings = settings
        self.processes = []
        self.directories = []
        self.containers = []

    def connection(self, host):
        return self.local if host is None else self.session

    def execute(self, host, arguments, *, input_data=None, timeout=30):
        result = self.connection(host).execute(
            host or "localhost", arguments, input_data=input_data, timeout=timeout,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout)[-4000:].decode(errors="replace")
            raise OperationError(f"{host or 'controller'}: command failed ({result.returncode}): {detail}")
        return result.stdout

    def start(self, host, arguments):
        procs = self.settings.get("relay_gomaxprocs", 0)
        if procs and arguments[:len(LOCAL_DOCKER)] != LOCAL_DOCKER:
            arguments = ["env", f"GOMAXPROCS={procs}", *arguments]
        process = self.connection(host).open_process(host or "localhost", arguments)
        self.processes.append(process)
        return process

    def directory(self, host):
        path = self.execute(host, ["mktemp", "-d", "/tmp/oci-relay.XXXXXXXXXX"]).decode().strip()
        if not re.fullmatch(r"/tmp/oci-relay\.[A-Za-z0-9]{10}", path):
            raise OperationError("host returned an invalid operation directory")
        self.directories.append((host, path))
        return path

    def write(self, host, path, data):
        # Only caller-generated paths, always shell-quoted; payload stays on stdin.
        self.execute(host, ["sh", "-c", "umask 077; cat > " + shlex.quote(path)], input_data=data)

    def json(self, host, path, value):
        self.write(host, path, json.dumps(value, separators=(",", ":")).encode())

    def architecture(self, host):
        fields = self.execute(host, ["uname", "-sm"]).decode().split()
        arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(fields[-1] if fields else "")
        if len(fields) != 2 or fields[0] != "Linux" or arch is None:
            raise OperationError(f"unsupported execution host platform on {host or 'controller'}")
        return arch

    def cache_directory(self, host):
        base = self.settings.get("remote_cache_dir")
        if not base:
            home = self.execute(host, ["sh", "-c", 'printf "%s" "$HOME"']).decode()
            base = home + "/.cache/oci-relay"
        if not isinstance(base, str) or not base.startswith("/") or "\x00" in base:
            raise OperationError("remote_cache_dir must be an absolute path")
        return base

    def stage_binary(self, host, path: Path, digest: str, version: str) -> str:
        base = self.cache_directory(host)
        destination = f"{base}/{version}/{digest}/oci-relay"
        # Cache hits need one command after locating the host cache. Execute
        # only after checking the full pinned hash, never just path existence.
        check = 'observed=$(sha256sum -- "$1" 2>/dev/null) || exit 42\n' \
                '[ "${observed%% *}" = "$2" ] || exit 42\nexec "$1" version'
        result = self.connection(host).execute(
            host or "localhost", ["sh", "-c", check, "relay-cache", destination, digest], timeout=30,
        )
        if result.returncode == 42:
            self.execute(host, ["mkdir", "-p", str(Path(destination).parent)])
            temporary = destination + ".tmp." + uuid.uuid4().hex
            try:
                self.connection(host).upload(host or "localhost", [str(path)], temporary)
                observed = self.execute(host, ["sha256sum", temporary]).decode().split()
                if not observed or observed[0] != digest or file_digest(path) != digest:
                    raise OperationError("staged relay checksum mismatch")
                self.execute(host, ["chmod", "0555", temporary])
                self.execute(host, ["mv", "-f", temporary, destination])
            finally:
                self.connection(host).execute(host or "localhost", ["rm", "-f", "--", temporary], timeout=10)
            output = self.execute(host, [destination, "version"])
        elif result.returncode:
            raise OperationError(f"{host or 'controller'}: cached relay check failed ({result.returncode})")
        else:
            output = result.stdout
        observed = json.loads(output)
        if observed.get("version") != version or observed.get("protocol") != 1:
            raise OperationError("staged engine version/protocol does not match plugin")
        if not hasattr(self, "binary_capabilities"):
            self.binary_capabilities = {}
        self.binary_capabilities[host] = observed.get("capabilities", [])
        logger.info("OCI Relay engine on %s: version=%s commit=%s protocol=%s",
                    host or "controller", observed["version"], observed.get("commit", "unknown"), observed["protocol"])
        logger.debug("OCI Relay binary on %s: sha256=%s capabilities=%s",
                     host or "controller", digest, self.binary_capabilities[host])
        return destination

    def stage_decoder(self, host, path, digest):
        destination = f"{self.cache_directory(host)}/helpers/{digest}/unpigz"
        result = self.connection(host).execute(host or "localhost", ["sha256sum", destination], timeout=30)
        if result.returncode == 0 and result.stdout.decode().split()[0] == digest:
            return destination
        self.execute(host, ["mkdir", "-p", str(Path(destination).parent)])
        temporary = destination + ".tmp." + uuid.uuid4().hex
        try:
            self.connection(host).upload(host or "localhost", [str(path)], temporary)
            observed = self.execute(host, ["sha256sum", temporary]).decode().split()
            if not observed or observed[0] != digest or file_digest(path) != digest:
                raise OperationError("staged decoder checksum mismatch")
            self.execute(host, ["chmod", "0555", temporary])
            self.execute(host, ["mv", "-f", temporary, destination])
        finally:
            self.connection(host).execute(host or "localhost", ["rm", "-f", "--", temporary], timeout=10)
        return destination

    def start_source(self, host, binary, directory, plan):
        """Pin native layers with an owned container; expose only read-only store trees.

        The relay is the container entrypoint. The image's application is never
        started, and the helper receives no Docker socket.
        """
        if plan["source"] not in {"docker-classic", "docker-containerd"}:
            self.json(host, directory + "/plan.json", plan)
            return self.start(host, [binary, "run", "--plan", directory + "/plan.json"])
        from sparkrun.plugins import ImageDistributionUnsupported

        info = docker_facts(self, host)
        rejection = (containerd_store_reason if plan["source"] == "docker-containerd" else classic_store_reason)(info)
        if rejection:
            raise ImageDistributionUnsupported(plan["source"] + ": " + rejection)
        image_id = self.execute(host, LOCAL_DOCKER + ["image", "inspect", "--format={{.Id}}", plan["image"]]).decode().strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise OperationError("Docker returned an invalid image ID")
        uid = int(self.execute(host, ["id", "-u"]))
        gid = int(self.execute(host, ["id", "-g"]))
        plan = dict(plan, image=image_id, docker_root="/oci-relay-store", engine_version=info["version"],
                    socket_uid=uid, socket_gid=gid)
        self.json(host, directory + "/plan.json", plan)
        return self._native_helper(host, binary, directory, image_id, info,
                                   ["run", "--plan", directory + "/plan.json"])

    def start_receiver(self, host, binary, directory, arguments, inventory):
        if not self.settings.get("allow_native_store", True):
            return self.start(host, arguments)
        info = docker_facts(self, host)
        store = "containerd" if containerd_store_reason(info) is None else "overlay2"
        if store == "overlay2" and classic_store_reason(info) is not None:
            return self.start(host, arguments)
        wanted = inventory.get("diff_ids", [])
        platform = inventory.get("platform", {})
        if not wanted or len(wanted) > 4095 or platform.get("os") != "linux":
            return self.start(host, arguments)
        budget = self.settings.get("cache_discovery_seconds", 10)
        discovered = json.loads(self.execute(host, [binary, "inventory", "--timeout-seconds", str(budget)],
            input_data=json.dumps({"diff_ids": wanted, "platform": platform}).encode(), timeout=budget + 15))
        path = directory + "/cache-inventory.json"
        self.json(host, path, discovered)
        arguments = [*arguments, "--cache-inventory", path]
        image_id = discovered.get("helper_image")
        # Empty stores need no helper. Containerd can probe exact CAS paths even
        # without matching image metadata, using any compatible local image.
        if not image_id or (store == "overlay2" and not discovered.get("candidates")):
            return self.start(host, arguments)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise OperationError("invalid cache helper image ID")
        command = [*arguments[1:], "--native-store", store, "--native-root", "/oci-relay-store",
                   "--engine-version", info["version"], "--native-base", image_id]
        return self._native_helper(host, binary, directory, image_id, info, command, receiver=True)

    def _native_helper(self, host, binary, directory, image_id, info, command, *, receiver=False):
        name = ("oci-relay-receiver-" if receiver else "oci-relay-source-") + uuid.uuid4().hex
        arguments = LOCAL_DOCKER + ["create", "--name", name, "--interactive", "--read-only", "--network", "host",
                     "--user", "0:0", "--cap-drop", "ALL", "--cap-add", "DAC_OVERRIDE",
                     "--cap-add", "CHOWN", "--security-opt", "no-new-privileges",
                     "--runtime", "runc", "--workdir", "/", "--no-healthcheck",
                     "--env", "NVIDIA_VISIBLE_DEVICES=void", "--label",
                     "com.scitrera.oci-relay.receiver=true" if receiver else "com.scitrera.oci-relay.source=true",
                     "--entrypoint", "/oci-relay-bin"]
        if self.settings.get("relay_gomaxprocs", 0):
            arguments.extend(["--env", f"GOMAXPROCS={self.settings['relay_gomaxprocs']}"])
        mounts = native_mounts(info, self.settings) + [
            (binary, "/oci-relay-bin", True), (directory, directory, receiver),
        ]
        if receiver:
            # Only receivers need the local Docker API for import/finalization.
            mounts.append(("/var/run/docker.sock", "/var/run/docker.sock", True))
            if "--unpigz" in command:
                helper = command[command.index("--unpigz") + 1]
                scratch = command[command.index("--decode-spool-dir") + 1]
                mounts.extend([(helper, helper, True), (scratch, scratch, False)])
        for source, target, readonly in mounts:
            if not source.startswith("/") or any(c in source for c in ',\n\r\x00'):
                raise OperationError("invalid native helper bind path")
            arguments.extend(["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")])
        arguments.extend([image_id, *command])
        # Track the unpredictable, operation-owned name before invoking Docker:
        # a disconnected create can succeed without returning its container ID.
        self.containers.append((host, name))
        container = self.execute(host, arguments, timeout=90).decode().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container):
            raise OperationError("Docker returned an invalid helper container ID")
        return self.start(host, LOCAL_DOCKER + ["start", "--attach", "--interactive", container])

    def close(self):
        errors = []
        for process in reversed(self.processes):
            try:
                process.close()
            except Exception as error:
                errors.append(str(error))
        for host, container in reversed(self.containers):
            try:
                self.execute(host, LOCAL_DOCKER + ["rm", "--force", "--volumes", container], timeout=30)
            except Exception as error:
                if "No such container" not in str(error):
                    errors.append(str(error))
        for host, path in reversed(self.directories):
            try:
                self.execute(host, ["rm", "-rf", "--", path], timeout=15)
            except Exception as error:
                errors.append(str(error))
        self.local.close()
        return errors


class Lines:
    """Drain a stream continuously; bound both line length and retained diagnostics."""

    def __init__(self, stream, *, events=False):
        self.queue = queue.Queue(maxsize=16)
        self.tail = deque(maxlen=64)
        self.error = None
        self.finished = threading.Event()
        self.events = events
        self.progress = None
        self.progress_lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self.thread.start()

    def _read(self, stream):
        try:
            # FileIO streams from HostProcess are unbuffered.
            import io

            with io.BufferedReader(stream) as reader:
                limit = (1 << 20) if self.events else (64 << 10)
                while line := reader.readline(limit + 1):
                    if len(line) > limit:
                        raise OperationError("relay output line exceeds limit")
                    text = line.decode(errors="replace").rstrip()
                    self.tail.append(text[-2048:])
                    if self.events:
                        try:
                            event = json.loads(text)
                            if not isinstance(event, dict):
                                raise OperationError("relay event must be an object")
                            if event.get("type") == "progress":
                                with self.progress_lock:
                                    self.progress = event
                                continue
                            self.queue.put_nowait(event)
                        except queue.Full as error:
                            raise OperationError("relay event queue exceeded limit") from error
        except Exception as error:
            self.error = error
        finally:
            self.finished.set()

    def event(self, deadline, health=None):
        while time.monotonic() < deadline:
            if self.error:
                raise OperationError(str(self.error))
            try:
                return self.queue.get(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if health is not None:
                    health()
                with self.progress_lock:
                    progress, self.progress = self.progress, None
                if progress is not None:
                    return progress
                if self.finished.is_set():
                    raise OperationError("relay exited without the required result: " + "\n".join(self.tail)[-4000:])
        raise OperationError("relay readiness/result deadline exceeded")


def pump(source, destination):
    try:
        while chunk := source.read(64 << 10):
            # FileIO.write may make a short write.
            view = memoryview(chunk)
            while view:
                n = destination.write(view)
                if not n:
                    return
                view = view[n:]
    except (OSError, ValueError):
        pass
    finally:
        try:
            destination.close()
        except (OSError, ValueError):
            pass
