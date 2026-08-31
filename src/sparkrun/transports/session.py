"""Executable host sessions supplied by cluster transports.

The session is the runtime peer of ``Transport.prepare``: preparation makes a
target reachable, while a session executes exact argv and credentialed image
operations through that prepared path.  It is deliberately independent of any
one integration: what a session offers is a property of the transport, not of
whoever happens to be calling it.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
from urllib.parse import quote as urlquote

from sparkrun.orchestration.ssh import build_ssh_cmd, should_run_locally
from sparkrun.utils.shell import quote


@dataclass(frozen=True)
class HostCommandResult:
    host: str
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class HostSessionError(RuntimeError):
    """The manager could not execute an operation on a prepared host."""


class HostSession(Protocol):
    """Executable session contract implemented by cluster transports."""

    provider_name: str

    def execute(
        self,
        host: str,
        arguments: list[str],
        *,
        input_data: bytes | None = None,
        combined: bool = False,
        timeout: float | None = None,
    ) -> HostCommandResult: ...

    def upload(self, host: str, sources: list[str], destination: str, *, recursive: bool = False) -> None: ...

    def docker_registry(self, host: str, operation: str, reference: str) -> None: ...

    def close(self) -> None: ...


class SshHostSession:
    """Concurrent, cancellable argv execution through Sparkrun's SSH config."""

    provider_name = "sparkrun-ssh"

    def __init__(
        self,
        *,
        ssh_user: str | None = None,
        ssh_key: str | None = None,
        ssh_options: list[str] | None = None,
        connect_timeout: int = 10,
    ):
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.ssh_options = list(ssh_options or ())
        self.connect_timeout = connect_timeout
        self._lock = Lock()
        self._processes: set[subprocess.Popen] = set()
        self._closed = False

    def execute(
        self,
        host: str,
        arguments: list[str],
        *,
        input_data: bytes | None = None,
        combined: bool = False,
        timeout: float | None = None,
    ) -> HostCommandResult:
        if not host or not arguments or any(not isinstance(value, str) or "\x00" in value for value in arguments):
            raise HostSessionError("host command is invalid")
        if should_run_locally(host, self.ssh_user):
            command = list(arguments)
        else:
            command = build_ssh_cmd(
                host,
                ssh_user=self.ssh_user,
                ssh_key=self.ssh_key,
                ssh_options=self.ssh_options,
                connect_timeout=self.connect_timeout,
            )
            command.append(" ".join(str(quote(argument)) for argument in arguments))
        return self._run(command, host, input_data=input_data, combined=combined, timeout=timeout)

    def upload(self, host: str, sources: list[str], destination: str, *, recursive: bool = False) -> None:
        if not host or not sources or not destination or any("\x00" in value for value in [*sources, destination]):
            raise HostSessionError("host upload is invalid")
        if should_run_locally(host, self.ssh_user):
            command = ["cp"]
            if recursive:
                command.append("-R")
            command.extend(sources)
            command.append(destination)
        else:
            command = ["scp", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.connect_timeout}"]
            if self.ssh_key:
                command.extend(["-i", self.ssh_key])
            command.extend(self.ssh_options)
            if recursive:
                command.append("-r")
            command.extend(sources)
            target = f"{self.ssh_user}@{host}" if self.ssh_user else host
            command.append(target + ":" + destination)
        result = self._run(command, host)
        self._require_success("upload", result)

    def docker_registry(self, host: str, operation: str, reference: str) -> None:
        if operation not in {"pull", "push"} or not host or not reference or "\x00" in reference:
            raise HostSessionError("Docker registry operation is invalid")
        environment = None
        if should_run_locally(host, self.ssh_user):
            command = ["docker", "image", operation, reference]
        else:
            authority = host
            if self.ssh_user:
                authority = urlquote(self.ssh_user, safe="") + "@" + host
            command = ["docker", "--host", "ssh://" + authority, "image", operation, reference]
            ssh = build_ssh_cmd(
                host,
                ssh_user=self.ssh_user,
                ssh_key=self.ssh_key,
                ssh_options=self.ssh_options,
                connect_timeout=self.connect_timeout,
            )[:-1]
            environment = {**os.environ, "DOCKER_SSH_COMMAND": " ".join(str(quote(value)) for value in ssh)}
        result = self._run(command, host, environment=environment)
        self._require_success("Docker image " + operation, result)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            processes = tuple(self._processes)
        for process in processes:
            self._terminate(process)

    def _run(
        self,
        command: list[str],
        host: str,
        *,
        input_data: bytes | None = None,
        combined: bool = False,
        timeout: float | None = None,
        environment: dict[str, str] | None = None,
    ) -> HostCommandResult:
        with self._lock:
            if self._closed:
                raise HostSessionError("host session is closed")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if combined else subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
        except OSError as error:
            raise HostSessionError(f"start host command on {host}: {error}") from error
        with self._lock:
            if self._closed:
                self._terminate(process)
                raise HostSessionError("host session closed while command started")
            self._processes.add(process)
        try:
            try:
                stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                self._terminate(process)
                stdout, stderr = process.communicate()
                raise HostSessionError(f"host command on {host} timed out") from error
            return HostCommandResult(host, process.returncode, stdout or b"", stderr or b"")
        finally:
            with self._lock:
                self._processes.discard(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _require_success(operation: str, result: HostCommandResult) -> None:
        if result.returncode == 0:
            return
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        suffix = ": " + detail[-2000:] if detail else ""
        raise HostSessionError(f"{operation} on {result.host} exited {result.returncode}{suffix}")


__all__ = ["HostCommandResult", "HostSession", "HostSessionError", "SshHostSession"]
