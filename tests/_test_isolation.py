"""Test-only home isolation and a backstop for accidental external I/O.

Local sockets and fixture executables remain real. Python audit events catch
Popen even when a test replaces subprocess.run, and retained attempts expose
best-effort application code that catches the exception. Native children do
not inherit Python hooks: their fixtures must supply local inputs/stub tools.
"""

from __future__ import annotations

import ipaddress
import ntpath
import os
import re
from pathlib import Path
import shutil
import shlex
import socket
import threading

_REAL_WHICH = shutil.which


def isolate_environment(monkeypatch, home: Path, *, clean_overrides=False):
    home.mkdir(parents=True, exist_ok=True)
    if clean_overrides:
        for name in tuple(os.environ):
            # An explicitly supplied uv artifact cache is a dependency input
            # for offline wheel tests; CI supplies a job-local directory.
            if (name.endswith(("_CONFIG_DIR", "_CACHE_DIR")) and name != "UV_CACHE_DIR") or name in {
                "SPARKRUN_APPLICATION_PROFILE",
                "SPARKRUN_APPLICATION_CONFIG",
                "STATEFUL_ROOT",
            }:
                monkeypatch.delenv(name, raising=False)
    # Path.home/expanduser and child interpreters must agree, including Windows.
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    for name, suffix in (
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_CACHE_HOME", ".cache"),
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_STATE_HOME", ".local/state"),
        ("HF_HOME", ".cache/huggingface"),
        ("HF_HUB_CACHE", ".cache/huggingface/hub"),
        ("HUGGINGFACE_HUB_CACHE", ".cache/huggingface/hub"),
        ("HF_TOKEN_PATH", ".cache/huggingface/token"),
    ):
        monkeypatch.setenv(name, str(home / suffix))
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "1")
    monkeypatch.setenv("SPARKRUN_NO_EXTERNAL_PLUGINS", "1")
    monkeypatch.setenv("SPARKRUN_NO_INSTALLED_PLUGINS", "1")


def numeric_host(host):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


class IOGuard:
    def __init__(self):
        self.active = True
        self.attempts: list[str] = []
        self.fixture_roots: list[Path] = []
        self.allowed_commands: set[str] = set()
        self._local = threading.local()

    def block(self, message):
        self.attempts.append(message)
        raise OSError("Unmocked external I/O: " + message)

    def __call__(self, event, args):
        if not self.active or getattr(self._local, "busy", False):
            return
        if event not in {
            "subprocess.Popen",
            "socket.connect",
            "socket.sendto",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
        }:
            return
        self._local.busy = True
        try:
            self.check(event, args)
        finally:
            self._local.busy = False

    def check(self, event, args):
        if event == "subprocess.Popen":
            executable, argv, cwd, env = args
            # Windows audits a rendered command line and often executable=None;
            # POSIX supplies argv and the resolved executable separately.
            if executable is None:
                executable = re.match(r'\s*("[^"]*"|\S+)', argv)[1].strip('"') if isinstance(argv, str) else argv[0]
            command = Path(ntpath.basename(os.fsdecode(executable))).stem.lower()
            if command in self.allowed_commands:
                return
            guarded = {
                "ssh",
                "ssh-keyscan",
                "scp",
                "sftp",
                "rsync",
                "docker",
                "kubectl",
                "uvx",
                "curl",
                "wget",
                "systemctl",
                "systemd-run",
                "sudo",
            }
            if command not in guarded:
                return
            if isinstance(argv, str):
                # Only inspect guarded tools. Python/shell payload quoting is
                # opaque to this guard and need not obey shlex's grammar.
                try:
                    argv = [arg[1:-1] if arg.startswith('"') and arg.endswith('"') else arg for arg in shlex.split(argv, posix=False)]
                except ValueError:
                    self.block("unparseable %s command line" % command)
            resolved = Path(_REAL_WHICH(executable, path=(env or os.environ).get("PATH")) or executable).resolve()
            if any(resolved.is_relative_to(root) for root in self.fixture_roots):
                return
            if isinstance(argv, (list, tuple)):
                if command == "ssh" and "-G" in argv[1:]:
                    return  # OpenSSH's local config parser, no connection.
                if len(argv) == 2 and argv[1] in {"--help", "--version", "-V"}:
                    return
                if (
                    command == "rsync"
                    and len(argv) >= 3
                    and all(
                        any(Path(value).is_absolute() and Path(value).resolve().is_relative_to(root) for root in self.fixture_roots)
                        for value in argv[-2:]
                    )
                ):
                    return
            self.block("process %r" % (argv,))
        elif event in {"socket.connect", "socket.sendto"}:
            sock, address = args[0], args[-1]
            if sock.family not in {socket.AF_INET, socket.AF_INET6}:
                return
            host = address[0]
            ip = numeric_host(host)
            if host == "localhost" or (ip is not None and ip.is_loopback):
                return
            if event == "socket.connect" and sock.type == socket.SOCK_DGRAM:
                return  # Route selection only; sending a datagram is still blocked.
            self.block("%s %r" % (event, address))
        else:
            host = args[0]
            ip = numeric_host(host)
            if host in {None, "", "localhost", "0.0.0.0", "::"} or (ip is not None and (event != "socket.gethostbyaddr" or ip.is_loopback)):
                return
            self.block("%s %r" % (event, host))
