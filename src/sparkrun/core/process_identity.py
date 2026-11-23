"""Strong local process identities for leases and supervisor recovery."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    if pid <= 0:
        return None
    if sys.platform != "linux":
        import psutil

        try:
            process = psutil.Process(pid)
            if process.status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}:
                return None
            return {"pid": pid, "platform": sys.platform, "created": process.create_time()}
        except psutil.NoSuchProcess:
            return None
        except psutil.AccessDenied as exc:
            # Failed inspection is not proof of death.
            raise PermissionError(f"Cannot inspect process {pid}") from exc
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] in {"Z", "X"}:
            return None
        return {"pid": pid, "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "created": stat[19]}
    except PermissionError:
        raise
    except (OSError, IndexError, ValueError):
        return None


def process_command(pid):
    """Best-effort command line of *pid*, for messages only; ``None`` when unreadable."""
    try:
        if sys.platform == "linux":
            argv = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
            return " ".join(part.decode(errors="replace") for part in argv if part) or None
        import psutil

        return " ".join(psutil.Process(int(pid)).cmdline()) or None
    except Exception:  # A message detail; never worth failing the caller over.
        return None


def alive(identity):
    return bool(identity and process_identity(identity.get("pid", -1)) == identity)


def descendants(identity):
    """Snapshot descendants by creation identity, regardless of sessions."""
    if not alive(identity):
        return []
    if sys.platform != "linux":
        import psutil

        try:
            found = [value for child in psutil.Process(identity["pid"]).children(recursive=True) if (value := process_identity(child.pid))]
            return found if alive(identity) else []
        except psutil.NoSuchProcess:
            return []
    children = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            child = process_identity(int(path.name))
            if child:
                children.setdefault(int(fields[1]), []).append(child)
        except (OSError, ValueError, IndexError):
            continue
    found = []
    pending = [identity]
    while pending:
        for child in children.get(pending.pop()["pid"], []):
            found.append(child)
            pending.append(child)
    return found if alive(identity) else []


def terminate_tree(identity, *, include_root=True):
    """Force only verified members of an owned tree; never trust a PID alone."""
    import signal

    if not alive(identity):
        return
    if sys.platform != "linux":
        import psutil

        root = psutil.Process(identity["pid"])
        if include_root and alive(identity):
            root.suspend()
        try:
            for child in reversed(descendants(identity)):
                try:
                    process = psutil.Process(child["pid"])
                    if alive(child):
                        process.kill()
                except psutil.NoSuchProcess:
                    pass
        finally:
            if include_root and alive(identity):
                root.kill()
        return
    # Stop the root from creating further children during the snapshot.
    if include_root:
        os.kill(identity["pid"], signal.SIGSTOP)
    try:
        members = descendants(identity)
        for child in reversed(members):
            if alive(child):
                try:
                    os.kill(child["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
    finally:
        if include_root and alive(identity):
            os.kill(identity["pid"], signal.SIGKILL)


def user_identity():
    """Stable OS account identity, independent of environment variables."""
    if os.name == "nt":
        from sparkrun.utils.windows import user_id

        return user_id()
    return str(os.getuid())


class DetachedProcessOptions(TypedDict, total=False):
    creationflags: int
    start_new_session: bool


def detached_process_options() -> DetachedProcessOptions:
    """A background API/worker without an inherited console or POSIX session."""
    if os.name == "nt":
        import subprocess

        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}
