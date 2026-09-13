"""Non-destructive local process probes for POSIX and Windows control nodes."""

from __future__ import annotations

import os
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def _windows_kernel():
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _windows_process_exists(pid: int) -> bool:
    import ctypes

    kernel = _windows_kernel()
    # SYNCHRONIZE is sufficient to observe process termination. Never request
    # terminate rights: unlike POSIX, Windows os.kill(pid, 0) kills the process.
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: no such PID
            return False
        raise ctypes.WinError(error)
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        if result == 0xFFFFFFFF:  # WAIT_FAILED
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 0x00000102  # WAIT_TIMEOUT: still running
    finally:
        kernel.CloseHandle(handle)


def process_exists(pid: int | None, *, inaccessible: bool = False) -> bool:
    """Observe a positive PID, optionally treating denied access as existence."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            return _windows_process_exists(pid)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return inaccessible
