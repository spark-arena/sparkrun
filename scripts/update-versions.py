#!/usr/bin/env python3
"""Drop-in shim for scitrera-repo-tools `sync-versions`.

Copy this file into any repo's `scripts/` directory (or run it from anywhere)
to sync versions defined in that repo's `versions.yaml`.

Resolution order:

  1. Package already importable in the current Python  ->  call directly
     (fast path; exact version is whatever the current env has installed).

  2. `uvx` (or `uv`) on PATH  ->  run via uvx with no persistent install.
     Set `REPO_TOOLS_SOURCE` to override the package source. Default:
     `git+https://github.com/scitrera/repo-tools.git`. Pin to a tag or
     PyPI version, e.g.:
         REPO_TOOLS_SOURCE='scitrera-repo-tools==0.1.0'

  3. Otherwise  ->  print install instructions and exit 1.

Usage:
    python scripts/update-versions.py            # sync
    python scripts/update-versions.py --check    # CI-friendly dry-run
    python scripts/update-versions.py --verbose

All flags pass through to `sync-versions`.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List

DEFAULT_SOURCE = "git+https://github.com/scitrera/repo-tools.git"


def _try_uvx(args: List[str]) -> None:
    """Replace the current process with a uvx invocation.

    Returns only if neither uvx nor uv is on PATH.
    """
    uv = shutil.which("uvx") or shutil.which("uv")
    if uv is None:
        return
    source = os.environ.get("REPO_TOOLS_SOURCE", DEFAULT_SOURCE)
    name = os.path.basename(uv)
    if name == "uv":
        cmd = [uv, "tool", "run", "--from", source, "sync-versions", *args]
    else:
        cmd = [uv, "--from", source, "sync-versions", *args]
    os.execvp(cmd[0], cmd)  # never returns


def _try_import(args: List[str]) -> bool:
    try:
        from scitrera_repo_tools.version_sync.cli import main as sync_main
    except ImportError:
        return False
    sys.argv = ["sync-versions", *args]
    sync_main()
    return True


def main(argv: List[str]) -> int:
    if _try_import(argv):
        return 0
    _try_uvx(argv)  # never returns if uvx/uv is on PATH
    source = os.environ.get("REPO_TOOLS_SOURCE", DEFAULT_SOURCE)
    sys.stderr.write(
        "scitrera-repo-tools is not available.\n"
        "Install one of:\n"
        f"  uv tool install --from {source} scitrera-repo-tools\n"
        f"  pip install {source}\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
