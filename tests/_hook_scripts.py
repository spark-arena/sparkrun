"""Read the hook script out of the command an executor generated for it."""

from __future__ import annotations

import base64
import re
import shlex


def hook_payload(script: str) -> str:
    """The bash text a hook command runs, with the executor's base64 layer removed.

    ``Executor.exec_cmd`` ships the hook script as ``printf %s <b64> | base64
    -d -- | bash``; assertions about what runs in the container read this.
    """
    inner = shlex.split(script)[-1]
    match = re.search(r"printf %s (\S+) \|", inner)
    assert match, "not an executor exec command: %r" % script
    return base64.b64decode(match.group(1)).decode()
