"""Shell safety utilities.

Provides helpers for safely interpolating values into shell command strings.
"""

from __future__ import annotations

import base64
import re
import shlex


def b64_encode_cmd(cmd: str) -> str:
    """Base64 encode a command string to avoid shell escaping issues.

    Useful when passing complex commands (e.g., with nested quotes or JSON)
    across SSH boundaries or into ``docker exec``.
    """
    return base64.b64encode(cmd.encode("utf-8")).decode("utf-8")


def b64_wrap_bash(cmd: str, quoted: bool = True) -> str:
    """Wrap a command in a base64 pipeline that decodes and executes via bash.

    Produces a string like: `printf '%s' <b64> | base64 -d -- | bash`

    If quoted is True, then the result is properly shell quoted as well.
    """
    b64_cmd = b64_encode_cmd(cmd)
    # Using printf instead of echo is safer against strings starting with dashes.
    # Adding -- to base64 -d prevents interpretation of the b64 string as flags.
    # Using --noprofile --norc with bash ensures a clean execution environment.
    result = f"printf '%s' '{b64_cmd}' | base64 -d -- | bash --noprofile --norc"
    if quoted:
        return quote(result)
    return result


def quote(value: str) -> str:
    """Return a shell-safe quoted version of *value*.

    Wraps :func:`shlex.quote` for convenience.
    """
    return shlex.quote(value)


def quote_dict(d: dict) -> dict:
    """Return a copy of *d* with all string values shell-quoted."""
    return {k: shlex.quote(v) if isinstance(v, str) else v for k, v in d.items()}


def validate_unix_username(user: str) -> str:
    """Validate and return a Unix username, or raise ValueError.

    Accepts names matching ``[a-z_][a-z0-9_-]*$?`` — the POSIX
    portable character set for user names.

    Args:
        user: Username string to validate.

    Returns:
        The validated username (unchanged).

    Raises:
        ValueError: If *user* contains characters outside the allowed set.
    """
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*\$?", user):
        raise ValueError("Invalid username: %r" % user)
    return user
