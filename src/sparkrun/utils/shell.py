"""Shell safety utilities.

Provides helpers for safely interpolating values into shell command strings.
"""

from __future__ import annotations

import base64
import re
import shlex
from typing import Any


def quote(value: str) -> str:
    """Return a shell-safe quoted version of *value*.

    Wraps :func:`shlex.quote` for convenience.
    """
    return shlex.quote(value)


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


def args_list_to_shell_str(args: list[str]) -> str:
    """Prepare a list of arguments for passing to a shell command."""
    # NOTE: no isinstance guard here, bubble up failures
    return " ".join(quote(arg) for arg in args if arg) or ""


def quote_list(source_list: list) -> list:
    """Return a copy of *source_list* with all values shell-quoted."""
    # NOTE: no isinstance guard here, bubble up failures
    return [quote(v) for v in source_list if v]


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


def render_args_as_flags(args: dict[str, Any]) -> list[str]:
    """Render a dict of args as CLI flags (--kebab-case-key value).

    Booleans become bare flags (present when True, absent when False).
    Lists emit repeated flags for each element.
    """
    parts: list[str] = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                parts.extend([flag, str(item)])
            continue
        parts.extend([flag, str(value)])
    return parts
