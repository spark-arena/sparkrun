"""Shell safety utilities.

Provides helpers for safely interpolating values into shell command strings.
"""

from __future__ import annotations

import base64
import re
import shlex
from typing import Any


class Quoted(str):
    """A string that has already been shell-quoted.

    Used as a sentinel to prevent double-quoting when a value passes through
    multiple layers that each call :func:`quote`.  Long-term the goal is to
    quote only at the shell-assembly boundary, making this unnecessary.
    """


def quote(value: str) -> Quoted:
    """Return a shell-safe quoted version of *value*.

    If *value* is already a :class:`Quoted` instance it is returned unchanged,
    making this function idempotent.

    Wraps :func:`shlex.quote` for convenience.
    """
    if isinstance(value, Quoted):
        return value
    return Quoted(shlex.quote(value))


def b64_encode_cmd(cmd: str) -> str:
    """Base64 encode a command string to avoid shell escaping issues.

    Useful when passing complex commands (e.g., with nested quotes or JSON)
    across SSH boundaries or into ``docker exec``.
    """
    return base64.b64encode(cmd.encode("utf-8")).decode("utf-8")


def b64_wrap_bash(cmd: str, quoted: bool = True) -> str:
    """Wrap a command in a base64 pipeline that decodes and executes via bash.

    Produces a string like: `printf %s <b64> | base64 -d -- | bash`

    If quoted is True, then the result is properly shell quoted as well.
    The b64 payload is inherently shell-safe ([A-Za-z0-9+/=]) so quoting
    around the printf args is unnecessary and avoids ugly escaped output.
    """
    b64_cmd = b64_encode_cmd(cmd)
    # Using printf instead of echo is safer against strings starting with dashes.
    # Adding -- to base64 -d prevents interpretation of the b64 string as flags.
    # Using --noprofile --norc with bash ensures a clean execution environment.
    # No quotes around %s or the b64 payload — b64 is [A-Za-z0-9+/=], all shell-safe.
    result = f"printf %s {b64_cmd} | base64 -d -- | bash --noprofile --norc"
    if quoted:
        return quote(result)
    return result


def b64_wrap_python(script: str, quoted: bool = True) -> str:
    """Wrap a Python script in a base64 pipeline that decodes and executes via python3.

    Produces a string like: `printf '%s' <b64> | base64 -d -- | python3`

    Avoids all shell-escaping issues when delivering Python scripts across SSH
    boundaries (same motivation as :func:`b64_wrap_bash`).

    If quoted is True, the result is shell-quoted for safe embedding in
    further shell commands.
    """
    b64_script = b64_encode_cmd(script)
    result = f"printf %s {b64_script} | base64 -d -- | python3"
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


# Characters that can perform command injection when a path is interpolated
# into a shell script (even inside double quotes).  Tilde, slash, dot, dash,
# underscore, plus, at, colon, comma, equals, and spaces are intentionally
# allowed — they are safe inside double-quoted shell strings and common in
# legitimate paths.
_SHELL_INJECTION_RE = re.compile(r"[;&|`$!\"'<>()\{\}\n\\]")


def assert_safe_path(value: str) -> str:
    """Validate that *value* is free of shell-injection metacharacters.

    Intended for paths that will be interpolated into shell scripts inside
    double quotes.  Raises :class:`ValueError` if dangerous characters are
    found.

    Returns *value* unchanged on success so it can be used inline::

        script = 'rsync "%s"/ dest/' % assert_safe_path(source)
    """
    match = _SHELL_INJECTION_RE.search(value)
    if match:
        raise ValueError("Unsafe character %r in path %r — possible shell injection" % (match.group(), value))
    return value


def safe_remote_path(value: str) -> str:
    """Validate and prepare a path for interpolation into a remote shell script.

    1. Validates that *value* contains no injection metacharacters (via
       :func:`assert_safe_path`).
    2. Converts a leading ``~/`` to ``$HOME/`` so the path expands correctly
       inside double-quoted shell strings.  (Tilde expansion only works in
       unquoted contexts in bash, but ``$HOME`` expands inside double quotes.)

    Returns the transformed path, ready for ``"%(path)s"`` interpolation::

        script = 'docker cp "%s"/. ctr:dest/' % safe_remote_path(source)
    """
    assert_safe_path(value)
    if value.startswith("~/"):
        return "$HOME/" + value[2:]
    if value == "~":
        return "$HOME"
    return value


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
