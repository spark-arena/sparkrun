"""Shell safety utilities.

Provides helpers for safely interpolating values into shell command strings.
"""

from __future__ import annotations

import re
import shlex


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
