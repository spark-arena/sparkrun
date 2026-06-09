"""Shell safety utilities.

Provides helpers for safely interpolating values into shell command strings.
"""

from __future__ import annotations

import base64
import os
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


def validate_hostname(host: str) -> str:
    """Validate and return a hostname or IP address, or raise ValueError.

    Accepts RFC-1123-style hostnames and IPv4/IPv6 addresses.  The pattern
    allows letters, digits, hyphens, underscores, and dots — no shell
    metacharacters.  Labels must start and end with an alphanumeric character.

    Args:
        host: Hostname or IP address string to validate.

    Returns:
        The validated hostname (unchanged).

    Raises:
        ValueError: If *host* contains characters outside the allowed set or
            does not match the expected structure.
    """
    if not host:
        raise ValueError("Hostname must not be empty")
    if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9._-]{0,253}[a-zA-Z0-9])?", host):
        raise ValueError("Invalid hostname %r — contains shell-unsafe characters or bad structure" % host)
    return host


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


# Conservative allowlist for paths interpolated into a sudoers rule.  A
# sudoers file is security-critical: a value containing whitespace, a newline,
# or shell/sudoers metacharacters could append extra rules or commands.  Only
# characters that appear in legitimate absolute filesystem paths are permitted.
_SUDOERS_PATH_RE = re.compile(r"[A-Za-z0-9_./-]+")


def validate_sudoers_path(value: str) -> str:
    """Validate an absolute path destined for interpolation into a sudoers rule.

    Requires an absolute path containing only ``[A-Za-z0-9_./-]`` — no
    whitespace, newlines, or shell/sudoers metacharacters.  This is stricter
    than :func:`assert_safe_path` because the interpolation target (a
    ``/etc/sudoers.d`` rule) is a privilege-escalation primitive.

    Args:
        value: Path string to validate.

    Returns:
        The validated path (unchanged).

    Raises:
        ValueError: If *value* is empty, not absolute, or contains characters
            outside the allowed set.
    """
    if not value:
        raise ValueError("Sudoers path must not be empty")
    if not value.startswith("/"):
        raise ValueError("Sudoers path must be absolute: %r" % value)
    if not _SUDOERS_PATH_RE.fullmatch(value):
        raise ValueError("Unsafe character in sudoers path %r — possible privilege escalation" % value)
    return value


# Host paths that must never be bind-mounted into a workload container.
# Mounting any of these hands the container control of the host (root fs, the
# docker control socket, kernel/dev pseudo-filesystems, system config, …).
_FORBIDDEN_MOUNT_PATHS = frozenset(
    {
        "/",
        "/etc",
        "/boot",
        "/proc",
        "/sys",
        "/dev",
        "/root",
        "/var/run/docker.sock",
        "/run/docker.sock",
    }
)


def assert_safe_mount_source(path: str) -> str:
    """Reject host paths that are catastrophic to bind-mount into a container.

    Defense-in-depth guard for ``docker -v`` sources that originate (even
    indirectly) from a recipe or cluster config.  Mounting the host root, the
    docker control socket, SSH keys, or kernel pseudo-filesystems gives the
    containerized workload control of the host, so these are refused outright —
    independent of recipe trust.  This complements (does not replace) the
    trust gate that blocks untrusted recipes from supplying mounts at all.

    Returns *path* unchanged on success; raises :class:`ValueError` otherwise.
    """
    if not path:
        raise ValueError("Empty bind-mount source path")
    real = os.path.realpath(os.path.expanduser(path))
    if real in _FORBIDDEN_MOUNT_PATHS:
        raise ValueError("Refusing to bind-mount sensitive host path %r (resolved to %r)" % (path, real))
    if os.path.basename(real) == "docker.sock":
        raise ValueError("Refusing to bind-mount the docker control socket %r" % path)
    ssh_dir = os.path.realpath(os.path.expanduser("~/.ssh"))
    if real == ssh_dir or real.startswith(ssh_dir + os.sep):
        raise ValueError("Refusing to bind-mount the SSH key directory %r" % path)
    return path


_ALLOWED_GIT_URL_SCHEMES = ("https://", "git@", "ssh://", "file://")


def validate_git_url(url: str) -> str:
    """Validate a git URL against an allowlist of safe schemes.

    Rejects URLs that could be interpreted as git command-line options (dash-
    leading strings), use unsafe schemes like ``http://``, or are empty.
    This mitigates CVE-2017-1000117-style option injection via URL arguments.

    Allowed schemes: ``https://``, ``git@``, ``ssh://``, ``file://``.

    Args:
        url: The git URL to validate.

    Returns:
        The URL (stripped of leading/trailing whitespace) if valid.

    Raises:
        ValueError: If *url* is empty, starts with a dash, or uses a
            disallowed scheme.
    """
    url = url.strip()
    if not url:
        raise ValueError("Git URL must not be empty")
    if url.startswith("-"):
        raise ValueError("Git URL must not start with '-' (option injection risk): %r" % url)
    if not any(url.startswith(scheme) for scheme in _ALLOWED_GIT_URL_SCHEMES):
        raise ValueError("Git URL %r uses a disallowed scheme. Allowed schemes: %s" % (url, ", ".join(_ALLOWED_GIT_URL_SCHEMES)))
    return url


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
