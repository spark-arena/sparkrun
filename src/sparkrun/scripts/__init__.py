"""Embedded bash scripts for remote execution.

Scripts are stored as .sh files alongside this module and loaded
via :func:`read_script` at runtime.

Scripts may share code with a ``# sparkrun:include <file.sh>`` directive on a
line of its own, which :func:`read_script` replaces with that file's contents
(see :data:`INCLUDE_DIRECTIVE`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from importlib import resources

from sparkrun.utils.resource_loader import load_resource
from sparkrun.utils.shell import quote

#: Line prefix marking a script include.  A line whose stripped form starts
#: with this is replaced by the named script's contents.  Deliberately a bash
#: comment, so an un-processed script is still valid (it just lacks the helper)
#: and ``shellcheck`` can read the file as-is.
INCLUDE_DIRECTIVE = "# sparkrun:include "


def _resolve_includes(text: str, _seen: frozenset[str] = frozenset()) -> str:
    """Expand ``# sparkrun:include`` directives in *text*, recursively.

    Raises:
        ValueError: On a circular include.
        FileNotFoundError: If an included script does not exist.
    """
    if INCLUDE_DIRECTIVE not in text:
        return text

    out: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped.startswith(INCLUDE_DIRECTIVE):
            out.append(line)
            continue
        name = stripped[len(INCLUDE_DIRECTIVE) :].strip()
        if name in _seen:
            raise ValueError("Circular script include: %s" % name)
        body = _resolve_includes(load_resource(__name__, name), _seen | {name})
        out.append(body if body.endswith("\n") else body + "\n")
    return "".join(out)


def read_script(name: str) -> str:
    """Read a bash script from the scripts package.

    ``# sparkrun:include`` directives are expanded before the script is
    returned, so callers always receive a self-contained script.

    Args:
        name: Script filename (e.g. ``"ip_detect.sh"``).

    Returns:
        Script content as a string.
    """
    from sparkrun.core.application_profile import get_application_profile

    script = _resolve_includes(load_resource(__name__, name))
    return script.replace("@RESOURCE_NAMESPACE@", get_application_profile().resource_namespace).replace(
        "@CACHE_NAMESPACE@", get_application_profile().cache_namespace
    )


def inject_shell_vars(script: str, **values: str | None) -> str:
    """Return *script* with ``NAME=value`` assignments inserted at the top.

    Used to pass configuration into a script that is piped to ``bash -s``,
    which takes no arguments and inherits no environment from the control
    machine.  Values are shell-quoted; ``None`` and empty values are skipped,
    so a caller with nothing to pin gets the script back untouched.

    Assignments go *after* the shebang (so the script stays directly
    executable) and before every statement, which is what lets an included
    helper read them — see :func:`inject_mgmt_vars` and ``_mgmt_iface.sh``.
    A helper read this way must therefore not default the variable itself:
    it is included partway down the script, so its own assignment would run
    *after* the injected one and clobber it.
    """
    assignments = ["%s=%s" % (name, quote(str(value))) for name, value in sorted(values.items()) if value]
    if not assignments:
        return script
    lines = script.split("\n")
    at = 1 if lines and lines[0].startswith("#!") else 0
    return "\n".join(lines[:at] + assignments + lines[at:])


# A peer address as ``_mgmt_iface.sh`` can use it: an IPv4 address or a
# hostname (resolved on the host).  Anything else is dropped rather than
# passed along, so a comma can never smuggle in a second entry.
_PEER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def mgmt_peer_list(hosts: Iterable[str] | None) -> str | None:
    """Normalize cluster *hosts* into the ``SPARKRUN_MGMT_PEERS`` value.

    Strips ``user@`` prefixes, drops loopback and anything that is not a plain
    address or hostname, and dedupes in order.  Returns ``None`` when nothing
    usable remains, which :func:`inject_shell_vars` treats as "not set".
    """
    peers: list[str] = []
    for host in hosts or ():
        peer = str(host).rsplit("@", 1)[-1].strip()
        if not peer or peer in peers or peer == "localhost" or peer.startswith("127.") or not _PEER_RE.match(peer):
            continue
        peers.append(peer)
    return ",".join(peers) or None


def inject_mgmt_vars(script: str, *, mgmt_interface: str | None = None, peers: Iterable[str] | None = None) -> str:
    """Pass management-interface inputs to a script including ``_mgmt_iface.sh``.

    *mgmt_interface* is the cluster's pinned interface (``SPARKRUN_MGMT_IFACE``);
    *peers* are the cluster's hosts as sparkrun reaches them
    (``SPARKRUN_MGMT_PEERS``), which the helper measures each interface's
    routes against.  The probed host may appear in *peers*: the helper skips
    its own addresses.  With neither, the script is returned untouched.

    Apply this to the finished script: a ``str.format()`` template must be
    formatted first.
    """
    return inject_shell_vars(script, SPARKRUN_MGMT_IFACE=mgmt_interface, SPARKRUN_MGMT_PEERS=mgmt_peer_list(peers))


def get_script_path(name: str):
    """Return a context manager that yields a filesystem :class:`~pathlib.Path` for a script.

    Usage::

        with get_script_path("mesh_ssh_keys.sh") as path:
            subprocess.run(["bash", str(path), ...])

    The context manager guarantees the path exists on disk even when the
    package is installed inside a zip archive.
    """
    return resources.as_file(resources.files(__package__).joinpath(name))
