"""Changed SSH host keys on the control machine: find them, and replace one on approval.

A re-imaged Spark presents a new host key, and OpenSSH then refuses every
connection to it ("REMOTE HOST IDENTIFICATION HAS CHANGED"), which stops
``setup ssh`` at its first probe. The same message is what a man-in-the-middle
looks like, so the decision is the operator's: this module reports old and new
fingerprints and replaces an entry only for a fingerprint the operator
approved. It never decides on its own (``SshProbe.host_key_failed`` is still
never auto-fixed).

A replacement changes exactly one thing: the recorded key of the approved
**type**, under the exact name ssh checks. Other key types, other names on a
shared line, wildcard entries and ``@cert-authority`` / ``@revoked`` markers
are left alone. That is why ``known_hosts`` is edited here rather than with
``ssh-keygen -R``, which removes every type for the host (re-adding them would
trust keys nobody looked at) and deletes pattern and multi-name lines that
cover other hosts too.

Scope is the control machine's ``known_hosts`` as ssh resolves it (the first
``UserKnownHostsFile``); host-to-host entries the mesh writes are refreshed by
``scripts/known_hosts_refresh.sh``. A stale key in a second user file or in
``GlobalKnownHostsFile`` is not found here; ssh then reports it as before.

Console-free and Windows-native like the rest of :mod:`sparkrun.api.setup`:
only ``ssh`` (for ``-G``) and ``ssh-keyscan``. Matching hashed entries and
writing them (``HashKnownHosts``) is done here, so detection and replacement
read the file the same way.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ._errors import OpenSshUnavailable, SshAccessError

logger = logging.getLogger(__name__)

#: A name sparkrun will write a trust anchor under: a host, an address, or
#: ``[host]:port``. Patterns (``*``, ``?``, ``!``) and lists (``,``) are not.
_LOOKUP_RE = re.compile(r"^(?:\[[A-Za-z0-9._:%-]+\]:[0-9]{1,5}|[A-Za-z0-9._:%-]+)$")

#: The lock ``scripts/known_hosts_refresh.sh`` takes (in ``~/.ssh``), so the two writers serialize.
_LOCK_NAME = ".known_hosts.sparkrun.lock"


@dataclass(frozen=True)
class HostKeyChange:
    """A host whose presented key differs from what ``known_hosts`` records for that key type."""

    host: str
    #: The ``known_hosts`` name ssh checks: ``HostKeyAlias``, ``host`` or ``[host]:port``.
    lookup: str
    key_type: str
    #: SHA256 fingerprints recorded for *key_type* (what ssh trusts now).
    recorded: tuple[str, ...]
    #: SHA256 fingerprint the host presents now (what replacing would trust).
    presented: str
    known_hosts: str


@dataclass(frozen=True)
class HostKeyReplacement:
    """The outcome of :func:`replace_host_key`."""

    host: str
    lookup: str
    key_type: str
    fingerprint: str
    known_hosts: str


@dataclass(frozen=True)
class _Target:
    host: str
    hostname: str
    port: str
    lookup: str
    known_hosts: Path
    hashed: bool


@dataclass(frozen=True)
class _Entry:
    """One parsed ``known_hosts`` line that names *lookup* (exactly, or by pattern)."""

    index: int
    marker: str
    key_type: str
    blob: str
    #: True when the line names *lookup* only through a pattern (shared with other hosts).
    by_pattern: bool


def key_fingerprint(key_blob: str) -> str:
    """OpenSSH's ``SHA256:...`` fingerprint of a base64 public-key blob.

    Raises:
        ValueError: *key_blob* is not base64.
    """
    try:
        raw = base64.b64decode(key_blob, validate=True)
    except binascii.Error as e:
        raise ValueError("not a base64 key blob") from e
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")


def _fingerprint_or_none(blob: str) -> str | None:
    """The fingerprint, or ``None`` for a corrupt blob (one bad line must not hide the rest)."""
    try:
        return key_fingerprint(blob)
    except ValueError:
        return None


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise OpenSshUnavailable(
            "%s not found on PATH. Install the OpenSSH client (Windows: Settings > Apps > Optional Features > OpenSSH Client)." % tool
        )
    return path


def _resolve_target(host, user, key, options, timeout) -> _Target:
    """Where ssh would look for *host*'s key, from ssh's own resolved config.

    Raises:
        SshAccessError: The resolved name is not one a key can safely be written under.
        ValueError / OSError / subprocess.SubprocessError: ``ssh -G`` failed.
    """
    from sparkrun.orchestration.ssh import effective_ssh_config

    cfg = effective_ssh_config(host, ssh_user=user, ssh_key=key, ssh_options=options, timeout=timeout)

    def first(name, default):
        values = cfg.get(name) or []
        return values[0].split()[0] if values and values[0].split() else default

    for proxy in ("proxyjump", "proxycommand"):
        if first(proxy, "none").lower() != "none":
            # ssh-keyscan connects directly; through a proxy that may be another machine.
            raise SshAccessError("%s is reached through %s; its host key is not checked here" % (host, proxy))
    hostname = first("hostname", host)
    port = first("port", "22")
    alias = first("hostkeyalias", "")
    lookup = alias or (hostname if port == "22" else "[%s]:%s" % (hostname, port))
    if not _LOOKUP_RE.match(lookup) or hostname.startswith("-") or not port.isdigit():
        raise SshAccessError("%s resolves to %r, which is not a host name sparkrun will record a key under" % (host, lookup))
    known_hosts = _first_known_hosts((cfg.get("userknownhostsfile") or ["~/.ssh/known_hosts"])[0])
    return _Target(host, hostname, port, lookup, known_hosts, first("hashknownhosts", "no").lower() == "yes")


def _first_known_hosts(value: str) -> Path:
    """The first file of ``ssh -G``'s space-joined ``userknownhostsfile`` value.

    The files are not quoted there, so a path containing a space (a Windows
    profile directory) is recovered as the longest leading run of words that
    names an existing file, before falling back to the first word.
    """
    words = value.split()
    for count in range(len(words), 0, -1):
        candidate = Path(os.path.expanduser(" ".join(words[:count])))
        if candidate.is_file():
            return candidate
    return Path(os.path.expanduser(words[0] if words else "~/.ssh/known_hosts"))


def _scan(target: _Target, timeout: int) -> dict[str, str]:
    """``key_type -> base64 blob`` the host presents now (empty if it did not answer)."""
    keyscan = _require("ssh-keyscan")
    result = subprocess.run(
        [keyscan, "-T", str(timeout), "-p", target.port, "--", target.hostname],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout + 10,
    )
    return parse_keyscan(result.stdout, target.hostname, target.port)


def parse_keyscan(output: str, hostname: str, port: str = "22") -> dict[str, str]:
    """``key_type -> base64 blob`` from ``ssh-keyscan`` output, for *hostname* only.

    ssh-keyscan prints the host field first; a line naming anything else is
    not this host's key, whatever produced it.
    """
    names = {hostname.lower(), "[%s]:%s" % (hostname.lower(), port)}
    presented: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and not line.startswith("#") and parts[0].lower() in names:
            presented.setdefault(parts[1], parts[2])
    return presented


def _hashed_name_matches(field: str, lookup: str) -> bool:
    """Whether a ``|1|salt|hash`` host field is the HMAC-SHA1 of *lookup*."""
    parts = field.split("|")
    if len(parts) != 4 or parts[1] != "1":
        return False
    try:
        salt = base64.b64decode(parts[2], validate=True)
        expected = base64.b64decode(parts[3], validate=True)
    except binascii.Error:
        return False
    return hmac.compare_digest(hmac.new(salt, lookup.encode(), hashlib.sha1).digest(), expected)


def _hashed_name(lookup: str) -> str:
    """A ``HashKnownHosts`` host field for *lookup* (fresh salt, as ``ssh-keygen -H`` writes)."""
    salt = secrets.token_bytes(20)
    digest = hmac.new(salt, lookup.encode(), hashlib.sha1).digest()
    return "|1|%s|%s" % (base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def _pattern_matches(pattern: str, name: str) -> bool:
    """OpenSSH's host pattern: ``*`` and ``?`` only; everything else, ``[`` included, is literal."""
    regex = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
    return re.fullmatch(regex, name, flags=re.DOTALL) is not None


def _names_match(field: str, lookup: str) -> tuple[bool, bool]:
    """``(exact, by_pattern)`` for a host field against *lookup*, with ssh's negation rule.

    *by_pattern* is True whenever a pattern on the line matches, even beside an
    exact name: removing only the exact name would leave the stale key trusted.
    """
    if field.startswith("|"):
        return _hashed_name_matches(field, lookup), False
    lowered = lookup.lower()
    exact = by_pattern = negated = False
    for name in field.lower().split(","):
        if name.startswith("!"):
            negated = negated or _pattern_matches(name[1:], lowered)
        elif name == lowered:
            exact = True
        elif ("*" in name or "?" in name) and _pattern_matches(name, lowered):
            by_pattern = True
    if negated:
        return False, False
    return exact, by_pattern


def _entries(lines: list[str], lookup: str) -> list[_Entry]:
    """Lines of a ``known_hosts`` file that apply to *lookup*, as ssh would match them."""
    found = []
    for index, line in enumerate(lines):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        marker = ""
        if parts[0].startswith("@"):
            marker, parts = parts[0], parts[1:]
        if len(parts) < 3:
            continue
        exact, by_pattern = _names_match(parts[0], lookup)
        if exact or by_pattern:
            found.append(_Entry(index, marker, parts[1], parts[2], by_pattern))
    return found


def _read_lines(path: Path) -> list[str]:
    """Lines split on newlines only, bytes preserved (surrogateescape) for the rewrite."""
    if not path.is_file():
        return []
    text = path.read_bytes().decode("utf-8", errors="surrogateescape")
    return [line[:-1] if line.endswith("\r") else line for line in text.split("\n") if line != ""]


def _change_for(target: _Target, timeout: int) -> list[HostKeyChange]:
    presented = _scan(target, timeout)
    if not presented:
        return []
    recorded: dict[str, set[str]] = {}
    for entry in _entries(_read_lines(target.known_hosts), target.lookup):
        if not entry.marker:
            recorded.setdefault(entry.key_type, set()).add(entry.blob)
    changes = []
    for key_type, blob in sorted(presented.items()):
        known = recorded.get(key_type)
        # A type ssh has never recorded is new, not changed: ssh would add it.
        if known and blob not in known:
            changes.append(
                HostKeyChange(
                    host=target.host,
                    lookup=target.lookup,
                    key_type=key_type,
                    recorded=tuple(sorted(fp for fp in map(_fingerprint_or_none, known) if fp)),
                    presented=key_fingerprint(blob),
                    known_hosts=str(target.known_hosts),
                )
            )
    return changes


def find_changed_host_keys(
    hosts: list[str],
    user: str | None = None,
    *,
    key: str | None = None,
    options: list[str] | None = None,
    timeout: int = 5,
    max_workers: int | None = None,
) -> list[HostKeyChange]:
    """Hosts whose presented host key differs from this machine's ``known_hosts``.

    Read-only: scans each host (no authentication) and compares, per key type,
    with what ssh would check. A host that does not answer, or that has no
    recorded key of a presented type, is not reported: there is nothing to
    replace. Advisory: a host that cannot be checked is skipped (ssh reports it
    on the next connection), so this never fails setup on its own.

    Raises:
        OpenSshUnavailable: ``ssh`` / ``ssh-keyscan`` are missing.
    """
    if not hosts:
        return []
    _require("ssh")
    _require("ssh-keyscan")

    def one(host):
        try:
            return _change_for(_resolve_target(host, user, key, options, timeout), timeout)
        except Exception:  # noqa: BLE001
            logger.debug("Host key check skipped for %s", host, exc_info=True)
            return []

    workers = max(1, min(max_workers or 16, len(hosts)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [change for changes in pool.map(one, hosts) for change in changes]


def replace_host_key(
    host: str,
    fingerprint: str,
    user: str | None = None,
    *,
    key: str | None = None,
    options: list[str] | None = None,
    timeout: int = 5,
    backup: bool = True,
) -> HostKeyReplacement:
    """Trust the key *host* presents with *fingerprint*, and nothing else.

    Re-scans first, and writes only if the host still presents a key with the
    approved *fingerprint* (``SHA256:...``, as :func:`find_changed_host_keys`
    reported it). Then, under the lock ``known_hosts_refresh.sh`` uses, the
    recorded keys of that one type under that one name are dropped (a line
    naming other hosts too keeps them) and the approved key is recorded, hashed
    when ``HashKnownHosts`` says so. The previous file is kept as ``.old``
    unless *backup* is False (a caller replacing several types of one host
    keeps the original by backing up only the first).

    Raises:
        SshAccessError: The host did not answer or no longer presents the
            approved key; the key is revoked (``@revoked``); the stale key comes
            from a wildcard entry shared with other hosts; or ``known_hosts``
            could not be read or written.
        OpenSshUnavailable: The OpenSSH tools are missing.
    """
    try:
        target = _resolve_target(host, user, key, options, timeout)
        presented = _scan(target, timeout)
    except (SshAccessError, OpenSshUnavailable):
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise SshAccessError("Could not scan %s's host key: %s" % (host, e)) from e
    if not presented:
        raise SshAccessError("%s did not answer a host key scan; nothing replaced" % host)
    approved = [(key_type, blob) for key_type, blob in presented.items() if _fingerprint_or_none(blob) == fingerprint]
    if not approved:
        raise SshAccessError(
            "%s no longer presents the approved key %s (it presents %s); nothing replaced"
            % (host, fingerprint, ", ".join(sorted(fp for fp in map(_fingerprint_or_none, presented.values()) if fp)))
        )
    key_type, blob = approved[0]

    # Write through a symlinked known_hosts rather than replacing the link.
    path = target.known_hosts.resolve() if target.known_hosts.is_symlink() else target.known_hosts
    try:
        from sparkrun.utils.fs import file_lock

        path.parent.mkdir(parents=True, exist_ok=True)
        lock_dir = Path.home() / ".ssh"
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        with file_lock(lock_dir / _LOCK_NAME):
            lines = _read_lines(path)
            entries = _entries(lines, target.lookup)
            if any(e.marker == "@revoked" and e.blob == blob for e in entries):
                raise SshAccessError("%s's key %s is marked @revoked in %s; nothing replaced" % (host, fingerprint, path))
            if any(e.by_pattern and not e.marker and e.key_type == key_type and e.blob != blob for e in entries):
                raise SshAccessError(
                    "%s's stale %s key comes from a wildcard entry in %s shared with other hosts; edit it by hand" % (host, key_type, path)
                )
            updated = _without_stale(lines, entries, target.lookup, key_type)
            name = _hashed_name(target.lookup) if target.hashed else target.lookup
            updated.append("%s %s %s" % (name, key_type, blob))
            if backup and path.is_file():
                shutil.copy2(path, str(path) + ".old")
            _write_lines(path, updated)
    except SshAccessError:
        raise
    except OSError as e:
        raise SshAccessError("Could not update %s: %s" % (path, e)) from e
    logger.info("Replaced the %s known_hosts entry for %s (%s)", key_type, target.lookup, fingerprint)
    return HostKeyReplacement(host, target.lookup, key_type, fingerprint, str(path))


def _without_stale(lines: list[str], entries: list[_Entry], lookup: str, key_type: str) -> list[str]:
    """*lines* minus the stale *key_type* keys recorded under exactly *lookup*.

    A line naming several hosts keeps the others (``spark1,10.0.0.1 ...`` loses
    only ``spark1``), so trust recorded for an alias is not dropped with it.
    """
    stale = {e.index for e in entries if not e.marker and not e.by_pattern and e.key_type == key_type}
    out = []
    for index, line in enumerate(lines):
        if index not in stale:
            out.append(line)
            continue
        field, rest = line.split(None, 1)
        if field.startswith("|"):
            continue
        others = [n for n in field.split(",") if n.lower() != lookup.lower()]
        if others:
            out.append("%s %s" % (",".join(others), rest))
    return out


def _write_lines(path: Path, lines: list[str]) -> None:
    """Replace *path* atomically with *lines*, LF-terminated, keeping its mode."""
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape", newline="\n") as stream:
            stream.write("".join(line + "\n" for line in lines))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
