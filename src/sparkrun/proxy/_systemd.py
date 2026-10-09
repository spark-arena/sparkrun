"""systemd unit management for the local inference proxy.

Console-free primitives behind ``sparkrun proxy systemd``: unit naming and
lookup, rendering (unit file, sudoers grant, privileged install/remove
scripts) and thin ``systemctl`` / ``loginctl`` wrappers.  Everything here runs
on the control machine; nothing is SSH'd.

Two scopes:

- **system** (default): ``/etc/systemd/system/<name>.service`` with
  ``User=<you>``.  Starts at boot regardless of logins.  Installing needs
  sudo once; a scoped sudoers grant then lets the owner start / stop / restart
  exactly that unit without a password, so ``proxy start`` / ``stop`` can route
  through it and its state stays truthful.
- **user**: ``~/.config/systemd/user/<name>.service``.  No sudo, but it is
  boot-started only while logind *lingering* is enabled for the user.

**Naming.**  A system unit is ``sparkrun-proxy.service`` unless that name
already belongs to another user, in which case it is
``sparkrun-proxy-<user>.service``.  Nothing records which one was chosen: the
lookup checks exactly those two candidates and takes the first whose ``User=``
is the caller and whose profile marker matches, so no record can drift from
the files.  A unit named for us but owned by someone else is never ours.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sparkrun.core.application_profile import child_environment, get_application_profile, resource_name
from sparkrun.proxy._supervisor import SUPERVISOR_ENV

SYSTEM_UNIT_DIR = Path("/etc/systemd/system")
SUDOERS_DIR = Path("/etc/sudoers.d")
MARKER_PREFIX = "# sparkrun.distribution="

SCOPE_SYSTEM = "system"
SCOPE_USER = "user"

#: Characters allowed in a path written into a unit file or sudoers rule.  A
#: unit file treats ``%`` as a specifier and whitespace as a separator, and a
#: sudoers rule is a privilege boundary, so anything else is refused rather
#: than escaped.
_SAFE_PATH = re.compile(r"/[A-Za-z0-9_./+@-]*")
#: The POSIX portable username set, without the trailing-``$`` machine-account
#: form ``validate_unix_username`` also accepts: the name lands in a unit name
#: and a sudoers file name, where ``$`` has no business.
_SAFE_USER = re.compile(r"[a-z_][a-z0-9_-]*")

_COMMAND_TIMEOUT_S = 60


class SystemdError(Exception):
    """A systemd operation could not be performed (message is user-facing)."""


@dataclass(frozen=True)
class UnitSpec:
    """One proxy unit: where it lives and who owns it."""

    scope: str
    name: str
    path: Path
    user: str

    @property
    def stem(self) -> str:
        return self.name[: -len(".service")]

    @property
    def sudoers_path(self) -> Path | None:
        """Grant file for a system unit (``None`` for user scope).

        Named after the unit stem, which never contains ``.`` — sudo silently
        skips ``/etc/sudoers.d`` entries whose name contains one.
        """
        return SUDOERS_DIR / self.stem if self.scope == SCOPE_SYSTEM else None

    @property
    def supervisor_value(self) -> str:
        """Value of :data:`SUPERVISOR_ENV` the unit sets for its process."""
        return "systemd:%s:%s" % (self.scope, self.name)


# -- Environment ---------------------------------------------------------------


def current_user() -> str:
    """The invoking user's login name, validated for unit/sudoers use."""
    import pwd

    name = pwd.getpwuid(os.getuid()).pw_name
    return validate_user(name)


def validate_user(name: str) -> str:
    if not _SAFE_USER.fullmatch(name or ""):
        raise SystemdError("Username %r cannot be used in a unit name or sudoers rule." % name)
    return name


def validate_path(value: str, what: str) -> str:
    """Refuse a path that cannot be written literally into a unit or sudoers file."""
    if not _SAFE_PATH.fullmatch(value or ""):
        raise SystemdError(
            "%s %r contains characters that cannot be written into a systemd unit (allowed: letters, digits, '/._+@-')." % (what, value)
        )
    return value


def systemd_unavailable_reason() -> str | None:
    """Why systemd units cannot be managed here, or ``None`` when they can."""
    import sys

    if not sys.platform.startswith("linux"):
        return "systemd units are only supported on Linux control machines."
    if not Path("/run/systemd/system").is_dir():
        return "this machine is not running systemd (no /run/systemd/system)."
    if shutil.which("systemctl") is None:
        return "systemctl was not found on PATH."
    return None


def systemctl_path() -> str:
    """Absolute ``systemctl`` path; sudoers rules must name commands absolutely."""
    found = shutil.which("systemctl")
    if not found:
        raise SystemdError("systemctl was not found on PATH.")
    return validate_path(os.path.realpath(found), "systemctl path")


def user_unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


# -- Naming and lookup ---------------------------------------------------------


def base_name() -> str:
    """Unit stem for this application profile (``sparkrun-proxy``)."""
    return resource_name("-proxy")


def candidates(scope: str, user: str) -> list[UnitSpec]:
    """The unit names that could belong to *user*, in preference order."""
    if scope == SCOPE_USER:
        name = base_name() + ".service"
        return [UnitSpec(SCOPE_USER, name, user_unit_dir() / name, user)]
    plain = base_name() + ".service"
    suffixed = "%s-%s.service" % (base_name(), user)
    return [
        UnitSpec(SCOPE_SYSTEM, plain, SYSTEM_UNIT_DIR / plain, user),
        UnitSpec(SCOPE_SYSTEM, suffixed, SYSTEM_UNIT_DIR / suffixed, user),
    ]


def read_owner(path: Path) -> tuple[str | None, str | None]:
    """``(profile id, User=)`` recorded in a unit file; ``(None, None)`` if unreadable."""
    try:
        text = path.read_text()
    except OSError:
        return None, None
    profile = None
    user = None
    for line in text.splitlines():
        if line.startswith(MARKER_PREFIX):
            profile = line[len(MARKER_PREFIX) :].strip()
        elif line.startswith("User="):
            user = line[len("User=") :].strip()
    return profile, user


def is_ours(spec: UnitSpec) -> bool:
    """True when *spec*'s file exists and belongs to this profile and user."""
    if not spec.path.exists():
        return False
    profile, user = read_owner(spec.path)
    if profile != get_application_profile().id:
        return False
    # A user unit has no User= line: living in our own config dir is ownership.
    return spec.scope == SCOPE_USER or user == spec.user


def find_installed(scope: str, user: str) -> UnitSpec | None:
    """The caller's installed unit in *scope*, or ``None``."""
    for spec in candidates(scope, user):
        if is_ours(spec):
            return spec
    return None


def choose_install_target(scope: str, user: str) -> UnitSpec:
    """Where an install for *user* goes: an existing unit of ours, else the first free name."""
    existing = find_installed(scope, user)
    if existing is not None:
        return existing
    for spec in candidates(scope, user):
        if not spec.path.exists():
            return spec
    raise SystemdError(
        "Every candidate unit name is taken by another user or application: %s" % ", ".join(str(c.path) for c in candidates(scope, user))
    )


# -- Rendering -----------------------------------------------------------------


@dataclass(frozen=True)
class UnitInputs:
    """Per-install values written into the unit."""

    sparkrun_path: str
    home: str
    group: str
    path_env: str
    config_path: str | None = None


def render_unit(spec: UnitSpec, inputs: UnitInputs) -> str:
    """Render a complete, self-contained unit for *spec*."""
    profile = get_application_profile()
    env = child_environment(config_path=inputs.config_path)
    env[SUPERVISOR_ENV] = spec.supervisor_value
    env["PATH"] = inputs.path_env
    lines = [
        MARKER_PREFIX + profile.id,
        "[Unit]",
        "Description=%s inference proxy (%s)" % (profile.command, spec.user),
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
    ]
    if spec.scope == SCOPE_SYSTEM:
        lines += ["User=%s" % spec.user, "Group=%s" % inputs.group, "Environment=HOME=%s" % inputs.home]
    lines += ["ExecStart=%s proxy start --foreground" % inputs.sparkrun_path]
    lines += ["Environment=%s=%s" % (key, value) for key, value in sorted(env.items())]
    lines += [
        "WorkingDirectory=%s" % inputs.home,
        "Restart=on-failure",
        "RestartSec=10",
        # SIGTERM to sparkrun only: it stops the discovery daemon before the
        # gateway, so the daemon cannot respawn what is being stopped.  The
        # rest of the cgroup is reaped once sparkrun exits.
        "KillMode=mixed",
        "TimeoutStopSec=60",
        # A gateway whose foreground path predates supervision returns 130 on
        # SIGTERM; that is a requested stop, not a failure.
        "SuccessExitStatus=130",
        "StandardOutput=journal",
        "StandardError=journal",
        "SyslogIdentifier=%s" % spec.stem,
        "",
        "[Install]",
        "WantedBy=%s" % ("multi-user.target" if spec.scope == SCOPE_SYSTEM else "default.target"),
        "",
    ]
    return "\n".join(lines)


def render_sudoers(spec: UnitSpec, systemctl: str) -> str:
    """Grant *spec*'s owner exactly start / stop / restart of that one unit."""
    commands = ", ".join("%s %s %s" % (systemctl, action, spec.name) for action in ("start", "stop", "restart"))
    return "%s%s\n# Installed by: %s proxy systemd install\n%s ALL=(root) NOPASSWD: %s\n" % (
        MARKER_PREFIX,
        get_application_profile().id,
        get_application_profile().command,
        spec.user,
        commands,
    )


def _owner_guard(path: Path, user: str | None) -> str:
    """Refuse to replace or delete *path* unless it carries our marker (and ``User=``)."""
    from sparkrun.utils.shell import quote

    user_check = ""
    if user is not None:
        user_check = (
            '\n    if ! grep -qx %s "$_artifact"; then echo "Refusing to touch $_artifact: owned by another user" >&2; exit 1; fi'
            % quote("User=" + user)
        )
    return (
        "_artifact=%s\n"
        'if [ -e "$_artifact" ]; then\n'
        '    if ! grep -qx %s "$_artifact"; then echo "Refusing to touch $_artifact: owned by another application" >&2; exit 1; fi%s\n'
        "fi"
    ) % (quote(str(path)), quote(MARKER_PREFIX + get_application_profile().id), user_check)


def render_system_install(spec: UnitSpec, unit_text: str, sudoers_text: str) -> str:
    """Privileged script: write unit + sudoers atomically, validate, enable."""
    from sparkrun.utils.shell import quote

    sudoers = spec.sudoers_path
    assert sudoers is not None
    return "\n".join(
        [
            "set -euo pipefail",
            _owner_guard(spec.path, spec.user),
            _owner_guard(sudoers, None),
            "_unit_tmp=$(mktemp %s)" % quote(str(spec.path) + ".XXXXXX"),
            # sudo ignores sudoers.d entries containing '.', so the temp file
            # cannot be read as a live rule before visudo has checked it.
            "_sudoers_tmp=$(mktemp %s)" % quote(str(sudoers) + ".XXXXXX"),
            'trap \'rm -f "$_unit_tmp" "$_sudoers_tmp"\' EXIT',
            "cat > \"$_unit_tmp\" << 'SPARKRUN_UNIT_EOF'\n%sSPARKRUN_UNIT_EOF" % unit_text,
            "cat > \"$_sudoers_tmp\" << 'SPARKRUN_SUDOERS_EOF'\n%sSPARKRUN_SUDOERS_EOF" % sudoers_text,
            'chmod 0440 "$_sudoers_tmp"',
            'if ! visudo -cf "$_sudoers_tmp" >/dev/null; then echo "sudoers validation failed" >&2; exit 1; fi',
            'chmod 0644 "$_unit_tmp"',
            'mv -f "$_unit_tmp" %s' % quote(str(spec.path)),
            'mv -f "$_sudoers_tmp" %s' % quote(str(sudoers)),
            "systemctl daemon-reload",
            "systemctl enable %s" % quote(spec.name),
            'echo "Installed and enabled %s"' % spec.name,
        ]
    )


def render_system_remove(spec: UnitSpec) -> str:
    """Privileged script: stop + disable, then delete unit and grant (owner-guarded)."""
    from sparkrun.utils.shell import quote

    sudoers = spec.sudoers_path
    assert sudoers is not None
    return "\n".join(
        [
            "set -euo pipefail",
            _owner_guard(spec.path, spec.user),
            _owner_guard(sudoers, None),
            "systemctl disable --now %s || true" % quote(spec.name),
            "rm -f %s %s" % (quote(str(spec.path)), quote(str(sudoers))),
            "systemctl daemon-reload",
            "systemctl reset-failed %s 2>/dev/null || true" % quote(spec.name),
            'echo "Removed %s"' % spec.name,
        ]
    )


# -- systemctl / loginctl --------------------------------------------------------


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=_COMMAND_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(argv, 1, "", str(exc))


def systemctl_argv(spec: UnitSpec, *args: str) -> list[str]:
    if spec.scope == SCOPE_USER:
        return ["systemctl", "--user", *args]
    return ["systemctl", *args]


def query(spec: UnitSpec, verb: str) -> str:
    """``is-enabled`` / ``is-active`` answer (no privilege needed)."""
    result = _run(systemctl_argv(spec, verb, spec.name))
    return (result.stdout or "").strip() or "unknown"


def control_argv(spec: UnitSpec, action: str) -> list[str]:
    """argv for start / stop / restart, matching the sudoers grant exactly."""
    if action not in ("start", "stop", "restart"):
        raise ValueError("unsupported action %r" % action)
    if spec.scope == SCOPE_USER:
        return ["systemctl", "--user", action, spec.name]
    return ["sudo", "-n", systemctl_path(), action, spec.name]


def control(spec: UnitSpec, action: str) -> subprocess.CompletedProcess:
    """Start / stop / restart *spec* without prompting (sudoers grant for system scope)."""
    return _run(control_argv(spec, action))


def user_command(spec: UnitSpec, *args: str) -> subprocess.CompletedProcess:
    """Unprivileged ``systemctl --user`` (install/remove of a user unit)."""
    return _run(systemctl_argv(spec, *args))


def journal_tail(spec: UnitSpec, lines: int = 20) -> str:
    argv = ["journalctl", "--no-pager", "-n", str(lines)]
    argv += ["--user-unit", spec.name] if spec.scope == SCOPE_USER else ["-u", spec.name]
    result = _run(argv)
    return (result.stdout or "").rstrip()


def linger_enabled(user: str) -> bool | None:
    """Whether logind keeps *user*'s manager running without a session (None: unknown)."""
    marker = Path("/var/lib/systemd/linger") / user
    if marker.parent.is_dir():
        return marker.exists()
    result = _run(["loginctl", "show-user", user, "--property=Linger", "--value"])
    value = (result.stdout or "").strip()
    return {"yes": True, "no": False}.get(value)


def enable_linger(user: str) -> bool:
    """Enable lingering, unprivileged first (polkit allows it for oneself), then ``sudo -n``."""
    for argv in (["loginctl", "enable-linger", user], ["sudo", "-n", "loginctl", "enable-linger", user]):
        if _run(argv).returncode == 0:
            return True
    return False


__all__ = [
    "SCOPE_SYSTEM",
    "SCOPE_USER",
    "SystemdError",
    "UnitInputs",
    "UnitSpec",
    "candidates",
    "choose_install_target",
    "control",
    "control_argv",
    "current_user",
    "enable_linger",
    "find_installed",
    "is_ours",
    "journal_tail",
    "linger_enabled",
    "query",
    "render_sudoers",
    "render_system_install",
    "render_system_remove",
    "render_unit",
    "systemctl_path",
    "systemd_unavailable_reason",
    "user_command",
    "user_unit_dir",
    "validate_path",
]
