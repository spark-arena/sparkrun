"""Console-free management of the proxy's systemd unit (``proxy systemd``).

The unit runs ``<app> proxy start --foreground`` with no other flags, so the
proxy it starts is whatever ``proxy.yaml`` says at that moment (port, host,
gateway, discovery scope) — the unit never needs rewriting when settings
change.  See :mod:`sparkrun.proxy._systemd` for naming, scopes and the sudoers
grant.

Privileged steps (installing or removing a *system* unit) try ``sudo -n``
first and raise :class:`SudoPasswordRequired` when a password is needed, so a
caller with a terminal can prompt and retry with ``sudo_password`` while a GUI
caller can collect it its own way.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.proxy._systemd import UnitSpec

    from ._ops import DiscoveryScope

logger = logging.getLogger(__name__)

#: How long ``install --now`` waits for the proxy port before reporting it as
#: not yet listening.  LiteLLM's first start resolves its environment through
#: uvx, which can take a while; the wait is informational, never a failure.
LISTEN_WAIT_S = 90.0


class ProxyServiceError(SparkrunError):
    """The proxy unit could not be installed, controlled, or removed."""


class SudoPasswordRequired(ProxyServiceError):
    """A privileged step needs a sudo password; retry with ``sudo_password``."""


@dataclass(frozen=True)
class ProxyServiceOptions:
    """Inputs for :func:`install_service`."""

    #: ``"system"`` (default; boot-started, sudo to install) or ``"user"``.
    scope: str = "system"
    #: Discovery cluster to save as ``proxy.cluster`` (explicit only).
    cluster: str | None = None
    #: Start (or restart) the unit now, replacing an ad-hoc proxy.
    now: bool = False
    dry_run: bool = False
    sudo_password: str | None = None


@dataclass(frozen=True)
class ProxyServiceInstallResult:
    unit: str
    scope: str
    unit_path: str
    unit_text: str
    sudoers_path: str | None = None
    sudoers_text: str | None = None
    #: True when an existing unit of ours was rewritten.
    updated: bool = False
    started: bool = False
    #: Whether the proxy port answered after ``--now`` (None: not checked).
    listening: bool | None = None
    stopped_adhoc_pid: int | None = None
    #: User scope: whether lingering is enabled (None: unknown / system scope).
    linger: bool | None = None
    persisted: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    dry_run: bool = False


@dataclass(frozen=True)
class ProxyServiceStatus:
    installed: bool
    unit: str | None = None
    scope: str | None = None
    unit_path: str | None = None
    enabled: str | None = None
    active: str | None = None
    #: System scope: whether the start/stop/restart grant works (None: unknown).
    grant_ok: bool | None = None
    linger: bool | None = None
    journal: str = ""
    discovery: "DiscoveryScope | None" = None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "discovery"}
        if self.discovery is not None:
            data["discovery"] = {"source": self.discovery.source, "cluster": self.discovery.cluster, "hosts": list(self.discovery.hosts)}
        return data


@dataclass(frozen=True)
class ProxyServiceUninstallResult:
    unit: str
    scope: str
    uninstalled: bool
    dry_run: bool = False
    notes: tuple[str, ...] = ()


# -- Lookup ----------------------------------------------------------------------


def find_service(scope: str | None = None) -> "UnitSpec | None":
    """The caller's installed proxy unit, or ``None``.

    With *scope* ``None`` both scopes are checked; having both is ambiguous
    and raises rather than guessing which one a command should act on.
    """
    from sparkrun.proxy import _systemd

    if _systemd.systemd_unavailable_reason() is not None:
        return None
    try:
        user = _systemd.current_user()
    except _systemd.SystemdError:
        return None
    scopes = [scope] if scope else [_systemd.SCOPE_SYSTEM, _systemd.SCOPE_USER]
    found = [spec for spec in (_systemd.find_installed(s, user) for s in scopes) if spec is not None]
    if len(found) > 1:
        raise ProxyServiceError(
            "Both a system and a user proxy unit are installed (%s). Uninstall one with "
            "'proxy systemd uninstall --system' or 'proxy systemd uninstall --user'." % ", ".join(str(s.path) for s in found)
        )
    return found[0] if found else None


def spec_from_record(record: Any) -> "UnitSpec | None":
    """The unit a running proxy's state names, when it is still ours."""
    from sparkrun.proxy import _systemd

    return _systemd.unit_for_record(record)


# -- Install -----------------------------------------------------------------------


def install_service(options: ProxyServiceOptions | None = None, *, sctx: "SparkrunContext | None" = None) -> ProxyServiceInstallResult:
    """Install (or update) the caller's proxy unit and enable it.

    Raises:
        SudoPasswordRequired: a system-scope install needs a password.
        ProxyServiceError: systemd is unavailable, a name is taken, inputs
            cannot be written into a unit, or a privileged step failed.
    """
    from sparkrun.proxy import _systemd

    options = options or ProxyServiceOptions()
    sctx = resolve_sctx(sctx)
    reason = _systemd.systemd_unavailable_reason()
    if reason:
        raise ProxyServiceError("Cannot install a proxy unit: %s" % reason)
    if options.scope not in (_systemd.SCOPE_SYSTEM, _systemd.SCOPE_USER):
        raise ProxyServiceError("Unknown scope %r." % options.scope)
    if options.scope == _systemd.SCOPE_SYSTEM and os.geteuid() == 0:
        raise ProxyServiceError("Run this as the user who owns the proxy, not as root; sudo is used only for the privileged steps.")

    warnings: list[str] = []
    try:
        user = _systemd.current_user()
        other_scope = _systemd.SCOPE_USER if options.scope == _systemd.SCOPE_SYSTEM else _systemd.SCOPE_SYSTEM
        other = _systemd.find_installed(other_scope, user)
        if other is not None:
            # Two units would both start a proxy on the same port at boot.
            raise _systemd.SystemdError(
                "A %s proxy unit is already installed (%s); uninstall it first with 'proxy systemd uninstall --%s'."
                % (other_scope, other.path, other_scope)
            )
        spec = _systemd.choose_install_target(options.scope, user)
        inputs = _unit_inputs(sctx, warnings)
        unit_text = _systemd.render_unit(spec, inputs)
        sudoers_text = _systemd.render_sudoers(spec, _systemd.systemctl_path()) if spec.scope == _systemd.SCOPE_SYSTEM else None
    except _systemd.SystemdError as exc:
        raise ProxyServiceError(str(exc)) from exc

    from ._ops import _load_cluster

    if options.cluster and _load_cluster(options.cluster, sctx) is None:
        raise ProxyServiceError("Unknown cluster %r." % options.cluster)
    if os.environ.get("SSH_AUTH_SOCK"):
        warnings.append(
            "The service has no ssh-agent. If discovery relies on an agent or a passphrase-protected key, "
            "set an unencrypted key with 'ssh.key' in config.yaml."
        )

    updated = spec.path.exists()
    common: dict[str, Any] = {
        "unit": spec.name,
        "scope": spec.scope,
        "unit_path": str(spec.path),
        "unit_text": unit_text,
        "sudoers_path": str(spec.sudoers_path) if spec.sudoers_path else None,
        "sudoers_text": sudoers_text,
        "updated": updated,
    }
    if options.dry_run:
        return ProxyServiceInstallResult(warnings=tuple(warnings), dry_run=True, **common)

    persisted: tuple[str, ...] = ()
    if options.cluster and options.cluster != sctx.proxy_config.cluster:
        sctx.proxy_config.set_proxy(cluster=options.cluster)
        sctx.proxy_config.save()
        persisted = ("cluster",)

    linger = None
    if spec.scope == _systemd.SCOPE_SYSTEM:
        _run_privileged(
            _systemd.render_system_install(spec, unit_text, sudoers_text or ""), options.sudo_password, "install the proxy unit"
        )
    else:
        linger = _install_user_unit(spec, unit_text, warnings)

    started = False
    listening = None
    stopped_pid = None
    if options.now:
        stopped_pid = _stop_adhoc_proxy(spec, sctx)
        active = _systemd.query(spec, "is-active") == "active"
        _control_or_raise(spec, "restart" if active else "start", options.sudo_password)
        started = True
        listening = _wait_listening(sctx)

    return ProxyServiceInstallResult(
        started=started,
        listening=listening,
        stopped_adhoc_pid=stopped_pid,
        linger=linger,
        persisted=persisted,
        warnings=tuple(warnings),
        **common,
    )


def _unit_inputs(sctx: "SparkrunContext", warnings: list[str]):
    """Resolve the absolute paths and identity written into the unit."""
    import grp
    import pwd

    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.core.config import get_config_root
    from sparkrun.proxy import _systemd

    from ._ops import resolve_gateway

    command = get_application_profile().command
    # Prefer the entry point actually running, so the unit runs the same
    # sparkrun the user invoked rather than whichever install is first on PATH.
    argv0 = Path(sys.argv[0])
    if argv0.name == command and argv0.is_file():
        sparkrun_path = str(argv0.absolute())
    else:
        sparkrun_path = shutil.which(command)
    if not sparkrun_path:
        raise _systemd.SystemdError("Could not find the %r executable on PATH; the unit needs its absolute path." % command)
    sparkrun_path = _systemd.validate_path(os.path.abspath(sparkrun_path), "%s path" % command)
    if "/.venv/" in sparkrun_path:
        warnings.append("%s resolves inside a virtualenv (%s); the unit breaks if that checkout moves." % (command, sparkrun_path))

    path_dirs = [os.path.dirname(sparkrun_path)]
    gateway = resolve_gateway(None, sctx=sctx)
    uvx = shutil.which("uvx")
    if uvx:
        path_dirs.append(os.path.dirname(os.path.abspath(uvx)))
    elif gateway == "litellm":
        raise _systemd.SystemdError("The litellm gateway runs through uvx, which was not found on PATH. Install uv first.")
    if gateway == "sparkroute":
        warnings.append(
            "With the sparkroute gateway, routes are not reconciled when the unit starts the gateway; "
            "they are applied at the first discovery sweep."
        )
    path_dirs += ["/usr/local/bin", "/usr/bin", "/bin"]
    path_env = ":".join(dict.fromkeys(_systemd.validate_path(d, "PATH entry") for d in path_dirs))

    entry = pwd.getpwuid(os.getuid())
    home = _systemd.validate_path(entry.pw_dir, "home directory")
    group = _systemd.validate_user(grp.getgrgid(entry.pw_gid).gr_name)

    config_path = None
    configured = getattr(sctx.config, "config_path", None)
    if configured and Path(configured).expanduser().resolve() != (get_config_root() / "config.yaml").resolve():
        config_path = _systemd.validate_path(str(Path(configured).expanduser().resolve()), "config path")
    return _systemd.UnitInputs(sparkrun_path=sparkrun_path, home=home, group=group, path_env=path_env, config_path=config_path)


def _install_user_unit(spec: "UnitSpec", unit_text: str, warnings: list[str]) -> bool | None:
    from sparkrun.proxy import _systemd

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spec.path.with_name("." + spec.path.name + ".tmp")
    tmp.write_text(unit_text)
    os.replace(tmp, spec.path)
    for args in (("daemon-reload",), ("enable", spec.name)):
        result = _systemd.user_command(spec, *args)
        if result.returncode != 0:
            raise ProxyServiceError("systemctl --user %s failed: %s" % (" ".join(args), (result.stderr or result.stdout).strip()))
    linger = _systemd.linger_enabled(spec.user)
    if linger is not True:
        linger = _systemd.enable_linger(spec.user)
        if not linger:
            warnings.append(
                "Could not enable lingering for %s, so the proxy starts at your next login rather than at boot. "
                "Fix with: sudo loginctl enable-linger %s" % (spec.user, spec.user)
            )
    return linger


def _run_privileged(script: str, password: str | None, what: str) -> None:
    from sparkrun.orchestration.sudo import is_sudo_auth_failure, run_sudo_script_on_host

    result = run_sudo_script_on_host("localhost", script, password, timeout=120)
    if result.success:
        return
    if password is None and is_sudo_auth_failure(result):
        raise SudoPasswordRequired("A sudo password is needed to %s." % what)
    detail = (result.stderr or result.stdout or "").strip()
    raise ProxyServiceError("Failed to %s: %s" % (what, detail or "exit code %d" % result.returncode))


def _control_or_raise(spec: "UnitSpec", action: str, password: str | None = None) -> None:
    """Run start/stop/restart; for system scope fall back to *password* if the grant fails."""
    from sparkrun.proxy import _systemd
    from sparkrun.utils.shell import quote

    result = _systemd.control(spec, action)
    if result.returncode == 0:
        return
    if spec.scope == _systemd.SCOPE_SYSTEM and password is not None:
        _run_privileged("systemctl %s %s" % (action, quote(spec.name)), password, "%s %s" % (action, spec.name))
        return
    detail = (result.stderr or result.stdout or "").strip()
    if spec.scope == _systemd.SCOPE_SYSTEM and "password" in detail.lower():
        raise SudoPasswordRequired("The sudoers grant for %s is missing. Run: sudo systemctl %s %s" % (spec.name, action, spec.name))
    raise ProxyServiceError("systemctl %s %s failed: %s" % (action, spec.name, detail or "exit code %d" % result.returncode))


def control_service(spec: "UnitSpec", action: str, *, sudo_password: str | None = None) -> None:
    """Start / stop / restart the caller's unit (public wrapper for ``api.proxy``)."""
    _control_or_raise(spec, action, sudo_password)


def _stop_adhoc_proxy(spec: "UnitSpec", sctx: "SparkrunContext") -> int | None:
    """Stop a proxy running outside the unit so the unit can take its port."""
    from ._ops import _running_engine, _stop_and_wait

    engine = _running_engine(sctx)
    if not engine.is_running():
        return None
    state = engine.get_state() or {}
    if spec_from_record(state.get("supervisor")) == spec:
        return None
    pid = engine.current_pid()
    if not _stop_and_wait(engine):
        raise ProxyServiceError("The running proxy (PID %s) did not stop; stop it with 'proxy stop' and retry." % pid)
    return pid


def _wait_listening(sctx: "SparkrunContext", timeout: float = LISTEN_WAIT_S) -> bool:
    cfg = sctx.proxy_config
    host = cfg.host if cfg.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, cfg.port), timeout=2):
                return True
        except OSError:
            time.sleep(1.0)
    return False


# -- Status / uninstall ------------------------------------------------------------------


def service_status(scope: str | None = None, *, journal_lines: int = 20, sctx: "SparkrunContext | None" = None) -> ProxyServiceStatus:
    """Report the caller's unit: state, grant, linger, recent journal, discovery scope."""
    from sparkrun.proxy import _systemd

    from ._ops import ProxyStartOptions, resolve_discovery_scope

    sctx = resolve_sctx(sctx)
    reason = _systemd.systemd_unavailable_reason()
    if reason:
        return ProxyServiceStatus(installed=False, unavailable_reason=reason)
    discovery = None
    try:
        discovery, _cluster, _warnings = resolve_discovery_scope(ProxyStartOptions(), sctx)
    except Exception:
        logger.debug("Could not resolve the discovery scope", exc_info=True)
    spec = find_service(scope)
    if spec is None:
        return ProxyServiceStatus(installed=False, discovery=discovery)
    grant_ok = None
    linger = None
    if spec.scope == _systemd.SCOPE_SYSTEM:
        grant_ok = _grant_works(spec)
    else:
        linger = _systemd.linger_enabled(spec.user)
    return ProxyServiceStatus(
        installed=True,
        unit=spec.name,
        scope=spec.scope,
        unit_path=str(spec.path),
        enabled=_systemd.query(spec, "is-enabled"),
        active=_systemd.query(spec, "is-active"),
        grant_ok=grant_ok,
        linger=linger,
        journal=_systemd.journal_tail(spec, journal_lines) if journal_lines > 0 else "",
        discovery=discovery,
    )


def _grant_works(spec: "UnitSpec") -> bool | None:
    """Ask sudo whether the exact start command is permitted, without running it.

    ``/etc/sudoers.d`` is root-only, so the file cannot simply be stat'ed.
    """
    import subprocess

    from sparkrun.proxy import _systemd

    argv = _systemd.control_argv(spec, "start")
    try:
        result = subprocess.run(["sudo", "-n", "-l", *argv[2:]], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.returncode == 0


def uninstall_service(
    scope: str | None = None, *, dry_run: bool = False, sudo_password: str | None = None, sctx: "SparkrunContext | None" = None
) -> ProxyServiceUninstallResult:
    """Stop, disable and delete the caller's unit (and its sudoers grant)."""
    from sparkrun.proxy import _systemd

    reason = _systemd.systemd_unavailable_reason()
    if reason:
        raise ProxyServiceError("Cannot manage proxy units: %s" % reason)
    spec = find_service(scope)
    if spec is None:
        raise ProxyServiceError("No proxy unit is installed for this user.")
    notes: list[str] = []
    if spec.scope == _systemd.SCOPE_USER:
        notes.append("Lingering was left as it is; other user services may rely on it.")
    if dry_run:
        return ProxyServiceUninstallResult(unit=spec.name, scope=spec.scope, uninstalled=False, dry_run=True, notes=tuple(notes))
    if spec.scope == _systemd.SCOPE_SYSTEM:
        _run_privileged(_systemd.render_system_uninstall(spec), sudo_password, "uninstall the proxy unit")
    else:
        _systemd.user_command(spec, "disable", "--now", spec.name)
        spec.path.unlink(missing_ok=True)
        _systemd.user_command(spec, "daemon-reload")
    return ProxyServiceUninstallResult(unit=spec.name, scope=spec.scope, uninstalled=True, notes=tuple(notes))


__all__ = [
    "ProxyServiceError",
    "ProxyServiceInstallResult",
    "ProxyServiceOptions",
    "ProxyServiceUninstallResult",
    "ProxyServiceStatus",
    "SudoPasswordRequired",
    "control_service",
    "find_service",
    "install_service",
    "uninstall_service",
    "service_status",
    "spec_from_record",
]
