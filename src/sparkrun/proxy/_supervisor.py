"""Process and state-file machinery shared by every gateway implementation.

A *gateway* is a long-lived local process sparkrun spawns, records, and later
stops.  That part is identical whether the process is ``uvx litellm`` or some
other server; only the argv and the config format differ.  This module holds
the common half so two implementations cannot drift.

The state file matters most.  ``<cache>/proxy/state.yaml`` is written by
whichever gateway started, and read back by management paths that must act on
*what is running* rather than on what is currently configured — including
:func:`sparkrun.api.proxy._ops._running_engine`, which reads the file in order
to decide which engine to construct.  Two implementations writing that format
independently would eventually disagree about it, and the symptom would be a
proxy nobody can stop.

:class:`GatewayState` is the read-only half, split out so a caller that only
needs "what is running" — the auto-discover daemon, a status probe — can ask
without constructing a nameless supervisor whose :attr:`gateway_name` is blank.

The file carries the master key, so it is written 0600 inside a 0700 directory,
mirroring ``arena.auth.save_refresh_token``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from sparkrun.utils.fs import open_private_write

logger = logging.getLogger(__name__)

#: How long a restart waits for a signalled gateway to actually exit.
RESTART_EXIT_TIMEOUT = 15.0

#: Grace period after spawn before deciding the process survived startup.
RESTART_STARTUP_GRACE = 2.0


class GatewayOperationError(RuntimeError):
    """A running gateway could not adopt a requested model/config change.

    The base class every gateway raises for a *management* failure, so
    :func:`sparkrun.api.proxy.sync` can translate one exception type instead of
    catching bare ``RuntimeError`` — which would report an unrelated engine bug
    as a routine "the proxy could not be updated".

    ``sparkrun.proxy.engine.ProxyRestartError`` is the LiteLLM member.
    """


def _restrict_file_permissions(path: Path) -> None:
    """Best-effort chmod 0600 (no-op where the platform lacks POSIX modes)."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.debug("Could not restrict permissions on %s", path, exc_info=True)


def _restrict_dir_permissions(path: Path) -> None:
    """Best-effort chmod 0700."""
    try:
        os.chmod(path, 0o700)
    except OSError:
        logger.debug("Could not restrict permissions on %s", path, exc_info=True)


def _is_zombie(pid: int) -> bool:
    """True when *pid* has exited but has not been reaped yet (Linux).

    A zombie still answers ``os.kill(pid, 0)``, so without this check a
    dead-but-unreaped gateway reads as "still running" forever.
    """
    try:
        with open("/proc/%d/stat" % pid) as f:
            stat = f.read()
    except OSError:
        return False
    # The comm field is parenthesised and may contain spaces; state is the
    # first field after the closing paren.
    close = stat.rfind(")")
    if close == -1:
        return False
    fields = stat[close + 1 :].split()
    return bool(fields) and fields[0] == "Z"


def _wait_for_exit(pid: int, timeout: float) -> bool:
    """Poll until *pid* is gone. Returns True if it exited within *timeout*.

    Handles the case where the gateway is a child of this process — reaping
    it so it does not linger as a zombie that ``os.kill(pid, 0)`` reports
    as alive.  A gateway spawned by an earlier CLI invocation is not our
    child, and ``waitpid`` simply reports that.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return True
        except ChildProcessError:
            pass  # not our child — the normal daemonized case
        except OSError:
            pass

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Alive but not ours to signal — treat as still running.
            pass

        if _is_zombie(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


class GatewayState:
    """Read-only view of the shared gateway state file.

    Answers "what is running" without naming an implementation.  Split from
    :class:`GatewaySupervisor` so the auto-discover daemon and other probes do
    not have to instantiate a supervisor with an empty :attr:`gateway_name`
    just to read a PID.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        if state_dir is None:
            from sparkrun.core.config import DEFAULT_CACHE_DIR

            state_dir = DEFAULT_CACHE_DIR / "proxy"
        self.state_dir = state_dir
        self.state_file = state_dir / "state.yaml"

    def get_state(self) -> dict[str, Any] | None:
        """Read full gateway state. Returns None if not found or unreadable."""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file) as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def _read_pid(self) -> int | None:
        """Read the gateway PID from the state file."""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file) as f:
                state = yaml.safe_load(f)
            return int(state["pid"]) if state and "pid" in state else None
        except Exception:
            return None

    def _read_autodiscover_pid(self) -> int | None:
        """Read the auto-discover PID from the state file."""
        state = self.get_state()
        if state and "autodiscover_pid" in state:
            return int(state["autodiscover_pid"])
        return None

    def current_pid(self) -> int | None:
        """PID recorded in the state file, or None."""
        return self._read_pid()

    def recorded_gateway(self) -> str | None:
        """Name of the gateway implementation that wrote this state, if any."""
        state = self.get_state() or {}
        name = state.get("gateway")
        return str(name) if name else None


class GatewaySupervisor(GatewayState):
    """Spawn / record / stop one local gateway process.

    Subclasses supply :attr:`gateway_name` (recorded in the state file so
    management paths bind to the running implementation) and :attr:`log_name`,
    and are responsible for building the argv and environment they pass to
    :meth:`_launch_background`.
    """

    #: Selector this implementation answers to (``proxy.gateway`` in proxy.yaml).
    gateway_name = ""

    #: Filename for the process's captured stdout/stderr inside ``state_dir``.
    log_name = "gateway.log"

    #: Whether sparkrun's auto-discover daemon may run alongside this gateway.
    #: False for gateways that own desired state themselves — two components
    #: with independent opinions about what should be running will fight.
    supports_autodiscover = True

    #: Bind settings every gateway carries; subclass ``__init__`` sets them.
    host = ""
    port = 0
    host_configured = False

    #: Whether this gateway is driven by ``proxy.yaml`` as a whole and wants
    #: the config object (plus session context) at construction.  Declared as
    #: a capability rather than branched on by name, so the API layer stays
    #: free of per-gateway special cases (``_engine_class`` remains the one
    #: place a name resolves to an implementation).
    #:
    #: Load-bearing on the *management* paths, which resolve their engine from
    #: the state file: without the config a reconcile computes an **empty**
    #: desired state, so ``proxy alias add`` would silently delete every
    #: deployment it was not explicitly told about.
    wants_proxy_config = False

    #: Why the last :meth:`list_models_via_api` could not answer, or "".
    #:
    #: Empty means the query succeeded, *including* a legitimately empty model
    #: list — collapsing the two would report an authenticated management
    #: failure as "no models registered".  Non-secret; it reaches the CLI.
    model_query_error = ""

    def __init__(self, state_dir: Path | None = None) -> None:
        super().__init__(state_dir)
        self._autodiscover_config_path = self.state_dir / "autodiscover.yaml"
        # Handle for a process this instance spawned, so a later restart can
        # reap it.  None when the running gateway belongs to another process
        # (the usual case for the CLI talking to a daemonized gateway).
        self._proc: subprocess.Popen | None = None

    # -- Configuration ------------------------------------------------------

    def prepare_config(self, endpoints: list, aliases: dict[str, str], *, write: bool = True) -> tuple[Path | None, set[str], set[str]]:
        """Generate — and unless *write* is False, persist — this gateway's config.

        Config generation belongs to the implementation because what a
        gateway's config *is* differs: a rendering of the endpoints sparkrun
        discovered, or a list of desired bindings the gateway resolves itself.
        Computing it in the API layer for one gateway and adapting for the rest
        is exactly the branch this seam exists to avoid.

        Args:
            endpoints: Healthy discovered endpoints.  Meaningful only for
                gateways sparkrun points at what it found; a binding-driven
                gateway ignores them.
            aliases: ``proxy.yaml`` alias -> model-group mapping.
            write: When False, compute everything but touch no files.  A dry
                run still reports which aliases *would* apply, and answering
                that from the same code that renders the real config is what
                keeps the preview honest.

        Returns:
            ``(config_path, aliases_applied, aliases_pending)``; the path is
            ``None`` when *write* is False.
        """
        raise NotImplementedError

    @property
    def data_plane_authenticated(self) -> bool:
        """Whether reaching the inference port requires a credential.

        Drives the wording of :meth:`_warn_insecure_bind`, and defaults to
        **False** because that is the safe assumption: a gateway that does
        authenticate says so, rather than every gateway being trusted to have
        opted out of the warning by accident.
        """
        return False

    def _warn_insecure_bind(self) -> None:
        """Warn loudly when binding every interface without an explicit choice.

        Legacy compatibility: with no bind host configured, the proxy keeps the
        historical ``0.0.0.0`` default so existing deployments do not break.
        That exposes every served model to the whole network, so it is said out
        loud on each start — an exposure nobody chose is exactly the one that
        needs announcing.  An explicit bind host silences it, including an
        explicit ``0.0.0.0``.
        """
        if self.host_configured:
            return
        if self.host not in ("0.0.0.0", "::"):
            return

        if self.data_plane_authenticated:
            logger.warning(
                "\n"
                "============================================================\n"
                "  sparkrun proxy: binding %s:%d (ALL network interfaces)\n"
                "  Authentication IS enabled (master key set), but every\n"
                "  discovered inference backend is reachable network-wide.\n"
                "  To restrict exposure, bind to localhost instead:\n"
                "      sparkrun proxy start --host 127.0.0.1\n"
                "  (the chosen bind host is persisted to proxy.yaml)\n"
                "============================================================",
                self.host,
                self.port,
            )
        else:
            logger.warning(
                "\n"
                "============================================================\n"
                "  SECURITY WARNING: sparkrun proxy is binding %s:%d\n"
                "  (ALL network interfaces) with NO authentication.\n"
                "  Every discovered inference backend is exposed UNAUTHENTICATED\n"
                "  to the entire network.\n"
                "\n"
                "  To secure the proxy and silence this warning:\n"
                "    * bind to localhost:   sparkrun proxy start --host 127.0.0.1\n"
                "    * require a token:     sparkrun proxy start --master-key <secret>\n"
                "  (both settings are persisted to proxy.yaml for future runs)\n"
                "============================================================",
                self.host,
                self.port,
            )

    # -- Model management ---------------------------------------------------
    #
    # The surface behind ``sparkrun proxy alias`` / ``sync`` / ``models``.
    # These live on the base class rather than only on the engine that happens
    # to implement them because ``api.proxy`` resolves an engine from the
    # *state file* — so calling a LiteLLM-only method against some other
    # running gateway would be an ``AttributeError`` at the point of use, which
    # is both a worse error and one that surfaces far from its cause.  A
    # gateway that genuinely cannot do one of these says so here, naming itself.

    def sync_models(self, endpoints: list, aliases: dict[str, str] | None = None) -> tuple[int, int]:
        """Make the gateway serve exactly *endpoints* (plus *aliases*).

        Returns:
            ``(added, removed)`` model entries relative to the previous config.
        """
        raise NotImplementedError("gateway %r cannot synchronize its model list" % self.gateway_name)

    def sync_aliases(self, aliases: dict[str, str]) -> tuple[int, int]:
        """Apply *aliases* to the gateway, leaving its model set alone.

        Returns:
            ``(added, removed)`` alias entries.
        """
        raise NotImplementedError("gateway %r cannot synchronize aliases" % self.gateway_name)

    def list_models_via_api(self) -> list[dict[str, Any]]:
        """Return the models the running gateway reports."""
        raise NotImplementedError("gateway %r cannot report its served models" % self.gateway_name)

    def register_loaded_model(
        self,
        recipe: str,
        overrides: dict[str, Any] | None = None,
        cluster: str | None = None,
    ) -> tuple[int, int] | None:
        """Register one successfully-loaded recipe with the gateway.

        ``None`` means the gateway is discovery-driven and the caller should
        perform its ordinary endpoint sync.  A catalog-driven gateway overrides
        this to persist the route needed to adopt the warm endpoint now and
        activate the same workload again after it goes cold.
        """
        return None

    def unregister_loaded_model(self, recipe: str) -> tuple[int, int] | None:
        """Remove a recipe loaded through the proxy command.

        ``None`` has the same discovery-driven meaning as
        :meth:`register_loaded_model`.
        """
        return None

    # -- Process lifecycle --------------------------------------------------

    @property
    def log_path(self) -> Path:
        return self.state_dir / self.log_name

    def _launch_background(self, cmd: list[str], env: dict[str, str]) -> int | None:
        """Spawn the gateway detached and confirm it survived startup.

        Returns:
            The new PID, or None if the process exited during the grace period.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Redirect output to log file so startup errors are visible
        log_path = self.log_path
        log_file = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        except OSError:
            log_file.close()
            logger.error("Failed to spawn gateway process", exc_info=True)
            return None

        # Wait briefly and verify the process survived startup
        time.sleep(RESTART_STARTUP_GRACE)
        poll = proc.poll()
        if poll is not None:
            log_file.close()
            # Process already exited — show error
            try:
                tail = log_path.read_text()[-2000:]
            except OSError:
                tail = ""
            logger.error(
                "Gateway exited immediately (code %d). Log tail:\n%s",
                poll,
                tail,
            )
            return None

        self._proc = proc
        return proc.pid

    def _await_exit(self, pid: int, timeout: float) -> bool:
        """Wait for gateway *pid* to exit, reaping it when it is our child.

        A process we spawned stays a zombie until reaped, and
        ``os.kill(pid, 0)`` *succeeds* on a zombie — so PID polling alone
        would burn the whole timeout and then escalate to SIGKILL against a
        process that already exited.  The auto-discover daemon spawns every
        replacement gateway itself, so from its second restart onward this is
        the normal path, not an edge case.
        """
        proc = self._proc
        if proc is not None and proc.pid == pid:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            self._proc = None
            return True
        return _wait_for_exit(pid, timeout)

    # -- Auto-discovery sidecar --------------------------------------------

    def start_autodiscover(
        self,
        proxy_pid: int,
        interval: int = 30,
        removal_grace_sweeps: int = 2,
        host_list: list[str] | None = None,
        ssh_kwargs: dict | None = None,
        cache_dir: str | None = None,
    ) -> int | None:
        """Spawn the gateway-neutral endpoint-discovery sidecar.

        The child resolves the gateway recorded in :attr:`state_file` on each
        reconciliation through :func:`sparkrun.api.proxy.sync`; it does not
        instantiate a particular gateway implementation itself.  That is also
        why the config it is handed no longer carries the master key: the
        credential belongs to whichever engine the state file names.

        Returns:
            PID of the auto-discover process, or ``None`` on failure.
        """
        cfg: dict[str, Any] = {
            "proxy_pid": proxy_pid,
            "gateway": self.gateway_name,
            "state_dir": str(self.state_dir),
            "interval": interval,
            "removal_grace_sweeps": removal_grace_sweeps,
        }
        if host_list:
            cfg["host_list"] = host_list
        if ssh_kwargs is not None:
            cfg["ssh_kwargs"] = ssh_kwargs
        if cache_dir:
            cfg["cache_dir"] = cache_dir

        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)
        # The config names hosts and SSH parameters, so create it owner-only
        # (no default-umask window) and refuse to follow a symlink at the target.
        fd = open_private_write(self._autodiscover_config_path)
        with os.fdopen(fd, "w") as stream:
            yaml.safe_dump(cfg, stream, default_flow_style=False)
        _restrict_file_permissions(self._autodiscover_config_path)

        log_path = self.state_dir / "autodiscover.log"
        log_file = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "sparkrun.proxy.autodiscover", str(self._autodiscover_config_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            logger.warning("Failed to start auto-discover process", exc_info=True)
            return None
        finally:
            log_file.close()

        logger.info(
            "Auto-discover started (PID %d), gateway=%s, interval=%ds, log=%s",
            proc.pid,
            self.gateway_name,
            interval,
            log_path,
        )
        return proc.pid

    def stop_autodiscover(self) -> None:
        """Stop the background auto-discovery process if running."""
        autodiscover_pid = self._read_autodiscover_pid()
        if autodiscover_pid is None:
            return
        try:
            os.kill(autodiscover_pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to auto-discover PID %d", autodiscover_pid)
        except ProcessLookupError:
            logger.debug("Auto-discover PID %d already gone", autodiscover_pid)
        except PermissionError:
            logger.warning("Permission denied stopping auto-discover PID %d", autodiscover_pid)
        self._autodiscover_config_path.unlink(missing_ok=True)

    def _before_stop(self) -> None:
        """Hook run before the gateway is signalled.

        Stops the shared discovery sidecar (it monitors the gateway PID anyway,
        but an explicit stop is cleaner).  A gateway with its own out-of-band
        state extends this rather than overriding :meth:`stop`.
        """
        self.stop_autodiscover()

    def stop(self, dry_run: bool = False) -> bool:
        """Stop the running gateway (SIGTERM via PID).

        Returns:
            True if a process was stopped.
        """
        pid = self._read_pid()
        if pid is None:
            logger.info("No proxy PID found in state file")
            return False

        if dry_run:
            logger.info("[dry-run] Would send SIGTERM to PID %d", pid)
            return True

        self._before_stop()

        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to proxy PID %d", pid)
            # Reap it *only* if we spawned it, so it does not linger as a
            # zombie (which still answers os.kill(pid, 0) and reads as
            # running).  Never block on a gateway owned by another process —
            # `sparkrun proxy stop` must stay instant.
            if self._proc is not None and self._proc.pid == pid:
                self._await_exit(pid, RESTART_EXIT_TIMEOUT)
            self._clear_state()
            return True
        except ProcessLookupError:
            logger.info("Proxy PID %d not running (stale state)", pid)
            self._clear_state()
            return False
        except PermissionError:
            logger.error("Permission denied sending signal to PID %d", pid)
            return False

    def is_running(self) -> bool:
        """Check if the gateway process is alive."""
        pid = self._read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    # -- State file ---------------------------------------------------------

    def _state_payload(self) -> dict[str, Any]:
        """Implementation-specific state fields (host, port, master key, …)."""
        return {}

    def _save_state(self, pid: int, autodiscover_pid: int | None = None) -> None:
        """Save gateway state to disk.

        The state file can contain the master key, so it is written with
        owner-only (0o600) permissions and its parent dir is restricted to
        0o700 — mirroring ``arena.auth.save_refresh_token``.
        """
        import datetime

        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)
        state: dict[str, Any] = {"pid": pid}
        state.update(self._state_payload())
        # Which gateway implementation owns this process.  Management paths
        # (stop / status / sync) read it back so they act on what is *running*
        # rather than on what is currently configured.
        state["gateway"] = self.gateway_name
        state["started_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if autodiscover_pid is not None:
            state["autodiscover_pid"] = autodiscover_pid
        with open(self.state_file, "w") as f:
            yaml.safe_dump(state, f, default_flow_style=False)
        _restrict_file_permissions(self.state_file)

    def update_autodiscover_pid(self, autodiscover_pid: int) -> None:
        """Record the auto-discover PID in state (call after start)."""
        pid = self._read_pid()
        if pid is not None:
            self._save_state(pid, autodiscover_pid=autodiscover_pid)

    def _clear_state(self) -> None:
        """Remove state file."""
        self.state_file.unlink(missing_ok=True)


__all__ = [
    "RESTART_EXIT_TIMEOUT",
    "RESTART_STARTUP_GRACE",
    "GatewayOperationError",
    "GatewayState",
    "GatewaySupervisor",
    "_is_zombie",
    "_restrict_dir_permissions",
    "_restrict_file_permissions",
    "_wait_for_exit",
]
