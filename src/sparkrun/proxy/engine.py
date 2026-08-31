"""LiteLLM proxy engine — config generation and subprocess lifecycle.

Launches ``uvx litellm`` as a subprocess and manages its lifecycle.

The generated config file is the single source of truth for the proxy's
model list: LiteLLM's runtime mutation endpoints need a DB-backed model
store (PostgreSQL + a generated prisma client), so applying a change means
rewriting the config and restarting.  The management API is still used
read-only, to report what the proxy is currently serving.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from sparkrun.proxy import (
    DEFAULT_MASTER_KEY,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
)
from sparkrun.proxy.discovery import DiscoveredEndpoint
from sparkrun.proxy.gateway import require_gateway_enabled
from sparkrun.utils.fs import open_private_write

logger = logging.getLogger(__name__)

# How long a restart waits for the old proxy to exit before escalating to
# SIGKILL, and how long the replacement gets to survive startup.
RESTART_EXIT_TIMEOUT = 15.0
RESTART_STARTUP_GRACE = 2.0


class ProxyRestartError(RuntimeError):
    """Raised when a config change could not be applied to the running proxy.

    The config file on disk has already been updated when this is raised —
    the *running* process is what failed to pick it up.
    """


def _restrict_file_permissions(path: Path) -> None:
    """Best-effort chmod a secrets-bearing file to owner-only (0o600)."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.debug("Could not restrict permissions on %s", path, exc_info=True)


def _restrict_dir_permissions(path: Path) -> None:
    """Best-effort chmod a secrets-bearing dir to owner-only (0o700)."""
    try:
        os.chmod(path, 0o700)
    except OSError:
        logger.debug("Could not restrict permissions on %s", path, exc_info=True)


def build_litellm_config(
    endpoints: list[DiscoveredEndpoint],
    master_key: str | None = DEFAULT_MASTER_KEY,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate a litellm proxy config dict from discovered endpoints.

    When *master_key* is set, LiteLLM performs stateless bearer-token
    authentication against ``general_settings.master_key`` — no backing
    database is required.  Features that need a DB (virtual keys,
    budgets, request logging) are intentionally out of scope.

    This config file is the **only** way sparkrun mutates the proxy's model
    list.  LiteLLM's runtime management endpoints (``/model/new``,
    ``/model/delete``) all require a DB-backed model store, which in turn
    requires PostgreSQL plus a generated prisma client — so they answer
    ``500 No DB Connected`` here and cannot be used.  See
    :meth:`ProxyEngine.apply_desired_state`.

    Args:
        endpoints: Discovered inference endpoints.
        master_key: Bearer token for stateless auth.  When ``None``,
            no authentication is configured.
        aliases: Optional ``alias name -> target model`` mapping.  Each
            alias is emitted as an extra ``model_list`` entry pointing at
            every backend serving the target, so clients can address the
            alias exactly like a real model.  An alias whose target has no
            healthy backend is skipped (it reappears when the target
            returns).

    Returns:
        Dict suitable for writing as litellm YAML config.
    """
    model_list: list[dict[str, Any]] = []
    seen: set[str] = set()

    for ep in endpoints:
        if not ep.healthy:
            continue

        # Use actual served models from /v1/models if available
        model_names = ep.actual_models if ep.actual_models else [ep.model]

        for model_name in model_names:
            # Deduplicate: same model name on same host:port
            dedup_key = "%s@%s:%d" % (model_name, ep.host, ep.port)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            entry: dict[str, Any] = {
                "model_name": model_name,
                "litellm_params": {
                    "model": "openai/%s" % model_name,
                    "api_base": "http://%s:%d/v1" % (ep.host, ep.port),
                    "api_key": ep.api_key or "not-needed",
                },
            }
            # Advertise the model's true context window so LiteLLM exposes it to
            # clients via /v1/models (and /model/info). Without this the gateway
            # has no model-level context length and clients can't see the real
            # window (only a per-key cap the gateway may enforce). ``model_info``
            # is the supported LiteLLM field; ``max_input_tokens`` maps to the
            # server-reported ``max_model_len``.
            if ep.max_model_len:
                entry["model_info"] = {"max_input_tokens": ep.max_model_len}
            model_list.append(entry)

    # Alias entries are appended after the real models so the lookup below
    # only ever sees genuine backends (an alias of an alias is not a thing).
    if aliases:
        backends_by_name: dict[str, list[dict[str, Any]]] = {}
        for entry in model_list:
            backends_by_name.setdefault(entry["model_name"], []).append(entry["litellm_params"])

        for alias_name, target_model in sorted(aliases.items()):
            targets = backends_by_name.get(target_model)
            if not targets:
                logger.debug(
                    "Alias %r skipped: target %r has no healthy backend",
                    alias_name,
                    target_model,
                )
                continue
            for params in targets:
                model_list.append(
                    {
                        "model_name": alias_name,
                        "litellm_params": {
                            "model": "openai/%s" % target_model,
                            "api_base": params["api_base"],
                            "api_key": params.get("api_key") or "not-needed",
                        },
                    }
                )

    general_settings: dict[str, Any] = {}
    if master_key:
        general_settings["master_key"] = master_key

    config: dict[str, Any] = {
        "model_list": model_list,
        "litellm_settings": {
            "drop_params": True,
        },
    }

    if general_settings:
        config["general_settings"] = general_settings

    return config


def _model_keys(config: dict[str, Any]) -> set[tuple[str, str, int | None]]:
    """Identity of a config's model list as ``{(model_name, api_base, window)}``.

    This is the comparison that decides whether a restart is warranted, so
    it deliberately ignores ordering and any field a restart would not
    change (``api_key`` is carried over from the same discovery source).

    The advertised context window **is** such a field: LiteLLM reads
    ``model_info`` at startup, so a backend that changed its
    ``max_model_len`` (or one whose window we simply never captured before)
    only reaches clients once the config is rewritten and the proxy
    replaced.  Leaving it out made the whole ``model_info`` emission inert
    for any already-running proxy — the model set was unchanged, so
    ``apply_desired_state`` returned early and never wrote the window.  The
    cost is that a *changed* window reads as one added plus one removed
    entry rather than a modification, which is the same shape a re-homed
    backend already produces.
    """
    keys: set[tuple[str, str, int | None]] = set()
    for entry in config.get("model_list") or []:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params") or {}
        window = (entry.get("model_info") or {}).get("max_input_tokens")
        keys.add(
            (
                str(entry.get("model_name", "")),
                str(params.get("api_base", "")),
                window if isinstance(window, int) else None,
            )
        )
    return keys


def _is_zombie(pid: int) -> bool:
    """True when *pid* has exited but has not been reaped yet (Linux).

    A zombie still answers ``os.kill(pid, 0)``, so without this check a
    dead-but-unreaped proxy reads as "still running" forever.
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

    Handles the case where the proxy is a child of this process — reaping
    it so it does not linger as a zombie that ``os.kill(pid, 0)`` reports
    as alive.  A proxy spawned by an earlier CLI invocation is not our
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


def _parse_api_base(api_base: str) -> tuple[str, int] | None:
    """Split ``http://host:port/v1`` back into ``(host, port)``."""
    try:
        parsed = urllib.parse.urlparse(api_base)
        if parsed.hostname and parsed.port:
            return parsed.hostname, int(parsed.port)
    except (ValueError, TypeError):
        pass
    return None


def write_config(config_dict: dict[str, Any], config_path: Path | None = None) -> Path:
    """Write litellm config to disk.

    Args:
        config_dict: Config dict from ``build_litellm_config()``.
        config_path: Output path (default: ``~/.cache/sparkrun/proxy/litellm_config.yaml``).

    Returns:
        Path to the written config file.
    """
    if config_path is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR

        config_path = DEFAULT_CACHE_DIR / "proxy" / "litellm_config.yaml"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    _restrict_dir_permissions(config_path.parent)
    # The litellm config carries general_settings.master_key plus every upstream
    # endpoint api_key, so it must be owner-only.  Create it 0600 and refuse to
    # follow a symlink at the target path (mirrors orchestration.job_metadata);
    # _restrict_file_permissions afterwards repairs perms on a pre-existing file
    # written by an older version under the default umask.
    fd = open_private_write(config_path)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
    _restrict_file_permissions(config_path)

    logger.debug("Wrote litellm config to %s", config_path)
    return config_path


class ProxyEngine:
    """Manages the litellm proxy subprocess and its management API.

    This is the LiteLLM *gateway* implementation.  :attr:`gateway_name` and
    :attr:`required_feature_flag` are the seam a second gateway plugs into
    (see :mod:`sparkrun.proxy.gateway`); ``start()`` is the single point that
    enforces the flag.
    """

    #: Selector this implementation answers to (``proxy.gateway`` in proxy.yaml).
    gateway_name = "litellm"

    #: Feature flag gating this gateway; enabled on every channel.
    required_feature_flag = "gateway.litellm"

    def __init__(
        self,
        host: str = DEFAULT_PROXY_HOST,
        port: int = DEFAULT_PROXY_PORT,
        master_key: str | None = DEFAULT_MASTER_KEY,
        state_dir: Path | None = None,
        host_configured: bool = False,
    ):
        self.host = host
        self.port = port
        self.master_key = master_key
        # Whether the bind host was explicitly configured by the user.  When
        # False and host is the legacy 0.0.0.0 default, start() warns loudly.
        self.host_configured = host_configured

        if state_dir is None:
            from sparkrun.core.config import DEFAULT_CACHE_DIR

            state_dir = DEFAULT_CACHE_DIR / "proxy"
        self.state_dir = state_dir
        self.state_file = state_dir / "state.yaml"
        self.config_path = state_dir / "litellm_config.yaml"
        self._autodiscover_config_path = state_dir / "autodiscover.yaml"
        # Handle for a proxy this instance spawned, so a later restart can
        # reap it.  None when the running proxy belongs to another process
        # (the usual case for the CLI talking to a daemonized proxy).
        self._proc: subprocess.Popen | None = None

    def start(
        self,
        config_path: Path | None = None,
        foreground: bool = False,
        dry_run: bool = False,
        autodiscover_kwargs: dict | None = None,
    ) -> int:
        """Launch the LiteLLM proxy server via uvx.

        Uses ``uvx --from 'litellm[proxy]==1.82.6' litellm`` to run the
        LiteLLM proxy server without requiring a permanent install.

        Note: ``litellm`` is the server command; ``litellm-proxy`` is the
        separate management CLI for interacting with a running proxy.

        Args:
            config_path: Path to litellm config YAML.
            foreground: Run in foreground (blocking).
            dry_run: Print command without executing.
            autodiscover_kwargs: When set, start a background auto-discover
                process after the proxy launches.  Keys: ``interval``,
                ``host_list``, ``ssh_kwargs``, ``cache_dir``.

        Returns:
            0 on success, non-zero on failure.

        Raises:
            GatewayUnavailableError: this gateway's feature flag is off.
        """
        # The one enforcement point for the gateway flag: bringing a gateway
        # *up*.  Stop / status / model sync / the auto-discover daemon's
        # restart path stay ungated so a proxy started while the flag was on
        # remains manageable (and stoppable) if it is later turned off.
        # Checked before --dry-run so a dry run can't advertise a start that
        # would be refused.
        require_gateway_enabled(self.gateway_name)

        cmd = self._build_command(config_path)
        if cmd is None:
            return 1

        if dry_run:
            logger.info("[dry-run] Would run: %s", " ".join(cmd))
            return 0

        self._warn_insecure_bind()

        if self.is_running():
            logger.warning("Proxy already running (PID %s)", self._read_pid())
            return 1

        self.state_dir.mkdir(parents=True, exist_ok=True)
        env = self._build_env()

        if foreground:
            proc = subprocess.Popen(cmd, env=env)
            self._save_state(proc.pid)
            if autodiscover_kwargs:
                ad_pid = self.start_autodiscover(
                    proxy_pid=proc.pid,
                    **autodiscover_kwargs,
                )
                if ad_pid:
                    self.update_autodiscover_pid(ad_pid)
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                return 130
            finally:
                self.stop_autodiscover()
                self._clear_state()
        else:
            pid = self._launch_background(cmd, env)
            if pid is None:
                return 1

            self._save_state(pid)

            if autodiscover_kwargs:
                ad_pid = self.start_autodiscover(
                    proxy_pid=pid,
                    **autodiscover_kwargs,
                )
                if ad_pid:
                    self.update_autodiscover_pid(ad_pid)

            logger.info("Proxy started (PID %d) on %s:%d", pid, self.host, self.port)
            logger.info("Log: %s", self.state_dir / "litellm.log")
            return 0

    def _build_command(self, config_path: Path | None = None) -> list[str] | None:
        """Build the ``uvx litellm`` argv. Returns None when uvx is missing."""
        uvx = shutil.which("uvx")
        if not uvx:
            logger.error("uvx not found on PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/")
            return None

        if config_path is None:
            config_path = self.config_path

        return [
            uvx,
            "--from",
            "litellm[proxy]==1.82.6",
            "litellm",
            "--config",
            str(config_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

    def _build_env(self) -> dict[str, str]:
        """Environment for the litellm subprocess.

        master_key auth uses LiteLLM's stateless bearer-token check, configured
        via ``general_settings.master_key`` in the YAML emitted by
        :func:`build_litellm_config`.  Features that need a database (virtual
        keys, budgets, request logging, the ``/ui``) are out of scope: LiteLLM's
        bundled ``schema.prisma`` declares ``provider = "postgresql"`` and its
        client is not shipped in ``litellm[proxy]``.

        ``DATABASE_URL`` is *stripped*, not merely left unset.  litellm treats
        its mere presence as "use a database" and aborts startup with
        ``ModuleNotFoundError: No module named 'prisma'`` — so an operator who
        happens to export it for an unrelated application would otherwise be
        unable to start the proxy at all, with a wholly unrelated error.
        """
        env = os.environ.copy()
        if env.pop("DATABASE_URL", None) is not None:
            logger.debug("Ignoring inherited DATABASE_URL; the proxy runs without a database")
        return env

    def _launch_background(self, cmd: list[str], env: dict[str, str]) -> int | None:
        """Spawn the proxy detached and confirm it survived startup.

        Returns:
            The new PID, or None if the process exited during the grace period.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Redirect output to log file so startup errors are visible
        log_path = self.state_dir / "litellm.log"
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
            logger.error("Failed to spawn proxy process", exc_info=True)
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
                "Proxy exited immediately (code %d). Log tail:\n%s",
                poll,
                tail,
            )
            return None

        self._proc = proc
        return proc.pid

    def _await_proxy_exit(self, pid: int, timeout: float) -> bool:
        """Wait for proxy *pid* to exit, reaping it when it is our child.

        A process we spawned stays a zombie until reaped, and
        ``os.kill(pid, 0)`` *succeeds* on a zombie — so PID polling alone
        would burn the whole timeout and then escalate to SIGKILL against a
        process that already exited.  The auto-discover daemon spawns every
        replacement proxy itself, so from its second restart onward this is
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

    def _warn_insecure_bind(self) -> None:
        """Emit a loud warning when binding 0.0.0.0 without explicit config.

        Legacy compatibility: when no bind host has been explicitly configured
        (``host_configured`` is False) the proxy keeps binding the legacy
        ``0.0.0.0`` default so existing deployments do not break.  This exposes
        every discovered inference backend to the whole network, so we warn
        loudly on every start and explain how to silence it.
        """
        if self.host_configured:
            return
        if self.host not in ("0.0.0.0", "::"):
            return

        if self.master_key:
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

    def start_autodiscover(
        self,
        proxy_pid: int,
        interval: int = 30,
        host_list: list[str] | None = None,
        ssh_kwargs: dict | None = None,
        cache_dir: str | None = None,
    ) -> int | None:
        """Spawn the background auto-discovery process.

        Writes a config file and launches
        ``python -m sparkrun.proxy.autodiscover`` as a detached subprocess.

        Returns:
            PID of the auto-discover process, or None on failure.
        """
        cfg = {
            "proxy_pid": proxy_pid,
            "proxy_port": self.port,
            "master_key": self.master_key,
            "interval": interval,
        }
        if host_list:
            cfg["host_list"] = host_list
        if ssh_kwargs is not None:
            cfg["ssh_kwargs"] = ssh_kwargs
        if cache_dir:
            cfg["cache_dir"] = cache_dir

        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)
        # The autodiscover config carries the master_key, so create it owner-only
        # (no default-umask window) and refuse to follow a symlink at the target.
        fd = open_private_write(self._autodiscover_config_path)
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
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
            logger.info(
                "Auto-discover started (PID %d), interval=%ds, log=%s",
                proc.pid,
                interval,
                log_path,
            )
            return proc.pid
        except Exception:
            log_file.close()
            logger.warning("Failed to start auto-discover process", exc_info=True)
            return None

    def stop_autodiscover(self) -> None:
        """Stop the background auto-discovery process if running."""
        ad_pid = self._read_autodiscover_pid()
        if ad_pid is None:
            return
        try:
            os.kill(ad_pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to auto-discover PID %d", ad_pid)
        except ProcessLookupError:
            logger.debug("Auto-discover PID %d already gone", ad_pid)
        except PermissionError:
            logger.warning("Permission denied stopping auto-discover PID %d", ad_pid)
        # Clean up config file
        self._autodiscover_config_path.unlink(missing_ok=True)

    def stop(self, dry_run: bool = False) -> bool:
        """Stop the running proxy and auto-discover (SIGTERM via PID).

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

        # Stop auto-discover first (it monitors proxy PID anyway,
        # but explicit stop is cleaner)
        self.stop_autodiscover()

        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to proxy PID %d", pid)
            # Reap it *only* if we spawned it, so it does not linger as a
            # zombie (which still answers os.kill(pid, 0) and reads as
            # running).  Never block on a proxy owned by another process —
            # `sparkrun proxy stop` must stay instant.
            if self._proc is not None and self._proc.pid == pid:
                self._await_proxy_exit(pid, RESTART_EXIT_TIMEOUT)
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
        """Check if the proxy process is alive."""
        pid = self._read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    # -- Management API client --

    def list_models_via_api(self) -> list[dict[str, Any]]:
        """Query registered models via GET /model/info.

        Returns:
            List of model info dicts from litellm.
        """
        try:
            data = self._api_request("GET", "/model/info")
            return data.get("data", [])
        except Exception:
            logger.debug("Failed to list models via management API", exc_info=True)
            return []

    def apply_desired_state(
        self,
        endpoints: list[DiscoveredEndpoint],
        aliases: dict[str, str] | None = None,
        *,
        restart: bool = True,
    ) -> tuple[int, int]:
        """Make the proxy serve exactly *endpoints* (plus *aliases*).

        The litellm config file is the single source of truth for the model
        list.  LiteLLM's runtime mutation endpoints (``/model/new``,
        ``/model/delete``) are **not** usable here: they require a DB-backed
        model store, which needs PostgreSQL plus a generated prisma client,
        so against a sparkrun-launched proxy they answer
        ``500 No DB Connected``.  Applying a change therefore means
        rewriting the config and restarting the process.

        The restart is skipped entirely when the desired model set already
        matches what is on disk, so a steady-state auto-discover sweep costs
        nothing and never interrupts serving.

        ``general_settings`` is carried over verbatim from the existing
        config so a restart can never rotate the running proxy's master key,
        even when called from a bare ``ProxyEngine()``.

        Args:
            endpoints: Healthy discovered endpoints the proxy should serve.
            aliases: Alias name -> target model mapping to re-apply.
            restart: When False, write the config but leave the running
                proxy alone (the change lands on its next start).

        Returns:
            Tuple of (added_count, removed_count) — model entries that
            appeared and disappeared relative to the previous config.

        Raises:
            ProxyRestartError: The config was rewritten but the running
                proxy could not be replaced.
        """
        if self.is_running():
            self._adopt_running_identity()

        desired = build_litellm_config(endpoints, master_key=self.master_key, aliases=aliases)
        current = self._load_current_config()

        # Preserve auth and any other operator-set general_settings.
        if current.get("general_settings"):
            desired["general_settings"] = current["general_settings"]

        desired_keys = _model_keys(desired)
        current_keys = _model_keys(current)
        added = len(desired_keys - current_keys)
        removed = len(current_keys - desired_keys)

        if not added and not removed:
            logger.debug("Proxy model list already matches discovered endpoints")
            return 0, 0

        write_config(desired, self.config_path)
        logger.info(
            "Proxy config updated: +%d model entry/entries, -%d",
            added,
            removed,
        )

        if not self.is_running():
            logger.debug("Proxy not running — new config applies on next start")
            return added, removed

        if not restart:
            logger.warning(
                "Proxy config updated but restart was suppressed; run 'sparkrun proxy start --restart' to serve the new model list."
            )
            return added, removed

        if self._restart_proxy() is None:
            raise ProxyRestartError("Proxy config was updated but the proxy failed to restart; see %s" % (self.state_dir / "litellm.log"))
        return added, removed

    def sync_models(
        self,
        endpoints: list[DiscoveredEndpoint],
        aliases: dict[str, str] | None = None,
    ) -> tuple[int, int]:
        """Synchronize proxy models with discovered endpoints.

        Thin wrapper over :meth:`apply_desired_state`.  When *aliases* is
        None the configured aliases are read from ``proxy.yaml`` — omitting
        them would silently drop every alias from the regenerated config.

        Args:
            endpoints: Healthy discovered endpoints.
            aliases: Alias mapping to preserve; read from config if None.

        Returns:
            Tuple of (added_count, removed_count).
        """
        if aliases is None:
            aliases = self._configured_aliases()
        return self.apply_desired_state(endpoints, aliases)

    def sync_aliases(self, aliases: dict[str, str]) -> tuple[int, int]:
        """Ensure all configured aliases are served by the proxy.

        The endpoint half of the desired state is recovered from the current
        config, so this only adds/removes alias entries.

        Args:
            aliases: Alias name -> target model name mapping.

        Returns:
            Tuple of (added_count, removed_count).
        """
        return self.apply_desired_state(self._endpoints_from_config(), aliases)

    def _restart_proxy(self) -> int | None:
        """Replace the running proxy with one reading the current config.

        The auto-discover daemon is deliberately **not** stopped or
        respawned: it re-reads the proxy PID from the state file each sweep,
        so it follows the restart.  Spawning a second one here would leave
        two daemons racing to rewrite the same config.

        Returns:
            The new proxy PID, or None on failure.
        """
        cmd = self._build_command()
        if cmd is None:
            return None

        old_pid = self._read_pid()
        ad_pid = self._read_autodiscover_pid()

        if old_pid is not None:
            try:
                os.kill(old_pid, signal.SIGTERM)
                logger.info("Restarting proxy: SIGTERM to PID %d", old_pid)
            except ProcessLookupError:
                old_pid = None
            except PermissionError:
                logger.error("Permission denied signalling proxy PID %d", old_pid)
                return None

        if old_pid is not None and not self._await_proxy_exit(old_pid, RESTART_EXIT_TIMEOUT):
            logger.warning(
                "Proxy PID %d did not exit within %.0fs; escalating to SIGKILL",
                old_pid,
                RESTART_EXIT_TIMEOUT,
            )
            try:
                os.kill(old_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not self._await_proxy_exit(old_pid, 5.0):
                logger.error("Proxy PID %d survived SIGKILL; not starting a replacement", old_pid)
                return None

        self._proc = None

        pid = self._launch_background(cmd, self._build_env())
        if pid is None:
            # The old proxy is already gone; clear state so callers and the
            # auto-discover daemon see "not running" rather than a stale PID.
            self._clear_state()
            return None

        self._save_state(pid, autodiscover_pid=ad_pid)
        logger.info("Proxy restarted (PID %d) on %s:%d", pid, self.host, self.port)
        return pid

    def _adopt_running_identity(self) -> None:
        """Take host/port/master_key from the running proxy's state file.

        ``ProxyEngine()`` is frequently constructed bare (the CLI does it in
        half a dozen places), so its defaults can disagree with the proxy
        that is actually running.  A restart must not silently move the
        proxy's port or rotate its master key, so the live values win.
        """
        state = self.get_state() or {}
        if "port" in state:
            self.port = int(state["port"])
        if "host" in state:
            self.host = str(state["host"])
        if "master_key" in state:
            self.master_key = state["master_key"]

    def _load_current_config(self) -> dict[str, Any]:
        """Read the litellm config currently on disk ({} when absent)."""
        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            logger.debug("Could not read litellm config %s", self.config_path, exc_info=True)
            return {}
        return data if isinstance(data, dict) else {}

    def _configured_aliases(self) -> dict[str, str]:
        """Read aliases from ``proxy.yaml`` (empty when unreadable)."""
        try:
            from sparkrun.proxy.config import ProxyConfig

            return ProxyConfig().aliases
        except Exception:
            logger.debug("Could not read configured aliases", exc_info=True)
            return {}

    def _endpoints_from_config(self) -> list[DiscoveredEndpoint]:
        """Recover the endpoint half of the desired state from the config.

        Alias entries are skipped — they are regenerated from the alias map,
        and an alias is recognised by its ``litellm_params.model`` naming a
        *different* model than its own ``model_name``.

        Every field the config carries must be recovered here, because this
        is the *whole* input to the rebuild: ``sync_aliases`` feeds these
        endpoints straight back into :func:`build_litellm_config`, so
        anything dropped is erased from the config the moment an alias is
        added or removed.  The advertised context window was, which meant a
        single ``proxy alias add`` stripped ``model_info`` from every model —
        and, because the discovery sweep only rewrites when the model set
        itself changes, it never came back.
        """
        endpoints: list[DiscoveredEndpoint] = []
        for entry in self._load_current_config().get("model_list") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("model_name", ""))
            params = entry.get("litellm_params") or {}
            if params.get("model") != "openai/%s" % name:
                continue
            hostport = _parse_api_base(str(params.get("api_base", "")))
            if hostport is None:
                continue
            host, port = hostport
            window = (entry.get("model_info") or {}).get("max_input_tokens")
            endpoints.append(
                DiscoveredEndpoint(
                    cluster_id="",
                    model=name,
                    served_model_name=name,
                    runtime="",
                    host=host,
                    port=port,
                    healthy=True,
                    actual_models=[name],
                    api_key=params.get("api_key"),
                    max_model_len=window if isinstance(window, int) else None,
                )
            )
        return endpoints

    def current_pid(self) -> int | None:
        """PID of the running proxy per the state file, or None."""
        return self._read_pid()

    def _api_request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """Make an HTTP request to the litellm management API."""
        url = "http://localhost:%d%s" % (self.port, path)
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.master_key:
            headers["Authorization"] = "Bearer %s" % self.master_key

        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    # -- State management --

    def _save_state(self, pid: int, autodiscover_pid: int | None = None) -> None:
        """Save proxy state to disk.

        The state file contains the master_key, so it is written with
        owner-only (0o600) permissions and its parent dir is restricted to
        0o700 — mirroring ``arena.auth.save_refresh_token``.
        """
        import datetime

        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)
        state = {
            "pid": pid,
            "port": self.port,
            "host": self.host,
            # Which gateway implementation owns this process.  Management
            # paths (stop / status / sync) read it back so they act on what
            # is *running* rather than on what is currently configured.
            "gateway": self.gateway_name,
            "master_key": self.master_key,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if autodiscover_pid is not None:
            state["autodiscover_pid"] = autodiscover_pid
        with open(self.state_file, "w") as f:
            yaml.safe_dump(state, f, default_flow_style=False)
        _restrict_file_permissions(self.state_file)

    def _read_pid(self) -> int | None:
        """Read PID from state file."""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file) as f:
                state = yaml.safe_load(f)
            return int(state["pid"]) if state and "pid" in state else None
        except Exception:
            return None

    def _read_autodiscover_pid(self) -> int | None:
        """Read auto-discover PID from state file."""
        state = self.get_state()
        if state and "autodiscover_pid" in state:
            return int(state["autodiscover_pid"])
        return None

    def update_autodiscover_pid(self, autodiscover_pid: int) -> None:
        """Record the auto-discover PID in state (call after start)."""
        pid = self._read_pid()
        if pid is not None:
            self._save_state(pid, autodiscover_pid=autodiscover_pid)

    def _clear_state(self) -> None:
        """Remove state file."""
        self.state_file.unlink(missing_ok=True)

    def get_state(self) -> dict[str, Any] | None:
        """Read full proxy state. Returns None if not found."""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file) as f:
                return yaml.safe_load(f)
        except Exception:
            return None
