"""LiteLLM proxy engine — config generation, subprocess lifecycle, management API.

Launches ``uvx litellm`` as a subprocess and manages its lifecycle.
Uses the litellm management API for runtime model add/query operations.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from sparkrun.proxy import (
    DEFAULT_ENABLE_UI,
    DEFAULT_MASTER_KEY,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    DEFAULT_UI_USERNAME,
)
from sparkrun.proxy.discovery import DiscoveredEndpoint

logger = logging.getLogger(__name__)


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
) -> dict[str, Any]:
    """Generate a litellm proxy config dict from discovered endpoints.

    When *master_key* is set, LiteLLM performs stateless bearer-token
    authentication against ``general_settings.master_key`` — no backing
    database is required.  Features that need a DB (virtual keys,
    budgets, request logging) are intentionally out of scope.

    Args:
        endpoints: Discovered inference endpoints.
        master_key: Bearer token for stateless auth.  When ``None``,
            no authentication is configured.

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

            model_list.append(
                {
                    "model_name": model_name,
                    "litellm_params": {
                        "model": "openai/%s" % model_name,
                        "api_base": "http://%s:%d/v1" % (ep.host, ep.port),
                        "api_key": ep.api_key or "not-needed",
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
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
    _restrict_file_permissions(config_path)

    logger.debug("Wrote litellm config to %s", config_path)
    return config_path


class ProxyEngine:
    """Manages the litellm proxy subprocess and its management API."""

    def __init__(
        self,
        host: str = DEFAULT_PROXY_HOST,
        port: int = DEFAULT_PROXY_PORT,
        master_key: str | None = DEFAULT_MASTER_KEY,
        state_dir: Path | None = None,
        enable_ui: bool = DEFAULT_ENABLE_UI,
        ui_username: str | None = None,
        ui_password: str | None = None,
        host_configured: bool = False,
    ):
        self.host = host
        self.port = port
        self.master_key = master_key
        # Whether the bind host was explicitly configured by the user.  When
        # False and host is the legacy 0.0.0.0 default, start() warns loudly.
        self.host_configured = host_configured
        self.enable_ui = enable_ui
        self.ui_username = ui_username
        self.ui_password = ui_password

        if state_dir is None:
            from sparkrun.core.config import DEFAULT_CACHE_DIR

            state_dir = DEFAULT_CACHE_DIR / "proxy"
        self.state_dir = state_dir
        self.state_file = state_dir / "state.yaml"
        self.config_path = state_dir / "litellm_config.yaml"
        self._autodiscover_config_path = state_dir / "autodiscover.yaml"

        if self.enable_ui and self.master_key is None:
            raise ValueError("enable_ui requires master_key to be set (LiteLLM UI login needs the master key)")

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
        """
        uvx = shutil.which("uvx")
        if not uvx:
            logger.error("uvx not found on PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/")
            return 1

        if config_path is None:
            config_path = self.config_path

        cmd = [
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

        if dry_run:
            logger.info("[dry-run] Would run: %s", " ".join(cmd))
            return 0

        self._warn_insecure_bind()

        if self.is_running():
            logger.warning("Proxy already running (PID %s)", self._read_pid())
            return 1

        self.state_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()

        # master_key auth uses LiteLLM's stateless bearer-token check, configured
        # via general_settings.master_key in the YAML emitted by build_litellm_config.
        # No DB env-vars are injected: virtual keys, budgets, and request logging —
        # features that require a backing DB — are intentionally out of scope.
        #
        # Opt-in UI mode (enable_ui=True) re-adds DATABASE_URL because LiteLLM's
        # /ui requires a DB.  master_key precondition is enforced in __init__.
        if self.enable_ui:
            env["DATABASE_URL"] = "sqlite:///%s" % (self.state_dir / "litellm.db")
            env["UI_USERNAME"] = self.ui_username or DEFAULT_UI_USERNAME
            env["UI_PASSWORD"] = self.ui_password or self.master_key

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
            # Redirect output to log file so startup errors are visible
            log_path = self.state_dir / "litellm.log"
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )

            # Wait briefly and verify the process survived startup
            import time

            time.sleep(2)
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
                return poll or 1

            self._save_state(proc.pid)

            if autodiscover_kwargs:
                ad_pid = self.start_autodiscover(
                    proxy_pid=proc.pid,
                    **autodiscover_kwargs,
                )
                if ad_pid:
                    self.update_autodiscover_pid(ad_pid)

            logger.info("Proxy started (PID %d) on %s:%d", proc.pid, self.host, self.port)
            logger.info("Log: %s", log_path)
            return 0

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
        fd = os.open(self._autodiscover_config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
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

    def add_model_via_api(self, endpoint: DiscoveredEndpoint) -> bool:
        """Add a model to the running proxy via POST /model/new.

        Returns:
            True if the model was added successfully.
        """
        model_names = endpoint.actual_models if endpoint.actual_models else [endpoint.model]

        success = True
        for model_name in model_names:
            payload = {
                "model_name": model_name,
                "litellm_params": {
                    "model": "openai/%s" % model_name,
                    "api_base": "http://%s:%d/v1" % (endpoint.host, endpoint.port),
                    "api_key": endpoint.api_key or "not-needed",
                },
            }

            try:
                self._api_request("POST", "/model/new", payload)
            except Exception:
                logger.debug(
                    "Failed to add model %s via management API",
                    model_name,
                    exc_info=True,
                )
                success = False

        return success

    def remove_model_via_api(self, model_id: str) -> bool:
        """Remove a model from the running proxy via POST /model/delete.

        Args:
            model_id: The model ID from ``list_models_via_api()``.

        Returns:
            True if the model was removed successfully.
        """
        try:
            self._api_request("POST", "/model/delete", {"id": model_id})
            return True
        except Exception:
            logger.debug(
                "Failed to remove model %s via management API",
                model_id,
                exc_info=True,
            )
            return False

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

    def sync_models(self, endpoints: list[DiscoveredEndpoint]) -> tuple[int, int]:
        """Synchronize proxy models with discovered endpoints.

        Adds models from healthy endpoints that aren't registered yet,
        and removes registered models whose backends are no longer
        present in the discovered endpoints.

        Args:
            endpoints: Healthy discovered endpoints.

        Returns:
            Tuple of (added_count, removed_count).
        """
        # Build set of expected api_base URLs from healthy endpoints
        healthy_bases: set[str] = set()
        for ep in endpoints:
            healthy_bases.add("http://%s:%d/v1" % (ep.host, ep.port))

        # Query currently registered models
        registered = self.list_models_via_api()

        # Remove stale models (backend no longer healthy)
        removed = 0
        for m in registered:
            model_id = m.get("model_info", {}).get("id")
            api_base = m.get("litellm_params", {}).get("api_base", "")
            if api_base and api_base not in healthy_bases and model_id:
                model_name = m.get("model_name", model_id)
                if self.remove_model_via_api(model_id):
                    logger.info("Removed stale model: %s (%s)", model_name, api_base)
                    removed += 1

        # Add new models from healthy endpoints
        # Re-query after removals to get current state
        if removed:
            registered = self.list_models_via_api()

        registered_keys: set[str] = set()
        for m in registered:
            name = m.get("model_name", "")
            api_base = m.get("litellm_params", {}).get("api_base", "")
            if name and api_base:
                registered_keys.add("%s@%s" % (name, api_base))

        added = 0
        for ep in endpoints:
            model_names = ep.actual_models if ep.actual_models else [ep.model]
            api_base = "http://%s:%d/v1" % (ep.host, ep.port)
            for model_name in model_names:
                key = "%s@%s" % (model_name, api_base)
                if key not in registered_keys:
                    if self.add_model_via_api(ep):
                        added += 1
                    break  # add_model_via_api handles all models for the endpoint

        return added, removed

    def add_alias_via_api(self, alias_name: str, target_model: str) -> bool:
        """Add an alias by registering it as a model pointing to the same backend(s).

        Finds all registered backends for *target_model* and adds
        *alias_name* entries pointing to the same backends via the
        management API.  No proxy restart required.

        Returns:
            True if at least one alias entry was added.
        """
        registered = self.list_models_via_api()
        backends = [m.get("litellm_params", {}) for m in registered if m.get("model_name") == target_model]

        if not backends:
            logger.warning(
                "Cannot add alias %r: target model %r not found in proxy",
                alias_name,
                target_model,
            )
            return False

        success = False
        for params in backends:
            api_base = params.get("api_base", "")
            payload = {
                "model_name": alias_name,
                "litellm_params": {
                    "model": "openai/%s" % target_model,
                    "api_base": api_base,
                    "api_key": params.get("api_key") or "not-needed",
                },
            }
            try:
                self._api_request("POST", "/model/new", payload)
                success = True
            except Exception:
                logger.debug(
                    "Failed to add alias %s via management API",
                    alias_name,
                    exc_info=True,
                )

        return success

    def remove_alias_via_api(self, alias_name: str) -> int:
        """Remove all model entries matching *alias_name* from the proxy.

        Returns:
            Number of entries removed.
        """
        registered = self.list_models_via_api()
        removed = 0
        for m in registered:
            if m.get("model_name") == alias_name:
                model_id = m.get("model_info", {}).get("id")
                if model_id and self.remove_model_via_api(model_id):
                    removed += 1
        return removed

    def sync_aliases(self, aliases: dict[str, str]) -> tuple[int, int]:
        """Ensure all configured aliases are registered with the proxy.

        Adds missing alias entries and removes alias entries whose
        alias name is no longer in the configuration.

        Args:
            aliases: Alias name -> target model name mapping.

        Returns:
            Tuple of (added_count, removed_count).
        """
        registered = self.list_models_via_api()

        # Build lookup: model_name -> list of litellm_params
        registered_by_name: dict[str, list[dict]] = {}
        for m in registered:
            name = m.get("model_name", "")
            registered_by_name.setdefault(name, []).append(m)

        added = 0
        removed = 0

        # Add missing aliases
        for alias_name, target_model in aliases.items():
            if alias_name in registered_by_name:
                continue  # Already registered
            if self.add_alias_via_api(alias_name, target_model):
                added += 1

        # Remove stale aliases: entries that were previously added as
        # aliases but are no longer in the alias config.  We identify
        # these as model entries whose model_name is NOT a real served
        # model (not a target of any alias) and NOT in the current
        # alias config.
        target_models = set(aliases.values())
        real_model_names = set()
        for m in registered:
            # A "real" model has model_name matching the openai/ suffix
            litellm_model = m.get("litellm_params", {}).get("model", "")
            if litellm_model == "openai/%s" % m.get("model_name", ""):
                real_model_names.add(m.get("model_name", ""))

        for m in registered:
            name = m.get("model_name", "")
            if name in real_model_names:
                continue  # Real model, not an alias
            if name in aliases:
                continue  # Still a configured alias
            if name in target_models:
                continue  # It's a target model name
            # This looks like a stale alias — but only remove if the
            # model param doesn't match its name (i.e. it was an alias)
            litellm_model = m.get("litellm_params", {}).get("model", "")
            if litellm_model != "openai/%s" % name:
                model_id = m.get("model_info", {}).get("id")
                if model_id and self.remove_model_via_api(model_id):
                    removed += 1

        return added, removed

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
