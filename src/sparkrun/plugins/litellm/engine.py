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
from sparkrun.proxy._supervisor import (
    SUPERVISOR_ENV,
    supervisor_value as _supervisor_value,
    RESTART_EXIT_TIMEOUT as _RESTART_EXIT_TIMEOUT,
    RESTART_STARTUP_GRACE as _RESTART_STARTUP_GRACE,
    GatewayOperationError,
    GatewaySupervisor,
    _is_zombie,  # noqa: F401 — re-exported; imported from here historically
    _restrict_dir_permissions,
    _restrict_file_permissions,
    _wait_for_exit,  # noqa: F401 — re-exported; imported from here historically
)
from sparkrun.plugins.litellm.media_mode import MEDIA_MODES, media_mode
from sparkrun.proxy.contracts import GatewayQueryError, ProxyModel
from sparkrun.proxy.discovery import DiscoveredEndpoint
from sparkrun.proxy.gateway import require_gateway_enabled
from sparkrun.utils.fs import atomic_private_write, open_private_write

logger = logging.getLogger(__name__)

# How long a restart waits for the old proxy to exit before escalating to
# SIGKILL, and how long the replacement gets to survive startup.  Owned by
# _supervisor (shared with every gateway); re-exported here because callers and
# tests have always imported them from this module.
RESTART_EXIT_TIMEOUT = _RESTART_EXIT_TIMEOUT
RESTART_STARTUP_GRACE = _RESTART_STARTUP_GRACE

#: ``gateways.litellm.balancing`` values sparkrun implements itself, mapped to
#: the callback LiteLLM loads (``<module>.<instance>``, resolved next to the
#: generated config).  See ``plugins/litellm/balancer.py``.
BALANCING_CALLBACKS = {"least-in-flight": "sparkrun_balancer.handler"}

#: File name the balancer module is written under, beside the generated config.
BALANCER_MODULE_FILE = "sparkrun_balancer.py"

#: The routing strategy a balancing callback runs under.  LiteLLM only calls
#: ``async_filter_deployments`` on its async routing path, which it takes for
#: five strategies; any other (``usage-based-routing``, say) bypasses the
#: callback entirely, so a configured strategy is replaced with this one.
BALANCING_ROUTING_STRATEGY = "simple-shuffle"

#: How long a restart routed through a systemd unit waits for the unit's new
#: foreground process to record its gateway (it runs a discovery sweep first).
UNIT_RESTART_TIMEOUT = 120.0


class ProxyRestartError(GatewayOperationError):
    """Raised when a config change could not be applied to the running proxy.

    The config file on disk has already been updated when this is raised —
    the *running* process is what failed to pick it up.
    """


_warned: set[str] = set()


def _warn_once(message: str, *args: Any) -> None:
    """Log a ``proxy.yaml`` problem once per process.

    The auto-discover daemon re-reads the settings on every sweep, so a plain
    warning would repeat for as long as the file is wrong.
    """
    text = message % args
    if text not in _warned:
        _warned.add(text)
        logger.warning("%s", text)


def build_litellm_config(
    endpoints: list[DiscoveredEndpoint],
    master_key: str | None = DEFAULT_MASTER_KEY,
    aliases: dict[str, str] | None = None,
    router_settings: dict[str, Any] | None = None,
    balancing: str | None = None,
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
        router_settings: Optional LiteLLM ``router_settings`` emitted
            verbatim (e.g. ``{"routing_strategy": "least-busy"}``).  Without
            it LiteLLM uses ``simple-shuffle``: a random backend per request,
            blind to how many requests each one is already serving.
        balancing: Optional sparkrun-side balancing policy (a key of
            :data:`BALANCING_CALLBACKS`).  ``least-in-flight`` registers the
            ``sparkrun_balancer`` callback, which sends each request to the
            backend of its model group with the fewest requests in flight.
            Unknown values are ignored here; the engine warns when reading them.
            A ``routing_strategy`` in *router_settings* is replaced with
            :data:`BALANCING_ROUTING_STRATEGY` so LiteLLM keeps calling it.

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
            # A diffusion backend advertises what it does (T2I/T2V/...) on its
            # /v1/models entry; LiteLLM speaks ``model_info.mode`` instead, so
            # without this translation an image or video pipeline is written as
            # an ordinary chat deployment and the gateway's own /v1/models
            # misreports it.  No task vocabulary (every chat backend) -> no
            # ``mode`` key, and the entry is left exactly as it was before.
            mode = media_mode(*ep.task_types, primary=ep.task_type)
            if mode:
                entry.setdefault("model_info", {})["mode"] = mode
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

    callback = BALANCING_CALLBACKS.get(balancing or "")
    if callback:
        config["litellm_settings"]["callbacks"] = [callback]

    if general_settings:
        config["general_settings"] = general_settings

    if router_settings:
        config["router_settings"] = dict(router_settings)
        if callback and "routing_strategy" in router_settings:
            config["router_settings"]["routing_strategy"] = BALANCING_ROUTING_STRATEGY

    return config


def _model_keys(config: dict[str, Any]) -> set[tuple[str, str, int | None, str | None]]:
    """Identity of a config's model list as ``{(model_name, api_base, window, mode)}``.

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

    ``mode`` is in the identity for the same reason and by the same argument:
    it lives in ``model_info``, LiteLLM reads it once at startup for
    capability routing, and a proxy whose config was written before the tag
    existed would otherwise never be rewritten — the name, base and window of
    a media model are all unchanged when the mode is what is missing.
    """
    keys: set[tuple[str, str, int | None, str | None]] = set()
    for entry in config.get("model_list") or []:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params") or {}
        model_info = entry.get("model_info") or {}
        window = model_info.get("max_input_tokens")
        mode = model_info.get("mode")
        keys.add(
            (
                str(entry.get("model_name", "")),
                str(params.get("api_base", "")),
                window if isinstance(window, int) else None,
                mode if isinstance(mode, str) else None,
            )
        )
    return keys


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
        from sparkrun.core.config import resolve_sparkrun_cache_dir

        config_path = resolve_sparkrun_cache_dir() / "proxy" / "litellm_config.yaml"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    _restrict_dir_permissions(config_path.parent)
    # The balancer module goes in first: a LiteLLM started from a config must
    # never find the callback it names missing.
    balanced = bool(set(BALANCING_CALLBACKS.values()) & set((config_dict.get("litellm_settings") or {}).get("callbacks") or []))
    if balanced:
        write_balancer_module(config_path.parent)
    # The litellm config carries general_settings.master_key plus every upstream
    # endpoint api_key, so it must be owner-only.  Create it 0600 and refuse to
    # follow a symlink at the target path (mirrors orchestration.job_metadata);
    # _restrict_file_permissions afterwards repairs perms on a pre-existing file
    # written by an older version under the default umask.
    fd = open_private_write(config_path)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
    _restrict_file_permissions(config_path)
    if not balanced:
        # Only once the config no longer names it, for the same reason.
        (config_path.parent / BALANCER_MODULE_FILE).unlink(missing_ok=True)

    logger.debug("Wrote litellm config to %s", config_path)
    return config_path


def write_balancer_module(directory: Path) -> Path:
    """Copy the balancer callback beside the config, where LiteLLM imports it from.

    LiteLLM runs in its own ``uvx`` environment and resolves a config
    callback ``<module>.<instance>`` as ``<config dir>/<module>.py``, so the
    module has to exist as a file there rather than as an importable
    ``sparkrun`` package.  Rewritten on every config write so an upgraded
    sparkrun never leaves an older copy behind, and published atomically so
    a LiteLLM starting meanwhile never imports half a module.
    """
    from importlib import resources

    source = resources.files("sparkrun.plugins.litellm").joinpath("balancer.py").read_text()
    path = directory / BALANCER_MODULE_FILE
    atomic_private_write(path, source)
    return path


def _model_from_info(row: object) -> ProxyModel:
    """Translate one LiteLLM /model/info entry into the shared model record."""
    if not isinstance(row, dict):
        raise GatewayQueryError("LiteLLM returned an invalid model entry")
    name = row.get("model_name")
    info = row.get("model_info")
    info = {} if info is None else info
    if not isinstance(info, dict):
        raise GatewayQueryError("LiteLLM returned invalid model metadata")
    params = row.get("litellm_params")
    if params is None:
        params = info.get("litellm_params")
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise GatewayQueryError("LiteLLM returned invalid model parameters")
    base = params.get("api_base", "")
    if not isinstance(name, str) or not name or not isinstance(base, str):
        raise GatewayQueryError("LiteLLM returned an invalid model name or endpoint")
    length = next((info[key] for key in ("max_input_tokens", "max_tokens", "max_model_len") if info.get(key) is not None), None)
    if length is not None and (type(length) is not int or length < 0):
        raise GatewayQueryError("LiteLLM returned an invalid model context length")
    return ProxyModel(name, base, length)


def _master_key(value: str | None) -> str | None:
    """LiteLLM-only credential fallback, shared by launch and management."""
    return value if value is not None else (os.environ.get("LITELLM_MASTER_KEY") or DEFAULT_MASTER_KEY)


class ProxyEngine(GatewaySupervisor):
    """Manages the litellm proxy subprocess and its management API.

    This is the LiteLLM *gateway* implementation.  Process, state-file and
    auto-discover machinery lives on
    :class:`~sparkrun.proxy._supervisor.GatewaySupervisor`, shared with every
    other gateway; what remains here is LiteLLM's own argv, environment and
    config format.  :attr:`gateway_name` / :attr:`required_feature_flag` are
    the selector and gate (see :mod:`sparkrun.proxy.gateway`); ``start()`` is
    the single point that enforces the flag.
    """

    #: Selector this implementation answers to (``proxy.gateway`` in proxy.yaml).
    gateway_name = "litellm"

    #: Feature flag gating this gateway; enabled on every channel.
    required_feature_flag = "gateway.litellm"

    log_name = "litellm.log"

    def __init__(
        self,
        host: str = DEFAULT_PROXY_HOST,
        port: int = DEFAULT_PROXY_PORT,
        master_key: str | None = DEFAULT_MASTER_KEY,
        state_dir: Path | None = None,
        host_configured: bool = False,
    ):
        super().__init__(state_dir)
        self.host = host
        self.port = port
        self.master_key = _master_key(master_key)
        # Whether the bind host was explicitly configured by the user.  When
        # False and host is the legacy 0.0.0.0 default, start() warns loudly.
        self.host_configured = host_configured

        self.config_path = self.state_dir / "litellm_config.yaml"

    @property
    def data_plane_authenticated(self) -> bool:
        """LiteLLM authenticates the inference port iff a master key is set."""
        return bool(self.master_key)

    def _state_payload(self) -> dict[str, Any]:
        """Bind settings and credential recorded for management paths."""
        return {"port": self.port, "host": self.host, "master_key": self.master_key}

    def prepare_config(self, endpoints: list, aliases: dict[str, str], *, write: bool = True) -> tuple[Path | None, set[str], set[str]]:
        """Render *endpoints* into a LiteLLM config; see the base method.

        An alias only takes effect when a discovered endpoint answers to its
        model group, so the applied/pending split is read back out of the
        rendered config rather than assumed from *aliases*.
        """
        config_dict = self._build_desired_config(endpoints, aliases)
        applied = {entry["model_name"] for entry in config_dict["model_list"]} & set(aliases)
        if write:
            self.claim_state_directory()
        path = write_config(config_dict, self.config_path) if write else None
        return path, applied, set(aliases) - applied

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

        self.claim_state_directory()
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
                self._start_autodiscover_or_abort(proc.pid, autodiscover_kwargs, child=proc)
            # Follows the gateway across the daemon's managed restarts and
            # owns the cleanup (see GatewaySupervisor.supervise_foreground).
            return self.supervise_foreground(proc)
        else:
            pid = self._launch_background(cmd, env)
            if pid is None:
                return 1

            self._save_state(pid)

            if autodiscover_kwargs:
                self._start_autodiscover_or_abort(pid, autodiscover_kwargs)

            logger.info("Proxy started (PID %d) on %s:%d", pid, self.host, self.port)
            logger.info("Log: %s", self.log_path)
            return 0

    def _start_autodiscover_or_abort(self, pid: int, autodiscover_kwargs: dict, *, child: subprocess.Popen | None = None) -> None:
        """Start discovery for the gateway just spawned, or stop that gateway.

        A start that fails here would otherwise leave a recorded, running
        gateway behind an error, which a service unit restarts into the same
        failure.
        """
        try:
            ad_pid = self.start_autodiscover(proxy_pid=pid, **autodiscover_kwargs)
        except BaseException:
            logger.error("Auto-discover could not start; stopping the gateway just started (PID %d)", pid)
            self.stop()
            if child is not None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        if ad_pid:
            self.update_autodiscover_pid(ad_pid)

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
        # Keep inherited server auth consistent with the key rendered into
        # config/state. An explicit key wins over the caller's environment.
        if self.master_key:
            env["LITELLM_MASTER_KEY"] = self.master_key
        else:
            env.pop("LITELLM_MASTER_KEY", None)
        if env.pop("DATABASE_URL", None) is not None:
            logger.debug("Ignoring inherited DATABASE_URL; the proxy runs without a database")
        return env

    # -- Management API client --

    def query_models(self) -> tuple[ProxyModel, ...]:
        """Query /model/info and translate LiteLLM's wire format at the provider.

        Empty success returns (). Operational failures raise GatewayQueryError;
        programming exceptions and interrupts propagate unchanged.
        """
        try:
            data = self._api_request("GET", "/model/info")
        except urllib.error.HTTPError as exc:
            raise GatewayQueryError("LiteLLM model query failed (HTTP %d)" % exc.code) from exc
        except OSError as exc:
            raise GatewayQueryError("LiteLLM management API is unreachable") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GatewayQueryError("LiteLLM management API returned malformed JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise GatewayQueryError("LiteLLM returned an invalid model list")
        return tuple(_model_from_info(row) for row in data["data"])

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

        desired = self._build_desired_config(endpoints, aliases)
        current = self._load_current_config()

        # Preserve auth and any other operator-set general_settings.
        if current.get("general_settings"):
            desired["general_settings"] = current["general_settings"]

        desired_keys = _model_keys(desired)
        current_keys = _model_keys(current)
        added = len(desired_keys - current_keys)
        removed = len(current_keys - desired_keys)

        # LiteLLM reads router_settings and its callbacks only at startup, so a
        # changed routing or balancing policy needs the same rewrite + restart
        # as a changed model set.
        router_changed = (desired.get("router_settings") or {}) != (current.get("router_settings") or {})
        balancing_changed = (desired["litellm_settings"].get("callbacks") or []) != (
            (current.get("litellm_settings") or {}).get("callbacks") or []
        )

        if not added and not removed and not router_changed and not balancing_changed:
            logger.debug("Proxy model list already matches discovered endpoints")
            return 0, 0

        write_config(desired, self.config_path)
        logger.info(
            "Proxy config updated: +%d model entry/entries, -%d%s%s",
            added,
            removed,
            ", router_settings changed" if router_changed else "",
            ", balancing changed" if balancing_changed else "",
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
            raise ProxyRestartError("Proxy config was updated but the proxy failed to restart; see %s" % (self.log_path))
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

        A gateway supervised by a systemd unit is restarted **through the
        unit** when this runs outside it (``proxy alias add`` in a shell):
        spawning the replacement here would put it in the shell's session,
        outside the unit's cgroup and beyond its stop.  Inside the unit (the
        daemon inherits its environment) the replacement is spawned directly
        and keeps the unit's record.

        Returns:
            The new proxy PID, or None on failure (including a stop that
            arrived mid-restart, which wins).
        """
        state = self.get_state() or {}
        from sparkrun.core.process_identity import alive, process_identity

        controller = state.get("controller")
        if controller and controller != process_identity():
            if not alive(controller):
                raise ProxyRestartError("Gateway supervisor is unavailable; restart the proxy explicitly")
            import uuid

            request = uuid.uuid4().hex
            self._mark_state(restart_requested=request)
            deadline = time.monotonic() + RESTART_EXIT_TIMEOUT + 30
            while time.monotonic() < deadline:
                current = self.get_state() or {}
                if not current or current.get("stop_requested"):
                    return None
                if current.get("restart_completed") == request:
                    if current.get("restart_error"):
                        raise ProxyRestartError(current["restart_error"])
                    return current.get("pid")
                time.sleep(0.1)
            raise ProxyRestartError("Gateway supervisor restart timed out; inspect proxy status before retrying")
        supervisor = state.get("supervisor") or None
        if supervisor and os.environ.get(SUPERVISOR_ENV) != _supervisor_value(supervisor):
            pid = self._restart_through_unit(supervisor)
            if pid is not None:
                return pid
            logger.warning("Could not restart through systemd unit %s; restarting the gateway directly", supervisor.get("unit"))

        cmd = self._build_command()
        if cmd is None:
            return None

        old_pid = self._read_pid()
        ad_pid = self._read_autodiscover_pid()

        if old_pid is not None:
            # Record intent first, so a foreground supervisor watching this
            # PID waits for the replacement instead of reporting a crash.
            self._mark_state(restarting=True)
            try:
                os.kill(old_pid, signal.SIGTERM)
                logger.info("Restarting proxy: SIGTERM to PID %d", old_pid)
            except ProcessLookupError:
                old_pid = None
            except PermissionError:
                logger.error("Permission denied signalling proxy PID %d", old_pid)
                self._mark_state(restarting=False)
                return None

        if old_pid is not None and not self._await_exit(old_pid, RESTART_EXIT_TIMEOUT):
            logger.warning(
                "Proxy PID %d did not exit within %.0fs; escalating to SIGKILL",
                old_pid,
                RESTART_EXIT_TIMEOUT,
            )
            try:
                os.kill(old_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not self._await_exit(old_pid, 5.0):
                logger.error("Proxy PID %d survived SIGKILL; not starting a replacement", old_pid)
                self._mark_state(restarting=False)
                return None

        self._proc = None

        if self._stop_requested():
            logger.info("Proxy stop requested during restart; not starting a replacement")
            return None

        pid = self._launch_background(cmd, self._build_env())
        if pid is None:
            # The old proxy is already gone. Keep its (dead) record but say
            # the restart failed, so a foreground supervisor reports a crash
            # rather than a requested stop; everyone else reads a dead PID as
            # "not running".
            self._mark_state(restarting=False, restart_failed=True)
            return None

        if not self._commit_replacement(pid, ad_pid, supervisor or "env"):
            logger.info("Proxy stop requested during restart; stopping replacement PID %d", pid)
            self._terminate_gateway(pid, self._proc)
            self._proc = None
            return None
        logger.info("Proxy restarted (PID %d) on %s:%d", pid, self.host, self.port)
        return pid

    def _stop_requested(self) -> bool:
        """A stop recorded (or the record cleared) since this restart began."""
        state = self.get_state()
        return state is None or bool(state.get("stop_requested"))

    def _restart_through_unit(self, supervisor: dict) -> int | None:
        """``systemctl restart`` the supervising unit; return the new gateway PID."""
        from sparkrun.proxy import _systemd

        spec = _systemd.unit_for_record(supervisor)
        if spec is None:
            return None
        old_pid = self._read_pid()
        result = _systemd.control(spec, "restart")
        if result.returncode != 0:
            logger.warning("systemctl restart %s failed: %s", spec.name, (result.stderr or result.stdout).strip())
            return None
        # The unit's new foreground process runs discovery before recording
        # its gateway, so allow for an SSH sweep.
        deadline = time.monotonic() + UNIT_RESTART_TIMEOUT
        while time.monotonic() < deadline:
            pid = self._read_pid()
            if pid is not None and pid != old_pid and self.is_running():
                logger.info("Proxy restarted through %s (PID %d)", spec.name, pid)
                return pid
            time.sleep(0.5)
        logger.error("%s restarted but recorded no gateway within %.0fs", spec.name, UNIT_RESTART_TIMEOUT)
        return None

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
            self.master_key = _master_key(state["master_key"])

    def _load_current_config(self) -> dict[str, Any]:
        """Read the litellm config currently on disk ({} when absent)."""
        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            logger.debug("Could not read litellm config %s", self.config_path, exc_info=True)
            return {}
        return data if isinstance(data, dict) else {}

    def _build_desired_config(self, endpoints: list, aliases: dict[str, str] | None) -> dict[str, Any]:
        """Render *endpoints* with the routing and balancing configured in ``proxy.yaml``."""
        router_settings = self._configured_router_settings()
        balancing = self._configured_balancing()
        strategy = router_settings.get("routing_strategy")
        if balancing and strategy is not None and strategy != BALANCING_ROUTING_STRATEGY:
            _warn_once(
                "gateways.%s.balancing %r decides the backend; router_settings.routing_strategy %r is replaced with %r, "
                "the strategy under which LiteLLM calls the balancer",
                self.gateway_name,
                balancing,
                strategy,
                BALANCING_ROUTING_STRATEGY,
            )
        return build_litellm_config(endpoints, self.master_key, aliases=aliases, router_settings=router_settings, balancing=balancing)

    def _configured_router_settings(self) -> dict[str, Any]:
        """Read ``gateways.litellm.router_settings`` from ``proxy.yaml`` (empty when unset or unreadable)."""
        try:
            from sparkrun.proxy.config import ProxyConfig

            value = ProxyConfig().gateway_settings(self.gateway_name).get("router_settings")
        except Exception:
            logger.debug("Could not read configured router_settings", exc_info=True)
            return {}
        if value is None:
            return {}
        if not isinstance(value, dict):
            _warn_once("Ignoring gateways.%s.router_settings in proxy.yaml: expected a mapping", self.gateway_name)
            return {}
        return value

    def _configured_balancing(self) -> str | None:
        """Read ``gateways.litellm.balancing`` from ``proxy.yaml`` (``None`` when unset, unknown or unreadable)."""
        try:
            from sparkrun.proxy.config import ProxyConfig

            value = ProxyConfig().gateway_settings(self.gateway_name).get("balancing")
        except Exception:
            logger.debug("Could not read configured balancing", exc_info=True)
            return None
        if value is None:
            return None
        if not isinstance(value, str) or value not in BALANCING_CALLBACKS:
            _warn_once(
                "Ignoring gateways.%s.balancing %r in proxy.yaml: expected one of %s",
                self.gateway_name,
                value,
                ", ".join(sorted(BALANCING_CALLBACKS)),
            )
            return None
        return value

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
            # ``mode`` has to come back out, or the first ``proxy alias add``
            # erases it from every media model and — because the discovery
            # sweep only rewrites when the model set changes — it never
            # returns.  A media mode is itself a token media_mode() accepts,
            # so it is recovered verbatim instead of reverse-mapped to a
            # backend-side enum this endpoint never advertised.
            mode = (entry.get("model_info") or {}).get("mode")
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
                    task_type=mode if mode in MEDIA_MODES else None,
                )
            )
        return endpoints

    def _api_request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """Make an authenticated request using the running LiteLLM identity."""
        self._adopt_running_identity()
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
