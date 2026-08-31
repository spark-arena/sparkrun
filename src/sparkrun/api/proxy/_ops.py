"""Console-free proxy/gateway operations.

Implementation behind :mod:`sparkrun.api.proxy`.  Every function here is the
CLI's and the desktop sidecar's single path to the gateway: which gateway is
used, how it starts and stops, and how its served model list is reconciled.

Layering: ``cli -> api.proxy -> sparkrun.proxy -> {core, orchestration}``.
Imports of :mod:`sparkrun.proxy` are deferred into the functions —
``sparkrun.proxy.discovery`` imports :mod:`sparkrun.api`, so a module-level
import would be circular.

**Gate placement.** Bringing a gateway *up* (:func:`start`) is gated by the
gateway's feature flag; ``stop`` / ``status`` / ``models`` / ``sync`` /
``alias_*`` are not.  A proxy started while the flag was on must stay
manageable — and stoppable — if the flag is later turned off, and the
auto-discover daemon keeps driving the engine it was started with.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx

from ._errors import GatewayUnavailable, ProxyAlreadyRunning, ProxyStartFailed, ProxyUpdateFailed

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext
    from sparkrun.proxy.discovery import DiscoveredEndpoint

logger = logging.getLogger(__name__)

#: How long :func:`start` waits for a superseded proxy to exit before giving up.
RESTART_WAIT_SECONDS = 10.0


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxyModel:
    """One model entry served by the gateway."""

    model_name: str
    api_base: str = ""
    #: Max context length (tokens) of the served model, when known (from the
    #: backend's ``max_model_len``). ``None`` when unavailable.
    max_model_len: int | None = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {"model_name": self.model_name, "api_base": self.api_base or "?"}
        if self.max_model_len is not None:
            d["max_model_len"] = self.max_model_len
        return d


@dataclass(frozen=True)
class ProxyEndpoint:
    """A discovered inference endpoint, flattened for callers."""

    host: str
    port: int
    models: tuple[str, ...]
    runtime: str = ""
    cluster_id: str = ""
    healthy: bool = True


@dataclass(frozen=True)
class ProxyStatus:
    """Snapshot of the gateway process and what it is serving."""

    running: bool
    gateway: str | None = None
    pid: int | None = None
    host: str | None = None
    port: int | None = None
    started_at: str | None = None
    autodiscover_pid: int | None = None
    autodiscover_running: bool = False
    models: tuple[ProxyModel, ...] = ()
    #: False when no state file exists at all (never started / cleaned up).
    known: bool = True

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "running": self.running,
            "gateway": self.gateway,
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "started_at": self.started_at,
            "models": [m.to_dict() for m in self.models],
        }
        if self.autodiscover_pid is not None:
            data["autodiscover"] = {"pid": self.autodiscover_pid, "running": self.autodiscover_running}
        return data


@dataclass(frozen=True)
class ProxyStartOptions:
    """Inputs for :func:`start`.

    Every ``None`` means "not supplied" — the persisted ``proxy.yaml`` value
    (or its default) is used, and nothing is written back for that key.
    """

    gateway: str | None = None
    port: int | None = None
    host: str | None = None
    master_key: str | None = None
    #: Restrict discovery to these hosts (already parsed; no CLI syntax here).
    host_filter: list[str] | None = None
    #: Named cluster, used for the SSH user during live discovery.
    cluster: str | None = None
    ssh_kwargs: dict | None = None
    auto_discover: bool | None = None
    discover_interval: int | None = None
    foreground: bool = False
    #: Replace a running proxy instead of raising :class:`ProxyAlreadyRunning`.
    restart: bool = False
    dry_run: bool = False
    #: Persist explicitly-supplied settings to ``proxy.yaml``.
    persist: bool = True


@dataclass(frozen=True)
class ProxyStartResult:
    """Outcome of :func:`start`."""

    gateway: str
    host: str
    port: int
    started: bool
    dry_run: bool = False
    foreground_rc: int | None = None
    endpoints: tuple[ProxyEndpoint, ...] = ()
    #: Aliases that resolved to a live backend, and those still waiting.
    aliases_applied: tuple[str, ...] = ()
    aliases_pending: tuple[str, ...] = ()
    auto_discover: bool = False
    discover_interval: int = 0
    config_path: str | None = None
    #: True when a previously-running proxy was stopped to make way for this one.
    restarted: bool = False
    #: ``proxy.yaml`` keys updated this call.
    persisted: tuple[str, ...] = ()
    #: Non-fatal observations (e.g. an obsolete config key).
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProxyStopResult:
    """Outcome of :func:`stop`."""

    stopped: bool
    was_running: bool
    pid: int | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class ProxySyncResult:
    """Model-list reconciliation outcome."""

    added: int = 0
    removed: int = 0
    #: False when no proxy was running (the config was still updated).
    proxy_running: bool = True

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


@dataclass(frozen=True)
class ProxyAliasResult:
    """Outcome of an alias mutation."""

    alias: str
    target: str | None = None
    #: False for a removal of an alias that did not exist.
    saved: bool = True
    applied: int = 0
    removed: int = 0
    proxy_running: bool = False
    aliases: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Gateway resolution / engine construction
# --------------------------------------------------------------------------


def list_gateways(*, sctx: "SparkrunContext | None" = None) -> list[str]:
    """Return the gateway names whose feature flag resolves on."""
    from sparkrun.proxy.gateway import list_gateways as _list

    return _list(config=_config(sctx))


def resolve_gateway(name: str | None = None, *, sctx: "SparkrunContext | None" = None) -> str:
    """Resolve which gateway to use, honoring the ``proxy.gateway`` pin.

    Args:
        name: Explicit override; when ``None`` the pin in ``proxy.yaml`` is
            consulted, then the default.

    Raises:
        GatewayUnavailable: unknown, disabled, or ambiguous.
    """
    from sparkrun.proxy.gateway import GatewayError, resolve_gateway as _resolve

    if not name and sctx is not None:
        name = sctx.proxy_config.gateway

    try:
        return _resolve(name, config=_config(sctx))
    except GatewayError as exc:
        raise _as_gateway_unavailable(exc) from exc


def _config(sctx: "SparkrunContext | None") -> "SparkrunConfig | None":
    """The SparkrunConfig backing feature-flag resolution, when we have one."""
    return sctx.config if sctx is not None else None


def _as_gateway_unavailable(exc: Exception) -> GatewayUnavailable:
    """Translate a :mod:`sparkrun.proxy.gateway` error to the api hierarchy."""
    return GatewayUnavailable(
        str(exc),
        gateway=getattr(exc, "gateway", None),
        available=tuple(getattr(exc, "available", ()) or ()),
    )


def _engine_class(gateway: str):
    """Return the engine class implementing *gateway*.

    The one place a gateway name becomes an implementation.  A second gateway
    becomes another branch here (and later, a plugin lookup) — no caller
    signature changes.
    """
    if gateway == "litellm":
        from sparkrun.proxy.engine import ProxyEngine

        return ProxyEngine
    raise GatewayUnavailable("No implementation registered for gateway %r" % gateway, gateway=gateway)


def _new_engine(gateway: str, **kwargs):
    """Construct the engine for *gateway* (does not gate; ``start`` does)."""
    return _engine_class(gateway)(**kwargs)


def _running_engine(sctx: "SparkrunContext | None" = None):
    """Engine bound to the proxy recorded in the state file.

    Deliberately ungated and independent of the *configured* gateway: it
    reflects what is actually running, so a proxy stays manageable after its
    flag is disabled or the configured gateway is changed underneath it.
    """
    from sparkrun.proxy.gateway import DEFAULT_GATEWAY

    probe = _new_engine(DEFAULT_GATEWAY)
    state = probe.get_state() or {}
    gateway = str(state.get("gateway") or DEFAULT_GATEWAY)
    if gateway == DEFAULT_GATEWAY:
        return probe
    return _new_engine(gateway)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def start(options: ProxyStartOptions | None = None, *, sctx: "SparkrunContext | None" = None) -> ProxyStartResult:
    """Discover endpoints, write the gateway config, and start the gateway.

    Raises:
        GatewayUnavailable: the resolved gateway is disabled or unknown.
        ProxyAlreadyRunning: a proxy is up and ``options.restart`` is False.
        ProxyStartFailed: the gateway process did not come up (or the
            superseded one refused to exit).
    """
    from sparkrun.proxy.engine import build_litellm_config, write_config
    from sparkrun.proxy.gateway import GatewayError

    options = options or ProxyStartOptions()
    sctx = resolve_sctx(sctx)
    proxy_cfg = sctx.proxy_config

    gateway = resolve_gateway(options.gateway, sctx=sctx)

    effective_port = options.port or proxy_cfg.port
    effective_host = options.host or proxy_cfg.host
    # "Explicitly configured" = supplied now or already persisted. Drives the
    # legacy-0.0.0.0 security warning inside the engine.
    host_configured = options.host is not None or proxy_cfg.host_configured
    effective_key = options.master_key if options.master_key is not None else proxy_cfg.master_key

    warnings: list[str] = []
    if proxy_cfg.enable_ui:
        warnings.append(
            "proxy.enable_ui in proxy.yaml is obsolete and ignored. LiteLLM's /ui requires a "
            "PostgreSQL database and a generated prisma client, neither of which sparkrun "
            "provisions. Remove the key to silence this."
        )

    # Persist explicit overrides before anything can fail, so intent sticks
    # regardless of whether the proxy actually gets (re)started now.
    persisted: tuple[str, ...] = ()
    if options.persist and not options.dry_run:
        persisted = tuple(_persist_overrides(proxy_cfg, options))

    live_hosts, ssh_kwargs = _discovery_args(options, sctx)
    endpoints = _discover(host_filter=options.host_filter, host_list=live_hosts, ssh_kwargs=ssh_kwargs, sctx=sctx)
    healthy = [ep for ep in endpoints if ep.healthy]

    aliases = proxy_cfg.aliases
    config_dict = build_litellm_config(healthy, effective_key, aliases=aliases)
    applied = {e["model_name"] for e in config_dict["model_list"]} & set(aliases)
    pending = set(aliases) - applied

    auto_discover = proxy_cfg.auto_discover if options.auto_discover is None else options.auto_discover
    interval = options.discover_interval or proxy_cfg.discover_interval

    common = {
        "gateway": gateway,
        "host": effective_host,
        "port": effective_port,
        "endpoints": tuple(_to_endpoint(ep) for ep in endpoints),
        "aliases_applied": tuple(sorted(applied)),
        "aliases_pending": tuple(sorted(pending)),
        "auto_discover": auto_discover,
        "discover_interval": interval,
        "persisted": persisted,
        "warnings": tuple(warnings),
    }

    if options.dry_run:
        return ProxyStartResult(started=False, dry_run=True, **common)

    config_path = write_config(config_dict)

    engine = _new_engine(
        gateway,
        host=effective_host,
        port=effective_port,
        master_key=effective_key,
        host_configured=host_configured,
    )

    restarted = False
    if engine.is_running():
        pid = engine.current_pid()
        if not options.restart:
            raise ProxyAlreadyRunning(
                "Proxy is already running (PID %s) on port %d." % (pid, engine.port),
                pid=pid,
                port=engine.port,
                persisted=persisted,
            )
        if not _stop_and_wait(engine):
            raise ProxyStartFailed("Proxy did not stop cleanly within %.0fs; aborting restart." % RESTART_WAIT_SECONDS)
        restarted = True

    ad_kwargs = None
    if auto_discover:
        ad_kwargs = {"interval": interval, "host_list": live_hosts, "ssh_kwargs": ssh_kwargs}

    try:
        rc = engine.start(config_path=config_path, foreground=options.foreground, autodiscover_kwargs=ad_kwargs)
    except GatewayError as exc:  # engine-level backstop for the same gate
        raise _as_gateway_unavailable(exc) from exc

    if options.foreground:
        # Blocking mode: start() returns the proxy's own exit code.
        return ProxyStartResult(started=True, foreground_rc=rc, restarted=restarted, config_path=str(config_path), **common)

    if rc != 0:
        raise ProxyStartFailed("Gateway %s failed to start (exit code %d)." % (gateway, rc), exit_code=rc)

    return ProxyStartResult(started=True, restarted=restarted, config_path=str(config_path), **common)


def stop(*, dry_run: bool = False, sctx: "SparkrunContext | None" = None) -> ProxyStopResult:
    """Stop the running proxy and its auto-discover daemon.

    Ungated on purpose: teardown must work even after the gateway's feature
    flag has been turned off.
    """
    engine = _running_engine(sctx)
    pid = engine.current_pid()

    if not engine.is_running():
        return ProxyStopResult(stopped=False, was_running=False, pid=pid, dry_run=dry_run)

    stopped = engine.stop(dry_run=dry_run)
    return ProxyStopResult(stopped=bool(stopped), was_running=True, pid=pid, dry_run=dry_run)


def status(*, sctx: "SparkrunContext | None" = None) -> ProxyStatus:
    """Report gateway process state and the models it currently serves."""
    engine = _running_engine(sctx)
    state = engine.get_state()

    if not state:
        return ProxyStatus(running=False, known=False)

    running = engine.is_running()

    ad_pid_raw = state.get("autodiscover_pid")
    ad_pid: int | None
    try:
        ad_pid = int(ad_pid_raw) if ad_pid_raw else None
    except (TypeError, ValueError):
        ad_pid = None

    return ProxyStatus(
        running=running,
        gateway=str(state.get("gateway") or engine.gateway_name),
        pid=state.get("pid"),
        host=state.get("host"),
        port=state.get("port"),
        started_at=state.get("started_at"),
        autodiscover_pid=ad_pid,
        autodiscover_running=_pid_alive(ad_pid),
        models=_models_via_api(engine) if running else (),
    )


def models(*, sctx: "SparkrunContext | None" = None) -> tuple[ProxyModel, ...]:
    """Return the models the running proxy reports (empty when stopped)."""
    engine = _running_engine(sctx)
    if not engine.is_running():
        return ()
    return _models_via_api(engine)


def sync(
    *,
    endpoints: "list[DiscoveredEndpoint] | None" = None,
    aliases: dict[str, str] | None = None,
    host_filter: list[str] | None = None,
    require_running: bool = False,
    sctx: "SparkrunContext | None" = None,
) -> ProxySyncResult:
    """Reconcile the gateway's model list with what is actually running.

    Rewrites the gateway config and restarts the proxy when the desired model
    set differs; a steady state costs nothing.  When *endpoints* is ``None``
    a discovery sweep is run first.

    Ungated: this manages an already-running gateway.

    Args:
        require_running: When True, do nothing (and skip discovery) if no
            proxy is running.  Callers that only want to *follow* a live
            proxy — ``proxy load`` / ``unload`` — pass True; the default
            still updates the config so the change lands on the next start.

    Raises:
        ProxyUpdateFailed: the config was rewritten but the proxy could not
            be replaced.
    """
    from sparkrun.proxy.engine import ProxyRestartError

    engine = _running_engine(sctx)
    running = engine.is_running()

    if require_running and not running:
        return ProxySyncResult(proxy_running=False)

    if endpoints is None:
        discovered = _discover(host_filter=host_filter, sctx=sctx)
        endpoints = [ep for ep in discovered if ep.healthy]

    try:
        added, removed = engine.sync_models(endpoints, aliases)
    except ProxyRestartError as exc:
        raise ProxyUpdateFailed(str(exc)) from exc

    return ProxySyncResult(added=added, removed=removed, proxy_running=running)


# --------------------------------------------------------------------------
# Aliases
# --------------------------------------------------------------------------


def list_aliases(*, sctx: "SparkrunContext | None" = None) -> dict[str, str]:
    """Return the configured ``alias -> target model`` mapping."""
    return resolve_sctx(sctx).proxy_config.aliases


def add_alias(alias: str, target: str, *, sctx: "SparkrunContext | None" = None) -> ProxyAliasResult:
    """Add (or update) an alias and apply it to a running proxy.

    Raises:
        ProxyUpdateFailed: the alias was saved but the running proxy could
            not be updated.
    """
    sctx = resolve_sctx(sctx)
    proxy_cfg = sctx.proxy_config
    proxy_cfg.add_alias(alias, target)
    proxy_cfg.save()

    applied, removed, running = _apply_aliases(proxy_cfg.aliases, sctx)
    return ProxyAliasResult(
        alias=alias,
        target=target,
        saved=True,
        applied=applied,
        removed=removed,
        proxy_running=running,
        aliases=proxy_cfg.aliases,
    )


def remove_alias(alias: str, *, sctx: "SparkrunContext | None" = None) -> ProxyAliasResult:
    """Remove an alias and drop it from a running proxy.

    ``saved=False`` in the result means the alias did not exist.
    """
    sctx = resolve_sctx(sctx)
    proxy_cfg = sctx.proxy_config

    if not proxy_cfg.remove_alias(alias):
        return ProxyAliasResult(alias=alias, saved=False, aliases=proxy_cfg.aliases)

    proxy_cfg.save()
    applied, removed, running = _apply_aliases(proxy_cfg.aliases, sctx)
    return ProxyAliasResult(
        alias=alias,
        saved=True,
        applied=applied,
        removed=removed,
        proxy_running=running,
        aliases=proxy_cfg.aliases,
    )


def _apply_aliases(aliases: dict[str, str], sctx: "SparkrunContext") -> tuple[int, int, bool]:
    """Push *aliases* to the running proxy. Returns (added, removed, running)."""
    from sparkrun.proxy.engine import ProxyRestartError

    engine = _running_engine(sctx)
    if not engine.is_running():
        return 0, 0, False

    try:
        added, removed = engine.sync_aliases(aliases)
    except ProxyRestartError as exc:
        raise ProxyUpdateFailed(str(exc)) from exc
    return added, removed, True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _discover(
    *,
    host_filter: list[str] | None = None,
    host_list: list[str] | None = None,
    ssh_kwargs: dict | None = None,
    sctx: "SparkrunContext | None" = None,
) -> "list[DiscoveredEndpoint]":
    """Run one endpoint-discovery sweep (deferred import: circular otherwise)."""
    from sparkrun.proxy.discovery import discover_endpoints

    return discover_endpoints(host_filter=host_filter, host_list=host_list, ssh_kwargs=ssh_kwargs, sctx=sctx)


def _discovery_args(options: ProxyStartOptions, sctx: "SparkrunContext") -> tuple[list[str] | None, dict | None]:
    """Resolve (hosts, ssh_kwargs) for live container discovery.

    ``(None, None)`` means "no host context" — discovery falls back to
    metadata-only mode.  Best-effort: a config/cluster read that fails
    degrades to metadata-only rather than failing the start.
    """
    if options.ssh_kwargs is not None:
        return (options.host_filter or None), options.ssh_kwargs

    try:
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        config = sctx.config
        live_hosts = options.host_filter or config.default_hosts or None
        if not live_hosts:
            return None, None

        ssh_kwargs = build_ssh_kwargs(config)

        # Apply the cluster's SSH user without mutating the shared config
        # (sctx.config is reused by every other call in this session).
        if options.cluster:
            cluster_def = sctx.cluster_manager.get(options.cluster)
            cluster_user = getattr(cluster_def, "user", None) if cluster_def else None
            if cluster_user:
                ssh_kwargs = dict(ssh_kwargs, ssh_user=cluster_user)

        return list(live_hosts), ssh_kwargs
    except Exception:
        logger.debug("Could not resolve live discovery args; falling back to metadata-only", exc_info=True)
        return None, None


def _persist_overrides(proxy_cfg, options: ProxyStartOptions) -> list[str]:
    """Write explicitly-supplied settings to ``proxy.yaml``.

    Only keys whose supplied value differs from the saved one are written, so
    a no-op invocation does not touch the file.

    Returns:
        The key names that were updated.
    """
    candidates: list[tuple[str, object, object]] = [
        ("port", options.port, proxy_cfg.port),
        ("host", options.host, proxy_cfg.host),
        ("master_key", options.master_key, proxy_cfg.master_key),
        ("discover_interval", options.discover_interval, proxy_cfg.discover_interval),
        ("gateway", options.gateway, proxy_cfg.gateway),
    ]

    updates: dict[str, object] = {}
    changed: list[str] = []
    for key, supplied, current in candidates:
        if supplied is None or supplied == current:
            continue
        updates[key] = supplied
        changed.append(key)

    if updates:
        proxy_cfg.set_proxy(**updates)
        proxy_cfg.save()

    return changed


def _stop_and_wait(engine) -> bool:
    """Stop *engine* and poll until the process is really gone."""
    import time

    engine.stop()

    waited = 0.0
    interval = 0.5
    while engine.is_running() and waited < RESTART_WAIT_SECONDS:
        time.sleep(interval)
        waited += interval

    return not engine.is_running()


def _models_via_api(engine) -> tuple[ProxyModel, ...]:
    """Normalize the management API's model rows into :class:`ProxyModel`."""
    out: list[ProxyModel] = []
    for m in engine.list_models_via_api():
        params = m.get("litellm_params") or m.get("model_info", {}).get("litellm_params", {})
        info = m.get("model_info") or {}
        mml = info.get("max_input_tokens") or info.get("max_tokens") or info.get("max_model_len")
        out.append(
            ProxyModel(
                model_name=m.get("model_name", "?"),
                api_base=params.get("api_base", ""),
                max_model_len=mml if isinstance(mml, int) else None,
            )
        )
    return tuple(out)


def _to_endpoint(ep: "DiscoveredEndpoint") -> ProxyEndpoint:
    """Flatten a discovery record into the api's endpoint shape."""
    return ProxyEndpoint(
        host=ep.host,
        port=ep.port,
        models=tuple(ep.actual_models or ([ep.model] if ep.model else ())),
        runtime=ep.runtime,
        cluster_id=ep.cluster_id,
        healthy=ep.healthy,
    )


def _pid_alive(pid: int | None) -> bool:
    """True when *pid* names a live process we may signal."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError):
        return False


__all__ = [
    "ProxyAliasResult",
    "ProxyEndpoint",
    "ProxyModel",
    "ProxyStartOptions",
    "ProxyStartResult",
    "ProxyStatus",
    "ProxyStopResult",
    "ProxySyncResult",
    "add_alias",
    "list_aliases",
    "list_gateways",
    "models",
    "remove_alias",
    "resolve_gateway",
    "start",
    "status",
    "stop",
    "sync",
]
