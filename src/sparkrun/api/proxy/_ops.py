"""Console-free proxy/gateway operations.

Implementation behind :mod:`sparkrun.api.proxy`.  Every function here is the
CLI's and the desktop sidecar's single path to the gateway: which gateway is
used, how it starts and stops, and how its served model list is reconciled.

Layering: ``cli -> api.proxy -> sparkrun.proxy -> {core, orchestration}``.
Gateway implementation imports are deferred into the functions —
``sparkrun.proxy.discovery`` imports :mod:`sparkrun.api`, so a module-level
import would be circular.

**Gate placement.** Bringing a gateway *up* (:func:`start`) is gated by the
gateway's feature flag; ``stop`` / ``status`` / ``models`` / ``sync`` /
``alias_*`` are not.  A proxy started while the flag was on must stay
stoppable if the flag is later turned off. Provider-dependent management still
requires a loaded implementation; process recovery cannot reconcile models.
"""

from __future__ import annotations

import logging
import os
import signal
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import SparkrunError
from sparkrun.proxy.contracts import (
    ProxyModel,
    GatewayQueryError,
    GatewayOperationError,
    GatewayConsole,
    GatewayConsoleCredentials,
    GatewayAdminToken,
)

from ._errors import GatewayUnavailable, ProxyAlreadyRunning, ProxyStartFailed, ProxyUnsupported, ProxyUpdateFailed, ProxyQueryFailed
from ._recovery import require_implementation

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.proxy._supervisor import GatewaySupervisor
    from sparkrun.proxy.discovery import DiscoveredEndpoint

logger = logging.getLogger(__name__)

#: How long :func:`start` waits for a superseded proxy to exit before giving up.
RESTART_WAIT_SECONDS = 10.0


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxyEndpoint:
    """A discovered inference endpoint, flattened for callers."""

    host: str
    port: int
    models: tuple[str, ...]
    runtime: str = ""
    cluster_id: str = ""
    healthy: bool = True
    cluster_name: str | None = None
    recipe_revision: str = ""
    native_protocols: tuple[str, ...] = ("openai",)
    capabilities: tuple[str, ...] = ()
    plugin_items: dict[str, Any] = field(default_factory=dict)


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
    #: Non-secret diagnostic when model enumeration failed.  Empty means the
    #: management query succeeded, *including* a legitimately empty model list —
    #: without the distinction an authenticated management failure renders
    #: identically to "no models registered", which is the wrong thing to tell
    #: someone whose models are in fact serving.
    model_query_error: str = ""
    #: False when no state file exists at all (never started / cleaned up).
    known: bool = True
    #: systemd unit supervising the running gateway, when there is one.
    managed_by: str | None = None

    def require_models(self) -> tuple[ProxyModel, ...]:
        """Return observed models, raising instead of treating failed queries as empty."""
        if self.model_query_error:
            raise ProxyQueryFailed("Model list unavailable: %s" % self.model_query_error)
        return self.models

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
        if self.model_query_error:
            data["model_query_error"] = self.model_query_error
        if self.managed_by:
            data["managed_by"] = self.managed_by
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
    #: Named cluster scoping discovery (hosts + SSH user).  Saved to
    #: ``proxy.yaml`` as ``proxy.cluster`` when ``persist`` is set.
    cluster: str | None = None
    #: Remove a saved ``proxy.cluster`` so discovery follows the default cluster.
    clear_cluster: bool = False
    ssh_kwargs: dict | None = None
    auto_discover: bool | None = None
    discover_interval: int | None = None
    discover_removal_grace_sweeps: int | None = None
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
    discover_removal_grace_sweeps: int = 0
    #: Generated configuration file; None for fileless gateways and dry runs.
    config_path: str | None = None
    #: True when a previously-running proxy was stopped to make way for this one.
    restarted: bool = False
    #: ``proxy.yaml`` keys updated this call.
    persisted: tuple[str, ...] = ()
    #: Non-fatal observations (e.g. an obsolete config key).
    warnings: tuple[str, ...] = ()
    #: Which layer decided the discovery scope (see :func:`resolve_discovery_scope`).
    discovery: "DiscoveryScope | None" = None
    #: systemd unit the start was delegated to (no process started here).
    unit: str | None = None


@dataclass(frozen=True)
class DiscoveryScope:
    """Where endpoint discovery looks, and which layer decided it.

    ``source`` is one of ``"--hosts"``, ``"--cluster"``, ``"proxy.yaml"``,
    ``"default cluster"``, ``"default hosts"`` or ``"none"`` (metadata-only).
    """

    source: str
    cluster: str | None = None
    hosts: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.cluster:
            return "cluster %s (from %s)" % (self.cluster, self.source)
        if self.hosts:
            return "hosts %s (from %s)" % (", ".join(self.hosts), self.source)
        return "job metadata only (no cluster or hosts configured)"


@dataclass(frozen=True)
class ProxyStopResult:
    """Outcome of :func:`stop`."""

    stopped: bool
    was_running: bool
    pid: int | None = None
    dry_run: bool = False
    #: systemd unit the stop went through (it stays enabled for the next boot).
    unit: str | None = None


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

    return _list(config=resolve_sctx(sctx).config)


def resolve_gateway(name: str | None = None, *, sctx: "SparkrunContext | None" = None) -> str:
    """Resolve which gateway to use, honoring the ``proxy.gateway`` pin.

    Args:
        name: Explicit override; when ``None`` the pin in ``proxy.yaml`` is
            consulted, then the default.

    Raises:
        GatewayUnavailable: unknown, disabled, or ambiguous.
    """
    from sparkrun.proxy.gateway import GatewayError, resolve_gateway as _resolve

    sctx = resolve_sctx(sctx)
    if not name:
        name = sctx.proxy_config.gateway

    try:
        return _resolve(name, config=sctx.config)
    except GatewayError as exc:
        raise _as_gateway_unavailable(exc) from exc


def _as_gateway_unavailable(exc: Exception) -> GatewayUnavailable:
    """Translate a :mod:`sparkrun.proxy.gateway` error to the api hierarchy."""
    return GatewayUnavailable(
        str(exc),
        gateway=getattr(exc, "gateway", None),
        available=tuple(getattr(exc, "available", ()) or ()),
    )


def _engine_class(gateway: str):
    """Return the engine class implementing *gateway*.

    Delegates to the registry in :mod:`sparkrun.proxy.gateway`, which plugins
    populate — so adding a gateway needs no change here, and an implementation
    may live outside the ``sparkrun.proxy`` tree entirely.
    """
    from sparkrun.proxy.gateway import GatewayError, gateway_class

    try:
        return gateway_class(gateway)
    except GatewayError as exc:
        raise _as_gateway_unavailable(exc) from exc


def _running_engine(sctx: "SparkrunContext | None" = None) -> GatewaySupervisor:
    """Initialize plugins, then bind management to the recorded gateway.

    Selection is ungated and independent of the saved gateway preference.
    Missing plugins or failed bootstrap retain process-level status and stop.
    """
    from sparkrun.core.application_profile import initialize_child_application_profile
    from sparkrun.proxy._supervisor import GatewayState
    from ._recovery import ProcessRecoverySupervisor
    from sparkrun.proxy.gateway import DEFAULT_GATEWAY

    try:
        sctx = resolve_sctx(sctx)
    except SparkrunError as exc:
        # Bootstrap may fail after selecting the application. Confirm its
        # identity without loading plugins before accessing process state; an
        # invalid profile must never fall back to another application's cache.
        try:
            initialize_child_application_profile()
        except Exception as profile_error:
            raise exc from profile_error
        sctx = None
        logger.warning("%s; only process-level gateway management is available.", exc)

    probe = GatewayState()
    state = probe.get_state() or {}
    gateway = str(state.get("gateway") or DEFAULT_GATEWAY)
    if sctx is not None:
        try:
            engine_cls = _engine_class(gateway)
        except GatewayUnavailable:
            logger.warning(
                "No implementation is loaded for the running gateway %r; only process-level management is available.",
                gateway,
            )
        else:
            try:
                return engine_cls(**_gateway_context_kwargs(engine_cls, sctx))
            except GatewayOperationError as exc:
                logger.warning("Gateway %r could not initialize: %s; only process-level management is available.", gateway, exc)

    orphan = ProcessRecoverySupervisor(state_dir=probe.state_dir)
    orphan.gateway_name = gateway
    orphan.host = str(state.get("host") or "")
    orphan.port = int(state.get("port") or 0)
    return orphan


def _gateway_context_kwargs(engine_cls: type, sctx: "SparkrunContext") -> dict[str, Any]:
    """Pass application configuration only to gateways that request it.

    Start and management share this construction contract. Context resolution
    precedes class lookup so installed plugins participate in both paths.
    """
    if not getattr(engine_cls, "wants_proxy_config", False):
        return {}
    return {"proxy_config": sctx.proxy_config, "sctx": sctx}


def _engine_config_kwargs(gateway: str, sctx: "SparkrunContext | None") -> dict[str, Any]:
    """Compatibility with the pinned gateway snapshot's private helper contract.

    Runtime construction uses the resolved class directly; retain this name
    while the vendored integration tests still consume it.
    """
    sctx = resolve_sctx(sctx)
    try:
        engine_cls = _engine_class(gateway)
    except GatewayUnavailable:
        return {}
    return _gateway_context_kwargs(engine_cls, sctx)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def start(options: ProxyStartOptions | None = None, *, sctx: "SparkrunContext | None" = None) -> ProxyStartResult:
    """Discover endpoints, write the gateway config, and start the gateway.

    Raises:
        GatewayUnavailable: the resolved gateway is disabled or unknown.
        ProxyAlreadyRunning: a proxy is up and ``options.restart`` is False.
        ProxyStartFailed: gateway configuration, launch, or shutdown failed.
            The original operational failure is retained as the cause.
    """
    from sparkrun.proxy.gateway import GatewayError

    try:
        return _start(options or ProxyStartOptions(), sctx=resolve_sctx(sctx))
    except GatewayError as exc:
        raise _as_gateway_unavailable(exc) from exc
    except GatewayOperationError as exc:
        raise ProxyStartFailed(str(exc)) from exc


def _start(options: ProxyStartOptions, *, sctx: "SparkrunContext") -> ProxyStartResult:
    """Execute the lifecycle inside the public operational-error boundary."""
    proxy_cfg = sctx.proxy_config

    gateway = resolve_gateway(options.gateway, sctx=sctx)

    effective_port = options.port or proxy_cfg.port
    effective_host = options.host or proxy_cfg.host
    # "Explicitly configured" = supplied now or already persisted. Drives the
    # legacy-0.0.0.0 security warning inside the engine.
    host_configured = options.host is not None or proxy_cfg.host_configured
    effective_key = options.master_key if options.master_key is not None else proxy_cfg.master_key
    removal_grace_sweeps = (
        proxy_cfg.discover_removal_grace_sweeps if options.discover_removal_grace_sweeps is None else options.discover_removal_grace_sweeps
    )
    if removal_grace_sweeps < 1:
        raise ProxyStartFailed("Discovery removal grace must be at least one sweep.")

    warnings: list[str] = []
    if proxy_cfg.enable_ui:
        warnings.append(
            "proxy.enable_ui in proxy.yaml is obsolete and ignored. LiteLLM's /ui requires a "
            "PostgreSQL database and a generated prisma client, neither of which sparkrun "
            "provisions. Remove the key to silence this."
        )

    if options.cluster and options.clear_cluster:
        raise ProxyStartFailed("--cluster and --clear-cluster are mutually exclusive.")
    # An explicit cluster is saved, so a typo must fail here rather than be
    # persisted and then warned about on every later start.
    if options.cluster and _load_cluster(options.cluster, sctx) is None:
        raise ProxyStartFailed("Unknown cluster %r." % options.cluster)

    # Persist explicit overrides before anything can fail, so intent sticks
    # regardless of whether the proxy actually gets (re)started now.
    persisted: tuple[str, ...] = ()
    if options.persist and not options.dry_run:
        persisted = tuple(_persist_overrides(proxy_cfg, options))

    unit = _unit_to_delegate(options)
    if unit is not None:
        return _start_via_unit(unit, options, gateway, effective_host, effective_port, persisted, warnings, sctx)

    scope, host_filter, live_hosts, ssh_kwargs, cluster_def, scope_warnings = _discovery_args(options, sctx)
    warnings.extend(scope_warnings)
    endpoints = _discover(host_filter=host_filter, host_list=live_hosts, ssh_kwargs=ssh_kwargs, cluster_def=cluster_def, sctx=sctx)
    healthy = [ep for ep in endpoints if ep.healthy]

    aliases = proxy_cfg.aliases

    # Build the engine before the config: what a gateway's config *is* — a
    # rendering of discovered endpoints, or a list of desired bindings — is the
    # implementation's business, so config generation hangs off the engine
    # rather than being computed here for one gateway and adapted for the rest.
    engine_kwargs: dict[str, Any] = {
        "host": effective_host,
        "port": effective_port,
        "master_key": effective_key,
        "host_configured": host_configured,
    }
    engine_cls = _engine_class(gateway)
    engine_kwargs.update(_gateway_context_kwargs(engine_cls, sctx))
    engine = engine_cls(**engine_kwargs)

    auto_discover = proxy_cfg.auto_discover if options.auto_discover is None else options.auto_discover
    if auto_discover and not getattr(engine, "supports_autodiscover", True):
        # The gateway's own config owns desired state; sparkrun's daemon has an
        # independent opinion about the same endpoints and the two would fight.
        # Say so rather than silently dropping a configured setting.
        warnings.append("auto-discover is not used with the %s gateway: its own config owns the desired state. Ignoring." % gateway)
        auto_discover = False
    interval = options.discover_interval or proxy_cfg.discover_interval

    # Refuse a foreign state directory before invoking plugin preparation.
    if not options.dry_run:
        engine.claim_state_directory()

    # Validate without changing generated files or gateway snapshots while a
    # previous process may still be serving them. The same preview supplies
    # the alias results for a dry run.
    config_path, applied, pending = engine.prepare_config(healthy, aliases, write=False)

    restarted = False
    if not options.dry_run:
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
        config_path, applied, pending = engine.prepare_config(healthy, aliases, write=True)

    common = {
        "gateway": gateway,
        "host": effective_host,
        "port": effective_port,
        "endpoints": tuple(_to_endpoint(ep) for ep in endpoints),
        "aliases_applied": tuple(sorted(applied)),
        "aliases_pending": tuple(sorted(pending)),
        "auto_discover": auto_discover,
        "discover_interval": interval,
        "discover_removal_grace_sweeps": removal_grace_sweeps,
        "persisted": persisted,
        "warnings": tuple(warnings),
        "discovery": scope,
        "config_path": str(config_path) if config_path is not None and not options.dry_run else None,
    }

    if options.dry_run:
        return ProxyStartResult(started=False, dry_run=True, **common)

    ad_kwargs = None
    if auto_discover:
        ad_kwargs = {
            "interval": interval,
            "removal_grace_sweeps": removal_grace_sweeps,
            "host_list": live_hosts,
            "ssh_kwargs": ssh_kwargs,
            "cluster": scope.cluster,
            "application_config_path": sctx.config.config_path,
        }

    rc = engine.start(config_path=config_path, foreground=options.foreground, autodiscover_kwargs=ad_kwargs)

    if options.foreground:
        # Blocking mode: start() returns the proxy's own exit code.
        return ProxyStartResult(started=True, foreground_rc=_foreground_exit_status(rc), restarted=restarted, **common)

    if rc != 0:
        raise ProxyStartFailed("Gateway %s failed to start (exit code %d)." % (gateway, rc), exit_code=rc)

    return ProxyStartResult(started=True, restarted=restarted, **common)


def stop(*, dry_run: bool = False, sctx: "SparkrunContext | None" = None) -> ProxyStopResult:
    """Stop the running proxy and its auto-discover daemon.

    Ungated on purpose: teardown must work even after the gateway's feature
    flag has been turned off. Declared stop failures raise ProxyUpdateFailed.
    """
    with _gateway_update_errors():
        engine = _running_engine(sctx)
        pid = engine.current_pid()

        if not engine.is_running():
            return ProxyStopResult(stopped=False, was_running=False, pid=pid, dry_run=dry_run)

        # A unit-supervised proxy is stopped through the unit, so systemd
        # records "inactive" instead of a process killed behind its back.  The
        # signal path below remains the fallback: the foreground supervisor
        # reads a requested stop as a clean exit, so the unit does not restart.
        from ._service import ProxyServiceError, control_service, spec_from_record

        unit = spec_from_record((engine.get_state() or {}).get("supervisor"))
        if unit is not None:
            if dry_run:
                return ProxyStopResult(stopped=True, was_running=True, pid=pid, dry_run=True, unit=unit.name)
            try:
                control_service(unit, "stop")
                return ProxyStopResult(stopped=True, was_running=True, pid=pid, unit=unit.name)
            except ProxyServiceError as exc:
                logger.warning("Could not stop %s through systemd (%s); signalling the gateway instead", unit.name, exc)

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

    served_models: tuple[ProxyModel, ...] = ()
    model_query_error = ""
    if running:
        try:
            served_models = engine.query_models()
        except GatewayQueryError as exc:
            model_query_error = str(exc) or "Gateway model query failed"
    return ProxyStatus(
        running=running,
        gateway=str(state.get("gateway") or engine.gateway_name),
        pid=state.get("pid"),
        host=state.get("host"),
        port=state.get("port"),
        started_at=state.get("started_at"),
        autodiscover_pid=ad_pid,
        autodiscover_running=_pid_alive(ad_pid),
        models=served_models,
        model_query_error=model_query_error,
        managed_by=_managed_by(state),
    )


def _managed_by(state: dict) -> str | None:
    record = state.get("supervisor")
    if isinstance(record, dict) and record.get("unit"):
        return str(record["unit"])
    return None


def models(*, sctx: "SparkrunContext | None" = None) -> tuple[ProxyModel, ...]:
    """Return served models (empty when stopped); raise ProxyQueryFailed if unavailable."""
    return status(sctx=sctx).require_models()


@contextmanager
def _gateway_update_errors():
    """Translate declared update failures, retaining causes and programming errors."""
    try:
        yield
    except GatewayOperationError as exc:
        raise ProxyUpdateFailed(str(exc)) from exc


def sync(
    *,
    endpoints: "list[DiscoveredEndpoint] | None" = None,
    aliases: dict[str, str] | None = None,
    host_filter: list[str] | None = None,
    require_running: bool = False,
    sctx: "SparkrunContext | None" = None,
) -> ProxySyncResult:
    """Reconcile the gateway's model list with what is actually running.

    When *endpoints* is ``None`` a discovery sweep is run first.  How the
    change is applied is the gateway's business: LiteLLM rewrites its config
    and restarts, another implementation may update its control plane in
    place.  A steady state costs nothing either way.

    Ungated: this manages an already-running gateway.

    Args:
        require_running: When True, do nothing (and skip discovery) if no
            proxy is running.  Callers that only want to *follow* a live
            proxy — ``proxy load`` / ``unload`` — pass True; the default
            still updates the config so the change lands on the next start.

    Raises:
        ProxyUpdateFailed: the running gateway could not adopt the change.
        GatewayUnavailable: only process recovery is available. Refused before
            discovery; require_running=True remains a no-op when stopped.
    """
    with _gateway_update_errors():
        engine = _running_engine(sctx)
        running = engine.is_running()

        if require_running and not running:
            return ProxySyncResult(proxy_running=False)

        require_implementation(engine)
        if endpoints is None:
            discovered = _discover(host_filter=host_filter, sctx=sctx)
            endpoints = [ep for ep in discovered if ep.healthy]

        added, removed = engine.sync_models(endpoints, aliases)

        return ProxySyncResult(added=added, removed=removed, proxy_running=running)


def register_loaded_model(
    recipe: str,
    *,
    overrides: dict[str, Any] | None = None,
    cluster: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> ProxySyncResult:
    """Register a recipe after ``proxy load`` successfully made it ready.

    A discovery-driven gateway (the engine returns ``None``) simply rescans
    live endpoints, which is byte-identical to calling :func:`sync` directly.
    A catalog-driven gateway persists an activatable binding instead, so the
    same workload can be brought back after it goes cold. A running gateway
    without a loaded implementation raises GatewayUnavailable before discovery.
    """
    with _gateway_update_errors():
        engine = _running_engine(sctx)
        if not engine.is_running():
            return ProxySyncResult(proxy_running=False)
        require_implementation(engine)
        result = engine.register_loaded_model(recipe, overrides, cluster)
        if result is None:
            return sync(require_running=True, sctx=sctx)
        added, removed = result
        return ProxySyncResult(added=added, removed=removed, proxy_running=True)


def unregister_loaded_model(
    recipe: str,
    *,
    sctx: "SparkrunContext | None" = None,
) -> ProxySyncResult:
    """Remove a recipe after ``proxy unload`` stopped its workload.

    ``None`` from the engine has the same discovery-driven meaning as in
    :func:`register_loaded_model`, including GatewayUnavailable during recovery.
    """
    with _gateway_update_errors():
        engine = _running_engine(sctx)
        if not engine.is_running():
            return ProxySyncResult(proxy_running=False)
        require_implementation(engine)
        result = engine.unregister_loaded_model(recipe)
        if result is None:
            return sync(require_running=True, sctx=sctx)
        added, removed = result
        return ProxySyncResult(added=added, removed=removed, proxy_running=True)


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
        GatewayUnavailable: the alias was saved, but the running provider is
            unavailable; restore it before synchronizing the saved settings.
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

    ``saved=False`` in the result means the alias did not exist. As with
    add_alias, ProxyUpdateFailed/GatewayUnavailable means the edit was saved
    but could not be applied to the running provider.
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
    with _gateway_update_errors():
        engine = _running_engine(sctx)
        if not engine.is_running():
            return 0, 0, False

        require_implementation(engine)
        added, removed = engine.sync_aliases(aliases)
        return added, removed, True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _discover(
    *,
    host_filter: list[str] | None = None,
    host_list: list[str] | None = None,
    ssh_kwargs: dict | None = None,
    cluster_def: "ClusterDefinition | None" = None,
    sctx: "SparkrunContext | None" = None,
) -> "list[DiscoveredEndpoint]":
    """Run one endpoint-discovery sweep (deferred import: circular otherwise)."""
    from sparkrun.proxy.discovery import discover_endpoints

    return discover_endpoints(host_filter=host_filter, host_list=host_list, ssh_kwargs=ssh_kwargs, cluster_def=cluster_def, sctx=sctx)


def _load_cluster(name: str, sctx: "SparkrunContext") -> "ClusterDefinition | None":
    """Load a named cluster, or ``None`` when it does not exist or cannot be read."""
    try:
        return sctx.cluster_manager.get(name)
    except Exception:
        logger.debug("Could not load cluster %r", name, exc_info=True)
        return None


def resolve_discovery_scope(
    options: ProxyStartOptions, sctx: "SparkrunContext"
) -> "tuple[DiscoveryScope, ClusterDefinition | None, list[str]]":
    """Decide where discovery looks.  First hit wins:

    1. ``--hosts`` / ``--hosts-file`` (this run only; never saved);
    2. ``--cluster`` (saved as ``proxy.cluster``);
    3. ``proxy.cluster`` from ``proxy.yaml`` (skipped by ``--clear-cluster``);
    4. the default cluster (``sparkrun cluster set-default``), resolved now so
       a later change is followed rather than frozen;
    5. ``config.default_hosts``;
    6. nothing: metadata-only discovery.

    The default cluster used to be skipped entirely, so a user whose only host
    source was ``cluster set-default`` got metadata-only discovery as the
    control node's own login.  A chosen cluster contributes its whole
    definition (SSH user, executor, transport), not just its hosts.

    Returns:
        ``(scope, cluster_def, warnings)``.  A saved ``proxy.cluster`` that no
        longer exists is a warning, not an error: a proxy starting at boot
        should come up, and discovery is best-effort anyway.
    """
    warnings: list[str] = []
    proxy_cfg = sctx.proxy_config
    named = options.cluster
    if options.host_filter:
        cluster_def = _load_cluster(named, sctx) if named else None
        return DiscoveryScope("--hosts", hosts=tuple(options.host_filter)), cluster_def, warnings

    candidates: list[tuple[str, str]] = []
    if named:
        candidates.append((named, "--cluster"))
    elif proxy_cfg.cluster and not options.clear_cluster:
        candidates.append((proxy_cfg.cluster, "proxy.yaml"))
    for name, source in candidates:
        cluster_def = _load_cluster(name, sctx)
        if cluster_def is not None:
            return DiscoveryScope(source, cluster=name, hosts=tuple(cluster_def.hosts or ())), cluster_def, warnings
        warnings.append(
            "proxy.yaml names discovery cluster %r, which no longer exists; using the default instead "
            "(run 'proxy start --cluster <name>' or '--clear-cluster' to fix)." % name
        )

    from sparkrun.api._resolve import _default_cluster

    default = _default_cluster(sctx, None)
    if default is not None and default.name:
        return DiscoveryScope("default cluster", cluster=default.name, hosts=tuple(default.hosts or ())), default, warnings

    default_hosts = list(sctx.config.default_hosts or ())
    if default_hosts:
        return DiscoveryScope("default hosts", hosts=tuple(default_hosts)), None, warnings
    return DiscoveryScope("none"), None, warnings


def _discovery_args(options: ProxyStartOptions, sctx: "SparkrunContext"):
    """Resolve the scope plus what :func:`_discover` needs to act on it.

    Returns ``(scope, host_filter, live_hosts, ssh_kwargs, cluster_def,
    warnings)``.  ``live_hosts is None`` means metadata-only discovery.  An
    explicit host list or cluster also filters endpoints to those hosts, as
    ``--cluster`` always has; the ``default hosts`` fallback keeps its old
    liveness-only role.
    """
    scope, cluster_def, warnings = resolve_discovery_scope(options, sctx)
    live_hosts = list(scope.hosts) or None
    host_filter = list(scope.hosts) if scope.source not in ("default hosts", "none") else None
    if live_hosts is None:
        return scope, host_filter, None, None, None, warnings
    if options.ssh_kwargs is not None:
        return scope, host_filter, live_hosts, options.ssh_kwargs, cluster_def, warnings
    try:
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        ssh_kwargs = build_ssh_kwargs(sctx.config)
    except Exception:
        logger.debug("Could not build SSH settings; falling back to metadata-only", exc_info=True)
        return scope, host_filter, None, None, None, warnings
    # Apply the cluster's SSH user without mutating the shared config
    # (sctx.config is reused by every other call in this session).
    cluster_user = getattr(cluster_def, "user", None) if cluster_def is not None else None
    if cluster_user:
        ssh_kwargs = dict(ssh_kwargs, ssh_user=cluster_user)
    return scope, host_filter, live_hosts, ssh_kwargs, cluster_def, warnings


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
        (
            "discover_removal_grace_sweeps",
            options.discover_removal_grace_sweeps,
            proxy_cfg.discover_removal_grace_sweeps,
        ),
        ("discover_interval", options.discover_interval, proxy_cfg.discover_interval),
        ("gateway", options.gateway, proxy_cfg.gateway),
        ("cluster", options.cluster, proxy_cfg.cluster),
    ]

    updates: dict[str, object] = {}
    changed: list[str] = []
    for key, supplied, current in candidates:
        if supplied is None or supplied == current:
            continue
        updates[key] = supplied
        changed.append(key)

    if options.clear_cluster and proxy_cfg.cluster is not None:
        proxy_cfg.unset_proxy("cluster")
        changed.append("cluster (cleared)")

    if updates or options.clear_cluster:
        if updates:
            proxy_cfg.set_proxy(**updates)
        proxy_cfg.save()

    return changed


def _unit_to_delegate(options: ProxyStartOptions):
    """The caller's installed proxy unit, when this start should go through it.

    Never for ``--foreground`` (that is what the unit itself runs, and an
    explicit request to run here) or inside a supervised process.
    """
    from sparkrun.proxy._supervisor import SUPERVISOR_ENV

    from ._service import ProxyServiceError, find_service

    if options.foreground or os.environ.get(SUPERVISOR_ENV):
        return None
    try:
        return find_service()
    except ProxyServiceError as exc:
        raise ProxyStartFailed(str(exc)) from exc


def _start_via_unit(unit, options, gateway, host, port, persisted, warnings, sctx) -> ProxyStartResult:
    """Start (or restart) the installed unit instead of an ad-hoc process.

    An ad-hoc proxy would collide with the unit at the next boot.  Settings
    supplied this run were already saved to ``proxy.yaml``, which is what the
    unit reads when it starts.
    """
    from sparkrun.proxy import _systemd

    from ._service import ProxyServiceError, control_service

    if options.host_filter:
        warnings.append("--hosts/--hosts-file apply to one invocation and are ignored by systemd unit %s; use --cluster." % unit.name)
    active = _systemd.query(unit, "is-active") == "active"
    if active and not options.restart:
        raise ProxyAlreadyRunning(
            "Proxy is running as systemd unit %s." % unit.name,
            pid=_running_engine(sctx).current_pid(),
            port=port,
            persisted=persisted,
        )
    if not options.dry_run:
        try:
            control_service(unit, "restart" if active else "start")
        except ProxyServiceError as exc:
            raise ProxyStartFailed(str(exc)) from exc
    return ProxyStartResult(
        gateway=gateway,
        host=host,
        port=port,
        started=not options.dry_run,
        dry_run=options.dry_run,
        restarted=active,
        persisted=persisted,
        warnings=tuple(warnings),
        unit=unit.name,
    )


def _foreground_exit_status(rc: int | None) -> int:
    """Exit status for a foreground gateway that has returned.

    ``ProxyEngine`` already supervises and returns a status. A gateway whose
    foreground path still returns the raw ``Popen.returncode`` reports a
    ``proxy stop`` as ``-SIGTERM``; that is a requested stop, not a failure a
    service manager should answer with a restart. A crash is never SIGTERM
    (Python errors exit 1, the OOM killer sends SIGKILL, faults SIGSEGV).
    """
    from sparkrun.proxy._supervisor import exit_status

    if rc == -signal.SIGTERM:
        return 0
    return exit_status(rc)


def _stop_and_wait(engine) -> bool:
    """Wait for the original process, independently of state-file cleanup."""
    pid = engine.current_pid()
    engine.stop()
    if pid is None:
        return True
    # stop() can remove state before a draining process releases listeners and
    # SQLite locks. is_running() would then mistake a missing record for exit.
    if not engine._await_exit(pid, RESTART_WAIT_SECONDS):
        return False
    # An asynchronous stop retains state until exit; do not clear a concurrent
    # replacement's record if another command has already written a new PID.
    if engine.current_pid() == pid:
        engine._clear_state()
    return True


def _to_endpoint(ep: "DiscoveredEndpoint") -> ProxyEndpoint:
    """Flatten a discovery record into the api's endpoint shape."""
    return ProxyEndpoint(
        host=ep.host,
        port=ep.port,
        models=tuple(ep.actual_models or ([ep.model] if ep.model else ())),
        runtime=ep.runtime,
        cluster_id=ep.cluster_id,
        healthy=ep.healthy,
        cluster_name=getattr(ep, "cluster_name", None),
        recipe_revision=getattr(ep, "recipe_revision", ""),
        native_protocols=tuple(getattr(ep, "native_protocols", None) or ("openai",)),
        capabilities=tuple(getattr(ep, "capabilities", None) or ()),
        plugin_items=deepcopy(getattr(ep, "plugin_items", None) or {}),
    )


def _pid_alive(pid: int | None) -> bool:
    """True when *pid* names a live process we may signal."""
    if not pid:
        return False
    from sparkrun.utils.process import process_exists

    return process_exists(pid)


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
    "ui",
    "admin_token",
    "ProxyUiResult",
    "list_aliases",
    "list_gateways",
    "models",
    "register_loaded_model",
    "remove_alias",
    "resolve_gateway",
    "start",
    "status",
    "stop",
    "sync",
    "unregister_loaded_model",
]


@dataclass(frozen=True)
class ProxyUiResult:
    """Where the gateway's admin console is, and how to get into it."""

    url: str
    running: bool
    #: Console credential when requested; the provider may issue or reuse it.
    token: str | None = None
    #: Address the console's listener is bound to.  Differs from the host in
    #: :attr:`url` for a wildcard bind, which is not connectable as written.
    bind_host: str = ""
    #: True when the console is reachable from off this machine.
    exposed: bool = False
    auth_required: bool = True


def ui(*, issue_token: bool = False, sctx: "SparkrunContext | None" = None) -> ProxyUiResult:
    """Locate the running gateway's admin console, optionally minting access.

    Ungated, like every other management path: a console started while the
    flag was on stays reachable.

    Args:
        issue_token: Ask the gateway to create or return console credentials.
            This can enable authentication. Kept for CLI compatibility;
            ``admin_token`` is the explicit read/rotate/clear API.

    Raises:
        ProxyUnsupported: the running gateway serves no admin console.
    """
    with _gateway_update_errors():
        engine = _running_engine(sctx)
        if not isinstance(engine, GatewayConsole):
            raise ProxyUnsupported("The %s gateway does not serve an admin console." % engine.gateway_name)
        url = engine.ui_url
        if not url:
            raise ProxyUnsupported("The %s gateway does not serve an admin console." % engine.gateway_name)
        token = None
        if issue_token:
            if not isinstance(engine, GatewayConsoleCredentials):
                raise ProxyUnsupported("The %s gateway cannot issue console credentials." % engine.gateway_name)
            token = engine.issue_ui_credential()
        return ProxyUiResult(
            url=str(url),
            running=engine.is_running(),
            token=token,
            bind_host=engine.admin_bind_host,
            exposed=engine.admin_exposed,
            auth_required=engine.admin_auth_required,
        )


def admin_token(*, rotate: bool = False, clear: bool = False, sctx: "SparkrunContext | None" = None) -> str | None:
    """Return, rotate, or clear the managed gateway's live admin token.

    None means admin authentication is currently open. Rotation generates a
    high-entropy bearer token and takes effect immediately; clearing removes
    that requirement immediately.
    """
    if rotate and clear:
        raise ValueError("rotate and clear are mutually exclusive")
    with _gateway_update_errors():
        engine = _running_engine(sctx)
        if not isinstance(engine, GatewayAdminToken):
            raise ProxyUnsupported("The %s gateway has no managed admin token." % getattr(engine, "gateway_name", "configured"))
        return engine.admin_token(rotate=rotate, clear=clear)
