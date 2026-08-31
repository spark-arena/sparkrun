"""Gateway selection — which implementation backs ``sparkrun proxy``.

A *gateway* is the process that fronts every discovered inference endpoint
behind one OpenAI-compatible API.  Core ships one implementation,
:class:`~sparkrun.proxy.engine.ProxyEngine` (LiteLLM); others register here.

Implementations become resolvable via :func:`register_gateway`, which is what
lets one live outside the ``sparkrun.proxy`` tree entirely.  The registry is
in-process rather than a SAF extension point because an engine is *constructed
with arguments* (host, port, master key, state dir) rather than resolved as a
stateless singleton — the same reason ``platforms`` and ``models.kv`` stayed
in-process.  Registration carries a **loader** rather than the class, so
``proxy.engine`` can import this module without a cycle and a registration
never drags in the implementation it names.

Note that enabling a second gateway's flag does **not** switch to it: with the
default enabled, the default wins.  Selecting a non-default gateway is an
explicit ``gateway:`` under ``proxy:`` in ``proxy.yaml``.

Two mechanisms, deliberately separate:

- **Availability** — each gateway declares a feature flag
  (``gateway.<name>``, see :mod:`sparkrun.core.features`).  ``gateway.litellm``
  is enabled on every channel; a future gateway would ship off by default.
- **Selection** — exactly one gateway is used at a time.  That is arbitrated
  *here*, at resolution: an explicit name (``proxy.gateway`` in ``proxy.yaml``,
  or a caller argument) must be known and enabled; with no name, the default
  wins when enabled, else the single remaining enabled gateway, else the caller
  is told to name one.  The flag registry has no notion of mutually-exclusive
  flags, so nothing stops a user enabling two — resolution refuses to guess
  instead.  This mirrors ``_default_executor_name`` in
  :mod:`sparkrun.orchestration.executor`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)

#: Gateway used when nothing names one.
DEFAULT_GATEWAY = "litellm"

#: Known gateway name -> the feature flag gating its availability.
#:
#: Maintained by :func:`register_gateway`; read by availability resolution.
GATEWAY_FEATURE_FLAGS: dict[str, str] = {}

#: Known gateway name -> zero-arg callable returning its engine class.
_GATEWAY_LOADERS: dict[str, "Callable[[], type]"] = {}


class GatewayError(RuntimeError):
    """Base class for gateway selection failures."""


class GatewayUnavailableError(GatewayError):
    """A gateway was requested (or needed) but is unknown or disabled.

    :attr:`gateway` names what was asked for (``None`` when the failure is
    "nothing is enabled"); :attr:`available` lists the enabled gateways.
    """

    def __init__(self, message: str, *, gateway: str | None = None, available: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.gateway = gateway
        self.available = available


class AmbiguousGatewayError(GatewayError):
    """Several gateways are enabled and none was named.

    :attr:`available` carries the candidates.
    """

    def __init__(self, message: str, *, available: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.available = available


def register_gateway(name: str, *, feature_flag: str, loader: "Callable[[], type]") -> None:
    """Make gateway *name* resolvable.

    Args:
        name: Selector used by ``proxy.gateway`` in ``proxy.yaml``.
        feature_flag: Flag gating availability; resolution refuses a gateway
            whose flag is off rather than silently falling back.
        loader: Zero-arg callable returning the engine class.  Deferred so a
            registration can name a class this module must not import at
            module scope, and so registering costs nothing at import time.

    Idempotent by name — re-registering replaces, which is what lets an
    out-of-tree plugin substitute an in-tree implementation.
    """
    GATEWAY_FEATURE_FLAGS[name] = feature_flag
    _GATEWAY_LOADERS[name] = loader


def gateway_class(name: str) -> type:
    """Return the engine class implementing gateway *name*.

    The one place a gateway name becomes an implementation.

    Raises:
        GatewayUnavailableError: No implementation is registered.  Deliberately
            distinct from "disabled": a name can be known to the flag registry
            while its plugin failed to load, and telling someone to enable a
            flag that is already on is a dead end.
    """
    loader = _GATEWAY_LOADERS.get(name)
    if loader is None:
        raise GatewayUnavailableError(
            "No implementation registered for gateway %r" % name,
            gateway=name,
            available=tuple(sorted(_GATEWAY_LOADERS)),
        )
    return loader()


def _load_litellm_engine() -> type:
    """Deferred import of the built-in LiteLLM engine."""
    from sparkrun.proxy.engine import ProxyEngine

    return ProxyEngine


# The built-in default.  Registered here rather than by a plugin because it is
# core: ``proxy`` must resolve to *something* even with every plugin absent.
register_gateway("litellm", feature_flag="gateway.litellm", loader=_load_litellm_engine)


def gateway_feature_flag(name: str) -> str | None:
    """Return the feature flag gating gateway *name*, or ``None`` if unknown."""
    return GATEWAY_FEATURE_FLAGS.get(name)


def is_gateway_enabled(name: str, *, config: "SparkrunConfig | None" = None) -> bool:
    """True when gateway *name* is known and its feature flag resolves on.

    Args:
        name: Gateway name (e.g. ``"litellm"``).
        config: Optional config to bind the flag resolution to.  When
            ``None`` the standard config path is read — the auto-discover
            daemon and other ``sctx``-less callers land here.
    """
    flag = gateway_feature_flag(name)
    if flag is None:
        return False
    if config is not None:
        return config.is_feature_enabled(flag)

    from sparkrun.core.features import feature_gate_enabled

    return feature_gate_enabled(flag)


def list_gateways(*, config: "SparkrunConfig | None" = None) -> list[str]:
    """Return the enabled gateway names, sorted."""
    return sorted(name for name in GATEWAY_FEATURE_FLAGS if is_gateway_enabled(name, config=config))


def resolve_gateway(name: str | None = None, *, config: "SparkrunConfig | None" = None) -> str:
    """Resolve which gateway to use.

    Args:
        name: Explicitly requested gateway (from ``proxy.gateway`` in
            ``proxy.yaml`` or a caller argument).  ``None`` / empty means
            "pick the default".
        config: Optional config binding for flag resolution.

    Returns:
        The resolved gateway name.

    Raises:
        GatewayUnavailableError: *name* is unknown or its flag is off, or no
            gateway is enabled at all.
        AmbiguousGatewayError: no name was given, the default is disabled, and
            more than one other gateway is enabled.
    """
    enabled = list_gateways(config=config)

    if name:
        if is_gateway_enabled(name, config=config):
            return name
        raise GatewayUnavailableError(
            _unavailable_message(name, enabled),
            gateway=name,
            available=tuple(enabled),
        )

    if DEFAULT_GATEWAY in enabled:
        return DEFAULT_GATEWAY

    if not enabled:
        raise GatewayUnavailableError(
            "No inference gateway is enabled. Enable one with: sparkrun setup features enable %s" % GATEWAY_FEATURE_FLAGS[DEFAULT_GATEWAY],
            available=(),
        )

    if len(enabled) == 1:
        return enabled[0]

    raise AmbiguousGatewayError(
        "Several gateways are enabled (%s) and the default (%s) is disabled. "
        "Name one with 'gateway: <name>' under 'proxy:' in proxy.yaml." % (", ".join(enabled), DEFAULT_GATEWAY),
        available=tuple(enabled),
    )


def require_gateway_enabled(name: str, *, config: "SparkrunConfig | None" = None) -> None:
    """Raise unless gateway *name* is known and enabled.

    The enforcement point for a gateway implementation's own bring-up (see
    :meth:`sparkrun.proxy.engine.ProxyEngine.start`).  Deliberately *not*
    applied to teardown, status, or the auto-discover daemon's restart path:
    a proxy that is already running must stay stoppable and manageable even
    after its flag has been turned off — the same rule
    ``cleanup_cluster_transport`` follows for transports.
    """
    if is_gateway_enabled(name, config=config):
        return
    enabled = list_gateways(config=config)
    raise GatewayUnavailableError(
        _unavailable_message(name, enabled),
        gateway=name,
        available=tuple(enabled),
    )


def _unavailable_message(name: str, enabled: list[str]) -> str:
    """Build the 'gateway X is not usable' message, with the remedy."""
    flag = gateway_feature_flag(name)
    if flag is None:
        known = ", ".join(sorted(GATEWAY_FEATURE_FLAGS)) or "(none)"
        return "Unknown gateway %r. Known gateways: %s" % (name, known)
    detail = "Gateway %r is disabled. Enable it with: sparkrun setup features enable %s" % (name, flag)
    if enabled:
        detail += " (currently enabled: %s)" % ", ".join(enabled)
    return detail


__all__ = [
    "DEFAULT_GATEWAY",
    "GATEWAY_FEATURE_FLAGS",
    "AmbiguousGatewayError",
    "GatewayError",
    "GatewayUnavailableError",
    "gateway_class",
    "gateway_feature_flag",
    "is_gateway_enabled",
    "list_gateways",
    "register_gateway",
    "require_gateway_enabled",
    "resolve_gateway",
]
