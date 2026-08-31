"""Public library API for the inference proxy (gateway).

Console-free surface over :mod:`sparkrun.proxy`, mirroring the
:mod:`sparkrun.api.tailscale` conventions: dataclass returns, typed
:class:`~sparkrun.api._errors.SparkrunError` subclasses, never writes to
stdout/stderr, never calls ``sys.exit``.

This is the seam that stays stable when a second gateway implementation is
added.  Callers name *what* they want (start the proxy, sync its models);
:mod:`sparkrun.proxy.gateway` decides *which* implementation serves it, and
:func:`resolve_gateway` / :func:`list_gateways` expose that decision.

Functions:

- :func:`start` — discover endpoints, write the gateway config, launch it.
- :func:`stop` — stop the running gateway and its auto-discover daemon.
- :func:`status` — process state plus the models currently served.
- :func:`models` — just the served model list.
- :func:`sync` — reconcile the served model list with what is running.
- :func:`add_alias` / :func:`remove_alias` / :func:`list_aliases`.
- :func:`resolve_gateway` / :func:`list_gateways` — gateway selection.
"""

from __future__ import annotations

from ._errors import (
    GatewayUnavailable,
    ProxyAlreadyRunning,
    ProxyStartFailed,
    ProxyUpdateFailed,
)
from ._ops import (
    ProxyAliasResult,
    ProxyEndpoint,
    ProxyModel,
    ProxyStartOptions,
    ProxyStartResult,
    ProxyStatus,
    ProxyStopResult,
    ProxySyncResult,
    add_alias,
    list_aliases,
    list_gateways,
    models,
    remove_alias,
    resolve_gateway,
    start,
    status,
    stop,
    sync,
)

__all__ = [
    # Functions
    "start",
    "stop",
    "status",
    "models",
    "sync",
    "add_alias",
    "remove_alias",
    "list_aliases",
    "resolve_gateway",
    "list_gateways",
    # Data models
    "ProxyStartOptions",
    "ProxyStartResult",
    "ProxyStopResult",
    "ProxyStatus",
    "ProxyModel",
    "ProxyEndpoint",
    "ProxySyncResult",
    "ProxyAliasResult",
    # Errors
    "GatewayUnavailable",
    "ProxyAlreadyRunning",
    "ProxyStartFailed",
    "ProxyUpdateFailed",
]
