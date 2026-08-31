"""Public proxy/gateway errors for :mod:`sparkrun.api.proxy`.

Failures from :mod:`sparkrun.proxy` (gateway selection, engine lifecycle) are
translated into these :class:`~sparkrun.api._errors.SparkrunError` subclasses at
the api boundary, so callers can ``except SparkrunError`` uniformly.
"""

from __future__ import annotations

from sparkrun.api._errors import SparkrunError


class GatewayUnavailable(SparkrunError):
    """The requested gateway is unknown, disabled, or ambiguous.

    Raised when :mod:`sparkrun.proxy.gateway` refuses to resolve: the name is
    unknown, its ``gateway.<name>`` feature flag is off, no gateway is enabled
    at all, or several are enabled and none was named.  :attr:`gateway` is the
    requested name (``None`` when none was given); :attr:`available` lists the
    enabled gateways.
    """

    def __init__(self, message: str, *, gateway: str | None = None, available: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.gateway = gateway
        self.available = available


class ProxyAlreadyRunning(SparkrunError):
    """A proxy is already running and ``restart`` was not requested.

    :attr:`pid` and :attr:`port` identify the live process.
    :attr:`persisted` names the ``proxy.yaml`` keys that were written before
    the conflict was detected — settings are saved regardless of whether the
    proxy is restarted now, so callers must still report them.
    """

    def __init__(
        self,
        message: str,
        *,
        pid: int | None = None,
        port: int | None = None,
        persisted: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.pid = pid
        self.port = port
        self.persisted = persisted


class ProxyStartFailed(SparkrunError):
    """The gateway process could not be started, or did not survive startup.

    :attr:`exit_code` carries the engine's return code when known.
    """

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ProxyUpdateFailed(SparkrunError):
    """The gateway config was rewritten but the running proxy did not adopt it.

    Wraps :class:`~sparkrun.proxy.engine.ProxyRestartError`: the change is on
    disk and applies on the next start, but the *running* process is stale.
    """


__all__ = [
    "GatewayUnavailable",
    "ProxyAlreadyRunning",
    "ProxyStartFailed",
    "ProxyUpdateFailed",
]
