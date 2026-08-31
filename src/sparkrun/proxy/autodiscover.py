"""Background auto-discovery process for the sparkrun proxy.

Periodically re-discovers inference endpoints and reconciles the running
gateway's model list with them.  Designed to run as a detached subprocess
spawned by :meth:`sparkrun.proxy._supervisor.GatewaySupervisor.start_autodiscover`.

**Gateway-neutral.** Reconciliation goes through :func:`sparkrun.api.proxy.sync`,
which resolves the implementation recorded in the shared gateway state file, so
this process never names a gateway class.  LiteLLM rewrites its config and
restarts; another gateway may update its control plane in place.  A sweep that
finds no change does nothing at all, so a steady-state cluster is never
disturbed.  Because the engine (and therefore its credential) is resolved from
the state file, this daemon's own config file no longer carries the master key.

Exits automatically when the proxy dies.  The proxy PID is re-read from the
state file each check rather than pinned at startup, so a restart — whether
performed by this process or by a ``sparkrun proxy`` command — is followed
instead of being mistaken for a shutdown.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SHUTDOWN = False

# Consecutive 1s checks with no live proxy before we conclude it is really
# gone.  A restart briefly has no PID at all, so a single miss must not be
# read as a shutdown; `proxy stop` signals us directly, so a generous value
# only ever delays the exit of an otherwise-idle daemon.
PROXY_GONE_TOLERANCE = 30


def _endpoint_identity(endpoint: Any) -> tuple[Any, ...]:
    """Stable identity for one workload across successive health sweeps.

    ``cluster_id`` when the endpoint has one: an address alone is not stable
    across a relaunch, and the cluster_id is the identity the rest of the
    system already keys workloads by.
    """
    cluster_id = str(getattr(endpoint, "cluster_id", "") or "").strip()
    if cluster_id:
        return ("cluster", cluster_id)
    return (
        "address",
        str(getattr(endpoint, "host", "") or ""),
        int(getattr(endpoint, "port", 0) or 0),
    )


@dataclass
class _TrackedEndpoint:
    endpoint: Any
    missed_sweeps: int = 0


class _EndpointRemovalGrace:
    """Keep a previously healthy endpoint through bounded transient misses.

    A health probe that times out once is not evidence a workload is gone, and
    evicting it costs a gateway restart plus a window where clients get a 404
    for a model that is serving fine.  An endpoint must be absent for
    *required_misses* consecutive sweeps before it is dropped; ``1`` restores
    the historical remove-on-first-miss behaviour.
    """

    def __init__(self, required_misses: int) -> None:
        self.required_misses = max(1, int(required_misses))
        self._tracked: dict[tuple[Any, ...], _TrackedEndpoint] = {}

    def reconcile(self, current: list[Any]) -> tuple[list[Any], int]:
        """Return ``(effective_endpoints, deferred_removal_count)``."""
        next_tracked: dict[tuple[Any, ...], _TrackedEndpoint] = {}
        effective: list[Any] = []
        for endpoint in current:
            identity = _endpoint_identity(endpoint)
            if identity not in next_tracked:
                effective.append(endpoint)
            next_tracked[identity] = _TrackedEndpoint(endpoint)

        deferred = 0
        for identity, tracked in self._tracked.items():
            if identity in next_tracked:
                continue
            missed_sweeps = tracked.missed_sweeps + 1
            if missed_sweeps < self.required_misses:
                next_tracked[identity] = _TrackedEndpoint(tracked.endpoint, missed_sweeps)
                effective.append(tracked.endpoint)
                deferred += 1

        self._tracked = next_tracked
        return effective, deferred


def _handle_signal(signum, _frame):
    global _SHUTDOWN
    _SHUTDOWN = True


def _proxy_alive(pid: int) -> bool:
    """Check if the proxy process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, just not ours to signal.
        return True


def run_autodiscover(config_path: str) -> None:
    """Run the auto-discovery loop.

    Args:
        config_path: Path to the auto-discovery config YAML written
            by :meth:`GatewaySupervisor.start_autodiscover`.
    """
    global _SHUTDOWN
    # Reset rather than read: the module global survives a previous run in the
    # same interpreter, which would otherwise leave a second call dead on
    # arrival (it never enters the loop).  Only tests call twice in-process.
    _SHUTDOWN = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    path = Path(config_path)
    if not path.exists():
        logger.error("Auto-discover config not found: %s", config_path)
        return

    with open(path) as f:
        cfg = yaml.safe_load(f)

    proxy_pid = cfg["proxy_pid"]
    interval = cfg.get("interval", 30)
    removal_grace_sweeps = max(1, int(cfg.get("removal_grace_sweeps", 2)))
    gateway = str(cfg.get("gateway") or "unknown")
    state_dir = Path(cfg["state_dir"]) if cfg.get("state_dir") else None
    host_list = cfg.get("host_list")
    ssh_kwargs = cfg.get("ssh_kwargs")
    cache_dir = cfg.get("cache_dir")

    from sparkrun import api
    from sparkrun.api._context import default_sctx
    from sparkrun.proxy._supervisor import GatewayState
    from sparkrun.proxy.discovery import discover_endpoints

    # Only the state file is read directly; which gateway to act on is resolved
    # from it by api.proxy.sync on every sweep, so a `proxy start --restart`
    # that swaps implementations is followed rather than fought.
    state_probe = GatewayState(state_dir=state_dir)
    sctx = default_sctx()
    removal_grace = _EndpointRemovalGrace(removal_grace_sweeps)

    logger.info(
        "Auto-discover started: gateway=%s, interval=%ds, removal_grace=%d sweep(s), proxy_pid=%d, hosts=%s",
        gateway,
        interval,
        removal_grace_sweeps,
        proxy_pid,
        host_list,
    )

    missing_streak = 0

    while not _SHUTDOWN:
        # Sleep in small increments so we respond to signals promptly
        for _ in range(interval):
            if _SHUTDOWN:
                break
            time.sleep(1)
            current_pid = state_probe.current_pid()
            if current_pid is not None and _proxy_alive(current_pid):
                missing_streak = 0
                proxy_pid = current_pid
                continue
            missing_streak += 1
            if missing_streak >= PROXY_GONE_TOLERANCE:
                logger.info("Proxy gone (last PID %s), exiting auto-discover", proxy_pid)
                return

        if _SHUTDOWN:
            break

        try:
            endpoints = discover_endpoints(
                check_health=True,
                host_list=host_list,
                ssh_kwargs=ssh_kwargs,
                cache_dir=cache_dir,
            )
            healthy = [ep for ep in endpoints if ep.healthy]
            effective, deferred_removals = removal_grace.reconcile(healthy)
            if deferred_removals:
                logger.info(
                    "Auto-discover: deferring removal of %d endpoint(s) during health grace",
                    deferred_removals,
                )

            # Models and aliases are reconciled in ONE gateway-neutral call:
            # for a config-file gateway they share a single file, so applying
            # them separately would rewrite and restart the proxy twice per
            # sweep.  The config is re-read each sweep so an alias added by
            # another CLI process between sweeps is picked up.
            sctx.proxy_config._load()
            result = api.proxy.sync(
                endpoints=effective,
                aliases=sctx.proxy_config.aliases,
                sctx=sctx,
            )
            if result.added or result.removed:
                logger.info(
                    "Auto-discover: gateway reconciled (+%d/-%d model entries)",
                    result.added,
                    result.removed,
                )
            else:
                logger.debug("Auto-discover: no changes")
        except Exception:
            logger.warning("Auto-discover sweep failed", exc_info=True)

    logger.info("Auto-discover shutting down")


def main() -> None:
    """Entry point for ``python -m sparkrun.proxy.autodiscover``."""
    if len(sys.argv) != 2:
        print("Usage: python -m sparkrun.proxy.autodiscover <config_path>", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [autodiscover] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_autodiscover(sys.argv[1])


if __name__ == "__main__":
    main()
