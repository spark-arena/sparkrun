"""Background auto-discovery process for the sparkrun proxy.

Periodically re-discovers inference endpoints and reconciles the proxy's
model list with them.  Designed to run as a detached subprocess spawned by
``ProxyEngine.start()``.

Reconciliation rewrites the litellm config and restarts the proxy, because
LiteLLM's runtime mutation endpoints require a DB sparkrun does not
provision (see ``ProxyEngine.apply_desired_state``).  A sweep that finds no
change does nothing at all, so a steady-state cluster is never disturbed.

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
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SHUTDOWN = False

# Consecutive 1s checks with no live proxy before we conclude it is really
# gone.  A restart briefly has no PID at all, so a single miss must not be
# read as a shutdown; `proxy stop` signals us directly, so a generous value
# only ever delays the exit of an otherwise-idle daemon.
PROXY_GONE_TOLERANCE = 30


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
            by ``ProxyEngine.start()``.
    """
    global _SHUTDOWN

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
    proxy_port = cfg.get("proxy_port", 4000)
    master_key = cfg.get("master_key")
    host_list = cfg.get("host_list")
    ssh_kwargs = cfg.get("ssh_kwargs")
    cache_dir = cfg.get("cache_dir")

    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.discovery import discover_endpoints
    from sparkrun.proxy.engine import ProxyEngine

    engine = ProxyEngine(port=proxy_port, master_key=master_key)
    # Daemon process — no sctx available; ProxyConfig() direct construction is the documented exception.
    proxy_cfg = ProxyConfig()

    logger.info(
        "Auto-discover started: interval=%ds, proxy_pid=%d, hosts=%s",
        interval,
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
            current_pid = engine.current_pid()
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

            # Models and aliases are reconciled in ONE call: they share a
            # single config file, so applying them separately would rewrite
            # and restart the proxy twice per sweep.  Aliases are re-read
            # each sweep so an alias added between sweeps is picked up.
            proxy_cfg._load()
            added, removed = engine.apply_desired_state(healthy, proxy_cfg.aliases)
            if added or removed:
                logger.info(
                    "Auto-discover: config updated (+%d/-%d model entries), proxy restarted",
                    added,
                    removed,
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
