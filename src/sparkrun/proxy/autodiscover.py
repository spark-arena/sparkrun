"""Background auto-discovery process for the sparkrun proxy.

Periodically re-discovers inference endpoints and syncs the proxy's
model list via the LiteLLM management API.  Designed to run as a
detached subprocess spawned by ``ProxyEngine.start()``.

Exits automatically when the proxy process (monitored by PID) dies.
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


def _handle_signal(signum, _frame):
    global _SHUTDOWN
    _SHUTDOWN = True


def _proxy_alive(pid: int) -> bool:
    """Check if the proxy process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


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
    proxy_cfg = ProxyConfig()

    logger.info(
        "Auto-discover started: interval=%ds, proxy_pid=%d, hosts=%s",
        interval,
        proxy_pid,
        host_list,
    )

    while not _SHUTDOWN:
        # Sleep in small increments so we respond to signals promptly
        for _ in range(interval):
            if _SHUTDOWN:
                break
            time.sleep(1)
            if not _proxy_alive(proxy_pid):
                logger.info("Proxy PID %d gone, exiting auto-discover", proxy_pid)
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
            added, removed = engine.sync_models(healthy)
            if added or removed:
                logger.info("Auto-discover sync: added=%d, removed=%d", added, removed)
            else:
                logger.debug("Auto-discover: no changes")

            # Apply configured aliases (re-read config each sweep
            # so alias add/remove between sweeps is picked up)
            proxy_cfg._load()
            aliases = proxy_cfg.aliases
            if aliases:
                a_added, a_removed = engine.sync_aliases(aliases)
                if a_added or a_removed:
                    logger.info("Auto-discover alias sync: added=%d, removed=%d", a_added, a_removed)
        except Exception:
            logger.debug("Auto-discover sweep failed", exc_info=True)

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
