"""Agent session state persistence.

Stores/loads agent session info (endpoint, cluster_id, recipe, etc.)
to ``~/.cache/sparkrun/agent-state.yaml`` so that ``agent stop`` and
``agent status`` can find the running agent model.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import yaml

from sparkrun.core.config import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

STATE_FILE = DEFAULT_CACHE_DIR / "agent-state.yaml"


def save_state(
        endpoint: str,
        cluster_id: str,
        recipe: str,
        host: str,
        port: int,
) -> None:
    """Persist agent session state to disk."""
    state = {
        "endpoint": endpoint,
        "cluster_id": cluster_id,
        "recipe": recipe,
        "host": host,
        "port": port,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        yaml.dump(state, f, default_flow_style=False)
    logger.debug("Agent state saved to %s", STATE_FILE)


def load_state() -> dict[str, Any] | None:
    """Load agent session state from disk, or None if not found."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        logger.debug("Failed to read agent state", exc_info=True)
        return None


def clear_state() -> None:
    """Remove agent session state file."""
    try:
        STATE_FILE.unlink(missing_ok=True)
        logger.debug("Agent state cleared")
    except OSError:
        logger.debug("Failed to clear agent state", exc_info=True)
