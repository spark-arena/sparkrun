"""Gradio web chat interface for the sparkrun agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import CodeAgent

logger = logging.getLogger(__name__)


def run_web_chat(agent: CodeAgent) -> None:
    """Launch the Gradio web chat interface.

    Requires the ``gradio`` package (install via ``pip install sparkrun[agent-web]``).

    Uses smolagents' built-in GradioUI for minimal integration effort.
    """
    try:
        from smolagents import GradioUI
    except ImportError:
        raise ImportError(
            "Gradio is required for the web UI. "
            "Install it with: pip install sparkrun[agent-web]"
        )

    logger.info("Starting Gradio web interface...")
    GradioUI(agent).launch()
