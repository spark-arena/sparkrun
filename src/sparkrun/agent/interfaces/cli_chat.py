"""Readline-based CLI chat interface for the sparkrun agent."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import CodeAgent

logger = logging.getLogger(__name__)


def run_cli_chat(agent: CodeAgent) -> None:
    """Start an interactive CLI REPL for chatting with the agent.

    Uses readline for line editing and history. Streams agent
    responses to stdout. Type 'quit', 'exit', or Ctrl-D to exit.
    """
    try:
        import readline  # noqa: F401 — enables line editing
    except ImportError:
        pass

    print("sparkrun agent ready. Type 'quit' to exit.")
    print()

    while True:
        try:
            user_input = input("sparkrun-agent> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        try:
            result = agent.run(user_input)
            if result:
                print(result)
                print()
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:
            logger.debug("Agent error", exc_info=True)
            print("Error: %s" % e, file=sys.stderr)
            print()
