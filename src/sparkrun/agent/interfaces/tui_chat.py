"""Textual TUI chat interface for the sparkrun agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

if TYPE_CHECKING:
    from smolagents import CodeAgent

logger = logging.getLogger(__name__)


class MessageWidget(Static):
    """A single chat message."""

    def __init__(self, role: str, content: str) -> None:
        prefix = "You" if role == "user" else "Agent"
        super().__init__("[bold]%s:[/bold] %s" % (prefix, content))
        self.add_class(role)


class AgentChatApp(App):
    """Textual chat application for the sparkrun agent."""

    TITLE = "sparkrun agent"
    CSS = """
    VerticalScroll {
        height: 1fr;
        padding: 0 1;
    }
    .user {
        color: $accent;
        margin-bottom: 1;
    }
    .assistant {
        color: $text;
        margin-bottom: 1;
    }
    Input {
        dock: bottom;
        margin: 0 1;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self, agent: CodeAgent, endpoint: str = "") -> None:
        super().__init__()
        self._agent = agent
        self._endpoint = endpoint

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-log")
        yield Input(placeholder="Ask sparkrun-agent something...")
        yield Footer()

    def on_mount(self) -> None:
        if self._endpoint:
            self.sub_title = self._endpoint
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        event.input.value = ""

        if user_text.lower() in ("quit", "exit", "q"):
            self.exit()
            return

        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(MessageWidget("user", user_text))

        # Run agent (blocking — fine for local model latency)
        try:
            result = self._agent.run(user_text)
            response = str(result) if result else "(no response)"
        except Exception as e:
            logger.debug("Agent error", exc_info=True)
            response = "Error: %s" % e

        log.mount(MessageWidget("assistant", response))
        log.scroll_end(animate=False)


def run_tui_chat(agent: CodeAgent, endpoint: str = "") -> None:
    """Launch the Textual TUI chat interface."""
    app = AgentChatApp(agent, endpoint=endpoint)
    app.run()
