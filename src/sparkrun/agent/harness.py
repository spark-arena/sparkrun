"""Agent harness — creates and configures the smolagents CodeAgent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import CodeAgent

logger = logging.getLogger(__name__)

SPARKRUN_CONTEXT = """\

You are sparkrun-agent, an AI assistant that helps manage inference workloads \
on NVIDIA DGX Spark systems.

You have access to tools that let you:
- Launch and stop inference workloads (run_inference, stop_inference)
- Check cluster status and view container logs (cluster_status, container_logs)
- Search, list, and inspect recipes (recipe_search, recipe_list, recipe_show)
- Create and validate new recipes (recipe_create, recipe_validate)
- Manage cluster definitions (cluster_list, cluster_show, cluster_create)
- Set up SSH mesh, networking, and permissions (setup_ssh_mesh, setup_cx7, setup_permissions)

Key concepts:
- A **recipe** defines how to run a model (model ID, runtime, container, defaults).
- A **cluster** is a named group of DGX Spark hosts.
- Each DGX Spark has 1 GPU with 128 GB unified memory.
- **Tensor parallelism** (--tp) maps to node count: --tp 2 means 2 hosts.
- Use --solo for single-node mode.

Important rules:
- Always use the appropriate tools rather than trying to execute shell commands directly.
- When the user asks to run a model, search for matching recipes first if they don't \
provide an exact recipe name.
- When calling tools, omit optional parameters you don't need — do NOT pass empty strings.
- Never assign a tool's result to a variable with the same name as the tool.
"""


def create_agent(base_url: str, *, system_prompt: str | None = None, verbose: bool = False) -> CodeAgent:
    """Create a smolagents CodeAgent with sparkrun tools.

    Args:
        base_url: OpenAI-compatible API base URL (e.g. ``http://10.0.0.1:52001/v1``).
        system_prompt: Optional custom system prompt (uses default if None).
        verbose: If True, show smolagents step-by-step execution details.

    Returns:
        Configured CodeAgent instance.
    """
    from smolagents import CodeAgent, OpenAIServerModel
    from smolagents.agents import LogLevel

    from sparkrun.agent.tools import discover_tools

    model = OpenAIServerModel(
        model_id="sparkrun-agent",
        api_base=base_url.rstrip("/"),
        api_key="not-needed",
    )

    tools = discover_tools()
    logger.info("Loaded %d agent tools", len(tools))

    agent = CodeAgent(
        tools=tools,
        model=model,
        verbosity_level=LogLevel.INFO if verbose else LogLevel.ERROR,
    )

    # Append sparkrun domain context to the default system prompt
    # (which contains critical formatting instructions for code blocks
    # and final_answer usage that the model needs).
    default_prompt = agent.prompt_templates["system_prompt"]
    agent.prompt_templates["system_prompt"] = default_prompt + (system_prompt or SPARKRUN_CONTEXT)

    return agent
