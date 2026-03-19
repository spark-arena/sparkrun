"""sparkrun agent tools — smolagents Tool wrappers for sparkrun CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import Tool


def discover_tools() -> list[Tool]:
    """Return all available sparkrun agent tools."""
    from .run_stop import RunInferenceTool, StopInferenceTool
    from .status_logs import ClusterStatusTool, ContainerLogsTool
    from .recipes import RecipeSearchTool, RecipeListTool, RecipeShowTool, RecipeCreateTool, RecipeValidateTool
    from .clusters import ClusterListTool, ClusterShowTool, ClusterCreateTool
    from .setup import SetupSSHMeshTool, SetupCX7Tool, SetupPermissionsTool
    from .openshell import OpenShellExecuteTool

    return [
        RunInferenceTool(),
        StopInferenceTool(),
        ClusterStatusTool(),
        ContainerLogsTool(),
        RecipeSearchTool(),
        RecipeListTool(),
        RecipeShowTool(),
        RecipeCreateTool(),
        RecipeValidateTool(),
        ClusterListTool(),
        ClusterShowTool(),
        ClusterCreateTool(),
        SetupSSHMeshTool(),
        SetupCX7Tool(),
        SetupPermissionsTool(),
        OpenShellExecuteTool(),
    ]
