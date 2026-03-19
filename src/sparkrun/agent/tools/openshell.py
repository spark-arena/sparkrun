"""OpenShell/NemoClaw integration stub for sandboxed code execution."""

from __future__ import annotations

from smolagents import Tool


class OpenShellExecuteTool(Tool):
    name = "openshell_execute"
    description = (
        "Execute code in a sandboxed environment via NVIDIA OpenShell/NemoClaw. "
        "NOTE: This tool is a placeholder for future OpenShell integration "
        "and is not yet functional."
    )
    inputs = {
        "code": {
            "type": "string",
            "description": "Code to execute in the sandbox",
        },
        "language": {
            "type": "string",
            "description": "Programming language (default: 'python')",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, code: str, language: str | None = None) -> str:
        return (
            "OpenShell integration is not yet available. "
            "This tool will provide sandboxed code execution via "
            "NVIDIA OpenShell/NemoClaw in a future release."
        )
