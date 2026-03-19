"""Tools for checking cluster status and viewing container logs."""

from __future__ import annotations

from ._base import SparkrunBaseTool


class ClusterStatusTool(SparkrunBaseTool):
    name = "cluster_status"
    description = (
        "Show sparkrun containers running on cluster hosts. "
        "Displays container names, status, and resource usage. "
        "Example: result = cluster_status() — call with no arguments to check all hosts. "
        "Only pass hosts or cluster if you need to narrow scope."
    )
    inputs = {
        "hosts": {
            "type": "string",
            "description": "Comma-separated host list. Omit to use default cluster.",
            "nullable": True,
        },
        "cluster": {
            "type": "string",
            "description": "Named cluster to check. Omit to use default.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, hosts: str | None = None, cluster: str | None = None) -> str:
        args = ["status"]
        if hosts:
            args.extend(["--hosts", hosts])
        if cluster:
            args.extend(["--cluster", cluster])
        return self._run_sparkrun(*args)


class ContainerLogsTool(SparkrunBaseTool):
    name = "container_logs"
    description = (
        "View recent logs from a running inference container. "
        "Useful for checking model loading progress or errors. "
        "Example: logs = container_logs(recipe_name='qwen3-1.7b-vllm')"
    )
    inputs = {
        "recipe_name": {
            "type": "string",
            "description": "Name of the recipe whose logs to view",
        },
        "tail": {
            "type": "integer",
            "description": "Number of log lines to show (default: 50)",
            "nullable": True,
        },
        "hosts": {
            "type": "string",
            "description": "Comma-separated host list (optional)",
            "nullable": True,
        },
        "cluster": {
            "type": "string",
            "description": "Named cluster to use (optional)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, recipe_name: str, tail: int | None = None,
                hosts: str | None = None, cluster: str | None = None) -> str:
        args = ["logs", recipe_name, "--no-follow"]
        if tail is not None:
            args.extend(["--tail", str(tail)])
        else:
            args.extend(["--tail", "50"])
        if hosts:
            args.extend(["--hosts", hosts])
        if cluster:
            args.extend(["--cluster", cluster])
        return self._run_sparkrun(*args, timeout=30)
