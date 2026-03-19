"""Tools for managing sparkrun cluster definitions."""

from __future__ import annotations

from ._base import SparkrunBaseTool


class ClusterListTool(SparkrunBaseTool):
    name = "cluster_list"
    description = (
        "List all saved cluster definitions with their hosts and configuration. "
        "Example: clusters = cluster_list()"
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        return self._run_sparkrun("cluster", "list")


class ClusterShowTool(SparkrunBaseTool):
    name = "cluster_show"
    description = (
        "Show detailed information about a specific saved cluster. "
        "Example: info = cluster_show(cluster_name='mylab')"
    )
    inputs = {
        "cluster_name": {
            "type": "string",
            "description": "Name of the cluster to show",
        },
    }
    output_type = "string"

    def forward(self, cluster_name: str) -> str:
        return self._run_sparkrun("cluster", "show", cluster_name)


class ClusterCreateTool(SparkrunBaseTool):
    name = "cluster_create"
    description = (
        "Create a new saved cluster definition with a name and list of hosts. "
        "Example: result = cluster_create(cluster_name='mylab', hosts='10.0.0.1,10.0.0.2') — "
        "add set_default=True to make it the default cluster."
    )
    inputs = {
        "cluster_name": {
            "type": "string",
            "description": "Name for the new cluster",
        },
        "hosts": {
            "type": "string",
            "description": "Comma-separated list of host IPs or hostnames",
        },
        "set_default": {
            "type": "boolean",
            "description": "Set this cluster as the default (default: false)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, cluster_name: str, hosts: str,
                set_default: bool | None = None) -> str:
        args = ["cluster", "create", cluster_name, "--hosts", hosts]
        if set_default:
            args.append("--default")
        return self._run_sparkrun(*args)
