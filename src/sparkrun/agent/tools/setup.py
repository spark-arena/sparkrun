"""Tools for sparkrun setup operations."""

from __future__ import annotations

from ._base import SparkrunBaseTool


class SetupSSHMeshTool(SparkrunBaseTool):
    name = "setup_ssh_mesh"
    description = (
        "Set up passwordless SSH between cluster hosts for multi-node inference. "
        "Example: result = setup_ssh_mesh(cluster='mylab') — "
        "or setup_ssh_mesh(hosts='10.0.0.1,10.0.0.2')"
    )
    inputs = {
        "hosts": {
            "type": "string",
            "description": "Comma-separated host list (optional if cluster is set)",
            "nullable": True,
        },
        "cluster": {
            "type": "string",
            "description": "Named cluster to use (optional)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, hosts: str | None = None, cluster: str | None = None) -> str:
        args = ["setup", "ssh-mesh"]
        if hosts:
            args.extend(["--hosts", hosts])
        if cluster:
            args.extend(["--cluster", cluster])
        return self._run_sparkrun(*args, timeout=120)


class SetupCX7Tool(SparkrunBaseTool):
    name = "setup_cx7"
    description = (
        "Configure ConnectX-7 network interfaces on cluster hosts "
        "for high-speed InfiniBand/Ethernet communication."
    )
    inputs = {
        "hosts": {
            "type": "string",
            "description": "Comma-separated host list (optional if cluster is set)",
            "nullable": True,
        },
        "cluster": {
            "type": "string",
            "description": "Named cluster to use (optional)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, hosts: str | None = None, cluster: str | None = None) -> str:
        args = ["setup", "cx7"]
        if hosts:
            args.extend(["--hosts", hosts])
        if cluster:
            args.extend(["--cluster", cluster])
        return self._run_sparkrun(*args, timeout=120)


class SetupPermissionsTool(SparkrunBaseTool):
    name = "setup_permissions"
    description = (
        "Fix Docker permissions on cluster hosts so the current user "
        "can run containers without sudo."
    )
    inputs = {
        "hosts": {
            "type": "string",
            "description": "Comma-separated host list (optional if cluster is set)",
            "nullable": True,
        },
        "cluster": {
            "type": "string",
            "description": "Named cluster to use (optional)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, hosts: str | None = None, cluster: str | None = None) -> str:
        args = ["setup", "permissions"]
        if hosts:
            args.extend(["--hosts", hosts])
        if cluster:
            args.extend(["--cluster", cluster])
        return self._run_sparkrun(*args, timeout=120)
