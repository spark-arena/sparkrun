"""Tools for launching and stopping inference workloads."""

from __future__ import annotations

from ._base import SparkrunBaseTool


class RunInferenceTool(SparkrunBaseTool):
    name = "run_inference"
    description = (
        "Launch an inference workload on the DGX Spark cluster. "
        "Example: result = run_inference(recipe_name='qwen3-1.7b-vllm') — "
        "launches with defaults. Add solo=True for single-node, "
        "tensor_parallel=2 for 2-host TP, gpu_mem=0.8 to limit GPU memory. "
        "Omit optional parameters you don't need."
    )
    inputs = {
        "recipe_name": {
            "type": "string",
            "description": "Name of the recipe to run (e.g. 'qwen3-1.7b-vllm')",
        },
        "hosts": {
            "type": "string",
            "description": "Comma-separated host list. Omit to use default cluster.",
            "nullable": True,
        },
        "cluster": {
            "type": "string",
            "description": "Named cluster to use. Omit to use default.",
            "nullable": True,
        },
        "solo": {
            "type": "boolean",
            "description": "Force single-node mode. Omit for default behavior.",
            "nullable": True,
        },
        "tensor_parallel": {
            "type": "integer",
            "description": "Override tensor parallelism (number of hosts). Omit for recipe default.",
            "nullable": True,
        },
        "gpu_mem": {
            "type": "number",
            "description": "GPU memory utilization 0.0-1.0. Omit for recipe default.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, recipe_name: str, hosts: str | None = None,
                cluster: str | None = None, solo: bool | None = None,
                tensor_parallel: int | None = None,
                gpu_mem: float | None = None) -> str:
        args = ["run", recipe_name, "--no-follow"]
        if hosts:
            args.extend(["--hosts", hosts])
        if cluster:
            args.extend(["--cluster", cluster])
        if solo:
            args.append("--solo")
        if tensor_parallel is not None:
            args.extend(["--tp", str(tensor_parallel)])
        if gpu_mem is not None:
            args.extend(["--gpu-mem", str(gpu_mem)])
        return self._run_sparkrun(*args, timeout=300)


class StopInferenceTool(SparkrunBaseTool):
    name = "stop_inference"
    description = (
        "Stop a running inference workload. "
        "Example: result = stop_inference(recipe_name='qwen3-1.7b-vllm') — "
        "stops that recipe. Use stop_all=True to stop all containers."
    )
    inputs = {
        "recipe_name": {
            "type": "string",
            "description": "Name of the recipe to stop (optional if stop_all=true)",
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
        "stop_all": {
            "type": "boolean",
            "description": "Stop all sparkrun containers (default: false)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, recipe_name: str | None = None, hosts: str | None = None,
                cluster: str | None = None, stop_all: bool | None = None) -> str:
        args = ["stop"]
        if stop_all:
            args.append("--all")
        elif recipe_name:
            args.append(recipe_name)
        else:
            return "Error: must provide recipe_name or set stop_all=true"
        if hosts:
            args.extend(["--hosts", hosts])
        if cluster:
            args.extend(["--cluster", cluster])
        return self._run_sparkrun(*args)
