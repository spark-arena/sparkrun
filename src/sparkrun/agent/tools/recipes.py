"""Tools for searching, listing, showing, creating, and validating recipes."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ._base import SparkrunBaseTool

logger = logging.getLogger(__name__)


class RecipeSearchTool(SparkrunBaseTool):
    name = "recipe_search"
    description = (
        "Search for inference recipes by name, model, or description. "
        "Returns matching recipes from all enabled registries. "
        "Example: results = recipe_search(query='llama')"
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query (e.g. 'llama', 'qwen', 'vllm')",
        },
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        return self._run_sparkrun("recipe", "search", query)


class RecipeListTool(SparkrunBaseTool):
    name = "recipe_list"
    description = (
        "List all available inference recipes from enabled registries. "
        "Example: all_recipes = recipe_list() — call with no arguments to list all. "
        "Optionally filter: recipe_list(runtime='vllm') or recipe_list(registry='spark-arena')"
    )
    inputs = {
        "registry": {
            "type": "string",
            "description": "Filter by registry name (optional)",
            "nullable": True,
        },
        "runtime": {
            "type": "string",
            "description": "Filter by runtime (e.g. 'vllm', 'sglang', 'llama-cpp')",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, registry: str | None = None, runtime: str | None = None) -> str:
        args = ["list"]
        if registry:
            args.extend(["--registry", registry])
        if runtime:
            args.extend(["--runtime", runtime])
        return self._run_sparkrun(*args)


class RecipeShowTool(SparkrunBaseTool):
    name = "recipe_show"
    description = (
        "Show detailed information about a specific recipe, including "
        "model, runtime, container image, defaults, and VRAM estimate. "
        "Example: info = recipe_show(recipe_name='qwen3-1.7b-vllm')"
    )
    inputs = {
        "recipe_name": {
            "type": "string",
            "description": "Name of the recipe to inspect",
        },
    }
    output_type = "string"

    def forward(self, recipe_name: str) -> str:
        return self._run_sparkrun("recipe", "show", recipe_name)


class RecipeCreateTool(SparkrunBaseTool):
    name = "recipe_create"
    description = (
        "Create a new sparkrun v2 recipe YAML file. Takes structured inputs "
        "and writes a valid recipe, then validates it."
    )
    inputs = {
        "name": {
            "type": "string",
            "description": "Recipe filename (without .yaml extension)",
        },
        "model": {
            "type": "string",
            "description": "HuggingFace model ID (e.g. 'meta-llama/Llama-2-7b-hf')",
        },
        "runtime": {
            "type": "string",
            "description": "Runtime name (e.g. 'vllm', 'sglang', 'llama-cpp')",
        },
        "container": {
            "type": "string",
            "description": "Container image (e.g. 'scitrera/dgx-spark-vllm:latest')",
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of the recipe",
            "nullable": True,
        },
        "tensor_parallel": {
            "type": "integer",
            "description": "Default tensor parallelism (default: 1)",
            "nullable": True,
        },
        "gpu_memory_utilization": {
            "type": "number",
            "description": "Default GPU memory utilization 0.0-1.0 (default: 0.9)",
            "nullable": True,
        },
        "port": {
            "type": "integer",
            "description": "Default serve port (default: 8000)",
            "nullable": True,
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to write the recipe file (default: current directory)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, name: str, model: str, runtime: str, container: str,
                description: str | None = None, tensor_parallel: int | None = None,
                gpu_memory_utilization: float | None = None, port: int | None = None,
                output_dir: str | None = None) -> str:
        recipe_data = {
            "sparkrun_version": "2",
            "name": name,
            "model": model,
            "runtime": runtime,
            "container": container,
            "defaults": {
                "host": "0.0.0.0",
                "port": port or 8000,
                "tensor_parallel": tensor_parallel or 1,
                "gpu_memory_utilization": gpu_memory_utilization or 0.9,
            },
        }
        if description:
            recipe_data["description"] = description

        # Write to file
        out_dir = Path(output_dir) if output_dir else Path.cwd()
        filename = name if name.endswith((".yaml", ".yml")) else "%s.yaml" % name
        out_path = out_dir / filename

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                yaml.dump(recipe_data, f, default_flow_style=False, sort_keys=False)
        except OSError as e:
            return "Error writing recipe: %s" % e

        # Validate
        validation_result = self._run_sparkrun("recipe", "validate", str(out_path))
        return "Recipe written to %s\n%s" % (out_path, validation_result)


class RecipeValidateTool(SparkrunBaseTool):
    name = "recipe_validate"
    description = "Validate a recipe YAML file for correctness."
    inputs = {
        "recipe_path": {
            "type": "string",
            "description": "Path to the recipe YAML file to validate",
        },
    }
    output_type = "string"

    def forward(self, recipe_path: str) -> str:
        return self._run_sparkrun("recipe", "validate", recipe_path)
