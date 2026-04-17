"""Benchmarking plugin base class, spec loading, and result export."""

from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

import yaml
from scitrera_app_framework import Plugin, Variables

from sparkrun.core.bootstrap import EXT_BENCHMARKING_FRAMEWORKS

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.launcher import LaunchResult

logger = logging.getLogger(__name__)


class BenchmarkingPlugin(Plugin):
    """Abstract base for benchmarking frameworks (SAF multi-extension plugin).

    Mirrors :class:`~sparkrun.runtimes.base.RuntimePlugin` in structure.
    Subclasses register via entry points under ``sparkrun.benchmarking``.
    """

    eager = False
    framework_name: str = ""
    default_args: dict[str, Any] = {}
    passthrough_args: set[str] = set()

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.benchmarking.%s" % self.framework_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_BENCHMARKING_FRAMEWORKS

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: Logger) -> BenchmarkingPlugin:
        return self

    # --- Framework interface ---

    @abstractmethod
    def check_prerequisites(self) -> list[str]:
        """Check and return missing prerequisites (empty list = ready).

        For example, llama-benchy checks for uvx availability.
        """
        ...

    @abstractmethod
    def build_benchmark_command(
        self,
        target_url: str,
        model: str,
        args: dict[str, Any],
        result_file: str | None = None,
    ) -> list[str]:
        """Build the benchmark command argv list.

        Args:
            target_url: The inference endpoint URL (e.g. http://host:8000/v1).
            model: Model name for the --model flag.
            args: Merged profile + CLI override args.
            result_file: Path where the framework should save results.

        Returns:
            Command argv list suitable for subprocess.
        """
        ...

    def interpret_arg(self, key: str, value: str) -> Any:
        """Interpret a CLI string arg into the correct type.

        Default: comma-separated strings become lists, else coerce_value().
        Subclasses override for framework-specific type handling.
        """
        from sparkrun.utils import coerce_value

        if "," in value:
            return [coerce_value(v.strip()) for v in value.split(",")]
        return coerce_value(value)

    @abstractmethod
    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        """Parse benchmark output into structured results dict.

        Args:
            stdout: Captured standard output.
            stderr: Captured standard error.
            result_file: Path to saved results file (if any).

        Returns:
            Structured results dict.
        """
        ...

    def get_default_args(self) -> dict[str, Any]:
        """Return default benchmark args when no profile is provided."""
        return dict(self.default_args)

    def estimate_test_count(self, args: dict[str, Any]) -> int | None:
        """Estimate the number of test combinations from the args.

        Returns None if the count cannot be determined.  Subclasses should
        override this when the framework's test matrix is predictable.
        """
        return None

    def __repr__(self) -> str:
        return "%s(framework_name=%r)" % (self.__class__.__name__, self.framework_name)


def _build_cluster_meta(recipe, overrides, cluster_id, host_list):
    """Build cluster metadata dict with only non-default parallelism values."""
    from sparkrun.core.parallelism import extract_parallelism_meta

    config_chain = recipe.build_config_chain(overrides)
    meta = {
        "cluster_id": cluster_id,
        "node_count": len(host_list),
    }
    meta.update(extract_parallelism_meta(config_chain))
    return meta


@dataclass
class BenchmarkResult:
    """Result of a benchmark run with output file paths."""

    # benchmark results
    success: bool = False
    results: dict[str, Any] | None = None
    outputs: Optional[dict[str, Any]] = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    # recipe/launch info
    recipe_name: Optional[str] = None
    launch_result: Optional["LaunchResult"] = None

    # store detailed recipe info for "non-launch" scenarios
    recipe: Optional["Recipe"] = None
    overrides: Optional[dict[str, Any]] = None
    cluster_id: Optional[str] = None
    host_list: Optional[list[str]] = None
    container_image: Optional[str] = None

    # benchmark info
    framework: Optional["BenchmarkingPlugin"] = None
    profile: Optional[str] = None
    benchmark_args: Optional[dict[str, Any]] = None

    @property
    def output_csv(self):
        return self.outputs.get("csv") if self.outputs else None

    @output_csv.setter
    def output_csv(self, value):
        if self.outputs is None:
            self.outputs = {}
        self.outputs["csv"] = value

    @property
    def output_json(self):
        return self.outputs.get("json") if self.outputs else None

    @output_json.setter
    def output_json(self, value):
        if self.outputs is None:
            self.outputs = {}
        self.outputs["json"] = value

    @property
    def output_yaml(self):
        return self.outputs.get("yaml") if self.outputs else None

    @output_yaml.setter
    def output_yaml(self, value):
        if self.outputs is None:
            self.outputs = {}
        self.outputs["yaml"] = value

    def generate_metadata(self):
        from sparkrun.models.download import parse_gguf_model_spec
        from sparkrun.utils.cli_formatters import RUNTIME_DISPLAY as _RUNTIME_DISPLAY

        # Use launch_result fields when available, fall back to direct fields
        # (e.g. when --skip-run was used and no launch occurred).
        if launch_result := self.launch_result:
            recipe = launch_result.recipe
            overrides = launch_result.overrides
            cluster_id = launch_result.cluster_id
            host_list = launch_result.host_list
            container_image = launch_result.container_image
            runtime_info = launch_result.runtime_info
        else:
            recipe = self.recipe
            overrides = self.overrides or {}
            cluster_id = self.cluster_id
            host_list = self.host_list or []
            container_image = self.container_image
            runtime_info = {}

        framework = self.framework
        profile = self.profile
        benchmark_args = self.benchmark_args

        if recipe is None:
            logger.warning("No recipe associated with benchmark result.")
            return {}
        
        # Resolve container image to a pinned long-term reference when possible
        container_pinned = False
        recipe_container = container_image or recipe.container
        if launch_result and launch_result.builder:
            try:
                resolved_image, pinned = launch_result.builder.resolve_long_term_image(
                    container_image=launch_result.container_image,
                    runtime_info=launch_result.runtime_info,
                    recipe=recipe,
                )
                if pinned:
                    recipe_container = resolved_image
                    container_pinned = True
                    logger.info("Pinned container image: %s", recipe_container)
            except Exception:
                logger.debug("Long-term image resolution failed", exc_info=True)

        recipe_export = str(recipe.export(overrides=None))
        recipe_hash = hashlib.sha256(recipe_export.encode("utf-8")).hexdigest()

        hf_model = parse_gguf_model_spec(recipe.model)[0]
        model_meta: dict[str, Any] = {}
        if recipe.metadata.get("model_dtype"):
            model_meta["dtype"] = recipe.metadata["model_dtype"]
        if recipe.model_revision:
            model_meta["revision"] = recipe.model_revision
        if recipe.metadata.get("model_params"):
            model_meta["params"] = recipe.metadata["model_params"]
        if recipe.metadata.get("num_layers"):
            model_meta["num_layers"] = recipe.metadata["num_layers"]
        if recipe.metadata.get("num_kv_heads"):
            model_meta["num_kv_heads"] = recipe.metadata["num_kv_heads"]
        if recipe.metadata.get("head_dim"):
            model_meta["head_dim"] = recipe.metadata["head_dim"]
        if recipe.metadata.get("quantization"):
            model_meta["quantization"] = recipe.metadata["quantization"]
        if recipe.metadata.get("quant_bits"):
            model_meta["quant_bits"] = recipe.metadata["quant_bits"]
        if recipe.metadata.get("kv_dtype"):
            model_meta["kv_dtype"] = recipe.metadata["kv_dtype"]

        metadata = {
            "recipe": {
                "name": recipe.name,
                "qualified_name": recipe.qualified_name,
                "type": "sparkrun",
                "model": recipe.model,  # will include quant if applicable
                "hf_model": hf_model,  # will exclude quant if applicable
                "raw_container": recipe.container,
                "container": recipe_container,
                "container_pinned": container_pinned,
                "runtime": _RUNTIME_DISPLAY.get(recipe.runtime, recipe.runtime),
                "runtime_full": recipe.runtime,
                "registry": recipe.source_registry,
                "registry_git": recipe.source_registry_url or "",
                "hash": recipe_hash,
            },
            "timing": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            },
            "cluster": _build_cluster_meta(recipe, overrides, cluster_id, host_list),
            "benchmark": {
                "framework": framework.framework_name if framework else "unknown",
                "profile": profile,
                "args": benchmark_args,
            },
            "model": model_meta,
            "runtime_info": runtime_info,
        }

        # TODO: more care for sparkrun version and/or other metadata
        # noinspection PyBroadException
        try:
            from sparkrun import __version__ as sparkrun_version

            metadata["sparkrun"] = {
                "version": sparkrun_version,
            }
        except Exception:
            pass

        return metadata


def export_results(
    *,
    recipe: Recipe,
    hosts: list[str],
    tp: int,
    cluster_id: str,
    framework_name: str,
    profile_name: str | None,
    args: dict[str, Any],
    results: dict[str, Any],
    output_path: str | Path,
    runtime_info: dict[str, str] | None = None,
) -> Path:
    """Export benchmark results to a YAML file.

    Args:
        recipe: The recipe that was benchmarked.
        hosts: Hosts used for inference.
        tp: Tensor parallelism value.
        cluster_id: Cluster ID from the inference run.
        framework_name: Name of the benchmarking framework.
        profile_name: Profile name (or None if defaults used).
        args: Benchmark args that were used.
        results: Structured results from the framework.
        output_path: Path to write the YAML file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    # noinspection PyProtectedMember
    recipe_text = str(recipe.export(path=None) or "")
    recipe_hash = hashlib.sha256(recipe_text.encode("utf-8")).hexdigest()

    # Build model metadata from recipe metadata (includes auto-detected
    # values written back by Recipe.estimate_vram).
    model_meta: dict[str, Any] = {}
    if recipe.metadata.get("model_dtype"):
        model_meta["dtype"] = recipe.metadata["model_dtype"]
    if recipe.model_revision:
        model_meta["revision"] = recipe.model_revision
    if recipe.metadata.get("model_params"):
        model_meta["params"] = recipe.metadata["model_params"]
    if recipe.metadata.get("num_layers"):
        model_meta["num_layers"] = recipe.metadata["num_layers"]
    if recipe.metadata.get("num_kv_heads"):
        model_meta["num_kv_heads"] = recipe.metadata["num_kv_heads"]
    if recipe.metadata.get("head_dim"):
        model_meta["head_dim"] = recipe.metadata["head_dim"]
    if recipe.metadata.get("quantization"):
        model_meta["quantization"] = recipe.metadata["quantization"]
    if recipe.metadata.get("quant_bits"):
        model_meta["quant_bits"] = recipe.metadata["quant_bits"]
    if recipe.metadata.get("kv_dtype"):
        model_meta["kv_dtype"] = recipe.metadata["kv_dtype"]

    data = {
        "sparkrun_benchmark": {
            "version": "1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recipe": {
                "name": recipe.name,
                "qualified_name": recipe.qualified_name,
                "type": "sparkrun",
                "model": recipe.model,
                "container": recipe.container,
                "runtime": recipe.runtime,
                "registry": recipe.source_registry,
                "registry_git": recipe.source_registry_url or "",
                "text": recipe_text,
                "hash": recipe_hash,
            },
            "model": model_meta,
            "cluster": {
                "tp": tp,
                "cluster_id": cluster_id,
                "runtime_info": runtime_info or {},
            },
            "benchmark": {
                "framework": framework_name,
                "profile": profile_name,
                "args": args,
            },
            "results": results,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)

    logger.info("Results exported to %s", output_path)
    return output_path
