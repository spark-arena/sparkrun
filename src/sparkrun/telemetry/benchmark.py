"""Telemetry event builder for benchmark runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .types import TelemetryEvent
from .events import model_quantization_summary
from .util import parallelism_summary, recipe_source, string_value, system_info

if TYPE_CHECKING:
    from sparkrun.api._benchmark_models import BenchmarkOptions, BenchmarkResult


def build_benchmark_event(*, result: BenchmarkResult, options: BenchmarkOptions, recipe=None) -> TelemetryEvent:
    """Build an anonymous telemetry event for a completed benchmark API call."""
    event_recipe = recipe if recipe is not None else options.recipe
    event: TelemetryEvent = {
        "event_type": "benchmark",
        "success": bool(result.success),
        "category": string_value(result.category),
        "framework": string_value(result.framework),
        "profile": string_value(result.profile),
        "arena": bool(options.arena),
        "dry_run": bool(options.dry_run),
        "skip_run": bool(options.skip_run),
        "resumed": bool(result.resumed),
        "host_count": len(tuple(result.host_list or ())),
        "result_keys": _sorted_keys(result.results),
        "output_formats": _sorted_keys(result.outputs),
        "bench_arg_keys": _sorted_keys(_bench_args(result, options)),
        "recipe_source": recipe_source(event_recipe),
        "submission_id_present": bool(result.submission_id),
        "container_image_sha_pinned": bool(result.container_image_sha_pinned),
        "container_image_longterm_pinned": bool(result.container_image_longterm_pinned),
        "system": system_info(),
    }
    model = string_value(getattr(event_recipe, "model", None))
    if model is not None:
        event["model"] = model
    parallelism = parallelism_summary(event_recipe, options.overrides, fallback_to_overrides=True)
    if parallelism:
        event["parallelism"] = parallelism
    quantization = model_quantization_summary(result.metadata) or model_quantization_summary(event_recipe, options.overrides)
    if quantization:
        event["model_quantization"] = quantization
    return event


def _bench_args(result: BenchmarkResult, options: BenchmarkOptions):
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    value = metadata.get("bench_args")
    return value if isinstance(value, Mapping) else options.bench_args


def _sorted_keys(value, *, limit: int = 50) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    keys = sorted(text for key in value if (text := string_value(key)) is not None)
    return keys[:limit]
