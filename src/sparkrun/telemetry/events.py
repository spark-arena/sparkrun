"""Telemetry event builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import time

from .types import TelemetryEvent
from .util import (
    attr_string,
    int_value,
    parallelism_summary,
    recipe_source,
    registry_summary,
    string_value,
    system_info,
)

_SETUP_STEPS = ("cluster", "ssh", "cx7", "ssh_remesh", "docker", "sudoers", "earlyoom")
_MODEL_QUANTIZATION_KEYS = ("quantization", "quant_bits", "model_dtype", "kv_dtype")
logger = logging.getLogger(__name__)


def build_update_event(
    *,
    command: str,
    old_version: str,
    new_version: str | None,
    upgraded: bool,
    registries: Sequence,
    self_upgrade_attempted: bool = True,
    channel: str | None = None,
    requested_channel: str | None = None,
) -> TelemetryEvent:
    event = {
        "event_type": "update",
        "command": command,
        "old_version": old_version,
        "new_version": new_version or old_version,
        "version_changed": bool(new_version and new_version != old_version),
        "self_upgrade_attempted": bool(self_upgrade_attempted),
        "self_upgrade_succeeded": upgraded,
        "system": system_info(),
    }
    if channel is not None:
        event["channel"] = channel
    if requested_channel is not None:
        event["requested_channel"] = requested_channel
    event.update(registry_summary(registries))
    return event


def build_setup_wizard_event(
    *,
    wizard_run_kind: str,
    results: Mapping[str, str | int | float | bool | None],
    cluster_node_count: int,
    dry_run: bool,
    cx7_detected: bool,
) -> TelemetryEvent:
    return {
        "event_type": "setup_wizard",
        "wizard_run_kind": wizard_run_kind,
        "cluster_node_count": max(0, int(cluster_node_count)),
        "dry_run": bool(dry_run),
        "cx7_detected": bool(cx7_detected),
        "step_choices": [{"step": step, "choice": _setup_step_choice(step, results.get(step))} for step in _SETUP_STEPS],
        "system": system_info(),
    }


def model_quantization_summary(source, overrides=None) -> TelemetryEvent:
    metadata = _model_quantization_metadata(source)
    if metadata is None:
        _populate_recipe_quantization_metadata(source, overrides)
        metadata = _recipe_metadata(source)
    if metadata is None:
        return {}

    event: TelemetryEvent = {}
    quantization = string_value(metadata.get("quantization"))
    quant_bits = int_value(metadata.get("quant_bits"))
    model_dtype = string_value(metadata.get("model_dtype"))
    kv_dtype = string_value(metadata.get("kv_dtype"))
    if quantization is not None:
        event["quantization"] = quantization
    if quant_bits is not None:
        event["quant_bits"] = quant_bits
    if model_dtype is not None:
        event["model_dtype"] = model_dtype
    if kv_dtype is not None:
        event["kv_dtype"] = kv_dtype
    return event


def build_run_event(*, result, recipe, cluster, options) -> TelemetryEvent:
    host_list = _host_list(result)
    rc = int_value(getattr(result, "rc", None), default=0) or 0
    event: TelemetryEvent = {
        "event_type": "run",
        "success": rc == 0,
        "return_code": rc,
        "dry_run": bool(getattr(result, "dry_run", False)),
        "runtime": attr_string(result, "runtime"),
        "executor": attr_string(result, "executor"),
        "scheduler": attr_string(result, "scheduler"),
        "is_solo": bool(getattr(result, "is_solo", False)),
        "model": attr_string(recipe, "model"),
        "recipe_source": recipe_source(recipe, _metadata(result)),
        "parallelism": _parallelism(recipe, options),
        "cluster": _cluster_summary(cluster, host_list),
        "system": system_info(),
    }
    quantization = model_quantization_summary(recipe, getattr(options, "overrides", {}) or {})
    if quantization:
        event["model_quantization"] = quantization
    duration_ms = _duration_ms(getattr(result, "started_at", None))
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    return event


def _setup_step_choice(step: str, value) -> str:
    if value is None:
        return "skipped"
    text = str(value).lower()
    if "skipped" in text:
        return "opted_out"
    if "failed" in text or "partial" in text or "error" in text:
        return "failed"
    if step == "cluster" and ("default" in text or "existing" in text or "already" in text):
        return "already_configured"
    if "already" in text:
        return "already_configured"
    return "opted_in"


def _host_list(result) -> tuple[str, ...]:
    raw = getattr(result, "host_list", ())
    if isinstance(raw, tuple):
        return tuple(str(host) for host in raw)
    if isinstance(raw, list):
        return tuple(str(host) for host in raw)
    return ()


def _metadata(result) -> Mapping[str, str | int | float | bool | None] | None:
    metadata = getattr(result, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else None


def _parallelism(recipe, options) -> TelemetryEvent:
    overrides = getattr(options, "overrides", {}) or {}
    return parallelism_summary(recipe, overrides)


def _cluster_summary(cluster, host_list: Sequence[str]) -> TelemetryEvent:
    chips: dict[tuple[str, str], int] = {}
    gpu_count = 0
    for host in host_list:
        hardware = _hardware_for(cluster, host)
        accelerators = getattr(hardware, "accelerators", ()) if hardware is not None else ()
        for accelerator in accelerators:
            vendor = string_value(getattr(accelerator, "vendor", None)) or "unknown"
            model = string_value(getattr(accelerator, "model", None)) or "unknown"
            count = int_value(getattr(accelerator, "count", None), default=1) or 1
            gpu_count += count
            key = (vendor, model)
            chips[key] = chips.get(key, 0) + count
    return {
        "node_count": len(host_list),
        "gpu_count": gpu_count,
        "gpu_chips": [
            {"vendor": vendor, "model": model, "count": count} for (vendor, model), count in sorted(chips.items(), key=lambda item: item[0])
        ],
    }


def _hardware_for(cluster, host: str):
    hardware_for = getattr(cluster, "hardware_for", None)
    if not callable(hardware_for):
        return None
    try:
        return hardware_for(host)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _duration_ms(started_at) -> int | None:
    if not isinstance(started_at, (int, float)):
        return None
    return max(0, int((time.time() - float(started_at)) * 1000))


def _model_quantization_metadata(source) -> Mapping | None:
    if not isinstance(source, Mapping):
        return None
    nested = source.get("model_quantization")
    if isinstance(nested, Mapping):
        return nested
    return source if any(key in source for key in _MODEL_QUANTIZATION_KEYS) else None


def _recipe_metadata(recipe) -> Mapping | None:
    metadata = getattr(recipe, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else None


def _populate_recipe_quantization_metadata(recipe, overrides) -> None:
    metadata = _recipe_metadata(recipe)
    if metadata is not None and all(key in metadata for key in _MODEL_QUANTIZATION_KEYS):
        return
    estimate_vram = getattr(recipe, "estimate_vram", None)
    if not callable(estimate_vram):
        return
    cli_overrides = dict(overrides) if isinstance(overrides, Mapping) else None
    try:
        estimate_vram(cli_overrides=cli_overrides)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        logger.debug("Could not estimate recipe VRAM for telemetry metadata", exc_info=True)
