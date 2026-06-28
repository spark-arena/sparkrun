"""Shared helpers for anonymous telemetry events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import platform

from sparkrun.core.parallelism import PARALLELISM_KEYS, extract_parallelism
from sparkrun.core.registry import BOOTSTRAP_REGISTRY_URLS, FALLBACK_DEFAULT_REGISTRIES

from .types import TelemetryEvent


def normalize_url(url: str) -> str:
    """Normalize a registry URL for default-registry comparisons."""
    return url.strip().rstrip("/").removesuffix(".git")


_DEFAULT_REGISTRY_NAMES = {entry.name for entry in FALLBACK_DEFAULT_REGISTRIES}
_DEFAULT_REGISTRY_URLS = {normalize_url(url) for url in BOOTSTRAP_REGISTRY_URLS} | {
    normalize_url(entry.url) for entry in FALLBACK_DEFAULT_REGISTRIES
}


def string_value(value) -> str | None:
    """Return a non-empty stripped string for telemetry dimensions."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def attr_string(source, name: str) -> str | None:
    """Read one optional string-like attribute from a loosely typed domain object."""
    return string_value(getattr(source, name, None))


def attr_bool(source, name: str) -> bool:
    """Read one optional boolean-like attribute from a loosely typed domain object."""
    return bool(getattr(source, name, False))


def int_value(value, *, default: int | None = None) -> int | None:
    """Parse an integer telemetry value, returning the provided default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def system_info() -> TelemetryEvent:
    """Return anonymous OS and architecture telemetry."""
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "architecture": platform.machine(),
    }


def registry_is_default(entry) -> bool:
    """Return whether a configured registry matches a built-in/default registry."""
    name = attr_string(entry, "name")
    url = attr_string(entry, "url")
    if name in _DEFAULT_REGISTRY_NAMES:
        return True
    return normalize_url(url) in _DEFAULT_REGISTRY_URLS if url else False


def registry_summary(registries: Sequence) -> TelemetryEvent:
    """Summarize registry counts without exposing registry names or URLs."""
    total = len(registries)
    enabled = [entry for entry in registries if bool(getattr(entry, "enabled", True))]
    non_default = [entry for entry in registries if not registry_is_default(entry)]
    enabled_non_default = [entry for entry in enabled if not registry_is_default(entry)]
    return {
        "registry_count": total,
        "enabled_registry_count": len(enabled),
        "non_default_registry_count": len(non_default),
        "enabled_non_default_registry_count": len(enabled_non_default),
        "has_non_default_registries": bool(non_default),
    }


def recipe_source(recipe, metadata: Mapping[str, str | int | float | bool | None] | None = None) -> TelemetryEvent:
    """Classify a recipe source without exposing the recipe path, URL, or registry name."""
    if isinstance(recipe, str):
        value = recipe.strip()
        is_url = value.startswith(("http://", "https://"))
        from_spark_arena = value.startswith("@spark-arena/") or "spark-arena.com" in value
        is_file = value.startswith(("/", "./", "../", "~")) or value.endswith((".yaml", ".yml", ".json"))
        if from_spark_arena:
            source_kind = "spark_arena"
        elif is_url:
            source_kind = "url"
        elif is_file:
            source_kind = "file"
        else:
            source_kind = "reference"
        return _source_event(
            source_kind=source_kind,
            from_spark_arena=from_spark_arena,
            from_registry=False,
            from_default_registry=False,
        )

    source_registry = attr_string(recipe, "source_registry")
    source_registry_url = attr_string(recipe, "source_registry_url")
    source_path = attr_string(recipe, "source_path")
    recipe_ref = string_value(metadata.get("recipe_ref")) if metadata is not None else None
    is_url = attr_bool(recipe, "is_url_sourced") or bool(source_path and source_path.startswith(("http://", "https://")))
    from_spark_arena = bool(
        (source_path and "spark-arena.com" in source_path)
        or (recipe_ref and (recipe_ref.startswith("@spark-arena/") or "spark-arena.com" in recipe_ref))
    )
    from_registry = source_registry is not None
    from_default_registry = bool(
        (source_registry in _DEFAULT_REGISTRY_NAMES)
        or (source_registry_url is not None and normalize_url(source_registry_url) in _DEFAULT_REGISTRY_URLS)
    )
    if from_spark_arena:
        source_kind = "spark_arena"
    elif from_registry:
        source_kind = "registry"
    elif source_path and not is_url:
        source_kind = "file"
    elif is_url:
        source_kind = "url"
    else:
        source_kind = "inline"
    return _source_event(
        source_kind=source_kind,
        from_spark_arena=from_spark_arena,
        from_registry=from_registry,
        from_default_registry=from_default_registry,
    )


def parallelism_summary(
    recipe,
    overrides,
    *,
    fallback_to_overrides: bool = False,
) -> TelemetryEvent:
    """Extract parallelism dimensions from a recipe config chain or overrides."""
    override_mapping = dict(overrides) if isinstance(overrides, Mapping) else {}
    build_config_chain = getattr(recipe, "build_config_chain", None)
    if callable(build_config_chain):
        try:
            parallelism = extract_parallelism(build_config_chain(override_mapping))
        except (AttributeError, TypeError, ValueError):
            return _parallelism_from_overrides(override_mapping) if fallback_to_overrides else {}
        return {
            "tensor_parallel": parallelism.tensor_parallel,
            "pipeline_parallel": parallelism.pipeline_parallel,
            "data_parallel": parallelism.data_parallel,
            "expert_parallel": parallelism.expert_parallel,
            "context_parallel": parallelism.context_parallel,
            "world_size": parallelism.world_size(),
        }
    return _parallelism_from_overrides(override_mapping) if fallback_to_overrides else {}


def _parallelism_from_overrides(overrides: Mapping) -> TelemetryEvent:
    event: TelemetryEvent = {}
    for key, _alias in PARALLELISM_KEYS:
        value = int_value(overrides.get(key))
        if value is not None:
            event[key] = value
    return event


def _source_event(
    *,
    source_kind: str,
    from_spark_arena: bool,
    from_registry: bool,
    from_default_registry: bool,
) -> TelemetryEvent:
    return {
        "kind": source_kind,
        "from_file": source_kind == "file",
        "from_spark_arena": from_spark_arena,
        "from_registry": from_registry,
        "from_default_registry": from_default_registry,
        "from_custom_registry": from_registry and not from_default_registry,
    }
