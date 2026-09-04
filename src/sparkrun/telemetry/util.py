"""Shared helpers for anonymous telemetry events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import logging
from pathlib import PurePath
import platform

from sparkrun.core.parallelism import PARALLELISM_KEYS, extract_parallelism
from sparkrun.core.registry import BOOTSTRAP_REGISTRY_URLS, FALLBACK_DEFAULT_REGISTRIES, _normalize_registry_url

from .types import TelemetryEvent

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """Normalize a registry URL for default-registry comparisons.

    Delegates to the canonical registry normalizer rather than keeping a
    weaker copy: this only ever decides whether a registry is one we ship (a
    count in :func:`registry_summary`, a ``source_kind`` label), so a spelling
    it failed to recognise inflated ``non_default_registry_count`` for users
    whose URLs are perfectly ordinary. No URL is emitted either way.
    """
    return _normalize_registry_url(url)


_DEFAULT_REGISTRY_NAMES = {entry.name for entry in FALLBACK_DEFAULT_REGISTRIES}
_DEFAULT_REGISTRY_URLS = {normalize_url(url) for url in BOOTSTRAP_REGISTRY_URLS} | {
    normalize_url(entry.url) for entry in FALLBACK_DEFAULT_REGISTRIES
}


#: The only types a telemetry dimension may be rendered from.  Deliberately a
#: closed list: every value here is read off a loosely typed domain object with
#: ``getattr``, so an unconditional ``str()`` renders whatever the caller
#: happened to be holding — which is how ``"<MagicMock name='mock.category'
#: id=...>"`` once reached the collector as a benchmark's category.  A dimension
#: that goes missing is a far cheaper failure than one carrying an object repr.
#:
#: ``os.PathLike`` is *not* used for the path case: it is a runtime protocol
#: satisfied by anything with ``__fspath__``, which ``MagicMock`` synthesizes —
#: it would readmit exactly what this excludes.
_SCALAR_TYPES = (str, bool, int, float, PurePath, Enum)


def _scalar(value):
    """Return *value* narrowed to a JSON-safe scalar, or None if it is not one."""
    if value is None or not isinstance(value, _SCALAR_TYPES):
        return None
    if isinstance(value, Enum):
        inner = value.value
        return inner if isinstance(inner, (str, bool, int, float)) else None
    return value


def string_value(value) -> str | None:
    """Return a non-empty stripped string for telemetry dimensions.

    Anything that is not a plain scalar yields ``None`` rather than its repr.
    """
    scalar = _scalar(value)
    if scalar is None:
        if value is not None:
            logger.debug("Dropping non-scalar telemetry value of type %s", type(value).__name__)
        return None
    text = str(scalar).strip()
    return text or None


# --------------------------------------------------------------------------
# Model identifiers
# --------------------------------------------------------------------------

#: Placeholders substituted for a model identifier that must not be sent.
#: They are deliberately distinct so the collected data can tell "this was a
#: local path" from "this was a private repo" from "we could not establish
#: visibility" — all three are useful signal, none of them names the model.
MODEL_LOCAL_PATH = "<local-path>"
MODEL_PRIVATE = "<hf-private>"
MODEL_UNKNOWN = "<unknown-visibility>"


def _looks_like_local_path(model: str) -> bool:
    """True when *model* refers to weights on disk rather than a Hub repo id.

    A Hub repo id is ``org/name`` — at most one slash, no leading separator,
    no drive letter, no ``~``.  Anything absolute, relative-with-dots, or
    Windows-drive-prefixed is a filesystem path.
    """
    if model.startswith(("/", "~", "./", "../", "\\")):
        return True
    # Windows drive letter, e.g. C:\models\foo
    if len(model) >= 2 and model[1] == ":" and model[0].isalpha():
        return True
    return model.count("/") > 1


def model_identifier(model, *, revision=None, probe: bool = True) -> str | None:
    """Return the model identifier safe to send, or a coarse placeholder.

    The raw value is emitted **only** for a repo confirmed publicly readable
    on the Hub.  Everything else collapses to one of :data:`MODEL_LOCAL_PATH`,
    :data:`MODEL_PRIVATE`, or :data:`MODEL_UNKNOWN`.

    This fails closed: an unresolvable lookup yields ``MODEL_UNKNOWN`` rather
    than the model name, so being offline or rate-limited can never turn into
    disclosure.

    Args:
        model: The recipe's ``model`` value.
        revision: Optional pinned revision, forwarded to the visibility probe.
        probe: When False, skip the network lookup entirely and report
            ``MODEL_UNKNOWN`` for anything not obviously a local path.  Used
            when telemetry is disabled, so an opted-out user never pays for a
            lookup that exists only to serve telemetry.
    """
    text = string_value(model)
    if text is None:
        return None

    if _looks_like_local_path(text):
        return MODEL_LOCAL_PATH

    if not probe:
        return MODEL_UNKNOWN

    from sparkrun.models.vram import (
        MODEL_VISIBILITY_PRIVATE,
        MODEL_VISIBILITY_PUBLIC,
        fetch_model_visibility,
    )

    visibility = fetch_model_visibility(text, string_value(revision))
    if visibility == MODEL_VISIBILITY_PUBLIC:
        return text
    if visibility == MODEL_VISIBILITY_PRIVATE:
        return MODEL_PRIVATE
    return MODEL_UNKNOWN


def attr_string(source, name: str) -> str | None:
    """Read one optional string-like attribute from a loosely typed domain object."""
    return string_value(getattr(source, name, None))


def attr_bool(source, name: str) -> bool:
    """Read one optional boolean-like attribute from a loosely typed domain object."""
    return bool(getattr(source, name, False))


def int_value(value, *, default: int | None = None) -> int | None:
    """Parse an integer telemetry value, returning the provided default on failure.

    Narrowed to scalars first, for the reason :data:`_SCALAR_TYPES` documents:
    ``int()`` on an arbitrary object invokes ``__int__``, which ``MagicMock``
    synthesizes — so an unnarrowed conversion reports a confident ``1`` for a
    test double.
    """
    try:
        return int(_scalar(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def system_info() -> TelemetryEvent:
    """Return anonymous OS and architecture telemetry."""
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "architecture": platform.machine(),
    }


def registry_is_plugin_declared(entry) -> bool:
    """Return whether a registry came from a plugin declaration, not the user."""
    return bool(attr_string(entry, "declared_by"))


def registry_is_default(entry) -> bool:
    """Return whether a configured registry matches a built-in/default registry.

    A plugin-declared registry counts as default.  ``non_default_*`` means "the
    user added a third-party registry", and a first-party plugin contributing
    its own is not that — without this, enabling a shipped plugin would flip
    ``has_non_default_registries`` and silently change the meaning of a metric
    already in published series.  They are counted separately below instead.
    """
    if registry_is_plugin_declared(entry):
        return True
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
    plugin_declared = [entry for entry in registries if registry_is_plugin_declared(entry)]
    return {
        "registry_count": total,
        "enabled_registry_count": len(enabled),
        "non_default_registry_count": len(non_default),
        "enabled_non_default_registry_count": len(enabled_non_default),
        "has_non_default_registries": bool(non_default),
        "plugin_registry_count": len(plugin_declared),
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
