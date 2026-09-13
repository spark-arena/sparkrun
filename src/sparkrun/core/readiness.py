"""Effective readiness policy: recipe > global config > built-in defaults."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from scitrera_app_framework.api import EnvPlacement, Variables

# Engine initialization, weight loading and graph capture can all precede port
# binding. Generous budgets avoid mistaking a slow engine for a dead workload;
# the probes also check container liveness and honor cancellation.
DEFAULT_PORT_READY_TIMEOUT_S = 1800.0
DEFAULT_HEALTH_READY_TIMEOUT_S = 900.0
OPENAI_CHAT_STREAM = "openai-chat-stream-v1"
OPENAI_RESPONSES_STREAM = "openai-responses-stream-v1"
ANTHROPIC_MESSAGES_STREAM = "anthropic-messages-stream-v1"
INFERENCE_STYLE_APIS = {
    OPENAI_CHAT_STREAM: "chat_completions",
    OPENAI_RESPONSES_STREAM: "responses",
    ANTHROPIC_MESSAGES_STREAM: "messages",
}
INFERENCE_STYLES = frozenset(INFERENCE_STYLE_APIS)


@dataclass(frozen=True)
class ReadinessObserver:
    """Executor-owned observation transport, separate from inference protocol.

    A start boundary describes the observer's timing capability, not policy.
    Future observers may omit it when only inference readiness is available.
    Only the Linux Docker host observer is implemented today.
    """

    adapter: str
    location: str
    start_boundary: str | None = None


DOCKER_HOST_OBSERVER = ReadinessObserver("docker-host-v1", "rank0-host", "docker.State.StartedAt")


class ObservationUnavailable(RuntimeError):
    """The selected observation transport is unsupported on this target."""


@dataclass(frozen=True)
class ReadinessSettings:
    port_timeout_s: float = DEFAULT_PORT_READY_TIMEOUT_S
    health_timeout_s: float = DEFAULT_HEALTH_READY_TIMEOUT_S
    inference: bool = True
    inference_style: str = "auto"
    inference_timeout_s: float = 120.0
    inference_prompt: str = "Reply with exactly: sparkrun-ready"


def _normalize(key: str, value: Any) -> Any:
    if key == "inference_style":
        if not isinstance(value, str) or value not in INFERENCE_STYLES | {"auto"}:
            raise ValueError("readiness.inference_style must be auto or a supported style: %s" % ", ".join(sorted(INFERENCE_STYLES)))
        return value
    if key == "inference":
        if type(value) is not bool:
            raise ValueError("readiness.inference must be a boolean")
        return value
    if key == "inference_prompt":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("readiness.inference_prompt must be a non-empty string")
        return value
    if key in {"port_timeout_s", "health_timeout_s", "inference_timeout_s"}:
        try:
            seconds = float(value)
        except (TypeError, ValueError, OverflowError):
            seconds = math.nan
        if (
            isinstance(value, bool)
            or math.isnan(seconds)
            or (key == "inference_timeout_s" and (not math.isfinite(seconds) or seconds <= 0))
        ):
            raise ValueError(
                "readiness.%s must be %s" % (key, "a finite positive timeout" if key == "inference_timeout_s" else "a timeout")
            )
        return seconds if seconds > 0 else math.inf
    raise ValueError("unknown readiness setting %r" % key)


def parse_recipe_readiness(value: Any) -> dict[str, Any]:
    """Validate the explicit recipe layer without baking in inherited defaults."""
    if not isinstance(value, Mapping):
        raise ValueError("readiness must be a mapping")
    for key, setting in value.items():
        _normalize(key, setting)
    return dict(value)


def resolve_readiness_settings(*, config=None, recipe=None) -> ReadinessSettings:
    """Resolve one immutable policy without mutating the global or recipe layer.

    Existing global settings are permissive: invalid values fall back to the
    built-in default for that field, except explicit inference styles. Explicit
    recipe errors fail at load time.
    Environment variables and runtime flag defaults do not enter this chain.
    """
    defaults = asdict(ReadinessSettings())
    get = getattr(config, "get", None)
    raw_global = get("readiness", {}) if callable(get) else {}
    global_layer = {}
    if isinstance(raw_global, Mapping):
        for key, value in raw_global.items():
            if key == "inference_style":
                # Validate the effective style, not a lower-priority value
                # that the recipe may replace (including with auto).
                global_layer[key] = value
                continue
            try:
                global_layer[key] = _normalize(key, value)
            except ValueError:
                continue
    raw_recipe = getattr(recipe, "readiness", {})
    recipe_layer = {key: _normalize(key, value) for key, value in parse_recipe_readiness(raw_recipe).items()}
    chain = Variables(sources=(recipe_layer, global_layer, defaults), env_placement=EnvPlacement.IGNORED)
    resolved = {key: chain.get(key) for key in defaults}
    resolved["inference_style"] = _normalize("inference_style", resolved["inference_style"])
    return ReadinessSettings(**resolved)


def resolve_inference_style(settings: ReadinessSettings, runtime, *, recipe=None) -> str | None:
    """Constrain effective policy to runtime capabilities; no fourth config layer.

    Runtime styles are preference ordered. No declaration opts out. An explicit
    incompatible style is an error, while auto preserves legacy endpoint checks.
    """
    # The same recipe/runtime declarations feed catalog, discovery and probes.
    # Validate explicit API declarations even when inference probing is disabled.
    native_apis = getattr(runtime, "native_apis", None)
    apis = native_apis(recipe) if recipe is not None and callable(native_apis) else None
    if not settings.inference:
        return None
    styles = getattr(runtime, "readiness_styles", ())
    styles = tuple(s for s in styles if isinstance(s, str) and s in INFERENCE_STYLES) if isinstance(styles, (tuple, list)) else ()
    if apis is not None:
        styles = tuple(style for style in styles if INFERENCE_STYLE_APIS[style] in apis)
    if settings.inference_style == "auto":
        return styles[0] if styles else None
    if settings.inference_style not in styles:
        raise ValueError(
            "Runtime %r does not support readiness.inference_style %r"
            % (getattr(runtime, "runtime_name", "unknown"), settings.inference_style)
        )
    return settings.inference_style


def validate_readiness_policy(*, config=None, recipe=None, runtime=None) -> None:
    """Validate before launch side effects, including supplied/precomputed plans."""
    resolve_inference_style(resolve_readiness_settings(config=config, recipe=recipe), runtime, recipe=recipe)
