# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Passive recipe defaults owned by SparkRoute, never by the serving runtime."""

from __future__ import annotations

from copy import deepcopy
import json
import re
import unicodedata
from typing import Any

# Keep aligned with the gateway config validator; unknown extensions use x-.
CAPABILITIES = frozenset(
    """tools vision audio_input audio_output file_input files developer_messages
structured_outputs json_mode reasoning logprobs seed multiple_choices parallel_tool_calls prediction
service_tier provider_hosted_tools stream_usage stored_completions responses responses_compact
background_responses conversations single_vector_embedding token_counting prompt_caching citations
provider_guardrails provider_prompts""".split()
)
OPERATIONS = frozenset(
    """chat_completions responses responses_compact messages messages_count_tokens
 generate_content stream_generate_content count_tokens converse converse_stream embeddings embed_content batch_embed_contents""".split()
)
PROTECTED_FIELDS = frozenset(
    """extra_body input messages model prompt stream stream_options tool_choice tools
contents system system_instruction systeminstruction instructions history conversation previous_response_id""".split()
)
SELECTOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class SparkrouteRecipeError(ValueError):
    """A recipe's SparkRoute defaults cannot be projected safely."""


def parse_sparkroute(value: Any, *, source: str = "recipe") -> dict[str, Any]:
    """Validate and copy capabilities plus selector → API → JSON parameters."""

    def fail(message):
        raise SparkrouteRecipeError(f"{source}: sparkroute.{message}")

    def json_value(item, depth=0):
        if depth > 32:
            fail("request_profiles JSON nesting exceeds 32 levels")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                fail("request_profiles JSON object keys must be strings")
            for child in item.values():
                json_value(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                json_value(child, depth + 1)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            fail("request_profiles values must be JSON-compatible")

    if not isinstance(value, dict):
        fail("configuration must be a mapping")
    if set(value) - {"capabilities", "request_profiles"}:
        fail("configuration supports only capabilities and request_profiles")
    caps = value.get("capabilities", [])
    if not isinstance(caps, list) or len(caps) > 64:
        fail("capabilities must be a list of at most 64 names")
    for cap in caps:
        if not isinstance(cap, str) or not re.fullmatch(r"[a-z0-9_.-]{1,64}", cap):
            fail("capabilities must contain lowercase capability names")
        if cap not in CAPABILITIES and not (cap.startswith("x-") and len(cap) > 2):
            fail(f"capabilities contains unknown capability {cap!r}; extensions must use x-")
    if len(set(caps)) != len(caps):
        fail("capabilities contains duplicate names")
    profiles = value.get("request_profiles", {})
    if not isinstance(profiles, dict) or len(profiles) > 64:
        fail("request_profiles must map at most 64 selectors to API parameter objects")
    for selector, overrides in profiles.items():
        if not isinstance(selector, str) or not SELECTOR.fullmatch(selector):
            fail("request_profiles selectors must be 1–64 letters/digits, dots, underscores or hyphens, starting with a letter/digit")
        if not isinstance(overrides, dict) or not overrides:
            fail(f"request_profiles.{selector} must map API names to parameter objects")
        for operation, parameters in overrides.items():
            if not isinstance(operation, str) or operation not in OPERATIONS:
                fail(f"request_profiles.{selector} contains an unsupported API operation")
            if not isinstance(parameters, dict) or not parameters or len(parameters) > 64:
                fail(f"request_profiles.{selector}.{operation} requires a JSON object with 1–64 parameters")
            for key in parameters:
                if (
                    not isinstance(key, str)
                    or not key
                    or len(key.encode()) > 128
                    or key.strip() != key
                    or any(unicodedata.category(c) == "Cc" for c in key)
                ):
                    fail(f"request_profiles.{selector}.{operation} contains an invalid parameter name")
                if key.lower() in PROTECTED_FIELDS:
                    fail(f"request_profiles.{selector}.{operation}.{key} is a protected request structure field")
    json_value(profiles)
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
        if len(encoded.encode()) > 256 * 1024:
            fail("configuration exceeds 256 KiB")
    except (TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, SparkrouteRecipeError):
            raise
        fail("request_profiles values must be finite, valid JSON")
    result = {}
    if "capabilities" in value:
        result["capabilities"] = sorted(caps)
    if "request_profiles" in value:
        result["request_profiles"] = deepcopy(profiles)
    return result


def recipe_sparkroute(recipe) -> dict[str, Any]:
    return parse_sparkroute(recipe.plugin_item("sparkroute", {}), source=str(getattr(recipe, "qualified_name", "recipe")))


def catalog_sparkroute(details: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    # Core transports plugin data generically. Only this integration defines
    # the SparkRoute-specific bridge field; other plugins' data stays local.
    from types import SimpleNamespace
    from .metadata import build_model_metadata

    details = dict(details)
    facets = details.get("metadata") or {}
    params_b = facets.get("parameters_b")
    recipe = SimpleNamespace(
        metadata={
            "model_params": params_b * 1e9 if isinstance(params_b, (int, float)) else None,
            "quantization": facets.get("quantization"),
        },
        runtime=details.get("runtime"),
        build_config_chain=lambda: {**(details.get("defaults") or {}), **(overrides or {})},
    )
    details["model_metadata"] = build_model_metadata(recipe, [details.get("model", "")]).get(details.get("model", ""), {})
    items = details.pop("plugin_items", {})
    settings = parse_sparkroute(items.get("sparkroute", {}), source=details.get("name", "recipe"))
    return {
        **details,
        "sparkroute": settings,
        "capabilities": sorted(set(details.get("capabilities", [])) | set(settings.get("capabilities", []))),
    }


class SparkrouteRecipeHandler:
    """Own the top-level recipe item using sparkrun's plugin lifecycle."""

    def parse(self, value, recipe):
        return parse_sparkroute(value, source=str(recipe.qualified_name))

    def validate(self, value, recipe):
        try:
            parse_sparkroute(value)
        except SparkrouteRecipeError as error:
            return [str(error).removeprefix("recipe: sparkroute.")]
        return []

    def export(self, value, recipe):
        return deepcopy(value)


RECIPE_HANDLER = SparkrouteRecipeHandler()
