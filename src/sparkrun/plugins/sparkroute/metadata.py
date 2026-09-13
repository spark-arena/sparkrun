# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Public model-card metadata published on bridge endpoints.

Implements the producer half of ``SPARKRUN_MODEL_METADATA_CONTRACT.md`` in the
SparkRoute repository.  Sparkrun already knows what each local runtime
serves; this lets the gateway's ``smallest`` / ``largest`` / ``lowest_cost``
selectors use that, instead of an operator hand-authoring size, price, context
and tag values a second time.

The extension is **advisory and additive**.  It cannot register an endpoint,
add a logical model, enable a disabled one, change routing weight, or deliver a
credential — and the gateway discards invalid metadata while keeping the
inference endpoint.  So the rule here is that nothing in this module may make a
valid endpoint unpublishable: :func:`build_model_metadata` returns ``{}`` rather
than raising, and every field is dropped unless it is known and valid.

Three properties are load-bearing:

* **Report, never infer.** An unknown field is omitted; a model-family default
  would be a plausible number with no source, and the gateway would rank real
  deployments against it.  ``context`` in particular is the *effective* runtime
  limit from the recipe's config chain, not the nominal family limit.
* **Offline.** This runs inside the one-shot ``gateway-bridge`` subprocess on
  the inference path, so it reads only what the recipe already carries.  It
  never fetches from HuggingFace — ``model_params`` is present precisely when a
  previous VRAM estimate wrote it back into recipe metadata.
* **Bounded and public.** Sizes are capped per the contract, and only
  presentation-safe values are emitted.  Secrets, credential references,
  endpoint URLs, raw model-card JSON and error text must never appear here;
  everything below is derived from a fixed set of numeric/enum recipe fields.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

#: Contract ceilings.  An endpoint publishes at most this many entries, each
#: with at most this many tags, each tag at most this many bytes.
MAX_METADATA_ENTRIES = 256
MAX_TAGS = 128
MAX_TAG_BYTES = 128

#: Sparkrun workloads run on the operator's own hardware, so there is no
#: per-token charge to report.  The contract distinguishes a *known* free price
#: (numeric zero) from an unknown one (omitted), and this is the former — which
#: is what lets a ``lowest_cost`` selector prefer local inference over a paid
#: API rather than treating it as unpriced.
LOCAL_PRICE_PER_MILLION_TOKENS = 0.0

#: Marks every endpoint this bridge reports, so a selector can express "local
#: only" without enumerating clusters.
LOCAL_TAG = "local"


def build_model_metadata(recipe: Any, served_models: list[str], *, overrides: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return the ``model_metadata`` map for one endpoint.

    Args:
        recipe: The launched recipe, rehydrated from job metadata.  May be
            ``None`` when the job predates recipe-state persistence.
        served_models: Model IDs the live ``/v1/models`` probe returned.  The
            contract requires every metadata key to appear here.

    Returns:
        ``model id -> public metadata``, empty when nothing is known.  Never
        raises: metadata is advisory, and a valid endpoint must stay publishable
        even when its model card cannot be read.
    """
    if recipe is None or not served_models:
        return {}
    try:
        values = _public_values(recipe, overrides)
    except Exception:  # noqa: BLE001 - advisory data must never fail discovery
        logger.debug("Could not derive model metadata for the gateway bridge", exc_info=True)
        return {}
    if not values:
        return {}
    # One workload, one model card: several served IDs are that model under
    # different names (``--served-model-name``), so each carries the same
    # values.  The gateway reduces duplicate reports conservatively anyway.
    return {model: dict(values) for model in served_models[:MAX_METADATA_ENTRIES]}


def _public_values(recipe: Any, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derive the public strategy fields from a recipe, omitting unknowns."""
    metadata = dict(getattr(recipe, "metadata", None) or {})
    values: dict[str, Any] = {}

    size_b = _size_in_billions(metadata.get("model_params"))
    if size_b is not None:
        values["size_b"] = size_b

    context = _context_length(recipe, overrides)
    if context is not None:
        values["context"] = context

    values["input_price"] = LOCAL_PRICE_PER_MILLION_TOKENS
    values["output_price"] = LOCAL_PRICE_PER_MILLION_TOKENS

    tags = _tags(recipe, metadata)
    if tags:
        values["tags"] = tags
    return values


def _size_in_billions(model_params: Any) -> float | None:
    """Convert a raw parameter count to the contract's billions-of-params.

    ``model_params`` is written back into recipe metadata by
    :meth:`Recipe.estimate_vram`, so it is present when a launch estimated VRAM
    and absent otherwise — which is exactly when the size is unknown.
    """
    if model_params is None:
        return None
    try:
        params = float(model_params)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(params) or params <= 0:
        return None
    # Three decimals keeps sub-billion models ("0.5B") distinguishable without
    # emitting float noise the gateway would have to carry.
    size_b = round(params / 1e9, 3)
    return size_b if size_b > 0 else None


def _context_length(recipe: Any, overrides: dict[str, Any] | None = None) -> int | None:
    """Effective runtime context limit, per the contract's explicit warning.

    Read from the recipe's config chain rather than the model's nominal family
    limit: a recipe that pins ``max_model_len: 65536`` on a 262k-context model
    serves 65536, and reporting the larger number would have the gateway route
    requests the runtime rejects.  ``auto`` means the runtime decides, which is
    not something sparkrun can report.
    """
    try:
        config = recipe.build_config_chain()
    except Exception:  # noqa: BLE001 - an unreadable chain just means "unknown"
        logger.debug("Could not build the config chain for model metadata", exc_info=True)
        return None
    raw = (overrides or {}).get("max_model_len", config.get("max_model_len"))
    if raw is None or str(raw).strip().lower() == "auto":
        return None
    try:
        context = int(raw)
    except (TypeError, ValueError):
        return None
    return context if context > 0 else None


def _tags(recipe: Any, metadata: dict[str, Any]) -> list[str]:
    """Presentation-safe tags describing how this endpoint is served.

    Deliberately a small fixed set — locality, runtime, and quantization — all
    of which are enum-ish recipe fields rather than free text, so nothing
    operator-authored can leak through as a tag.
    """
    candidates = [
        LOCAL_TAG,
        str(getattr(recipe, "runtime", "") or ""),
        str(metadata.get("quantization") or ""),
    ]
    tags: list[str] = []
    for candidate in candidates:
        tag = _safe_tag(candidate)
        if tag and tag not in tags:
            tags.append(tag)
    return sorted(tags[:MAX_TAGS])


def _safe_tag(value: str) -> str | None:
    """Normalize one tag, or ``None`` when it is not publishable."""
    tag = value.strip().lower()
    if not tag:
        return None
    # Printable and single-line: a tag is rendered in the gateway's admin
    # console and in selector simulations.
    if any(not character.isprintable() for character in tag):
        return None
    if len(tag.encode()) > MAX_TAG_BYTES:
        return None
    return tag


__all__ = [
    "LOCAL_PRICE_PER_MILLION_TOKENS",
    "LOCAL_TAG",
    "MAX_METADATA_ENTRIES",
    "MAX_TAGS",
    "MAX_TAG_BYTES",
    "build_model_metadata",
]


def reduce_model_metadata(current: dict[str, Any], next_values: dict[str, Any]) -> dict[str, Any]:
    """Conservative model attributes for a generated deployment spanning jobs."""
    result = dict(current)
    for key in ("size_b", "context", "input_price", "output_price"):
        if key in next_values:
            value = next_values[key]
            result[key] = (min if key == "context" else max)(result[key], value) if key in result else value
    tags = sorted(set(result.get("tags", [])) | set(next_values.get("tags", [])))
    if tags:
        result["tags"] = tags[:MAX_TAGS]
    return result
