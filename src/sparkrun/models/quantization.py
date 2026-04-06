"""Quantization detection and metadata for HuggingFace models.

Extracts quantization information from multiple sources:
- ``config.json`` → ``quantization_config`` block (standard HF mechanism)
- ``hf_quant_config.json`` (modelopt supplement, e.g. NVIDIA NVFP4 models)
- Recipe-level ``metadata.quantization`` overrides
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QuantizationInfo:
    """Detected quantization metadata for a model."""

    method: str  # "awq", "gptq", "fp8", "nvfp4", "mxfp4", "bitsandbytes", "compressed-tensors", "none"
    bits: int | None  # 4, 8, None (for fp8/mxfp4 where bits is implicit)
    weight_dtype: str  # Effective dtype for VRAM: "awq4", "gptq", "fp8", "nvfp4", "mxfp4", "int4", "int8"
    kv_cache_quant: str | None = None  # From hf_quant_config.json (e.g. "fp8")
    group_size: int | None = None  # Quantization group size


def fetch_hf_quant_config(
    model_id: str,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any] | None:
    """Fetch ``hf_quant_config.json`` from HuggingFace Hub.

    Some models (e.g. NVIDIA modelopt NVFP4) ship a separate
    ``hf_quant_config.json`` with quantization details not present in
    ``config.json``.

    Returns the parsed JSON dict, or ``None`` if the file doesn't exist.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
        import json

        from sparkrun.models.download import _hub_cache

        kwargs: dict[str, Any] = {"repo_id": model_id, "filename": "hf_quant_config.json"}
        if revision:
            kwargs["revision"] = revision
        if cache_dir:
            kwargs["cache_dir"] = _hub_cache(cache_dir)
        try:
            disable_progress_bars()
            config_path = hf_hub_download(**kwargs)
        finally:
            enable_progress_bars()
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Could not fetch hf_quant_config.json for %s: %s", model_id, e)
        return None


def _resolve_from_hf_quant_config(data: dict[str, Any]) -> QuantizationInfo | None:
    """Parse modelopt-style ``hf_quant_config.json``.

    Expected structure::

        {
            "quantization": {
                "quant_algo": "nvfp4",
                "kv_cache_quant_algo": "fp8",
                "group_size": 64
            }
        }

    When ``quant_algo`` is ``"mixed_precision"``, the dominant algorithm
    is resolved from ``quantized_layers`` by counting per-layer
    ``quant_algo`` occurrences.
    """
    quant_block = data.get("quantization")
    if not isinstance(quant_block, dict):
        return None

    algo = str(quant_block.get("quant_algo", "")).lower().strip()
    if not algo:
        return None

    kv_cache_algo = quant_block.get("kv_cache_quant_algo")
    if isinstance(kv_cache_algo, str):
        kv_cache_algo = kv_cache_algo.lower().strip() or None
    else:
        kv_cache_algo = None

    group_size = quant_block.get("group_size")

    # For mixed precision, resolve the dominant algo from quantized_layers
    if algo == "mixed_precision":
        algo, group_size = _resolve_mixed_precision(quant_block)
        if not algo:
            return None

    if group_size is not None:
        group_size = int(group_size)

    # Map algo to weight_dtype and bits
    weight_dtype, bits = _algo_to_dtype_bits(algo)

    return QuantizationInfo(
        method=algo,
        bits=bits,
        weight_dtype=weight_dtype,
        kv_cache_quant=kv_cache_algo,
        group_size=group_size,
    )


def _resolve_mixed_precision(quant_block: dict[str, Any]) -> tuple[str, int | None]:
    """Resolve the dominant quantization algorithm from mixed-precision layers.

    Counts ``quant_algo`` occurrences in ``quantized_layers`` and returns
    the most common algorithm (lowercased) and its most common ``group_size``.
    """
    layers = quant_block.get("quantized_layers")
    if not isinstance(layers, dict) or not layers:
        return "", None

    algo_counts: dict[str, int] = {}
    group_sizes: dict[str, list[int]] = {}
    for _name, layer_cfg in layers.items():
        if not isinstance(layer_cfg, dict):
            continue
        la = str(layer_cfg.get("quant_algo", "")).lower().strip()
        if la:
            algo_counts[la] = algo_counts.get(la, 0) + 1
            gs = layer_cfg.get("group_size")
            if gs is not None:
                group_sizes.setdefault(la, []).append(int(gs))

    if not algo_counts:
        return "", None

    dominant = max(algo_counts, key=algo_counts.get)
    # Pick the most common group_size for the dominant algo
    gs_list = group_sizes.get(dominant)
    dominant_gs: int | None = None
    if gs_list:
        dominant_gs = max(set(gs_list), key=gs_list.count)

    return dominant, dominant_gs


def _algo_to_dtype_bits(algo: str) -> tuple[str, int | None]:
    """Map a quantization algorithm name to (weight_dtype, bits)."""
    if algo in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        return "fp8", 8
    if algo == "nvfp4":
        return "nvfp4", 4
    if algo == "mxfp4":
        return "mxfp4", 4
    if algo == "awq":
        return "awq4", 4
    if algo in ("gptq", "marlin"):
        return "gptq", 4
    if algo == "int4":
        return "int4", 4
    if algo == "int8":
        return "int8", 8
    # Fallback: use the algo name itself as dtype
    return algo, None


def _resolve_from_quantization_config(qc: dict[str, Any]) -> QuantizationInfo | None:
    """Resolve quantization from a HuggingFace ``quantization_config`` block.

    Enhanced version of the former ``_resolve_quant_dtype()`` that returns
    full :class:`QuantizationInfo` instead of just a dtype string.
    """
    method = str(qc.get("quant_method", "")).lower().strip()
    if not method:
        return None

    if method == "fp8":
        return QuantizationInfo(method="fp8", bits=8, weight_dtype="fp8")

    bits = qc.get("bits")

    if method == "awq":
        b = 4 if bits is None else int(bits)
        return QuantizationInfo(method="awq", bits=b, weight_dtype="awq4" if b == 4 else "awq%d" % b)

    if method in ("gptq", "marlin"):
        b = 4 if bits is None else int(bits)
        return QuantizationInfo(method=method, bits=b, weight_dtype="gptq" if b == 4 else "int%d" % b)

    if method == "bitsandbytes":
        if qc.get("load_in_4bit") or qc.get("quant_type") == "nf4":
            return QuantizationInfo(method="bitsandbytes", bits=4, weight_dtype="int4")
        if qc.get("load_in_8bit"):
            return QuantizationInfo(method="bitsandbytes", bits=8, weight_dtype="int8")
        return None

    if method == "mxfp4":
        return QuantizationInfo(method="mxfp4", bits=4, weight_dtype="mxfp4")

    if method == "nvfp4":
        return QuantizationInfo(method="nvfp4", bits=4, weight_dtype="nvfp4")

    if method == "compressed-tensors":
        return _resolve_compressed_tensors(qc)

    if method in ("auto-round", "autoround", "auto_round"):
        b = 4 if bits is None else int(bits)
        data_type = str(qc.get("data_type", "int")).lower()
        if data_type == "int":
            wd = "int%d" % b
        else:
            wd = "fp%d" % b
        return QuantizationInfo(
            method="auto-round",
            bits=b,
            weight_dtype=wd,
            group_size=int(qc["group_size"]) if qc.get("group_size") is not None else None,
        )

    return None


def _resolve_compressed_tensors(qc: dict[str, Any]) -> QuantizationInfo:
    """Resolve quantization details from a compressed-tensors config.

    Inspects ``config_groups.*.weights`` to derive the actual format/bits.
    The packing format is resolved from multiple locations (first match wins):
    ``weights.strategy``, ``group.format``, top-level ``qc.format``.

    For mixed-precision configs (multiple groups), groups that use a broad
    catch-all target (e.g. ``"Linear"``) are preferred over groups that
    target specific layers, since the catch-all typically represents the
    dominant quantization (e.g. int4 for language weights vs fp8 for a
    smaller vision encoder).

    Falls back to ``"compressed-tensors"`` as method if unrecognizable.
    """
    top_format = str(qc.get("format", "")).lower()
    config_groups = qc.get("config_groups")
    if isinstance(config_groups, dict):
        # Sort groups so broad catch-all targets (e.g. "Linear") come first.
        # A group whose targets list contains a short non-dotted name is
        # likely a catch-all; groups targeting "model.visual.blocks.0.attn.qkv"
        # are layer-specific.  This ensures mixed-precision configs resolve to
        # the dominant quantization.
        def _group_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
            _name, grp = item
            if not isinstance(grp, dict):
                return (1, _name)
            targets = grp.get("targets", [])
            has_catchall = any(
                isinstance(t, str) and "." not in t
                for t in targets
            )
            return (0 if has_catchall else 1, _name)

        sorted_groups = sorted(config_groups.items(), key=_group_sort_key)
        for _group_name, group in sorted_groups:
            if not isinstance(group, dict):
                continue
            weights = group.get("weights")
            if not isinstance(weights, dict):
                continue
            # Resolve packing format: weights.strategy > group.format > top-level format
            fmt = str(weights.get("strategy", "")).lower()
            if fmt in ("", "group", "tensor"):
                fmt = str(group.get("format", "")).lower() or top_format
            wtype = str(weights.get("type", "")).lower()
            num_bits = weights.get("num_bits")
            group_size = weights.get("group_size")
            # pack-quantized with int type is typically AWQ-style
            if fmt == "pack-quantized" and wtype == "int" and num_bits is not None:
                b = int(num_bits)
                if b == 4:
                    return QuantizationInfo(
                        method="awq",
                        bits=4,
                        weight_dtype="awq4",
                        group_size=int(group_size) if group_size is not None else None,
                    )
                return QuantizationInfo(method="compressed-tensors", bits=b, weight_dtype="int%d" % b)
            if fmt == "channel-quantized" and wtype == "int" and num_bits is not None:
                b = int(num_bits)
                return QuantizationInfo(method="compressed-tensors", bits=b, weight_dtype="int%d" % b)
            if wtype == "float" and num_bits is not None:
                b = int(num_bits)
                if b == 8:
                    return QuantizationInfo(method="fp8", bits=8, weight_dtype="fp8")
                return QuantizationInfo(method="compressed-tensors", bits=b, weight_dtype="fp%d" % b)

    # Fallback: unrecognizable structure
    return QuantizationInfo(method="compressed-tensors", bits=None, weight_dtype="compressed-tensors")


def _gguf_normalize_quant(name: str) -> str | None:
    """Normalize a GGUF quant variant name to a recognized dtype.

    Tries an exact match first (e.g. ``q4_k_m``), then strips the
    ``_s``/``_m``/``_l`` mix suffix to try the base K-quant name
    (e.g. ``q4_k_m`` → ``q4_k``).

    Returns the matched dtype string or ``None`` if unrecognized.
    """
    from sparkrun.models.vram import bytes_per_element

    qlower = name.lower().strip().replace("-", "_")
    # Exact match (e.g. q4_k_m, q8_0, iq4_xs)
    if bytes_per_element(qlower) is not None:
        return qlower
    # Strip mix suffix: q3_k_m -> q3_k, q5_k_s -> q5_k
    if qlower.endswith(("_s", "_m", "_l")):
        base = qlower[:-2]
        if bytes_per_element(base) is not None:
            return base
    return None


def resolve_from_gguf(model_id: str) -> QuantizationInfo | None:
    """Derive quantization info from a GGUF model spec's quant variant.

    GGUF models use colon syntax: ``Qwen/Qwen3-1.7B-GGUF:Q4_K_M``.
    The suffix after ``:`` is the quantization variant.

    Returns ``QuantizationInfo`` if a recognized GGUF quant suffix is found,
    or ``None`` for non-GGUF models.
    """
    if ":" not in model_id:
        return None
    _repo, quant = model_id.rsplit(":", 1)
    if not quant:
        return None
    dtype = _gguf_normalize_quant(quant)
    if dtype is not None:
        return QuantizationInfo(method="gguf", bits=None, weight_dtype=dtype)
    return None


def resolve_quantization(
    *,
    hf_config: dict[str, Any] | None = None,
    hf_quant_config: dict[str, Any] | None = None,
    recipe_quant: str | None = None,
    model_id: str | None = None,
) -> QuantizationInfo | None:
    """Merge quantization info from all available sources.

    Priority for method/bits:
      1. ``recipe_quant`` — explicit recipe author override (``metadata.quantization``)
      2. ``hf_config`` → ``quantization_config`` block (standard HF)
      3. ``hf_quant_config`` (modelopt supplement)
      4. ``model_id`` — GGUF colon syntax (e.g. ``repo:Q4_K_M``)

    The ``hf_quant_config`` is checked last for method because ``config.json``
    is the standard HF mechanism, but ``hf_quant_config.json`` is the *only*
    source for ``kv_cache_quant_algo`` and ``group_size`` so those are always
    picked up when the file exists.
    """
    info_from_config: QuantizationInfo | None = None
    info_from_quant_file: QuantizationInfo | None = None

    # Source 3: hf_quant_config.json
    if hf_quant_config:
        info_from_quant_file = _resolve_from_hf_quant_config(hf_quant_config)

    # Source 2: config.json quantization_config
    if hf_config:
        qc = hf_config.get("quantization_config")
        if isinstance(qc, dict):
            info_from_config = _resolve_from_quantization_config(qc)

    # Source 1 (strongest): recipe metadata.quantization
    if recipe_quant:
        rq = recipe_quant.lower().strip()
        if rq not in ("none", "auto", ""):
            weight_dtype, bits = _algo_to_dtype_bits(rq)
            result = QuantizationInfo(method=rq, bits=bits, weight_dtype=weight_dtype)
            # Absorb kv_cache_quant and group_size from hf_quant_config if available
            if info_from_quant_file:
                result.kv_cache_quant = info_from_quant_file.kv_cache_quant
                result.group_size = info_from_quant_file.group_size
            return result

    # Pick the best method source
    result = info_from_config or info_from_quant_file

    # Source 4 (weakest): GGUF model spec
    if result is None and model_id:
        result = resolve_from_gguf(model_id)

    if result is None:
        return None

    # If we got method from config.json but hf_quant_config has supplemental info, merge
    if result is info_from_config and info_from_quant_file:
        if result.kv_cache_quant is None:
            result.kv_cache_quant = info_from_quant_file.kv_cache_quant
        if result.group_size is None:
            result.group_size = info_from_quant_file.group_size

    return result
