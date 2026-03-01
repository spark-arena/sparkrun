"""VRAM estimation for inference workloads on DGX Spark systems."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Bytes per element for common dtypes
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "float16": 2.0,
    "fp16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "fp8_e5m2": 1.0,
    "fp8_e4m3": 1.0,
    "int4": 0.5,
    "nvfp4": 0.5,
    "awq4": 0.5,
    "awq8": 1.0,
    "gptq": 0.5,
    "mxfp4": 0.5,
    # TODO: GGUF quants
    'q3_k_m': 0.4,
    'q4_k_m': 0.608,
    'q6_k': 0.823,
    'q8_0': 1.0,
}

# Shorthand suffixes for parameter counts
_PARAM_SUFFIXES = {
    "T": 1_000_000_000_000,
    "B": 1_000_000_000,
    "M": 1_000_000,
    "K": 1_000,
}

# DGX Spark: unified memory shared between CPU and GPU.
# Total system memory is ~128 GB (127601452 KiB ≈ 121.7 GiB).
# Usable GPU memory depends on gpu_memory_utilization and OS overhead.
# We use 121 GiB as an "available for inference" figure.
DGX_SPARK_VRAM_GB = 121.0


@dataclass
class VRAMEstimate:
    """Result of a VRAM estimation."""

    model_weights_gb: float
    kv_cache_per_token_bytes: float | None
    kv_cache_total_gb: float | None
    total_per_gpu_gb: float
    max_model_len: int | None
    tensor_parallel: int
    warnings: list[str] = field(default_factory=list)

    # Input parameters used (for display)
    model_params: int | None = None
    model_dtype: str | None = None
    kv_dtype: str | None = None
    num_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None

    # GPU memory budget fields
    gpu_memory_utilization: float | None = None
    usable_gpu_memory_gb: float | None = None
    available_kv_gb: float | None = None
    max_context_tokens: int | None = None
    context_multiplier: float | None = None

    @property
    def fits_dgx_spark(self) -> bool:
        """Whether the estimated VRAM fits within DGX Spark GPU memory."""
        return self.total_per_gpu_gb <= DGX_SPARK_VRAM_GB


def bytes_per_element(dtype: str) -> float | None:
    """Return bytes per element for a dtype string, or None if unknown."""
    return _DTYPE_BYTES.get(dtype.lower().strip().replace("-", "_"))


def parse_param_count(value: int | float | str) -> int | None:
    """Parse a parameter count from integer or shorthand string.

    Supports: 7000000000, 7.0e9, "7B", "70B", "0.5B", "480M", "7_000_000_000"

    Returns:
        Parsed integer count, or None if unparseable.
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().replace("_", "")
        # Try numeric parse first
        try:
            return int(float(value))
        except ValueError:
            pass
        # Try suffix parse (case-insensitive suffix)
        for suffix, multiplier in _PARAM_SUFFIXES.items():
            if value.upper().endswith(suffix):
                try:
                    num = float(value[: -len(suffix)])
                    return int(num * multiplier)
                except ValueError:
                    pass
    return None


def fetch_model_config(
        model_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
) -> dict[str, Any] | None:
    """Fetch model config.json from HuggingFace Hub without downloading weights.

    Args:
        model_id: HuggingFace model identifier.
        revision: Optional revision (branch, tag, or commit hash).
        cache_dir: Optional HuggingFace cache directory override.

    Returns the config dict or None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
        import json

        kwargs: dict[str, Any] = {"repo_id": model_id, "filename": "config.json"}
        if revision:
            kwargs["revision"] = revision
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        config_path = hf_hub_download(**kwargs)
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Could not fetch HF config for %s: %s", model_id, e)
        return None


def fetch_safetensors_size(
        model_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
) -> int | None:
    """Fetch total parameter storage size from safetensors index metadata.

    Downloads only the small ``model.safetensors.index.json`` file, which
    contains a ``metadata.total_size`` field giving the total bytes of all
    parameter tensors on disk.

    Args:
        model_id: HuggingFace model identifier.
        revision: Optional revision (branch, tag, or commit hash).
        cache_dir: Optional HuggingFace cache directory override.

    Returns:
        Total size in bytes, or ``None`` if unavailable.
    """
    try:
        from huggingface_hub import hf_hub_download
        import json

        kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "filename": "model.safetensors.index.json",
        }
        if revision:
            kwargs["revision"] = revision
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        index_path = hf_hub_download(**kwargs)
        with open(index_path) as f:
            index = json.load(f)
        total_size = index.get("metadata", {}).get("total_size")
        if total_size is not None:
            return int(total_size)
    except Exception as e:
        logger.debug("Could not fetch safetensors index for %s: %s", model_id, e)
    return None


def _extract_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture info from a single config dict (top-level or nested)."""
    info: dict[str, Any] = {}

    # dtype: check torch_dtype first, then dtype
    for key in ("torch_dtype", "dtype"):
        if key in cfg:
            info["model_dtype"] = cfg[key]
            break

    # num_layers: varies by architecture
    for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if key in cfg:
            info["num_layers"] = cfg[key]
            break

    # num_kv_heads: GQA architectures first, then MHA fallback
    for key in ("num_key_value_heads", "num_kv_heads"):
        if key in cfg:
            info["num_kv_heads"] = cfg[key]
            break
    if "num_kv_heads" not in info:
        for key in ("num_attention_heads", "n_head"):
            if key in cfg:
                info["num_kv_heads"] = cfg[key]
                break

    # head_dim: explicit or derived from hidden_size / num_attention_heads
    if "head_dim" in cfg:
        info["head_dim"] = cfg["head_dim"]
    elif "hidden_size" in cfg:
        # Try all known attention head key names for derivation
        for key in ("num_attention_heads", "n_head"):
            if key in cfg and cfg[key] > 0:
                info["head_dim"] = cfg["hidden_size"] // cfg[key]
                break

    return info


def extract_model_info(hf_config: dict[str, Any]) -> dict[str, Any]:
    """Extract model architecture info from a HuggingFace config.json.

    Handles naming variants across architectures (Llama, Qwen, Mistral, GPT-NeoX, etc.).
    For multimodal models that nest text architecture under ``text_config``,
    ``llm_config``, or ``language_config``, those nested dicts are checked
    as a fallback when top-level extraction yields incomplete results.

    Returns:
        Dict with keys: model_dtype, num_layers, num_kv_heads, head_dim (present only if found).
    """
    info = _extract_from_config(hf_config)

    # For multimodal / composite models the text architecture lives in a
    # nested sub-config.  Check common nesting keys when the top-level
    # extraction is missing architecture fields.
    _NEEDED = {"model_dtype", "num_layers", "num_kv_heads", "head_dim"}
    if not _NEEDED.issubset(info.keys()):
        for nested_key in ("text_config", "llm_config", "language_config"):
            nested = hf_config.get(nested_key)
            if isinstance(nested, dict):
                nested_info = _extract_from_config(nested)
                # Fill in only missing fields (top-level takes precedence)
                for k, v in nested_info.items():
                    if k not in info:
                        info[k] = v
                break  # only use the first matching nested config

    return info


def estimate_vram(
        *,
        model_params: int | None = None,
        model_dtype: str | None = None,
        kv_dtype: str | None = None,
        num_layers: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_model_len: int | None = None,
        tensor_parallel: int = 1,
        model_vram: float | None = None,
        kv_vram_per_token: float | None = None,
        gpu_memory_utilization: float | None = None,
) -> VRAMEstimate:
    """Estimate VRAM usage for an inference workload.

    Args:
        model_params: Total parameter count.
        model_dtype: Weight dtype (e.g. "float16", "int4", "fp8").
        kv_dtype: KV cache dtype (default: "bfloat16").
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        max_model_len: Maximum sequence length for KV cache sizing.
        tensor_parallel: Tensor parallelism degree.
        model_vram: Direct override for model weight VRAM in GB (not scaled by TP).
        kv_vram_per_token: Direct override for KV cache in GB per token (scaled by max_model_len and TP).
        gpu_memory_utilization: Fraction of GPU memory the runtime is allowed to use (e.g. 0.9).

    Returns:
        VRAMEstimate with per-GPU totals and any warnings.
    """
    warnings: list[str] = []
    kv_dtype = kv_dtype or "bfloat16"
    tp = max(tensor_parallel, 1)

    # --- Model weight VRAM ---
    model_weights_gb = 0.0
    if model_vram is not None:
        # Direct override: user provides total model VRAM for single-GPU equivalent
        model_weights_gb = model_vram
    elif model_params and model_dtype:
        bpe = bytes_per_element(model_dtype)
        if bpe is not None:
            model_weights_gb = model_params * bpe / (1024 ** 3)
        else:
            warnings.append("Unknown dtype %r; cannot estimate model weight VRAM" % model_dtype)
    elif not model_params:
        warnings.append("model_params not available; model weight estimate is zero")
    elif not model_dtype:
        warnings.append("model_dtype not available; model weight estimate is zero")

    # --- KV cache VRAM ---
    kv_cache_per_token_bytes: float | None = None
    kv_cache_total_gb: float | None = None

    if kv_vram_per_token is not None:
        # Direct override: user provides GB per token
        kv_cache_per_token_bytes = kv_vram_per_token * (1024 ** 3)  # convert to bytes for display
        if max_model_len:
            kv_cache_total_gb = kv_vram_per_token * max_model_len
    elif num_layers and num_kv_heads and head_dim:
        kv_bpe = bytes_per_element(kv_dtype)
        if kv_bpe is not None:
            # Per token: 2 (K+V) * num_layers * num_kv_heads * head_dim * bytes
            kv_cache_per_token_bytes = 2.0 * num_layers * num_kv_heads * head_dim * kv_bpe
            if max_model_len:
                kv_cache_total_gb = kv_cache_per_token_bytes * max_model_len / (1024 ** 3)
        else:
            warnings.append("Unknown KV cache dtype %r" % kv_dtype)
    else:
        missing = []
        if not num_layers:
            missing.append("num_layers")
        if not num_kv_heads:
            missing.append("num_kv_heads")
        if not head_dim:
            missing.append("head_dim")
        warnings.append(
            "Missing architecture info (%s); KV cache estimate unavailable" % ", ".join(missing)
        )

    # --- Per-GPU total ---
    # Model weights split across TP GPUs
    per_gpu_weights_gb = model_weights_gb / tp

    # KV heads also split across TP GPUs (each GPU handles num_kv_heads/tp heads)
    per_gpu_kv_gb = (kv_cache_total_gb / tp) if kv_cache_total_gb else 0.0

    total_per_gpu_gb = per_gpu_weights_gb + per_gpu_kv_gb

    # --- GPU memory budget analysis ---
    # Compute how much memory the runtime can actually use, and how much
    # is left for KV cache after model weights are loaded.
    usable_gpu_memory_gb: float | None = None
    available_kv_gb: float | None = None
    max_context_tokens: int | None = None
    context_multiplier: float | None = None

    if gpu_memory_utilization is not None and gpu_memory_utilization > 0:
        usable_gpu_memory_gb = DGX_SPARK_VRAM_GB * gpu_memory_utilization
        available_kv_gb = usable_gpu_memory_gb - per_gpu_weights_gb

        if available_kv_gb < 0:
            warnings.append(
                "Model weights (%.1f GB) exceed usable GPU memory "
                "(%.1f GB at %.0f%% utilization)"
                % (per_gpu_weights_gb, usable_gpu_memory_gb,
                   gpu_memory_utilization * 100)
            )
            available_kv_gb = 0.0

        # Estimate max context tokens that fit in available KV space
        if kv_cache_per_token_bytes and kv_cache_per_token_bytes > 0:
            per_gpu_kv_per_token_gb = (kv_cache_per_token_bytes / tp) / (1024 ** 3)
            if per_gpu_kv_per_token_gb > 0:
                max_context_tokens = int(available_kv_gb / per_gpu_kv_per_token_gb)

                if max_model_len and max_model_len > 0:
                    context_multiplier = max_context_tokens / max_model_len

    return VRAMEstimate(
        model_weights_gb=model_weights_gb,
        kv_cache_per_token_bytes=kv_cache_per_token_bytes,
        kv_cache_total_gb=kv_cache_total_gb,
        total_per_gpu_gb=total_per_gpu_gb,
        max_model_len=max_model_len,
        tensor_parallel=tp,
        warnings=warnings,
        model_params=model_params,
        model_dtype=model_dtype,
        kv_dtype=kv_dtype,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        gpu_memory_utilization=gpu_memory_utilization,
        usable_gpu_memory_gb=usable_gpu_memory_gb,
        available_kv_gb=available_kv_gb,
        max_context_tokens=max_context_tokens,
        context_multiplier=context_multiplier,
    )
