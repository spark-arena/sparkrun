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
    'w4a16_awq': 0.5,
    "awq8": 1.0,
    "gptq": 0.5,
    "mxfp4": 0.5,
    # GGUF quants — bytes per weight from llama.cpp ggml type_size / block_size.
    # Basic quants
    "q4_0": 0.5625,
    "q4_1": 0.625,
    "q5_0": 0.6875,
    "q5_1": 0.75,
    "q8_0": 1.0625,
    "q8_1": 1.125,
    # K-quants (base types — dominant tensor type in K-quant mixes)
    "q2_k": 0.3125,
    "q3_k": 0.4375,
    "q4_k": 0.5625,
    "q5_k": 0.6875,
    "q6_k": 0.8125,
    "q8_k": 1.0625,
    # K-quant mixes (suffixed names used by llama.cpp quantize CLI).
    # The _s/_m suffix selects which layers use the base vs higher-precision quant;
    # bytes-per-element is the same as the base type for estimation purposes.
    # Uncommon _l variants fall back to the base via _gguf_normalize_quant().
    "q2_k_s": 0.3125,
    "q3_k_s": 0.4375,
    "q3_k_m": 0.4375,
    "q4_k_s": 0.5625,
    "q4_k_m": 0.5625,
    "q5_k_s": 0.6875,
    "q5_k_m": 0.6875,
    # IQ (importance-matrix quants)
    "iq1_s": 0.1875,
    "iq1_m": 0.1875,
    "iq2_xxs": 0.25,
    "iq2_xs": 0.3125,
    "iq2_s": 0.3125,
    "iq3_xxs": 0.4063,
    "iq3_s": 0.4375,
    "iq4_nl": 0.5625,
    "iq4_xs": 0.5625,
    # Ternary
    "tq1_0": 0.1875,
    "tq2_0": 0.3125,
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
    pipeline_parallel: int = 1
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

    def to_dict(self) -> dict[str, Any]:
        """Convert the estimate to a JSON-serializable dictionary."""
        from dataclasses import asdict
        result = asdict(self)
        result["fits_dgx_spark"] = self.fits_dgx_spark
        return result


_DTYPE_CANONICAL: dict[str, str] = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def normalize_dtype(dtype: str) -> str:
    """Normalize a dtype string to its canonical form.

    Maps common short aliases (``bf16`` → ``bfloat16``, ``fp16`` → ``float16``,
    ``fp32`` → ``float32``) to full names.  Unknown dtypes are returned
    lower-cased but otherwise unchanged.
    """
    key = dtype.lower().strip().replace("-", "_")
    return _DTYPE_CANONICAL.get(key, key)


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
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
        import json

        from sparkrun.models.download import _hub_cache

        kwargs: dict[str, Any] = {"repo_id": model_id, "filename": "config.json"}
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
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
        import json

        from sparkrun.models.download import _hub_cache

        hub_kwargs: dict[str, Any] = {"repo_id": model_id}
        if revision:
            hub_kwargs["revision"] = revision
        if cache_dir:
            hub_kwargs["cache_dir"] = _hub_cache(cache_dir)

        _SAFETENSORS_DTYPE_BYTES: dict[str, int] = {
            "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
            "F8_E4M3": 1, "F8_E5M2": 1,
            "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
        }

        def _compute_api_bytes() -> int | None:
            """Compute total bytes from HF model_info per-dtype param counts."""
            try:
                from huggingface_hub import model_info as _model_info

                mi_kwargs: dict[str, Any] = {"repo_id": model_id}
                if revision:
                    mi_kwargs["revision"] = revision
                mi = _model_info(**mi_kwargs)
                if mi.safetensors is not None:
                    total = 0
                    for dtype_name, count in mi.safetensors.parameters.items():
                        elem_size = _SAFETENSORS_DTYPE_BYTES.get(dtype_name, 2)
                        total += count * elem_size
                    if total > 0:
                        return total
            except Exception as e:
                logger.debug("model_info API failed for %s: %s", model_id, e)
            return None

        # Try 1: sharded model with index file.
        # Use the index weight_map to identify model files, then sum
        # actual file sizes from list_repo_tree (LFS metadata).  This
        # handles both stale total_size (e.g. copied from pre-quantized)
        # and repos with extra safetensors (e.g. original/ copies).
        # Falls back to index total_size if list_repo_tree is unavailable.
        try:
            disable_progress_bars()
            try:
                index_path = hf_hub_download(
                    **hub_kwargs, filename="model.safetensors.index.json"
                )
            finally:
                enable_progress_bars()
            with open(index_path) as f:
                index = json.load(f)

            # Try to compute actual file sizes from repo tree
            model_files = set(index.get("weight_map", {}).values())
            if model_files:
                try:
                    from huggingface_hub import list_repo_tree

                    tree_kwargs: dict[str, Any] = {"repo_id": model_id}
                    if revision:
                        tree_kwargs["revision"] = revision
                    file_total = 0
                    matched = 0
                    for entry in list_repo_tree(**tree_kwargs):
                        if hasattr(entry, "rfilename") and entry.rfilename in model_files:
                            if entry.size and entry.size > 0:
                                file_total += entry.size
                                matched += 1
                    if matched > 0 and file_total > 0:
                        logger.debug(
                            "Got %d bytes from file sizes (%d/%d files) for %s",
                            file_total, matched, len(model_files), model_id,
                        )
                        return file_total
                except Exception as e:
                    logger.debug("list_repo_tree failed for %s: %s", model_id, e)

            # Fall back to index total_size
            total_size = index.get("metadata", {}).get("total_size")
            if total_size is not None:
                logger.debug("Using index total_size %d for %s", total_size, model_id)
                return int(total_size)
        except Exception as e:
            logger.debug("safetensors index failed for %s: %s", model_id, e)

        # Try 2: single-file model — read safetensors header for total param size
        try:
            import os

            disable_progress_bars()
            try:
                sf_path = hf_hub_download(
                    **hub_kwargs, filename="model.safetensors"
                )
            finally:
                enable_progress_bars()
            # safetensors header: first 8 bytes = header size (u64 LE),
            # then JSON header with tensor metadata including shape/dtype
            import struct

            with open(sf_path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))

            total_bytes = 0
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                shape = info.get("shape", [])
                dtype = info.get("dtype", "")
                elem_size = _SAFETENSORS_DTYPE_BYTES.get(dtype, 2)
                num_elements = 1
                for dim in shape:
                    num_elements *= dim
                total_bytes += num_elements * elem_size
            if total_bytes > 0:
                return total_bytes
        except Exception as e:
            logger.debug("safetensors header parse failed for %s: %s", model_id, e)

        # Try 3: API per-dtype for models without index or header
        api_bytes = _compute_api_bytes()
        if api_bytes is not None:
            logger.debug("Got %d bytes from model_info API for %s", api_bytes, model_id)
            return api_bytes

        # Try 4: fall back to file size as rough approximation
        try:
            import os

            disable_progress_bars()
            try:
                sf_path = hf_hub_download(
                    **hub_kwargs, filename="model.safetensors"
                )
            finally:
                enable_progress_bars()
            size = os.path.getsize(sf_path)
            if size > 0:
                logger.debug("Using file size as param size estimate for %s", model_id)
                return size
        except Exception as e:
            logger.debug("safetensors file size fallback failed for %s: %s", model_id, e)

    except Exception as e:
        logger.debug("Could not fetch safetensors size for %s: %s", model_id, e)
    return None


def fetch_safetensors_params(
        model_id: str,
        revision: str | None = None,
) -> int | None:
    """Fetch total parameter count from HuggingFace model safetensors metadata.

    Uses the HuggingFace Hub API (``model_info``) which returns parameter counts
    per dtype without downloading any model files.  This is the preferred method
    for single-file safetensors models that lack an index file.

    Args:
        model_id: HuggingFace model identifier.
        revision: Optional revision (branch, tag, or commit hash).

    Returns:
        Total parameter count, or ``None`` if unavailable.
    """
    try:
        from huggingface_hub import model_info as _model_info

        kwargs: dict[str, Any] = {"repo_id": model_id}
        if revision:
            kwargs["revision"] = revision
        info = _model_info(**kwargs)
        if info.safetensors is not None:
            total = info.safetensors.total
            if total and total > 0:
                logger.debug(
                    "Got %d params from safetensors metadata for %s", total, model_id
                )
                return int(total)
    except Exception as e:
        logger.debug("Could not fetch safetensors params for %s: %s", model_id, e)
    return None


def _resolve_quant_dtype(quantization_config: dict[str, Any]) -> str | None:
    """Derive a model weight dtype from a HuggingFace quantization_config block.

    Handles common quant methods: fp8, awq, gptq, marlin, bitsandbytes,
    mxfp4, nvfp4, compressed-tensors.
    Returns a dtype string recognized by :func:`bytes_per_element`, or ``None``
    if the method is unrecognized.

    .. note::
       This is a thin wrapper around
       :func:`sparkrun.models.quantization._resolve_from_quantization_config`
       kept for backward compatibility.  New code should use
       :func:`~sparkrun.models.quantization.resolve_quantization` instead.
    """
    from sparkrun.models.quantization import _resolve_from_quantization_config

    info = _resolve_from_quantization_config(quantization_config)
    return info.weight_dtype if info else None


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

    # Extract quantization dtype from quantization_config if present.
    # This is more accurate than torch_dtype for quantized models (e.g.
    # an FP8 model will have torch_dtype=bfloat16 but quant_method=fp8).
    qc = hf_config.get("quantization_config")
    if isinstance(qc, dict):
        from sparkrun.models.quantization import _resolve_from_quantization_config

        qi = _resolve_from_quantization_config(qc)
        if qi:
            info["quant_dtype"] = qi.weight_dtype
            info["quant_info"] = qi

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
        pipeline_parallel: int = 1,
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
        pipeline_parallel: Pipeline parallelism degree.
        model_vram: Direct override for model weight VRAM in GB (not scaled by TP/PP).
        kv_vram_per_token: Direct override for KV cache in GB per token (scaled by max_model_len and TP*PP).
        gpu_memory_utilization: Fraction of GPU memory the runtime is allowed to use (e.g. 0.9).

    Returns:
        VRAMEstimate with per-GPU totals and any warnings.
    """
    warnings: list[str] = []
    kv_dtype = kv_dtype or "bfloat16"
    tp = max(tensor_parallel, 1)
    pp = max(pipeline_parallel, 1)
    shard_factor = tp * pp

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
        warnings.append("Missing architecture info (%s); KV cache estimate unavailable" % ", ".join(missing))

    # --- Per-GPU total ---
    # Model weights split across TP * PP GPUs
    per_gpu_weights_gb = model_weights_gb / shard_factor

    # KV heads also split across TP * PP GPUs
    per_gpu_kv_gb = (kv_cache_total_gb / shard_factor) if kv_cache_total_gb else 0.0

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
                "(%.1f GB at %.0f%% utilization)" % (per_gpu_weights_gb, usable_gpu_memory_gb, gpu_memory_utilization * 100)
            )
            available_kv_gb = 0.0

        # Estimate max context tokens that fit in available KV space
        if kv_cache_per_token_bytes and kv_cache_per_token_bytes > 0:
            per_gpu_kv_per_token_gb = (kv_cache_per_token_bytes / shard_factor) / (1024 ** 3)
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
        pipeline_parallel=pp,
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
