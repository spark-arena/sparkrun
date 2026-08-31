"""VRAM estimation for inference workloads on DGX Spark systems.

Model weights, the GPU memory budget, and the arithmetic that combines them with
a KV cache estimate.  The KV estimate itself is architecture-specific and comes
from :mod:`sparkrun.models.kv` — nothing in this module names an attention
architecture.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sparkrun.models.dtypes import bytes_per_element, kv_bytes_per_element, normalize_dtype
from sparkrun.models.kv import ArchInfo, KVSizing, arch_marker_names, extract_arch_fields, resolve_kv_strategy

logger = logging.getLogger(__name__)

# Re-exported so ``from sparkrun.models.vram import bytes_per_element`` keeps
# working; the tables themselves live in the dtypes leaf so a KV strategy can
# ask for an element width without importing the estimator that calls it.
__all__ = [
    "DEFAULT_VRAM_GB",
    "DGX_SPARK_VRAM_GB",
    "MODEL_VISIBILITY_PRIVATE",
    "MODEL_VISIBILITY_PUBLIC",
    "MODEL_VISIBILITY_UNKNOWN",
    "VRAMEstimate",
    "bytes_per_element",
    "estimate_vram",
    "extract_model_info",
    "fetch_model_config",
    "fetch_model_visibility",
    "fetch_safetensors_params",
    "fetch_safetensors_size",
    "kv_bytes_per_element",
    "normalize_dtype",
    "parse_param_count",
]

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
#
# Used as the default per-host VRAM budget by the single-platform fit
# path (:attr:`VRAMEstimate.fits_dgx_spark`).  Heterogeneous-cluster
# fits should call :func:`sparkrun.models.fit.check_fit` instead, which
# reads ``memory_gb`` from each host's
# :class:`~sparkrun.core.hardware.HostHardware`.
DEFAULT_VRAM_GB = 121.0
DGX_SPARK_VRAM_GB = DEFAULT_VRAM_GB  # alias retained for callers that pre-date DEFAULT_VRAM_GB


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

    # KV cache architecture (see sparkrun.models.kv)
    kv_arch: str = "dense"
    """Name of the :class:`~sparkrun.models.kv.KVCacheStrategy` that sized the cache.

    Reports the architecture that was *detected*, independently of whether it
    could be sized — an incomplete config yields ``kv_arch="mla"`` with a
    ``None`` KV estimate, not a model relabelled as dense.
    """

    kv_arch_label: str | None = None
    """Human-readable architecture line for display, or ``None`` for the generic
    layers/heads/head_dim summary."""

    kv_cache_replicated: bool = False
    """Whether the KV cache is duplicated on every tensor-parallel rank.

    True for MLA: the compressed latent has no head dimension to shard, so each
    TP rank holds the full cache and ``tensor_parallel`` does not reduce the
    per-GPU KV footprint (pipeline parallelism still splits it by layer).
    """

    kv_estimate_is_floor: bool = False
    """Whether auxiliary caches exist that this estimate does not count."""

    # GPU memory budget fields
    gpu_memory_utilization: float | None = None
    total_gpu_memory_gb: float | None = None
    usable_gpu_memory_gb: float | None = None
    available_kv_gb: float | None = None
    max_context_tokens: int | None = None
    context_multiplier: float | None = None

    @property
    def fits_dgx_spark(self) -> bool:
        """Whether the estimated per-GPU VRAM fits within DGX Spark memory.

        Legacy single-platform helper.  For heterogeneous-cluster fit checks
        use :func:`sparkrun.models.fit.check_fit`, which inspects each
        host's actual accelerator memory from
        :class:`~sparkrun.core.hardware.HostHardware`.
        """
        return self.total_per_gpu_gb <= DGX_SPARK_VRAM_GB

    def to_dict(self) -> dict[str, Any]:
        """Convert the estimate to a JSON-serializable dictionary."""
        from dataclasses import asdict

        result = asdict(self)
        result["fits_dgx_spark"] = self.fits_dgx_spark
        return result


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


#: Resolved visibility of a HuggingFace repo, as reported by ``model_info``.
MODEL_VISIBILITY_PUBLIC = "public"
MODEL_VISIBILITY_PRIVATE = "private"
MODEL_VISIBILITY_UNKNOWN = "unknown"

#: Per-process memo for :func:`fetch_model_visibility` so a single command
#: never asks the Hub about the same repo twice.
_VISIBILITY_MEMO: dict[tuple[str, str | None], str] = {}


def fetch_model_visibility(model_id: str, revision: str | None = None) -> str:
    """Return whether *model_id* is a publicly readable HuggingFace repo.

    One of :data:`MODEL_VISIBILITY_PUBLIC`, :data:`MODEL_VISIBILITY_PRIVATE`
    (also covers *gated* repos), or :data:`MODEL_VISIBILITY_UNKNOWN`.

    This reads ``ModelInfo.private`` / ``.gated`` rather than inferring from
    whether a fetch succeeded.  The distinction matters: ``huggingface_hub``
    picks up an ambient ``HF_TOKEN`` or stored login, so a *successful* lookup
    says nothing about visibility — a user with a token resolves their own
    private repos perfectly well.

    Every failure mode — offline, rate-limited, typo'd id, no such repo —
    collapses to ``unknown``, so callers must treat ``unknown`` as "not
    established" rather than "not public".
    """
    key = (model_id, revision)
    memo = _VISIBILITY_MEMO.get(key)
    if memo is not None:
        return memo

    verdict = MODEL_VISIBILITY_UNKNOWN
    try:
        from huggingface_hub import model_info as _model_info

        kwargs: dict[str, Any] = {"repo_id": model_id}
        if revision:
            kwargs["revision"] = revision
        mi = _model_info(**kwargs)
        # `gated` is False, "auto", or "manual" — anything truthy means the
        # repo id is not freely readable and is treated as non-public.
        if bool(getattr(mi, "private", False)) or bool(getattr(mi, "gated", False)):
            verdict = MODEL_VISIBILITY_PRIVATE
        else:
            verdict = MODEL_VISIBILITY_PUBLIC
    except Exception as e:
        logger.debug("Could not resolve HF visibility for %s: %s", model_id, e)

    _VISIBILITY_MEMO[key] = verdict
    return verdict


def fetch_safetensors_size(
    model_id: str,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> int | None:
    """Fetch total parameter storage size from safetensors metadata.

    Only consults metadata endpoints — never downloads weight files.  In
    order of preference:

    1. ``model.safetensors.index.json`` (small file) plus ``list_repo_tree``
       LFS sizes for sharded models.
    2. ``list_repo_tree`` for the size of a single ``model.safetensors`` file.
    3. The HuggingFace ``model_info`` API for per-dtype byte counts.

    The ``model_info`` per-dtype counts can be inaccurate for packed quant
    formats, so the raw LFS file size from ``list_repo_tree`` is preferred
    when available.

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

        tree_kwargs: dict[str, Any] = {"repo_id": model_id}
        if revision:
            tree_kwargs["revision"] = revision

        _SAFETENSORS_DTYPE_BYTES: dict[str, int] = {
            "F64": 8,
            "F32": 4,
            "F16": 2,
            "BF16": 2,
            "F8_E4M3": 1,
            "F8_E5M2": 1,
            "I64": 8,
            "I32": 4,
            "I16": 2,
            "I8": 1,
            "U8": 1,
            "BOOL": 1,
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
                index_path = hf_hub_download(**hub_kwargs, filename="model.safetensors.index.json")
            finally:
                enable_progress_bars()
            with open(index_path) as f:
                index = json.load(f)

            # Try to compute actual file sizes from repo tree
            model_files = set(index.get("weight_map", {}).values())
            if model_files:
                try:
                    from huggingface_hub import list_repo_tree

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
                            file_total,
                            matched,
                            len(model_files),
                            model_id,
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

        # Try 2: list_repo_tree for single-file model.safetensors size.
        # Metadata-only LFS lookup — no weight files are downloaded.  Preferred
        # over the model_info API because the on-disk file size reflects packed
        # quant formats (e.g. NVFP4) accurately, whereas the API's per-dtype
        # counts can mis-report for non-standard dtypes.
        try:
            from huggingface_hub import list_repo_tree

            for entry in list_repo_tree(**tree_kwargs):
                if hasattr(entry, "rfilename") and entry.rfilename == "model.safetensors":
                    if entry.size and entry.size > 0:
                        logger.debug(
                            "Using single-file size %d from list_repo_tree for %s",
                            entry.size,
                            model_id,
                        )
                        return int(entry.size)
                    break
        except Exception as e:
            logger.debug("list_repo_tree single-file lookup failed for %s: %s", model_id, e)

        # Try 3: API per-dtype as a last resort when LFS metadata is unavailable.
        api_bytes = _compute_api_bytes()
        if api_bytes is not None:
            logger.debug("Got %d bytes from model_info API for %s", api_bytes, model_id)
            return api_bytes

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
                logger.debug("Got %d params from safetensors metadata for %s", total, model_id)
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

    # Architecture-specific fields (MLA's latent markers, and whatever a future
    # strategy declares).  Each KV strategy owns its own extraction, so this
    # module never learns an architecture's config keys.  The universal fields
    # resolved above are passed along because a derivation may need them — MLA
    # resolves its cached width partly from ``head_dim``.
    info.update(extract_arch_fields(cfg, info))

    # Extracted here rather than only at the top level so a multimodal wrapper's
    # *text* model_type is reachable — it is the one that selects the KV slot
    # layout, and it lives in the nested config alongside the architecture markers.
    if cfg.get("model_type"):
        info["model_type"] = cfg["model_type"]

    return info


# Architecture keys that make an estimate possible at all.  Their absence is
# what sends :func:`extract_model_info` looking in a nested sub-config.
_CORE_ARCH_KEYS = frozenset({"model_dtype", "num_layers", "num_kv_heads", "head_dim"})


def extract_model_info(hf_config: dict[str, Any]) -> dict[str, Any]:
    """Extract model architecture info from a HuggingFace config.json.

    Handles naming variants across architectures (Llama, Qwen, Mistral, GPT-NeoX, etc.).
    For multimodal models that nest text architecture under ``text_config``,
    ``llm_config``, or ``language_config``, those nested dicts are checked
    as a fallback when top-level extraction yields incomplete results.

    Returns:
        Dict with keys: model_dtype, num_layers, num_kv_heads, head_dim,
        model_type, plus whichever architecture markers a KV strategy declares
        (:func:`sparkrun.models.kv.arch_marker_names`) and found — e.g.
        kv_lora_rank / qk_rope_head_dim / compress_ratios for MLA.
    """
    info = _extract_from_config(hf_config)

    # For multimodal / composite models the text architecture lives in a nested
    # sub-config.  Consult it when the top level is missing core architecture
    # fields *or* carries no architecture markers — a wrapper around an MLA text
    # model can be complete for the core keys while hiding every MLA field
    # below, and gating on the core keys alone would silently size it as
    # ordinary attention (a ~14x overestimate that refuses placements).
    markers = arch_marker_names()
    needs_core = not _CORE_ARCH_KEYS.issubset(info.keys())
    needs_arch = markers.isdisjoint(info.keys())
    if needs_core or needs_arch:
        for nested_key in ("text_config", "llm_config", "language_config"):
            nested = hf_config.get(nested_key)
            if isinstance(nested, dict):
                nested_info = _extract_from_config(nested)
                # Fill in only missing fields (top-level takes precedence)
                for k, v in nested_info.items():
                    if k not in info:
                        info[k] = v
                # The KV slot layout is a property of the *text* model, so when
                # the architecture markers came from the nested config its
                # model_type outranks the wrapper's (deepseek_v4, not
                # deepseek_vl_v2).
                if nested_info.get("model_type") and not markers.isdisjoint(nested_info.keys()):
                    info["model_type"] = nested_info["model_type"]
                break  # only use the first matching nested config

    if not info.get("model_type") and hf_config.get("model_type"):
        info["model_type"] = hf_config["model_type"]

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
    total_gpu_memory_gb: float | None = None,
    model_type: str | None = None,
    arch: Mapping[str, Any] | None = None,
) -> VRAMEstimate:
    """Estimate VRAM usage for an inference workload.

    Args:
        model_params: Total parameter count.
        model_dtype: Weight dtype (e.g. "float16", "int4", "fp8").
        kv_dtype: KV cache dtype. ``None`` means "unset" — the estimator falls
            back to ``"bfloat16"`` for computation but leaves ``VRAMEstimate.kv_dtype``
            as ``None`` so display code can distinguish an explicit dtype from a
            defaulted one (issue #248).
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        max_model_len: Maximum sequence length for KV cache sizing.
        tensor_parallel: Tensor parallelism degree.
        pipeline_parallel: Pipeline parallelism degree.
        model_vram: Direct override for model weight VRAM in GB (not scaled by TP/PP).
        kv_vram_per_token: Direct override for KV cache in GB per token (scaled by max_model_len,
            then divided by TP*PP — or by PP alone when the architecture replicates its cache
            across TP ranks, as MLA does).
        gpu_memory_utilization: Fraction of GPU memory the runtime is allowed to use (e.g. 0.9).
        total_gpu_memory_gb: Per-GPU memory of the *target* accelerator (e.g. 48 for an
            RTX A6000). Defaults to the DGX Spark figure when unset, preserving the
            legacy single-platform estimate.
        model_type: HuggingFace ``model_type``. A strong prior for which KV architecture
            a model uses, and what selects a family-specific slot layout.
        arch: Architecture-specific parameters, keyed by
            :attr:`~sparkrun.models.kv.ArchField.name` — e.g.
            ``{"kv_lora_rank": 512, "qk_rope_head_dim": 64}`` for MLA. Which keys
            are meaningful is declared by the registered KV strategies
            (:func:`sparkrun.models.kv.arch_fields`), never by this signature.

    Returns:
        VRAMEstimate with per-GPU totals and any warnings.
    """
    warnings: list[str] = []
    # Apply the bfloat16 fallback only at computation sites, not on the value
    # returned in VRAMEstimate.kv_dtype.  Keeping the original (possibly None)
    # lets the CLI formatter distinguish an explicit dtype from a defaulted one
    # and show "bfloat16 (default)" — without this, the fallback was baked into
    # est.kv_dtype itself and the display code's (default) branch never fired
    # (issue #248).
    kv_dtype_effective = kv_dtype or "bfloat16"
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
            model_weights_gb = model_params * bpe / (1024**3)
        else:
            warnings.append("Unknown dtype %r; cannot estimate model weight VRAM" % model_dtype)
    elif not model_params:
        warnings.append("model_params not available; model weight estimate is zero")
    elif not model_dtype:
        warnings.append("model_dtype not available; model weight estimate is zero")

    # --- KV cache VRAM ---
    # Which architecture this model uses, and therefore how its cache is sized
    # and sharded, is decided once here and owned by sparkrun.models.kv.
    #
    # Detection is separate from sizing on purpose.  A model whose config is too
    # incomplete to size is still that architecture: reporting it as dense
    # instead would mislabel it in `to_dict()` and the CLI, and would flip the
    # sharding rule that a `kv_vram_per_token` override still depends on.
    arch_info = ArchInfo(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_type=model_type,
        kv_dtype=kv_dtype_effective,
        extra=dict(arch or {}),
    )
    strategy, detection = resolve_kv_strategy(arch_info)
    warnings.extend(detection.warnings)

    if kv_vram_per_token is not None:
        # Direct override: the user supplies GB per token, so sizing is theirs.
        # The *sharding* rule is not — it is a property of the architecture, and
        # TP-dividing a replicated cache under-claims memory and lets the
        # scheduler over-commit the placement.
        per_token_bytes = kv_vram_per_token * (1024**3)
        sizing = KVSizing(
            total_bytes=per_token_bytes * max_model_len if max_model_len else None,
            per_token_bytes=per_token_bytes,
            replicated_across_tp=strategy.replicates_kv,
        )
    else:
        sizing = strategy.size(arch_info, max_model_len=max_model_len)

    warnings.extend(sizing.warnings)
    if sizing.unsizable_reason:
        warnings.append(sizing.unsizable_reason)

    kv_cache_per_token_bytes = sizing.per_token_bytes
    kv_cache_total_gb = sizing.total_bytes / (1024**3) if sizing.total_bytes is not None else None

    # --- Per-GPU total ---
    # Model weights split across TP * PP GPUs
    per_gpu_weights_gb = model_weights_gb / shard_factor

    # KV heads also split across TP * PP GPUs — except for an architecture whose
    # cache is replicated per rank (MLA's compressed latent has no head dimension
    # to shard).  Pipeline parallelism still splits it by layer either way.
    kv_shard_factor = pp if sizing.replicated_across_tp else shard_factor
    per_gpu_kv_gb = (kv_cache_total_gb / kv_shard_factor) if kv_cache_total_gb else 0.0

    total_per_gpu_gb = per_gpu_weights_gb + per_gpu_kv_gb

    # --- GPU memory budget analysis ---
    # Compute how much memory the runtime can actually use, and how much
    # is left for KV cache after model weights are loaded.
    usable_gpu_memory_gb: float | None = None
    available_kv_gb: float | None = None
    max_context_tokens: int | None = None
    context_multiplier: float | None = None

    # Target accelerator memory: caller-supplied (e.g. 48 GB A6000) or the
    # DGX Spark default. Keeps the budget honest on non-DGX clusters.
    _total_gpu_gb = total_gpu_memory_gb if (total_gpu_memory_gb and total_gpu_memory_gb > 0) else DGX_SPARK_VRAM_GB

    if gpu_memory_utilization is not None and gpu_memory_utilization > 0:
        usable_gpu_memory_gb = _total_gpu_gb * gpu_memory_utilization
        available_kv_gb = usable_gpu_memory_gb - per_gpu_weights_gb

        if available_kv_gb < 0:
            warnings.append(
                "Model weights (%.1f GB) exceed usable GPU memory "
                "(%.1f GB at %.0f%% utilization)" % (per_gpu_weights_gb, usable_gpu_memory_gb, gpu_memory_utilization * 100)
            )
            available_kv_gb = 0.0

        # Estimate max context tokens that fit in available KV space.  The
        # strategy owns the inversion: linear by default, but a windowed or
        # per-sequence cache must not be extrapolated as if it were.  The budget
        # handed over is the whole (unsharded) cache's share, since that is what
        # the strategy sizes.
        whole_cache_budget_bytes = available_kv_gb * kv_shard_factor * (1024**3)
        max_context_tokens = strategy.tokens_for_budget(arch_info, sizing, whole_cache_budget_bytes)
        if max_context_tokens is not None and max_model_len and max_model_len > 0:
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
        kv_arch=strategy.name,
        kv_arch_label=strategy.label,
        kv_cache_replicated=sizing.replicated_across_tp,
        kv_estimate_is_floor=sizing.is_floor,
        gpu_memory_utilization=gpu_memory_utilization,
        total_gpu_memory_gb=_total_gpu_gb,
        usable_gpu_memory_gb=usable_gpu_memory_gb,
        available_kv_gb=available_kv_gb,
        max_context_tokens=max_context_tokens,
        context_multiplier=context_multiplier,
    )
