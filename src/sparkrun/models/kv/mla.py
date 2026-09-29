"""Multi-head Latent Attention (DeepSeek V2 / V3 / V4, Kimi) KV cache sizing.

MLA stores one *compressed latent* per token per layer instead of a K and V
entry per attention head, so the dense
``2 * num_layers * num_kv_heads * head_dim`` formula overestimates it by one to
two orders of magnitude.  The latent has no head dimension to shard, so every
tensor-parallel rank holds a full copy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from sparkrun.models.dtypes import dtype_key, kv_bytes_per_element
from sparkrun.models.kv._base import ArchField, ArchInfo, KVCacheStrategy, KVDetection, KVSizing

__all__ = [
    "MLA_ARCH_FIELDS",
    "MlaKVStrategy",
    "mla_kv_bytes_per_token",
    "mla_latent_dim",
    "reconcile_compress_ratios",
]

# Fixed-width KV slot layouts used by DeepSeek Multi-head Latent Attention
# runtimes.  These backends pack the compressed latent, its block scales and
# the RoPE tail into one padded uint8 slot per token per layer, so the
# footprint is a constant rather than ``2 * heads * head_dim * bytes``.
#
# Keyed by KV cache dtype, then by ``model_type`` (``None`` = the fallback for
# any MLA model that isn't special-cased).
# Both figures are verifiable in upstream vLLM (`vllm/v1/kv_cache_interface.py`,
# `MLAAttentionSpec.real_page_size_bytes`); only the `nvfp4_ds_mla` *spelling*
# is fork-only, shipping in the DGX Spark overlay (Anemll/dspark-vllm-gx10).
_MLA_SLOT_BYTES: dict[str, dict[str | None, int]] = {
    # V3.2 sparse MLA: kv_lora_rank 512 in fp8 (512 B) + 4 fp32 block scales
    # (16 B) + qk_rope_head_dim 64 in bfloat16 (128 B) = 656 B.
    # DeepSeek V4 instead packs 448 B NoPE + 128 B RoPE + 8 B fp8 scale = 584 B.
    "fp8_ds_mla": {None: 656, "deepseek_v4": 584},
    # V4 reuses that same 584 B envelope for the NVFP4 variant — the padding,
    # not the element width, sets the slot size.
    #
    # The non-V4 fallback is deliberately the *fp8* 656 B figure, not a true
    # NVFP4 one (which would be nearer 416-448 B): no non-V4 model ships this
    # layout, so rather than invent a number we over-estimate, which refuses a
    # borderline placement instead of OOMing it. Revisit if one appears.
    "nvfp4_ds_mla": {None: 656, "deepseek_v4": 584},
}

# HuggingFace ``model_type`` values that mean Multi-head Latent Attention.
# These are architecture family names, not domain strings: 'deepseek_v3' means
# DeepSeek-V3 does MLA.  Matching on *prefix* deliberately, so a future
# deepseek_v5 / deepseek_v6 keeps working without a code change.
_MLA_MODEL_TYPE_PREFIXES = ("deepseek_v", "deepseek2", "kimiko")

#: The architecture parameters MLA reads. Declared at module level so
#: ``RECIPES.md`` and the recipe metadata plumbing can name them without
#: instantiating the strategy.
MLA_ARCH_FIELDS: tuple[ArchField, ...] = (
    ArchField(
        "layer_types",
        hf_keys=("layer_types",),
        kind=list,
        item_kind=str,
        doc="Per-layer attention types; mixed linear/MLA cache sizing is unsupported.",
    ),
    ArchField(
        "kv_lora_rank",
        hf_keys=("kv_lora_rank",),
        doc="Compressed-latent width, excluding the RoPE tail. Switches KV sizing to the MLA path.",
    ),
    ArchField(
        "qk_rope_head_dim",
        hf_keys=("qk_rope_head_dim",),
        doc="RoPE tail width, cached alongside the latent.",
    ),
    ArchField(
        "compress_ratios",
        hf_keys=("compress_ratios",),
        kind=list,
        doc="DeepSeek V4 per-layer cache compression; layers at ratio <= 1 are sliding-window layers.",
    ),
    ArchField(
        "index_head_dim",
        hf_keys=("index_head_dim",),
        doc="Sparse-attention indexer width. Not sized; its presence marks the estimate as a floor.",
    ),
)


def is_mla_model_type(model_type: str | None) -> bool:
    """Whether *model_type* names an MLA-family architecture.

    ``model_type`` on its own is not a KV provenance guarantee — nothing says a
    model named ``deepseek_*`` *must* run MLA, and a backend could run it with
    full attention.  But it is a strong prior: every shipping DeepSeek/Kimi
    config intends MLA, and the packed-slot layouts for those families exist
    precisely because they cache a latent.  Used to let a user who pins
    ``metadata.model_type`` get a consistent MLA verdict even when the latent
    markers are not pinned alongside it (the ``kv_vram_per_token`` path).
    """
    if not model_type:
        return False
    t = model_type.lower()
    return any(t.startswith(p) for p in _MLA_MODEL_TYPE_PREFIXES)


def layer_types_affect_sizing(metadata: Mapping[str, Any]) -> bool:
    """Whether recipe ``metadata.layer_types`` changes the KV estimate.

    It is read in one place: :meth:`MlaKVStrategy.size` refuses to size a
    hybrid linear/MLA cache. So it matters only for an MLA model (the same
    markers :meth:`MlaKVStrategy.detect` uses) with ``linear_attention``
    layers. Recipe export drops it otherwise: it is a list the length of the
    model, and once the other architecture fields are in ``metadata`` it is
    not re-detected, so dropping it where it *is* read would silently turn a
    refusal into a wrong estimate.
    """
    if "linear_attention" not in (metadata.get("layer_types") or ()):
        return False
    return bool(
        metadata.get("kv_lora_rank")
        or metadata.get("qk_rope_head_dim")
        or dtype_key(str(metadata.get("kv_dtype") or "")) in _MLA_SLOT_BYTES
        or is_mla_model_type(metadata.get("model_type"))
    )


def mla_latent_dim(*, kv_lora_rank: int | None = None, head_dim: int | None = None, qk_rope_head_dim: int | None = None) -> int | None:
    """Resolve the non-RoPE part of an MLA model's cached width.

    The two DeepSeek generations spell the same quantity differently, and the
    difference is easy to double-count:

    - **V2/V3** name the latent ``kv_lora_rank`` and cache ``qk_rope_head_dim``
      *in addition* to it — 512 + 64 = 576 elements for DeepSeek-V3.
    - **V4** has no ``kv_lora_rank``.  Its ``head_dim`` is the *whole* cached
      width, with the RoPE tail carved out of it: upstream vLLM computes
      ``nope_head_dim = head_dim - qk_rope_head_dim`` (512 − 64 = 448) and
      documents the slot as "448B NoPE + 128B RoPE + 8B fp8 scale = 584B".

    Returning the NoPE width for both shapes lets callers add the tail exactly
    once, so V4 sizes to ``head_dim`` rather than ``head_dim + qk_rope_head_dim``.

    Returns:
        The NoPE width in elements, or ``None`` when it can't be resolved.
    """
    if kv_lora_rank:
        return int(kv_lora_rank)
    if not head_dim:
        return None
    # V4 shape: head_dim already contains the tail, so carve it back out.
    if qk_rope_head_dim and head_dim > qk_rope_head_dim:
        return int(head_dim) - int(qk_rope_head_dim)
    return int(head_dim)


def reconcile_compress_ratios(compress_ratios: Sequence[int] | None, num_layers: int | None) -> tuple[Sequence[int] | None, str | None]:
    """Trim a ``compress_ratios`` list to the model's layer count.

    Upstream vLLM consults ``compress_ratios[layer_id]`` only for
    ``layer_id < num_hidden_layers`` — DeepSeek-V4-Flash ships 46 entries for
    43 layers, the extra ones covering MTP / non-standard layers.  Summing the
    whole list would count caches that are never allocated.

    Returns:
        ``(ratios, note)`` — the ratios to size from, and a human-readable note
        when the mismatch is one worth surfacing.  A trailing tail of ``<= 1``
        entries (the normal V4 shape) changes nothing and is dropped silently;
        a *short* list, or a trimmed tail that held real compressed layers,
        would change the estimate and is reported.
    """
    if not compress_ratios or not num_layers:
        return compress_ratios, None

    count = len(compress_ratios)
    if count == num_layers:
        return compress_ratios, None
    if count < num_layers:
        return compress_ratios, "compress_ratios lists %d layers but the model has %d; the remainder is unsized" % (count, num_layers)

    trimmed = compress_ratios[:num_layers]
    dropped = [r for r in compress_ratios[num_layers:] if r and r > 1]
    if dropped:
        return (
            trimmed,
            "compress_ratios lists %d layers but the model has %d; %d compressed layer(s) beyond the layer count were ignored"
            % (
                count,
                num_layers,
                len(dropped),
            ),
        )
    return trimmed, None


def mla_kv_bytes_per_token(
    *,
    kv_dtype: str,
    num_layers: int | None = None,
    kv_lora_rank: int | None = None,
    qk_rope_head_dim: int | None = None,
    compress_ratios: Sequence[int] | None = None,
    model_type: str | None = None,
) -> float | None:
    """Bytes of KV cache per token, summed over all layers.

    Two sizings are supported:

    - **Fixed-slot layouts** (``fp8_ds_mla`` / ``nvfp4_ds_mla``): the backend
      packs the latent, its block scales and the RoPE tail into a padded uint8
      slot of constant width — see :data:`_MLA_SLOT_BYTES`.
    - **Everything else**: ``(kv_lora_rank + qk_rope_head_dim)`` elements per
      layer at the KV dtype's element width.

    ``compress_ratios`` is DeepSeek V4's per-layer cache compression (from the
    HF config key of the same name).  A layer with ratio ``r > 1`` stores one
    slot per ``r`` tokens; layers with ratio ``<= 1`` are sliding-window layers
    whose cache is bounded by ``sliding_window`` rather than ``max_model_len``,
    so they contribute nothing at this scale and are excluded.  Pass it through
    :func:`reconcile_compress_ratios` first — when a list is present it, not
    ``num_layers``, decides how many layers are summed.

    Returns:
        Bytes per token, or ``None`` when neither sizing has enough information.
    """
    key = dtype_key(kv_dtype)

    slots = _MLA_SLOT_BYTES.get(key)
    if slots is not None:
        per_layer: float = slots.get(model_type, slots[None])
    else:
        if not kv_lora_rank:
            return None
        bpe = kv_bytes_per_element(key)
        if bpe is None:
            return None
        per_layer = (kv_lora_rank + (qk_rope_head_dim or 0)) * bpe

    if compress_ratios:
        total = sum(per_layer / r for r in compress_ratios if r and r > 1)
        # Every entry was <= 1, so no layer contributes a latent cache.  That
        # is 0 bytes, not "0 GB of KV needed" — a zero estimate passes every
        # fit check, so the workload would be placed with no KV headroom and
        # OOM at runtime.  Report it as unsizable and let the caller warn.
        return total or None
    if not num_layers:
        return None
    return per_layer * num_layers


class MlaKVStrategy(KVCacheStrategy):
    """Sizes DeepSeek-family compressed-latent KV caches."""

    name: ClassVar[str] = "mla"
    label: ClassVar[str | None] = "MLA (compressed latent KV cache)"
    priority: ClassVar[int] = 10
    replicates_kv: ClassVar[bool] = True

    def arch_fields(self) -> tuple[ArchField, ...]:
        return MLA_ARCH_FIELDS

    def kv_layouts(self) -> frozenset[str]:
        return frozenset(_MLA_SLOT_BYTES)

    def extract(self, cfg: Mapping[str, Any], core: Mapping[str, Any]) -> dict[str, Any]:
        """Read the MLA markers, normalizing V2/V3 and V4 to one latent width.

        ``qk_rope_head_dim`` is the marker that a model caches a latent at all,
        so the latent is only derived when it is present: ``head_dim`` alone
        means ordinary attention.  ``compress_ratios`` and ``index_head_dim``
        are read independently — V4 has all three, V3.2 has the indexer but no
        per-layer compression.
        """
        found: dict[str, Any] = {}
        if "qk_rope_head_dim" in cfg:
            found["qk_rope_head_dim"] = cfg["qk_rope_head_dim"]
            latent = mla_latent_dim(
                kv_lora_rank=cfg.get("kv_lora_rank"),
                head_dim=core.get("head_dim"),
                qk_rope_head_dim=cfg["qk_rope_head_dim"],
            )
            if latent:
                found["kv_lora_rank"] = latent

        ratios = cfg.get("compress_ratios")
        if isinstance(ratios, list):
            found["compress_ratios"] = ratios

        # DeepSeek sparse attention (V3.2 / V4) keeps a second, separate indexer
        # cache alongside the latent.  We don't size it, but its presence is what
        # makes the estimate a floor rather than a total.
        if "index_head_dim" in cfg:
            found["index_head_dim"] = cfg["index_head_dim"]
        if "layer_types" in cfg:
            found["layer_types"] = cfg["layer_types"]
        return found

    def detect(self, arch: ArchInfo) -> KVDetection | None:
        kv_lora_rank = arch.get("kv_lora_rank")
        qk_rope_head_dim = arch.get("qk_rope_head_dim")
        layout = dtype_key(arch.kv_dtype or "") in _MLA_SLOT_BYTES
        family = is_mla_model_type(arch.model_type)

        if kv_lora_rank or qk_rope_head_dim:
            source = "architecture"
        elif layout:
            source = "kv_layout"
        elif family:
            source = "model_type"
        else:
            return None

        warnings: tuple[str, ...] = ()
        # `qk_rope_head_dim` on its own is a weak signal.  Auto-detection always
        # resolves a latent alongside it (every shipping DeepSeek/Kimi config names
        # kv_lora_rank, or is V4-shaped), so this only arises from hand-pinned
        # metadata — and there it is worth flagging, because sizing a non-MLA model
        # this way drops the 2 * num_kv_heads factor and *under*-estimates, which
        # OOMs at runtime rather than refusing the placement.
        if not kv_lora_rank and qk_rope_head_dim and not layout:
            warnings = (
                "MLA sizing inferred from qk_rope_head_dim alone, using head_dim as the cached width; "
                "pin metadata.kv_lora_rank (or kv_vram_per_token) if this model does not use a compressed KV cache",
            )
        # The mirror image: kv_lora_rank on its own silently drops the RoPE tail.
        # The generic path sizes (kv_lora_rank + qk_rope_head_dim) elements, so
        # omitting the tail reads `kv_lora_rank * bytes` — an ~11% under-estimate
        # for DeepSeek-V3 (62,464 vs 70,272), again in the OOM direction.  Same
        # reachability: auto-detection always pairs the fields, so this is a
        # hand-pinned-metadata footgun.
        elif kv_lora_rank and not qk_rope_head_dim and not layout:
            warnings = (
                "MLA sizing inferred from kv_lora_rank alone; the RoPE tail (qk_rope_head_dim) is not counted. "
                "Pin metadata.qk_rope_head_dim if this model caches the tail alongside the latent",
            )
        # An MLA KV layout is authoritative on its own — but if the model shows no
        # architectural MLA marker at all, it is being forced onto a non-MLA model,
        # which sizes the latent instead of the real heads and *under*-estimates by
        # ~170x (Qwen3-32B: 656 B/layer/token vs ~113 KB).  Reachable only by an
        # explicit `kv_dtype: nvfp4_ds_mla` on such a model; worth a loud warning.
        elif layout and not (kv_lora_rank or qk_rope_head_dim):
            warnings = (
                "KV layout %r forces MLA sizing but the model has no MLA architecture markers; "
                "this under-estimates a non-MLA model. Pin metadata.kv_lora_rank or remove the layout" % arch.kv_dtype,
            )

        return KVDetection(source=source, warnings=warnings)

    def size(self, arch: ArchInfo, *, max_model_len: int | None) -> KVSizing:
        if "linear_attention" in (arch.get("layer_types") or ()):
            return KVSizing(
                total_bytes=None,
                replicated_across_tp=True,
                is_floor=True,
                unsizable_reason="Hybrid linear/MLA cache sizing is unavailable; runtime must verify context capacity",
            )
        kv_dtype = arch.kv_dtype or ""
        raw_ratios = arch.get("compress_ratios")
        index_head_dim = arch.get("index_head_dim")

        # Same normalization :meth:`extract` applies, so a recipe that pins
        # head_dim + qk_rope_head_dim in metadata gets the same cached width as
        # auto-detection would have written, without naming kv_lora_rank.
        latent = mla_latent_dim(
            kv_lora_rank=arch.get("kv_lora_rank"),
            head_dim=arch.head_dim,
            qk_rope_head_dim=arch.get("qk_rope_head_dim"),
        )

        ratios, ratio_note = reconcile_compress_ratios(raw_ratios, arch.num_layers)
        per_token = mla_kv_bytes_per_token(
            kv_dtype=kv_dtype,
            num_layers=arch.num_layers,
            kv_lora_rank=latent,
            qk_rope_head_dim=arch.get("qk_rope_head_dim"),
            compress_ratios=ratios,
            model_type=arch.model_type,
        )

        if per_token is None:
            return KVSizing(
                total_bytes=None,
                replicated_across_tp=True,
                unsizable_reason="Cannot size MLA KV cache (%s); KV cache estimate unavailable"
                % self._unsizable_reason(arch, latent, raw_ratios),
            )

        warnings: list[str] = []
        if ratio_note:
            warnings.append(ratio_note)

        # Name the auxiliary caches this estimate leaves out, so the number is
        # understood as a floor.  Keying this on compress_ratios alone would
        # silently skip DeepSeek V3.2, which has a sparse indexer but no
        # per-layer compression.
        excluded = []
        if raw_ratios and any(not r or r <= 1 for r in raw_ratios):
            excluded.append("sliding-window")
        if index_head_dim:
            excluded.append("sparse-indexer")
        if excluded:
            warnings.append(
                "MLA estimate covers the compressed latent cache only; the %s cache%s not included"
                % (" and ".join(excluded), "s are" if len(excluded) > 1 else " is")
            )

        return KVSizing(
            total_bytes=per_token * max_model_len if max_model_len else None,
            per_token_bytes=per_token,
            replicated_across_tp=True,
            is_floor=bool(excluded),
            warnings=tuple(warnings),
        )

    def _unsizable_reason(self, arch: ArchInfo, latent: int | None, ratios: Sequence[int] | None) -> str:
        """Name the actual gap.

        Blaming the dtype unconditionally is misleading when the real problem is
        a missing latent dim, a missing layer count, or a ratio list with
        nothing above 1.
        """
        layout = dtype_key(arch.kv_dtype or "") in _MLA_SLOT_BYTES
        if ratios and not any(r and r > 1 for r in ratios):
            return "no compress_ratios entry above 1, so no layer holds a latent cache"
        if not layout and not latent:
            return "no kv_lora_rank/head_dim to size the compressed latent from"
        if not layout and kv_bytes_per_element(arch.kv_dtype or "") is None:
            return "unknown KV cache dtype %r" % arch.kv_dtype
        if not arch.num_layers:
            return "num_layers unavailable"
        return "insufficient architecture info"
