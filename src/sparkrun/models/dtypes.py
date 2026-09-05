"""Element-width tables for model weights and KV caches.

A leaf module: it imports nothing from ``sparkrun`` and is imported by
:mod:`sparkrun.models.vram`, :mod:`sparkrun.models.quantization` and every
:mod:`sparkrun.models.kv` strategy.  Keeping it separate is what lets a KV
strategy ask for an element width without importing the estimator that calls it.

Two tables, because a dtype's *KV cache* packing is not always its *weight*
packing — see :data:`_KV_DTYPE_BYTES`.
"""

from __future__ import annotations

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
    "mxfp8": 1.0,
    "int4": 0.5,
    "awq": 0.5,
    "nvfp4": 0.5,
    "awq4": 0.5,
    "fp4": 0.5,
    "w4a16_awq": 0.5,
    "w4a16_nvfp4": 0.5,
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

# Bytes per element for dtypes whose *KV cache* packing differs from their
# weight packing.  Consulted by :func:`kv_bytes_per_element` before
# :data:`_DTYPE_BYTES`.
#
# NVFP4 KV cache stores fp8 block scales alongside the fp4 data (one scale
# per 16 elements), so the packed last dimension is
# ``head_size // 2 + head_size // 16`` — 0.5625 bytes per element, not 0.5.
_KV_DTYPE_BYTES: dict[str, float] = {
    "nvfp4": 0.5625,
    "w4a16_nvfp4": 0.5625,
}

_DTYPE_CANONICAL: dict[str, str] = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}

# Weight formats whose width is a per-checkpoint bits-per-weight figure rather
# than a fixed packing, so no table entry can cover them: exllamav3 publishes
# 2.05 / 3.05 / 4.05 bpw builds of the same model as separate branches of one
# repo.  The width therefore travels *in* the dtype string — ``exl3:2.05`` — and
# is read back by :func:`bytes_per_element`.
#
# Scoped to a declared set so an arbitrary ``foo:1.5`` stays unknown, and
# weight-only by design: these are never KV-cache layouts, which is why the
# parse is absent from :func:`kv_bytes_per_element`.
#
# The figure is nominal, matching every other entry here.  Real checkpoints run
# a few percent above it (a separately-quantized ``head_bits`` plus trellis
# scale overhead), which is within what a VRAM *estimate* is for.
_BPW_FAMILIES = frozenset({"exl2", "exl3"})

# Below 1 nothing ships; above 16 the caller means a float dtype and is reading
# the wrong table.  Bounds exist so a typo'd ``exl3:205`` is rejected rather
# than silently sizing the model 100x too large.
_BPW_MIN = 1.0
_BPW_MAX = 16.0


def _bits_per_weight_bytes(key: str) -> float | None:
    """Bytes per weight for a ``<family>:<bits>`` dtype key, or None."""
    family, sep, bits = key.partition(":")
    if not sep or family not in _BPW_FAMILIES:
        return None
    try:
        parsed = float(bits)
    except ValueError:
        return None
    if not (_BPW_MIN <= parsed <= _BPW_MAX):
        return None
    return parsed / 8.0


def dtype_key(dtype: str) -> str:
    """Fold a dtype spelling to its lookup key.

    Case, surrounding whitespace and ``-``/``_`` are all insignificant, so
    ``nvfp4-ds-mla`` and ``NVFP4_DS_MLA`` reach the same table entry.  Every
    table lookup in this package goes through here, so a new table cannot
    accidentally normalize differently.
    """
    return dtype.lower().strip().replace("-", "_")


def normalize_dtype(dtype: str) -> str:
    """Normalize a dtype string to its canonical form.

    Maps common short aliases (``bf16`` → ``bfloat16``, ``fp16`` → ``float16``,
    ``fp32`` → ``float32``) to full names.  Unknown dtypes are returned
    lower-cased but otherwise unchanged.
    """
    key = dtype_key(dtype)
    return _DTYPE_CANONICAL.get(key, key)


def bytes_per_element(dtype: str) -> float | None:
    """Return bytes per element for a dtype string, or None if unknown.

    Covers the fixed table plus the bits-parameterized ``<family>:<bits>``
    spelling (``exl3:2.05``) — see :data:`_BPW_FAMILIES`.  A bare family name
    stays unknown, because it is not sizable: the bpw *is* the width.
    """
    key = dtype_key(dtype)
    known = _DTYPE_BYTES.get(key)
    return known if known is not None else _bits_per_weight_bytes(key)


def kv_bytes_per_element(dtype: str) -> float | None:
    """Return bytes per KV-cache element for a dtype string, or None if unknown.

    Same as :func:`bytes_per_element` except for dtypes whose KV cache packing
    carries extra per-block scale bytes (see :data:`_KV_DTYPE_BYTES`).

    Returns ``None`` for a *packed slot layout* such as ``fp8_ds_mla``: those
    have no per-element width at all.  Ask
    :func:`sparkrun.models.kv.is_valid_kv_dtype` instead when the question is
    "may a recipe name this".
    """
    key = dtype_key(dtype)
    override = _KV_DTYPE_BYTES.get(key)
    return override if override is not None else _DTYPE_BYTES.get(key)
