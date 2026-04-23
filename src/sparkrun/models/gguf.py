"""Remote GGUF header parser for quantization detection.

Reads only the header and tensor-info section of a remote GGUF file
(via HTTP range requests) to determine the dominant quantization type
without downloading the full model weights.

.. note::
   This module is **not yet wired into the auto-detection pipeline**.
   It is available for explicit use but will be integrated in a future release.
"""

from __future__ import annotations

import logging
import struct
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# GGUF metadata value types
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# ggml_type enum → display name.  Covers all types from ggml.h as of 2025-06.
GGML_TYPE_NAMES: dict[int, str] = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "i8",
    25: "i16",
    26: "i32",
    27: "i64",
    28: "f64",
    29: "iq1_m",
    30: "bf16",
    31: "q4_0_4_4",
    32: "q4_0_4_8",
    33: "q4_0_8_8",
    34: "tq1_0",
    35: "tq2_0",
    36: "iq4_nl_4_4",
    37: "iq4_nl_4_8",
    38: "iq4_nl_8_8",
}

# Types that are not quantized weights (used for embeddings, norms, etc.)
_NON_QUANT_TYPES = {"f32", "f16", "bf16", "f64", "i8", "i16", "i32", "i64"}


class _NeedMoreBytes(Exception):
    """Raised when the buffer doesn't contain enough data to parse."""


class _Buf:
    """Minimal buffer reader for little-endian GGUF binary parsing."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _need(self, n: int) -> None:
        if self.pos + n > len(self.data):
            raise _NeedMoreBytes

    def read(self, n: int) -> bytes:
        self._need(n)
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def str(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")


def _skip_metadata_value(buf: _Buf, value_type: int) -> None:
    """Skip over a single GGUF metadata value in the buffer."""
    if value_type in (_GGUF_TYPE_UINT8, _GGUF_TYPE_INT8, _GGUF_TYPE_BOOL):
        buf.read(1)
    elif value_type in (_GGUF_TYPE_UINT16, _GGUF_TYPE_INT16):
        buf.read(2)
    elif value_type in (_GGUF_TYPE_UINT32, _GGUF_TYPE_INT32, _GGUF_TYPE_FLOAT32):
        buf.read(4)
    elif value_type in (_GGUF_TYPE_UINT64, _GGUF_TYPE_INT64, _GGUF_TYPE_FLOAT64):
        buf.read(8)
    elif value_type == _GGUF_TYPE_STRING:
        buf.str()
    elif value_type == _GGUF_TYPE_ARRAY:
        elem_type = buf.u32()
        n = buf.u64()
        for _ in range(n):
            _skip_metadata_value(buf, elem_type)
    else:
        raise ValueError("Unknown GGUF metadata value type: %d" % value_type)


def parse_gguf_tensor_types(data: bytes) -> Counter[str]:
    """Parse GGUF header + tensor_infos from raw bytes.

    Args:
        data: Raw bytes from the beginning of a GGUF file (at least the
            header, metadata, and tensor_info sections).

    Returns:
        Counter mapping ggml type names to occurrence counts.

    Raises:
        ValueError: If the data is not a valid GGUF file or version is unsupported.
        _NeedMoreBytes: If the data is too short to parse all tensor_infos.
    """
    buf = _Buf(data)

    magic = buf.read(4)
    if magic != b"GGUF":
        raise ValueError("Not a GGUF file (magic: %r)" % magic)

    version = buf.u32()
    if version not in (2, 3):
        raise ValueError("Unsupported GGUF version: %d" % version)

    tensor_count = buf.u64()
    metadata_kv_count = buf.u64()

    # Skip all metadata key-value pairs
    for _ in range(metadata_kv_count):
        buf.str()  # key
        value_type = buf.u32()
        _skip_metadata_value(buf, value_type)

    # Parse tensor_infos: name, n_dims, dims[], ggml_type, offset
    counts: Counter[str] = Counter()
    for _ in range(tensor_count):
        buf.str()  # tensor name
        n_dimensions = buf.u32()
        for _ in range(n_dimensions):
            buf.u64()  # dimension
        ggml_type = buf.u32()
        buf.u64()  # offset into tensor_data

        type_name = GGML_TYPE_NAMES.get(ggml_type, "ggml_type_%d" % ggml_type)
        counts[type_name] += 1

    return counts


def dominant_quantization(counts: Counter[str]) -> str | None:
    """Determine the dominant quantization type from tensor type counts.

    Excludes non-quantized types (f32, f16, bf16, etc.) which are typically
    used for embeddings, norms, and output heads — not the bulk of weights.

    Returns the most common quantized type name, or ``None`` if no quantized
    tensors are found.
    """
    quant_counts = {k: v for k, v in counts.items() if k not in _NON_QUANT_TYPES}
    if not quant_counts:
        return None
    return max(quant_counts, key=lambda k: quant_counts[k])


def fetch_remote_gguf_quant(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    start_bytes: int = 256 * 1024,
    max_bytes: int = 16 * 1024 * 1024,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """Read the header of a remote GGUF file and detect quantization.

    Uses HTTP range requests to download only the beginning of the file
    from HuggingFace Hub, progressively doubling the fetch size until the
    header and tensor_info section can be fully parsed.

    Args:
        repo_id: HuggingFace repository ID (e.g. ``Qwen/Qwen3-1.7B-GGUF``).
        filename: Name of the ``.gguf`` file in the repo.
        revision: Optional git revision (branch, tag, or commit hash).
        token: Optional HuggingFace API token.
        start_bytes: Initial number of bytes to fetch (default 256 KiB).
        max_bytes: Maximum bytes to fetch before giving up (default 16 MiB).
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict with keys ``majority_quantization`` (str), ``tensor_type_counts``
        (dict), ``bytes_fetched`` (int), or ``None`` on failure.
    """
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils._headers import build_hf_headers
        from urllib.request import Request, urlopen
    except ImportError:
        logger.debug("huggingface_hub not available for GGUF header fetch")
        return None

    try:
        kwargs: dict[str, Any] = {"repo_id": repo_id, "filename": filename}
        if revision:
            kwargs["revision"] = revision
        url = hf_hub_url(**kwargs)
        hf_headers = build_hf_headers(token=token)

        n = start_bytes
        while n <= max_bytes:
            headers = dict(hf_headers)
            headers["Range"] = "bytes=0-%d" % (n - 1)
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                content = resp.read()

            try:
                counts = parse_gguf_tensor_types(content)
                majority = dominant_quantization(counts)
                return {
                    "majority_quantization": majority,
                    "tensor_type_counts": dict(counts),
                    "bytes_fetched": len(content),
                }
            except _NeedMoreBytes:
                n *= 2

        logger.warning(
            "GGUF header for %s/%s exceeded %d bytes; giving up",
            repo_id,
            filename,
            max_bytes,
        )
    except Exception as e:
        logger.debug("Failed to fetch GGUF header for %s/%s: %s", repo_id, filename, e)

    return None
