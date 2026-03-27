"""Tests for sparkrun.models.gguf module (GGUF header parser)."""

from __future__ import annotations

import struct
from collections import Counter

import pytest

from sparkrun.models.gguf import (
    GGML_TYPE_NAMES,
    _NeedMoreBytes,
    dominant_quantization,
    parse_gguf_tensor_types,
)


def _build_gguf_header(
    *,
    version: int = 3,
    metadata: dict[str, bytes] | None = None,
    tensors: list[tuple[str, list[int], int]] | None = None,
) -> bytes:
    """Build a minimal GGUF binary header for testing.

    Args:
        version: GGUF version (2 or 3).
        metadata: Dict of key -> (type_u32 + value_bytes) pairs.
        tensors: List of (name, dimensions, ggml_type) tuples.

    Returns raw bytes of a parseable GGUF header.
    """
    parts: list[bytes] = []
    meta_items = metadata or {}
    tensor_items = tensors or []

    # Magic
    parts.append(b"GGUF")
    # Version
    parts.append(struct.pack("<I", version))
    # Tensor count
    parts.append(struct.pack("<Q", len(tensor_items)))
    # Metadata KV count
    parts.append(struct.pack("<Q", len(meta_items)))

    # Metadata KV pairs
    for key, type_and_value in meta_items.items():
        key_bytes = key.encode("utf-8")
        parts.append(struct.pack("<Q", len(key_bytes)))
        parts.append(key_bytes)
        parts.append(type_and_value)

    # Tensor infos
    for name, dims, ggml_type in tensor_items:
        name_bytes = name.encode("utf-8")
        parts.append(struct.pack("<Q", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<I", len(dims)))
        for d in dims:
            parts.append(struct.pack("<Q", d))
        parts.append(struct.pack("<I", ggml_type))
        parts.append(struct.pack("<Q", 0))  # offset

    return b"".join(parts)


def _make_uint32_meta(value: int) -> bytes:
    """Create a GGUF metadata value blob for a uint32."""
    return struct.pack("<I", 4) + struct.pack("<I", value)


class TestParseGgufTensorTypes:
    """Test parse_gguf_tensor_types with synthetic GGUF headers."""

    def test_basic_q4_k_tensors(self):
        data = _build_gguf_header(
            tensors=[
                ("weight.0", [4096, 4096], 12),  # Q4_K
                ("weight.1", [4096, 4096], 12),  # Q4_K
                ("weight.2", [4096, 4096], 12),  # Q4_K
                ("norm.0", [4096], 0),  # F32
            ]
        )
        counts = parse_gguf_tensor_types(data)
        assert counts["q4_k"] == 3
        assert counts["f32"] == 1

    def test_mixed_types(self):
        data = _build_gguf_header(
            tensors=[
                ("w.0", [100], 12),  # Q4_K
                ("w.1", [100], 14),  # Q6_K
                ("w.2", [100], 12),  # Q4_K
                ("embed", [100], 1),  # F16
            ]
        )
        counts = parse_gguf_tensor_types(data)
        assert counts["q4_k"] == 2
        assert counts["q6_k"] == 1
        assert counts["f16"] == 1

    def test_no_tensors(self):
        data = _build_gguf_header(tensors=[])
        counts = parse_gguf_tensor_types(data)
        assert len(counts) == 0

    def test_with_metadata(self):
        data = _build_gguf_header(
            metadata={"general.architecture": _make_uint32_meta(42)},
            tensors=[("w.0", [10], 8)],  # Q8_0
        )
        counts = parse_gguf_tensor_types(data)
        assert counts["q8_0"] == 1

    def test_version_2(self):
        data = _build_gguf_header(
            version=2,
            tensors=[("w.0", [10], 2)],  # Q4_0
        )
        counts = parse_gguf_tensor_types(data)
        assert counts["q4_0"] == 1

    def test_invalid_magic(self):
        data = b"NOTG" + b"\x00" * 100
        with pytest.raises(ValueError, match="Not a GGUF file"):
            parse_gguf_tensor_types(data)

    def test_unsupported_version(self):
        data = b"GGUF" + struct.pack("<I", 99) + b"\x00" * 100
        with pytest.raises(ValueError, match="Unsupported GGUF version"):
            parse_gguf_tensor_types(data)

    def test_truncated_data_raises(self):
        """Truncated data should raise _NeedMoreBytes."""
        data = _build_gguf_header(tensors=[("weight.0", [4096, 4096], 12)])
        with pytest.raises(_NeedMoreBytes):
            parse_gguf_tensor_types(data[:20])

    def test_unknown_ggml_type(self):
        data = _build_gguf_header(tensors=[("w.0", [10], 999)])
        counts = parse_gguf_tensor_types(data)
        assert counts["ggml_type_999"] == 1

    def test_iq_types(self):
        data = _build_gguf_header(
            tensors=[
                ("w.0", [10], 16),  # IQ2_XXS
                ("w.1", [10], 20),  # IQ4_NL
                ("w.2", [10], 23),  # IQ4_XS
            ]
        )
        counts = parse_gguf_tensor_types(data)
        assert counts["iq2_xxs"] == 1
        assert counts["iq4_nl"] == 1
        assert counts["iq4_xs"] == 1

    def test_multidimensional_tensors(self):
        data = _build_gguf_header(
            tensors=[
                ("w.0", [128, 256, 64], 12),  # Q4_K, 3D
                ("w.1", [1024], 8),  # Q8_0, 1D
            ]
        )
        counts = parse_gguf_tensor_types(data)
        assert counts["q4_k"] == 1
        assert counts["q8_0"] == 1

    def test_many_tensors(self):
        """Simulate a model with many layers."""
        tensors = [
            ("blk.%d.attn_q.weight" % i, [4096, 4096], 12)  # Q4_K
            for i in range(80)
        ] + [
            ("output_norm.weight", [4096], 0),  # F32
            ("token_embd.weight", [4096, 32000], 1),  # F16
        ]
        data = _build_gguf_header(tensors=tensors)
        counts = parse_gguf_tensor_types(data)
        assert counts["q4_k"] == 80
        assert counts["f32"] == 1
        assert counts["f16"] == 1


class TestDominantQuantization:
    """Test dominant_quantization filtering and selection."""

    def test_excludes_non_quant_types(self):
        counts = Counter({"q4_k": 100, "f32": 5, "f16": 3})
        assert dominant_quantization(counts) == "q4_k"

    def test_all_non_quant(self):
        counts = Counter({"f32": 10, "f16": 5, "bf16": 2})
        assert dominant_quantization(counts) is None

    def test_empty(self):
        assert dominant_quantization(Counter()) is None

    def test_picks_most_common(self):
        counts = Counter({"q4_k": 50, "q6_k": 100, "f32": 5})
        assert dominant_quantization(counts) == "q6_k"

    def test_single_quant_type(self):
        counts = Counter({"q8_0": 300, "f32": 1})
        assert dominant_quantization(counts) == "q8_0"

    def test_iq_types_included(self):
        counts = Counter({"iq4_xs": 200, "f32": 5, "f16": 2})
        assert dominant_quantization(counts) == "iq4_xs"

    def test_unknown_type_included(self):
        """Unknown ggml types should still be counted as quant."""
        counts = Counter({"ggml_type_99": 100, "f32": 5})
        assert dominant_quantization(counts) == "ggml_type_99"


class TestGgmlTypeNames:
    """Verify GGML_TYPE_NAMES coverage."""

    def test_common_types_present(self):
        expected = ["f32", "f16", "q4_0", "q4_k", "q6_k", "q8_0", "bf16"]
        for name in expected:
            assert name in GGML_TYPE_NAMES.values(), "Missing: %s" % name

    def test_all_values_lowercase(self):
        for v in GGML_TYPE_NAMES.values():
            assert v == v.lower(), "Not lowercase: %s" % v

    def test_iq_types_present(self):
        expected_iq = ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs"]
        for name in expected_iq:
            assert name in GGML_TYPE_NAMES.values(), "Missing IQ type: %s" % name

    def test_ternary_types_present(self):
        assert "tq1_0" in GGML_TYPE_NAMES.values()
        assert "tq2_0" in GGML_TYPE_NAMES.values()
