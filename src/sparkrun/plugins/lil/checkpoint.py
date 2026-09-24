"""Checkpoint-derived decisions, ported from the lil launcher.

These are the parts of a lil launch that come from the checkpoint rather than
the manifest: which kernel backend the MTP draft's experts need, the default
tensor-parallel size under the ``fit`` policy, and the CUDA-graph capture
sizes for a speculator's verification shape. Each function names the lil
source it mirrors (``internal/launcher/{checkpoint,builder}.go``) so a drift
check has somewhere to look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: ``MTPMoEBackendFromQuantization``.
_BACKEND_FOR_QUANTIZATION = {"nvfp4": "b12x", "mxfp4": "b12x", "mxfp8": "humming", "bf16": "b12x"}

_LAYER_EXPERTS = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:$|\.)")


class CheckpointDecisionError(ValueError):
    """The checkpoint does not determine a value lil would require."""


@dataclass(frozen=True)
class MTPBackendDecision:
    quantization: str
    backend: str
    evidence: tuple[str, ...] = field(default_factory=tuple)


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    """``modelConfig``: a multimodal checkpoint keeps the language model under ``text_config``."""
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def attention_heads(config: dict[str, Any]) -> int | None:
    heads = _integer(model_config(config).get("num_attention_heads"))
    return heads if heads and heads > 0 else None


def backend_for_quantization(quantization: str, evidence: str) -> MTPBackendDecision:
    """``MTPMoEBackendFromQuantization``."""
    backend = _BACKEND_FOR_QUANTIZATION.get(quantization)
    if backend is None:
        raise CheckpointDecisionError("no default MTP MoE backend for quantization %r; declare speculators.mtp.moe_backend" % quantization)
    return MTPBackendDecision(quantization, backend, (evidence,))


def normalize_quantization(value: Any) -> str:
    """``normalizeQuantization``."""
    if not isinstance(value, str):
        return ""
    normalized = value.lower().replace("_", "").replace("-", "")
    if "nvfp4" in normalized:
        return "nvfp4"
    if "mxfp8" in normalized:
        return "mxfp8"
    if "mxfp4" in normalized or normalized == "fp4":
        return "mxfp4"
    if normalized in ("fp8", "e4m3", "float8e4m3fn"):
        return "fp8"
    if normalized in ("bf16", "bfloat16"):
        return "bf16"
    return ""


def _group_quantization(name: str, group: dict[str, Any]) -> str:
    """``groupQuantization``."""
    by_name = normalize_quantization(name)
    if by_name:
        return by_name
    weights = group.get("weights")
    if not isinstance(weights, dict):
        return ""
    bits, size, kind = _integer(weights.get("num_bits")), _integer(weights.get("group_size")), weights.get("type")
    if bits == 4 and size == 16 and kind == "float":
        return "nvfp4"
    if bits == 8 and size == 32 and kind == "float":
        return "mxfp8"
    return ""


def _mtp_layers(config: dict[str, Any]) -> set[int]:
    """``mtpLayerIndices``: the MTP layers follow the last hidden layer."""
    model = model_config(config)
    start = _integer(model.get("num_hidden_layers"))
    count = _integer(model.get("num_nextn_predict_layers"))
    if count is None:
        count = _integer(model.get("mtp_num_hidden_layers"))
    if start is None or count is None or count <= 0:
        return set()
    return set(range(start, start + count))


def _is_mtp_expert_target(target: str, layers: set[int]) -> bool:
    """``isMTPExpertTarget``."""
    normalized = target.lower()
    if "mtp" in normalized and ".experts" in normalized:
        return True
    match = _LAYER_EXPERTS.search(normalized)
    return match is not None and int(match.group(1)) in layers


def _unquantized_decision(config: dict[str, Any], default_dtype: str) -> MTPBackendDecision:
    """``unquantizedMTPDecision``."""
    model = model_config(config)
    dtype = next(
        (d for d in (model.get("dtype"), model.get("torch_dtype"), config.get("dtype"), config.get("torch_dtype")) if d), default_dtype
    )
    normalized = str(dtype).lower().removeprefix("torch.")
    if normalized not in ("bf16", "bfloat16"):
        raise CheckpointDecisionError("cannot select an MTP MoE backend for unquantized checkpoint dtype %r" % dtype)
    return backend_for_quantization("bf16", "no quantized MTP expert target; dtype=%s" % dtype)


def _block_fp8_decision(config: dict[str, Any], quantization_config: dict[str, Any]) -> MTPBackendDecision | None:
    """``blockFP8CheckpointDecision``: DeepSeek's block-FP8 projections with an ``expert_dtype``."""
    if str(quantization_config.get("quant_method") or "").lower() != "fp8":
        return None
    expert_dtype = config.get("expert_dtype") or "fp4"
    quantization = normalize_quantization(expert_dtype)
    if not quantization:
        return None
    return backend_for_quantization(quantization, "quantization_config.quant_method=fp8 with expert_dtype=%s" % expert_dtype)


def mtp_backend_from_config(config: dict[str, Any], default_dtype: str) -> MTPBackendDecision:
    """``MTPMoEBackendFromConfig``: the MTP experts' kernel backend from ``config.json``."""
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return _unquantized_decision(config, default_dtype)
    layers = _mtp_layers(config)
    findings: list[tuple[str, str]] = []
    quantized_layers = quantization_config.get("quantized_layers")
    if isinstance(quantized_layers, dict):
        for target, settings in quantized_layers.items():
            if _is_mtp_expert_target(target, layers):
                algo = settings.get("quant_algo") if isinstance(settings, dict) else None
                findings.append((normalize_quantization(algo), target))
    if not findings:
        groups = quantization_config.get("config_groups")
        if isinstance(groups, dict):
            for name, group in groups.items():
                targets = group.get("targets") if isinstance(group, dict) else None
                for target in targets if isinstance(targets, list) else ():
                    if isinstance(target, str) and _is_mtp_expert_target(target, layers):
                        findings.append((_group_quantization(name, group), target))
    if not findings:
        block = _block_fp8_decision(config, quantization_config)
        return block if block is not None else _unquantized_decision(config, default_dtype)
    unknown = sorted(target for quantization, target in findings if not quantization)
    if unknown:
        raise CheckpointDecisionError("cannot determine MTP MoE quantization for checkpoint targets: %s" % ", ".join(unknown))
    algorithms = {quantization for quantization, _ in findings}
    if len(algorithms) != 1:
        parts = sorted("%s=%s" % (target, quantization) for quantization, target in findings)
        raise CheckpointDecisionError("MTP MoE checkpoint targets use conflicting quantization: %s" % ", ".join(parts))
    quantization = findings[0][0]
    decision = backend_for_quantization(quantization, "checkpoint metadata")
    return MTPBackendDecision(quantization, decision.backend, tuple(sorted(target for _, target in findings)))


def mtp_backend_decision(policy: dict[str, Any], config: dict[str, Any] | None, dtype: str) -> MTPBackendDecision | None:
    """``mtpBackendDecision``: manifest declarations first, checkpoint as evidence.

    ``config`` is ``None`` when the checkpoint could not be read (offline, or
    the Hub unreachable). lil refuses to launch then. The translator degrades
    to what the manifest alone says, and returns ``None`` when that is nothing,
    leaving vLLM's own choice in effect.
    """
    declared_quantization = policy.get("moe_quantization") or ""
    declared_backend = policy.get("moe_backend")
    decision = error = None
    if config is not None:
        try:
            decision = mtp_backend_from_config(config, dtype)
        except CheckpointDecisionError as exc:
            error = exc
    if declared_quantization and decision is not None and decision.quantization and decision.quantization != declared_quantization:
        raise CheckpointDecisionError(
            "MTP MoE quantization mismatch: manifest declares %s, checkpoint contains %s" % (declared_quantization, decision.quantization)
        )
    if declared_backend:
        quantization = (decision.quantization if decision else "") or declared_quantization
        return MTPBackendDecision(quantization, declared_backend, ("lil.yaml moe_backend",))
    if decision is not None:
        return decision
    if declared_quantization:
        reason = "checkpoint metadata inconclusive: %s" % error if error else "checkpoint metadata unavailable"
        return backend_for_quantization(declared_quantization, "lil.yaml (%s)" % reason)
    if error is not None:
        raise error
    return None


# --- tensor-parallel fit (builder.go: estimateMemory / DefaultTPSize) ------------------------------

_GIB = 1 << 30


def fits(device_bytes: int, utilization: float, weight_bytes: int, tp: int) -> bool:
    """``estimateMemory(...).Fits``: sharded weights plus a runtime reserve leave KV headroom."""
    sharded = -(-weight_bytes // tp)
    allocatable = int(device_bytes * utilization)
    reserve = max(8 * _GIB, -(-device_bytes * 8 // 100))
    return allocatable - sharded - reserve > 0


def fit_tensor_parallel(heads: int, weight_bytes: int, device_bytes: int, utilization: float, max_tp: int) -> int | None:
    """``DefaultTPSize`` under the ``fit`` policy: the smallest TP that divides the heads and fits."""
    for size in range(1, max_tp + 1):
        if heads % size == 0 and fits(device_bytes, utilization, weight_bytes, size):
            return size
    return None


# --- CUDA-graph capture sizes (builder.go: cudagraphCaptureSizes / compilationConfig) ---------------


def cudagraph_capture_sizes(max_num_seqs: int, max_num_batched_tokens: int, query_len: int, mixed_multiplier: int) -> list[int]:
    """Every uniform verification batch through ``max_num_seqs``, plus a mixed-batch ladder."""
    max_mixed = min(max_num_seqs * query_len * mixed_multiplier, max_num_batched_tokens, 1024)
    sizes: set[int] = set()

    def add(size: int) -> None:
        if 0 < size <= max_num_batched_tokens:
            sizes.add(size)

    for count in range(1, max_num_seqs + 1):
        add(count * query_len)
    for size in (1, 2, 4):
        if size <= max_mixed:
            add(size)
    for size in range(8, min(max_mixed, 255) + 1, 8):
        add(size)
    for size in range(256, max_mixed + 1, 16):
        add(size)
    add(max_mixed)
    return sorted(sizes)


def verification_shape(speculator: str, tokens: int) -> tuple[int, int]:
    """``(query_len, mixed_multiplier)`` for a speculator actually running ``tokens`` drafts."""
    if tokens == 0 or speculator in ("none", "dflash"):
        return 1, 2
    if speculator == "dspark":
        return tokens + 1, 1
    return tokens + 1, 2
