"""Translate a layered lil entry into a sparkrun v2 recipe.

The mapping follows the lil launcher's ``buildVLLMArgv`` flag for flag, onto
sparkrun's structured vLLM defaults (``runtimes/_vllm_common.py``), so the
result renders through the ordinary flag map and ``-o`` works on every key.
Anything lil decides per launch that sparkrun decides per recipe becomes
generated ``overrides:``:

* **Loader:** lil loads with ``b12x`` on sm_121 and ``instanttensor``
  elsewhere, unless the manifest names one.
* **Topology tuning:** PCIe all-reduce on a multi-GPU box, and
  ``--disable-custom-all-reduce`` plus no PCIe all-reduce across nodes.
* **Memory utilization:** lil's values are tuned for discrete cards, so they
  apply everywhere except unified-memory accelerators (GB10), where the
  platform default stands.
* **Speculators:** the manifest's default runs unless ``-o speculator=…``
  selects another, including ``none``.
* **lil overrides:** translated as ``kind`` → ``nodes``, ``arch`` → ``arch``,
  ``tp`` → ``tp``, and ``speculator`` → ``config.speculator``.

Deliberate differences from a lil launch, each because sparkrun owns the
concern:

* **Tensor parallelism** is decided once, when the recipe is loaded (it is
  workload identity). The ``fit`` policy uses DGX Spark GB10 usable memory;
  pass ``--tp`` otherwise.
* **Runtime-managed values** (NCCL, interfaces, ``CUTE_DSL_ARCH``,
  ``CUDA_HOME``, ``PYTHONPATH``) come from sparkrun and its platform tier,
  never from here.
* **``draft_tensor_parallel_size``** is left to vLLM, so ``--tp`` stays consistent.
* **``cudagraph_capture_sizes``** are computed per override for the capacity
  and speculator that override implies. A ``-o max_num_seqs=`` beyond it runs
  larger batches uncaptured rather than wrong.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .checkpoint import (
    CheckpointDecisionError,
    attention_heads,
    cudagraph_capture_sizes,
    fit_tensor_parallel,
    mtp_backend_decision,
    verification_shape,
)
from .manifest import LilEntry, LilManifestError, deep_merge

logger = logging.getLogger(__name__)

#: The image lil's vLLM fork ships in for DGX Spark.
LIL_CONTAINER = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest"
DEFAULT_PORT = 8000

#: Launcher defaults (``profiles.go`` constants).
DEFAULT_DTYPE = "bfloat16"
DEFAULT_KV_CACHE_DTYPE = "fp8"
DEFAULT_KERNEL_BACKEND = "b12x"
DEFAULT_MTP_TOKENS = 3

#: ``defaultCapacityLayer``.
DEFAULT_CAPACITY: dict[str, Any] = {
    "gpu_memory_utilization": None,
    "kv_cache_memory_bytes": None,
    "max_model_len": "auto",
    "max_num_seqs": 8,
    "max_num_batched_tokens": 4096,
}

#: ``commonHostEnvironment``, minus what sparkrun or the platform owns
#: (``PYTORCH_CUDA_ALLOC_CONF`` is a platform default; ``VLLM_PLUGINS`` is set
#: only with the b12x loader).
COMMON_ENV = {
    "OMP_NUM_THREADS": "16",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_USE_AOT_COMPILE": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "1",
    "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
    "VLLM_USE_STANDALONE_COMPILE": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "B12X_POLICY_MODE": "auto",
}
#: ``DefaultLocalEnvironment`` minus NCCL (sparkrun computes comm env).
MULTI_GPU_HOST_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "1",
    "VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
    "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": "64KB",
    "VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE": "84KB",
}
#: ``DefaultSparkEnvironment`` minus NCCL.
MULTI_NODE_ENV = {"VLLM_ENABLE_PCIE_ALLREDUCE": "0"}

#: Largest tensor-parallel size the ``fit`` policy considers (an 8-node cluster).
MAX_FIT_TP = 8


@dataclass(frozen=True)
class CheckpointFacts:
    """What the checkpoint says; ``None`` fields are unknown (offline, or the Hub unreachable)."""

    config: dict[str, Any] | None = None
    weight_bytes: int | None = None


def _present(mapping: dict[str, Any], key: str) -> bool:
    return mapping.get(key) is not None


def _capacity(entry: LilEntry, layer: dict[str, Any] | None = None) -> dict[str, Any]:
    capacity = deep_merge(DEFAULT_CAPACITY, entry.section("capacity"))
    return deep_merge(capacity, layer) if layer else capacity


def _capacity_defaults(capacity: dict[str, Any], *, include_utilization: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes"):
        if key in capacity:
            value = capacity[key]
            out[key] = str(value) if key in ("max_model_len", "kv_cache_memory_bytes") and value is not None else value
    if include_utilization and "gpu_memory_utilization" in capacity:
        out["gpu_memory_utilization"] = capacity["gpu_memory_utilization"]
    return out


def _speculative_config(
    method: str, entry: LilEntry, facts: CheckpointFacts, *, strict: bool, provenance: dict[str, Any]
) -> tuple[dict[str, Any] | None, int]:
    """``speculativeConfig``: ``(config, draft tokens)`` for *method*; ``(None, 0)`` when nothing runs."""
    speculators = entry.section("speculators")
    if method == "none":
        return None, 0
    policy = speculators.get(method)
    if not isinstance(policy, dict):
        raise LilManifestError("%s does not define a %s speculator" % (entry.name, method))
    if method == "mtp":
        tokens = policy["tokens"] if policy.get("tokens") is not None else DEFAULT_MTP_TOKENS
        if tokens == 0:
            return None, 0
        config: dict[str, Any] = {"method": "mtp", "num_speculative_tokens": tokens}
        dtype = entry.section("precision").get("dtype") or DEFAULT_DTYPE
        try:
            decision = mtp_backend_decision(policy, facts.config, dtype)
        except CheckpointDecisionError as error:
            if strict:
                raise LilManifestError("%s: %s" % (entry.name, error)) from error
            decision = None
        if decision is not None:
            config["moe_backend"] = decision.backend
            provenance["mtp_moe"] = {
                "quantization": decision.quantization,
                "backend": decision.backend,
                "evidence": list(decision.evidence),
            }
        elif strict:
            logger.warning("%s: MTP MoE backend undetermined (no checkpoint metadata); leaving vLLM's choice", entry.name)
        if policy.get("model") == "target":
            config["model"] = entry.model
            if entry.revision:
                config["revision"] = entry.revision
        for key, flag in (
            ("attention", "attention_backend"),
            ("draft_sample_method", "draft_sample_method"),
            ("rejection_sample_method", "rejection_sample_method"),
        ):
            if _present(policy, key):
                config[flag] = policy[key]
        return config, tokens
    if method == "dflash":
        tokens = policy.get("tokens") or 0
        if tokens == 0:
            return None, 0
        config = {"method": "dflash", "model": policy["model"], "num_speculative_tokens": tokens, "kv_cache_dtype": "auto"}
        if _present(policy, "attention"):
            config["attention_backend"] = policy["attention"]
        ignored = sorted(set(policy) - {"tokens", "model", "attention"})
        if ignored:
            provenance.setdefault("ignored", []).append("speculators.dflash.%s" % ", ".join(ignored))
        return config, tokens
    if method == "dspark":
        tokens = policy.get("tokens") or 0
        if tokens == 0:
            return None, 0
        config = {"method": "dspark", "model": entry.model, "num_speculative_tokens": tokens}
        if entry.revision:
            config["revision"] = entry.revision
        for key, flag in (
            ("draft_sample_method", "draft_sample_method"),
            ("rejection_sample_method", "rejection_sample_method"),
            ("attention", "attention_backend"),
            ("adaptive_verification_cost_scale", "adaptive_verification_cost_scale"),
        ):
            if _present(policy, key):
                config[flag] = policy[key]
        if policy.get("adaptive_verification"):
            config["enable_adaptive_verification"] = True
        return config, tokens
    raise LilManifestError("%s: unsupported speculator %r" % (entry.name, method))


def _compilation(
    entry: LilEntry, capacity: dict[str, Any], speculator: str, tokens: int, layer: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """``compilationConfig``: the merged mapping plus capture sizes for this shape."""
    base = entry.section("compilation")
    if not base and not layer:
        return None
    config = deep_merge(base, layer or {})
    query_len, multiplier = verification_shape(speculator, tokens)
    config["cudagraph_capture_sizes"] = cudagraph_capture_sizes(
        int(capacity["max_num_seqs"]), int(capacity["max_num_batched_tokens"]), query_len, multiplier
    )
    return config


def _env(values: dict[str, Any] | None) -> dict[str, str]:
    return {str(k): str(v) for k, v in (values or {}).items() if v is not None}


def _when(condition: dict[str, Any]) -> dict[str, Any]:
    """A lil ``when:`` in sparkrun selectors."""
    when: dict[str, Any] = {}
    kind = condition.get("kind")
    if kind == "local":
        when["nodes"] = 1
    elif kind == "spark_rdma":
        when["nodes"] = {"gt": 1}
    elif kind:
        raise LilManifestError("override kind must be local or spark_rdma, got %r" % kind)
    if condition.get("arch"):
        when["arch"] = condition["arch"]
    if condition.get("tp"):
        when["tp"] = condition["tp"]
    if condition.get("speculator"):
        when["config"] = {"speculator": condition["speculator"]}
    if not when:
        raise LilManifestError("override has an empty when:")
    return when


def _fit_tp(entry: LilEntry, facts: CheckpointFacts, provenance: dict[str, Any]) -> int | None:
    from sparkrun.platforms.dgx_spark import DGX_SPARK_MEMORY_GB, DGX_SPARK_SCHEDULING_FRACTION

    heads = attention_heads(facts.config) if facts.config else None
    if heads is None or not facts.weight_bytes:
        return None
    device_bytes = int(DGX_SPARK_MEMORY_GB * (1 << 30))
    tp = fit_tensor_parallel(heads, facts.weight_bytes, device_bytes, DGX_SPARK_SCHEDULING_FRACTION, MAX_FIT_TP)
    provenance["tensor_parallel"] = {
        "policy": "fit",
        "target": "DGX Spark GB10",
        "attention_heads": heads,
        "weight_bytes": facts.weight_bytes,
        "size": tp,
    }
    return tp


def translate(entry: LilEntry, facts: CheckpointFacts, *, strict: bool = True) -> dict[str, Any]:
    """The v2 recipe for *entry*. ``strict=False`` (listing) degrades instead of raising."""
    serving = entry.section("serving")
    precision = entry.section("precision")
    cache = entry.section("cache")
    kernels = entry.section("kernels")
    loading = entry.section("loading")
    speculators = entry.section("speculators")
    provenance: dict[str, Any] = {"entry": entry.name}
    if entry.family:
        provenance["family"] = entry.family
    if entry.catalog_commit:
        provenance["catalog_commit"] = entry.catalog_commit

    defaults: dict[str, Any] = {"port": DEFAULT_PORT, "host": "0.0.0.0", "served_model_name": serving["served_model_name"]}
    tp = _fit_tp(entry, facts, provenance)
    if tp is not None:
        defaults["tensor_parallel"] = tp
    elif strict:
        logger.warning("%s: tensor parallelism undetermined (no checkpoint facts); pass --tp", entry.name)

    # --- serving (buildVLLMArgv order) ---
    if serving.get("trust_remote_code"):
        defaults["trust_remote_code"] = True
    for key in ("tokenizer_mode", "reasoning_parser", "tool_call_parser", "generation_config"):
        if _present(serving, key):
            defaults[key] = serving[key]
    if serving.get("auto_tool_choice", bool(serving.get("tool_call_parser"))):
        defaults["enable_auto_tool_choice"] = True
    if serving.get("hf_overrides"):
        defaults["hf_overrides"] = deepcopy(serving["hf_overrides"])
    if serving.get("chat_template_kwargs"):
        defaults["default_chat_template_kwargs"] = deepcopy(serving["chat_template_kwargs"])
    defaults["async_scheduling"] = bool(serving.get("async_scheduling", False))
    defaults["scheduler_reserve_full_isl"] = bool(serving.get("scheduler_reserve_full_isl", True))
    defaults["enable_prefix_caching"] = bool(serving.get("prefix_caching", True))
    if defaults["enable_prefix_caching"] and _present(cache, "prefix_cache_retention_interval"):
        defaults["prefix_cache_retention_interval"] = cache["prefix_cache_retention_interval"]
    defaults["enable_chunked_prefill"] = bool(serving.get("chunked_prefill", True))
    for key in (
        "long_prefill_token_threshold",
        "prefill_schedule_interval",
        "prefill_compute_share",
        "prefill_compute_half_life",
        "max_parallel_prefills",
        "prefill_policy",
        "decode_refill_target",
        "jit_monitor_mode",
    ):
        if _present(serving, key):
            defaults[key] = serving[key]
    for key, flag in (
        ("prompt_tokens_details", "enable_prompt_tokens_details"),
        ("force_include_usage", "enable_force_include_usage"),
        ("request_id_headers", "enable_request_id_headers"),
    ):
        if serving.get(key):
            defaults[flag] = True
    multimodal = serving.get("multimodal") if isinstance(serving.get("multimodal"), dict) else {}
    if _present(multimodal, "encoder_tp_mode"):
        defaults["mm_encoder_tp_mode"] = multimodal["encoder_tp_mode"]
    if _present(multimodal, "processor_cache_gb"):
        defaults["mm_processor_cache_gb"] = multimodal["processor_cache_gb"]
    if multimodal.get("limit_per_prompt"):
        defaults["limit_mm_per_prompt"] = deepcopy(multimodal["limit_per_prompt"])

    # --- precision, cache, kernels ---
    defaults["dtype"] = precision.get("dtype") or DEFAULT_DTYPE
    defaults["kv_cache_dtype"] = cache.get("kv_cache_dtype") or DEFAULT_KV_CACHE_DTYPE
    if _present(precision, "quantization"):
        defaults["quantization"] = precision["quantization"]
    for key, flag in (
        ("attention", "attention_backend"),
        ("gdn_decode", "gdn_decode_kernel"),
        ("gdn_prefill", "gdn_prefill_backend"),
        ("kda_prefill", "kda_prefill_backend"),
        ("mamba", "mamba_backend"),
    ):
        if _present(kernels, key):
            defaults[flag] = kernels[key]
    defaults["linear_backend"] = kernels.get("linear") or DEFAULT_KERNEL_BACKEND
    defaults["moe_backend"] = kernels.get("moe") or DEFAULT_KERNEL_BACKEND
    if _present(kernels, "flashinfer_autotune"):
        defaults["enable_flashinfer_autotune"] = bool(kernels["flashinfer_autotune"])
    for key in ("block_size", "mamba_cache_mode", "recurrent_checkpoint_policy"):
        if _present(cache, key):
            defaults[key] = cache[key]
    if _present(cache, "indexer_kv_dtype"):
        defaults["attention_config"] = {"indexer_kv_dtype": cache["indexer_kv_dtype"]}

    # --- loading, engram ---
    env = dict(COMMON_ENV)
    overrides: list[dict[str, Any]] = []
    load_format = loading.get("load_format")
    if load_format:
        defaults["load_format"] = load_format
        if load_format == "b12x":
            env["VLLM_PLUGINS"] = "b12x_loader"
    else:
        defaults["load_format"] = "instanttensor"
        overrides.append({"when": {"arch": "sm_121"}, "defaults": {"load_format": "b12x"}, "env": {"VLLM_PLUGINS": "b12x_loader"}})
    if _present(loading, "safetensors_load_strategy"):
        defaults["safetensors_load_strategy"] = loading["safetensors_load_strategy"]
    if loading.get("loader_extra_config"):
        defaults["model_loader_extra_config"] = deepcopy(loading["loader_extra_config"])
    engram = entry.document.get("engram")
    if isinstance(engram, dict):
        defaults["engram_config"] = {
            "cpu_offload": engram.get("cpu_offload", True),
            "table_memory": engram.get("table_memory", "device"),
            "disk_resident_scales": engram.get("disk_resident_scales", False),
        }

    # --- capacity, speculation, compilation ---
    base_capacity = _capacity(entry)
    # A null capacity means "vLLM decides" in lil. At the base that is simply an
    # absent key; only an override layer needs an explicit None, to undo a value.
    defaults.update({k: v for k, v in _capacity_defaults(base_capacity, include_utilization=False).items() if v is not None})
    default_speculator = speculators.get("default") or "none"
    spec_config, spec_tokens = _speculative_config(default_speculator, entry, facts, strict=strict, provenance=provenance)
    if spec_config is not None:
        defaults["speculative_config"] = spec_config
    compilation = _compilation(entry, base_capacity, default_speculator, spec_tokens)
    if compilation is not None:
        defaults["compilation_config"] = compilation
    env.update(_env(entry.section("environment")))

    # --- topology tuning ---
    overrides.append({"when": {"nodes": 1, "gpus_per_node": {"gt": 1}}, "env": dict(MULTI_GPU_HOST_ENV)})
    overrides.append({"when": {"nodes": {"gt": 1}}, "defaults": {"disable_custom_all_reduce": True}, "env": dict(MULTI_NODE_ENV)})
    if base_capacity.get("gpu_memory_utilization") is not None:
        overrides.append(
            {
                "when": {"capability": {"ne": "unified-memory"}},
                "defaults": {"gpu_memory_utilization": base_capacity["gpu_memory_utilization"]},
            }
        )

    # --- speculator alternatives: -o speculator=<method> ---
    alternatives = [m for m in ("mtp", "dflash", "dspark") if isinstance(speculators.get(m), dict) and m != default_speculator]
    if default_speculator != "none":
        alternatives.append("none")
    if alternatives:
        defaults["speculator"] = default_speculator
    for method in alternatives:
        try:
            alt_config, alt_tokens = _speculative_config(method, entry, facts, strict=strict, provenance={})
        except LilManifestError as error:
            if strict:
                raise
            logger.debug("%s: skipping %s alternative: %s", entry.name, method, error)
            continue
        layer: dict[str, Any] = {"speculative_config": alt_config}
        alt_compilation = _compilation(entry, base_capacity, method, alt_tokens)
        if alt_compilation is not None:
            layer["compilation_config"] = alt_compilation
        overrides.append({"when": {"config": {"speculator": method}}, "defaults": layer})

    # --- the manifest's own overrides, in order (later wins, as in lil) ---
    for index, raw in enumerate(entry.document.get("overrides") or []):
        if not isinstance(raw, dict) or not isinstance(raw.get("when"), dict):
            raise LilManifestError("%s: overrides[%d] needs a when: mapping" % (entry.name, index))
        translated: dict[str, Any] = {"when": _when(raw["when"])}
        layer_defaults: dict[str, Any] = {}
        capacity_layer = raw.get("capacity") if isinstance(raw.get("capacity"), dict) else None
        if capacity_layer:
            layer_defaults.update(_capacity_defaults(capacity_layer, include_utilization=True))
        speculator = raw["when"].get("speculator") or default_speculator
        tokens = spec_tokens
        if speculator != default_speculator:
            tokens = _speculative_config(speculator, entry, facts, strict=False, provenance={})[1]
        if capacity_layer or raw.get("compilation") or raw["when"].get("speculator"):
            compiled = _compilation(entry, _capacity(entry, capacity_layer), speculator, tokens, raw.get("compilation"))
            if compiled is not None:
                layer_defaults["compilation_config"] = compiled
        if layer_defaults:
            translated["defaults"] = layer_defaults
        if raw.get("environment"):
            translated["env"] = _env(raw["environment"])
        if len(translated) > 1:
            overrides.append(translated)

    requires = entry.section("requires").get("arch")
    if requires:
        provenance["requires_arch"] = list(requires)

    recipe: dict[str, Any] = {
        "recipe_version": "2",
        "model": entry.model,
        "runtime": "vllm-distributed",
        "container": LIL_CONTAINER,
        "description": entry.document.get("description") or "",
        "defaults": defaults,
        "env": env,
        "overrides": overrides,
        "metadata": {"source": "lil", "lil": provenance},
    }
    if entry.revision:
        recipe["model_revision"] = entry.revision
    return recipe
