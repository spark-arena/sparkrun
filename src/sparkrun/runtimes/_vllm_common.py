"""Shared mixin for vLLM runtimes (vllm-ray and vllm-distributed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from sparkrun.runtimes._util import default_env_hf_offline

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


class VllmMixin:
    """Shared methods for vLLM runtimes.

    Provides tuning config auto-mounting and version detection
    that are identical between vllm-ray and vllm-distributed.
    """

    def get_common_env(self):
        return default_env_hf_offline()

    def get_extra_volumes(self) -> dict[str, str]:
        """Mount vLLM tuning configs if available."""
        from sparkrun.tuning.vllm import get_vllm_tuning_volumes

        return get_vllm_tuning_volumes() or {}

    def get_extra_env(self) -> dict[str, str]:
        """Set VLLM_TUNED_CONFIG_FOLDER if tuning configs exist."""
        from sparkrun.tuning.vllm import get_vllm_tuning_env

        env = super().get_extra_env()
        env.update(get_vllm_tuning_env() or {})
        return env

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        cmds["vllm"] = "python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown"
        return cmds

    def resolve_overrides_for_auto(
        self,
        recipe: "Recipe",
        overrides: dict,
    ) -> dict:
        """Resolve ``max_model_len: "auto"`` to a concrete integer.

        Checks the config chain for ``max_model_len``.  If the effective
        value is the string ``"auto"``, runs the recipe's VRAM estimator
        to calculate the maximum context length that fits within the GPU
        memory budget, and returns a *new* overrides dict with that
        integer injected so the config chain will pick it up.

        If estimation fails (missing model metadata, etc.) the override
        is left out — ``max_model_len`` will be ``None`` in the config
        chain, which omits ``--max-model-len`` from the CLI and lets
        vLLM pick its own default.

        Returns:
            A (possibly updated) overrides dict.
        """
        config = recipe.build_config_chain(overrides)
        raw = config.get("max_model_len")
        if raw is None or str(raw).lower() != "auto":
            return overrides

        import logging

        log = logging.getLogger(__name__)
        log.info("max_model_len is 'auto'; calculating from VRAM budget …")

        try:
            est = recipe.estimate_vram(cli_overrides=overrides, auto_detect=True)
        except Exception:
            log.warning(
                "VRAM estimation failed; omitting --max-model-len "
                "(vLLM will use its default)",
                exc_info=True,
            )
            return {**overrides, "max_model_len": None}

        if est.max_context_tokens is not None and est.max_context_tokens > 0:
            log.info(
                "Resolved max_model_len='auto' → %d tokens "
                "(%.1f GB KV headroom on %.1f GB usable)",
                est.max_context_tokens,
                est.available_kv_gb or 0,
                est.usable_gpu_memory_gb or 0,
            )
            return {**overrides, "max_model_len": est.max_context_tokens}

        log.warning(
            "VRAM estimation could not determine max_context_tokens; "
            "omitting --max-model-len (vLLM will use its default)"
        )
        return {**overrides, "max_model_len": None}

    def detect_spec_config_draft_model(self, recipe: "Recipe") -> str | None:
        try:
            # TODO: support various ways that speculative config can be specified
            # noinspection PyProtectedMember
            spec_cfg = recipe._effective_default("speculative_config")
            spec_cfg_dict = json.loads(spec_cfg)
            if spec_cfg_dict["method"] == "dflash":
                return spec_cfg_dict.get("model", None)
        except Exception:
            return None


# Standard vLLM CLI flags and their recipe default keys
VLLM_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "tensor_parallel": "-tp",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-model-len",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "max_num_seqs": "--max-num-seqs",
    "served_model_name": "--served-model-name",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "enforce_eager": "--enforce-eager",
    "enable_prefix_caching": "--enable-prefix-caching",
    "trust_remote_code": "--trust-remote-code",
    "distributed_executor_backend": "--distributed-executor-backend",
    "pipeline_parallel": "-pp",
    "data_parallel": "--data-parallel-size",
    "kv_cache_dtype": "--kv-cache-dtype",
    "otlp_traces_endpoint": "--otlp-traces-endpoint",
}

# Boolean flags (present = True, absent = False)
VLLM_BOOL_FLAGS = {
    "enforce_eager",
    "enable_prefix_caching",
    "trust_remote_code",
    "enable_auto_tool_choice",
}
