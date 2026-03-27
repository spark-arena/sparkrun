"""Shared mixin for vLLM runtimes (vllm-ray and vllm-distributed)."""

from __future__ import annotations

from sparkrun.runtimes._util import default_env_hf_offline


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
    "kv_cache_dtype": "--kv-cache-dtype",
}

# Boolean flags (present = True, absent = False)
VLLM_BOOL_FLAGS = {
    "enforce_eager",
    "enable_prefix_caching",
    "trust_remote_code",
    "enable_auto_tool_choice",
}
