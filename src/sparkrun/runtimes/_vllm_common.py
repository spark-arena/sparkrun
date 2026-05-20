"""Shared mixin for vLLM runtimes (vllm-ray and vllm-distributed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from sparkrun.runtimes._util import default_env_hf_offline, resolve_api_key

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

    def resolve_api_key(
        self,
        recipe: "Recipe",
        overrides: dict | None = None,
    ) -> str | None:
        """Resolve the vLLM ``--api-key`` value for proxy/discovery use.

        Delegates to :func:`sparkrun.runtimes._util.resolve_api_key` with
        ``env_var="VLLM_API_KEY"`` and ``flag_name="--api-key"``.
        """
        return resolve_api_key(recipe, overrides, "VLLM_API_KEY", "--api-key")

    def _build_base_command(
        self,
        recipe: "Recipe",
        config,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build the ``vllm serve`` command without cluster-specific arguments.

        Emits ``vllm serve <model> [-tp N] [--flag value ...]`` from the
        config chain.  ``tensor_parallel`` and ``distributed_executor_backend``
        are always added to the skip set since callers append them
        explicitly (or omit them) based on the clustering strategy.
        """
        parts = ["vllm", "serve", recipe.model]

        tp = config.get("tensor_parallel")
        if tp:
            parts.extend(["-tp", str(tp)])

        skip = {"tensor_parallel", "distributed_executor_backend"}
        skip.update(skip_keys)
        parts.extend(
            self.build_flags_from_map(
                config,
                VLLM_FLAG_MAP,
                bool_keys=VLLM_BOOL_FLAGS,
                skip_keys=skip,
            )
        )

        return " ".join(parts)

    def _build_command(
        self,
        recipe: "Recipe",
        config,
        is_cluster: bool,
        num_nodes: int,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        *,
        cluster_backend: str | None = None,
        master_port: int = 25000,
    ) -> str:
        """Build the ``vllm serve`` command from structured config.

        The non-cluster path produces the bare ``vllm serve`` invocation.
        Cluster mode appends either:

        * ``--distributed-executor-backend <backend>`` when
          *cluster_backend* is set (Ray runtime), or
        * ``--nnodes <num_nodes> --master-addr <head_ip> --master-port
          <master_port>`` when *head_ip* is supplied (native distributed).

        For native distributed, ``--node-rank`` is intentionally omitted —
        that is the responsibility of :meth:`generate_node_command`.

        Args:
            recipe: The loaded recipe.
            config: Resolved config chain (``recipe.build_config_chain(...)``).
            is_cluster: Whether the workload is multi-node.
            num_nodes: Total node count (used for ``--nnodes``).
            head_ip: Head IP for native distributed cluster.  Ignored when
                *cluster_backend* is set.
            skip_keys: Config keys to omit from flag emission.
            cluster_backend: Optional distributed-executor backend
                (e.g. ``"ray"``); when set, appends
                ``--distributed-executor-backend`` instead of native
                ``--nnodes``/``--master-addr`` flags.
            master_port: Master coordination port for native distributed.
        """
        base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        if not is_cluster:
            return base

        if cluster_backend:
            return base + " --distributed-executor-backend %s" % cluster_backend

        if head_ip:
            return base + " --nnodes %d --master-addr %s --master-port %d" % (num_nodes, head_ip, master_port)

        return base

    def detect_spec_config_draft_model(self, recipe: "Recipe") -> str | None:
        try:
            # TODO: support various ways that speculative config can be specified
            # noinspection PyProtectedMember
            spec_cfg = recipe._effective_default("speculative_config")
            spec_cfg_dict = json.loads(spec_cfg) or {}
            # intended primarily for dflash, but we allow any "model" field for future extensibility
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
    "api_key": "--api-key",
}

# Boolean flags (present = True, absent = False)
VLLM_BOOL_FLAGS = {
    "enforce_eager",
    "enable_prefix_caching",
    "trust_remote_code",
    "enable_auto_tool_choice",
}
