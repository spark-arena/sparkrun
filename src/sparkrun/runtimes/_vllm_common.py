"""Shared mixin for vLLM runtimes (vllm-ray and vllm-distributed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from sparkrun.runtimes._util import default_env_hf_offline, ptrace_executor_config, resolve_api_key

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


class VllmMixin:
    """Shared methods for vLLM runtimes.

    Provides tuning config auto-mounting and version detection
    that are identical between vllm-ray and vllm-distributed.
    """

    def get_common_env(self):
        return default_env_hf_offline()

    def default_executor_config(self) -> dict:
        """Allow attaching a stack sampler to a hung vLLM engine.

        vLLM's own hang-debugging guidance is ``py-spy dump --pid <EngineCore
        pid>``, which needs ``CAP_SYS_PTRACE`` because EngineCore is a sibling
        of whatever shell the operator execs in.  See
        :func:`~sparkrun.runtimes._util.ptrace_executor_config`.
        """
        return {**super().default_executor_config(), **ptrace_executor_config()}

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

    def runtime_cache_paths(self, *, fingerprint: str = "") -> dict:
        """Persist vLLM's compile / JIT caches across container restarts.

        All of these are content-addressed internally (torch.compile hashes its
        config, Triton and FlashInfer key by kernel signature, CuTeDSL
        fingerprints its sources and toolchain), which is why vLLM does not
        need ``key_by_image`` — see :mod:`sparkrun.core.runtime_cache`.

        ``VLLM_CACHE_ROOT`` and the FlashInfer pair are named explicitly rather
        than left to the ``XDG_CACHE_HOME`` catch-all because those libraries
        expand ``~/.cache/...`` directly instead of consulting XDG.

        **FlashInfer needs two entries, and the one that works is the
        workspace base.**  ``FLASHINFER_CACHE_DIR`` reads like the lever but is
        not an environment variable at all — in ``flashinfer/jit/env.py`` it is
        a module *attribute* derived as ``FLASHINFER_WORKSPACE_BASE /
        ".cache" / "flashinfer"``, with the base defaulting to
        ``Path.home()``.  So the JIT output (``cached_ops``, ``generated``)
        followed the container's throwaway ``HOME`` and was recompiled on every
        launch.  Verified against flashinfer 0.6.11 and 0.6.18.  Both vars
        point at the same subtree: the base grows a ``.cache/flashinfer/...``
        tree under it, and ``FLASHINFER_CACHE_DIR`` is kept — harmless today,
        and correct if a later release starts honoring it.

        ``CUTE_DSL_CACHE_DIR`` is NVIDIA's CuTeDSL generated-IR cache
        (``nvidia_cutlass_dsl``), which vLLM's ``vllm_flash_attn.cute``,
        FlashInfer's sparse kernels and the b12x kernel stack all compile
        through.  Unset it goes to ``$TMPDIR/<user>/cutlass_python_cache`` —
        neither XDG nor ``HOME``, so nothing else here reaches it.
        ``FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`` is the peer used by the vendored
        FlashAttention CuTe copies; that cache is opt-in
        (``FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1``) but pointing it costs an
        empty directory and makes enabling it actually persist.

        b12x itself needs no entry: its compile cache resolves
        ``B12X_COMPILE_CACHE_DIR`` → ``$XDG_CACHE_HOME/b12x/compile``, so the
        catch-all already covers it.  Relocating these is safe with respect to
        b12x's own cache key, which treats ``B12X_COMPILE_CACHE_DIR`` and
        ``CUTE_DSL_CACHE_DIR`` as operational and excludes them from the digest.
        """
        from sparkrun.core.runtime_cache import CachePath

        return {
            "VLLM_CACHE_ROOT": CachePath("vllm"),
            "TORCHINDUCTOR_CACHE_DIR": CachePath("inductor"),
            "TRITON_CACHE_DIR": CachePath("triton"),
            "FLASHINFER_CACHE_DIR": CachePath("flashinfer"),
            "FLASHINFER_WORKSPACE_BASE": CachePath("flashinfer"),
            "CUTE_DSL_CACHE_DIR": CachePath("cute_dsl"),
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": CachePath("flash_attn_cute_dsl"),
        }

    def finalize_host_comm_env(self, host_env: dict[str, str]) -> dict[str, str]:
        """Advertise vLLM on the host's resolved ``NODE_IP``.

        vLLM infers its message-queue / distributed host address from the
        default route unless ``VLLM_HOST_IP`` is set.  Mirror the finalized
        per-host ``NODE_IP`` (which the init-network selection may have pinned
        to the IB/CX7 fabric) so vLLM binds the same network as the rendezvous
        address instead of the default-route interface.
        """
        host_env = super().finalize_host_comm_env(host_env)
        node_ip = host_env.get("NODE_IP")
        if node_ip:
            host_env = {**host_env, "VLLM_HOST_IP": node_ip}
        return host_env

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        cmds["vllm"] = "python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown"
        return cmds

    def _apply_distributed_backend(self, command: str, config, skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Sync a rendered command's ``--distributed-executor-backend`` value.

        ``render_command`` leaves a literal ``--distributed-executor-backend
        ray`` in a recipe ``command`` untouched, so a CLI override
        (``-o distributed_executor_backend=mp``) would otherwise be ignored.
        When the config chain resolves a value for the key — i.e. it was set
        in ``defaults`` or via ``-o`` — force the command to that value;
        when it resolves nothing (the value lives only in the literal
        command), leave the command as-is.
        """
        if "distributed_executor_backend" in skip_keys:
            return command
        value = config.get("distributed_executor_backend")
        if value is None:
            return command
        return self.reconcile_flag_in_command(command, "--distributed-executor-backend", value, override=True)

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
    # Serving-behaviour flags. These are spelled in nearly every real recipe's
    # ``command:`` template; without them a command-less recipe silently serves
    # a differently-configured server (no tool parsing, default attention
    # backend, default weight loader).
    "attention_backend": "--attention-backend",
    "load_format": "--load-format",
    "reasoning_parser": "--reasoning-parser",
    "tool_call_parser": "--tool-call-parser",
    "chat_template": "--chat-template",
    "speculative_config": "--speculative-config",
    "tokenizer_mode": "--tokenizer-mode",
    "mm_encoder_tp_mode": "--mm-encoder-tp-mode",
    "block_size": "--block-size",
    "seed": "--seed",
    # NOTE: ``enable_auto_tool_choice`` / ``enable_chunked_prefill`` /
    # ``async_scheduling`` are booleans, but they must ALSO appear here.
    # ``build_flags_from_map`` iterates this map and consults ``bool_keys``
    # only to decide how to *render* a key it has already found, so a key
    # listed solely in VLLM_BOOL_FLAGS is unreachable. Keep the two in sync.
    "enable_auto_tool_choice": "--enable-auto-tool-choice",
    "enable_chunked_prefill": "--enable-chunked-prefill",
    "async_scheduling": "--async-scheduling",
}

# Boolean flags (present = True, absent = False).
# Every entry here MUST also have an entry in VLLM_FLAG_MAP — see the note above.
VLLM_BOOL_FLAGS = {
    "enforce_eager",
    "enable_prefix_caching",
    "trust_remote_code",
    "enable_auto_tool_choice",
    "enable_chunked_prefill",
    "async_scheduling",
}
