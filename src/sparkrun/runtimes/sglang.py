"""Native SGLang runtime for sparkrun."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes._util import default_env_hf_offline, ptrace_executor_config, resolve_api_key
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv

logger = logging.getLogger(__name__)

# SGLang CLI flag mapping
_SGLANG_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "tensor_parallel": "--tp-size",
    "pipeline_parallel": "--pp-size",
    "gpu_memory_utilization": "--mem-fraction-static",
    "max_model_len": "--context-length",
    "max_num_seqs": "--max-running-requests",
    "served_model_name": "--served-model-name",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "trust_remote_code": "--trust-remote-code",
    "chunked_prefill": "--chunked-prefill-size",
    "kv_cache_dtype": "--kv-cache-dtype",
    "tokenizer_path": "--tokenizer-path",
    "api_key": "--api-key",
    # Serving-behaviour flags, spelled in nearly every real SGLang recipe's
    # ``command:`` template. Without them a command-less recipe silently serves
    # a differently-configured server.
    "attention_backend": "--attention-backend",
    "load_format": "--load-format",
    "reasoning_parser": "--reasoning-parser",
    "tool_call_parser": "--tool-call-parser",
    "mm_feature_transport": "--mm-feature-transport",
    # Speculative decoding (NEXTN / EAGLE / DSPARK).
    "speculative_algorithm": "--speculative-algorithm",
    "speculative_draft_model_path": "--speculative-draft-model-path",
    "speculative_draft_model_revision": "--speculative-draft-model-revision",
    "speculative_num_steps": "--speculative-num-steps",
    "speculative_eagle_topk": "--speculative-eagle-topk",
    "speculative_num_draft_tokens": "--speculative-num-draft-tokens",
    "speculative_dspark_block_size": "--speculative-dspark-block-size",
    # CUDA graph / torch.compile batch sizing.
    "cuda_graph_bs": "--cuda-graph-bs",
    "cuda_graph_max_bs": "--cuda-graph-max-bs",
    "cuda_graph_max_bs_decode": "--cuda-graph-max-bs-decode",
    "torch_compile_max_bs": "--torch-compile-max-bs",
    "num_continuous_decode_steps": "--num-continuous-decode-steps",
    # Kernel backends and hybrid-SSM knobs.
    "fp8_gemm_backend": "--fp8-gemm-backend",
    "fp4_gemm_backend": "--fp4-gemm-backend",
    "moe_runner_backend": "--moe-runner-backend",
    "mamba_ssm_dtype": "--mamba-ssm-dtype",
    "mamba_full_memory_ratio": "--mamba-full-memory-ratio",
    # NOTE: the boolean keys below must ALSO be listed here (``trust_remote_code``
    # already is, above). ``build_flags_from_map`` iterates this map and consults
    # ``bool_keys`` only to decide how to *render* a key it has already found, so
    # a key listed solely in _SGLANG_BOOL_FLAGS is unreachable. Keep them in sync.
    "enable_torch_compile": "--enable-torch-compile",
    "disable_radix_cache": "--disable-radix-cache",
    "disable_prefill_cuda_graph": "--disable-prefill-cuda-graph",
}

# Boolean flags (present = True, absent = False).
# Every entry here MUST also have an entry in _SGLANG_FLAG_MAP — see note above.
_SGLANG_BOOL_FLAGS = {
    "trust_remote_code",
    "enable_torch_compile",
    "disable_radix_cache",
    "disable_prefill_cuda_graph",
}


class SglangRuntime(RuntimePlugin):
    """Native SGLang runtime using prebuilt container images.

    SGLang uses its own distributed init mechanism for multi-node inference,
    not Ray.  Each node runs the full ``sglang serve`` command with
    ``--dist-init-addr``, ``--nnodes``, and ``--node-rank`` arguments.
    """

    runtime_name = "sglang"
    default_image_prefix = "scitrera/dgx-spark-sglang"

    # Native distribution: each node runs its own serve process and rendezvous
    # is over the wire, so per-machine tuned images are meaningful here.  Ranks
    # still share NCCL/torch ABI expectations — the images are expected to be
    # differently-tuned builds of the same stack, not different stacks.
    supports_heterogeneous_images = True

    # Native distribution: each node runs its own serve process and rendezvous
    # is over the wire, so per-machine tuned images are meaningful here.  Ranks
    # still share NCCL/torch ABI expectations — the images are expected to be
    # differently-tuned builds of the same stack, not different stacks.
    supports_heterogeneous_images = True

    def cluster_strategy(self) -> str:
        """SGLang uses native multi-node distribution, not Ray."""
        return "native"

    def known_config_keys(self) -> frozenset[str]:
        """Flag-map keys plus the SGLang keys read outside it.

        The speculative draft-model keys are consumed by ``prepare()`` /
        distribution rather than emitted from the map.  See
        :func:`sparkrun.core.launcher.report_unmapped_config_keys`.
        """
        return frozenset(_SGLANG_FLAG_MAP) | {
            "speculative_draft_model",
            "speculative_draft_model_path",
            "speculative_draft_model_revision",
        }

    def serve_flag_map(self):
        return _SGLANG_FLAG_MAP

    def resolve_api_key(
        self,
        recipe: "Recipe",
        overrides: dict | None = None,
    ) -> str | None:
        """Resolve the SGLang ``--api-key`` value for proxy/discovery use.

        Delegates to :func:`sparkrun.runtimes._util.resolve_api_key` with
        ``env_var="SGLANG_API_KEY"`` and ``flag_name="--api-key"``.
        """
        return resolve_api_key(recipe, overrides, "SGLANG_API_KEY", "--api-key")

    def prepare(
        self,
        recipe: Recipe,
        hosts: list[str],
        config: "SparkrunConfig | None" = None,
        dry_run: bool = False,
        transfer_mode: str = "auto",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Pre-sync the speculative draft model when configured."""
        draft_model = self._detect_speculative_draft_model(recipe)
        if draft_model:
            recipe.distribution_config.add_model(draft_model, revision=self._detect_speculative_draft_revision(recipe))

    @staticmethod
    def _detect_speculative_draft_model(recipe: "Recipe") -> str | None:
        """Resolve the speculative draft model from recipe defaults.

        Accepts either ``speculative_draft_model_path`` (canonical, matches
        the CLI flag) or ``speculative_draft_model`` (alias).  Returns
        ``None`` when neither is set.
        """
        for key in ("speculative_draft_model_path", "speculative_draft_model"):
            # noinspection PyProtectedMember
            val = recipe._effective_default(key)
            if val:
                return str(val)
        return None

    @staticmethod
    def _detect_speculative_draft_revision(recipe: "Recipe") -> str | None:
        """Resolve the pin for the draft model's *own* repo, if declared.

        Distribution-only, like the draft model path itself — SGLang has no
        serve flag for it.  Absent a declaration the draft model is fetched
        unpinned; the recipe's ``model_revision`` is emphatically not a
        substitute, since that SHA exists only in the served model's repo.
        """
        # noinspection PyProtectedMember
        val = recipe._effective_default("speculative_draft_model_revision")
        return str(val) if val else None

    def generate_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the sglang serve command.

        For cluster mode this produces the *base* command without
        ``--node-rank``.  Use :meth:`generate_node_command` to get the
        per-node variant.
        """
        config = recipe.build_config_chain(overrides)
        self._normalize_config(config)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--served-model-name",
                skip_keys,
            )
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    _SGLANG_FLAG_MAP,
                    _SGLANG_BOOL_FLAGS,
                )
            return rendered

        return self._build_command(recipe, config, is_cluster, num_nodes, head_ip, skip_keys=skip_keys)

    def generate_node_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        head_ip: str,
        num_nodes: int,
        node_rank: int,
        init_port: int = 25000,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        hosts: list[str] | None = None,
        placement=None,
    ) -> str:
        """Generate the sglang command for a specific node.

        Produces the full ``sglang serve`` invocation with the
        node-specific ``--dist-init-addr``, ``--nnodes``, and
        ``--node-rank`` flags appended.
        """
        config = recipe.build_config_chain(overrides)
        self._normalize_config(config)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--served-model-name",
                skip_keys,
            )
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    _SGLANG_FLAG_MAP,
                    _SGLANG_BOOL_FLAGS,
                )
            base = rendered
        else:
            base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        # SGLang forms a SINGLE global torch.distributed world rendezvoused at
        # the head node: one --dist-init-addr for the whole job, with TP/PP/DP
        # grouping handled internally by sglang.  The rendezvous host is therefore
        # ALWAYS the head node.  We deliberately do NOT forward hosts/placement
        # here: with the default replica_size=1, _resolve_master_addr maps
        # node_rank -> hosts[node_rank] (each node's own IP), so every worker would
        # point --dist-init-addr at itself and only rank 0 would bind the store —
        # manifesting as "1/N clients joined" rendezvous timeouts.
        node_args = self._make_node_command_args(
            head_ip=head_ip,
            num_nodes=num_nodes,
            node_rank=node_rank,
            init_port=init_port,
        )

        # Append sglang multi-node arguments.  SGLang combines master_addr
        # and master_port into a single --dist-init-addr HOST:PORT flag.
        parts = [
            base,
            "--dist-init-addr %s:%s" % (node_args["master_addr"], node_args["master_port"]),
            "--nnodes %s" % node_args["num_nodes"],
            "--node-rank %s" % node_args["node_rank"],
        ]
        return " ".join(parts)

    @staticmethod
    def _inject_gguf_model(config) -> None:
        """Ensure ``{model}`` in command templates resolves to the GGUF file path.

        When a GGUF model has been pre-synced, the CLI stores the
        container-internal path as ``_gguf_model_path`` in overrides.
        This helper copies that value into the ``model`` key so that
        ``{model}`` in recipe command templates renders the local file
        path instead of the raw HF repo spec (which includes the
        sparkrun-specific ``:quant`` suffix that runtimes cannot parse).
        """
        gguf_path = config.get("_gguf_model_path")
        if gguf_path:
            config.put("model", str(gguf_path))

    @staticmethod
    def _normalize_config(config) -> None:
        """Apply pre-render config normalizations (GGUF path + speculative alias)."""
        SglangRuntime._inject_gguf_model(config)
        # Accept ``speculative_draft_model`` as an alias for the canonical
        # ``speculative_draft_model_path`` key so users can use either in
        # recipe defaults; flag emission only looks at the canonical key.
        if not config.get("speculative_draft_model_path"):
            alias = config.get("speculative_draft_model")
            if alias:
                config.set("speculative_draft_model_path", alias)

    def _build_base_command(self, recipe: Recipe, config, skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the sglang command without cluster-specific arguments."""
        # For GGUF models, use the resolved file path instead of the HF repo name
        model_path = config.get("_gguf_model_path") or recipe.model
        # ``sglang serve`` is the current entrypoint; ``python3 -m
        # sglang.launch_server`` is the legacy spelling it replaced. Recipes
        # carrying either form in an explicit ``command:`` still work — the
        # runtime detector (``core.recipe._CMD_SGLANG_RE``) matches both.
        parts = ["sglang", "serve", "--model-path", str(model_path)]

        tp = config.get("tensor_parallel")
        if tp:
            parts.extend(["--tp-size", str(tp)])

        skip = {"tensor_parallel"}
        skip.update(skip_keys)
        parts.extend(
            self.build_flags_from_map(
                config,
                _SGLANG_FLAG_MAP,
                bool_keys=_SGLANG_BOOL_FLAGS,
                skip_keys=skip,
            )
        )

        return " ".join(parts)

    def _build_command(
        self,
        recipe: Recipe,
        config,
        is_cluster: bool,
        num_nodes: int,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build the sglang serve command from structured config.

        For cluster mode, includes ``--dist-init-addr`` and ``--nnodes`` but
        NOT ``--node-rank`` (that is added per-node by the orchestrator or
        by :meth:`generate_node_command`).
        """
        base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        if is_cluster and head_ip:
            base += " --dist-init-addr %s:25000 --nnodes %d" % (head_ip, num_nodes)

        return base

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        cmds["sglang"] = "python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo unknown"
        return cmds

    def get_common_env(self):
        return default_env_hf_offline()

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return SGLang-specific cluster environment variables."""
        return {
            **RuntimePlugin.get_cluster_env(self, head_ip, num_nodes),
            "NCCL_CUMEM_ENABLE": "0",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",  # confirmed for v0.5.9 on 20260205 by DB
        }

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate SGLang-specific recipe fields."""
        from sparkrun.models.download import is_gguf_model

        issues = super().validate_recipe(recipe)

        if recipe.model and is_gguf_model(recipe.model):
            tokenizer = (recipe.defaults or {}).get("tokenizer_path")
            cmd = recipe.command or ""
            cmd_has_tokenizer = "--tokenizer-path" in cmd or "{tokenizer_path}" in cmd

            # Both are declared errors: SGLang refuses to load a GGUF model
            # without a tokenizer path, so either shape fails at startup —
            # after the weights have been downloaded and fanned out.
            if not tokenizer and not cmd_has_tokenizer:
                issues.append(
                    self.recipe_error(
                        "GGUF model detected but no tokenizer path configured. "
                        "SGLang requires --tokenizer-path pointing to the base (non-GGUF) HF model. "
                        "Set 'tokenizer_path' in defaults (e.g. tokenizer_path: Qwen/Qwen3-1.7B) "
                        "or add --tokenizer-path to the command template."
                    )
                )
            if tokenizer and cmd and not cmd_has_tokenizer:
                issues.append(
                    self.recipe_error(
                        "GGUF recipe has 'tokenizer_path' in defaults but the command "
                        "template does not reference {tokenizer_path} or --tokenizer-path, "
                        "so the value is dropped. "
                        "Add '--tokenizer-path {tokenizer_path}' to the command template."
                    )
                )

        return issues

    def default_executor_config(self) -> dict[str, Any]:
        """Allow attaching a stack sampler to a hung SGLang process.

        SGLang's watchdog reports *that* a scheduler stalled but not where;
        pinning it down means ``py-spy dump`` against the scheduler /
        detokenizer / TP-worker process, none of which is a descendant of an
        operator's exec shell.  See
        :func:`~sparkrun.runtimes._util.ptrace_executor_config`.
        """
        return {**super().default_executor_config(), **ptrace_executor_config()}

    # --- Tuning config auto-mount ---

    def get_extra_volumes(self) -> dict[str, str]:
        """Mount SGLang tuning configs if available."""
        from sparkrun.tuning.sglang import get_sglang_tuning_volumes

        return get_sglang_tuning_volumes() or {}

    def get_extra_env(self) -> dict[str, str]:
        """Set SGLANG_MOE_CONFIG_DIR if tuning configs exist."""
        from sparkrun.tuning.sglang import get_sglang_tuning_env

        env = super().get_extra_env()
        env.update(get_sglang_tuning_env() or {})
        return env

    # --- Compilation cache ---

    def runtime_cache_paths(self, *, fingerprint: str = "") -> dict:
        """Persist SGLang's torch.compile / Triton / FlashInfer / SGLang caches.

        SGLang's own graph cache goes through torch.compile, so Inductor covers
        it.  All of these are content-addressed or version-keyed internally and
        safe to share across images.

        ``FLASHINFER_WORKSPACE_BASE`` and the CuTeDSL pair carry the same
        rationale they do in :meth:`VllmMixin.runtime_cache_paths`, and for the
        same libraries — ``FLASHINFER_CACHE_DIR`` is not an env var upstream
        reads, and the CuTeDSL generated-IR cache defaults into ``$TMPDIR``.

        ``SGLANG_CACHE_DIR`` is the root of everything SGLang caches *itself*,
        and is separate from ``FLASHINFER_CACHE_DIR`` even though both have
        "flashinfer" trees.  ``FLASHINFER_CACHE_DIR`` is FlashInfer's own JIT
        cubin cache; the FlashInfer **autotune** results SGLang collects on
        startup are SGLang's, written to
        ``$SGLANG_CACHE_DIR/flashinfer/autotune/<fi-version>/sm<arch>/<cfg-hash>/rank_*.json``.
        Leaving it unset sent them to ``~/.cache/sglang`` — i.e. ``/tmp`` under
        the ``$SHELL_USER`` container's ``HOME=/tmp`` — so every launch re-ran
        the autotune sweep.  The XDG catch-all does not cover it: the default is
        a literal ``os.path.expanduser("~/.cache/sglang")``.

        ``SGLANG_JIT_CACHE_DIR`` is set explicitly rather than left to track
        ``SGLANG_CACHE_DIR``, because it does not: unset, SGLang's JIT build
        cache falls back to a hardcoded ``~/.cache/sglang/jit``.  (The DeepGEMM
        cache, ``SGLANG_DG_CACHE_DIR``, *is* resolved lazily from
        ``SGLANG_CACHE_DIR`` upstream, so it needs no entry.)

        ``TILELANG_CACHE_DIR`` is the same class of miss from a different
        library — SGLang's TileLang kernels default to ``~/.tilelang/cache``,
        outside both XDG and the SGLang root.  Harmless when TileLang is absent
        from the image.

        ``TVM_FFI_CACHE_DIR`` covers SGLang's generated TVM-FFI extension
        libraries.  TVM-FFI defaults to the literal ``~/.cache/tvm-ffi`` and
        does not consult XDG, so a CRIU restore otherwise cannot reopen the
        mapped JIT ``.so`` after the capture container is gone.
        """
        from sparkrun.core.runtime_cache import CachePath

        return {
            "TORCHINDUCTOR_CACHE_DIR": CachePath("inductor"),
            "TRITON_CACHE_DIR": CachePath("triton"),
            "FLASHINFER_CACHE_DIR": CachePath("flashinfer"),
            "FLASHINFER_WORKSPACE_BASE": CachePath("flashinfer"),
            "CUTE_DSL_CACHE_DIR": CachePath("cute_dsl"),
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": CachePath("flash_attn_cute_dsl"),
            "SGLANG_CACHE_DIR": CachePath("sglang"),
            "SGLANG_JIT_CACHE_DIR": CachePath("sglang/jit"),
            "TILELANG_CACHE_DIR": CachePath("tilelang"),
            "TVM_FFI_CACHE_DIR": CachePath("tvm_ffi"),
        }

    # --- Cluster stop ---

    def _stop_cluster(
        self,
        hosts: list[str],
        cluster_id: str,
        config=None,
        dry_run: bool = False,
    ) -> int:
        """Stop an SGLang native cluster."""
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)

    # --- Cluster launch ---

    def _run_cluster(
        self,
        hosts: list[str],
        image: str,
        serve_command: str = "",
        recipe=None,
        overrides=None,
        *,
        cluster_id: str = "sparkrun0",
        env: dict[str, str] | None = None,
        cache_dir: str | None = None,
        config=None,
        dry_run: bool = False,
        detached: bool = True,
        comm_env: "ClusterCommEnv | None" = None,
        init_port: int = 25000,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        **kwargs,
    ) -> int:
        """Orchestrate a multi-node SGLang cluster using native distribution."""
        return self._run_native_cluster(
            hosts=hosts,
            image=image,
            serve_command=serve_command,
            recipe=recipe,
            overrides=overrides,
            cluster_id=cluster_id,
            env=env,
            cache_dir=cache_dir,
            config=config,
            dry_run=dry_run,
            detached=detached,
            comm_env=comm_env,
            init_port=init_port,
            skip_keys=skip_keys,
            banner_title="SGLang Cluster Launcher",
            port_label="Init Port",
            node_label="sglang node",
            **kwargs,
        )
