"""Native SGLang runtime for sparkrun."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple, TYPE_CHECKING

from scitrera_app_framework import ext_parse_bool

from sparkrun.runtimes._util import default_env_hf_offline, ptrace_executor_config, resolve_api_key
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv

logger = logging.getLogger(__name__)

#: Config key for SGLang's DP-attention mode, and the flag it renders to.
#: Read outside the flag map as well — see :meth:`SglangRuntime._dp_attention_enabled`.
_DP_ATTENTION_KEY = "enable_dp_attention"
_DP_ATTENTION_FLAG = "--enable-dp-attention"
_DP_SIZE_FLAG = "--dp-size"

# SGLang CLI flag mapping
_SGLANG_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "tensor_parallel": "--tp-size",
    "pipeline_parallel": "--pp-size",
    # NOTE: emitted by the caller, never by ``build_flags_from_map`` — see
    # ``_append_dp_size``.  ``--dp-size`` describes replicas inside ONE launch,
    # so whether it is legal depends on the launch topology, not on the recipe
    # alone.  Listed here so ``known_config_keys`` / ``strip_flags_from_command``
    # / the hardcoded-flag check all see it.
    "data_parallel": _DP_SIZE_FLAG,
    _DP_ATTENTION_KEY: _DP_ATTENTION_FLAG,
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
    _DP_ATTENTION_KEY,
    "trust_remote_code",
    "enable_torch_compile",
    "disable_radix_cache",
    "disable_prefill_cuda_graph",
}


class _SglangTopology(NamedTuple):
    """How a launch's ranks group into SGLang distributed worlds.

    SGLang spells data parallelism two incompatible ways, and which one a
    launch is using decides both the flags and the orchestration:

    * ``--dp-size N`` alone — ``N`` replicas *inside one launch*, ``tp * pp *
      N`` GPUs total (sparkrun's ``world_size`` formula).  Upstream refuses
      it across nodes: ``assert not (dp_size > 1 and nnodes != 1 and not
      enable_dp_attention)``.
    * ``--dp-size N --enable-dp-attention`` — ``N`` is a *partition of the tp
      world* (``dp == tp``), so the job still costs ``tp * pp`` GPUs, not
      ``tp * pp * N``.  This is the only multi-node spelling upstream allows,
      and it is why :meth:`SglangRuntime.world_size` stops multiplying by dp.

    Anything else with ``dp > 1`` is the separate-launch/router shape: each
    replica is an independent server, joined by a router rather than by a
    rendezvous.  Injecting ``--nnodes``/``--node-rank`` there trips
    ``assert (tp_size * pp_size) % nnodes == 0`` before the server binds a
    port (issue #284).
    """

    replica_size: int
    """``tp * pp`` — ranks in one distributed world."""

    dp: int
    dp_attention: bool

    @property
    def independent_replicas(self) -> bool:
        """True when this launch is N standalone servers, not one world."""
        return self.dp > 1 and not self.dp_attention


class SglangRuntime(RuntimePlugin):
    """Native SGLang runtime using prebuilt container images.

    SGLang uses its own distributed init mechanism for multi-node inference,
    not Ray.  Each node runs the full ``sglang serve`` command with
    ``--dist-init-addr``, ``--nnodes``, and ``--node-rank`` arguments — except
    under pure data parallelism, where there is no shared world to join at
    all.  See :class:`_SglangTopology`.
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

    # --- Parallelism topology ---

    @staticmethod
    def _dp_attention_enabled(recipe: "Recipe | None", config=None) -> bool:
        """Whether this workload runs SGLang's DP-attention mode.

        Checked in three places, and the third is load-bearing rather than a
        nicety: a recipe is free to hardcode ``--enable-dp-attention`` in its
        ``command:`` template, where the config chain cannot see it.  Missing
        it there would classify a working DeepSeek/Qwen-MoE recipe as
        "independent replicas", strip its rendezvous flags, and break a launch
        that works today.
        """
        value = None
        if config is not None:
            value = config.get(_DP_ATTENTION_KEY)
        if value is None and recipe is not None:
            # noinspection PyProtectedMember
            value = recipe._effective_default(_DP_ATTENTION_KEY)
        if value is not None:
            return ext_parse_bool(value)
        return _DP_ATTENTION_FLAG in ((recipe.command or "") if recipe is not None else "")

    @classmethod
    def _resolve_topology(cls, recipe: "Recipe | None", config) -> _SglangTopology:
        """Resolve the launch's :class:`_SglangTopology` from the config chain."""
        from sparkrun.core.parallelism import extract_parallelism

        p = extract_parallelism(config)
        return _SglangTopology(
            replica_size=max(1, p.tensor_parallel * p.pipeline_parallel),
            dp=max(1, p.data_parallel),
            dp_attention=cls._dp_attention_enabled(recipe, config),
        )

    def _append_dp_size(
        self,
        command: str,
        topo: _SglangTopology,
        num_nodes: int,
        skip_keys: set[str] | frozenset[str],
    ) -> str:
        """Append ``--dp-size`` when this launch unit actually owns the replicas.

        Emitted from here rather than from the flag map because legality is a
        property of the *launch*, not of the recipe: upstream refuses
        ``dp_size > 1`` with ``nnodes > 1`` unless DP attention is on, and on a
        1-GPU-per-host cluster a multi-node ``--dp-size 2`` would ask each host
        to run two replicas on its single GPU.
        """
        if topo.dp <= 1 or "data_parallel" in skip_keys:
            return command
        if num_nodes > 1 and not topo.dp_attention:
            return command
        return self.reconcile_flag_in_command(command, _DP_SIZE_FLAG, topo.dp)

    def world_size(
        self,
        parallelism: "ParallelismConfig",
        *,
        recipe: "Recipe",
        cluster: "ClusterDefinition",
    ) -> int:
        """``tp * pp * dp``, except under DP attention where dp is not a multiplier.

        With ``--enable-dp-attention`` the dp dimension partitions the tensor
        world rather than replicating it (upstream: "the dp size should be
        equal to the tp size"), so the default formula would size a 2-node,
        ``tp 16 / dp 16`` DeepSeek layout at 256 ranks.
        """
        if self._dp_attention_enabled(recipe):
            return max(1, parallelism.model_shard_factor)
        return super().world_size(parallelism, recipe=recipe, cluster=cluster)

    def managed_rendezvous_flags(self) -> tuple[str, ...]:
        """The three flags :meth:`generate_node_command` appends for the world.

        ``--dp-size`` is deliberately absent: whether it is legal is a property
        of the *launch* rather than of the recipe (see ``_append_dp_size``), and
        a recipe writing it in ``command:`` is the documented DP-attention
        spelling that ``_dp_attention_enabled`` reads back.
        """
        return ("--dist-init-addr", "--nnodes", "--node-rank")

    def model_revision_flags(self) -> tuple[str, ...]:
        """SGLang spells the model repo pin ``--revision``, as vLLM does.

        Same spelling, separate declaration: the overlap is a coincidence of
        two engines borrowing HuggingFace's vocabulary, not a shared base.
        """
        return ("--revision",)

    def native_rendezvous_port(
        self,
        recipe: "Recipe | None",
        overrides: dict[str, Any] | None = None,
        *,
        num_nodes: int = 1,
        init_port: int = 25000,
    ) -> int | None:
        """``None`` under pure data parallelism — there is no rendezvous to gate on.

        Each replica is a standalone server that binds only its serve port, so
        nothing ever listens on *init_port* and the shared launch path would
        wait out its whole budget before declaring the head dead.
        """
        if recipe is None:
            return init_port
        topo = self._resolve_topology(recipe, recipe.build_config_chain(overrides or {}))
        if topo.independent_replicas and topo.replica_size <= 1:
            return None
        return init_port

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
        topo = self._resolve_topology(recipe, config)

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
            return self._append_dp_size(rendered, topo, num_nodes, skip_keys)

        return self._build_command(recipe, config, is_cluster, num_nodes, head_ip, topo, skip_keys=skip_keys)

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

        Produces the full ``sglang serve`` invocation with the node-specific
        ``--dist-init-addr``, ``--nnodes`` and ``--node-rank`` flags appended —
        scoped to the distributed world *this* node belongs to, which is not
        always the whole cluster.  See :class:`_SglangTopology`:

        * ``dp == 1`` (or DP attention): one world spanning every node.
        * ``dp > 1, tp * pp > 1``: one world *per replica*; ``--nnodes`` is the
          replica's size, ``--node-rank`` is the rank within it, and the
          rendezvous address is that replica's own head.
        * ``dp > 1, tp * pp == 1``: no world at all — a standalone replica,
          joined to its peers by a router rather than by torch.distributed.
        """
        config = recipe.build_config_chain(overrides)
        self._normalize_config(config)
        topo = self._resolve_topology(recipe, config)

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

        base = self._append_dp_size(base, topo, num_nodes, skip_keys)

        # Ranks in one world, and this node's rank within it.  For a single
        # global world (dp == 1, or DP attention) that is the whole cluster;
        # for independent replicas it is the replica.
        if topo.independent_replicas:
            world_nodes = topo.replica_size
            rank_in_world = node_rank % world_nodes
        else:
            world_nodes = num_nodes
            rank_in_world = node_rank

        # A one-node world has nothing to rendezvous with, and saying otherwise
        # is fatal: sglang asserts ``(tp_size * pp_size) % nnodes == 0`` before
        # it binds a port, so --nnodes 2 on a tp=pp=1 replica aborts the server
        # rather than degrading (issue #284).
        if world_nodes <= 1:
            return base

        # The *global* node_rank goes in (paired with replica_size=world_nodes)
        # because that is what selects the world: _resolve_master_addr
        # floor-divides the two, so a single global world resolves to hosts[0]
        # for every rank while a per-replica world resolves to the first host of
        # *that* replica.  Passing rank_in_world here would collapse every
        # replica onto hosts[0]; passing the default replica_size=1 would map
        # node_rank -> hosts[node_rank] (each node's own IP), leaving only rank 0
        # bound to the store — the "1/N clients joined" rendezvous timeout.
        # The emitted --node-rank is the intra-world one, as vLLM-distributed
        # does with intra_replica_rank.
        node_args = self._make_node_command_args(
            head_ip=head_ip,
            num_nodes=world_nodes,
            node_rank=node_rank,
            init_port=init_port,
            hosts=hosts,
            placement=placement,
            replica_size=world_nodes,
        )

        # Append sglang multi-node arguments.  SGLang combines master_addr
        # and master_port into a single --dist-init-addr HOST:PORT flag.
        parts = [
            base,
            "--dist-init-addr %s:%s" % (node_args["master_addr"], node_args["master_port"]),
            "--nnodes %s" % node_args["num_nodes"],
            "--node-rank %d" % rank_in_world,
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

        # ``data_parallel`` is emitted by _append_dp_size, not from the map:
        # whether --dp-size is legal depends on the launch topology.
        skip = {"tensor_parallel", "data_parallel"}
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
        topo: _SglangTopology | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build the sglang serve command from structured config.

        For cluster mode, includes ``--dist-init-addr`` and ``--nnodes`` but
        NOT ``--node-rank`` (that is added per-node by the orchestrator or
        by :meth:`generate_node_command`).  ``--nnodes`` counts the nodes in
        *one* distributed world, which under pure data parallelism is one node
        and therefore no rendezvous at all — see :meth:`generate_node_command`.
        """
        base = self._build_base_command(recipe, config, skip_keys=skip_keys)
        if topo is None:
            topo = self._resolve_topology(recipe, config)
        base = self._append_dp_size(base, topo, num_nodes, skip_keys)

        world_nodes = topo.replica_size if topo.independent_replicas else num_nodes
        if is_cluster and head_ip and world_nodes > 1:
            base += " --dist-init-addr %s:25000 --nnodes %d" % (head_ip, world_nodes)

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
        issues.extend(self._validate_parallelism(recipe))

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

    def _validate_parallelism(self, recipe: Recipe) -> list:
        """Check the two SGLang parallelism constraints sparkrun can see up front.

        Both are upstream assertions that fire *after* the image and weights
        have been distributed, which is an expensive place to learn about a
        typo in ``defaults:``.
        """
        from sparkrun.core.parallelism import extract_parallelism

        issues = []
        config = recipe.build_config_chain()
        p = extract_parallelism(config)
        replica_size = max(1, p.tensor_parallel * p.pipeline_parallel)

        if self._dp_attention_enabled(recipe):
            if p.data_parallel != p.tensor_parallel:
                issues.append(
                    self.recipe_error(
                        "enable_dp_attention requires data_parallel == tensor_parallel "
                        "(got data_parallel=%d, tensor_parallel=%d). SGLang's DP attention "
                        "partitions the tensor world rather than replicating it, so the two "
                        "sizes must match." % (p.data_parallel, p.tensor_parallel)
                    )
                )
        elif p.data_parallel > 1:
            issues.append(
                self.recipe_warning(
                    "data_parallel=%d without enable_dp_attention: SGLang cannot span replicas "
                    "across nodes (--dp-size is refused with --nnodes > 1). Placed on one host "
                    "the %d replicas share a single endpoint; placed across hosts (%d node(s) "
                    "each) they are independent servers, each with its own endpoint on port %s — "
                    "front those with sglang_router or `sparkrun proxy` to get one address."
                    % (
                        p.data_parallel,
                        p.data_parallel,
                        replica_size,
                        config.get("port") or 30000,
                    )
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
