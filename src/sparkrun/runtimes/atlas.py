"""Atlas runtime for sparkrun.

Atlas (https://github.com/Avarok-Cybersecurity/atlas) is a pure-Rust LLM inference
server. It bootstraps multi-rank deployments via NCCL using
``--rank``/``--world-size``/``--master-addr``/``--master-port`` flags,
without Ray. This runtime therefore uses the ``"native"`` clustering
strategy, identical in shape to the SGLang and vllm-distributed
runtimes.

Atlas composes tensor parallelism (``--tp-size``) and expert parallelism
(``--ep-size``) on the same physical ranks (see
``crates/spark-server/src/cli.rs``):

* ``world_size == tp_size * ep_size``  — orthogonal mesh
* ``world_size == tp_size == ep_size`` — overlapping groups (used by the
  validated MiniMax M2.7 EP=2 + TP=2 config on two GB10 nodes)

The runtime auto-derives ``world_size`` from ``tensor_parallel`` and
``ep_size`` so users only set the two parallelism dims and sparkrun
maps them to physical hosts.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes._util import default_env_hf_offline
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv

logger = logging.getLogger(__name__)

# Standardized sparkrun config keys → Atlas CLI flags.
# Keys here follow the conventions documented in RECIPES.md "Common
# Defaults Keys" so that recipes are portable across runtimes where
# possible.  Boolean toggles are listed in `_ATLAS_BOOL_FLAGS` and emit
# only the flag (no value).
_ATLAS_FLAG_MAP = {
    # Sparkrun standard keys
    "port": "--port",
    "host": "--host",  # Atlas exposes --bind with `--host` as an alias
    "tensor_parallel": "--tp-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-seq-len",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-prefill-tokens",
    "served_model_name": "--model-name",
    "kv_cache_dtype": "--kv-cache-dtype",
    "ep_size": "--ep-size",  # TODO: change to expert_parallel after alias config [FUTURE]
    # Atlas-specific keys (passed through as-is)
    "max_batch_size": "--max-batch-size",
    "block_size": "--block-size",
    "kv_high_precision_layers": "--kv-high-precision-layers",
    "tool_call_parser": "--tool-call-parser",
    "tool_max_tokens": "--tool-max-tokens",
    "scheduling_policy": "--scheduling-policy",
    "tbt_deadline_ms": "--tbt-deadline-ms",
    "max_prefill_tokens": "--max-prefill-tokens",
    "oom_guard_mb": "--oom-guard-mb",
    "ssm_cache_slots": "--ssm-cache-slots",
    "ssm_checkpoint_interval": "--ssm-checkpoint-interval",
    "mtp_quantization": "--mtp-quantization",
    "mtp_vocab": "--mtp-vocab",
    "num_drafts": "--num-drafts",
    "draft_model": "--draft-model",
    "dflash_gamma": "--dflash-gamma",
    "dflash_window_size": "--dflash-window-size",
    "max_thinking_budget": "--max-thinking-budget",
    "model_from_path": "--model-from-path",
    "cache_dir": "--cache-dir",
    "gpu_ordinal": "--gpu-ordinal",
    # Boolean toggles — present when truthy, omitted otherwise.  Listed
    # in `_ATLAS_BOOL_FLAGS` so build/strip helpers treat them correctly.
    "enable_prefix_caching": "--enable-prefix-caching",
    "speculative": "--speculative",
    "self_speculative": "--self-speculative",
    "ngram_speculative": "--ngram-speculative",
    "dflash": "--dflash",
    "disable_thinking": "--disable-thinking",
    "high_speed_swap": "--high-speed-swap",
    "require_auth": "--require-auth",
}

_ATLAS_BOOL_FLAGS = frozenset(
    {
        "enable_prefix_caching",
        "speculative",
        "self_speculative",
        "ngram_speculative",
        "dflash",
        "disable_thinking",
        "high_speed_swap",
        "require_auth",
    }
)


class AtlasRuntime(RuntimePlugin):
    """Native Atlas Spark runtime using the public ``avarok/atlas-gb10`` image.

    Atlas handles its own multi-rank bootstrap via NCCL — each node runs
    the full ``atlas serve`` command with node-specific ``--rank``,
    ``--world-size``, ``--master-addr``, and ``--master-port`` flags.
    The non-head ranks bind their HTTP listener to ``--port 0`` so only
    rank 0 exposes the OpenAI API.
    """

    runtime_name = "atlas"
    default_image_prefix = "avarok/atlas-gb10"

    def cluster_strategy(self) -> str:
        """Atlas handles its own multi-rank distribution via NCCL, not Ray."""
        return "native"

    def prefer_ib_for_init_addr(self) -> bool:
        """Added to give option to usb IB IP for Atlas instead of MGMT IP during troubleshooting"""
        return False

    def get_common_env(self):
        return default_env_hf_offline()

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
        # noinspection PyProtectedMember
        draft_model = recipe._effective_default("draft_model")
        if draft_model:
            recipe.distribution_config.add_model(str(draft_model))

    # --- Command generation ---

    def generate_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the ``atlas serve`` command for solo or cluster mode.

        For cluster mode this is the head-node (rank 0) command. Worker
        ranks are produced by :meth:`generate_node_command`.
        """
        config = recipe.build_config_chain(overrides)

        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--model-name",
                skip_keys,
            )
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    _ATLAS_FLAG_MAP,
                    _ATLAS_BOOL_FLAGS,
                )
            return rendered

        return self._build_command(recipe, config, is_cluster, num_nodes, head_ip, node_rank=0, skip_keys=skip_keys)

    def generate_node_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        head_ip: str,
        num_nodes: int,
        node_rank: int,
        init_port: int = 29500,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        hosts: list[str] | None = None,
    ) -> str:
        """Generate the per-rank ``atlas serve`` command.

        Non-head ranks override ``--port 0`` so they don't try to bind
        the HTTP listener — only rank 0 exposes the OpenAI API.
        """
        config = recipe.build_config_chain(overrides)

        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--model-name",
                skip_keys,
            )
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    _ATLAS_FLAG_MAP,
                    _ATLAS_BOOL_FLAGS,
                )
            base = rendered
        else:
            base = self._build_base_command(recipe, config, node_rank=node_rank, skip_keys=skip_keys)

        parts = [
            base,
            "--rank %d" % node_rank,
            "--world-size %d" % num_nodes,
            "--master-addr %s" % head_ip,
            "--master-port %d" % init_port,
        ]
        return " ".join(parts)

    # --- Internal helpers ---

    def _build_base_command(
        self,
        recipe: Recipe,
        config,
        node_rank: int = 0,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build the ``atlas serve`` command without cluster-coordination flags.

        Worker ranks (node_rank > 0) override ``--port 0`` to suppress HTTP
        binding; only the head exposes the OpenAI API.
        """
        parts = ["spark", "serve", recipe.model]

        skip = set(skip_keys)
        # Worker ranks always bind --port 0 regardless of the recipe value.
        if node_rank > 0:
            skip.add("port")
            parts.extend(["--port", "0"])

        parts.extend(
            self.build_flags_from_map(
                config,
                _ATLAS_FLAG_MAP,
                bool_keys=_ATLAS_BOOL_FLAGS,
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
        node_rank: int = 0,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build the head-rank command, including coordination flags in cluster mode."""
        base = self._build_base_command(recipe, config, node_rank=node_rank, skip_keys=skip_keys)

        init_port = int(config.get("init_port", 29500))
        if is_cluster and head_ip:
            base += " --rank 0 --world-size %d --master-addr %s --master-port %d" % (num_nodes, head_ip, init_port)

        return base

    # --- Parallelism ---

    def compute_required_nodes(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> int | None:
        """Atlas world_size = max(tp*ep, tp, ep).

        Atlas supports overlapping ``tp == ep`` topology (both groups
        live on the same physical ranks) as well as the orthogonal
        ``tp * ep`` mesh. Sparkrun's stock ``compute_required_nodes``
        only considers ``tensor_parallel * pipeline_parallel *
        data_parallel``, so we override to factor in ``ep_size``.
        """
        config = recipe.build_config_chain(overrides or {})
        tp = _coerce_int(config.get("tensor_parallel"), default=1)
        ep = _coerce_int(config.get("ep_size"), default=1)

        if tp <= 1 and ep <= 1:
            return None

        if tp == ep and tp > 1:
            # Overlapping groups: both TP and EP share the same ranks.
            result = tp
        else:
            result = tp * ep

        if result > 1:
            raise ValueError("Atlas runtime currently only supports single node. Support for multiple nodes is coming soon.")

        return result

    # --- Cluster env ---

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """NCCL settings validated for GB10 RoCEv2 fabrics.

        Mirrors ``scripts/start-minimax-ep2.sh`` from the Atlas repo —
        the same env that's in production use for the MiniMax M2.7 EP=2
        and Qwen3.5-122B EP=2 deployments. Recipe ``env`` overrides any
        of these values.
        """
        return {
            **RuntimePlugin.get_cluster_env(self, head_ip, num_nodes),
            # Interface/HCA should always come from cluster config -- IF THESE ARE NEEDED, then we need to do deeper work on NCCL
            # "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
            # "NCCL_IB_HCA": "rocep1s0f0",
            "NCCL_IB_GID_INDEX": "",  # clear
            "NCCL_CROSS_NIC": "",  # clear
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_ROCE_VERSION_NUM": "2",
            "NCCL_IB_ADDR_FAMILY": "AF_INET",
            "NCCL_IB_TIMEOUT": "22",
            "NCCL_IB_RETRY_CNT": "7",
            "NCCL_NET_GDR_LEVEL": "0",
            "NCCL_NET_GDR_C2C": "0",
            "NCCL_DMABUF_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_CUMEM_HOST_ENABLE": "0",
            "NCCL_PROTO": "Simple",
            "NCCL_ALGO": "Ring",
            "NCCL_BUFFSIZE": "33554432",
            "NCCL_MIN_NCHANNELS": "1",
            "NCCL_MAX_NCHANNELS": "2",
        }

    def get_extra_docker_opts(self) -> list[str]:
        """RDMA device + capabilities required by Atlas's NCCL/io_uring paths.

        ``--high-speed-swap`` uses ``IORING_SETUP_SQPOLL`` (kernel ≥ 5.13)
        and Docker's default seccomp profile blocks ``io_uring_*``, so we
        run the storage path unconfined. ``IPC_LOCK`` + ``memlock=-1``
        unblock ``ibv_reg_mr``; ``SYS_NICE`` is needed by the SQPOLL
        kernel thread.
        """
        # TODO: this should transition to be executor_config pass-through [FUTURE; works well enough today]
        return [
            # Allow bash
            "--entrypoint",
            '""',
            "--cap-add=IPC_LOCK",
            "--cap-add=SYS_NICE",
            "--security-opt",
            "seccomp=unconfined",
        ]

    # --- Version reporting ---

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        # TODO: atlas does not have versions; adjust later
        cmds["atlas"] = "spark --version 2>/dev/null || echo unknown"
        return cmds

    # --- Cluster lifecycle ---

    def _stop_cluster(
        self,
        hosts: list[str],
        cluster_id: str,
        config=None,
        dry_run: bool = False,
    ) -> int:
        """Stop an Atlas native cluster."""
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)

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
        ib_ip_map: dict[str, str] | None = None,
        init_port: int = 29500,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        **kwargs,
    ) -> int:
        """Orchestrate a multi-rank Atlas cluster using native NCCL distribution."""
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
            ib_ip_map=ib_ip_map,
            init_port=init_port,
            skip_keys=skip_keys,
            banner_title="Atlas Cluster Launcher",
            port_label="NCCL Master Port",
            node_label="atlas rank",
            **kwargs,
        )


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
