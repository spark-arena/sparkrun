"""Native tokenary runtime for sparkrun.

tokenary is an experimental Rust LLM inference engine
with native multi-node tensor parallelism over NCCL. One process per node:
rank 0 is the head (scheduler + OpenAI HTTP API), ranks > 0 are SPMD workers that
join the NCCL world and never bind HTTP. See the engine's docs/MULTINODE.md.

Each node runs ``tokenary`` with its own rank/world-size/master flags:
  rank 0:   tokenary --m {model} --server --port {p} --rank 0 --world-size {N} \
                     --master-addr {head} --master-port {mp} --tp-size {N}
  rank R>0: tokenary --m {model} --rank R --world-size {N} \
                     --master-addr {head} --master-port {mp} --tp-size {N}
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes._util import default_env_hf_offline
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv

logger = logging.getLogger(__name__)

# tokenary CLI flag mapping (recipe config key -> tokenary flag). `port` and
# `tensor_parallel` are emitted per-node (rank-0-only / == world_size), so they
# are NOT in this map.
_TOKENARY_FLAG_MAP = {
    "host": "--host",
    "max_model_len": "--max-model-len",
    "max_num_seqs": "--max-num-seqs",
    "max_tokens": "--max-tokens",
    "dtype": "--dtype",
    "kv_cache_dtype": "--kvcache-dtype",
    "gpu_memory_utilization": "--kv-fraction",
    "prefill_chunk_size": "--prefill-chunk-size",
    # Boolean toggles must also appear here: build_flags_from_map iterates the
    # flag_map to find each flag string, then emits flag-only for bool_keys.
    "disable_prefix_cache": "--disable-prefix-cache",
    "disable_reasoning": "--disable-reasoning",
    "disable_cuda_graph": "--disable-cuda-graph",
}

_TOKENARY_BOOL_FLAGS = {
    "disable_prefix_cache",
    "disable_reasoning",
    "disable_cuda_graph",
}


class TokenaryRuntime(RuntimePlugin):
    """Native tokenary runtime using prebuilt CUDA-13 container images.

    Multi-node TP uses tokenary's own NCCL bootstrap (a one-time TCP rendezvous
    on ``--master-addr:--master-port`` distributes the NCCL Id; per-step inputs
    are NCCL-broadcast). Not Ray, not MPI.
    """

    runtime_name = "tokenary"
    default_image_prefix = "scitrera/tokenary"

    def cluster_strategy(self) -> str:
        return "native"

    # --- command construction ---

    def _build_base_command(
        self,
        recipe: "Recipe",
        config,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """tokenary base command: model id + mapped serving flags.

        Excludes port / tensor_parallel / the distributed flags — those are
        appended per-node by :meth:`generate_node_command` (or by
        :meth:`generate_command` for the solo case).
        """
        parts = ["tokenary", "--model", str(recipe.model)]
        skip = {"port", "tensor_parallel"}
        skip.update(skip_keys)
        parts.extend(
            self.build_flags_from_map(
                config,
                _TOKENARY_FLAG_MAP,
                bool_keys=_TOKENARY_BOOL_FLAGS,
                skip_keys=skip,
            )
        )
        return " ".join(parts)

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Solo (single-node) serve command. Rank 0 serves the API."""
        config = recipe.build_config_chain(overrides)
        rendered = recipe.render_command(config)
        if rendered:
            return rendered
        base = self._build_base_command(recipe, config, skip_keys=skip_keys)
        tp = config.get("tensor_parallel") or 1
        port = config.get("port") or 8000
        return "%s --server --port %s --tp-size %s" % (base, port, tp)

    def generate_node_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        head_ip: str,
        num_nodes: int,
        node_rank: int,
        init_port: int = 27101,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        hosts: list[str] | None = None,
        placement=None,
    ) -> str:
        """Per-node serve command for native multi-node TP (1 GPU per node).

        Pure tensor parallelism: ``tp_size == world_size == num_nodes``.
        Rank 0 additionally binds the HTTP API.
        """
        config = recipe.build_config_chain(overrides)
        rendered = recipe.render_command(config)
        base = rendered if rendered else self._build_base_command(recipe, config, skip_keys=skip_keys)

        node_args = self._make_node_command_args(
            head_ip=head_ip,
            num_nodes=num_nodes,
            node_rank=node_rank,
            init_port=init_port,
            hosts=hosts,
            placement=placement,
            replica_size=num_nodes,
        )

        parts = [
            base,
            "--rank %s" % node_args["node_rank"],
            "--world-size %s" % node_args["num_nodes"],
            "--master-addr %s" % node_args["master_addr"],
            "--master-port %s" % node_args["master_port"],
            "--tp-size %s" % node_args["num_nodes"],
        ]
        if int(node_args["node_rank"]) == 0:
            # Head node serves the OpenAI-compatible API; workers bind no HTTP.
            port = config.get("port") or 8000
            parts.append("--server --port %s" % port)
        return " ".join(parts)

    # --- environment ---

    def get_common_env(self):
        return default_env_hf_offline()

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """NCCL cluster env. tokenary reads NCCL tuning straight from the env."""
        return {
            **RuntimePlugin.get_cluster_env(self, head_ip, num_nodes),
            "NCCL_CUMEM_ENABLE": "0",
        }

    # --- cluster lifecycle (delegate to the native orchestrator) ---

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
        init_port: int = 27101,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        **kwargs,
    ) -> int:
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
            banner_title="tokenary Cluster Launcher",
            port_label="Master Port",
            node_label="tokenary node",
            **kwargs,
        )

    def _stop_cluster(
        self,
        hosts: list[str],
        cluster_id: str,
        config=None,
        dry_run: bool = False,
    ) -> int:
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)
