"""Native vLLM distributed runtime for sparkrun.

Uses vLLM's built-in multi-node support (``--nnodes``, ``--node-rank``,
``--master-addr``, ``--master-port``, ``--headless``) instead of Ray.
Follows the same orchestration pattern as SGLang's native distribution.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin
from sparkrun.runtimes._vllm_common import VllmMixin, VLLM_FLAG_MAP, VLLM_BOOL_FLAGS

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.comm_env import ClusterCommEnv

logger = logging.getLogger(__name__)


class VllmDistributedRuntime(VllmMixin, RuntimePlugin):
    """vLLM runtime using native distributed mode (no Ray).

    Each node runs the full ``vllm serve`` command with node-specific
    ``--nnodes``, ``--node-rank``, ``--master-addr``, and ``--master-port``
    arguments.  Worker nodes additionally receive ``--headless``.
    """

    runtime_name = "vllm-distributed"
    default_image_prefix = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5"

    def get_family(self) -> str:
        return "vllm"

    def cluster_strategy(self) -> str:
        """vLLM distributed uses native multi-node distribution, not Ray."""
        return "native"

    def generate_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the vllm serve command.

        For cluster mode this produces the *base* command without
        ``--node-rank``.  Use :meth:`generate_node_command` to get the
        per-node variant.
        """
        config = recipe.build_config_chain(overrides)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--served-model-name",
                skip_keys,
            )
            # Augment OTel flags if present in config but missing from rendered command
            for key in ["otlp_traces_endpoint", "collect_detailed_traces"]:
                if key in skip_keys:
                    continue
                val = config.get(key)
                if val is not None and VLLM_FLAG_MAP[key] not in rendered:
                    rendered += f" {VLLM_FLAG_MAP[key]} {val}"

            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    VLLM_FLAG_MAP,
                    VLLM_BOOL_FLAGS,
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
    ) -> str:
        """Generate the vllm serve command for a specific node.

        Handles the three parallelism regimes on DGX Spark (1 GPU/node):

        * ``tp*pp > 1, dp == 1``: cross-node tensor/pipeline parallel
          within a single replica.  Appends ``--nnodes``, ``--node-rank``,
          ``--master-addr``, ``--master-port``; workers add ``--headless``.
        * ``tp*pp == 1, dp > 1``: pure data-parallel replication.  Each
          node is its own replica; appends ``--data-parallel-size``,
          ``--data-parallel-rank``, ``--data-parallel-address``,
          ``--data-parallel-rpc-port``.  No tp/pp torch-distributed flags.
        * ``tp*pp > 1, dp > 1`` (hybrid): both sets of flags.  The node's
          ``--master-addr`` points at the first host of *its* dp replica,
          and ``--node-rank`` is the rank *within* that replica (0..tp*pp-1),
          not the global node index.
        """
        from sparkrun.core.parallelism import extract_parallelism

        config = recipe.build_config_chain(overrides)
        p = extract_parallelism(config)
        replica_size = p.tensor_parallel * p.pipeline_parallel
        dp = p.data_parallel

        # Rank math — see CLAUDE.md / plan "Rank math" section.
        # When dp == 1 this collapses to node_rank = global rank, tp_master = head_ip.
        if replica_size <= 0:
            replica_size = 1  # defensive: tp or pp misconfigured as 0
        dp_rank = node_rank // replica_size
        intra_replica_rank = node_rank % replica_size
        if hosts and len(hosts) >= (dp_rank + 1) * replica_size:
            tp_master_addr = hosts[dp_rank * replica_size]
        else:
            # Fallback: no host list available (solo / unit tests).
            tp_master_addr = head_ip

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(
                rendered,
                config,
                "--served-model-name",
                skip_keys,
            )
            # Augment OTel flags if present in config but missing from rendered command
            for key in ["otlp_traces_endpoint", "collect_detailed_traces"]:
                if key in skip_keys:
                    continue
                val = config.get(key)
                if val is not None and VLLM_FLAG_MAP[key] not in rendered:
                    rendered += f" {VLLM_FLAG_MAP[key]} {val}"

            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    VLLM_FLAG_MAP,
                    VLLM_BOOL_FLAGS,
                )
            base = rendered
        else:
            base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        parts = [base]

        # Torch-distributed coordination for cross-node tp/pp (intra-replica).
        if replica_size > 1:
            parts.extend(
                [
                    "--nnodes %d" % replica_size,
                    "--node-rank %d" % intra_replica_rank,
                    "--master-addr %s" % tp_master_addr,
                    "--master-port %d" % init_port,
                ]
            )
            if intra_replica_rank > 0:
                parts.append("--headless")

        # vLLM data-parallel coordination (inter-replica).
        if dp > 1:
            dp_rpc_port = int(config.get("data_parallel_rpc_port") or 13345)
            dp_address = hosts[0] if hosts else head_ip
            # Only inject --data-parallel-size when the recipe template
            # didn't already supply it (mirrors how we handle -tp today).
            if "--data-parallel-size" not in (base or ""):
                parts.append("--data-parallel-size %d" % dp)
            parts.extend(
                [
                    "--data-parallel-rank %d" % dp_rank,
                    "--data-parallel-address %s" % dp_address,
                    "--data-parallel-rpc-port %d" % dp_rpc_port,
                ]
            )

        return " ".join(parts)

    def _build_base_command(self, recipe: Recipe, config, skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the vllm serve command without cluster-specific arguments."""
        parts = ["vllm", "serve", recipe.model]

        tp = config.get("tensor_parallel")
        if tp:
            parts.extend(["-tp", str(tp)])

        # Add flags from defaults (skip tp and distributed_executor_backend)
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
        recipe: Recipe,
        config,
        is_cluster: bool,
        num_nodes: int,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build the vllm serve command from structured config.

        For cluster mode, includes ``--nnodes``, ``--master-addr``, and
        ``--master-port`` but NOT ``--node-rank`` (that is added per-node
        by :meth:`generate_node_command`).
        """
        base = self._build_base_command(recipe, config, skip_keys=skip_keys)

        if is_cluster and head_ip:
            base += " --nnodes %d --master-addr %s --master-port 25000" % (num_nodes, head_ip)

        return base

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return vLLM distributed-specific cluster environment variables.

        Sets ``OMP_NUM_THREADS=4`` by default to avoid thread
        over-subscription on multi-node clusters.  Recipe ``env`` can
        override any of these values (runtime defaults are merged first,
        recipe env wins).
        """
        return {
            **RuntimePlugin.get_cluster_env(self, head_ip, num_nodes),
            "NCCL_CUMEM_ENABLE": "0",
            "OMP_NUM_THREADS": "4",
        }

    # --- Cluster stop ---

    def _stop_cluster(
        self,
        hosts: list[str],
        cluster_id: str,
        config=None,
        dry_run: bool = False,
    ) -> int:
        """Stop a vLLM distributed native cluster."""
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
        """Orchestrate a multi-node vLLM cluster using native distribution."""
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
            banner_title="vLLM Distributed Cluster Launcher",
            port_label="Master Port",
            node_label="vllm node",
            **kwargs,
        )
