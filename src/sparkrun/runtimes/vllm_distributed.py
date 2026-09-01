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
    from sparkrun.core.config import SparkrunConfig
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

    # See SglangRuntime: native distribution, so per-machine tuned images work.
    # vllm-ray is a *sibling* (both are VllmMixin + RuntimePlugin), not a
    # subclass, so it does not pick this up — which is what we want: Ray needs
    # one build across head and workers.
    supports_heterogeneous_images = True

    def get_family(self) -> str:
        return "vllm"

    def cluster_strategy(self) -> str:
        """vLLM distributed uses native multi-node distribution, not Ray."""
        return "native"

    # TODO: pure DP (``tp*pp == 1, dp > 1``) emits ``--data-parallel-*`` and no
    # ``--master-port``, so nothing here looks like it binds ``init_port`` — if
    # so the head gate in ``_cluster_ops`` waits out its budget and reports a
    # healthy launch as dead, and the fix is a ``native_rendezvous_port``
    # override returning ``None`` for that regime (see SglangRuntime, #284).
    # Needs a live 2-node ``--dp 2`` run to confirm before changing anything.

    def prepare(
        self,
        recipe: Recipe,
        hosts: list[str],
        config: "SparkrunConfig | None" = None,
        dry_run: bool = False,
        transfer_mode: str = "auto",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Detect and add a draft model to distribution config if needed"""
        draft_model, draft_revision = self.detect_spec_config_draft(recipe)
        if draft_model:
            recipe.distribution_config.add_model(draft_model, revision=draft_revision)

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
            # A defaults/-o value for the backend wins over a literal in the
            # command template (e.g. -o distributed_executor_backend=mp over a
            # legacy command that hardcodes --distributed-executor-backend ray).
            rendered = self._apply_distributed_backend(rendered, config, skip_keys)
            rendered = self._augment_vllm_served_model_name(
                rendered,
                recipe,
                config,
                skip_keys,
            )
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
        placement=None,
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
        if replica_size <= 0:
            replica_size = 1  # defensive: tp or pp misconfigured as 0

        # Rank math — see CLAUDE.md / plan "Rank math" section.
        # When dp == 1 this collapses to node_rank = global rank, tp_master = head_ip.
        dp_rank = node_rank // replica_size
        intra_replica_rank = node_rank % replica_size

        # Canonical distributed-init args.  We pass the *global* node_rank
        # + replica_size so _resolve_master_addr can pick the correct
        # replica head; the emitted --nnodes/--node-rank values below use
        # the intra-replica numbers, not the global ones.
        node_args = self._make_node_command_args(
            head_ip=head_ip,
            num_nodes=replica_size,
            node_rank=node_rank,
            init_port=init_port,
            hosts=hosts,
            placement=placement,
            replica_size=replica_size,
        )

        return self._generate_parallel_command(
            recipe=recipe,
            config=config,
            skip_keys=skip_keys,
            replica_size=replica_size,
            replica_node_count=int(node_args["num_nodes"]),
            replica_node_rank=intra_replica_rank,
            master_addr=node_args["master_addr"],
            master_port=int(node_args["master_port"]),
            dp=dp,
            dp_rank=dp_rank,
            dp_address=hosts[0] if hosts else head_ip,
        )

    def generate_launch_unit_command(
        self,
        recipe: Recipe,
        overrides: dict[str, Any],
        head_ip: str,
        init_port: int,
        worker_ranks: tuple[int, ...],
        service_index: int,
        service_unit_index: int,
        service_unit_count: int,
        service_hosts: list[str],
    ) -> str:
        """Generate one vLLM process tree for several local GPU workers.

        Scheduler ranks identify accelerator *workers*. vLLM's ``--node-rank``
        identifies *process trees*, and the two numbers diverge as soon as a
        host owns more than one GPU — which ``generate_node_command`` cannot
        express, since it takes a single rank and infers everything else.
        Materialization passes both namespaces explicitly instead.
        """
        from sparkrun.core.parallelism import extract_parallelism

        if not worker_ranks or service_unit_count <= 0 or not 0 <= service_unit_index < service_unit_count:
            raise ValueError("invalid vLLM launch-unit topology")
        config = recipe.build_config_chain(overrides)
        p = extract_parallelism(config)
        replica_size = p.tensor_parallel * p.pipeline_parallel
        if replica_size <= 0:
            raise ValueError("vLLM tensor/pipeline replica size must be positive")
        expected_service = {rank // replica_size for rank in worker_ranks}
        if expected_service != {service_index}:
            raise ValueError("a vLLM launch unit cannot cross data-parallel replicas")
        if len(service_hosts) != service_unit_count:
            raise ValueError("vLLM service host inventory is incomplete")
        return self._generate_parallel_command(
            recipe=recipe,
            config=config,
            skip_keys=frozenset(),
            replica_size=replica_size,
            replica_node_count=service_unit_count,
            replica_node_rank=service_unit_index,
            master_addr=service_hosts[0],
            master_port=init_port,
            dp=p.data_parallel,
            dp_rank=service_index,
            dp_address=head_ip,
        )

    def _generate_parallel_command(
        self,
        *,
        recipe: Recipe,
        config,
        skip_keys: set[str] | frozenset[str],
        replica_size: int,
        replica_node_count: int,
        replica_node_rank: int,
        master_addr: str,
        master_port: int,
        dp: int,
        dp_rank: int,
        dp_address: str,
    ) -> str:
        """Render vLLM flags from explicit worker and launch-unit topology."""

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            # A defaults/-o value for the backend wins over a literal in the
            # command template (e.g. -o distributed_executor_backend=mp over a
            # legacy command that hardcodes --distributed-executor-backend ray).
            rendered = self._apply_distributed_backend(rendered, config, skip_keys)
            rendered = self._augment_vllm_served_model_name(
                rendered,
                recipe,
                config,
                skip_keys,
            )
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
                    "--nnodes %d" % replica_node_count,
                    "--node-rank %d" % replica_node_rank,
                    "--master-addr %s" % master_addr,
                    "--master-port %d" % master_port,
                ]
            )
            if replica_node_rank > 0:
                parts.append("--headless")

        # vLLM data-parallel coordination (inter-replica).
        if dp > 1:
            dp_rpc_port = int(config.get("data_parallel_rpc_port") or 13345)
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
