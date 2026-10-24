"""TensorFold DeepSeek-V4.1 (TP2/TP4) runtime for sparkrun.

This is the community-fork lane, not the upstream TensorFold release.  As of
TensorFold 1.1.3 the released native CUDA binary does not serve DeepSeek-V4.1
(upstream PR #827 is still open), so this lane pins the community build from
``bertholomus/TensorFold`` (branch ``deepseek-v41-zig``, commit ``3ff3b2a``)
plus the three small patches used by the
``tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks`` deployment.

What makes it different from the upstream ``tensorfold`` runtime:

* The follower ranks run a **separate binary** (``tf-dsv41-lanes``) with
  positional arguments, while rank 0 runs ``tensorfold-native serve`` with
  ``TF_TP_*`` environment variables.  There is no shared serve flag vocabulary
  between the two shapes.
* The engine consumes **per-rank caches** and a kernel kit, not the raw
  checkpoint: a bundle directory is prepared once (CPU only) from the EXL3
  pack and mounted read-only on every node.  See
  ``scripts/tensorfold-dsv41-prepare.sh`` and ``docs/TENSORFOLD.md``.
* The weights are big: a TP2 pair needs ~99 GiB per node plus the rank cache,
  and the Engram tables must be local.  The bundle layout is documented in
  ``docs/TENSORFOLD.md``; sparkrun deliberately does not download or build it.

Bundle layout (absolute path in the recipe's ``model:``, identical on every
node; a shared NFS mount is the natural home):

.. code-block:: text

    <bundle>/model/                 # config/tokenizer/index/encoding/vision shards/template
    <bundle>/cache/rank-cache/      # rank<N>of<M>-<hash>.bin
    <bundle>/cache/dsv41_token_map.json
    <bundle>/kit/                   # kernel kit (aot/, MANIFEST, ...)
    <bundle>/engram/                # shards 47/48 of the base checkpoint (local per node)

The engine binaries live in the image at ``/opt/tensorfold-dsv41/bin``.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.core.validation import RecipeIssue
from sparkrun.runtimes.base import RuntimePlugin, render_flag_value
from sparkrun.runtimes.tensorfold import TensorFoldRuntime
from sparkrun.utils.data import integer_setting
from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

#: Supported plane counts for the community TP lane.
_DSV41_WORLDS = frozenset({2, 4})

#: Default engine locations inside the image (see docker/tensorfold-dsv41).
DEFAULT_COMMAND_BINARY = "/opt/tensorfold-dsv41/bin/tensorfold-native"
DEFAULT_FOLLOWER_BINARY = "/opt/tensorfold-dsv41/bin/tf-dsv41-lanes"

#: Recipe config keys understood in addition to the upstream TensorFold map.
_DSV41_OWNED_KEYS = frozenset(
    {
        "command_binary",
        "follower_binary",
        "streams",
        "pool",
        "round_rows",
        "window",
        "tf_2d",
        "rdma_hca",
        "net_if",
        "rdma_gid_index",
        "nccl_ib_gid_index",
        "cuda_kernels",
        "tf_env",
    }
)

#: Config keys that are emitted as inline environment assignments, with one
#: env-var name each.  Values are quoted when substituted.
_DSV41_ENV_KEYS: dict[str, str] = {
    "rdma_hca": "TF_RDMA_DEVICES",
    "net_if": "NCCL_SOCKET_IFNAME",
    "rdma_gid_index": "TF_RDMA_GID_INDEX",
    "nccl_ib_gid_index": "NCCL_IB_GID_INDEX",
}


class TensorFoldDSV41Runtime(TensorFoldRuntime):
    """TensorFold DeepSeek-V4.1 community build, TP2/TP4 over a prepared bundle."""

    runtime_name = "tensorfold-dsv41"
    default_image_prefix = "ghcr.io/spark-arena/tensorfold-dsv41"

    def known_config_keys(self) -> frozenset[str]:
        return super().known_config_keys() | _DSV41_OWNED_KEYS

    # --- placement ----------------------------------------------------------

    @staticmethod
    def _world(recipe: "Recipe", overrides: dict[str, Any] | None = None) -> int:
        chain = recipe.build_config_chain(overrides or {})
        value = chain.get("tensor_parallel")
        if value is None:
            return 1
        try:
            return max(1, integer_setting(value, key="tensor_parallel"))
        except (TypeError, ValueError):
            return 1

    def world_size(
        self,
        parallelism: "ParallelismConfig",
        *,
        recipe: "Recipe",
        cluster: "ClusterDefinition",
    ) -> int:
        if parallelism.tensor_parallel in _DSV41_WORLDS:
            return parallelism.tensor_parallel
        return 1

    def native_start_order(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> str:
        if recipe is None:
            return "head_first"
        return "workers_first" if self._world(recipe, overrides) in _DSV41_WORLDS else "head_first"

    def native_rendezvous_port(
        self, recipe: "Recipe | None", overrides: dict[str, Any] | None = None, *, num_nodes: int = 1, init_port: int = 25000
    ) -> int | None:
        return init_port

    def native_rendezvous_port_span(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> int:
        # NCCL_COMM_ID/TP_PORT is used by rank 0; the follower control channel
        # opens the next port.
        return 2

    # --- bundle helpers -----------------------------------------------------

    @staticmethod
    def _bundle(recipe: "Recipe", config) -> str:
        """Absolute prepared-bundle root (the recipe's pre-placed ``model:``)."""
        bundle = config.get("_prepared_model_path") or config.get("resolved_model_path") or recipe.model
        if not bundle:
            raise ValueError("tensorfold-dsv41 needs a prepared bundle path in model:")
        return str(bundle).rstrip("/")

    @staticmethod
    def _bundle_path(bundle: str, config, key: str, default: str) -> str:
        value = config.get(key)
        return str(value) if value not in (None, "") else "%s/%s" % (bundle, default)

    @staticmethod
    def _env_assignment(name: str, value: object) -> str:
        return "%s=%s" % (name, quote(str(value)))

    # --- command construction ----------------------------------------------

    def _common_flags(self, config, skip_keys: set[str] | frozenset[str]) -> list[str]:
        """Serve flags shared by rank 0 (the upstream flag map)."""
        from sparkrun.runtimes.tensorfold import _TENSORFOLD_BOOL_FLAGS, _TENSORFOLD_FLAG_MAP, _TENSORFOLD_NEGATABLE_FLAGS, _SKIP_KEYS

        skip = set(_SKIP_KEYS) | set(skip_keys)
        return self.build_flags_from_map(
            config,
            _TENSORFOLD_FLAG_MAP,
            bool_keys=_TENSORFOLD_BOOL_FLAGS,
            skip_keys=skip,
            negatable_keys=_TENSORFOLD_NEGATABLE_FLAGS,
        )

    def _rank0_command(self, config, bundle: str, world: int) -> str:
        binary = config.get("command_binary")
        binary = DEFAULT_COMMAND_BINARY if binary is None else str(binary)
        model = "%s/model" % bundle
        host = config.get("host") or "0.0.0.0"
        port = config.get("port") or 8000
        parts = ([binary] if binary else []) + ["serve", quote(model)]
        parts.extend(["--host", render_flag_value(host), "--port", render_flag_value(port)])
        served_name = config.get("served_model_name")
        if served_name is not None:
            parts.extend(["--name", render_flag_value(served_name)])
        api_key = config.get("api_key")
        if api_key not in (None, ""):
            parts.extend(["--api-key", render_flag_value(api_key)])
        parts.append("--no-update-check")
        parts.extend(["--backend", "cuda"])
        parts.extend(self._common_flags(config, frozenset()))
        return " ".join(parts)

    def _follower_command(self, config, bundle: str, world: int, rank: int, head_ip: str, init_port: int) -> str:
        binary = config.get("follower_binary")
        binary = DEFAULT_FOLLOWER_BINARY if binary is None else str(binary)
        cache = self._bundle_path(bundle, config, "cache_dir", "cache/rank-cache")
        kit = self._bundle_path(bundle, config, "kit_dir", "kit")
        engram = self._bundle_path(bundle, config, "engram_dir", "engram")
        token_map = self._bundle_path(bundle, config, "token_map", "cache/dsv41_token_map.json")
        streams = config.get("streams") or config.get("max_num_seqs") or 4
        pool = config.get("pool") or config.get("max_model_len") or 1048576
        rdma = config.get("rdma_hca") or ""
        parts = [
            binary,
            quote("%s/model" % bundle),
            quote(cache),
            str(rank),
            str(world),
            quote(head_ip),
            str(init_port),
            quote(kit),
            "--engram",
            quote(engram),
            "--token-map",
            quote(token_map),
            "--pool",
            render_flag_value(pool),
            "--graphs",
            "1",
            "--side",
            "1",
            "--prefetch",
            "1",
            "--aio",
            "1",
            "--streams",
            render_flag_value(streams),
        ]
        if rdma:
            parts.extend(["--rdma", quote(str(rdma))])
        return " ".join(parts)

    def _rank_env(self, config, bundle: str, world: int, rank: int, head_ip: str, init_port: int) -> list[str]:
        cache = self._bundle_path(bundle, config, "cache_dir", "cache/rank-cache")
        kit = self._bundle_path(bundle, config, "kit_dir", "kit")
        engram = self._bundle_path(bundle, config, "engram_dir", "engram")
        token_map = self._bundle_path(bundle, config, "token_map", "cache/dsv41_token_map.json")
        kernels = config.get("cuda_kernels") or "%s/aot" % kit
        env: dict[str, object] = {
            "TF_TP_RANK": rank,
            "TF_TP_WORLD": world,
            "TF_TP_MASTER": head_ip,
            "TF_TP_PORT": init_port,
            "TF_DS_RANK_CACHE": cache,
            "TF_DS_KIT": kit,
            "TF_DS_ENGRAM": engram,
            "TF_DS_TOKEN_MAP": token_map,
            "TENSORFOLD_CUDA_KERNELS": kernels,
            "TF_DS_GRAPHS": 1,
            "TF_DS_HC_SIDE": 1,
            "TF_DS_L2_PREFETCH": 1,
            "TF_DS_ENGRAM_AIO": 1,
            "TF_DS_STREAMS": config.get("streams") or config.get("max_num_seqs") or 4,
        }
        for key, env_name in _DSV41_ENV_KEYS.items():
            value = config.get(key)
            if value not in (None, ""):
                env[env_name] = value
        if world == 4:
            env["TF_DS_2D_GU"] = config.get("tf_2d") or "parity"
            env["TF_DS_WINDOW"] = config.get("window") or 2097152
            env["TF_DS_ROUND_ROWS"] = config.get("round_rows") or 48
        extra = config.get("tf_env")
        if isinstance(extra, dict):
            for key, value in extra.items():
                if isinstance(key, str) and value is not None:
                    env[key] = value
        return [self._env_assignment(name, value) for name, value in env.items()]

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        raise ValueError(
            "tensorfold-dsv41 is a multi-node runtime: use --tp 2 or --tp 4 with a TP2/TP4 bundle "
            "(DeepSeek-V4.1 has no qualified single-rank serving path in this lane)"
        )

    def generate_node_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        head_ip: str,
        num_nodes: int,
        node_rank: int,
        init_port: int = 25000,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        hosts: list[str] | None = None,
        placement=None,
    ) -> str:
        config = recipe.build_config_chain(overrides)
        if num_nodes not in _DSV41_WORLDS:
            raise ValueError("tensorfold-dsv41 supports tp=2 or tp=4 (got %d nodes)" % num_nodes)
        bundle = self._bundle(recipe, config)
        env = self._rank_env(config, bundle, num_nodes, node_rank, head_ip, init_port)
        command = (
            self._rank0_command(config, bundle, num_nodes)
            if node_rank == 0
            else self._follower_command(config, bundle, num_nodes, node_rank, head_ip, init_port)
        )
        return " ".join(env + [command])

    # --- validation ---------------------------------------------------------

    @staticmethod
    def _is_prepared_bundle(recipe: "Recipe") -> bool:
        import os

        bundle = getattr(getattr(recipe, "cluster_config", None), "resolved_model_path", None) or recipe.model
        return isinstance(bundle, str) and os.path.isabs(bundle)

    def validate_recipe(self, recipe: "Recipe") -> list[str | RecipeIssue]:
        # Skip TensorFoldRuntime's upstream-TP checks: this lane is TP2/TP4 by
        # construction and uses a prepared bundle, not a family key.
        issues = RuntimePlugin.validate_recipe(self, recipe)
        config = recipe.build_config_chain({})
        dims = self._parallelism_dimensions(config)
        tp = dims["tensor_parallel"]
        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                issues.append(self.recipe_error("%s is not supported by tensorfold-dsv41" % key))
        if tp not in _DSV41_WORLDS:
            issues.append(
                self.recipe_error(
                    "tensorfold-dsv41 is TP2/TP4 only: set defaults.tensor_parallel to 2 or 4 "
                    "(single-rank DeepSeek-V4.1 is not a qualified TensorFold path)"
                )
            )
        if not self._is_prepared_bundle(recipe):
            issues.append(
                self.recipe_error(
                    "tensorfold-dsv41 needs a prepared bundle directory as an absolute model: path "
                    "(or cluster_config.resolved_model_path); see docs/TENSORFOLD.md"
                )
            )
        if recipe.command:
            issues.append(
                self.recipe_error("a command: template is not supported: rank 0 and the tf-dsv41-lanes followers take different argv")
            )
        if recipe.min_nodes and recipe.min_nodes > tp:
            issues.append(self.recipe_error("min_nodes=%d exceeds tensor_parallel=%d" % (recipe.min_nodes, tp)))
        return issues

    def prepare(
        self,
        recipe: "Recipe",
        hosts: list[str],
        config=None,
        dry_run: bool = False,
        transfer_mode: str = "auto",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        chain = recipe.build_config_chain(overrides or {})
        dims = self._parallelism_dimensions(chain)
        if dims["tensor_parallel"] not in _DSV41_WORLDS:
            raise RuntimeError(
                "tensorfold-dsv41 is TP2/TP4 only (got tensor_parallel=%d); single-rank DeepSeek-V4.1 is not qualified"
                % dims["tensor_parallel"]
            )
        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                raise RuntimeError("tensorfold-dsv41 does not support %s" % key)
        if not self._is_prepared_bundle(recipe):
            raise RuntimeError("tensorfold-dsv41 needs an absolute prepared-bundle model: path (or cluster_config.resolved_model_path)")
        if len(hosts) != dims["tensor_parallel"]:
            raise RuntimeError(
                "tensorfold-dsv41 needs exactly tensor_parallel hosts (got %d hosts for tp=%d)" % (len(hosts), dims["tensor_parallel"])
            )

    # --- diagnostics --------------------------------------------------------

    def version_commands(self) -> dict[str, str]:
        commands = super().version_commands()
        commands["tensorfold_dsv41"] = "/opt/tensorfold-dsv41/bin/tensorfold-native --version 2>/dev/null | head -1 || echo unknown"
        commands["tensorfold_dsv41_lanes"] = "test -x /opt/tensorfold-dsv41/bin/tf-dsv41-lanes && echo present || echo missing"
        return commands

    # --- cluster lifecycle --------------------------------------------------

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
        comm_env=None,
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
            banner_title="TensorFold DeepSeek-V4.1 Cluster Launcher",
            port_label="TP Port",
            node_label="TensorFold rank",
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
