"""TensorFold runtime for sparkrun.

TensorFold (https://github.com/ashhart/TensorFold) is a Zig/CUDA native LLM
server with an OpenAI-compatible API.  It is distributed as a release archive,
not as a Python package, and the server itself is ``tensorfold-native serve``.

One runtime covers the engine's deployment shapes; the recipe selects the shape
with ``defaults.tf_family``:

* **Solo** (no family) — one GB10 (one GPU per DGX Spark).  The engine reads the
  checkpoint from the standard HuggingFace cache or from a local directory.  An
  absolute ``model:`` path (or ``cluster_config.resolved_model_path``) is
  sparkrun's pre-placed/NFS mode: distribution is skipped and the directory is
  identity-mounted into the container.
* **Native TP2, family ``glm`` / ``nemotron``** — the upstream native CUDA build
  has no generic ``--tp`` flag; these families place their own ranks through
  environment variables (``TF_GLM_RANK``/``TF_GLM_WORLD`` and
  ``TF_NEMOTRON_RANK``/``TF_NEMOTRON_WORLD``, both with ``NCCL_COMM_ID``).
  Rank 0 serves HTTP and the followers must already be running when rank 0
  starts, so :meth:`native_start_order` returns ``"workers_first"``.
* **Native TP2/TP4, family ``dsv41``** — the DeepSeek-V4.1 community fork.  Its
  followers run a separate ``tf-dsv41-lanes`` binary and the engine reads a
  prepared per-rank-cache bundle rather than the raw checkpoint.  This is the
  same runtime, not a second plugin: only the command shape and validation
  differ, and both are selected by the same family key.

Container image
---------------
The runtime relies on the platform's per-runtime default image (see
``platforms/dgx_spark.py``) for the upstream build.  That image only needs the
release archive plus a CUDA/NCCL capable base and is built outside this
repository, like the other first-party runtime images.  The DeepSeek fork is not
a first-party image: its recipes name an explicit ``container:`` (the community
build is compiled outside sparkrun) and validation refuses the lane without one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

from sparkrun.core.readiness import (
    ANTHROPIC_MESSAGES_STREAM,
    OPENAI_CHAT_STREAM,
    OPENAI_RESPONSES_STREAM,
)
from sparkrun.core.validation import RecipeIssue
from sparkrun.runtimes import _tensorfold_jayleaton as _jayleaton
from sparkrun.runtimes._util import default_env_hf_offline, parse_flag_value_from_command
from sparkrun.runtimes.base import RuntimePlugin, render_flag_value
from sparkrun.utils.data import integer_setting
from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# Runtime family -> (rank env var, world-size env var).  Only families whose
# native CUDA ranks are placed by environment variables are listed; the engine
# has no generic --tp spelling (docs/multi-rank-receipts.md upstream).
_TP_FAMILIES: dict[str, tuple[str, str]] = {
    "glm": ("TF_GLM_RANK", "TF_GLM_WORLD"),
    "nemotron": ("TF_NEMOTRON_RANK", "TF_NEMOTRON_WORLD"),
}

#: The DeepSeek-V4.1 community fork has its own follower binary and bundle
#: layout.  It is a family of the same runtime, not a separate plugin.
_DSV41_FAMILY = "dsv41"

#: Every family that implies a multi-rank launch.
_ALL_FAMILIES = frozenset({*_TP_FAMILIES, _DSV41_FAMILY, _jayleaton.FAMILY})

#: Recipe ``defaults`` key naming the engine's rank-placement family.
_FAMILY_KEY = "tf_family"

#: Supported rank counts for the DeepSeek fork (upstream GLM/Nemotron allow other
#: worlds but sparkrun qualifies the released ones).
_DSV41_WORLDS = frozenset({2, 4})

#: Default engine locations inside the community DeepSeek image.
_DEFAULT_DSV41_COMMAND = "/opt/tensorfold-dsv41/bin/tensorfold-native"
_DEFAULT_DSV41_FOLLOWER = "/opt/tensorfold-dsv41/bin/tf-dsv41-lanes"

#: Recipe config key -> TensorFold serve flag.  Kept deliberately close to the
#: verified ``serve --help`` surface; exotic flags belong in a recipe's
#: ``command:`` template (solo only) or an image-specific fork.
_TENSORFOLD_FLAG_MAP: dict[str, str] = {
    "max_model_len": "--context",
    "max_num_seqs": "--parallel",
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "min_p": "--min-p",
    "thinking": "--thinking",
    "vision": "--vision",
    "no_drafts": "--no-drafts",
    "drafter": "--drafter",
    "drafter_bits": "--drafter-bits",
    "backend": "--backend",
    "prompt_cache_gib": "--prompt-cache-gib",
    "reasoning_effort": "--reasoning-effort",
    "loop_guard": "--loop-guard",
    "metrics_open": "--metrics-open",
    "dashboard": "--dashboard",
    "alias": "--alias",
    "chat_template": "--chat-template",
}

#: Flags that take no value (presence means "on").
_TENSORFOLD_BOOL_FLAGS = frozenset({"thinking", "vision", "no_drafts", "loop_guard", "metrics_open", "dashboard"})

#: Boolean flags whose ``False`` must render ``--no-<flag>``.  ``thinking`` is
#: the only engine flag that defaults on for some templates; omitting it would
#: silently keep thinking enabled when the recipe asked for it off.
_TENSORFOLD_NEGATABLE_FLAGS = frozenset({"thinking"})

#: Flags that only the serving rank may receive: the worker/follower rank has
#: no HTTP listener, and the DeepSeek follower binary is a different argv shape.
_RANK0_ONLY_FLAG_KEYS = frozenset({"metrics_open", "dashboard"})

#: Keys this runtime consumes outside the flag map.  ``report_unmapped_config_keys``
#: uses this to distinguish "the engine never sees this" from "handled elsewhere".
_TENSORFOLD_OWNED_KEYS = frozenset(
    {
        "host",
        "port",
        "served_model_name",
        "api_key",
        "command_binary",
        _FAMILY_KEY,
        "tensor_parallel",
        "pipeline_parallel",
        "data_parallel",
        "expert_parallel",
    }
)

#: DeepSeek-fork config keys (bundle layout, rank caches, fabric, TP4 2D split).
_DSV41_OWNED_KEYS = frozenset(
    {
        "follower_binary",
        "cache_dir",
        "kit_dir",
        "engram_dir",
        "token_map",
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
        "served_k",
        "tf_env",
    }
)

#: Config keys that are emitted as inline environment assignments, with one
#: env-var name each.  Values are quoted when substituted.  The fork's TP2 and
#: TP4 branches spell the GID knob differently (TF_RDMA_GID_INDEX vs TF_RDMA_GID);
#: the TP4 branch's spelling is supplied through ``tf_env`` in the recipe.
_DSV41_ENV_KEYS: dict[str, str] = {
    "rdma_hca": "TF_RDMA_DEVICES",
    "net_if": "NCCL_SOCKET_IFNAME",
    "rdma_gid_index": "TF_RDMA_GID_INDEX",
    "nccl_ib_gid_index": "NCCL_IB_GID_INDEX",
}

#: Config keys consumed from the structured path (and therefore not "dropped"
#: when reporting unmapped defaults).
_SKIP_KEYS = frozenset({"host", "port", "served_model_name", "api_key", "command_binary", _FAMILY_KEY})


class TensorFoldRuntime(RuntimePlugin):
    """TensorFold Zig/CUDA server: solo, native TP2, and the DeepSeek fork."""

    runtime_name = "tensorfold"
    default_image_prefix = "ghcr.io/spark-arena/tensorfold"

    # The native CUDA fatbins shipped in the release are sm_80/86/120/121;
    # sparkrun's first-class target is the DGX Spark GB10 (sm_121).  A runtime
    # for another accelerator should set an explicit image and remove this
    # gate in the same change that qualifies it.
    requires_capability = frozenset({"gb10"})

    readiness_styles = (OPENAI_CHAT_STREAM, OPENAI_RESPONSES_STREAM, ANTHROPIC_MESSAGES_STREAM)
    readiness_health_path = "/health"

    # --- capabilities / env -------------------------------------------------

    def cluster_strategy(self) -> str:
        """TensorFold clusters use the native rank orchestration, not Ray."""
        return "native"

    def native_api_options(self) -> list[str]:
        """TensorFold serves OpenAI Chat/Completions/Responses and Anthropic Messages."""
        return ["chat_completions", "responses", "messages"]

    def get_common_env(self):
        """HF offline env so the mounted cache is the only model source."""
        return {
            **default_env_hf_offline(),
            "TENSORFOLD_NO_UPDATE_CHECK": "1",
        }

    def get_extra_docker_opts(self) -> list[str]:
        """Memlock/capabilities for GB10 unified memory and NCCL pinning."""
        return ["--cap-add=IPC_LOCK", "--ulimit", "memlock=-1"]

    # --- config surface -----------------------------------------------------

    def known_config_keys(self) -> frozenset[str]:
        return frozenset(_TENSORFOLD_FLAG_MAP) | _TENSORFOLD_OWNED_KEYS | _DSV41_OWNED_KEYS | _jayleaton.owned_keys()

    def serve_flag_map(self):
        return _TENSORFOLD_FLAG_MAP

    def managed_rendezvous_flags(self) -> tuple[str, ...]:
        """TensorFold's rendezvous is environment-based, not serve-flag based."""
        return ()

    # --- config helpers -----------------------------------------------------

    @staticmethod
    def _config(recipe: "Recipe", overrides: dict[str, Any] | None):
        return recipe.build_config_chain(overrides or {})

    @staticmethod
    def _family(config) -> str | None:
        value = config.get(_FAMILY_KEY)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("%s must be a string (glm, nemotron or dsv41)" % _FAMILY_KEY)
        family = value.strip().lower()
        if family not in _ALL_FAMILIES:
            raise ValueError("%s must be one of %s, not %r" % (_FAMILY_KEY, ", ".join(sorted(_ALL_FAMILIES)), value))
        return family

    @staticmethod
    def _tensor_parallel(config) -> int:
        value = config.get("tensor_parallel")
        if value is None:
            return 1
        try:
            return max(1, integer_setting(value, key="tensor_parallel"))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _parallelism_dimensions(config) -> dict[str, int]:
        result: dict[str, int] = {}
        for key in ("tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel"):
            value = config.get(key)
            if value is None:
                result[key] = 1
                continue
            try:
                result[key] = max(1, integer_setting(value, key=key))
            except (TypeError, ValueError):
                result[key] = 1
        return result

    def _requires_tp(self, config) -> bool:
        return self._tensor_parallel(config) > 1

    @staticmethod
    def _is_dsv41(config) -> bool:
        return TensorFoldRuntime._family(config) == _DSV41_FAMILY

    @staticmethod
    def _is_jayleaton(config) -> bool:
        return _jayleaton.is_jayleaton(config)

    @staticmethod
    def _is_prepared_bundle(recipe: "Recipe") -> bool:
        bundle = getattr(getattr(recipe, "cluster_config", None), "resolved_model_path", None) or recipe.model
        return isinstance(bundle, str) and os.path.isabs(bundle)

    # --- placement / orchestration -----------------------------------------

    def world_size(
        self,
        parallelism: "ParallelismConfig",
        *,
        recipe: "Recipe",
        cluster: "ClusterDefinition",
    ) -> int:
        """Ranks this launch needs.

        Every multi-rank family maps one rank to one host (one GPU per Spark).
        Without a family TensorFold is a single-node engine; ``prepare`` and
        ``validate_recipe`` refuse a multi-rank request the engine cannot use.
        """
        config = self._config(recipe, {})
        if self._is_jayleaton(config):
            return _jayleaton.world_size(parallelism, config)
        if self._family(config) in _ALL_FAMILIES and self._requires_tp(config):
            return max(1, parallelism.tensor_parallel)
        return 1

    def native_start_order(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> str:
        """TensorFold followers must be up before rank 0 starts serving."""
        if recipe is None:
            return "head_first"
        config = self._config(recipe, overrides)
        if self._is_jayleaton(config):
            return _jayleaton.start_order(config)
        if self._family(config) in _ALL_FAMILIES and self._requires_tp(config):
            return "workers_first"
        return "head_first"

    def native_rendezvous_port(
        self, recipe: "Recipe | None", overrides: dict[str, Any] | None = None, *, num_nodes: int = 1, init_port: int = 25000
    ) -> int | None:
        """The port carried in ``NCCL_COMM_ID``/``TF_TP_PORT`` (some families use +1)."""
        if recipe is not None:
            config = self._config(recipe, overrides)
            if self._is_jayleaton(config):
                return _jayleaton.rendezvous_port(config, init_port)
        return init_port

    def native_rendezvous_port_span(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> int:
        """TensorFold's follower control channel also needs ``port + 1``."""
        if recipe is None:
            return 1
        config = self._config(recipe, overrides)
        if self._is_jayleaton(config):
            return _jayleaton.rendezvous_port_span(config)
        return 2 if (self._family(config) in _ALL_FAMILIES and self._requires_tp(config)) else 1

    # --- upstream command construction -------------------------------------

    def _build_command(
        self,
        recipe: "Recipe",
        config,
        *,
        bind: bool,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build one upstream TensorFold serve command.

        *bind* is true only for the serving rank: it emits the host/port/name/
        api-key flags that make the rank reachable.  Follower ranks run the same
        engine without an HTTP listener.
        """
        binary = config.get("command_binary")
        binary = "tensorfold-native" if binary is None else str(binary)
        model = config.get("_prepared_model_path") or recipe.model
        parts = ([binary] if binary else []) + ["serve", quote(str(model))]

        if bind:
            host = config.get("host") or "0.0.0.0"
            port = config.get("port") or 8000
            parts.extend(["--host", render_flag_value(host), "--port", render_flag_value(port)])
            served_name = config.get("served_model_name")
            if served_name is not None:
                parts.extend(["--name", render_flag_value(served_name)])
            api_key = config.get("api_key")
            if api_key is not None and str(api_key) != "":
                parts.extend(["--api-key", render_flag_value(api_key)])
        parts.append("--no-update-check")

        skip = set(_SKIP_KEYS) | set(skip_keys)
        if not bind:
            skip |= _RANK0_ONLY_FLAG_KEYS
        parts.extend(
            self.build_flags_from_map(
                config,
                _TENSORFOLD_FLAG_MAP,
                bool_keys=_TENSORFOLD_BOOL_FLAGS,
                skip_keys=skip,
                negatable_keys=_TENSORFOLD_NEGATABLE_FLAGS,
            )
        )
        return " ".join(parts)

    # --- DeepSeek fork bundle/command helpers ------------------------------

    @staticmethod
    def _bundle(recipe: "Recipe", config) -> str:
        """Absolute prepared-bundle root (the recipe's pre-placed ``model:``)."""
        bundle = config.get("_prepared_model_path") or config.get("resolved_model_path") or recipe.model
        if not bundle:
            raise ValueError("the dsv41 family needs a prepared bundle path in model:")
        return str(bundle).rstrip("/")

    @staticmethod
    def _bundle_path(bundle: str, config, key: str, default: str) -> str:
        value = config.get(key)
        return str(value) if value not in (None, "") else "%s/%s" % (bundle, default)

    @staticmethod
    def _env_assignment(name: str, value: object) -> str:
        return "%s=%s" % (name, quote(str(value)))

    def _dsv41_common_flags(self, config, skip_keys: set[str] | frozenset[str]) -> list[str]:
        skip = set(_SKIP_KEYS) | set(skip_keys)
        return self.build_flags_from_map(
            config,
            _TENSORFOLD_FLAG_MAP,
            bool_keys=_TENSORFOLD_BOOL_FLAGS,
            skip_keys=skip,
            negatable_keys=_TENSORFOLD_NEGATABLE_FLAGS,
        )

    def _dsv41_rank0_command(self, config, bundle: str) -> str:
        binary = config.get("command_binary")
        binary = _DEFAULT_DSV41_COMMAND if binary is None else str(binary)
        parts = ([binary] if binary else []) + ["serve", quote("%s/model" % bundle)]
        host = config.get("host") or "0.0.0.0"
        port = config.get("port") or 8000
        parts.extend(["--host", render_flag_value(host), "--port", render_flag_value(port)])
        served_name = config.get("served_model_name")
        if served_name is not None:
            parts.extend(["--name", render_flag_value(served_name)])
        api_key = config.get("api_key")
        if api_key not in (None, ""):
            parts.extend(["--api-key", render_flag_value(api_key)])
        parts.append("--no-update-check")
        parts.extend(["--backend", "cuda"])
        parts.extend(self._dsv41_common_flags(config, frozenset()))
        return " ".join(parts)

    def _dsv41_follower_command(self, config, bundle: str, world: int, rank: int, head_ip: str, init_port: int) -> str:
        binary = config.get("follower_binary")
        binary = _DEFAULT_DSV41_FOLLOWER if binary is None else str(binary)
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

    def _dsv41_rank_env(self, config, bundle: str, world: int, rank: int, head_ip: str, init_port: int) -> list[str]:
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
            # The 2D-split reference deployment serves a solo request per rank.
            env["TF_DS_SERVED_K"] = config.get("served_k") or "solo"
        extra = config.get("tf_env")
        if isinstance(extra, dict):
            for key, value in extra.items():
                if isinstance(key, str) and value is not None:
                    env[key] = value
        return [self._env_assignment(name, value) for name, value in env.items()]

    # --- command generation ------------------------------------------------

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the solo serve command (upstream shape only)."""
        config = self._config(recipe, overrides)
        if self._is_jayleaton(config):
            return _jayleaton.generate_command(recipe, config, is_cluster=is_cluster)
        if self._is_dsv41(config):
            # The native path never executes this string: each rank gets its
            # own argv from generate_node_command() (the followers use a
            # different binary).  The launcher nevertheless requires a
            # non-empty command for validation/metadata, so describe rank 0's
            # shape rather than refusing the whole cluster here.
            if is_cluster:
                return self._dsv41_rank0_command(config, self._bundle(recipe, config))
            raise ValueError(
                "the dsv41 TensorFold family is multi-node only: use --tp 2 or --tp 4 with a prepared bundle "
                "(single-rank DeepSeek-V4.1 has no qualified serving path in this lane)"
            )
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(rendered, config, "--name", skip_keys)
            if skip_keys:
                rendered = self.strip_flags_from_command(rendered, skip_keys, _TENSORFOLD_FLAG_MAP, _TENSORFOLD_BOOL_FLAGS)
            return rendered
        return self._build_command(recipe, config, bind=True, skip_keys=skip_keys)

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
        """Generate the per-rank command for the selected family."""
        config = self._config(recipe, overrides)
        family = self._family(config)
        if family is None:
            raise ValueError(
                "TensorFold multi-node launch needs defaults.tf_family (glm, nemotron or dsv41); "
                "the native CUDA engine has no generic tensor-parallel flag"
            )
        if family == _jayleaton.FAMILY:
            return _jayleaton.generate_node_command(
                recipe,
                config,
                head_ip=head_ip,
                num_nodes=num_nodes,
                node_rank=node_rank,
                init_port=init_port,
            )
        if family == _DSV41_FAMILY:
            if num_nodes not in _DSV41_WORLDS:
                raise ValueError("the dsv41 family supports tp=2 or tp=4 (got %d nodes)" % num_nodes)
            bundle = self._bundle(recipe, config)
            env = self._dsv41_rank_env(config, bundle, num_nodes, node_rank, head_ip, init_port)
            command = (
                self._dsv41_rank0_command(config, bundle)
                if node_rank == 0
                else self._dsv41_follower_command(config, bundle, num_nodes, node_rank, head_ip, init_port)
            )
            return " ".join(env + [command])

        rank_env, world_env = _TP_FAMILIES[family]
        command = self._build_command(recipe, config, bind=node_rank == 0, skip_keys=skip_keys)
        env_prefix = " ".join(
            [
                "%s=%d" % (rank_env, node_rank),
                "%s=%d" % (world_env, num_nodes),
                "NCCL_COMM_ID=%s" % quote("%s:%d" % (head_ip, init_port)),
            ]
        )
        return env_prefix + " " + command

    # --- validation ---------------------------------------------------------

    def validate_recipe(self, recipe: "Recipe") -> list[str | RecipeIssue]:
        config = self._config(recipe, {})
        if self._is_jayleaton(config):
            return _jayleaton.validate(recipe, config, self)
        if self._is_dsv41(config):
            return self._validate_dsv41(recipe, config)
        return self._validate_upstream(recipe, config)

    def _validate_upstream(self, recipe: "Recipe", config) -> list[str | RecipeIssue]:
        issues: list[str | RecipeIssue] = list(RuntimePlugin.validate_recipe(self, recipe))
        dims = self._parallelism_dimensions(config)
        tp = dims["tensor_parallel"]
        family = self._family(config)

        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                issues.append(self.recipe_error("%s is not supported; TensorFold TP is family-specific (see tf_family)" % key))

        if tp > 1:
            if family is None:
                issues.append(
                    self.recipe_error(
                        "multi-node TP needs defaults.tf_family (glm, nemotron or dsv41); the native CUDA engine has no generic --tp flag"
                    )
                )
            if family == "glm" and tp > 16:
                issues.append(self.recipe_error("GLM-5.3-Flash refuses a world larger than 16 ranks"))
            if family == "nemotron" and tp > 2:
                issues.append(
                    self.recipe_warning(
                        "Nemotron 3.5 Super VL is qualified for two ranks only; >2 ranks is experimental and must divide the expert shares"
                    )
                )
            if recipe.command:
                issues.append(
                    self.recipe_error(
                        "a command: template is not supported for multi-node TP; use structured defaults so rank 0 and followers get different bind flags"
                    )
                )
            if recipe.min_nodes and tp < recipe.min_nodes:
                issues.append(self.recipe_error("tensor_parallel=%d is below min_nodes=%d" % (tp, recipe.min_nodes)))
        elif family is not None:
            issues.append(self.recipe_warning("%s only applies to a multi-node launch; it is ignored solo" % _FAMILY_KEY))

        return issues

    def _validate_dsv41(self, recipe: "Recipe", config) -> list[str | RecipeIssue]:
        issues: list[str | RecipeIssue] = list(RuntimePlugin.validate_recipe(self, recipe))
        dims = self._parallelism_dimensions(config)
        tp = dims["tensor_parallel"]
        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                issues.append(self.recipe_error("%s is not supported by the dsv41 family" % key))
        if tp not in _DSV41_WORLDS:
            issues.append(
                self.recipe_error(
                    "the dsv41 family is TP2/TP4 only: set defaults.tensor_parallel to 2 or 4 "
                    "(single-rank DeepSeek-V4.1 is not a qualified TensorFold path)"
                )
            )
        if not self._is_prepared_bundle(recipe):
            issues.append(
                self.recipe_error(
                    "the dsv41 family needs a prepared bundle directory as an absolute model: path "
                    "(or cluster_config.resolved_model_path); see docs/TENSORFOLD.md"
                )
            )
        if recipe.command:
            issues.append(
                self.recipe_error(
                    "a command: template is not supported for the dsv41 family: rank 0 and the tf-dsv41-lanes followers take different argv"
                )
            )
        if not (recipe.container or recipe.containers):
            issues.append(
                self.recipe_error("the dsv41 family needs an explicit container: image; the community fork image is built outside sparkrun")
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
        """Refuse a launch the selected family cannot serve, before assets move."""
        chain = self._config(recipe, overrides)
        dims = self._parallelism_dimensions(chain)
        family = self._family(chain)
        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                raise RuntimeError("TensorFold does not support %s" % key)
        if family == _jayleaton.FAMILY:
            _jayleaton.prepare(recipe, hosts, chain)
            return
        if family == _DSV41_FAMILY:
            if dims["tensor_parallel"] not in _DSV41_WORLDS:
                raise RuntimeError(
                    "the dsv41 family is TP2/TP4 only (got tensor_parallel=%d); single-rank DeepSeek-V4.1 is not qualified"
                    % dims["tensor_parallel"]
                )
            if not self._is_prepared_bundle(recipe):
                raise RuntimeError("the dsv41 family needs an absolute prepared-bundle model: path (or cluster_config.resolved_model_path)")
            if not (recipe.container or recipe.containers):
                raise RuntimeError("the dsv41 family needs an explicit container: image (built outside sparkrun)")
            if len(hosts) != dims["tensor_parallel"]:
                raise RuntimeError(
                    "the dsv41 family needs exactly tensor_parallel hosts (got %d hosts for tp=%d)" % (len(hosts), dims["tensor_parallel"])
                )
            return
        if dims["tensor_parallel"] > 1 and family is None:
            raise RuntimeError(
                "TensorFold multi-node TP needs tf_family (glm, nemotron or dsv41); set it in the recipe defaults or run solo with --tp 1"
            )
        if dims["tensor_parallel"] > 1 and recipe.command:
            raise RuntimeError("TensorFold multi-node TP does not support a recipe command: template")

    # --- proxy/diagnostic hooks --------------------------------------------

    def resolve_api_key(self, recipe: "Recipe", overrides: dict[str, Any] | None = None) -> str | None:
        """Surface the upstream API key so proxy discovery can health-check."""
        chain = self._config(recipe, overrides)
        value = chain.get("api_key")
        if value:
            return str(value)
        parsed = parse_flag_value_from_command(recipe.command, "--api-key")
        return parsed or None

    def version_commands(self) -> dict[str, str]:
        commands = super().version_commands()
        # The images are deliberately Python-free; base Python probes will print
        # "unknown" and that is fine.  The engine version is what matters.
        commands["tensorfold"] = "tensorfold-native --version 2>/dev/null | head -1 || echo unknown"
        commands["nccl_lib"] = "ldconfig -p 2>/dev/null | grep -m1 'libnccl\\.so\\.2' | awk '{print $NF}' || echo unknown"
        commands["tensorfold_dsv41"] = "/opt/tensorfold-dsv41/bin/tensorfold-native --version 2>/dev/null | head -1 || echo unknown"
        commands["tensorfold_dsv41_lanes"] = "test -x /opt/tensorfold-dsv41/bin/tf-dsv41-lanes && echo present || echo missing"
        commands.update(_jayleaton.version_commands())
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
            banner_title="TensorFold Cluster Launcher",
            port_label="NCCL/TP Port",
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
