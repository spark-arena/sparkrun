"""TensorFold runtime for sparkrun.

TensorFold (https://github.com/ashhart/TensorFold) is a Zig/CUDA native LLM
server with an OpenAI-compatible API.  It is distributed as a release archive,
not as a Python package, and the server itself is ``tensorfold-native serve``.

Two deployment shapes are supported:

* **Solo** — one GB10 (one GPU per DGX Spark).  The engine reads the checkpoint
  from the standard HuggingFace cache or from a local directory.  An absolute
  ``model:`` path (or ``cluster_config.resolved_model_path``) is sparkrun's
  pre-placed/NFS mode: distribution is skipped and the directory is
  identity-mounted into the container.
* **Native multi-node TP** — the engine families that place their own ranks
  through environment variables.  As of TensorFold 1.1.3 the native CUDA
  build has no generic ``--tp`` flag; each family wires its own ranks:

  ==============  ======================================  =================
  family          rank variables                          rendezvous
  ==============  ======================================  =================
  ``glm``         ``TF_GLM_RANK`` / ``TF_GLM_WORLD``      ``NCCL_COMM_ID``
  ``nemotron``    ``TF_NEMOTRON_RANK`` / ``TF_NEMOTRON_WORLD``  ``NCCL_COMM_ID``
  ==============  ======================================  =================

  Rank 0 serves HTTP; the follower ranks must already be running when rank 0
  starts, so :meth:`native_start_order` returns ``"workers_first"`` and the
  orchestrator deliberately skips the head-port gate.

The engine's third TP dialect — the DeepSeek-V4.1 community fork, whose
followers run a separate ``tf-dsv41-lanes`` binary — is implemented by
:class:`sparkrun.runtimes.tensorfold_dsv41.TensorFoldDSV41Runtime`, not here.

Container image
---------------
The runtime relies on the platform's per-runtime default image (see
``platforms/dgx_spark.py``).  That image only needs the release archive plus a
CUDA/NCCL capable base; see ``docker/tensorfold`` and
``scripts/build-tensorfold-image.sh``.  A recipe may override ``container:`` to
pin an exact engine version.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.core.validation import RecipeIssue
from sparkrun.core.readiness import (
    ANTHROPIC_MESSAGES_STREAM,
    OPENAI_CHAT_STREAM,
    OPENAI_RESPONSES_STREAM,
)
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

#: Recipe ``defaults`` key naming the engine's rank-placement family.
_FAMILY_KEY = "tf_family"

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
    "alias": "--alias",
    "chat_template": "--chat-template",
}

#: Flags that take no value (presence means "on").
_TENSORFOLD_BOOL_FLAGS = frozenset({"thinking", "vision", "no_drafts", "loop_guard", "metrics_open"})

#: Boolean flags whose ``False`` must render ``--no-<flag>``.  ``thinking`` is
#: the only engine flag that defaults on for some templates; omitting it would
#: silently keep thinking enabled when the recipe asked for it off.
_TENSORFOLD_NEGATABLE_FLAGS = frozenset({"thinking"})

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

#: Runtime configuration keys consumed from the structured path (and therefore
#: not "dropped" when reporting unmapped defaults).
_SKIP_KEYS = frozenset({"host", "port", "served_model_name", "api_key", "command_binary", _FAMILY_KEY})


class TensorFoldRuntime(RuntimePlugin):
    """TensorFold Zig/CUDA server, solo or native multi-node TP."""

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
        return frozenset(_TENSORFOLD_FLAG_MAP) | _TENSORFOLD_OWNED_KEYS

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
            raise ValueError("%s must be a string (glm or nemotron)" % _FAMILY_KEY)
        family = value.strip().lower()
        if family not in _TP_FAMILIES:
            raise ValueError("%s must be one of %s, not %r" % (_FAMILY_KEY, ", ".join(sorted(_TP_FAMILIES)), value))
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

    # --- placement / orchestration -----------------------------------------

    def world_size(
        self,
        parallelism: "ParallelismConfig",
        *,
        recipe: "Recipe",
        cluster: "ClusterDefinition",
    ) -> int:
        """Ranks this launch needs.

        With a TP family the world is ``tensor_parallel`` (one GPU per Spark,
        so one host per rank).  Without one TensorFold is a single-node engine;
        ``prepare``/``validate_recipe`` refuse a multi-rank request rather than
        silently honouring a ``--tp`` the engine cannot use.
        """
        config = self._config(recipe, {})
        if self._family(config) is not None and self._requires_tp(config):
            return max(1, parallelism.tensor_parallel)
        return 1

    def native_start_order(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> str:
        """TensorFold followers must be up before rank 0 starts serving."""
        if recipe is None:
            return "head_first"
        config = self._config(recipe, overrides)
        if self._requires_tp(config) and self._family(config):
            return "workers_first"
        return "head_first"

    def native_rendezvous_port(
        self, recipe: "Recipe | None", overrides: dict[str, Any] | None = None, *, num_nodes: int = 1, init_port: int = 25000
    ) -> int | None:
        """The port carried in ``NCCL_COMM_ID`` (and, for some families, port+1)."""
        return init_port

    def native_rendezvous_port_span(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> int:
        """TensorFold's follower control channel also needs ``port + 1``."""
        if recipe is None:
            return 1
        config = self._config(recipe, overrides)
        return 2 if (self._requires_tp(config) and self._family(config)) else 1

    # --- command construction ----------------------------------------------

    def _build_command(
        self,
        recipe: "Recipe",
        config,
        *,
        bind: bool,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build one TensorFold serve command.

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

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Generate the solo serve command (rank 0 shape)."""
        config = self._config(recipe, overrides)
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
        """Generate the per-rank command for the family's native TP launch."""
        config = self._config(recipe, overrides)
        family = self._family(config)
        if family is None:
            raise ValueError(
                "TensorFold multi-node launch needs defaults.tf_family (glm or nemotron); "
                "the native CUDA engine has no generic tensor-parallel flag"
            )
        rank_env, world_env = _TP_FAMILIES[family]
        # Serving rank only: full bind flags.  Followers: same engine command,
        # no HTTP listener.  Both get the common serve flags.
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
        issues = super().validate_recipe(recipe)
        config = self._config(recipe, {})
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
                        "multi-node TP needs defaults.tf_family (glm or nemotron); the native CUDA engine has no generic --tp flag"
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

    def prepare(
        self,
        recipe: "Recipe",
        hosts: list[str],
        config=None,
        dry_run: bool = False,
        transfer_mode: str = "auto",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Refuse a CLI-only multi-node request that the engine cannot serve.

        ``validate_recipe`` sees recipe defaults only, so ``--tp 2`` without a
        TP family in the recipe would otherwise reach the scheduler and be
        discovered after distribution.  This is the same check on the effective
        config chain, before any asset movement.
        """
        chain = self._config(recipe, overrides)
        dims = self._parallelism_dimensions(chain)
        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                raise RuntimeError("TensorFold does not support %s" % key)
        if dims["tensor_parallel"] > 1 and self._family(chain) is None:
            raise RuntimeError(
                "TensorFold multi-node TP needs tf_family (glm or nemotron); set it in the recipe defaults or run solo with --tp 1"
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
        # The image is deliberately Python-free; base Python probes will print
        # "unknown" and that is fine.  The engine version is what matters.
        commands["tensorfold"] = "tensorfold-native --version 2>/dev/null | head -1 || echo unknown"
        commands["nccl_lib"] = "ldconfig -p 2>/dev/null | grep -m1 'libnccl\\.so\\.2' | awk '{print $NF}' || echo unknown"
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
            port_label="NCCL Comm Port",
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
