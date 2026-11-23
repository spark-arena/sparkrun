"""TensorFold runtime for sparkrun.

TensorFold (https://github.com/ashhart/TensorFold) is a Zig/CUDA native LLM
server with an OpenAI-compatible API.  It is distributed as a release archive,
not as a Python package, and the server itself is ``tensorfold-native serve``.

The runtime is deliberately upstream-only: one engine, one launch shape.  There
are no per-model or per-fork families here.  The recipe supplies the model and,
for a multi-node launch, ``tensor_parallel``; this runtime emits the upstream
``--tp/--rank/--master`` flags and leaves any family-specific refusal (for
example a model that serves only one rank) to the engine, which is the authority
on what it supports.

Deployment shapes
-----------------
* **Solo** — one GB10 (one GPU per DGX Spark).  The engine reads the checkpoint
  from the standard HuggingFace cache or from a local directory.  An absolute
  ``model:`` path (or ``cluster_config.resolved_model_path``) is sparkrun's
  pre-placed/NFS mode: distribution is skipped and the directory is
  identity-mounted into the container.
* **Native TP** — one rank per Spark, placed with ``--tp/--rank/--master``.  The
  follower ranks start first and the serving rank starts last, because rank 0
  opens the HTTP listener only after its ranks join.

Container image
---------------
The runtime relies on the platform's per-runtime default image (see
``platforms/dgx_spark.py``) for the upstream build.  The image only needs the
release archive plus a CUDA/NCCL-capable base and is built outside this
repository, like the other first-party runtime images.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.core.readiness import (
    ANTHROPIC_MESSAGES_STREAM,
    OPENAI_CHAT_STREAM,
    OPENAI_RESPONSES_STREAM,
)
from sparkrun.core.validation import RecipeIssue
from sparkrun.runtimes._util import default_env_hf_offline, parse_flag_value_from_command
from sparkrun.runtimes.base import RuntimePlugin, render_flag_value
from sparkrun.utils.data import integer_setting
from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

#: Recipe config key -> TensorFold serve flag.  Kept deliberately close to the
#: verified ``serve --help`` surface; exotic flags belong in a recipe's
#: ``command:`` template (solo only).
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

#: Flags that only the serving rank may receive: follower ranks have no HTTP
#: listener.  The upstream server exposes both on rank 0 only.
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
        "tensor_parallel",
        "pipeline_parallel",
        "data_parallel",
        "expert_parallel",
    }
)

#: Config keys consumed from the structured path, so they are not "dropped".
_SKIP_KEYS = frozenset({"host", "port", "served_model_name", "api_key", "command_binary"})


class TensorFoldRuntime(RuntimePlugin):
    """TensorFold's upstream native CUDA server."""

    runtime_name = "tensorfold"
    default_image_prefix = "ghcr.io/spark-arena/tensorfold"
    required_feature_flag = "runtime.tensorfold"

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
        return frozenset(_TENSORFOLD_FLAG_MAP) | _TENSORFOLD_OWNED_KEYS

    def serve_flag_map(self):
        return _TENSORFOLD_FLAG_MAP

    def managed_rendezvous_flags(self) -> tuple[str, ...]:
        """TensorFold's rendezvous is carried by explicit rank/master flags."""
        return ()

    # --- config helpers -----------------------------------------------------

    @staticmethod
    def _config(recipe: "Recipe", overrides: dict[str, Any] | None):
        return recipe.build_config_chain(overrides or {})

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
        """Ranks this launch needs: one per tensor-parallel host."""
        config = self._config(recipe, {})
        if self._requires_tp(config):
            return max(1, parallelism.tensor_parallel)
        return 1

    def native_start_order(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> str:
        """Rank 0 starts last; its followers must be running before it binds."""
        if recipe is None:
            return "head_first"
        config = self._config(recipe, overrides)
        return "workers_first" if self._requires_tp(config) else "head_first"

    def native_rendezvous_port(
        self, recipe: "Recipe | None", overrides: dict[str, Any] | None = None, *, num_nodes: int = 1, init_port: int = 25000
    ) -> int | None:
        """Rank 0's ``--master-port``; the orchestrator reserves ``init_port``."""
        return init_port

    def native_rendezvous_port_span(self, recipe: "Recipe | None" = None, overrides: dict[str, Any] | None = None) -> int:
        """One rendezvous port; the HTTP port is separate and allocated by sparkrun."""
        return 1

    # --- command construction ----------------------------------------------

    def _build_command(
        self,
        recipe: "Recipe",
        config,
        *,
        bind: bool,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        num_nodes: int = 1,
        node_rank: int = 0,
        head_ip: str | None = None,
        init_port: int = 25000,
    ) -> str:
        """Build one upstream TensorFold serve command.

        *bind* is true only for the serving rank: it emits the host/port/name/
        api-key flags that make the rank reachable.  Follower ranks run the same
        engine with the same ``--tp/--rank/--master`` placement flags but no
        HTTP listener.
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
            if api_key not in (None, ""):
                parts.extend(["--api-key", render_flag_value(api_key)])
        parts.append("--no-update-check")

        if num_nodes > 1:
            if not head_ip:
                raise ValueError("a multi-node TensorFold launch needs the head address for --master")
            parts.extend(
                [
                    "--tp",
                    str(num_nodes),
                    "--rank",
                    str(node_rank),
                    "--master",
                    quote(str(head_ip)),
                    "--master-port",
                    str(init_port),
                ]
            )

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
        """Generate the serving command.

        A cluster launch executes ``generate_node_command`` for every rank; this
        method still has to return a non-empty preview because the launcher uses
        it for metadata and preflight checks.
        """
        config = self._config(recipe, overrides)
        if is_cluster and self._requires_tp(config):
            # No single command is executed for a cluster; describe rank 0's
            # shape without inventing a rank placement.
            return self._build_command(recipe, config, bind=True, skip_keys=skip_keys)

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
        """Generate one upstream TensorFold rank command."""
        config = self._config(recipe, overrides)
        return self._build_command(
            recipe,
            config,
            bind=node_rank == 0,
            skip_keys=skip_keys,
            num_nodes=num_nodes,
            node_rank=node_rank,
            head_ip=head_ip,
            init_port=init_port,
        )

    # --- validation ---------------------------------------------------------

    def validate_recipe(self, recipe: "Recipe") -> list[str | RecipeIssue]:
        config = self._config(recipe, {})
        issues: list[str | RecipeIssue] = list(RuntimePlugin.validate_recipe(self, recipe))
        dims = self._parallelism_dimensions(config)
        tp = dims["tensor_parallel"]

        if "tf_family" in config:
            issues.append(
                self.recipe_error(
                    "tf_family was removed: the TensorFold runtime is upstream-only and places ranks with --tp/--rank/--master"
                )
            )

        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                issues.append(self.recipe_error("%s is not supported by TensorFold" % key))

        if tp > 1:
            if recipe.command:
                issues.append(
                    self.recipe_error(
                        "a command: template is not supported for multi-node TP; use structured defaults so rank 0 and followers get different bind flags"
                    )
                )
            if recipe.min_nodes and tp < recipe.min_nodes:
                issues.append(self.recipe_error("tensor_parallel=%d is below min_nodes=%d" % (tp, recipe.min_nodes)))

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
        """Refuse a launch the upstream engine cannot place, before assets move."""
        chain = self._config(recipe, overrides)
        dims = self._parallelism_dimensions(chain)

        if "tf_family" in chain:
            raise RuntimeError(
                "tf_family was removed: the TensorFold runtime is upstream-only and places ranks with --tp/--rank/--master"
            )
        for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
            if dims[key] > 1:
                raise RuntimeError("TensorFold does not support %s" % key)
        if dims["tensor_parallel"] > 1:
            if recipe.command:
                raise RuntimeError("TensorFold multi-node TP does not support a recipe command: template")
            tp = dims["tensor_parallel"]
            if len(hosts) != tp:
                raise RuntimeError("TensorFold needs exactly one host per tensor_parallel rank (got %d hosts for tp=%d)" % (len(hosts), tp))

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
