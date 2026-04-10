"""Native SGLang runtime for sparkrun."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes._util import default_env_hf_offline
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
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
}

_SGLANG_BOOL_FLAGS = {
    "trust_remote_code",
    "enable_torch_compile",
    "disable_radix_cache",
}


class SglangRuntime(RuntimePlugin):
    """Native SGLang runtime using prebuilt container images.

    SGLang uses its own distributed init mechanism for multi-node inference,
    not Ray.  Each node runs the full ``sglang.launch_server`` command with
    ``--dist-init-addr``, ``--nnodes``, and ``--node-rank`` arguments.
    """

    runtime_name = "sglang"
    default_image_prefix = "scitrera/dgx-spark-sglang"

    def cluster_strategy(self) -> str:
        """SGLang uses native multi-node distribution, not Ray."""
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
        """Generate the sglang launch_server command.

        For cluster mode this produces the *base* command without
        ``--node-rank``.  Use :meth:`generate_node_command` to get the
        per-node variant.
        """
        config = recipe.build_config_chain(overrides)
        self._inject_gguf_model(config)

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
    ) -> str:
        """Generate the sglang command for a specific node.

        Produces the full ``sglang.launch_server`` invocation with the
        node-specific ``--dist-init-addr``, ``--nnodes``, and
        ``--node-rank`` flags appended.
        """
        config = recipe.build_config_chain(overrides)
        self._inject_gguf_model(config)

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

        # Append sglang multi-node arguments
        parts = [
            base,
            "--dist-init-addr %s:%d" % (head_ip, init_port),
            "--nnodes %d" % num_nodes,
            "--node-rank %d" % node_rank,
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

    def _build_base_command(self, recipe: Recipe, config, skip_keys: set[str] | frozenset[str] = frozenset()) -> str:
        """Build the sglang command without cluster-specific arguments."""
        # For GGUF models, use the resolved file path instead of the HF repo name
        model_path = config.get("_gguf_model_path") or recipe.model
        parts = ["python3", "-m", "sglang.launch_server", "--model-path", str(model_path)]

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
        """Build the sglang launch_server command from structured config.

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

            if not tokenizer and not cmd_has_tokenizer:
                issues.append(
                    "[sglang] GGUF model detected but no tokenizer path configured. "
                    "SGLang requires --tokenizer-path pointing to the base (non-GGUF) HF model. "
                    "Set 'tokenizer_path' in defaults (e.g. tokenizer_path: Qwen/Qwen3-1.7B) "
                    "or add --tokenizer-path to the command template."
                )
            if tokenizer and cmd and not cmd_has_tokenizer:
                issues.append(
                    "[sglang] GGUF recipe has 'tokenizer_path' in defaults but the command "
                    "template does not reference {tokenizer_path} or --tokenizer-path. "
                    "Add '--tokenizer-path {tokenizer_path}' to the command template."
                )

        return issues

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
