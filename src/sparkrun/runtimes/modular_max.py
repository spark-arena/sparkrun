"""Modular MAX runtime for sparkrun.

`MAX <https://docs.modular.com/max/>`_ is Modular's OpenAI-compatible inference
server, invoked as ``max serve --model <id>``.  Unlike vLLM/SGLang/tokenary, MAX
has **no multi-node / distributed tensor parallelism**: it runs as a single
process on a single node and uses local GPUs via its ``--devices`` flag
(``gpu``, ``gpu:all``, ``gpu:0,1``).

Consequences for sparkrun (which otherwise maps ``--tp N`` to N nodes on a
1-GPU-per-host DGX Spark cluster):

* This runtime is **single-node only**.  :meth:`world_size` always returns ``1``
  so the scheduler places it solo and the multi-node (``_run_cluster``) path is
  never reached.
* ``tensor_parallel`` is interpreted as **N local GPUs on one host** and emitted
  as ``--devices gpu:0,...,(N-1)`` — never as additional ranks/hosts.
* On a single-GPU host (every DGX Spark), ``tensor_parallel > 1`` is rejected by
  :meth:`prepare` after probing the host's accelerator count.

Container images: ``modular/max-nvidia-full`` (default) and ``modular/max-nvidia-base``.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes._util import default_env_hf_offline
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# MAX `serve` CLI flag mapping (recipe config key -> MAX flag).  Only flags
# verified against the MAX source (max/python/max/.../model_config.py and
# cli/config.py) are included.  ``port`` / ``tensor_parallel`` / ``devices`` are
# emitted explicitly by :meth:`ModularMaxRuntime._build_command`, so they are
# excluded from the structured flag pass.
_MAX_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "max_model_len": "--max-length",
    "max_num_seqs": "--max-batch-size",
    "served_model_name": "--served-model-name",
    "quantization": "--quantization-encoding",
    "devices": "--devices",
}

# No boolean (value-less) flags are verified for MAX serve yet.
_MAX_BOOL_FLAGS: frozenset[str] = frozenset()


class ModularMaxRuntime(RuntimePlugin):
    """Single-node Modular MAX runtime using prebuilt container images.

    The solo launch path (container with ``sleep infinity`` + ``docker exec``)
    runs the string returned by :meth:`generate_command`, i.e.
    ``max serve --model <id> --port <p> [--devices ...] [flags...]``.  The base
    class mounts the HuggingFace cache at ``/cache/huggingface`` and sets
    ``HF_HOME``, so a pre-synced model is found with no extra wiring.
    """

    runtime_name = "modular-max"
    default_image_prefix = "modular/max-nvidia-full"

    # --- placement: single node, always ---

    # noinspection PyUnusedLocal
    def world_size(
        self,
        parallelism: "ParallelismConfig",
        *,
        recipe: "Recipe",
        cluster: "ClusterDefinition",
    ) -> int:
        """MAX is one process on one node — world size is always ``1``.

        Tensor parallelism is satisfied by *local* GPUs (``--devices``), never by
        additional ranks/hosts.  Returning ``1`` forces the scheduler to place
        the workload solo and guarantees the base ``_run_cluster``
        (``NotImplementedError``) path is never reached.  The per-host GPU budget
        for ``tensor_parallel`` is validated separately in :meth:`prepare`.
        """
        return 1

    # --- command construction ---

    @staticmethod
    def _resolve_devices(config) -> str | None:
        """Resolve the MAX ``--devices`` value.

        Priority: an explicit ``devices`` recipe key (passed through verbatim,
        e.g. ``gpu:all``) wins; otherwise ``tensor_parallel > 1`` expands to
        ``gpu:0,1,...,(tp-1)``.  Returns ``None`` (omit the flag, MAX defaults to
        a single GPU) when neither applies.
        """
        explicit = config.get("devices")
        if explicit:
            return str(explicit)
        tp = config.get("tensor_parallel")
        try:
            tp = int(tp) if tp is not None else 1
        except (TypeError, ValueError):
            tp = 1
        if tp > 1:
            return "gpu:" + ",".join(str(i) for i in range(tp))
        return None

    def _build_command(
        self,
        recipe: "Recipe",
        config,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build ``max serve`` from structured config."""
        parts = ["max", "serve", "--model", str(recipe.model)]

        port = config.get("port") or 8000
        parts.extend(["--port", str(port)])

        devices = self._resolve_devices(config)
        if devices:
            parts.extend(["--devices", devices])

        # ``port`` / ``devices`` already emitted above; ``tensor_parallel`` is
        # expressed via ``--devices`` and has no standalone MAX flag.
        skip = {"port", "devices", "tensor_parallel"}
        skip.update(skip_keys)
        parts.extend(
            self.build_flags_from_map(
                config,
                _MAX_FLAG_MAP,
                bool_keys=_MAX_BOOL_FLAGS,
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
        """Generate the ``max serve`` command (always single-node).

        ``is_cluster`` / ``num_nodes`` / ``head_ip`` are accepted for interface
        compatibility but unused: :meth:`world_size` pins this runtime to a
        single node, so it is only ever invoked in solo mode.
        """
        config = recipe.build_config_chain(overrides)

        # Honour an explicit command template if the recipe provides one.
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
                    _MAX_FLAG_MAP,
                    _MAX_BOOL_FLAGS,
                )
            return rendered

        return self._build_command(recipe, config, skip_keys=skip_keys)

    # --- environment ---

    def get_common_env(self):
        """HF offline env so MAX reads pre-synced weights from the mounted cache."""
        return default_env_hf_offline()

    # --- validation ---

    def validate_recipe(self, recipe: "Recipe") -> list[str]:
        """Reject multi-node intent — MAX cannot span nodes."""
        issues = super().validate_recipe(recipe)
        if recipe.min_nodes and recipe.min_nodes > 1:
            issues.append(
                "[modular-max] is single-node only: min_nodes=%d (or cluster_only) "
                "is not supported. MAX uses local GPUs via --devices, not multi-node "
                "tensor parallelism." % recipe.min_nodes
            )
        return issues

    # --- pre-launch GPU-count guard ---

    def prepare(
        self,
        recipe: "Recipe",
        hosts: list[str],
        config: "SparkrunConfig | None" = None,
        dry_run: bool = False,
        transfer_mode: str = "auto",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Reject ``tensor_parallel > 1`` when the target host lacks the GPUs.

        MAX maps tensor parallelism onto *local* GPUs, so a request for
        ``--tp N`` requires a single host with at least N accelerators.  On every
        DGX Spark (1 GPU per host) this rejects ``--tp > 1`` outright; on a
        multi-GPU host it permits ``--tp`` up to the local GPU count.
        """
        config_chain = recipe.build_config_chain(overrides)
        tp = config_chain.get("tensor_parallel")
        try:
            tp = int(tp) if tp is not None else 1
        except (TypeError, ValueError):
            tp = 1
        if tp <= 1 or dry_run or not hosts:
            return

        from sparkrun.core.hardware_probe import probe_hosts
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        hardware = probe_hosts(list(hosts), build_ssh_kwargs(config))
        for host in hosts:
            hw = hardware.get(host)
            gpus = hw.total_gpus if hw else 0
            if gpus < tp:
                raise RuntimeError(
                    "modular-max is single-node only: tensor_parallel=%d requires one "
                    "host with >=%d GPUs, but %s reports %d. MAX cannot span nodes — "
                    "reduce --tp or target a host with more local GPUs." % (tp, tp, host, gpus)
                )
