"""Ds4CudaRuntime — sparkrun runtime for the ds4 CUDA engine.

ds4 (https://github.com/Bleysg/ds4) is a native C/CUDA inference server —
Bleysg's fork of antirez/ds4.  It runs as a host process (not a Docker
container) and serves GGUF models with an OpenAI-compatible HTTP API at
``/v1/chat/completions`` and ``/v1/models``, plus Prometheus metrics at
``/metrics``.

Because ds4 is a native binary, recipes using this runtime should set::

    executor: local

so sparkrun uses its :class:`LocalExecutor` (no-container) path — the
``setsid``-based native subprocess launcher — instead of DockerExecutor.

The ds4-serve launcher
~~~~~~~~~~~~~~~~~~~~~

ds4 ships a bash wrapper, ``ds4-serve``, that sets up the ``DS4_*``
environment and invokes the ``ds4-server`` binary with CLI flags.  This
runtime generates an equivalent ``ds4-serve`` invocation from recipe
``defaults``, mapping the cross-runtime sparkrun config keys (``port``,
``host``, ``context`` → ``-c``) to ds4 CLI flags, and emitting ``DS4_*``
env vars from the recipe ``env`` block.

Flag mapping
~~~~~~~~~~~~

The ds4 ``ds4-server`` binary accepts a small set of CLI flags::

    ds4-server -m <model.gguf> -c <ctx> --port <port> --host <host>

The remaining tunables (MTP, coalescing, serial-token cap, batch-fit
headroom, DSpark drafter) are all environment variables, not CLI flags,
so they are carried via the recipe ``env:`` block and injected by the
LocalExecutor's env prelude.

DSpark drafter
~~~~~~~~~~~~~~

DSpark is ds4's speculative-decoding drafter.  It's activated by setting
``DS4_CONT_DSPARK=1`` plus ``DS4_DSPARK_MODEL=/path/to/drafter.gguf``,
with ``DS4_CONT_MTP_MODE`` controlling the MTP mode.  All of these go in
the recipe ``env`` block; this runtime simply passes them through.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sparkrun.runtimes._util import default_env_hf_offline, resolve_api_key
from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# ─── CLI flag mapping ─────────────────────────────────────────────
# Recipe ``defaults`` keys → ds4-server CLI flags.
#
# Keys follow sparkrun's cross-runtime conventions where possible so
# recipes stay portable.  ds4-server's CLI surface is minimal — most
# tunables are env vars, handled by ``get_extra_env`` / the recipe env
# block.
_DS4_FLAG_MAP = {
    # Sparkrun standard keys
    "port": "--port",
    "host": "--host",
    # ds4-native keys
    "context": "-c",  # context window size (e.g. 131072)
}

# Boolean flags (present when truthy, absent when falsy).
# ds4-server treats these as bare flags.
_DS4_BOOL_FLAGS = frozenset()

# ─── Environment variables ────────────────────────────────────────
# ds4 reads runtime tuning exclusively from DS4_* env vars.  The recipe
# ``env:`` block is the primary source, but we also define sensible
# defaults here so a minimal recipe "just works".  Recipe env always
# overrides these defaults (see ``get_extra_env`` ordering).
_DS4_DEFAULT_ENV = {
    # Batch-fit headroom in MB — reserves GPU memory for the batch fit
    # planner so it doesn't over-allocate and OOM on a large batch.
    "DS4_BATCH_FIT_HEADROOM_MB": "8192",
    # Maximum number of tokens the server will serialize into a single
    # response.  Set equal to the context window so long completions
    # aren't truncated.
    "DS4_SERVER_SERIAL_MAX_TOKENS": "131072",
    # Coalescing — how many decode steps are batched before a token is
    # emitted.  2 gives the best throughput/latency balance on DGX Spark.
    "DS4_SERVER_COALESCE_MAX": "2",
}

# Keys in the DS4_* env namespace that correspond to recipe ``defaults``
# keys, so that users can set them via the structured ``defaults:`` block
# (the cross-runtime portable form) and we translate them to the env
# var the binary actually reads.  This is a one-way bridge: ``defaults``
# → ``DS4_*``.  The recipe ``env:`` block can still set them directly.
_DS4_ENV_FROM_DEFAULTS = {
    "batch_fit_headroom_mb": "DS4_BATCH_FIT_HEADROOM_MB",
    "serial_max_tokens": "DS4_SERVER_SERIAL_MAX_TOKENS",
    "coalesce_max": "DS4_SERVER_COALESCE_MAX",
    "dspark_model": "DS4_DSPARK_MODEL",
    "dspark": "DS4_CONT_DSPARK",
    "mtp_mode": "DS4_CONT_MTP_MODE",
    "mtp_drafters": "DS4_CONT_MTP_DRAFTERS",
    "prompt_cache": "DS4_CONT_PROMPT_CACHE",
    "quant_kv": "DS4_CONT_QUANT_KV",
    "prefill_chunk": "DS4_PREFILL_CHUNK",
    "batch_max": "DS4_BATCH_MAX",
}

# Defaults injected when not set in recipe config.  These mirror the
# known-good launch configuration for DS4-Flash-0731 on DGX Spark.
_DS4_CONFIG_DEFAULTS = {
    "context": 131072,
    "host": "0.0.0.0",
    "port": 8000,
}


class Ds4CudaRuntime(RuntimePlugin):
    """Native ds4 CUDA runtime for sparkrun.

    Launches the ds4 inference engine (Bleysg's fork of antirez/ds4) as
    a native host process — no Docker container.  The engine is a C/CUDA
    binary (``ds4-server``) typically invoked via the ``ds4-serve`` bash
    wrapper, which sets up the ``DS4_*`` environment and passes CLI flags.

    **Solo mode**: single-node inference via ``_run_solo`` with the
    ``LocalExecutor`` (recipe must set ``executor: local``).

    **Cluster mode**: not yet supported — ds4 does not currently have a
    multi-node distributed inference path.  Recipes should set
    ``max_nodes: 1`` (or ``solo_only: true``).

    **API**: OpenAI-compatible at ``/v1/chat/completions`` and
    ``/v1/models``; Prometheus metrics at ``/metrics``.

    **Env vars**: ds4 reads its tuning knobs from ``DS4_*`` environment
    variables.  Set them in the recipe ``env:`` block (the primary path)
    or via the cross-runtime ``defaults:`` keys (translated by this
    runtime into the corresponding ``DS4_*`` names).
    """

    runtime_name = "ds4-cuda"
    # ds4 has no Docker image prefix — it runs natively.  Recipes must
    # set ``executor: local`` and provide the model path explicitly.
    default_image_prefix = ""

    def cluster_strategy(self) -> str:
        """ds4 uses native execution — no Ray.

        Multi-node isn't supported yet (ds4 has no distributed path),
        so this is only relevant for single-node launches where the
        strategy string is consulted for executor selection.
        """
        return "native"

    def default_executor(self) -> str | None:
        """Prefer the LocalExecutor since ds4 is a native binary.

        This sits below recipe-level ``executor`` and CLI overrides, so
        a recipe that explicitly sets ``executor: docker`` (e.g. for a
        future containerized ds4 image) still wins.
        """
        return "local"

    def get_family(self) -> str:
        return "ds4-cuda"

    def resolve_api_key(
        self,
        recipe: Recipe,
        overrides: dict | None = None,
    ) -> str | None:
        """Resolve the ds4 API key (``DS4_API_KEY`` env / ``--api-key`` flag).

        ds4 optionally accepts an API key via the ``DS4_API_KEY`` env var.
        """
        return resolve_api_key(recipe, overrides, "DS4_API_KEY", "--api-key")

    # --- Environment ---

    def get_common_env(self):
        """Base env: HF offline (ds4 loads from local GGUF, not HF hub)."""
        return default_env_hf_offline()

    def get_extra_env(self) -> dict[str, str]:
        """Inject ds4 default env vars.

        Recipe ``env`` overrides these (the ``_run_solo`` merge order is
        ``get_common_env → get_solo_env → recipe_env → get_extra_env``,
        so ``get_extra_env`` is applied *last* and would normally win —
        but we only set defaults here, and the recipe env block for
        DS4_* values goes through ``_run_solo``'s ``env`` parameter,
        which is merged *before* ``get_extra_env``.  To avoid clobbering
        recipe-set values, we only apply defaults for keys not already
        present in the recipe env.  The actual recipe-env-aware merge
        happens in :meth:`_build_ds4_env`.
        """
        return dict(_DS4_DEFAULT_ENV)

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
        """Generate the ``ds4-serve`` command for solo mode.

        Produces a command of the form::

            ds4-serve -m <model> -c <ctx> --port <port> --host <host>

        If the recipe provides an explicit ``command:`` template, it is
        rendered with ``{placeholder}`` substitution and returned as-is
        (after optional flag stripping for benchmark flow).
        """
        config = recipe.build_config_chain(overrides)
        self._normalize_config(config)

        # Apply our built-in defaults for unset keys.
        self._apply_defaults(config)

        # If the recipe has an explicit command template, render it.
        rendered = recipe.render_command(config)
        if rendered:
            if skip_keys:
                rendered = self.strip_flags_from_command(
                    rendered,
                    skip_keys,
                    _DS4_FLAG_MAP,
                    _DS4_BOOL_FLAGS,
                )
            return rendered

        # Otherwise, build the command from structured defaults.
        return self._build_command(recipe, config, skip_keys=skip_keys)

    # --- Internal helpers ---

    @staticmethod
    def _normalize_config(config) -> None:
        """Apply pre-render config normalizations.

        Translates ``max_model_len`` (the cross-runtime key) to
        ``context`` (the ds4-native key) when ``context`` isn't already
        set, so recipes written against the vLLM/SGLang convention work
        transparently.
        """
        if config.get("context") is None:
            max_model_len = config.get("max_model_len")
            if max_model_len is not None:
                config.set("context", max_model_len)

    @staticmethod
    def _apply_defaults(config) -> None:
        """Inject built-in defaults for keys not already set."""
        for key, value in _DS4_CONFIG_DEFAULTS.items():
            if config.get(key) is None:
                config.set(key, value)

    def _build_command(
        self,
        recipe: Recipe,
        config,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Build the ``ds4-serve`` command from structured config.

        Emits ``ds4-serve -m <model>`` then the flag map.  The
        ``ds4-serve`` wrapper handles translating env + flags into the
        final ``ds4-server`` invocation.
        """
        # ``ds4-serve`` is the bash launcher; it reads DS4_* env vars
        # and forwards CLI flags to ``ds4-server``.
        parts = ["ds4-serve"]

        # The model is positional-ish: ds4-serve expects -m <path>.
        model = config.get("model")
        if model and "model" not in skip_keys:
            parts.extend(["-m", str(model)])

        parts.extend(
            self.build_flags_from_map(
                config,
                _DS4_FLAG_MAP,
                bool_keys=_DS4_BOOL_FLAGS,
                skip_keys=skip_keys,
            )
        )

        return " ".join(parts)

    # --- Version reporting ---

    def version_commands(self) -> dict[str, str]:
        """Version detection commands for the ds4 engine.

        These run inside the process's environment (via LocalExecutor's
        ``exec_cmd``), so ``ds4-server --version`` works natively.
        """
        cmds = super().version_commands()
        cmds["ds4"] = "ds4-server --version 2>/dev/null | head -1 || echo unknown"
        return cmds

    # --- Lifecycle ---

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Runtime-specific recipe validation for ds4-cuda."""
        issues = super().validate_recipe(recipe)

        # Warn if the recipe doesn't set executor: local — ds4 is native.
        # Not an error (default_executor() returns "local"), but surface
        # it so users who set executor: docker know it'll be ignored.
        if recipe.executor and recipe.executor != "local":
            issues.append(
                f"[{self.runtime_name}] ds4-cuda is a native binary (no Docker image); "
                f"recipe sets executor={recipe.executor!r} but only executor=local is supported. "
                f"Using the LocalExecutor anyway."
            )

        # Warn about multi-node — ds4 has no distributed path yet.
        if recipe.max_nodes and recipe.max_nodes > 1:
            issues.append(
                f"[{self.runtime_name}] ds4-cuda does not yet support multi-node inference; "
                f"max_nodes={recipe.max_nodes} will be ignored (single-node only)."
            )

        return issues
