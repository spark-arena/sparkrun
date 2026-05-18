"""Public facade for the executor subsystem.

Concrete executor implementations live under
:mod:`sparkrun.orchestration.executors` (the package).  This module
re-exports the ABC + config + extension point, and adds the
resolution helpers used by both the launcher and the lifecycle
commands (``sparkrun stop`` / ``sparkrun logs``).

Single source of truth for "give me an executor":

- :func:`resolve_executor` — layered config chain (CLI → recipe →
  runtime → per-executor adjustments → SparkrunConfig → per-executor
  defaults).  Used by ``core.launcher`` and the lifecycle helpers in
  ``cli._stop_logs``.
- :func:`get_executor` — look up a registered executor class by
  ``executor_name``.  Mirrors :func:`get_runtime` / :func:`get_builder`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scitrera_app_framework import Variables, get_extensions
from scitrera_app_framework.api import EnvPlacement

from sparkrun.orchestration.executors._base import (
    EXT_EXECUTOR,
    Executor,
    ExecutorConfig,
    accelerator_vendor_for,
)
from sparkrun.orchestration.executors.docker import DOCKER_DEFAULTS, DockerExecutor

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)


#: Alias for :data:`DockerExecutor.default_config()`.  Importable from
#: this module so that callers and tests have a stable public name for
#: the Docker-flavoured defaults without reaching into the
#: implementation package.
EXECUTOR_DEFAULTS = DOCKER_DEFAULTS


__all__ = [
    "EXT_EXECUTOR",
    "EXECUTOR_DEFAULTS",
    "DOCKER_DEFAULTS",
    "Executor",
    "ExecutorConfig",
    "accelerator_vendor_for",
    "get_executor",
    "list_executors",
    "resolve_executor",
]


# ---------------------------------------------------------------------------
# Plugin lookup (mirrors get_runtime / get_builder).
# ---------------------------------------------------------------------------


def get_executor(name: str, v: Variables | None = None) -> type[Executor]:
    """Look up a registered :class:`Executor` *class* by ``executor_name``.

    Unlike :func:`get_runtime`/:func:`get_builder`, this returns the
    **class** rather than the SAF singleton instance: executors carry
    per-launch state on ``self.config``, so callers always instantiate
    a fresh one with the resolved config.

    Falls back to a hard-coded mapping when SAF isn't initialized
    (e.g. test harnesses that build executors directly without going
    through ``init_sparkrun``).
    """
    if v is None:
        try:
            from sparkrun.core.bootstrap import get_variables

            v = get_variables()
        except Exception:  # pragma: no cover - degraded path
            v = None

    if v is not None:
        try:
            all_executors = get_extensions(EXT_EXECUTOR, v=v)
            for _plugin_name, plugin in all_executors.items():
                if plugin.executor_name == name:
                    return type(plugin)
        except Exception:
            logger.debug("Falling back to static executor lookup for %r", name, exc_info=True)

    # Static fallback — keeps :func:`resolve_executor` working in test
    # paths and other harnesses that bypass the full ``init_sparkrun``
    # plugin-discovery bootstrap.
    if name == "docker":
        return DockerExecutor
    if name == "local":
        from sparkrun.orchestration.executors.local import LocalExecutor

        return LocalExecutor
    if name == "k8s":
        from sparkrun.orchestration.executors.k8s import K8sExecutor

        return K8sExecutor

    raise ValueError("Unknown executor: %r" % name)


def list_executors(v: Variables | None = None) -> list[str]:
    """Return registered executor names (sorted)."""
    if v is None:
        from sparkrun.core.bootstrap import get_variables

        v = get_variables()
    all_executors = get_extensions(EXT_EXECUTOR, v=v)
    return sorted(plugin.executor_name for plugin in all_executors.values())


# ---------------------------------------------------------------------------
# Resolution helpers (chain construction).
# ---------------------------------------------------------------------------


def _coerce_str(value) -> str | None:
    """Return *value* coerced to ``str`` iff it's a real string-ish.

    Guards against MagicMock / other non-string objects sneaking
    through the chain (common in launcher unit tests).
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value
    return None


def _coerce_dict(value) -> dict:
    """Return *value* iff it's a real dict, else ``{}``.

    Guards against MagicMock attributes returning auto-magic stand-ins.
    """
    return dict(value) if isinstance(value, dict) else {}


def _recipe_exec_dict(recipe: "Recipe | None") -> dict:
    """Flatten a recipe's executor selector + config into a chain layer."""
    if recipe is None:
        return {}
    cfg = _coerce_dict(getattr(recipe, "executor_config", None))
    selector = _coerce_str(getattr(recipe, "executor", "")) or ""
    if selector and "executor" not in cfg:
        cfg["executor"] = selector
    return cfg


def _runtime_exec_dict(runtime: "RuntimePlugin | None") -> dict:
    """Flatten ``runtime.default_executor()`` into a chain layer."""
    if runtime is None:
        return {}
    fn = getattr(runtime, "default_executor", None)
    if not callable(fn):
        return {}
    try:
        val = fn()
    except Exception:
        return {}
    val = _coerce_str(val)
    return {"executor": val} if val else {}


def _config_exec_dict(config: "SparkrunConfig | None") -> dict:
    """Flatten SparkrunConfig executor defaults into a chain layer."""
    if config is None:
        return {}
    cfg = _coerce_dict(getattr(config, "executor_config", None))
    selector = _coerce_str(getattr(config, "default_executor", None))
    if selector and "executor" not in cfg:
        cfg["executor"] = selector
    return cfg


def _resolve_executor_name(
    *,
    cli_overrides: dict | None,
    recipe: "Recipe | None",
    runtime: "RuntimePlugin | None",
    config: "SparkrunConfig | None",
) -> str:
    """Pick the executor name from the chain (CLI → recipe → runtime → config → docker).

    Unknown names log a warning and fall back to ``"docker"``.
    """
    for layer in (
        cli_overrides,
        _recipe_exec_dict(recipe),
        _runtime_exec_dict(runtime),
        _config_exec_dict(config),
    ):
        if not layer:
            continue
        name = layer.get("executor") or layer.get("executor_type")
        name = _coerce_str(name)
        if not name:
            continue
        name = name.strip().lower()
        if name in ExecutorConfig._KNOWN_EXECUTORS:
            return name
        logger.warning("Unknown executor name %r; falling back to 'docker'", name)
        return "docker"
    return "docker"


def resolve_executor(
    *,
    recipe: "Recipe | None" = None,
    runtime: "RuntimePlugin | None" = None,
    config: "SparkrunConfig | None" = None,
    cli_overrides: dict | None = None,
    rootless: bool = True,
    auto_user: bool = True,
    v: Variables | None = None,
) -> Executor:
    """Single entry point that produces an :class:`Executor` for a launch.

    Layers the resolution chain (highest → lowest):

        1. ``cli_overrides``
        2. ``recipe.executor`` + ``recipe.executor_config``
        3. ``runtime.default_executor()``
        4. ``cls.apply_runtime_adjustments(rootless=, auto_user=)``
        5. ``config.default_executor`` + ``config.executor_config``
        6. ``cls.default_config()``
        7. :class:`ExecutorConfig` dataclass field defaults

    The selected executor class comes from :func:`get_executor` (SAF
    plugin registry); the resulting :class:`ExecutorConfig` is built
    by :meth:`ExecutorConfig.from_chain`.  Returns a fresh
    per-launch instance.
    """
    name = _resolve_executor_name(
        cli_overrides=cli_overrides,
        recipe=recipe,
        runtime=runtime,
        config=config,
    )
    cls = get_executor(name, v)

    chain = Variables(
        sources=(
            cli_overrides or {},
            _recipe_exec_dict(recipe),
            _runtime_exec_dict(runtime),
            cls.apply_runtime_adjustments(rootless=rootless, auto_user=auto_user),
            _config_exec_dict(config),
            cls.default_config(),
        ),
        env_placement=EnvPlacement.IGNORED,
    )
    exec_cfg = ExecutorConfig.from_chain(chain)
    return cls(exec_cfg)
