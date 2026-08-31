"""Recipe-local execution strategies and deterministic preparation hooks.

The core owns strategy selection and the replacement barrier.  Plugins may
contribute preparation steps only through top-level recipe items they own;
merely installing a plugin never changes normal ``sparkrun run`` behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from sparkrun.api._models import RunOptions, RunPlan
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.timing import Timeline


@dataclass(frozen=True)
class ExecutionContext:
    """Read-only inputs available while an execution is prepared."""

    options: "RunOptions"
    plan: "RunPlan"
    sctx: "SparkrunContext"


PreparationAction = Callable[[ExecutionContext, Mapping[str, Any]], Any]
PreparationCleanup = Callable[[ExecutionContext, Any], None]


@dataclass(frozen=True)
class PreparationStep:
    """One named, dependency-ordered unit of launch preparation."""

    name: str
    action: PreparationAction
    requires: tuple[str, ...] = ()
    cleanup: PreparationCleanup | None = None


@dataclass(frozen=True)
class LaunchAssetPolicy:
    """How the shared launcher should prepare assets for a strategy."""

    images_by_node: tuple[str, ...] | None = None
    distribute_images: bool = True
    prepare_model: bool = True
    run_builder: bool = True
    prepare_runtime: bool = True
    probe_images: bool = True
    sync_tuning: bool = True
    clear_page_cache: bool = True


@dataclass(frozen=True)
class PreparedExecution:
    """Strategy decision produced before the shared asset pipeline runs."""

    strategy: str
    assets: LaunchAssetPolicy = field(default_factory=LaunchAssetPolicy)
    state: Any = None
    receipts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivationContext:
    """Final launcher state exposed after assets are resident, before eviction."""

    execution: ExecutionContext
    prepared: PreparedExecution
    cluster_id: str
    hosts: tuple[str, ...]
    container_image: str
    images_by_node: tuple[str, ...]
    effective_cache_dir: str
    serve_port: int
    serve_command: str
    comm_env: Any = None
    ib_ip_map: Mapping[str, str] = field(default_factory=dict)
    ib_iface_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivationResult:
    """Result returned by a strategy in place of ``RuntimePlugin.run``."""

    rc: int
    runtime_info: Mapping[str, str] = field(default_factory=dict)


class RecipeExecutionStrategy(Protocol):
    """Lifecycle implemented by a recipe-owned execution strategy."""

    name: str

    def preparation_steps(self, context: ExecutionContext) -> Sequence[PreparationStep]: ...

    def finalize_preparation(
        self,
        context: ExecutionContext,
        receipts: Mapping[str, Any],
    ) -> PreparedExecution: ...

    def prepare_activation(self, context: ActivationContext) -> Any: ...

    def activate(self, context: ActivationContext, receipt: Any) -> ActivationResult: ...


def resolve_recipe_execution(context: ExecutionContext) -> tuple[RecipeExecutionStrategy | None, tuple[PreparationStep, ...]]:
    """Select the one strategy and all hooks owned by present recipe items."""

    from sparkrun.core.recipe_items import registered_recipe_items

    strategies: list[RecipeExecutionStrategy] = []
    steps: list[PreparationStep] = []
    for registration in registered_recipe_items():
        if registration.key not in context.plan.recipe.plugin_items:
            continue
        if registration.execution_strategy is not None:
            strategies.append(registration.execution_strategy)
            steps.extend(registration.execution_strategy.preparation_steps(context))
        if registration.preparation_steps is not None:
            steps.extend(registration.preparation_steps(context))
    if len(strategies) > 1:
        raise ValueError("recipe selects multiple execution strategies: %s" % ", ".join(sorted(strategy.name for strategy in strategies)))
    return (strategies[0] if strategies else None), tuple(steps)


def run_preparation_steps(
    context: ExecutionContext,
    steps: Sequence[PreparationStep],
    *,
    timeline: "Timeline | None" = None,
    parent: int | None = None,
) -> dict[str, Any]:
    """Run a small deterministic DAG and clean completed work on failure."""

    from sparkrun.core.timing import timed

    indexed: dict[str, PreparationStep] = {}
    for step in steps:
        if not step.name or step.name in indexed:
            raise ValueError("preparation step names must be non-empty and unique: %r" % step.name)
        indexed[step.name] = step
    for step in steps:
        unknown = set(step.requires) - set(indexed)
        if unknown:
            raise ValueError("preparation step %r requires unknown step(s): %s" % (step.name, ", ".join(sorted(unknown))))

    receipts: dict[str, Any] = {}
    completed: list[PreparationStep] = []
    remaining = list(steps)
    try:
        while remaining:
            runnable = next((step for step in remaining if all(name in receipts for name in step.requires)), None)
            if runnable is None:
                raise ValueError("preparation step dependencies contain a cycle")
            with timed(timeline, runnable.name, parent=parent, step=runnable.name):
                receipts[runnable.name] = runnable.action(context, receipts)
            completed.append(runnable)
            remaining.remove(runnable)
    except BaseException:
        for step in reversed(completed):
            if step.cleanup is not None:
                try:
                    with timed(timeline, "%s.cleanup" % step.name, parent=parent, step=step.name):
                        step.cleanup(context, receipts[step.name])
                except Exception:
                    # Cleanup is compensating best-effort work and must not
                    # hide the preparation failure that triggered it.
                    pass
        raise
    return receipts


__all__ = [
    "ActivationContext",
    "ActivationResult",
    "ExecutionContext",
    "LaunchAssetPolicy",
    "PreparationStep",
    "PreparedExecution",
    "RecipeExecutionStrategy",
    "resolve_recipe_execution",
    "run_preparation_steps",
]
