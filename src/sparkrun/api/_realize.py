"""Realize a recipe for concrete hardware: the plain v2 recipe a launch would run."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from sparkrun.api._models import RunOptions, RunPlan

if TYPE_CHECKING:
    from sparkrun.api._pin import Pin


@dataclass(frozen=True)
class RealizedRecipe:
    """A recipe with everything hardware- and shape-dependent decided.

    :attr:`recipe` is an ordinary v2 recipe mapping (``Recipe.export`` shape):
    matched ``overrides:`` layers and CLI serve overrides are baked into
    ``defaults`` / ``env`` / ``container``, and the ``overrides:`` block is
    gone. Platform tiers (``default_runtime_flags``, ``default_env``) are
    **not** baked in: they are the platform's to apply at launch, and a recipe
    restating them is what ``restated-platform-env`` exists to flag.
    """

    recipe: dict[str, Any]
    plan: RunPlan
    dropped_keys: tuple[str, ...] = field(default=())
    """Selector-only defaults removed because no override reads them any more."""
    pins: tuple["Pin", ...] = field(default=())
    """What was pinned (empty with ``pin=False``)."""
    unpinned: tuple[str, ...] = field(default=())
    """References a pin does not cover (e.g. a separate draft model)."""


def _realized_for(run_plan: RunPlan) -> dict[str, Any]:
    """Provenance for ``metadata``: what the recipe was realized against (no host names)."""
    from sparkrun.platforms import resolve_accelerator_platform

    accelerators: list[str] = []
    platforms: list[str] = []
    for host in run_plan.host_list:
        hardware = run_plan.host_hardware.get(host) or run_plan.cluster.hardware_for(host)
        for accel in hardware.accelerators:
            label = "%s/%s" % (accel.vendor, accel.model)
            if label not in accelerators:
                accelerators.append(label)
            platform = resolve_accelerator_platform(accel, hardware)
            name = (platform.display_name or platform.platform_name) if platform is not None else None
            if name and name not in platforms:
                platforms.append(name)
    info: dict[str, Any] = {"nodes": len(run_plan.host_list)}
    if platforms:
        info["platform"] = platforms[0] if len(platforms) == 1 else platforms
    if accelerators:
        info["accelerator"] = accelerators[0] if len(accelerators) == 1 else accelerators
    return info


def realize_recipe(options: RunOptions, *, pin: bool = True, offline: bool = False, sctx=None) -> RealizedRecipe:
    """Plan *options* against the real cluster and export the recipe it would run.

    Runs :func:`sparkrun.api.plan` (hardware probe, one status sweep,
    placement) and nothing else: no image, model or container is touched.
    The plan is never a dry run, because a dry run skips the hardware probe
    and would realize against inventory or platform assumptions instead of
    the hosts.

    With *pin* (the default) the image becomes ``repo@sha256:…`` and the model
    revision a commit (:func:`sparkrun.api._pin.pin_realized_recipe`). Online,
    those come from the registry and the Hub: what the tags name now.
    *offline* reads them from the placed hosts instead (their resident image
    and cached model), and needs no network beyond the cluster.
    """
    from sparkrun.api._context import resolve_sctx
    from sparkrun.api._pin import pin_realized_recipe, unpinned_references
    from sparkrun.api._run import plan
    from sparkrun.core.launcher import selector_only_config_keys

    sctx = resolve_sctx(sctx)

    run_plan = plan(replace(options, dry_run=False), sctx=sctx)
    recipe = run_plan.recipe
    data = recipe.to_dict(overrides=dict(options.overrides or {}) or None, effective=True)

    dropped = sorted(selector_only_config_keys(recipe, run_plan.runtime) - set(options.overrides or {}))
    defaults = data.get("defaults")
    if dropped and isinstance(defaults, dict):
        for key in dropped:
            defaults.pop(key, None)
        if not defaults:
            data.pop("defaults")

    metadata = dict(data.get("metadata") or {})
    metadata["realized_for"] = _realized_for(run_plan)
    data["metadata"] = metadata
    pins = pin_realized_recipe(data, run_plan, options, offline=offline, sctx=sctx) if pin else ()
    return RealizedRecipe(
        recipe=data, plan=run_plan, dropped_keys=tuple(dropped), pins=pins, unpinned=tuple(unpinned_references(data)) if pin else ()
    )
