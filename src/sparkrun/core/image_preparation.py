"""Shared container-image preparation for launches and integrations.

The phase deliberately stops at a typed receipt.  Normal inference launches
feed that receipt into their existing combined image/model distribution path;
an integration that stages images *without* launching (a capture step, an
image-only preflight) can request the images alone plus immutable per-node
identities, rather than reimplementing the builder / image-plan ordering and
being free to get it subtly different.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from sparkrun.core.images import ImagePlan, ImagePlanError, derive_container_entries, resolve_image_plan

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.builders.base import BuilderPlugin
    from sparkrun.core.cluster_manager import ClusterDefinition, ModelDistributionPrefs
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.timing import Timeline
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.runtimes.base import RuntimePlugin


logger = logging.getLogger(__name__)
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


class ImagePreparationError(RuntimeError):
    """The image phase could not produce launchable, resident images."""


@dataclass(frozen=True)
class PreparedImageSet:
    """Builder output and per-node launch image plan."""

    source_image: str
    image_plan: ImagePlan
    builder: BuilderPlugin | None = None

    @property
    def default_image(self) -> str:
        return self.image_plan.default_image

    @property
    def images_by_node(self) -> tuple[str, ...]:
        return self.image_plan.images_by_node

    @property
    def head_image(self) -> str:
        return self.image_plan.head_image()


@dataclass(frozen=True)
class StagedImageSet:
    """Prepared images after container distribution completed."""

    prepared: PreparedImageSet
    content_images_by_node: tuple[str, ...]
    comm_env: ClusterCommEnv | None = None
    ib_ip_map: dict[str, str] | None = None
    mgmt_ip_map: dict[str, str] | None = None
    ib_iface_map: dict[str, str] | None = None


def builder_transforms_image(recipe: Recipe, v: Variables | None = None) -> bool:
    """Best-effort image-transform check used by the pre-side-effect guard."""
    if not getattr(recipe, "builder", ""):
        return False
    try:
        from sparkrun.core.bootstrap import get_builder

        builder = get_builder(recipe.builder, v)
    except Exception:
        logger.debug("Could not resolve builder '%s' for image-transform check", recipe.builder, exc_info=True)
        return False
    return bool(getattr(builder, "transforms_image", True))


def validate_image_configuration(
    recipe: Recipe,
    runtime: RuntimePlugin,
    *,
    v: Variables | None = None,
    run_builder: bool = True,
    transform_check: Callable[[Recipe, Variables | None], bool] = builder_transforms_image,
) -> None:
    """Fail before side effects on incompatible per-machine image settings."""
    if not getattr(recipe, "containers", None):
        return

    from sparkrun.core.recipe import RecipeError

    if not runtime.supports_heterogeneous_images:
        raise RecipeError(
            "Recipe declares per-machine container images (`containers:`), which the '%s' runtime "
            "does not support: its ranks must all run the same build (Ray requires one build across "
            "head and workers; MPI ranks must share an ABI). Remove the `containers:` block, or use "
            "a runtime that supports it (sglang, vllm-distributed, llama-cpp)." % runtime.runtime_name
        )
    if run_builder and transform_check(recipe, v):
        raise RecipeError(
            "Recipe declares per-machine container images (`containers:`) together with builder '%s', "
            "which builds the image it is given. A builder produces one image, so the two cannot be "
            "combined — build the per-machine images out of band and reference them by tag, or drop "
            "the `containers:` block." % recipe.builder
        )


def prepare_images(
    recipe: Recipe,
    runtime: RuntimePlugin,
    host_list: list[str],
    overrides: dict[str, Any],
    *,
    config: SparkrunConfig | None = None,
    v: Variables | None = None,
    cluster: ClusterDefinition | None = None,
    dry_run: bool = False,
    transfer_mode: str = "local",
    ssh_kwargs: dict | None = None,
    run_builder: bool = True,
    images_by_node: Sequence[str] | None = None,
    strategy_name: str = "",
    source_image: str | None = None,
    validate: bool = True,
    transform_check: Callable[[Recipe, Variables | None], bool] = builder_transforms_image,
    builder_context: Mapping[str, Any] | None = None,
) -> PreparedImageSet:
    """Run the optional builder and resolve the authoritative per-node plan."""
    if not host_list:
        raise ImagePreparationError("image preparation requires at least one target host")
    if validate:
        validate_image_configuration(
            recipe,
            runtime,
            v=v,
            run_builder=run_builder,
            transform_check=transform_check,
        )

    source = source_image if source_image is not None else runtime.resolve_container(recipe, overrides)
    default_image = source
    builder: BuilderPlugin | None = None
    if getattr(recipe, "builder", "") and run_builder:
        from sparkrun.core.bootstrap import get_builder

        # A recipe-selected builder is part of the launch contract. Unknown
        # and unavailable builders are fatal; silently skipping would launch
        # an image/environment the recipe did not describe.
        builder = get_builder(recipe.builder, v)
        if builder is not None:
            default_image = builder.prepare(
                default_image,
                recipe,
                host_list,
                config=config,
                dry_run=dry_run,
                transfer_mode=transfer_mode,
                ssh_kwargs=ssh_kwargs,
                builder_context=builder_context,
            )

    from sparkrun.core.recipe import RecipeError

    try:
        image_plan = resolve_image_plan(
            recipe,
            default_image,
            host_list,
            cluster_hosts=list(cluster.hosts) if cluster is not None else None,
        )
    except ImagePlanError as error:
        raise RecipeError(str(error)) from error

    if images_by_node is not None:
        resolved = tuple(str(image).strip() for image in images_by_node)
        if len(resolved) != len(host_list):
            owner = "execution strategy %r" % strategy_name if strategy_name else "image preparation override"
            raise RecipeError("%s prepared %d image(s) for %d host(s)" % (owner, len(resolved), len(host_list)))
        if not all(resolved):
            raise RecipeError("prepared image references must be non-empty")
        image_plan = ImagePlan(default_image=resolved[0], images_by_node=resolved)

    containers = getattr(getattr(recipe, "distribution_config", None), "containers", None)
    if containers is not None:
        if images_by_node is not None:
            containers.enabled = True
            containers.entries = derive_container_entries(image_plan, host_list)
        elif image_plan.heterogeneous and not containers.explicit:
            containers.entries = derive_container_entries(image_plan, host_list)

    if image_plan.heterogeneous:
        logger.info(
            "Per-machine container images: %d distinct image(s) across %d host(s)",
            len(image_plan.distinct),
            len(host_list),
        )
    return PreparedImageSet(source_image=source, image_plan=image_plan, builder=builder)


def stage_prepared_images(
    prepared: PreparedImageSet,
    recipe: Recipe,
    host_list: list[str],
    cache_dir: str,
    config: SparkrunConfig,
    *,
    dry_run: bool = False,
    recipe_name: str | None = None,
    transfer_mode: str = "local",
    transfer_interface: str | None = None,
    local_cache_dir: str | None = None,
    pre_ib=None,
    topology: str | None = None,
    prefs: ModelDistributionPrefs | None = None,
    require_content_ids: bool = False,
    ssh_kwargs: dict | None = None,
    stage_models: bool = False,
    timeline: "Timeline | None" = None,
) -> StagedImageSet:
    """Distribute prepared images, optional model assets, and pin node IDs."""
    from sparkrun.orchestration.distribution import distribute_from_config

    comm_env, ib_ip_map, mgmt_ip_map, ib_iface_map = distribute_from_config(
        recipe,
        prepared.head_image,
        host_list,
        cache_dir,
        config,
        dry_run,
        recipe_name=recipe_name,
        transfer_mode=transfer_mode,
        transfer_interface=transfer_interface,
        local_cache_dir=local_cache_dir,
        pre_ib=pre_ib,
        topology=topology,
        prefs=prefs,
        skip_model=not stage_models,
        skip_container=False,
        timeline=timeline,
    )
    if require_content_ids:
        content_images = resolve_content_images(
            prepared.images_by_node,
            host_list,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
        )
    else:
        content_images = prepared.images_by_node
    return StagedImageSet(
        prepared=prepared,
        content_images_by_node=content_images,
        comm_env=comm_env,
        ib_ip_map=ib_ip_map,
        mgmt_ip_map=mgmt_ip_map,
        ib_iface_map=ib_iface_map,
    )


def resolve_content_images(
    images_by_node: Sequence[str],
    host_list: Sequence[str],
    *,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> tuple[str, ...]:
    """Return immutable, locally runnable image references for every node."""
    if len(images_by_node) != len(host_list):
        raise ImagePreparationError("image identity resolution requires one image per host")
    if dry_run:
        return tuple(images_by_node)

    resolved: list[str | None] = [None] * len(host_list)
    pending: dict[Any, int] = {}
    with ThreadPoolExecutor(max_workers=min(max(len(host_list), 1), 16)) as pool:
        for index, (host, image) in enumerate(zip(host_list, images_by_node)):
            preserve = bool(_PINNED_IMAGE.fullmatch(image) or _IMAGE_ID.fullmatch(image))
            pending[pool.submit(_resolve_host_image_id, host, image, ssh_kwargs or {}, preserve)] = index
        for future in as_completed(pending):
            resolved[pending[future]] = future.result()

    missing = [str(index) for index, image in enumerate(resolved) if not image]
    if missing:
        raise ImagePreparationError("could not resolve immutable image identities for node(s): %s" % ", ".join(missing))
    return tuple(str(image) for image in resolved)


def _resolve_host_image_id(host: str, image: str, ssh_kwargs: dict, preserve_reference: bool) -> str:
    from sparkrun.orchestration.primitives import run_command_on_host
    from sparkrun.utils.shell import quote

    result = run_command_on_host(
        host,
        "docker image inspect --format '{{.Id}}' %s" % quote(image),
        ssh_kwargs=ssh_kwargs,
        timeout=30,
        quiet=True,
    )
    value = str(getattr(result, "stdout", "") or "").strip().splitlines()[:1]
    identity = value[0].strip() if value else ""
    if not getattr(result, "success", False) or not _IMAGE_ID.fullmatch(identity):
        detail = str(getattr(result, "stderr", "") or "").strip()
        raise ImagePreparationError(
            "host %r could not resolve prepared image %r%s" % (host, image, ": " + detail[-1000:] if detail else "")
        )
    return image if preserve_reference else identity


__all__ = [
    "ImagePreparationError",
    "PreparedImageSet",
    "StagedImageSet",
    "builder_transforms_image",
    "prepare_images",
    "resolve_content_images",
    "stage_prepared_images",
    "validate_image_configuration",
]
