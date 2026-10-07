"""Shared builder preparation and container-image planning.

The result carries a launch-local distribution policy for the shared transfer
layer. Image preparation does not transfer assets or open a second transport.
Standalone integrations can stage that plan and require immutable per-host
identities through the shared distribution transaction below.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from sparkrun.core.image_distribution import image_distribution_operation, resolve_distributed_image
from sparkrun.core.images import (
    ImagePlan,
    ImagePlanError,
    derive_container_entries,
    resolve_image_plan,
    resolve_runtime_image_plan,
    validate_runtime_image_plan,
)

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.builders.base import BuilderPlugin
    from sparkrun.core.cluster_manager import ClusterDefinition, ModelDistributionPrefs
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe, DistributionResourceConfig, DistributionContainerEntry
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.core.timing import Timeline
    from sparkrun.runtimes.base import RuntimePlugin


logger = logging.getLogger(__name__)
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


class ImagePreparationError(RuntimeError):
    """The image phase could not produce launchable, resident images."""


@dataclass(frozen=True)
class PreparedImageSet:
    """Builder output and per-node launch image plan."""

    source_image: str | None
    image_plan: ImagePlan | None
    builder: BuilderPlugin | None = None
    container_distribution: DistributionResourceConfig[DistributionContainerEntry] | None = None
    """Launch-local transfer policy; never installed on the reusable recipe.

    No image plan means the executor does not use container assets.
    """

    @property
    def default_image(self) -> str | None:
        return self.image_plan.default_image if self.image_plan is not None else None

    @property
    def images_by_node(self) -> tuple[str, ...]:
        return self.image_plan.images_by_node if self.image_plan is not None else ()

    @property
    def head_image(self) -> str | None:
        return self.image_plan.head_image() if self.image_plan is not None else None


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
    *,
    config: SparkrunConfig | None = None,
    v: Variables | None = None,
    cluster: ClusterDefinition | None = None,
    host_hardware=None,
    dry_run: bool = False,
    transfer_mode: str = "local",
    ssh_kwargs: dict | None = None,
    run_builder: bool = True,
    needs_image: bool = True,
    images_by_node: Sequence[str] | None = None,
    validate: bool = True,
    transform_check: Callable[[Recipe, Variables | None], bool] = builder_transforms_image,
    builder_context: Mapping[str, Any] | None = None,
) -> PreparedImageSet:
    """Run the optional builder and resolve the authoritative per-node plan."""
    if not host_list:
        raise ImagePreparationError("image preparation requires at least one target host")
    if validate and needs_image and images_by_node is None:
        validate_image_configuration(
            recipe,
            runtime,
            v=v,
            run_builder=run_builder,
            transform_check=transform_check,
        )

    from copy import deepcopy
    from sparkrun.core.recipe import RecipeError

    try:
        image_plan = (
            resolve_runtime_image_plan(
                recipe, runtime, host_list, cluster=cluster, host_hardware=host_hardware, images_by_node=images_by_node
            )
            if needs_image
            else None
        )
    except ImagePlanError as error:
        raise RecipeError(str(error)) from error
    if not needs_image and images_by_node is not None:
        raise RecipeError("prepared container images require an executor that uses images")
    source = image_plan.head_image() if image_plan is not None else None
    builder: BuilderPlugin | None = None
    if getattr(recipe, "builder", "") and run_builder:
        from sparkrun.core.bootstrap import get_builder

        builder = get_builder(recipe.builder, v)
        if builder is not None:
            transforms = bool(getattr(builder, "transforms_image", True))
            if transforms and image_plan is None:
                raise RecipeError("An image-transforming builder requires an executor that uses images")
            if transforms and images_by_node is not None:
                raise RecipeError("Prepared images cannot be combined with an image-transforming builder; disable the builder")
            if transforms and image_plan is not None and image_plan.heterogeneous:
                raise RecipeError("A builder requires one source image; set `container:` or build per-host images separately")
            built_image = builder.prepare(
                source or "",
                recipe,
                host_list,
                config=config,
                dry_run=dry_run,
                transfer_mode=transfer_mode,
                ssh_kwargs=ssh_kwargs,
                builder_context=builder_context,
            )
            if image_plan is not None and built_image != source:
                try:
                    image_plan = validate_runtime_image_plan(
                        resolve_image_plan(recipe, built_image, host_list, cluster_hosts=list(cluster.hosts) if cluster else None),
                        runtime,
                        host_list,
                    )
                except ImagePlanError as error:
                    raise RecipeError(str(error)) from error

    container_distribution: DistributionResourceConfig[DistributionContainerEntry] | None = None
    containers = recipe.distribution_config.containers
    if image_plan is not None and (images_by_node is not None or (containers is not None and not containers.explicit)):
        container_distribution = deepcopy(containers)
        if images_by_node is not None:
            container_distribution.enabled = True
        container_distribution.entries = [*derive_container_entries(image_plan, host_list)]
    if image_plan is not None and image_plan.heterogeneous:
        logger.info("Per-machine container images: %d distinct image(s) across %d host(s)", len(image_plan.distinct), len(host_list))
    return PreparedImageSet(source_image=source, image_plan=image_plan, builder=builder, container_distribution=container_distribution)


@image_distribution_operation
def stage_prepared_images(
    prepared: PreparedImageSet,
    recipe: Recipe,
    host_list: list[str],
    cache_dir: str,
    config: SparkrunConfig,
    *,
    dry_run: bool = False,
    recipe_name: str = "",
    transfer_mode: str = "local",
    transfer_interface: str | None = None,
    local_cache_dir: str | None = None,
    pre_ib=None,
    topology: str | None = None,
    prefs: ModelDistributionPrefs | None = None,
    require_content_ids: bool = False,
    ssh_kwargs: dict | None = None,
    stage_models: bool = False,
    mgmt_interface: str | None = None,
    cluster_name: str = "",
    timeline: Timeline | None = None,
    offline: bool = False,
) -> StagedImageSet:
    """Stage a prepared container plan without launching a workload.

    Pass an operation-scoped config (``config.for_cluster(cluster)``). The
    optional SSH mapping must describe that same transport. Distribution owns
    image/model transfers and consumes the prepared, launch-local container
    policy; neither reusable recipe templates nor the policy are mutated.
    Required image identities are verified on every host after image transfer
    and before model transfer. Dry runs never inspect a Docker daemon.
    *offline* is forwarded to distribution: nothing is pulled or downloaded,
    only existing copies are distributed.
    """
    from sparkrun.orchestration.distribution import distribute_from_config
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    if prepared.image_plan is None or not host_list or len(prepared.images_by_node) != len(host_list):
        raise ImagePreparationError("image staging requires one prepared container image per host")
    connection = build_ssh_kwargs(config)
    if ssh_kwargs is not None and ssh_kwargs != connection:
        raise ImagePreparationError("image staging SSH settings must match the operation-scoped config")
    content_images = prepared.images_by_node

    def verify_images() -> None:
        nonlocal content_images
        content_images = resolve_content_images(
            prepared.images_by_node,
            host_list,
            ssh_kwargs=connection,
            dry_run=dry_run,
        )

    comm_env, ib_ip_map, mgmt_ip_map, ib_iface_map = distribute_from_config(
        recipe,
        prepared.image_plan.head_image(),
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
        mgmt_interface=mgmt_interface,
        cluster_name=cluster_name,
        prefs=prefs,
        skip_model=not stage_models,
        skip_container=False,
        after_container_sync=verify_images if require_content_ids else None,
        container_distribution=prepared.container_distribution,
        timeline=timeline,
        offline=offline,
    )
    if not require_content_ids:
        content_images = tuple(
            resolve_distributed_image(image, host, ssh_kwargs=connection, dry_run=dry_run)
            for host, image in zip(host_list, prepared.images_by_node, strict=True)
        )

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
        for index, (host, image) in enumerate(zip(host_list, images_by_node, strict=True)):
            image = resolve_distributed_image(image, host, ssh_kwargs=ssh_kwargs)
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
    if _IMAGE_ID.fullmatch(image) and identity != image:
        raise ImagePreparationError("host %r resolved prepared image %r to a different content ID" % (host, image))
    return image if preserve_reference else identity


__all__ = [
    "ImagePreparationError",
    "PreparedImageSet",
    "StagedImageSet",
    "builder_transforms_image",
    "prepare_images",
    "stage_prepared_images",
    "resolve_content_images",
    "validate_image_configuration",
]
