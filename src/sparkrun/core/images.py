"""Per-machine container image resolution.

A recipe normally names one ``container:`` and every node runs it.  A recipe
that serves pre-optimized, machine-tuned images instead declares a
``containers:`` block binding an image to a **hostname**:

.. code-block:: yaml

    container: nvcr.io/nvidia/vllm:25.09        # fallback for unlisted machines
    containers:
      - image: myorg/vllm-spark:node-01
        host: spark-01
      - image: myorg/vllm-spark:node-02
        host: spark-02

Keying by hostname rather than by rank or node index is deliberate.  The image
is a property of the *machine*, so a rank-indexed map would be silently wrong
the moment the scheduler ordered hosts differently — and "silently wrong image"
is the entire failure mode this feature has to avoid.  Hostnames are also
declarative, which is what lets :func:`generate_intent_id` hash the map without
becoming placement-dependent (see :attr:`ImagePlan.declared`).

Note this is *not* spelled inside ``layout:``.  ``RecipeLayout.placements`` is
honored **verbatim** by every scheduler, so putting the image there would pin
placement as a side effect of naming an image — wrong when the tuned images
exist on every cluster machine and a ``--tp 2`` launch should still be free to
pick the two idlest.  A recipe that wants both declares ``layout:`` as well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sparkrun.core.recipe import DistributionContainerEntry, Recipe

logger = logging.getLogger(__name__)


class ImagePlanError(ValueError):
    """A ``containers:`` block that cannot be resolved to a runnable plan.

    Always a recipe/cluster mismatch the user must fix — a typo'd hostname, a
    duplicate entry, or a machine with neither an entry nor a ``container:``
    fallback.  Never raised for a recipe without a ``containers:`` block.
    """


def parse_container_entries(raw: Any) -> list[dict[str, str]]:
    """Normalize a raw recipe ``containers:`` value to ``[{host, image}, …]``.

    Permissive by design (the recipe loader parses without a cluster in hand);
    every real validation happens in :func:`resolve_image_plan`, which is the
    only place the cluster's host list is known.  Entries missing either key are
    dropped here rather than raised on, so a malformed block surfaces as the
    actionable "machine X has no image" from the resolver instead of a parse
    error that cannot name the machine.
    """
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        host = str(item.get("host") or "").strip()
        image = str(item.get("image") or "").strip()
        if not host or not image:
            logger.warning("Ignoring containers: entry missing host or image: %r", item)
            continue
        entries.append({"host": host, "image": image})
    return entries


@dataclass(frozen=True)
class ImagePlan:
    """Which container image each node of a launch runs.

    Two maps are kept deliberately, because they answer different questions and
    conflating them breaks workload identity:

    * :attr:`declared` is what the *recipe* said — sorted ``(host, image)``
      pairs, independent of which hosts this launch happens to use.  This is
      what :func:`~sparkrun.orchestration.job_metadata.generate_intent_id`
      hashes; hashing the resolved map instead would make the intent depend on
      placement, so ``stop`` / ``logs`` / ``--ensure`` would stop matching a
      workload whenever the scheduler picked a different host subset.
    * :attr:`images_by_node` is what this launch actually runs, positionally
      aligned with the resolved host list.  This drives container launch and
      the distribution derivation.
    """

    default_image: str
    """The recipe's ``container:`` (post-override, post-builder).  May be empty
    only when every selected host has an explicit entry."""

    declared: tuple[tuple[str, str], ...] = ()
    """Sorted ``(host, image)`` pairs exactly as declared.  Empty ⇒ no
    ``containers:`` block, and every path must behave as it did before this
    feature existed."""

    images_by_node: tuple[str, ...] = ()
    """Image per node, aligned with the resolved host list."""

    @property
    def heterogeneous(self) -> bool:
        """True when this launch runs more than one distinct image."""
        return len(set(self.images_by_node)) > 1

    @property
    def distinct(self) -> tuple[str, ...]:
        """Distinct images in use, in first-node order (stable for logging)."""
        seen: dict[str, None] = {}
        for img in self.images_by_node:
            seen.setdefault(img, None)
        return tuple(seen)

    def image_for_node(self, index: int) -> str:
        """Image for the node at *index*, falling back to the default."""
        if 0 <= index < len(self.images_by_node):
            return self.images_by_node[index]
        return self.default_image

    def head_image(self) -> str:
        """Image the head node runs — the scalar every legacy caller wants."""
        return self.images_by_node[0] if self.images_by_node else self.default_image


def resolve_image_plan(
    recipe: "Recipe",
    default_image: str,
    host_list: list[str],
    cluster_hosts: list[str] | None = None,
) -> ImagePlan:
    """Resolve a recipe's images against the hosts this launch will use.

    Args:
        recipe: Loaded recipe; ``recipe.containers`` supplies the declarations.
        default_image: Resolved ``container:`` — *after* the builder phase, so a
            builder that rewrote the image ref is reflected.
        host_list: Hosts this launch was placed on, in rank order.
        cluster_hosts: The cluster's full host list, used to validate declared
            hostnames.  ``None`` skips that check (explicit ``--hosts`` runs,
            where there is no cluster definition to check against).

    Raises:
        ImagePlanError: A declared host is not in the cluster, a host is
            declared twice, or a selected host has neither an entry nor a
            ``container:`` fallback.
    """
    declared_raw = getattr(recipe, "containers", None) or []

    if not declared_raw:
        # No block: byte-identical to pre-feature behavior.  Note images_by_node
        # is still populated so callers have one uniform accessor.
        return ImagePlan(
            default_image=default_image,
            declared=(),
            images_by_node=tuple(default_image for _ in host_list),
        )

    by_host: dict[str, str] = {}
    for entry in declared_raw:
        host = entry["host"]
        if host in by_host:
            raise ImagePlanError(
                "Recipe declares container image for host '%s' more than once. "
                "Machine-specific images must be unambiguous; remove the duplicate." % host
            )
        by_host[host] = entry["image"]

    # A typo'd hostname would otherwise fall through to the generic `container:`
    # and silently run an untuned image on a machine the user believed covered.
    # Checked against the *cluster's* hosts, not the selected subset, so
    # declaring more machines than a given launch uses stays legal (and is the
    # expected case — the block usually covers the whole cluster).
    if cluster_hosts:
        unknown = sorted(set(by_host) - set(cluster_hosts))
        if unknown:
            raise ImagePlanError(
                "Recipe declares container images for host(s) not in this cluster: %s. "
                "Known hosts: %s." % (", ".join(unknown), ", ".join(cluster_hosts))
            )

    images: list[str] = []
    missing: list[str] = []
    fell_back: list[str] = []
    for host in host_list:
        img = by_host.get(host)
        if img is None:
            if not default_image:
                missing.append(host)
                continue
            fell_back.append(host)
            img = default_image
        images.append(img)

    if missing:
        raise ImagePlanError(
            "No container image for host(s) %s: they have no `containers:` entry and the "
            "recipe declares no `container:` fallback." % ", ".join(missing)
        )

    if fell_back:
        # Never silent: on a machine-tuned cluster, running the generic image is
        # a material difference the user should see without --verbose.
        logger.info(
            "Host(s) %s have no machine-specific image; using the recipe default '%s'",
            ", ".join(fell_back),
            default_image,
        )

    return ImagePlan(
        default_image=default_image,
        declared=tuple(sorted((h, i) for h, i in by_host.items())),
        images_by_node=tuple(images),
    )


def derive_container_entries(plan: ImagePlan, host_list: list[str]) -> list["DistributionContainerEntry"]:
    """Derive distribution entries from a resolved plan.

    One entry per *distinct* image, targeting the node indices that run it.
    This is what keeps "what to ship" and "what to run" from ever disagreeing:
    the distribution view is derived from the launch view rather than being
    declared alongside it.  A hand-written ``distribution_config.containers``
    still wins — the caller checks its ``explicit`` flag before applying this.
    """
    from sparkrun.core.recipe import DistributionContainerEntry

    targets: dict[str, list[int]] = {}
    for index, image in enumerate(plan.images_by_node[: len(host_list)]):
        targets.setdefault(image, []).append(index)

    return [DistributionContainerEntry(name=image, target=indices) for image, indices in targets.items()]
