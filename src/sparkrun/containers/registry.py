"""
Container image registry operations.

# TODO: [FUTURE]: allow alternatives to docker!

"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def pull_image(image: str, dry_run: bool = False) -> int:
    """Pull a container image from a registry.

    Args:
        image: Image reference to pull (e.g. ``"nvcr.io/nvidia/vllm:latest"``).
        dry_run: If True, show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    if dry_run:
        logger.info("[dry-run] Would pull image: %s", image)
        return 0

    logger.info("Pulling image: %s...", image)
    result = subprocess.run(
        ["docker", "pull", image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to pull image %s: %s", image, result.stderr[:200])
    return result.returncode


def image_exists_locally(image: str) -> bool:
    """Check if a container image exists locally.

    Args:
        image: Image reference to check.

    Returns:
        True if the image exists in the local Docker image store.
    """
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def get_image_identity(image: str) -> tuple[str | None, list[str]]:
    """Get the Docker image ID and RepoDigests for a local image.

    Image IDs are derived from the *local* image configuration and so vary
    across hosts that use different Docker storage drivers (e.g. overlay2
    vs the containerd overlayfs snapshotter), even for the same registry
    image.  RepoDigests, by contrast, encode the registry manifest hash and
    are stable across storage drivers — but they are absent for images that
    were built locally and never pushed, or that were transferred via
    ``docker save | docker load``.

    Args:
        image: Image reference to inspect.

    Returns:
        Tuple ``(image_id, repo_digests)``.  ``image_id`` is
        ``"sha256:abc..."`` or ``None`` if the image is not present
        locally.  ``repo_digests`` is the list of ``"repo@sha256:..."``
        entries (possibly empty).
    """
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}|{{range .RepoDigests}}{{.}} {{end}}",
            image,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None, []
    image_id, _, digests_str = result.stdout.strip().partition("|")
    digests = digests_str.split()
    return (image_id or None), digests


def get_image_id(image: str) -> str | None:
    """Get the Docker image ID for a local image.

    Convenience wrapper around :func:`get_image_identity` that returns only
    the image ID.  Prefer :func:`get_image_identity` when comparing images
    across hosts that may have differing Docker storage drivers.

    Args:
        image: Image reference to inspect.

    Returns:
        Image ID string (e.g. ``"sha256:abc123..."``) or None if the
        image does not exist locally.
    """
    return get_image_identity(image)[0]


def ensure_image(image: str, dry_run: bool = False, force_pull: bool = False) -> int:
    """Ensure an image exists locally, pulling if needed.

    Args:
        image: Image reference.
        dry_run: If True, show what would be done without executing.
        force_pull: If True, force pull even if it exists locally.

    Returns:
        Exit code (0 = success).
    """
    is_latest_or_nightly = image.endswith(":latest") or "nightly" in image
    if not force_pull and not is_latest_or_nightly:
        if image_exists_locally(image):
            logger.info("Image already available: %s", image)
            return 0
    elif is_latest_or_nightly:
        logger.info("Image uses 'latest' or 'nightly' tag; forcing pull: %s", image)
    
    return pull_image(image, dry_run=dry_run)
