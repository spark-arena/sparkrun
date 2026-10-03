"""
Container image registry operations.

# TODO: [FUTURE]: allow alternatives to docker!

"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from typing import NamedTuple

from sparkrun.utils.images import is_pullable_image_ref, parse_image_ref

logger = logging.getLogger(__name__)


class ImageIdentity(NamedTuple):
    """Every identity Docker reports for a locally-present image.

    Each field answers a different question, and no single one is reliable
    across hosts, which is why cache decisions consult all of them rather
    than picking a favourite:

    ``image_id``
        The local ``.Id``.  It is recomputed by the *local storage driver*,
        so two hosts on different drivers report **different** IDs for
        byte-identical content (issue #152).  Trustworthy only when the
        drivers agree.

    ``repo_digests``
        ``repo@sha256:...`` entries recorded at pull time.  Registry-
        canonical and driver-agnostic, so this is the best cross-host
        signal available — but it is **empty** for any image built locally
        or moved with ``docker save | docker load``, because neither path
        carries registry metadata.

    ``content_sig``
        A digest of the image's actual content: its configuration plus its
        ordered layer chain.  Both are read verbatim out of the image
        manifest and are stored identically by every driver, so this covers
        the two failure modes above *at once* — different drivers and no
        RepoDigests, which is exactly what a ``save | load`` fan-out
        produces.  ``None`` only when the host could not report it.
    """

    image_id: str | None
    repo_digests: list[str]
    content_sig: str | None = None

    @classmethod
    def empty(cls) -> "ImageIdentity":
        """The identity of an image that is not present."""
        return cls(None, [], None)


# Content-bearing ``docker image inspect`` template, shared by the local and
# the remote probe so the two can never drift apart.
#
# Field order is deliberate: the configuration is last because it is JSON
# and may legitimately contain the ``|`` separator inside a string value,
# whereas the ID, the digest list and the layer list cannot.  A bounded
# split therefore keeps the configuration intact.
#
# The template quotes nothing, which lets the remote probe wrap it in single
# quotes for the shell without escaping anything inside it.
IMAGE_INSPECT_FORMAT = (
    "{{.Id}}|{{range .RepoDigests}}{{.}} {{end}}|{{range .RootFS.Layers}}{{.}} {{end}}|{{json .Config}}"
)

# What ships in :func:`get_image_identity` predates ``content_sig``.  Used as
# a fallback so a host on a Docker too old to render the newer fields
# degrades to the old answer instead of reporting the image as missing.
LEGACY_IMAGE_INSPECT_FORMAT = "{{.Id}}|{{range .RepoDigests}}{{.}} {{end}}"


def content_signature(layers_str: str, config_json: str) -> str | None:
    """Digest an image's content: configuration plus ordered layer chain.

    Both inputs are what Docker read out of the image manifest, so the
    result depends on the image and not on the host that stored it.  The
    configuration is re-serialised canonically because JSON makes no key
    ordering promise.

    Returns ``None`` when neither input is available, so a caller can tell
    "no content signature" apart from "content differs".
    """
    layers = (layers_str or "").strip()
    config_json = (config_json or "").strip()
    if not layers and not config_json:
        return None

    if config_json:
        try:
            config_json = json.dumps(json.loads(config_json), sort_keys=True, separators=(",", ":"))
        except (ValueError, TypeError):
            # Unparsable config: hash it verbatim rather than dropping the
            # field.  A false *mismatch* only costs a redundant transfer,
            # whereas discarding it could mask a real change.
            logger.debug("content_signature: config JSON unparsable; hashing verbatim")

    # NUL cannot occur in either field, so it separates them with no risk of
    # one field bleeding into the other's digest.
    return "sha256:" + hashlib.sha256(("\x00".join([config_json, layers])).encode()).hexdigest()


def parse_image_identity(raw: str) -> ImageIdentity:
    """Parse the output of :data:`IMAGE_INSPECT_FORMAT`.

    Empty *raw* means the image is absent.  Output from a Docker too old for
    the trailing fields parses to an identity without a ``content_sig``, so
    callers keep falling back to the ID/digest signals they already used.
    """
    raw = (raw or "").strip()
    if not raw:
        return ImageIdentity.empty()

    # Bounded split: everything past the third separator is the JSON config,
    # which may itself contain the separator.
    parts = raw.split("|", 3)
    image_id = parts[0].strip() or None
    repo_digests = parts[1].split() if len(parts) > 1 else []
    layers_str = parts[2] if len(parts) > 2 else ""
    config_json = parts[3] if len(parts) > 3 else ""
    return ImageIdentity(image_id, repo_digests, content_signature(layers_str, config_json))


def pull_image(image: str, dry_run: bool = False, required: bool = True) -> int:
    """Pull a container image from a registry.

    Args:
        image: Image reference to pull (e.g. ``"nvcr.io/nvidia/vllm:latest"``).
        dry_run: If True, show what would be done without executing.
        required: If False, suppress error output since this isn't "required"

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
        if required:
            logger.error("Failed to pull image %s: %s", image, result.stderr[:200])
        else:
            logger.info("[NON-CRITICAL] Failed to pull image %s: %s", image, result.stderr[:200])
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


def get_image_identity(image: str) -> ImageIdentity:
    """Get every Docker-reported identity for a local image.

    Image IDs are derived from the *local* image configuration and so vary
    across hosts that use different Docker storage drivers (e.g. overlay2
    vs the containerd overlayfs snapshotter), even for the same registry
    image.  RepoDigests, by contrast, encode the registry manifest hash and
    are stable across storage drivers — but they are absent for images that
    were built locally and never pushed, or that were transferred via
    ``docker save | docker load``.  The content signature covers both gaps.

    Args:
        image: Image reference to inspect.

    Returns:
        An :class:`ImageIdentity`.  ``image_id`` is ``"sha256:abc..."`` or
        ``None`` if the image is not present locally.  ``repo_digests`` is
        the list of ``"repo@sha256:..."`` entries (possibly empty).
        ``content_sig`` is ``None`` only when the host could not report the
        layer chain or configuration.
    """
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", IMAGE_INSPECT_FORMAT, image],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return parse_image_identity(result.stdout)

    # The rich template failed.  That is either "no such image" or a daemon
    # too old for ``.RootFS.Layers``/``json .Config`` -- and the two are
    # indistinguishable from the exit code alone.  Retry with the
    # pre-content_sig template so a legacy daemon still yields the
    # identities it *can* report.  Treating a template error as "image
    # absent" would force a pointless re-transfer of an image that is
    # right there; the retry costs one extra local inspect only on this
    # path.
    legacy = subprocess.run(
        ["docker", "image", "inspect", "--format", LEGACY_IMAGE_INSPECT_FORMAT, image],
        capture_output=True,
        text=True,
    )
    if legacy.returncode != 0:
        return ImageIdentity.empty()

    identity = parse_image_identity(legacy.stdout)
    if identity.image_id is not None:
        logger.debug(
            "image inspect needed legacy format for %r (content signature unavailable on this host)",
            image,
        )
    return identity


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
    # if force_pull, then we pull regardless of local presence or tag; failure to pull on explicit force_pull is an error
    if force_pull:
        logger.info("Force pull requested for image: %s", image)
        return pull_image(image, dry_run=dry_run)

    # "latest" means a mutable tag, so an opportunistic re-pull is worthwhile.
    # Parse the reference rather than testing `":" not in image`: that read the
    # port of a ported registry (`myreg.io:5000/foo`) as a tag and so never
    # refreshed those, and it missed a digest-pinned ref entirely.
    ref = parse_image_ref(image)
    is_latest = is_pullable_image_ref(image) and ref.digest is None and ref.tag in (None, "latest")
    exists_locally = image_exists_locally(image)

    # if image exists and uses latest tag, then we can opportunistically pull it but not failure mode without force_pull
    if exists_locally and is_latest:
        logger.info("Image uses 'latest' tag, attempting non-critical pull: %s", image)
        attempt = pull_image(image, dry_run=dry_run, required=False)
        if attempt == 0:
            logger.info("Fresh Image pulled: %s", image)
        else:
            logger.warning("Failed to pull updated image: %s", image)
        return 0

    # if otherwise image exists and not force_pull, then we're happy
    if exists_locally:
        logger.info("Image already available: %s", image)
        return 0

    # otherwise, we need to pull
    return pull_image(image, dry_run=dry_run)
