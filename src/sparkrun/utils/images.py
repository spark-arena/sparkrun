"""Container image reference parsing.

A leaf module (no sparkrun imports) implementing the parts of Docker's
reference grammar that sparkrun actually decides on.  It lives in ``utils``
rather than ``containers`` because the *builders* need the same answers and
must not depend on the container layer.

Two properties of the grammar are load-bearing and are the reason this is
shared code rather than a per-call-site check:

- **A registry host is not implied by a slash.** ``vllm/vllm-openai`` is a
  Docker Hub repository; the ``docker.io/`` prefix is optional sugar that
  docker fills in.  Requiring an explicit host to call a ref "pullable" is
  what made sparkrun refuse the canonical spelling of nearly every upstream
  image (issue: eugr substituting its nightly for ``vllm/vllm-openai:<tag>``).
- **A colon is not always a tag.** ``myreg.io:5000/foo`` is an untagged image
  on a ported registry.  Only a colon in the *last* path component separates a
  tag, which is why ``":" in ref`` is never a correct test.
"""

from __future__ import annotations

from typing import NamedTuple


class ImageRef(NamedTuple):
    """A parsed container image reference.

    ``tag`` and ``digest`` are ``None`` when absent — which is the distinction
    every caller here cares about, since docker silently resolves a missing tag
    to the mutable ``latest``.
    """

    repository: str
    tag: str | None
    digest: str | None


def parse_image_ref(image: str) -> ImageRef:
    """Split *image* into repository, tag and digest.

    Only a ``:`` in the final path component is a tag separator; a ``:`` in the
    first component is a registry port.
    """
    ref = (image or "").strip()
    repo, _, digest = ref.partition("@")
    last_component = repo.rsplit("/", 1)[-1]
    if ":" in last_component:
        repository, _, tag = repo.rpartition(":")
        return ImageRef(repository, tag, digest or None)
    return ImageRef(repo, None, digest or None)


def image_has_explicit_version(image: str) -> bool:
    """True when *image* pins a tag or digest.

    An untagged ref is not an error — docker resolves it to ``latest`` — but it
    is never what an inference recipe means, so callers warn on it.
    """
    ref = parse_image_ref(image)
    return bool(ref.tag or ref.digest)


def is_pullable_image_ref(image: str) -> bool:
    """True when *image* names a remote repository docker can pull.

    A ref containing ``/`` names either an explicit registry host (first
    component has a ``.`` or ``:``, or is ``localhost``) or an implicit Docker
    Hub namespace (``vllm/vllm-openai``).  Both are pullable.

    A bare single-component name (``vllm-node``, ``sparkrun-eugr-vllm``) is
    reported **not** pullable.  It is genuinely ambiguous — Docker Hub's
    ``library/`` namespace spells official images that way too — and locally
    built tags are overwhelmingly the common case for that shape in sparkrun,
    so treating it as local preserves the builders' build-if-absent behavior.
    """
    repository = parse_image_ref(image).repository
    return "/" in repository
