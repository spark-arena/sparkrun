"""Resolve a container image ref to a pinned ``repo@sha256:…`` digest.

Two sources, because "pin" asks two different questions:

* :func:`resolve_registry_digest`: what the tag points to **now**, from the
  registry (OCI distribution API, no pull). The default for ``export recipe
  --realize``.
* :func:`resolve_host_digest`: what the hosts **already have**, from the
  image's ``RepoDigests`` on each of them. The offline answer: it never leaves
  the cluster, and it pins what is actually resident rather than what a
  mutable tag has moved on to.

Both return the **manifest digest** the registry serves for the tag, which for
a multi-arch image is the index digest. That is what ``docker pull
repo@sha256:…`` expects, and it resolves to the right platform on each host.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from sparkrun.utils.images import parse_image_ref

logger = logging.getLogger(__name__)

_DOCKER_HUB = "registry-1.docker.io"
_DOCKER_HUB_AUTH_KEYS = ("https://index.docker.io/v1/", "index.docker.io", "docker.io", _DOCKER_HUB)
_HTTP_TIMEOUT = 15
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DigestResolutionError(RuntimeError):
    """An image could not be pinned; the message says why and what to do."""


def split_registry(image: str) -> tuple[str, str, str | None, str | None]:
    """``(registry, repository, tag, digest)`` for *image*, with Docker Hub spelled out.

    Docker's own rule: the first path component is a registry host only if it
    contains ``.`` or ``:`` or is ``localhost``. Official Hub images live
    under ``library/``.
    """
    ref = parse_image_ref(image)
    first, _, rest = ref.repository.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, rest
    else:
        registry, repository = _DOCKER_HUB, ref.repository
        if "/" not in repository:
            repository = "library/" + repository
    if registry in ("docker.io", "index.docker.io"):
        registry = _DOCKER_HUB
    return registry, repository, ref.tag, ref.digest


def pinned_ref(image: str, digest: str) -> str:
    """*image* with its tag replaced by *digest* (``repo@sha256:…``), spelled as written."""
    return "%s@%s" % (parse_image_ref(image).repository, digest)


def is_pinned(image: str) -> bool:
    return bool(parse_image_ref(image).digest)


def _docker_config_auth(registry: str) -> str | None:
    """A basic-auth value for *registry* from ``~/.docker/config.json``, if stored inline.

    Credential helpers (``credsStore`` / ``credHelpers``) are not consulted:
    running an arbitrary helper binary to export an image ref is out of
    scope. A private image behind one fails with a message pointing at
    ``--offline``, which reads the digest the hosts already pulled.
    """
    path = Path.home() / ".docker" / "config.json"
    try:
        auths = json.loads(path.read_text()).get("auths") or {}
    except (OSError, ValueError):
        return None
    keys = _DOCKER_HUB_AUTH_KEYS if registry == _DOCKER_HUB else (registry, "https://" + registry)
    for key in keys:
        entry = auths.get(key)
        if isinstance(entry, dict) and entry.get("auth"):
            return str(entry["auth"])
    return None


def _parse_challenge(header: str) -> dict[str, str]:
    scheme, _, params = header.partition(" ")
    if scheme.lower() != "bearer":
        return {}
    return dict(re.findall(r'(\w+)="([^"]*)"', params))


def _bearer_token(challenge: dict[str, str], repository: str, basic_auth: str | None) -> str | None:
    realm = challenge.get("realm")
    if not realm or not realm.startswith("https://"):
        return None
    query = {"service": challenge.get("service", ""), "scope": challenge.get("scope") or "repository:%s:pull" % repository}
    url = realm + "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v})
    headers = {"User-Agent": "sparkrun"}
    if basic_auth:
        headers["Authorization"] = "Basic " + basic_auth
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        logger.debug("Token request to %s failed: %s", realm, error)
        return None
    return data.get("token") or data.get("access_token")


def _manifest_digest(registry: str, repository: str, reference: str, authorization: str | None) -> tuple[int, str | None, str]:
    """``(status, digest, www_authenticate)`` for one manifest request."""
    import hashlib

    url = "https://%s/v2/%s/manifests/%s" % (registry, repository, reference)
    headers = {"Accept": _MANIFEST_ACCEPT, "User-Agent": "sparkrun"}
    if authorization:
        headers["Authorization"] = authorization
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=_HTTP_TIMEOUT) as resp:
            digest = resp.headers.get("Docker-Content-Digest")
            body = resp.read()
            # A registry that omits the header still serves the exact bytes the
            # digest is computed over.
            return resp.status, digest or "sha256:" + hashlib.sha256(body).hexdigest(), ""
    except urllib.error.HTTPError as error:
        return error.code, None, error.headers.get("WWW-Authenticate", "")


def resolve_registry_digest(image: str) -> str:
    """The manifest digest the registry currently serves for *image*'s tag.

    Raises :class:`DigestResolutionError` rather than returning ``None``: a
    pin that silently fell back to the mutable tag would be the opposite of
    what was asked for.
    """
    registry, repository, tag, digest = split_registry(image)
    if digest:
        return digest
    reference = tag or "latest"
    try:
        status, found, challenge = _manifest_digest(registry, repository, reference, None)
        if status == 401 and challenge:
            basic = _docker_config_auth(registry)
            token = _bearer_token(_parse_challenge(challenge), repository, basic)
            if token:
                status, found, challenge = _manifest_digest(registry, repository, reference, "Bearer " + token)
            elif basic and not _parse_challenge(challenge):
                status, found, challenge = _manifest_digest(registry, repository, reference, "Basic " + basic)
    except (urllib.error.URLError, OSError) as error:
        raise DigestResolutionError(
            "cannot reach %s to resolve %s (%s); use --offline to pin the digest the hosts already have" % (registry, image, error)
        ) from error
    if status == 200 and found and _DIGEST_RE.match(found):
        return found
    if status in (401, 403):
        raise DigestResolutionError(
            "%s refused access to %s (HTTP %d). Only inline credentials in ~/.docker/config.json are read, not "
            "credential helpers; use --offline to pin the digest the hosts already pulled" % (registry, image, status)
        )
    if status == 404:
        raise DigestResolutionError("%s has no tag '%s' for %s" % (registry, reference, repository))
    raise DigestResolutionError("unexpected answer from %s for %s (HTTP %s)" % (registry, image, status))


def _repo_digests(repo_digests: list[str], repository: str) -> set[str]:
    """The ``sha256:`` digests recorded for *repository* (any repository when none match)."""
    pairs = [d.rsplit("@", 1) for d in repo_digests if "@" in d]
    return {digest for repo, digest in pairs if repo == repository} or {digest for _repo, digest in pairs}


def _local_identity(image: str) -> tuple[str | None, list[str]]:
    from sparkrun.containers.registry import get_image_identity

    try:
        return get_image_identity(image)
    except OSError as error:  # no docker on the control machine
        logger.debug("Local image identity unavailable for %s: %s", image, error)
        return None, []


def resolve_host_digest(image: str, hosts: list[str], ssh_kwargs: dict | None = None) -> str:
    """The digest of *image* as resident on *hosts*, which must all hold the same build.

    Every host must have the image (an offline launch cannot pull it), and
    they must agree: pinning one build while another host runs a different
    one would silently change what that host serves.

    An image distributed by ``docker save | docker load`` (sparkrun's push and
    delegated modes) carries **no** ``RepoDigests`` on the hosts, only the
    image ID, which the load preserves. So a host without a digest is matched
    by image ID to a holder that has one: another host, or this control
    machine, where a push-mode image was pulled. Hosts that each record a
    digest agree by digest instead, because image IDs can differ across
    storage drivers for the same registry image.
    """
    from sparkrun.containers.distribute import _check_remote_image_identities

    ssh_kwargs = ssh_kwargs or {}
    identities = _check_remote_image_identities(
        image,
        list(hosts),
        ssh_user=ssh_kwargs.get("ssh_user"),
        ssh_key=ssh_kwargs.get("ssh_key"),
        ssh_options=ssh_kwargs.get("ssh_options"),
    )
    missing = [h for h in hosts if h not in identities]
    if missing:
        raise DigestResolutionError("%s is not present on %s, so there is no local digest to pin" % (image, ", ".join(missing)))

    repository = parse_image_ref(image).repository
    recorded = {host: _repo_digests(digests, repository) for host, (_id, digests) in identities.items()}
    if all(recorded.values()):
        common = set.intersection(*recorded.values())
        if len(common) == 1:
            return common.pop()
        detail = "; ".join("%s: %s" % (h, ", ".join(sorted(d))) for h, d in sorted(recorded.items()))
        raise DigestResolutionError("the hosts hold different builds of %s (%s); pull one everywhere first" % (image, detail))

    image_ids = {image_id for image_id, _digests in identities.values()}
    if len(image_ids) != 1 or None in image_ids:
        detail = "; ".join("%s: %s" % (h, i or "?") for h, (i, _d) in sorted(identities.items()))
        raise DigestResolutionError("the hosts hold different builds of %s (image ids %s)" % (image, detail))
    image_id = image_ids.pop()
    local_id, local_digests = _local_identity(image)
    holders = [digests for i, digests in [*identities.values(), (local_id, local_digests)] if i == image_id]
    candidates = set().union(*(_repo_digests(d, repository) for d in holders))
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates:
        raise DigestResolutionError(
            "no copy of %s (image id %s) records a registry digest: it was built or loaded locally everywhere, "
            "so there is nothing to pin offline" % (image, image_id[:19])
        )
    raise DigestResolutionError("image id %s of %s maps to several digests (%s)" % (image_id[:19], image, ", ".join(sorted(candidates))))
