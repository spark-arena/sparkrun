"""Internal helpers for GHCR (GitHub Container Registry) interactions.

Generic utilities for fetching build indexes, listing tags, and
inspecting OCI labels on public GHCR packages.  Builder plugins
(eugr, future sglang, etc.) supply their own URLs and cache names.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# GHCR v2 API base
_GHCR_API = "https://ghcr.io/v2"

# GHCR token endpoint for anonymous bearer auth
_GHCR_TOKEN_URL = "https://ghcr.io/token?scope=repository:{package}:pull&service=ghcr.io"

# Timeout for HTTP requests (seconds)
_HTTP_TIMEOUT = 15


def _ghcr_anonymous_token(package: str) -> str | None:
    """Acquire an anonymous bearer token for a public GHCR package.

    GHCR requires a bearer token even for public packages.  This calls
    the token endpoint and returns the token string, or ``None`` on failure.
    """
    url = _GHCR_TOKEN_URL.format(package=package)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sparkrun"})
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("token")
    except Exception:
        logger.debug("Failed to acquire GHCR token for %s", package, exc_info=True)
        return None


def fetch_build_index(
    url: str,
    cache_dir: Path | None = None,
    cache_name: str = "build-index.json",
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch and optionally cache a JSON build-index file.

    The index is expected to be a JSON array of objects.  Each builder
    defines its own schema (e.g. ``tag``, ``variant``, ``repo_commit``).

    Args:
        url: URL to fetch the build-index JSON from.
        cache_dir: Local directory for caching.  ``None`` disables caching.
        cache_name: Filename for the cached copy (allows multiple builders
            to cache independently in the same directory).
        force_refresh: Bypass cache and re-fetch from network.

    Returns an empty list on any failure (network, parse, etc.).
    """
    cached = None
    if cache_dir is not None:
        cached = cache_dir / cache_name
        if cached.exists() and not force_refresh:
            try:
                data = json.loads(cached.read_text())
                if isinstance(data, list):
                    return data
            except Exception:
                logger.debug("Cached %s unreadable, re-fetching", cache_name)

    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "sparkrun"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, list):
            logger.debug("%s is not a JSON array", cache_name)
            return []
        # Cache the result
        if cached is not None:
            try:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(body)
            except Exception:
                logger.debug("Failed to cache %s", cache_name, exc_info=True)
        return data
    except Exception:
        logger.debug("Failed to fetch %s from %s", cache_name, url, exc_info=True)
        return []


def ghcr_list_tags(image_name: str, tag_pattern: str = r"\d{10}") -> list[str]:
    """List available tags for a GHCR public package.

    Args:
        image_name: Full package path without the registry prefix,
            e.g. ``spark-arena/dgx-vllm-eugr-nightly-tf5``.
        tag_pattern: Regex pattern to filter tags.  Defaults to
            ``YYYYMMDDNN`` (10-digit date+sequence).

    Returns a list of matching tag strings, or empty list on failure.
    """
    url = "%s/%s/tags/list" % (_GHCR_API, image_name)
    try:
        headers = {"User-Agent": "sparkrun"}
        token = _ghcr_anonymous_token(image_name)
        if token:
            headers["Authorization"] = "Bearer %s" % token
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tags = data.get("tags", [])
        return [t for t in tags if re.fullmatch(tag_pattern, t)]
    except Exception:
        logger.debug("Failed to list GHCR tags for %s", image_name, exc_info=True)
        return []


def ghcr_get_labels(image_name: str, tag: str) -> dict[str, str]:
    """Get OCI labels for a specific image tag via GHCR v2 manifest.

    Fetches the manifest, then the config blob to extract labels.
    Returns a flat dict of label key-value pairs, or empty on failure.
    """
    manifest_url = "%s/%s/manifests/%s" % (_GHCR_API, image_name, tag)
    try:
        headers = {"User-Agent": "sparkrun"}
        token = _ghcr_anonymous_token(image_name)
        if token:
            headers["Authorization"] = "Bearer %s" % token

        # Fetch manifest (request OCI or Docker manifest)
        req = urllib.request.Request(
            manifest_url,
            headers={
                **headers,
                "Accept": ("application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"),
            },
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            manifest = json.loads(resp.read().decode("utf-8"))

        # Handle manifest list / index — pick first amd64 manifest
        media_type = manifest.get("mediaType", "")
        if "manifest.list" in media_type or "image.index" in media_type:
            for m in manifest.get("manifests", []):
                platform = m.get("platform", {})
                if platform.get("architecture") == "amd64":
                    return ghcr_get_labels(image_name, m["digest"])
            return {}

        # Get config digest from manifest
        config = manifest.get("config", {})
        config_digest = config.get("digest", "")
        if not config_digest:
            return {}

        # Fetch config blob
        blob_url = "%s/%s/blobs/%s" % (_GHCR_API, image_name, config_digest)
        req = urllib.request.Request(blob_url, headers=headers)
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            config_data = json.loads(resp.read().decode("utf-8"))

        # Labels live in config.config.Labels (OCI image config spec)
        labels = config_data.get("config", {}).get("Labels", {})
        return labels if isinstance(labels, dict) else {}
    except Exception:
        logger.debug(
            "Failed to get labels for %s:%s",
            image_name,
            tag,
            exc_info=True,
        )
        return {}
