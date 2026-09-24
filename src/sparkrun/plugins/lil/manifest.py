"""lil manifests: ``<Entry>/lil.yaml`` over the vendored model bases.

A catalog entry is layered the way the lil launcher layers it
(``LoadModelEntry``): the bases' global ``defaults``, then the entry's
``family`` (resolved through its parent chain), then the entry itself.
Mappings merge. Scalars, lists and explicit nulls replace.

The bases are **not** in the catalog. The launcher embeds them, so they are
vendored here (``bases.yaml``) at a pinned upstream commit.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

#: Where ``bases.yaml`` was vendored from. Update with the file.
BASES_UPSTREAM = {
    "repository": "https://github.com/local-inference-lab/lil",
    "path": "configs/models/_bases.yaml",
    "commit": "c861cf92dd777b4b46c64f0ea18bca2294fd730b",
    "sha256": "e71d07e08549739422776325e47d26904fdae4f8c913b72f8f2b20f28ef87872",
}

BASES_PATH = Path(__file__).with_name("bases.yaml")

_HF_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_KEYS = ("schema_version", "kind", "family", "model", "revision")


class LilManifestError(ValueError):
    """A catalog entry that the lil launcher would also reject."""


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """``deepMerge``: mappings merge recursively; everything else replaces, ``None`` included."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@lru_cache(maxsize=1)
def _bases() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = yaml.safe_load(BASES_PATH.read_text())
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise LilManifestError("vendored lil bases have an unexpected schema_version")
    return dict(document.get("defaults") or {}), dict(document.get("families") or {})


def bases_digest() -> str:
    return hashlib.sha256(BASES_PATH.read_bytes()).hexdigest()


def _resolve_family(families: dict[str, dict[str, Any]], name: str, visiting: tuple[str, ...] = ()) -> dict[str, Any]:
    """``resolveFamily``: a family may name a parent family."""
    if name in visiting:
        raise LilManifestError("family inheritance cycle: %s" % " -> ".join((*visiting, name)))
    raw = families.get(name)
    if raw is None:
        raise LilManifestError("unknown model family %r" % name)
    resolved: dict[str, Any] = {}
    parent = raw.get("family")
    if parent is not None:
        if not isinstance(parent, str) or not parent:
            raise LilManifestError("family %r has an invalid parent" % name)
        resolved = _resolve_family(families, parent, (*visiting, name))
    overlay = {k: v for k, v in raw.items() if k != "family"}
    return deep_merge(resolved, overlay)


@dataclass(frozen=True)
class LilEntry:
    """One ``kind: model`` catalog entry, layered and ready to translate."""

    name: str
    model: str
    revision: str | None
    family: str | None
    document: dict[str, Any]
    catalog_commit: str | None = None

    def section(self, key: str) -> dict[str, Any]:
        value = self.document.get(key)
        return value if isinstance(value, dict) else {}


def read_manifest(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, dict):
        raise LilManifestError("%s is not a YAML mapping" % path)
    return document


def is_model_manifest(document: Any) -> bool:
    return isinstance(document, dict) and document.get("schema_version") == SCHEMA_VERSION and document.get("kind") == "model"


def load_entry(path: Path, *, catalog_commit: str | None = None) -> LilEntry:
    """Parse and layer one entry. Rejects what ``LoadModelEntry`` rejects that matters here."""
    path = Path(path)
    document = read_manifest(path)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise LilManifestError("%s: schema_version must be %d" % (path, SCHEMA_VERSION))
    if document.get("kind") != "model":
        raise LilManifestError("%s: kind must be model (drafts are not launchable)" % path)
    model = document.get("model")
    if not isinstance(model, str) or not _HF_MODEL_ID.match(model):
        raise LilManifestError("%s: model must name the repository in owner/name form" % path)
    revision = document.get("revision") or None
    if revision is not None and (not isinstance(revision, str) or not _COMMIT.match(revision)):
        raise LilManifestError("%s: revision must be a 40-character commit SHA" % path)

    defaults, families = _bases()
    family = document.get("family")
    resolved = _resolve_family(families, family) if family else {}
    overlay = {k: v for k, v in document.items() if k not in _IDENTITY_KEYS}
    layered = deep_merge(deep_merge(defaults, resolved), overlay)

    serving = layered.get("serving")
    if not isinstance(serving, dict) or not isinstance(serving.get("served_model_name"), str) or not serving["served_model_name"]:
        raise LilManifestError("%s: serving.served_model_name must be a non-empty string" % path)
    return LilEntry(
        name=path.parent.name,
        model=model,
        revision=revision,
        family=family or None,
        document=layered,
        catalog_commit=catalog_commit,
    )


def read_git_head(checkout: Path | None) -> str | None:
    """The commit a git checkout is at, read from ``.git`` without running git. ``None`` if unknown."""
    if checkout is None:
        return None
    git = Path(checkout) / ".git"
    try:
        if git.is_file():  # a worktree or submodule pointer
            pointer = git.read_text().strip()
            if not pointer.startswith("gitdir:"):
                return None
            git = (Path(checkout) / pointer.split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if _COMMIT.match(head):
            return head
        if not head.startswith("ref:"):
            return None
        ref = head.split(":", 1)[1].strip()
        loose = git / ref
        if loose.is_file():
            value = loose.read_text().strip()
            return value if _COMMIT.match(value) else None
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref and _COMMIT.match(parts[0]):
                    return parts[0]
    except OSError:
        return None
    return None
