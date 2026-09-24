"""lil catalog compatibility: serve ``@lil/<Entry>`` recipes from lil manifests.

The `lil launcher <https://github.com/local-inference-lab/lil>`_ reads launch
manifests from the ``local-inference-lab/lil-catalog`` repository on Hugging
Face (one ``<Entry>/lil.yaml`` per model). This plugin declares that catalog
as the ``lil`` registry (``subpath: "/"``, ``format: lil``) and registers a
recipe format that translates each manifest into an ordinary v2 recipe for
the B12X vLLM image. ``sparkrun run @lil/<Entry>`` launches it, and
``sparkrun export recipe @lil/<Entry>`` writes it out.

Pieces:

* :mod:`.manifest`: layering over the vendored model bases (``bases.yaml``)
* :mod:`.checkpoint`: checkpoint-derived decisions, ported from the launcher
* :mod:`.translate`: manifest to recipe

Checkpoint facts (``config.json``, stored weight size) are read **cache
first** through sparkrun's shared Hub path, so a model already downloaded to
this machine needs no network. Listing never touches the network.

Gated by the ``registry.lil`` feature flag (on for the alpha channel).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sparkrun import __version__ as __version__

SPARKRUN_PLUGIN_API_VERSION = 1

FEATURE_FLAG = "registry.lil"
FORMAT_NAME = "lil"
REGISTRY_NAME = "lil"
CATALOG_URL = "https://huggingface.co/local-inference-lab/lil-catalog"
MANIFEST_NAME = "lil.yaml"

logger = logging.getLogger(__name__)


def iter_manifests(root: Path) -> list[Path]:
    """Launchable entries under a catalog checkout. Drafts are catalog-only."""
    from .manifest import is_model_manifest, read_manifest

    found = []
    for path in sorted(Path(root).glob("*/%s" % MANIFEST_NAME)):
        try:
            if is_model_manifest(read_manifest(path)):
                found.append(path)
        except Exception as error:  # a malformed entry must not hide the rest of the catalog
            logger.debug("lil: skipping unreadable manifest %s: %s", path, error)
    return found


def entry_name(path: Path, root: Path | None) -> str:
    """The entry directory is the launch name (as in ``lil render <Entry>``)."""
    return Path(path).parent.name


def claims(path: Path, data: dict[str, Any]) -> bool:
    from .manifest import is_model_manifest

    return Path(path).name == MANIFEST_NAME and is_model_manifest(data) and isinstance(data.get("serving"), dict)


def _hf_cache_dir() -> str | None:
    try:
        from sparkrun.core.config import SparkrunConfig

        return str(SparkrunConfig().hf_cache_dir)
    except Exception as error:
        logger.debug("lil: could not resolve the HF cache dir: %s", error)
        return None


def gather_facts(model: str, revision: str | None, *, offline: bool):
    """Checkpoint facts, cache first. ``offline`` never touches the network."""
    from sparkrun.models.vram import cached_hub_file, fetch_model_config, fetch_safetensors_size

    from .translate import CheckpointFacts

    cache_dir = _hf_cache_dir()
    config = fetch_model_config(model, revision=revision, cache_dir=cache_dir, local_only=offline)
    weight_bytes = None
    index = cached_hub_file(model, "model.safetensors.index.json", revision, cache_dir)
    if index is not None:
        try:
            with open(index) as f:
                total = json.load(f).get("metadata", {}).get("total_size")
            weight_bytes = int(total) if total else None
        except (OSError, ValueError, TypeError) as error:
            logger.debug("lil: cached safetensors index for %s unreadable: %s", model, error)
    if weight_bytes is None and not offline:
        weight_bytes = fetch_safetensors_size(model, revision=revision, cache_dir=cache_dir)
    return CheckpointFacts(config=config, weight_bytes=weight_bytes)


def load(path: Path, *, registry_manager=None, offline: bool, allow_local: bool = True) -> dict[str, Any]:
    """``RecipeFormat.load``: one manifest as a v2 recipe."""
    from .manifest import load_entry, read_git_head
    from .translate import translate

    path = Path(path)
    entry = load_entry(path, catalog_commit=read_git_head(path.parent.parent))
    return translate(entry, gather_facts(entry.model, entry.revision, offline=offline), strict=not offline)


def register(v) -> None:
    from sparkrun.core.recipe_formats import RecipeFormat, register_recipe_format
    from sparkrun.core.registry import REPO_ROOT_SUBPATH, RegistryEntry
    from sparkrun.core.registry_defaults import register_default_registry

    register_recipe_format(
        RecipeFormat(name=FORMAT_NAME, owner="lil", iter_files=iter_manifests, name_of=entry_name, load=load, claims=claims)
    )
    register_default_registry(
        RegistryEntry(
            name=REGISTRY_NAME,
            url=CATALOG_URL,
            subpath=REPO_ROOT_SUBPATH,
            description="lil launch manifests (local-inference-lab/lil-catalog), translated for the B12X vLLM image",
            format=FORMAT_NAME,
        ),
        owner="lil",
    )
