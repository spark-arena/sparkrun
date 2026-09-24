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
    resolved_root = Path(root).resolve()
    for path in sorted(Path(root).glob("*/%s" % MANIFEST_NAME)):
        if not path.resolve().is_relative_to(resolved_root):
            logger.warning("lil: ignoring %s, which resolves outside the catalog", path)
            continue
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


def _facts_memo_path(model: str, revision: str | None) -> Path | None:
    try:
        from sparkrun.core.config import SparkrunConfig

        root = Path(SparkrunConfig().cache_dir) / "lil" / "facts"
    except Exception as error:
        logger.debug("lil: no cache dir for checkpoint facts: %s", error)
        return None
    import hashlib

    return root / ("%s.json" % hashlib.sha256(("%s@%s" % (model, revision or "")).encode()).hexdigest()[:24])


def _read_memo(path: Path | None):
    from .translate import CheckpointFacts

    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        return CheckpointFacts(config=data["config"], weight_bytes=int(data["weight_bytes"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.debug("lil: ignoring unreadable facts memo %s: %s", path, error)
        return None


def _write_memo(path: Path | None, facts) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"config": facts.config, "weight_bytes": facts.weight_bytes}))
        tmp.replace(path)
    except OSError as error:
        logger.debug("lil: could not record checkpoint facts at %s: %s", path, error)


def _local_weight_bytes(model: str, revision: str | None, cache_dir: str | None) -> int | None:
    """Stored size of every shard, from a *complete* local snapshot. Same quantity the Hub reports."""
    from sparkrun.models.vram import cached_hub_file

    config = cached_hub_file(model, "config.json", revision, cache_dir)
    if config is None:
        return None
    snapshot = Path(config).parent
    index = snapshot / "model.safetensors.index.json"
    try:
        if index.is_file():
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        else:
            shards = {p.name for p in snapshot.glob("*.safetensors")}
        paths = [snapshot / shard for shard in shards]
        if not paths or not all(p.is_file() for p in paths):
            return None  # a partial download must not be sized as the whole model
        return sum(p.stat().st_size for p in paths)
    except (OSError, ValueError) as error:
        logger.debug("lil: could not size the local snapshot of %s: %s", model, error)
        return None


def gather_facts(model: str, revision: str | None, *, offline: bool):
    """Checkpoint facts, cache first. ``offline`` never touches the network.

    Two properties matter because the facts decide ``tensor_parallel``, which
    is workload identity (``stop`` / ``logs`` / ``--ensure`` recompute it):

    * **One quantity from every source**: the stored size of the weight
      shards, whether read from a complete local snapshot or reported by the
      Hub. (The index's ``total_size`` counts tensor bytes only, and switching
      between the two after a download could flip the ``fit`` TP.)
    * **Recorded once known.** The first complete answer is memoised under
      sparkrun's cache, so later loads agree with the launch even when the Hub
      is unreachable or the budget has tripped.
    """
    from sparkrun.models.vram import fetch_model_config, fetch_safetensors_size

    from .translate import CheckpointFacts

    memo = _facts_memo_path(model, revision)
    known = _read_memo(memo)
    if known is not None:
        return known
    cache_dir = _hf_cache_dir()
    config = fetch_model_config(model, revision=revision, cache_dir=cache_dir, local_only=offline)
    weight_bytes = _local_weight_bytes(model, revision, cache_dir)
    if weight_bytes is None and not offline:
        weight_bytes = fetch_safetensors_size(model, revision=revision, cache_dir=cache_dir)
    facts = CheckpointFacts(config=config, weight_bytes=weight_bytes)
    if config is not None and weight_bytes:
        _write_memo(memo, facts)
    return facts


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
