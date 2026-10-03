"""Persistent runtime cache for compilation and autotune artifacts.

Inference runtimes compile a great deal on first start — torch.compile /
inductor graphs, Triton cubins, FlashInfer JIT modules, TRT-LLM autotuner
tactics — and containers are ``--rm``, so all of it is discarded on exit and
recomputed on the next launch.  This module resolves a *host* directory that is
mounted at :data:`RUNTIME_CACHE_CONTAINER_PATH` so those artifacts survive.

Design notes live in ``.slop/runtime-cache-design.md``.  Two properties matter
enough to restate here:

**Directory keying is hygiene, never correctness.**  ``key_by_image`` defaults
*off*, so nothing may depend on the directory key to avoid loading a stale
artifact.  torch.compile / inductor / Triton / FlashInfer are content-addressed
internally and are correct in a shared tree — that is what lets image-keying
default off.  TRT-LLM's autotuner is the exception: a single file that records
its version and GPU but checks neither on load, so it carries the recipe
fingerprint in its *filename* (see :class:`CachePath` with ``file=True``)
regardless of the directory key.  A runtime whose cache genuinely cannot be
shared declares ``key_by_image: True`` from
:meth:`~sparkrun.runtimes.base.RuntimePlugin.runtime_cache_defaults` rather
than relying on the global default.

**The container path is constant.**  All keying is host-side, so recipes, env
blocks and serve commands never spell a key.  Turning a key knob changes the
host directory only.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)

#: Container-side mount point.  Sibling of ``/cache/huggingface`` (see
#: :func:`sparkrun.orchestration.primitives.build_volumes`), same convention.
RUNTIME_CACHE_CONTAINER_PATH = "/cache/runtime"

#: Host-side directory name under the sparkrun cache root.
RUNTIME_CACHE_SUBDIR = "runtime-cache"

#: Marker file touched in the leaf directory on every launch.  Pruning ages
#: trees by *this*, not by directory mtime: reading a cache does not update a
#: directory's mtime, so a warm tree used daily but never written would look
#: idle and be deleted.
LAST_USED_MARKER = ".sparkrun-last-used"

#: Environment variable used as the catch-all.  Libraries that honor XDG (and
#: many do) land inside the mount without sparkrun enumerating them.
XDG_CACHE_ENV = "XDG_CACHE_HOME"

DEFAULT_PRUNE_MAX_AGE_DAYS = 30

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_COLLAPSE = re.compile(r"-{2,}")

#: Cap on a single sanitized path component.  Model ids and image tags can be
#: long; the short hash carries the identity, so the readable half is only for
#: a human scanning the directory.
_MAX_LABEL_LEN = 48


@dataclass(frozen=True)
class CachePath:
    """A path a runtime wants persisted, relative to the mount point.

    Args:
        subpath: Path relative to :data:`RUNTIME_CACHE_CONTAINER_PATH`.
        file: When True the value names a *file* — only its parent directory is
            created.  TRT-LLM's ``TLLM_AUTOTUNER_CACHE_PATH`` is the motivating
            case; everything else is a directory.
    """

    subpath: str
    file: bool = False


@dataclass(frozen=True)
class RuntimeCacheSettings:
    """Resolved knobs for the runtime cache.

    Defaults are the shipped baseline: on, model-keyed, not image-keyed,
    pruning enabled.  See :func:`resolve_runtime_cache_settings` for the
    precedence chain that produces this.
    """

    enabled: bool = True
    dir: str | None = None
    key_by_image: bool = False
    key_by_model: bool = True
    prune_enabled: bool = True
    prune_max_age_days: int = DEFAULT_PRUNE_MAX_AGE_DAYS


@dataclass(frozen=True)
class RuntimeCacheMounts:
    """Everything a launch needs to mount, populate and sweep the cache.

    ``volumes`` and ``env`` are merged into the container launch by the runtime
    (solo path) or :class:`~sparkrun.runtimes._cluster_ops.ClusterContext`
    (cluster path).  ``dirs`` and the prune fields drive
    :meth:`~sparkrun.orchestration.executors._base.Executor.ensure_runtime_cache`.
    """

    leaf: str
    """Absolute host path mounted at :data:`RUNTIME_CACHE_CONTAINER_PATH`."""

    family_root: str
    """Absolute host path whose children the prune sweep ages.

    The tree *above* the key components — pruning compares sibling keyed trees
    against each other, never unrelated runtimes.
    """

    volumes: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    dirs: list[str] = field(default_factory=list)
    """Absolute host directories to create before the container starts.

    Docker creates a missing ``-v`` source **root-owned**, which breaks
    rootless docker and breaks the ``local`` executor outright, so these are
    created explicitly rather than left to the daemon.
    """

    prune_enabled: bool = True
    prune_max_age_days: int = DEFAULT_PRUNE_MAX_AGE_DAYS


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def _short_hash(text: str, length: int = 8) -> str:
    """Stable short digest of *text* (identity half of a key component)."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:length]


def sanitize_key_component(text: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """Reduce *text* to a readable, filesystem- and shell-safe label.

    ``/`` becomes ``__`` so an org-qualified model id stays legible as one
    component; everything outside ``[A-Za-z0-9._-]`` collapses to ``-``.  The
    result is only the *readable* half of a key — :func:`model_key` and
    :func:`image_key` append a hash so truncation and collapsing can never
    collide two distinct inputs.
    """
    cleaned = text.strip().replace("/", "__")
    cleaned = _UNSAFE_CHARS.sub("-", cleaned)
    cleaned = _COLLAPSE.sub("-", cleaned).strip("-._")
    cleaned = cleaned[:max_len].strip("-._")
    return cleaned or "unknown"


def model_key(recipe: "Recipe") -> str:
    """Key component identifying the recipe's model.

    Folds in ``model_revision`` and any GGUF ``repo:quant`` suffix, since both
    change the compiled artifacts.  An absolute-path model hashes the path.
    """
    model = str(getattr(recipe, "model", "") or "")
    revision = str(getattr(recipe, "model_revision", "") or "")
    identity = "%s@%s" % (model, revision)
    label = sanitize_key_component(model.rstrip("/").rsplit("/", 1)[-1] if model.startswith("/") else model)
    return "%s-%s" % (label, _short_hash(identity))


def image_key(image: str, identity: str | None = None) -> str:
    """Key component identifying the container image.

    *identity* is the image ID / repo digest collected during container
    distribution (:func:`sparkrun.containers.distribute.get_image_identity`).
    It is preferred because a mutable tag like ``:latest`` is the same string
    across two genuinely different images.  When it is unavailable — a dry run,
    or an image not yet pulled — the ref string is hashed instead, which still
    separates distinct tags.
    """
    image = str(image or "")
    tag = image.rsplit("/", 1)[-1] if image else ""
    return "%s-%s" % (sanitize_key_component(tag), _short_hash(identity or image))


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------

_BOOL_FIELDS = ("enabled", "key_by_image", "key_by_model")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def _apply_layer(settings: RuntimeCacheSettings, layer: dict[str, Any] | None) -> RuntimeCacheSettings:
    """Fold one configuration *layer* onto *settings* (layer wins where set)."""
    if not layer:
        return settings

    updates: dict[str, Any] = {}
    for key in _BOOL_FIELDS:
        if key in layer:
            coerced = _coerce_bool(layer[key])
            if coerced is None:
                logger.warning("runtime_cache.%s: expected a boolean, got %r — ignoring", key, layer[key])
            else:
                updates[key] = coerced

    if "dir" in layer and layer["dir"]:
        updates["dir"] = str(layer["dir"])

    prune = layer.get("prune")
    if isinstance(prune, dict):
        if "enabled" in prune:
            coerced = _coerce_bool(prune["enabled"])
            if coerced is None:
                logger.warning("runtime_cache.prune.enabled: expected a boolean, got %r — ignoring", prune["enabled"])
            else:
                updates["prune_enabled"] = coerced
        if "max_age_days" in prune:
            try:
                updates["prune_max_age_days"] = max(0, int(prune["max_age_days"]))
            except (TypeError, ValueError):
                logger.warning("runtime_cache.prune.max_age_days: expected an int, got %r — ignoring", prune["max_age_days"])
    elif isinstance(prune, bool):
        # ``prune: false`` shorthand.
        updates["prune_enabled"] = prune

    return replace(settings, **updates) if updates else settings


def resolve_runtime_cache_settings(
    *,
    runtime: "RuntimePlugin | None" = None,
    config: "SparkrunConfig | None" = None,
    cluster: "ClusterDefinition | None" = None,
    recipe: "Recipe | None" = None,
    cli_override: dict[str, Any] | None = None,
    env_disabled: bool = False,
) -> RuntimeCacheSettings:
    """Resolve runtime-cache settings across the layered chain.

    Precedence, highest first::

        recipe `runtime_cache:` → CLI → cluster → config `runtime_cache:`
            → RuntimePlugin.runtime_cache_defaults() → baseline

    The runtime-plugin layer sits where ``runtime.default_executor()`` sits in
    :func:`sparkrun.orchestration.executor.resolve_executor`'s chain, and is
    what makes ``key_by_image: False`` a safe *global* default: ``trtllm``
    declares ``key_by_image: True`` for itself, so the one known-unvalidated
    cache is protected without every TRT-LLM recipe repeating it.

    *env_disabled* reflects ``SPARKRUN_NO_RUNTIME_CACHE`` and beats every
    layer — it is a kill switch, not a preference.
    """
    settings = RuntimeCacheSettings()

    if runtime is not None:
        try:
            settings = _apply_layer(settings, runtime.runtime_cache_defaults())
        except Exception:  # pragma: no cover — a broken plugin must not break launch
            logger.debug("runtime_cache: runtime defaults unavailable", exc_info=True)

    if config is not None:
        settings = _apply_layer(settings, getattr(config, "runtime_cache", None))

    if cluster is not None:
        settings = _apply_layer(settings, getattr(cluster, "runtime_cache", None))

    settings = _apply_layer(settings, cli_override)

    if recipe is not None:
        settings = _apply_layer(settings, getattr(recipe, "runtime_cache", None))

    if env_disabled:
        settings = replace(settings, enabled=False)

    return settings


def runtime_cache_disabled_by_env() -> bool:
    """True when ``SPARKRUN_NO_RUNTIME_CACHE`` is set to a truthy value.

    Mirrors ``SPARKRUN_NO_IMAGE_PROBE`` / ``SPARKRUN_NO_SESSION_GUARD``.
    """
    import os

    raw = os.environ.get("SPARKRUN_NO_RUNTIME_CACHE", "")
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Mount construction
# ---------------------------------------------------------------------------


def resolve_runtime_cache_root(settings: RuntimeCacheSettings, sparkrun_cache_dir: str) -> str:
    """Absolute host root for the runtime cache.

    *sparkrun_cache_dir* is the target host's sparkrun cache directory (see
    :func:`sparkrun.core.launcher.resolve_effective_runtime_cache_dir`, which
    probes it remotely).  An explicit ``dir`` setting wins outright.
    """
    if settings.dir:
        return settings.dir.rstrip("/")
    return "%s/%s" % (sparkrun_cache_dir.rstrip("/"), RUNTIME_CACHE_SUBDIR)


def probe_image_identity(
    image: str,
    hosts: list[str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> str | None:
    """Resolve *image*'s content identity (image ID / repo digest) on the head host.

    Only meaningful when ``key_by_image`` is on, and only *called* then — this
    is an SSH round-trip, and the shared-tree default has no use for it.

    Hashing the image *ref* is not a substitute where it matters most: a
    mutable tag like ``:latest`` is the same string before and after a re-pull,
    so a ref-keyed directory would hand the new image the old image's cache —
    precisely the staleness issue #256 reports for the TRT-LLM autotuner, and
    the reason ``trtllm`` turns image-keying on.  A pinned tag is unambiguous
    either way.

    Returns ``None`` when the identity cannot be established (dry run, image
    not yet pulled, unreachable host); :func:`image_key` then falls back to the
    ref, which still separates distinct tags.
    """
    if dry_run or not image or not hosts:
        return None

    try:
        from sparkrun.containers.distribute import _check_remote_image_identities

        ssh_kwargs = ssh_kwargs or {}
        identities = _check_remote_image_identities(
            image,
            hosts[:1],  # the image is identical everywhere by the time we ask
            ssh_user=ssh_kwargs.get("ssh_user"),
            ssh_key=ssh_kwargs.get("ssh_key"),
            ssh_options=ssh_kwargs.get("ssh_options"),
        )
    except Exception:
        logger.debug("runtime_cache: image identity probe failed", exc_info=True)
        return None

    for identity in identities.values():
        # Prefer the local image ID; a repo digest is the fallback for an image
        # pulled but not built locally.
        #
        # Deliberately *not* switched to ``identity.content_sig``: that would be
        # a more driver-agnostic cache key, but it silently orphans every
        # runtime cache already keyed by image ID.  Cache-key semantics are a
        # separate change from the distribution staleness this fixes.
        return identity.image_id or (identity.repo_digests[0] if identity.repo_digests else None)
    return None


def build_runtime_cache_mounts(
    *,
    runtime: "RuntimePlugin",
    recipe: "Recipe | None",
    settings: RuntimeCacheSettings,
    root: str,
    image: str = "",
    image_identity: str | None = None,
    fingerprint: str = "",
) -> RuntimeCacheMounts | None:
    """Build the mount/env/mkdir plan, or ``None`` when the cache is disabled.

    ``None`` means "behave exactly as before this feature existed" — no mount,
    no env, no host mutation.
    """
    if not settings.enabled:
        return None

    family = sanitize_key_component(runtime.get_family() or runtime.runtime_name or "runtime")
    family_root = "%s/%s" % (root.rstrip("/"), family)

    components: list[str] = []
    if settings.key_by_image:
        components.append(image_key(image, image_identity))
    if settings.key_by_model and recipe is not None:
        components.append(model_key(recipe))

    leaf = "/".join([family_root, *components]) if components else family_root

    try:
        declared = runtime.runtime_cache_paths(fingerprint=fingerprint) or {}
    except Exception:  # pragma: no cover — a broken plugin degrades to XDG only
        logger.debug("runtime_cache: runtime paths unavailable", exc_info=True)
        declared = {}

    env: dict[str, str] = {XDG_CACHE_ENV: RUNTIME_CACHE_CONTAINER_PATH}
    dirs: list[str] = [leaf]

    for var, spec in sorted(declared.items()):
        subpath = spec.subpath.strip("/") if isinstance(spec, CachePath) else str(spec).strip("/")
        is_file = bool(getattr(spec, "file", False))
        if not subpath:
            continue
        env[var] = "%s/%s" % (RUNTIME_CACHE_CONTAINER_PATH, subpath)
        host_path = "%s/%s" % (leaf, subpath)
        # A file's *parent* is what must exist; the runtime writes the file.
        dirs.append(host_path.rsplit("/", 1)[0] if is_file else host_path)

    # Stable order, no duplicates — this list drives a generated shell script.
    unique_dirs = sorted(set(dirs))

    return RuntimeCacheMounts(
        leaf=leaf,
        family_root=family_root,
        volumes={leaf: RUNTIME_CACHE_CONTAINER_PATH},
        env=env,
        dirs=unique_dirs,
        prune_enabled=settings.prune_enabled,
        prune_max_age_days=settings.prune_max_age_days,
    )
