"""Generic mod resolution and pre_exec injection.

A "mod" is a directory containing a ``run.sh`` script (plus any
supporting files such as patches or templates) that gets copied into a
container and executed before the serve command. Mods are a
builder/runtime-agnostic mechanism for small, named container tweaks.

For each ``mods:`` entry on a recipe, the resolver searches in this
order and uses the first hit:

1. **Explicit scoped reference** — ``@registry/<rel>`` resolves under
   that registry's configured ``mods_subpath``.
2. **Adjacent to the recipe file** — tries ``<rel>`` and ``mods/<rel>``
   relative to ``dirname(recipe.source_path)``.
3. **Same registry as the recipe** — uses ``recipe.source_registry``
   plus that registry's ``mods_subpath``.
4. **Eugr fallback** — falls back to the registry literally named
   ``eugr`` (the original source of truth for mods).

A leading ``mods/`` prefix is stripped during normalization, so
``fix-foo`` and ``mods/fix-foo`` resolve identically.

Each resolved mod produces two ``pre_exec`` entries on the recipe (the
same shape the eugr builder used historically):

* a ``copy`` dict that the hooks system materializes via ``docker cp``
  (optionally with ``source_host`` for delegated-mode head-node sources),
* an exec string that runs ``./run.sh`` inside the container with
  ``WORKSPACE_DIR=$PWD``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.orchestration.ssh import run_rsync
from sparkrun.utils import parse_scoped_name

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.registry import RegistryManager

logger = logging.getLogger(__name__)

EUGR_FALLBACK_REGISTRY = "eugr"
_CONTAINER_MODS_BASE = "/workspace/mods"
_HEAD_STAGING_BASE = "~/.cache/sparkrun/mods-staging"


class ModNotFoundError(Exception):
    """Raised when a mod reference cannot be resolved to a source path.

    Attributes:
        name: The original (un-normalized) mod reference.
        tried: Filesystem paths the resolver attempted.
    """

    def __init__(self, name: str, tried: list[str]) -> None:
        self.name = name
        self.tried = tried
        tried_lines = "\n  ".join(tried) if tried else "(no candidate paths)"
        super().__init__("Could not resolve mod %r. Tried:\n  %s" % (name, tried_lines))


@dataclass
class ResolvedMod:
    """A mod reference resolved to a concrete source location.

    Attributes:
        name: Basename used as the in-container directory under
            ``/workspace/mods/<name>``.
        source_path: Absolute path on the source host.
        source_host: Hostname owning ``source_path``; ``None`` means the
            local control machine.
    """

    name: str
    source_path: str
    source_host: str | None


def _normalize_ref(ref: str) -> str:
    """Strip a single leading ``mods/`` from a mod reference."""
    return ref.removeprefix("mods/")


def _mod_dir_if_valid(path: Path) -> Path | None:
    """Return *path* if it looks like a mod directory (has a ``run.sh``)."""
    if path.is_dir() and (path / "run.sh").exists():
        return path
    return None


def _resolve_scoped(
    ref: str,
    registry_mgr: RegistryManager,
    tried: list[str],
) -> Path | None:
    """Resolve an ``@registry/...`` reference to a control-machine path."""
    registry_name, rel = parse_scoped_name(ref)
    if registry_name is None:
        return None
    try:
        entry = registry_mgr.get_registry(registry_name)
    except Exception:
        tried.append("@%s (registry not found)" % registry_name)
        return None
    if not entry.mods_subpath:
        tried.append("@%s (no mods_subpath configured)" % registry_name)
        return None
    cache_dir = registry_mgr._cache_dir(entry.name)
    rel_clean = _normalize_ref(rel)
    candidate = cache_dir / entry.mods_subpath / rel_clean
    tried.append(str(candidate))
    return _mod_dir_if_valid(candidate)


def _resolve_adjacent(
    ref: str,
    recipe_source_path: str | None,
    tried: list[str],
) -> Path | None:
    """Resolve a mod adjacent to the recipe file on the control machine."""
    if not recipe_source_path:
        return None
    base = Path(recipe_source_path).parent
    rel_clean = _normalize_ref(ref)
    for candidate in (base / rel_clean, base / "mods" / rel_clean):
        tried.append(str(candidate))
        hit = _mod_dir_if_valid(candidate)
        if hit is not None:
            return hit
    return None


def _resolve_in_registry(
    ref: str,
    registry_name: str | None,
    registry_mgr: RegistryManager,
    tried: list[str],
    *,
    default_subpath: str | None = None,
    sync_if_missing: bool = False,
) -> Path | None:
    """Resolve a mod inside a specific registry's ``mods_subpath``.

    Args:
        default_subpath: Used when the registry entry has no
            ``mods_subpath`` configured. Lets the eugr fallback work for
            legacy ``registries.yaml`` files that predate the field.
        sync_if_missing: If True and the candidate path doesn't exist,
            trigger a registry clone/pull and re-check. Used for the
            eugr fallback so users don't have to manually sync.
    """
    if not registry_name:
        return None
    try:
        entry = registry_mgr.get_registry(registry_name)
    except Exception:
        tried.append("@%s (registry not registered)" % registry_name)
        return None
    mods_subpath = entry.mods_subpath or default_subpath
    if not mods_subpath:
        tried.append("@%s (no mods_subpath configured)" % registry_name)
        return None
    cache_dir = registry_mgr._cache_dir(entry.name)
    rel_clean = _normalize_ref(ref)
    candidate = cache_dir / mods_subpath / rel_clean
    tried.append(str(candidate))
    hit = _mod_dir_if_valid(candidate)
    if hit is not None:
        return hit
    if sync_if_missing:
        # Pass an entry with mods_subpath populated so the sparse-checkout
        # actually fetches the mods directory — legacy registries.yaml
        # entries without mods_subpath would otherwise pull only the
        # recipe subpath and the mod files would stay absent.
        if not entry.mods_subpath:
            from dataclasses import replace

            sync_entry = replace(entry, mods_subpath=mods_subpath)
        else:
            sync_entry = entry
        try:
            registry_mgr._clone_or_pull(sync_entry)
        except Exception as exc:
            tried.append("@%s sync failed: %s" % (registry_name, exc))
            return None
        hit = _mod_dir_if_valid(candidate)
        if hit is not None:
            return hit
        tried.append("@%s sync completed but mod still missing at %s" % (registry_name, candidate))
    return None


def _build_pre_exec_entries(resolved: ResolvedMod) -> list:
    """Return the (copy, exec) pre_exec pair for a resolved mod."""
    dest = "%s/%s" % (_CONTAINER_MODS_BASE, resolved.name)
    copy_entry: dict[str, str] = {"copy": resolved.source_path, "dest": dest}
    if resolved.source_host is not None:
        copy_entry["source_host"] = resolved.source_host
    exec_entry = "export WORKSPACE_DIR=$PWD && cd %s && chmod +x run.sh && ./run.sh" % dest
    return [copy_entry, exec_entry]


def _resolve_local(
    ref: str,
    recipe: Recipe,
    registry_mgr: RegistryManager,
) -> ResolvedMod:
    """Resolve a mod reference to a control-machine path (no SSH).

    Raises ``ModNotFoundError`` on miss. Caller is responsible for
    transferring the result to a remote host if required.
    """
    tried: list[str] = []

    hit = _resolve_scoped(ref, registry_mgr, tried)
    if hit is None:
        hit = _resolve_adjacent(ref, recipe.source_path, tried)
    if hit is None and parse_scoped_name(ref)[0] is None:
        # rule 3: same registry as the recipe (skipped for explicit @scoped refs)
        hit = _resolve_in_registry(ref, recipe.source_registry, registry_mgr, tried)
    if hit is None and parse_scoped_name(ref)[0] is None:
        # rule 4: eugr fallback (also skipped for explicit @scoped refs).
        # Use "mods" as the default subpath even if the local registries.yaml
        # predates the mods_subpath field, and sync the registry on demand
        # so users don't have to pre-clone eugr just for mods.
        hit = _resolve_in_registry(
            ref,
            EUGR_FALLBACK_REGISTRY,
            registry_mgr,
            tried,
            default_subpath="mods",
            sync_if_missing=True,
        )
    if hit is None:
        raise ModNotFoundError(ref, tried)

    return ResolvedMod(name=hit.name, source_path=str(hit.resolve()), source_host=None)


def _rsync_to_head(
    local_path: str,
    name: str,
    head: str,
    ssh_kwargs: dict | None,
    dry_run: bool,
) -> str:
    """Push a local mod directory to a deterministic staging dir on *head*.

    Returns the head-side absolute path (with ``~`` left for shell to expand).
    """
    dest = "%s/%s" % (_HEAD_STAGING_BASE, name)
    logger.info("Staging mod %r to %s:%s", name, head, dest)
    kw = ssh_kwargs or {}
    if dry_run:
        return dest
    result = run_rsync(
        local_path,
        head,
        dest,
        ssh_user=kw.get("ssh_user"),
        ssh_key=kw.get("ssh_key"),
        ssh_options=kw.get("ssh_options"),
        timeout=120,
    )
    if not result.success:
        raise RuntimeError("Failed to stage mod %r to %s: %s" % (name, head, result.stderr.strip() if result.stderr else "(no output)"))
    return dest


def _resolve_delegated(
    ref: str,
    recipe: Recipe,
    registry_mgr: RegistryManager,
    head: str,
    ssh_kwargs: dict | None,
    dry_run: bool,
    ensured_registries: dict[str, str],
) -> ResolvedMod:
    """Resolve a mod for delegated mode: source files live on *head*.

    Registry-backed mods are materialized via
    ``RegistryManager.ensure_registry_on_host``; adjacent recipes' mods
    are rsynced to a staging directory on the head node.
    """

    def _ensure_registry(name: str) -> str:
        path = ensured_registries.get(name)
        if path is None:
            path = registry_mgr.ensure_registry_on_host(name, head, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
            ensured_registries[name] = path
        return path

    # rule 1: explicit @registry/...
    registry_name, rel = parse_scoped_name(ref)
    if registry_name is not None:
        try:
            entry = registry_mgr.get_registry(registry_name)
        except Exception as exc:
            raise ModNotFoundError(ref, ["@%s (registry not found)" % registry_name]) from exc
        if not entry.mods_subpath:
            raise ModNotFoundError(ref, ["@%s (no mods_subpath configured)" % registry_name])
        remote_root = _ensure_registry(registry_name)
        rel_clean = _normalize_ref(rel)
        remote_path = "%s/%s/%s" % (remote_root, entry.mods_subpath, rel_clean)
        name = Path(rel_clean).name
        return ResolvedMod(name=name, source_path=remote_path, source_host=head)

    # rule 2: adjacent to recipe (local) -> rsync to head
    tried: list[str] = []
    local_hit = _resolve_adjacent(ref, recipe.source_path, tried)
    if local_hit is not None:
        staged = _rsync_to_head(str(local_hit.resolve()), local_hit.name, head, ssh_kwargs, dry_run)
        return ResolvedMod(name=local_hit.name, source_path=staged, source_host=head)

    # rule 3: same registry as recipe
    if recipe.source_registry:
        try:
            entry = registry_mgr.get_registry(recipe.source_registry)
        except Exception:
            entry = None
        if entry is not None and entry.mods_subpath:
            remote_root = _ensure_registry(recipe.source_registry)
            rel_clean = _normalize_ref(ref)
            remote_path = "%s/%s/%s" % (remote_root, entry.mods_subpath, rel_clean)
            tried.append(remote_path + " (on %s)" % head)
            # Trust delegated existence — if missing the docker cp will fail
            # with a clear error; we cannot easily stat over SSH per-mod
            # without paying a round-trip per ref. Verify only at the
            # registry level (ensure_registry_on_host succeeded).
            return ResolvedMod(name=Path(rel_clean).name, source_path=remote_path, source_host=head)

    # rule 4: eugr fallback — default subpath to "mods" so legacy
    # registries.yaml files (no mods_subpath field) still work.
    try:
        eugr_entry = registry_mgr.get_registry(EUGR_FALLBACK_REGISTRY)
    except Exception:
        eugr_entry = None
        tried.append("@%s (registry not registered)" % EUGR_FALLBACK_REGISTRY)
    if eugr_entry is not None:
        eugr_subpath = eugr_entry.mods_subpath or "mods"
        remote_root = _ensure_registry(EUGR_FALLBACK_REGISTRY)
        rel_clean = _normalize_ref(ref)
        remote_path = "%s/%s/%s" % (remote_root, eugr_subpath, rel_clean)
        tried.append(remote_path + " (on %s)" % head)
        return ResolvedMod(name=Path(rel_clean).name, source_path=remote_path, source_host=head)

    raise ModNotFoundError(ref, tried)


def resolve_and_inject_mods(
    recipe: Recipe,
    registry_mgr: RegistryManager,
    *,
    config: SparkrunConfig | None = None,
    transfer_mode: str = "local",
    head: str | None = None,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> None:
    """Resolve ``recipe.mods`` and append the matching ``pre_exec`` entries.

    Idempotent: sets ``recipe._mods_resolved = True`` after a successful
    pass; subsequent invocations are no-ops. ``recipe.mods`` is preserved
    in place so ``Recipe.to_dict()`` keeps emitting the original ``mods:``
    field, while mod-derived ``pre_exec`` entries are reset from raw input
    on export (the same mechanism that strips builder-added entries).

    Args:
        recipe: Recipe to mutate (``pre_exec`` is extended in place).
        registry_mgr: Used to resolve registry-backed mods.
        config: Optional sparkrun config (reserved for future use).
        transfer_mode: ``"local"`` or ``"delegated"`` — controls whether
            sources land on the control machine or the head node.
        head: Head host for delegated mode (required when delegated).
        ssh_kwargs: SSH options for delegated mode.
        dry_run: Skip side effects (no rsync, no git clone) and return
            the would-be paths.

    Raises:
        ModNotFoundError: A reference could not be resolved.
        ValueError: ``transfer_mode="delegated"`` without ``head``.
    """
    del config  # unused for now; kept for API symmetry with builders
    if not recipe.mods:
        return
    if getattr(recipe, "_mods_resolved", False):
        return

    delegated = transfer_mode == "delegated"
    if delegated and not head:
        raise ValueError("resolve_and_inject_mods: head is required when transfer_mode='delegated'")

    ensured_registries: dict[str, str] = {}
    resolved: list[ResolvedMod] = []
    for ref in recipe.mods:
        if delegated:
            assert head is not None  # narrowed by the guard above
            resolved.append(
                _resolve_delegated(
                    ref,
                    recipe,
                    registry_mgr,
                    head=head,
                    ssh_kwargs=ssh_kwargs,
                    dry_run=dry_run,
                    ensured_registries=ensured_registries,
                )
            )
        else:
            resolved.append(_resolve_local(ref, recipe, registry_mgr))

    for r in resolved:
        recipe.pre_exec.extend(_build_pre_exec_entries(r))

    logger.info("Resolved %d mod(s) into pre_exec entries", len(resolved))
    recipe._mods_resolved = True
