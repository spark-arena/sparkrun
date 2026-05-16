"""Tests for the generic mod resolver in :mod:`sparkrun.core.mods`.

Covers resolution order, normalization, pre_exec injection shape (local
vs. delegated), Recipe.to_dict round-trip preservation, and delegated-mode
behavior (ensure_registry_on_host invocation, rsync staging).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from sparkrun.core.mods import (
    ModNotFoundError,
    ResolvedMod,
    resolve_and_inject_mods,
)
from sparkrun.core.recipe import Recipe
from sparkrun.core.registry import RegistryEntry, RegistryManager


def _write_mod(base: Path, name: str, body: str = "echo hi") -> Path:
    """Create a mod directory under *base*/<name>/ with a run.sh script."""
    mod_dir = base / name
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "run.sh").write_text("#!/bin/bash\n%s\n" % body)
    return mod_dir


def _make_registry_manager(
    tmp_path: Path,
    entries: list[RegistryEntry],
) -> RegistryManager:
    """Return a RegistryManager wired to a tmp config root + cache root."""
    config_root = tmp_path / "config"
    cache_root = tmp_path / "cache"
    config_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    mgr = RegistryManager(config_root=config_root, cache_root=cache_root)
    mgr._save_registries(entries)
    return mgr


def _populate_registry_cache(
    mgr: RegistryManager,
    entry: RegistryEntry,
    mods: dict[str, str] | None = None,
) -> Path:
    """Materialize <cache>/<name>/<mods_subpath>/<each mod>/run.sh."""
    cache_dir = mgr._cache_dir(entry.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if entry.mods_subpath and mods:
        mods_dir = cache_dir / entry.mods_subpath
        mods_dir.mkdir(parents=True, exist_ok=True)
        for name, body in mods.items():
            _write_mod(mods_dir, name, body)
    return cache_dir


def _recipe_with_mods(mods: list[str], source_path: Path | None = None) -> Recipe:
    recipe = Recipe.from_dict(
        {
            "name": "test",
            "model": "some/model",
            "runtime": "vllm-distributed",
            "container": "scitrera/dgx-spark-vllm:latest",
            "mods": mods,
        }
    )
    if source_path is not None:
        recipe.source_path = str(source_path)
    return recipe


# ---------------------------------------------------------------------------
# Local-mode resolution rules
# ---------------------------------------------------------------------------


def test_resolve_adjacent_mod(tmp_path: Path):
    """Rule 2: mod sitting next to the recipe at mods/<name>/run.sh resolves."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    recipe = _recipe_with_mods(["foo"], source_path=recipe_path)
    mgr = _make_registry_manager(tmp_path, [])

    resolve_and_inject_mods(recipe, mgr)

    # mods list is preserved on the recipe so it round-trips via to_dict();
    # idempotency is tracked separately via recipe._mods_resolved.
    assert recipe.mods == ["foo"]
    assert recipe._mods_resolved is True
    assert len(recipe.pre_exec) == 2
    copy_entry = recipe.pre_exec[0]
    assert isinstance(copy_entry, dict)
    assert copy_entry["copy"].endswith("/mods/foo")
    assert copy_entry["dest"] == "/workspace/mods/foo"
    assert "source_host" not in copy_entry
    assert "run.sh" in recipe.pre_exec[1]


def test_resolve_scoped_mod_explicit(tmp_path: Path):
    """Rule 1: @registry/foo resolves under that registry's mods_subpath."""
    entry = RegistryEntry(name="alt", url="https://example/alt.git", subpath="recipes", mods_subpath="mods")
    mgr = _make_registry_manager(tmp_path, [entry])
    _populate_registry_cache(mgr, entry, mods={"foo": "echo alt"})

    recipe = _recipe_with_mods(["@alt/foo"])
    resolve_and_inject_mods(recipe, mgr)

    assert "alt" in recipe.pre_exec[0]["copy"]
    assert recipe.pre_exec[0]["copy"].endswith("/mods/foo")


def test_resolve_same_registry(tmp_path: Path):
    """Rule 3: recipe.source_registry's mods_subpath supplies the mod."""
    entry = RegistryEntry(name="home", url="https://example/home.git", subpath="recipes", mods_subpath="mods")
    mgr = _make_registry_manager(tmp_path, [entry])
    _populate_registry_cache(mgr, entry, mods={"foo": "echo home"})

    recipe = _recipe_with_mods(["foo"])
    recipe.source_registry = "home"
    resolve_and_inject_mods(recipe, mgr)

    assert recipe.pre_exec[0]["copy"].endswith("/mods/foo")
    assert "home" in recipe.pre_exec[0]["copy"]


def test_resolve_eugr_fallback(tmp_path: Path):
    """Rule 4: missing elsewhere, look in the registry literally named 'eugr'."""
    eugr_entry = RegistryEntry(name="eugr", url="https://example/eugr.git", subpath="recipes", mods_subpath="mods")
    mgr = _make_registry_manager(tmp_path, [eugr_entry])
    _populate_registry_cache(mgr, eugr_entry, mods={"legacy": "echo legacy"})

    recipe = _recipe_with_mods(["legacy"])  # no source_registry, no adjacent
    resolve_and_inject_mods(recipe, mgr)

    assert recipe.pre_exec[0]["copy"].endswith("/mods/legacy")


def test_resolve_eugr_fallback_legacy_registry_without_mods_subpath(tmp_path: Path):
    """Legacy registries.yaml entries (no mods_subpath) still get the fallback.

    Older sparkrun installs have an eugr RegistryEntry that pre-dates the
    mods_subpath field. The eugr-fallback path must default to ``"mods"``
    rather than silently skipping.
    """
    eugr_entry = RegistryEntry(
        name="eugr",
        url="https://example/eugr.git",
        subpath="recipes",
        # NOTE: mods_subpath intentionally omitted (legacy layout)
    )
    mgr = _make_registry_manager(tmp_path, [eugr_entry])
    _populate_registry_cache(
        mgr,
        # Materialize at the conventional <cache>/eugr/mods/<name>/run.sh
        RegistryEntry(
            name="eugr",
            url=eugr_entry.url,
            subpath="recipes",
            mods_subpath="mods",
        ),
        mods={"fix-foo": "echo fix"},
    )

    recipe = _recipe_with_mods(["mods/fix-foo"])
    resolve_and_inject_mods(recipe, mgr)
    assert recipe.pre_exec[0]["copy"].endswith("/mods/fix-foo")


def test_resolve_eugr_fallback_syncs_when_cache_missing(tmp_path: Path):
    """If the eugr cache isn't populated, the resolver triggers a clone."""
    eugr_entry = RegistryEntry(
        name="eugr",
        url="https://example/eugr.git",
        subpath="recipes",
        mods_subpath="mods",
    )
    mgr = _make_registry_manager(tmp_path, [eugr_entry])
    # Do NOT pre-populate the cache; simulate _clone_or_pull creating it
    # by writing the mod into the cache when called.

    def _fake_clone_or_pull(entry):
        target = mgr._cache_dir(entry.name) / entry.mods_subpath / "synced"
        target.mkdir(parents=True, exist_ok=True)
        (target / "run.sh").write_text("#!/bin/bash\necho synced\n")
        return True

    recipe = _recipe_with_mods(["synced"])

    with mock.patch.object(RegistryManager, "_clone_or_pull", side_effect=_fake_clone_or_pull) as mock_sync:
        resolve_and_inject_mods(recipe, mgr)

    mock_sync.assert_called_once()
    assert recipe.pre_exec[0]["copy"].endswith("/mods/synced")


def test_resolve_order_priority_adjacent_beats_registry(tmp_path: Path):
    """Adjacent mods win over same-registry mods when both exist."""
    entry = RegistryEntry(name="home", url="https://example/home.git", subpath="recipes", mods_subpath="mods")
    mgr = _make_registry_manager(tmp_path, [entry])
    _populate_registry_cache(mgr, entry, mods={"foo": "echo from-registry"})

    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    adjacent_dir = tmp_path / "mods"
    _write_mod(adjacent_dir, "foo", body="echo adjacent")

    recipe = _recipe_with_mods(["foo"], source_path=recipe_path)
    recipe.source_registry = "home"
    resolve_and_inject_mods(recipe, mgr)

    # Adjacent path wins over registry path
    copy_path = recipe.pre_exec[0]["copy"]
    assert "mods/foo" in copy_path
    assert "home" not in copy_path  # registry cache name not in path


def test_resolve_order_priority_scoped_beats_adjacent(tmp_path: Path):
    """Explicit @scoped reference bypasses adjacent-search."""
    entry = RegistryEntry(name="alt", url="https://example/alt.git", subpath="recipes", mods_subpath="mods")
    mgr = _make_registry_manager(tmp_path, [entry])
    _populate_registry_cache(mgr, entry, mods={"foo": "echo alt"})

    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo", body="echo adjacent")

    recipe = _recipe_with_mods(["@alt/foo"], source_path=recipe_path)
    resolve_and_inject_mods(recipe, mgr)

    assert "alt" in recipe.pre_exec[0]["copy"]


def test_resolve_strips_mods_prefix(tmp_path: Path):
    """`foo` and `mods/foo` resolve to identical pre_exec entries."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    mgr = _make_registry_manager(tmp_path, [])

    r1 = _recipe_with_mods(["foo"], source_path=recipe_path)
    r2 = _recipe_with_mods(["mods/foo"], source_path=recipe_path)
    resolve_and_inject_mods(r1, mgr)
    resolve_and_inject_mods(r2, mgr)

    assert r1.pre_exec == r2.pre_exec
    # And no double 'mods/mods/' in the source path
    assert "mods/mods/" not in r1.pre_exec[0]["copy"]


def test_resolve_not_found_raises_with_paths(tmp_path: Path):
    """ModNotFoundError lists every candidate path the resolver tried."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods(["nope"], source_path=recipe_path)

    with pytest.raises(ModNotFoundError) as excinfo:
        resolve_and_inject_mods(recipe, mgr)
    err = excinfo.value
    assert err.name == "nope"
    assert err.tried, "ModNotFoundError should record the attempted paths"
    # Adjacent paths should be present in the list
    joined = " ".join(err.tried)
    assert "mods/nope" in joined or "/nope" in joined


def test_resolve_is_idempotent(tmp_path: Path):
    """A second call after resolution is a no-op (recipe.mods cleared)."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods(["foo"], source_path=recipe_path)

    resolve_and_inject_mods(recipe, mgr)
    pre_exec_after_first = list(recipe.pre_exec)
    resolve_and_inject_mods(recipe, mgr)
    assert recipe.pre_exec == pre_exec_after_first


# ---------------------------------------------------------------------------
# Pre-exec entry shape (local vs. delegated)
# ---------------------------------------------------------------------------


def test_pre_exec_entries_shape_local(tmp_path: Path):
    """Local mode: copy dict has no `source_host` key."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods(["foo"], source_path=recipe_path)

    resolve_and_inject_mods(recipe, mgr, transfer_mode="local")

    copy_entry = recipe.pre_exec[0]
    assert isinstance(copy_entry, dict)
    assert set(copy_entry.keys()) == {"copy", "dest"}
    assert copy_entry["dest"] == "/workspace/mods/foo"
    assert recipe.pre_exec[1] == ("export WORKSPACE_DIR=$PWD && cd /workspace/mods/foo && chmod +x run.sh && ./run.sh")


def test_pre_exec_entries_shape_delegated_scoped(tmp_path: Path):
    """Delegated mode: copy dict carries `source_host=<head>` for registry mods."""
    entry = RegistryEntry(name="alt", url="https://example/alt.git", subpath="recipes", mods_subpath="mods")
    mgr = _make_registry_manager(tmp_path, [entry])
    _populate_registry_cache(mgr, entry, mods={"foo": "echo alt"})

    recipe = _recipe_with_mods(["@alt/foo"])

    with mock.patch.object(RegistryManager, "ensure_registry_on_host", return_value="/remote/registries/_url_abc") as mock_ensure:
        resolve_and_inject_mods(recipe, mgr, transfer_mode="delegated", head="head-host", ssh_kwargs={})

    mock_ensure.assert_called_once()
    copy_entry = recipe.pre_exec[0]
    assert copy_entry["source_host"] == "head-host"
    assert copy_entry["dest"] == "/workspace/mods/foo"
    assert copy_entry["copy"] == "/remote/registries/_url_abc/mods/foo"


def test_delegated_rsyncs_adjacent_mod_to_head(tmp_path: Path):
    """Adjacent mod in delegated mode is rsync'd to the head staging dir."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods(["foo"], source_path=recipe_path)

    with mock.patch("sparkrun.core.mods.run_rsync") as _mock_rsync:
        _mock_rsync.return_value = mock.Mock(success=True, stderr="")
        resolve_and_inject_mods(recipe, mgr, transfer_mode="delegated", head="head-host", ssh_kwargs={})

    _mock_rsync.assert_called_once()
    args, kwargs = _mock_rsync.call_args
    # run_rsync(local_path, host, dest, ...)
    assert "head-host" in args
    copy_entry = recipe.pre_exec[0]
    assert copy_entry["source_host"] == "head-host"
    assert copy_entry["copy"].endswith("/foo")


def test_delegated_requires_head(tmp_path: Path):
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods(["foo"])
    with pytest.raises(ValueError, match="head"):
        resolve_and_inject_mods(recipe, mgr, transfer_mode="delegated", head=None)


def test_delegated_dry_run_skips_rsync(tmp_path: Path):
    """Dry-run still computes paths but never calls rsync or git on the head."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods(["foo"], source_path=recipe_path)

    with mock.patch("sparkrun.core.mods.run_rsync") as _mock_rsync:
        resolve_and_inject_mods(recipe, mgr, transfer_mode="delegated", head="head-host", dry_run=True)
    _mock_rsync.assert_not_called()
    assert recipe.pre_exec[0]["source_host"] == "head-host"


# ---------------------------------------------------------------------------
# Empty inputs and edge cases
# ---------------------------------------------------------------------------


def test_resolve_no_mods_is_noop(tmp_path: Path):
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods([])
    resolve_and_inject_mods(recipe, mgr)
    assert recipe.pre_exec == []
    assert recipe.mods == []


def test_resolve_mod_dir_without_run_sh_misses(tmp_path: Path):
    """A directory exists but lacks run.sh — should not match."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    (tmp_path / "mods" / "foo").mkdir(parents=True)  # no run.sh
    mgr = _make_registry_manager(tmp_path, [])
    recipe = _recipe_with_mods(["foo"], source_path=recipe_path)

    with pytest.raises(ModNotFoundError):
        resolve_and_inject_mods(recipe, mgr)


# ---------------------------------------------------------------------------
# Recipe.to_dict round-trip
# ---------------------------------------------------------------------------


def test_recipe_to_dict_preserves_mods_after_resolution(tmp_path: Path):
    """Export contains the original mods list; pre_exec excludes mod-derived entries."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    mgr = _make_registry_manager(tmp_path, [])

    recipe = Recipe.from_dict(
        {
            "name": "test",
            "model": "m",
            "runtime": "vllm-distributed",
            "container": "img:tag",
            "mods": ["foo"],
            # No user-authored pre_exec
        }
    )
    recipe.source_path = str(recipe_path)
    resolve_and_inject_mods(recipe, mgr)

    assert len(recipe.pre_exec) == 2  # mods injected entries are live in memory
    exported = recipe.to_dict()
    # mods round-trips
    assert exported.get("mods") == ["foo"]
    # mod-derived pre_exec entries are NOT exported (Recipe.to_dict() resets
    # pre_exec from raw input — which had no pre_exec key)
    assert "pre_exec" not in exported


def test_recipe_to_dict_preserves_user_pre_exec_only(tmp_path: Path):
    """If the user authored pre_exec, export keeps that — not mod-derived entries."""
    recipe_path = tmp_path / "my.yaml"
    recipe_path.write_text("name: x\n")
    _write_mod(tmp_path / "mods", "foo")
    mgr = _make_registry_manager(tmp_path, [])

    user_pre_exec = ["echo hello"]
    recipe = Recipe.from_dict(
        {
            "name": "test",
            "model": "m",
            "runtime": "vllm-distributed",
            "container": "img:tag",
            "mods": ["foo"],
            "pre_exec": list(user_pre_exec),
        }
    )
    recipe.source_path = str(recipe_path)
    resolve_and_inject_mods(recipe, mgr)

    assert len(recipe.pre_exec) == 3  # user 1 + injected 2
    exported = recipe.to_dict()
    assert exported["mods"] == ["foo"]
    assert exported["pre_exec"] == user_pre_exec


def test_recipe_getstate_setstate_roundtrips_mods():
    recipe = Recipe.from_dict(
        {
            "name": "test",
            "model": "m",
            "runtime": "vllm-distributed",
            "container": "img:tag",
            "mods": ["foo", "mods/bar"],
        }
    )
    state = recipe.__getstate__()
    assert state["mods"] == ["foo", "mods/bar"]

    restored = Recipe._deserialize(state)
    assert restored.mods == ["foo", "mods/bar"]


# ---------------------------------------------------------------------------
# ResolvedMod sanity
# ---------------------------------------------------------------------------


def test_resolved_mod_dataclass():
    r = ResolvedMod(name="foo", source_path="/x/foo", source_host=None)
    assert r.name == "foo"
    assert r.source_host is None
