"""``subpath: "/"``: a registry whose assets sit at the repository root."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml

from sparkrun.core.registry import (
    RECIPE_ASSET,
    REPO_ROOT_SUBPATH,
    RegistryEntry,
    RegistryError,
    RegistryManager,
    assert_safe_registry_subpath,
    iter_asset_files,
    resolve_registry_subpath,
)


def test_root_is_the_only_absolute_spelling_accepted():
    assert_safe_registry_subpath(REPO_ROOT_SUBPATH)
    for bad in ("/recipes", "//", "/etc"):
        with pytest.raises(RegistryError, match="'/' alone means the repository root"):
            assert_safe_registry_subpath(bad)


def test_root_resolves_to_the_checkout_never_the_filesystem(tmp_path):
    checkout = tmp_path / "cache" / "reg"
    assert resolve_registry_subpath(checkout, REPO_ROOT_SUBPATH) == checkout
    assert resolve_registry_subpath(checkout, "recipes") == checkout / "recipes"
    assert resolve_registry_subpath(checkout, "") is None


@pytest.fixture
def root_registry(tmp_path: Path):
    config, cache = tmp_path / "config", tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    mgr = RegistryManager(config, cache)
    mgr._manifest_discovery_attempted = True
    entry = RegistryEntry(name="rooted", url="https://example.invalid/rooted.git", subpath=REPO_ROOT_SUBPATH)
    mgr._save_registries([entry])
    checkout = cache / "rooted"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".sparkrun").mkdir()
    (checkout / ".sparkrun" / "registry.yaml").write_text(yaml.safe_dump({"registries": [{"name": "rooted"}]}))
    (checkout / ".git" / "config.yaml").write_text("x: 1")
    (checkout / "top.yaml").write_text(yaml.safe_dump({"model": "org/top", "runtime": "vllm", "container": "img"}))
    (checkout / "fam").mkdir()
    (checkout / "fam" / "nested.yaml").write_text(yaml.safe_dump({"model": "org/nested", "runtime": "vllm", "container": "img"}))
    return mgr, entry, checkout


def test_asset_dir_is_the_checkout(root_registry):
    mgr, entry, checkout = root_registry
    assert mgr.asset_dir(entry, RECIPE_ASSET) == checkout


def test_scans_skip_hidden_directories(root_registry):
    mgr, _, checkout = root_registry
    assert iter_asset_files(checkout, RECIPE_ASSET) == [checkout / "fam" / "nested.yaml", checkout / "top.yaml"]
    assert mgr.find_recipe_in_registries("registry") == []  # the manifest is not a recipe
    assert [r["name"] for r in mgr.search_recipes("")] == ["@rooted/nested", "@rooted/top"]
    assert mgr.qualified_recipe_name("rooted", checkout / "fam" / "nested.yaml") == "@rooted/fam/nested"


def test_root_subpath_round_trips(root_registry):
    mgr, _, _ = root_registry
    [entry] = mgr._load_registries()
    assert entry.subpath == REPO_ROOT_SUBPATH


def test_sparse_checkout_is_disabled_for_a_root_subpath(tmp_path):
    with mock.patch("sparkrun.core.registry.subprocess.run") as run:
        RegistryManager._apply_sparse_paths(tmp_path, ["/", ".sparkrun"], {})
        RegistryManager._apply_sparse_paths(tmp_path, ["recipes", ".sparkrun"], {})
    assert run.call_args_list[0].args[0][-2:] == ["sparse-checkout", "disable"]
    assert run.call_args_list[1].args[0][-4:] == ["sparse-checkout", "set", "recipes", ".sparkrun"]


def test_raw_url_has_no_root_segment():
    from sparkrun.cli._registry import _build_raw_url

    assert _build_raw_url("https://github.com/o/r.git", "/", "a/b.yaml") == "https://raw.githubusercontent.com/o/r/main/a/b.yaml"
    assert _build_raw_url("https://github.com/o/r", "recipes", "b.yaml") == "https://raw.githubusercontent.com/o/r/main/recipes/b.yaml"
