"""Foreign recipe formats: the registry ``format`` field and the RecipeFormat seam.

A toy format stands in for a real one (lil): each entry is ``<Name>/toy.yaml``,
drafts (``kind: draft``) are not launchable, and a manifest translates to v2
recipe data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparkrun.core.recipe import Recipe, RecipeError, find_recipe, list_recipes
from sparkrun.core.recipe_formats import (
    DEFAULT_RECIPE_FORMAT,
    RecipeFormat,
    get_recipe_format,
    register_recipe_format,
    unregister_recipe_format,
)
from sparkrun.core.registry import RegistryEntry, RegistryError, RegistryManager, assert_safe_registry_entry

CALLS: list[bool] = []


def _toy_iter(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/toy.yaml") if (yaml.safe_load(p.read_text()) or {}).get("kind") != "draft")


def _toy_name(path: Path, root: Path | None) -> str:
    return path.parent.name


def _toy_load(path: Path, *, registry_manager=None, offline: bool, allow_local: bool = True) -> dict:
    CALLS.append(offline)
    data = yaml.safe_load(path.read_text())
    return {
        "model": data["weights"],
        "runtime": "vllm-distributed",
        "container": "img:toy",
        "description": data.get("about", ""),
        "defaults": {"port": 8000, "max_num_seqs": data.get("seqs", 4)},
    }


def _toy_claims(path: Path, data: dict) -> bool:
    return path.name == "toy.yaml" and "weights" in data


TOY = RecipeFormat(name="toy", owner="tests", iter_files=_toy_iter, name_of=_toy_name, load=_toy_load, claims=_toy_claims)


@pytest.fixture
def toy_format():
    CALLS.clear()
    register_recipe_format(TOY)
    yield TOY
    unregister_recipe_format("toy")


@pytest.fixture
def toy_registry(tmp_path: Path):
    """A manager with one sparkrun registry and one toy-format registry, both cached."""
    config, cache = tmp_path / "config", tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    mgr = RegistryManager(config, cache)
    mgr._manifest_discovery_attempted = True
    native = RegistryEntry(name="native", url="https://example.invalid/native.git", subpath="recipes")
    toy = RegistryEntry(name="toyreg", url="https://example.invalid/toy.git", subpath="catalog", format="toy")
    mgr._save_registries([native, toy])

    native_dir = cache / "native" / "recipes"
    native_dir.mkdir(parents=True)
    (cache / "native" / ".git").mkdir()
    (native_dir / "plain.yaml").write_text(yaml.safe_dump({"model": "org/plain", "runtime": "vllm", "container": "img:x"}))

    toy_dir = cache / "toyreg" / "catalog"
    toy_dir.mkdir(parents=True)
    (cache / "toyreg" / ".git").mkdir()
    for name, body in {
        "Alpha": {"weights": "org/alpha", "about": "first", "seqs": 2},
        "Beta": {"weights": "org/beta"},
        "Draft": {"kind": "draft", "weights": "org/draft"},
    }.items():
        (toy_dir / name).mkdir()
        (toy_dir / name / "toy.yaml").write_text(yaml.safe_dump(body))
    # A stray YAML the sparkrun scanner would pick up; the toy format must not.
    (toy_dir / "README.yaml").write_text(yaml.safe_dump({"model": "not/a/recipe", "runtime": "vllm", "container": "x"}))
    return mgr, toy_dir


# --- registration ----------------------------------------------------------------------------


def test_registration_is_idempotent_and_owner_exclusive(toy_format):
    from sparkrun.core.installed_plugins import PluginConflictError

    register_recipe_format(TOY)  # same owner: fine
    with pytest.raises(PluginConflictError, match="already registered by tests"):
        register_recipe_format(RecipeFormat(name="toy", owner="someone-else", iter_files=_toy_iter, name_of=_toy_name, load=_toy_load))


def test_the_builtin_format_cannot_be_replaced():
    with pytest.raises(ValueError, match="built in"):
        register_recipe_format(RecipeFormat(name=DEFAULT_RECIPE_FORMAT, owner="x", iter_files=_toy_iter, name_of=_toy_name, load=_toy_load))
    assert get_recipe_format(DEFAULT_RECIPE_FORMAT) is None


# --- registry entry --------------------------------------------------------------------------


def test_format_round_trips_and_default_is_not_written(toy_registry):
    mgr, _ = toy_registry
    by_name = {e.name: e for e in mgr._load_registries()}
    assert by_name["toyreg"].format == "toy"
    assert by_name["native"].format == DEFAULT_RECIPE_FORMAT
    written = yaml.safe_load(mgr._registries_path.read_text())["registries"]
    assert [r.get("format") for r in written] == [None, "toy"]


@pytest.mark.parametrize("fmt", ["Toy", "../x", "", "x" * 40, "a b"])
def test_unsafe_formats_are_rejected(fmt):
    with pytest.raises(RegistryError, match="format"):
        assert_safe_registry_entry(RegistryEntry(name="r", url="https://example.invalid/r.git", subpath="recipes", format=fmt))


# --- lookup, listing, loading ---------------------------------------------------------------------


def test_lookup_resolves_by_the_formats_name(toy_registry, toy_format):
    mgr, toy_dir = toy_registry
    assert find_recipe("@toyreg/Alpha", registry_manager=mgr) == toy_dir / "Alpha" / "toy.yaml"
    assert find_recipe("Beta", registry_manager=mgr) == toy_dir / "Beta" / "toy.yaml"
    with pytest.raises(RecipeError):
        find_recipe("@toyreg/Draft", registry_manager=mgr)  # not launchable, so not resolvable
    with pytest.raises(RecipeError):
        find_recipe("@toyreg/README", registry_manager=mgr)  # a stray file is not a recipe


def test_listing_uses_the_format_offline(toy_registry, toy_format):
    mgr, _ = toy_registry
    rows = {r["name"]: r for r in mgr.search_recipes("")}
    assert set(rows) == {"@native/plain", "@toyreg/Alpha", "@toyreg/Beta"}
    assert rows["@toyreg/Alpha"]["model"] == "org/alpha"
    assert CALLS and all(CALLS)  # every listing call was offline


def test_list_recipes_includes_foreign_registries_once(toy_registry, toy_format):
    mgr, _ = toy_registry
    names = [r["name"] for r in list_recipes(registry_manager=mgr)]
    assert sorted(names) == ["@native/plain", "@toyreg/Alpha", "@toyreg/Beta"]


def test_get_recipe_paths_leaves_foreign_registries_out(toy_registry, toy_format):
    mgr, toy_dir = toy_registry
    assert toy_dir not in mgr.get_recipe_paths()


def test_qualified_name_uses_the_format(toy_registry, toy_format):
    mgr, toy_dir = toy_registry
    assert mgr.qualified_recipe_name("toyreg", toy_dir / "Alpha" / "toy.yaml") == "@toyreg/Alpha"


def test_loading_translates_to_a_v2_recipe(toy_registry, toy_format):
    mgr, toy_dir = toy_registry
    recipe = Recipe.load(toy_dir / "Alpha" / "toy.yaml", registry_manager=mgr)
    assert recipe.name == "Alpha"
    assert recipe.model == "org/alpha" and recipe.defaults["max_num_seqs"] == 2
    assert recipe.runtime == "vllm-distributed"
    assert CALLS[-1] is False  # a launch load may use the network


def test_known_format_skips_registry_lookup(toy_registry, toy_format):
    _, toy_dir = toy_registry
    recipe = Recipe.load(toy_dir / "Beta" / "toy.yaml", recipe_format="toy")
    assert recipe.model == "org/beta"


def test_direct_path_is_claimed_by_content(tmp_path, toy_format):
    manifest = tmp_path / "Gamma" / "toy.yaml"
    manifest.parent.mkdir()
    manifest.write_text(yaml.safe_dump({"weights": "org/gamma"}))
    recipe = Recipe.load(manifest)
    assert (recipe.name, recipe.model) == ("Gamma", "org/gamma")


def test_export_of_a_foreign_recipe_is_plain_v2(toy_registry, toy_format):
    mgr, toy_dir = toy_registry
    exported = yaml.safe_load(Recipe.load(toy_dir / "Alpha" / "toy.yaml", registry_manager=mgr).export())
    assert exported["model"] == "org/alpha" and exported["recipe_version"] == "2"
    assert "weights" not in exported


# --- no plugin: inert, never misread ------------------------------------------------------------------


def test_without_the_plugin_the_registry_lists_and_resolves_nothing(toy_registry):
    mgr, toy_dir = toy_registry
    assert [r["name"] for r in mgr.search_recipes("")] == ["@native/plain"]
    with pytest.raises(RecipeError):
        find_recipe("@toyreg/Alpha", registry_manager=mgr)
    with pytest.raises(RecipeError):
        find_recipe("@toyreg/README", registry_manager=mgr)  # not rescanned as sparkrun YAML either


def test_without_the_plugin_loading_its_file_is_refused(toy_registry):
    mgr, toy_dir = toy_registry
    with pytest.raises(RecipeError, match="no loaded plugin provides"):
        Recipe.load(toy_dir / "README.yaml", registry_manager=mgr)
    with pytest.raises(RecipeError, match="no loaded plugin provides"):
        Recipe.load(toy_dir / "Alpha" / "toy.yaml", recipe_format="toy")


def test_translation_errors_name_the_format(tmp_path, toy_format):
    manifest = tmp_path / "Broken" / "toy.yaml"
    manifest.parent.mkdir()
    manifest.write_text(yaml.safe_dump({"weights": "org/x", "seqs": 1}))

    def _boom(path, *, registry_manager=None, offline, allow_local=True):
        raise KeyError("capacity")

    register_recipe_format(
        RecipeFormat(name="toy2", owner="tests", iter_files=_toy_iter, name_of=_toy_name, load=_boom, claims=_toy_claims)
    )
    unregister_recipe_format("toy")
    try:
        with pytest.raises(RecipeError, match="Could not translate toy2 manifest"):
            Recipe.load(manifest)
    finally:
        unregister_recipe_format("toy2")


def test_known_format_loads_under_the_same_name_it_lists(toy_registry):
    """M2: a format whose name depends on the registry root names a load and a listing identically."""
    mgr, toy_dir = toy_registry

    def _rooted(path, root):
        return "no-root" if root is None else path.parent.relative_to(root).as_posix()

    register_recipe_format(RecipeFormat(name="toy", owner="tests", iter_files=_toy_iter, name_of=_rooted, load=_toy_load))
    try:
        listed = {r["file"] for r in mgr.search_recipes("")}
        loaded = Recipe.load(toy_dir / "Alpha" / "toy.yaml", registry_manager=mgr, recipe_format="toy")
        assert loaded.name == "Alpha" and "Alpha" in listed
    finally:
        unregister_recipe_format("toy")


def test_the_format_is_told_when_neighbouring_files_are_off_limits(tmp_path, toy_format):
    manifest = tmp_path / "Delta" / "toy.yaml"
    manifest.parent.mkdir()
    manifest.write_text(yaml.safe_dump({"weights": "org/delta"}))
    seen = []
    original = TOY.load

    def _spy(path, **kw):
        seen.append(kw["allow_local"])
        return original(path, **kw)

    register_recipe_format(RecipeFormat(name="toy", owner="tests", iter_files=_toy_iter, name_of=_toy_name, load=_spy, claims=_toy_claims))
    Recipe.load(manifest)
    assert seen == [True]


def test_registry_show_explains_an_inert_format(toy_registry, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    mgr, _ = toy_registry
    monkeypatch.setattr("sparkrun.core.config.SparkrunConfig.get_registry_manager", lambda self: mgr)
    result = CliRunner().invoke(main, ["registry", "show", "toyreg"])
    assert "Format:      toy (no loaded plugin provides it" in result.output, result.output
