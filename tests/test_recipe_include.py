"""Recipe ``include:`` — merge rules, resolution, provenance, trust and export."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparkrun.core.recipe import Recipe, RecipeError, is_recipe_file, recipe_summary
from sparkrun.core.recipe_include import MAX_INCLUDE_DEPTH, IncludeSource, merge_recipe_data, resolve_recipe_includes

_BASE = {
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "vllm-distributed",
    "container": "vllm/vllm-openai:latest",
    "defaults": {
        "port": 8000,
        "max_num_seqs": 8,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
    },
    "env": {"A": "1", "B": "2"},
    "metadata": {"description": "base", "model_params": "1.7B"},
    "mods": ["fix-a"],
    "overrides": [{"when": {"nodes": 1}, "defaults": {"max_num_seqs": 1}}],
}


def _write(directory: Path, name: str, data: dict) -> Path:
    path = directory / ("%s.yaml" % name)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# --- merge rules -----------------------------------------------------------------


def test_merge_mappings_by_key_and_replace_scalars():
    merged = merge_recipe_data(_BASE, {"container": "img:new", "defaults": {"max_num_seqs": 4}, "metadata": {"description": "outer"}})
    assert merged["container"] == "img:new"
    assert merged["defaults"]["max_num_seqs"] == 4
    assert merged["defaults"]["port"] == 8000  # inherited
    assert merged["metadata"] == {"description": "outer", "model_params": "1.7B"}


def test_default_values_are_atomic():
    """Swapping speculators must not blend two configs into one."""
    merged = merge_recipe_data(_BASE, {"defaults": {"speculative_config": {"method": "dflash", "model": "d/raft"}}})
    assert merged["defaults"]["speculative_config"] == {"method": "dflash", "model": "d/raft"}


def test_lists_replace_and_null_deletes():
    merged = merge_recipe_data(_BASE, {"mods": ["fix-b"], "env": {"A": None}, "metadata": None})
    assert merged["mods"] == ["fix-b"]
    assert merged["env"] == {"B": "2"}
    assert "metadata" not in merged


def test_overrides_append_base_first():
    outer = {"overrides": [{"when": {"tp": 2}, "defaults": {"max_num_seqs": 2}}]}
    merged = merge_recipe_data(_BASE, outer)
    assert [o["when"] for o in merged["overrides"]] == [{"nodes": 1}, {"tp": 2}]


def test_merge_does_not_mutate_inputs():
    base = {"defaults": {"x": 1}, "overrides": [{"when": {"tp": 1}, "defaults": {"y": 1}}]}
    outer = {"defaults": {"x": 2}, "overrides": [{"when": {"tp": 2}, "defaults": {"y": 2}}]}
    merge_recipe_data(base, outer)
    assert base == {"defaults": {"x": 1}, "overrides": [{"when": {"tp": 1}, "defaults": {"y": 1}}]}


# --- sibling resolution ----------------------------------------------------------


def test_sibling_include_loads_merged_recipe(tmp_path):
    _write(tmp_path, "base", _BASE)
    outer = _write(tmp_path, "variant", {"include": "base", "defaults": {"max_num_seqs": 2}})
    recipe = Recipe.load(outer)
    assert recipe.name == "variant"
    assert recipe.model == "Qwen/Qwen3-1.7B"
    assert recipe.defaults["max_num_seqs"] == 2
    assert recipe.defaults["port"] == 8000
    assert [s.ref for s in recipe.include_chain] == ["base"]
    assert recipe.include_chain[0].registry is None
    assert len(recipe.overrides) == 1  # inherited


def test_sibling_include_accepts_extension(tmp_path):
    _write(tmp_path, "base", _BASE)
    outer = _write(tmp_path, "variant", {"include": "base.yaml"})
    assert Recipe.load(outer).model == _BASE["model"]


@pytest.mark.parametrize("ref", ["../base", "sub/base", "/etc/passwd", "..", ".hidden", "-rf", "C:base"])
def test_sibling_include_cannot_leave_the_directory(tmp_path, ref):
    outer = _write(tmp_path, "variant", {"include": ref})
    with pytest.raises(RecipeError, match="same directory"):
        Recipe.load(outer)


def test_missing_include_is_an_error(tmp_path):
    outer = _write(tmp_path, "variant", {"include": "nope"})
    with pytest.raises(RecipeError, match="not found"):
        Recipe.load(outer)


def test_nested_chain_records_every_level(tmp_path):
    _write(tmp_path, "a", _BASE)
    _write(tmp_path, "b", {"include": "a", "defaults": {"max_num_seqs": 5}})
    outer = _write(tmp_path, "c", {"include": "b", "env": {"C": "3"}})
    recipe = Recipe.load(outer)
    assert [s.ref for s in recipe.include_chain] == ["a", "b"]
    assert recipe.defaults["max_num_seqs"] == 5
    assert recipe.env == {"A": "1", "B": "2", "C": "3"}


def test_include_cycle_is_detected(tmp_path):
    _write(tmp_path, "a", {"include": "b", **_BASE})
    outer = _write(tmp_path, "b", {"include": "a"})
    with pytest.raises(RecipeError, match="cycle"):
        Recipe.load(outer)


def test_self_include_is_a_cycle(tmp_path):
    outer = _write(tmp_path, "a", {"include": "a", **_BASE})
    with pytest.raises(RecipeError, match="cycle"):
        Recipe.load(outer)


def test_include_depth_is_capped(tmp_path):
    _write(tmp_path, "r0", _BASE)
    for i in range(1, MAX_INCLUDE_DEPTH + 2):
        _write(tmp_path, "r%d" % i, {"include": "r%d" % (i - 1)})
    with pytest.raises(RecipeError, match="deeper than"):
        Recipe.load(tmp_path / ("r%d.yaml" % (MAX_INCLUDE_DEPTH + 1)))


def test_v1_recipes_cannot_take_part(tmp_path):
    _write(tmp_path, "old", {"recipe_version": "1", **_BASE})
    outer = _write(tmp_path, "variant", {"include": "old"})
    with pytest.raises(RecipeError, match="v2 feature"):
        Recipe.load(outer)


def test_sibling_includes_refused_without_a_trustworthy_directory(tmp_path):
    _write(tmp_path, "base", _BASE)
    outer = _write(tmp_path, "variant", {"include": "base"})
    with pytest.raises(RecipeError, match="fully qualified"):
        Recipe.load(outer, allow_local_includes=False)


def test_unresolved_include_cannot_be_constructed_directly():
    with pytest.raises(RecipeError, match="Recipe.load"):
        Recipe({"include": "base", **_BASE})


@pytest.mark.parametrize("ref", [None, "", 42, ["a"]])
def test_include_must_be_a_name(tmp_path, ref):
    outer = _write(tmp_path, "variant", {"include": ref, **_BASE})
    with pytest.raises(RecipeError, match="must be a recipe name"):
        Recipe.load(outer)


# --- registry resolution ---------------------------------------------------------


class _Registries:
    """Minimal registry manager: one registry, recipes looked up by stem."""

    def __init__(self, root: Path, name: str = "reg", url: str = "https://example.invalid/reg.git", trusted: bool = False):
        self.root, self.name, self.url, self.trusted = root, name, url, trusted

    def find_recipe_in_registries(self, name, include_hidden=False):
        path = self.root / ("%s.yaml" % name)
        return [(self.name, path)] if path.exists() else []

    def get_registry(self, name, allow_discovery=True):
        return SimpleNamespace(name=self.name, url=self.url, enabled=True, trusted=self.trusted)

    def qualified_recipe_name(self, registry, path):
        return "@%s/%s" % (registry, Path(path).stem)

    def _recipe_dir(self, entry):
        return self.root

    def registry_for_path(self, path, *, allow_discovery=True, entries=None):
        return self.name if Path(path).resolve().is_relative_to(self.root.resolve()) else None


def test_registry_include_records_provenance_and_scopes_mods(tmp_path):
    reg_root = tmp_path / "reg"
    reg_root.mkdir()
    _write(reg_root, "base", _BASE)
    local = tmp_path / "local"
    local.mkdir()
    outer = _write(local, "variant", {"include": "@reg/base"})
    recipe = Recipe.load(outer, registry_manager=_Registries(reg_root))
    source = recipe.include_chain[-1]
    assert (source.registry, source.registry_url) == ("reg", "https://example.invalid/reg.git")
    # Resolved where the base's author put it, not next to the includer.
    assert recipe.mods == ["@reg/fix-a"]


def test_registry_base_sibling_includes_stay_inside_the_registry(tmp_path):
    reg_root = tmp_path / "reg"
    reg_root.mkdir()
    _write(reg_root, "core", _BASE)
    _write(reg_root, "base", {"include": "core", "defaults": {"max_num_seqs": 6}})
    outer = _write(tmp_path, "variant", {"include": "@reg/base"})
    recipe = Recipe.load(outer, registry_manager=_Registries(reg_root))
    assert [s.ref for s in recipe.include_chain] == ["core", "@reg/base"]
    assert recipe.defaults["max_num_seqs"] == 6


def test_registry_includes_allowed_for_url_sources(tmp_path):
    reg_root = tmp_path / "reg"
    reg_root.mkdir()
    _write(reg_root, "base", _BASE)
    outer = _write(tmp_path, "variant", {"include": "@reg/base"})
    assert Recipe.load(outer, registry_manager=_Registries(reg_root), allow_local_includes=False).model == _BASE["model"]


# --- trust ------------------------------------------------------------------------


@pytest.mark.parametrize("trusted", [True, False])
def test_local_recipe_is_only_as_trusted_as_its_registry_base(tmp_path, trusted):
    from sparkrun.core.launcher import resolve_recipe_trust

    reg_root = tmp_path / "reg"
    reg_root.mkdir()
    _write(reg_root, "base", _BASE)
    registries = _Registries(reg_root, trusted=trusted)
    recipe = Recipe.load(_write(tmp_path, "variant", {"include": "@reg/base"}), registry_manager=registries)
    sctx = SimpleNamespace(registry_manager=registries)
    assert resolve_recipe_trust(recipe, False, sctx=sctx) is trusted


def test_trust_refused_when_the_registry_url_changed(tmp_path):
    from sparkrun.core.launcher import resolve_recipe_trust

    reg_root = tmp_path / "reg"
    reg_root.mkdir()
    _write(reg_root, "base", _BASE)
    recipe = Recipe.load(_write(tmp_path, "variant", {"include": "@reg/base"}), registry_manager=_Registries(reg_root, trusted=True))
    moved = _Registries(reg_root, url="https://example.invalid/elsewhere.git", trusted=True)
    assert resolve_recipe_trust(recipe, False, sctx=SimpleNamespace(registry_manager=moved)) is False


def test_sibling_include_keeps_local_trust(tmp_path):
    from sparkrun.core.launcher import resolve_recipe_trust

    _write(tmp_path, "base", _BASE)
    recipe = Recipe.load(_write(tmp_path, "variant", {"include": "base"}))
    assert resolve_recipe_trust(recipe, False) is True


# --- identity, export, state -----------------------------------------------------------


def test_fingerprint_follows_the_base(tmp_path):
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    _write(tmp_path, "base", _BASE)
    outer = _write(tmp_path, "variant", {"include": "base"})
    before = derive_recipe_fingerprint(Recipe.load(outer))
    _write(tmp_path, "base", {**_BASE, "defaults": {**_BASE["defaults"], "max_num_seqs": 99}})
    assert derive_recipe_fingerprint(Recipe.load(outer)) != before


def test_flattened_recipe_equals_hand_written_equivalent(tmp_path):
    """A recipe via include is the same workload as its flattened form."""
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint, generate_intent_id

    _write(tmp_path, "base", _BASE)
    via_include = Recipe.load(_write(tmp_path, "variant", {"include": "base", "defaults": {"max_num_seqs": 2}}))
    flat_path = tmp_path / "flat" / "variant.yaml"
    flat_path.parent.mkdir()
    flat_path.write_text(via_include.export())
    flat = Recipe.load(flat_path)
    assert "include" not in yaml.safe_load(flat_path.read_text())
    assert derive_recipe_fingerprint(flat) == derive_recipe_fingerprint(via_include)
    assert generate_intent_id(flat) == generate_intent_id(via_include)


def test_state_round_trip_keeps_include_provenance(tmp_path):
    _write(tmp_path, "base", _BASE)
    recipe = Recipe.load(_write(tmp_path, "variant", {"include": "base"}))
    restored = Recipe._deserialize_yaml(recipe._serialize_yaml())
    assert restored.include_chain == recipe.include_chain
    assert restored._include_declared == {"include": "base"}
    assert isinstance(restored.include_chain[0], IncludeSource)


def test_export_keep_include_emits_the_file_as_written(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    _write(tmp_path, "base", _BASE)
    outer = _write(tmp_path, "variant", {"include": "base", "defaults": {"max_num_seqs": 2}})
    monkeypatch.chdir(tmp_path)
    kept = CliRunner().invoke(main, ["export", "recipe", str(outer), "--keep-include"])
    assert kept.exit_code == 0, kept.output
    assert yaml.safe_load(kept.output) == {"include": "base", "defaults": {"max_num_seqs": 2}}
    flat = CliRunner().invoke(main, ["export", "recipe", str(outer)])
    assert flat.exit_code == 0, flat.output
    exported = yaml.safe_load(flat.output)
    assert "include" not in exported and exported["model"] == _BASE["model"]


# --- listing / discovery -----------------------------------------------------------------


def test_listing_resolves_sibling_includes(tmp_path):
    _write(tmp_path, "base", _BASE)
    outer = _write(tmp_path, "variant", {"include": "base"})
    assert is_recipe_file(outer)
    assert recipe_summary(outer)["model"] == _BASE["model"]


def test_listing_never_resolves_registry_includes(tmp_path):
    """A catalog scan must not reach for the registry manager (it would clone)."""
    outer = _write(tmp_path, "variant", {"include": "@reg/base"})
    assert is_recipe_file(outer)
    assert recipe_summary(outer)["model"] == ""


def test_resolve_without_include_is_identity():
    data = {"model": "m"}
    merged, chain = resolve_recipe_includes(data, None)
    assert merged is data and chain == ()


# --- validation -----------------------------------------------------------------------------


def test_nested_include_suggestion(tmp_path):
    from sparkrun.core.validation import SUGGESTION, check_nested_include

    _write(tmp_path, "a", _BASE)
    _write(tmp_path, "b", {"include": "a"})
    one = Recipe.load(_write(tmp_path, "one", {"include": "a"}))
    two = Recipe.load(_write(tmp_path, "two", {"include": "b"}))
    assert check_nested_include(one) == []
    [issue] = check_nested_include(two)
    assert (issue.severity, issue.code) == (SUGGESTION, "nested-include")
    assert "b → a" in issue.summary


# --- review hardening ---------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["@reg//etc/passwd", "@reg/../../x", "@reg/a/../b", "@reg/", "@reg/-x", "@reg/C:x"])
def test_registry_include_names_cannot_leave_the_registry(tmp_path, ref):
    """The name is third-party content joined onto a registry dir; it must not reach other files."""
    reg_root = tmp_path / "reg"
    reg_root.mkdir()
    outer = _write(tmp_path, "variant", {"include": ref})
    with pytest.raises(RecipeError, match="not a valid @registry/recipe name"):
        Recipe.load(outer, registry_manager=_Registries(reg_root))


def test_registry_include_resolving_outside_its_registry_is_refused(tmp_path):
    """Belt and braces: a lookup that returns a path outside the registry is not trusted as that registry."""
    reg_root = tmp_path / "reg"
    reg_root.mkdir()
    elsewhere = _write(tmp_path, "secret", _BASE)

    class _Leaky(_Registries):
        def find_recipe_in_registries(self, name, include_hidden=False):
            return [(self.name, elsewhere)]

    outer = _write(tmp_path, "variant", {"include": "@reg/secret"})
    with pytest.raises(RecipeError, match="outside its registry"):
        Recipe.load(outer, registry_manager=_Leaky(reg_root))


def test_sibling_symlink_cannot_escape_the_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside, "private", _BASE)
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "base.yaml").symlink_to(outside / "private.yaml")
    outer = _write(recipes, "variant", {"include": "base"})
    with pytest.raises(RecipeError, match="outside its registry or directory"):
        Recipe.load(outer)


def test_sibling_symlink_within_the_directory_is_fine(tmp_path):
    _write(tmp_path, "real", _BASE)
    (tmp_path / "base.yaml").symlink_to(tmp_path / "real.yaml")
    assert Recipe.load(_write(tmp_path, "variant", {"include": "base"})).model == _BASE["model"]


def test_changing_the_model_drops_inherited_metadata(tmp_path):
    """Architecture numbers describe the base's model and would size the wrong one."""
    _write(tmp_path, "base", _BASE)
    same = Recipe.load(_write(tmp_path, "same", {"include": "base"}))
    other = Recipe.load(_write(tmp_path, "other", {"include": "base", "model": "Qwen/Qwen3-8B"}))
    assert same.metadata.get("model_params") == "1.7B"
    assert "model_params" not in other.metadata


def test_compose_style_include_list_is_not_a_recipe(tmp_path):
    compose = tmp_path / "compose.yaml"
    compose.write_text(yaml.safe_dump({"include": ["other.yaml"], "services": {"a": {"image": "x"}}}))
    assert not is_recipe_file(compose)
