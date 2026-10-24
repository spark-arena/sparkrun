"""Offline checks for the bundled TensorFold recipes.

The recipes ship as examples for a hardware run, so their job here is to be
loadable and structurally valid: parse, match a known runtime, and pass that
runtime's blocking validation (model, parallelism, bundle path) without a host
list.  A drift in the runtime's config surface shows up as a recipe that no
longer validates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparkrun.core.recipe import Recipe
from sparkrun.core.validation import ERROR
from sparkrun.runtimes.tensorfold import TensorFoldRuntime
from sparkrun.runtimes.tensorfold_dsv41 import TensorFoldDSV41Runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE_DIR = REPO_ROOT / "recipes"

_RUNTIMES = {
    "tensorfold": TensorFoldRuntime,
    "tensorfold-dsv41": TensorFoldDSV41Runtime,
}


def _recipe_files() -> list[Path]:
    return sorted(RECIPE_DIR.glob("tensorfold-*.yaml"))


def test_recipes_exist():
    assert _recipe_files(), "expected tensorfold recipe examples under recipes/"


@pytest.mark.parametrize("path", _recipe_files(), ids=lambda p: p.name)
def test_recipe_parses_and_validates(path: Path):
    data = yaml.safe_load(path.read_text())
    assert data["recipe_version"] == "2"
    recipe = Recipe.from_dict(data)

    runtime_name = data["runtime"]
    assert runtime_name in _RUNTIMES, "recipe names an unknown runtime: %s" % runtime_name
    runtime = _RUNTIMES[runtime_name]()

    issues = [issue for issue in runtime.validate_recipe(recipe) if getattr(issue, "severity", None) == ERROR]
    assert issues == [], "%s has blocking runtime issues: %s" % (path.name, [str(i) for i in issues])


@pytest.mark.parametrize("path", _recipe_files(), ids=lambda p: p.name)
def test_recipe_declares_model_and_single_image_plan(path: Path):
    data = yaml.safe_load(path.read_text())
    assert data.get("model"), "recipe has no model"
    # Every example either relies on the platform default image or names one
    # explicitly; both are valid, but a concrete `containers:` block would make
    # a recipe host-specific and these examples are portable.
    assert "containers" not in data


def test_tp_recipes_carry_a_family_and_rank_count():
    for name, family, world in (
        ("tensorfold-glm-5.3-flash-tp2.yaml", "glm", 2),
        ("tensorfold-nemotron-3.5-super-vl-tp2.yaml", "nemotron", 2),
    ):
        data = yaml.safe_load((RECIPE_DIR / name).read_text())
        defaults = data["defaults"]
        assert defaults["tf_family"] == family
        assert defaults["tensor_parallel"] == world
        assert data["min_nodes"] == world == data["max_nodes"]


def test_dsv41_recipes_bundle_is_absolute_and_world_matches_nodes():
    for name, world in (
        ("tensorfold-dsv41-flash-tp2.yaml", 2),
        ("tensorfold-dsv41-flash-tp4.yaml", 4),
    ):
        data = yaml.safe_load((RECIPE_DIR / name).read_text())
        assert data["model"].startswith("/"), "the DeepSeek lane needs an absolute prepared bundle path"
        assert data["defaults"]["tensor_parallel"] == world
        assert data["min_nodes"] == world == data["max_nodes"]
