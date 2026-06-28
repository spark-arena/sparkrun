from __future__ import annotations

from sparkrun.telemetry.util import recipe_source


def test_recipe_source_classifies_string_references():
    assert recipe_source("@spark-arena/qwen")["kind"] == "spark_arena"
    assert recipe_source("https://example.test/recipe.yaml")["kind"] == "url"
    assert recipe_source("./recipe.yaml")["kind"] == "file"
    assert recipe_source("private-recipe")["kind"] == "reference"
