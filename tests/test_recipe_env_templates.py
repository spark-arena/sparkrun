"""One recipe env mapping preserves interpolation and literal override semantics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from sparkrun.core.env_templates import render_env_template, canonical_env, select_env_templates
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.core.recipe_include import merge_recipe_data
from sparkrun.core.resolve import apply_env_overrides, apply_recipe_overrides
from sparkrun.core.validation import check_managed_cache_env, check_managed_comm_env, check_unknown_top_level_keys
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint


def _recipe(**values):
    return Recipe({"model": "org/model", "runtime": "sglang", "container": "image:test", **values})


def rendered(recipe, **config):
    return {**recipe.env, **{key: render_env_template(value, config, {"node_rank": 2}) for key, value in recipe.env_templates.items()}}


def test_one_env_block_recognizes_only_reserved_tokens(monkeypatch):
    monkeypatch.setenv("CONTROL_SECRET", "must-not-expand")
    env = {
        "PORT": "{config.port}",
        "RANK": "{launch.node_rank}",
        "JSON": '{"a":{"b":1},"port":"{config.port}"}',
        "SHELL": "$CONTROL_SECRET/${HOME}/${config.port}",
        "BRACES": "{other} {{json}} {env.CONTROL_SECRET} } {",
        "LITERAL": "{{launch.node_rank}}",
    }
    recipe = _recipe(env=env)
    assert recipe.env == env
    assert set(recipe.env_templates) == {"PORT", "RANK", "JSON", "LITERAL"}
    actual = rendered(recipe, port=9123)
    assert actual["PORT"] == "9123" and actual["RANK"] == "2"
    assert actual["JSON"] == '{"a":{"b":1},"port":"9123"}'
    assert actual["LITERAL"] == "{launch.node_rank}"
    assert actual["SHELL"] == env["SHELL"] and actual["BRACES"] == env["BRACES"]
    assert check_unknown_top_level_keys(recipe, None) == []
    assert "env_templates" not in recipe.to_dict()


def test_separate_unreleased_field_is_rejected():
    with pytest.raises(RecipeError, match="put templates directly in env"):
        _recipe(env_templates={"PORT": "{config.port}"})


@pytest.mark.parametrize("name", ["", "1VALUE", "SPACE KEY", "BAD=KEY", "BAD-KEY", "BAD.KEY", "BÄD", 1])
def test_template_names_must_be_portable_env_names(name):
    with pytest.raises(RecipeError, match="environment variable names"):
        _recipe(env={name: "{config.port}"})


@pytest.mark.parametrize(
    "template", ["{launch.unknown}", "{config.port!r}", "{config.port:04}", "{config.port[0]}", "{config.port.attr}", "{launch.model_path"]
)
def test_invalid_reserved_token_fails_without_echoing_value(template):
    with pytest.raises(RecipeError, match="env.VALUE") as error:
        _recipe(env={"VALUE": "private-prefix:" + template})
    assert "private-prefix" not in str(error.value)


def test_structure_rechecks_mutations_and_conditional_env():
    recipe = _recipe(env={"PORT": "{config.port}"}, overrides=[{"when": {"config": {"port": 9000}}, "env": {"OTHER": "{launch.unknown}"}}])
    recipe.env["PORT"] = "9000"
    assert any("env.OTHER" in issue for issue in recipe.validate_structure())


def test_structure_does_not_guess_unavailable_launch_values():
    assert _recipe(env={"MODEL_PATH": "{launch.model_path}", "FUTURE": "{config.runtime_default}"}).validate_structure() == []


@pytest.mark.parametrize("transport", ["deepcopy", "state", "yaml", "export", "dict"])
def test_templates_round_trip_without_rendering(transport):
    recipe = _recipe(env={"MODEL_PATH": "{launch.model_path}", "JSON": '{"port":"{config.port}"}', "LITERAL": "{{launch.model_path}}"})
    recipe.snapshot_declared_values()
    if transport == "deepcopy":
        restored = deepcopy(recipe)
    elif transport == "state":
        restored = Recipe._deserialize(recipe.__getstate__())
    elif transport == "yaml":
        restored = Recipe._deserialize_yaml(recipe._serialize_yaml())
    elif transport == "export":
        restored = Recipe(yaml.safe_load(recipe.export()))
    else:
        restored = Recipe(recipe.to_dict())
    assert restored.env == recipe.env
    assert restored.env_templates == recipe.env_templates
    restored.env["MODEL_PATH"] = "changed"
    assert recipe.env["MODEL_PATH"] == "{launch.model_path}"


def test_older_saved_state_remains_loadable():
    state = _recipe().__getstate__()
    state["_declared"] = {"defaults": {}, "env": {}, "container": "image:test"}
    restored = Recipe._deserialize(state)
    restored.restore_declared_values()
    assert restored.env_templates == {}
    assert "env_templates" not in restored.to_dict()


def test_snapshot_keeps_declared_templates_for_relaunch():
    recipe = _recipe(env={"VALUE": "{config.value}"})
    recipe.snapshot_declared_values()
    recipe.env["VALUE"] = "{config.changed}"
    clone = Recipe._deserialize_yaml(recipe._serialize_yaml())
    assert clone.to_dict()["env"] == {"VALUE": "{config.value}"}
    assert clone.to_dict(effective=True)["env"] == {"VALUE": "{config.changed}"}
    clone.restore_declared_values()
    assert clone.env_templates == {"VALUE": "{config.value}"}


@pytest.mark.parametrize(
    "value",
    [
        "{config.port}",
        "{{config.port}}",
        "{{{config.port}}}",
        "{launch.unknown}",
        "{config.unclosed",
        "{{config.unclosed",
        "{config.x[0]}",
        '{"x":"{config.port}"}',
        "${config.port}",
        "",
    ],
)
@pytest.mark.parametrize("cli", ["env", "override"])
def test_cli_literals_survive_export_reload_and_fingerprints(value, cli):
    recipe = _recipe(env={"VALUE": "{launch.node_rank}"})
    if cli == "env":
        apply_env_overrides(recipe, ["VALUE=" + value])
    else:
        apply_recipe_overrides(("env.VALUE=" + value,), recipe=recipe)
    recipe.snapshot_declared_values()
    restored = Recipe(yaml.safe_load(recipe.export()))
    assert rendered(recipe)["VALUE"] == value
    assert rendered(restored)["VALUE"] == value
    assert derive_recipe_fingerprint(recipe) == derive_recipe_fingerprint(restored)
    clone = Recipe._deserialize_yaml(recipe._serialize_yaml())
    assert rendered(clone)["VALUE"] == value
    assert derive_recipe_fingerprint(clone) == derive_recipe_fingerprint(recipe)


def test_template_and_same_text_cli_literal_have_distinct_identity():
    template = _recipe(env={"VALUE": "{launch.node_rank}"})
    literal = deepcopy(template)
    apply_env_overrides(literal, ["VALUE={launch.node_rank}"])
    assert template.env == literal.env
    assert derive_recipe_fingerprint(template) != derive_recipe_fingerprint(literal)


def test_late_cli_override_does_not_reinterpret_declared_template():
    recipe = _recipe(env={"PORT": "{config.port}"})
    recipe.snapshot_declared_values()
    apply_env_overrides(recipe, ["PORT=9000"])
    assert recipe.to_dict()["env"] == {"PORT": "{config.port}"}
    assert recipe.to_dict(effective=True)["env"] == {"PORT": "9000"}


def test_conditional_env_is_automatically_detected_after_resolution():
    from sparkrun.core.recipe_overrides import OverrideContext, apply_recipe_override_layers

    recipe = _recipe(env={"VALUE": "literal"}, overrides=[{"when": {"config": {"use_rank": True}}, "env": {"VALUE": "{launch.node_rank}"}}])
    context = OverrideContext(shape={}, runtime="sglang", runtime_family="sglang", config={"use_rank": True}, hosts={})
    apply_recipe_override_layers(recipe, context)
    assert rendered(recipe)["VALUE"] == "2"
    context.config["use_rank"] = False
    apply_recipe_override_layers(recipe, context)
    assert rendered(recipe)["VALUE"] == "literal"


def test_include_env_templates_replace_and_null_deletes(tmp_path):
    base = {
        "model": "org/model",
        "runtime": "sglang",
        "env": {"MODEL": "{launch.model_path}", "PORT": "{config.port}", "REMOVE": "literal"},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"include": "base", "env": {"PORT": "9000", "REMOVE": None}}))
    recipe = Recipe.load(child, resolve=False)
    assert recipe.env == {"MODEL": "{launch.model_path}", "PORT": "9000"}
    assert recipe.env_templates == {"MODEL": "{launch.model_path}"}
    merged = merge_recipe_data({"env": {"KEY": {"old": 1}}}, {"env": {"KEY": {"new": 2}}})
    assert merged["env"]["KEY"] == {"new": 2}


def test_templated_managed_communication_keys_are_reported():
    issues = check_managed_comm_env(_recipe(env={"NCCL_IB_HCA": "{config.hca}"}))
    assert len(issues) == 1
    assert issues[0].message.startswith("env: sets NCCL_IB_HCA")


def _cache_runtime():
    return SimpleNamespace(
        runtime_name="test",
        get_extra_env=lambda: {"HF_HOME": "/cache/huggingface"},
        runtime_cache_paths=lambda: {"TRITON_CACHE_DIR": "triton"},
    )


def test_templates_cannot_silently_replace_managed_model_cache_env():
    issues = check_managed_cache_env(_recipe(env={"HF_HOME": "{config.other}"}), _cache_runtime())
    assert [issue.code for issue in issues] == ["overridden-cache-env"]


@pytest.mark.parametrize("template", ["{launch.runtime_cache_dir}", "{launch.runtime_cache_dir}/triton", "{launch.runtime_cache_dir}/a/"])
def test_templates_under_resolved_runtime_cache_are_not_reported_as_moving_cache(template):
    assert check_managed_cache_env(_recipe(env={"TRITON_CACHE_DIR": template}), _cache_runtime()) == []


@pytest.mark.parametrize("template", ["{config.other}", "{launch.runtime_cache_dir}/../outside", "{{launch.runtime_cache_dir}}/triton"])
def test_templates_that_can_move_runtime_cache_keep_diagnostic(template):
    issues = check_managed_cache_env(_recipe(env={"TRITON_CACHE_DIR": template}), _cache_runtime())
    assert [issue.code for issue in issues] == ["managed-cache-env"]


def test_canonical_literal_encoding_uses_only_reserved_tokens():
    env = {"VALUE": '{"key":"$HOME/{other}/{config.port}"}'}
    source = canonical_env(env, {"VALUE"})
    assert source == {"VALUE": '{"key":"$HOME/{other}/{{config.port}}"}'}
    assert select_env_templates(source) == source
    assert render_env_template(source["VALUE"], {}, {}) == env["VALUE"]
