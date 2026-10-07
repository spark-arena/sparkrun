"""Minimum core versions gate execution, not recipe inspection or export."""

from copy import deepcopy
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
import pickle
from types import SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner
import pytest
import yaml

from sparkrun.core import version
from sparkrun.core.recipe import Recipe, RecipeError


DATA = {"model": "test/model", "runtime": "sglang", "container": "test/image"}


def recipe(minimum="0.4.0"):
    return Recipe.from_dict({**DATA, "min_sparkrun_version": minimum})


@pytest.mark.parametrize("minimum", ["", " ", ">=0.4.0", "0.4.*", "latest", True, 0.4, 4, [], {}])
def test_invalid_minimum_is_a_recipe_error(minimum):
    with pytest.raises(RecipeError, match="min_sparkrun_version must be a version string"):
        recipe(minimum)


@pytest.mark.parametrize(
    "installed,minimum",
    [
        ("0.4.0", "0.4.0"),
        ("0.4.1", "0.4.0"),
        ("0.10.0", "0.9.0"),
        ("0.4.0+g123abc", "0.4.0"),
        ("0.4.0.post1", "0.4.0"),
        ("0.4.0rc2", "0.4.0rc1"),
        ("0.4.0", "0.4.0rc1"),
        ("0.5.0.dev1", "0.4.0"),
    ],
)
def test_compatible_versions(installed, minimum, monkeypatch):
    monkeypatch.setattr(version, "base_version", lambda package: installed)
    model = recipe(minimum)
    version.require_recipe_version(model)
    assert model.validate_structure() == []


@pytest.mark.parametrize(
    "installed,minimum",
    [
        ("0.3.99", "0.4.0"),
        ("0.9.0", "0.10.0"),
        ("0.4.0rc1", "0.4.0"),
        ("0.4.0.dev1", "0.4.0"),
        ("0.4.0", "0.4.0.post1"),
    ],
)
def test_incompatible_versions_report_required_installed_and_upgrade(installed, minimum, monkeypatch):
    monkeypatch.setattr(version, "base_version", lambda package: installed)
    with pytest.raises(ValueError, match="requires Sparkrun") as caught:
        version.require_recipe_version(recipe(minimum))
    message = str(caught.value)
    assert "installed version is " + installed in message
    assert "Sparkrun >= " + minimum in message
    assert "sparkrun setup update" in message


@pytest.mark.parametrize("data", [DATA, {**DATA, "min_sparkrun_version": None}])
def test_optional_field_does_not_require_package_metadata(data, monkeypatch):
    metadata = Mock(side_effect=AssertionError("no constraint needs no lookup"))
    monkeypatch.setattr(version, "base_version", metadata)
    model = Recipe.from_dict(data)
    version.require_recipe_version(model)
    assert model.validate_structure() == []
    assert "min_sparkrun_version" not in model.to_dict()
    metadata.assert_not_called()


def test_unparseable_installed_version_fails_closed(monkeypatch):
    monkeypatch.setattr(version, "base_version", lambda package: "unknown-build")
    with pytest.raises(ValueError, match="Cannot verify min_sparkrun_version.*Upgrade"):
        version.require_recipe_version(recipe())


def test_missing_metadata_does_not_bypass_requirement(monkeypatch):
    monkeypatch.setattr(version, "version", Mock(side_effect=PackageNotFoundError))
    with pytest.raises(ValueError, match="installed version is 0.0.0-dev"):
        version.require_recipe_version(recipe())


def test_compares_core_metadata_not_application_or_display_version(monkeypatch):
    monkeypatch.setattr(version, "get_application_profile", lambda: SimpleNamespace(package="branded-app", command="branded"))
    lookup = Mock(side_effect=lambda package: "0.3.0" if package == "sparkrun" else "99.0.0")
    monkeypatch.setattr(version, "version", lookup)
    monkeypatch.setattr(version, "display_version", Mock(side_effect=AssertionError("display is not machine comparison")))
    with pytest.raises(ValueError, match="branded setup update"):
        version.require_recipe_version(recipe())
    lookup.assert_called_once_with("sparkrun")


def test_requirement_survives_export_serialization_and_older_state(monkeypatch):
    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    model = recipe()
    for restored in (
        Recipe.from_dict(model.to_dict()),
        Recipe.from_dict(yaml.safe_load(model.export())),
        Recipe._deserialize_yaml(model._serialize_yaml()),
        pickle.loads(pickle.dumps(model)),
        deepcopy(model),
    ):
        assert restored.min_sparkrun_version == "0.4.0"
        assert "min_sparkrun_version" not in restored.runtime_config
        assert restored.to_dict()["min_sparkrun_version"] == "0.4.0"
        with pytest.raises(ValueError, match="requires Sparkrun"):
            version.require_recipe_version(restored)
    state = model.__getstate__()
    state.pop("min_sparkrun_version")
    assert Recipe._deserialize(state).min_sparkrun_version == "0.4.0"  # recover older raw source
    state["_raw"].pop("min_sparkrun_version")
    assert Recipe._deserialize(state).min_sparkrun_version is None


def test_include_inherits_minimum_and_flattened_export_preserves_it(tmp_path, monkeypatch):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({**DATA, "min_sparkrun_version": "0.4.0"}))
    outer = tmp_path / "derived.yaml"
    outer.write_text("include: base\ndefaults:\n  port: 9000\n")
    model = Recipe.load(outer)
    assert model.min_sparkrun_version == "0.4.0"
    assert model.to_dict()["min_sparkrun_version"] == "0.4.0"
    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    with pytest.raises(ValueError, match="requires Sparkrun"):
        version.require_recipe_version(model)


def test_validation_reports_incompatible_version_as_error(monkeypatch):
    from sparkrun.core.validation import ERROR, validate_recipe
    from sparkrun.runtimes.sglang import SglangRuntime

    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    findings = validate_recipe(recipe(), runtime=SglangRuntime())
    issue = next(issue for issue in findings if "min_sparkrun_version" in issue.message)
    assert issue.severity == ERROR
    assert "setup update" in issue.message


@pytest.mark.parametrize("dry_run", [False, True])
def test_api_plan_fails_before_transport_preparation(dry_run, monkeypatch):
    import sparkrun.api as api

    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    prepare = Mock(side_effect=AssertionError("must reject before transport preparation"))
    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", prepare)
    with pytest.raises(api.SparkrunError, match="requires Sparkrun >= 0.4.0"):
        api.plan(api.RunOptions(recipe=recipe(), hosts=("h1",), dry_run=dry_run))
    prepare.assert_not_called()


def test_supplied_plan_is_rechecked_before_strategy_preparation(monkeypatch):
    import sparkrun.api as api
    from test_api_materialize import _fixture

    options, plan, _ = _fixture()
    plan.recipe.min_sparkrun_version = "0.4.0"
    monkeypatch.setattr(version, "base_version", lambda package: "0.4.0")
    version.require_recipe_version(plan.recipe)  # valid when planned
    serialized = Recipe._deserialize_yaml(plan.recipe._serialize_yaml())
    plan = replace(plan, recipe=serialized)
    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    strategy = Mock(side_effect=AssertionError("must reject before plugin preparation"))
    launch = Mock(side_effect=AssertionError("must reject before launch/replacement"))
    monkeypatch.setattr("sparkrun.core.execution.resolve_recipe_execution", strategy)
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    with pytest.raises(api.SparkrunError, match="requires Sparkrun >= 0.4.0"):
        api.run(options, plan=plan)
    strategy.assert_not_called()
    launch.assert_not_called()


def test_materialize_rechecks_supplied_plan(monkeypatch):
    import sparkrun.api as api
    from test_api_materialize import _fixture

    options, plan, sctx = _fixture()
    plan.recipe.min_sparkrun_version = "0.4.0"
    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    with pytest.raises(ValueError, match="requires Sparkrun >= 0.4.0"):
        api.materialize(options, plan=plan, sctx=sctx)


def test_direct_launcher_rejects_before_any_runtime_or_config_work(monkeypatch):
    from sparkrun.core.launcher import launch_inference

    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    with pytest.raises(ValueError, match="requires Sparkrun >= 0.4.0"):
        launch_inference(recipe=recipe(), runtime=None, host_list=["h1"], overrides={}, config=None)


@pytest.mark.parametrize("command", [["recipe", "validate"], ["run", "--hosts", "localhost"]])
def test_cli_rejects_with_upgrade_guidance(command, tmp_path, monkeypatch):
    from sparkrun.cli import main

    path = tmp_path / "future.yaml"
    path.write_text(yaml.safe_dump({**DATA, "min_sparkrun_version": "0.4.0"}))
    monkeypatch.setattr(version, "base_version", lambda package: "0.3.0")
    with patch("sparkrun.core.launcher.launch_inference") as launch:
        result = CliRunner().invoke(main, [*command, str(path)])
    assert result.exit_code != 0
    assert "requires Sparkrun" in result.output
    assert "setup update" in " ".join(result.output.split())
    launch.assert_not_called()
