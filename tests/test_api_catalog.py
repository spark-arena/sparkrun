from pathlib import Path

import pytest
import yaml

from sparkrun import api
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint


@pytest.fixture
def catalog(tmp_path):
    sctx = api.default_sctx()
    root = sctx.config.config_path.parent / "recipes"
    root.mkdir(parents=True)
    data = {
        "recipe_version": "2",
        "model": "test/model",
        "runtime": "sglang",
        "container": "test/image:latest",
        "defaults": {"tensor_parallel": 1, "port": 30000},
        "metadata": {"model_params": 1000000},
    }
    for name in ["one/same", "two/same"]:
        path = root / (name + ".yaml")
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(data))
    sctx.cluster_manager.create("lab", ["10.0.0.1"])
    sctx.cluster_manager.set_default("lab")
    return sctx, root, data


def test_catalog_preserves_same_stem_identity_and_paginates(catalog, monkeypatch):
    sctx, root, _ = catalog
    monkeypatch.setattr(sctx.registry_manager, "ensure_initialized", lambda: pytest.fail("search initialized registries"))
    first = api.catalog_recipes(limit=1, sctx=sctx)
    second = api.catalog_recipes(offset=1, limit=1, sctx=sctx)
    assert first["total"] == 2
    assert first["next_offset"] == 1
    assert second["next_offset"] is None
    assert first["recipes"][0]["reference"] != second["recipes"][0]["reference"]
    chosen = second["recipes"][0]
    detail = api.get_recipe_details(chosen["reference"], sctx=sctx)
    assert detail["source_path"] == str(root / "two/same.yaml")
    assert detail["model"] == "test/model"
    assert detail["native_protocols"]
    assert api.list_clusters(sctx=sctx)[0]["default"] is True


def test_details_and_launch_share_image_env_override_fingerprint(catalog):
    sctx, root, _ = catalog
    overrides = {"image": "test/new:latest", "env.TEST": "hello", "tensor_parallel": "1", "max_model_len": "4096"}
    detail = api.get_recipe_details(str(root / "one/same.yaml"), overrides, sctx=sctx)
    recipe, normalized = api.resolve_catalog_recipe(detail["reference"], overrides, sctx=sctx)
    assert recipe.container == "test/new:latest"
    assert recipe.env["TEST"] == "hello"
    assert normalized["max_model_len"] == 4096
    assert "image" not in normalized
    assert detail["recipe_revision"] == derive_recipe_fingerprint(recipe, normalized)
    assert "env" not in detail and "container" not in detail


def test_selection_detects_changed_or_deleted_file(catalog):
    sctx, root, data = catalog
    detail = api.get_recipe_details(str(root / "one/same.yaml"), sctx=sctx)
    data["model"] = "test/changed"
    (root / "one/same.yaml").write_text(yaml.safe_dump(data))
    changed = api.get_recipe_details(detail["reference"], sctx=sctx)
    assert changed["recipe_revision"] != detail["recipe_revision"]
    (root / "one/same.yaml").unlink()
    with pytest.raises(api.RecipeNotFound):
        api.get_recipe_details(detail["reference"], sctx=sctx)


def test_import_is_not_implicitly_trusted(catalog):
    sctx, _, data = catalog
    data["pre_exec"] = ["echo must-be-reviewed"]
    detail = api.import_recipe(yaml.safe_dump(data), sctx=sctx)
    assert detail["trusted"] is False
    assert any(issue["code"] == "recipe_trust_required" for issue in detail["issues"])
    recipe, _ = api.resolve_catalog_recipe(detail["reference"], sctx=sctx)
    assert recipe.is_url_sourced
    assert Path(detail["source_path"]).is_file()


def test_invalid_selection_and_page_are_rejected(catalog):
    sctx, _, _ = catalog
    with pytest.raises(api.SparkrunError):
        api.catalog_recipes(limit=500, sctx=sctx)
    with pytest.raises(api.RecipeNotFound):
        api.get_recipe_details("https://example.invalid/recipe.yaml", sctx=sctx)


def test_abandoned_imports_expire_but_committed_imports_remain(catalog):
    import os
    import time

    sctx, _, data = catalog
    abandoned = api.import_recipe(yaml.safe_dump(data), sctx=sctx)
    data["model"] = "test/retained"
    retained = api.import_recipe(yaml.safe_dump(data), sctx=sctx)
    api.retain_catalog_recipe(retained["reference"], sctx=sctx)
    for detail in [abandoned, retained]:
        old = time.time() - 8 * 86400
        os.utime(detail["source_path"], (old, old))
    assert api.cleanup_catalog_imports(sctx=sctx) == 1
    assert Path(retained["source_path"]).is_file()
    assert not Path(abandoned["source_path"]).exists()


def test_import_path_cannot_bypass_trust_and_auxiliary_files_are_explicit(catalog):
    sctx, _, data = catalog
    data["mods"] = ["relative-patch"]
    detail = api.import_recipe(yaml.safe_dump(data), sctx=sctx)
    reopened = api.get_recipe_details(detail["source_path"], sctx=sctx)
    assert not reopened["trusted"]
    assert any(issue["code"] == "import_auxiliary_files" for issue in reopened["issues"])


def test_unknown_plugin_item_blocks_unattended_preview(catalog):
    sctx, _, data = catalog
    data["missing_plugin"] = {"artifact": "test"}
    detail = api.import_recipe(yaml.safe_dump(data), sctx=sctx)
    assert any(issue["code"] == "unknown-top-level-key" and issue["severity"] == "error" for issue in detail["issues"])


def test_recipe_path_with_spaces_uses_short_canonical_reference(catalog):
    sctx, root, data = catalog
    path = root / "a path with spaces.yaml"
    path.write_text(yaml.safe_dump(data))
    detail = api.get_recipe_details(str(path), sctx=sctx)
    assert len(detail["reference"]) < 256 and " " not in detail["reference"]
    assert api.get_recipe_details(detail["reference"], sctx=sctx)["source_path"] == str(path)


def test_preview_preserves_hf_model_and_served_name_default(catalog):
    sctx, root, data = catalog
    data["defaults"]["served_model_name"] = "coding-default"
    (root / "one/same.yaml").write_text(yaml.safe_dump(data))
    detail = api.get_recipe_details(str(root / "one/same.yaml"), sctx=sctx)
    assert detail["defaults"]["served_model_name"] == "coding-default"
    assert detail["hf_model"] == "test/model"
    assert detail["model"] == "coding-default"


def test_facets_only_use_declared_metadata_and_filter_before_pagination(catalog):
    sctx, root, data = catalog
    data["metadata"]["quantization"] = "fp8"
    data["defaults"]["max_model_len"] = 32768
    (root / "one/same.yaml").write_text(yaml.safe_dump(data))
    result = api.catalog_recipes(filters={"quantization": "fp8", "context_length": "32768"}, limit=1, sctx=sctx)
    assert result["total"] == 1
    assert result["recipes"][0]["source_path"] == str(root / "one/same.yaml")
    assert result["facets"]["quantization"] == ["fp8", "unknown"]
    assert api.catalog_recipes(filters={"quantization": "unknown"}, sctx=sctx)["total"] == 1
    with pytest.raises(api.SparkrunError):
        api.catalog_recipes(filters={"guessed_model_size": "32"}, sctx=sctx)


def test_registry_add_never_grants_trust_or_clones(catalog, monkeypatch):
    sctx, _, _ = catalog
    monkeypatch.setattr(sctx.registry_manager, "_clone_or_pull", lambda *_: pytest.fail("registry configuration cloned"))
    api.configure_registry("add", "private-test", url="https://github.com/example/recipes.git", sctx=sctx)
    entry = sctx.registry_manager.get_registry("private-test")
    assert not entry.trusted
    with pytest.raises(api.SparkrunError):
        api.configure_registry("trust", "private-test", sctx=sctx)
    api.configure_registry("trust", "private-test", acknowledge_trust=True, sctx=sctx)
    assert sctx.registry_manager.get_registry("private-test").trusted
    api.configure_registry("disable", "private-test", sctx=sctx)
    assert not sctx.registry_manager.get_registry("private-test").enabled
    api.configure_registry("remove", "private-test", sctx=sctx)
    assert not any(r["name"] == "private-test" for r in api.list_registries(sctx=sctx))


def test_capacity_keeps_unreachable_hosts_unknown(catalog, monkeypatch):
    from sparkrun.core.cluster_status import ClusterStatus

    sctx, _, _ = catalog
    monkeypatch.setattr("sparkrun.api._status.status", lambda *_, **__: ClusterStatus())
    result = api.catalog_cluster_capacity("lab", sctx=sctx)
    assert result["advisory"] is True
    assert result["hosts"][0]["reachable"] is False
    assert result["hosts"][0]["free_slots"] is None


@pytest.mark.parametrize("runtime", ["vllm", "vllm-ray", "eugr-vllm"])
def test_vllm_native_api_options_and_version_defaults(catalog, runtime):
    sctx, root, data = catalog
    data["runtime"] = runtime
    path = root / "one/same.yaml"
    data["container"] = "vllm/vllm-openai:v0.12.0"
    path.write_text(yaml.safe_dump(data))
    details = api.get_recipe_details(str(path), sctx=sctx)
    assert details["native_api_options"] == ["chat_completions", "responses", "messages"]
    assert details["native_protocols"] == ["openai", "anthropic"]
    assert "responses" in details["capabilities"]
    data["container"] = "custom/image:latest"
    path.write_text(yaml.safe_dump(data))
    unknown = api.get_recipe_details(str(path), sctx=sctx)
    assert unknown["native_protocols"] == ["openai", "anthropic"]
    assert "responses" in unknown["capabilities"]
    data["metadata"]["native_apis"] = ["chat_completions"]
    path.write_text(yaml.safe_dump(data))
    declared = api.get_recipe_details(str(path), sctx=sctx)
    assert declared["native_protocols"] == ["openai"]
    assert "responses" not in declared["capabilities"]


def test_plugin_annotations_survive_catalog_without_changing_launch_revision(catalog, passive_recipe_item):
    sctx, root, data = catalog
    path = root / "one/same.yaml"
    original = api.get_recipe_details(str(path), sctx=sctx)
    data[passive_recipe_item] = {"labels": ["vision"], "options": {"low": {"temperature": 0.2}}}
    path.write_text(yaml.safe_dump(data))
    changed = api.get_recipe_details(str(path), sctx=sctx)
    assert changed["plugin_items"] == {passive_recipe_item: data[passive_recipe_item]}
    assert changed["recipe_revision"] == original["recipe_revision"]
    assert "tests.annotations" in changed["required_plugins"]
    assert not any(issue["code"] == "unknown-top-level-key" for issue in changed["issues"])


def test_plugin_parse_failures_use_catalog_error_surface(catalog, passive_recipe_item):
    from sparkrun.core.recipe import RecipeError

    sctx, root, data = catalog
    path = root / "one/same.yaml"
    data[passive_recipe_item] = []
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(api.SparkrunError, match="Recipe is invalid") as error:
        api.get_recipe_details(str(path), sctx=sctx)
    assert isinstance(error.value.__cause__, RecipeError)
