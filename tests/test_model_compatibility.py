"""Legacy means automatic metadata validation with bounded compatibility fallback."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from sparkrun.core import model_distribution as md
from sparkrun.core.recipe import Recipe
from sparkrun.models import acquisition, preparation
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest, ModelInventoryUnavailable
from sparkrun.models.manifest import manifest_path
from sparkrun.orchestration import distribution
from test_model_artifacts import artifact, cache, DATA, SHA


@pytest.fixture
def inventory(monkeypatch):
    manifest = artifact()
    fetch = Mock(return_value={"revision": SHA, "files": [asdict(f) for f in manifest.files]})
    monkeypatch.setattr(preparation, "fetch_inventory", fetch)
    return fetch


@md.model_distribution_operation
def prepare(root, config=None, *, revision=None, offline=False, dry_run=False, mode="local", source=None):
    return distribution._prepare_manifest_model(
        "org/model", ["localhost"], str(root), str(source or root), mode, {}, revision, None, dry_run, offline=offline
    )


@pytest.mark.parametrize("policy", [None, "legacy", "metadata"])
@pytest.mark.parametrize("entrypoint", ["resources", "recipe"])
def test_public_default_paths_automatically_migrate_cache(tmp_path, monkeypatch, inventory, policy, entrypoint):
    cache(tmp_path, artifact())
    settings = {} if policy is None else {"model_cache_validation": policy}
    config = SimpleNamespace(cache_dir=tmp_path, get=settings.get)
    download = Mock(side_effect=AssertionError("manifest-backed preparation must not use legacy downloader"))
    monkeypatch.setattr("sparkrun.models.download.download_model", download)
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a: {})
    monkeypatch.setattr("sparkrun.containers.registry.ensure_image", lambda *a, **kw: 0)
    monkeypatch.setattr(distribution, "_get_hf_token", lambda: None)
    if entrypoint == "resources":
        distribution.distribute_resources("test:tag", "org/model", ["localhost"], str(tmp_path), config, dry_run=False)
    else:
        recipe = Recipe.from_dict({"model": "org/model", "runtime": "sglang", "container": "test:tag"})
        distribution.distribute_from_config(recipe, "test:tag", ["localhost"], str(tmp_path), config, dry_run=False)
    inventory.assert_called_once()
    for revision in (None, "main", SHA):
        saved = manifest_path(str(tmp_path), ModelArtifactRequest("org/model", revision))
        assert ModelArtifactManifest.from_json(saved.read_text()).identity == artifact().identity
    download.assert_not_called()


@pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull"])
def test_default_migrates_every_transfer_mode(tmp_path, monkeypatch, inventory, mode):
    cache(tmp_path, artifact())
    remote = Mock(return_value=inventory.return_value)
    monkeypatch.setattr(preparation, "run_adapter", remote)
    assert prepare(tmp_path, mode=mode) == []
    assert (remote if mode in {"delegated", "pull"} else inventory).call_count == 1
    assert manifest_path(str(tmp_path), ModelArtifactRequest("org/model", SHA)).exists()


def test_push_persists_inventory_on_controller_and_target(tmp_path, inventory):
    source, target = tmp_path / "source", tmp_path / "target"
    cache(source, artifact())
    assert prepare(target, mode="push", source=source) == []
    for root in (source, target):
        for revision in (None, SHA):
            assert manifest_path(str(root), ModelArtifactRequest("org/model", revision)).exists()
    assert (target / artifact().relative_snapshot / "tokenizer.json").read_bytes() == DATA["tokenizer.json"]


@pytest.mark.parametrize("revision", [None, "main", SHA])
def test_saved_manifest_is_used_offline_and_reobserved(tmp_path, inventory, revision):
    snapshot = cache(tmp_path, artifact())
    assert prepare(tmp_path) == []
    inventory.side_effect = AssertionError("offline requested metadata")
    assert prepare(tmp_path, revision=revision, offline=True) == []
    (snapshot / "tokenizer.json").unlink()
    failures = prepare(tmp_path, revision=revision, offline=True)
    assert failures and "tokenizer.json" in failures[0].detail
    assert "offline" in failures[0].detail


def test_pinned_inventory_reused_online_without_hub(tmp_path, inventory):
    cache(tmp_path, artifact())
    assert prepare(tmp_path) == []
    inventory.side_effect = AssertionError("pinned inventory was re-fetched")
    assert prepare(tmp_path, revision=SHA) == []


def test_mutable_revision_refreshes_when_available(tmp_path, inventory):
    cache(tmp_path, artifact())
    assert prepare(tmp_path) == []
    latest = replace(artifact(), revision="b" * 40)
    cache(tmp_path, latest)
    inventory.return_value = {"revision": latest.revision, "files": [asdict(f) for f in latest.files]}
    assert prepare(tmp_path) == []
    saved = manifest_path(str(tmp_path), ModelArtifactRequest("org/model"))
    assert ModelArtifactManifest.from_json(saved.read_text()).revision == latest.revision
    assert inventory.call_count == 2


def test_default_inventory_selection_excludes_redundant_weights(tmp_path, inventory):
    snapshot = cache(tmp_path, artifact())
    (snapshot / "model.safetensors.index.json").write_bytes(b"{}")
    inventory.return_value["files"].extend(
        [{"path": "model.safetensors.index.json", "size": 2}, {"path": "metal/model.bin", "size": 1_000_000_000}]
    )
    inventory.return_value["weight_map"] = {"tensor": "model.safetensors"}
    assert prepare(tmp_path) == []
    saved = ModelArtifactManifest.from_json(manifest_path(str(tmp_path), ModelArtifactRequest("org/model")).read_text())
    assert saved.selection == "safetensors-index-v1"
    assert "metal/model.bin" not in {f.path for f in saved.files}


def test_metadata_outage_uses_saved_inventory_but_never_excuses_known_damage(tmp_path, monkeypatch, inventory, caplog):
    snapshot = cache(tmp_path, artifact())
    assert prepare(tmp_path) == []
    inventory.side_effect = ModelInventoryUnavailable("Hub unavailable")
    assert prepare(tmp_path) == []
    assert "validating saved revision" in caplog.text
    (snapshot / "tokenizer.json").write_bytes(b"bad size")
    monkeypatch.setattr(acquisition, "repair_snapshot_files", Mock(side_effect=ModelArtifactError("repair unavailable")))
    failures = prepare(tmp_path)
    assert failures and "repair unavailable" in failures[0].detail
    assert "using existing cache/download checks" not in caplog.text


@pytest.mark.parametrize("policy", ["metadata", "checksum"])
def test_strict_mutable_refresh_failure_never_uses_compatibility(tmp_path, inventory, policy):
    cache(tmp_path, artifact())
    assert prepare(tmp_path) == []
    inventory.side_effect = ModelInventoryUnavailable("Hub unavailable")
    failures = prepare(tmp_path, SimpleNamespace(get={"model_cache_validation": policy}.get))
    assert failures and "Hub unavailable" in failures[0].detail


def test_failed_first_repair_still_saves_inventory_for_next_offline_run(tmp_path, monkeypatch, inventory, caplog):
    snapshot = cache(tmp_path, artifact())
    (snapshot / "tokenizer.json").unlink()
    repair = Mock(side_effect=ModelArtifactError("repair unavailable"))
    monkeypatch.setattr(acquisition, "repair_snapshot_files", repair)
    failures = prepare(tmp_path)
    assert failures and "repair unavailable" in failures[0].detail
    assert manifest_path(str(tmp_path), ModelArtifactRequest("org/model")).exists()
    inventory.side_effect = AssertionError("offline origin request")
    failures = prepare(tmp_path, offline=True)
    assert failures and "tokenizer.json" in failures[0].detail
    assert "using existing cache/download checks" not in caplog.text
    repair.assert_called_once()


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("policy", [None, "legacy", "metadata", "checksum"])
def test_only_legacy_may_fall_back_without_inventory(tmp_path, inventory, caplog, offline, policy):
    cache(tmp_path, artifact())
    inventory.side_effect = ModelInventoryUnavailable("Hub unavailable")
    config = None if policy is None else SimpleNamespace(get={"model_cache_validation": policy}.get)
    failures = prepare(tmp_path, config, offline=offline)
    if policy in {None, "legacy"}:
        assert failures is None
        assert "unverified" in caplog.text and "legacy compatibility" in caplog.text
    else:
        assert failures and failures[0].host == "localhost"
    assert not manifest_path(str(tmp_path), ModelArtifactRequest("org/model")).exists()
    if offline:
        inventory.assert_not_called()


def test_provider_never_receives_unverified_fallback(tmp_path, inventory):
    cache(tmp_path, artifact())
    inventory.side_effect = ModelInventoryUnavailable("Hub unavailable")
    config = SimpleNamespace(get={"model_distribution_provider": "example"}.get)
    failures = prepare(tmp_path, config)
    assert failures and "Hub unavailable" in failures[0].detail


def test_malformed_manifest_never_downgrades(tmp_path, inventory):
    cache(tmp_path, artifact())
    saved = manifest_path(str(tmp_path), ModelArtifactRequest("org/model"))
    saved.parent.mkdir()
    saved.write_text("not a manifest")
    failures = prepare(tmp_path, offline=True)
    assert failures and "manifest" in failures[0].detail
    inventory.assert_not_called()


def test_invalid_authoritative_selection_never_downgrades(tmp_path, inventory):
    cache(tmp_path, artifact())
    assert prepare(tmp_path) == []
    inventory.return_value["files"].append({"path": "model.safetensors.index.json", "size": 5})
    inventory.return_value["weight_map"] = {"tensor": "missing.safetensors"}
    failures = prepare(tmp_path)
    assert failures and "missing or invalid shards" in failures[0].detail


def test_compatibility_dry_run_neither_probes_nor_fetches(tmp_path, monkeypatch, inventory):
    monkeypatch.setattr(preparation.ModelHostIO, "__init__", Mock(side_effect=AssertionError("dry run probed hosts")))
    assert prepare(tmp_path, dry_run=True) == []
    inventory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "output,unavailable",
    [
        (b"", True),
        (b"SPARKRUN_MODEL_ADAPTER_READY\nSPARKRUN_MODEL_ERROR=unavailable\n", True),
        (b"SPARKRUN_MODEL_ADAPTER_READY\n", False),
        (b"SPARKRUN_MODEL_ERROR=invalid\n", False),
    ],
)
def test_remote_errors_preserve_fallback_boundary(output, unavailable, repair):
    io = Mock()
    io.execute.return_value = SimpleNamespace(returncode=1, stdout=output, stderr=b"failure with secret-token")
    with pytest.raises(ModelArtifactError) as error:
        acquisition.run_adapter(
            io, "head", ModelArtifactRequest("org/model"), "/hf", token="secret-token", manifest=artifact() if repair else None
        )
    assert isinstance(error.value, ModelInventoryUnavailable) == (unavailable and not repair)
    assert "secret-token" not in str(error.value)


@pytest.mark.parametrize("network", [False, True])
def test_generated_remote_adapter_classifies_actual_errors(monkeypatch, capsys, network):
    exception = httpx.ConnectError("metadata unavailable") if network else ValueError("invalid inventory")
    api = Mock()
    api.model_info.side_effect = exception
    monkeypatch.setattr("huggingface_hub.HfApi", Mock(return_value=api))
    shell = acquisition.hub_script(ModelArtifactRequest("org/model"), "/hf", token=None)
    program = shell.split("<<'SPARKRUN_HUB_ADAPTER'\n", 1)[1].rsplit("\nSPARKRUN_HUB_ADAPTER", 1)[0]
    with pytest.raises(SystemExit):
        exec(compile(program, "remote-adapter", "exec"), {})
    assert "SPARKRUN_MODEL_ERROR=" + ("unavailable" if network else "invalid") in capsys.readouterr().out


@pytest.mark.parametrize("network", [False, True])
def test_controller_index_errors_preserve_fallback_boundary(monkeypatch, network):
    from sparkrun.models import manifest

    api = Mock()
    api.model_info.return_value = SimpleNamespace(
        sha=SHA,
        siblings=[SimpleNamespace(rfilename="model.safetensors.index.json", size=2, lfs=None, blob_id="b" * 40)],
    )
    monkeypatch.setattr("huggingface_hub.HfApi", Mock(return_value=api))
    error = httpx.ConnectError("Hub unavailable") if network else ValueError("index checksum mismatch")
    monkeypatch.setattr(manifest, "read_weight_index", Mock(side_effect=error))
    with pytest.raises(ModelArtifactError) as result:
        manifest.fetch_inventory(ModelArtifactRequest("org/model"))
    assert isinstance(result.value, ModelInventoryUnavailable) == network


def test_preplaced_cache_uses_configured_path_and_retains_model_identity(tmp_path):
    for name, data in DATA.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / ".sparkrun-model-manifest.json").write_text(artifact().to_json())
    bindings = preparation.validate_preplaced("org/model", ["localhost"], model_path=str(tmp_path))
    assert bindings[0].model == "org/model"
    assert bindings[0].model_path == str(tmp_path)
