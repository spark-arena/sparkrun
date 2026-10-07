from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner

from sparkrun.api import model_cache
from sparkrun.models.acquisition import repair_snapshot_files
from sparkrun.models.artifacts import ModelArtifactRequest
from test_model_artifacts import artifact, cache, DATA, SHA


def test_import_inspect_then_detect_deletion(tmp_path):
    request = ModelArtifactRequest("org/model", SHA)
    assert model_cache.inspect_model(request, cache_dir=str(tmp_path))[0].status == "unverified"
    manifest = artifact()
    snapshot = cache(tmp_path, manifest)
    assert model_cache.import_manifest(request, manifest, cache_dir=str(tmp_path), checksums=True)[0].complete
    (snapshot / "config.json").unlink()
    report = model_cache.inspect_model(request, cache_dir=str(tmp_path))[0]
    assert [(f.path, f.reason) for f in report.failures] == [("config.json", "missing")]


def test_import_incomplete_does_not_claim_success(tmp_path):
    request = ModelArtifactRequest("org/model", SHA)
    result = model_cache.import_manifest(request, artifact(), cache_dir=str(tmp_path))
    assert result[0].status == "incomplete"
    assert model_cache.inspect_model(request, cache_dir=str(tmp_path))[0].status == "incomplete"


def test_repair_regular_snapshot_does_not_unlink_shared_blob(tmp_path, monkeypatch):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest, shared=True)
    blob = (snapshot / "model.safetensors").resolve()
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").write_bytes(b"invalid")

    def download(repo, name, **kwargs):
        assert kwargs["revision"] == SHA
        path = Path(kwargs["local_dir"]) / name
        path.write_bytes(DATA[name])
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    repair_snapshot_files(manifest.repo, SHA, str(tmp_path), manifest.endpoint, None, ("model.safetensors",))
    assert (snapshot / "model.safetensors").read_bytes() == DATA["model.safetensors"]
    assert blob.read_bytes() == DATA["model.safetensors"]
    assert not list(snapshot.parent.glob(".sparkrun-repair-*"))


def test_cli_inspect_is_offline_and_nonzero_when_unverified(tmp_path, monkeypatch):
    from sparkrun.cli._model_cache import model_cache as cli
    from sparkrun.cli import _model_cache

    monkeypatch.setattr(_model_cache, "_get_context", lambda ctx: Mock(config=Mock(ssh_user=None, ssh_key=None, ssh_options=None)))
    monkeypatch.setattr("sparkrun.models.manifest.fetch_inventory", Mock(side_effect=AssertionError("network")))
    result = CliRunner().invoke(cli, ["inspect", "org/model", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert '"status": "unverified"' in result.output


def test_small_index_is_checked_against_typed_inventory(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from sparkrun.models.manifest import read_weight_index

    raw = json.dumps({"weight_map": {"tensor": "model.safetensors"}}).encode()

    def download(*args, **kwargs):
        p = Path(kwargs["local_dir"]) / "model.safetensors.index.json"
        p.write_bytes(raw)
        return str(p)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    entry = {
        "path": "model.safetensors.index.json",
        "size": len(raw),
        "checksum_algorithm": "git-sha1",
        "checksum": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
    }
    assert read_weight_index("org/model", SHA, "https://huggingface.co", None, entry) == {"tensor": "model.safetensors"}
    entry["checksum"] = "0" * 40
    with pytest.raises(ValueError, match="checksum"):
        read_weight_index("org/model", SHA, "https://huggingface.co", None, entry)
