"""Manifest completeness, typed checksums, and real filesystem transfer semantics."""

import hashlib
import os
import subprocess
from dataclasses import replace

import pytest

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactFile, ModelArtifactManifest, ModelArtifactRequest
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.manifest import check_request, manifest_path, select_manifest
from sparkrun.models.observation import observation_script, parse_observations
from sparkrun.models.preparation import prepare_model

SHA = "a" * 40
DATA = {"config.json": b"{}", "model.safetensors": b"weights", "tokenizer.json": b"tokenizer"}


def artifact(data=DATA):
    return ModelArtifactManifest(
        "org/model",
        SHA,
        "https://huggingface.co",
        "full-snapshot-v1",
        tuple(ModelArtifactFile(name, len(value), "sha256", hashlib.sha256(value).hexdigest()) for name, value in sorted(data.items())),
    )


def cache(root, manifest, data=DATA, *, shared=False):
    snapshot = root / manifest.relative_snapshot
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, contents in data.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if shared:
            digest = hashlib.sha256(contents).hexdigest()
            backing = root / "hub/blobs" / digest[:2] / digest
            backing.parent.mkdir(parents=True, exist_ok=True)
            backing.write_bytes(contents)
            blob = root / "hub/models--org--model/blobs" / digest
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.symlink_to(os.path.relpath(backing, blob.parent))
            path.symlink_to(os.path.relpath(blob, path.parent))
        else:
            path.write_bytes(contents)
    return snapshot


def observe(manifest, snapshot, checksums=False):
    result = subprocess.run(
        ["bash"], input=observation_script(manifest, str(snapshot), checksums=checksums), text=True, capture_output=True
    )
    return parse_observations("local", manifest, result.stdout, returncode=result.returncode, checksums=checksums)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a//b", "a/", "", "a\0b"])
def test_manifest_rejects_noncanonical_paths(path):
    with pytest.raises(ModelArtifactError):
        ModelArtifactFile(path, 2)


def test_manifest_roundtrip_and_selection_identity():
    manifest = artifact()
    assert ModelArtifactManifest.from_json(manifest.to_json()) == manifest
    assert replace(manifest, revision="b" * 40).identity != manifest.identity
    with pytest.raises(ModelArtifactError, match="unique"):
        replace(manifest, files=manifest.files + manifest.files)


@pytest.mark.parametrize("shared", [False, True])
def test_expected_missing_file_is_detected_even_with_good_weights(tmp_path, shared):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest, shared=shared)
    (snapshot / "tokenizer.json").unlink()
    report = observe(manifest, snapshot)
    assert not report.complete
    assert [(f.path, f.reason) for f in report.failures] == [("tokenizer.json", "missing")]


def test_dangling_shards_and_file_types(tmp_path):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to("/not/a/blob")
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").mkdir()
    reasons = {f.path: f.reason for f in observe(manifest, snapshot).failures}
    assert reasons == {"config.json": "not_regular", "model.safetensors": "missing"}


def test_typed_git_checksum_and_unusual_filename(tmp_path):
    data = b"not an LFS blob"
    digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    name = "nested/newline\nquote' tab\t.bin"
    manifest = ModelArtifactManifest(
        "org/model", SHA, "https://huggingface.co", "full-snapshot-v1", (ModelArtifactFile(name, len(data), "git-sha1", digest),)
    )
    snapshot = cache(tmp_path, manifest, {name: data})
    assert observe(manifest, snapshot, True).complete
    assert (snapshot / name).read_bytes() == data


def test_same_size_corruption_requires_checksum_level(tmp_path):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest)
    (snapshot / "model.safetensors").write_bytes(b"corrupt")
    assert observe(manifest, snapshot).complete
    report = observe(manifest, snapshot, True)
    assert not report.complete
    assert report.failures[0].reason == "checksum_mismatch"


def test_malformed_observation_does_not_claim_missing_or_complete():
    manifest = artifact()
    assert parse_observations("x", manifest, "0\tok\n").status == "unavailable"


def test_safetensors_index_excludes_alternatives_but_keeps_ancillary_data():
    paths = [
        "model.safetensors.index.json",
        "model-00000-of-00002.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "original/model.safetensors",
        "metal/model.bin",
        "config.json",
        "tokenizer.bin",
    ]
    inventory = {"revision": SHA, "files": [dict(path=p, size=10) for p in paths], "weight_map": {"a": paths[2], "b": paths[3]}}
    request = ModelArtifactRequest("org/model")
    manifest = select_manifest(request, inventory)
    assert {f.path for f in manifest.files} == {paths[0], paths[2], paths[3], "config.json", "tokenizer.bin"}
    assert manifest.selection == "safetensors-index-v1"
    check_request(ModelArtifactManifest.from_json(manifest.to_json()), request)
    assert len(select_manifest(replace(request, file_selection="all"), inventory).files) == len(paths)
    with pytest.raises(ModelArtifactError, match="selection"):
        check_request(manifest, replace(request, file_selection="all"))


def test_safetensors_index_cannot_reference_an_absent_shard():
    with pytest.raises(ModelArtifactError, match="missing"):
        select_manifest(
            ModelArtifactRequest("org/model"),
            {
                "revision": SHA,
                "files": [dict(path="model.safetensors.index.json", size=10)],
                "weight_map": {"weight": "missing.safetensors"},
            },
        )


def test_missing_gguf_part_one_cannot_select_part_two():
    with pytest.raises(ModelArtifactError, match="incomplete"):
        select_manifest(
            ModelArtifactRequest("org/model-GGUF:Q4"), {"revision": SHA, "files": [dict(path="Q4/model-Q4-00002-of-00002.gguf", size=10)]}
        )


@pytest.mark.parametrize("checksums", [False, True])
def test_real_copy_repairs_only_required_files_and_preserves_shared_source(tmp_path, checksums):
    source, target = tmp_path / "source", tmp_path / "target"
    manifest = artifact()
    source_snapshot = cache(source, manifest, shared=True)
    target_snapshot = cache(target, manifest)
    if checksums:
        (target_snapshot / "model.safetensors").write_bytes(b"corrupt")
    else:
        (target_snapshot / "model.safetensors").unlink()
    # No manifest entry means no transfer of this unrelated large variant.
    (source_snapshot / "unused-variant.bin").write_bytes(b"unused")
    outcome = prepare_model(
        "org/model",
        ["localhost"],
        cache_dir=str(target),
        local_cache_dir=str(source),
        revision=SHA,
        offline=True,
        checksums=checksums,
        manifest=manifest,
    )
    assert outcome is not None and outcome.bindings[0].report.complete
    assert (source_snapshot / "model.safetensors").is_symlink()
    assert (target_snapshot / "model.safetensors").read_bytes() == b"weights"
    assert not (target_snapshot / "unused-variant.bin").exists()
    assert manifest_path(str(target), ModelArtifactRequest("org/model", SHA)).exists()


def test_saved_manifest_still_detects_later_deletion(tmp_path):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest)
    prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), revision=SHA, offline=True, manifest=manifest)
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(ModelArtifactError, match="offline.*tokenizer"):
        prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), revision=SHA, offline=True)


def test_offline_manifestless_cache_is_unverified(tmp_path):
    cache(tmp_path, artifact())
    with pytest.raises(ModelArtifactError, match="unverified"):
        prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), revision=SHA, offline=True)


def test_shared_cache_host_aliases_do_not_mutate_source(tmp_path):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest, shared=True)
    prepared = prepare_model(
        "org/model",
        ["localhost", "127.0.0.1"],
        cache_dir=str(tmp_path),
        revision=SHA,
        transfer_mode="delegated",
        offline=True,
        manifest=manifest,
    )
    assert len(prepared.bindings) == 2
    assert (snapshot / "model.safetensors").is_symlink()
    assert (snapshot / "model.safetensors").read_bytes() == b"weights"


def test_dry_run_neither_probes_nor_creates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelHostIO, "__init__", lambda *a, **k: pytest.fail("dry run probed hosts"))
    assert prepare_model("org/model", ["bad-host"], cache_dir=str(tmp_path / "absent"), dry_run=True) is None
    assert not (tmp_path / "absent").exists()


def test_preplaced_manifest_is_required_for_validated_binding(tmp_path, caplog):
    from sparkrun.models.preparation import validate_preplaced

    assert validate_preplaced(str(tmp_path), ["localhost"]) == ()
    assert "unverified" in caplog.text
    manifest = artifact()
    for name, data in DATA.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / ".sparkrun-model-manifest.json").write_text(manifest.to_json())
    bindings = validate_preplaced(str(tmp_path), ["localhost"], checksums=True)
    assert bindings[0].model_path == str(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ModelArtifactError, match="incomplete"):
        validate_preplaced(str(tmp_path), ["localhost"])


def test_gguf_auto_projector_is_bound_to_same_inventory():
    request = ModelArtifactRequest("org/model:Q8_0")
    files = [{"path": p, "size": 1} for p in ["model-Q8_0.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf"]]
    manifest = select_manifest(request, {"revision": SHA, "files": files})
    assert manifest.weight_path == "model-Q8_0.gguf"
    assert manifest.projector_path == "mmproj-F16.gguf"


def test_index_keeps_unknown_nested_components():
    request = ModelArtifactRequest("org/model")
    inventory = {
        "revision": SHA,
        "files": [
            {"path": p, "size": 1}
            for p in [
                "model.safetensors.index.json",
                "model-00000.safetensors",
                "vision/model.safetensors",
                "tokenizer.bin",
                "metal/model.bin",
            ]
        ],
        "weight_map": {"parameter": "model-00000.safetensors"},
    }
    manifest = select_manifest(request, inventory)
    assert {f.path for f in manifest.files} == {
        "model.safetensors.index.json",
        "model-00000.safetensors",
        "vision/model.safetensors",
        "tokenizer.bin",
    }
