"""All acquisition topologies share immutable selection and real file copies."""

from dataclasses import replace
from pathlib import Path

import pytest

from sparkrun.models import preparation, transport
from sparkrun.models.host_io import ModelHostIO
from test_model_artifacts import artifact, cache, DATA


@pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull"])
def test_all_modes_repair_only_missing_files_and_preserve_source(tmp_path, monkeypatch, mode):
    manifest = artifact()
    locations = {None: tmp_path / "control", "head": tmp_path / "head", "worker": tmp_path / "worker"}
    source = cache(locations[None], manifest, shared=True)
    head = cache(locations["head"], manifest, shared=True)
    worker = cache(locations["worker"], manifest)
    (worker / "tokenizer.json").unlink()
    repairs = []

    class LocalHosts(ModelHostIO):
        def root(self, host, path):
            return str(locations[host])

        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    original = transport.copy_script

    def local_copy(request, target, ssh_kwargs, **kwargs):
        return original(replace(request, source_host=None), replace(target, host="localhost"), {}, **kwargs)

    def acquire(io, host, request, root, **kwargs):
        repairs.append((host, kwargs["files"]))
        assert kwargs["manifest"].identity == manifest.identity
        for name in kwargs["files"]:
            (Path(root) / manifest.relative_snapshot / name).write_bytes(DATA[name])

    monkeypatch.setattr(preparation, "ModelHostIO", LocalHosts)
    monkeypatch.setattr(transport, "copy_script", local_copy)
    monkeypatch.setattr(preparation, "run_adapter", acquire)
    prepared = preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=manifest, transfer_mode=mode)
    assert len(prepared.bindings) == 2
    assert all(b.report.complete for b in prepared.bindings)
    assert (source / "model.safetensors").is_symlink()
    assert (head / "model.safetensors").is_symlink()
    assert (worker / "tokenizer.json").read_bytes() == DATA["tokenizer.json"]
    assert repairs == ([("worker", ("tokenizer.json",))] if mode == "pull" else [])
