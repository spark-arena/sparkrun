"""Environment interpolation at the prepared-assets and runtime boundaries."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparkrun.core import env_templates as et
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.scripts.resolve_model_path import resolve_model_path

SHA = "a" * 40
OTHER_SHA = "b" * 40


def recipe(**kw):
    return Recipe.from_dict({"model": "org/model", "runtime": "sglang", "defaults": {"port": 8000}, **kw})


def snapshot(cache: Path, sha=SHA, repo="org/model", ref="main"):
    root = cache / "hub" / ("models--" + repo.replace("/", "--"))
    path = root / "snapshots" / sha
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "weights.safetensors").write_bytes(b"fixture")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
    if ref:
        target = root / "refs" / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sha)
    return path


def test_renderer_is_explicit_single_pass_data_only(monkeypatch):
    monkeypatch.setenv("SECRET", "must-not-read")
    value = "quote' \"\n$SECRET $(touch ignored) {launch.node_rank}"
    assert (
        et.render_env_template("{json} {config.value} {launch.node_rank}", {"value": value}, {"node_rank": 2}) == "{json} " + value + " 2"
    )
    assert et.render_env_template("$SECRET", {}, {}) == "$SECRET"
    assert et.validate_env_template("{config.port}/{launch.node_rank}") == {"port"}


@pytest.mark.parametrize(
    "template", ["{config.x.y}", "{config.x[0]}", "{launch.unknown}", "{config.x!r}", "{config.x:02d}", "{launch.unclosed"]
)
def test_invalid_templates_rejected(template):
    with pytest.raises(ValueError):
        et.render_env_template(template, {"x": 1}, {})


@pytest.mark.parametrize("value", [None, {}, [], "\x00"])
def test_invalid_config_values_rejected(value):
    with pytest.raises(ValueError):
        et.render_env_template("{config.x}", {"x": value}, {})


def test_probe_exact_ref_and_pinned_commit_without_ref(tmp_path):
    path = snapshot(tmp_path, ref="releases/stable")
    assert resolve_model_path("org/model", "releases/stable", str(tmp_path)) == {"path": str(path), "revision": SHA}
    assert resolve_model_path("org/model", SHA, str(tmp_path))["path"] == str(path)
    with pytest.raises(ValueError, match="no cached ref"):
        resolve_model_path("org/model", "", str(tmp_path))
    with pytest.raises(ValueError, match="not present"):
        resolve_model_path("org/model", OTHER_SHA, str(tmp_path))


def test_probe_uses_selected_commit_on_every_node(tmp_path):
    first = snapshot(tmp_path, ref="main")
    snapshot(tmp_path, OTHER_SHA, ref="main")
    assert resolve_model_path("org/model", "", str(tmp_path), SHA)["path"] == str(first)
    first.rename(first.with_name("removed"))
    with pytest.raises(ValueError, match="not present"):
        resolve_model_path("org/model", "", str(tmp_path), SHA)


def test_probe_accepts_hf_blob_symlinks_but_rejects_missing_weights(tmp_path):
    path = snapshot(tmp_path)
    blob = tmp_path / "blob"
    (path / "weights.safetensors").rename(blob)
    (path / "weights.safetensors").symlink_to(blob)
    assert resolve_model_path("org/model", "", str(tmp_path))["revision"] == SHA
    blob.unlink()
    with pytest.raises(ValueError, match="missing or invalid"):
        resolve_model_path("org/model", "", str(tmp_path))


def test_probe_rejects_index_path_escape(tmp_path):
    path = snapshot(tmp_path)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../bad"}}))
    with pytest.raises(ValueError, match="invalid weight"):
        resolve_model_path("org/model", "", str(tmp_path))


def test_probe_gguf_selects_one_complete_quantization(tmp_path):
    path = snapshot(tmp_path, repo="org/model-GGUF")
    for name in ["model-Q4-00001-of-00002.gguf", "model-Q4-00002-of-00002.gguf", "model-Q8.gguf", "mmproj.gguf"]:
        (path / name).touch()
    assert resolve_model_path("org/model-GGUF:Q4", "", str(tmp_path))["path"].endswith("model-Q4-00001-of-00002.gguf")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model_path("org/model-GGUF", "", str(tmp_path))
    (path / "model-Q4-00002-of-00002.gguf").unlink()
    with pytest.raises(ValueError, match="missing shards"):
        resolve_model_path("org/model-GGUF:Q4", "", str(tmp_path))


def test_remote_probe_executes_as_data_with_quoted_paths(tmp_path, monkeypatch):
    cache = tmp_path / "cache ' $stuff"
    path = snapshot(cache)
    calls = []

    def remote(host, script, **kwargs):
        calls.append(host)
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=5)
        return RemoteResult(host, result.returncode, result.stdout, result.stderr)

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", remote)
    assert et.probe_model_path("worker", "org/model", "", str(cache), {}) == et.PreparedModelPath(str(path), SHA)
    assert calls == ["worker"]


def test_prepare_probes_only_requested_fields_and_pins_all_nodes(monkeypatch):
    calls = []

    def probe(host, model, revision, cache_dir, ssh_kwargs, *, expected_revision=""):
        calls.append((host, model, expected_revision))
        return et.PreparedModelPath("/hf/hub/snapshots/" + SHA, SHA)

    monkeypatch.setattr(et, "probe_model_path", probe)
    r = recipe(env={"M": "{launch.model_path}", "N": "{launch.node_rank}"})
    plan = et.prepare_env_templates(r, {}, ["z", "a", "z"], "cid", "/hf", {})
    assert calls == [("z", "org/model", ""), ("a", "org/model", SHA)]
    assert plan.hosts == ("z", "a")
    r.env["M"] = "{literal CLI value}"
    calls.clear()
    plan = et.prepare_env_templates(r, {}, ["z"], "cid", "/hf", {})
    assert not calls
    assert plan.templates == {"N": "{launch.node_rank}"}
    r.env["N"] = "override"
    assert et.prepare_env_templates(r, {}, ["z"], "cid", "/hf", {}) is None


def test_gguf_command_override_does_not_replace_prepared_model(monkeypatch):
    seen = []
    monkeypatch.setattr(et, "probe_model_path", lambda h, m, *a, **k: seen.append(m) or et.PreparedModelPath("/hf/file.gguf", SHA))
    r = recipe(model="org/model-GGUF:Q4", env={"M": "{launch.model_path}"})
    et.prepare_env_templates(
        r, {"model": "/cache/huggingface/file.gguf", "_gguf_model_path": "/cache/huggingface/file.gguf"}, ["h"], "id", "/hf", {}
    )
    assert seen == ["org/model-GGUF:Q4"]
    with pytest.raises(ValueError, match="prepared recipe.model"):
        et.prepare_env_templates(r, {"model": "other/model"}, ["h"], "id", "/hf", {})


@pytest.mark.parametrize(
    "extra",
    [
        ["-v", "/other:/cache/huggingface"],
        ["--mount=type=bind,src=/x,dst=/cache/runtime"],
        ["--tmpfs", "/cache"],
        ["--volumes-from=c"],
        ["-vmy-cache:/cache/runtime"],
    ],
)
def test_path_templates_refuse_unstructured_mounts(extra):
    with pytest.raises(ValueError, match="structured executor_config.volumes"):
        et.prepare_env_templates(
            recipe(env={"M": "{launch.model_path}"}), {}, ["h"], "id", "/hf", {}, dry_run=True, extra_docker_opts=extra
        )


def test_model_mount_mapping_uses_longest_source_and_detects_shadow():
    assert et.workload_model_path("/hf/model/snapshot", {"/hf": "/cache", "/hf/model": "/model:ro"}, "docker") == "/model/snapshot"
    assert et.workload_model_path("/hf/model", {}, "local") == "/hf/model"
    with pytest.raises(ValueError, match="shadowed"):
        et.workload_model_path("/hf/model", {"/hf": "/cache", "/other": "/cache/model"}, "docker")
    with pytest.raises(ValueError, match="not visible"):
        et.workload_model_path("/missing/model", {"/hf": "/cache"}, "docker")


def test_per_node_render_is_reused_and_context_cannot_change():
    p = et.EnvTemplatePlan(
        {"M": "{launch.model_path}", "N": "{launch.node_rank}/{launch.num_nodes}"},
        {},
        ("z", "a"),
        "id",
        models={h: et.PreparedModelPath("/hf/model", SHA) for h in ["z", "a"]},
    )
    volumes = {"/hf": "/cache/huggingface"}
    assert p.render("a", volumes, None, DockerExecutor()) == {"M": "/cache/huggingface/model", "N": "1/2"}
    assert p.render("z", volumes, None, DockerExecutor())["N"] == "0/2"
    with pytest.raises(ValueError, match="context changed"):
        p.render("a", {"/hf": "/changed"}, None, DockerExecutor())


def test_runtime_cache_resolves_same_mount_and_local_host_path():
    cache = SimpleNamespace(leaf="/host/compile", volumes={"/host/compile": "/cache/runtime"})
    p = et.EnvTemplatePlan({"C": "{launch.runtime_cache_dir}/mod"}, {}, ("h",), "id")
    assert p.render("h", cache.volumes, cache, DockerExecutor()) == {"C": "/cache/runtime/mod"}
    assert replace(p, rendered={}, contexts={}).render("h", cache.volumes, cache, LocalExecutor()) == {"C": "/host/compile/mod"}
    for volumes in [{}, {**cache.volumes, "/other": "/cache/runtime"}, {**cache.volumes, "/other": "/cache/runtime/mod"}]:
        with pytest.raises(ValueError, match="mount"):
            p.render("h", volumes, cache, DockerExecutor())
    with pytest.raises(ValueError, match="enabled runtime cache"):
        p.render("h", {}, None, DockerExecutor())


def test_dry_run_never_probes_and_leaves_dynamic_model_fields_explicit(monkeypatch):
    monkeypatch.setattr(et, "probe_model_path", lambda *a, **k: pytest.fail("dry run must not probe"))
    r = recipe(env={"M": "{launch.model_path}", "R": "{launch.model_revision}", "P": "{config.port}"})
    p = et.prepare_env_templates(r, {"port": 9000}, ["h"], "id", "/hf", {}, dry_run=True)
    assert p.render("h", {}, None, DockerExecutor()) == {
        "M": "<unresolved:launch.model_path>",
        "R": "<unresolved:launch.model_revision>",
        "P": "9000",
    }


def test_scoped_context_is_reset_after_failure():
    r = recipe(env={"N": "{launch.node_rank}"})
    p = et.prepare_env_templates(r, {}, ["h"], "id", "/hf", {})
    with pytest.raises(RuntimeError), et.env_template_scope(p):
        assert et.render_launch_env(r, "h", {}, None, DockerExecutor()) == {"N": "0"}
        raise RuntimeError("fixture")
    with pytest.raises(ValueError, match="prepared context"):
        et.render_launch_env(r, "h", {}, None, DockerExecutor())


def test_fingerprint_tracks_templates_without_materialized_paths():
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    r = recipe(env={"N": "{launch.node_rank}"})
    fingerprint = derive_recipe_fingerprint(r)
    p = et.prepare_env_templates(r, {}, ["h"], "id", "/hf", {})
    p.render("h", {}, None, DockerExecutor())
    assert derive_recipe_fingerprint(r) == fingerprint
    assert derive_recipe_fingerprint(recipe(env={"N": "{launch.num_nodes}"})) != fingerprint


@pytest.mark.parametrize("kind", ["solo", "sglang", "vllm-distributed", "vllm-ray", "llama-cpp", "sglang-hybrid"])
def test_create_and_serve_receive_same_rank_environment(kind, monkeypatch):
    from _runtime_fixtures import StubRuntime
    from sparkrun.core.runtime_cache import build_runtime_cache_mounts, RuntimeCacheSettings
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.runtimes import _cluster_ops
    from sparkrun.runtimes.sglang import SglangRuntime
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
    from sparkrun.runtimes.vllm_ray import VllmRayRuntime
    from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

    factories = {
        "solo": StubRuntime,
        "sglang": SglangRuntime,
        "vllm-distributed": VllmDistributedRuntime,
        "vllm-ray": VllmRayRuntime,
        "llama-cpp": LlamaCppRuntime,
    }
    runtime = factories[kind.removesuffix("-hybrid")]()
    hybrid = kind.endswith("-hybrid")
    hosts = ["z-head"] if kind == "solo" else ["z-head", "a-worker", "b-worker", "c-worker"] if hybrid else ["z-head", "a-worker"]
    r = recipe(
        runtime=runtime.runtime_name,
        defaults={"port": 9000, "tensor_parallel": 2 if hybrid else len(hosts), "data_parallel": 2 if hybrid else 1},
        env={
            "LITERAL": "{{launch.node_rank}}",
            "RANK": "{launch.node_rank}",
            "MODEL_PATH": "{launch.model_path}",
            "MOD_CACHE": "{launch.runtime_cache_dir}/mod",
        },
        pre_exec=["echo prepare"],
    )
    cache = build_runtime_cache_mounts(runtime=runtime, recipe=r, settings=RuntimeCacheSettings(), root="/compile", image="fixture")
    plan = et.EnvTemplatePlan(r.env_templates, {}, tuple(hosts), "id", models={h: et.PreparedModelPath("/hf/model", SHA) for h in hosts})
    creation, serving, hooks = [], [], []
    original_create = DockerExecutor.run_cmd
    original_serve = DockerExecutor.generate_exec_serve_script

    def create(self, *a, **kw):
        creation.append(kw)
        return original_create(self, *a, **kw)

    def serve(self, *a, **kw):
        serving.append(kw)
        return original_serve(self, *a, **kw)

    def remote(host, script, **kw):
        if script.startswith("docker exec --user root"):
            from test_hook_environment import _docker_exec

            hooks.append((host, _docker_exec(script)))
        return RemoteResult(host, 0, "10.0.0.1\n", "")

    monkeypatch.setattr(DockerExecutor, "run_cmd", create)
    monkeypatch.setattr(DockerExecutor, "generate_exec_serve_script", serve)
    for name in ("cleanup_ranked_containers", "cleanup_named_containers"):
        monkeypatch.setattr(_cluster_ops, name, lambda *a, **k: None)
    monkeypatch.setattr(_cluster_ops, "find_port", lambda ctx, h, p: p)
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda ctx: "10.0.0.1")
    monkeypatch.setattr(_cluster_ops, "resolve_hosts_for_init", lambda ctx, ip: ctx.hosts)
    monkeypatch.setattr(_cluster_ops, "detect_ib_with_ips", lambda *a, **k: (ClusterCommEnv.empty(), {}, {}))
    monkeypatch.setattr(_cluster_ops, "resolve_comm_env", lambda *a, **k: ClusterCommEnv.empty())
    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", remote)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", remote)
    monkeypatch.setattr("sparkrun.orchestration.primitives.detect_host_ip", lambda *a, **k: "10.0.0.1")
    monkeypatch.setattr("sparkrun.orchestration.primitives.wait_for_port", lambda *a, **k: True)
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **k: {})

    with et.env_template_scope(plan):
        assert (
            runtime.run(
                hosts=hosts,
                image="fixture",
                serve_command="echo serve",
                recipe=r,
                overrides={},
                cluster_id="id",
                env=r.env,
                cache_dir="/hf",
                dry_run=True,
                trust=True,
                comm_env=ClusterCommEnv.empty(),
                runtime_cache=cache,
                backends={},
            )
            == 0
        )
    assert len(creation) == len(hosts)
    assert {c["env"]["RANK"] for c in creation} == {str(i) for i in range(len(hosts))}
    for call in creation + serving:
        assert call["env"]["MODEL_PATH"] == "/cache/huggingface/model"
        assert call["env"]["MOD_CACHE"] == "/cache/runtime/mod"
        assert call["env"]["LITERAL"] == "{launch.node_rank}"
    assert serving
    for host, (cname, overlay, _) in hooks:
        parent = next(c for c in creation if c["container_name"] == cname)
        assert overlay["SPARKRUN_NODE_RANK"] == parent["env"]["RANK"] == str(hosts.index(host))
        assert overlay["SPARKRUN_RUNTIME_CACHE_DIR"] + "/mod" == parent["env"]["MOD_CACHE"]


@pytest.mark.parametrize("failure", ["missing_config", "disabled_cache", "missing_model", "mpi_node_rank", "valid_config"])
def test_launcher_rejects_invalid_context_before_replacement(failure, monkeypatch, tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import launch_inference
    from sparkrun.runtimes.sglang import SglangRuntime
    from sparkrun.runtimes.trtllm import TrtllmRuntime

    runtime = TrtllmRuntime() if failure == "mpi_node_rank" else SglangRuntime()
    templates = {
        "missing_config": "{config.not_set}",
        "disabled_cache": "{launch.runtime_cache_dir}",
        "missing_model": "{launch.model_path}",
        "mpi_node_rank": "{launch.node_rank}",
        "valid_config": "{config.port}/{launch.node_rank}",
    }
    r = recipe(container="fixture", runtime=runtime.runtime_name, env={"VALUE": templates[failure]}, runtime_cache=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cache_dir: " + str(tmp_path / "cache"))
    config = SparkrunConfig(config_path)
    events = []
    monkeypatch.setattr(runtime, "prepare", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "get_extra_volumes", lambda: {})

    def run(**kwargs):
        events.append("run")
        assert et.render_launch_env(r, "head", {}, None, DockerExecutor()) == {"VALUE": "9001/0"}
        assert et.render_launch_env(r, "worker", {}, None, DockerExecutor()) == {"VALUE": "9001/1"}
        return 0

    monkeypatch.setattr(runtime, "run", run)
    monkeypatch.setattr(runtime, "_collect_runtime_info", lambda *a, **k: {})
    monkeypatch.setattr("sparkrun.core.launcher.resolve_effective_cache_dir", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda *a, **k: SimpleNamespace(mode="local"))
    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", lambda *a, **k: (None, {}, {}, {}))
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", lambda *a, **k: events.append("save"))
    monkeypatch.setattr("sparkrun.core.launcher._verify_mount_sources", lambda *a, **k: None)

    def fail_probe(*a, **k):
        raise ValueError("model snapshot unavailable")

    monkeypatch.setattr(et, "probe_model_path", fail_probe)

    def launch():
        return launch_inference(
            recipe=r,
            runtime=runtime,
            host_list=["head", "worker"],
            overrides={"port": 9001},
            config=config,
            sync_tuning=False,
            dry_run=False,
            before_start=lambda: events.append("replace"),
            trust=True,
        )

    if failure == "valid_config":
        assert launch().rc == 0
        assert events == ["replace", "save", "run"]
        assert r.env_templates == {"VALUE": "{config.port}/{launch.node_rank}"}
        with pytest.raises(ValueError, match="prepared context"):
            et.render_launch_env(r, "head", {}, None, DockerExecutor())
    else:
        pattern = {
            "missing_config": "not_set",
            "disabled_cache": "enabled runtime cache",
            "missing_model": "snapshot unavailable",
            "mpi_node_rank": "MPI launches",
        }[failure]
        with pytest.raises(ValueError, match=pattern):
            launch()
        assert events == []


def test_nested_model_mount_and_duplicate_source_destinations_fail():
    from sparkrun.orchestration.executors._base import ExecutorConfig

    with pytest.raises(ValueError, match="shadowed"):
        et.workload_model_path("/hf/model", {"/hf": "/cache", "/other": "/cache/model/weights"}, "docker")
    p = et.EnvTemplatePlan({"M": "{launch.model_path}"}, {}, ("h",), "id", models={"h": et.PreparedModelPath("/hf/model", SHA)})
    executor = DockerExecutor(ExecutorConfig(volumes=["/hf:/one", "/hf:/two"]))
    with pytest.raises(ValueError, match="multiple destinations"):
        p.render("h", {}, None, executor)


@pytest.mark.parametrize("index", [[], {"weight_map": {}}, {"weight_map": {"x": 1}}])
def test_probe_rejects_malformed_weight_indexes(tmp_path, index):
    path = snapshot(tmp_path)
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="no valid weight map"):
        resolve_model_path("org/model", "", str(tmp_path))


@pytest.mark.parametrize("source_registry", [None, "experimental"])
def test_delegated_transfer_stages_mod_then_resolves_remote_node_paths(monkeypatch, tmp_path, source_registry):
    """Delegated asset transfer still uses the standard env/mount launch boundary."""
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import launch_inference
    from sparkrun.runtimes.sglang import SglangRuntime

    root = tmp_path / "registry"
    recipe_dir = root / "experimental-recipes/model"
    recipe_dir.mkdir(parents=True)
    mod = recipe_dir / "prepare-model"
    mod.mkdir(parents=True)
    (mod / "run.sh").write_text("#!/bin/bash\ntrue\n")
    r = recipe(
        container="fixture",
        mods=["prepare-model"],
        env={"MODEL_PATH": "{launch.model_path}", "RANK": "{launch.node_rank}", "CACHE": "{launch.runtime_cache_dir}"},
    )
    r.source_path = str(recipe_dir / "recipe.yaml")
    r.source_registry = source_registry
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cache_dir: " + str(tmp_path / "controller-cache"))
    config = SparkrunConfig(config_path)
    runtime = SglangRuntime()
    events = []
    monkeypatch.setattr(config, "get_registry_manager", lambda: SimpleNamespace())
    monkeypatch.setattr(runtime, "prepare", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "get_extra_volumes", lambda: {})
    monkeypatch.setattr(runtime, "_collect_runtime_info", lambda *a, **k: {})
    monkeypatch.setattr("sparkrun.core.launcher.resolve_effective_runtime_cache_dir", lambda *a, **k: "/remote/sparkrun-cache")
    monkeypatch.setattr("sparkrun.core.launcher._verify_mount_sources", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.core.asset_preparation.prepare_tuning", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda *a, **k: SimpleNamespace(mode="delegated"))
    monkeypatch.setattr(DockerExecutor, "ensure_runtime_cache", lambda *a, **k: None)

    def rsync(path, host, destination, **kwargs):
        assert Path(path) == mod
        assert host == "head"
        events.append("stage-mod")
        return RemoteResult(host, 0, "", "")

    def distribute(*args, **kwargs):
        assert kwargs["transfer_mode"] == "delegated"
        assert args[3] == "/remote/hf"
        events.append("distribute-assets")
        return None, {}, {}, {}

    def probe(host, model, revision, cache_dir, ssh_kwargs, *, expected_revision=""):
        assert "distribute-assets" in events
        assert cache_dir == "/remote/hf"
        assert expected_revision == ("" if host == "head" else SHA)
        events.append("probe-" + host)
        return et.PreparedModelPath("/remote/hf/hub/models--org--model/snapshots/" + SHA, SHA)

    def run(**kwargs):
        cache = kwargs["runtime_cache"]
        assert cache.leaf.startswith("/remote/sparkrun-cache/")
        volumes = {"/remote/hf": "/cache/huggingface", **cache.volumes}
        for i, host in enumerate(("head", "worker")):
            rendered = et.render_launch_env(r, host, volumes, cache, DockerExecutor())
            assert rendered == {
                "MODEL_PATH": "/cache/huggingface/hub/models--org--model/snapshots/" + SHA,
                "RANK": str(i),
                "CACHE": "/cache/runtime",
            }
        assert r.pre_exec[0]["source_host"] == "head"
        assert r.pre_exec[0]["dest"] == "/workspace/mods/prepare-model"
        events.append("run")
        return 0

    monkeypatch.setattr("sparkrun.core.mods.run_rsync", rsync)
    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    monkeypatch.setattr(et, "probe_model_path", probe)
    monkeypatch.setattr(runtime, "run", run)
    result = launch_inference(
        recipe=r,
        runtime=runtime,
        host_list=["head", "worker"],
        overrides={},
        config=config,
        cache_dir="/remote/hf",
        local_cache_dir="/controller/hf",
        transfer_mode="delegated",
        sync_tuning=False,
        trust=True,
        before_start=lambda: events.append("replace"),
    )
    assert result.rc == 0
    assert events == ["stage-mod", "distribute-assets", "probe-head", "probe-worker", "replace", "run"]
