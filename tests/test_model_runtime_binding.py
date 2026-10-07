"""The command, env, and workload access probe consume one prepared artifact."""

import shlex
from unittest.mock import Mock

import pytest

from sparkrun.core import model_distribution as md
from sparkrun.core.env_templates import prepare_env_templates
from sparkrun.core.recipe import Recipe
from sparkrun.models.artifacts import CacheValidationReport, ValidatedModelBinding
from sparkrun.models.runtime import bind_runtime_models, validate_runtime_command, verify_workload_access
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from test_model_artifacts import artifact, cache, SHA


@pytest.fixture
def binding_scope():
    token = md._BINDINGS.set({})
    yield
    md._BINDINGS.reset(token)


def bind(root, model="org/model", manifest=None):
    manifest = manifest or artifact()
    binding = ValidatedModelBinding(
        model, "localhost", str(root), manifest, CacheValidationReport("localhost", manifest.identity, "metadata", "complete")
    )
    md.record_binding(binding)
    return binding


@pytest.mark.parametrize("runtime_class", [SglangRuntime, VllmDistributedRuntime])
def test_structured_command_and_env_use_same_revision(tmp_path, monkeypatch, binding_scope, runtime_class):
    runtime = runtime_class()
    recipe = Recipe.from_dict(
        {
            "model": "org/model",
            "runtime": runtime.runtime_name,
            "container": "test",
            "env": {"MODEL": "{launch.model_path}", "REV": "{launch.model_revision}"},
        }
    )
    binding = bind(tmp_path)
    executor = DockerExecutor(ExecutorConfig(accelerator_vendor="cpu", privileged=False))
    volumes = {str(tmp_path): "/cache/huggingface"}
    overrides = {}
    bind_runtime_models(recipe, overrides, runtime, ["localhost"], {"localhost": executor}, volumes)
    command = runtime.generate_command(recipe, overrides=overrides, is_cluster=False)
    validate_runtime_command(recipe, overrides, runtime, command)
    assert overrides["_prepared_model_path"] in shlex.split(command)
    assert "--served-model-name org/model" in command
    assert SHA in command
    monkeypatch.setattr("sparkrun.core.env_templates.probe_model_path", Mock(side_effect=AssertionError("reprobe")))
    plan = prepare_env_templates(recipe, overrides, ["localhost"], "job", str(tmp_path), {})
    assert plan.models["localhost"].host_path == binding.model_path
    assert plan.render("localhost", volumes, None, executor) == {"MODEL": overrides["_prepared_model_path"], "REV": SHA}
    assert recipe.model_revision is None


def test_custom_command_cannot_load_unvalidated_revision(binding_scope, tmp_path):
    bind(tmp_path)
    recipe = Recipe.from_dict(
        {"model": "org/model", "runtime": "sglang", "container": "test", "command": "sglang serve org/model --revision main"}
    )
    runtime = SglangRuntime()
    overrides = {}
    bind_runtime_models(recipe, overrides, runtime, ["localhost"], {"localhost": DockerExecutor()}, {str(tmp_path): "/cache/huggingface"})
    with pytest.raises(ValueError, match="validated model"):
        validate_runtime_command(recipe, overrides, runtime, recipe.command)


def test_workload_visibility_detects_deleted_file(tmp_path, binding_scope):
    binding = bind(tmp_path)
    snapshot = cache(tmp_path, binding.manifest)
    executor = Mock(executor_name="local", config=ExecutorConfig())
    del executor.model_access_command
    verify_workload_access(["localhost"], {"localhost": executor}, ["test"], {}, {})
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="inaccessible.*tokenizer.json"):
        verify_workload_access(["localhost"], {"localhost": executor}, ["test"], {}, {})


def test_docker_probe_uses_effective_user_and_mounts_without_gpu():
    executor = DockerExecutor(ExecutorConfig(user="1234:5678", accelerator_vendor="nvidia", privileged=False))
    executor.bind_image_references({"requested:tag": "sha256:" + "a" * 64})
    command = executor.model_access_command(
        "requested:tag", "echo probe", {"/cache": "/cache/huggingface:ro"}, ["--gpus", "all", "--entrypoint=ignored"]
    )
    tokens = shlex.split(command)
    assert tokens[tokens.index("--user") + 1] == "1234:5678"
    assert "/cache:/cache/huggingface:ro" in tokens
    assert "--pull=never" in tokens
    assert "--gpus" not in tokens and not any("nvidia.com/gpu" in t for t in tokens)
    assert tokens[tokens.index("--entrypoint") + 1] == "/bin/bash"
    assert executor.config.entrypoint is None


def test_shadow_mount_refused_before_access(tmp_path, binding_scope):
    bind(tmp_path)
    executor = DockerExecutor(ExecutorConfig(volumes=["/other:/cache/huggingface/hub"]))
    with pytest.raises(ValueError, match="shadow"):
        verify_workload_access(["localhost"], {"localhost": executor}, ["image"], {str(tmp_path): "/cache/huggingface"}, {})
