"""Runtime protocol and executor observation capabilities resolve separately."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sparkrun.core.readiness import (
    DOCKER_HOST_OBSERVER,
    OPENAI_CHAT_STREAM,
    OPENAI_RESPONSES_STREAM,
    ANTHROPIC_MESSAGES_STREAM,
    ObservationUnavailable,
    ReadinessSettings,
    resolve_inference_style,
    resolve_readiness_settings,
)
from sparkrun.core.launcher import wait_for_serve_ready
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.core.validation import check_readiness
from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.executors.k8s import K8sExecutor
from sparkrun.orchestration.startup import validate_observation
from sparkrun.scripts import startup_probe


def model(**readiness):
    return Recipe({"model": "test/model", "runtime": "vllm-distributed", "container": "image", "readiness": readiness})


def runtime(**changes):
    return SimpleNamespace(
        runtime_name="custom",
        readiness_styles=(OPENAI_CHAT_STREAM,),
        readiness_health_path="/health",
        executor=DockerExecutor(),
        get_head_container_name=lambda *a, **kw: "head",
        **changes,
    )


def receipt(**changes):
    return {
        "format": 1,
        "measurement": "rank0-acceptance-v1",
        "observer": "rank0",
        "container_id": "current",
        "container_started_unix_ns": 1,
        "first_token_unix_ns": 2,
        "first_token_field": "content",
        "inference_ready": True,
        "response_validated": True,
        **changes,
    }


def launch(rt=None, **readiness):
    return SimpleNamespace(
        runtime=rt or runtime(),
        recipe=model(**readiness),
        config=None,
        startup_observation={},
        cluster_id="job",
        host_list=["localhost"],
        is_solo=True,
        serve_port=8000,
        runtime_info={},
        overrides={},
        timeline=None,
    )


def test_explicit_style_resolves_through_three_layers_and_is_not_a_runtime_flag():
    config = SimpleNamespace(get=lambda *a: {"inference_style": OPENAI_CHAT_STREAM, "inference_timeout_s": 19})
    recipe = model(inference_style="auto")
    settings = resolve_readiness_settings(config=config, recipe=recipe)
    assert settings.inference_style == "auto" and settings.inference_timeout_s == 19
    assert resolve_inference_style(settings, runtime()) == OPENAI_CHAT_STREAM
    assert recipe.to_dict()["readiness"] == {"inference_style": "auto"}
    assert "inference_style" not in recipe.runtime_config


@pytest.mark.parametrize("style", [OPENAI_RESPONSES_STREAM, ANTHROPIC_MESSAGES_STREAM])
@pytest.mark.parametrize("previous", ["receipt", "accepted", "none"])
def test_vllm_launch_probes_selected_api_and_served_name(style, previous):
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
    from sparkrun.core.timing import Timeline

    rt = VllmDistributedRuntime()
    rt.executor = DockerExecutor()
    result = launch(rt=rt, inference_style=style)
    result.overrides = {"served_model_name": "launch-alias"}
    result.timeline = Timeline()
    if previous == "receipt":
        result.startup_observation = receipt()
    elif previous == "accepted":
        result.runtime_info = {"inference_readiness": "accepted"}
    response = receipt(measurement="sparkrun-rank0-v1", inference_style=style)
    with patch("sparkrun.orchestration.startup.run_probe", return_value=response) as probe:
        assert wait_for_serve_ready(result).ready
    assert probe.call_count == 1
    assert probe.call_args.args[1]["inference_style"] == style
    assert probe.call_args.args[1]["model"] == "launch-alias"
    assert result.startup_observation["inference_style"] == style
    assert result.timeline.find("serve.startup_ttft") is not None
    # A fixed ColdSnap Chat receipt must never be relabeled as another API.
    with pytest.raises(ValueError, match="incompatible"):
        validate_observation(receipt(inference_style=style))


@pytest.mark.parametrize("style", ["unknown", [], None])
def test_native_receipt_rejects_invalid_protocol(style):
    with pytest.raises(ValueError, match="incompatible inference_style"):
        validate_observation(receipt(measurement="sparkrun-rank0-v1", inference_style=style))


@pytest.mark.parametrize("style", [None, False, [], "", "typo"])
def test_invalid_style_is_not_silently_defaulted(style):
    with pytest.raises(RecipeError, match="inference_style"):
        model(inference_style=style)
    with pytest.raises(ValueError, match="inference_style"):
        resolve_readiness_settings(config=SimpleNamespace(get=lambda *a: {"inference_style": style}))


def test_recipe_replaces_an_incompatible_lower_priority_style():
    config = SimpleNamespace(get=lambda *a: {"inference_style": "unknown"})
    assert resolve_readiness_settings(config=config, recipe=model(inference_style="auto")).inference_style == "auto"


def test_declared_styles_match_packaged_protocol_handlers():
    from sparkrun.core.readiness import INFERENCE_STYLES

    assert INFERENCE_STYLES == frozenset(startup_probe.INFERENCE_PROBES)


def test_runtime_opt_in_and_opt_out_do_not_depend_on_family_names():
    result = launch()  # custom runtime opts in without pretending to be vLLM
    with patch("sparkrun.orchestration.startup.run_probe", return_value=receipt()) as probe:
        assert wait_for_serve_ready(result).ready
    assert probe.call_args.args[1]["inference_style"] == OPENAI_CHAT_STREAM
    result = launch()
    result.runtime.readiness_styles = ()
    result.runtime.get_family = lambda: "vllm"
    with patch("sparkrun.core.launcher.wait_for_endpoint_ready", return_value="legacy") as endpoint:
        assert wait_for_serve_ready(result) == "legacy"
    endpoint.assert_called_once()


def test_incompatible_explicit_style_fails_validation_but_disabled_inference_does_not():
    unsupported = SimpleNamespace(runtime_name="other")
    issues = check_readiness(model(inference_style=OPENAI_CHAT_STREAM), unsupported, None)
    assert len(issues) == 1 and issues[0].is_error and issues[0].code == "readiness-incompatible"
    assert check_readiness(model(inference_style=OPENAI_CHAT_STREAM, inference=False), unsupported, None) == []
    assert resolve_inference_style(ReadinessSettings(), unsupported) is None


def test_precomputed_plan_is_validated_before_strategy_preparation():
    from sparkrun.api import RunOptions, SparkrunError, run

    recipe = model(inference_style=OPENAI_CHAT_STREAM)
    plan = SimpleNamespace(recipe=recipe, runtime=SimpleNamespace(runtime_name="other"))
    sctx = SimpleNamespace(config=None)
    with patch("sparkrun.core.execution.resolve_recipe_execution") as prepare:
        with pytest.raises(SparkrunError, match="does not support"):
            run(RunOptions(recipe=recipe), plan=plan, sctx=sctx)
    prepare.assert_not_called()


@pytest.mark.parametrize("executor", [LocalExecutor(), K8sExecutor(), DockerExecutor(ExecutorConfig(network="bridge"))])
def test_executor_without_observer_keeps_legacy_readiness(executor):
    assert executor.readiness_observer() is None
    result = launch()
    result.runtime.executor = executor
    with (
        patch("sparkrun.orchestration.startup.run_probe") as probe,
        patch("sparkrun.core.launcher.wait_for_endpoint_ready", return_value="legacy"),
    ):
        assert wait_for_serve_ready(result) == "legacy"
    probe.assert_not_called()
    assert DockerExecutor().readiness_observer() == DOCKER_HOST_OBSERVER


def test_target_side_unsupported_environment_falls_back_without_startup_metrics():
    result = launch()
    with (
        patch("sparkrun.orchestration.startup.run_probe", side_effect=ObservationUnavailable("unsupported target")),
        patch("sparkrun.core.launcher.wait_for_endpoint_ready", return_value="legacy"),
    ):
        assert wait_for_serve_ready(result) == "legacy"
    assert not result.startup_observation


def test_coldsnap_profile_normalization_reuses_receipt_without_mutating_it():
    result = launch(inference_style=OPENAI_CHAT_STREAM)
    original = receipt()
    assert validate_observation(original) is original
    result.startup_observation = original
    with patch("sparkrun.orchestration.startup.run_probe") as probe:
        ready = wait_for_serve_ready(result)
    probe.assert_not_called()
    assert "inference_style" not in original
    assert ready.startup_observation["inference_style"] == OPENAI_CHAT_STREAM
    assert ready.startup_observation["executor"] == "docker"
    assert ready.startup_observation["observer_location"] == "rank0-container"
    assert ready.startup_observation["start_boundary"] == "docker.State.StartedAt"


@pytest.mark.parametrize(
    "changes", [{"inference_style": "unknown"}, {"executor": "local"}, {"observer_location": "control"}, {"start_boundary": "pod.created"}]
)
def test_profile_cannot_be_relabelled(changes):
    with pytest.raises(ValueError, match="incompatible"):
        validate_observation(receipt(**changes))


@pytest.mark.parametrize("target", ["remote", "context", "rootless", "desktop", "non-linux", "native"])
def test_docker_observer_checks_actual_target_environment(monkeypatch, target):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setattr(startup_probe.sys, "platform", "darwin" if target == "non-linux" else "linux")
    endpoint = "unix:///var/run/docker.sock"
    if target == "remote":
        monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    if target == "context":
        monkeypatch.setenv("DOCKER_HOST", endpoint)
        monkeypatch.setenv("DOCKER_CONTEXT", "remote-context")
        endpoint = "ssh://remote.invalid"
    info = {"OSType": "linux", "OperatingSystem": "Ubuntu", "SecurityOptions": []}
    if target == "rootless":
        info["SecurityOptions"] = ["name=rootless"]
    if target == "desktop":
        info["OperatingSystem"] = "Docker Desktop"
    read = Mock(side_effect=[json.dumps(endpoint).encode(), json.dumps(info).encode()])
    monkeypatch.setattr(startup_probe.subprocess, "check_output", read)
    if target == "native":
        startup_probe.verify_host_observer()
        assert read.call_count == 2
    else:
        with pytest.raises(startup_probe.ObservationUnavailable):
            startup_probe.verify_host_observer()


def test_actual_container_network_must_match_declared_observer():
    with patch.object(startup_probe.subprocess, "check_output", return_value=b'{"HostConfig":{"NetworkMode":"bridge"}}'):
        with pytest.raises(startup_probe.ObservationUnavailable, match="host network"):
            startup_probe.inspect_container("head")


def test_unknown_probe_style_fails_before_docker_or_http():
    with patch.object(startup_probe, "verify_host_observer") as verify:
        with pytest.raises(ValueError, match="unsupported inference"):
            startup_probe.observe({"inference_style": "unknown"})
    verify.assert_not_called()
