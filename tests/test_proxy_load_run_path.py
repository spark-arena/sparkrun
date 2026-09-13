"""Proxy loading must preserve the ordinary run plan and readiness policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun import api
from sparkrun.cli import main
from sparkrun.core.launcher import LaunchResult
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint, load_job_metadata, save_job_metadata


@pytest.fixture
def load_env(tmp_path, monkeypatch):
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import DEFAULT_CONFIG_DIR

    manager = ClusterManager(DEFAULT_CONFIG_DIR)
    manager.create("lab", ["10.0.4.30", "10.0.4.31"], user="model-user", scheduler="greedy")
    manager.set_default("lab")
    recipe = tmp_path / "model.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "sparkrun_version": "2",
                "model": "test/model",
                "runtime": "sglang",
                "container": "test/image:latest",
                "defaults": {"port": 30000, "tensor_parallel": 1},
                "metadata": {"model_params": 1000000, "model_dtype": "float16"},
            }
        )
    )
    calls = []
    order = []

    def launch(**kwargs):
        order.append("launch")
        calls.append(kwargs)
        assert kwargs["recipe_fingerprint"] == derive_recipe_fingerprint(kwargs["recipe"], kwargs["overrides"])
        # Like real auto-port/platform adjustment, mutate only launch inputs.
        kwargs["overrides"]["port"] = 30007
        kwargs["recipe"].defaults["platform_only_flag"] = True
        if not kwargs["dry_run"]:
            save_job_metadata(
                kwargs["cluster_id_override"],
                kwargs["recipe"],
                kwargs["host_list"],
                kwargs["overrides"],
                cache_dir=str(kwargs["config"].cache_dir),
                cluster_name=kwargs["cluster"].name,
                ssh_user=kwargs["config"].ssh_user,
                recipe_fingerprint=kwargs["recipe_fingerprint"],
            )
        return LaunchResult(
            rc=0,
            cluster_id=kwargs["cluster_id_override"],
            host_list=kwargs["host_list"],
            is_solo=kwargs["is_solo"],
            runtime=kwargs["runtime"],
            recipe=kwargs["recipe"],
            overrides=kwargs["overrides"],
            container_image="test/image:latest",
            effective_cache_dir="/tmp/cache",
            serve_port=30007,
            config=kwargs["config"],
            timeline=kwargs["sctx"].timing,
        )

    def wait(result, **kwargs):
        order.append("ready")
        assert result.serve_port == 30007
        return SimpleNamespace(ready=True)

    def register(*args, **kwargs):
        order.append("register")
        return SimpleNamespace(added=1)

    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    readiness = Mock(side_effect=wait)
    registration = Mock(side_effect=register)
    status = Mock(return_value=SimpleNamespace(running=True, autodiscover_running=False))
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", readiness)
    monkeypatch.setattr(api.proxy, "register_loaded_model", registration)
    monkeypatch.setattr(api.proxy, "status", status)
    return SimpleNamespace(recipe=recipe, calls=calls, order=order, readiness=readiness, registration=registration, status=status)


@pytest.mark.parametrize(
    ("flags", "cluster"),
    [
        ([], "lab"),
        (["--cluster", "lab"], "lab"),
        (["--cluster", "lab", "--hosts", "10.0.4.31"], "lab"),
        (["--hosts", "10.0.4.31"], None),
    ],
)
def test_proxy_load_preserves_cluster_metadata_and_registration(load_env, flags, cluster):
    env = load_env
    result = CliRunner().invoke(main, ["proxy", "load", str(env.recipe), *flags, "--max-model-len", "4096", "--port", "30001"])
    assert result.exit_code == 0, result.output
    assert len(env.calls) == 1
    launch = env.calls[0]
    assert launch["auto_port"] is True
    assert launch["follow"] is False
    if "--hosts" not in flags:
        assert launch["placement"] is not None
    metadata = load_job_metadata(launch["cluster_id_override"], cache_dir=str(launch["config"].cache_dir))
    assert metadata.get("cluster") == cluster
    assert metadata["recipe_fingerprint"] == launch["recipe_fingerprint"]
    if cluster:
        assert metadata["ssh_user"] == "model-user"
    if "--hosts" in flags:
        assert launch["host_list"] == ["10.0.4.31"]
    registered = env.registration.call_args
    assert registered.args == (str(env.recipe),)
    assert registered.kwargs["cluster"] == cluster
    assert registered.kwargs["overrides"] == {"max_model_len": 4096, "port": 30001}
    assert env.order == ["launch", "ready", "register"]


@pytest.mark.parametrize("reason", ["port", "health", "inference", "cancelled"])
def test_proxy_load_does_not_register_before_readiness(load_env, reason):
    env = load_env
    env.readiness.side_effect = None
    env.readiness.return_value = SimpleNamespace(
        ready=False, reason=reason, port=30007, head_host="10.0.4.30", health_url="http://test/v1/models"
    )
    result = CliRunner().invoke(main, ["proxy", "load", str(env.recipe)])
    assert result.exit_code == 0, result.output
    env.registration.assert_not_called()
    assert "Warning:" in result.output


def test_proxy_load_dry_run_does_not_probe_or_register(load_env):
    result = CliRunner().invoke(main, ["proxy", "load", str(load_env.recipe), "--dry-run"])
    assert result.exit_code == 0, result.output
    load_env.status.assert_not_called()
    load_env.readiness.assert_not_called()
    load_env.registration.assert_not_called()


def test_proxy_load_without_running_proxy_does_not_wait(load_env):
    load_env.status.return_value.running = False
    result = CliRunner().invoke(main, ["proxy", "load", str(load_env.recipe)])
    assert result.exit_code == 0, result.output
    load_env.readiness.assert_not_called()
    load_env.registration.assert_not_called()


def test_proxy_load_reports_api_failure(load_env, monkeypatch):
    monkeypatch.setattr(api, "run", Mock(side_effect=api.SparkrunError("cannot place workload")))
    result = CliRunner().invoke(main, ["proxy", "load", str(load_env.recipe)])
    assert result.exit_code == 1
    assert "Error: cannot place workload" in result.output
    load_env.readiness.assert_not_called()
    load_env.registration.assert_not_called()


def test_proxy_load_uses_the_shared_post_launch_lifecycle(load_env, monkeypatch):
    data = yaml.safe_load(load_env.recipe.read_text())
    data["post_exec"] = ["echo initialized"]
    load_env.recipe.write_text(yaml.safe_dump(data))
    lifecycle = Mock(side_effect=lambda *a, **kw: load_env.order.append("post_launch"))
    monkeypatch.setattr("sparkrun.core.launcher.post_launch_lifecycle", lifecycle)
    result = CliRunner().invoke(main, ["proxy", "load", str(load_env.recipe)])
    assert result.exit_code == 0, result.output
    lifecycle.assert_called_once()
    assert load_env.order == ["launch", "post_launch", "ready", "register"]
