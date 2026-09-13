"""Proxy unload shares the stop API and retires already-stopped registrations."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from sparkrun import api
from sparkrun.cli import main


@pytest.fixture
def unload_env(tmp_path, monkeypatch):
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import DEFAULT_CONFIG_DIR

    manager = ClusterManager(DEFAULT_CONFIG_DIR)
    manager.create("lab", ["10.0.4.30", "10.0.4.31"], user="model-user")
    manager.set_default("lab")
    recipe = tmp_path / "model.yaml"
    recipe.write_text('sparkrun_version: "2"\nmodel: test/model\nruntime: sglang\ncontainer: test/image:latest\n')
    stop = Mock(return_value=api.StopResult(cluster_id="job", hosts_targeted=("10.0.4.30",), containers_removed=1))
    status = Mock(return_value=SimpleNamespace(running=True))
    unregister = Mock(return_value=SimpleNamespace(removed=1))
    monkeypatch.setattr(api, "stop", stop)
    monkeypatch.setattr(api.proxy, "status", status)
    monkeypatch.setattr(api.proxy, "unregister_loaded_model", unregister)
    return SimpleNamespace(recipe=recipe, stop=stop, status=status, unregister=unregister)


def invoke(env, *flags):
    return CliRunner().invoke(main, ["proxy", "unload", str(env.recipe), *flags])


@pytest.mark.parametrize(
    ("flags", "cluster", "hosts"),
    [
        ([], "lab", ("10.0.4.30", "10.0.4.31")),
        (["--cluster", "lab", "--hosts", "10.0.4.31"], "lab", ("10.0.4.31",)),
        (["--hosts", "10.0.4.31"], None, ("10.0.4.31",)),
    ],
)
def test_unload_uses_stop_api_with_effective_cluster(unload_env, flags, cluster, hosts):
    result = invoke(unload_env, *flags)
    assert result.exit_code == 0, result.output
    kwargs = unload_env.stop.call_args.kwargs
    assert kwargs["recipe"].model == "test/model"
    assert kwargs["cluster"] == cluster
    assert kwargs["hosts"] == hosts
    if cluster:
        assert kwargs["sctx"].config.ssh_user == "model-user"
    unload_env.unregister.assert_called_once_with(str(unload_env.recipe), sctx=kwargs["sctx"])


def test_unload_stopped_recipe_still_unregisters(unload_env):
    unload_env.stop.side_effect = api.JobNotFound("already gone")
    result = invoke(unload_env)
    assert result.exit_code == 0, result.output
    assert "No running workload" in result.output
    unload_env.unregister.assert_called_once()


@pytest.mark.parametrize("error", [api.SparkrunError("host unreachable"), api.AmbiguousWorkload("ambiguous", ["one", "two"])])
def test_unload_keeps_registration_when_stop_cannot_resolve(unload_env, error):
    unload_env.stop.side_effect = error
    result = invoke(unload_env)
    assert result.exit_code == 1
    assert "Proxy registration was kept" in result.output
    unload_env.unregister.assert_not_called()


def test_unload_keeps_registration_when_teardown_fails(unload_env):
    unload_env.stop.return_value = api.StopResult(cluster_id="job", hosts_targeted=("h1",), containers_removed=0, hosts_failed=("h1",))
    result = invoke(unload_env)
    assert result.exit_code == 1
    assert "NOT fully stopped" in result.output
    unload_env.unregister.assert_not_called()


def test_unload_dry_run_never_stops_or_mutates_proxy(unload_env):
    result = invoke(unload_env, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Would stop" in result.output
    unload_env.stop.assert_not_called()
    unload_env.status.assert_not_called()
    unload_env.unregister.assert_not_called()


def test_unload_reports_proxy_update_failure(unload_env):
    unload_env.unregister.side_effect = api.proxy.ProxyUpdateFailed("dependent virtual model")
    result = invoke(unload_env)
    assert result.exit_code == 1
    assert "Error: dependent virtual model" in result.output


def test_unload_without_proxy_reports_unchanged_registration(unload_env):
    unload_env.status.return_value.running = False
    result = invoke(unload_env)
    assert result.exit_code == 0, result.output
    assert "saved proxy registration was not changed" in result.output
    unload_env.unregister.assert_not_called()
