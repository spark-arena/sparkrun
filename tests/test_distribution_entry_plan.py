"""Local and remote transports must stage the same resolved resource policy."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.orchestration.distribution import DistributionError, TransferModeResult, distribute_from_config
from sparkrun.orchestration.infiniband import IBDetectionResult


@pytest.fixture
def distribution_env(monkeypatch, tmp_path, unavailable_model_inventory):
    recipe = Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "org/model",
            "runtime": "vllm-distributed",
            "container": "org/main",
            "builder_config": {"rebuild": True},
            "distribution_config": {
                "containers": {
                    "entries": [
                        {"name": "org/main", "target": [0]},
                        {"name": "org/helper", "target": [-1]},
                        {"name": "org/elsewhere", "target": [1]},
                    ]
                },
                "models": {"entries": [{"name": "org/model", "target": [0]}, {"name": "org/draft", "target": [1]}]},
            },
        }
    )
    ensure, download = Mock(return_value=0), Mock(return_value=0)
    remote_images, remote_model = Mock(return_value=[]), Mock(return_value=[])
    after = Mock()
    monkeypatch.setattr("sparkrun.orchestration.distribution._is_cross_user", lambda kw: False)
    monkeypatch.setattr("sparkrun.orchestration.distribution._get_hf_token", lambda: "")
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr("sparkrun.containers.registry.ensure_image", ensure)
    monkeypatch.setattr("sparkrun.models.download.download_model", download)
    monkeypatch.setattr("sparkrun.orchestration.distribution._distribute_image_plan", remote_images)
    monkeypatch.setattr("sparkrun.orchestration.distribution._distribute_single_model", remote_model)
    ib = IBDetectionResult(comm_env=ClusterCommEnv.empty(), ib_ip_map={}, mgmt_ip_map={})
    kwargs = dict(
        recipe=recipe,
        image="org/main",
        host_list=["node"],
        cache_dir=str(tmp_path),
        config=SimpleNamespace(cache_dir=tmp_path),
        dry_run=False,
        pre_ib=TransferModeResult(mode="local", ib_result=ib),
        transfer_mode="local",
        after_container_sync=after,
    )
    return SimpleNamespace(
        recipe=recipe, ensure=ensure, download=download, remote_images=remote_images, remote_model=remote_model, after=after, kwargs=kwargs
    )


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("policy", ["recipe", "prepared"])
def test_same_entries_targets_and_force_pull_for_both_transports(distribution_env, monkeypatch, local, policy):
    env = distribution_env
    monkeypatch.setattr("sparkrun.orchestration.distribution.is_local_host", lambda h: local)
    before = deepcopy(env.recipe.distribution_config)
    if policy == "prepared":
        env.kwargs["container_distribution"] = deepcopy(before.containers)
        env.recipe.distribution_config.containers.enabled = False
        before = deepcopy(env.recipe.distribution_config)
    distribute_from_config(**env.kwargs)
    if local:
        assert [c.args[0] for c in env.ensure.call_args_list] == ["org/main", "org/helper"]
        assert all(c.kwargs["force_pull"] for c in env.ensure.call_args_list)
        assert [c.args[0] for c in env.download.call_args_list] == ["org/model"]
        env.remote_images.assert_not_called()
    else:
        assert env.remote_images.call_args.args[0] == [("org/main", ["node"]), ("org/helper", ["node"])]
        assert env.remote_images.call_args.kwargs["force_pull"] is True
        assert [(c.args[0], c.args[1]) for c in env.remote_model.call_args_list] == [("org/model", ["node"])]
        env.ensure.assert_not_called()
    env.after.assert_called_once()
    assert env.recipe.distribution_config == before
    if policy == "prepared":
        assert env.kwargs["container_distribution"].enabled
        assert len(env.kwargs["container_distribution"].entries) == 3


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_disabled_and_skipped_resources_still_allow_post_image_callback(distribution_env, monkeypatch, local, disabled):
    env = distribution_env
    monkeypatch.setattr("sparkrun.orchestration.distribution.is_local_host", lambda h: local)
    if disabled:
        env.recipe.distribution_config.containers.enabled = False
        env.recipe.distribution_config.models.enabled = False
    else:
        env.kwargs.update(skip_container=True, skip_model=True)
    distribute_from_config(**env.kwargs)
    for operation in (env.ensure, env.download, env.remote_images, env.remote_model):
        operation.assert_not_called()
    env.after.assert_called_once()


def test_missing_required_local_image_prevents_callback_and_model_download(distribution_env, monkeypatch):
    env = distribution_env
    monkeypatch.setattr("sparkrun.orchestration.distribution.is_local_host", lambda h: True)
    env.ensure.side_effect = [0, 1]
    with pytest.raises(DistributionError, match="org/helper"):
        distribute_from_config(**env.kwargs)
    env.after.assert_not_called()
    env.download.assert_not_called()
