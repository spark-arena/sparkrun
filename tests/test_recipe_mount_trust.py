"""Security: untrusted recipes must not bind-mount arbitrary host paths.

Covers the ``_enforce_recipe_mount_trust`` gate (cluster_config /
executor_config.volumes require a trusted recipe) and the
``assert_safe_mount_source`` defense-in-depth denylist that refuses
catastrophic host mount sources regardless of trust.
"""

from __future__ import annotations

import os

import pytest

from sparkrun.core.launcher import _enforce_recipe_mount_trust
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.utils.shell import assert_safe_mount_source


def _recipe(**extra):
    d = {
        "recipe_version": "2",
        "name": "esc",
        "model": "Qwen/Qwen3-1.7B",
        "runtime": "vllm-distributed",
        "container": "img:latest",
    }
    d.update(extra)
    return Recipe.from_dict(d)


# ---------------------------------------------------------------------------
# Trust gate
# ---------------------------------------------------------------------------


def test_untrusted_recipe_with_cluster_config_is_rejected():
    recipe = _recipe(cluster_config={"resolved_model_path": "/nfs/models/qwen3"})
    with pytest.raises(RecipeError, match="cluster_config"):
        _enforce_recipe_mount_trust(recipe, trusted=False)


def test_untrusted_recipe_with_cache_dir_override_is_rejected():
    recipe = _recipe(cluster_config={"remote_cache_dir": "/nfs/hf"})
    with pytest.raises(RecipeError, match="cluster_config"):
        _enforce_recipe_mount_trust(recipe, trusted=False)


def test_untrusted_recipe_with_executor_volumes_is_rejected():
    recipe = _recipe(executor_config={"volumes": ["/:/host"]})
    with pytest.raises(RecipeError, match="executor_config.volumes"):
        _enforce_recipe_mount_trust(recipe, trusted=False)


def test_trusted_recipe_with_mounts_is_allowed():
    # --trust / local / default-registry recipes opt in; the gate is a no-op.
    recipe = _recipe(
        cluster_config={"resolved_model_path": "/nfs/models/qwen3"},
        executor_config={"volumes": ["/mnt/quant:/mnt/quant"]},
    )
    _enforce_recipe_mount_trust(recipe, trusted=True)  # must not raise


def test_untrusted_recipe_without_mounts_is_allowed():
    _enforce_recipe_mount_trust(_recipe(), trusted=False)  # must not raise


# ---------------------------------------------------------------------------
# Defense-in-depth mount-source denylist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "/",
        "/etc",
        "/root",
        "/proc",
        "/sys",
        "/dev",
        "/boot",
        "/var/run/docker.sock",
        "/run/docker.sock",
        "/etc/../etc",  # normalizes back to /etc
    ],
)
def test_assert_safe_mount_source_rejects_sensitive_paths(bad):
    with pytest.raises(ValueError):
        assert_safe_mount_source(bad)


def test_assert_safe_mount_source_rejects_ssh_dir():
    ssh = os.path.expanduser("~/.ssh")
    with pytest.raises(ValueError):
        assert_safe_mount_source(ssh)
    with pytest.raises(ValueError):
        assert_safe_mount_source(os.path.join(ssh, "id_rsa"))


def test_assert_safe_mount_source_allows_normal_paths():
    for ok in ("/nfs/models/qwen3", "/mnt/quant/calib", "/data/hf"):
        assert assert_safe_mount_source(ok) == ok


def test_docker_volume_emission_rejects_catastrophic_source():
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.docker import DockerExecutor

    with pytest.raises(ValueError):
        DockerExecutor(ExecutorConfig(volumes=["/:/host"])).run_cmd("img:1")
