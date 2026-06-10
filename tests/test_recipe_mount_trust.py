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
    with pytest.raises(RecipeError, match="executor_config"):
        _enforce_recipe_mount_trust(recipe, trusted=False)


@pytest.mark.parametrize(
    "exec_cfg",
    [
        {"privileged": True},  # --privileged → full container escape
        {"cap_add": ["SYS_ADMIN"]},  # raise capabilities past the rootless baseline
        {"security_opt": []},  # drop the no-new-privileges hardening
        {"devices": ["/dev/mem"]},  # raw kernel-memory / block-device access
        {"user": "0:0"},  # run the container as root
        {"volumes": ["/etc:/etc"]},  # extra host bind mount
    ],
)
def test_untrusted_recipe_with_privileged_executor_keys_is_rejected(exec_cfg):
    """C1 regression: an untrusted recipe cannot re-enable any isolation-breaking
    executor_config key — not just ``volumes``.  These sit above the rootless
    ``apply_runtime_adjustments`` layer, so the trust gate is the only thing
    stopping a "run this link" recipe from emitting ``docker run --privileged``
    / ``--device`` / ``--user 0:0`` and taking over the host."""
    recipe = _recipe(executor_config=exec_cfg)
    with pytest.raises(RecipeError, match="executor_config"):
        _enforce_recipe_mount_trust(recipe, trusted=False)


def test_untrusted_recipe_with_benign_executor_keys_is_allowed():
    """Innocuous resource knobs are not gated — they can't break isolation."""
    recipe = _recipe(executor_config={"shm_size": "16gb", "ipc": "host", "memory_limit": "64g"})
    _enforce_recipe_mount_trust(recipe, trusted=False)  # must not raise


def test_trusted_recipe_with_mounts_is_allowed():
    # --trust / local / default-registry recipes opt in; the gate is a no-op.
    recipe = _recipe(
        cluster_config={"resolved_model_path": "/nfs/models/qwen3"},
        executor_config={"volumes": ["/mnt/quant:/mnt/quant"], "privileged": True, "devices": ["/dev/kfd"]},
    )
    _enforce_recipe_mount_trust(recipe, trusted=True)  # must not raise


def test_untrusted_recipe_without_mounts_is_allowed():
    _enforce_recipe_mount_trust(_recipe(), trusted=False)  # must not raise


# ---------------------------------------------------------------------------
# Executor selector gate (S1): only Docker keeps the container sandbox.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("executor", ["local", "k8s", "Local", " LOCAL "])
def test_untrusted_recipe_selecting_non_docker_executor_is_rejected(executor):
    """An untrusted recipe may not select a non-Docker executor.

    ``local`` runs the serve command natively (``setsid bash -c``) and ``k8s``
    via ``kubectl run`` — neither sandboxes the command in a rootless container,
    so honouring the selector for a "run this link" recipe is direct host RCE.
    """
    recipe = _recipe(executor=executor)
    with pytest.raises(RecipeError, match="executor"):
        _enforce_recipe_mount_trust(recipe, trusted=False)


def test_untrusted_recipe_selecting_non_docker_via_executor_config_is_rejected():
    """The selector smuggled through executor_config is gated too."""
    recipe = _recipe(executor_config={"executor": "local"})
    with pytest.raises(RecipeError, match="executor"):
        _enforce_recipe_mount_trust(recipe, trusted=False)


@pytest.mark.parametrize("executor", ["", "docker", "DOCKER"])
def test_untrusted_recipe_with_docker_executor_is_allowed(executor):
    """The default Docker executor (or unset) keeps the sandbox → allowed."""
    recipe = _recipe(executor=executor)
    _enforce_recipe_mount_trust(recipe, trusted=False)  # must not raise


def test_trusted_recipe_may_select_local_executor():
    """Trust is the explicit opt-in for the non-container executors."""
    recipe = _recipe(executor="local")
    _enforce_recipe_mount_trust(recipe, trusted=True)  # must not raise


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


@pytest.mark.parametrize(
    "bad",
    [
        "/etc/sudoers.d",  # subtree of /etc — exact-match denylist would miss it
        "/var/lib/docker",  # docker data root
        "/var/lib/docker/volumes/x",
        "/run/secrets",  # subtree of /run
        "/proc/self/environ",
        "relative/path",  # not absolute
        "models",  # not absolute
        "/data/../etc",  # contains ..
    ],
)
def test_assert_safe_mount_source_rejects_subtrees_and_relative(bad):
    with pytest.raises(ValueError):
        assert_safe_mount_source(bad)


def test_assert_safe_mount_source_rejects_other_users_ssh_dir():
    # Matched by path component, not just the control machine's own ~/.ssh.
    with pytest.raises(ValueError):
        assert_safe_mount_source("/home/someone-else/.ssh/id_ed25519")


def test_assert_safe_mount_source_allows_normal_paths():
    for ok in ("/nfs/models/qwen3", "/mnt/quant/calib", "/data/hf"):
        assert assert_safe_mount_source(ok) == ok


def test_docker_volume_emission_rejects_catastrophic_source():
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.docker import DockerExecutor

    with pytest.raises(ValueError):
        DockerExecutor(ExecutorConfig(volumes=["/:/host"])).run_cmd("img:1")
