from __future__ import annotations

from types import SimpleNamespace

import pytest

from sparkrun.core.image_preparation import (
    ImagePreparationError,
    PreparedImageSet,
    prepare_images,
    resolve_content_images,
    stage_prepared_images,
)
from sparkrun.core.images import ImagePlan
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.ssh import RemoteResult


HOSTS = ["node-a", "node-b"]
PINNED = "registry.example/vllm@sha256:" + "a" * 64


class _Runtime:
    runtime_name = "stub"
    supports_heterogeneous_images = True

    def resolve_container(self, recipe, overrides):
        return overrides.get("container", recipe.container)


def _recipe(**extra):
    return Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "org/model",
            "model_revision": "model-revision",
            "runtime": "vllm-distributed",
            "container": PINNED,
            **extra,
        }
    )


def test_prepare_images_runs_builder_and_returns_authoritative_plan(monkeypatch):
    recipe = _recipe(builder="snapshot")
    calls = []

    class _Builder:
        def prepare(self, image, recipe, hosts, **kwargs):
            calls.append((image, hosts, kwargs["transfer_mode"]))
            return "myorg/snapshot-vllm:built"

    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", lambda *_args, **_kwargs: _Builder())
    prepared = prepare_images(
        recipe,
        _Runtime(),
        HOSTS,
        {},
        transfer_mode="delegated",
    )

    assert prepared.source_image == PINNED
    assert prepared.head_image == "myorg/snapshot-vllm:built"
    assert prepared.images_by_node == ("myorg/snapshot-vllm:built",) * 2
    assert calls == [(PINNED, HOSTS, "delegated")]


def test_prepare_images_strategy_override_skips_builder_and_drives_distribution(monkeypatch):
    recipe = _recipe(builder="snapshot")
    capsules = ("registry/capsule0@sha256:" + "b" * 64, "registry/capsule1@sha256:" + "c" * 64)
    monkeypatch.setattr(
        "sparkrun.core.bootstrap.get_builder",
        lambda *_args, **_kwargs: pytest.fail("restore resolved builder"),
    )

    prepared = prepare_images(
        recipe,
        _Runtime(),
        HOSTS,
        {},
        run_builder=False,
        images_by_node=capsules,
        strategy_name="snapshot",
    )

    assert prepared.builder is None
    assert prepared.images_by_node == capsules
    assert recipe.distribution_config.containers.enabled is True
    assert [entry.name for entry in recipe.distribution_config.containers.entries] == list(capsules)


def test_resolve_content_images_preserves_pins_and_pins_builder_tags(monkeypatch):
    images = (PINNED, "myorg/snapshot-vllm:built")
    calls = []

    def run(host, command, **_kwargs):
        calls.append((host, command))
        identity = "sha256:" + ("1" if host == "node-a" else "2") * 64
        return RemoteResult(host=host, returncode=0, stdout=identity + "\n", stderr="")

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", run)
    resolved = resolve_content_images(images, HOSTS, ssh_kwargs={"ssh_user": "drew"})

    assert resolved == (PINNED, "sha256:" + "2" * 64)
    assert {host for host, _command in calls} == set(HOSTS)


def test_resolve_content_images_fails_when_prepared_image_is_missing(monkeypatch):
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, *_args, **_kwargs: RemoteResult(host=host, returncode=1, stdout="", stderr="not found"),
    )
    with pytest.raises(ImagePreparationError, match="node-a.*not found"):
        resolve_content_images((PINNED,), ("node-a",))


def test_stage_prepared_images_reuses_distribution_and_returns_content_ids(monkeypatch):
    from sparkrun.core.timing import Timeline

    recipe = _recipe()
    prepared = PreparedImageSet(PINNED, ImagePlan(PINNED, images_by_node=(PINNED, PINNED)))
    distribution = {}
    timeline = Timeline()

    def distribute(*args, **kwargs):
        distribution.update(kwargs)
        return "comm", {"node-a": "ib-a"}, {"node-a": "mgmt-a"}, {"node-a": "ib0"}

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    monkeypatch.setattr(
        "sparkrun.core.image_preparation.resolve_content_images",
        lambda *_args, **_kwargs: ("sha256:" + "1" * 64, "sha256:" + "2" * 64),
    )
    staged = stage_prepared_images(
        prepared,
        recipe,
        HOSTS,
        "/cache/hf",
        SimpleNamespace(),
        require_content_ids=True,
        timeline=timeline,
    )

    assert distribution["skip_model"] is True
    assert distribution["skip_container"] is False
    assert distribution["timeline"] is timeline
    assert staged.comm_env == "comm"
    assert staged.content_images_by_node == ("sha256:" + "1" * 64, "sha256:" + "2" * 64)


def test_stage_prepared_images_can_include_model_assets(monkeypatch):
    recipe = _recipe()
    prepared = PreparedImageSet(PINNED, ImagePlan(PINNED, images_by_node=(PINNED, PINNED)))
    distribution = {}

    def distribute(*args, **kwargs):
        distribution.update(kwargs)
        return None, {}, {}, {}

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    stage_prepared_images(
        prepared,
        recipe,
        HOSTS,
        "/cache/hf",
        SimpleNamespace(),
        stage_models=True,
    )

    assert distribution["skip_model"] is False
    assert distribution["skip_container"] is False
    assert "model_revision" not in distribution
