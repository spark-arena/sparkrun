"""Verified provider IDs reach each Docker launch without mutating recipe pins."""

from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from sparkrun.core import image_distribution as api
from sparkrun.orchestration.executors.docker import DockerExecutor

PIN = "registry.test/image:canary@sha256:" + "a" * 64
IDS = {"a": "sha256:" + "b" * 64, "b": "sha256:" + "c" * 64}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(api, "_PROVIDERS", {})
    bindings = api._RUNTIME_IMAGES.set(None)
    config = api._CONFIG.set(None)
    yield
    api._RUNTIME_IMAGES.reset(bindings)
    api._CONFIG.reset(config)


def pull():
    return api.try_image_pull(
        image=PIN,
        source_host=None,
        targets=list(IDS),
        transfer_hosts=None,
        ssh_user=None,
        ssh_key=None,
        ssh_options=None,
        timeout=30,
        dry_run=False,
        offline=False,
        force_pull=False,
    )


def test_nested_distribution_context_preserves_ids_then_clears_them():
    api.register_image_distribution_provider(
        "test",
        SimpleNamespace(
            copy=lambda r: None,
            pull=lambda r: api.ImageCopyResult(dict.fromkeys(r.targets, "complete"), runtime_images=IDS),
        ),
    )

    @api.image_distribution_operation
    def distribute(config):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(copy_context().run, pull).result() == []

    @api.image_distribution_operation
    def launch(config):
        distribute(config)
        assert {h: api.resolve_distributed_image(PIN, h) for h in IDS} == IDS
        assert api.resolve_distributed_image(PIN, "a", dry_run=True) == PIN

    launch({})
    assert api._RUNTIME_IMAGES.get() is None
    assert api.resolve_distributed_image(PIN, "a") == PIN


@pytest.mark.parametrize(
    "runtime_images,errors,outcomes",
    [
        ({"stranger": IDS["a"]}, {}, {"a": "complete"}),
        ({"a": "mutable:tag"}, {}, {"a": "complete"}),
        ({"a": "sha256:short"}, {}, {"a": "complete"}),
        ({"a": IDS["a"]}, {"a": "failed"}, {"a": "complete"}),
        ({"a": IDS["a"]}, {}, {"a": "failed"}),
    ],
)
def test_provider_cannot_bind_unknown_failed_or_mutable_images(runtime_images, errors, outcomes):
    token = api._RUNTIME_IMAGES.set({})
    try:
        with pytest.raises(ValueError):
            api._record_runtime_images(PIN, api.ImageCopyResult(outcomes, errors, runtime_images), dry_run=False)
        assert api._RUNTIME_IMAGES.get() == {}
    finally:
        api._RUNTIME_IMAGES.reset(token)


def test_offline_pre_pull_is_an_explicit_optional_capability():
    requests = []
    api.register_image_distribution_provider(
        "test",
        SimpleNamespace(
            supports_offline_pull=True,
            copy=lambda r: None,
            pull=lambda r: requests.append(r) or api.ImageCopyResult(dict.fromkeys(r.targets, "already_present"), runtime_images=IDS),
        ),
    )
    assert (
        api.try_image_pull(
            image=PIN,
            source_host=None,
            targets=list(IDS),
            transfer_hosts=None,
            ssh_user=None,
            ssh_key=None,
            ssh_options=None,
            timeout=30,
            dry_run=False,
            offline=True,
            force_pull=False,
        )
        == []
    )
    assert requests[0].offline is True


def test_runtime_launch_uses_each_stores_immutable_id_and_never_pulls():
    executors = {h: DockerExecutor() for h in IDS}
    for host, executor in executors.items():
        executor.bind_image_references({PIN: IDS[host]})
    head = executors["a"]
    head.bind_host_executors(executors)
    for host in IDS:
        command = head.for_host(host).run_cmd(PIN, "true")
        assert IDS[host] in command and "--pull=never" in command and PIN not in command
        assert "--pull=never" not in head.for_host(host).run_cmd("other:tag")
        for flag in ["--pull=always", "--pull always"]:
            with pytest.raises(ValueError, match="verified runtime image"):
                head.for_host(host).run_cmd(PIN, extra_opts=[flag])


def test_later_operation_resolves_pin_through_provider_and_honors_builtin(monkeypatch):
    calls = []
    api.register_image_distribution_provider(
        "test",
        SimpleNamespace(
            copy=lambda r: None,
            local_image=lambda r: calls.append(r) or IDS["a"],
        ),
    )
    assert api.resolve_distributed_image(PIN, "a") == IDS["a"]
    assert calls[0].source_host == "a" and calls[0].offline
    api._CONFIG.set({"container_distribution_provider": "builtin"})
    assert api.resolve_distributed_image(PIN, "a") == PIN
    assert len(calls) == 1


def test_staging_inspects_each_hosts_bound_id(monkeypatch):
    from sparkrun.core.image_preparation import resolve_content_images

    seen = []

    def run(host, command, **kwargs):
        seen.append((host, command))
        return SimpleNamespace(success=True, stdout=IDS[host])

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", run)

    @api.image_distribution_operation
    def stage(config):
        api._record_runtime_images(PIN, api.ImageCopyResult(dict.fromkeys(IDS, "complete"), runtime_images=IDS), dry_run=False)
        return resolve_content_images([PIN, PIN], list(IDS))

    assert stage({}) == tuple(IDS.values())
    assert len(seen) == 2 and all(IDS[host] in command and PIN not in command for host, command in seen)


@pytest.mark.parametrize("container_images", [True, False])
def test_launch_hands_off_ids_to_probes_and_executors_but_preserves_metadata(monkeypatch, tmp_path, container_images):
    from sparkrun.core import launcher
    from sparkrun.core.recipe import Recipe
    from test_launcher import _StubRuntime

    class Config(dict):
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"
        missing_mount_source_policy = "fail"
        readiness_port_timeout_s = 240
        readiness_health_timeout_s = 600

        def get_registry_manager(self):
            return None

    class Runtime(_StubRuntime):
        supports_heterogeneous_images = False  # Store IDs do not make the recipe heterogeneous.

        def resolve_container(self, recipe, **kwargs):
            return recipe.container

        def run(self, **kwargs):
            if not container_images:
                return super().run(**kwargs)
            assert kwargs["image"] == PIN and kwargs["images_by_node"] is None
            for host in IDS:
                command = kwargs["executor"].for_host(host).run_cmd(kwargs["image"], "true")
                assert IDS[host] in command and "--pull=never" in command and PIN not in command
            return super().run(**kwargs)

    recipe = Recipe.from_dict(
        {"name": "pin-handoff", "runtime": "vllm-distributed", "model": "org/model", "container": PIN, "defaults": {"port": 8000}}
    )
    metadata = []
    probes = []
    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda *a, **kw: SimpleNamespace(mode="local"))

    def distribute(*a, **kw):
        api._record_runtime_images(PIN, api.ImageCopyResult(dict.fromkeys(IDS, "complete"), runtime_images=IDS), dry_run=False)
        kw["after_container_sync"]()
        return None, {}, {}, {}

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", lambda *a, **kw: metadata.append(kw))
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.derive_cluster_id", lambda *a, **kw: "sparkrun_pin_test")
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr(
        launcher, "_verify_image_command_passthrough", lambda recipe, image, hosts, *a, **kw: probes.append((hosts[0], image))
    )
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", lambda **kw: DockerExecutor())
    monkeypatch.setattr(DockerExecutor, "prepare_launch", lambda *a, **kw: None)
    monkeypatch.setattr(DockerExecutor, "needs_image", container_images)
    monkeypatch.setattr("sparkrun.core.asset_preparation.prepare_tuning", lambda *a, **kw: None)
    launcher.launch_inference(recipe=recipe, runtime=Runtime(), host_list=list(IDS), overrides={}, config=Config(), sync_tuning=False)
    assert dict(probes) == (IDS if container_images else {})
    assert recipe.container == PIN
    assert metadata
    if container_images:
        assert all(m["container_image"] == PIN for m in metadata)
    assert api._RUNTIME_IMAGES.get() is None


def test_operation_uses_sctx_config_when_no_direct_config_is_supplied():
    config = {"container_distribution_provider": "builtin"}

    @api.image_distribution_operation
    def launch(*, config=None, sctx=None):
        assert api._CONFIG.get() is sctx.config

    launch(sctx=SimpleNamespace(config=config))
    assert api._CONFIG.get() is None


def test_offline_preflight_accepts_relay_imports_without_upstream_repo_digests(monkeypatch):
    from sparkrun.core import launcher
    from sparkrun.core.recipe import Recipe

    api.register_image_distribution_provider(
        "test",
        SimpleNamespace(
            copy=lambda r: None,
            local_image=lambda r: IDS.get(r.source_host),
        ),
    )
    seen = []

    def identities(image, hosts, **kw):
        seen.append((image, hosts))
        assert all(image == IDS[host] for host in hosts)
        return {host: (image, []) for host in hosts}

    monkeypatch.setattr("sparkrun.containers.distribute._check_remote_image_identities", identities)
    monkeypatch.setattr("sparkrun.containers.registry.image_exists_locally", lambda image: False)
    recipe = Recipe.from_dict({"model": "org/model", "runtime": "vllm-distributed", "container": PIN})
    runtime = SimpleNamespace(
        runtime_name="stub", supports_heterogeneous_images=False, resolve_container=lambda recipe, **kw: recipe.container
    )
    launcher._offline_preflight(
        recipe,
        runtime,
        list(IDS),
        cluster=None,
        launch_hardware=None,
        config={},
        v=None,
        ssh_kwargs={},
        transfer_result=SimpleNamespace(mode="local", auto_delegated=False),
        cache_dir=None,
        local_cache_dir=None,
        skip_model=True,
        skip_images=False,
        shared_model_cache=False,
    )
    assert len(seen) == 2


def test_optional_staging_also_returns_locally_runnable_pins(monkeypatch):
    from sparkrun.core.image_preparation import prepare_images, stage_prepared_images
    from sparkrun.core.recipe import Recipe

    recipe = Recipe.from_dict({"model": "org/model", "runtime": "vllm-distributed", "container": PIN})
    runtime = SimpleNamespace(
        runtime_name="stub", supports_heterogeneous_images=False, resolve_container=lambda recipe, **kw: recipe.container
    )
    prepared = prepare_images(recipe, runtime, list(IDS))

    def distribute(*args, **kwargs):
        api._record_runtime_images(PIN, api.ImageCopyResult(dict.fromkeys(IDS, "complete"), runtime_images=IDS), dry_run=False)
        return None, {}, {}, {}

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    config = SimpleNamespace(ssh_user=None, ssh_key=None, ssh_options=[])
    staged = stage_prepared_images(prepared, recipe, list(IDS), "/cache", config)
    assert staged.content_images_by_node == tuple(IDS.values())
    assert staged.prepared.images_by_node == (PIN, PIN)


@pytest.mark.parametrize("operation", ["pull", "copy", "resolve"])
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_provider_session_ownership_on_success_and_failure(monkeypatch, operation, borrowed, fails):
    from unittest.mock import Mock

    session = SimpleNamespace(close=Mock())
    factory = Mock(return_value=session)
    monkeypatch.setattr("sparkrun.transports.session.SshHostSession", factory)
    requests = []

    def provider(request):
        requests.append(request)
        assert request.session is session
        if fails:
            raise RuntimeError("transfer failed")
        if operation == "resolve":
            return IDS["a"]
        return api.ImageCopyResult({"a": "complete"}, runtime_images={"a": IDS["a"]})

    api.register_image_distribution_provider("test", SimpleNamespace(copy=provider, pull=provider, local_image=provider))
    kwargs = {"session": session} if borrowed else {}

    def invoke():
        if operation == "resolve":
            return api.resolve_distributed_image(PIN, "a", **kwargs)
        return getattr(api, "try_image_" + operation)(
            image=PIN,
            source_host=None,
            targets=["a"],
            transfer_hosts=None,
            ssh_user=None,
            ssh_key=None,
            ssh_options=None,
            timeout=30,
            dry_run=False,
            offline=False,
            **({"force_pull": False} if operation == "pull" else {}),
            **kwargs,
        )

    if fails:
        with pytest.raises(RuntimeError, match="transfer failed"):
            invoke()
    else:
        assert invoke() == (IDS["a"] if operation == "resolve" else [])
    assert len(requests) == 1
    assert factory.call_count == (0 if borrowed else 1)
    assert session.close.call_count == (0 if borrowed else 1)
