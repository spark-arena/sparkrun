"""Tests for ``sparkrun.core.launcher`` backend resolution (A1)."""

from __future__ import annotations

import pytest

from sparkrun.core.backend_select import BackendBundle
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.launcher import resolve_per_host_backends
from sparkrun.orchestration.collectives import NcclBackend, RcclBackend


def _nvidia_hw() -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10")])


def _amd_hw() -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x")])


def _apple_hw() -> HostHardware:
    return HostHardware(accelerators=[AcceleratorSpec(vendor="apple", model="m5")])


# ---------------------------------------------------------------------------
# Default DGX fallback (no cluster threaded)
# ---------------------------------------------------------------------------


def test_resolve_per_host_backends_no_cluster_defaults_to_dgx_nvidia():
    """When no cluster is provided, every host defaults to DGX Spark / NVIDIA."""
    backends = resolve_per_host_backends(["10.0.0.1", "10.0.0.2"], cluster=None)
    assert set(backends.keys()) == {"10.0.0.1", "10.0.0.2"}
    for host, bundle in backends.items():
        assert isinstance(bundle, BackendBundle)
        assert bundle.accelerator_vendor == "nvidia"
        assert isinstance(bundle.collective, NcclBackend)


def test_resolve_per_host_backends_empty_host_list_empty_map():
    assert resolve_per_host_backends([], cluster=None) == {}


# ---------------------------------------------------------------------------
# Cluster-aware resolution
# ---------------------------------------------------------------------------


def test_resolve_per_host_backends_uses_cluster_hardware():
    """Cluster hosts_hardware drives per-host vendor selection."""
    cluster = ClusterDefinition(
        name="mixed",
        hosts=["nvidia-host", "amd-host"],
        hosts_hardware={
            "nvidia-host": _nvidia_hw(),
            "amd-host": _amd_hw(),
        },
    )
    backends = resolve_per_host_backends(cluster.hosts, cluster=cluster)
    assert backends["nvidia-host"].accelerator_vendor == "nvidia"
    assert isinstance(backends["nvidia-host"].collective, NcclBackend)
    assert backends["amd-host"].accelerator_vendor == "amd"
    assert isinstance(backends["amd-host"].collective, RcclBackend)


def test_resolve_per_host_backends_missing_entry_falls_back_to_dgx():
    """Hosts without an explicit hosts_hardware entry use DGX Spark default."""
    cluster = ClusterDefinition(
        name="partial",
        hosts=["explicit-amd", "implicit-host"],
        hosts_hardware={"explicit-amd": _amd_hw()},
    )
    backends = resolve_per_host_backends(cluster.hosts, cluster=cluster)
    assert backends["explicit-amd"].accelerator_vendor == "amd"
    # Implicit host -> DGX Spark fallback -> NVIDIA / NCCL
    assert backends["implicit-host"].accelerator_vendor == "nvidia"


def test_resolve_per_host_backends_unknown_vendor_skipped_silently():
    """A host with an unsupported vendor is omitted (runtime falls back to legacy IB path)."""
    cluster = ClusterDefinition(
        name="apple-mix",
        hosts=["nvidia-host", "apple-host"],
        hosts_hardware={
            "nvidia-host": _nvidia_hw(),
            "apple-host": _apple_hw(),
        },
    )
    backends = resolve_per_host_backends(cluster.hosts, cluster=cluster)
    assert "nvidia-host" in backends
    assert "apple-host" not in backends


# ---------------------------------------------------------------------------
# Threading into runtime.run via launch_inference
# ---------------------------------------------------------------------------


class _StubRuntime:
    """Minimal RuntimePlugin-ish stub that records ``run()`` kwargs."""

    runtime_name = "stub"
    requires_capability: frozenset = frozenset()
    last_kwargs: dict = {}

    def is_delegating_runtime(self):
        return False

    def resolve_container(self, recipe, overrides=None):
        return "stub:latest"

    def prepare(self, *args, **kwargs):
        return None

    def get_head_container_name(self, cluster_id, is_solo=False):
        return "%s_solo" % cluster_id

    def generate_command(self, **kwargs):
        return "echo serve"

    def resolve_api_key(self, recipe, overrides=None):
        return None

    def _collect_runtime_info(self, *args, **kwargs):
        return {}

    def run(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        return 0


def test_launch_inference_threads_backends_to_runtime_run(monkeypatch, tmp_path):
    """launch_inference resolves backends and passes them to runtime.run()."""
    from sparkrun.core import launcher
    from sparkrun.core.launcher import launch_inference

    # Make all the heavy-lift / network helpers no-ops.
    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.distribute_from_config",
        lambda *a, **kw: (None, {}, {}),
    )
    # save_job_metadata is imported lazily inside launch_inference.
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.save_job_metadata",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.derive_cluster_id",
        lambda *a, **kw: "sparkrun_testabc12345",
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.build_ssh_kwargs",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(
        launcher,
        "resolve_effective_cache_dir",
        lambda *a, **kw: str(tmp_path),
    )
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr(
        "sparkrun.orchestration.executor.resolve_executor",
        lambda **kw: type("Ex", (), {})(),
    )

    # Fake config
    class _Cfg:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    # Fake recipe
    class _Recipe:
        runtime = "stub"
        model = "stub-model"
        env = {}
        builder = None
        mods = []
        source_registry = None
        source_registry_url = None
        defaults = {"port": 8000}
        pre_exec = []
        post_exec = []
        post_commands = []
        layout = None
        stop_after_post = False
        executor = ""
        executor_config = None
        is_url_sourced = False
        cluster_config = None
        qualified_name = "stub-recipe"
        name = "stub-recipe"
        container = "stub:latest"
        model_revision = None

        def build_config_chain(self, overrides=None):
            class _CC:
                def get(self, k, default=None):
                    return (overrides or {}).get(k, self_outer.defaults.get(k, default))

            self_outer = self
            return _CC()

        def __getstate__(self):
            return {}

    runtime = _StubRuntime()
    cluster = ClusterDefinition(
        name="t",
        hosts=["nv-host"],
        hosts_hardware={"nv-host": _nvidia_hw()},
    )

    result = launch_inference(
        recipe=_Recipe(),
        runtime=runtime,
        host_list=["nv-host"],
        overrides={},
        config=_Cfg(),
        cluster=cluster,
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
    )

    # backends in LaunchResult and threaded to runtime.run
    assert "nv-host" in result.backends
    assert isinstance(result.backends["nv-host"], BackendBundle)
    threaded = _StubRuntime.last_kwargs.get("backends")
    assert threaded is not None
    assert "nv-host" in threaded
    assert isinstance(threaded["nv-host"].collective, NcclBackend)


# ---------------------------------------------------------------------------
# Platform validate_host warnings are logged but do not raise
# ---------------------------------------------------------------------------


def test_launch_inference_logs_platform_warnings_without_raising(monkeypatch, tmp_path, caplog):
    """validate_host warnings appear in the log at WARNING level but do not abort launch."""
    import logging

    from sparkrun.core import launcher
    from sparkrun.core.launcher import launch_inference

    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.distribute_from_config",
        lambda *a, **kw: (None, {}, {}),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.save_job_metadata",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.derive_cluster_id",
        lambda *a, **kw: "sparkrun_testabc12345",
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.build_ssh_kwargs",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(
        launcher,
        "resolve_effective_cache_dir",
        lambda *a, **kw: str(tmp_path),
    )
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr(
        "sparkrun.orchestration.executor.resolve_executor",
        lambda **kw: type("Ex", (), {})(),
    )

    # Build a host with a GB10 accelerator but WITHOUT RoCEv2 — DgxSparkPlatform
    # will emit a warning about the missing capability.
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    hw_no_roce = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", capabilities=frozenset({"cuda"}))])
    cluster = ClusterDefinition(
        name="warn-test",
        hosts=["dgx-host"],
        hosts_hardware={"dgx-host": hw_no_roce},
    )

    class _Cfg:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    class _Recipe:
        runtime = "stub"
        model = "stub-model"
        env = {}
        builder = None
        mods = []
        source_registry = None
        source_registry_url = None
        defaults = {"port": 8000}
        pre_exec = []
        post_exec = []
        post_commands = []
        layout = None
        stop_after_post = False
        executor = ""
        executor_config = None
        is_url_sourced = False
        cluster_config = None
        qualified_name = "stub-recipe"
        name = "stub-recipe"
        container = "stub:latest"
        model_revision = None
        requires_capability: frozenset = frozenset()

        def build_config_chain(self, overrides=None):
            class _CC:
                def get(self, k, default=None):
                    return (overrides or {}).get(k, self_outer.defaults.get(k, default))

            self_outer = self
            return _CC()

    runtime = _StubRuntime()

    with caplog.at_level(logging.WARNING, logger="sparkrun.core.launcher"):
        result = launch_inference(
            recipe=_Recipe(),
            runtime=runtime,
            host_list=["dgx-host"],
            overrides={},
            config=_Cfg(),
            cluster=cluster,
            is_solo=True,
            dry_run=True,
            sync_tuning=False,
        )

    # Launch must succeed (return 0 from stub runtime)
    assert result.rc == 0

    # At least one warning mentioning the host and the missing capability
    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("dgx-host" in w and "rdma:roce-v2" in w for w in warning_texts), (
        "Expected a warning about missing rdma:roce-v2 for dgx-host, got: %s" % warning_texts
    )


# ---------------------------------------------------------------------------
# post_launch_lifecycle: health-check / hook failure & stop paths
# ---------------------------------------------------------------------------


class _LifecycleRecipe:
    """Recipe stub for post_launch_lifecycle with configurable hook fields."""

    def __init__(self, *, post_exec=None, post_commands=None, stop_after_post=False, port=8000):
        self.post_exec = post_exec or []
        self.post_commands = post_commands or []
        self.stop_after_post = stop_after_post
        self.source_registry = None
        self._port = port

    # resolve_recipe_trust introspects these.
    is_url_sourced = False

    def build_config_chain(self, overrides=None):
        return {"port": self._port}


class _LifecycleRuntime:
    """Runtime stub recording stop() invocations."""

    def __init__(self):
        self.stop_calls: list[dict] = []

    def stop(self, **kwargs):
        self.stop_calls.append(dict(kwargs))
        return 0


def _make_launch_result(recipe, runtime):
    from sparkrun.core.launcher import LaunchResult

    class _Cfg:
        pass

    return LaunchResult(
        rc=0,
        cluster_id="sparkrun_lifecyclecid",
        host_list=["localhost"],
        is_solo=True,
        runtime=runtime,
        recipe=recipe,
        overrides={},
        container_image="img:latest",
        effective_cache_dir="/tmp/cache",
        serve_port=8000,
        config=_Cfg(),
    )


def _patch_lifecycle_common(monkeypatch, *, port_ready=True, healthy=True):
    """Patch the orchestration helpers post_launch_lifecycle imports lazily.

    is_local_host -> True keeps head_ip on 127.0.0.1 (no detect_host_ip SSH).
    """
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **k: {})
    monkeypatch.setattr("sparkrun.utils.is_local_host", lambda host: True)
    monkeypatch.setattr("sparkrun.orchestration.docker.generate_container_name", lambda cid, suffix: "%s_%s" % (cid, suffix))
    monkeypatch.setattr("sparkrun.orchestration.docker.generate_node_container_name", lambda cid, rank: "%s_node_%d" % (cid, rank))
    monkeypatch.setattr("sparkrun.orchestration.health.wait_for_port", lambda *a, **k: port_ready)
    monkeypatch.setattr("sparkrun.orchestration.health.wait_for_healthy", lambda *a, **k: healthy)


def test_post_launch_lifecycle_port_timeout_exits_1(monkeypatch):
    """wait_for_port returning False -> SystemExit(1) with a port error."""
    from sparkrun.core.launcher import post_launch_lifecycle

    _patch_lifecycle_common(monkeypatch, port_ready=False)
    recipe = _LifecycleRecipe(post_commands=["echo hi"])
    runtime = _LifecycleRuntime()
    result = _make_launch_result(recipe, runtime)

    with pytest.raises(SystemExit) as exc:
        post_launch_lifecycle(result, remote_cache_dir="/tmp/cache")
    assert exc.value.code == 1
    # Stop must not have been called — we never reached stop_after_post.
    assert runtime.stop_calls == []


def test_post_launch_lifecycle_health_timeout_exits_1(monkeypatch):
    """wait_for_healthy returning False -> SystemExit(1) with a health error."""
    from sparkrun.core.launcher import post_launch_lifecycle

    _patch_lifecycle_common(monkeypatch, port_ready=True, healthy=False)
    recipe = _LifecycleRecipe(post_commands=["echo hi"])
    runtime = _LifecycleRuntime()
    result = _make_launch_result(recipe, runtime)

    with pytest.raises(SystemExit) as exc:
        post_launch_lifecycle(result, remote_cache_dir="/tmp/cache")
    assert exc.value.code == 1


def test_post_launch_lifecycle_hook_runtime_error_exits_1(monkeypatch):
    """A RuntimeError from a post hook surfaces as SystemExit(1)."""
    from sparkrun.core.launcher import post_launch_lifecycle

    _patch_lifecycle_common(monkeypatch)
    monkeypatch.setattr("sparkrun.orchestration.hooks.build_hook_context", lambda *a, **k: {})

    def _boom(*a, **k):
        raise RuntimeError("post_commands failed")

    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_commands", _boom)
    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_exec", lambda *a, **k: None)

    recipe = _LifecycleRecipe(post_commands=["false"])
    runtime = _LifecycleRuntime()
    result = _make_launch_result(recipe, runtime)

    with pytest.raises(SystemExit) as exc:
        post_launch_lifecycle(result, remote_cache_dir="/tmp/cache")
    assert exc.value.code == 1


def test_post_launch_lifecycle_happy_path_runs_hooks_no_exit(monkeypatch):
    """Port + health OK, hooks succeed, no stop_after_post -> returns normally."""
    from sparkrun.core.launcher import post_launch_lifecycle

    _patch_lifecycle_common(monkeypatch)
    monkeypatch.setattr("sparkrun.orchestration.hooks.build_hook_context", lambda *a, **k: {})

    exec_calls: list = []
    cmd_calls: list = []
    monkeypatch.setattr(
        "sparkrun.orchestration.hooks.run_post_exec",
        lambda *a, **k: exec_calls.append(k.get("trust")),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.hooks.run_post_commands",
        lambda *a, **k: cmd_calls.append(k.get("trust")),
    )

    recipe = _LifecycleRecipe(post_exec=["echo inside"], post_commands=["echo outside"])
    runtime = _LifecycleRuntime()
    result = _make_launch_result(recipe, runtime)

    # Must NOT raise SystemExit (no stop_after_post).
    post_launch_lifecycle(result, remote_cache_dir="/tmp/cache")

    # Both hook runners fired; local recipe (source_registry=None) is trusted.
    assert exec_calls == [True]
    assert cmd_calls == [True]
    assert runtime.stop_calls == []


def test_post_launch_lifecycle_stop_after_post_stops_and_exits_0(monkeypatch):
    """stop_after_post -> runtime.stop is invoked and the process exits 0."""
    from sparkrun.core.launcher import post_launch_lifecycle

    _patch_lifecycle_common(monkeypatch)
    monkeypatch.setattr("sparkrun.orchestration.hooks.build_hook_context", lambda *a, **k: {})
    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_exec", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_commands", lambda *a, **k: None)

    recipe = _LifecycleRecipe(post_commands=["echo hi"], stop_after_post=True)
    runtime = _LifecycleRuntime()
    result = _make_launch_result(recipe, runtime)

    with pytest.raises(SystemExit) as exc:
        post_launch_lifecycle(result, remote_cache_dir="/tmp/cache")
    assert exc.value.code == 0
    assert len(runtime.stop_calls) == 1
    assert runtime.stop_calls[0]["cluster_id"] == "sparkrun_lifecyclecid"


def test_post_launch_lifecycle_dry_run_skips_health_waits(monkeypatch):
    """dry_run=True skips the port/health waits entirely (no SystemExit)."""
    from sparkrun.core.launcher import post_launch_lifecycle

    # Make the health helpers explode if called — dry_run must not reach them.
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **k: {})
    monkeypatch.setattr("sparkrun.utils.is_local_host", lambda host: True)
    monkeypatch.setattr("sparkrun.orchestration.docker.generate_container_name", lambda cid, suffix: "%s_%s" % (cid, suffix))
    monkeypatch.setattr("sparkrun.orchestration.docker.generate_node_container_name", lambda cid, rank: "%s_node_%d" % (cid, rank))

    def _must_not_call(*a, **k):
        raise AssertionError("health wait must be skipped under dry_run")

    monkeypatch.setattr("sparkrun.orchestration.health.wait_for_port", _must_not_call)
    monkeypatch.setattr("sparkrun.orchestration.health.wait_for_healthy", _must_not_call)
    monkeypatch.setattr("sparkrun.orchestration.hooks.build_hook_context", lambda *a, **k: {})
    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_exec", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_commands", lambda *a, **k: None)

    recipe = _LifecycleRecipe(post_commands=["echo hi"])
    runtime = _LifecycleRuntime()
    result = _make_launch_result(recipe, runtime)

    post_launch_lifecycle(result, remote_cache_dir="/tmp/cache", dry_run=True)


# ---------------------------------------------------------------------------
# launch_inference: best-effort metadata persistence (except blocks)
# ---------------------------------------------------------------------------


def test_launch_inference_save_job_metadata_failure_is_best_effort(monkeypatch, tmp_path):
    """save_job_metadata raising must NOT abort a non-dry-run launch.

    Exercises the best-effort ``except Exception`` guard around the
    initial metadata persistence in launch_inference: the launch still
    completes and returns rc=0 from the stub runtime even though metadata
    persistence blew up.
    """
    from sparkrun.core import launcher
    from sparkrun.core.launcher import launch_inference

    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )
    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.distribute_from_config",
        lambda *a, **kw: (None, {}, {}),
    )

    save_calls: list = []

    def _save_boom(*a, **kw):
        save_calls.append(1)
        raise OSError("disk full")

    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", _save_boom)
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.derive_cluster_id",
        lambda *a, **kw: "sparkrun_metafailcid01",
    )
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", lambda **kw: type("Ex", (), {})())
    # Tuning sync/distribute are best-effort too; stub them to no-ops.
    monkeypatch.setattr("sparkrun.tuning.sync.sync_registry_tuning", lambda *a, **kw: 0)
    monkeypatch.setattr("sparkrun.tuning.distribute.distribute_tuning_to_hosts", lambda *a, **kw: [])

    class _Cfg:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    class _Recipe:
        runtime = "stub"
        model = "stub-model"
        env = {}
        builder = None
        mods = []
        source_registry = None
        source_registry_url = None
        defaults = {"port": 8000}
        pre_exec = []
        post_exec = []
        post_commands = []
        layout = None
        stop_after_post = False
        executor = ""
        executor_config = None
        is_url_sourced = False
        cluster_config = None
        qualified_name = "stub-recipe"
        name = "stub-recipe"
        container = "stub:latest"
        model_revision = None
        requires_capability: frozenset = frozenset()

        def build_config_chain(self, overrides=None):
            class _CC:
                def get(self, k, default=None):
                    return (overrides or {}).get(k, self_outer.defaults.get(k, default))

            self_outer = self
            return _CC()

    runtime = _StubRuntime()

    # dry_run=False so save_job_metadata is actually reached.
    result = launch_inference(
        recipe=_Recipe(),
        runtime=runtime,
        host_list=["nv-host"],
        overrides={},
        config=_Cfg(),
        is_solo=True,
        dry_run=False,
        sync_tuning=False,
    )

    assert result.rc == 0
    assert save_calls, "save_job_metadata should have been attempted"
