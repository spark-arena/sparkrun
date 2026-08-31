"""Tests for ``sparkrun.core.launcher`` backend resolution (A1)."""

from __future__ import annotations

import pytest

from sparkrun.core.backend_select import BackendBundle
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.launcher import (
    apply_platform_runtime_flag_defaults,
    resolve_per_host_backends,
    resolve_platform_env_defaults,
)
from sparkrun.core.recipe import Recipe
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
# Platform/runtime/accelerator flag defaults (GB10 llama.cpp mmap off)
# ---------------------------------------------------------------------------


def _llama_recipe(defaults=None):
    return Recipe.from_dict(
        {
            "name": "vl",
            "model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "defaults": defaults or {},
        }
    )


def test_platform_flag_defaults_applies_mmap_off_for_gb10_llama_cpp():
    recipe = _llama_recipe()
    applied = apply_platform_runtime_flag_defaults(recipe, "llama-cpp", _nvidia_hw())
    assert applied == {"mmap": False}
    assert recipe.defaults["mmap"] is False


def test_platform_flag_defaults_respects_explicit_recipe_value():
    """An explicit recipe mmap:true is preserved (setdefault semantics)."""
    recipe = _llama_recipe({"mmap": True})
    applied = apply_platform_runtime_flag_defaults(recipe, "llama-cpp", _nvidia_hw())
    assert applied == {}
    assert recipe.defaults["mmap"] is True


def test_platform_flag_defaults_non_llama_runtime_noop():
    recipe = _llama_recipe()
    applied = apply_platform_runtime_flag_defaults(recipe, "vllm-distributed", _nvidia_hw())
    assert applied == {}
    assert "mmap" not in recipe.defaults


def test_platform_flag_defaults_non_gb10_noop():
    recipe = _llama_recipe()
    applied = apply_platform_runtime_flag_defaults(recipe, "llama-cpp", _amd_hw())
    assert applied == {}
    assert "mmap" not in recipe.defaults


# ---------------------------------------------------------------------------
# Platform/runtime container env defaults (GB10 torch expandable segments)
# ---------------------------------------------------------------------------


class _StubRuntimeForEnv:
    def __init__(self, runtime_name: str, family: str | None = None):
        self.runtime_name = runtime_name
        self._family = family or runtime_name

    def get_family(self) -> str:
        return self._family


_EXPANDABLE = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@pytest.mark.parametrize(
    "runtime_name,family",
    [
        ("vllm-ray", "vllm"),
        ("vllm-distributed", "vllm"),
        ("eugr-vllm", "vllm"),  # family match — variant is never enumerated
        ("sglang", "sglang"),
    ],
)
def test_platform_env_defaults_for_torch_runtimes_on_gb10(runtime_name, family):
    got = resolve_platform_env_defaults(_StubRuntimeForEnv(runtime_name, family), _nvidia_hw())
    assert got == _EXPANDABLE


@pytest.mark.parametrize("runtime_name", ["llama-cpp", "trtllm", "modular-max"])
def test_platform_env_defaults_skips_non_torch_runtimes(runtime_name):
    """Runtimes that manage their own memory get no allocator env."""
    assert resolve_platform_env_defaults(_StubRuntimeForEnv(runtime_name), _nvidia_hw()) == {}


def test_platform_env_defaults_non_gb10_noop():
    """A non-GB10 NVIDIA host is served by the generic platform — no env."""
    h100 = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100")])
    assert resolve_platform_env_defaults(_StubRuntimeForEnv("vllm-ray", "vllm"), h100) == {}


def test_platform_env_defaults_unclaimed_hardware_noop():
    assert resolve_platform_env_defaults(_StubRuntimeForEnv("vllm-ray", "vllm"), _amd_hw()) == {}


def test_platform_env_defaults_no_hardware_noop():
    assert resolve_platform_env_defaults(_StubRuntimeForEnv("vllm-ray", "vllm"), None) == {}
    assert resolve_platform_env_defaults(_StubRuntimeForEnv("vllm-ray", "vllm"), HostHardware()) == {}


def test_platform_env_defaults_platform_error_is_swallowed(monkeypatch):
    """A misbehaving platform hook contributes nothing rather than failing the launch."""
    from sparkrun.platforms.dgx_spark import DgxSparkPlatform

    def _raise(self, runtime_name, accelerator, *, runtime_family=None):
        raise ValueError("boom")

    monkeypatch.setattr(DgxSparkPlatform, "default_env", _raise)
    assert resolve_platform_env_defaults(_StubRuntimeForEnv("vllm-ray", "vllm"), _nvidia_hw()) == {}


def test_platform_env_defaults_returns_a_copy():
    """Callers must not be able to mutate the platform's table via the result."""
    runtime = _StubRuntimeForEnv("sglang", "sglang")
    got = resolve_platform_env_defaults(runtime, _nvidia_hw())
    got["PYTORCH_CUDA_ALLOC_CONF"] = "mutated"
    assert resolve_platform_env_defaults(runtime, _nvidia_hw()) == _EXPANDABLE


def test_platform_flag_defaults_then_command_emits_no_mmap():
    """End-to-end: GB10 default flows into the rendered llama.cpp command."""
    from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

    recipe = _llama_recipe({"port": 8001})
    apply_platform_runtime_flag_defaults(recipe, "llama-cpp", _nvidia_hw())
    cmd = LlamaCppRuntime().generate_command(recipe, {}, is_cluster=False)
    assert "--no-mmap" in cmd


def test_platform_flag_defaults_explicit_true_suppresses_no_mmap():
    from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

    recipe = _llama_recipe({"port": 8001, "mmap": True})
    apply_platform_runtime_flag_defaults(recipe, "llama-cpp", _nvidia_hw())
    cmd = LlamaCppRuntime().generate_command(recipe, {}, is_cluster=False)
    assert "--no-mmap" not in cmd


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
        lambda *a, **kw: (None, {}, {}, {}),
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
        lambda *a, **kw: (None, {}, {}, {}),
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
        lambda *a, **kw: (None, {}, {}, {}),
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


# ---------------------------------------------------------------------------
# wait_for_serve_ready
# ---------------------------------------------------------------------------


def _patch_serve_ready(monkeypatch, *, port_ready=True, healthy=True):
    """Patch the probes wait_for_serve_ready imports lazily, recording calls."""
    calls: list[str] = []

    def _port(*_a, **_k):
        calls.append("port")
        return port_ready

    def _health(*_a, **_k):
        calls.append("health")
        return healthy

    monkeypatch.setattr("sparkrun.utils.is_local_host", lambda host: True)
    monkeypatch.setattr("sparkrun.orchestration.docker.generate_container_name", lambda cid, suffix: "%s_%s" % (cid, suffix))
    monkeypatch.setattr("sparkrun.orchestration.docker.generate_node_container_name", lambda cid, rank: "%s_node_%d" % (cid, rank))
    monkeypatch.setattr("sparkrun.orchestration.health.wait_for_port", _port)
    monkeypatch.setattr("sparkrun.orchestration.health.wait_for_healthy", _health)
    return calls


def test_wait_for_serve_ready_ready(monkeypatch):
    """Both probes passing -> ready, no reason, and a usable health URL."""
    from sparkrun.core.launcher import wait_for_serve_ready

    calls = _patch_serve_ready(monkeypatch)
    result = _make_launch_result(_LifecycleRecipe(), _LifecycleRuntime())

    readiness = wait_for_serve_ready(result)

    assert readiness.ready is True
    assert readiness.reason == ""
    assert readiness.head_ip == "127.0.0.1"
    assert readiness.health_url == "http://127.0.0.1:8000/v1/models"
    assert readiness.container == "sparkrun_lifecyclecid_solo"
    assert calls == ["port", "health"]


def test_wait_for_serve_ready_port_failure_skips_health(monkeypatch):
    """A dead port must short-circuit.

    wait_for_healthy treats consecutive connection refusals as "the server
    died", which is exactly what a still-initializing server looks like —
    so it is only sound once the port is confirmed listening.
    """
    from sparkrun.core.launcher import wait_for_serve_ready

    calls = _patch_serve_ready(monkeypatch, port_ready=False)
    result = _make_launch_result(_LifecycleRecipe(), _LifecycleRuntime())

    readiness = wait_for_serve_ready(result)

    assert readiness.ready is False
    assert readiness.reason == "port"
    assert calls == ["port"], "wait_for_healthy must not run before the port is up"


def test_wait_for_serve_ready_health_failure(monkeypatch):
    """Port up but never HTTP 200 -> reason='health'."""
    from sparkrun.core.launcher import wait_for_serve_ready

    calls = _patch_serve_ready(monkeypatch, healthy=False)
    result = _make_launch_result(_LifecycleRecipe(), _LifecycleRuntime())

    readiness = wait_for_serve_ready(result)

    assert readiness.ready is False
    assert readiness.reason == "health"
    assert calls == ["port", "health"]


def test_wait_for_serve_ready_dry_run_probes_nothing(monkeypatch):
    """--dry-run reports ready without touching the network."""
    from sparkrun.core.launcher import wait_for_serve_ready

    calls = _patch_serve_ready(monkeypatch, port_ready=False, healthy=False)
    result = _make_launch_result(_LifecycleRecipe(), _LifecycleRuntime())

    readiness = wait_for_serve_ready(result, dry_run=True)

    assert readiness.ready is True
    assert calls == []


def test_wait_for_serve_ready_uses_serve_port_not_recipe_port(monkeypatch):
    """The probed port is the *resolved* one.

    ``auto_port=True`` (how ``proxy load`` launches) can move the server off
    the recipe's declared port; probing the recipe value would poll a port
    nothing is listening on.
    """
    import dataclasses

    from sparkrun.core.launcher import wait_for_serve_ready

    probed: list[int] = []

    _patch_serve_ready(monkeypatch)
    monkeypatch.setattr(
        "sparkrun.orchestration.health.wait_for_port",
        lambda host, port, **_k: (probed.append(port), True)[1],
    )

    # Recipe declares 8000; auto_port moved the server to 8001.
    result = _make_launch_result(_LifecycleRecipe(port=8000), _LifecycleRuntime())
    result = dataclasses.replace(result, serve_port=8001)

    readiness = wait_for_serve_ready(result)

    assert probed == [8001]
    assert readiness.port == 8001


def test_wait_for_serve_ready_multinode_uses_head_container(monkeypatch):
    """A non-solo launch watches the rank-0 container, not the solo name."""
    import dataclasses

    from sparkrun.core.launcher import wait_for_serve_ready

    _patch_serve_ready(monkeypatch)
    result = _make_launch_result(_LifecycleRecipe(), _LifecycleRuntime())
    result = dataclasses.replace(result, is_solo=False, host_list=["h1", "h2"])

    readiness = wait_for_serve_ready(result)

    assert readiness.container == "sparkrun_lifecyclecid_node_0"
    assert readiness.head_host == "h1"


# ---------------------------------------------------------------------------
# Phase 2 (builder) error handling
# ---------------------------------------------------------------------------
#
# The builder phase tolerates exactly one failure: an *unknown* builder, which
# warns and skips. A gated builder and a failing prepare() must both abort the
# launch — for an environment builder (a venv the serve command depends on),
# "skipping" means serving under the wrong interpreter.


def _builder_phase_harness(monkeypatch, tmp_path):
    """Mock just enough of launch_inference's preamble to reach phase 2."""
    from sparkrun.core import launcher

    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.derive_cluster_id", lambda *a, **kw: "sparkrun_test00000000")
    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))

    class _Cfg:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    class _Runtime:
        runtime_name = "stub"

        def resolve_container(self, recipe, overrides):
            return "stub:latest"

        def get_family(self):
            return "stub"

        def run(self, *a, **kw):
            return type("R", (), {"containers": {}, "head_host": "h1"})()

    class _Recipe:
        runtime = "stub"
        model = "stub-model"
        env = {}
        builder = "some-builder"
        builder_config = {}
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
            merged = dict(self.defaults)
            merged.update(overrides or {})

            class _CC:
                def get(self, k, default=None):
                    return merged.get(k, default)

            return _CC()

        def __getstate__(self):
            return {}

    def _launch():
        from sparkrun.core.launcher import launch_inference

        return launch_inference(
            recipe=_Recipe(),
            runtime=_Runtime(),
            host_list=["h1"],
            overrides={},
            config=_Cfg(),
            cluster=None,
            is_solo=True,
            dry_run=True,
            sync_tuning=False,
        )

    return _launch


def test_builder_phase_reraises_a_gated_builder(monkeypatch, tmp_path):
    """A recipe naming a real-but-disabled builder aborts rather than warning."""
    from sparkrun.builders.base import BuilderUnavailableError

    launch = _builder_phase_harness(monkeypatch, tmp_path)

    def _gated(name, v=None):
        raise BuilderUnavailableError("Builder %r is disabled by feature flag 'x'." % name)

    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", _gated)
    with pytest.raises(BuilderUnavailableError):
        launch()


def test_builder_phase_does_not_swallow_a_prepare_failure(monkeypatch, tmp_path):
    """A ValueError out of prepare() is a build failure, not "builder not
    found" — reporting it as the latter launched the workload anyway."""
    launch = _builder_phase_harness(monkeypatch, tmp_path)

    class _Boom:
        def prepare(self, *a, **kw):
            raise ValueError("bad builder_config")

    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", lambda name, v=None: _Boom())
    with pytest.raises(ValueError, match="bad builder_config"):
        launch()


class _PastPhase2(Exception):
    """Sentinel: execution reached the step after the builder phase."""


def test_builder_phase_still_skips_an_unknown_builder(monkeypatch, tmp_path, caplog):
    """Back-compat: an unknown builder warns and the launch continues.

    Asserted with a sentinel raised from the first call *after* phase 2 rather
    than by completing a launch — the point is only that phase 2 neither
    raised nor aborted, and stubbing the whole runtime protocol to prove it
    would test the stub.
    """
    import logging

    from sparkrun.core import launcher

    launch = _builder_phase_harness(monkeypatch, tmp_path)

    def _unknown(name, v=None):
        raise ValueError("Unknown builder: %r" % name)

    def _sentinel(*a, **kw):
        raise _PastPhase2()

    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", _unknown)
    monkeypatch.setattr(launcher, "apply_platform_runtime_flag_defaults", _sentinel)

    with caplog.at_level(logging.WARNING), pytest.raises(_PastPhase2):
        launch()
    assert "not found, skipping" in caplog.text
