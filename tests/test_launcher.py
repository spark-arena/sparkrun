"""Tests for ``sparkrun.core.launcher`` backend resolution (A1)."""

from __future__ import annotations

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
        "sparkrun.orchestration.job_metadata.generate_cluster_id",
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
