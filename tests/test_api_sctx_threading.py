"""Tests for SparkrunContext threading through the sparkrun.api surface.

Each ``api.*`` function accepts an optional ``sctx`` and propagates it
to internal helpers and to ``launch_inference``.  These tests verify:

- ``sparkrun.api.default_sctx`` produces a valid context.
- ``sparkrun.api.SparkrunContext`` is re-exported.
- Each api entry accepts ``sctx=`` (signature check).
- Resolve helpers honor sctx-provided managers (registry / cluster).
- ``api.run`` threads sctx into ``launch_inference``.
- ``api.list_jobs`` honors ``sctx.config.cache_dir`` as fallback.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import sparkrun.api as api


# --------------------------------------------------------------------------
# Public re-exports + default factory
# --------------------------------------------------------------------------


def test_sparkrun_context_re_exported():
    assert hasattr(api, "SparkrunContext")
    assert hasattr(api, "default_sctx")


def test_default_sctx_builds_complete_context():
    sctx = api.default_sctx()
    assert isinstance(sctx, api.SparkrunContext)
    assert sctx.variables is not None
    assert sctx.config is not None
    # cached_property: accessing twice returns the same instance.
    assert sctx.cluster_manager is sctx.cluster_manager
    assert sctx.registry_manager is sctx.registry_manager


# --------------------------------------------------------------------------
# Every api function accepts sctx=
# --------------------------------------------------------------------------


def test_api_run_accepts_sctx_kwarg():
    sig = inspect.signature(api.run)
    assert "sctx" in sig.parameters
    assert sig.parameters["sctx"].default is None


def test_api_stop_accepts_sctx_kwarg():
    sig = inspect.signature(api.stop)
    assert "sctx" in sig.parameters
    assert sig.parameters["sctx"].default is None


def test_api_logs_accepts_sctx_kwarg():
    sig = inspect.signature(api.logs)
    assert "sctx" in sig.parameters
    assert sig.parameters["sctx"].default is None


def test_api_status_accepts_sctx_kwarg():
    sig = inspect.signature(api.status)
    assert "sctx" in sig.parameters
    assert sig.parameters["sctx"].default is None


def test_api_schedule_accepts_sctx_kwarg():
    sig = inspect.signature(api.schedule)
    assert "sctx" in sig.parameters
    assert sig.parameters["sctx"].default is None


def test_api_list_jobs_accepts_sctx_kwarg():
    sig = inspect.signature(api.list_jobs)
    assert "sctx" in sig.parameters
    assert sig.parameters["sctx"].default is None


# --------------------------------------------------------------------------
# Resolve helpers honor sctx-provided managers
# --------------------------------------------------------------------------


def test_resolve_cluster_uses_sctx_cluster_manager(tmp_path):
    """``resolve_cluster`` looks up cluster names via ``sctx.cluster_manager``."""
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.core.cluster_manager import ClusterManager

    # Build a custom ClusterManager and an sctx that exposes it.
    mgr = ClusterManager(tmp_path)
    mgr.create(name="custom-cluster", hosts=["h1", "h2"])

    # Inject the manager by constructing a stub sctx (lazy attribute access).
    class _StubSctx:
        @property
        def cluster_manager(self):
            return mgr

    resolved = resolve_cluster("custom-cluster", sctx=_StubSctx())  # type: ignore[arg-type]
    assert resolved.name == "custom-cluster"
    assert resolved.hosts == ["h1", "h2"]


def test_resolve_cluster_uses_sctx_config_default_hosts():
    """``resolve_cluster`` falls back to ``sctx.config.default_hosts`` when no hosts/cluster."""
    from sparkrun.api._resolve import resolve_cluster

    class _StubConfig:
        default_hosts = ["fallback-1", "fallback-2"]

    class _StubSctx:
        config = _StubConfig()

    resolved = resolve_cluster(None, None, sctx=_StubSctx())  # type: ignore[arg-type]
    assert resolved.name == ""  # anonymous
    assert resolved.hosts == ["fallback-1", "fallback-2"]


def test_resolve_cluster_explicit_overrides_sctx_default():
    """Explicit ``hosts_input`` wins over ``sctx.config.default_hosts``."""
    from sparkrun.api._resolve import resolve_cluster

    class _StubConfig:
        default_hosts = ["fallback-1"]

    class _StubSctx:
        config = _StubConfig()

    resolved = resolve_cluster(None, ["override"], sctx=_StubSctx())  # type: ignore[arg-type]
    assert resolved.hosts == ["override"]


# --------------------------------------------------------------------------
# api.run threads sctx into launch_inference
# --------------------------------------------------------------------------


def test_api_run_forwards_sctx_to_launch_inference():
    """When a caller passes sctx=, api.run must forward it (and sctx.variables) to launch_inference."""
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

    custom_sctx = SparkrunContext(
        variables=api.default_sctx().variables,
        config=api.default_sctx().config,
        verbose=True,
    )

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), solo=True, dry_run=True)

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, *, recipe, cluster):
            return parallelism.total_gpus

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return type(
            "FakeLaunchResult",
            (),
            {
                "rc": 0,
                "cluster_id": "sparkrun_zz",
                "host_list": ["h1"],
                "is_solo": True,
                "runtime": _FakeRuntime(),
                "recipe": recipe,
                "overrides": {},
                "container_image": "img:t",
                "effective_cache_dir": "/tmp",
                "serve_port": 8000,
                "config": None,
                "recipe_ref": None,
                "comm_env": None,
                "ib_ip_map": {},
                "serve_command": "",
                "runtime_info": {},
                "builder": None,
                "backends": {},
            },
        )()

    with (
        patch("sparkrun.core.launcher.launch_inference", side_effect=_capture),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
    ):
        api.run(opts, sctx=custom_sctx)

    # api.run must pass sctx-supplied config and SAF variables down.
    assert captured.get("sctx") is custom_sctx
    assert captured.get("v") is custom_sctx.variables
    assert captured.get("config") is custom_sctx.config


def test_api_run_builds_default_sctx_when_omitted():
    """When sctx is omitted, api.run still wires v/sctx (built from default_sctx)."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), solo=True, dry_run=True)

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, *, recipe, cluster):
            return parallelism.total_gpus

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return type(
            "FakeLaunchResult",
            (),
            {
                "rc": 0,
                "cluster_id": "sparkrun_yy",
                "host_list": ["h1"],
                "is_solo": True,
                "runtime": _FakeRuntime(),
                "recipe": recipe,
                "overrides": {},
                "container_image": "img:t",
                "effective_cache_dir": "/tmp",
                "serve_port": 8000,
                "config": None,
                "recipe_ref": None,
                "comm_env": None,
                "ib_ip_map": {},
                "serve_command": "",
                "runtime_info": {},
                "builder": None,
                "backends": {},
            },
        )()

    with (
        patch("sparkrun.core.launcher.launch_inference", side_effect=_capture),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
    ):
        api.run(opts)

    # api.run must always pass non-None sctx + v even when caller omits sctx.
    assert captured.get("sctx") is not None
    assert captured.get("v") is not None
    assert captured.get("config") is not None


# --------------------------------------------------------------------------
# api.list_jobs honors sctx.config.cache_dir
# --------------------------------------------------------------------------


def test_list_jobs_uses_sctx_cache_dir_when_no_explicit_arg(tmp_path: Path):
    """When cache_dir is not passed, sctx.config.cache_dir is the fallback."""
    import yaml

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "aaaaaaaaaaaa.yaml").write_text(
        yaml.safe_dump(
            {
                "cluster_id": "sparkrun_aaaaaaaaaaaa",
                "recipe": "test",
                "runtime": "vllm",
                "hosts": ["h1"],
                "started_at": 100.0,
            }
        )
    )

    class _StubConfig:
        cache_dir = tmp_path

    class _StubSctx:
        config = _StubConfig()

    result = api.list_jobs(sctx=_StubSctx())  # type: ignore[arg-type]
    cluster_ids = {j.cluster_id for j in result}
    assert "sparkrun_aaaaaaaaaaaa" in cluster_ids


def test_list_jobs_explicit_cache_dir_wins_over_sctx(tmp_path: Path):
    """Explicit cache_dir takes precedence over sctx.config.cache_dir."""
    import yaml

    # sctx points to one location...
    sctx_cache = tmp_path / "sctx-cache"
    (sctx_cache / "jobs").mkdir(parents=True)

    # ...but the caller passes a different one with actual data.
    explicit_cache = tmp_path / "explicit-cache"
    explicit_jobs = explicit_cache / "jobs"
    explicit_jobs.mkdir(parents=True)
    (explicit_jobs / "bbbbbbbbbbbb.yaml").write_text(
        yaml.safe_dump(
            {
                "cluster_id": "sparkrun_bbbbbbbbbbbb",
                "recipe": "test",
                "started_at": 200.0,
            }
        )
    )

    class _StubConfig:
        cache_dir = sctx_cache

    class _StubSctx:
        config = _StubConfig()

    result = api.list_jobs(cache_dir=str(explicit_cache), sctx=_StubSctx())  # type: ignore[arg-type]
    cluster_ids = {j.cluster_id for j in result}
    assert "sparkrun_bbbbbbbbbbbb" in cluster_ids


# --------------------------------------------------------------------------
# Click independence: sctx layer doesn't drag Click
# --------------------------------------------------------------------------


def test_sctx_layer_does_not_import_click():
    """SparkrunContext lives in core/, not cli/, and uses no Click types."""
    import subprocess
    import sys

    code = (
        "import sys, importlib;"
        "m = importlib.import_module('sparkrun.api');"
        "sctx = m.default_sctx();"
        "assert sctx.variables is not None;"
        "assert sctx.config is not None;"
        "assert 'click' not in sys.modules, 'click must not be pulled in via sctx default'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, "stdout=%s\nstderr=%s" % (result.stdout, result.stderr)
