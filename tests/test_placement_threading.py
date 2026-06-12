"""Tests for ``placement`` threading from API → launcher → runtime → ClusterContext (Task 8).

The architectural invariant verified here: when ``api.run`` (or any
caller of ``launch_inference``) provides a precomputed
:class:`RankAssignment`, it is used verbatim — the runtime layer
does NOT recompute placement.  When ``placement=None``, the cluster
context falls back to its internal computation (back-compat for the
in-tree callers that haven't been threaded yet).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.runtimes._cluster_ops import ClusterContext


def _stub_runtime():
    """Build a minimal RuntimePlugin-shaped object that ClusterContext.build can use.

    We don't instantiate a real RuntimePlugin (its initialization
    pulls SAF + many hooks); a MagicMock that responds to the few
    methods build() touches is sufficient.
    """
    rt = MagicMock()
    rt.get_extra_volumes.return_value = {}
    rt.get_cluster_env.return_value = {}
    rt.get_common_env.return_value = {}
    rt.get_extra_env.return_value = {}
    return rt


def _stub_cluster_with_4gpu_host(host: str):
    """Build a ClusterDefinition with a single 4-GPU host."""
    from sparkrun.core.cluster_manager import ClusterDefinition

    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200", count=4)])
    return ClusterDefinition(name="multi-gpu", hosts=[host], hosts_hardware={host: hw})


def _stub_recipe():
    """Build a Recipe-shaped object with a build_config_chain method."""
    rt = MagicMock()
    rt.build_config_chain.return_value = {"tensor_parallel": 4}
    rt.layout = None
    return rt


# --------------------------------------------------------------------------
# Direct ClusterContext.build threading
# --------------------------------------------------------------------------


def test_cluster_context_uses_provided_placement_verbatim():
    """When ``placement=`` is passed, build() must NOT recompute."""
    runtime = _stub_runtime()
    fake_placement = RankAssignment(
        by_rank=(RankSlot(host="h1", local_gpu=0), RankSlot(host="h1", local_gpu=1)),
        hosts_used=("h1",),
    )
    ctx = ClusterContext.build(
        runtime=runtime,
        hosts=["h1"],
        image="test:img",
        cluster_id="cid",
        env=None,
        cache_dir=None,
        config=None,
        dry_run=True,
        placement=fake_placement,
    )
    assert ctx.placement is fake_placement


def test_cluster_context_computes_when_placement_none_with_cluster_recipe():
    """Back-compat: ``placement=None`` + cluster + recipe → build computes one."""
    runtime = _stub_runtime()
    cluster = _stub_cluster_with_4gpu_host("h1")
    recipe = _stub_recipe()

    ctx = ClusterContext.build(
        runtime=runtime,
        hosts=["h1"],
        image="test:img",
        cluster_id="cid",
        env=None,
        cache_dir=None,
        config=None,
        dry_run=True,
        cluster=cluster,
        recipe=recipe,
    )
    assert ctx.placement is not None
    assert ctx.placement.hosts_used == ("h1",)
    assert ctx.placement.total_ranks == 4  # tensor_parallel=4 on a 4-GPU host


def test_cluster_context_skips_computation_when_no_cluster_or_recipe():
    """No placement, no cluster, no recipe → ctx.placement stays None."""
    runtime = _stub_runtime()
    ctx = ClusterContext.build(
        runtime=runtime,
        hosts=["h1"],
        image="test:img",
        cluster_id="cid",
        env=None,
        cache_dir=None,
        config=None,
        dry_run=True,
    )
    assert ctx.placement is None


def test_cluster_context_provided_placement_wins_over_computation():
    """When both placement= AND cluster+recipe are given, placement= wins
    (the architectural property: caller-supplied placement is authoritative)."""
    runtime = _stub_runtime()
    cluster = _stub_cluster_with_4gpu_host("h1")
    recipe = _stub_recipe()

    explicit = RankAssignment(
        by_rank=(RankSlot(host="h1", local_gpu=2),),
        hosts_used=("h1",),
    )
    ctx = ClusterContext.build(
        runtime=runtime,
        hosts=["h1"],
        image="test:img",
        cluster_id="cid",
        env=None,
        cache_dir=None,
        config=None,
        dry_run=True,
        cluster=cluster,
        recipe=recipe,
        placement=explicit,
    )
    # If recomputation happened, total_ranks would be 4 (tp=4) with
    # local_gpus 0..3.  The explicit single-rank assignment proves
    # the caller's input was used verbatim.
    assert ctx.placement is explicit
    assert ctx.placement.total_ranks == 1


# --------------------------------------------------------------------------
# launch_inference accepts placement= and forwards it via kwargs
# --------------------------------------------------------------------------


def test_launch_inference_accepts_placement_kwarg():
    """``launch_inference`` must accept ``placement=`` in its signature."""
    import inspect

    from sparkrun.core.launcher import launch_inference

    sig = inspect.signature(launch_inference)
    assert "placement" in sig.parameters, "launch_inference should accept placement= kwarg"
    # Default value must be None to preserve back-compat.
    assert sig.parameters["placement"].default is None


# --------------------------------------------------------------------------
# RuntimePlugin.run accepts placement via **kwargs and forwards it
# --------------------------------------------------------------------------


def test_runtime_run_pops_placement_from_kwargs_in_base():
    """The base RuntimePlugin native-cluster helper pops ``placement``
    from kwargs and passes it to ClusterContext.build.  Verified by
    reading the source (since instantiating a full runtime + cluster
    here would require an SAF-bootstrapped environment)."""
    from pathlib import Path

    base_src = Path("src/sparkrun/runtimes/base.py").read_text()
    assert 'placement = kwargs.pop("placement", None)' in base_src
    assert "placement=placement" in base_src


def test_runtime_run_pops_placement_in_overriding_runtimes():
    """vLLM Ray / llama.cpp / TRT-LLM overrides also forward placement."""
    from pathlib import Path

    for path in (
        "src/sparkrun/runtimes/vllm_ray.py",
        "src/sparkrun/runtimes/llama_cpp.py",
        "src/sparkrun/runtimes/trtllm.py",
    ):
        src = Path(path).read_text()
        assert 'placement = kwargs.pop("placement", None)' in src, "%s missing placement kwarg pop" % path
        assert "placement=placement" in src, "%s missing placement forwarding" % path


# --------------------------------------------------------------------------
# Single-compute invariant: api.run computes placement once
# --------------------------------------------------------------------------


def test_api_run_passes_placement_to_launch_inference():
    """When api.run computes a placement, it is forwarded to launch_inference
    via the ``placement=`` kwarg — verifying the single-compute path."""
    from unittest.mock import patch

    import sparkrun.api as api
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
                "cluster_id": "sparkrun_x",
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

    # Solo + single-host means we don't run the scheduler — placement is
    # None.  The architectural invariant we're verifying is that the
    # kwarg is present in the call signature (even when None).
    assert "placement" in captured, "launch_inference must receive a ``placement`` kwarg"


def test_api_run_passes_placement_when_multi_host_with_parallelism():
    """Multi-host + parallelism configured → api.run computes placement
    and threads it (non-None) into launch_inference."""
    from unittest.mock import patch

    import sparkrun.api as api
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe

    recipe = Recipe(
        {
            "sparkrun_version": "2",
            "runtime": "vllm",
            "model": "test/m",
            "defaults": {"tensor_parallel": 2},
        }
    )
    cluster = ClusterDefinition(name="dual", hosts=["h1", "h2"])
    opts = api.RunOptions(recipe=recipe, cluster=cluster, dry_run=True)

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
                "cluster_id": "sparkrun_y",
                "host_list": ["h1", "h2"],
                "is_solo": False,
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

    placement = captured.get("placement")
    assert placement is not None, "Multi-host + tp=2 should yield a non-None placement"
    assert placement.hosts_used == ("h1", "h2")
    assert placement.total_ranks == 2
