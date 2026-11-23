"""One container-env composition for solo, cluster and ``api.materialize``.

The three paths used to merge the env tiers separately, and materialize left
out the runtime-cache env, so it could describe a container env the launch
never set. These tests pin the tier order and compare materialize against the
env each launch path actually hands its executor.
"""

from __future__ import annotations

from types import SimpleNamespace

import sparkrun.api as api
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.launch_env import compose_launch_env
from sparkrun.core.launcher import resolve_launch_runtime_cache, resolve_platform_env_by_host
from sparkrun.core.recipe import Recipe
from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.utils import merge_env

_TIERS = ("comm", "runtime_cache", "runtime_common", "runtime_mode", "platform", "recipe", "templates", "runtime_extra")


def test_each_tier_beats_every_tier_below_it():
    for index, tier in enumerate(_TIERS):
        env = compose_launch_env(**{lower: {"K": lower} for lower in _TIERS[: index + 1]})
        assert env["K"] == tier


def test_platform_env_ranks_below_declared_env_and_above_runtime_defaults():
    env = compose_launch_env(
        runtime_common={"A": "runtime", "B": "runtime"},
        platform={"A": "platform", "B": "platform"},
        recipe={"B": "recipe"},
    )
    assert env == {"A": "platform", "B": "recipe"}


def test_hf_cache_vars_beat_the_xdg_catch_all_and_the_recipe():
    env = compose_launch_env(
        runtime_cache={"XDG_CACHE_HOME": "/cache/runtime"},
        recipe={"XDG_CACHE_HOME": "/elsewhere", "HF_HOME": "/recipe"},
        runtime_extra={"HF_HOME": "/cache/huggingface"},
    )
    assert env == {"XDG_CACHE_HOME": "/elsewhere", "HF_HOME": "/cache/huggingface"}


# ---------------------------------------------------------------------------
# materialize vs. the env the launch hands the executor
# ---------------------------------------------------------------------------

_GB10 = HostHardware(accelerators=[AcceleratorSpec("nvidia", "gb10", memory_gb=121, capabilities=frozenset({"cuda"}))])
_COMM = {
    "h1": {"NCCL_IB_HCA": "rocep1s0f0", "NCCL_CUMEM_ENABLE": "1", "NODE_IP": "10.0.0.1"},
    "h2": {"NCCL_IB_HCA": "rocep1s0f1", "NCCL_CUMEM_ENABLE": "1", "NODE_IP": "10.0.0.2"},
}


def _plan(hosts: tuple[str, ...]):
    recipe = Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "Qwen/Qwen3.5-0.8B",
            "model_revision": "model-commit",
            "runtime": "vllm-distributed",
            "container": "org/capsule@sha256:%s" % ("a" * 64),
            "defaults": {"tensor_parallel": len(hosts), "port": 8000},
            # One key overlapping the platform tier, so the order is observable.
            "env": {"RECIPE_FLAG": "yes", "CUTE_DSL_ARCH": "recipe-arch"},
        }
    )
    cluster = ClusterDefinition(
        name="c",
        hosts=list(hosts),
        cache_dir="/cache/hf",
        sparkrun_cache_dir="/sr-cache",
        hosts_hardware={host: _GB10 for host in hosts},
    )
    plan = api.RunPlan(
        recipe=recipe,
        runtime=VllmDistributedRuntime(),
        cluster=cluster,
        candidate_hosts=hosts,
        host_list=hosts,
        is_solo=len(hosts) == 1,
        placement=RankAssignment(by_rank=tuple(RankSlot(h, 0) for h in hosts), hosts_used=hosts),
        workload_id="sparkrun_test_test",
    )
    options = api.RunOptions(recipe=recipe, hosts=hosts, init_port=29731)
    sctx = SimpleNamespace(config=SimpleNamespace(hf_cache_dir="/fallback", cache_dir="/sr-cache"))
    return options, plan, sctx


def _launch_inputs(plan, sctx):
    """What ``launch_inference`` resolves and hands ``runtime.run`` (dry run)."""
    runtime_cache = resolve_launch_runtime_cache(
        runtime=plan.runtime,
        recipe=plan.recipe,
        overrides={},
        config=sctx.config,
        cluster=plan.cluster,
        host_list=list(plan.host_list),
        ssh_kwargs={},
        image=plan.recipe.container,
        dry_run=True,
    )
    assert runtime_cache is not None
    platform_env_by_host = resolve_platform_env_by_host(plan.runtime, plan.cluster.hosts_hardware, plan.recipe.env)
    return runtime_cache, platform_env_by_host


def test_materialize_reports_the_cluster_launch_env():
    from sparkrun.runtimes._cluster_ops import ClusterContext

    options, plan, sctx = _plan(("h1", "h2"))
    runtime_cache, platform_env_by_host = _launch_inputs(plan, sctx)
    ctx = ClusterContext.build(
        plan.runtime,
        list(plan.host_list),
        plan.recipe.container,
        "cid",
        dict(plan.recipe.env),
        "/cache/hf",
        None,
        True,
        recipe=plan.recipe,
        runtime_cache=runtime_cache,
        platform_env_by_host=platform_env_by_host,
    )
    comm_env = ClusterCommEnv.from_per_host(_COMM)

    spec = api.materialize(options, plan=plan, comm_env=comm_env, sctx=sctx)

    for unit in spec.units:
        # The executor layers the (finalized) comm env beneath the container env.
        launched = merge_env(plan.runtime.finalize_host_comm_env(comm_env.get_env(unit.host)), ctx.env_for_host(unit.host))
        assert unit.environment == launched
        assert unit.environment["XDG_CACHE_HOME"] == "/cache/runtime"
        assert unit.environment["HF_HOME"] == "/cache/huggingface"
        assert unit.environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert unit.environment["CUTE_DSL_ARCH"] == "recipe-arch"
        # The runtime's cluster env beats the comm env, as in the executor.
        assert unit.environment["NCCL_CUMEM_ENABLE"] == "0"
        assert any(m.target == "/cache/runtime" and m.source == runtime_cache.leaf for m in unit.mounts)


def test_materialize_reports_the_solo_launch_env(monkeypatch):
    from sparkrun.orchestration.executors.docker import DockerExecutor

    options, plan, sctx = _plan(("h1",))
    runtime_cache, platform_env_by_host = _launch_inputs(plan, sctx)
    comm_env = ClusterCommEnv.from_per_host({"h1": _COMM["h1"]})

    captured = {}

    def _capture(self, *, env=None, nccl_env=None, **kwargs):
        captured.update(merge_env(nccl_env, env))
        return "true"

    monkeypatch.setattr(DockerExecutor, "generate_launch_script", _capture)
    plan.runtime._run_solo(
        "h1",
        plan.recipe.container,
        "serve",
        env=dict(plan.recipe.env),
        cache_dir="/cache/hf",
        dry_run=True,
        comm_env=comm_env,
        recipe=plan.recipe,
        runtime_cache=runtime_cache,
        platform_env=platform_env_by_host["h1"],
    )
    assert captured

    (unit,) = api.materialize(options, plan=plan, comm_env=comm_env, sctx=sctx).units
    # The solo path hands the executor the comm env unfinalized; vLLM's
    # finalization only adds keys, which the container env never sets.
    assert {k: v for k, v in unit.environment.items() if k in captured} == captured
    assert set(unit.environment) - set(captured) <= {"VLLM_HOST_IP"}


def test_solo_serve_exec_caps_only_detached_dispatch(monkeypatch):
    """Foreground exec is the serve process; a wall-clock cap kills a healthy server.

    ``export systemd`` units run ``run --foreground``, so a 60s cap here made
    every solo service exit rc 124 a minute into weight loading.
    """
    import sparkrun.orchestration.primitives as primitives
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.ssh import RemoteResult

    options, plan, sctx = _plan(("h1",))
    runtime_cache, platform_env_by_host = _launch_inputs(plan, sctx)
    monkeypatch.setattr(DockerExecutor, "generate_launch_script", lambda self, **kwargs: "true")

    real = primitives.run_script_on_host
    timeouts = []

    def _record(host, script, *, timeout=None, **kwargs):
        if "exec" in script and "base64 -d" in script:
            timeouts.append(timeout)
            return RemoteResult(host=host, returncode=0, stdout="", stderr="")
        return real(host, script, timeout=timeout, **kwargs)

    monkeypatch.setattr(primitives, "run_script_on_host", _record)

    for detached in (True, False):
        plan.runtime._run_solo(
            "h1",
            plan.recipe.container,
            "serve",
            env=dict(plan.recipe.env),
            cache_dir="/cache/hf",
            dry_run=True,
            detached=detached,
            recipe=plan.recipe,
            runtime_cache=runtime_cache,
            platform_env=platform_env_by_host["h1"],
        )

    assert timeouts == [60, None]
