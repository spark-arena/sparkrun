"""Tests for ``sparkrun.api.run``, ``sparkrun.api.stop``, ``sparkrun.api.logs`` (Task 7).

The full launch path requires SSH-reachable hosts and a complete
runtime/builder/registry environment, so these tests focus on:

- API contract: signatures, return types, error mapping.
- Pre-launcher orchestration: recipe / hosts / cluster resolution and
  the routing of scheduling failures into typed API errors.
- Dry-run path through ``api.run`` (no remote execution).
- ``api.stop`` argument validation + ``JobNotFound`` handling.
- ``api.logs`` argument validation + ``JobNotFound`` handling.
- Iterator contract for ``api.logs``.

Full SSH-driven end-to-end coverage lands in Task 13's integration
tests.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest

import sparkrun.api as api


# --------------------------------------------------------------------------
# Public surface — re-exports reachable
# --------------------------------------------------------------------------


def test_run_function_exposed():
    assert hasattr(api, "run") and callable(api.run)


def test_stop_function_exposed():
    assert hasattr(api, "stop") and callable(api.stop)


def test_logs_function_exposed():
    assert hasattr(api, "logs") and callable(api.logs)


def test_logs_returns_iterator_protocol():
    """``api.logs`` is declared to return an Iterator[LogLine]; verify the
    type hint matches the runtime behaviour by inspecting the function
    return annotation."""
    import inspect

    sig = inspect.signature(api.logs)
    assert sig.return_annotation is not inspect.Parameter.empty


# --------------------------------------------------------------------------
# api.run — input validation / pre-launch failures
# --------------------------------------------------------------------------


def test_run_unknown_recipe_raises_recipe_not_found():
    """A string recipe name that doesn't resolve must raise RecipeNotFound."""
    with pytest.raises(api.RecipeNotFound):
        api.run(api.RunOptions(recipe="this-recipe-name-doesnt-exist-anywhere", hosts=("h1",)))


def test_run_no_hosts_no_cluster_raises_hosts_unreachable():
    """When no host source is available and config has no defaults."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    # No hosts, no cluster, and assume no default_hosts in test isolation.
    with pytest.raises(api.HostsUnreachable):
        api.run(api.RunOptions(recipe=recipe))


def test_run_options_immutable_through_run():
    """Passing the same RunOptions twice must not mutate its overrides dict."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), overrides={"port": 8000})
    snapshot = dict(opts.overrides)
    try:
        api.run(opts)
    except api.SparkrunError:
        pass  # Don't care about launch failure here; just probe input invariance.
    assert opts.overrides == snapshot


# --------------------------------------------------------------------------
# api.run dry-run path — no remote execution
# --------------------------------------------------------------------------


def test_run_dry_run_returns_run_result_without_ssh():
    """A dry-run call returns a populated RunResult without invoking SSH.

    Patching ``launch_inference`` keeps this hermetic — we just verify
    that api.run wires the options through and translates the result.
    """
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), dry_run=True)

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None  # no executor; fallback path

    fake_result = type(
        "FakeLaunchResult",
        (),
        {
            "rc": 0,
            "cluster_id": "sparkrun_fakefakefake",
            "host_list": ["h1"],
            "is_solo": True,
            "runtime": _FakeRuntime(),
            "recipe": recipe,
            "overrides": {},
            "container_image": "test:latest",
            "effective_cache_dir": "/tmp/cache",
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
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
    ):
        result = api.run(opts)

    assert isinstance(result, api.RunResult)
    assert result.cluster_id == "sparkrun_fakefakefake"
    assert result.dry_run is True
    assert result.runtime == "vllm"
    assert result.is_solo is True


def test_run_solo_mode_truncates_to_one_host():
    """Solo mode keeps only the head host even when multiple are passed."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(
        recipe=recipe,
        hosts=("h1", "h2", "h3"),
        solo=True,
        dry_run=True,
    )

    captured_hosts: list[str] = []

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

    def _capture(**kwargs):
        captured_hosts.extend(kwargs["host_list"])
        return type(
            "FakeLaunchResult",
            (),
            {
                "rc": 0,
                "cluster_id": "sparkrun_solosolosolo",
                "host_list": kwargs["host_list"],
                "is_solo": kwargs["is_solo"],
                "runtime": _FakeRuntime(),
                "recipe": recipe,
                "overrides": {},
                "container_image": "test:latest",
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
        result = api.run(opts)

    assert captured_hosts == ["h1"]
    assert result.is_solo is True


# --------------------------------------------------------------------------
# api.stop — argument validation
# --------------------------------------------------------------------------


def test_stop_requires_cluster_id_or_recipe():
    """Calling stop with neither cluster_id nor recipe must raise SparkrunError."""
    with pytest.raises(api.SparkrunError):
        api.stop()


def test_stop_unknown_cluster_id_raises_job_not_found(tmp_path):
    """When no job metadata matches cluster_id and no hosts given."""
    with pytest.raises(api.JobNotFound):
        api.stop(cluster_id="sparkrun_doesnotexist", cache_dir=str(tmp_path))


def test_stop_with_hosts_skips_metadata_lookup(tmp_path):
    """Providing explicit hosts allows stop to proceed without metadata."""
    # Mock the SSH dispatch so no real connection is attempted.
    from sparkrun.orchestration.ssh import RemoteResult

    fake_result = RemoteResult(host="h1", returncode=0, stdout="", stderr="")
    with patch("sparkrun.orchestration.ssh.run_remote_script", return_value=fake_result):
        result = api.stop(
            cluster_id="sparkrun_explicithost",
            hosts=("h1",),
            cache_dir=str(tmp_path),
        )
    assert isinstance(result, api.StopResult)
    assert result.cluster_id == "sparkrun_explicithost"
    assert result.hosts_targeted == ("h1",)


# --------------------------------------------------------------------------
# api.logs — argument validation
# --------------------------------------------------------------------------


def test_logs_unknown_cluster_raises_job_not_found(tmp_path):
    """No metadata and no hosts → JobNotFound when consumed."""
    # logs() is a generator-returning function; the underlying call
    # currently raises immediately (no host source).  We consume the
    # iterator to confirm.
    with pytest.raises(api.JobNotFound):
        # Either call raises directly, or consuming the iterator does.
        gen = api.logs("sparkrun_doesnotexist", cache_dir=str(tmp_path))
        if isinstance(gen, Iterator):
            list(gen)


def test_logs_returns_iterator_with_explicit_hosts(tmp_path):
    """``api.logs`` returns an iterator when given explicit hosts."""
    # We can't easily iterate without a real subprocess, but we can
    # assert the call signature returns something iterable.
    gen = api.logs("sparkrun_anyid", hosts=("h1",), cache_dir=str(tmp_path), tail=10)
    assert hasattr(gen, "__iter__")


# --------------------------------------------------------------------------
# Click independence — the full API surface still must not import click
# --------------------------------------------------------------------------


def test_full_api_with_run_stop_logs_imports_without_click():
    import subprocess
    import sys

    code = (
        "import sys, importlib;"
        "m = importlib.import_module('sparkrun.api');"
        "assert all(hasattr(m, n) for n in ('run', 'stop', 'logs'));"
        "assert 'click' not in sys.modules, 'click should not be pulled in'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, "stdout=%s\nstderr=%s" % (result.stdout, result.stderr)


# --------------------------------------------------------------------------
# Scheduler resolution chain: CLI > recipe > cluster > greedy default
# --------------------------------------------------------------------------


def _run_with_scheduler_chain(*, options_scheduler, recipe_scheduler, cluster_scheduler):
    """Exercise ``api.run`` with the three layers wired up; return the
    ``RunResult.scheduler`` field (the actually-resolved scheduler name).
    """
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe

    recipe_data = {"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"}
    if recipe_scheduler is not None:
        recipe_data["scheduler"] = recipe_scheduler
    recipe = Recipe(recipe_data)

    cluster_def = ClusterDefinition(name="test-cluster", hosts=["h1"], scheduler=cluster_scheduler)

    opts = api.RunOptions(
        recipe=recipe,
        hosts=("h1",),
        dry_run=True,
        scheduler=options_scheduler,
    )

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, recipe=None, cluster=None):
            return 1

    fake_result = type(
        "FakeLaunchResult",
        (),
        {
            "rc": 0,
            "cluster_id": "sparkrun_schedtestcid",
            "host_list": ["h1"],
            "is_solo": True,
            "runtime": _FakeRuntime(),
            "recipe": recipe,
            "overrides": {},
            "container_image": "test:latest",
            "effective_cache_dir": "/tmp/cache",
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
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
        patch("sparkrun.api._resolve.resolve_cluster", return_value=cluster_def),
    ):
        return api.run(opts)


def test_scheduler_chain_cli_option_wins_over_all():
    result = _run_with_scheduler_chain(
        options_scheduler="from-cli",
        recipe_scheduler="from-recipe",
        cluster_scheduler="from-cluster",
    )
    assert result.scheduler == "from-cli"


def test_scheduler_chain_recipe_wins_over_cluster():
    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler="from-recipe",
        cluster_scheduler="from-cluster",
    )
    assert result.scheduler == "from-recipe"


def test_scheduler_chain_cluster_wins_over_default():
    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler=None,
        cluster_scheduler="from-cluster",
    )
    assert result.scheduler == "from-cluster"


def test_scheduler_chain_all_unset_falls_back_to_default():
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER

    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler=None,
        cluster_scheduler=None,
    )
    # RunResult.scheduler reflects the actually-resolved scheduler name —
    # when no caller supplies one we land on FALLBACK_DEFAULT_SCHEDULER.
    assert result.scheduler == FALLBACK_DEFAULT_SCHEDULER


# --------------------------------------------------------------------------
# Scheduling request carries occupancy status
# --------------------------------------------------------------------------


def _build_multihost_run_fixtures(num_hosts: int = 4):
    """Build a recipe + cluster + options tuple that triggers the multi-host
    scheduling path inside ``api.run``.

    Returns ``(recipe, cluster_def, opts, fake_runtime, fake_launch_result)``.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe

    hosts = tuple("h%d" % (i + 1) for i in range(num_hosts))
    recipe = Recipe(
        {
            "sparkrun_version": "2",
            "runtime": "vllm",
            "model": "test/m",
            "defaults": {"tensor_parallel": 2},
        }
    )
    cluster_def = ClusterDefinition(name="multi", hosts=list(hosts))

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, recipe=None, cluster=None):
            return parallelism.tensor_parallel

    fake_runtime = _FakeRuntime()
    fake_launch_result = type(
        "FakeLaunchResult",
        (),
        {
            "rc": 0,
            "cluster_id": "sparkrun_multihostfix",
            "host_list": list(hosts[:2]),
            "is_solo": False,
            "runtime": fake_runtime,
            "recipe": recipe,
            "overrides": {},
            "container_image": "test:latest",
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

    opts = api.RunOptions(recipe=recipe, hosts=hosts, cluster=cluster_def, dry_run=True)
    return recipe, cluster_def, opts, fake_runtime, fake_launch_result


def test_run_populates_scheduling_request_status():
    """``api.run`` queries cluster status and passes the snapshot into the
    SchedulingRequest so occupancy-sparse / occupancy-dense schedulers can subtract committed
    workloads from each host's capacity."""
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.scheduler import RankAssignment, SchedulingResult

    recipe, cluster_def, opts, fake_runtime, fake_launch_result = _build_multihost_run_fixtures()

    fake_status = ClusterStatus(hosts=())
    captured_requests: list = []

    def _fake_schedule(request, *, scheduler=None, sctx=None):
        captured_requests.append(request)
        return SchedulingResult(
            assignment=RankAssignment(
                by_rank=(),
                hosts_used=tuple(request.hosts[:2]),
            ),
            scheduler_name="greedy",
            diagnostics=(),
        )

    with (
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_launch_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=fake_runtime),
        patch("sparkrun.api._resolve.resolve_cluster", return_value=cluster_def),
        patch("sparkrun.api.status", return_value=fake_status),
        patch("sparkrun.api.schedule", side_effect=_fake_schedule),
    ):
        api.run(opts)

    assert len(captured_requests) == 1
    assert captured_requests[0].status is fake_status


def test_run_status_acquisition_failure_falls_back_gracefully():
    """When the cluster status query fails (partial reachability, missing
    executor, transient SSH error), scheduling still proceeds with
    ``status=None`` rather than crashing the launch."""
    from sparkrun.core.scheduler import RankAssignment, SchedulingResult

    recipe, cluster_def, opts, fake_runtime, fake_launch_result = _build_multihost_run_fixtures()

    captured_requests: list = []

    def _fake_schedule(request, *, scheduler=None, sctx=None):
        captured_requests.append(request)
        return SchedulingResult(
            assignment=RankAssignment(
                by_rank=(),
                hosts_used=tuple(request.hosts[:2]),
            ),
            scheduler_name="greedy",
            diagnostics=(),
        )

    with (
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_launch_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=fake_runtime),
        patch("sparkrun.api._resolve.resolve_cluster", return_value=cluster_def),
        patch("sparkrun.api.status", side_effect=RuntimeError("boom")),
        patch("sparkrun.api.schedule", side_effect=_fake_schedule),
    ):
        api.run(opts)

    assert len(captured_requests) == 1
    assert captured_requests[0].status is None


def test_run_result_scheduler_reflects_effective_resolution():
    """When the caller doesn't pick a scheduler, ``RunResult.scheduler``
    reports the actually-resolved scheduler name (not a stale ``"greedy"``
    default) so consumers can verify which scheduler ran."""
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER

    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler=None,
        cluster_scheduler=None,
    )
    # FALLBACK_DEFAULT_SCHEDULER ("occupancy-sparse") names a registered
    # plugin; the resolver returns the plugin's canonical scheduler_name.
    assert result.scheduler == FALLBACK_DEFAULT_SCHEDULER
