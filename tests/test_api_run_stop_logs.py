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
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.ssh import RemoteResult


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


def test_discover_cluster_id_sweeps_whole_host_scope(monkeypatch):
    """``stop <recipe>`` intent-discovery sweeps the whole host scope, so a job
    launched under the native ``local`` executor (invisible to ``docker ps``) is
    still found — not just docker workloads."""
    from sparkrun.api._resolve import discover_cluster_id_by_intent
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.executors.local import LocalExecutor

    intent = "aabbccddeeff0011"
    cid = "sparkrun_%s_0123456789ab" % intent

    # docker sees nothing; the workload only exists as a local-executor process.
    monkeypatch.setattr(DockerExecutor, "query_status", lambda self, hosts, **kw: ClusterStatus(hosts=(), executor="docker"))
    monkeypatch.setattr(
        LocalExecutor,
        "query_status",
        lambda self, hosts, **kw: ClusterStatus(
            hosts=tuple(HostOccupancy(host=h, workloads=(RunningWorkload(cluster_id=cid),)) for h in hosts),
            executor="local",
        ),
    )

    found = discover_cluster_id_by_intent(
        intent,
        ["h1"],
        cluster_def=ClusterDefinition(name="c", hosts=["h1"]),
        cache_dir=None,
        sctx=None,
    )
    assert found == cid  # discovered via the local sweep, not docker


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


def test_logs_without_metadata_refuses_to_guess(tmp_path):
    """Hosts alone aren't enough: without job metadata there is no runtime, and
    without a runtime sparkrun cannot know the container name or whether the
    output lives on container stdout or in the in-container serve log.

    This used to return an iterator that silently read ``{cid}_solo`` via
    ``docker logs`` — the wrong container for Ray runtimes and an empty stream
    for every sleep-infinity + exec runtime.  Failing with an actionable
    message beats streaming nothing.
    """
    with pytest.raises(api.JobNotFound) as exc:
        api.logs("sparkrun_anyid", hosts=("h1",), cache_dir=str(tmp_path), tail=10)
    assert "recipe" in str(exc.value)


def test_logs_returns_iterator_when_metadata_names_the_runtime(tmp_path):
    """With metadata recording the runtime, ``api.logs`` returns a lazy iterator.

    Resolution happens eagerly (so bad targets raise at call time); only the
    reading is deferred, so no subprocess is spawned by the call itself.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    # A *resolved* runtime name, which is what save_job_metadata records for a
    # loaded recipe ("vllm" is resolved to vllm-ray / vllm-distributed at load).
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    gen = api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)
    assert isinstance(gen, Iterator)


def test_logs_rejects_unknown_scope(tmp_path):
    with pytest.raises(api.SparkrunError):
        api.logs("sparkrun_anyid", hosts=("h1",), scope="everything", cache_dir=str(tmp_path))


def test_logs_requires_cluster_id_or_recipe():
    with pytest.raises(api.SparkrunError):
        api.logs()


def _make_status(host_workloads, errors=None):
    """Build a ClusterStatus for precheck tests.

    ``host_workloads`` maps host → list of (cluster_id, container_names).
    A host with no entry is unreachable (goes into ``errors``).
    """
    from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload

    hosts = []
    for host, workloads in host_workloads.items():
        ws = []
        for cid, container_names in workloads:
            containers = tuple(ContainerDetail(name=n, role="solo", status="Up", image="img") for n in container_names)
            ws.append(RunningWorkload(cluster_id=cid, containers=containers))
        hosts.append(HostOccupancy(host=host, workloads=tuple(ws)))
    return ClusterStatus(hosts=tuple(hosts), executor="docker", errors=dict(errors or {}))


def test_logs_stopped_container_preserves_metadata(tmp_path):
    """Container stopped but still exists → JobNotFound + metadata preserved.

    ``query_status`` reports only running workloads, so the precheck asks
    ``executor.describe_terminated`` what became of this one.  The container is
    still on the host, so metadata is kept and the executor's own investigation
    hints are rendered.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → host reachable, no workloads (container not running)
    # describe_terminated probe → docker ps -a lists it, still stopped on the host
    empty_snapshot = _make_status({"h1": []})
    probe = [RemoteResult(host="h1", returncode=0, stdout="%s_solo\tExited (1) 2 minutes ago\n" % cluster_id, stderr="")]
    with (
        patch.object(DockerExecutor, "query_status", return_value=empty_snapshot),
        patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=probe),
    ):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    assert "not currently running" in msg
    assert cluster_id in msg
    assert "Exited (1) 2 minutes ago" in msg
    # Hints come from the executor, not from this module.
    assert "docker logs" in msg
    assert "docker inspect" in msg
    assert "sparkrun stop" in msg
    # Metadata is preserved for investigation.
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None


def test_logs_removed_container_cleans_metadata(tmp_path):
    """Container fully gone (auto-removed) → JobNotFound + metadata removed.

    When the executor confirms nothing remains, the metadata is stale and is
    removed so ``logs <TAB>`` stops suggesting the dead workload.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → host reachable, no workloads (container not running)
    # describe_terminated probe → docker ps -a lists nothing (container gone)
    empty_snapshot = _make_status({"h1": []})
    gone = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]
    with (
        patch.object(DockerExecutor, "query_status", return_value=empty_snapshot),
        patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=gone),
    ):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    assert "nothing remains to read" in msg
    assert cluster_id in msg
    assert "stale job metadata has been removed" in msg
    # auto_remove defaults on, so absence is `--rm` doing its job rather than a
    # workload that never ran — say which, and how to keep it next time.
    assert "auto-removed on exit" in msg
    assert "auto_remove=false" in msg
    # Metadata was cleaned up.
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is None


def test_logs_precheck_worker_alive_when_head_dead(tmp_path):
    """Head dead but workers alive → helpful error pointing to --all-sources.

    In a multi-node job the head may crash while workers are still alive.
    The precheck detects this via the status snapshot (head host has no
    matching workload, worker host does) and raises a JobNotFound with a
    pointer to ``--all-sources`` rather than letting the reader fail on
    the dead head container with a raw docker error.  Metadata is preserved.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    # h1: reachable but no workloads (head dead)
    # h2: has the worker container running
    snapshot = _make_status(
        {
            "h1": [],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1", "h2"), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    assert "partially running" in msg
    assert "h2" in msg
    assert "--all-sources" in msg
    # Metadata is preserved.
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None


def test_logs_precheck_all_sources_proceeds_when_head_dead(tmp_path):
    """With --all-sources, head dead + worker alive → proceed (don't raise).

    When the user passes ``--all-sources`` (scope=SCOPE_ALL), the reader
    can read from the surviving workers, so the precheck should let it
    through rather than raising the "partially running" error.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    # h1: reachable but no workloads (head dead)
    # h2: has the worker container running
    snapshot = _make_status(
        {
            "h1": [],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        gen = api.logs(cluster_id, hosts=("h1", "h2"), cache_dir=str(tmp_path), tail=10, scope="all")
        # With --all-sources, worker is alive → proceed (no JobNotFound).
        assert isinstance(gen, Iterator)


def test_logs_skips_precheck_on_ssh_failure(tmp_path):
    """Host unreachable (in ClusterStatus.errors) → precheck skipped.

    When ``query_status`` can't reach a host, it records it in
    ``ClusterStatus.errors`` and omits it from ``hosts``.  The precheck
    treats this as inconclusive (``for_host()`` returns ``None``) and
    skips, letting the log reader surface its own error.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → h1 unreachable (in errors, absent from hosts)
    snapshot = _make_status({}, errors={"h1": "unreachable"})
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        gen = api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)
        # Precheck skipped → returns a lazy iterator (no JobNotFound raised).
        assert isinstance(gen, Iterator)


def test_logs_precheck_skips_on_query_exception(tmp_path):
    """If executor.query_status raises, precheck is skipped (best-effort).

    The precheck must never crash — a broken query_status should let the
    log reader surface its own error rather than preempting with a
    misleading 'not running' message.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    with patch.object(DockerExecutor, "query_status", side_effect=RuntimeError("internal error")):
        gen = api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)
        assert isinstance(gen, Iterator)


def test_logs_precheck_exists_cmd_failure_preserves_metadata(tmp_path):
    """An inconclusive ``describe_terminated`` probe → preserve metadata.

    After the status snapshot shows nothing running, the precheck asks the
    executor what became of the workload.  When that probe cannot answer (SSH
    failure rc=255, no container engine rc=127, timeout), the executor reports
    nothing for that source — and "cannot tell" must never be read as
    "confirmed gone", because that is the verdict that deletes metadata.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → host reachable, no workloads (container not running)
    # describe_terminated probe → rc=255 (SSH failure), so no verdict at all
    empty_snapshot = _make_status({"h1": []})
    ssh_fail_followup = [RemoteResult(host="h1", returncode=255, stdout="", stderr="ssh: connection refused")]
    with (
        patch.object(DockerExecutor, "query_status", return_value=empty_snapshot),
        patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=ssh_fail_followup),
    ):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    # Inconclusive must not read as "gone" — that verdict deletes metadata.
    assert "not currently running" in msg
    assert "nothing remains to read" not in msg
    # Metadata preserved.
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None


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
    # FALLBACK_DEFAULT_SCHEDULER ("greedy") names a registered plugin; the
    # resolver returns the plugin's canonical scheduler_name when nothing in
    # the CLI/recipe/cluster chain selects one.
    assert result.scheduler == FALLBACK_DEFAULT_SCHEDULER


# --------------------------------------------------------------------------
# cluster_id placement token: deterministic for greedy, random for occupancy
# --------------------------------------------------------------------------


def _capture_launch_cluster_id(opts):
    """Run ``api.run(opts)`` with a hermetic launcher and return the
    ``cluster_id_override`` it composed and handed to ``launch_inference``.

    The fake launch result echoes the override back as its ``cluster_id`` —
    mirroring the real launcher, which honours ``cluster_id_override`` — so
    callers can assert on either the captured override or ``RunResult.cluster_id``.
    """
    from sparkrun.core.recipe import Recipe  # noqa: F401  (imported for parity / side effects)

    captured: dict = {}

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, recipe=None, cluster=None):
            return 1

    def _capture(**kwargs):
        captured["cluster_id"] = kwargs["cluster_id_override"]
        return type(
            "FakeLaunchResult",
            (),
            {
                "rc": 0,
                "cluster_id": kwargs["cluster_id_override"],
                "host_list": kwargs["host_list"],
                "is_solo": kwargs["is_solo"],
                "runtime": _FakeRuntime(),
                "recipe": kwargs["recipe"],
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
        api.run(opts)

    return captured["cluster_id"]


def test_run_greedy_cluster_id_is_deterministic():
    """The default (greedy) scheduler yields a deterministic cluster_id:
    relaunching the same recipe on the same hosts reuses the same id so the
    prior deployment is replaced — sparkrun 0.2.x semantics."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), dry_run=True)

    first = _capture_launch_cluster_id(opts)
    second = _capture_launch_cluster_id(opts)

    assert first == second


def test_run_greedy_cluster_id_matches_lookup_derivation():
    """The greedy launch id equals what the lookup paths (stop / status /
    --ensure) compute via ``derive_cluster_id`` — so they can find and
    replace the running job."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import derive_cluster_id

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    hosts = ("h2", "h1")  # unsorted on purpose: derivation sorts before hashing
    opts = api.RunOptions(recipe=recipe, hosts=hosts, dry_run=True)

    launched = _capture_launch_cluster_id(opts)

    assert launched == derive_cluster_id(recipe, list(hosts))


def test_run_occupancy_cluster_id_is_random():
    """A status-aware scheduler (occupancy-sparse) keeps a fresh random
    placement token per launch so the same intent placed on different host
    sets never collides.  The intent half stays stable; only the placement
    token differs."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import parse_cluster_id

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), dry_run=True, scheduler="occupancy-sparse")

    first = _capture_launch_cluster_id(opts)
    second = _capture_launch_cluster_id(opts)

    assert first != second
    first_intent, first_token = parse_cluster_id(first)
    second_intent, second_token = parse_cluster_id(second)
    assert first_intent == second_intent  # same recipe → same intent
    assert first_token != second_token  # random placement token per launch
