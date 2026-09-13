"""Regression tests: ``sparkrun run`` decides its placement exactly once.

``sparkrun run`` used to place *twice* — once in the CLI (so the banner, the
per-host VRAM fit block and diagnostics could name the target hosts) and again
inside :func:`sparkrun.api.run`.  The CLI pass **narrowed** the candidate host
list, so the second pass could only choose among the survivors.

That made the two passes' inputs load-bearing, and they diverged: the CLI
helper never forwarded the resolved scheduler, so the first pass ran the
``None`` → greedy default while the launch used the cluster's
``occupancy-sparse``.  On a 4-host cluster with two busy hosts, greedy handed
``api.run`` the leading two hosts regardless of load; ``occupancy-sparse`` then
found one occupied, had no other candidates left, and the launch died with::

    Error: cluster has insufficient free capacity for 2 node(s): Cluster cannot
    satisfy 2 ranks under current occupancy: only placed 1 of 2 ranks across 2 host(s)

…while two idle hosts sat unused (reported against 0.3.3).

The CLI now calls :func:`sparkrun.api.plan` once and hands the result to
``api.run(plan=…)``, so the failure mode is structural rather than a matter of
keeping two call sites in sync.  These tests pin both properties: the scenario
above must launch, and there must remain only one placement pass.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload
from sparkrun.runtimes.sglang import SglangRuntime

_RECIPE_NAME = "test-sched-thread-cluster"

_RECIPE_DATA = {
    "sparkrun_version": "2",
    "name": "Scheduler threading regression recipe",
    "description": "tp=2 cluster recipe used to exercise the CLI host trim",
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "sglang",
    "mode": "cluster",
    "min_nodes": 1,
    "max_nodes": 8,
    "container": "scitrera/dgx-spark-sglang:latest",
    "defaults": {
        "port": 30000,
        "host": "0.0.0.0",
        "tensor_parallel": 2,
    },
    "metadata": {
        "model_params": 1700000000,
        "model_dtype": "float16",
    },
}

_HOSTS = ["10.0.4.30", "10.0.4.31", "10.0.4.32", "10.0.4.33"]
#: The two hosts occupied in :func:`_busy_status` — the *leading* two, which is
#: exactly the pair a greedy trim would pick.
_BUSY = _HOSTS[:2]
_FREE = _HOSTS[2:]


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cluster_env(tmp_path, monkeypatch, v):
    """A 4-host ``occupancy-sparse`` cluster plus a discoverable tp=2 recipe."""
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

    mgr = ClusterManager(config_root)
    mgr.create("wopr", list(_HOSTS), scheduler="occupancy-sparse")

    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    recipe_file = recipe_dir / ("%s.yaml" % _RECIPE_NAME)
    recipe_file.write_text(yaml.safe_dump(_RECIPE_DATA))

    import sparkrun.core.recipe

    original_discover = sparkrun.core.recipe.discover_cwd_recipes
    monkeypatch.setattr(
        sparkrun.core.recipe,
        "discover_cwd_recipes",
        lambda directory=None: [recipe_file] + original_discover(directory),
    )
    return config_root


def _busy_status(hosts) -> ClusterStatus:
    """Snapshot where :data:`_BUSY` are fully occupied and the rest are idle."""
    return ClusterStatus(
        hosts=tuple(
            HostOccupancy(
                host=h,
                workloads=(RunningWorkload(cluster_id="sparkrun_deadbeef_cafe"),) if h in _BUSY else (),
                used_slots=1 if h in _BUSY else 0,
                free_slots=0 if h in _BUSY else 1,
            )
            for h in hosts
        ),
        executor="docker",
    )


@pytest.fixture
def busy_cluster_status(monkeypatch):
    """Point every ``api.status`` call site at :func:`_busy_status`."""
    _stub = lambda hosts, **kwargs: _busy_status(list(hosts))  # noqa: E731
    monkeypatch.setattr("sparkrun.api._status.status", _stub)
    monkeypatch.setattr("sparkrun.api.status", _stub)


def test_placement_uses_the_clusters_scheduler(runner, cluster_env, busy_cluster_status, monkeypatch):
    """Every ``api.schedule`` call names the cluster's scheduler, not greedy.

    This is the original root cause in isolation: a pass that scheduled with
    ``scheduler=None`` fell back to greedy and ignored occupancy entirely.
    """
    import sparkrun.api as api

    seen: list[str | None] = []
    real_schedule = api.schedule

    def _spy(request, *, scheduler=None, sctx=None):
        seen.append(scheduler)
        return real_schedule(request, scheduler=scheduler, sctx=sctx)

    monkeypatch.setattr(api, "schedule", _spy)

    with mock.patch.object(SglangRuntime, "run", return_value=0):
        result = runner.invoke(main, ["run", _RECIPE_NAME, "--cluster", "wopr", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert seen, "the CLI never scheduled"
    assert set(seen) == {"occupancy-sparse"}


def test_reported_scenario_launches_on_the_idle_hosts(runner, cluster_env, busy_cluster_status):
    """End-to-end reproduction: a tp=2 launch lands on the two idle hosts.

    Before the fix the greedy pass selected ``_BUSY`` (the leading two hosts),
    ``api.run`` re-scheduled over only those and raised "insufficient free
    capacity" — with ``_FREE`` never considered.
    """
    with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
        result = runner.invoke(main, ["run", _RECIPE_NAME, "--cluster", "wopr", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "insufficient free capacity" not in result.output

    # Banner reflects the idle pair...
    assert "Head:    %s" % _FREE[0] in result.output
    assert "Workers: %s" % _FREE[1] in result.output
    # ...and so does the placement actually handed to the runtime.
    assert list(mock_run.call_args.kwargs["hosts"]) == _FREE


def test_cli_places_exactly_once(runner, cluster_env, busy_cluster_status):
    """A launch runs the placement authority once — not once per renderer.

    The banner cannot disagree with the launch if there is only one decision.
    This is the structural guarantee behind the fix: the CLI calls
    ``api.plan`` and hands the result to ``api.run(plan=…)``, so no pass can
    narrow the candidate set out from under a later one.
    """
    from sparkrun.api._hosts import resolve_effective_hosts as _real_resolve

    calls: list[list[str]] = []

    def _spy(host_list, *args, **kwargs):
        out = _real_resolve(host_list, *args, **kwargs)
        calls.append(list(out[0]))
        return out

    with mock.patch("sparkrun.api._hosts.resolve_effective_hosts", _spy):
        with mock.patch.object(SglangRuntime, "run", return_value=0):
            result = runner.invoke(main, ["run", _RECIPE_NAME, "--cluster", "wopr", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1, "placement ran %d times, expected exactly 1" % len(calls)
    assert calls[0] == _FREE


def test_cluster_occupancy_is_swept_once(runner, cluster_env, monkeypatch):
    """One launch → one occupancy sweep.

    Each placement pass queried live status over SSH, so the old two-pass
    flow paid for the cluster round-trip twice — and a workload starting
    between the two sweeps could make the second disagree with the first.
    """
    sweeps: list[list[str]] = []

    def _stub(hosts, **kwargs):
        sweeps.append(list(hosts))
        return _busy_status(list(hosts))

    monkeypatch.setattr("sparkrun.api._status.status", _stub)
    monkeypatch.setattr("sparkrun.api.status", _stub)

    with mock.patch.object(SglangRuntime, "run", return_value=0):
        result = runner.invoke(main, ["run", _RECIPE_NAME, "--cluster", "wopr", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert len(sweeps) == 1, "occupancy swept %d times: %r" % (len(sweeps), sweeps)
    assert sweeps[0] == _HOSTS


def test_api_run_sees_full_candidate_set(runner, cluster_env, busy_cluster_status):
    """``api.run`` is handed every host, not the CLI's chosen subset.

    ``candidate_hosts`` feeds the deterministic placement token and the
    superseded-deployment sweep, both of which must agree with ``stop`` /
    ``status`` — which only ever know the cluster's full host list.
    """
    import sparkrun.api as api

    seen: dict[str, object] = {}
    real_plan = api.plan

    def _spy(options, *, sctx=None):
        plan = real_plan(options, sctx=sctx)
        seen["options_hosts"] = list(options.hosts or ())
        seen["candidates"] = list(plan.candidate_hosts)
        seen["targets"] = list(plan.host_list)
        return plan

    with mock.patch.object(api, "plan", _spy):
        with mock.patch.object(SglangRuntime, "run", return_value=0):
            result = runner.invoke(main, ["run", _RECIPE_NAME, "--cluster", "wopr", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert seen["options_hosts"] == _HOSTS
    assert seen["candidates"] == _HOSTS
    assert seen["targets"] == _FREE


@pytest.mark.parametrize("host_option", ["--hosts", "--hosts-file"])
def test_named_cluster_survives_explicit_host_subset(runner, cluster_env, busy_cluster_status, tmp_path, host_option):
    from sparkrun import api

    host_file = tmp_path / "selected-hosts"
    host_file.write_text("\n".join(_FREE))
    value = str(host_file) if host_option == "--hosts-file" else ",".join(_FREE)
    with mock.patch.object(api, "run", wraps=api.run) as run:
        with mock.patch.object(SglangRuntime, "run", return_value=0):
            result = runner.invoke(main, ["run", _RECIPE_NAME, "--cluster", "wopr", host_option, value, "--dry-run"])
    assert result.exit_code == 0, result.output
    plan = run.call_args.kwargs["plan"]
    assert plan.cluster.name == "wopr"
    assert plan.scheduler == "occupancy-sparse"
    assert list(plan.host_list) == _FREE
    assert run.call_args.args[0].auto_port is False
