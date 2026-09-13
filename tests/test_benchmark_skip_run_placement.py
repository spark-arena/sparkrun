"""Regression tests: ``sparkrun benchmark --skip-run`` targets the running workload.

``--skip-run`` was the one benchmark path that never placed.  Every other path
routes through ``api.plan`` → ``resolve_effective_hosts``, so a ``tp: 1`` recipe
on a 4-host cluster resolved to one host exactly as ``sparkrun run`` does; the
``--skip-run`` branch instead kept the full candidate list and reported
``Mode: cluster (4 nodes)`` for a solo workload.

Three things rode on that host list, in ascending order of damage:

* the banner (cosmetic),
* ``head_host = host_list[0]`` — the benchmark aimed at the *first host of the
  cluster*, reaching the right server only when the workload happened to land
  there,
* ``export_results(hosts=host_list, …)`` — a one-node measurement published as
  a four-node one.

The fix asks the question ``--skip-run`` actually poses ("where is this
workload serving?") via ``api.find_running_intent``, the same intent-keyed
lookup ``--ensure`` uses.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.api._benchmark import _resolve_running_deployment
from sparkrun.api._intent import IntentMatch
from sparkrun.cli import main

_RECIPE_NAME = "test-benchmark-skip-run"

#: ``tensor_parallel: 1`` — a solo workload on a multi-host cluster, which is
#: the shape that exposed the bug.
_RECIPE_DATA = {
    "sparkrun_version": "2",
    "name": "Benchmark skip-run placement recipe",
    "description": "single-rank recipe benchmarked via --skip-run",
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "sglang",
    "mode": "cluster",
    "container": "scitrera/dgx-spark-sglang:latest",
    "defaults": {
        "port": 30000,
        "host": "0.0.0.0",
        "tensor_parallel": 1,
    },
    "metadata": {
        "model_params": 1700000000,
        "model_dtype": "float16",
    },
}

_HOSTS = ["10.0.4.30", "10.0.4.31", "10.0.4.32", "10.0.4.33"]

#: The workload is deliberately NOT on ``_HOSTS[0]`` — the old code took
#: ``host_list[0]`` and would have looked correct on the first host by accident.
_RUNNING_HOST = "10.0.4.32"


# ---------------------------------------------------------------------------
# Unit: the decision helper
# ---------------------------------------------------------------------------


class _Recipe:
    """Minimal stand-in for the recipe surface the helper reads."""

    mode = "cluster"
    defaults: dict = {}
    runtime = "sglang"
    model = "org/model"
    container = "img:tag"


class _Emitter:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _resolve(match_or_exc, *, solo=False, hosts=None, emitter=None):
    """Drive the helper with ``find_running_intent`` stubbed to *match_or_exc*."""
    seen: dict = {}

    def _fake(intent_id, query_hosts, *, cluster=None, sctx=None, status=None):
        seen["intent_id"] = intent_id
        seen["hosts"] = list(query_hosts)
        if isinstance(match_or_exc, Exception):
            raise match_or_exc
        return match_or_exc

    import sparkrun.api as api

    with mock.patch.object(api, "find_running_intent", _fake):
        out = _resolve_running_deployment(
            _Recipe(),
            {},
            list(hosts if hosts is not None else _HOSTS),
            solo=solo,
            cluster=None,
            sctx=None,
            emitter=emitter or _Emitter(),
        )
    return out, seen


def test_adopts_the_running_deployments_hosts_and_cluster_id():
    """The reported bug: benchmark the host serving the workload, not the cluster."""
    match = IntentMatch(intent_id="abc", cluster_id="sparkrun_abc_def456", hosts=(_RUNNING_HOST,))
    (hosts, is_solo, cluster_id), _ = _resolve(match)

    assert hosts == [_RUNNING_HOST]
    assert is_solo is True
    assert cluster_id == "sparkrun_abc_def456"


def test_lookup_sees_the_full_candidate_list():
    """A deployment on a host this benchmark would not have chosen still counts.

    Narrowing before the lookup is how ``--ensure`` used to miss jobs entirely.
    """
    match = IntentMatch(intent_id="abc", cluster_id="sparkrun_abc_def456", hosts=(_RUNNING_HOST,))
    _, seen = _resolve(match)

    assert seen["hosts"] == _HOSTS


def test_multi_host_deployment_is_not_reported_as_solo():
    match = IntentMatch(intent_id="abc", cluster_id="sparkrun_abc_def456", hosts=tuple(_HOSTS[:2]))
    (hosts, is_solo, _cid), _ = _resolve(match)

    assert hosts == _HOSTS[:2]
    assert is_solo is False


def test_no_match_falls_back_to_candidates_with_a_warning():
    """ "Couldn't find it" must not become "refuse to benchmark" — but must be said."""
    emitter = _Emitter()
    (hosts, is_solo, cluster_id), _ = _resolve(None, emitter=emitter)

    assert hosts == _HOSTS
    assert is_solo is False
    assert cluster_id is None
    assert any("no running workload matched" in w for w in emitter.warnings)


def test_failed_status_query_degrades_instead_of_raising():
    """An unreachable cluster is "couldn't tell", not a benchmark failure."""
    emitter = _Emitter()
    (hosts, _is_solo, cluster_id), _ = _resolve(RuntimeError("ssh exploded"), emitter=emitter)

    assert hosts == _HOSTS
    assert cluster_id is None
    assert emitter.warnings


def test_fallback_still_honours_solo():
    emitter = _Emitter()
    (hosts, is_solo, _cid), _ = _resolve(None, solo=True, emitter=emitter)

    assert hosts == [_HOSTS[0]]
    assert is_solo is True


def test_several_matching_deployments_are_reported():
    match = IntentMatch(
        intent_id="abc",
        cluster_id="sparkrun_abc_def456",
        hosts=(_RUNNING_HOST,),
        other_cluster_ids=("sparkrun_abc_999999",),
    )
    emitter = _Emitter()
    (hosts, _is_solo, cluster_id), _ = _resolve(match, emitter=emitter)

    assert hosts == [_RUNNING_HOST]
    assert cluster_id == "sparkrun_abc_def456"
    assert any("2 deployments" in w for w in emitter.warnings)


# ---------------------------------------------------------------------------
# End to end: the banner the user actually sees
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cluster_status(monkeypatch):
    """A 4-host snapshot with the recipe's workload running on ``_RUNNING_HOST``.

    The workload's ``intent_id`` is left ``None`` so the match goes through the
    cluster_id prefix — the path a real ``docker ps`` sweep takes when the
    container carries no intent label.
    """
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    state: dict = {"intent_id": None}

    def _stub(hosts, **kwargs):
        occ = []
        for h in hosts:
            workloads = ()
            if h == _RUNNING_HOST and state["intent_id"]:
                workloads = (RunningWorkload(cluster_id="sparkrun_%s_aabbccddeeff" % state["intent_id"]),)
            occ.append(
                HostOccupancy(
                    host=h,
                    workloads=workloads,
                    used_slots=1 if workloads else 0,
                    free_slots=0 if workloads else 1,
                )
            )
        return ClusterStatus(hosts=tuple(occ), executor="docker")

    monkeypatch.setattr("sparkrun.api._status.status", _stub)
    monkeypatch.setattr("sparkrun.api.status", _stub)
    return state


@pytest.fixture
def bench_env(tmp_path, monkeypatch, v, cluster_status):
    """A 4-host cluster plus a discoverable single-rank recipe."""
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

    ClusterManager(config_root).create("wopr", list(_HOSTS))

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


def _invoke(runner, extra_argv, *, arm=None):
    """Run ``sparkrun benchmark``, optionally arming the status stub.

    The workload's cluster_id must carry the *real* intent id the flow derives,
    so rather than recomputing it here (and silently diverging the day
    ``generate_intent_id`` changes) the lookup is wrapped in a pass-through spy
    that hands the id to the snapshot stub before delegating to the real
    implementation.
    """
    import sparkrun.api as api

    real = api.find_running_intent

    def _spy(intent_id, hosts, **kwargs):
        if arm is not None:
            arm["intent_id"] = intent_id
        return real(intent_id, hosts, **kwargs)

    argv = ["benchmark", _RECIPE_NAME, "--cluster", "wopr", "--dry-run", "--fresh"] + extra_argv
    with mock.patch.object(api, "find_running_intent", _spy):
        return runner.invoke(main, argv)


def test_skip_run_banner_names_the_workloads_host(runner, bench_env, cluster_status):
    """The end-user symptom: a solo workload reported as ``cluster (4 nodes)``."""
    result = _invoke(runner, ["--skip-run"], arm=cluster_status)

    assert result.exit_code == 0, result.output
    assert "Hosts:                 %s" % _RUNNING_HOST in result.output
    assert "Mode:                  solo" in result.output
    assert "cluster (4 nodes)" not in result.output


def test_skip_run_without_a_running_workload_warns(runner, bench_env, cluster_status):
    """Nothing found → previous behaviour, but said out loud rather than implied."""
    result = _invoke(runner, ["--skip-run"])

    assert result.exit_code == 0, result.output
    assert "no running workload matched" in result.output


def test_without_skip_run_placement_is_unchanged(runner, bench_env, cluster_status):
    """The planning path must not acquire the discovery behaviour."""
    result = _invoke(runner, [])

    assert result.exit_code == 0, result.output
    assert "Mode:                  solo" in result.output
    assert "no running workload matched" not in result.output


@pytest.mark.parametrize(
    ("flags", "expected_cluster"),
    [
        ([], "wopr"),
        (["--cluster", "wopr"], "wopr"),
        (["--cluster", "wopr", "--hosts", _HOSTS[-1]], "wopr"),
        (["--hosts", _HOSTS[-1]], ""),
    ],
)
def test_benchmark_launch_keeps_effective_cluster(runner, bench_env, cluster_status, flags, expected_cluster):
    from sparkrun import api
    from sparkrun.core.cluster_manager import ClusterManager

    ClusterManager(bench_env).set_default("wopr")
    with mock.patch.object(api, "run", wraps=api.run) as run:
        result = runner.invoke(main, ["benchmark", _RECIPE_NAME, *flags, "--dry-run", "--fresh"])
    assert result.exit_code == 0, result.output
    assert run.call_count == 1
    assert run.call_args.kwargs["plan"].cluster.name == expected_cluster
