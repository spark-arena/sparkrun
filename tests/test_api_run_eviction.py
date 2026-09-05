"""Tests for the relaunch-eviction half of ``sparkrun.api.run`` (issue #223).

``resolve_effective_hosts(..., exclude_intent_id=...)`` subtracts the launching
intent's own occupancy from the scheduling snapshot on the premise that the
relaunch *replaces* the prior deployment.  Under a status-aware scheduler the
new ``cluster_id`` carries a fresh random placement token, so the runtime's
"Step 1: clean up existing containers" (which removes containers by the *new*
cluster_id's names) can never match the old ones.
:func:`sparkrun.api._run._evict_superseded_deployments` is what makes the
premise true; these tests pin its scope.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import sparkrun.api as api
from sparkrun.api._run import _evict_superseded_deployments
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

INTENT = "aaaaaaaaaaaa"
OTHER_INTENT = "bbbbbbbbbbbb"

PRIOR = "sparkrun_%s_1111111111111111" % INTENT
NEW = "sparkrun_%s_2222222222222222" % INTENT
FOREIGN = "sparkrun_%s_3333333333333333" % OTHER_INTENT


def _status(*host_workloads: tuple[str, list[RunningWorkload]]) -> ClusterStatus:
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=host, workloads=tuple(loads), used_slots=len(loads)) for host, loads in host_workloads),
        executor="docker",
    )


def _workload(cluster_id: str, intent_id: str | None) -> RunningWorkload:
    return RunningWorkload(cluster_id=cluster_id, intent_id=intent_id)


@pytest.fixture
def cluster():
    return ClusterDefinition(name="c", hosts=["h1", "h2", "h3", "h4"])


def _evict(status, cluster, *, target_hosts, stop_results=None, stop_side_effect=None):
    """Run the helper against *status*, capturing the ``api.stop`` calls."""
    calls: list[dict] = []

    def _fake_stop(**kwargs):
        calls.append(kwargs)
        if stop_side_effect is not None:
            raise stop_side_effect
        return (stop_results or {}).get(kwargs["cluster_id"], _StopOk(kwargs["cluster_id"]))

    with (
        patch("sparkrun.orchestration.executor.query_status_for_cluster", return_value=status),
        patch.object(api, "stop", _fake_stop),
    ):
        evicted, _observed = _evict_superseded_deployments(
            intent_id=INTENT,
            cluster_id_for_launch=NEW,
            candidate_hosts=list(cluster.hosts),
            target_hosts=target_hosts,
            cluster_def=cluster,
            config=None,
            sctx=None,
        )
    return evicted, calls


class _StopOk:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.hosts_failed = ()


class _StopPartial:
    def __init__(self, cluster_id, hosts_failed):
        self.cluster_id = cluster_id
        self.hosts_failed = tuple(hosts_failed)


# --------------------------------------------------------------------------
# What gets evicted
# --------------------------------------------------------------------------


def test_prior_deployment_on_target_hosts_is_evicted(cluster):
    """The reported bug: a still-running deployment of the same recipe that
    occupies the hosts the relaunch was just placed on must be torn down."""
    status = _status(*[(h, [_workload(PRIOR, INTENT)]) for h in cluster.hosts])

    evicted, calls = _evict(status, cluster, target_hosts=list(cluster.hosts))

    assert evicted == [PRIOR]
    assert len(calls) == 1
    assert calls[0]["cluster_id"] == PRIOR
    assert calls[0]["hosts"] == ["h1", "h2", "h3", "h4"]


def test_eviction_covers_every_host_the_deployment_occupies(cluster):
    """An overlapping deployment is torn down across all of its hosts, not
    only the ones the new placement happens to reuse — half a distributed job
    is dead weight either way."""
    status = _status(
        ("h1", [_workload(PRIOR, INTENT)]),
        ("h2", [_workload(PRIOR, INTENT)]),
        ("h3", [_workload(PRIOR, INTENT)]),
        ("h4", [_workload(PRIOR, INTENT)]),
    )

    evicted, calls = _evict(status, cluster, target_hosts=["h1", "h2"])

    assert evicted == [PRIOR]
    assert calls[0]["hosts"] == ["h1", "h2", "h3", "h4"]


def test_intent_recovered_from_cluster_id_prefix_when_label_absent(cluster):
    """Containers launched before the sparkrun labels existed report
    ``intent_id=None``; the cluster_id prefix still identifies them."""
    status = _status(("h1", [_workload(PRIOR, None)]))

    evicted, _ = _evict(status, cluster, target_hosts=["h1"])

    assert evicted == [PRIOR]


# --------------------------------------------------------------------------
# What is left alone
# --------------------------------------------------------------------------


def test_foreign_intent_is_never_evicted(cluster):
    """A different recipe sharing the cluster is a capacity question for the
    scheduler — a launch may not unilaterally kill someone else's job."""
    status = _status(*[(h, [_workload(FOREIGN, OTHER_INTENT)]) for h in cluster.hosts])

    evicted, calls = _evict(status, cluster, target_hosts=list(cluster.hosts))

    assert evicted == []
    assert calls == []


def test_same_intent_on_disjoint_hosts_is_left_running(cluster):
    """Running the same intent twice on disjoint host subsets is a supported
    use of the random placement token, so a non-overlapping sibling
    deployment survives."""
    status = _status(
        ("h3", [_workload(PRIOR, INTENT)]),
        ("h4", [_workload(PRIOR, INTENT)]),
    )

    evicted, calls = _evict(status, cluster, target_hosts=["h1", "h2"])

    assert evicted == []
    assert calls == []


def test_launching_cluster_id_is_not_evicted(cluster):
    """Never tear down the deployment we are in the middle of creating."""
    status = _status(("h1", [_workload(NEW, INTENT)]))

    evicted, calls = _evict(status, cluster, target_hosts=["h1"])

    assert evicted == []
    assert calls == []


def test_mixed_snapshot_evicts_only_the_overlapping_sibling(cluster):
    status = _status(
        ("h1", [_workload(PRIOR, INTENT), _workload(FOREIGN, OTHER_INTENT)]),
        ("h2", [_workload("sparkrun_%s_4444444444444444" % INTENT, INTENT)]),
    )

    evicted, calls = _evict(status, cluster, target_hosts=["h1"])

    assert evicted == [PRIOR]
    assert [c["cluster_id"] for c in calls] == [PRIOR]


# --------------------------------------------------------------------------
# Best-effort semantics — never block the launch
# --------------------------------------------------------------------------


def test_status_query_failure_is_swallowed(cluster):
    with patch(
        "sparkrun.orchestration.executor.query_status_for_cluster",
        side_effect=RuntimeError("hosts unreachable"),
    ):
        assert (
            _evict_superseded_deployments(
                intent_id=INTENT,
                cluster_id_for_launch=NEW,
                candidate_hosts=list(cluster.hosts),
                target_hosts=list(cluster.hosts),
                cluster_def=cluster,
                config=None,
                sctx=None,
            )
            # Nothing evicted, and — importantly — ``None`` rather than an
            # empty set for the observed workloads: "couldn't look" must not
            # read as "looked, nothing there", since the post-launch metadata
            # prune treats the latter as licence to delete.
            == ([], None)
        )


def test_strict_status_query_failure_aborts(cluster):
    with patch(
        "sparkrun.orchestration.executor.query_status_for_cluster",
        side_effect=RuntimeError("hosts unreachable"),
    ):
        with pytest.raises(RuntimeError, match="could not query cluster status"):
            _evict_superseded_deployments(
                intent_id=INTENT,
                cluster_id_for_launch=NEW,
                candidate_hosts=list(cluster.hosts),
                target_hosts=list(cluster.hosts),
                cluster_def=cluster,
                config=None,
                sctx=None,
                strict=True,
            )


def test_stop_failure_is_swallowed(cluster):
    status = _status(("h1", [_workload(PRIOR, INTENT)]))

    evicted, calls = _evict(status, cluster, target_hosts=["h1"], stop_side_effect=RuntimeError("ssh down"))

    assert evicted == []
    assert len(calls) == 1  # attempted, then gave up on that deployment


def test_strict_stop_failure_aborts(cluster):
    status = _status(("h1", [_workload(PRIOR, INTENT)]))

    with (
        patch("sparkrun.orchestration.executor.query_status_for_cluster", return_value=status),
        patch.object(api, "stop", side_effect=RuntimeError("ssh down")),
    ):
        with pytest.raises(RuntimeError, match="could not stop earlier deployment"):
            _evict_superseded_deployments(
                intent_id=INTENT,
                cluster_id_for_launch=NEW,
                candidate_hosts=list(cluster.hosts),
                target_hosts=["h1"],
                cluster_def=cluster,
                config=None,
                sctx=None,
                strict=True,
            )


def test_unconfirmed_teardown_still_counts_as_attempted(cluster, caplog):
    """A partially-confirmed teardown warns (the containers may still hold
    GPU memory) but does not abort the launch."""
    status = _status(("h1", [_workload(PRIOR, INTENT)]))

    with caplog.at_level("WARNING"):
        evicted, _ = _evict(
            status,
            cluster,
            target_hosts=["h1"],
            stop_results={PRIOR: _StopPartial(PRIOR, ["h1"])},
        )

    assert evicted == [PRIOR]
    assert "did not confirm" in caplog.text


def test_strict_unconfirmed_teardown_aborts(cluster):
    status = _status(("h1", [_workload(PRIOR, INTENT)]))

    with (
        patch("sparkrun.orchestration.executor.query_status_for_cluster", return_value=status),
        patch.object(api, "stop", return_value=_StopPartial(PRIOR, ["h1"])),
    ):
        with pytest.raises(RuntimeError, match="was not confirmed on h1"):
            _evict_superseded_deployments(
                intent_id=INTENT,
                cluster_id_for_launch=NEW,
                candidate_hosts=list(cluster.hosts),
                target_hosts=["h1"],
                cluster_def=cluster,
                config=None,
                sctx=None,
                strict=True,
            )


# --------------------------------------------------------------------------
# Wiring into api.run
# --------------------------------------------------------------------------


class _FakeRuntime:
    runtime_name = "vllm"
    executor = None

    def world_size(self, parallelism, recipe=None, cluster=None):
        return 1


def _run_with_stubbed_launcher(opts):
    """``api.run(opts)`` with a hermetic launcher, returning the eviction mock."""

    def _fake_launch(**kwargs):
        # Honour the real launcher's contract: ``before_start`` fires once,
        # immediately before containers start — i.e. after every step that can
        # fail slowly (distribution, model download).  A fake that skipped it
        # would let the eviction move back to the top of ``api.run`` unnoticed.
        hook = kwargs.get("before_start")
        if hook is not None and not kwargs.get("dry_run"):
            hook()
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
                "timeline": None,
            },
        )()

    with (
        patch("sparkrun.core.launcher.launch_inference", side_effect=_fake_launch),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
        patch("sparkrun.api._run._evict_superseded_deployments", return_value=([], set())) as evict,
    ):
        api.run(opts)
    return evict


def _run_options(**kwargs):
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    return api.RunOptions(recipe=recipe, hosts=("h1",), scheduler="occupancy-sparse", **kwargs)


def test_dry_run_never_evicts():
    """``--dry-run`` computes everything but must not tear anything down."""
    _run_with_stubbed_launcher(_run_options(dry_run=True)).assert_not_called()


def test_real_run_evicts_before_starting_containers():
    """Positive control for the test above: a non-dry-run reaches the
    eviction step, and hands it the launch's own cluster_id so it can't tear
    down the deployment it is creating."""
    evict = _run_with_stubbed_launcher(_run_options(dry_run=False, follow=False))

    evict.assert_called_once()
    kwargs = evict.call_args.kwargs
    assert kwargs["target_hosts"] == ["h1"]
    assert kwargs["cluster_id_for_launch"].startswith("sparkrun_%s_" % kwargs["intent_id"])


def test_launch_that_dies_before_starting_containers_evicts_nothing():
    """An interrupted launch must leave the running deployment alone.

    Eviction used to run at the top of ``api.run``, before image distribution
    and the model download — steps that take minutes and are routinely
    Ctrl-C'd.  A launch killed there tore down the serving workload it was
    replacing and then died without replacing it, leaving the cluster empty.
    Now the teardown is the last thing before containers start, so anything
    that fails earlier is harmless.
    """

    def _die_before_start(**kwargs):
        # Model a Ctrl-C during "Distributing resources": the launcher raises
        # without ever reaching its ``before_start`` hook.
        raise KeyboardInterrupt

    with (
        patch("sparkrun.core.launcher.launch_inference", side_effect=_die_before_start),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
        patch("sparkrun.api._run._evict_superseded_deployments", return_value=([], set())) as evict,
    ):
        with pytest.raises(KeyboardInterrupt):
            api.run(_run_options(dry_run=False, follow=False))

    evict.assert_not_called()
