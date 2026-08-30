"""A job's *connection identity* survives into ``stop`` / ``logs`` (issue #277).

``sparkrun stop <cluster_id>`` (no ``--cluster``) recovers its hosts from job
metadata, so the invocation names no cluster — and ``resolve_cluster`` answers a
bare host list with an *anonymous* ``ClusterDefinition``: no SSH user, no
executor pin, no ``executor_config``, no transport.  On a cluster whose ``user:``
differs from the control node's login that meant every teardown SSH failed with
``Permission denied`` while ``stop`` printed a success line, leaving the workload
serving and holding most of a Spark's unified memory.

The fix has two halves, and both are tested here:

* **Write** — ``save_job_metadata`` records the cluster name and the SSH user
  the launch actually connected as.
* **Read** — ``resolve_cluster_for_job`` prefers an explicitly named cluster,
  else the one the *job* recorded, else the recorded ``ssh_user`` alone.

The launcher's half of the write path (all three save sites forwarding the
identity) lives in ``test_launcher.py``, where the launch harness already is.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import sparkrun.api as api
from sparkrun.api._resolve import resolve_cluster_for_job
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import load_job_metadata, save_job_metadata
from sparkrun.orchestration.ssh import RemoteResult

CLUSTER_ID = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
CLUSTER_USER = "cluster-user"


def _recipe() -> Recipe:
    return Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})


def _cluster_manager() -> ClusterManager:
    """A manager rooted where ``resolve_cluster`` looks when given no sctx.

    ``get_config_root(None)`` returns ``DEFAULT_CONFIG_DIR``, which the
    ``isolate_stateful`` fixture points at the test's ``tmp_path`` — so this
    writes into the sandbox, never the developer's real config.
    """
    from sparkrun.core.config import get_config_root

    return ClusterManager(get_config_root())


# --------------------------------------------------------------------------
# Write side — what the launch records
# --------------------------------------------------------------------------


def test_save_records_cluster_and_ssh_user(tmp_path: Path):
    save_job_metadata(
        CLUSTER_ID,
        _recipe(),
        ["h1", "h2"],
        cache_dir=str(tmp_path),
        cluster_name="lab",
        ssh_user=CLUSTER_USER,
    )
    meta = load_job_metadata(CLUSTER_ID, cache_dir=str(tmp_path))
    assert meta["cluster"] == "lab"
    assert meta["ssh_user"] == CLUSTER_USER


def test_save_omits_unknown_connection_identity(tmp_path: Path):
    """An anonymous ``--hosts`` launch with no configured user records neither.

    Omitted rather than written empty: the read side has to be able to tell
    "there was no cluster" from "this predates sparkrun recording one", and an
    empty ``ssh_user`` would otherwise be applied as a real username.
    """
    save_job_metadata(CLUSTER_ID, _recipe(), ["h1"], cache_dir=str(tmp_path), cluster_name="", ssh_user=None)
    meta = load_job_metadata(CLUSTER_ID, cache_dir=str(tmp_path))
    assert "cluster" not in meta
    assert "ssh_user" not in meta


# --------------------------------------------------------------------------
# Read side — resolve_cluster_for_job
# --------------------------------------------------------------------------


def test_explicit_cluster_outranks_the_recorded_one():
    mgr = _cluster_manager()
    mgr.create("lab", ["h1", "h2"], user=CLUSTER_USER)
    mgr.create("other", ["h9"], user="other-user")

    resolved = resolve_cluster_for_job("other", ["h1"], meta={"cluster": "lab", "ssh_user": CLUSTER_USER})
    assert resolved.name == "other"
    assert resolved.user == "other-user"


def test_recorded_cluster_is_recovered_with_the_jobs_hosts():
    """The recorded cluster supplies *how to connect*; hosts stay the job's.

    Placement is what the metadata's ``hosts`` records — a load-aware scheduler
    may have put the workload on a subset — so recovering the cluster must not
    widen the teardown back to the cluster's full host list.
    """
    mgr = _cluster_manager()
    mgr.create("lab", ["h1", "h2", "h3"], user=CLUSTER_USER)

    resolved = resolve_cluster_for_job(None, ["h1", "h2"], meta={"cluster": "lab"})
    assert resolved.name == "lab"
    assert resolved.user == CLUSTER_USER
    assert resolved.hosts == ["h1", "h2"]


def test_recorded_cluster_carries_its_executor_pin():
    """Not just the user: the whole definition comes back.

    The executor pin is the difference between tearing down on the substrate
    that launched the workload and reporting "no such container" on another.
    """
    mgr = _cluster_manager()
    mgr.create("lab", ["h1"], user=CLUSTER_USER, executor="local")

    resolved = resolve_cluster_for_job(None, ["h1"], meta={"cluster": "lab"})
    assert resolved.executor == "local"


def test_deleted_cluster_falls_back_to_the_recorded_ssh_user():
    """A job outliving its cluster keeps the part that matters most.

    The executor pin and transport are genuinely lost, but the SSH user decides
    whether the hosts can be reached at all — so it is recorded separately and
    survives the cluster's deletion.
    """
    resolved = resolve_cluster_for_job(None, ["h1"], meta={"cluster": "deleted-cluster", "ssh_user": CLUSTER_USER})
    assert resolved.user == CLUSTER_USER
    assert resolved.hosts == ["h1"]


def test_recorded_ssh_user_applies_to_an_anonymous_launch():
    """No cluster was ever named, but the launch still knew who it connected as."""
    resolved = resolve_cluster_for_job(None, ["h1"], meta={"ssh_user": CLUSTER_USER})
    assert resolved.name == ""
    assert resolved.user == CLUSTER_USER


def test_recorded_ssh_user_never_overrides_a_resolved_cluster_user():
    """Current configuration beats history when both answer.

    The recorded user is what the *launch* used; the cluster's is what the user
    has configured *now*.  A cluster whose user was corrected must not keep
    being reached as the old one.
    """
    mgr = _cluster_manager()
    mgr.create("lab", ["h1"], user="current-user")

    resolved = resolve_cluster_for_job(None, ["h1"], meta={"cluster": "lab", "ssh_user": "stale-user"})
    assert resolved.user == "current-user"


def test_empty_cluster_name_means_unnamed_not_a_cluster_called_empty():
    """``HostContext.cluster_name`` spells "unnamed" as ``""``.

    Forwarded verbatim, that would be looked up as a cluster literally named
    ``""`` and raise — so it must fall through to the job's own record.
    """
    resolved = resolve_cluster_for_job("", ["h1"], meta={"ssh_user": CLUSTER_USER})
    assert resolved.user == CLUSTER_USER


def test_no_recorded_identity_is_unchanged_from_before():
    """Metadata written by an older sparkrun resolves exactly as it used to."""
    resolved = resolve_cluster_for_job(None, ["h1"], meta={"hosts": ["h1"]})
    assert resolved.name == ""
    assert resolved.user is None


# --------------------------------------------------------------------------
# End to end — the reported failure
# --------------------------------------------------------------------------


def _stop_capturing_ssh(cache_dir: Path, **stop_kwargs) -> dict:
    """Run ``api.stop`` with teardown mocked; return the ssh_kwargs it used."""
    captured: dict = {}

    def _cleanup(container_map, ssh_kwargs=None, executor=None, **kwargs):
        captured.update(ssh_kwargs or {})
        return {host: RemoteResult(host=host, returncode=0, stdout="REMOVED 1", stderr="") for host in container_map}

    with (
        patch("sparkrun.orchestration.primitives.cleanup_containers_by_host", side_effect=_cleanup),
        patch("sparkrun.api._stop._discover_executor_name", return_value=None),
    ):
        api.stop(cache_dir=str(cache_dir), **stop_kwargs)
    return captured


def test_stop_by_cluster_id_connects_as_the_launching_cluster(tmp_path: Path):
    """The reported bug: ``stop <id>`` with no ``--cluster`` used to SSH as $USER.

    The hosts come from metadata, so nothing named the cluster and the teardown
    ran as the control node's own login — every SSH refused, while ``stop``
    reported success.
    """
    mgr = _cluster_manager()
    mgr.create("lab", ["h1", "h2"], user=CLUSTER_USER)
    save_job_metadata(
        CLUSTER_ID,
        _recipe(),
        ["h1", "h2"],
        cache_dir=str(tmp_path),
        cluster_name="lab",
        ssh_user=CLUSTER_USER,
    )

    ssh_kwargs = _stop_capturing_ssh(tmp_path, cluster_id=CLUSTER_ID)
    assert ssh_kwargs.get("ssh_user") == CLUSTER_USER


def test_stop_by_cluster_id_uses_recorded_user_when_the_cluster_is_gone(tmp_path: Path):
    save_job_metadata(
        CLUSTER_ID,
        _recipe(),
        ["h1"],
        cache_dir=str(tmp_path),
        cluster_name="deleted-cluster",
        ssh_user=CLUSTER_USER,
    )

    ssh_kwargs = _stop_capturing_ssh(tmp_path, cluster_id=CLUSTER_ID)
    assert ssh_kwargs.get("ssh_user") == CLUSTER_USER


def test_stop_with_explicit_hosts_still_recovers_the_job_cluster(tmp_path: Path):
    """``--hosts`` narrows *where* to stop, not *how* to connect."""
    mgr = _cluster_manager()
    mgr.create("lab", ["h1", "h2"], user=CLUSTER_USER)
    save_job_metadata(CLUSTER_ID, _recipe(), ["h1", "h2"], cache_dir=str(tmp_path), cluster_name="lab", ssh_user=CLUSTER_USER)

    ssh_kwargs = _stop_capturing_ssh(tmp_path, cluster_id=CLUSTER_ID, hosts=("h1",))
    assert ssh_kwargs.get("ssh_user") == CLUSTER_USER


def test_logs_by_cluster_id_connects_as_the_launching_cluster(tmp_path: Path):
    """``logs`` shares the gap and the fix — same anonymous-cluster resolution."""
    from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload
    from sparkrun.orchestration.executors.docker import DockerExecutor

    mgr = _cluster_manager()
    mgr.create("lab", ["h1"], user=CLUSTER_USER)
    save_job_metadata(CLUSTER_ID, _recipe(), ["h1"], cache_dir=str(tmp_path), cluster_name="lab", ssh_user=CLUSTER_USER)

    snapshot = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                workloads=(
                    RunningWorkload(
                        cluster_id=CLUSTER_ID,
                        containers=(ContainerDetail(name=CLUSTER_ID + "_solo", role="solo", status="Up", image="img"),),
                    ),
                ),
            ),
        ),
        executor="docker",
    )

    captured: dict = {}

    def _read(executor, sources, follow=False, tail=None, ssh_kwargs=None):
        captured.update(ssh_kwargs or {})
        return iter(())

    with (
        patch.object(DockerExecutor, "query_status", return_value=snapshot),
        patch("sparkrun.orchestration.logs.read_log_sources", side_effect=_read),
    ):
        list(api.logs(CLUSTER_ID, cache_dir=str(tmp_path), tail=10))

    assert captured.get("ssh_user") == CLUSTER_USER


@pytest.mark.parametrize("recorded_user", [CLUSTER_USER, None])
def test_stop_never_raises_on_metadata_without_a_cluster(tmp_path: Path, recorded_user):
    """Pre-fix metadata (no ``cluster`` key) still stops, exactly as before."""
    save_job_metadata(CLUSTER_ID, _recipe(), ["h1"], cache_dir=str(tmp_path), ssh_user=recorded_user)

    ssh_kwargs = _stop_capturing_ssh(tmp_path, cluster_id=CLUSTER_ID)
    assert ssh_kwargs.get("ssh_user") == recorded_user
