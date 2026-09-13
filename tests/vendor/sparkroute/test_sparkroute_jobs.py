# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

import json
import os
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import sparkrun.api as api
from sparkrun.plugins.sparkroute import jobs, operations
from sparkrun.plugins.sparkroute.protocol import Binding, ProtocolError, Request


def activation(recipe="catalog:one", cluster="lab", revision="fingerprint"):
    return Request("test", "ensure_ready", binding=Binding(recipe, revision, (cluster,)), wait=False)


def test_workers_coalesce_same_workload_across_recipe_aliases_but_not_clusters(monkeypatch):
    sctx = api.default_sctx()
    with (
        mock.patch.object(jobs.subprocess, "Popen", return_value=SimpleNamespace(pid=12345)) as spawn,
        mock.patch.object(jobs, "_alive", return_value=True),
    ):
        first = jobs.start_operation(activation(), sctx=sctx)
        alias = jobs.start_operation(activation("catalog:alias"), sctx=sctx)
        other_cluster = jobs.start_operation(activation(cluster="other"), sctx=sctx)
        other_model = jobs.start_operation(activation(revision="another"), sctx=sctx)
    assert first["operation_id"] == alias["operation_id"]
    assert len({first["operation_id"], other_cluster["operation_id"], other_model["operation_id"]}) == 3
    assert spawn.call_count == 3
    assert spawn.call_args.kwargs["stdin"] == jobs.subprocess.DEVNULL
    assert "request" not in first and "pid" not in first
    if os.name != "nt":
        assert spawn.call_args.kwargs["start_new_session"] is True
        assert jobs._path(sctx).stat().st_mode & 0o077 == 0


def test_dead_worker_reuses_operation_and_preserves_placement():
    sctx = api.default_sctx()
    with mock.patch.object(jobs.subprocess, "Popen", return_value=SimpleNamespace(pid=12345)):
        first = jobs.start_operation(activation(), sctx=sctx)
    placement = {"cluster_id": "job", "cluster": "lab", "hosts": ["host"], "port": 8001, "solo": True}
    with jobs._connect(jobs._path(sctx)) as db:
        db.execute("UPDATE operations SET placement=?", (json.dumps(placement),))
    with (
        mock.patch.object(jobs.subprocess, "Popen", return_value=SimpleNamespace(pid=54321)) as spawn,
        mock.patch.object(jobs, "_alive", return_value=False),
    ):
        status = jobs.operation_status(first["operation_id"], sctx=sctx)
        assert status["state"] == "failed" and status["phase"] == "interrupted"
        spawn.assert_not_called()
        recovered = jobs.start_operation(activation(), sctx=sctx)
    assert recovered["operation_id"] == first["operation_id"]
    assert recovered["cluster_id"] == "job" and "hosts" not in recovered
    spawn.assert_called_once()
    with jobs._connect(jobs._path(sctx)) as db:
        assert json.loads(db.execute("SELECT placement FROM operations").fetchone()[0]) == placement


def test_failed_activation_retains_placement_on_retry():
    sctx = api.default_sctx()
    with mock.patch.object(jobs.subprocess, "Popen", return_value=SimpleNamespace(pid=12345)):
        first = jobs.start_operation(activation(), sctx=sctx)
        with jobs._connect(jobs._path(sctx)) as db:
            db.execute("UPDATE operations SET state='failed', placement='{}'")
        second = jobs.start_operation(activation(), sctx=sctx)
    assert first["operation_id"] == second["operation_id"]


def test_worker_runs_without_original_request_and_persists_refresh_results(monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "1")
    sctx = api.default_sctx()
    with mock.patch.object(jobs.subprocess, "Popen", return_value=SimpleNamespace(pid=os.getpid())):
        first = jobs.start_operation(Request("r", "catalog_refresh"), sctx=sctx)
    with mock.patch.object(api, "refresh_registries", return_value={"updated": {"one": True, "two": False}, "failed": ["two"]}):
        jobs.run_worker(sctx.config.config_path, first["operation_id"])
    status = jobs.operation_status(first["operation_id"], sctx=sctx)
    assert status["state"] == "succeeded"
    assert status["result"]["failed"] == ["two"]


def test_recovery_waits_for_existing_job_without_relaunching():
    placement = {"cluster_id": "job", "cluster": "lab", "hosts": ["host"], "port": 8001, "solo": True}
    snapshot = SimpleNamespace(errors={}, hosts=[SimpleNamespace(workloads=[SimpleNamespace(cluster_id="job")])])
    with (
        mock.patch.object(api, "status", return_value=snapshot),
        mock.patch.object(api, "run") as run,
        mock.patch("sparkrun.api._resolve.resolve_runtime", return_value=object()),
        mock.patch("sparkrun.core.launcher.wait_for_endpoint_ready") as ready,
        mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
        mock.patch.object(operations, "_discover", return_value=[{"port": 8001}]),
    ):
        result = operations._recover_launch(
            placement,
            activation().binding,
            SimpleNamespace(post_exec=[], post_commands=[]),
            "fingerprint",
            SimpleNamespace(config=object()),
            operations.time.monotonic() + 30,
        )
    assert result["endpoint"]["port"] == 8001
    assert ready.call_args.kwargs["port"] == 8001
    run.assert_not_called()


def test_unreachable_recovery_cannot_launch_a_duplicate():
    with mock.patch.object(api, "status", return_value=SimpleNamespace(errors={"host": "unreachable"}, hosts=[])):
        with pytest.raises(ProtocolError, match="Cannot confirm"):
            operations._recover_launch({"hosts": ["host"], "cluster": "lab"}, activation().binding, object(), "revision", object(), 0)


def test_discovery_and_stop_use_authoritative_named_cluster():
    entries = [
        SimpleNamespace(cluster_id=name, metadata={"recipe_fingerprint": "fp", "cluster": cluster, "owner": "sparkroute"})
        for name, cluster in [("a", "lab"), ("b", "other"), ("old", None)]
    ]
    with (
        mock.patch.object(api, "list_jobs", return_value=entries),
        mock.patch.object(operations, "discover_endpoints", return_value=[]) as discover,
    ):
        operations._discover(object(), fingerprint="fp", cluster_candidates=("lab",))
        assert discover.call_args.kwargs["cluster_ids"] == {"a"}
        with pytest.raises(ProtocolError, match="does not belong"):
            operations._stop("fp", "b", object(), cluster_candidates=("lab",))


def test_retry_recovers_persisted_job_after_successful_worker_is_gone():
    job = SimpleNamespace(
        cluster_id="recorded-job", hosts=["host"], metadata={"recipe_fingerprint": "fingerprint", "cluster": "lab", "port": 8123}
    )
    with (
        mock.patch.object(operations, "_discover", return_value=[]),
        mock.patch.object(jobs, "previous_placement", return_value=None),
        mock.patch.object(api, "list_jobs", return_value=[job]),
        mock.patch.object(operations, "_recover_launch", return_value={"state": "ready"}) as recover,
        mock.patch.object(api, "run") as run,
    ):
        assert operations._ensure_ready(activation(), activation().binding, object(), "fingerprint", object()) == {"state": "ready"}
    assert recover.call_args.args[0]["port"] == 8123
    assert recover.call_args.args[0]["cluster_id"] == "recorded-job"
    run.assert_not_called()


def test_synchronous_bridge_waits_on_same_durable_operation():
    request = Request("test", "ensure_ready", binding=activation().binding, wait=True)
    with (
        mock.patch.object(operations, "_require_feature_enabled"),
        mock.patch.object(operations, "_resolve_binding", return_value=(object(), "fingerprint")),
        mock.patch.object(jobs, "start_operation", return_value={"operation_id": "same", "state": "running"}) as start,
        mock.patch.object(jobs, "wait_operation", return_value={"state": "ready"}) as wait,
        mock.patch.object(api, "run") as run,
    ):
        assert operations.execute(request) == {"state": "ready"}
    assert wait.call_args.args[0]["operation_id"] == "same"
    start.assert_called_once()
    run.assert_not_called()


def test_wait_timeout_does_not_cancel_worker():
    with mock.patch.object(jobs, "operation_status") as status:
        with pytest.raises(ProtocolError, match="continues in the background"):
            jobs.wait_operation({"operation_id": "same", "state": "running"}, 0, sctx=object())
    status.assert_not_called()


def test_real_detached_worker_persists_failure_without_launching(monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "1")
    sctx = api.default_sctx()
    request = activation(recipe="catalog:" + "0" * 32)
    operation = jobs.start_operation(request, sctx=sctx)
    deadline = time.monotonic() + 20
    while operation["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        operation = jobs.operation_status(operation["operation_id"], sctx=sctx)
    assert operation["state"] == "failed"
    assert operation["error"]["code"] == "recipe_not_found"
    assert (jobs._path(sctx).parent / (operation["operation_id"] + ".log")).is_file()
