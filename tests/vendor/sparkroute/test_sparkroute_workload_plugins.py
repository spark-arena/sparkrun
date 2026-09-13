# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock
import sys

import pytest
import sparkrun.api as api
from sparkrun.plugins.sparkroute import jobs, workload_plugins as plugins
from sparkrun.plugins.sparkroute.protocol import Binding, ProtocolError, Request


def fixture(monkeypatch, *, owned=True, used=True):
    sctx = api.default_sctx()
    job = SimpleNamespace(
        cluster_id="job",
        metadata={
            "cluster": "lab",
            "recipe_fingerprint": "revision",
            "owner": "sparkroute" if owned else "user",
            "runtime_info": {"execution_strategy": "coldsnap", "capture_id": "capture"} if used else {},
        },
    )
    monkeypatch.setattr(api, "list_jobs", lambda **_: [job])
    monkeypatch.setattr(
        plugins, "plugin_availability", lambda: [{"name": "coldsnap", "installed": True, "enabled": True, "lifecycle": True}]
    )
    calls = []

    def control(action, selected, **kwargs):
        calls.append((action, selected.cluster_id))
        return {"state": "sleeping" if action == "sleep" else "running", "cluster_id": selected.cluster_id}

    monkeypatch.setitem(sys.modules, "sparkrun.plugins.coldsnap.api", SimpleNamespace(control_job=control))
    request = Request("test", "sleep", binding=Binding("recipe", "revision", ("lab",)), cluster_id="job")
    return sctx, job, request, calls


def test_installed_plugin_does_not_claim_usage_or_controls(monkeypatch):
    sctx, job, _, _ = fixture(monkeypatch, used=False)
    result = plugins.describe_job(job, sctx=sctx)
    assert result["plugins_in_use"] == [] and result["lifecycle_actions"] == []


def test_sleep_requires_ownership_and_exact_binding(monkeypatch):
    sctx, job, request, calls = fixture(monkeypatch, owned=False)
    with pytest.raises(ProtocolError, match="externally owned"):
        plugins.control_workload(request, sctx=sctx)
    job.metadata["owner"] = "sparkroute"
    with pytest.raises(ProtocolError, match="binding and cluster"):
        plugins.control_workload(replace(request, binding=Binding("recipe", "revision", ("other",))), sctx=sctx)
    assert calls == []


def test_sleep_wake_state_survives_bridge_processes(monkeypatch):
    sctx, job, request, calls = fixture(monkeypatch)
    assert plugins.control_workload(request, sctx=sctx)["lifecycle_state"] == "sleeping"
    assert plugins.describe_job(job, sctx=sctx)["lifecycle_state"] == "sleeping"
    assert plugins.control_workload(replace(request, operation="wake"), sctx=sctx)["lifecycle_state"] == "running"
    assert calls == [("sleep", "job"), ("wake", "job")]


def test_uncertain_sleep_is_not_readvertised_as_running(monkeypatch):
    sctx, job, request, _ = fixture(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "sparkrun.plugins.coldsnap.api", SimpleNamespace(control_job=mock.Mock(side_effect=RuntimeError("lost reply")))
    )
    with pytest.raises(ProtocolError):
        plugins.control_workload(request, sctx=sctx)
    assert plugins.describe_job(job, sctx=sctx)["lifecycle_state"] == "sleeping_pending"


def test_durable_sleep_and_activation_cannot_run_concurrently(monkeypatch):
    sctx, _, request, _ = fixture(monkeypatch)
    with (
        mock.patch.object(jobs.subprocess, "Popen", return_value=SimpleNamespace(pid=12345)),
        mock.patch.object(jobs, "_alive", return_value=True),
    ):
        jobs.start_operation(request, sctx=sctx)
        with pytest.raises(ProtocolError, match="already running"):
            jobs.start_operation(replace(request, operation="ensure_ready", cluster_id=""), sctx=sctx)
