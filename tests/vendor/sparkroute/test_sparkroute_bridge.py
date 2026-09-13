# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Tests for the hidden ``sparkrun gateway-bridge`` command.

The bridge is a one-request JSON stdio protocol the gateway drives as a child
process. Two properties dominate:

* **the response key sets are a hard wire contract** — the gateway decodes
  results with Go's ``DisallowUnknownFields`` (``pkg/sparkrun/bridge.go``), so
  an extra field fails the whole decode rather than being ignored;
* **ownership is not identity** — a recipe fingerprint says "same serve
  configuration", so adoption may be fingerprint-based while teardown must not.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.plugins.sparkroute import bridge, operations
from sparkrun.plugins.sparkroute.protocol import Binding, ProtocolError, Request, parse_request, read_request


@pytest.fixture(autouse=True)
def _gateway_feature_enabled(monkeypatch):
    """The integration ships off; most tests here exercise it turned on.

    The gating test re-sets this to ``0`` via the same ``monkeypatch``
    instance, so its own value wins.

    Also clears ``PluggableGroup``'s per-instance attach guard. The flag gates
    plugin *loading*, and the CLI loads plugins once per process on first
    command resolution — so an earlier test that resolved ``main`` while the
    flag was off would otherwise leave the command permanently unattached. A
    real CLI invocation is one process, so this is a test-suite concern only.
    """
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "1")
    monkeypatch.setattr(main, "_cli_ext_loaded", False, raising=False)


def _request(operation: str = "capabilities", **extra):
    value = {"schema_version": 4, "request_id": "request-1", "operation": operation}
    value.update(extra)
    return value


def _run_worker_activation(request):
    sctx = operations.api.default_sctx()
    recipe, fingerprint = operations._resolve_binding(request.binding, sctx)
    return operations._ensure_ready(request, request.binding, recipe, fingerprint, sctx)


def _binding():
    return {
        "recipe": "@local/qwen",
        "recipe_revision": "abc123abc123",
        "cluster_candidates": ["spark-a"],
        "overrides": {"tensor_parallel": "2"},
    }


def _job(cluster_id: str, *, owner: str | None = None, fingerprint: str = "abc123abc123", recipe_state: dict | None = None):
    metadata = {"recipe_fingerprint": fingerprint}
    if owner is not None:
        metadata["owner"] = owner
    if recipe_state is not None:
        metadata["recipe_state"] = recipe_state
    return SimpleNamespace(cluster_id=cluster_id, metadata=metadata)


def _recipe_state() -> dict:
    """Serialized recipe state as a launch persists it into job metadata."""
    from sparkrun.core.recipe import Recipe

    return Recipe(
        {
            "model": "Qwen/Qwen3-32B",
            "runtime": "vllm",
            "container": "vllm/vllm-openai:latest",
            "defaults": {"max_model_len": 65536},
            "metadata": {"model_params": 32_000_000_000},
        }
    ).__getstate__()


def _endpoint(cluster_id: str):
    return SimpleNamespace(
        cluster_id=cluster_id,
        actual_models=["served"],
        served_model_name="served",
        model="model",
        host="10.0.0.2",
        port=8000,
        recipe_name="@local/qwen",
        runtime="vllm",
    )


# ---------------------------------------------------------------------------
# Command wiring
# ---------------------------------------------------------------------------


def test_hidden_command_is_not_in_help_but_is_invokable():
    runner = CliRunner()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    assert "gateway-bridge" not in help_result.output

    result = runner.invoke(main, ["gateway-bridge"], input=json.dumps(_request()))
    assert result.exit_code == 0
    response = json.loads(result.output)
    assert response["ok"] is True
    assert response["result"]["protocol_version"] == 4
    assert "ensure_ready" in response["result"]["operations"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_protocol_rejects_unknown_and_unbounded_input(monkeypatch):
    monkeypatch.setattr("sparkrun.plugins.sparkroute.protocol.MAX_REQUEST_BYTES", 8)
    with pytest.raises(ProtocolError, match="size limit") as exc_info:
        read_request(io.BytesIO(b"{}" * 5))
    assert exc_info.value.code == "request_too_large"

    with pytest.raises(ProtocolError) as exc_info:
        parse_request(_request("resolve", binding=_binding(), unexpected=True))
    assert exc_info.value.code == "invalid_request"


def test_stdio_returns_structured_operation_error():
    request = json.dumps(_request("resolve", binding=_binding())).encode()
    output = io.StringIO()
    with mock.patch.object(bridge, "execute", side_effect=ProtocolError("recipe_not_found", "not found")):
        exit_code = bridge.run_stdio(io.BytesIO(request), output)
    assert exit_code == 0
    assert json.loads(output.getvalue()) == {
        "schema_version": 4,
        "request_id": "request-1",
        "ok": False,
        "error": {"code": "recipe_not_found", "message": "not found", "retryable": False},
    }


def test_cluster_id_must_be_a_safe_identifier():
    with pytest.raises(ProtocolError) as exc_info:
        parse_request(_request("status", binding=_binding(), cluster_id="cluster id; rm -rf /"))
    assert exc_info.value.code == "invalid_request"


def test_wait_is_rejected_outside_ensure_ready():
    assert parse_request(_request("ensure_ready", binding=_binding(), wait=False)).wait is False
    with pytest.raises(ProtocolError) as exc_info:
        parse_request(_request("status", binding=_binding(), wait=False))
    assert exc_info.value.code == "invalid_request"


def test_an_unsupported_version_is_refused_at_the_version_it_asked_for():
    """Echoing the requested version is what lets a strict client's correlation
    check pass, so it reads the refusal and can downgrade."""
    with pytest.raises(ProtocolError) as exc_info:
        parse_request({"schema_version": 99, "request_id": "r", "operation": "capabilities"})
    assert exc_info.value.code == "unsupported_version"
    assert exc_info.value.schema_version == 99

    output = io.StringIO()
    bridge.run_stdio(io.BytesIO(json.dumps({"schema_version": 99, "request_id": "r", "operation": "capabilities"}).encode()), output)
    assert json.loads(output.getvalue())["schema_version"] == 99


def test_version_is_checked_before_unknown_fields():
    """A newer caller should be told the version is unsupported, not pointed at
    its own fields."""
    with pytest.raises(ProtocolError) as exc_info:
        parse_request({"schema_version": 99, "request_id": "r", "operation": "capabilities", "future_field": 1})
    assert exc_info.value.code == "unsupported_version"


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def test_mutating_operations_are_gated_but_introspection_is_not(monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "0")

    request = Request(request_id="request-1", operation="ensure_ready", binding=Binding(recipe="@local/qwen"))
    with mock.patch.object(operations.api, "run") as run:
        with pytest.raises(ProtocolError) as exc_info:
            operations.execute(request)
    assert exc_info.value.code == "feature_disabled"
    assert "setup features enable gateway.sparkroute" in exc_info.value.message
    run.assert_not_called()

    with (
        mock.patch.object(operations.api, "default_sctx", return_value=object()),
        mock.patch.object(operations, "_discover", return_value=[]),
    ):
        assert operations.execute(Request(request_id="r", operation="discover")) == {"endpoints": []}


# ---------------------------------------------------------------------------
# Discovery and ownership
# ---------------------------------------------------------------------------


def test_discovery_never_exports_api_key():
    job = SimpleNamespace(cluster_id="c", metadata={"recipe_fingerprint": "abc123abc123", "api_key": "secret"})
    endpoint = _endpoint("c")
    endpoint.api_key = "secret"
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[job]),
        mock.patch.object(operations, "discover_endpoints", return_value=[endpoint]),
    ):
        result = operations._discover(object(), fingerprint="abc123abc123")
    assert result[0]["served_models"] == ["served"]
    assert "secret" not in json.dumps(result)


def test_discovery_requires_live_served_model_ids():
    endpoint = _endpoint("cluster")
    endpoint.actual_models = []
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[_job("cluster")]),
        mock.patch.object(operations, "discover_endpoints", return_value=[endpoint]),
    ):
        assert operations._discover(object(), fingerprint="abc123abc123") == []


def test_discovery_scopes_the_health_probe_to_matching_jobs():
    jobs = [_job("owned", owner=operations.GATEWAY_OWNER), _job("other", fingerprint="ffffffffffff")]
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=jobs),
        mock.patch.object(operations, "discover_endpoints", return_value=[_endpoint("owned")]) as discover,
    ):
        result = operations._discover(object(), fingerprint="abc123abc123")
    # The non-matching job must never reach the probe, not merely be dropped
    # from the results afterwards.
    assert discover.call_args.kwargs["cluster_ids"] == {"owned"}
    assert [e["cluster_id"] for e in result] == ["owned"]
    assert result[0][operations._OWNED_KEY] is True


def test_stop_refuses_a_workload_the_bridge_did_not_launch():
    with mock.patch.object(operations.api, "list_jobs", return_value=[_job("human")]):
        with pytest.raises(ProtocolError) as exc_info:
            operations._stop("abc123abc123", "human", object())
    assert exc_info.value.code == "job_not_owned"


def test_stop_without_a_cluster_id_only_stops_owned_workloads():
    jobs = [_job("human"), _job("mine", owner=operations.GATEWAY_OWNER)]
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=jobs),
        mock.patch.object(operations.api, "stop", return_value=SimpleNamespace(success=True)) as stop,
    ):
        result = operations._stop("abc123abc123", "", object())
    assert result["cluster_ids"] == ["mine"]
    assert [c.kwargs["cluster_id"] for c in stop.call_args_list] == ["mine"]


# ---------------------------------------------------------------------------
# ensure_ready
# ---------------------------------------------------------------------------


def test_global_discovery_does_not_resolve_a_binding():
    with (
        mock.patch.object(operations.api, "default_sctx", return_value=object()),
        mock.patch.object(operations, "_discover", return_value=[]) as discover,
        mock.patch.object(operations, "_resolve_binding") as resolve,
    ):
        result = operations.execute(Request(request_id="r", operation="discover"))
    assert result == {"endpoints": []}
    discover.assert_called_once()
    resolve.assert_not_called()


def test_ensure_ready_adopts_matching_endpoint_without_launch():
    request = Request(request_id="r", operation="ensure_ready", binding=Binding(recipe="@local/qwen"))
    endpoint = {"cluster_id": "cluster"}
    with (
        mock.patch.object(operations, "_resolve_binding", return_value=(object(), "abc123abc123")),
        mock.patch.object(operations.api, "default_sctx", return_value=object()),
        mock.patch.object(operations, "_discover", return_value=[endpoint]),
        mock.patch.object(operations.api, "run") as run,
    ):
        result = _run_worker_activation(request)
    assert result == {"state": "ready", "endpoint": endpoint, "adopted": True}
    run.assert_not_called()


def test_ensure_ready_prefers_adopting_a_workload_it_owns():
    request = Request(request_id="r", operation="ensure_ready", binding=Binding(recipe="@local/qwen"))
    foreign = {"cluster_id": "human", operations._OWNED_KEY: False}
    mine = {"cluster_id": "mine", operations._OWNED_KEY: True}
    with (
        mock.patch.object(operations, "_resolve_binding", return_value=(object(), "abc123abc123")),
        mock.patch.object(operations.api, "default_sctx", return_value=object()),
        mock.patch.object(operations, "_discover", return_value=[foreign, mine]),
        mock.patch.object(operations.api, "run") as run,
    ):
        result = _run_worker_activation(request)
    assert result == {"state": "ready", "endpoint": {"cluster_id": "mine"}, "adopted": True}
    run.assert_not_called()


def test_ensure_ready_uses_shared_plan_run_and_readiness():
    request = Request(
        request_id="r", operation="ensure_ready", binding=Binding(recipe="@local/qwen", cluster_candidates=("spark-a",)), timeout_seconds=30
    )
    recipe = SimpleNamespace(_sparkroute_launch_overrides={"tensor_parallel": 2}, post_exec=[], post_commands=[], defaults={"port": 8000})
    plan = SimpleNamespace(cluster_id="new", cluster=SimpleNamespace(name="spark-a"), host_list=["host"], recipe=recipe, is_solo=True)
    run_result = SimpleNamespace(
        rc=0, cluster_id="new", recipe_fingerprint="abc123abc123", launch_result=object(), host_list=["host"], serve_port=8001, is_solo=True
    )
    endpoint = {"cluster_id": "new", "host": "127.0.0.1", "port": 8001, operations._OWNED_KEY: True}
    with (
        mock.patch.object(operations, "_resolve_binding", return_value=(recipe, "abc123abc123")),
        mock.patch.object(operations.api, "default_sctx", return_value=SimpleNamespace(config=object())),
        mock.patch.object(operations, "_discover", side_effect=[[], [endpoint]]) as discover,
        mock.patch.object(operations.api, "plan", return_value=plan) as planner,
        mock.patch.object(operations.api, "run", return_value=run_result) as run,
        mock.patch("sparkrun.core.launcher.wait_for_serve_ready", return_value=SimpleNamespace(ready=True)) as readiness,
        mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
    ):
        result = _run_worker_activation(request)
    options = planner.call_args.args[0]
    assert options.owner == operations.GATEWAY_OWNER
    assert options.cluster == "spark-a" and options.auto_port and options.trust is False
    assert options.overrides == {"tensor_parallel": 2}
    assert run.call_args.kwargs["plan"] is plan
    assert readiness.call_args.args[0] is run_result.launch_result
    assert "port_timeout_s" not in readiness.call_args.kwargs  # preserve configured readiness policy
    assert discover.call_args.kwargs == {"fingerprint": "abc123abc123", "cluster_id": "new", "cluster_candidates": ("spark-a",)}
    assert result["endpoint"]["port"] == 8001 and result["adopted"] is False


def test_ensure_ready_background_request_returns_operation_without_launching():
    request = Request(request_id="r", operation="ensure_ready", binding=Binding(recipe="@local/qwen"), wait=False)
    with (
        mock.patch.object(operations, "_resolve_binding", return_value=(object(), "abc123abc123")),
        mock.patch.object(operations.api, "default_sctx", return_value=object()),
        mock.patch("sparkrun.plugins.sparkroute.jobs.start_operation", return_value={"state": "running", "operation_id": "123"}) as start,
        mock.patch.object(operations.api, "run") as run,
    ):
        assert operations.execute(request) == {"state": "running", "operation_id": "123"}
    start.assert_called_once()
    run.assert_not_called()


# ---------------------------------------------------------------------------
# Wire contract
# ---------------------------------------------------------------------------
#
# The gateway decodes every bridge result with Go's DisallowUnknownFields (see
# pkg/sparkrun/bridge.go:decodeStrict), so an *extra* field is not ignored — it
# fails the whole decode and the operation errors. These key sets are therefore
# a hard contract, and adding to one means adding to the Go struct first.


def test_endpoint_projection_matches_the_gateway_struct():
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[_job("cluster", owner=operations.GATEWAY_OWNER)]),
        mock.patch.object(operations, "discover_endpoints", return_value=[_endpoint("cluster")]),
    ):
        internal = operations._discover(object(), fingerprint="abc123abc123")

    # sparkrun.Endpoint in pkg/sparkrun/bridge.go
    assert set(operations._project(internal[0])) == {
        "state",
        "cluster_id",
        "job_id",
        "host",
        "port",
        "protocol",
        "served_models",
        "recipe",
        "recipe_revision",
        "runtime",
        "owned",
        "plugins_in_use",
        "lifecycle_actions",
    }
    # Ownership is tracked internally and must not reach the wire.
    assert operations._OWNED_KEY in internal[0]


def test_endpoint_projection_carries_optional_model_metadata():
    """``model_metadata`` is the one optional key on the Go ``Endpoint``, and
    its own value keys are ``sparkrun.ModelMetadata`` — also decoded strictly."""
    job = _job("cluster", owner=operations.GATEWAY_OWNER, recipe_state=_recipe_state())
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[job]),
        mock.patch.object(operations, "discover_endpoints", return_value=[_endpoint("cluster")]),
    ):
        projected = operations._project(operations._discover(object(), fingerprint="abc123abc123")[0])

    assert set(projected) == {
        "state",
        "cluster_id",
        "job_id",
        "host",
        "port",
        "protocol",
        "served_models",
        "recipe",
        "recipe_revision",
        "runtime",
        "owned",
        "plugins_in_use",
        "lifecycle_actions",
        "model_metadata",
    }
    # The key must be an exact served model identity, per the contract.
    assert set(projected["model_metadata"]) <= set(projected["served_models"])
    # sparkrun.ModelMetadata in pkg/sparkrun/bridge.go
    assert set(projected["model_metadata"]["served"]) <= {
        "size_b",
        "input_price",
        "output_price",
        "context",
        "tags",
    }


def test_endpoint_without_a_recipe_state_omits_model_metadata():
    """Older jobs carry no serialized recipe. The Go field is ``omitempty``, so
    omission is valid — but an empty object would claim "reported, all
    unknown" rather than "not reported"."""
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[_job("cluster")]),
        mock.patch.object(operations, "discover_endpoints", return_value=[_endpoint("cluster")]),
    ):
        projected = operations._project(operations._discover(object(), fingerprint="abc123abc123")[0])
    assert "model_metadata" not in projected


def test_unreadable_recipe_state_does_not_break_discovery():
    """Metadata is advisory: a malformed model card must not cost the endpoint."""
    job = _job("cluster", recipe_state={"not": "a recipe"})
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[job]),
        mock.patch.object(operations, "discover_endpoints", return_value=[_endpoint("cluster")]),
    ):
        endpoints = operations._discover(object(), fingerprint="abc123abc123")
    assert len(endpoints) == 1
    assert endpoints[0]["state"] == "ready"


def test_ensure_ready_result_matches_the_gateway_struct():
    request = Request(request_id="r", operation="ensure_ready", binding=Binding(recipe="@local/qwen"))
    with (
        mock.patch.object(operations, "_resolve_binding", return_value=(object(), "abc123abc123")),
        mock.patch.object(operations.api, "default_sctx", return_value=object()),
        mock.patch.object(operations, "_discover", return_value=[{"cluster_id": "c", operations._OWNED_KEY: True}]),
    ):
        result = _run_worker_activation(request)
    # sparkrun.EnsureResult in pkg/sparkrun/bridge.go
    assert set(result) == {"state", "endpoint", "adopted"}
    assert operations._OWNED_KEY not in result["endpoint"]


def test_capabilities_result_matches_the_gateway_struct():
    result = operations.execute(Request(request_id="r", operation="capabilities"))
    # sparkrun.Capabilities in pkg/sparkrun/bridge.go
    assert set(result) == {"protocol_version", "operations", "sparkrun_version"}


def test_stop_result_matches_the_gateway_struct():
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[_job("mine", owner=operations.GATEWAY_OWNER)]),
        mock.patch.object(operations.api, "stop", return_value=SimpleNamespace(success=True)),
    ):
        result = operations._stop("abc123abc123", "", object())
    # sparkrun.StopResult in pkg/sparkrun/bridge.go
    assert set(result) == {"state", "cluster_ids"}


@pytest.mark.parametrize("operation", ["discover", "status", "ensure_ready"])
def test_cluster_name_is_exposed_for_endpoint_operations(operation):
    job = _job("opaque-job")
    job.metadata["cluster"] = "spark-a"
    binding = Binding(recipe="@local/qwen", recipe_revision="abc123abc123")
    with (
        mock.patch.object(operations.api, "default_sctx", return_value=object()),
        mock.patch.object(operations.api, "list_jobs", return_value=[job]),
        mock.patch.object(operations, "discover_endpoints", return_value=[_endpoint("opaque-job")]),
        mock.patch.object(operations, "_resolve_binding", return_value=(object(), "abc123abc123")),
    ):
        request = Request(request_id="r", operation=operation, binding=binding)
        result = _run_worker_activation(request) if operation == "ensure_ready" else operations.execute(request)
    endpoint = result["endpoints"][0] if operation == "discover" else result["endpoint"]
    assert endpoint["cluster_id"] == endpoint["job_id"] == "opaque-job"
    assert endpoint["cluster_name"] == "spark-a"
    assert "_owned" not in endpoint


@pytest.mark.parametrize("cluster", [None, "", "bad\nlabel", "x" * 1025])
def test_missing_or_invalid_cluster_names_do_not_block_discovery(cluster):
    job = _job("opaque-job")
    job.metadata["cluster"] = cluster
    with (
        mock.patch.object(operations.api, "list_jobs", return_value=[job]),
        mock.patch.object(operations, "discover_endpoints", return_value=[_endpoint("opaque-job")]),
    ):
        endpoint = operations._project(operations._discover(object())[0])
    assert "cluster_name" not in endpoint
    assert endpoint["cluster_id"] == "opaque-job"


@pytest.mark.parametrize("version", [1, 2, 99])
def test_parse_errors_preserve_recoverable_request_id_and_version(version):
    request = _request("resolve", schema_version=version, binding={})
    output = io.StringIO()
    bridge.run_stdio(io.BytesIO(json.dumps(request).encode()), output)
    error = json.loads(output.getvalue())
    assert error["ok"] is False
    assert error["request_id"] == "request-1"
    assert error["schema_version"] == version
