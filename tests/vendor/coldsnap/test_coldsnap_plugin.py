# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.execution import ExecutionContext, resolve_recipe_execution, run_preparation_steps
from sparkrun.core.progress import PROGRESS
from sparkrun.core.recipe import Recipe
from sparkrun.core.scheduler import RankAssignment, RankSlot
from sparkrun.core.timing import Timeline
from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint
from sparkrun.plugins.coldsnap import register
from sparkrun.plugins.coldsnap.artifacts import (
    DEFAULT_ARTIFACT_GENERATIONS,
    promote_generation,
    resolve_artifact_store,
    resolve_generation_limit,
)
from sparkrun.plugins.coldsnap.cli import _resolve_materialization_policy, _timing_tree_depth, build_command
from sparkrun.plugins.coldsnap.config import ColdSnapRecipe
from sparkrun.plugins.coldsnap.policy import resolve_coldsnap_policy
from sparkrun.plugins.coldsnap.providers import (
    _download_worker_pack,
    _read_native_pack_status,
    _resolve_capture_local_pack,
    _resolve_node_cached_pack,
    _stage_payload_verifier,
    resolve_request_weight_mode,
    stage_native_packs,
)
from sparkrun.plugins.coldsnap.request import _coldsnap_command, build_request
from sparkrun.plugins.coldsnap.runtime_cache import (
    CaptureRuntimeCacheStage,
    _capture_staging_root,
    cleanup_capture_runtime_cache,
    stage_capture_runtime_cache,
)
from sparkrun.plugins.coldsnap.service import ColdSnapService, RestoreDescriptor, prepare_capture_images
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.transports.session import HostCommandResult


def _setup():
    register(None)
    digest = "a" * 64
    recipe = Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "Qwen/Qwen3.5-0.8B",
            "model_revision": "model-commit",
            "runtime": "vllm-distributed",
            "container": "org/capsule@sha256:%s" % digest,
            "defaults": {"tensor_parallel": 2},
            "coldsnap": {
                "format": 1,
                "capsule": {
                    "repository": "registry.example/coldsnap/qwen",
                },
                "weights": {
                    "mode": "auto",
                    "recovery": {"loader_backend": "torch"},
                    "native": {
                        "repository": "org/qwen-native",
                        "revision": "native-commit",
                    },
                },
            },
        }
    )
    cluster = ClusterDefinition(
        name="c",
        hosts=["h1", "h2"],
        cache_dir="/cache/hf",
        sparkrun_cache_dir="/cache/sparkrun",
    )
    placement = RankAssignment(
        by_rank=(RankSlot("h1", 0), RankSlot("h2", 0)),
        hosts_used=("h1", "h2"),
    )
    plan = api.RunPlan(
        recipe=recipe,
        runtime=VllmDistributedRuntime(),
        cluster=cluster,
        candidate_hosts=("h1", "h2"),
        host_list=("h1", "h2"),
        is_solo=False,
        placement=placement,
        intent_id="b" * 16,
        placement_token="c" * 12,
        cluster_id="sparkrun_%s_%s" % ("b" * 16, "c" * 12),
    )
    options = api.RunOptions(recipe=recipe, hosts=("h1", "h2"), init_port=29731)
    sctx = SimpleNamespace(
        config=SimpleNamespace(
            cache_dir="/cache/sparkrun",
            hf_cache_dir="/fallback",
            ssh_user=None,
            ssh_key=None,
            ssh_options=None,
        )
    )
    return recipe, options, plan, sctx


def _write_pending_capture(store, capture_id, *, state="committed"):
    output = store.capture_output(capture_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": 8,
        "kind": "coldsnap-snapshot-artifact",
        "state": state,
        "capture_id": capture_id,
    }
    output.write_text(json.dumps(artifact), encoding="utf-8")
    return output, artifact


def _artifact_launch(hosts=("h1", "h2")):
    return {
        "units": [{"id": "unit-%d" % index, "host": host} for index, host in enumerate(hosts)],
        "execution": {"workers": [{"id": "worker-%d" % index, "unit": "unit-%d" % index} for index in range(len(hosts))]},
    }


def _artifact_packs():
    return [
        {
            "role": "model-weight-payload",
            "owner": "worker/worker-%d" % index,
            "path": "model-payloads/sha256/%s.pack" % (str(index) * 64),
            "bytes": 100 + index,
            "sha256": "sha256:" + str(index) * 64,
        }
        for index in range(2)
    ]


def _validated_pack_record(worker: str, path: str, size: int, digest: str) -> dict:
    return {
        "format": 1,
        "kind": "coldsnap-payload-validation-result",
        "decision": "accept",
        "worker": worker,
        "path": path,
        "bytes": size,
        "sha256": digest,
        "validation": {
            "record": path + ".coldsnap-validation.json",
            "provider": "sha256-cache-v1",
            "content_evidence": "cached-full-sha256",
            "device": 1,
            "inode": 1,
            "size": size,
            "mtime_ns": 1,
            "bytes_hashed": 0,
        },
    }


def _test_payload_verifier(tmp_path: Path) -> Path:
    """Provide the adapter's JSON contract without reimplementing its trust policy."""

    verifier = tmp_path / "coldsnap-payload-verifier"
    verifier.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("operation")
parser.add_argument("--path", required=True)
parser.add_argument("--record", required=True)
parser.add_argument("--expected-sha256", required=True)
parser.add_argument("--expected-bytes", required=True, type=int)
parser.add_argument("--worker", required=True)
args = parser.parse_args()
if args.operation != "payload-verify":
    raise SystemExit("unexpected operation")
value = os.stat(args.path, follow_symlinks=False)
print(json.dumps({
    "format": 1,
    "kind": "coldsnap-payload-validation-result",
    "decision": "accept",
    "worker": args.worker,
    "path": args.path,
    "bytes": args.expected_bytes,
    "sha256": args.expected_sha256,
    "validation": {
        "record": args.record,
        "provider": "sha256-cache-v1",
        "content_evidence": "full-sha256-this-operation",
        "reason": "test-contract",
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "bytes_hashed": value.st_size,
    },
}, sort_keys=True))
""",
        encoding="utf-8",
    )
    verifier.chmod(0o755)
    return verifier


class _VerifierStageSession:
    def __init__(self, *, fail_first_install=False, fail_all_installs=False, identity=None):
        self.files: dict[tuple[str, str], bytes] = {}
        self.closed = False
        self.lock = threading.Lock()
        self.fail_first_install = fail_first_install
        self.fail_all_installs = fail_all_installs
        self.install_attempts: dict[str, int] = {}
        self.identity = identity
        self.identity_checks = []

    def execute(self, host, arguments, **_kwargs):
        with self.lock:
            if arguments[1:] == ["version", "--json"]:
                self.identity_checks.append(host)
                return HostCommandResult(host, 0, stdout=json.dumps(self.identity).encode())
            if arguments[0] == "install":
                self.install_attempts[host] = self.install_attempts.get(host, 0) + 1
                if self.fail_all_installs or (self.fail_first_install and self.install_attempts[host] == 1):
                    return HostCommandResult(
                        host,
                        1,
                        stderr=(
                            b"identity_sign: private key /home/u/.ssh/id_ed25519 contents do not match public\ninstall: Permission denied\n"
                        ),
                    )
                return HostCommandResult(host, 0)
            if arguments[0] == "chmod":
                return HostCommandResult(host, 0)
            if arguments[0] == "sha256sum":
                payload = self.files.get((host, arguments[1]))
                if payload is None:
                    return HostCommandResult(host, 1, stderr=b"missing")
                digest = hashlib.sha256(payload).hexdigest().encode()
                return HostCommandResult(host, 0, stdout=digest + b"  " + arguments[1].encode() + b"\n")
            if arguments[:2] == ["mv", "-f"]:
                self.files[(host, arguments[3])] = self.files.pop((host, arguments[2]))
                return HostCommandResult(host, 0)
            raise AssertionError(arguments)

    def upload(self, host, sources, destination, **_kwargs):
        with self.lock:
            self.files[(host, destination)] = Path(sources[0]).read_bytes()

    def close(self):
        self.closed = True


def test_payload_verifier_is_staged_once_per_data_host(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    request = build_request("restore", options, plan=plan, sctx=sctx)
    verifier = tmp_path / "adapter"
    verifier.write_bytes(b"release-matched-adapter")
    verifier.chmod(0o755)
    session = _VerifierStageSession()
    monkeypatch.setattr("sparkrun.transports.open_cluster_host_session", lambda *_args, **_kwargs: session)

    target = _stage_payload_verifier(
        verifier,
        request=request,
        plan=plan,
        state_root="/cache/coldsnap",
        ssh_kwargs={},
    )

    assert target.endswith("/coldsnap-payload-verifier")
    assert {host for host, path in session.files if path == target} == {"h1", "h2"}
    assert {payload for (host, path), payload in session.files.items() if path == target} == {b"release-matched-adapter"}
    assert session.closed


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("matching", [False, True])
def test_payload_verifier_checks_remote_release_identity_even_on_cache_hit(tmp_path, monkeypatch, warm, matching):
    _recipe, options, plan, sctx = _setup()
    request = build_request("restore", options, plan=plan, sctx=sctx)
    verifier = tmp_path / "adapter"
    verifier.write_bytes(b"release-matched-adapter")
    verifier.chmod(0o755)
    digest = hashlib.sha256(verifier.read_bytes()).hexdigest()
    expected = {"version": "0.3.21", "commit": "a" * 40}
    (tmp_path / "manifest.json").write_text(json.dumps({**expected, "sha256": {"adapter": digest}}))
    session = _VerifierStageSession(identity=expected if matching else {**expected, "commit": "b" * 40})
    target = "/cache/coldsnap/tools/payload-verifier/%s/coldsnap-payload-verifier" % digest
    if warm:
        for host in plan.host_list:
            session.files[(host, target)] = verifier.read_bytes()
    monkeypatch.setattr("sparkrun.transports.open_cluster_host_session", lambda *_args, **_kwargs: session)

    def stage():
        return _stage_payload_verifier(verifier, request=request, plan=plan, state_root="/cache/coldsnap", ssh_kwargs={})

    if matching:
        assert stage() == target
        assert sorted(session.identity_checks) == sorted(plan.host_list)
    else:
        with pytest.raises(RuntimeError, match="release identity mismatch"):
            stage()
    assert session.closed


def test_payload_verifier_repairs_root_owned_cache_and_retries(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    request = build_request("restore", options, plan=plan, sctx=sctx)
    verifier = tmp_path / "adapter"
    verifier.write_bytes(b"release-matched-adapter")
    verifier.chmod(0o755)
    session = _VerifierStageSession(fail_first_install=True)
    repairs = []
    monkeypatch.setattr("sparkrun.transports.open_cluster_host_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr(
        "sparkrun.orchestration.sudo.ensure_remote_dir_ownership",
        lambda path, hosts, **kwargs: repairs.append((path, tuple(hosts), kwargs["resource_label"], kwargs["session"] is session)) or [],
    )

    _stage_payload_verifier(
        verifier,
        request=request,
        plan=plan,
        state_root="/cache/sparkrun/coldsnap",
        ownership_root="/cache/sparkrun",
        ssh_kwargs={},
    )

    assert sorted(repairs) == [
        ("/cache/sparkrun", ("h1",), "ColdSnap cache", True),
        ("/cache/sparkrun", ("h2",), "ColdSnap cache", True),
    ]
    assert session.install_attempts == {"h1": 2, "h2": 2}


def test_payload_verifier_reports_permission_fix_separately_from_ssh_warning(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    request = build_request("restore", options, plan=plan, sctx=sctx)
    verifier = tmp_path / "adapter"
    verifier.write_bytes(b"release-matched-adapter")
    verifier.chmod(0o755)
    session = _VerifierStageSession(fail_all_installs=True)
    monkeypatch.setattr("sparkrun.transports.open_cluster_host_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr(
        "sparkrun.orchestration.sudo.ensure_remote_dir_ownership",
        lambda *_args, **_kwargs: ["h1"],
    )

    with pytest.raises(RuntimeError) as error:
        _stage_payload_verifier(
            verifier,
            request=request,
            plan=plan,
            state_root="/home/u/.cache/sparkrun/coldsnap",
            ownership_root="/home/u/.cache/sparkrun",
            ssh_kwargs={},
        )

    message = str(error.value)
    assert "sparkrun setup fix-permissions --cluster c --cache-dir /home/u/.cache/sparkrun" in message
    assert "install: Permission denied" in message
    assert "SSH reports that the local id_ed25519.pub does not match" in message
    assert "identity_sign:" not in message


def _worker_index(worker):
    return int(str(worker).removeprefix("worker-"))


def test_coldsnap_owns_top_level_recipe_item():
    recipe, *_ = _setup()
    assert isinstance(recipe.plugin_item("coldsnap"), ColdSnapRecipe)
    assert "coldsnap" not in recipe.runtime_config
    assert "artifact" not in recipe.to_dict()["coldsnap"]
    assert recipe.to_dict()["coldsnap"]["weights"]["mode"] == "auto"
    assert recipe.to_dict()["coldsnap"]["weights"]["recovery"]["loader_backend"] == "torch"


def test_coldsnap_artifact_reference_is_typed_and_exported_only_when_explicit():
    base, *_ = _setup()
    document = base.to_dict()
    reference = "oci://registry.example/coldsnap/qwen@sha256:" + "d" * 64
    document["coldsnap"]["artifact"] = {"reference": reference}
    recipe = Recipe.from_dict(document)
    assert recipe.plugin_item("coldsnap").artifact.reference == reference
    assert recipe.to_dict()["coldsnap"]["artifact"] == {"reference": reference}

    document["coldsnap"]["artifact"] = {"reference": "./artifact.json"}
    recipe = Recipe.from_dict(document)
    assert "coldsnap.artifact.reference is invalid" in recipe.validate()


def test_capture_output_option_is_an_optional_override():
    capture = build_command().commands["capture"]
    output = next(parameter for parameter in capture.params if parameter.name == "output")

    assert not output.required
    assert output.default == ""


@pytest.mark.parametrize("snapshot_driver", ["n580", "n610"])
@pytest.mark.parametrize("operation", ["restore", "sleep", "wake", "status", "publish", "publish-native"])
def test_request_retains_captured_legacy_thread_default(tmp_path, monkeypatch, snapshot_driver, operation):
    _recipe, options, plan, sctx = _setup()
    # Model a host that no longer injects the historical CPU-thread default.
    monkeypatch.setattr(plan.runtime, "get_cluster_env", lambda **_kwargs: {})
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "launch": {
                    "engine": "vllm",
                    "units": [
                        {"id": "unit-1", "host": "old-h2", "environment": {"OMP_NUM_THREADS": "4"}},
                        {"id": "unit-0", "host": "old-h1", "environment": {"OMP_NUM_THREADS": "4", "OLD_MODEL_SETTING": "1"}},
                    ],
                },
            }
        )
    )
    request = build_request(
        operation,
        options,
        plan=plan,
        sctx=sctx,
        artifact=str(artifact),
        snapshot_driver=snapshot_driver,
        native_repository="org/native",
        native_revision="commit",
    )
    assert [unit["environment"]["OMP_NUM_THREADS"] for unit in request["launch"]["units"]] == ["4", "4"]
    assert all("OLD_MODEL_SETTING" not in unit["environment"] for unit in request["launch"]["units"])


@pytest.mark.parametrize("explicit", ["4", "12", ""])
def test_request_preserves_explicit_thread_override(tmp_path, monkeypatch, explicit):
    recipe, options, plan, sctx = _setup()
    monkeypatch.setattr(plan.runtime, "get_cluster_env", lambda **_kwargs: {})
    recipe.env["OMP_NUM_THREADS"] = explicit
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "launch": {
                    "engine": "vllm",
                    "units": [
                        {"id": "unit-0", "environment": {"OMP_NUM_THREADS": "4"}},
                        {"id": "unit-1", "environment": {"OMP_NUM_THREADS": "4"}},
                    ],
                },
            }
        )
    )
    request = build_request("restore", options, plan=plan, sctx=sctx, artifact=str(artifact))
    assert [unit["environment"]["OMP_NUM_THREADS"] for unit in request["launch"]["units"]] == [explicit, explicit]


@pytest.mark.parametrize("captured_value", [None, "8"])
def test_request_does_not_invent_or_inherit_other_thread_defaults(tmp_path, monkeypatch, captured_value):
    _recipe, options, plan, sctx = _setup()
    monkeypatch.setattr(plan.runtime, "get_cluster_env", lambda **_kwargs: {})
    artifact = tmp_path / "artifact.json"
    environment = {} if captured_value is None else {"OMP_NUM_THREADS": captured_value}
    artifact.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "launch": {"engine": "vllm", "units": [{"id": "unit-0", "environment": environment}]},
            }
        )
    )
    request = build_request("restore", options, plan=plan, sctx=sctx, artifact=str(artifact))
    assert all("OMP_NUM_THREADS" not in unit["environment"] for unit in request["launch"]["units"])


def test_capture_does_not_read_old_thread_defaults(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    monkeypatch.setattr(plan.runtime, "get_cluster_env", lambda **_kwargs: {})
    artifact = tmp_path / "artifact.json"
    artifact.write_text("not a descriptor")
    request = build_request("capture", options, plan=plan, sctx=sctx, artifact=str(artifact))
    assert all("OMP_NUM_THREADS" not in unit["environment"] for unit in request["launch"]["units"])


def test_build_request_uses_resolved_unit_commands_without_profile_file():
    _recipe, options, plan, sctx = _setup()
    request = build_request(
        "restore",
        options,
        plan=plan,
        sctx=sctx,
    )
    assert request["kind"] == "coldsnap-operation-request"
    assert request["format"] == 4
    assert request["snapshot_driver"] == {"id": "n610"}
    assert [unit["id"] for unit in request["launch"]["units"]] == ["unit-0", "unit-1"]
    assert [worker["id"] for worker in request["launch"]["execution"]["workers"]] == ["worker-0", "worker-1"]
    assert request["policy"]["weights"]["mode"] == "auto"
    assert request["policy"]["weights"]["recovery"] == {"loader_backend": "torch"}
    fingerprint = derive_recipe_fingerprint(plan.recipe, options.overrides)
    assert request["artifact"] == "/cache/sparkrun/coldsnap/artifacts/%s/%s/drivers/n610/current.json" % (
        plan.intent_id,
        fingerprint,
    )
    assert "validation" not in request
    assert "process" not in request["policy"]
    assert "files_by_worker" not in request["policy"]["weights"]["native"]
    assert request["policy"]["weights"]["native"]["materialize"] == "off"
    assert request["policy"]["capsule"] == {
        "repository": "registry.example/coldsnap/qwen",
    }
    assert request["policy"]["compatibility"] == {"enforce_captured_driver_floor": False}
    assert request["workload"] == {
        "cluster_id": plan.cluster_id,
        "intent_id": plan.intent_id,
        "recipe": plan.recipe.qualified_name,
        "runtime": "vllm-distributed",
        "model": "Qwen/Qwen3.5-0.8B",
        "served_model_name": "Qwen/Qwen3.5-0.8B",
        "log_path": "/tmp/sparkrun_serve.log",
    }
    assert request["policy"]["cache"]["paths"] == ["/var/cache/coldsnap/runtime"]
    assert "/cache/huggingface" not in request["policy"]["cache"]["paths"]
    assert request["launch"]["units"][1]["command"][:4] == [
        "bash",
        "--noprofile",
        "--norc",
        "-c",
    ]
    assert "--headless" in request["launch"]["units"][1]["command"][4]
    assert "--load-format coldsnap" not in request["launch"]["units"][1]["command"][4]
    assert "--enable-sleep-mode" in request["launch"]["units"][1]["command"][4]


def test_coldsnap_sglang_command_enables_live_memory_control_once():
    command = ["bash", "--noprofile", "--norc", "-c", "sglang serve --model-path org/model"]
    first = _coldsnap_command(command, "sglang")
    second = _coldsnap_command(first, "sglang")

    assert first[4].endswith("--enable-memory-saver")
    assert second[4].count("--enable-memory-saver") == 1
    assert "--enable-sleep-mode" not in second[4]


def test_sglang_native_materialization_capability_is_explicit():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["runtime"] = "sglang"
    recipe = Recipe.from_dict(document)
    options = replace(options, recipe=recipe)
    plan = replace(plan, recipe=recipe, runtime=SglangRuntime())

    request = build_request("restore", options, plan=plan, sctx=sctx)
    assert request["policy"]["weights"]["native"]["materialize"] == "off"
    with pytest.raises(ValueError, match="SGLang does not support.*required"):
        build_request(
            "restore",
            options,
            plan=plan,
            sctx=sctx,
            native_materialization="required",
        )


def test_build_request_preserves_explicit_coldsnap_runtime_overrides():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["coldsnap"]["process"] = {
        "backend": "cuda-criu",
        "kv_discard": False,
        "async_graphs": False,
    }
    document["coldsnap"]["validation"] = {
        "health_path": "/ready",
        "prompt": "Say ready",
        "expected": "ready",
    }
    recipe = Recipe.from_dict(document)
    options = replace(options, recipe=recipe)
    plan = replace(plan, recipe=recipe)

    request = build_request(
        "restore",
        options,
        plan=plan,
        sctx=sctx,
        native_materialization="off",
    )

    assert request["policy"]["process"] == {
        "backend": "cuda-criu",
        "kv_discard": False,
        "async_graphs": False,
    }
    assert request["policy"]["weights"]["native"]["materialize"] == "off"
    assert request["validation"] == {
        "health_path": "/ready",
        "prompt": "Say ready",
        "expected": "ready",
    }


def test_build_request_omits_undeclared_runtime_defaults_except_restore_generation_off():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["coldsnap"] = {
        "capsule": {"repository": "registry.example/coldsnap/qwen"},
    }
    recipe = Recipe.from_dict(document)
    options = replace(options, recipe=recipe)
    plan = replace(plan, recipe=recipe)

    request = build_request("restore", options, plan=plan, sctx=sctx)

    assert request["policy"] == {
        "weights": {"native": {"materialize": "off"}},
        "cache": {"seed": True, "paths": ["/var/cache/coldsnap/runtime"]},
        "capsule": {"repository": "registry.example/coldsnap/qwen"},
        "compatibility": {"enforce_captured_driver_floor": False},
    }
    assert "validation" not in request


def test_cluster_coldsnap_policy_controls_state_root_and_recovery_io():
    _recipe, options, plan, sctx = _setup()
    plan.cluster.sparkrun_cache_dir = "/mnt/sparkrun"
    plan.cluster.plugins = {
        "coldsnap": {
            "state_root": "/mnt/coldsnap",
            "io": {"recovery_read": "buffered"},
        }
    }

    policy = resolve_coldsnap_policy(
        cluster=plan.cluster,
        sctx=sctx,
        hosts=list(plan.host_list),
        probe_remote=False,
    )
    request = build_request("restore", options, plan=plan, sctx=sctx)

    assert policy.sparkrun_cache_dir == "/mnt/sparkrun"
    assert policy.state_root == "/mnt/coldsnap"
    assert policy.recovery_read == "buffered"
    assert request["policy"]["weights"]["recovery"]["loader_backend"] == "torch"

    document = plan.recipe.to_dict()
    del document["coldsnap"]["weights"]["recovery"]
    recipe = Recipe.from_dict(document)
    inherited = build_request(
        "restore",
        replace(options, recipe=recipe),
        plan=replace(plan, recipe=recipe),
        sctx=sctx,
    )
    assert inherited["policy"]["weights"]["recovery"]["loader_backend"] == "buffered"


def test_site_policy_coexists_with_other_user_coldsnap_settings():
    _recipe, _options, plan, sctx = _setup()
    sctx.config.plugins = {
        "coldsnap": {
            "artifact_generations": 5,
            "controller": {"version": "0.3.3"},
            "io": {"recovery_read": "buffered"},
        }
    }

    policy = resolve_coldsnap_policy(
        cluster=plan.cluster,
        sctx=sctx,
        hosts=list(plan.host_list),
        probe_remote=False,
    )

    assert policy.recovery_read == "buffered"
    assert policy.sources["io.recovery_read"] == "user-plugin"


def test_shape_calibration_policy_round_trips_and_reaches_engine_neutral_request():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["coldsnap"]["process"]["shape_calibration"] = "enabled"
    configured = Recipe.from_dict(document)
    request = build_request(
        "capture",
        replace(options, recipe=configured),
        plan=replace(plan, recipe=configured),
        sctx=sctx,
    )

    assert configured.to_dict()["coldsnap"]["process"]["shape_calibration"] == "enabled"
    assert request["policy"]["process"]["shape_calibration"] == "enabled"


def test_graph_policy_round_trips_and_reaches_engine_neutral_request():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["coldsnap"]["process"]["graph_policy"] = "preserve-nccl-exec"
    configured = Recipe.from_dict(document)
    request = build_request(
        "capture",
        replace(options, recipe=configured),
        plan=replace(plan, recipe=configured),
        sctx=sctx,
    )

    assert configured.to_dict()["coldsnap"]["process"]["graph_policy"] == "preserve-nccl-exec"
    assert request["policy"]["process"]["graph_policy"] == "preserve-nccl-exec"


def test_target_local_scope_is_emitted_only_for_n580_capture():
    _recipe, options, plan, sctx = _setup()

    request = build_request(
        "capture",
        options,
        plan=plan,
        sctx=sctx,
        snapshot_driver="n580",
        artifact_scope="target-local",
    )

    assert request["policy"]["process"]["artifact_scope"] == "target-local"
    with pytest.raises(ValueError, match="n580 capture"):
        build_request(
            "restore",
            options,
            plan=plan,
            sctx=sctx,
            snapshot_driver="n580",
            artifact_scope="target-local",
        )

    document = plan.recipe.to_dict()
    document["runtime"] = "sglang"
    recipe = Recipe.from_dict(document)
    sglang_options = replace(options, recipe=recipe)
    sglang_plan = replace(plan, recipe=recipe, runtime=SglangRuntime())
    with pytest.raises(ValueError, match="target-local.*requires vLLM"):
        build_request(
            "capture",
            sglang_options,
            plan=sglang_plan,
            sctx=sctx,
            snapshot_driver="n580",
            artifact_scope="target-local",
        )


def test_omitted_weight_mode_uses_selected_driver_default_for_asset_staging():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["coldsnap"] = {
        "capsule": {"repository": "registry.example/coldsnap/qwen"},
    }
    recipe = Recipe.from_dict(document)
    options = replace(options, recipe=recipe)
    plan = replace(plan, recipe=recipe)

    n580 = build_request("restore", options, plan=plan, sctx=sctx, snapshot_driver="n580")
    n610 = build_request("restore", options, plan=plan, sctx=sctx, snapshot_driver="n610")

    assert "mode" not in n580["policy"]["weights"]
    assert "mode" not in n610["policy"]["weights"]
    assert resolve_request_weight_mode(n580) == "recovery"
    assert resolve_request_weight_mode(n610) == "auto"
    outcome = stage_native_packs(
        n580,
        plan=plan,
        sctx=sctx,
        downloader=lambda **_kwargs: pytest.fail("unconfigured native provider attempted download"),
    )
    assert outcome.selected_mode == "recovery"
    assert outcome.request["policy"]["weights"]["mode"] == "recovery"

    document = recipe.to_dict()
    document["runtime"] = "sglang"
    sglang_recipe = Recipe.from_dict(document)
    sglang = build_request(
        "restore",
        replace(options, recipe=sglang_recipe),
        plan=replace(plan, recipe=sglang_recipe, runtime=SglangRuntime()),
        sctx=sctx,
        snapshot_driver="n580",
    )
    assert resolve_request_weight_mode(sglang) == "auto"


@pytest.mark.parametrize("snapshot_driver", ["n580", "n610"])
def test_service_progress_always_names_snapshot_driver(caplog, snapshot_driver):
    _recipe, options, plan, sctx = _setup()
    request = build_request(
        "restore",
        options,
        plan=plan,
        sctx=sctx,
        snapshot_driver=snapshot_driver,
    )
    service = ColdSnapService(
        "/opt/coldsnap/bin/coldsnap",
        run_command=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    sctx.timing = Timeline()
    caplog.set_level(logging.DEBUG, logger="sparkrun.plugins.coldsnap.service")

    service._invoke(request, prepare_only=False, capture_output=False, sctx=sctx)

    messages = [record.getMessage() for record in caplog.records]
    assert ("ColdSnap [%s]: development controller /opt/coldsnap/bin/coldsnap; running restore" % snapshot_driver) in messages
    assert any("snapshot_driver=%s" % snapshot_driver in message for message in messages)
    controller = sctx.timing.find("coldsnap.controller")
    assert controller.attrs == {
        "operation": "restore",
        "phase": "execute",
        "snapshot_driver": snapshot_driver,
        "units": 2,
        "workers": 2,
    }


def test_managed_controller_version_is_visible_at_operation_start(caplog):
    _recipe, options, plan, sctx = _setup()
    request = build_request("restore", options, plan=plan, sctx=sctx, snapshot_driver="n610")
    tool = SimpleNamespace(
        path=Path("/cache/tools/coldsnap"),
        environment={},
        version="0.3.13",
        source="cache",
    )
    service = ColdSnapService(
        run_command=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        tool_resolver=lambda _config: tool,
    )
    sctx.timing = Timeline()
    caplog.set_level(logging.DEBUG, logger="sparkrun.plugins.coldsnap.service")

    service._invoke(request, prepare_only=False, capture_output=False, sctx=sctx)

    assert "ColdSnap [n610]: controller v0.3.13 (cache); running restore" in [record.getMessage() for record in caplog.records]


def test_capture_uses_unique_immutable_generation_paths():
    _recipe, options, plan, sctx = _setup()

    first = build_request("capture", options, plan=plan, sctx=sctx)
    second = build_request("capture", options, plan=plan, sctx=sctx)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)

    assert first["id"] != second["id"]
    assert first["output"] == str(store.capture_output(first["id"]))
    assert second["output"] == str(store.capture_output(second["id"]))
    assert first["output"] != second["output"]


def test_requests_preserve_fast_recipe_loader_for_compatibility():
    recipe, options, plan, sctx = _setup()
    recipe.defaults["load_format"] = "instanttensor"

    capture = build_request("capture", options, plan=plan, sctx=sctx)
    restore = build_request("restore", options, plan=plan, sctx=sctx)

    assert "--load-format instanttensor" in capture["launch"]["units"][0]["command"][4]
    assert "--load-format instanttensor" in restore["launch"]["units"][0]["command"][4]


def test_publish_uses_current_artifact_and_new_pending_generation():
    _recipe, options, plan, sctx = _setup()
    first = build_request("publish", options, plan=plan, sctx=sctx)
    second = build_request("publish", options, plan=plan, sctx=sctx)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)

    assert first["operation"] == "publish"
    assert first["artifact"] == str(store.current)
    assert first["output"] == str(store.capture_output(first["id"]))
    assert first["policy"]["capsule"] == {
        "repository": "registry.example/coldsnap/qwen",
    }
    assert first["id"] != second["id"]


def test_publish_native_uses_current_artifact_and_explicit_hf_destination():
    _recipe, options, plan, sctx = _setup()
    request = build_request(
        "publish-native",
        options,
        plan=plan,
        sctx=sctx,
        native_repository="org/qwen-coldsnap-native",
        native_revision="main",
    )
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)

    assert request["operation"] == "publish-native"
    assert request["artifact"] == str(store.current)
    assert request["output"] == str(store.capture_output(request["id"]))
    assert request["policy"]["weights"]["native"] == {
        "repository": "org/qwen-coldsnap-native",
        "revision": "main",
    }


def test_publish_native_requires_explicit_hf_destination():
    _recipe, options, plan, sctx = _setup()
    with pytest.raises(ValueError, match="--hf-repo and --revision"):
        build_request("publish-native", options, plan=plan, sctx=sctx)


def test_publish_requires_recipe_capsule_repository():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["coldsnap"]["capsule"] = {}
    plan = replace(plan, recipe=Recipe.from_dict(document))

    with pytest.raises(ValueError, match="capsule.repository"):
        build_request("publish", options, plan=plan, sctx=sctx)


def test_publish_cli_is_explicit_and_has_no_weight_selector():
    publish = build_command().commands["publish"]
    assert {parameter.name for parameter in publish.params} >= {
        "artifact",
        "recipe",
        "cluster",
        "hosts",
        "dry_run",
        "show_timings",
        "coldsnap_binary",
    }
    assert "weights" not in {parameter.name for parameter in publish.params}

    publish_native = build_command().commands["publish-native"]
    assert {parameter.name for parameter in publish_native.params} >= {
        "artifact",
        "hf_repo",
        "revision",
        "recipe",
        "cluster",
        "hosts",
        "dry_run",
        "show_timings",
        "coldsnap_binary",
    }
    assert "weights" not in {parameter.name for parameter in publish_native.params}


def test_request_rematerialization_preserves_only_safe_operation_ids():
    _recipe, options, plan, sctx = _setup()
    request = build_request(
        "capture",
        options,
        plan=plan,
        output="/tmp/capture.json",
        operation_id="sparkrun-capture-fixed",
        sctx=sctx,
    )
    assert request["id"] == "sparkrun-capture-fixed"

    with pytest.raises(ValueError, match="operation_id"):
        build_request(
            "capture",
            options,
            plan=plan,
            operation_id="../unsafe",
            sctx=sctx,
        )


def test_explicit_artifact_paths_bypass_managed_defaults():
    _recipe, options, plan, sctx = _setup()

    capture = build_request("capture", options, plan=plan, output="/tmp/manual-capture.json", sctx=sctx)
    restore = build_request("restore", options, plan=plan, artifact="/tmp/manual-restore.json", sctx=sctx)

    assert capture["output"] == "/tmp/manual-capture.json"
    assert restore["artifact"] == "/tmp/manual-restore.json"


def test_lifecycle_requests_bind_exact_workload_and_warm_state():
    _recipe, options, plan, sctx = _setup()
    cluster_id = "sparkrun_%s_%s" % ("b" * 16, "d" * 12)

    for operation in ("sleep", "wake", "status"):
        request = build_request(
            operation,
            options,
            plan=plan,
            artifact="/tmp/artifact.json",
            workload_cluster_id=cluster_id,
            sctx=sctx,
        )
        assert request["operation"] == operation
        assert request["artifact"] == "/tmp/artifact.json"
        assert request["workload"]["cluster_id"] == cluster_id
        assert request["lifecycle"] == {"activation_state": "running"}

    warm = build_request("restore", options, plan=plan, activation_state="warm", sctx=sctx)
    assert warm["lifecycle"] == {"activation_state": "warm"}


def test_warm_request_rejects_n580():
    _recipe, options, plan, sctx = _setup()
    with pytest.raises(ValueError, match="n610"):
        build_request("restore", options, plan=plan, activation_state="warm", snapshot_driver="n580", sctx=sctx)


@pytest.mark.parametrize("operation", ["capture", "publish", "restore"])
def test_explicit_operations_require_hardware_preflight(monkeypatch, operation):
    _recipe, options, plan, sctx = _setup()
    calls = []

    def reject(*_args, **_kwargs):
        calls.append(operation)
        raise RuntimeError("hardware rejected")

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", reject)
    service = ColdSnapService(run_command=lambda *_args, **_kwargs: pytest.fail("ColdSnap was invoked"))

    with pytest.raises(RuntimeError, match="hardware rejected"):
        service.execute_explicit(operation, options, plan=plan, sctx=sctx)
    assert calls == [operation]


def test_explicit_render_only_does_not_probe_hosts(monkeypatch):
    _recipe, options, plan, sctx = _setup()
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: pytest.fail("render-only probed hosts"),
    )

    request, removed, failures = ColdSnapService().execute_explicit(
        "restore",
        options,
        plan=plan,
        sctx=sctx,
        render_only=True,
        snapshot_driver="n580",
    )

    assert request["operation"] == "restore"
    assert request["snapshot_driver"] == {"id": "n580"}
    assert removed == ()
    assert failures == ()


def test_restore_cli_uses_shared_run_strategy(monkeypatch):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    seen = []
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: seen.append(("plan", options)) or plan)
    monkeypatch.setattr(
        api,
        "run",
        lambda options, plan=None, sctx=None: seen.append(("run", options, plan)) or SimpleNamespace(rc=0),
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.ColdSnapService.execute_explicit",
        lambda *_args, **_kwargs: pytest.fail("restore bypassed the execution strategy"),
    )

    result = CliRunner().invoke(
        build_command(),
        [
            "restore",
            "recipe.yaml",
            "--cluster",
            "g610",
            "--weights",
            "recovery",
            "--materialize-native",
            "off",
            "--artifact",
            "/tmp/portable-artifact.json",
            "--coldsnap-binary",
            "/opt/coldsnap/bin/coldsnap",
            "--snapshot-driver",
            "n580",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [entry[0] for entry in seen] == ["plan", "run"]
    options = seen[1][1]
    assert options.dry_run is False
    assert options.follow is False
    assert options.strategy_options == {
        "artifact": "/tmp/portable-artifact.json",
        "weights": "recovery",
        "binary": "/opt/coldsnap/bin/coldsnap",
        "materialize_native": "off",
        "snapshot_driver": "n580",
    }
    assert seen[1][2] is plan


def test_warm_cli_uses_restore_strategy_with_explicit_warm_state(monkeypatch):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    seen = []
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr(
        api,
        "run",
        lambda options, plan=None, sctx=None: seen.append(options) or SimpleNamespace(rc=0),
    )

    result = CliRunner().invoke(build_command(), ["warm", "recipe.yaml", "--cluster", "g610"])

    assert result.exit_code == 0, result.output
    assert seen[0].strategy_options == {"activation_state": "warm"}
    assert "weights and KV remain unhydrated" in result.output


@pytest.mark.parametrize("operation,state", [("sleep", "sleeping"), ("wake", "running"), ("status", "warm")])
def test_lifecycle_cli_uses_exact_service_operation(monkeypatch, operation, state):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    seen = []
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)

    def execute(_self, selected, options, **kwargs):
        seen.append((selected, options, kwargs))
        return {"id": "request"}, {
            "kind": "coldsnap-inference-lifecycle",
            "engine": "vllm",
            "state": state,
            "cluster_id": plan.cluster_id,
            "capture_id": "capture",
            "seconds": 1.25,
        }

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.ColdSnapService.execute_lifecycle", execute)

    result = CliRunner().invoke(build_command(), [operation, "recipe.yaml", "--cluster", "g610"])

    assert result.exit_code == 0, result.output
    assert seen[0][0] == operation
    assert "state=%s" % state in result.output
    assert "ColdSnap timings" in result.output


def test_coldsnap_cli_can_suppress_timing_output(monkeypatch):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.ColdSnapService.execute_lifecycle",
        lambda *_args, **_kwargs: (
            {"id": "request"},
            {"state": "warm", "cluster_id": plan.cluster_id, "capture_id": "capture", "seconds": 0.1},
        ),
    )

    result = CliRunner().invoke(build_command(), ["status", "recipe.yaml", "--cluster", "g610", "--no-timings"])

    assert result.exit_code == 0, result.output
    assert "ColdSnap timings" not in result.output


def test_lifecycle_service_resolves_running_intent_before_control(monkeypatch, tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "capture_id": "capture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    target = "sparkrun_%s_%s" % (plan.intent_id, "d" * 12)
    events = []
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: SimpleNamespace(snapshot_driver="n610"),
    )
    monkeypatch.setattr(
        "sparkrun.api._resolve.discover_cluster_id_by_intent",
        lambda intent, hosts, **_kwargs: events.append((intent, tuple(hosts))) or target,
    )

    def invoke(_self, request, **_kwargs):
        events.append(request)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "format": 1,
                    "kind": "coldsnap-inference-lifecycle",
                    "engine": "vllm",
                    "operation_id": request["id"],
                    "operation": "sleep",
                    "cluster_id": target,
                    "capture_id": "capture",
                    "snapshot_driver": "n610",
                    "state": "sleeping",
                    "units": [{}, {}],
                }
            )
        )

    monkeypatch.setattr(ColdSnapService, "_invoke", invoke)

    request, report = ColdSnapService().execute_lifecycle("sleep", options, plan=plan, sctx=sctx, artifact=str(artifact))

    assert events[0] == (plan.intent_id, tuple(plan.cluster.hosts))
    assert request["workload"]["cluster_id"] == target
    assert report["state"] == "sleeping"


def test_coldsnap_cli_has_concise_help_for_every_command_and_option():
    import click

    from sparkrun.cli._common import CLUSTER_NAME, RECIPE_NAME

    command = build_command()
    assert set(command.commands) == {
        "capture",
        "delete",
        "materialize",
        "native-status",
        "publish",
        "publish-native",
        "restore",
        "sleep",
        "status",
        "wake",
        "warm",
    }
    for name, subcommand in command.commands.items():
        summary = (subcommand.help or "").splitlines()[0]
        assert summary, name
        assert len(summary) <= 72, (name, summary)
        assert "RECIPE is a recipe name or YAML path." in subcommand.help
        recipe = next(parameter for parameter in subcommand.params if parameter.name == "recipe")
        assert recipe.type is RECIPE_NAME
        cluster = next(parameter for parameter in subcommand.params if parameter.name == "cluster")
        assert cluster.type is CLUSTER_NAME
        dry_run = next(parameter for parameter in subcommand.params if parameter.name == "dry_run")
        assert set(dry_run.opts) == {"--dry-run", "-n"}
        assert all(parameter.name != "render_only" for parameter in subcommand.params)
        for parameter in subcommand.params:
            if isinstance(parameter, click.Option):
                assert parameter.help, (name, parameter.name)

    path_options = [
        parameter
        for subcommand in command.commands.values()
        for parameter in subcommand.params
        if parameter.name in {"artifact", "output", "coldsnap_binary"}
    ]
    assert path_options
    assert all(isinstance(parameter.type, click.Path) for parameter in path_options)
    assert all(any(item.type == "file" for item in parameter.type.shell_complete(None, parameter, "./")) for parameter in path_options)


def test_coldsnap_cli_version_reports_host_and_plugin_versions(monkeypatch):
    from click.testing import CliRunner

    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.cli._version_identity",
        lambda: ("0.4.0-alpha", "0.1.0"),
    )

    result = CliRunner().invoke(build_command(), ["--version"])

    assert result.exit_code == 0, result.output
    assert result.output == "sparkrun v0.4.0-alpha\nColdSnap plugin v0.1.0\n"


def test_coldsnap_subcommand_opens_with_progress_version_banner(monkeypatch, caplog):
    from click.testing import CliRunner

    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.cli._version_identity",
        lambda: ("0.4.0-alpha", "0.1.0"),
    )

    with caplog.at_level(PROGRESS, logger="sparkrun.plugins.coldsnap.cli"):
        result = CliRunner().invoke(build_command(), ["capture", "--help"])

    assert result.exit_code == 0, result.output
    assert [record.message for record in caplog.records] == [
        "sparkrun v0.4.0-alpha",
        "ColdSnap plugin v0.1.0",
    ]


@pytest.mark.parametrize(
    ("verbosity", "expected"),
    [(0, 2), (1, 3), (2, 4), (3, None), (4, None), (-1, 2)],
)
def test_coldsnap_timing_tree_depth_tracks_global_verbosity(verbosity, expected):
    class Context:
        obj = {"verbose": verbosity}

        def find_root(self):
            return self

    assert _timing_tree_depth(Context()) == expected


@pytest.mark.parametrize(
    "driver,native,residual,expected",
    [
        ("n580", "auto", "auto", ("off", "required")),
        ("n610", "auto", "auto", ("required", "off")),
        ("n580", "required", "off", ("required", "off")),
        ("n580", "off", "required", ("off", "required")),
    ],
)
def test_materialization_policy_resolves_driver_specific_vllm_defaults(driver, native, residual, expected):
    assert _resolve_materialization_policy(driver, "vllm", native, residual) == expected


def test_materialization_policy_rejects_unsupported_or_empty_plans():
    with pytest.raises(ValueError, match="requires vLLM"):
        _resolve_materialization_policy("n580", "unsupported", "auto", "auto")
    with pytest.raises(ValueError, match="require snapshot driver n580"):
        _resolve_materialization_policy("n610", "vllm", "required", "required")
    with pytest.raises(ValueError, match="resolved no local assets"):
        _resolve_materialization_policy("n610", "vllm", "off", "off")


def _sglang_setup():
    recipe, options, plan, sctx = _setup()
    document = recipe.to_dict()
    document["runtime"] = "sglang"
    recipe = Recipe.from_dict(document)
    return recipe, replace(options, recipe=recipe), replace(plan, recipe=recipe, runtime=SglangRuntime()), sctx


@pytest.mark.parametrize("operation", ["status", "sleep", "wake"])
@pytest.mark.parametrize("driver", ["n580", "n610"])
@pytest.mark.parametrize("explicit", [False, True])
def test_sglang_lifecycle_follows_materialized_capture_but_respects_explicit_artifact(monkeypatch, tmp_path, operation, driver, explicit):
    _recipe, options, plan, sctx = _sglang_setup()
    source, local = tmp_path / "source.json", tmp_path / "local.json"
    for path in (source, local):
        path.write_text(json.dumps({"kind": "coldsnap-snapshot-artifact", "capture_id": path.stem}))
    expected = source if explicit else local
    monkeypatch.setattr(ColdSnapService, "_restore_artifact_path", lambda *args, **kwargs: source)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.resolve_artifact_store", lambda **kwargs: object())
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.select_local_materialization", lambda *args, **kwargs: local)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts",
        lambda *args, **kwargs: SimpleNamespace(verified=True, snapshot_driver=driver, hardware={}),
    )
    monkeypatch.setattr("sparkrun.api._resolve.discover_cluster_id_by_intent", lambda *args, **kwargs: plan.cluster_id)

    def invoke(self, request, **kwargs):
        assert request["artifact"] == str(expected)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "format": 1,
                    "kind": "coldsnap-inference-lifecycle",
                    "engine": "sglang",
                    "operation_id": request["id"],
                    "operation": operation,
                    "cluster_id": plan.cluster_id,
                    "capture_id": expected.stem,
                    "snapshot_driver": driver,
                    "state": "running",
                    "units": [{}, {}],
                }
            )
        )

    monkeypatch.setattr(ColdSnapService, "_invoke", invoke)
    request, report = ColdSnapService().execute_lifecycle(
        operation, options, plan=plan, sctx=sctx, artifact=str(source) if explicit else ""
    )
    assert report["capture_id"] == expected.stem


@pytest.mark.parametrize("driver", ["n580", "n610"])
@pytest.mark.parametrize("native,residual", [("auto", "auto"), ("required", "off"), ("off", "required")])
def test_sglang_materialize_captures_and_verifies_without_write_behind(monkeypatch, tmp_path, driver, native, residual):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _sglang_setup()
    events = []
    candidate = tmp_path / "candidate.json"
    hardware = SimpleNamespace(snapshot_driver=driver, hardware={})
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.compatibility.verify_coldsnap_hosts", lambda *args, **kwargs: hardware)

    def capture(self, options, **kwargs):
        assert kwargs["native_weights"] is (native != "off")
        assert kwargs["hardware"] is hardware
        events.append("capture")
        kwargs["verify"](candidate)
        events.append("promote")
        return candidate

    def run(options, **kwargs):
        assert options.strategy_options["artifact"] == str(candidate)
        assert options.strategy_options["materialize_native"] == "off"
        assert options.strategy_options["weights"] == ("recovery" if native == "off" else "native")
        events.append("verify")
        assert options.cluster_id_override == kwargs["plan"].cluster_id != plan.cluster_id
        return SimpleNamespace(rc=0, cluster_id=options.cluster_id_override)

    def stop(**kwargs):
        assert kwargs["cluster_id"] != plan.cluster_id
        assert kwargs["hosts"] == plan.host_list
        events.append("cleanup")
        return SimpleNamespace(success=True)

    monkeypatch.setattr(ColdSnapService, "materialize_sglang", capture)
    monkeypatch.setattr(api, "run", run)
    monkeypatch.setattr(api, "stop", stop)
    result = CliRunner().invoke(
        build_command(),
        [
            "materialize",
            "recipe.yaml",
            "--cluster",
            "cluster",
            "--snapshot-driver",
            driver,
            "--native-weights",
            native,
            "--residual-overlay",
            residual,
        ],
    )
    assert result.exit_code == 0, result.output
    assert events == ["capture", "verify", "cleanup", "promote"]
    assert "materialization ready" in result.output


@pytest.mark.parametrize("driver", ["n580", "n610"])
def test_sglang_materialize_dry_run_is_capture_plan_without_cluster_changes(monkeypatch, driver):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _sglang_setup()
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.compatibility.verify_coldsnap_hosts", lambda *args, **kwargs: pytest.fail("probe"))
    monkeypatch.setattr(ColdSnapService, "materialize_sglang", lambda *args, **kwargs: pytest.fail("capture"))
    result = CliRunner().invoke(build_command(), ["materialize", "recipe.yaml", "--dry-run", "--snapshot-driver", driver])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["native_weights"] == report["residual_overlay"] == "required"
    assert report["method"] == "capture-and-verify"
    assert report["hardware_verified"] is False


@pytest.mark.parametrize("failure", ["restore", "cleanup"])
def test_sglang_materialize_failed_verification_cleans_up_and_never_reports_ready(monkeypatch, tmp_path, failure):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _sglang_setup()
    events = []
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.compatibility.verify_coldsnap_hosts",
        lambda *args, **kwargs: SimpleNamespace(snapshot_driver="n610", hardware={}),
    )

    def materialize(self, options, **kwargs):
        kwargs["verify"](tmp_path / "candidate.json")
        pytest.fail("failed verification reached promotion")

    monkeypatch.setattr(ColdSnapService, "materialize_sglang", materialize)
    monkeypatch.setattr(api, "run", lambda *args, **kwargs: SimpleNamespace(rc=1 if failure == "restore" else 0))
    monkeypatch.setattr(
        api, "stop", lambda **kwargs: events.append("stop") or SimpleNamespace(success=failure != "cleanup", errors=("stop failed",))
    )
    result = CliRunner().invoke(build_command(), ["materialize", "recipe.yaml", "--cluster", "cluster"])
    assert result.exit_code == 1
    assert events == ["stop"]
    assert "materialization ready" not in result.output


@pytest.mark.parametrize("native", [False, True])
def test_sglang_materialization_service_uses_capture_writer_and_does_not_promote_current(monkeypatch, tmp_path, native):
    _recipe, options, plan, sctx = _sglang_setup()
    source, local = tmp_path / "source.json", tmp_path / "local.json"
    store = SimpleNamespace(overlay_pending=tmp_path / "pending")
    hardware = SimpleNamespace(snapshot_driver="n610", hardware={})
    events = []
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.resolve_artifact_store", lambda **kwargs: store)
    monkeypatch.setattr(ColdSnapService, "_restore_artifact_path", lambda *args, **kwargs: source)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.validate_materialization_source", lambda path: events.append("source"))
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.select_local_materialization", lambda *args, **kwargs: None)

    def capture(self, operation, options, **kwargs):
        events.append("capture")
        assert operation == "capture"
        assert kwargs["weight_mode"] == ("auto" if native else "recovery")
        assert kwargs["artifact_scope"] == "portable"
        assert kwargs["output"]  # Explicit output suppresses managed current.json/generation promotion.
        Path(kwargs["output"]).write_text("{}")

    def promote(store_arg, source_arg, path, **kwargs):
        assert source_arg == source and store_arg is store
        assert kwargs["require_native"] is native
        kwargs["verify"](path)
        events.append("promote")
        return local

    monkeypatch.setattr(ColdSnapService, "execute_explicit", capture)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.promote_local_materialization", promote)
    result = ColdSnapService().materialize_sglang(
        options, plan=plan, sctx=sctx, artifact="", hardware=hardware, native_weights=native, verify=lambda path: events.append("verify")
    )
    assert result == local
    assert events == ["source", "capture", "verify", "promote"]


def test_sglang_materialization_reuses_verified_capture_without_recapture(monkeypatch, tmp_path):
    _recipe, options, plan, sctx = _sglang_setup()
    source, local = tmp_path / "source.json", tmp_path / "local.json"
    events = []
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.resolve_artifact_store", lambda **kwargs: object())
    monkeypatch.setattr(ColdSnapService, "_restore_artifact_path", lambda *args, **kwargs: source)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.validate_materialization_source", lambda path: None)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.select_local_materialization", lambda *args, **kwargs: local)
    monkeypatch.setattr(ColdSnapService, "execute_explicit", lambda *args, **kwargs: pytest.fail("recapture"))
    result = ColdSnapService().materialize_sglang(
        options,
        plan=plan,
        sctx=sctx,
        artifact="",
        hardware=SimpleNamespace(snapshot_driver="n580", hardware={}),
        native_weights=True,
        verify=lambda path: events.append(path),
    )
    assert result == local and events == [local]


@pytest.mark.parametrize("mode", ["auto", "native", "recovery"])
@pytest.mark.parametrize("driver", ["n580", "n610"])
def test_sglang_normal_run_selects_paired_local_capture_without_write_behind(tmp_path, monkeypatch, mode, driver):
    _recipe, options, plan, sctx = _sglang_setup()
    portable, local = tmp_path / "portable.json", tmp_path / "local.json"
    for path in (portable, local):
        value = _write_strategy_artifact(path)
        value["snapshot_driver"]["id"] = driver
        value["launch"] = {"engine": "sglang"}
        path.write_text(json.dumps(value))
    options = replace(options, strategy_options={"weights": mode})
    monkeypatch.setattr(ColdSnapService, "_restore_artifact_path", lambda *args, **kwargs: portable)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.resolve_artifact_store", lambda **kwargs: object())
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.select_local_materialization", lambda *args, **kwargs: local)
    hardware = SimpleNamespace(verified=True, snapshot_driver=driver, hardware={})
    result = ColdSnapService().describe_restore(ExecutionContext(options=options, plan=plan, sctx=sctx), hardware)
    assert result.request["artifact"] == str(local)
    assert result.request["policy"]["weights"]["mode"] == mode
    assert result.request["policy"]["weights"]["native"]["materialize"] == "off"


def test_materialize_n610_auto_requires_only_native_weights(monkeypatch):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    seen = []
    monkeypatch.setattr(api, "stop", lambda **kwargs: SimpleNamespace(success=True))
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr(
        api,
        "run",
        lambda options, plan=None, sctx=None: seen.append(options) or SimpleNamespace(rc=0),
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.compatibility.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: SimpleNamespace(snapshot_driver="n610", hardware={}),
    )
    monkeypatch.setattr(
        ColdSnapService,
        "materialize_local_overlay",
        lambda *_args, **_kwargs: pytest.fail("n610 auto attempted residual capture"),
    )

    result = CliRunner().invoke(
        build_command(),
        ["materialize", "recipe.yaml", "--cluster", "g610"],
    )

    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert seen[0].strategy_options == {
        "weights": "auto",
        "materialize_native": "required",
        "snapshot_driver": "n610",
    }
    assert "materialization ready: native weights" in result.output


def test_materialize_n580_auto_captures_residuals_without_native_weights(monkeypatch, tmp_path):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    seen = []
    monkeypatch.setattr(api, "stop", lambda **kwargs: SimpleNamespace(success=True))
    captures = []
    overlay = tmp_path / "overlay.json"
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr(
        api,
        "run",
        lambda options, plan=None, sctx=None: seen.append(options) or SimpleNamespace(rc=0),
    )
    hardware = SimpleNamespace(snapshot_driver="n580", hardware={"h1": object(), "h2": object()})
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.compatibility.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: hardware,
    )
    monkeypatch.setattr(
        ColdSnapService,
        "materialize_local_overlay",
        lambda self, options, **kwargs: captures.append((options, kwargs)) or (overlay, {}, True),
    )

    result = CliRunner().invoke(
        build_command(),
        ["materialize", "recipe.yaml", "--cluster", "g580"],
    )

    assert result.exit_code == 0, result.output
    assert len(captures) == 1
    assert captures[0][1]["snapshot_driver"] == "n580"
    assert captures[0][1]["hardware"] is hardware
    assert len(seen) == 1
    assert seen[0].strategy_options == {
        "artifact": str(overlay),
        "weights": "recovery",
        "materialize_native": "off",
        "snapshot_driver": "n580",
    }
    assert "materialization ready: target-local residuals" in result.output


def test_target_local_materialization_captures_only_recovery_residuals(monkeypatch, tmp_path):
    _recipe, options, plan, sctx = _setup()
    portable = tmp_path / "portable.json"
    portable.write_text("{}", encoding="utf-8")
    overlay = tmp_path / "overlay.json"
    store = SimpleNamespace(overlay_pending=tmp_path / "pending")
    hardware = SimpleNamespace(snapshot_driver="n580", hardware={"h1": object(), "h2": object()})
    observed = {}

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.resolve_artifact_store", lambda **_kwargs: store)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.select_local_overlay", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.promote_local_overlay",
        lambda *_args, **_kwargs: (overlay, {"format": 3}),
    )
    monkeypatch.setattr(
        ColdSnapService,
        "_restore_artifact_path",
        lambda *_args, **_kwargs: portable,
    )

    def capture(_self, operation, _options, **kwargs):
        observed.update(kwargs)
        assert operation == "capture"
        Path(kwargs["output"]).write_text("{}", encoding="utf-8")
        return {}, (), ()

    monkeypatch.setattr(ColdSnapService, "execute_explicit", capture)

    result, record, created = ColdSnapService().materialize_local_overlay(
        options,
        plan=plan,
        sctx=sctx,
        snapshot_driver="n580",
        hardware=hardware,
    )

    assert (result, record, created) == (overlay, {"format": 3}, True)
    assert observed["weight_mode"] == "recovery"
    assert observed["artifact_scope"] == "target-local"


def test_materialize_required_residual_failure_is_fatal(monkeypatch):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.compatibility.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: SimpleNamespace(snapshot_driver="n580", hardware={}),
    )

    def fail_overlay(*_args, **_kwargs):
        raise RuntimeError("target-local capture failed")

    monkeypatch.setattr(ColdSnapService, "materialize_local_overlay", fail_overlay)
    monkeypatch.setattr(api, "run", lambda *_args, **_kwargs: pytest.fail("failed overlay was verified"))

    result = CliRunner().invoke(
        build_command(),
        ["materialize", "recipe.yaml", "--cluster", "g580"],
    )

    assert result.exit_code == 1
    assert "target-local capture failed" in result.output


def test_materialize_dry_run_reports_unverified_auto_resolution(monkeypatch):
    from click.testing import CliRunner

    _recipe, _options, plan, sctx = _setup()
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr(api, "plan", lambda options, sctx=None: plan)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.compatibility.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: pytest.fail("dry-run probed hardware"),
    )

    result = CliRunner().invoke(
        build_command(),
        ["materialize", "recipe.yaml", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report == {
        "engine": "vllm",
        "format": 1,
        "hardware_verified": False,
        "kind": "sparkrun-coldsnap-materialization-plan",
        "native_weights": "required",
        "residual_overlay": "off",
        "snapshot_driver": "n610",
    }


def test_restore_strategy_options_override_artifact_and_weights(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    _write_strategy_artifact(artifact)
    options = replace(
        options,
        strategy_options={"artifact": str(artifact), "weights": "recovery"},
    )

    descriptor = ColdSnapService().describe_restore(ExecutionContext(options=options, plan=plan, sctx=sctx))

    assert descriptor.request["artifact"] == str(artifact)
    assert descriptor.request["policy"]["weights"]["mode"] == "recovery"


def test_n580_native_restore_uses_portable_capsule_instead_of_recovery_overlay(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    portable = tmp_path / "portable.json"
    artifact = _write_strategy_artifact(portable)
    artifact["snapshot_driver"]["id"] = "n580"
    portable.write_text(json.dumps(artifact), encoding="utf-8")
    options = replace(options, strategy_options={"weights": "native"})
    selected = []
    monkeypatch.setattr(
        ColdSnapService,
        "_restore_artifact_path",
        lambda *_args, **_kwargs: portable,
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.select_local_overlay",
        lambda *_args, **_kwargs: selected.append(True) or pytest.fail("native restore selected recovery-only overlay"),
    )
    hardware = SimpleNamespace(verified=True, snapshot_driver="n580", hardware={})

    descriptor = ColdSnapService().describe_restore(ExecutionContext(options=options, plan=plan, sctx=sctx), hardware)

    assert selected == []
    assert descriptor.request["artifact"] == str(portable)
    assert descriptor.request["policy"]["weights"]["mode"] == "native"


def test_n580_auto_restore_pins_recovery_when_using_local_overlay(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    portable = tmp_path / "portable.json"
    overlay = tmp_path / "overlay.json"
    for path in (portable, overlay):
        artifact = _write_strategy_artifact(path)
        artifact["snapshot_driver"]["id"] = "n580"
        path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(
        ColdSnapService,
        "_restore_artifact_path",
        lambda *_args, **_kwargs: portable,
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.resolve_artifact_store",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.select_local_overlay",
        lambda *_args, **_kwargs: overlay,
    )
    hardware = SimpleNamespace(verified=True, snapshot_driver="n580", hardware={})

    descriptor = ColdSnapService().describe_restore(ExecutionContext(options=options, plan=plan, sctx=sctx), hardware)

    assert descriptor.request["artifact"] == str(overlay)
    assert descriptor.request["policy"]["weights"]["mode"] == "recovery"


def test_explicit_capture_uses_shared_prepared_image_identities(monkeypatch):
    _recipe, options, plan, sctx = _setup()
    images = ("sha256:" + "b" * 64, "sha256:" + "c" * 64)
    comm_env = ClusterCommEnv.from_per_host(
        {
            "h1": {"NODE_IP": "10.0.0.1"},
            "h2": {"NODE_IP": "10.0.0.2"},
        }
    )
    events = []

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", lambda *_args, **_kwargs: events.append("hardware"))
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.prepare_capture_images",
        lambda *_args, **_kwargs: (
            events.append("images")
            or SimpleNamespace(
                content_images_by_node=images,
                comm_env=comm_env,
            )
        ),
    )

    def stage_cache(request, **_kwargs):
        events.append("cache")
        staged = json.loads(json.dumps(request))
        staged["policy"]["cache"]["staged"] = [
            {"unit": "unit-0", "path": "/cache/staged/unit-0"},
            {"unit": "unit-1", "path": "/cache/staged/unit-1"},
        ]
        return CaptureRuntimeCacheStage(request=staged)

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.stage_capture_runtime_cache", stage_cache)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.cleanup_capture_runtime_cache",
        lambda _stage: events.append("cleanup"),
    )

    def stage(request, **_kwargs):
        events.append("weights")
        return SimpleNamespace(request=request, failures=())

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.stage_native_packs", stage)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.replace_capture_workload",
        lambda **_kwargs: events.append("replace") or (),
    )

    def invoke(arguments, **kwargs):
        events.append("invoke")
        payload = json.loads(kwargs["input"])
        assert arguments[:2] == ["coldsnap", "capture"]
        assert tuple(unit["image"] for unit in payload["launch"]["units"]) == images
        assert tuple(unit["image_digest"] for unit in payload["launch"]["units"]) == images
        assert payload["launch"]["units"][0]["environment"]["VLLM_HOST_IP"] == "10.0.0.1"
        assert payload["launch"]["units"][1]["environment"]["VLLM_HOST_IP"] == "10.0.0.2"
        assert payload["policy"]["cache"]["staged"] == [
            {"unit": "unit-0", "path": "/cache/staged/unit-0"},
            {"unit": "unit-1", "path": "/cache/staged/unit-1"},
        ]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request, removed, failures = ColdSnapService(
        "coldsnap",
        run_command=invoke,
        target_tool_resolver=lambda **_kwargs: SimpleNamespace(environment={}),
    ).execute_explicit(
        "capture",
        options,
        plan=plan,
        sctx=sctx,
        output="/tmp/manual-capture.json",
    )

    assert tuple(unit["image"] for unit in request["launch"]["units"]) == images
    assert events == ["hardware", "images", "cache", "weights", "replace", "invoke", "cleanup"]
    assert removed == ()
    assert failures == ()


def test_capture_runtime_cache_snapshots_each_unit_from_resolved_leaf(monkeypatch):
    _recipe, options, plan, sctx = _setup()
    request = build_request("capture", options, plan=plan, output="/tmp/capture.json", sctx=sctx)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.runtime_cache.resolve_effective_runtime_cache_dir",
        lambda *_args, **_kwargs: "/cache/sparkrun",
    )
    calls = []

    def runner(host, script, **kwargs):
        calls.append((host, script, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    stage = stage_capture_runtime_cache(
        request,
        options=options,
        plan=plan,
        sctx=sctx,
        images_by_node=("sha256:" + "b" * 64, "sha256:" + "c" * 64),
        runner=runner,
    )

    parent = "/cache/coldsnap/runtime-cache-staging/%s" % request["id"]
    assert stage.request is not request
    assert stage.request["policy"]["cache"]["staged"] == [
        {"unit": "unit-0", "path": parent + "/units/unit-0"},
        {"unit": "unit-1", "path": parent + "/units/unit-1"},
    ]
    assert {call[0] for call in calls} == {"h1", "h2"}
    assert all("cp -a --no-preserve=ownership --reflink=auto" in call[1] for call in calls)
    assert all("staging root is not writable" in call[1] for call in calls)
    assert all("/cache/sparkrun/runtime-cache/vllm/" in call[1] for call in calls)
    assert all(call[2]["session_guard"] is True for call in calls)
    assert {entry.parent for entry in stage.entries} == {parent}

    cleanup_calls = []

    def cleanup_runner(host, script, **kwargs):
        cleanup_calls.append((host, script, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    cleanup_capture_runtime_cache(stage, runner=cleanup_runner)
    assert {call[0] for call in cleanup_calls} == {"h1", "h2"}
    assert all("rm -rf" in call[1] for call in cleanup_calls)
    assert all("--entrypoint chown" in call[1] for call in cleanup_calls)
    assert all("--pull never" in call[1] for call in cleanup_calls)


def test_capture_runtime_cache_resolves_source_and_staging_per_host(monkeypatch):
    _recipe, options, plan, sctx = _setup()
    request = build_request("capture", options, plan=plan, output="/tmp/capture.json", sctx=sctx)
    resolved = []

    def resolve(hosts, *_args, **_kwargs):
        resolved.append(tuple(hosts))
        return "/cache-%s/sparkrun" % hosts[0]

    monkeypatch.setattr("sparkrun.plugins.coldsnap.runtime_cache.resolve_effective_runtime_cache_dir", resolve)
    calls = []

    def runner(host, script, **_kwargs):
        calls.append((host, script))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    stage = stage_capture_runtime_cache(request, options=options, plan=plan, sctx=sctx, runner=runner)

    assert resolved == [("h1",), ("h2",)]
    assert stage.request["policy"]["cache"]["staged"] == [
        {"unit": "unit-0", "path": "/cache-h1/coldsnap/runtime-cache-staging/%s/units/unit-0" % request["id"]},
        {"unit": "unit-1", "path": "/cache-h2/coldsnap/runtime-cache-staging/%s/units/unit-1" % request["id"]},
    ]
    scripts = dict(calls)
    assert "/cache-h1/sparkrun/runtime-cache/vllm/" in scripts["h1"]
    assert "/cache-h2/sparkrun/runtime-cache/vllm/" in scripts["h2"]


def test_capture_staging_root_is_sibling_for_standard_cache_and_nested_for_custom_cache():
    assert _capture_staging_root("/home/drew/.cache/sparkrun") == "/home/drew/.cache/coldsnap/runtime-cache-staging"
    assert _capture_staging_root("/mnt/operator-cache") == "/mnt/operator-cache/coldsnap/runtime-cache-staging"


def test_disabled_runtime_cache_leaves_capture_request_unstaged():
    _recipe, options, plan, sctx = _setup()
    options = replace(options, runtime_cache=False)
    request = build_request("capture", options, plan=plan, output="/tmp/capture.json", sctx=sctx)

    stage = stage_capture_runtime_cache(
        request,
        options=options,
        plan=plan,
        sctx=sctx,
        runner=lambda *_args, **_kwargs: pytest.fail("disabled cache touched a host"),
    )

    assert stage.request is request
    assert stage.entries == ()
    assert "staged" not in request["policy"]["cache"]


def test_failed_runtime_cache_snapshot_cleans_every_planned_unit(monkeypatch):
    _recipe, options, plan, sctx = _setup()
    request = build_request("capture", options, plan=plan, output="/tmp/capture.json", sctx=sctx)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.runtime_cache.resolve_effective_runtime_cache_dir",
        lambda *_args, **_kwargs: "/cache/sparkrun",
    )
    calls = []

    def runner(host, script, **_kwargs):
        calls.append((host, script))
        failed = host == "h1" and "cp -a" in script
        return SimpleNamespace(
            returncode=1 if failed else 0,
            stdout="",
            stderr="copy failed" if failed else "",
        )

    with pytest.raises(RuntimeError, match="unit unit-0.*copy failed"):
        stage_capture_runtime_cache(
            request,
            options=options,
            plan=plan,
            sctx=sctx,
            runner=runner,
        )

    cleanup_calls = [(host, script) for host, script in calls if "rm -rf" in script]
    assert {host for host, _script in cleanup_calls} == {"h1", "h2"}
    assert any("units/unit-0" in script for _host, script in cleanup_calls)
    assert any("units/unit-1" in script for _host, script in cleanup_calls)


def test_capture_image_phase_reuses_sparkrun_transfer_resolution(monkeypatch):
    _recipe, options, plan, sctx = _setup()
    prepared = object()
    staged = object()
    observed = {}

    def resolve(mode, hosts, **kwargs):
        observed["resolve"] = (mode, hosts, kwargs)
        return SimpleNamespace(mode="delegated")

    def prepare(*args, **kwargs):
        observed["prepare"] = (args, kwargs)
        return prepared

    def stage(*args, **kwargs):
        observed["stage"] = (args, kwargs)
        return staged

    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", resolve)
    monkeypatch.setattr("sparkrun.core.image_preparation.prepare_images", prepare)
    monkeypatch.setattr("sparkrun.core.image_preparation.stage_prepared_images", stage)

    assert prepare_capture_images(options, plan=plan, sctx=sctx, snapshot_driver="n580") is staged
    assert observed["resolve"][0] == "auto"
    assert observed["resolve"][1] == ["h1", "h2"]
    assert observed["prepare"][1]["run_builder"] is True
    assert observed["prepare"][1]["transfer_mode"] == "delegated"
    assert observed["prepare"][1]["builder_context"] == {"snapshot_driver": "n580", "engine": "vllm"}
    assert observed["stage"][0][0] is prepared
    assert observed["stage"][1]["require_content_ids"] is True
    assert observed["stage"][1]["stage_models"] is True


def test_capture_generation_is_atomically_promoted(tmp_path):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    output, artifact = _write_pending_capture(store, "capture-one")

    promote_generation(output, store)

    assert not output.exists()
    assert json.loads(store.generation("capture-one").read_text(encoding="utf-8")) == artifact
    assert json.loads(store.current.read_text(encoding="utf-8")) == artifact
    assert store.current.stat().st_mode & 0o777 == 0o600


def test_publication_promotes_same_capture_as_new_descriptor_generation(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    store.current.parent.mkdir(parents=True)
    store.current.write_text(
        json.dumps(
            {
                "format": 3,
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "capture_id": "capture-one",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.stage_native_packs",
        lambda *_args, **_kwargs: pytest.fail("publication staged weights"),
    )
    published_reference = "oci://registry.example/coldsnap/qwen@sha256:" + "e" * 64
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.publish_oci_artifact",
        lambda path, reference, **_kwargs: published_reference,
    )

    def invoke(arguments, **kwargs):
        request = json.loads(kwargs["input"])
        assert arguments[:2] == ["coldsnap", "publish"]
        Path(request["output"]).parent.mkdir(parents=True, exist_ok=True)
        Path(request["output"]).write_text(
            json.dumps(
                {
                    "format": 3,
                    "kind": "coldsnap-snapshot-artifact",
                    "state": "committed",
                    "capture_id": "capture-one",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request, removed, failures = ColdSnapService(
        "coldsnap",
        run_command=invoke,
        target_tool_resolver=lambda **_kwargs: SimpleNamespace(environment={}),
    ).execute_explicit(
        "publish",
        options,
        plan=plan,
        sctx=sctx,
    )

    assert request["policy"]["capsule"]["repository"] == "registry.example/coldsnap/qwen"
    assert request["published_artifact_reference"] == published_reference
    assert store.generation(request["id"]).is_file()
    assert json.loads(store.current.read_text(encoding="utf-8"))["capture_id"] == "capture-one"
    assert removed == ()
    assert failures == ()


def test_native_publication_promotes_provider_and_refreshes_portable_descriptor(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    store.current.parent.mkdir(parents=True)
    artifact = {
        "format": 8,
        "kind": "coldsnap-snapshot-artifact",
        "state": "committed",
        "capture_id": "capture-one",
        "capsule": {
            "images": [
                {
                    "unit": "unit-%d" % index,
                    "reference": "registry.example/coldsnap/qwen@sha256:" + "a" * 64,
                    "digest": "sha256:" + "a" * 64,
                }
                for index in range(2)
            ]
        },
        "weights": {
            "native": {},
            "model_payloads": {
                "repository": "",
                "revision": "",
                "objects": [],
            },
        },
    }
    store.current.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", lambda *_args, **_kwargs: None)
    published_reference = "oci://registry.example/coldsnap/qwen@sha256:" + "e" * 64
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.publish_oci_artifact",
        lambda path, reference, **_kwargs: published_reference,
    )

    def invoke(arguments, **kwargs):
        request = json.loads(kwargs["input"])
        assert arguments[:2] == ["coldsnap", "publish-native"]
        assert kwargs["env"]["COLDSNAP_HOST_PROVIDER"] == "external"
        assert "HF_TOKEN" not in kwargs["env"]
        updated = json.loads(store.current.read_text(encoding="utf-8"))
        updated["weights"]["model_payloads"].update(
            {
                "repository": request["policy"]["weights"]["native"]["repository"],
                "revision": "b" * 40,
            }
        )
        Path(request["output"]).parent.mkdir(parents=True, exist_ok=True)
        Path(request["output"]).write_text(json.dumps(updated), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    request, removed, failures = ColdSnapService(
        "coldsnap",
        run_command=invoke,
        target_tool_resolver=lambda **_kwargs: SimpleNamespace(environment={}),
    ).execute_explicit(
        "publish-native",
        options,
        plan=plan,
        sctx=sctx,
        native_repository="org/qwen-coldsnap-native",
        native_revision="main",
    )

    assert request["published_native_repository"] == "org/qwen-coldsnap-native"
    assert request["published_native_revision"] == "b" * 40
    assert request["published_artifact_reference"] == published_reference
    assert store.generation(request["id"]).is_file()
    assert json.loads(store.current.read_text(encoding="utf-8"))["weights"]["model_payloads"]["revision"] == "b" * 40
    assert removed == ()
    assert failures == ()


def test_invalid_generation_does_not_replace_current(tmp_path):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    store.current.parent.mkdir(parents=True)
    store.current.write_text('{"existing": true}', encoding="utf-8")
    output, _artifact = _write_pending_capture(store, "capture-bad", state="partial")

    with pytest.raises(RuntimeError, match="not a committed ColdSnap artifact"):
        promote_generation(output, store)

    assert json.loads(store.current.read_text(encoding="utf-8")) == {"existing": True}


def test_managed_capture_retains_latest_two_generations(tmp_path):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    removed = []

    for index in range(7):
        output, _artifact = _write_pending_capture(store, "capture-%d" % index)
        removed.extend(promote_generation(output, store))

    assert DEFAULT_ARTIFACT_GENERATIONS == 2
    assert sorted(path.name for path in store.generations.glob("*.json")) == [
        "capture-5.json",
        "capture-6.json",
    ]
    assert len(removed) == 5
    assert json.loads(store.current.read_text(encoding="utf-8"))["capture_id"] == "capture-6"


def test_generation_retention_uses_activation_order(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    activated = iter([100, 50])
    monkeypatch.setattr("sparkrun.plugins.coldsnap.artifacts.time.time_ns", lambda: next(activated))

    first, _artifact = _write_pending_capture(store, "z-first")
    os.utime(first, ns=(1, 1))
    promote_generation(first, store, keep_generations=1)
    first_activation = store.generation("z-first").stat().st_mtime_ns
    second, _artifact = _write_pending_capture(store, "a-second")
    os.utime(second, ns=(1, 1))
    promote_generation(second, store, keep_generations=1)

    assert [path.name for path in store.generations.glob("*.json")] == ["a-second.json"]
    assert store.generation("a-second").stat().st_mtime_ns > first_activation


def test_zero_generation_retention_preserves_current_only(tmp_path):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    output, artifact = _write_pending_capture(store, "capture-current-only")

    removed = promote_generation(output, store, keep_generations=0)

    assert removed == (store.generation("capture-current-only"),)
    assert not list(store.generations.glob("*.json"))
    assert json.loads(store.current.read_text(encoding="utf-8")) == artifact


def test_unlimited_generation_retention_prunes_nothing(tmp_path):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)

    for index in range(7):
        output, _artifact = _write_pending_capture(store, "capture-%d" % index)
        assert promote_generation(output, store, keep_generations=None) == ()

    assert len(list(store.generations.glob("*.json"))) == 7


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 2), (0, 0), (2, 2), ("unlimited", None), (" UNLIMITED ", None)],
)
def test_generation_retention_config(configured, expected):
    if configured is None:
        config = SimpleNamespace()
    else:
        config = SimpleNamespace(plugin_settings=lambda _name: {"artifact_generations": configured})
    assert resolve_generation_limit(config) == expected


@pytest.mark.parametrize("configured", [True, -1, 1.5, "five", None])
def test_invalid_generation_retention_config(configured):
    config = SimpleNamespace(plugin_settings=lambda _name: {"artifact_generations": configured})
    with pytest.raises(ValueError, match="artifact_generations"):
        resolve_generation_limit(config)


def test_coldsnap_policy_changes_recipe_fingerprint():
    recipe, *_ = _setup()
    document = recipe.to_dict()
    document["coldsnap"].setdefault("process", {})["async_graphs"] = False
    eager = Recipe.from_dict(document)

    assert derive_recipe_fingerprint(recipe) != derive_recipe_fingerprint(eager)


def test_coldsnap_can_reenable_captured_driver_floor_without_changing_omitted_export():
    recipe, options, plan, sctx = _setup()
    assert "compatibility" not in recipe.to_dict()["coldsnap"]

    document = recipe.to_dict()
    document["coldsnap"]["compatibility"] = {"enforce_captured_driver_floor": True}
    strict = Recipe.from_dict(document)
    strict_options = replace(options, recipe=strict)
    strict_plan = replace(plan, recipe=strict)

    request = build_request("restore", strict_options, plan=strict_plan, sctx=sctx)
    assert request["policy"]["compatibility"] == {"enforce_captured_driver_floor": True}
    assert derive_recipe_fingerprint(recipe) != derive_recipe_fingerprint(strict)


def test_native_pack_staging_is_worker_specific_and_preverified():
    _recipe, options, plan, sctx = _setup()
    request = build_request(
        "restore",
        options,
        plan=plan,
        artifact="./coldsnap-artifact.json",
        sctx=sctx,
    )
    request["policy"]["weights"]["native"]["files_by_worker"] = {
        "worker-0": "model-payloads/sha256/%s.pack" % ("0" * 64),
        "worker-1": "model-payloads/sha256/%s.pack" % ("1" * 64),
    }
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        worker = kwargs["worker"]
        index = _worker_index(worker)
        return _validated_pack_record(
            worker,
            "/cache/native/worker-%d.weights" % index,
            100 + index,
            "sha256:" + str(index) * 64,
        )

    outcome = stage_native_packs(request, plan=plan, sctx=sctx, downloader=download)
    assert outcome.selected_mode == "native"
    assert [entry["worker"] for entry in outcome.request["policy"]["weights"]["native"]["staged"]] == [
        "worker-0",
        "worker-1",
    ]
    assert {call["host"] for call in calls} == {"h1", "h2"}


def test_multi_gpu_units_download_only_their_own_worker_packs():
    recipe, options, plan, sctx = _setup()
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    plan.cluster.hosts_hardware = {
        host: HostHardware(
            accelerators=[AcceleratorSpec("nvidia", "h100", count=2, memory_gb=80.0, capabilities=frozenset({"cuda"}))],
        )
        for host in plan.host_list
    }
    document = recipe.to_dict()
    document["defaults"]["tensor_parallel"] = 4
    recipe = Recipe.from_dict(document)
    placement = RankAssignment(
        by_rank=(RankSlot("h1", 0), RankSlot("h1", 1), RankSlot("h2", 0), RankSlot("h2", 1)),
        hosts_used=("h1", "h2"),
    )
    plan = replace(plan, recipe=recipe, placement=placement)
    options = replace(options, recipe=recipe)
    request = build_request(
        "restore",
        options,
        plan=plan,
        artifact="./coldsnap-artifact.json",
        sctx=sctx,
    )
    request["policy"]["weights"]["native"]["files_by_worker"] = {
        "worker-%d" % index: "model-payloads/sha256/%s.pack" % (str(index) * 64) for index in range(4)
    }
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        worker = kwargs["worker"]
        index = _worker_index(worker)
        return _validated_pack_record(
            worker,
            "/cache/native/%s.weights" % worker,
            100 + index,
            "sha256:" + str(index) * 64,
        )

    outcome = stage_native_packs(request, plan=plan, sctx=sctx, downloader=download)

    assert outcome.selected_mode == "native"
    assert {(call["worker"], call["host"]) for call in calls} == {
        ("worker-0", "h1"),
        ("worker-1", "h1"),
        ("worker-2", "h2"),
        ("worker-3", "h2"),
    }
    assert len(request["launch"]["units"]) == 2
    assert len(request["launch"]["execution"]["workers"]) == 4


def test_auto_pack_failure_selects_recovery_but_native_mode_fails():
    _recipe, options, plan, sctx = _setup()
    request = build_request(
        "restore",
        options,
        plan=plan,
        artifact="./coldsnap-artifact.json",
        sctx=sctx,
    )
    request["policy"]["weights"]["native"]["files_by_worker"] = {
        "worker-0": "model-payloads/sha256/%s.pack" % ("0" * 64),
        "worker-1": "model-payloads/sha256/%s.pack" % ("1" * 64),
    }

    def missing(**_kwargs):
        raise FileNotFoundError("not in the pinned repository")

    outcome = stage_native_packs(request, plan=plan, sctx=sctx, downloader=missing)
    assert outcome.selected_mode == "recovery"
    assert outcome.request["policy"]["weights"]["mode"] == "recovery"
    request["policy"]["weights"]["mode"] = "native"

    with pytest.raises(RuntimeError, match="native ColdSnap model payload staging failed"):
        stage_native_packs(request, plan=plan, sctx=sctx, downloader=missing)


def test_unpublished_native_inventory_falls_back_to_materializing_recovery(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "weights": {
                    "native": {},
                    "model_payloads": {
                        "repository": "",
                        "revision": "",
                        "objects": _artifact_packs(),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    request = build_request(
        "restore",
        options,
        plan=plan,
        artifact=str(artifact),
        sctx=sctx,
        native_materialization="required",
    )

    def missing(**_kwargs):
        raise FileNotFoundError("node-local model payload is unavailable")

    outcome = stage_native_packs(request, plan=plan, sctx=sctx, node_cache_resolver=missing)

    assert outcome.selected_mode == "recovery"
    assert outcome.request["policy"]["weights"]["mode"] == "recovery"
    assert outcome.request["policy"]["weights"]["native"] == {"materialize": "required"}
    assert "model payload repository and pinned revision are not configured" in outcome.failures


def test_native_pack_metadata_can_come_from_committed_artifact(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "weights": {
                    "native": {},
                    "model_payloads": {
                        "repository": "org/from-artifact",
                        "revision": "artifact-commit",
                        "objects": [
                            {
                                "role": "model-weight-payload",
                                "owner": "worker/worker-0",
                                "path": "model-payloads/sha256/%s.pack" % ("0" * 64),
                                "bytes": 100,
                                "sha256": "sha256:" + "0" * 64,
                            },
                            {
                                "role": "model-weight-payload",
                                "owner": "worker/worker-1",
                                "path": "model-payloads/sha256/%s.pack" % ("1" * 64),
                                "bytes": 101,
                                "sha256": "sha256:" + "1" * 64,
                            },
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    request = build_request("restore", options, plan=plan, artifact=str(artifact), sctx=sctx)
    native = request["policy"]["weights"]["native"]
    native.update({"repository": "", "revision": "", "files_by_worker": {}})

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        index = _worker_index(kwargs["worker"])
        return _validated_pack_record(
            kwargs["worker"],
            "/cache/worker-%d.weights" % index,
            100,
            "sha256:" + str(index) * 64,
        )

    outcome = stage_native_packs(request, plan=plan, sctx=sctx, downloader=download)
    assert outcome.selected_mode == "native"
    assert {call["repository"] for call in calls} == {"org/from-artifact"}
    assert {call["revision"] for call in calls} == {"artifact-commit"}


def test_node_local_read_through_cache_precedes_capture_and_published_sources(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "weights": {
                    "native": {},
                    "model_payloads": {
                        "repository": "org/published",
                        "revision": "pinned",
                        "objects": _artifact_packs(),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    request = build_request("restore", options, plan=plan, artifact=str(artifact), sctx=sctx)
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        worker = kwargs["worker"]
        index = _worker_index(worker)
        return _validated_pack_record(
            worker,
            "/cache/coldsnap/model-payloads/sha256/%s.pack" % (str(index) * 64),
            100 + index,
            "sha256:" + str(index) * 64,
        )

    outcome = stage_native_packs(
        request,
        plan=plan,
        sctx=sctx,
        node_cache_resolver=resolve,
        local_resolver=lambda **_kwargs: pytest.fail("capture-local resolver was used"),
        downloader=lambda **_kwargs: pytest.fail("published provider was used"),
    )

    assert outcome.selected_mode == "native"
    assert {call["host"] for call in calls} == {"h1", "h2"}
    assert {entry["worker"] for entry in outcome.request["policy"]["weights"]["native"]["staged"]} == {
        "worker-0",
        "worker-1",
    }
    assert all(
        set(entry) == {"worker", "path", "bytes", "sha256", "validation"}
        for entry in outcome.request["policy"]["weights"]["native"]["staged"]
    )


def test_node_local_cache_resolver_delegates_to_staged_go_verifier(tmp_path, monkeypatch):
    digest = hashlib.sha256(b"payload").hexdigest()
    pack = tmp_path / "model-payloads" / "sha256" / (digest + ".pack")
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"payload")
    # A prototype-era marker is deliberately ignored. The current provider
    # establishes its own complete-content evidence instead of preserving a
    # backwards-compatibility acceptance path.
    marker = pack.with_name(pack.name + ".coldsnap-preverified.json")
    marker.write_text("{}\n", encoding="utf-8")
    marker.chmod(0o600)

    def run_script(_host, script, **_kwargs):
        completed = subprocess.run(("bash",), input=script, text=True, capture_output=True, check=False)
        return SimpleNamespace(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", run_script)
    record = _resolve_node_cached_pack(
        worker="worker-0",
        unit="unit-0",
        host="localhost",
        capture_id="",
        snapshot_driver="n610",
        state_root=str(tmp_path),
        expected={
            "role": "model-weight-payload",
            "owner": "worker/worker-0",
            "path": "model-payloads/sha256/%s.pack" % digest,
            "bytes": len(b"payload"),
            "sha256": "sha256:" + digest,
        },
        ssh_kwargs={},
        verifier=str(_test_payload_verifier(tmp_path)),
    )

    assert record["path"] == str(pack)
    validation = record["validation"]
    assert validation["record"] == str(pack) + ".coldsnap-validation.json"
    assert validation["provider"] == "sha256-cache-v1"
    assert validation["content_evidence"] == "full-sha256-this-operation"
    assert validation["reason"] == "test-contract"
    assert validation["bytes_hashed"] == len(b"payload")


def test_node_local_cache_miss_is_clean_successful_probe(tmp_path, monkeypatch):
    observed = []

    def run_script(_host, script, **_kwargs):
        completed = subprocess.run(("bash",), input=script, text=True, capture_output=True, check=False)
        observed.append(completed)
        return SimpleNamespace(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", run_script)
    with pytest.raises(RuntimeError, match="model payload is unavailable"):
        _resolve_node_cached_pack(
            worker="worker-0",
            unit="unit-0",
            host="localhost",
            capture_id="",
            snapshot_driver="n610",
            state_root=str(tmp_path),
            expected={
                "role": "model-weight-payload",
                "owner": "worker/worker-0",
                "path": "model-payloads/sha256/%s.pack" % ("a" * 64),
                "bytes": 1,
                "sha256": "sha256:" + "a" * 64,
            },
            ssh_kwargs={},
            verifier="/bin/false",
        )

    assert observed[0].returncode == 0
    assert observed[0].stderr == ""
    assert "Traceback" not in observed[0].stdout


def test_native_status_reports_ready_stale_and_in_progress_payloads(tmp_path, monkeypatch):
    payload = b"payload"
    digest = hashlib.sha256(payload).hexdigest()
    relative = "model-payloads/sha256/%s.pack" % digest
    pack = tmp_path / relative
    pack.parent.mkdir(parents=True)
    pack.write_bytes(payload)
    value = pack.stat()
    validation = pack.with_name(pack.name + ".coldsnap-validation.json")
    validation.write_text(
        json.dumps(
            {
                "format": 1,
                "kind": "coldsnap-payload-validation",
                "provider": "sha256-cache-v1",
                "blob": pack.name,
                "expected": {"bytes": len(payload), "sha256": "sha256:" + digest},
                "content_identity": {
                    "device": value.st_dev,
                    "inode": value.st_ino,
                    "size": value.st_size,
                    "mtime_ns": value.st_mtime_ns,
                },
            }
        ),
        encoding="utf-8",
    )
    validation.chmod(0o600)

    def run_script(_host, script, **_kwargs):
        completed = subprocess.run(("bash",), input=script, text=True, capture_output=True, check=False)
        return SimpleNamespace(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", run_script)
    arguments = {
        "worker": "worker-0",
        "host": "localhost",
        "state_root": str(tmp_path),
        "expected": {"path": relative, "bytes": len(payload), "sha256": "sha256:" + digest},
        "ssh_kwargs": {},
    }
    ready = _read_native_pack_status(**arguments)
    assert ready["state"] == "ready"
    assert ready["validation"]["content_evidence"] == "cached-full-sha256"

    os.utime(pack, ns=(value.st_atime_ns, value.st_mtime_ns + 1_000_000))
    stale = _read_native_pack_status(**arguments)
    assert stale["state"] == "revalidation-needed"

    pack.unlink()
    status = tmp_path / "model-payloads" / ".status" / "worker-0.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "state": "writing",
                "operation_id": "materialize-one",
                "path": str(pack),
                "bytes": len(payload),
                "sha256": "sha256:" + digest,
                "seconds": 2.5,
            }
        ),
        encoding="utf-8",
    )
    writing = _read_native_pack_status(**arguments)
    assert writing["state"] == "writing"
    assert writing["operation_id"] == "materialize-one"


def test_native_materialization_mode_is_typed():
    _recipe, options, plan, sctx = _setup()
    request = build_request(
        "restore",
        options,
        plan=plan,
        sctx=sctx,
        native_materialization="required",
    )
    assert request["policy"]["weights"]["native"]["materialize"] == "required"
    with pytest.raises(ValueError, match="native materialization"):
        build_request(
            "restore",
            options,
            plan=plan,
            sctx=sctx,
            native_materialization="eventually",
        )


def test_artifact_without_native_replay_does_not_stage_shared_payloads(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "n580-artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "weights": {
                    "model_payloads": {
                        "repository": "org/shared-models",
                        "revision": "pinned",
                        "objects": _artifact_packs(),
                    },
                    "recovery": {},
                }
            }
        ),
        encoding="utf-8",
    )
    request = build_request("restore", options, plan=plan, artifact=str(artifact), sctx=sctx)

    outcome = stage_native_packs(
        request,
        plan=plan,
        sctx=sctx,
        downloader=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model payload must not be staged without native replay")),
    )

    assert outcome.selected_mode == "recovery"
    assert outcome.failures == ("artifact has no driver-qualified native replay provider",)


def test_capture_local_native_packs_precede_unpublished_provider(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "capture_id": "capture-local-one",
                "snapshot_driver": {"id": "n610", "abi": 1},
                "launch": _artifact_launch(),
                "weights": {
                    "native": {},
                    "model_payloads": {"objects": _artifact_packs()},
                },
            }
        ),
        encoding="utf-8",
    )
    request = build_request("restore", options, plan=plan, artifact=str(artifact), sctx=sctx)
    request["policy"]["weights"]["native"].update({"repository": "", "revision": "", "files_by_worker": {}})
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        worker = kwargs["worker"]
        index = _worker_index(worker)
        return _validated_pack_record(
            worker,
            "/cache/coldsnap/%s/model-weights.pack" % worker,
            100 + index,
            "sha256:" + str(index) * 64,
        )

    def unexpected_download(**_kwargs):
        raise AssertionError("published provider must not be used")

    outcome = stage_native_packs(
        request,
        plan=plan,
        sctx=sctx,
        downloader=unexpected_download,
        local_resolver=resolve,
    )
    assert outcome.selected_mode == "native"
    assert [entry["worker"] for entry in outcome.request["policy"]["weights"]["native"]["staged"]] == [
        "worker-0",
        "worker-1",
    ]
    assert {call["host"] for call in calls} == {"h1", "h2"}
    assert {call["capture_id"] for call in calls} == {"capture-local-one"}


def test_capture_local_payload_delegates_to_staged_go_verifier(tmp_path, monkeypatch):
    capture_id = "capture-local-marker"
    pack = tmp_path / "captures" / capture_id / "drivers" / "n610" / "units" / "unit-0" / "hydration" / "worker" / "model-weights.pack"
    pack.parent.mkdir(parents=True)
    (pack.parent / "manifest.json").write_text(json.dumps({"worker_id": "worker-0"}), encoding="utf-8")
    payload = b"model-payload"
    pack.write_bytes(payload)
    expected = {
        "bytes": len(payload),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }

    def run_script(_host, script, **_kwargs):
        completed = subprocess.run(("bash",), input=script, text=True, capture_output=True, check=False)
        return SimpleNamespace(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", run_script)
    arguments = {
        "worker": "worker-0",
        "unit": "unit-0",
        "host": "local",
        "capture_id": capture_id,
        "snapshot_driver": "n610",
        "state_root": str(tmp_path),
        "expected": expected,
        "ssh_kwargs": {},
        "verifier": str(_test_payload_verifier(tmp_path)),
    }
    first = _resolve_capture_local_pack(**arguments)

    marker = Path(first["validation"]["record"])
    assert first["validation"]["content_evidence"] == "full-sha256-this-operation"
    assert first["validation"]["bytes_hashed"] == len(payload)
    assert marker == Path(str(pack) + ".coldsnap-validation.json")
    assert first["validation"]["reason"] == "test-contract"


@pytest.mark.parametrize("driver", ["n580", "n610"])
@pytest.mark.parametrize("scenario", ["missing", "missing-pack", "ambiguous", "invalid-manifest", "non-file", "verifier-failure"])
def test_capture_local_probe_distinguishes_absence_from_errors(tmp_path, monkeypatch, driver, scenario):
    hydration = tmp_path / "captures" / "capture-probe" / "drivers" / driver / "units" / "unit-0" / "hydration"
    if scenario != "missing":
        for index in range(2 if scenario == "ambiguous" else 1):
            directory = hydration / str(index)
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("invalid" if scenario == "invalid-manifest" else json.dumps({"worker_id": "worker-0"}))
            if scenario == "non-file":
                (directory / "model-weights.pack").mkdir()
            elif scenario != "missing-pack":
                (directory / "model-weights.pack").write_bytes(b"payload")
    verifier = tmp_path / "reject-payload"
    verifier.write_text("#!/bin/sh\necho 'payload digest mismatch' >&2\nexit 1\n")
    verifier.chmod(0o755)
    observed = []

    def run_script(_host, script, **kwargs):
        assert not kwargs.get("quiet"), "SSH and verification failures must remain visible"
        completed = subprocess.run(("bash",), input=script, text=True, capture_output=True, check=False)
        observed.append(completed)
        return SimpleNamespace(success=completed.returncode == 0, stdout=completed.stdout, stderr=completed.stderr)

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", run_script)
    errors = {
        "missing": "capture-local model payload not found",
        "missing-pack": "capture-local model payload not found",
        "ambiguous": "expected exactly one.*found 2",
        "invalid-manifest": "manifest is unreadable or invalid",
        "non-file": "model payload is not a regular file",
        "verifier-failure": "payload digest mismatch",
    }
    with pytest.raises(RuntimeError, match=errors[scenario]):
        _resolve_capture_local_pack(
            worker="worker-0",
            unit="unit-0",
            host="remote",
            capture_id="capture-probe",
            snapshot_driver=driver,
            state_root=str(tmp_path),
            expected={"bytes": 7, "sha256": "sha256:" + "a" * 64},
            ssh_kwargs={},
            verifier=str(verifier),
        )
    result = observed[0]
    if scenario in {"missing", "missing-pack"}:
        assert result.returncode == 0
        assert result.stderr == ""
        assert result.stdout.startswith("COLDSNAP_MISS ")
    else:
        assert result.returncode != 0
        assert "COLDSNAP_MISS " not in result.stdout


def test_downloaded_pack_remote_verifier_is_valid_python(monkeypatch):
    captured = []

    def capture_script(_host, script, **_kwargs):
        captured.append(script)
        return SimpleNamespace(success=False, stdout="", stderr="expected stop")

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", capture_script)
    with pytest.raises(RuntimeError, match="expected stop"):
        _download_worker_pack(
            worker="worker-0",
            host="remote",
            repository="org/model",
            revision="commit",
            filename="rank0.weights",
            expected={"bytes": 1, "sha256": "sha256:" + "a" * 64},
            cache_dir="/cache/hf",
            offline=True,
            ssh_kwargs={},
            verifier="/bin/false",
        )
    program = captured[0].split("python3 - <<'COLDSNAP_PY'\n", 1)[1].rsplit("\nCOLDSNAP_PY\n", 1)[0]
    ast.parse(program)
    assert '"payload-verify"' in program
    assert "hashlib.sha256" not in program


def test_capture_local_pack_host_mismatch_falls_back_to_published_provider(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "capture_id": "capture-local-two",
                "snapshot_driver": {"id": "n610", "abi": 1},
                "launch": _artifact_launch(("other-h1", "other-h2")),
                "weights": {
                    "native": {},
                    "model_payloads": {
                        "repository": "org/published",
                        "revision": "pinned",
                        "objects": _artifact_packs(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    request = build_request("restore", options, plan=plan, artifact=str(artifact), sctx=sctx)
    request["policy"]["weights"]["native"].update({"repository": "", "revision": "", "files_by_worker": {}})
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        worker = kwargs["worker"]
        index = _worker_index(worker)
        return _validated_pack_record(
            worker,
            "/cache/published/%s.weights" % worker,
            100 + index,
            "sha256:" + str(index) * 64,
        )

    outcome = stage_native_packs(
        request,
        plan=plan,
        sctx=sctx,
        downloader=download,
        local_resolver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("host-mismatched pack was probed")),
    )
    assert outcome.selected_mode == "native"
    assert len(downloads) == 2
    assert not outcome.failures


def test_capture_local_pack_mismatch_falls_back_in_auto_mode(tmp_path):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "coldsnap-snapshot-artifact",
                "state": "committed",
                "capture_id": "capture-local-three",
                "snapshot_driver": {"id": "n610", "abi": 1},
                "launch": _artifact_launch(),
                "weights": {
                    "native": {},
                    "model_payloads": {"objects": _artifact_packs()},
                },
            }
        ),
        encoding="utf-8",
    )
    request = build_request("restore", options, plan=plan, artifact=str(artifact), sctx=sctx)
    request["policy"]["weights"]["native"].update({"repository": "", "revision": "", "files_by_worker": {}})

    def resolve(**kwargs):
        worker = kwargs["worker"]
        index = _worker_index(worker)
        return _validated_pack_record(
            worker,
            "/cache/coldsnap/%s/model-weights.pack" % worker,
            100 + index,
            "sha256:" + "f" * 64,
        )

    outcome = stage_native_packs(request, plan=plan, sctx=sctx, local_resolver=resolve)
    assert outcome.selected_mode == "recovery"
    assert any("digest differs" in failure for failure in outcome.failures)


def _write_strategy_artifact(path):
    digest = "sha256:" + "d" * 64
    artifact = {
        "format": 8,
        "kind": "coldsnap-snapshot-artifact",
        "state": "committed",
        "capture_id": "capture-one",
        "snapshot_driver": {"id": "n610", "abi": 1},
        "capsule": {
            "images": [
                {"unit": "unit-0", "reference": "registry/capsule0@" + digest, "digest": digest},
                {"unit": "unit-1", "reference": "registry/capsule1@" + digest, "digest": digest},
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


def test_strategy_native_staging_uses_explicit_controller_adapter(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    controller = tmp_path / "coldsnap"
    adapter = tmp_path / "coldsnap-vllm-adapter"
    controller.write_text("controller", encoding="utf-8")
    adapter.write_text("adapter", encoding="utf-8")
    controller.chmod(0o755)
    adapter.chmod(0o755)
    options = replace(options, strategy_options={"binary": str(controller)})
    request = build_request(
        "restore",
        options,
        plan=plan,
        sctx=sctx,
        artifact=str(tmp_path / "artifact.json"),
    )

    def stage(current, **kwargs):
        assert kwargs["payload_verifier"]() == adapter
        return SimpleNamespace(request=current, selected_mode="recovery", failures=())

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.stage_native_packs", stage)

    def target_resolver(**kwargs):
        assert kwargs["binary"] == str(controller)
        assert kwargs["engine"] == "vllm"
        assert kwargs["hosts"] == plan.host_list
        return SimpleNamespace(verifier=adapter)

    service = ColdSnapService(
        tool_resolver=lambda _config: pytest.fail("managed controller was resolved"),
        target_tool_resolver=target_resolver,
    )
    state = service.stage_restore(
        ExecutionContext(options=options, plan=plan, sctx=sctx),
        RestoreDescriptor(request=request, artifact={}),
    )

    assert state.selected_mode == "recovery"


@pytest.mark.parametrize(("selected", "prepare_model"), [("native", False), ("recovery", True)])
@pytest.mark.parametrize("driver", ["n580", "n610"])
def test_coldsnap_strategy_selects_capsules_before_conditional_model_preparation(tmp_path, monkeypatch, selected, prepare_model, driver):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx, snapshot_driver=driver)
    artifact = _write_strategy_artifact(store.current)
    artifact["snapshot_driver"]["id"] = driver
    store.current.write_text(json.dumps(artifact))
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware
    from sparkrun.plugins.coldsnap.compatibility import ColdSnapHardwareReceipt

    hardware = {
        host: HostHardware(
            accelerators=[AcceleratorSpec("nvidia", "gb10")],
            source="detected",
            driver_versions={"nvidia": driver[1:] + ".1"},
            ib_info={"IB_DETECTED": "1"},
        )
        for host in plan.host_list
    }
    hardware_receipt = ColdSnapHardwareReceipt(hardware, verified=True, snapshot_driver=driver)

    def stage(request, **_kwargs):
        prepared = json.loads(json.dumps(request))
        prepared["policy"]["weights"]["mode"] = selected
        return SimpleNamespace(request=prepared, selected_mode=selected, failures=())

    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.stage_native_packs", stage)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts",
        lambda *_args, **_kwargs: hardware_receipt,
    )
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.ColdSnapService.prepare_capsules",
        lambda _self, _context, state: SimpleNamespace(
            request=state.request,
            receipt={"provider": state.selected_mode},
        ),
    )
    context = ExecutionContext(options=options, plan=plan, sctx=sctx)
    strategy, steps = resolve_recipe_execution(context)
    assert [step.name for step in steps][:2] == ["coldsnap.hardware", "coldsnap.artifact"]
    sctx.timing = Timeline()
    preparation_span = sctx.timing.begin("execution.prepare", strategy="coldsnap")
    receipts = run_preparation_steps(context, steps, timeline=sctx.timing, parent=preparation_span)
    sctx.timing.end(preparation_span)
    prepared = strategy.finalize_preparation(context, receipts)

    assert prepared.host_hardware == hardware
    assert prepared.host_hardware is not hardware
    assert all(prepared.host_hardware[host] is hardware[host] for host in plan.host_list)
    unverified_receipts = {**receipts, "coldsnap.hardware": replace(hardware_receipt, verified=False)}
    assert strategy.finalize_preparation(context, unverified_receipts).host_hardware == {}
    assert strategy.name == "coldsnap"
    assert prepared.assets.prepare_model is prepare_model
    assert prepared.assets.run_builder is False
    assert prepared.assets.prepare_runtime is True
    assert prepared.assets.probe_images is False
    assert prepared.assets.distribute_images is False
    assert prepared.assets.images_by_node == (
        "registry/capsule0@sha256:" + "d" * 64,
        "registry/capsule1@sha256:" + "d" * 64,
    )
    spans = {span["name"]: span for span in sctx.timing.export()["spans"]}
    assert spans["coldsnap.hardware"]["parent"] == preparation_span
    assert spans["coldsnap.artifact"]["parent"] == preparation_span
    assert spans["coldsnap.weights"]["parent"] == preparation_span
    assert spans["coldsnap.capsules"]["parent"] == preparation_span


def test_coldsnap_prepare_only_receipt_precedes_activation(tmp_path, monkeypatch):
    _recipe, options, plan, sctx = _setup()
    sctx.config.cache_dir = str(tmp_path)
    store = resolve_artifact_store(plan=plan, options=options, sctx=sctx)
    monkeypatch.setattr(plan.runtime, "get_cluster_env", lambda **_kwargs: {})
    artifact = _write_strategy_artifact(store.current)
    artifact["launch"] = {
        "engine": "vllm",
        "units": [{"id": "unit-%d" % index, "environment": {"OMP_NUM_THREADS": "4"}} for index in range(2)],
    }
    store.current.write_text(json.dumps(artifact))
    calls = []

    def run_command(arguments, **kwargs):
        calls.append((arguments, json.loads(kwargs["input"])))
        request = calls[-1][1]
        assert all(unit["environment"]["OMP_NUM_THREADS"] == "4" for unit in request["launch"]["units"])
        output = ""
        if "--prepare-only" in arguments:
            output = json.dumps(
                {
                    "format": 1,
                    "kind": "coldsnap-restore-preparation",
                    "operation_id": request["id"],
                    "capture_id": "capture-one",
                    "provider": "recovery",
                    "snapshot_driver": {"id": "n610", "abi": 1},
                }
            )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.service.stage_native_packs",
        lambda request, **_kwargs: SimpleNamespace(request=request, selected_mode="recovery", failures=()),
    )
    service = ColdSnapService(
        "coldsnap-test",
        run_command=run_command,
        target_tool_resolver=lambda **_kwargs: SimpleNamespace(environment={}),
    )
    execution = ExecutionContext(options=options, plan=plan, sctx=sctx)
    descriptor = service.describe_restore(execution)
    state = service.stage_restore(execution, descriptor)
    capsule_receipt = service.prepare_capsules(execution, state)
    prepared = service.finalize_restore(
        execution,
        {"coldsnap.weights": state, "coldsnap.capsules": capsule_receipt},
    )
    from sparkrun.core.execution import ActivationContext

    activation_context = ActivationContext(
        execution=execution,
        prepared=prepared,
        cluster_id=plan.cluster_id,
        hosts=plan.host_list,
        container_image=prepared.assets.images_by_node[0],
        images_by_node=prepared.assets.images_by_node,
        effective_cache_dir="/cache/hf",
        serve_port=8000,
        serve_command="vllm serve",
    )
    receipt = service.prepare_activation(activation_context)
    result = service.activate(activation_context, receipt)

    assert "--prepare-only" in calls[0][0]
    assert "--receipt-json" in calls[0][0]
    assert "--prepare-only" not in calls[1][0]
    assert result.rc == 0
    assert result.runtime_info["weight_provider"] == "recovery"


def test_coldsnap_recipe_rejects_coerced_types_and_internal_payload_paths():
    register(None)
    base = {
        "recipe_version": "2",
        "model": "Qwen/Qwen3.5-0.8B",
        "model_revision": "model-commit",
        "runtime": "vllm-distributed",
        "container": "org/capsule@sha256:%s" % ("a" * 64),
    }
    with pytest.raises(Exception, match="kv_discard must be a boolean"):
        Recipe.from_dict({**base, "coldsnap": {"process": {"kv_discard": "false"}}})
    with pytest.raises(Exception, match="format must be an integer"):
        Recipe.from_dict({**base, "coldsnap": {"format": 1.0}})
    with pytest.raises(Exception, match="cache.paths must be a list of strings"):
        Recipe.from_dict({**base, "coldsnap": {"cache": {"paths": "/root/.cache"}}})
    with pytest.raises(Exception, match=r"artifact must be a mapping"):
        Recipe.from_dict({**base, "coldsnap": {"artifact": "./legacy.json"}})
    with pytest.raises(Exception, match=r"weights\.native has unknown field.*publish"):
        Recipe.from_dict({**base, "coldsnap": {"weights": {"native": {"publish": True}}}})

    defaults = Recipe.from_dict({**base, "coldsnap": {}})
    assert defaults.plugin_item("coldsnap").weight_mode is None
    assert defaults.plugin_item("coldsnap").recovery.loader_backend is None
    assert defaults.plugin_item("coldsnap").process_backend is None
    assert defaults.plugin_item("coldsnap").enforce_captured_driver_floor is False
    assert defaults.plugin_item("coldsnap").health_path is None

    with pytest.raises(Exception, match="compatibility.enforce_captured_driver_floor must be a boolean"):
        Recipe.from_dict({**base, "coldsnap": {"compatibility": {"enforce_captured_driver_floor": "false"}}})

    unsupported = Recipe.from_dict({**base, "coldsnap": {"weights": {"recovery": {"loader_backend": "swizzler"}}}})
    assert "coldsnap.weights.recovery.loader_backend is unsupported" in unsupported.validate()

    with pytest.raises(Exception, match=r"weights\.native has unknown field.*files_by_worker"):
        Recipe.from_dict(
            {
                **base,
                "coldsnap": {
                    "weights": {
                        "native": {
                            "repository": "org/packs",
                            "revision": "commit",
                            "files_by_worker": {"worker-0": "rank0.weights"},
                        }
                    }
                },
            }
        )

    recipe = Recipe.from_dict(
        {
            **base,
            "coldsnap": {
                "cache": {"paths": ["/root/.cache/huggingface"]},
            },
        }
    )
    assert "coldsnap.cache.paths is invalid" in recipe.validate()

    recipe = Recipe.from_dict(
        {
            **base,
            "coldsnap": {
                "capsule": {"repository": "registry.example/capsules:latest"},
            },
        }
    )
    assert "coldsnap.capsule.repository is invalid" in recipe.validate()


def test_coldsnap_recipe_rejects_sparkrun_and_runtime_owned_env():
    register(None)
    recipe = Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "Qwen/Qwen3.5-0.8B",
            "model_revision": "model-commit",
            "runtime": "vllm-distributed",
            "container": "org/capsule@sha256:%s" % ("a" * 64),
            "env": {
                "COLDSNAP_RECOVERY_LOADER_BACKEND": "torch",
                "CUDA_CACHE_DISABLE": "0",
                "CUDA_CACHE_MAXSIZE": "1073741824",
                "CUDA_CACHE_PATH": "/tmp/coldsnap-derived-cache/cuda",
                "HF_HUB_OFFLINE": "1",
                "NCCL_IB_HCA": "mlx5_0",
                "NCCL_CUMEM_ENABLE": "1",
                "VLLM_CACHE_ROOT": "/recipe/cache",
            },
            "coldsnap": {},
        }
    )

    issues = recipe.validate()
    assert any("sparkrun-owned communication variables: NCCL_IB_HCA" in issue for issue in issues)
    assert any("sparkrun-owned runtime variables: HF_HUB_OFFLINE" in issue for issue in issues)
    assert any("ColdSnap-owned runtime variables:" in issue for issue in issues)
    assert any("NCCL_CUMEM_ENABLE" in issue for issue in issues)
    assert any("COLDSNAP_RECOVERY_LOADER_BACKEND" in issue for issue in issues)
    assert any("CUDA_CACHE_DISABLE" in issue for issue in issues)
    assert any("CUDA_CACHE_MAXSIZE" in issue for issue in issues)
    assert any("CUDA_CACHE_PATH" in issue for issue in issues)
    assert any("VLLM_CACHE_ROOT" in issue for issue in issues)


@pytest.mark.parametrize("wrong_job,wrong_capture", [(True, False), (False, True)])
def test_lifecycle_service_rejects_mismatched_job_or_capture(monkeypatch, tmp_path, wrong_job, wrong_capture):
    _recipe, options, plan, sctx = _setup()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"kind": "coldsnap-snapshot-artifact", "capture_id": "capture"}))
    target = "sparkrun_%s_%s" % (plan.intent_id, "d" * 12)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.service.verify_coldsnap_hosts", lambda *_, **__: SimpleNamespace(snapshot_driver="n610"))
    monkeypatch.setattr("sparkrun.api._resolve.discover_cluster_id_by_intent", lambda *_, **__: target)
    monkeypatch.setattr(ColdSnapService, "_invoke", lambda *_, **__: pytest.fail("mismatched lifecycle invoked controller"))
    with pytest.raises(RuntimeError, match="different job|activation receipt"):
        ColdSnapService().execute_lifecycle(
            "sleep",
            options,
            plan=plan,
            sctx=sctx,
            artifact=str(artifact),
            expected_cluster_id="other-job" if wrong_job else target,
            expected_capture_id="other-capture" if wrong_capture else "capture",
        )


def test_job_control_uses_saved_recipe_and_exact_activation_receipt(monkeypatch):
    from sparkrun.plugins.coldsnap.api import control_job

    recipe, _, plan, sctx = _setup()
    job = SimpleNamespace(
        cluster_id="exact-job",
        hosts=tuple(plan.host_list),
        metadata={
            "cluster": "lab",
            "recipe_state": recipe.__getstate__(),
            "port": 8111,
            "runtime_info": {
                "execution_strategy": "coldsnap",
                "capture_id": "exact-capture",
                "snapshot_driver": "n610",
                "artifact": "/test/committed.json",
            },
        },
    )
    captured = {}

    def planning(options, **kwargs):
        assert options.dry_run and options.overrides["port"] == 8111
        assert options.hosts == tuple(plan.host_list)
        return plan

    def execute(self, action, options, **kwargs):
        captured.update(kwargs)
        return {}, {"state": "sleeping", "cluster_id": "exact-job", "capture_id": "exact-capture"}

    monkeypatch.setattr("sparkrun.api.plan", planning)
    monkeypatch.setattr(ColdSnapService, "execute_lifecycle", execute)
    assert control_job("sleep", job, sctx=sctx)["state"] == "sleeping"
    assert captured["expected_cluster_id"] == "exact-job"
    assert captured["expected_capture_id"] == "exact-capture"
    assert captured["artifact"] == "/test/committed.json"


@pytest.mark.parametrize("operation", ["capture", "restore", "warm", "status"])
def test_cli_reports_planning_errors_without_traceback(monkeypatch, operation):
    from click.testing import CliRunner
    from sparkrun.core.timing import STATUS_ERROR

    _recipe, _options, _plan, sctx = _setup()
    finished = []
    message = "Recipe is invalid: include: '_missing.yaml' not found next to recipe.yaml"
    monkeypatch.setattr("sparkrun.api._context.default_sctx", lambda: sctx)
    monkeypatch.setattr("sparkrun.plugins.coldsnap.cli._begin_operation_timing", lambda *a, **kw: finished.append)

    def fail_plan(*args, **kwargs):
        raise api.SparkrunError(message)

    monkeypatch.setattr(api, "plan", fail_plan)
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.cli.ColdSnapService",
        lambda *a, **kw: pytest.fail("invalid plan started a ColdSnap operation"),
    )
    result = CliRunner().invoke(build_command(), [operation, "recipe.yaml", "--cluster", "g580"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error: " + message in result.output
    assert "Traceback" not in result.output
    assert finished == [STATUS_ERROR]
