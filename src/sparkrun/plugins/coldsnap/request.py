# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Translate a sparkrun plan and ``coldsnap:`` item into ColdSnap JSON."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sparkrun.api as api
from sparkrun.core.log_source import SERVE_LOG_PATH
from sparkrun.plugins.coldsnap.artifacts import _read_committed_artifact, resolve_artifact_store
from sparkrun.plugins.coldsnap.config import ColdSnapRecipe
from sparkrun.plugins.coldsnap.environment import RecipeEnvironment
from sparkrun.plugins.coldsnap.policy import resolve_coldsnap_policy


def build_request(
    operation: str,
    options: api.RunOptions,
    *,
    plan: api.RunPlan,
    artifact: str = "",
    output: str = "",
    weight_mode: str | None = None,
    comm_env=None,
    sctx=None,
    snapshot_driver: str = "n610",
    images_by_node: Sequence[str] | None = None,
    operation_id: str = "",
    native_repository: str = "",
    native_revision: str = "",
    native_materialization: str | None = None,
    artifact_scope: str = "portable",
    activation_state: str = "running",
    workload_cluster_id: str = "",
    defer_environment: bool = False,
) -> dict:
    if operation not in {"capture", "publish", "publish-native", "restore", "sleep", "wake", "status"}:
        raise ValueError("operation must be capture, publish, publish-native, restore, sleep, wake, or status")
    config = plan.recipe.plugin_item("coldsnap")
    if not isinstance(config, ColdSnapRecipe):
        raise ValueError("recipe does not declare a valid top-level coldsnap item")
    if operation == "publish" and not config.capsule.repository:
        raise ValueError("coldsnap.capsule.repository is required for publication")
    if operation == "publish-native" and (not native_repository or not native_revision):
        raise ValueError("native publication requires --hf-repo and --revision")
    if native_materialization is not None and native_materialization not in {"off", "async", "required"}:
        raise ValueError("native materialization must be off, async, or required")
    if operation == "restore" and native_materialization is None:
        # Ordinary launches consume prepared assets, without implicitly writing
        # native packs. Emit this even for older controllers that default to
        # async; explicit materialize operations supply their required policy.
        native_materialization = "off"
    if artifact_scope not in {"portable", "target-local"}:
        raise ValueError("artifact_scope must be portable or target-local")
    if artifact_scope == "target-local" and (operation != "capture" or snapshot_driver != "n580"):
        raise ValueError("target-local artifact scope currently requires n580 capture")
    if activation_state not in {"running", "warm"}:
        raise ValueError("activation_state must be running or warm")
    if operation != "restore" and activation_state != "running":
        raise ValueError("warm activation_state is valid only for restore")
    if activation_state == "warm" and snapshot_driver != "n610":
        raise ValueError("warm activation currently requires snapshot driver n610")
    issues = plan.recipe.validate()
    coldsnap_issues = [issue for issue in issues if issue.startswith("coldsnap.")]
    if coldsnap_issues:
        raise ValueError("invalid coldsnap recipe: %s" % "; ".join(coldsnap_issues))
    artifact_store = resolve_artifact_store(plan=plan, options=options, sctx=sctx, snapshot_driver=snapshot_driver)
    environment = RecipeEnvironment(
        operation,
        options,
        plan=plan,
        sctx=sctx,
        artifact=artifact or str(artifact_store.current),
        deferred=defer_environment,
    )
    spec = api.materialize(
        options,
        plan=plan,
        comm_env=comm_env,
        sctx=sctx,
        images_by_node=images_by_node,
        env_template_resolver=environment.resolve,
    )
    if spec.engine not in {"vllm", "sglang"}:
        raise ValueError("ColdSnap requires a vLLM or SGLang recipe; got %s" % spec.engine)
    if artifact_scope == "target-local" and spec.engine != "vllm":
        raise ValueError("target-local artifact scope currently requires vLLM")
    if spec.engine == "sglang" and native_materialization not in {None, "off"}:
        raise ValueError(
            "SGLang does not support native model-payload materialization mode %s; use off or pre-publish native payloads"
            % native_materialization
        )
    if not spec.model_revision:
        raise ValueError("ColdSnap requires an immutable model_revision")
    unpinned = [unit.id for unit in spec.units if not unit.image_digest]
    if unpinned:
        raise ValueError("ColdSnap requires digest-pinned per-unit OCI images; unpinned unit(s): %s" % ", ".join(unpinned))
    if operation_id and not _VALID_OPERATION_ID.fullmatch(operation_id):
        raise ValueError("ColdSnap operation_id must contain 1-128 safe identifier characters")
    request_id = operation_id or _request_id(plan.cluster_id, operation)
    if snapshot_driver not in {"n580", "n610"}:
        raise ValueError("snapshot_driver must be n580 or n610")
    mode = weight_mode if weight_mode is not None else config.weight_mode
    site_policy = resolve_coldsnap_policy(
        cluster=plan.cluster,
        sctx=sctx,
        hosts=list(plan.host_list),
        probe_remote=False,
    )
    launch_units = []
    for unit in spec.units:
        launch_units.append(
            {
                "id": unit.id,
                "index": unit.index,
                "host": unit.host,
                "devices": list(unit.devices),
                "image": unit.image,
                "image_digest": unit.image_digest,
                "command": _coldsnap_command(unit.command, spec.engine),
                "environment": dict(unit.environment),
                "mounts": [asdict(mount) for mount in unit.mounts],
            }
        )
    environment.stamp(launch_units)
    process_policy = {
        key: setting
        for key, setting in {
            "backend": config.process_backend,
            "kv_discard": config.kv_discard,
            "async_graphs": config.async_graphs,
            "graph_policy": config.graph_policy,
            "shape_calibration": config.shape_calibration,
            # Portable is the coordinator default. Only emit the manager's
            # explicit target-local capture specialization.
            "artifact_scope": artifact_scope if artifact_scope != "portable" else None,
        }.items()
        if setting is not None
    }
    native_policy = {
        "repository": native_repository if operation == "publish-native" else config.native.repository,
        "revision": native_revision if operation == "publish-native" else config.native.revision,
    }
    native_policy = {key: value for key, value in native_policy.items() if value}
    if native_materialization is not None:
        native_policy["materialize"] = native_materialization
    weights_policy: dict[str, Any] = {"native": native_policy}
    if mode is not None:
        weights_policy["mode"] = mode
    recovery_read = config.recovery.loader_backend
    if recovery_read is None and site_policy.sources.get("io.recovery_read") != "coldsnap-default":
        recovery_read = site_policy.recovery_read
    if recovery_read is not None:
        weights_policy["recovery"] = {"loader_backend": recovery_read}
    policy = {
        "weights": weights_policy,
        "cache": {"seed": config.cache_seed, "paths": list(config.cache_paths)},
        "capsule": {"repository": config.capsule.repository},
        "compatibility": {
            "enforce_captured_driver_floor": config.enforce_captured_driver_floor,
        },
    }
    if process_policy:
        policy["process"] = process_policy
    validation_policy = {
        key: setting
        for key, setting in {
            "health_path": config.health_path,
            "prompt": config.prompt,
            "expected": config.expected,
        }.items()
        if setting is not None
    }
    request = {
        "format": 4,
        "kind": "coldsnap-operation-request",
        "operation": operation,
        "id": request_id,
        "snapshot_driver": {"id": snapshot_driver},
        "launch": {
            "engine": spec.engine,
            "model": {"id": spec.model, "revision": spec.model_revision, "source": "huggingface"},
            "units": launch_units,
            "execution": asdict(spec.execution),
        },
        "policy": policy,
        "lifecycle": {"activation_state": activation_state},
        "workload": {
            "cluster_id": workload_cluster_id or plan.cluster_id,
            "intent_id": plan.intent_id,
            "recipe": getattr(plan.recipe, "qualified_name", None) or plan.recipe.name,
            "runtime": plan.runtime.runtime_name,
            "model": spec.model,
            "served_model_name": str(
                plan.recipe.build_config_chain(options.overrides).get("served_model_name") or plan.recipe.effective_served_model_name
            ),
            "log_path": SERVE_LOG_PATH,
        },
    }
    if validation_policy:
        request["validation"] = validation_policy
    if operation == "capture":
        request["output"] = output or str(artifact_store.capture_output(request_id))
    elif operation in {"restore", "sleep", "wake", "status"}:
        request["artifact"] = artifact or str(artifact_store.current)
    else:
        request["artifact"] = artifact or str(artifact_store.current)
        request["output"] = output or str(artifact_store.capture_output(request_id))
    if operation != "capture" and plan.runtime.runtime_name == "vllm-distributed":
        _preserve_legacy_thread_default(launch_units, Path(request["artifact"]))
    return request


def _preserve_legacy_thread_default(units: list[dict], artifact_path: Path) -> None:
    """Retain the removed host default for an already captured vLLM process.

    Older hosts injected OMP_NUM_THREADS=4 into distributed vLLM launches.
    An unset value on a newer host still restores those captured threads;
    explicitly requested values remain unchanged for controller validation.
    No other process environment or arbitrary thread value is inherited.
    """

    missing = {unit["id"]: unit for unit in units if "OMP_NUM_THREADS" not in unit["environment"]}
    if not missing:
        return
    path = artifact_path.expanduser()
    if path.is_dir():
        path /= "artifact.json"
    if not path.is_file():
        # Request-only rendering may precede artifact availability.
        return
    try:
        _, artifact = _read_committed_artifact(path.resolve())
    except RuntimeError:
        # This optional migration must not replace the service/controller's
        # authoritative artifact validation or lifecycle identity diagnostics.
        return
    launch = artifact.get("launch", {})
    if not isinstance(launch, Mapping) or launch.get("engine") != "vllm":
        return
    for captured in launch.get("units", []):
        if not isinstance(captured, Mapping):
            continue
        environment = captured.get("environment", {})
        unit = missing.get(captured.get("id"))
        if unit is not None and isinstance(environment, Mapping) and environment.get("OMP_NUM_THREADS") == "4":
            unit["environment"]["OMP_NUM_THREADS"] = "4"


def _request_id(cluster_id: str, operation: str) -> str:
    base = "".join(character if character.isalnum() or character in "._-" else "-" for character in cluster_id)
    base = base.strip(".-") or "sparkrun"
    suffix = "-" + operation
    if operation in {"capture", "publish", "publish-native"}:
        suffix += "-" + secrets.token_hex(6)
    return base[: 128 - len(suffix)] + suffix


_VALID_OPERATION_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def _coldsnap_command(command, engine: str) -> list[str]:
    values = list(command)
    if len(values) < 5 or values[:4] != ["bash", "--noprofile", "--norc", "-c"]:
        raise ValueError("ColdSnap requires sparkrun's resolved Bash command envelope")
    text = values[4].rstrip()
    if engine == "vllm" and not re.search(r"(?<!\S)--enable-sleep-mode(?:\s|$)", text):
        text += " --enable-sleep-mode"
    if engine == "sglang" and not re.search(r"(?<!\S)--enable-memory-saver(?:\s|$)", text):
        text += " --enable-memory-saver"
    values[4] = text
    return values


__all__ = ["build_request"]
