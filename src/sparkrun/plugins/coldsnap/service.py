# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Reusable ColdSnap capture/publish/restore service used by CLI and run strategy."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from sparkrun.core.config import resolve_hf_token
from sparkrun.core.execution import (
    ActivationContext,
    ActivationResult,
    ExecutionContext,
    LaunchAssetPolicy,
    PreparationStep,
    PreparedExecution,
)
from sparkrun.core.progress import PROGRESS, progress_heartbeat
from sparkrun.core.timing import timed
from sparkrun.plugins.coldsnap.environment import validate_environment_recipe
from sparkrun.plugins.coldsnap.artifacts import promote_generation, resolve_artifact_store, resolve_generation_limit
from sparkrun.plugins.coldsnap.compatibility import (
    SNAPSHOT_DRIVER_N610,
    ColdSnapHardwareReceipt,
    verify_coldsnap_hosts,
)
from sparkrun.plugins.coldsnap.controller_process import ControllerResult, run_controller
from sparkrun.plugins.coldsnap.host_provider import ColdSnapHostProvider
from sparkrun.plugins.coldsnap.target_tools import prepare_target_tools
from sparkrun.plugins.coldsnap.local_overlays import (
    promote_local_materialization,
    promote_local_overlay,
    select_local_materialization,
    select_local_overlay,
    target_identity,
    target_key,
    validate_materialization_source,
)
from sparkrun.plugins.coldsnap.oci_artifacts import (
    configured_artifact_reference,
    default_artifact_publish_reference,
    publish_oci_artifact,
    stage_oci_artifact,
)
from sparkrun.plugins.coldsnap.policy import resolve_coldsnap_policy
from sparkrun.plugins.coldsnap.providers import (
    StageOutcome,
    native_pack_status,
    resolve_request_weight_mode,
    stage_native_packs,
)
from sparkrun.plugins.coldsnap.request import build_request
from sparkrun.plugins.coldsnap.runtime_cache import (
    CaptureRuntimeCacheStage,
    cleanup_capture_runtime_cache,
    stage_capture_runtime_cache,
)
from sparkrun.plugins.coldsnap.timing import (
    OperationTimingEventStream,
    OperationTimingProgress,
    follow_operation_timing_events,
    import_operation_timing,
    read_operation_receipt,
    startup_observation,
)
from sparkrun.plugins.coldsnap.tool import (
    ControllerTool,
    ensure_controller_tool,
    explicit_controller_environment,
)

RunCommand = Callable[..., subprocess.CompletedProcess]
ToolResolver = Callable[[Any], ControllerTool]
logger = logging.getLogger(__name__)


def replace_capture_workload(*, plan, sctx) -> tuple[str, ...]:
    """Strictly replace an overlapping deployment immediately before capture.

    Image/model/cache preparation is intentionally complete by this point.
    ColdSnap capture launches host-network containers itself, outside the
    normal launcher ``before_start`` hook, so leaving an earlier deployment
    alive would make rank 0 fail its service-port bind and leave rank 1 waiting
    for the full capture timeout.
    """
    from sparkrun.api._run import _evict_superseded_deployments

    replacement_options = {"strict": True}
    if "include_current" in signature(_evict_superseded_deployments).parameters:
        # Capture submits its own containers instead of invoking the ordinary
        # runtime's same-ID cleanup. 0.4 exposes that replacement explicitly.
        replacement_options["include_current"] = True
    evicted, _observed = _evict_superseded_deployments(
        intent_id=plan.intent_id,
        cluster_id_for_launch=plan.cluster_id,
        candidate_hosts=list(plan.candidate_hosts),
        target_hosts=list(plan.host_list),
        cluster_def=plan.cluster,
        config=sctx.config,
        sctx=sctx,
        **replacement_options,
    )
    if evicted:
        logger.log(PROGRESS, "ColdSnap: replaced %d active workload(s) before capture", len(evicted))
    return tuple(evicted)


def prepare_capture_images(options, *, plan, sctx, snapshot_driver: str):
    """Build, distribute, and content-pin the images used for capture."""
    from sparkrun.core.image_preparation import prepare_images, stage_prepared_images
    from sparkrun.orchestration.distribution import resolve_auto_transfer_mode
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    hosts = list(plan.host_list)
    # Host 0.4 scopes connection settings per operation. Preserve that contract
    # on older supported hosts as well, without modifying the caller's config.
    config = sctx.config
    if callable(getattr(config, "for_cluster", None)):
        config = config.for_cluster(plan.cluster)
    elif plan.cluster.user:
        config = copy(config)
        config.ssh_user = plan.cluster.user
    ssh_kwargs = build_ssh_kwargs(config)
    # Runtime hooks declare auxiliary models on the recipe. Keep those changes
    # and legacy host image-plan updates local to this staging transaction.
    recipe = deepcopy(plan.recipe)
    topology = options.topology or plan.cluster.topology
    requested_transfer = options.transfer_mode or plan.cluster.transfer_mode or "auto"
    transfer = resolve_auto_transfer_mode(
        requested_transfer,
        hosts,
        ssh_kwargs=ssh_kwargs,
        dry_run=False,
        topology=topology,
    )
    engine = "sglang" if plan.runtime.runtime_name == "sglang" else "vllm"
    prepare_options = {
        "config": config,
        "v": getattr(sctx, "variables", None),
        "cluster": plan.cluster,
        "dry_run": False,
        "transfer_mode": transfer.mode,
        "ssh_kwargs": ssh_kwargs,
        "run_builder": True,
        "builder_context": {"snapshot_driver": snapshot_driver, "engine": engine},
    }
    # 0.4 removed this unused argument. Keep the supported 0.3 host contract
    # explicit without retrying a possibly side-effecting builder on TypeError.
    if "overrides" in signature(prepare_images).parameters:
        prepare_options["overrides"] = dict(options.overrides)
    prepared = prepare_images(recipe, plan.runtime, hosts, **prepare_options)
    # Capture launches the real engine rather than entering the normal
    # launcher pipeline. Run the engine's shared asset declaration hook here
    # so auxiliary models (for example SGLang/vLLM speculative drafts) join the
    # primary pinned snapshot in the same distribution transaction.
    plan.runtime.prepare(
        recipe,
        hosts,
        config=config,
        dry_run=False,
        transfer_mode=transfer.mode,
        overrides=dict(options.overrides),
    )
    cache_dir = options.cache_dir or plan.cluster.cache_dir or str(config.hf_cache_dir)
    return stage_prepared_images(
        prepared,
        recipe,
        hosts,
        cache_dir,
        config,
        dry_run=False,
        recipe_name=plan.recipe.name,
        transfer_mode=transfer.mode,
        transfer_interface=options.transfer_interface or plan.cluster.transfer_interface,
        local_cache_dir=options.local_cache_dir,
        pre_ib=transfer,
        topology=topology,
        require_content_ids=True,
        ssh_kwargs=ssh_kwargs,
        stage_models=True,
        timeline=getattr(sctx, "timing", None),
    )


@dataclass(frozen=True)
class RestoreDescriptor:
    request: dict[str, Any]
    artifact: dict[str, Any]


@dataclass(frozen=True)
class PreparedRestore:
    request: dict[str, Any]
    artifact: dict[str, Any]
    selected_mode: str
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class RestoreActivationReceipt:
    request: dict[str, Any]
    receipt: dict[str, Any]


class ColdSnapService:
    """One implementation of explicit and strategy-driven ColdSnap operations."""

    def __init__(
        self,
        binary: str = "",
        *,
        run_command: RunCommand = run_controller,
        tool_resolver: ToolResolver = ensure_controller_tool,
        host_provider_factory=ColdSnapHostProvider,
        target_tool_resolver=prepare_target_tools,
    ):
        self.binary = binary
        self.run_command = run_command
        self.tool_resolver = tool_resolver
        self.host_provider_factory = host_provider_factory
        self.target_tool_resolver = target_tool_resolver

    def describe_restore(
        self,
        context: ExecutionContext,
        hardware: ColdSnapHardwareReceipt | None = None,
    ) -> RestoreDescriptor:
        selected_driver: str = getattr(hardware, "snapshot_driver", SNAPSHOT_DRIVER_N610)
        snapshot_driver = selected_driver
        strategy_options = _strategy_options(context)
        artifact = self._restore_artifact_path(
            context.options,
            plan=context.plan,
            sctx=context.sctx,
            explicit=str(strategy_options.get("artifact") or ""),
            dry_run=context.options.dry_run,
            snapshot_driver=snapshot_driver,
        )
        requested_weight_mode = str(strategy_options.get("weights") or "") or None
        request = build_request(
            "restore",
            context.options,
            plan=context.plan,
            sctx=context.sctx,
            artifact=str(artifact),
            weight_mode=requested_weight_mode,
            native_materialization=str(strategy_options.get("materialize_native") or "") or None,
            activation_state=str(strategy_options.get("activation_state") or "running"),
            snapshot_driver=snapshot_driver,
        )
        resolved_weight_mode = resolve_request_weight_mode(request)
        sglang = request["launch"]["engine"] == "sglang"
        local_asset_kind = "local materialization" if sglang else "target-local residual overlay"
        if (
            hardware is not None
            and getattr(hardware, "verified", False)
            and not str(strategy_options.get("artifact") or "")
            # vLLM's target-local residual overlays are captured without a native
            # replay manifest. Explicit/cache-only native restores must keep
            # using the portable capsule while reusing any node-local payload.
            and (sglang or (snapshot_driver == "n580" and resolved_weight_mode not in {"native", "cache-only-auto"}))
        ):
            store = resolve_artifact_store(
                plan=context.plan,
                options=context.options,
                sctx=context.sctx,
                snapshot_driver=snapshot_driver,
            )
            try:
                selector = select_local_materialization if sglang else select_local_overlay
                overlay = selector(
                    store,
                    Path(artifact),
                    hardware=hardware.hardware,
                    hosts=context.plan.host_list,
                    snapshot_driver=snapshot_driver,
                    **({"require_native": resolved_weight_mode in {"native", "cache-only-auto"}} if sglang else {}),
                )
            except RuntimeError as error:
                # A target-local overlay is a disposable acceleration cache.
                # A stale/corrupt entry must never make the portable artifact
                # unavailable; explicit materialization can replace it later.
                logger.warning("ColdSnap: ignoring unusable %s: %s", local_asset_kind, error)
                overlay = None
            if overlay is not None:
                logger.log(PROGRESS, "ColdSnap: using verified %s %s", local_asset_kind, overlay)
                artifact = overlay
                # Auto mode normally prefers a verified native cache. This
                # overlay owns only recovery residual state, so make the
                # compatible provider decision before native staging.
                if resolved_weight_mode == "auto" and not sglang:
                    requested_weight_mode = "recovery"
                request = build_request(
                    "restore",
                    context.options,
                    plan=context.plan,
                    sctx=context.sctx,
                    artifact=str(artifact),
                    weight_mode=requested_weight_mode,
                    native_materialization=str(strategy_options.get("materialize_native") or "") or None,
                    activation_state=str(strategy_options.get("activation_state") or "running"),
                    snapshot_driver=snapshot_driver,
                )
        artifact_path = _artifact_file(request["artifact"])
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("read committed ColdSnap artifact %s: %s" % (artifact_path, error)) from error
        if artifact.get("kind") != "coldsnap-snapshot-artifact" or artifact.get("state") != "committed":
            raise RuntimeError("ColdSnap artifact is not a committed snapshot: %s" % artifact_path)
        artifact_driver = artifact.get("snapshot_driver", {}).get("id")
        if artifact_driver != snapshot_driver:
            raise RuntimeError(
                "ColdSnap artifact uses snapshot driver %s, but sparkrun selected %s" % (artifact_driver or "<missing>", snapshot_driver)
            )
        return RestoreDescriptor(request=request, artifact=artifact)

    def stage_restore(self, context: ExecutionContext, descriptor: RestoreDescriptor) -> PreparedRestore:
        if context.options.dry_run:
            request = deepcopy(descriptor.request)
            requested = resolve_request_weight_mode(request)
            selected = "native" if requested == "native" else "recovery"
            request["policy"]["weights"]["mode"] = selected
            return PreparedRestore(request, descriptor.artifact, selected)
        outcome = stage_native_packs(
            descriptor.request,
            plan=context.plan,
            sctx=context.sctx,
            payload_verifier=lambda: self._payload_verifier_path(
                context.plan,
                context.sctx,
                binary=str(_strategy_options(context).get("binary") or ""),
            ),
        )
        if outcome.failures:
            logger.warning(
                "ColdSnap model payloads unavailable; using safetensors recovery: %s",
                "; ".join(outcome.failures),
            )
        return PreparedRestore(
            request=outcome.request,
            artifact=descriptor.artifact,
            selected_mode=outcome.selected_mode,
            failures=outcome.failures,
        )

    def finalize_restore(self, context: ExecutionContext, receipts) -> PreparedExecution:
        prepared = receipts["coldsnap.weights"]
        if not isinstance(prepared, PreparedRestore):
            raise TypeError("ColdSnap weight preparation receipt is invalid")
        images = _capsule_images(
            prepared.artifact,
            prepared.request["launch"]["units"],
            context.plan.host_list,
        )
        return PreparedExecution(
            strategy="coldsnap",
            assets=LaunchAssetPolicy(
                images_by_node=images,
                # ColdSnap prepare-only has already pulled or verified each
                # capsule on its assigned host. Local-only image IDs do not
                # exist on the controller and must not enter normal fan-out.
                distribute_images=False,
                prepare_model=prepared.selected_mode != "native",
                run_builder=False,
                # Runtime preparation declares auxiliary launch assets such as
                # speculative draft models. Image conversion remains disabled
                # independently by run_builder=False, so recovery restores
                # stage every pinned model without rebuilding their capsules.
                prepare_runtime=True,
                probe_images=False,
                sync_tuning=False,
                clear_page_cache=False,
            ),
            state=prepared,
            receipts=dict(receipts),
            host_hardware=(
                dict(getattr(receipts.get("coldsnap.hardware"), "hardware", {}))
                if getattr(receipts.get("coldsnap.hardware"), "verified", False)
                else {}
            ),
        )

    def prepare_capsules(self, context: ExecutionContext, state: PreparedRestore) -> RestoreActivationReceipt:
        if context.options.dry_run:
            return RestoreActivationReceipt(
                request=state.request,
                receipt={"provider": state.selected_mode, "capture_id": state.artifact.get("capture_id"), "dry_run": True},
            )
        completed = self._invoke(
            state.request,
            prepare_only=True,
            capture_output=True,
            binary=str(_strategy_options(context).get("binary") or ""),
            sctx=context.sctx,
            cluster=context.plan.cluster,
            offline=context.options.offline,
        )
        try:
            receipt = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("ColdSnap prepare-only returned an invalid receipt") from error
        if receipt.get("kind") != "coldsnap-restore-preparation" or receipt.get("operation_id") != state.request["id"]:
            raise RuntimeError("ColdSnap prepare-only receipt does not match this restore")
        if receipt.get("provider") != state.selected_mode:
            raise RuntimeError("ColdSnap selected %s weights after sparkrun prepared %s" % (receipt.get("provider"), state.selected_mode))
        if receipt.get("snapshot_driver", {}).get("id") != state.request["snapshot_driver"]["id"]:
            raise RuntimeError("ColdSnap prepare-only receipt returned a different snapshot driver")
        return RestoreActivationReceipt(request=state.request, receipt=receipt)

    def prepare_activation(self, context: ActivationContext) -> RestoreActivationReceipt:
        state = context.prepared.state
        if not isinstance(state, PreparedRestore):
            raise TypeError("ColdSnap prepared execution state is invalid")
        prepared_capsules = context.prepared.receipts.get("coldsnap.capsules")
        if not isinstance(prepared_capsules, RestoreActivationReceipt):
            raise TypeError("ColdSnap capsule preparation receipt is invalid")
        request = build_request(
            "restore",
            context.execution.options,
            plan=context.execution.plan,
            artifact=state.request["artifact"],
            weight_mode=state.selected_mode,
            comm_env=context.comm_env,
            sctx=context.execution.sctx,
            snapshot_driver=state.request["snapshot_driver"]["id"],
            activation_state=str(_strategy_options(context.execution).get("activation_state") or "running"),
        )
        # Preserve the exact staged inventory selected before shared launch
        # assets were prepared. Re-materialization above is only for the final
        # sparkrun-owned communication environment.
        request["policy"]["weights"] = deepcopy(state.request["policy"]["weights"])
        request["policy"]["weights"]["mode"] = state.selected_mode
        return RestoreActivationReceipt(request=request, receipt=prepared_capsules.receipt)

    def activate(self, context: ActivationContext, activation: RestoreActivationReceipt) -> ActivationResult:
        if not isinstance(activation, RestoreActivationReceipt):
            raise TypeError("ColdSnap activation receipt is invalid")
        if context.execution.options.dry_run:
            return ActivationResult(0, {"execution_strategy": "coldsnap", "weight_provider": activation.receipt["provider"]})
        completed = self._invoke(
            activation.request,
            prepare_only=False,
            capture_output=False,
            binary=str(_strategy_options(context.execution).get("binary") or ""),
            sctx=context.execution.sctx,
            cluster=context.execution.plan.cluster,
            offline=context.execution.options.offline,
        )
        # Published 0.3.7 and older develop-next hosts lack this optional field.
        observation = getattr(completed, "startup_observation", {})
        readiness = (
            {"startup_observation": observation} if "startup_observation" in getattr(ActivationResult, "__dataclass_fields__", {}) else {}
        )
        return ActivationResult(
            completed.returncode,
            {
                "execution_strategy": "coldsnap",
                "weight_provider": str(activation.receipt["provider"]),
                "capture_id": str(activation.receipt.get("capture_id") or ""),
                "artifact": str(activation.request["artifact"]),
                "snapshot_driver": activation.request["snapshot_driver"]["id"],
                "lifecycle_state": activation.request.get("lifecycle", {}).get("activation_state", "running"),
                "inference_readiness": (
                    "accepted" if activation.request.get("lifecycle", {}).get("activation_state", "running") == "running" else "inactive"
                ),
            },
            **readiness,
        )

    def inspect_native_status(
        self,
        options,
        *,
        plan,
        sctx,
        artifact: str = "",
        render_only: bool = False,
        snapshot_driver: str | None = None,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        """Inspect asynchronous native-cache state without changing payloads."""

        timeline = getattr(sctx, "timing", None)
        with timed(timeline, "coldsnap.hardware", operation="native-status"):
            hardware = (
                ColdSnapHardwareReceipt(
                    hardware={},
                    verified=False,
                    snapshot_driver=snapshot_driver or SNAPSHOT_DRIVER_N610,
                )
                if render_only
                else verify_coldsnap_hosts(plan, sctx, snapshot_driver=snapshot_driver)
            )
        selected_driver = getattr(hardware, "snapshot_driver", SNAPSHOT_DRIVER_N610)
        with timed(timeline, "coldsnap.artifact", operation="native-status"):
            artifact_path = self._restore_artifact_path(
                options,
                plan=plan,
                sctx=sctx,
                explicit=artifact,
                dry_run=render_only,
                snapshot_driver=selected_driver,
            )
        request = build_request(
            "restore",
            options,
            plan=plan,
            sctx=sctx,
            artifact=str(artifact_path),
            snapshot_driver=selected_driver,
        )
        with timed(timeline, "coldsnap.weights.native_status", operation="native-status"):
            statuses = native_pack_status(
                request,
                plan=plan,
                sctx=sctx,
                probe_remote=not render_only,
            )
        return request, statuses

    def materialize_sglang(self, options, *, plan, sctx, artifact, hardware, native_weights, verify) -> Path:
        """Explicit capture/verify/promote; never recovery-loader write-behind."""
        snapshot_driver = hardware.snapshot_driver
        if str(plan.runtime.get_family()) != "sglang":
            raise ValueError("capture-based materialization requires SGLang")
        store = resolve_artifact_store(plan=plan, options=options, sctx=sctx, snapshot_driver=snapshot_driver)
        source = self._restore_artifact_path(
            options,
            plan=plan,
            sctx=sctx,
            explicit=artifact,
            dry_run=False,
            snapshot_driver=snapshot_driver,
        )
        validate_materialization_source(source)
        try:
            existing = select_local_materialization(
                store,
                source,
                hardware=hardware.hardware,
                hosts=plan.host_list,
                snapshot_driver=snapshot_driver,
                require_native=native_weights,
            )
        except RuntimeError as error:
            logger.warning("ColdSnap: replacing unusable local materialization: %s", error)
            existing = None
        if existing is not None:
            # Fail visibly if a previously verified artifact lost its runtime
            # assets; do not turn arbitrary inference failures into recaptures.
            verify(existing)
            return existing
        store.overlay_pending.mkdir(parents=True, exist_ok=True, mode=0o700)
        with TemporaryDirectory(prefix="materialize-sglang-", dir=store.overlay_pending) as temporary:
            output = Path(temporary) / "artifact.json"
            logger.log(
                PROGRESS, "ColdSnap: capturing SGLang local %s", "native weights and runtime state" if native_weights else "runtime state"
            )
            self.execute_explicit(
                "capture",
                options,
                plan=plan,
                sctx=sctx,
                output=str(output),
                weight_mode="auto" if native_weights else "recovery",
                snapshot_driver=snapshot_driver,
                # Use SGLang's existing capture boundary on both drivers, not
                # vLLM's n580 pre-worker-import residual optimization.
                artifact_scope="portable",
            )
            try:
                result = promote_local_materialization(
                    store,
                    source,
                    output,
                    hardware=hardware.hardware,
                    hosts=plan.host_list,
                    snapshot_driver=snapshot_driver,
                    require_native=native_weights,
                    verify=verify,
                )
            except BaseException:
                # Preserve the diagnostic descriptor, but never make a failed
                # verification selectable by run or overwrite the source.
                if output.is_file():
                    failed = store.overlay_pending / (output.parent.name + ".failed.json")
                    output.replace(failed)
                    logger.error("ColdSnap: unpromoted SGLang capture retained at %s", failed)
                raise
        logger.log(PROGRESS, "ColdSnap: verified SGLang local materialization %s", result)
        return result

    def materialize_local_overlay(
        self,
        options,
        *,
        plan,
        sctx,
        artifact: str = "",
        snapshot_driver: str | None = None,
        hardware: ColdSnapHardwareReceipt | None = None,
    ) -> tuple[Path | None, dict[str, Any] | None, bool]:
        """Build a target-compatible n580 residual overlay when useful."""

        hardware = hardware or verify_coldsnap_hosts(plan, sctx, snapshot_driver=snapshot_driver)
        snapshot_driver = hardware.snapshot_driver
        runtime_name = str(getattr(plan.runtime, "runtime_name", ""))
        if snapshot_driver != "n580" or not runtime_name.startswith("vllm"):
            return None, None, False
        store = resolve_artifact_store(
            plan=plan,
            options=options,
            sctx=sctx,
            snapshot_driver=snapshot_driver,
        )
        portable = self._restore_artifact_path(
            options,
            plan=plan,
            sctx=sctx,
            explicit=artifact,
            dry_run=False,
            snapshot_driver=snapshot_driver,
        )
        try:
            existing = select_local_overlay(
                store,
                portable,
                hardware=hardware.hardware,
                hosts=plan.host_list,
                snapshot_driver=snapshot_driver,
            )
        except RuntimeError as error:
            logger.warning("ColdSnap: replacing unusable target-local residual overlay: %s", error)
            existing = None
        if existing is not None:
            identity = target_identity(hardware.hardware, plan.host_list, snapshot_driver)
            record = json.loads(store.overlay_record(target_key(identity)).read_text(encoding="utf-8"))
            return existing, record, False

        store.overlay_pending.mkdir(parents=True, exist_ok=True, mode=0o700)
        with TemporaryDirectory(prefix="materialize-", dir=store.overlay_pending) as temporary:
            output = Path(temporary) / "artifact.json"
            logger.log(
                PROGRESS,
                "ColdSnap: capturing target-local residuals on %s",
                ", ".join(plan.host_list),
            )
            self.execute_explicit(
                "capture",
                options,
                plan=plan,
                sctx=sctx,
                output=str(output),
                # A local overlay owns only target-specific residual state.
                # Its effective descriptor retains the portable artifact's
                # weight-provider references during promotion, so exporting a
                # second model payload here wastes time and disk.
                weight_mode="recovery",
                render_only=False,
                snapshot_driver=snapshot_driver,
                artifact_scope="target-local",
            )
            overlay, record = promote_local_overlay(
                store,
                portable,
                output,
                hardware=hardware.hardware,
                hosts=plan.host_list,
                snapshot_driver=snapshot_driver,
            )
        if overlay is None:
            logger.log(PROGRESS, "ColdSnap: target residuals match the portable capsule; no overlay stored")
        else:
            logger.log(PROGRESS, "ColdSnap: stored target-local residual overlay %s", overlay)
        return overlay, record, True

    def execute_lifecycle(
        self,
        operation: str,
        options,
        *,
        plan,
        sctx,
        artifact: str = "",
        render_only: bool = False,
        snapshot_driver: str | None = None,
        expected_cluster_id: str = "",
        expected_capture_id: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Control one exact active ColdSnap workload selected by recipe intent."""
        if operation not in {"sleep", "wake", "status"}:
            raise ValueError("lifecycle operation must be sleep, wake, or status")
        logger.log(PROGRESS, "ColdSnap: resolving live workload for %s", operation)
        timeline = getattr(sctx, "timing", None)
        with timed(timeline, "coldsnap.hardware", operation=operation):
            hardware = (
                ColdSnapHardwareReceipt(hardware={}, verified=False, snapshot_driver=snapshot_driver or SNAPSHOT_DRIVER_N610)
                if render_only
                else verify_coldsnap_hosts(plan, sctx, snapshot_driver=snapshot_driver)
            )
        selected_driver: str = getattr(hardware, "snapshot_driver", SNAPSHOT_DRIVER_N610)
        snapshot_driver = selected_driver
        with timed(timeline, "coldsnap.artifact", operation=operation):
            artifact_path = self._restore_artifact_path(
                options,
                plan=plan,
                sctx=sctx,
                explicit=artifact,
                dry_run=render_only,
                snapshot_driver=snapshot_driver,
            )
            if not artifact and getattr(hardware, "verified", False) and str(plan.runtime.get_family()) == "sglang":
                store = resolve_artifact_store(plan=plan, options=options, sctx=sctx, snapshot_driver=snapshot_driver)
                try:
                    local = select_local_materialization(
                        store,
                        artifact_path,
                        hardware=hardware.hardware,
                        hosts=plan.host_list,
                        snapshot_driver=snapshot_driver,
                    )
                except RuntimeError as error:
                    logger.warning("ColdSnap: ignoring unusable local materialization for lifecycle control: %s", error)
                    local = None
                if local is not None:
                    artifact_path = local
        cluster_id = plan.cluster_id
        if not render_only:
            from sparkrun.api._resolve import discover_cluster_id_by_intent

            cluster_hosts = list(getattr(plan.cluster, "hosts", ()) or plan.host_list)
            with timed(timeline, "coldsnap.workload", operation=operation):
                cluster_id = discover_cluster_id_by_intent(
                    plan.intent_id,
                    cluster_hosts,
                    cluster_def=plan.cluster,
                    cache_dir=options.cache_dir,
                    sctx=sctx,
                )
        if expected_cluster_id and cluster_id != expected_cluster_id:
            raise RuntimeError("ColdSnap lifecycle resolved a different job; refusing to control it")
        request = build_request(
            operation,
            options,
            plan=plan,
            sctx=sctx,
            artifact=str(artifact_path),
            snapshot_driver=snapshot_driver,
            workload_cluster_id=cluster_id,
            defer_environment=render_only,
        )
        if render_only:
            return request, None
        artifact_file = _artifact_file(request["artifact"])
        with timed(timeline, "coldsnap.artifact.read", operation=operation):
            try:
                artifact_document = json.loads(artifact_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("read committed ColdSnap artifact %s: %s" % (artifact_file, error)) from error
        expected_capture = artifact_document.get("capture_id")
        if artifact_document.get("kind") != "coldsnap-snapshot-artifact" or not isinstance(expected_capture, str) or not expected_capture:
            raise RuntimeError("ColdSnap lifecycle artifact is not a snapshot descriptor")
        if expected_capture_id and expected_capture != expected_capture_id:
            raise RuntimeError("ColdSnap lifecycle artifact does not match the job's activation receipt")
        completed = self._invoke(
            request,
            prepare_only=False,
            capture_output=True,
            binary=str(options.strategy_options.get("binary") or "") if isinstance(options.strategy_options, Mapping) else "",
            sctx=sctx,
            cluster=plan.cluster,
            offline=options.offline,
        )
        try:
            report = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("ColdSnap lifecycle operation returned an invalid report") from error
        if (
            report.get("format") != 1
            or report.get("kind") != "coldsnap-inference-lifecycle"
            or report.get("engine") != request["launch"]["engine"]
            or report.get("operation_id") != request["id"]
            or report.get("operation") != operation
            or report.get("cluster_id") != cluster_id
            or report.get("capture_id") != expected_capture
            or report.get("snapshot_driver") != snapshot_driver
            or report.get("state") not in {"running", "sleeping", "warm", "failed"}
            or not isinstance(report.get("units"), list)
            or len(report["units"]) != len(request["launch"]["units"])
        ):
            raise RuntimeError("ColdSnap lifecycle report does not match this workload")
        return request, report

    def execute_explicit(
        self,
        operation: str,
        options,
        *,
        plan,
        sctx,
        artifact: str = "",
        output: str = "",
        weight_mode: str | None = None,
        comm_env=None,
        render_only: bool = False,
        native_repository: str = "",
        native_revision: str = "",
        snapshot_driver: str | None = None,
        artifact_scope: str = "portable",
    ) -> tuple[dict[str, Any], tuple[Path, ...], tuple[str, ...]]:
        validate_environment_recipe(plan.recipe)
        timeline = getattr(sctx, "timing", None)
        with timed(timeline, "coldsnap.hardware", operation=operation):
            hardware = (
                ColdSnapHardwareReceipt(hardware={}, verified=False, snapshot_driver=snapshot_driver or SNAPSHOT_DRIVER_N610)
                if render_only
                else verify_coldsnap_hosts(plan, sctx, snapshot_driver=snapshot_driver)
            )
        selected_driver: str = getattr(hardware, "snapshot_driver", SNAPSHOT_DRIVER_N610)
        snapshot_driver = selected_driver
        request = build_request(
            operation,
            options,
            plan=plan,
            artifact=artifact,
            output=output,
            weight_mode=weight_mode,
            comm_env=comm_env,
            sctx=sctx,
            native_repository=native_repository,
            native_revision=native_revision,
            snapshot_driver=snapshot_driver,
            artifact_scope=artifact_scope,
            defer_environment=True,
        )
        if render_only:
            return request, (), ()
        cache_stage = CaptureRuntimeCacheStage(request=request)
        if operation == "capture":
            with timed(timeline, "coldsnap.images", operation=operation):
                staged_images = prepare_capture_images(options, plan=plan, sctx=sctx, snapshot_driver=snapshot_driver)
            request = build_request(
                operation,
                options,
                plan=plan,
                output=request["output"],
                weight_mode=weight_mode,
                comm_env=staged_images.comm_env or comm_env,
                sctx=sctx,
                images_by_node=staged_images.content_images_by_node,
                operation_id=request["id"],
                snapshot_driver=snapshot_driver,
                artifact_scope=artifact_scope,
            )
            with timed(timeline, "coldsnap.runtime_cache", operation=operation):
                cache_stage = stage_capture_runtime_cache(
                    request,
                    options=options,
                    plan=plan,
                    sctx=sctx,
                    images_by_node=staged_images.content_images_by_node,
                )
            request = cache_stage.request
        try:
            managed_promotion = operation in {"capture", "publish", "publish-native"} and not output
            retention = resolve_generation_limit(sctx.config) if managed_promotion else None
            if operation in {"publish", "publish-native", "restore"}:
                with timed(timeline, "coldsnap.artifact", operation=operation):
                    if operation in {"publish-native", "restore"}:
                        request["artifact"] = str(
                            self._restore_artifact_path(
                                options,
                                plan=plan,
                                sctx=sctx,
                                explicit=artifact,
                                dry_run=False,
                                snapshot_driver=snapshot_driver,
                            )
                        )
                    _artifact_file(request["artifact"])
            if operation != "capture":
                request = build_request(
                    operation,
                    options,
                    plan=plan,
                    artifact=request["artifact"],
                    output=request.get("output", ""),
                    weight_mode=weight_mode,
                    comm_env=comm_env,
                    sctx=sctx,
                    native_repository=native_repository,
                    native_revision=native_revision,
                    snapshot_driver=snapshot_driver,
                    artifact_scope=artifact_scope,
                    operation_id=request["id"],
                )
            if operation in {"publish", "publish-native"}:
                outcome = StageOutcome(request=request, selected_mode=resolve_request_weight_mode(request))
            else:
                with timed(timeline, "coldsnap.weights", operation=operation):
                    outcome = stage_native_packs(
                        request,
                        plan=plan,
                        sctx=sctx,
                        payload_verifier=lambda: self._payload_verifier_path(plan, sctx),
                    )
            if operation == "capture":
                with timed(timeline, "coldsnap.workload.replace", operation=operation):
                    replace_capture_workload(plan=plan, sctx=sctx)
            self._invoke(
                outcome.request,
                prepare_only=False,
                capture_output=False,
                sctx=sctx,
                cluster=plan.cluster,
                offline=options.offline,
            )
            if operation == "publish":
                publication = default_artifact_publish_reference(plan=plan, options=options, snapshot_driver=snapshot_driver)
                if not publication:
                    raise RuntimeError("ColdSnap artifact publication requires a capsule repository")
                with timed(timeline, "coldsnap.publish", operation=operation):
                    outcome.request["published_artifact_reference"] = publish_oci_artifact(
                        Path(request["output"]), publication, snapshot_driver=snapshot_driver
                    )
            elif operation == "publish-native":
                published = json.loads(Path(request["output"]).read_text(encoding="utf-8"))
                payloads = published.get("weights", {}).get("model_payloads", {})
                outcome.request["published_native_repository"] = payloads.get("repository", "")
                outcome.request["published_native_revision"] = payloads.get("revision", "")
                if _has_registry_backed_capsules(published):
                    publication = default_artifact_publish_reference(plan=plan, options=options, snapshot_driver=snapshot_driver)
                    if publication:
                        with timed(timeline, "coldsnap.publish", operation=operation):
                            outcome.request["published_artifact_reference"] = publish_oci_artifact(
                                Path(request["output"]), publication, snapshot_driver=snapshot_driver
                            )
            removed: tuple[Path, ...] = ()
            if managed_promotion:
                store = resolve_artifact_store(plan=plan, options=options, sctx=sctx, snapshot_driver=snapshot_driver)
                generation_id = request["id"] if operation in {"publish", "publish-native"} else None
                with timed(timeline, "coldsnap.promote", operation=operation):
                    removed = tuple(
                        promote_generation(
                            Path(request["output"]),
                            store,
                            keep_generations=retention,
                            generation_id=generation_id,
                        )
                    )
            return outcome.request, removed, outcome.failures
        finally:
            with timed(timeline, "coldsnap.runtime_cache.cleanup", operation=operation):
                cleanup_capture_runtime_cache(cache_stage)

    def _payload_verifier_path(self, plan, sctx, *, binary: str = "") -> Path:
        """Resolve the selected engine adapter used as the remote Go verifier."""

        engine = "sglang" if plan.runtime.runtime_name == "sglang" else "vllm"
        return self.target_tool_resolver(
            hosts=plan.host_list,
            engine=engine,
            cluster=plan.cluster,
            sctx=sctx,
            binary=binary or self.binary,
        ).verifier

    def _restore_artifact_path(
        self,
        options,
        *,
        plan,
        sctx,
        explicit: str = "",
        dry_run: bool = False,
        snapshot_driver: str = "n610",
    ) -> Path:
        store = resolve_artifact_store(plan=plan, options=options, sctx=sctx, snapshot_driver=snapshot_driver)
        reference = explicit if explicit.startswith("oci://") else ""
        if not reference and not explicit:
            config = plan.recipe.plugin_item("coldsnap")
            configured = getattr(getattr(config, "artifact", None), "reference", "")
            # The locally promoted generation is authoritative on its capture
            # controller. The default OCI tag is a portability fallback for a
            # controller that has no local descriptor; an explicit recipe
            # reference always wins.
            if not configured and store.current.is_file():
                return store.current
            reference = configured_artifact_reference(plan=plan, options=options, snapshot_driver=snapshot_driver)
        if not reference:
            return Path(explicit).expanduser() if explicit else store.current
        if dry_run:
            if store.imported.is_file():
                return store.imported
            if store.current.is_file():
                return store.current
            raise RuntimeError("ColdSnap dry-run cannot inspect %s before it has been staged once" % reference)
        return stage_oci_artifact(reference, store.imported, snapshot_driver=snapshot_driver).path

    def _invoke(
        self,
        request: dict[str, Any],
        *,
        prepare_only: bool,
        capture_output: bool,
        binary: str = "",
        sctx=None,
        cluster=None,
        offline: bool | None = None,
    ):
        executable = binary or self.binary
        environment = None
        controller_label = "development controller %s" % executable if executable else ""
        if executable:
            explicit_environment = explicit_controller_environment(executable)
            if explicit_environment:
                environment = {**os.environ, **explicit_environment}
        else:
            if sctx is None:
                raise RuntimeError("ColdSnap managed controller resolution requires a sparkrun context")
            tool = self.tool_resolver(sctx.config)
            executable = str(tool.path)
            environment = {**os.environ, **tool.environment}
            logger.info("ColdSnap controller: %s v%s (%s)", tool.path, tool.version, tool.source)
            controller_label = "controller v%s (%s)" % (tool.version, tool.source)
        if cluster is not None:
            if request["operation"] in {"capture", "restore", "publish-native"}:
                target_tools = self.target_tool_resolver(
                    hosts=[str(unit["host"]) for unit in request["launch"]["units"]],
                    engine=request["launch"]["engine"],
                    cluster=cluster,
                    sctx=sctx,
                    binary=binary or self.binary,
                )
                environment = {**os.environ, **(environment or {}), **target_tools.environment}
            site_policy = resolve_coldsnap_policy(
                cluster=cluster,
                sctx=sctx,
                hosts=[str(unit["host"]) for unit in request["launch"]["units"]],
                probe_remote=True,
            )
            environment = {
                **os.environ,
                **(environment or {}),
                "COLDSNAP_REMOTE_STATE_ROOT": site_policy.state_root,
            }
            logger.info(
                "ColdSnap site policy: state_root=%s recovery_read=%s",
                site_policy.state_root,
                site_policy.recovery_read,
            )
        # Direct ColdSnap invocation retains its controller-owned token path.
        # Under sparkrun, credentialed operations are performed by the
        # operation-scoped host provider and the token never enters ColdSnap's
        # process environment.
        if request["operation"] == "publish-native" and cluster is None:
            token = resolve_hf_token()
            if not token:
                raise RuntimeError("ColdSnap native publication requires Hugging Face authentication")
            environment = {**os.environ, **(environment or {}), "HF_TOKEN": token}
        arguments = [executable, request["operation"], "--request-json", "-"]
        if prepare_only:
            arguments.append("--prepare-only")
        operation = "%s prepare-only" % request["operation"] if prepare_only else str(request["operation"])
        snapshot_driver = str(request.get("snapshot_driver", {}).get("id") or "<missing>")
        logger.log(
            PROGRESS,
            "ColdSnap [%s]: %s; running %s",
            snapshot_driver,
            controller_label,
            operation,
        )
        logger.info(
            "ColdSnap request: id=%s snapshot_driver=%s engine=%s units=%d workers=%d",
            request["id"],
            snapshot_driver,
            request["launch"]["engine"],
            len(request["launch"]["units"]),
            len(request["launch"]["execution"]["workers"]),
        )
        provider_context = (
            self.host_provider_factory(request, sctx=sctx, cluster=cluster, offline=offline) if cluster is not None else nullcontext(None)
        )
        with TemporaryDirectory(prefix="sparkrun-coldsnap-receipt-") as receipt_root:
            receipt_path = Path(receipt_root) / "operation.json"
            arguments.extend(["--receipt-json", str(receipt_path)])
            timeline = getattr(sctx, "timing", None)
            event_path = Path(receipt_root) / "timing.ndjson"
            event_stop = threading.Event()
            event_streams: list[OperationTimingEventStream] = []
            event_errors: list[BaseException] = []
            event_thread: threading.Thread | None = None
            arguments.extend(["--timing-events", str(event_path)])
            progress = OperationTimingProgress(request)

            def report_event(event: Mapping[str, Any]) -> None:
                update = progress.accept(event)
                if update is not None:
                    message, default_visible = update
                    logger.log(PROGRESS if default_visible else logging.INFO, "%s", message)

            def consume_events() -> None:
                try:
                    event_streams.append(follow_operation_timing_events(event_path, request, event_stop, on_event=report_event))
                except (OSError, RuntimeError, UnicodeError) as error:  # surfaced on the controller thread below
                    event_errors.append(error)

            event_thread = threading.Thread(
                target=consume_events,
                name="sparkrun-coldsnap-timing-events",
                daemon=True,
            )
            event_thread.start()
            logger.debug("ColdSnap command: %s", arguments)
            with (
                timed(
                    timeline,
                    "coldsnap.controller",
                    operation=request["operation"],
                    phase="prepare" if prepare_only else "execute",
                    snapshot_driver=snapshot_driver,
                    units=len(request["launch"]["units"]),
                    workers=len(request["launch"]["execution"]["workers"]),
                ) as controller_span,
                provider_context as provider,
            ):
                if provider is not None:
                    environment = {**os.environ, **(environment or {}), **provider.environment}
                    # Credentialed provider operations resolve controller secrets
                    # inside sparkrun. Do not leak those credentials into either
                    # the ColdSnap controller or its engine-adapter child.
                    environment.pop("HF_TOKEN", None)
                    environment.pop("HUGGING_FACE_HUB_TOKEN", None)
                try:
                    heartbeat_base = f"ColdSnap [{snapshot_driver}]: {operation}"
                    with progress_heartbeat(logger, lambda: progress.heartbeat_label(heartbeat_base)):
                        completed = self.run_command(
                            arguments,
                            input=json.dumps(request, indent=2, sort_keys=True) + "\n",
                            text=True,
                            check=False,
                            capture_output=capture_output,
                            env=environment,
                        )
                finally:
                    event_stop.set()
                    if event_thread is not None:
                        event_thread.join(timeout=2.0)
                        if event_thread.is_alive():
                            event_errors.append(RuntimeError("ColdSnap timing event reader did not stop"))
                receipt = read_operation_receipt(receipt_path, request, completed.returncode)
                # Injectable runners may return a plain CompletedProcess. Attach
                # the same receipt field declared by our controller result.
                cast(ControllerResult, completed).startup_observation = startup_observation(receipt)
                if event_errors:
                    logger.warning("ColdSnap timing event stream was not usable: %s", event_errors[0])
                elif event_streams and receipt is not None:
                    try:
                        event_streams[0].finish(receipt)
                    except RuntimeError as error:
                        logger.warning("ColdSnap timing event stream was not usable: %s", error)
                imported = import_operation_timing(
                    timeline,
                    receipt,
                    controller_span,
                )
                if imported:
                    logger.debug("ColdSnap controller timing: imported %d span(s)", imported)
                if completed.returncode:
                    detail = str(getattr(completed, "stderr", "") or "").strip()
                    raise RuntimeError(
                        "coldsnap %s%s failed with exit code %d%s"
                        % (
                            request["operation"],
                            " prepare-only" if prepare_only else "",
                            completed.returncode,
                            ": " + detail[-2000:] if detail else "",
                        )
                    )
        return completed


def _has_registry_backed_capsules(artifact: Mapping[str, Any]) -> bool:
    images = artifact.get("capsule", {}).get("images", [])
    return bool(images) and all(
        isinstance(image, Mapping)
        and isinstance(image.get("reference"), str)
        and isinstance(image.get("digest"), str)
        and "@" in image["reference"]
        and image["reference"].endswith("@" + image["digest"])
        for image in images
    )


def _strategy_options(context: ExecutionContext) -> Mapping[str, Any]:
    value = getattr(context.options, "strategy_options", None)
    return value if isinstance(value, Mapping) else {}


class ColdSnapExecutionStrategy:
    name = "coldsnap"
    owns_env_templates = True

    def __init__(self, service: ColdSnapService | None = None):
        self.service = service or ColdSnapService()

    def preparation_steps(self, context: ExecutionContext):
        validate_environment_recipe(context.plan.recipe)
        return (
            PreparationStep(
                "coldsnap.hardware",
                lambda current, _receipts: verify_coldsnap_hosts(
                    current.plan,
                    current.sctx,
                    dry_run=current.options.dry_run,
                    snapshot_driver=(str(_strategy_options(current).get("snapshot_driver") or "") or None),
                ),
            ),
            PreparationStep(
                "coldsnap.artifact",
                lambda current, receipts: self.service.describe_restore(current, receipts["coldsnap.hardware"]),
                requires=("coldsnap.hardware",),
            ),
            PreparationStep(
                "coldsnap.weights",
                lambda current, receipts: self.service.stage_restore(current, receipts["coldsnap.artifact"]),
                requires=("coldsnap.artifact",),
            ),
            PreparationStep(
                "coldsnap.capsules",
                lambda current, receipts: self.service.prepare_capsules(current, receipts["coldsnap.weights"]),
                requires=("coldsnap.weights",),
            ),
        )

    def finalize_preparation(self, context: ExecutionContext, receipts) -> PreparedExecution:
        return self.service.finalize_restore(context, receipts)

    def prepare_activation(self, context: ActivationContext):
        return self.service.prepare_activation(context)

    def activate(self, context: ActivationContext, receipt):
        return self.service.activate(context, receipt)


def _artifact_file(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "artifact.json"
    if not path.is_file():
        raise RuntimeError(
            "No ColdSnap artifact is active for this recipe fingerprint: %s. Capture it first with `sparkrun coldsnap capture`." % path
        )
    return path


def _capsule_images(artifact: dict[str, Any], launch_units, hosts) -> tuple[str, ...]:
    try:
        records = artifact["capsule"]["images"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("ColdSnap artifact has no capsule image inventory") from error
    if not isinstance(records, list) or len(records) != len(launch_units):
        raise RuntimeError("ColdSnap capsule image count differs from the planned launch-unit count")
    images_by_unit: dict[str, str] = {}
    for unit, record in zip(launch_units, records, strict=True):
        unit_id = unit.get("id")
        if not isinstance(unit_id, str) or not isinstance(record, dict) or record.get("unit") != unit_id:
            raise RuntimeError("ColdSnap capsule image inventory is out of launch-unit order")
        reference = record.get("reference")
        digest = record.get("digest")
        if not isinstance(reference, str) or not reference or not isinstance(digest, str) or not reference.endswith(digest):
            raise RuntimeError("ColdSnap capsule image for unit %s is not digest-pinned" % unit_id)
        images_by_unit[unit_id] = reference
    # The shared asset interface remains host-indexed. ColdSnap has already
    # prepared every unit image and bypasses normal distribution, so this is
    # only a representative image for shared launch metadata.
    images_by_host: dict[str, str] = {}
    for unit in launch_units:
        images_by_host.setdefault(str(unit["host"]), images_by_unit[str(unit["id"])])
    try:
        return tuple(images_by_host[str(host)] for host in hosts)
    except KeyError as error:
        raise RuntimeError("ColdSnap launch units do not cover every planned host") from error


__all__ = [
    "ColdSnapExecutionStrategy",
    "ColdSnapService",
    "PreparedRestore",
    "RestoreActivationReceipt",
    "RestoreDescriptor",
    "prepare_capture_images",
]
