"""Reviewed model-cache argument bindings, shared by registration and execution.

Python parameters are explicitly forwarded, adapted, injected, or excluded.
Signature inspection belongs in contract tests, never in wire-field discovery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from sparkrun.api._wire import object_schema, to_wire
from sparkrun.api._wire_schemas import RESULT_SCHEMAS
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
from sparkrun.daemon.request_fields import fields


LOCAL_CONTEXT = {
    "session": "The daemon uses its configured host transport, not a caller-owned Python session.",
    "sctx": "The adapter supplies SSH/configuration explicitly; retain the API's context default.",
}
ORIGIN_TOKEN = {"token": "Origin credentials are resolved server-side, never persisted in daemon arguments."}


def artifact_request(values):
    from sparkrun.models.artifacts import ModelArtifactRequest

    return {"request": ModelArtifactRequest(**values)}


def imported_manifest(values):
    import json
    from sparkrun.models.artifacts import ModelArtifactManifest

    values = dict(values)
    manifest = ModelArtifactManifest.from_json(json.dumps(values.pop("manifest"))) if "manifest" in values else None
    return {**artifact_request(values), "manifest": manifest}


def adoption_request(values):
    import os
    from sparkrun.models.artifacts import ModelArtifactRequest

    model, revision = values.get("model"), values.get("revision")
    endpoint = values.get("endpoint") or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if revision and not model:
        raise ValueError("revision needs a model")
    return {
        "request": ModelArtifactRequest(model, revision, endpoint=endpoint) if model else None,
        "endpoint": None if model else endpoint,
    }


@dataclass(frozen=True)
class ModelCacheBinding:
    name: str
    target: str
    scope: str
    forwarded: str
    required: str
    request_fields: str = ""
    adapter: Callable[[dict], dict] | None = None
    adapted_parameters: tuple[str, ...] = ()
    injected: tuple[str, ...] = ("ssh_kwargs",)
    excluded: Mapping[str, str] = field(default_factory=lambda: dict(LOCAL_CONTEXT))
    overrides: dict[str, Any] = field(default_factory=dict)

    def request_schema(self):
        properties = fields(self.request_fields + " " + self.forwarded)
        # fields() returns fresh schemas; operation-specific rules stay local.
        from copy import deepcopy

        properties.update(deepcopy(self.overrides))
        return object_schema(properties, self.required.split())

    def arguments(self, arguments, sctx):
        # Preserve omission and falsey values: Python defaults apply only when
        # the caller omitted a field, never when it supplied False or [].
        unknown = arguments.keys() - set((self.request_fields + " " + self.forwarded).split())
        if unknown:
            raise ValueError(f"{self.name}: unknown fields: {', '.join(sorted(unknown))}")
        data = {name: arguments[name] for name in self.forwarded.split() if name in arguments}
        if self.adapter is not None:
            values = {name: arguments[name] for name in self.request_fields.split() if name in arguments}
            adapted = self.adapter(values)
            if adapted.keys() != set(self.adapted_parameters):
                raise ValueError(f"{self.name}: adapter returned unexpected parameters")
            data.update(adapted)
        if "ssh_kwargs" in self.injected:
            from sparkrun.orchestration.primitives import build_ssh_kwargs

            data["ssh_kwargs"] = build_ssh_kwargs(sctx.config)
        if "config" in self.injected:
            data["config"] = sctx.config
        return data


BINDINGS = (
    ModelCacheBinding("model_cache.list", "model_cache.inventory_cache", "read", "cache_dir hosts", "cache_dir"),
    ModelCacheBinding(
        "model_cache.prune",
        "model_cache.prune_cache",
        "admin",
        "cache_dir hosts dry_run min_age_days max_bytes min_free_bytes",
        "cache_dir",
    ),
    ModelCacheBinding(
        "model_cache.pin",
        "model_cache.pin_revision",
        "operate",
        "cache_dir hosts pinned",
        "cache_dir model revision",
        request_fields="model revision endpoint",
        adapter=artifact_request,
        adapted_parameters=("request",),
    ),
    ModelCacheBinding(
        "model_cache.delete",
        "model_cache.delete_model",
        "admin",
        "cache_dir hosts dry_run force allowed_revisions",
        "cache_dir model",
        request_fields="model revision",
        adapter=artifact_request,
        adapted_parameters=("request",),
    ),
    ModelCacheBinding(
        "model_cache.adopt",
        "model_cache.adopt_cache",
        "admin",
        "cache_dir hosts all_models dry_run offline last_used pinned",
        "cache_dir",
        request_fields="model revision endpoint",
        adapter=adoption_request,
        adapted_parameters=("request", "endpoint"),
        excluded={**LOCAL_CONTEXT, **ORIGIN_TOKEN},
    ),
    ModelCacheBinding(
        "model_cache.reconcile",
        "model_cache.reconcile_claim",
        "admin",
        "cache_dir hosts claim confirm_inactive",
        "cache_dir model revision claim confirm_inactive",
        request_fields="model revision endpoint",
        adapter=artifact_request,
        adapted_parameters=("request",),
        overrides={"confirm_inactive": {"type": "boolean", "enum": [True]}},
    ),
    ModelCacheBinding(
        "model_cache.inspect",
        "model_cache.inspect_model",
        "read",
        "cache_dir hosts checksums",
        "model cache_dir",
        request_fields="model revision endpoint projector file_selection allow_patterns ignore_patterns",
        adapter=artifact_request,
        adapted_parameters=("request",),
    ),
    ModelCacheBinding(
        "model_cache.repair",
        "model_cache.repair_model",
        "operate",
        "cache_dir hosts local_cache_dir transfer_mode checksums offline dry_run",
        "model cache_dir",
        request_fields="model revision endpoint projector file_selection allow_patterns ignore_patterns",
        adapter=artifact_request,
        adapted_parameters=("request",),
        injected=("ssh_kwargs", "config"),
        excluded={**LOCAL_CONTEXT, **ORIGIN_TOKEN, "manifest": "Use model_cache.import to validate and store a supplied manifest."},
    ),
    ModelCacheBinding(
        "model_cache.import",
        "model_cache.import_model",
        "operate",
        "cache_dir hosts checksums fetch_manifest offline",
        "model cache_dir",
        request_fields="model revision endpoint projector file_selection allow_patterns ignore_patterns manifest",
        adapter=imported_manifest,
        adapted_parameters=("request", "manifest"),
        excluded={
            **LOCAL_CONTEXT,
            **ORIGIN_TOKEN,
            "confirm_manifest": "Daemon callers consent with fetch_manifest; Python confirmation callbacks are local-only.",
        },
    ),
    ModelCacheBinding(
        "model_cache.resolve",
        "model_cache.resolve_manifest",
        "read",
        "cache_dir",
        "model",
        request_fields="model revision endpoint projector file_selection allow_patterns ignore_patterns",
        adapter=artifact_request,
        adapted_parameters=("request",),
        injected=(),
        excluded=ORIGIN_TOKEN,
    ),
)


def invoke(binding, arguments, sctx, prompt=None):
    from sparkrun.daemon.ops import _resolve

    return to_wire(_resolve(binding.target)(**binding.arguments(arguments, sctx)))


def _load(binding):
    return partial(invoke, binding)


def register_model_cache_ops():
    from sparkrun.daemon.ops import _describe

    for binding in BINDINGS:
        register_daemon_op(
            DaemonOp(
                binding.name,
                partial(_load, binding),
                binding.request_schema(),
                result_schema=RESULT_SCHEMAS[binding.name],
                kind="job",
                scope=binding.scope,
                description=_describe(binding.name, binding.target, binding.name.replace(".", " ").replace("_", " ").capitalize() + "."),
            )
        )
