"""Core workload operations using the authenticated management identity."""

from functools import partial

from sparkrun.api._wire import object_schema
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op

_BINDING = {
    "recipe": {"type": "string", "minLength": 1, "maxLength": 4096},
    "recipe_revision": {"type": "string", "maxLength": 1024},
    "cluster_candidates": {"type": "array", "items": {"type": "string", "maxLength": 1024}, "maxItems": 64},
    "overrides": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 4096}},
}


def invoke(name, arguments, sctx, prompt=None):
    from sparkrun.api import workloads

    args = dict(arguments)
    owner = sctx.operation.manager
    if name == "plugins":
        return {"plugins": workloads.plugins(sctx=sctx)}
    if name == "inventory":
        return workloads.inventory(owner=owner, sctx=sctx)
    if name == "endpoints":
        return workloads.endpoints(sctx=sctx, **args)
    binding = workloads.WorkloadBinding(**{key: args.pop(key) for key in list(args) if key in _BINDING})
    if name == "ensure_ready":
        return workloads.ensure_ready(binding, owner=owner, sctx=sctx, **args)
    if name == "reconcile":
        return workloads.reconcile(binding, owner=owner, sctx=sctx, **args)
    if name == "stop":
        return workloads.stop(binding, owner=owner, sctx=sctx, **args)
    return workloads.control(binding, owner=owner, sctx=sctx, action=name, **args)


def _load(name):
    return partial(invoke, name)


def workload_target(name):
    """The ``sparkrun.api.workloads`` function behind ``workloads.<name>``, for its documentation."""
    return "workloads." + ("control" if name in {"status", "sleep", "wake"} else name)


def register_workload_ops():
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS
    from sparkrun.daemon.ops import _describe

    for name in ("ensure_ready", "stop", "status", "sleep", "wake", "inventory", "endpoints", "plugins", "reconcile"):
        fields = dict(_BINDING)
        required = ["recipe"]
        if name == "ensure_ready":
            fields["timeout"] = {"type": "number", "minimum": 0.01, "maximum": 3600}
        elif name in {"inventory", "plugins"}:
            fields, required = {}, []
        elif name == "endpoints":
            fields.pop("recipe")
            fields.pop("overrides")
            fields["workload_id"] = {"type": "string", "maxLength": 1024}
            required = []
        else:
            fields["workload_id"] = {"type": "string", "maxLength": 1024}
            required += ["recipe_revision", "workload_id"]
            if name == "reconcile":
                fields["acknowledge_effects"] = {"type": "boolean", "enum": [True]}
                required.append("acknowledge_effects")
        register_daemon_op(
            DaemonOp(
                "workloads." + name,
                partial(_load, name),
                object_schema(fields, required),
                result_schema=RESULT_SCHEMAS[
                    "discovery.snapshot"
                    if name == "endpoints"
                    else "workloads.control"
                    if name in {"status", "sleep", "wake"}
                    else "workloads." + name
                ],
                scope="admin" if name == "reconcile" else "read" if name in {"inventory", "endpoints", "status", "plugins"} else "operate",
                resources="none" if name in {"inventory", "endpoints", "status", "plugins"} else "controller",
                description=_describe(
                    "workloads." + name, workload_target(name), "Inspect or control an exact workload using the shared recovery workflow."
                ),
            )
        )
