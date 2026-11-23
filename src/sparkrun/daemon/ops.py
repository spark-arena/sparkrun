"""Explicit core operation adapters and their public request schemas."""

from __future__ import annotations

from functools import lru_cache, partial
import inspect
from typing import Any

from sparkrun.daemon.request_fields import fields
from sparkrun.daemon.artifacts import with_diagnostics

from sparkrun.api._wire import object_schema, to_wire, validate
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
from sparkrun.core.plugin_identity import core_registration
from sparkrun.api._wire_schemas import RESULT_SCHEMAS

S = {"type": "string", "maxLength": 4096}
B = {"type": "boolean"}
N = {"type": "number", "minimum": 0}
INTEGER = {"type": "integer", "minimum": 0}
A = {"type": "array", "items": S, "maxItems": 512}
D = {"type": "object"}


def nullable(schema) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


# Each row is a reviewed allowlist, not reflection over api.__all__.
# Long or potentially mutating reads run in workers until their query audit is complete.
SPECS = [
    ("catalog.recipes", "catalog_recipes", "query", "read", "query registry runtime local_only offset limit filters", ""),
    ("catalog.recipe", "get_recipe_details", "job", "read", "reference overrides expected_revision", "reference"),
    ("catalog.registries", "list_registries", "query", "read", "", ""),
    ("catalog.clusters", "list_clusters", "query", "read", "", ""),
    ("catalog.capacity", "catalog_cluster_capacity", "job", "read", "cluster", "cluster"),
    ("catalog.import", "import_recipe", "job", "operate", "content", "content"),
    ("catalog.retain", "retain_catalog_recipe", "job", "operate", "reference", "reference"),
    ("catalog.cleanup", "cleanup_catalog_imports", "job", "operate", "", ""),
    ("catalog.refresh", "refresh_registries", "job", "operate", "", ""),
    ("workloads.list", "list_jobs", "query", "read", "limit", ""),
    ("cluster.observe", "status", "job", "read", "hosts cluster executor", ""),
    ("cluster.report", "status_report", "job", "read", "hosts cluster executor", ""),
    ("workloads.find_intent", "find_running_intent", "job", "read", "intent_id hosts cluster", "intent_id hosts"),
    ("catalog.search", "search_recipes", "query", "read", "query registry runtime include_hidden include_local unique_names", ""),
    ("workloads.stop_matching", "stop", "job", "operate", "workload_id recipe hosts cluster overrides", ""),
    ("workloads.stop_all", "stop_all", "job", "operate", "hosts cluster dry_run", "hosts|cluster"),
    ("discovery.snapshot", "discovery_snapshot", "job", "read", "hosts cluster workload_ids metadata_only", ""),
    (
        "benchmarks.resume",
        "resume_benchmark",
        "job",
        "operate",
        "benchmark_id dry_run integrations export_files output_file timeout exit_on_first_fail api_key_env",
        "benchmark_id",
    ),
    ("cluster.create", "create_cluster", "job", "admin", "name hosts description user", "name hosts"),
    ("cluster.update", "update_cluster", "job", "admin", "name hosts description user", "name"),
    ("cluster.delete", "delete_cluster", "job", "admin", "name", "name"),
    ("cluster.default", "set_default_cluster", "job", "admin", "name", ""),
    ("proxy.status", "proxy.status", "job", "read", "", ""),
    ("proxy.models", "proxy.models", "job", "read", "", ""),
    ("proxy.stop", "proxy.stop", "job", "operate", "dry_run", ""),
    ("proxy.sync", "proxy.sync", "job", "operate", "aliases host_filter", ""),
    ("proxy.register_loaded_model", "proxy.register_loaded_model", "job", "operate", "recipe overrides cluster", "recipe"),
    ("proxy.unregister_loaded_model", "proxy.unregister_loaded_model", "job", "operate", "recipe", "recipe"),
    ("proxy.add_alias", "proxy.add_alias", "job", "admin", "alias target", "alias target"),
    ("proxy.remove_alias", "proxy.remove_alias", "job", "admin", "alias", "alias"),
    ("proxy.list_aliases", "proxy.list_aliases", "query", "read", "", ""),
    ("proxy.list_gateways", "proxy.list_gateways", "query", "read", "", ""),
    ("proxy.resolve_gateway", "proxy.resolve_gateway", "query", "read", "name", ""),
    ("proxy.service_status", "proxy.service_status", "job", "admin", "scope journal_lines", ""),
    ("proxy.uninstall_service", "proxy.uninstall_service", "job", "admin", "scope dry_run", ""),
    ("setup.probe_ssh_access", "setup.probe_ssh_access", "job", "read", "hosts user key timeout max_workers", "hosts"),
    ("setup.find_changed_host_keys", "setup.find_changed_host_keys", "job", "read", "hosts user key timeout max_workers", "hosts"),
    ("setup.replace_host_key", "setup.replace_host_key", "job", "admin", "host fingerprint user key timeout", "host fingerprint"),
    ("setup.ensure_local_key", "setup.ensure_local_key", "job", "admin", "preferred generate", ""),
    ("setup.resolve_control_key", "setup.resolve_control_key", "job", "admin", "host user generate", ""),
    ("setup.mesh_ssh_keys", "setup.mesh_ssh_keys_native", "job", "admin", "hosts user key timeout dry_run", "hosts"),
]

# Audited local reads that tolerate the short cache TTL. A new read query must
# opt in here; declaring its permission alone never makes it cacheable.
CACHEABLE_QUERIES = frozenset(
    {
        "catalog.recipes",
        "catalog.registries",
        "catalog.clusters",
        "catalog.search",
        "workloads.list",
        "proxy.list_aliases",
        "proxy.list_gateways",
        "proxy.resolve_gateway",
    }
)

OPTION_SPECS = [
    ("workloads.run", "run", "RunOptions", "operate"),
    ("workloads.plan", "plan", "RunOptions", "read"),
    ("workloads.materialize", "materialize", "RunOptions", "operate"),
    ("recipes.realize", "realize_recipe", "RunOptions", "operate"),
    ("recipes.build", "build", "BuildOptions", "operate"),
    ("recipes.plan_build", "plan_build", "BuildOptions", "read"),
    ("benchmarks.run", "benchmark", "BenchmarkOptions", "operate"),
    ("proxy.start", "proxy.start", "ProxyStartOptions", "operate"),
    ("proxy.install_service", "proxy.install_service", "ProxyServiceOptions", "admin"),
]


DESCRIPTIONS = {
    "proxy.discovery.ensure": "Start (or adopt) the daemon-run discovery job for the current proxy session generation.",
    "proxy.discovery.status": "Report the proxy discovery session and its job, flagging one that needs recovery.",
    "proxy.discovery.stop": "Stop the proxy discovery session and cancel its job.",
    "setup.plan": "Resolve the setup context and plan the setup steps for the selected hosts.",
    "workloads.stop": "Stop an exact persisted workload through its owning lifecycle.",
    "workloads.status": "Report the live state of an exact persisted workload.",
    "workloads.sleep": "Put an exact persisted workload to sleep, releasing its accelerators where the runtime supports it.",
    "workloads.wake": "Wake a sleeping workload and wait for it to serve again.",
    "model_cache.inspect": "Validate one model revision's cached files and bindings on the selected hosts.",
    "workloads.list": "List saved workload metadata without probing live health.",
    "workloads.run": "Plan and launch a workload, with optional readiness and post-launch hooks.",
    "workloads.plan": "Resolve placement and launch settings without launching a workload.",
    "workloads.materialize": "Resolve container launch units from a plan without starting them.",
    "workloads.stop_matching": "Stop workloads selected by ID, recipe or hosts; no integration-owner guard.",
    "workloads.stop_all": "Discover and stop all workloads on the given hosts or cluster (one is required; a teardown never defaults its target).",
    "workloads.find_intent": "Find an observed running deployment matching a launch intent.",
    "cluster.observe": "Observe raw per-host occupancy and coverage errors.",
    "cluster.report": "Observe occupancy and enrich grouped workload/idle-host results with saved metadata.",
    "catalog.search": "Search recipe summaries with simple filters; catalog.recipes provides paginated browsing.",
    "recipes.build": "Prepare images or native environments and stage recipe assets without starting inference.",
    "recipes.plan_build": "Plan environment preparation and asset staging without reserving inference ranks.",
    "recipes.realize": "Plan against the selected cluster and export a pinned, resolved recipe.",
    "benchmarks.run": "Launch or reuse inference and execute the selected benchmark framework.",
    "benchmarks.resume": "Resume a saved benchmark by benchmark ID, separate from its daemon job ID.",
}


OPTION_FIELDS = {
    "RunOptions": "recipe hosts cluster overrides strategy_options solo dry_run trust ensure scheduler transfer_mode cache_dir offline runtime_cache auto_port ray_port dashboard_port dashboard init_port executor executor_config rootful workload_id_override transfer_interface local_cache_dir sync_tuning extra_docker_opts topology preserve_model_perms skip_model_fan_out recipe_ref",
    "BuildOptions": "recipe hosts cluster overrides dry_run trust rebuild offline executor executor_config cache_dir local_cache_dir transfer_mode transfer_interface topology sync_tuning preserve_model_perms skip_model_fan_out builder_context",
    "BenchmarkOptions": "recipe category framework profile bench_args hosts cluster overrides resume skip_run no_stop exit_on_first_fail timeout api_key_env integrations output_file export_files solo dry_run scheduler rootful trust offline sync_tuning extra_docker_opts",
    "ProxyStartOptions": "host port gateway host_filter cluster clear_cluster auto_discover discovery_backend discover_interval discover_removal_grace_sweeps restart persist dry_run",
    "ProxyServiceOptions": "scope cluster now dry_run",
}


def _resolve(target):
    from sparkrun import api

    value: Any = api
    for part in target.split("."):
        value = getattr(value, part)
    return value


def unavailable_prompt(*args, **kwargs):
    raise RuntimeError("This operation requires a worker input channel")


def _benchmark_progress(sctx, event):
    # Scenario observations already arrive through OperationState.feedback.
    # Keep legacy benchmark events without publishing a second observation.
    if event.kind != "task_progress":
        sctx.operation.emit("progress", event=to_wire(event))


@with_diagnostics
def invoke(target, arguments, sctx, prompt=unavailable_prompt):
    from sparkrun import api

    fn = _resolve(target)
    arguments = dict(arguments)
    parameters = inspect.signature(fn).parameters
    if target == "search_recipes":
        arguments["ensure_initialized"] = False
    if "sctx" in parameters:
        arguments["sctx"] = sctx
    if "user" in parameters and "user" not in arguments:
        arguments["user"] = sctx.config.ssh_user
    if "progress" in parameters:
        arguments["progress"] = lambda name, success: sctx.operation.emit("progress", label=name, success=success)
    if "progress_callback" in parameters:
        arguments["progress_callback"] = lambda event: _benchmark_progress(sctx, event)
    if "decision_callback" in parameters:
        arguments["decision_callback"] = lambda decision: prompt("decision", decision.message, default=False)
    try:
        return to_wire(fn(**arguments))
    except api.proxy.SudoPasswordRequired:
        if target != "proxy.uninstall_service":
            raise
        arguments["sudo_password"] = prompt("secret", "Sudo password", default=None)
        return to_wire(fn(**arguments))


@with_diagnostics
def invoke_options(target, model, arguments, sctx, prompt=unavailable_prompt):
    from sparkrun import api

    arguments = dict(arguments)
    confirm = arguments.pop("confirm", False)
    ready = arguments.pop("wait_ready", True)
    hooks = arguments.pop("post_launch", True)
    cls = getattr(api.proxy, model) if model.startswith("Proxy") else getattr(api, model)
    if model == "RunOptions":
        arguments.update(detached=True, follow=False, owner=sctx.operation.manager)
    if model == "BenchmarkOptions":
        arguments["progress_callback"] = lambda event: _benchmark_progress(sctx, event)
        arguments["decision_callback"] = lambda decision: prompt("decision", decision.message, default=False)
        if "resume" in arguments:
            arguments["resume"] = api.ResumeMode(arguments["resume"])
    options = cls(**arguments)
    if target == "proxy.install_service":
        from dataclasses import replace

        try:
            return to_wire(api.proxy.install_service(options, sctx=sctx))
        except api.proxy.SudoPasswordRequired:
            options = replace(options, sudo_password=prompt("secret", "Sudo password", default=None))
            return to_wire(api.proxy.install_service(options, sctx=sctx))
    if target == "run":
        plan = api.plan(options, sctx=sctx)
        sctx.operation.emit("plan", plan=to_wire(plan))
        if confirm and not prompt("approval", "Launch this plan?", default=False):
            raise InterruptedError("Launch declined")
        result = api.run(options, sctx=sctx, plan=plan)
        if result.rc:
            raise api.SparkrunError(f"Launch {result.workload_id} failed with exit code {result.rc}")
        if not options.dry_run and not result.already_running and result.launch_result is not None:
            readiness = api.wait_ready(result, sctx=sctx) if ready else None
            if readiness is not None and not readiness.ready:
                raise api.SparkrunError("Server did not become ready: " + str(readiness.reason))
            if hooks and (plan.recipe.post_exec or plan.recipe.post_commands):
                api.post_launch(result, trust=options.trust, readiness=readiness, sctx=sctx)
        elif not options.dry_run and (ready or hooks):
            sctx.operation.emit("warning", message="Launch has no new lifecycle handle; readiness and post-launch hooks were not run.")
        return to_wire(result)
    return to_wire(_resolve(target)(options, sctx=sctx))


def _loader(target):
    return partial(invoke, target)


def _options_loader(target, model):
    return partial(invoke_options, target, model)


def _request_schema(properties, required):
    """``"a b"`` requires both; ``"a|b"`` requires at least one of them."""
    groups = required.split()
    schema = object_schema(properties, [name for name in groups if "|" not in name])
    alternatives = [group.split("|") for group in groups if "|" in group]
    if len(alternatives) > 1:
        raise ValueError("Only one alternative-required group is supported per operation")
    if alternatives:
        schema["anyOf"] = [{"required": [name]} for name in alternatives[0]]
    return schema


@lru_cache(maxsize=None)
def _describe(name, target, generic):
    """The hand-written description, else the library function's own summary, else *generic*.

    Computed once per process: a description is part of the operation's
    definition, and an equal re-registration must stay equal even if the
    library attribute was since replaced (a test double, a wrapping plugin).
    """
    from sparkrun.daemon.op_docs import api_target, summary

    return DESCRIPTIONS.get(name) or summary(api_target(target)) or generic


@core_registration()
def register_core_ops():
    for name, target, kind, scope, names, required in SPECS:
        properties = fields(names)
        if name == "benchmarks.resume":
            properties["collect_diagnostics"] = B
        if name == "cluster.default":
            properties["name"] = nullable(S)
        if name == "catalog.import":
            properties["content"] = {"type": "string", "maxLength": 262144}
        register_daemon_op(
            DaemonOp(
                name,
                partial(_loader, target),
                _request_schema(properties, required),
                result_schema=RESULT_SCHEMAS[name],
                kind=kind,
                scope=scope,
                cacheable=name in CACHEABLE_QUERIES,
                resources="none" if name.startswith("proxy.") else "auto",
                feature_flag="gateway"
                if name.startswith("proxy.")
                and name not in {"proxy.status", "proxy.stop", "proxy.service_status", "proxy.uninstall_service"}
                else None,
                description=_describe(name, target, f"{target.replace('_', ' ').capitalize()} through the library API."),
            )
        )
    for name, target, model, scope in OPTION_SPECS:
        properties = fields(OPTION_FIELDS[model])
        if "recipe" in properties:
            properties["recipe"] = S
        if target in {"run", "benchmark"}:
            properties["collect_diagnostics"] = B
        if target == "run":
            properties.update({"confirm": B, "wait_ready": B, "post_launch": B})
        register_daemon_op(
            DaemonOp(
                name,
                partial(_options_loader, target, model),
                object_schema(properties, [] if model.startswith("Proxy") else ["recipe"]),
                result_schema=RESULT_SCHEMAS[name],
                kind="job",
                scope=scope,
                resources="none" if name.startswith("proxy.") else "auto",
                feature_flag="gateway"
                if name.startswith("proxy.")
                and name not in {"proxy.status", "proxy.stop", "proxy.service_status", "proxy.uninstall_service"}
                else None,
                description=_describe(name, target, f"{name.replace('_', ' ').capitalize()} with an operation-local context."),
            )
        )
    from sparkrun.daemon.registry_ops import register_registry_ops

    register_registry_ops()
    from sparkrun.daemon.proxy import register_proxy_ops

    register_proxy_ops()
    from sparkrun.daemon.workloads import register_workload_ops

    register_workload_ops()
    from sparkrun.daemon.special_ops import register_special_ops

    register_special_ops()


def check_request(op, arguments):
    from sparkrun.api._wire import reject_secrets

    validate(arguments, op.request_schema)
    reject_secrets(arguments)
    return arguments
