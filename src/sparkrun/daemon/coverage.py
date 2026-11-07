"""Reviewed mapping of public library functions to wire operations or exclusions."""

from sparkrun.daemon.ops import SPECS, OPTION_SPECS
from sparkrun.daemon.registry_ops import REGISTRY_ACTIONS
from sparkrun.daemon.model_cache_ops import BINDINGS as MODEL_CACHE_BINDINGS

API_OPERATIONS = {target: name for name, target, *_ in SPECS}
API_OPERATIONS.update({target: name for name, target, *_ in OPTION_SPECS})
API_OPERATIONS.update({binding.target: binding.name for binding in MODEL_CACHE_BINDINGS})
API_OPERATIONS["validate_recipe"] = "recipes.validate"
API_OPERATIONS["configure_registry"] = " / ".join("catalog.registries." + action for action in REGISTRY_ACTIONS)
API_OPERATIONS.update(
    {
        # The offline primitive is the supplied-manifest branch of import_model.
        "model_cache.import_manifest": "model_cache.import",
        "collect_diagnostics": "diagnostics.collect",
        "wait_ready": "workloads.run",
        "post_launch": "workloads.run",
        "schedule": "workloads.plan",
        "logs": "/v1/streams/logs",
        "open_telemetry": "/v1/streams/monitor",
        "open_live_monitor": "/v1/streams/monitor",
        "live_monitor": "/v1/streams/monitor",
        "setup.probe_setup_hosts": "setup.probe_hosts",
        "setup.check_ssh_host_identities": "setup.plan",
        "setup.resolve_setup_context": "setup.plan",
        "setup.build_setup_plan": "setup.plan",
        "setup.evaluate_host": "setup.plan",
        "setup.apply_setup_step": "setup.run_steps",
        "setup.run_setup_steps": "setup.run_steps",
        "setup.run_setup_undo": "setup.undo",
        "setup.rdma_test": "setup.rdma_test",
        "tailscale.status": "tailscale.status",
        "tailscale.join": "tailscale.join",
        "tailscale.down": "tailscale.down",
        "tailscale.expose": "tailscale.expose",
    }
)
EXCLUSIONS = {
    "capture_diagnostics": "Python context manager; workloads.run and benchmarks.run/resume accept collect_diagnostics.",
    "open_log_target": "Python reader handle; /v1/streams/logs resolves each request and accepts the same host/container filters.",
    "open_exec_target": "Hands a local terminal to a shell inside a workload; over the wire it would be remote command execution.",
    "default_sctx": "Python application binding; daemon creates an operation-local context.",
    "resolve_sctx": "Python application binding; daemon creates an operation-local context.",
    "resolve_catalog_recipe": "Returns a live Recipe; catalog.recipe exposes a bounded preview and run resolves it server-side.",
    "resolve_recipe_filter": "Python filter helper; catalog.recipes and catalog.search accept the same filter inputs.",
    "untrusted_hooks": "Takes a live Recipe; workloads.run applies the same gate server-side and catalog.recipe reports trust.",
    "recipe_protocols": "Takes a live Recipe; catalog.recipe and discovery.snapshot report native_protocols and capabilities.",
    "replace_intent_deployments": "Experimental; takes a live RunPlan for processes started outside workloads.run, which replaces server-side.",
    "fresh_workload_id": "Experimental; returns a live RunPlan copy for the Python run(options, plan=...) path.",
    "proxy.admin_token": "Credential issuance is interactive/local; secrets are not persisted in job results.",
    "proxy.ui": "May issue a credential-bearing URL; use the local proxy ui command.",
    "setup.install_public_key_interactive": "Requires a terminal-owned ssh-copy-id/password conversation; use setup in a terminal.",
    "setup.rotate_ssh_host_keys": "Host-identity rotation has no wire operation; use the confirmed setup ssh-rotate-keys CLI or Python API.",
    "setup.plan_cx7": "Experimental; reports progress and asks for a topology through Python callbacks. Use setup cx7 or the Python API.",
    "setup.apply_cx7": "Experimental; applies a live Cx7Plan with caller-collected sudo credentials. Use setup cx7 or the Python API.",
    "setup.save_cx7_topology": "Experimental; takes a live Cx7Plan and is applied by apply_cx7.",
    "setup.validate_cx7_options": "Pure option check that plan_cx7 applies itself.",
    "setup.diagnose_ssh": "Requires a terminal-owned ssh password conversation; use setup ssh --diagnose in a terminal.",
    "setup.find_local_public_key": "Reads the caller's own ssh configuration; diagnose_ssh reports the result.",
    "setup.load_identity": "Reads the caller's own private key file; setup.resolve_control_key reports the identity in use.",
    "setup.validate_public_key": "Pure validation helper used before a key is embedded in a remote script.",
    "model_cache.reconcile_claims": "Composes status with model_cache.list and model_cache.reconcile; use the CLI or Python API.",
}
K8S_OPERATIONS = {
    name: "plugins.k8s." + name
    for name in (
        "ensure_kubectl",
        "cluster_info",
        "configure_service_account",
        "list_nodes",
        "kueue_status",
        "setup_kueue",
        "stop_jobset",
        "jobset_status",
    )
}
K8S_OPERATIONS.update({"logs": "/v1/streams/logs", "launch_jobset": "workloads.run", "check_feasibility": "workloads.plan"})
K8S_EXCLUSIONS = {
    "make_client": "Returns an executable Python client, not JSON.",
    "probe_nodes_fallback": "Internal hardware fallback covered by list_nodes.",
    "run_launcher_job": "Low-level arbitrary launcher specification; submit a recipe through workloads.run/recipes.build instead.",
}


WORKLOAD_OPERATIONS = {
    name: "workloads." + name
    for name in (
        "ensure_ready",
        "stop",
        "endpoints",
        "inventory",
        "plugins",
        "reconcile",
    )
}
WORKLOAD_OPERATIONS["control"] = "workloads.status / workloads.sleep / workloads.wake"
WORKLOAD_EXCLUSIONS = {"resolve_binding": "Returns a live Recipe; catalog.recipe exposes its normalized public details."}
