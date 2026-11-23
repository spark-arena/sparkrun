"""Adapters for APIs whose Python callbacks and objects are operation-local."""

from __future__ import annotations

from functools import partial

from sparkrun.api._wire import object_schema, to_wire
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
from sparkrun.api._wire_schemas import RESULT_SCHEMAS
from sparkrun.daemon.ops import _describe, unavailable_prompt


def execute(name, arguments, sctx, prompt=unavailable_prompt):
    if name.startswith("model_cache."):
        from sparkrun.daemon.model_cache_ops import BINDINGS, invoke

        return invoke(next(binding for binding in BINDINGS if binding.name == name), arguments, sctx, prompt)

    from sparkrun import api
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    data = dict(arguments)
    if name == "diagnostics.collect":
        from sparkrun.daemon.artifacts import collect

        return collect(data, sctx)
    cluster = (
        resolve_cluster(data.pop("cluster", None), data.get("hosts"), sctx=sctx) if name.startswith(("setup.", "tailscale.")) else None
    )
    if cluster is not None:
        sctx = sctx.for_cluster(cluster)
    ssh = build_ssh_kwargs(sctx.config)
    if name == "recipes.validate":
        from sparkrun.api._resolve import resolve_recipe

        return to_wire(
            api.validate_recipe(
                resolve_recipe(data["recipe"], sctx=sctx), overrides=data.get("overrides"), config=sctx.config, v=sctx.variables
            )
        )
    if name.startswith("tailscale."):
        assert cluster is not None
        action = name.split(".")[1]
        hosts = data.pop("hosts", list(cluster.hosts))
        if action != "expose":
            data["host_list"] = hosts
        # Secrets are requested at execution, never part of the persisted request.
        if action in {"join", "down", "expose"} and data.pop("ask_sudo", False):
            data["sudo_password"] = prompt("secret", "Sudo password", default=None)
        return to_wire(getattr(api.tailscale, action)(sctx=sctx, ssh_kwargs=ssh, **data))
    if name == "setup.rdma_test":
        assert cluster is not None
        hosts = data.pop("hosts", list(cluster.hosts))
        return to_wire(api.setup.rdma_test(sctx, hosts, ssh, cluster=cluster, **data))
    if name.startswith("setup."):
        assert cluster is not None
        from sparkrun.core.setup_actions import SetupActionContext
        from sparkrun.core.setup_manifest import ManifestManager

        hosts = data.get("hosts") or list(cluster.hosts)
        dry_run = data.get("dry_run", False)
        states, context = api.setup.probe_setup_hosts(hosts, ssh_kwargs=ssh, config=sctx.config, cluster=cluster, cluster_name=cluster.name)
        if name == "setup.probe_hosts":
            return to_wire(states)
        if name == "setup.plan":
            return {host: to_wire(api.setup.build_setup_plan(state, context)) for host, state in states.items()}
        action = SetupActionContext(user=sctx.config.ssh_user, ssh_kwargs=ssh, dry_run=dry_run)
        manager = ManifestManager(sctx.cluster_manager.clusters_dir)
        callbacks = {
            "approve": lambda step, targets: bool(prompt("approval", f"Apply {step.key} to {', '.join(targets)}?", default=False)),
            "credentials": lambda: prompt("secret", "Sudo password", default=None),
            "progress_callback": lambda event: sctx.operation.emit("progress", event=to_wire(event)),
        }
        only = set(data["only_steps"]) if data.get("only_steps") else None
        if name == "setup.undo":
            if not cluster.name:
                raise ValueError("Undo requires a named cluster")
            manifest = manager.load(cluster.name, strict=True)
            if manifest is None:
                raise ValueError("No setup manifest for this cluster")
            return to_wire(api.setup.run_setup_undo(manifest, action, manifest_mgr=manager, only_steps=only, **callbacks))
        return to_wire(
            api.setup.run_setup_steps(
                states,
                context,
                action,
                cluster=cluster,
                cluster_name=cluster.name,
                manifest_mgr=manager,
                only_steps=only,
                host_credentials=lambda host: prompt("secret", f"Sudo password for {host}", default=None),
                **callbacks,
            )
        )
    raise LookupError(name)


def _load(name):
    return partial(execute, name)


#: The library function behind each operation, for its documentation.
SPECIAL_TARGETS = {
    "diagnostics.collect": "collect_diagnostics",
    "recipes.validate": "validate_recipe",
    "setup.probe_hosts": "setup.probe_setup_hosts",
    "setup.plan": "setup.build_setup_plan",
    "setup.run_steps": "setup.run_setup_steps",
    "setup.undo": "setup.run_setup_undo",
    "setup.rdma_test": "setup.rdma_test",
    "tailscale.status": "tailscale.status",
    "tailscale.join": "tailscale.join",
    "tailscale.down": "tailscale.down",
    "tailscale.expose": "tailscale.expose",
}


def register_special_ops():
    from sparkrun.daemon.request_fields import fields

    specs = [
        ("diagnostics.collect", "read", "hosts cluster workload_id log_tail max_bytes dry_run", ""),
        ("recipes.validate", "read", "recipe overrides", "recipe"),
        ("setup.probe_hosts", "read", "hosts cluster", ""),
        ("setup.plan", "read", "hosts cluster", ""),
        ("setup.run_steps", "admin", "hosts cluster dry_run only_steps", ""),
        ("setup.undo", "admin", "cluster dry_run only_steps", "cluster"),
        (
            "setup.rdma_test",
            "operate",
            "hosts cluster suite parallel_pairs full image duration queue_pairs msg_size nccl_binary gid_index link_type force_container keep_containers dry_run",
            "",
        ),
        ("tailscale.status", "read", "hosts cluster", ""),
        ("tailscale.join", "admin", "hosts cluster tag ephemeral enable_ssh hostname dry_run ask_sudo", ""),
        ("tailscale.down", "admin", "hosts cluster remove dry_run ask_sudo", ""),
        ("tailscale.expose", "admin", "cluster proxy head_host port set_proxy_host ask_sudo", ""),
    ]
    for name, scope, names, required in specs:
        properties = fields(names)
        if name == "diagnostics.collect":
            properties["max_bytes"] = {"type": "integer", "minimum": 16384, "maximum": 67108864}
        register_daemon_op(
            DaemonOp(
                name,
                partial(_load, name),
                object_schema(properties, required.split()),
                result_schema=RESULT_SCHEMAS[name],
                kind="job",
                scope=scope,
                description=_describe(name, SPECIAL_TARGETS.get(name, ""), name.replace(".", " ").replace("_", " ").capitalize() + "."),
            )
        )

    from sparkrun.daemon.model_cache_ops import register_model_cache_ops

    register_model_cache_ops()
