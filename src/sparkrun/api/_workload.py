"""Open an *existing* workload for an operation that reaches into it.

The shared front half of :func:`sparkrun.api.open_log_target` and
:func:`sparkrun.api.open_exec_target`: which workload, on which hosts, run by
which runtime, under which executor, over which connection. Reading logs and
opening a shell differ only in what they do once they get there, so the
resolution is written once and both are guaranteed to address the same
containers through the same executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sparkrun.api._errors import JobNotFound, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.executors._base import Executor
    from sparkrun.runtimes.base import RuntimePlugin


@dataclass(frozen=True)
class WorkloadAccess:
    """A resolved workload plus everything needed to reach its containers."""

    cluster_id: str
    meta: dict[str, Any] | None
    hosts: list[str]
    cluster: "ClusterDefinition"
    runtime: "RuntimePlugin"
    executor: "Executor"
    ssh_kwargs: dict[str, Any]
    sctx: "SparkrunContext | None"
    cache_dir: str

    @property
    def native(self) -> bool:
        """The workload is a controller-native resource (read via its executor's control plane)."""
        return bool(self.meta and self.meta.get("native_resource") is not None)


def open_workload(
    workload_id: str | None,
    *,
    recipe: "str | Recipe | None",
    hosts: list[str] | tuple[str, ...] | None,
    overrides: dict | None,
    cluster: "str | ClusterDefinition | None",
    cache_dir: str | None,
    operation: str,
    sctx: "SparkrunContext | None",
) -> WorkloadAccess:
    """Resolve a running workload's runtime, executor and connection.

    Either ``workload_id`` or ``recipe`` addresses it (see
    :func:`~sparkrun.api._resolve.resolve_workload_target`). The executor comes
    from the job's recorded executor settings, and the connection is the job's
    own. A native resource needs no host transport; everything else is
    prepared once here.

    Raises:
        JobNotFound / AmbiguousWorkload / SparkrunError: as
            :func:`~sparkrun.api._resolve.resolve_workload_target`, plus an
            unresolvable runtime.
    """
    from sparkrun.api._resolve import bind_job_cluster, prepare_transport, resolve_workload_target, scope_operation
    from sparkrun.core._executor_destination import metadata_executor_overrides
    from sparkrun.core.config import resolve_configured_cache_dir
    from sparkrun.orchestration.executor import resolve_executor

    cache_dir = str(resolve_configured_cache_dir(cache_dir, config=sctx.config if sctx is not None else None))
    target_spec = resolve_workload_target(
        workload_id=workload_id,
        recipe=recipe,
        hosts=hosts,
        overrides=overrides,
        cluster=cluster,
        cache_dir=cache_dir,
        operation=operation,
        sctx=sctx,
    )
    cluster_id, meta, target_hosts, cluster_def, resolved_recipe = target_spec[:5]
    if resolved_recipe is not None and not hosts and meta and meta.get("hosts"):
        # The recipe form returns its discovery range (the whole cluster),
        # which is right for teardown but not for addressing containers: rank i
        # lives on the i-th *placed* host, and a load-aware scheduler may have
        # placed the job on a subset. The job recorded where it actually runs.
        target_hosts = list(meta["hosts"])
    if not target_spec.transport_prepared and not (meta and meta.get("native_resource") is not None):
        prepare_transport(cluster_def)

    runtime = resolve_runtime_for_job(meta, cluster_id, recipe=resolved_recipe, sctx=sctx)
    executor = resolve_executor(
        cluster=cluster_def,
        cli_overrides=metadata_executor_overrides(meta),
        rootless=False,
        auto_user=False,
        config=sctx.config if sctx is not None else None,
        v=sctx.variables if sctx is not None else None,
    )
    cluster_def = bind_job_cluster(cluster_def, executor.resolve_target(dry_run=True), meta)
    sctx, ssh_kwargs = scope_operation(cluster_def, sctx=sctx, prepare=False)
    return WorkloadAccess(cluster_id, meta, list(target_hosts), cluster_def, runtime, executor, dict(ssh_kwargs), sctx, cache_dir)


def resolve_runtime_for_job(meta: dict | None, cluster_id: str, *, recipe=None, sctx: "SparkrunContext | None"):
    """Resolve the runtime that launched this workload.

    The runtime is what knows where its containers and logs live, so getting
    this right is what keeps ``api.logs`` from reading a container that
    doesn't exist: a Ray job's head is ``{cid}_head`` with its serve output in
    an in-container file, while a native job's head is ``{cid}_node_0``.
    Guessing wrong yields "No such container" or an empty stream — which is
    exactly what the previous hardcoded implementation did.

    Prefers the *recipe* when the caller addressed the workload by one: the
    recipe carries the runtime directly, so the recipe form keeps working
    even when the job-metadata cache is missing (a job launched from another
    control machine, or a cleared cache).  Falls back to metadata for the
    cluster_id form, where the recipe isn't known.
    """
    from sparkrun.core.bootstrap import get_runtime

    runtime_name = getattr(recipe, "runtime", None) or (meta or {}).get("runtime")
    if not runtime_name:
        raise JobNotFound(
            "No job metadata (or no runtime recorded) for workload_id %r, so sparkrun can't tell how this workload "
            "was launched. Address it by recipe instead (with its hosts or cluster)." % cluster_id
        )
    try:
        return get_runtime(runtime_name, sctx.variables if sctx is not None else None)
    except ValueError as e:
        raise SparkrunError("Cannot resolve runtime %r for workload_id %r: %s" % (runtime_name, cluster_id, e)) from e
