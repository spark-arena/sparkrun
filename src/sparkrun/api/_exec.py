"""``sparkrun.api.open_exec_target`` — enter a running workload's container.

The interactive peer of :func:`sparkrun.api.open_log_target`, resolved through
the same :func:`~sparkrun.api._workload.open_workload` so both address the same
containers through the same executor. The runtime names the containers
(:meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources`, head first), the
executor builds the argv
(:meth:`~sparkrun.orchestration.executors._base.Executor.interactive_exec_argv`),
and the caller runs it: this module never touches a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sparkrun.api._errors import SparkrunError
from sparkrun.core.log_source import SCOPE_ALL, SCOPE_HEAD, LogSource

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.executors._base import Executor


@dataclass(frozen=True)
class ExecTarget:
    """One container of a running workload, ready to be entered.

    ``nodes`` lists every container head first; ``node`` indexes it, the same
    numbering ``sparkrun exec <workload>/<node>`` uses.
    """

    workload_id: str
    runtime: str
    node: int
    source: LogSource
    nodes: tuple[LogSource, ...]
    _executor: "Executor" = field(repr=False, compare=False)
    _ssh_kwargs: dict = field(repr=False, compare=False, default_factory=dict)

    @property
    def host(self) -> str:
        return self.source.host

    @property
    def container(self) -> str:
        return self.source.container

    def argv(self, command: str | None = None, *, tty: bool = False) -> list[str]:
        """Control-machine argv for a shell (or *command*, one shell string) in this container."""
        try:
            return self._executor.interactive_exec_argv(
                self.source.host,
                self.source.container,
                command,
                tty=tty,
                ssh_kwargs=self._ssh_kwargs,
                resource=self.source.resource,
            )
        except NotImplementedError as e:
            raise SparkrunError(str(e)) from None


def open_exec_target(
    workload_id: str | None = None,
    *,
    recipe: "str | Recipe | None" = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    node: int = 0,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> ExecTarget:
    """Resolve one container of a running workload for an interactive exec.

    Addressing is :func:`~sparkrun.api.open_log_target`'s: ``workload_id``, or
    ``recipe`` plus a host source (live intent discovery). *node* indexes the
    workload's containers head first. The selected container is checked for
    liveness with the same status sweep ``logs`` uses, so a dead one is
    reported with the executor's post-mortem hints instead of a raw
    substrate error.

    Raises:
        JobNotFound: No such workload, or the selected container is not running.
        AmbiguousWorkload: The recipe matches several running workloads.
        SparkrunError: *node* is out of range, or no workload was addressed.
    """
    from sparkrun.api._logs import _verify_log_source_alive
    from sparkrun.api._workload import open_workload

    if node < 0:
        raise SparkrunError("node must be >= 0, got %d" % node)
    access = open_workload(
        workload_id,
        recipe=recipe,
        hosts=hosts,
        overrides=overrides,
        cluster=cluster,
        cache_dir=cache_dir,
        operation="Exec resolution",
        sctx=sctx,
    )
    if access.native:
        try:
            sources = access.executor.native_log_sources(access.cluster_id, metadata=access.meta, scope=SCOPE_ALL)
        except (SparkrunError, ValueError):
            raise
        except Exception as exc:
            raise SparkrunError("Native workload resolution failed: %s" % exc) from exc
    else:
        sources = access.runtime.log_sources(access.cluster_id, access.hosts, is_solo=len(access.hosts) <= 1, scope=SCOPE_ALL)
    sources = tuple(sources)
    if node >= len(sources):
        listing = ", ".join("%d=%s/%s" % (i, s.host, s.container) for i, s in enumerate(sources))
        raise SparkrunError("Workload %s has no node %d (nodes: %s)" % (access.cluster_id, node, listing or "none"))
    source = sources[node]
    if not access.native:
        _verify_log_source_alive(
            access.executor,
            [source],
            access.ssh_kwargs,
            access.cluster_id,
            access.cache_dir,
            scope=SCOPE_HEAD,
            host_hardware=access.cluster.hosts_hardware or None,
        )
    return ExecTarget(
        workload_id=access.cluster_id,
        runtime=access.runtime.runtime_name,
        node=node,
        source=source,
        nodes=sources,
        _executor=access.executor,
        _ssh_kwargs=dict(access.ssh_kwargs),
    )


__all__ = ["ExecTarget", "open_exec_target"]
