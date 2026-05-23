"""``sparkrun.api.status`` — inspect running workloads via the resolved Executor.

The status surface routes through the *same* executor resolution chain
the launcher uses (CLI > recipe > cluster > runtime > SparkrunConfig),
so the inspector matches the launcher: whatever would have run a
workload is what's asked about it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def status(
    hosts: list[str],
    *,
    executor: str | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    ssh_kwargs: dict | None = None,
    sctx: "SparkrunContext | None" = None,
) -> "ClusterStatus":
    """Return a :class:`ClusterStatus` snapshot of *hosts*.

    Args:
        hosts: Host list to inspect.
        executor: Optional executor name override (CLI-level).  When
            ``None``, the executor is resolved via the standard chain
            with *cluster* providing the cluster row.
        cluster: Named cluster or pre-loaded definition.  When set,
            the cluster's ``executor`` / ``executor_config`` and
            ``hosts_hardware`` flow into the resolution and the
            query.  When ``None``, default executor is used and
            hardware falls back to DGX Spark per host.
        ssh_kwargs: Optional SSH connection kwargs (forwarded to the
            executor's ``query_status``).
        sctx: Optional shared :class:`SparkrunContext`.  Provides
            cluster manager + SAF variables for chained-call sharing.

    Returns:
        A :class:`ClusterStatus` snapshot.  Unreachable hosts are
        omitted from :attr:`ClusterStatus.hosts`; callers can detect
        this with ``status.for_host(h) is None``.
    """
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.orchestration.executor import resolve_executor

    # Always end up with a populated ClusterDefinition; hosts are the
    # explicit list passed in.
    cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
    cli_overrides = {"executor": executor} if executor else None
    resolved = resolve_executor(
        cluster=cluster_def,
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
        v=sctx.variables if sctx is not None else None,
    )
    return resolved.query_status(
        list(hosts),
        ssh_kwargs=ssh_kwargs,
        host_hardware=cluster_def.hosts_hardware or None,
    )


__all__ = ["status"]
