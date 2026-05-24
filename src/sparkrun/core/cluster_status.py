"""Cluster status dataclasses — the shape of "what's running where?"

This module is **data only**.  Production of :class:`ClusterStatus`
lives behind the :class:`~sparkrun.orchestration.executors._base.Executor`
ABC's ``query_status`` method — there is no separate extension point
for status providers.  Each Executor knows how to inspect its own
backend (Docker via ``docker ps``, K8s via ``kubectl get pods``, …).

Consumers (the future ``OccupancyAwareScheduler``, the
``sparkrun status`` CLI, the ``cluster monitor`` TUI) all flow
through ``sparkrun.api.status`` → ``executor.query_status``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunningWorkload:
    """A single sparkrun-launched workload occupying slots on one host.

    :attr:`memory_used_gb` and :attr:`util_fraction` are optional
    per-workload resource accounting fields populated by status
    providers that can observe them (e.g. parsing ``nvidia-smi``).
    Fractional-capable schedulers read these to compute available
    capacity for occupancy-aware placement.
    """

    cluster_id: str
    intent_id: str | None = None
    """Hex intent identifier (the deterministic prefix of cluster_id).

    Recovered from the ``sparkrun.intent_id`` container label when
    emitted; otherwise derived from the cluster_id's intent prefix or
    enriched from cached job metadata.  ``None`` indicates a workload
    whose container name does not parse as a canonical sparkrun
    identifier.
    """
    recipe_name: str | None = None
    runtime_name: str | None = None
    started_at: float | None = None
    ranks_on_host: int = 1
    container_ids: tuple[str, ...] = field(default_factory=tuple)
    memory_used_gb: float | None = None
    util_fraction: float | None = None


@dataclass(frozen=True)
class GpuOccupancy:
    """Per-accelerator occupancy detail on a host.

    Populated by status providers that can introspect per-GPU usage
    (e.g. via ``nvidia-smi --query-gpu=memory.used,utilization.gpu``).
    Consumed by fractional-capable schedulers to decide whether a new
    rank's :class:`~sparkrun.core.scheduler.ResourceRequest` fits on
    each accelerator.
    """

    gpu_index: int
    used_memory_gb: float = 0.0
    used_util_fraction: float = 0.0
    workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HostOccupancy:
    """Per-host accelerator occupancy at the time of the snapshot.

    :attr:`gpus` carries per-accelerator detail when the producing
    executor can supply it; otherwise it's empty and consumers should
    fall back to host-level :attr:`used_slots` / :attr:`free_slots`.
    """

    host: str
    workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)
    used_slots: int = 0
    free_slots: int = 0
    free_memory_gb: float | None = None
    gpus: tuple[GpuOccupancy, ...] = field(default_factory=tuple)

    @property
    def total_slots(self) -> int:
        """Total accelerator slots = used + free."""
        return self.used_slots + self.free_slots


@dataclass(frozen=True)
class ClusterStatus:
    """Snapshot of cluster occupancy produced by an Executor.

    ``hosts`` carries one entry per host inspected (in input order).
    ``executor`` is the registered name of the executor that produced
    the snapshot (e.g. ``"docker"``) — primarily useful for diagnostics
    and asymmetry detection (e.g. asking the docker executor about a
    k8s-managed host).
    """

    hosts: tuple[HostOccupancy, ...] = field(default_factory=tuple)
    queried_at: float = 0.0
    executor: str = ""

    def for_host(self, host: str) -> HostOccupancy | None:
        """Return the :class:`HostOccupancy` for *host*, or ``None`` if absent."""
        for entry in self.hosts:
            if entry.host == host:
                return entry
        return None

    def free_slots(self, host: str) -> int:
        """Free accelerator slots on *host*; 0 when host is absent."""
        entry = self.for_host(host)
        return entry.free_slots if entry is not None else 0

    def running_cluster_ids(self) -> tuple[str, ...]:
        """Distinct cluster_ids running anywhere in the snapshot."""
        seen: list[str] = []
        for entry in self.hosts:
            for w in entry.workloads:
                if w.cluster_id not in seen:
                    seen.append(w.cluster_id)
        return tuple(seen)


def empty_status(hosts: list[str], executor: str = "") -> ClusterStatus:
    """Build a zero-occupancy snapshot — every host fully free, no workloads.

    Used as a safe default by executors that don't implement
    introspection (and as a sentinel in tests).
    """
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=h) for h in hosts),
        queried_at=0.0,
        executor=executor,
    )
