"""Rank-to-host placement for heterogeneous clusters.

Phase 2 of the hardware-agnostic refactor: defines the placement engine
that maps a :class:`ParallelismConfig` onto a concrete cluster, honoring
an optional :class:`RecipeLayout` and per-host hardware capacity from
:class:`HostHardware`.

The pre-refactor assumption was "1 GPU per host", baked into
``ParallelismConfig.total_nodes`` and every runtime's ``hosts[i]``
indexing.  This engine generalizes that to "rank → (host, local-GPU
index)" while preserving identical output for the homogeneous DGX
Spark case so all existing runtimes continue to function unchanged.

Resolution algorithm:

1. If ``layout.placements`` is non-empty → honor verbatim.
2. Else compute per-host capacity from ``host_hardware`` (DGX Spark
   default via :func:`default_dgx_spark_hardware` for hosts without an
   explicit entry).
3. If every placed host shares one accelerator vendor → pack ranks
   sequentially (host *i* receives up to ``capacity_i`` ranks before
   moving to host *i+1*).
4. Else → :class:`LayoutRequiredError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from sparkrun.core.hardware import HostHardware, default_dgx_spark_hardware
from sparkrun.core.layout import RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig


class PlacementError(Exception):
    """Base class for placement engine errors."""


class LayoutRequiredError(PlacementError):
    """Raised when a heterogeneous cluster is missing an explicit ``recipe.layout``.

    Heterogeneous clusters can't be auto-fit (vendor-mixed clusters in
    particular may require splitting work along boundaries the engine
    can't safely infer).  The recipe must declare placements explicitly.
    """


class InsufficientCapacityError(PlacementError):
    """Raised when the cluster doesn't have enough accelerator slots for the requested parallelism."""


@dataclass(frozen=True)
class RankSlot:
    """One global rank's home: the host and local accelerator index."""

    host: str
    local_gpu: int


@dataclass(frozen=True)
class RankAssignment:
    """Concrete mapping from global rank to (host, local-GPU index)."""

    by_rank: tuple[RankSlot, ...]
    """``by_rank[i]`` is the slot for global rank *i*."""

    hosts_used: tuple[str, ...]
    """Distinct hosts that participate, in rank-major order (first appearance)."""

    def host_for_rank(self, rank: int) -> str:
        """Host that runs *rank*."""
        return self.by_rank[rank].host

    def local_gpu_for_rank(self, rank: int) -> int:
        """Local accelerator index that runs *rank* on its host."""
        return self.by_rank[rank].local_gpu

    def ranks_on_host(self, host: str) -> tuple[int, ...]:
        """Global ranks scheduled on *host*, in ascending order."""
        return tuple(i for i, slot in enumerate(self.by_rank) if slot.host == host)

    @property
    def total_ranks(self) -> int:
        return len(self.by_rank)

    @property
    def max_ranks_per_host(self) -> int:
        """Largest number of ranks assigned to any single host (slot count for MPI)."""
        if not self.by_rank:
            return 0
        counts: dict[str, int] = {}
        for slot in self.by_rank:
            counts[slot.host] = counts.get(slot.host, 0) + 1
        return max(counts.values())


def _hw_for(host: str, host_hardware: Mapping[str, HostHardware] | None) -> HostHardware:
    """Return the per-host hardware spec or the DGX Spark default fallback."""
    if host_hardware and host in host_hardware:
        return host_hardware[host]
    return default_dgx_spark_hardware()


def _host_capacity(hw: HostHardware) -> int:
    """Total accelerator count across all entries on this host."""
    return hw.total_gpus


def _host_vendor(hw: HostHardware) -> str | None:
    """Single vendor name iff this host advertises exactly one vendor, else ``None``."""
    vendors = hw.vendors
    return next(iter(vendors)) if len(vendors) == 1 else None


def compute_placement(
    parallelism: ParallelismConfig,
    hosts: list[str],
    *,
    host_hardware: Mapping[str, HostHardware] | None = None,
    layout: RecipeLayout | None = None,
) -> RankAssignment:
    """Map global ranks onto a cluster, honoring an optional explicit layout.

    Args:
        parallelism: ``tp/pp/dp/ep/cp`` dimensions (``total_gpus = tp*pp*dp``).
        hosts: Ordered host list as it appears on the cluster definition.
        host_hardware: Optional per-host accelerator metadata.  Missing
            entries default to DGX Spark (1× GB10, 121 GB).
        layout: Optional explicit recipe layout.  When provided with
            non-empty ``placements`` it is honored verbatim.

    Returns:
        A :class:`RankAssignment` covering ``parallelism.total_gpus`` ranks.

    Raises:
        LayoutRequiredError: Heterogeneous cluster (>1 vendor across
            placed hosts) without an explicit layout.
        InsufficientCapacityError: Cluster doesn't have enough slots
            for the requested parallelism (auto-fit path).
        PlacementError: Explicit layout doesn't cover ``total_gpus`` ranks
            or references unknown hosts.
    """
    total_gpus = parallelism.total_gpus

    if layout is not None and layout.placements:
        return _placement_from_layout(layout, hosts, total_gpus)

    return _auto_pack(hosts, host_hardware, total_gpus)


def _placement_from_layout(
    layout: RecipeLayout,
    hosts: list[str],
    total_gpus: int,
) -> RankAssignment:
    """Honor an explicit ``RecipeLayout.placements`` verbatim."""
    host_set = set(hosts)
    by_rank_dict: dict[int, RankSlot] = {}
    hosts_used: list[str] = []
    seen_hosts: set[str] = set()

    for placement in layout.placements:
        if placement.host not in host_set:
            raise PlacementError("Layout placement references host '%s' not present in cluster hosts %s" % (placement.host, hosts))
        if placement.host not in seen_hosts:
            hosts_used.append(placement.host)
            seen_hosts.add(placement.host)

        local_gpus = placement.local_gpus or tuple(range(len(placement.ranks)))
        if len(local_gpus) != len(placement.ranks):
            raise PlacementError(
                "Layout placement for host '%s' has %d ranks but %d local_gpus" % (placement.host, len(placement.ranks), len(local_gpus))
            )
        for rank, local_gpu in zip(placement.ranks, local_gpus):
            if rank in by_rank_dict:
                raise PlacementError("Layout assigns rank %d to multiple hosts" % rank)
            by_rank_dict[rank] = RankSlot(host=placement.host, local_gpu=int(local_gpu))

    if total_gpus > 0:
        missing = sorted(set(range(total_gpus)) - by_rank_dict.keys())
        if missing:
            raise PlacementError("Layout does not cover ranks %s (parallelism requires %d ranks)" % (missing, total_gpus))

    ordered = tuple(by_rank_dict[i] for i in sorted(by_rank_dict)) if not total_gpus else tuple(by_rank_dict[i] for i in range(total_gpus))
    return RankAssignment(by_rank=ordered, hosts_used=tuple(hosts_used))


def _auto_pack(
    hosts: list[str],
    host_hardware: Mapping[str, HostHardware] | None,
    total_gpus: int,
) -> RankAssignment:
    """Sequentially pack ranks onto hosts up to each host's capacity.

    Requires that all placed hosts share a single accelerator vendor;
    raises :class:`LayoutRequiredError` for multi-vendor clusters.

    A multi-vendor cluster is one where the *placed* hosts (those
    receiving at least one rank) advertise more than one distinct
    vendor across their accelerators.  Hosts beyond the parallelism
    budget are not consulted, so a cluster with mixed vendors can
    still be auto-fit as long as the leading slice it consumes is
    single-vendor.
    """
    if total_gpus <= 0:
        return RankAssignment(by_rank=(), hosts_used=())

    by_rank: list[RankSlot] = []
    hosts_used: list[str] = []
    placed_vendors: set[str] = set()
    remaining = total_gpus

    for host in hosts:
        if remaining <= 0:
            break
        hw = _hw_for(host, host_hardware)
        capacity = _host_capacity(hw)
        if capacity <= 0:
            continue
        vendor = _host_vendor(hw)
        if vendor is None:
            # Multi-vendor *single host* (e.g. Apple M5 + RTX): refuse to auto-fit.
            raise LayoutRequiredError(
                "Host '%s' advertises multiple accelerator vendors (%s); "
                "recipe.layout.placements must specify which ranks land on which accelerator" % (host, sorted(hw.vendors))
            )
        placed_vendors.add(vendor)
        if len(placed_vendors) > 1:
            raise LayoutRequiredError(
                "Cluster spans multiple accelerator vendors %s; "
                "recipe.layout.placements is required for heterogeneous-vendor clusters" % sorted(placed_vendors)
            )
        take = min(capacity, remaining)
        for local_idx in range(take):
            by_rank.append(RankSlot(host=host, local_gpu=local_idx))
        hosts_used.append(host)
        remaining -= take

    if remaining > 0:
        raise InsufficientCapacityError(
            "Cluster cannot satisfy %d ranks: only %d accelerator slot(s) available across %d host(s)"
            % (total_gpus, total_gpus - remaining, len(hosts))
        )

    return RankAssignment(by_rank=tuple(by_rank), hosts_used=tuple(hosts_used))
