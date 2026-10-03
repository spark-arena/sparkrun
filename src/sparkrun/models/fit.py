"""Cluster-aware fit check between a model VRAM estimate and a placement.

Splits "model VRAM requirement" (an intrinsic property of model +
parallelism, computed by :func:`sparkrun.models.vram.estimate_vram`)
from "does it fit on this cluster" (a property of the placement against
per-host accelerator memory resolved from inventory and platform defaults in
:class:`~sparkrun.core.hardware.HostHardware`).

All target fit reporting uses this result, including unknown-capacity states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sparkrun.core.allocations import EXCLUSIVE_GPU_THRESHOLD

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.scheduler import RankAssignment
    from sparkrun.models.vram import VRAMEstimate


@dataclass(frozen=True)
class HostFitDetail:
    """Fit result for a single host receiving one or more ranks."""

    host: str
    ranks_assigned: int
    """Number of ranks placed on this host (≥1)."""

    vram_per_rank_gb: float
    """Per-rank VRAM requirement (= :attr:`VRAMEstimate.total_per_gpu_gb`)."""

    accelerator_memory_gb: float | None
    """Smallest active memory budget across this host's assigned accelerators.

    Usable = nominal ``memory_gb × max_gpu_memory_utilization`` (the cap
    resolved by :func:`sparkrun.core.limits.resolve_max_gpu_memory_utilization`).
    This is the figure the fit decision is made against. ``None`` when neither
    inventory nor the platform supplies capacity for any accelerator on the host.
    Fit cannot then be verified; the host is reported as ``ok=True`` with a warning
    rather than failing the whole check.
    """

    headroom_gb: float | None
    """``accelerator_memory_gb (usable) - vram_per_rank_gb``, or ``None`` when unknown."""

    ok: bool
    """``True`` when the per-rank requirement fits the smallest usable accelerator on the host."""

    note: str = ""
    """Human-readable explanation when the result is unusual (unknown memory, etc.)."""

    nominal_memory_gb: float | None = None
    """Nominal per-accelerator memory of the limiting accelerator (pre-cap), or ``None``."""

    max_gpu_memory_utilization: float | None = None
    """Usable-memory cap applied to the limiting accelerator (``1.0`` = no cap), or ``None``."""

    memory_capacity_source: str | None = None
    hardware_source: str = "inventory"
    memory_limit_source: str | None = None
    memory_estimate_complete: bool = True
    memory_estimate_gaps: tuple[str, ...] = ()
    """:attr:`VRAMEstimate.memory_estimate_gaps`, so a report can name them."""
    allocation_mode: str = "exclusive"

    @property
    def memory_capacity_verification(self) -> str:
        from sparkrun.core.hardware_observations import capacity_verification

        return capacity_verification(self.accelerator_memory_gb, self.memory_capacity_source)

    @property
    def status(self) -> str:
        if not self.ok:
            return "exceeds"
        if (
            self.accelerator_memory_gb is None
            or not self.memory_estimate_complete
            or self.hardware_source == "assumed"
            or self.memory_capacity_verification in {"unknown", "estimated"}
        ):
            return "unknown"
        return "fits"

    @property
    def unverified_reasons(self) -> tuple[str, ...]:
        """Why :attr:`status` is ``"unknown"`` (empty otherwise), naming the actual gap.

        Capacity provenance and estimate completeness are independent causes;
        a generic "partial estimate or unverified capacity" sends the reader
        to check hardware that was in fact probed.
        """
        if self.status != "unknown":
            return ()
        reasons = []
        if self.accelerator_memory_gb is None:
            reasons.append("accelerator capacity unknown")
        elif self.memory_capacity_verification == "estimated" and self.hardware_source != "assumed":
            # Assumed hardware always has an estimated capacity; say it once.
            reasons.append("capacity is a platform estimate, not measured")
        if self.hardware_source == "assumed":
            reasons.append("hardware assumed, not probed")
        if not self.memory_estimate_complete:
            reasons.extend(self.memory_estimate_gaps or ("memory estimate incomplete",))
        return tuple(reasons)


@dataclass
class FitResult:
    """Aggregate fit decision across every host receiving ranks."""

    ok: bool
    """``True`` iff every host with resolved memory satisfies the per-rank requirement."""

    per_host: dict[str, HostFitDetail] = field(default_factory=dict)
    """Detail per host that participates in the placement."""

    warnings: list[str] = field(default_factory=list)
    """Soft issues: unknown accelerator memory, missing host metadata, etc."""

    @property
    def status(self) -> str:
        """Memory evidence, separate from permissive legacy/admission ``ok``."""
        if not self.ok:
            return "exceeds"
        if not self.per_host or any(d.status == "unknown" for d in self.per_host.values()):
            return "unknown"
        return "fits"

    @property
    def unverified_reasons(self) -> tuple[str, ...]:
        """Distinct per-host :attr:`HostFitDetail.unverified_reasons`, in host order."""
        reasons: list[str] = []
        for detail in self.per_host.values():
            reasons.extend(r for r in detail.unverified_reasons if r not in reasons)
        return tuple(reasons)

    @property
    def hosts_used(self) -> tuple[str, ...]:
        """Hosts receiving at least one rank, in placement order."""
        return tuple(self.per_host.keys())

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "per_host": {h: _detail_to_dict(d) for h, d in self.per_host.items()},
            "warnings": list(self.warnings),
        }


def _detail_to_dict(d: HostFitDetail) -> dict:
    return {
        "host": d.host,
        "ranks_assigned": d.ranks_assigned,
        "vram_per_rank_gb": d.vram_per_rank_gb,
        "accelerator_memory_gb": d.accelerator_memory_gb,
        "nominal_memory_gb": d.nominal_memory_gb,
        "max_gpu_memory_utilization": d.max_gpu_memory_utilization,
        "memory_limit_source": d.memory_limit_source,
        "memory_capacity_source": d.memory_capacity_source,
        "memory_capacity_verification": d.memory_capacity_verification,
        "hardware_source": d.hardware_source,
        "status": d.status,
        "memory_estimate_complete": d.memory_estimate_complete,
        "unverified_reasons": list(d.unverified_reasons),
        "allocation_mode": d.allocation_mode,
        "headroom_gb": d.headroom_gb,
        "ok": d.ok,
        "note": d.note,
    }


def _limiting_accelerator(hw, cluster, indices=None, estimate=None) -> tuple[float, float, float, str, str] | None:
    """Find the accelerator with the smallest *usable* memory on *hw*.

    Returns ``(usable_gb, nominal_gb, cap, source)`` for the limiting assigned
    accelerator, or ``None`` if any assigned accelerator has unknown capacity.  Usable memory
    applies the scheduling/fit cap resolved by
    :func:`sparkrun.core.limits.resolve_max_gpu_memory_utilization`.
    """
    from sparkrun.core.limits import resolve_memory_limit, resolve_accelerator_memory

    best: tuple[float, float, float, str, str] | None = None
    slots = [a for a in hw.accelerators for _ in range(a.count)]
    selected = range(len(slots)) if indices is None else indices
    if any(index < 0 or index >= len(slots) for index in selected):
        return None
    for index in selected:
        accel = slots[index]
        capacity, capacity_source = resolve_accelerator_memory(accel, hw)
        if capacity is None:
            return None
        cap, source = resolve_memory_limit(accel, hw, cluster)
        usable = estimate.fit_budget_gb(capacity, cap) if estimate is not None else capacity * cap
        if usable < capacity * cap:
            source = "runtime gpu_memory_utilization"
            cap = usable / capacity if capacity else 0.0
        if best is None or usable < best[0]:
            best = (usable, capacity, cap, source, capacity_source)
    return best


def check_fit(
    estimate: VRAMEstimate,
    cluster: ClusterDefinition,
    placement: RankAssignment | None = None,
) -> FitResult:
    """Check whether *estimate*'s per-rank VRAM requirement fits each placed host.

    Per-rank VRAM is :attr:`VRAMEstimate.total_per_gpu_gb`; this is the
    memory one GPU needs after tensor/pipeline sharding.  A host with
    multiple ranks is compared against its smallest assigned accelerator's
    active budget. Unassigned accelerators do not affect this report.
    This estimates runtime fit; shared allocation admission separately checks
    the sum of reservations on each accelerator.

    Hosts without inventory or platform capacity are reported ``ok=True``
    with a warning: sparkrun cannot verify fit without a capacity estimate.
    Use ``sparkrun cluster update --infer-hardware`` to refresh inventory;
    qualified platform defaults also apply to existing saved probe records.

    Args:
        estimate: Result of :func:`sparkrun.models.vram.estimate_vram`.
        cluster: Cluster definition (provides per-host hardware metadata).
        placement: Rank-to-host assignment from
            ``sparkrun.api.schedule`` / :func:`sparkrun.schedulers.greedy.pack`.
            None reports candidate devices with zero assigned ranks.

    Returns:
        :class:`FitResult` with per-host details and aggregate ``ok``.
    """
    vram_per_rank = float(estimate.total_per_gpu_gb)
    per_host: dict[str, HostFitDetail] = {}
    warnings: list[str] = []
    overall_ok = True

    for host in placement.hosts_used if placement is not None else cluster.hosts:
        ranks = placement.ranks_on_host(host) if placement is not None else ()
        hw = cluster.hardware_for(host)
        indices = {placement.local_gpu_for_rank(rank) for rank in ranks} if placement is not None else None
        limiting = _limiting_accelerator(hw, cluster, indices, estimate)
        mode = (
            "exclusive"
            if (placement is None or all(placement.by_rank[r].util_fraction >= EXCLUSIVE_GPU_THRESHOLD for r in ranks))
            else "shared"
        )

        if limiting is None:
            note = "accelerator memory capacity unavailable from inventory or platform; fit not verified"
            warnings.append("%s: %s" % (host, note))
            detail = HostFitDetail(
                host=host,
                ranks_assigned=len(ranks),
                vram_per_rank_gb=vram_per_rank,
                accelerator_memory_gb=None,
                headroom_gb=None,
                ok=True,
                note=note,
                memory_estimate_complete=estimate.memory_estimate_complete,
                memory_estimate_gaps=estimate.memory_estimate_gaps,
                memory_capacity_source="unknown",
                allocation_mode=mode,
                hardware_source=hw.source,
            )
        else:
            usable_mem, nominal_mem, cap, source, capacity_source = limiting
            headroom = usable_mem - vram_per_rank
            host_ok = vram_per_rank <= usable_mem
            if not host_ok:
                overall_ok = False
            detail = HostFitDetail(
                host=host,
                ranks_assigned=len(ranks),
                vram_per_rank_gb=vram_per_rank,
                accelerator_memory_gb=usable_mem,
                headroom_gb=headroom,
                ok=host_ok,
                nominal_memory_gb=nominal_mem,
                max_gpu_memory_utilization=cap,
                memory_limit_source=source,
                memory_capacity_source=capacity_source,
                memory_estimate_complete=estimate.memory_estimate_complete,
                memory_estimate_gaps=estimate.memory_estimate_gaps,
                allocation_mode=mode,
                hardware_source=hw.source,
            )
        if hw.source == "assumed":
            warnings.append("%s: hardware assumed by application policy; probe the target to verify fit" % host)
        per_host[host] = detail

    return FitResult(ok=overall_ok, per_host=per_host, warnings=warnings)
