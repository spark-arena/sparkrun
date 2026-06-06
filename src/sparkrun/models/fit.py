"""Cluster-aware fit check between a model VRAM estimate and a placement.

Splits "model VRAM requirement" (an intrinsic property of model +
parallelism, computed by :func:`sparkrun.models.vram.estimate_vram`)
from "does it fit on this cluster" (a property of the placement against
per-host accelerator memory captured in
:class:`~sparkrun.core.hardware.HostHardware`).

:attr:`VRAMEstimate.fits_dgx_spark` is retained for the single-platform
CLI output path; heterogeneous clusters must use :func:`check_fit`
since per-host budgets vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.placement import RankAssignment
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
    """Smallest per-accelerator *usable* memory across this host's accelerators.

    Usable = nominal ``memory_gb × max_gpu_memory_utilization`` (the cap
    resolved by :func:`sparkrun.core.limits.resolve_max_gpu_memory_utilization`).
    This is the figure the fit decision is made against.  ``None`` when no
    :class:`AcceleratorSpec` on the host declares ``memory_gb`` — fit cannot be
    verified and the host is reported as ``ok=True`` with a warning rather than
    failing the whole check.
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


@dataclass
class FitResult:
    """Aggregate fit decision across every host receiving ranks."""

    ok: bool
    """``True`` iff every host with declared memory satisfies the per-rank requirement."""

    per_host: dict[str, HostFitDetail] = field(default_factory=dict)
    """Detail per host that participates in the placement."""

    warnings: list[str] = field(default_factory=list)
    """Soft issues: unknown accelerator memory, missing host metadata, etc."""

    @property
    def hosts_used(self) -> tuple[str, ...]:
        """Hosts receiving at least one rank, in placement order."""
        return tuple(self.per_host.keys())

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
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
        "headroom_gb": d.headroom_gb,
        "ok": d.ok,
        "note": d.note,
    }


def _limiting_accelerator(hw, cluster) -> tuple[float, float, float] | None:
    """Find the accelerator with the smallest *usable* memory on *hw*.

    Returns ``(usable_gb, nominal_gb, cap)`` for that accelerator, or ``None``
    when no accelerator on the host declares ``memory_gb``.  Usable memory
    applies the scheduling/fit cap resolved by
    :func:`sparkrun.core.limits.resolve_max_gpu_memory_utilization`.
    """
    from sparkrun.core.limits import resolve_max_gpu_memory_utilization

    best: tuple[float, float, float] | None = None
    for accel in hw.accelerators:
        if accel.memory_gb is None:
            continue
        cap = resolve_max_gpu_memory_utilization(accel, hw, cluster)
        usable = accel.memory_gb * cap
        if best is None or usable < best[0]:
            best = (usable, accel.memory_gb, cap)
    return best


def check_fit(
    estimate: VRAMEstimate,
    cluster: ClusterDefinition,
    placement: RankAssignment,
) -> FitResult:
    """Check whether *estimate*'s per-rank VRAM requirement fits each placed host.

    Per-rank VRAM is :attr:`VRAMEstimate.total_per_gpu_gb`; this is the
    memory one GPU needs after tensor/pipeline sharding.  A host with
    multiple ranks is still expected to provide per-rank memory on each
    accelerator, so we compare per-rank against the smallest accelerator
    memory on the host (worst-case fit).

    Hosts without ``memory_gb`` metadata are reported ``ok=True`` with a
    warning — sparkrun can't verify the fit but won't block the launch
    on missing data.  Use ``sparkrun cluster update --infer-hardware``
    to populate it.

    Args:
        estimate: Result of :func:`sparkrun.models.vram.estimate_vram`.
        cluster: Cluster definition (provides per-host hardware metadata).
        placement: Rank-to-host assignment from
            :func:`sparkrun.core.placement.compute_placement`.

    Returns:
        :class:`FitResult` with per-host details and aggregate ``ok``.
    """
    vram_per_rank = float(estimate.total_per_gpu_gb)
    per_host: dict[str, HostFitDetail] = {}
    warnings: list[str] = []
    overall_ok = True

    for host in placement.hosts_used:
        ranks = placement.ranks_on_host(host)
        hw = cluster.hardware_for(host)
        limiting = _limiting_accelerator(hw, cluster)

        if limiting is None:
            note = "no memory_gb declared on host hardware; fit not verified"
            warnings.append("%s: %s" % (host, note))
            detail = HostFitDetail(
                host=host,
                ranks_assigned=len(ranks),
                vram_per_rank_gb=vram_per_rank,
                accelerator_memory_gb=None,
                headroom_gb=None,
                ok=True,
                note=note,
            )
        else:
            usable_mem, nominal_mem, cap = limiting
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
            )
        per_host[host] = detail

    return FitResult(ok=overall_ok, per_host=per_host, warnings=warnings)
