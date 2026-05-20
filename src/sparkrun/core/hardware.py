"""Per-host hardware metadata for heterogeneous clusters.

Defines :class:`AcceleratorSpec` (a single accelerator on a host) and
:class:`HostHardware` (all accelerators + fingerprint on one host).

Hosts without an explicit metadata entry default to DGX Spark via
:func:`default_dgx_spark_hardware` so existing single-platform clusters
keep working without recipe / cluster-file changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AcceleratorSpec:
    """A single accelerator (or group of identical accelerators) on a host.

    ``count`` lets one entry cover multiple identical GPUs (e.g. an 8×H200
    server is one ``AcceleratorSpec`` with ``count=8``).  Heterogeneous
    accelerators on a single host are represented as multiple entries.
    """

    vendor: str
    """Vendor identifier: ``"nvidia"``, ``"amd"``, ``"intel"``, ``"apple"``, ``"cpu"``."""

    model: str
    """Model identifier: ``"gb10"``, ``"rtx-pro-6000"``, ``"mi300x"``, ``"h200"``, ``"m5"``, etc."""

    count: int = 1
    """Number of identical accelerators of this type on the host."""

    memory_gb: float | None = None
    """Per-accelerator memory in GB (e.g. 80 for H100, 192 for MI300X, 121 for DGX Spark unified)."""

    capabilities: frozenset[str] = frozenset()
    """Free-form capability tags: ``"cuda"``, ``"rocm"``, ``"nvlink"``, ``"rdma:roce-v2"``, etc."""

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serializable form. Omits defaults to keep YAML small."""
        d: dict[str, Any] = {"vendor": self.vendor, "model": self.model}
        if self.count != 1:
            d["count"] = self.count
        if self.memory_gb is not None:
            d["memory_gb"] = self.memory_gb
        if self.capabilities:
            d["capabilities"] = sorted(self.capabilities)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcceleratorSpec:
        caps = data.get("capabilities") or ()
        return cls(
            vendor=str(data.get("vendor", "")),
            model=str(data.get("model", "")),
            count=int(data.get("count", 1)),
            memory_gb=float(data["memory_gb"]) if data.get("memory_gb") is not None else None,
            capabilities=frozenset(str(c) for c in caps),
        )


@dataclass
class HostHardware:
    """All accelerators on a single host, plus optional fingerprint metadata."""

    accelerators: list[AcceleratorSpec] = field(default_factory=list)
    """Ordered list of accelerators present on this host."""

    fingerprint: str | None = None
    """Stable hash identifying the detected hardware (populated by detection)."""

    notes: str = ""
    """Free-form notes for users (e.g. ``"manually configured"`` or ``"detected 2026-05-14"``)."""

    ib_info: dict | None = None
    """Raw InfiniBand detection results from the combined probe script.

    Populated by :func:`sparkrun.core.hardware_probe.probe_host` when both
    the accelerator fingerprint and IB detection run in a single SSH
    round-trip.  ``None`` when hardware was loaded from a cluster YAML file
    or constructed without an IB probe.

    .. todo::
        Convert to a proper dataclass (``IbProbeResult``) in a follow-up
        once the dict shape has stabilised across all callers.
    """

    @property
    def total_gpus(self) -> int:
        """Total accelerator count across all ``accelerators`` entries."""
        return sum(a.count for a in self.accelerators)

    @property
    def vendors(self) -> frozenset[str]:
        """Set of distinct vendors present on this host."""
        return frozenset(a.vendor for a in self.accelerators)

    def has_capability(self, capability: str) -> bool:
        """True if any accelerator on this host advertises *capability*."""
        return any(capability in a.capabilities for a in self.accelerators)

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serializable form. Omits empty optional fields."""
        d: dict[str, Any] = {"accelerators": [a.to_dict() for a in self.accelerators]}
        if self.fingerprint:
            d["fingerprint"] = self.fingerprint
        if self.notes:
            d["notes"] = self.notes
        # ib_info is intentionally not serialised to YAML — it is ephemeral
        # probe data, not stored cluster metadata.
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HostHardware:
        raw_accels = data.get("accelerators") or []
        accels = [AcceleratorSpec.from_dict(a) for a in raw_accels if isinstance(a, dict)]
        return cls(
            accelerators=accels,
            fingerprint=data.get("fingerprint") or None,
            notes=str(data.get("notes", "")),
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# DGX Spark GB10: 1 GPU per host, 121 GB available unified memory.
# Mirrors models.vram.DGX_SPARK_VRAM_GB.  Kept in this module so callers
# that resolve hardware never need to import the VRAM module.
_DGX_SPARK_VRAM_GB = 121.0


def default_dgx_spark_hardware() -> HostHardware:
    """Default ``HostHardware`` for hosts without explicit metadata.

    Treats every host as a DGX Spark (1× GB10, 121 GB unified memory,
    CUDA + RoCEv2 RDMA) so clusters that don't ship per-host hardware
    blocks keep working unchanged.
    """
    return HostHardware(
        accelerators=[
            AcceleratorSpec(
                vendor="nvidia",
                model="gb10",
                count=1,
                memory_gb=_DGX_SPARK_VRAM_GB,
                capabilities=frozenset({"cuda", "unified-memory", "rdma:roce-v2"}),
            )
        ],
        notes="default (no explicit hardware metadata)",
    )
