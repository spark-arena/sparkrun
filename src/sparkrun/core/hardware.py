"""Per-host hardware metadata for heterogeneous clusters.

Defines :class:`AcceleratorSpec` (a single accelerator on a host) and
:class:`HostHardware` (all accelerators + fingerprint on one host).

Missing metadata is resolved only through explicit application policy.
Assumed hardware carries provenance and is never device detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any


_COMPUTE_CAPABILITY_RE = re.compile(r"^(\d{1,2})\.(\d)$")
_ARCH_RE = re.compile(r"^sm_?(\d{2,3})[af]?$")


def normalize_compute_capability(value: Any) -> str | None:
    """Canonical ``"<major>.<minor>"`` form, or ``None`` when *value* is not one.

    Accepts what ``nvidia-smi --query-gpu=compute_cap`` prints (``"12.1"``) and
    the numeric YAML spelling a hand-written inventory produces (``12.1``).
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    match = _COMPUTE_CAPABILITY_RE.match(text)
    if match is None:
        return None
    return "%d.%s" % (int(match.group(1)), match.group(2))


def compute_capability_to_arch(capability: str | None) -> str | None:
    """``"12.1"`` → ``"sm_121"``; ``None`` for anything unparseable."""
    canonical = normalize_compute_capability(capability)
    if canonical is None:
        return None
    major, minor = canonical.split(".")
    return "sm_%s%s" % (major, minor)


def normalize_arch(value: Any) -> str | None:
    """Canonical ``sm_<NN>`` spelling of an architecture name.

    Drops the arch-specific / family suffix (``sm_120a`` → ``sm_120``), so the
    feature-set spellings CUDA toolchains use compare equal to the architecture
    itself. Also accepts a compute capability (``"12.0"``).
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().lower()
    match = _ARCH_RE.match(text)
    if match is not None:
        return "sm_%s" % match.group(1)
    return compute_capability_to_arch(text)


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

    max_gpu_memory_utilization: float | None = None
    """Upper bound (``0.0`` < x ≤ ``1.0``) on the fraction of :attr:`memory_gb`
    treated as usable for **scheduling and fit** decisions
    (``usable = memory_gb × max_gpu_memory_utilization``).

    This is the *memory* axis — it scales capacity for placement/room math —
    and is distinct from the scheduler's compute ``util_fraction``.  It also
    does **not** set the serving ``--gpu-memory-utilization`` /
    ``--mem-fraction-static`` flag.  ``None`` means "no per-accelerator cap"; a
    resolved default comes from the cluster config or platform tier (see
    :func:`sparkrun.core.limits.resolve_max_gpu_memory_utilization`), falling
    back to :data:`DEFAULT_MAX_GPU_MEMORY_UTILIZATION`."""

    memory_limit_source: str | None = None
    """Ephemeral resolved budget provenance; not persisted in raw inventory."""

    memory_capacity_source: str | None = None
    """Ephemeral capacity provenance; not persisted as an inventory override."""

    compute_capability: str | None = None
    """``"<major>.<minor>"`` as recorded by inventory (probed or hand-written).

    Not the authoritative value: compute capability is a fixed property of an
    accelerator model, so a platform that knows the model declares it
    (:meth:`~sparkrun.platforms.base.HardwarePlatformPlugin.default_compute_capability`)
    and that declaration wins. This field fills in for models no platform
    declares, and is what ``validate_host`` checks the declaration against.
    Resolve through :func:`sparkrun.platforms.resolve_compute_capability`."""

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serializable form. Omits defaults to keep YAML small."""
        d: dict[str, Any] = {"vendor": self.vendor, "model": self.model}
        if self.count != 1:
            d["count"] = self.count
        if self.memory_gb is not None:
            d["memory_gb"] = self.memory_gb
        if self.capabilities:
            d["capabilities"] = sorted(self.capabilities)
        if self.max_gpu_memory_utilization is not None:
            d["max_gpu_memory_utilization"] = self.max_gpu_memory_utilization
        if self.compute_capability is not None:
            d["compute_capability"] = self.compute_capability
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcceleratorSpec:
        caps = data.get("capabilities") or ()
        raw_max_util = data.get("max_gpu_memory_utilization")
        return cls(
            vendor=str(data.get("vendor", "")),
            model=str(data.get("model", "")),
            count=int(data.get("count", 1)),
            memory_gb=float(data["memory_gb"]) if data.get("memory_gb") is not None else None,
            capabilities=frozenset(str(c) for c in caps),
            max_gpu_memory_utilization=float(raw_max_util) if raw_max_util is not None else None,
            compute_capability=normalize_compute_capability(data.get("compute_capability")),
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

    driver_versions: dict[str, str] = field(default_factory=dict)
    """Detected host driver versions keyed by accelerator vendor.

    Driver versions are software compatibility metadata, so they are
    intentionally excluded from :attr:`fingerprint`.  They are persisted in
    cluster hardware inventory when available and may also be refreshed by a
    capability-specific live preflight.
    """

    source: str = "inventory"
    """Hardware provenance: inventory, detected, or assumed by application policy."""

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

    def usable_gpu_memory_slots(self) -> list[float | None]:
        """Per-local-GPU *usable* memory budget, one entry per accelerator slot.

        Expands :attr:`accelerators` into local-index order (an entry per GPU,
        honouring :attr:`AcceleratorSpec.count`).  Each entry is
        ``memory_gb × max_gpu_memory_utilization`` — the usable-memory cap
        resolved upstream and baked into
        :attr:`AcceleratorSpec.max_gpu_memory_utilization` (see
        :func:`sparkrun.core.limits.resolved_hardware_for_scheduling`).  A
        missing cap means ``1.0`` (no cap); an entry is ``None`` when the spec
        declares no ``memory_gb`` (callers treat that slot as "memory ignored").

        Shared by the greedy and occupancy schedulers so the cap math lives in
        exactly one place while the schedulers stay free of ``platforms`` /
        ``cluster_manager`` imports.
        """
        mem: list[float | None] = []
        for spec in self.accelerators:
            if spec.memory_gb is None:
                usable: float | None = None
            else:
                cap = spec.max_gpu_memory_utilization
                usable = spec.memory_gb * (cap if cap is not None else 1.0)
            for _ in range(spec.count):
                mem.append(usable)
        return mem

    def selected(self, indices) -> HostHardware:
        """Policy view of assigned physical accelerators, preserving host facts.

        The placement retains physical GPU indices; this view is only for
        platform, compatibility, image, and backend decisions.
        """
        slots = [a for a in self.accelerators for _ in range(a.count)]
        selected = sorted(set(indices))
        if any(i < 0 or i >= len(slots) for i in selected):
            raise ValueError("Assigned accelerator index is outside hardware inventory")
        return replace(self, accelerators=[replace(slots[i], count=1) for i in selected])

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serializable form. Omits empty optional fields."""
        d: dict[str, Any] = {"accelerators": [a.to_dict() for a in self.accelerators]}
        if self.source != "inventory":
            d["source"] = self.source
        if self.fingerprint:
            d["fingerprint"] = self.fingerprint
        if self.notes:
            d["notes"] = self.notes
        if self.driver_versions:
            d["driver_versions"] = dict(sorted(self.driver_versions.items()))
        # ib_info is intentionally not serialised to YAML — it is ephemeral
        # probe data, not stored cluster metadata.
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HostHardware:
        raw_accels = data.get("accelerators") or []
        accels = [AcceleratorSpec.from_dict(a) for a in raw_accels if isinstance(a, dict)]
        return cls(
            accelerators=accels,
            source=str(data.get("source") or "inventory"),
            fingerprint=data.get("fingerprint") or None,
            notes=str(data.get("notes", "")),
            driver_versions={str(vendor): str(version) for vendor, version in (data.get("driver_versions") or {}).items()},
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Generic scheduling policy when no configured or platform cap applies.
DEFAULT_MAX_GPU_MEMORY_UTILIZATION = 1.0


def resolve_hardware(hardware: HostHardware | None = None) -> HostHardware:
    """Resolve missing inventory through the application's explicit policy.

    Platform assumptions are qualified and marked at this boundary. Consumers
    must not substitute a device identity, capacity, or network capability.
    """
    if hardware is not None:
        return hardware
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.platforms import get_platform_by_name

    profile = get_application_profile()
    platform = get_platform_by_name(profile.hardware_fallback)
    if profile.hardware_fallback != "require-metadata" and platform is not None:
        assumed = platform.assumed_hardware()
        if assumed is not None:
            return replace(assumed, source="assumed", fingerprint=None)
    raise ValueError("Target hardware metadata is required by %s; probe the target hosts before launching" % profile.id)


def resolve_host_hardware(hosts, cluster=None, placement=None) -> dict[str, HostHardware]:
    """One hardware-policy view per target, restricted to assigned devices."""
    result = {}
    for host in hosts:
        hw = cluster.hardware_for(host) if cluster is not None else resolve_hardware()
        if placement is not None:
            hw = hw.selected(placement.local_gpu_for_rank(r) for r in placement.ranks_on_host(host))
        result[host] = hw
    return result


def resolve_fallback_hardware() -> HostHardware:
    """Compatibility spelling for the centralized hardware resolver."""
    return resolve_hardware()


def default_dgx_spark_hardware() -> HostHardware:
    """Legacy explicit Spark assumption; prefer inventory or resolve_hardware."""
    from sparkrun.platforms.dgx_spark import DgxSparkPlatform

    return DgxSparkPlatform().assumed_hardware()


def __getattr__(name: str):
    if name in {"DGX_SPARK_MEMORY_GB", "DGX_SPARK_SCHEDULING_FRACTION"}:
        from sparkrun.core._platform_compat import legacy_capacity

        return legacy_capacity(name)
    raise AttributeError(name)
