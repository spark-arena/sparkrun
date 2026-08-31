"""Usable-memory cap resolution for scheduling and fit.

Resolves a per-accelerator ``max_gpu_memory_utilization`` — the fraction of
nominal :attr:`AcceleratorSpec.memory_gb` treated as usable for **scheduling /
fit** decisions (``usable = memory_gb × cap``).  This is the *memory* axis and
is distinct from the scheduler's compute ``util_fraction``; it does not affect
the serving ``--gpu-memory-utilization`` flag.

Resolution precedence (highest first):

1. ``AcceleratorSpec.max_gpu_memory_utilization`` — per-host+accelerator
2. ``cluster.accelerator_memory_limits[accel.model]`` — per-accelerator-type
3. ``cluster.max_gpu_memory_utilization`` — cluster-wide default
4. ``platform.default_max_gpu_memory_utilization(accel)`` — platform default
   (e.g. DGX Spark GB10 → 0.85)
5. :data:`~sparkrun.core.hardware.DEFAULT_MAX_GPU_MEMORY_UTILIZATION` (``1.0``)

Only values in ``(0.0, 1.0]`` are accepted at each level; anything else is
ignored and resolution falls through to the next tier.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from sparkrun.core.hardware import (
    DEFAULT_MAX_GPU_MEMORY_UTILIZATION,
    AcceleratorSpec,
    HostHardware,
    default_dgx_spark_hardware,
)

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition

logger = logging.getLogger(__name__)


def _valid_fraction(value: float | None) -> float | None:
    """Return *value* if it is a usable fraction in ``(0.0, 1.0]``, else ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 < f <= 1.0:
        return f
    logger.warning("Ignoring out-of-range max_gpu_memory_utilization=%r (must be 0.0 < x <= 1.0)", value)
    return None


def resolve_max_gpu_memory_utilization(
    accel: AcceleratorSpec,
    host_hw: HostHardware,
    cluster: "ClusterDefinition | None",
) -> float:
    """Resolve the usable-memory cap for *accel* on *host_hw*.

    See the module docstring for the precedence chain.  Always returns a
    concrete fraction in ``(0.0, 1.0]`` (``1.0`` when nothing applies).
    """
    # 1. Per-host+accelerator (explicit on the spec).
    explicit = _valid_fraction(accel.max_gpu_memory_utilization)
    if explicit is not None:
        return explicit

    if cluster is not None:
        # 2. Per-accelerator-type map.
        per_type = _valid_fraction(cluster.accelerator_memory_limits.get(accel.model))
        if per_type is not None:
            return per_type
        # 3. Cluster-wide default.
        cluster_wide = _valid_fraction(cluster.max_gpu_memory_utilization)
        if cluster_wide is not None:
            return cluster_wide

    # 4. Platform default.
    platform_default = _resolve_platform_default(accel, host_hw)
    if platform_default is not None:
        return platform_default

    # 5. Hard fallback.
    return DEFAULT_MAX_GPU_MEMORY_UTILIZATION


def _resolve_platform_default(accel: AcceleratorSpec, host_hw: HostHardware) -> float | None:
    """Platform-tier default for *accel*, or ``None`` when no platform claims the host."""
    # Lazy import to avoid a core -> platforms import cycle (platforms imports
    # core.hardware at module load).
    from sparkrun.platforms import resolve_platform

    platform = resolve_platform(host_hw)
    if platform is None:
        return None
    try:
        return _valid_fraction(platform.default_max_gpu_memory_utilization(accel))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Platform %r default_max_gpu_memory_utilization raised: %s", getattr(platform, "platform_name", "?"), e)
        return None


def usable_memory_gb(
    accel: AcceleratorSpec,
    host_hw: HostHardware,
    cluster: "ClusterDefinition | None",
) -> float | None:
    """Usable memory (GB) for *accel* = ``memory_gb × resolved cap``.

    Returns ``None`` when the accelerator declares no ``memory_gb`` (capacity
    unknown — callers fall back to "memory not verified").
    """
    if accel.memory_gb is None:
        return None
    return accel.memory_gb * resolve_max_gpu_memory_utilization(accel, host_hw, cluster)


def resolved_hardware_for_scheduling(
    cluster: "ClusterDefinition | None",
    hosts: list[str],
) -> dict[str, HostHardware]:
    """Materialize per-host hardware with the usable-memory cap folded in.

    For every host in *hosts*, resolves the full cap chain (cluster + platform
    + per-accel) once and bakes the result into each
    :class:`AcceleratorSpec.max_gpu_memory_utilization` so the schedulers — which
    only see ``host_hardware`` and must stay free of ``platforms`` /
    ``cluster_manager`` imports — can apply a single resolved fraction.

    ``memory_gb`` is left at its nominal value; only the cap field is set.  The
    returned copies are ephemeral scheduling inputs and are never persisted.
    """
    resolved: dict[str, HostHardware] = {}
    for host in hosts:
        hw = cluster.hardware_for(host) if cluster is not None else default_dgx_spark_hardware()
        new_accels = [
            dataclasses.replace(
                accel,
                max_gpu_memory_utilization=resolve_max_gpu_memory_utilization(accel, hw, cluster),
            )
            for accel in hw.accelerators
        ]
        resolved[host] = HostHardware(
            accelerators=new_accels,
            fingerprint=hw.fingerprint,
            notes=hw.notes,
            ib_info=hw.ib_info,
        )
    return resolved
