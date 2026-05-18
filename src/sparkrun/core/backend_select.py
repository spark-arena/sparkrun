"""Map a :class:`HostHardware` fingerprint to a concrete backend bundle.

Resolves the combination of components a host needs to run (collective
backend + executor accelerator vendor flag) from its detected hardware.
If no matching backend exists, :class:`NoMatchingBackendError` lists
both what was detected and which backends are known, so users get an
actionable error instead of a silent NVIDIA fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from sparkrun.core.hardware import HostHardware
from sparkrun.orchestration.collectives import (
    CollectiveBackend,
    UnsupportedCollectiveError,
    get_backend,
)
from sparkrun.orchestration.executor import accelerator_vendor_for


class NoMatchingBackendError(Exception):
    """Raised when sparkrun can't pick backends for a fingerprinted host.

    Carries both the detected hardware summary and the list of known
    backends so the surface is immediately actionable: the user either
    fixes detection metadata or contributes a backend implementation.
    """

    def __init__(self, host_hardware: HostHardware, known_vendors: list[str]):
        self.host_hardware = host_hardware
        self.known_vendors = known_vendors
        accels = ", ".join("%s/%s" % (a.vendor, a.model) for a in host_hardware.accelerators) or "<none>"
        super().__init__(
            "No backend matches host hardware. Detected: %s. Known vendors: %s. "
            "Either set explicit cluster.hosts_hardware or contribute a backend "
            "(see sparkrun/orchestration/collectives/)." % (accels, ", ".join(known_vendors))
        )


@dataclass(frozen=True)
class BackendBundle:
    """Concrete backend choices for a single host."""

    accelerator_vendor: str
    """Vendor flag for :class:`~sparkrun.orchestration.executor.ExecutorConfig`."""

    collective: CollectiveBackend
    """Collective backend (NCCL/RCCL/HCCL) that produces this host's env block."""


# Vendors that have at least a backend scaffold today.
_KNOWN_VENDORS: list[str] = ["nvidia", "amd", "intel"]


def select_backends(host_hardware: HostHardware) -> BackendBundle:
    """Pick collective + executor flags for a fingerprinted host.

    Args:
        host_hardware: Per-host hardware metadata, typically produced
            by :func:`sparkrun.core.fingerprint.fingerprint_host` or
            populated explicitly in cluster YAML.

    Returns:
        A :class:`BackendBundle` ready to inject into per-host configs.

    Raises:
        NoMatchingBackendError: When the host advertises no accelerators,
            mixes vendors (ambiguous), or uses a vendor sparkrun has no
            backend for (e.g. ``"apple"``, ``"cpu"``).
    """
    vendor = accelerator_vendor_for(host_hardware)
    if vendor is None or vendor not in _KNOWN_VENDORS:
        raise NoMatchingBackendError(host_hardware, list(_KNOWN_VENDORS))

    try:
        collective = get_backend(vendor)
    except UnsupportedCollectiveError as e:
        raise NoMatchingBackendError(host_hardware, list(_KNOWN_VENDORS)) from e

    return BackendBundle(accelerator_vendor=vendor, collective=collective)


def known_vendors() -> list[str]:
    """List of vendors with at least a backend scaffold (testable inspection point)."""
    return list(_KNOWN_VENDORS)
