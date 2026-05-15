"""Runtime ↔ host compatibility checks (Phase 7).

Pairs a :class:`~sparkrun.runtimes.base.RuntimePlugin`'s
``requires_capability`` set with each host's
:class:`~sparkrun.core.hardware.HostHardware`, producing actionable
errors when a runtime can't safely target a host.

Capability matching: a runtime is satisfied for a host iff every entry
in :attr:`RuntimePlugin.requires_capability` appears either in some
accelerator's :attr:`AcceleratorSpec.capabilities` set on that host or
as an :attr:`AcceleratorSpec.model` name.  This dual lookup means
runtimes can pin to specific accelerators (``"gb10"``,
``"mi300x"``) without coordinating a separate capability-tag taxonomy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import HostHardware
    from sparkrun.core.placement import RankAssignment
    from sparkrun.runtimes.base import RuntimePlugin


class IncompatibleHardwareError(Exception):
    """Raised when a runtime cannot target one or more placed hosts."""

    def __init__(self, runtime_name: str, errors: list[str]):
        self.runtime_name = runtime_name
        self.errors = errors
        super().__init__("Runtime %r is incompatible with the placed cluster:\n  - %s" % (runtime_name, "\n  - ".join(errors)))


def _host_capability_set(hw: HostHardware) -> set[str]:
    """Capabilities present on this host: every accelerator's capabilities ∪ model name."""
    present: set[str] = set()
    for accel in hw.accelerators:
        present.update(accel.capabilities)
        if accel.model:
            present.add(accel.model)
    return present


def check_runtime_host_compatibility(
    runtime: RuntimePlugin,
    host: str,
    host_hardware: HostHardware,
) -> list[str]:
    """Return a list of error strings (empty when *runtime* may run on *host*).

    Args:
        runtime: Initialised runtime plugin.
        host: Host name (for error messages).
        host_hardware: Host's hardware metadata.

    Returns:
        Empty list when compatible.  When incompatible, one error per
        missing capability — kept granular so callers can surface a
        helpful "need X, Y; host has Z" message.
    """
    required = runtime.requires_capability
    if not required:
        return []

    present = _host_capability_set(host_hardware)
    missing = sorted(set(required) - present)
    if not missing:
        return []

    return [
        "host %r missing required capabilities %s for runtime %r (host advertises: %s)"
        % (host, missing, runtime.runtime_name, sorted(present) or "[]")
    ]


def check_runtime_cluster_compatibility(
    runtime: RuntimePlugin,
    cluster: ClusterDefinition,
    placement: RankAssignment | None = None,
) -> list[str]:
    """Aggregate compatibility errors across every host that receives ranks.

    Walks the placement when provided so only *placed* hosts are
    checked (a heterogeneous cluster with an explicit layout may
    deliberately exclude some hosts).  Falls back to checking every
    cluster host when no placement is supplied.
    """
    hosts = placement.hosts_used if placement is not None else tuple(cluster.hosts)
    errors: list[str] = []
    for host in hosts:
        errors.extend(check_runtime_host_compatibility(runtime, host, cluster.hardware_for(host)))
    return errors


def assert_runtime_cluster_compatibility(
    runtime: RuntimePlugin,
    cluster: ClusterDefinition,
    placement: RankAssignment | None = None,
) -> None:
    """Like :func:`check_runtime_cluster_compatibility` but raises on failure."""
    errors = check_runtime_cluster_compatibility(runtime, cluster, placement)
    if errors:
        raise IncompatibleHardwareError(runtime.runtime_name, errors)
