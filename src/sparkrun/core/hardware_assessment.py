"""Pre-launch checks over the hardware a launch will run on.

Everything here is a pure function of facts planning already holds — the
probed observations, the cluster they were merged into, and the placement —
so it runs in :func:`sparkrun.api.plan`, where its findings can sit beside the
fit table they explain and an incompatible host is refused before any launch
phase starts. ``launch_inference`` falls back to computing it for callers that
launch without a plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import HostHardware
    from sparkrun.core.scheduler import RankAssignment


@dataclass(frozen=True)
class HardwareAssessment:
    """What the target hardware says about this launch, decided once."""

    hosts: dict[str, dict] = field(default_factory=dict)
    """Host → :func:`~sparkrun.core.hardware_observations.hardware_evidence`
    for every launch host, in launch order (each fact carries its provenance).
    Empty when there is no cluster to read facts from."""
    detected: tuple[str, ...] = ()
    """Hosts probed this operation."""
    assumed: tuple[str, ...] = ()
    """Hosts whose hardware is the application policy's assumption, not a fact.
    Kept as data rather than warning text so a renderer can mark the hosts
    instead of repeating a sentence per host."""
    warnings: tuple[str, ...] = ()
    """Other concerns that do not block the launch, each naming its host(s)."""
    errors: tuple[str, ...] = ()
    """Runtime/host incompatibilities; non-empty means the launch must not start."""

    @property
    def evidence(self) -> dict[str, str]:
        """Host → one-line description of its *detected* hardware."""
        from sparkrun.core.hardware_observations import format_hardware_evidence

        return {host: format_hardware_evidence(self.hosts[host]) for host in self.detected if host in self.hosts}

    def assumed_warnings(self) -> list[str]:
        """The per-host sentence for :attr:`assumed`, for log-only consumers."""
        return ["Host %s hardware is assumed by application policy; probe it to verify identity and capacity" % h for h in self.assumed]


def assess_launch_hardware(
    runtime,
    host_list: Sequence[str],
    cluster: ClusterDefinition | None,
    placement: RankAssignment | None,
    observations: Mapping[str, HostHardware] | None = None,
) -> HardwareAssessment:
    """Assess the hardware *host_list* resolves to under *cluster* and *placement*.

    *observations* are this operation's probe results; only hosts detected
    there get an evidence line or take part in the driver comparison, because
    an inventory record is history, not a fact about the host right now.
    """
    from sparkrun.core.hardware import resolve_host_hardware
    from sparkrun.core.hardware_observations import hardware_evidence
    from sparkrun.platforms import resolve_accelerator_platform
    from sparkrun.runtimes.compatibility import check_runtime_host_compatibility

    observations = observations or {}
    detected = {h for h in host_list if h in observations and observations[h].source == "detected"}
    launch_hardware = resolve_host_hardware(list(host_list), cluster, placement)

    facts: dict[str, dict] = {}
    assumed: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    for host, hw in launch_hardware.items():
        if cluster is not None:
            facts[host] = hardware_evidence(cluster, host, placement)
        if getattr(runtime, "requires_capability", ()):
            errors.extend(check_runtime_host_compatibility(runtime, host, hw))
        host_warnings: list[str] = []
        for accel in hw.accelerators:
            platform = resolve_accelerator_platform(accel, hw)
            if platform is not None:
                host_warnings.extend(w for w in platform.validate_host(hw) if w not in host_warnings)
        warnings.extend("Host %s: %s" % (host, w) for w in host_warnings)
        if hw.source == "assumed":
            assumed.append(host)

    warnings.extend(driver_mismatch_warnings({h: observations[h] for h in host_list if h in detected}))
    return HardwareAssessment(
        hosts=facts,
        detected=tuple(h for h in host_list if h in detected),
        assumed=tuple(assumed),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def driver_mismatch_warnings(hardware: Mapping[str, HostHardware]) -> list[str]:
    """Warn when hosts in one launch run different accelerator driver versions.

    Ranks of a multi-node job share NCCL and the container's CUDA userspace,
    both of which sit on the host driver, so a launch spanning driver
    branches can fail in ways that point nowhere near the cause. A warning,
    not an error: mixed drivers often work.
    """
    by_vendor: dict[str, dict[str, list[str]]] = {}
    for host, hw in hardware.items():
        for vendor, version in hw.driver_versions.items():
            if version:
                by_vendor.setdefault(vendor, {}).setdefault(version, []).append(host)
    warnings = []
    for vendor, versions in sorted(by_vendor.items()):
        if len(versions) < 2:
            continue
        branches = {v.split(".", 1)[0] for v in versions}
        warnings.append(
            "%s driver %s differ across hosts: %s; multi-node collectives expect matching drivers"
            % (
                vendor,
                "branches" if len(branches) > 1 else "versions",
                ", ".join("%s (%s)" % (v, ", ".join(hosts)) for v, hosts in sorted(versions.items())),
            )
        )
    return warnings
