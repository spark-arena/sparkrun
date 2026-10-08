"""Console-free setup readiness and target context models."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.hardware import HostHardware
    from sparkrun.orchestration.networking import CX7HostDetection
    from sparkrun.orchestration.rdma import RdmaHostFacts

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"
GPU_ACCESS_CDI = "cdi"
DOCKER_IPC_HOST = "host"


@dataclass
class CheckItem:
    """The outcome of one setup check for one host."""

    key: str
    label: str
    status: str
    detail: str = ""
    guidance: str = ""


@dataclass
class CheckContext:
    """Cluster-wide facts a check may need beyond a host's own probe output."""

    cluster_name: str | None
    multi_host: bool

    gpu_access_modes: dict[str, str] = field(default_factory=dict)
    """host -> the ``gpu_access_mode`` a launch on that host would resolve to.

    Empty (or missing a host) when resolution failed; see :meth:`cdi_required`.
    """

    config: "SparkrunConfig | None" = None
    executor_names: dict[str, str] = field(default_factory=dict)
    strict: bool = False
    """Require measured hardware/executor facts before offering host changes."""

    ipc_exposure: dict[str, str] = field(default_factory=dict)
    """host -> the IPC namespace a launch on that host would put the workload in.

    ``"host"`` means the workload's POSIX semaphores and shared memory land in
    the *host's* ``/dev/shm`` — either a Docker ``ipc: host`` override or the
    ``local`` executor, which runs natively and has no namespace at all.  Any
    other value (or a missing host) means it does not; see
    :meth:`shares_host_ipc`.
    """

    extra_mesh_peers: tuple[str, ...] = ()
    """SSH-mesh members that are not setup targets (a control machine joining the mesh).

    Probed as peers alongside the targets, so ``ssh_mesh`` reports the mesh a
    frontend actually builds rather than only its host-to-host part.
    """

    @property
    def cluster_flag(self) -> str:
        """`` --cluster <name>`` suffix for guidance commands (empty if unknown)."""
        return " --cluster %s" % self.cluster_name if self.cluster_name else ""

    def cdi_required(self, host: str) -> bool:
        """Would a launch on *host* actually need a CDI spec?

        Only when the resolved ``gpu_access_mode`` is ``cdi`` — a cluster that
        requests GPUs with ``--gpus`` never reads ``/etc/cdi/nvidia.yaml``, so a
        missing or stale spec is not a gap for it.

        Fails **safe**: an unresolvable mode is treated as requiring CDI, which
        is the historical behavior (a missing spec is a hard failure). Unrecognized
        modes also require CDI, matching Docker's fallback.
        """
        # Match Docker's case/whitespace normalization and unknown-mode fallback.
        return (self.gpu_access_modes.get(host) or GPU_ACCESS_CDI).strip().lower() != "gpus"

    def shares_host_ipc(self, host: str) -> bool:
        """Would a launch on *host* put the workload on the host IPC namespace?

        Fails **quiet**, the opposite of :meth:`cdi_required`: an unresolvable
        executor reads as "no". The two differ because a false positive costs
        different things — a spurious CDI finding points at a real file, while
        a spurious IPC finding tells the operator to change logind on a host
        where nothing is at risk. A finding that fires on correct setups is how
        a report teaches people to skim it.
        """
        return self.ipc_exposure.get(host, "") == DOCKER_IPC_HOST


@dataclass
class HostState:
    """Everything a check needs about one host.

    ``facts`` is the parsed key=value output of ``setup_check.sh``. Richer
    signals that reuse existing sparkrun machinery hang off dedicated fields
    (e.g. ``cx7`` from :func:`detect_cx7_for_hosts`) rather than being
    re-derived from a shallow probe — this is what lets a check vet the
    *effective* config, not just a file's presence.
    """

    host: str
    facts: dict[str, str] = field(default_factory=dict)
    cx7: "CX7HostDetection | None" = None
    hardware: "HostHardware | None" = None
    reachable: bool = True
    error: str = ""
    rdma: "RdmaHostFacts | None" = None
    """Devices and tooling from ``rdma_probe.sh``, for :func:`_check_rdma`.

    A dedicated field for the same reason ``cx7`` is one: widening
    ``setup_check.sh`` to carry RDMA facts would put a second, shallower
    implementation of the probe beside the real one.
    """

    @property
    def has_nvidia(self) -> bool:
        if self.hardware is not None:
            return "nvidia" in self.hardware.vendors
        return self.facts.get("CHECK_GPU_PRESENT") == "1"
