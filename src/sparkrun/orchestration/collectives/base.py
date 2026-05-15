"""Abstract base for collective communication backends (NCCL, RCCL, HCCL)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class UnsupportedCollectiveError(Exception):
    """Raised when no collective backend matches the requested vendor."""


class CollectiveBackend(ABC):
    """Abstract collective backend (NCCL / RCCL / HCCL).

    A backend converts the result of a transport detection (today:
    :func:`sparkrun.orchestration.infiniband.parse_ib_detect_output`)
    into a set of environment variables to inject into the runtime
    container.  The legacy NCCL implementation lives in
    :mod:`sparkrun.orchestration.infiniband` and is wrapped by
    :class:`~.nccl.NcclBackend` to preserve byte-for-byte output.

    Subclasses must define :attr:`name` and implement
    :meth:`env_for_host` and :meth:`ring_overrides`.
    """

    name: str = ""
    """Canonical backend name: ``"nccl"``, ``"rccl"``, ``"hccl"``."""

    vendor: str = ""
    """Accelerator vendor this backend serves: ``"nvidia"`` / ``"amd"`` / ``"intel"``."""

    @abstractmethod
    def env_for_host(
        self,
        ib_info: dict[str, str],
        *,
        topology: str | None = None,
    ) -> dict[str, str]:
        """Build the per-host env block from a transport detection result.

        Args:
            ib_info: Parsed transport info (today: Mellanox InfiniBand
                / RoCEv2 from
                :func:`sparkrun.orchestration.infiniband.parse_ib_detect_output`).
                AMD / Intel implementations will accept the same shape
                and ignore NCCL-specific keys, or extend the contract
                as needed.
            topology: Optional cluster topology hint (e.g. ``"ring"``).

        Returns:
            Environment variables to inject into the host's runtime
            container.  Empty dict when no transport was detected.
        """
        ...

    @abstractmethod
    def ring_overrides(self, ib_info: dict[str, str]) -> dict[str, str]:
        """Backend-specific overrides for 3-node ring/mesh topologies."""
        ...
