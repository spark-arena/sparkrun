"""NVIDIA NCCL collective backend.

Thin wrapper over :func:`sparkrun.orchestration.infiniband.generate_nccl_env`
and :func:`generate_ring_nccl_overrides`.  Preserves the existing
behavior byte-for-byte so DGX Spark output is unchanged.
"""

from __future__ import annotations

from sparkrun.orchestration.collectives.base import CollectiveBackend


class NcclBackend(CollectiveBackend):
    """NVIDIA NCCL backend (default for ``vendor=="nvidia"`` and ``vendor is None``)."""

    name = "nccl"
    vendor = "nvidia"

    def env_for_host(
        self,
        ib_info: dict[str, str],
        *,
        topology: str | None = None,
    ) -> dict[str, str]:
        from sparkrun.orchestration.infiniband import generate_nccl_env

        return generate_nccl_env(ib_info, topology=topology)

    def ring_overrides(self, ib_info: dict[str, str]) -> dict[str, str]:
        from sparkrun.orchestration.infiniband import generate_ring_nccl_overrides

        return generate_ring_nccl_overrides(ib_info)
