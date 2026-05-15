"""AMD RCCL collective backend scaffold.

ROCm Communication Collectives Library — the AMD equivalent of NCCL.
This scaffold raises :class:`NotImplementedError` at the point of use
so any code path that resolves to an AMD host without a real RCCL
implementation fails loudly instead of silently emitting NCCL env vars.

Filling in this implementation requires:

- Mapping :func:`infiniband.parse_ib_detect_output` (Mellanox HCAs are
  shared across vendors) onto RCCL-specific env vars such as
  ``RCCL_NET_SHARED_BUFFERS``, ``RCCL_IB_HCA``, etc.
- Potentially adding HIP-side env (``HSA_OVERRIDE_GFX_VERSION``,
  ``HIP_VISIBLE_DEVICES``) when those should travel with the
  collective env block rather than the runtime container env.
"""

from __future__ import annotations

from sparkrun.orchestration.collectives.base import CollectiveBackend


class RcclBackend(CollectiveBackend):
    """AMD RCCL backend (scaffold; not yet implemented)."""

    name = "rccl"
    vendor = "amd"

    def env_for_host(
        self,
        ib_info: dict[str, str],
        *,
        topology: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError(
            "RCCL backend is not yet implemented. "
            "AMD ROCm hosts require RCCL-equivalent env vars (RCCL_NET_*, RCCL_IB_*) — "
            "contribute an implementation in sparkrun/orchestration/collectives/rccl.py."
        )

    def ring_overrides(self, ib_info: dict[str, str]) -> dict[str, str]:
        raise NotImplementedError("RCCL ring-topology overrides are not yet implemented.")
