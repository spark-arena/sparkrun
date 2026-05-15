"""Intel Gaudi HCCL collective backend scaffold.

Habana Collective Communications Library — the Intel Gaudi equivalent
of NCCL.  This scaffold raises :class:`NotImplementedError` at the
point of use so any code path that resolves to an Intel host without
a real HCCL implementation fails loudly instead of silently emitting
NCCL env vars.

Filling in this implementation requires translating
:func:`infiniband.parse_ib_detect_output` into HCCL env vars
(``HCCL_OVER_OFI=1``, ``HCCL_SOCKET_IFNAME``, etc.) and potentially
Gaudi-side env (``HABANA_VISIBLE_DEVICES``).
"""

from __future__ import annotations

from sparkrun.orchestration.collectives.base import CollectiveBackend


class HcclBackend(CollectiveBackend):
    """Intel Gaudi HCCL backend (scaffold; not yet implemented)."""

    name = "hccl"
    vendor = "intel"

    def env_for_host(
        self,
        ib_info: dict[str, str],
        *,
        topology: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError(
            "HCCL backend is not yet implemented. "
            "Intel Gaudi hosts require HCCL-equivalent env vars (HCCL_OVER_OFI, HCCL_SOCKET_IFNAME) — "
            "contribute an implementation in sparkrun/orchestration/collectives/hccl.py."
        )

    def ring_overrides(self, ib_info: dict[str, str]) -> dict[str, str]:
        raise NotImplementedError("HCCL ring-topology overrides are not yet implemented.")
