"""Collective communication backend abstraction.

:class:`CollectiveBackend` is the seam that lets AMD (RCCL) and Intel
Gaudi (HCCL) plug in alongside NVIDIA NCCL without ad-hoc branching in
every runtime.  The default :class:`~.nccl.NcclBackend` delegates
straight to :mod:`sparkrun.orchestration.infiniband` so NVIDIA-host
output is unchanged.  RCCL and HCCL ship as scaffolds that raise
:class:`NotImplementedError` with actionable messages — contributors
can fill them in without re-plumbing runtimes.
"""

from __future__ import annotations

from sparkrun.orchestration.collectives.base import (
    CollectiveBackend,
    UnsupportedCollectiveError,
)
from sparkrun.orchestration.collectives.hccl import HcclBackend
from sparkrun.orchestration.collectives.nccl import NcclBackend
from sparkrun.orchestration.collectives.rccl import RcclBackend

_BY_VENDOR: dict[str, type[CollectiveBackend]] = {
    "nvidia": NcclBackend,
    "amd": RcclBackend,
    "intel": HcclBackend,
}


def get_backend(vendor: str | None) -> CollectiveBackend:
    """Return the collective backend for a given accelerator vendor.

    ``vendor=None`` and ``vendor="nvidia"`` both map to NCCL — the
    legacy default — so existing DGX call sites behave identically.

    AMD / Intel return scaffolds that raise
    :class:`NotImplementedError` from :meth:`env_for_host`; this
    surfaces the missing implementation at the point of use rather
    than silently producing wrong NCCL env vars.

    Raises:
        UnsupportedCollectiveError: When *vendor* is not recognised
            (e.g. ``"apple"``, ``"cpu"``, fictional FPGAs).
    """
    key = (vendor or "nvidia").lower()
    cls = _BY_VENDOR.get(key)
    if cls is None:
        raise UnsupportedCollectiveError("No collective backend for vendor %r (known: %s)" % (vendor, sorted(_BY_VENDOR)))
    return cls()


__all__ = [
    "CollectiveBackend",
    "HcclBackend",
    "NcclBackend",
    "RcclBackend",
    "UnsupportedCollectiveError",
    "get_backend",
]
