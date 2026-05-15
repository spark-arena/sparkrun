"""Recipe layout primitives for explicit placement on heterogeneous clusters.

Phase 1 of the hardware-agnostic refactor: defines the data model only.
Nothing consumes :class:`RecipeLayout` yet beyond parse / round-trip;
the placement engine in :mod:`sparkrun.core.placement` (Phase 2) will
honor it.

A recipe may declare a ``layout`` block to:

- Require that selected hosts advertise specific capabilities
  (e.g. ``cuda``, ``rocm``).
- Assign global ranks to specific hosts and local accelerator indices
  (essential on heterogeneous clusters where auto-fit is intentionally
  not attempted).

YAML form::

    layout:
      requires:
        - capability: cuda
      placements:
        - host: spark-01
          ranks: [0]
        - host: spark-02
          ranks: [1]
        - host: rtx-box
          ranks: [2, 3]
          local_gpus: [0, 1]    # optional; defaults to range(len(ranks))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityRequirement:
    """A single capability requirement applied to placed hosts."""

    capability: str
    """Capability tag every placed host must advertise (matches :attr:`AcceleratorSpec.capabilities`)."""

    def to_dict(self) -> dict[str, Any]:
        return {"capability": self.capability}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityRequirement:
        return cls(capability=str(data.get("capability", "")))


@dataclass(frozen=True)
class Placement:
    """One host's slice of the global rank space."""

    host: str
    """Host name/address as it appears in :attr:`ClusterDefinition.hosts`."""

    ranks: tuple[int, ...] = ()
    """Global ranks assigned to this host (e.g. ``(0,)`` or ``(2, 3)``)."""

    local_gpus: tuple[int, ...] = ()
    """Local accelerator indices on this host (one per rank).

    Empty means "use indices ``0..len(ranks)-1``"; the placement engine
    materializes the default in Phase 2.
    """

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"host": self.host, "ranks": list(self.ranks)}
        if self.local_gpus:
            d["local_gpus"] = list(self.local_gpus)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Placement:
        ranks_raw = data.get("ranks") or ()
        local_raw = data.get("local_gpus") or ()
        return cls(
            host=str(data.get("host", "")),
            ranks=tuple(int(r) for r in ranks_raw),
            local_gpus=tuple(int(g) for g in local_raw),
        )


@dataclass
class RecipeLayout:
    """Explicit layout declaration for a recipe on a heterogeneous cluster."""

    requires: list[CapabilityRequirement] = field(default_factory=list)
    """Capabilities every placed host must advertise."""

    placements: list[Placement] = field(default_factory=list)
    """Explicit per-host rank assignments.  Empty means auto-place (homogeneous clusters only)."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.requires:
            d["requires"] = [r.to_dict() for r in self.requires]
        if self.placements:
            d["placements"] = [p.to_dict() for p in self.placements]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeLayout:
        raw_requires = data.get("requires") or []
        raw_placements = data.get("placements") or []
        return cls(
            requires=[CapabilityRequirement.from_dict(r) for r in raw_requires if isinstance(r, dict)],
            placements=[Placement.from_dict(p) for p in raw_placements if isinstance(p, dict)],
        )
