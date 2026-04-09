"""Cluster inter-node communication environment value object.

Bundles the NCCL/gloo/UCX/OMPI env vars used to configure transports
inside containers, with support for per-host overrides so heterogeneous
management interfaces (e.g. wired on the head, wifi on a worker) each
receive the right ``*_SOCKET_IFNAME`` / ``MN_IF_NAME`` / etc. values.

Carrying this around as a single object keeps the plumbing surface
small: runtimes pass ``comm_env`` through to ``_run_cluster`` and each
per-host launch site calls ``comm_env.get_env(host)`` to obtain the
merged env for that host.  Future transport vars are added here, not
in every function signature between the detector and the launcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ClusterCommEnv:
    """Inter-node communication environment for a cluster launch.

    Args:
        shared: Cluster-wide env applied to every host (e.g.
            ``NCCL_NET``, ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``,
            ``UCX_NET_DEVICES``).
        per_host: Per-host overrides keyed by hostname; applied on top
            of *shared*.  Typically holds the socket-interface names
            that differ between hosts (``GLOO_SOCKET_IFNAME``,
            ``TP_SOCKET_IFNAME``, ``MN_IF_NAME``, etc.).
    """

    shared: Mapping[str, str] = field(default_factory=dict)
    per_host: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def get_env(self, host: str) -> dict[str, str]:
        """Return the merged env for *host* (shared + per-host override).

        Returns a fresh ``dict`` safe for the caller to mutate.
        """
        merged: dict[str, str] = dict(self.shared)
        override = self.per_host.get(host)
        if override:
            merged.update(override)
        return merged

    def all_keys(self) -> set[str]:
        """Return the union of all env-var names across shared + per-host."""
        keys: set[str] = set(self.shared.keys())
        for override in self.per_host.values():
            keys.update(override.keys())
        return keys

    def hosts(self) -> list[str]:
        """Hosts with per-host overrides (not necessarily all cluster hosts)."""
        return list(self.per_host.keys())

    def is_empty(self) -> bool:
        return not self.shared and not self.per_host

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __len__(self) -> int:
        """Count of distinct env var names (for "(%d vars)" style logs)."""
        return len(self.all_keys())

    @classmethod
    def empty(cls) -> ClusterCommEnv:
        return cls()

    @classmethod
    def from_shared(cls, env: Mapping[str, str] | None) -> ClusterCommEnv:
        """Build from a single shared env dict (no per-host overrides).

        Convenient for solo-mode paths and for lifting a legacy
        ``dict[str, str]`` into the new type.
        """
        if not env:
            return cls.empty()
        return cls(shared=dict(env))

    @classmethod
    def from_per_host(
        cls,
        per_host: Mapping[str, Mapping[str, str]],
    ) -> ClusterCommEnv:
        """Build from a per-host env mapping, factoring out shared keys.

        Keys whose values are identical across *all* hosts are moved
        into ``shared``; keys that differ or are missing from any host
        remain in ``per_host``.  Hosts that end up with no unique keys
        are dropped from the per-host mapping.
        """
        if not per_host:
            return cls.empty()

        hosts = list(per_host.keys())
        if len(hosts) == 1:
            # Single host: everything is "shared" from its perspective.
            return cls(shared=dict(per_host[hosts[0]]))

        # Union of all keys seen anywhere
        all_keys: set[str] = set()
        for env in per_host.values():
            all_keys.update(env.keys())

        shared: dict[str, str] = {}
        for key in all_keys:
            # Require key present in *every* host with identical value
            first = per_host[hosts[0]].get(key)
            if first is None:
                continue
            if all(per_host[h].get(key) == first for h in hosts[1:]):
                shared[key] = first

        overrides: dict[str, dict[str, str]] = {}
        for host, env in per_host.items():
            diff = {k: v for k, v in env.items() if shared.get(k) != v}
            if diff:
                overrides[host] = diff

        return cls(shared=shared, per_host=overrides)
