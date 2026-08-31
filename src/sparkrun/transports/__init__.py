"""Cluster transports — the connectivity seam.

A transport owns how sparkrun *reaches / prepares* a cluster's hosts before the
generic SSH machinery runs.  ``ssh`` (the default) is a no-op; provider-backed
transports (``thunder``) refresh ephemeral connection details out-of-band.

Transports are SAF plugins discovered by
``find_types_in_modules("sparkrun.transports", Transport)`` in
:func:`sparkrun.core.bootstrap.init_sparkrun` and selected by
:attr:`Transport.transport_name` — the same mechanism as
:mod:`sparkrun.orchestration.executors`.  (Selection is by exact name, so unlike
the order-sensitive :mod:`sparkrun.platforms` registry there is no ordering to
preserve — which is why transports, but not platforms, moved onto SAF.)

Layering: ``cli/api → transports → {core, orchestration}``; ``orchestration``
never imports this package.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scitrera_app_framework import Variables, get_extensions

from sparkrun.transports.base import EXT_TRANSPORT, Transport, TransportError
from sparkrun.transports.ssh import SshTransport

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.transports.session import HostSession

logger = logging.getLogger(__name__)

DEFAULT_TRANSPORT = "ssh"


def _require_transport_enabled(name: str, transport: Transport) -> None:
    """Raise :class:`TransportError` if *transport* is feature-gated and disabled.

    The gate is declared by the transport itself via
    :attr:`Transport.required_feature_flag` (so external transports self-gate
    without any core registry). Resolves the flag context-free (env →
    config.yaml → channel default) via
    :func:`sparkrun.core.features.feature_gate_enabled`, so it works from the
    run/status/logs/stop hook where no ``SparkrunContext`` is threaded. Enforced
    here at the ``prepare`` call site — not via SAF ``is_multi_extension``
    hiding — so a gated selector yields a clear "enable it with ..." error
    rather than an "unknown transport".
    """
    flag = getattr(transport, "required_feature_flag", None)
    if not flag:
        return
    from sparkrun.core.features import feature_gate_enabled

    if not feature_gate_enabled(flag):
        raise TransportError(
            "The %r transport is experimental and disabled. Enable it with: sparkrun setup features enable %s" % (name, flag)
        )


def _transport_extensions(v: Variables | None = None) -> dict:
    """Return SAF transport plugins, initializing the framework if needed."""
    if v is None:
        from sparkrun.core.bootstrap import get_variables

        v = get_variables()
    return get_extensions(EXT_TRANSPORT, v=v)


def list_transports(v: Variables | None = None) -> list[str]:
    """Return the registered transport selector names."""
    return sorted(t.transport_name for t in _transport_extensions(v).values() if getattr(t, "transport_name", ""))


def resolve_transport(name: str | None, v: Variables | None = None) -> Transport:
    """Return a :class:`Transport` instance for selector *name*.

    ``None`` / empty resolves to the default ``ssh`` transport.  An unknown
    selector raises :class:`TransportError` (never a silent fallback — a cluster
    that declares ``transport: foo`` must not silently run over plain SSH).

    Transports are stateless, so the registered SAF singleton is returned
    directly (mirroring the platform registry; unlike executors, which carry
    per-launch config and are instantiated fresh).
    """
    key = name or DEFAULT_TRANSPORT
    for transport in _transport_extensions(v).values():
        if getattr(transport, "transport_name", "") == key:
            return transport
    raise TransportError("Unknown transport %r (known: %s)" % (key, ", ".join(list_transports(v))))


def prepare_cluster_transport(cluster: "ClusterDefinition | None", *, dry_run: bool = False) -> None:
    """Run the transport ``prepare`` step for *cluster* before any SSH.

    The single call site helper: resolve the cluster's transport and invoke
    :meth:`Transport.prepare`.  Short-circuits the default ``ssh`` transport
    without constructing anything so existing clusters pay zero cost.  A
    ``None`` cluster (or one with no ``transport`` attribute) is treated as
    plain SSH.
    """
    if cluster is None:
        return
    name = getattr(cluster, "transport", None) or DEFAULT_TRANSPORT
    if name == DEFAULT_TRANSPORT:
        return
    transport = resolve_transport(name)
    _require_transport_enabled(name, transport)
    transport.prepare(cluster, dry_run=dry_run)


def open_cluster_host_session(cluster: "ClusterDefinition | None", *, ssh_kwargs: dict | None = None) -> "HostSession":
    """Open the executable session owned by a cluster's prepared transport."""
    name = (getattr(cluster, "transport", None) or DEFAULT_TRANSPORT) if cluster is not None else DEFAULT_TRANSPORT
    transport = resolve_transport(name)
    _require_transport_enabled(name, transport)
    return transport.open_host_session(cluster, ssh_kwargs=ssh_kwargs)


def cleanup_cluster_transport(cluster: "ClusterDefinition | None", *, dry_run: bool = False) -> None:
    """Run the transport ``cleanup_cluster`` step for *cluster* on deletion.

    Mirrors :func:`prepare_cluster_transport`.  Deliberately **not** feature-
    gated — teardown of provider-owned state (ssh aliases, keys) must succeed
    even if the transport was later disabled.  Defensive: an unknown/absent
    transport (e.g. its plugin isn't loaded) is a no-op rather than a delete
    failure.
    """
    if cluster is None:
        return
    name = getattr(cluster, "transport", None) or DEFAULT_TRANSPORT
    if name == DEFAULT_TRANSPORT:
        return
    try:
        transport = resolve_transport(name)
    except TransportError:
        logger.debug("No transport %r available to clean up cluster %r; skipping", name, getattr(cluster, "name", "?"))
        return
    transport.cleanup_cluster(cluster, dry_run=dry_run)


__all__ = [
    "DEFAULT_TRANSPORT",
    "EXT_TRANSPORT",
    "SshTransport",
    "Transport",
    "TransportError",
    "cleanup_cluster_transport",
    "list_transports",
    "open_cluster_host_session",
    "prepare_cluster_transport",
    "resolve_transport",
]
