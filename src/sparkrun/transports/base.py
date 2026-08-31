"""Transport seam: how sparkrun reaches / prepares a cluster's hosts.

A :class:`Transport` owns the *connectivity and host-session* concern: how a
cluster is prepared and how exact-argv operations are subsequently executed on
its hosts. For ordinary clusters preparation is empty and the executable
session uses Sparkrun's shared SSH configuration.

Provider-backed clusters (e.g. Thunder Compute) override :meth:`prepare` to
materialize connection details — refresh ephemeral IP/port, provision SSH keys,
and write a managed ``ssh_config`` alias — so that once ``prepare`` returns,
every host in the cluster is a plain SSH host. A transport with another
substrate may also override :meth:`open_host_session`.

Transport (how you *reach* the host) is orthogonal to Executor
(``orchestration.executors`` — how you *run the workload* on the host).  A
Thunder-transport cluster still uses the default docker executor.

Discovery: transports are SAF :class:`~scitrera_app_framework.Plugin` classes
registered at the :data:`EXT_TRANSPORT` extension point and discovered via
``find_types_in_modules("sparkrun.transports", Transport)`` in
:func:`sparkrun.core.bootstrap.init_sparkrun` — the same mechanism used for
executors.  Selection is by :attr:`transport_name` (see
:func:`sparkrun.transports.resolve_transport`), so unlike the order-sensitive
platform registry there is no ordering to preserve.

Layering: this package depends on ``core`` + ``orchestration``; ``orchestration``
never imports ``transports`` (it stays the generic SSH/Docker leaf).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from scitrera_app_framework import Plugin, Variables

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.transports.session import HostSession

logger = logging.getLogger(__name__)

EXT_TRANSPORT = "sparkrun.transport"


class TransportError(Exception):
    """Raised when a transport cannot prepare a cluster for connectivity."""


class Transport(Plugin):
    """Base class for cluster transports.

    The default implementation is a no-op — subclasses override :meth:`prepare`
    only when the cluster's hosts need out-of-band setup before SSH.

    Each concrete transport is a SAF plugin selected by :attr:`transport_name`
    (matched against :attr:`ClusterDefinition.transport`); the registered SAF
    singleton is returned directly by
    :func:`sparkrun.transports.resolve_transport` since transports are
    stateless.
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    transport_name: ClassVar[str] = ""
    """Selector matching :attr:`ClusterDefinition.transport`."""

    # --- Optional feature gating ---
    # When set to a registered feature-flag name (e.g. ``"transports.thunder"``),
    # a cluster declaring this transport fails closed at the ``prepare`` call
    # site (:func:`sparkrun.transports.prepare_cluster_transport`) unless the
    # flag resolves on — never a silent SSH downgrade. ``None`` = always usable.
    required_feature_flag: ClassVar[str | None] = None

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.transport.%s" % self.transport_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TRANSPORT

    def is_enabled(self, v: Variables) -> bool:
        # Must return False for multi-extension plugins to prevent SAF's
        # single-extension cache from short-circuiting subsequent plugin
        # initializations under the same extension point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        # Transports are always exposed; feature-gated providers (e.g.
        # thunder) fail closed at the ``prepare`` call site
        # (``_require_transport_enabled``) instead, which yields a clear
        # "enable it with ..." message rather than an "unknown transport".
        return True

    def initialize(self, v: Variables, logger=None) -> "Transport":
        return self

    def prepare(self, cluster: "ClusterDefinition", *, dry_run: bool = False) -> None:
        """Ensure every host in *cluster* is reachable via plain ``ssh``.

        Called at run/connect init before any SSH fan-out.  The default is a
        no-op.  Implementations must be idempotent and safe to call repeatedly.
        When *dry_run* is True they must not mutate local state or make
        privileged/expensive remote calls beyond read-only lookups.
        """
        return None

    def open_host_session(self, cluster: "ClusterDefinition | None", *, ssh_kwargs: dict | None = None) -> "HostSession":
        """Open an executable session through this prepared transport.

        Provider transports may override this when their connection substrate
        is not ordinary SSH. The default intentionally uses Sparkrun's shared
        SSH configuration after :meth:`prepare` has made host aliases and keys
        available.
        """
        from sparkrun.transports.session import SshHostSession

        return SshHostSession(**(ssh_kwargs or {}))

    def cleanup_cluster(self, cluster: "ClusterDefinition", *, dry_run: bool = False) -> None:
        """Tear down transport-owned state for *cluster* on deletion.

        The counterpart to :meth:`prepare`: called from the generic cluster
        delete path so a provider transport can release out-of-band resources
        it materialized (e.g. remove the managed ssh alias / key).  Named
        ``cleanup_cluster`` rather than ``cleanup`` to make clear it targets the
        cluster, not a transport session.  The default is a no-op; must be
        idempotent and safe on a partially-prepared or already-gone cluster.
        """
        return None
