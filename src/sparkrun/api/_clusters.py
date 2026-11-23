"""Cluster mutations through the shared manager's locked write boundary."""

from sparkrun.api._context import resolve_sctx


def create_cluster(name: str, hosts: list[str], *, description: str = "", user: str | None = None, sctx=None):
    """Save a new named cluster definition and return it.

    Args:
        name: Cluster name (letters, digits, ``-`` and ``_``; starts alphanumeric).
        hosts: Host names or addresses, head first.
        description: Free-form note shown by ``cluster list``.
        user: SSH user for these hosts; omit to use the configured default.

    Raises:
        ClusterError: The name is invalid or already taken.
    """
    manager = resolve_sctx(sctx).cluster_manager
    manager.create(name, hosts, description=description, user=user)
    return manager.get(name)


def update_cluster(name: str, *, hosts: list[str] | None = None, description: str | None = None, user: str | None = None, sctx=None):
    """Change a saved cluster's hosts, description or SSH user and return it.

    Args:
        name: The cluster to change.
        hosts: Replacement host list, head first; omit to keep the current one.
        description: Replacement note; omit to keep it.
        user: Replacement SSH user; omit to keep it.

    Raises:
        ClusterError: No cluster has that name.
    """
    manager = resolve_sctx(sctx).cluster_manager
    manager.update(name, hosts=hosts, description=description, user=user)
    return manager.get(name)


def delete_cluster(name: str, *, sctx=None):
    """Delete a saved cluster definition (clearing the default if it pointed there).

    Args:
        name: The cluster to delete.

    Raises:
        ClusterError: No cluster has that name.
    """
    resolve_sctx(sctx).cluster_manager.delete(name)


def set_default_cluster(name: str | None = None, *, sctx=None):
    """Set the default cluster, or clear it when *name* is ``None``.

    ``SPARKRUN_CLUSTER`` still overrides the default in a shell that sets it.

    Args:
        name: The cluster to make the default, or ``None`` to unset it.

    Raises:
        ClusterError: No cluster has that name.
    """
    manager = resolve_sctx(sctx).cluster_manager
    manager.set_default(name) if name is not None else manager.unset_default()
