"""Named cluster management for SparkRun."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Name validation pattern: start with alphanumeric, contain alphanumeric/underscore/hyphen
CLUSTER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class ClusterError(Exception):
    """Raised when cluster operations fail."""

    pass


# Sentinel for "not provided" to distinguish from explicit None
_UNSET = object()


@dataclass
class ClusterDefinition:
    """Definition of a named cluster."""

    name: str
    hosts: list[str]
    description: str = ""
    user: str | None = None
    cache_dir: str | None = None


@dataclass
class ClusterGroup:
    """A group of containers forming a sparkrun cluster."""

    cluster_id: str
    members: list[tuple[str, str, str, str]]  # (host, role, status, image)
    meta: dict[str, Any]  # job metadata from cache


@dataclass
class ClusterStatusResult:
    """Result of querying sparkrun container status across hosts."""

    groups: dict[str, ClusterGroup]  # cluster_id -> group
    solo_entries: list[tuple[str, str, str, str]]  # (host, name, status, image)
    errors: dict[str, str]  # host -> error message
    idle_hosts: list[str]  # hosts with no containers and no errors
    pending_ops: list[dict[str, Any]]  # relevant pending operations
    total_containers: int
    host_count: int


class ClusterManager:
    """Manages named cluster definitions stored as YAML files."""

    def __init__(self, config_root: Path) -> None:
        """Initialize cluster manager.

        Args:
            config_root: Base configuration directory (e.g. ~/.config/sparkrun/).
                Cluster files will be stored in config_root/clusters/.
        """
        self.config_root = Path(config_root)
        self.clusters_dir = self.config_root / "clusters"
        self.default_file = self.clusters_dir / ".default"

        # Ensure clusters directory exists
        self.clusters_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ClusterManager initialized with clusters_dir: %s", self.clusters_dir)

    def _validate_name(self, name: str) -> None:
        """Validate cluster name against allowed pattern.

        Args:
            name: Cluster name to validate

        Raises:
            ClusterError: If name is invalid
        """
        if not CLUSTER_NAME_PATTERN.match(name):
            raise ClusterError(
                f"Invalid cluster name '{name}': must start with alphanumeric character "
                "and contain only alphanumeric, underscore, or hyphen characters"
            )

    def _cluster_path(self, name: str) -> Path:
        """Get path to cluster YAML file."""
        return self.clusters_dir / f"{name}.yaml"

    def create(self, name: str, hosts: list[str], description: str = "",
               user: str | None = None, cache_dir: str | None = None) -> None:
        """Create a new named cluster.

        Args:
            name: Cluster name
            hosts: List of host addresses
            description: Optional cluster description
            user: Optional SSH username for this cluster
            cache_dir: Optional HuggingFace cache directory for this cluster

        Raises:
            ClusterError: If cluster already exists or name is invalid
        """
        self._validate_name(name)

        cluster_path = self._cluster_path(name)
        if cluster_path.exists():
            raise ClusterError(f"Cluster '{name}' already exists")

        cluster_def = ClusterDefinition(name=name, hosts=hosts, description=description,
                                        user=user, cache_dir=cache_dir)
        self._write_cluster(cluster_def)
        logger.info("Created cluster '%s' with %d hosts", name, len(hosts))

    def get(self, name: str) -> ClusterDefinition:
        """Load cluster definition by name.

        Args:
            name: Cluster name

        Returns:
            ClusterDefinition for the requested cluster

        Raises:
            ClusterError: If cluster not found
        """
        cluster_path = self._cluster_path(name)
        if not cluster_path.exists():
            raise ClusterError(f"Cluster '{name}' not found")

        return self._read_cluster(cluster_path)

    def update(
            self,
            name: str,
            hosts: list[str] | None = None,
            description: str | None = None,
            user: str | None = _UNSET,
            cache_dir: str | None = _UNSET,
    ) -> None:
        """Update existing cluster definition.

        Args:
            name: Cluster name
            hosts: New host list (if provided)
            description: New description (if provided)
            user: SSH username (if provided; pass ``None`` explicitly to clear)
            cache_dir: HuggingFace cache directory (if provided; pass ``None`` explicitly to clear)

        Raises:
            ClusterError: If cluster does not exist
        """
        # Load existing cluster
        cluster_def = self.get(name)

        # Update provided fields
        if hosts is not None:
            cluster_def.hosts = hosts
            logger.debug("Updated hosts for cluster '%s'", name)

        if description is not None:
            cluster_def.description = description
            logger.debug("Updated description for cluster '%s'", name)

        if user is not _UNSET:
            cluster_def.user = user
            logger.debug("Updated user for cluster '%s'", name)

        if cache_dir is not _UNSET:
            cluster_def.cache_dir = cache_dir
            logger.debug("Updated cache_dir for cluster '%s'", name)

        # Write back
        self._write_cluster(cluster_def)
        logger.info("Updated cluster '%s'", name)

    def list_clusters(self) -> list[ClusterDefinition]:
        """List all defined clusters.

        Returns:
            List of ClusterDefinition objects sorted by name
        """
        clusters = []
        for yaml_file in self.clusters_dir.glob("*.yaml"):
            try:
                cluster_def = self._read_cluster(yaml_file)
                clusters.append(cluster_def)
            except Exception as e:
                logger.warning("Failed to load cluster from %s: %s", yaml_file, e)

        clusters.sort(key=lambda c: c.name)
        logger.debug("Listed %d clusters", len(clusters))
        return clusters

    def delete(self, name: str) -> None:
        """Delete a cluster definition.

        Args:
            name: Cluster name

        Raises:
            ClusterError: If cluster not found
        """
        cluster_path = self._cluster_path(name)
        if not cluster_path.exists():
            raise ClusterError(f"Cluster '{name}' not found")

        cluster_path.unlink()
        logger.info("Deleted cluster '%s'", name)

        # Clear default if it pointed to this cluster
        default_name = self.get_default()
        if default_name == name:
            self.unset_default()
            logger.debug("Cleared default pointer as it referenced deleted cluster '%s'", name)

    def set_default(self, name: str) -> None:
        """Set the default cluster.

        Args:
            name: Cluster name

        Raises:
            ClusterError: If cluster does not exist
        """
        # Verify cluster exists
        self.get(name)

        self.default_file.write_text(name)
        logger.info("Set default cluster to '%s'", name)

    def unset_default(self) -> None:
        """Clear the default cluster marker.

        Does not raise error if default is not set.
        """
        if self.default_file.exists():
            self.default_file.unlink()
            logger.debug("Unset default cluster")

    def get_default(self) -> str | None:
        """Get the default cluster name.

        Returns:
            Default cluster name if set and cluster exists, None otherwise
        """
        if not self.default_file.exists():
            return None

        default_name = self.default_file.read_text().strip()
        if not default_name:
            return None

        # Verify cluster still exists
        cluster_path = self._cluster_path(default_name)
        if not cluster_path.exists():
            logger.warning("Default cluster '%s' no longer exists, clearing default", default_name)
            self.unset_default()
            return None

        return default_name

    def _write_cluster(self, cluster_def: ClusterDefinition) -> None:
        """Write cluster definition to YAML file."""
        cluster_path = self._cluster_path(cluster_def.name)

        data: dict[str, Any] = {
            "name": cluster_def.name,
            "hosts": cluster_def.hosts,
            "description": cluster_def.description,
        }
        if cluster_def.user is not None:
            data["user"] = cluster_def.user
        if cluster_def.cache_dir is not None:
            data["cache_dir"] = cluster_def.cache_dir

        with cluster_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.debug("Wrote cluster definition to %s", cluster_path)

    def _read_cluster(self, cluster_path: Path) -> ClusterDefinition:
        """Read cluster definition from YAML file."""
        from sparkrun.utils import load_yaml

        data = load_yaml(cluster_path)
        if not data:
            raise ClusterError(f"Invalid cluster file format: {cluster_path}")

        return ClusterDefinition(
            name=data.get("name", ""),
            hosts=data.get("hosts", []),
            description=data.get("description", ""),
            user=data.get("user"),
            cache_dir=data.get("cache_dir"),
        )


# ---------------------------------------------------------------------------
# Cluster status query — business logic extracted from CLI
# ---------------------------------------------------------------------------

def query_cluster_status(
        host_list: list[str],
        ssh_kwargs: dict[str, Any],
        cache_dir: str,
) -> ClusterStatusResult:
    """Query sparkrun containers on hosts and classify them.

    Runs ``docker ps`` on each host in parallel, parses the output,
    groups containers into clusters vs solo entries, enriches with
    cached job metadata, and identifies idle hosts and pending ops.

    Args:
        host_list: Target hostnames/IPs to query.
        ssh_kwargs: SSH connection keyword arguments.
        cache_dir: Cache directory for job metadata and pending ops.

    Returns:
        A :class:`ClusterStatusResult` with all collected data.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sparkrun.orchestration.ssh import run_remote_command
    from sparkrun.orchestration.job_metadata import load_job_metadata
    from sparkrun.core.pending_ops import list_pending_ops

    docker_cmd = (
        "docker ps --filter 'name=sparkrun_' "
        "--format '{{.Names}}\\t{{.Status}}\\t{{.Image}}'"
    )

    # Query all hosts in parallel
    with ThreadPoolExecutor(max_workers=len(host_list)) as executor:
        futures = {
            executor.submit(
                run_remote_command, host, docker_cmd, timeout=15, **ssh_kwargs,
            ): host
            for host in host_list
        }
        results = {}
        for future in as_completed(futures):
            host = futures[future]
            results[host] = future.result()

    # Collect per-host container info: list of (name, status, image)
    host_containers: dict[str, list[tuple[str, str, str]]] = {}
    errors: dict[str, str] = {}
    for host in host_list:
        result = results[host]
        if not result.success:
            errors[host] = result.stderr.strip()
            continue
        entries = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                entries.append((parts[0], parts[1], parts[2]))
            else:
                entries.append((line.strip(), "", ""))
        host_containers[host] = entries

    # Build cluster groups: cluster_id -> [(host, role, status, image), ...]
    # Anything ending in _solo is standalone; everything else is grouped
    # by cluster_id (name up to last underscore-delimited role suffix).
    groups: dict[str, list[tuple[str, str, str, str]]] = {}
    solo_entries: list[tuple[str, str, str, str]] = []
    total_containers = 0

    for host in host_list:
        for name, status, image in host_containers.get(host, []):
            total_containers += 1
            if name.endswith("_solo"):
                solo_entries.append((host, name, status, image))
            else:
                # Extract cluster_id by stripping the role suffix.
                # Patterns: sparkrun_hash_head, sparkrun_hash_worker,
                #           sparkrun_hash_node_0 (SGLang)
                # Strategy: find the cluster_id prefix (sparkrun_{12-char hash})
                # and treat the rest as the role.
                prefix_end = name.find("_", len("sparkrun_"))
                if 0 < prefix_end < len(name) - 1:
                    cluster_id = name[:prefix_end]
                    role = name[prefix_end + 1:]
                else:
                    cluster_id = name
                    role = "?"
                groups.setdefault(cluster_id, []).append((host, role, status, image))

    # Enrich groups with job metadata
    cluster_groups: dict[str, ClusterGroup] = {}
    for cid, members in groups.items():
        meta = load_job_metadata(cid, cache_dir=cache_dir) or {}
        cluster_groups[cid] = ClusterGroup(cluster_id=cid, members=members, meta=meta)

    # Idle hosts: no containers and no errors
    idle_hosts = [h for h in host_list if h not in errors and not host_containers.get(h)]

    # Pending operations filtered to relevant hosts
    pending = list_pending_ops(cache_dir=cache_dir)
    host_set = set(host_list)
    relevant_ops = [
        op for op in pending
        if not op.get("hosts") or host_set & set(op["hosts"])
    ]

    return ClusterStatusResult(
        groups=cluster_groups,
        solo_entries=solo_entries,
        errors=errors,
        idle_hosts=idle_hosts,
        pending_ops=relevant_ops,
        total_containers=total_containers,
        host_count=len(host_list),
    )
