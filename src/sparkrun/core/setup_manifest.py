"""Setup manifest tracking for sparkrun setup operations.

Tracks which setup phases have been applied to a cluster, enabling
the uninstall command to reverse them. Manifests are stored as YAML
files alongside cluster definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PhaseRecord:
    """Record of a single setup phase applied to a cluster."""

    applied: bool
    timestamp: str  # ISO 8601
    hosts: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SetupManifest:
    """Manifest of all setup phases applied to a cluster."""

    version: int
    cluster: str
    created: str  # ISO 8601
    updated: str  # ISO 8601
    user: str
    hosts: list[str]
    phases: dict[str, PhaseRecord] = field(default_factory=dict)


class ManifestManager:
    """Manages setup manifest files alongside cluster definitions."""

    MANIFEST_VERSION = 1

    def __init__(self, clusters_dir: Path) -> None:
        self.clusters_dir = Path(clusters_dir)
        self.clusters_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, cluster_name: str) -> Path:
        return self.clusters_dir / ("%s.manifest.yaml" % cluster_name)

    def load(self, cluster_name: str) -> SetupManifest | None:
        """Load manifest for a cluster, or None if not found."""
        path = self._manifest_path(cluster_name)
        if not path.exists():
            return None

        try:
            with path.open() as f:
                data = yaml.safe_load(f)
            if not data or not isinstance(data, dict):
                return None

            phases = {}
            for name, pdata in data.get("phases", {}).items():
                phases[name] = PhaseRecord(
                    applied=pdata.get("applied", False),
                    timestamp=pdata.get("timestamp", ""),
                    hosts=pdata.get("hosts", []),
                    extra=pdata.get("extra", {}),
                )

            return SetupManifest(
                version=data.get("version", self.MANIFEST_VERSION),
                cluster=data.get("cluster", cluster_name),
                created=data.get("created", ""),
                updated=data.get("updated", ""),
                user=data.get("user", ""),
                hosts=data.get("hosts", []),
                phases=phases,
            )
        except Exception:
            logger.warning("Failed to load manifest for cluster '%s'", cluster_name, exc_info=True)
            return None

    def save(self, manifest: SetupManifest) -> None:
        """Save manifest to YAML file."""
        path = self._manifest_path(manifest.cluster)

        phases_data = {}
        for name, record in manifest.phases.items():
            phases_data[name] = {
                "applied": record.applied,
                "timestamp": record.timestamp,
                "hosts": record.hosts,
                "extra": record.extra,
            }

        data = {
            "version": manifest.version,
            "cluster": manifest.cluster,
            "created": manifest.created,
            "updated": manifest.updated,
            "user": manifest.user,
            "hosts": manifest.hosts,
            "phases": phases_data,
        }

        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.debug("Saved manifest for cluster '%s' to %s", manifest.cluster, path)

    def delete(self, cluster_name: str) -> None:
        """Delete manifest file for a cluster."""
        path = self._manifest_path(cluster_name)
        if path.exists():
            path.unlink()
            logger.info("Deleted manifest for cluster '%s'", cluster_name)

    def record_phase(
        self,
        cluster_name: str,
        user: str,
        hosts: list[str],
        phase: str,
        **extra: Any,
    ) -> None:
        """Record a completed setup phase, creating or updating the manifest.

        If a phase already exists, hosts are unioned and extra fields are
        merged (new keys added, existing keys preserved).
        """
        now = datetime.now(timezone.utc).isoformat()
        manifest = self.load(cluster_name)

        if manifest is None:
            manifest = SetupManifest(
                version=self.MANIFEST_VERSION,
                cluster=cluster_name,
                created=now,
                updated=now,
                user=user,
                hosts=list(hosts),
                phases={},
            )
        else:
            # Union top-level hosts
            existing_hosts = set(manifest.hosts)
            for h in hosts:
                if h not in existing_hosts:
                    manifest.hosts.append(h)
                    existing_hosts.add(h)
            manifest.updated = now
            if user:
                manifest.user = user

        # Upsert phase record
        if phase in manifest.phases:
            existing = manifest.phases[phase]
            # Union hosts
            host_set = set(existing.hosts)
            for h in hosts:
                if h not in host_set:
                    existing.hosts.append(h)
                    host_set.add(h)
            # Merge extra (existing keys preserved, new keys added)
            for k, v in extra.items():
                if k not in existing.extra:
                    existing.extra[k] = v
            existing.timestamp = now
            existing.applied = True
        else:
            manifest.phases[phase] = PhaseRecord(
                applied=True,
                timestamp=now,
                hosts=list(hosts),
                extra=dict(extra),
            )

        self.save(manifest)
        logger.info("Recorded phase '%s' for cluster '%s' (%d hosts)", phase, cluster_name, len(hosts))
