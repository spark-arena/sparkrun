"""Docker storage-driver probe + consistency check.

Detects each host's Docker storage driver and snapshotter so the CLI can
warn when a cluster has heterogeneous drivers across hosts.  Different
drivers (e.g. ``overlay2`` vs containerd-snapshotter ``overlayfs``)
produce different local Image IDs for the same registry image, which
causes ``sparkrun`` to unnecessarily re-sync containers — see
`#152 <https://github.com/spark-arena/sparkrun/issues/152>`_.

The runtime image-sync code (:func:`sparkrun.containers.distribute._images_match`)
already falls back to RepoDigest comparison so the bug is non-blocking,
but a proactive setup-time warning lets users normalize their fleet
before they hit the slow path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# `docker info` doesn't expose "is this the containerd snapshotter?" via
# a stable top-level field.  We get the driver name and the DriverStatus
# rows; the snapshotter shows itself as a row whose key is
# ``driver-type`` and whose value starts with ``io.containerd.snapshotter``.
_DOCKER_INFO_PROBE = (
    "docker info --format \"{{.Driver}}|{{range .DriverStatus}}{{index . 0}}={{index . 1}};{{end}}\" 2>/dev/null || echo 'UNAVAILABLE|'"
)


@dataclass(frozen=True)
class DockerDriverInfo:
    """Parsed Docker storage driver info for a single host."""

    driver: str
    """The top-level driver name reported by ``docker info`` (e.g.
    ``overlay2``, ``overlayfs``).  ``UNAVAILABLE`` when ``docker info``
    failed (Docker not installed / daemon not running / not in group).
    """

    snapshotter: bool
    """True when ``DriverStatus`` indicates the containerd snapshotter
    is in use (driver-type starts with ``io.containerd.snapshotter``)."""

    @property
    def signature(self) -> str:
        """Stable identifier for cross-host comparison.

        Two hosts with the same signature are guaranteed to compute the
        same local Image ID for any given registry image.  Hosts with
        different signatures may not.
        """
        return "%s+snapshotter" % self.driver if self.snapshotter else self.driver

    @property
    def is_available(self) -> bool:
        return self.driver != "UNAVAILABLE"


def parse_docker_info_output(raw: str) -> DockerDriverInfo:
    """Parse one host's stdout from :data:`_DOCKER_INFO_PROBE`."""
    line = raw.strip().splitlines()[-1] if raw.strip() else ""
    if "|" not in line:
        return DockerDriverInfo(driver="UNAVAILABLE", snapshotter=False)
    driver, status_blob = line.split("|", 1)
    driver = driver.strip() or "UNAVAILABLE"
    snapshotter = False
    for entry in status_blob.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        if key.strip() == "driver-type" and value.strip().startswith("io.containerd.snapshotter"):
            snapshotter = True
            break
    return DockerDriverInfo(driver=driver, snapshotter=snapshotter)


def detect_docker_drivers(
    hosts: list[str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> dict[str, DockerDriverInfo]:
    """Probe each host's Docker storage driver in parallel.

    Hosts where the probe fails (SSH error, Docker not running) are
    returned with :attr:`DockerDriverInfo.driver` set to ``"UNAVAILABLE"``.
    """
    if not hosts:
        return {}

    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    kw = ssh_kwargs or {}
    results = run_remote_scripts_parallel(
        hosts,
        _DOCKER_INFO_PROBE,
        timeout=15,
        dry_run=dry_run,
        quiet=True,
        **kw,
    )

    out: dict[str, DockerDriverInfo] = {}
    for result in results:
        if dry_run:
            out[result.host] = DockerDriverInfo(driver="overlay2", snapshotter=False)
            continue
        if not result.success:
            out[result.host] = DockerDriverInfo(driver="UNAVAILABLE", snapshotter=False)
            continue
        out[result.host] = parse_docker_info_output(result.stdout)
    return out


def check_driver_consistency(
    driver_map: dict[str, DockerDriverInfo],
) -> tuple[bool, dict[str, list[str]]]:
    """Group hosts by driver signature.

    Returns ``(is_consistent, groups)`` where ``groups`` maps each
    distinct signature to the list of hosts using it.  Hosts whose
    probe came back ``UNAVAILABLE`` are excluded from the consistency
    decision (we can't know what driver they'd use) but reported
    separately under the ``"UNAVAILABLE"`` group.
    """
    groups: dict[str, list[str]] = {}
    for host, info in driver_map.items():
        groups.setdefault(info.signature, []).append(host)

    distinct_available = {sig for sig in groups if sig != "UNAVAILABLE"}
    is_consistent = len(distinct_available) <= 1
    return is_consistent, groups


def format_driver_warning(groups: dict[str, list[str]]) -> str:
    """Render a multi-line remediation message for inconsistent drivers.

    The caller is expected to print this via ``click.echo(... err=True)``
    only when :func:`check_driver_consistency` reports inconsistency.
    """
    lines: list[str] = []
    lines.append("Warning: Docker storage drivers are inconsistent across hosts.")
    lines.append("")
    lines.append("  Per-host driver:")
    for sig in sorted(groups):
        for host in sorted(groups[sig]):
            lines.append("    %s: %s" % (host, sig))
    lines.append("")
    lines.append("  Different drivers compute different local Image IDs for the")
    lines.append("  same registry image (issue #152), which can trigger unnecessary")
    lines.append("  container re-syncs.  Sparkrun's RepoDigest fallback keeps things")
    lines.append("  working, but to fully avoid the slow path, normalize every host")
    lines.append("  on overlay2.  On each host using a different driver:")
    lines.append("")
    lines.append("    sudo systemctl stop docker docker.socket containerd")
    lines.append("    sudo rm -rf /var/lib/docker /var/lib/containerd")
    lines.append("    sudo mkdir -p /etc/docker")
    lines.append('    echo \'{"storage-driver": "overlay2"}\' | sudo tee /etc/docker/daemon.json')
    lines.append("    sudo systemctl start containerd docker")
    lines.append("")
    lines.append("  Note: the rm step deletes all local Docker data on that host;")
    lines.append("  cached images will be re-pulled on next use.")
    return "\n".join(lines)
