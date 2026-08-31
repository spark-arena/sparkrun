"""Container image distribution across cluster nodes."""

from __future__ import annotations

import logging

from sparkrun.orchestration.primitives import sync_resource_to_hosts
from sparkrun.scripts import read_script
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)


def sync_image_to_hosts(
    image: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    dry_run: bool = False,
    force_pull: bool = False,
) -> list[str]:
    """Ensure a container image is available on all hosts, pulling in parallel.

    Every host fetches from the registry itself, concurrently — no head node
    pulls on anyone's behalf and nothing crosses the control machine.  This is
    the ``pull`` transfer mode, and it is the only correct strategy when nodes
    run *different* images: ``docker save | ssh docker load`` would copy a
    machine-tuned image onto the wrong machine.

    Note the presence check inside ``image_sync.sh`` is metadata-only, so an
    image re-pushed under the same tag is **not** refreshed.  Pass *force_pull*
    (``sparkrun run --rebuild``) to bypass it.

    Args:
        image: Container image reference.
        hosts: List of remote hostnames or IPs.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        dry_run: If True, show what would be done without executing.
        force_pull: Re-pull even when a copy is already present.

    Returns:
        List of hostnames where the image sync failed.
    """
    script = read_script("image_sync.sh").format(image=quote(image), force_pull="1" if force_pull else "0")

    return sync_resource_to_hosts(
        script,
        hosts,
        "Image",
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )
