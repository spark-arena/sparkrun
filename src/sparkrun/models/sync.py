"""Model distribution across cluster nodes."""

from __future__ import annotations

import logging

from sparkrun.core.config import resolve_hf_cache_home
from sparkrun.orchestration.primitives import sync_resource_to_hosts
from sparkrun.scripts import read_script

logger = logging.getLogger(__name__)


def sync_model_to_hosts(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Download a model on all hosts in parallel.

    Uses ``huggingface-cli`` on each remote host to download the model
    if it is not already cached.

    Args:
        model_id: HuggingFace model identifier.
        hosts: List of remote hostnames or IPs.
        cache_dir: Override for the HuggingFace cache directory.
        revision: Optional revision (branch, tag, or commit hash).
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        dry_run: If True, show what would be done without executing.

    Returns:
        List of hostnames where the sync failed.
    """
    cache = resolve_hf_cache_home(cache_dir)
    revision_flag = "--revision %s " % revision if revision else ""

    script = read_script("model_sync.sh").format(
        model_id=model_id,
        cache=cache,
        revision_flag=revision_flag,
    )

    return sync_resource_to_hosts(
        script,
        hosts,
        "Model",
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )
