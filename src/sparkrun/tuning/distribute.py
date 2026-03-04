"""Distribute tuning configs from local cache to remote hosts."""

from __future__ import annotations

import logging

from sparkrun.core.hosts import is_local_host
from sparkrun.tuning.sync import _get_local_tuning_dir

logger = logging.getLogger(__name__)


def distribute_tuning_to_hosts(
    runtime: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
    transfer_mode: str = "local",
) -> list[str]:
    """Distribute local tuning configs to remote hosts via rsync.

    Pushes the local tuning config directory (populated by
    :func:`sparkrun.tuning.sync.sync_registry_tuning` or local tuning
    runs) to all remote hosts so that worker nodes have the same
    configs mounted into their containers.

    For ``push`` and ``delegated`` modes, rsyncs to the head node first,
    then runs a distribution rsync from head to workers.  Tuning configs
    are small, so the two-hop overhead is negligible.

    Args:
        runtime: Runtime name (e.g. ``"sglang"``, ``"vllm-ray"``).
        hosts: Target hostnames or IPs.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        dry_run: If True, log what would be done without executing.
        transfer_mode: Distribution strategy (``"local"``, ``"push"``,
            or ``"delegated"``).

    Returns:
        List of hostnames where distribution failed (empty = success).
    """
    tuning_dir = _get_local_tuning_dir(runtime)

    # No-op if local tuning directory doesn't exist or has no JSON files
    if not tuning_dir.is_dir() or not any(tuning_dir.rglob("*.json")):
        logger.debug("No local tuning configs for %s, skipping distribution", runtime)
        return []

    # Filter out localhost — no need to rsync to self
    remote_hosts = [h for h in hosts if not is_local_host(h)]
    if not remote_hosts:
        logger.debug("No remote hosts for tuning distribution")
        return []

    from sparkrun.orchestration.ssh import run_rsync_parallel, build_ssh_opts_string, run_remote_script

    source = str(tuning_dir)

    if transfer_mode in ("push", "delegated") and len(remote_hosts) > 1:
        # Two-hop: rsync to head, then head distributes to workers
        head = remote_hosts[0]
        workers = remote_hosts[1:]

        logger.info(
            "Distributing tuning configs (%s) via %s mode: head=%s, %d worker(s)",
            runtime, transfer_mode, head, len(workers),
        )

        # Step 1: rsync to head
        head_results = run_rsync_parallel(
            source, [head], source,
            ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
            rsync_options=["-az", "--delete", "--partial"],
            dry_run=dry_run,
        )
        head_failed = [r.host for r in head_results if not r.success]
        if head_failed:
            logger.warning("Tuning config push to head failed: %s", head)
            return list(remote_hosts)

        # Step 2: distribute from head to workers
        ssh_opts = build_ssh_opts_string(
            ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        )
        user_prefix = "%s@" % ssh_user if ssh_user else ""
        targets_str = " ".join(workers)
        dist_script = (
            'set -euo pipefail\n'
            'SOURCE="{source}"\n'
            'for TARGET in {targets}; do\n'
            '  rsync -az --delete --partial -e "ssh {ssh_opts}" '
            '"$SOURCE/" {user_prefix}$TARGET:"$SOURCE/"\n'
            'done\n'
        ).format(
            source=source, targets=targets_str,
            ssh_opts=ssh_opts, user_prefix=user_prefix,
        )

        dist_result = run_remote_script(
            head, dist_script,
            ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
            timeout=120, dry_run=dry_run,
        )
        if not dist_result.success:
            logger.warning("Tuning config distribution from head failed (rc=%d)",
                           dist_result.returncode)
            return list(workers)

        logger.info("Tuning configs distributed via %s mode to all %d host(s)",
                     transfer_mode, len(remote_hosts))
        return []

    # Default (local mode) or single remote host: direct rsync to all
    logger.info(
        "Distributing tuning configs (%s) to %d host(s)",
        runtime, len(remote_hosts),
    )

    results = run_rsync_parallel(
        source, remote_hosts, source,
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        rsync_options=["-az", "--delete", "--partial"],
        dry_run=dry_run,
    )

    failed = [r.host for r in results if not r.success]
    if failed:
        logger.warning("Tuning config distribution failed on hosts: %s", failed)
    else:
        logger.info("Tuning configs distributed to all %d host(s)", len(remote_hosts))

    return failed
