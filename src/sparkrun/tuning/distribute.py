"""Distribute tuning configs from local cache to remote hosts."""

from __future__ import annotations

import logging
import os

from sparkrun.utils import is_local_host
from sparkrun.tuning._common import tuning_configs_present
from sparkrun.tuning.sync import _get_local_tuning_dir, _get_remote_tuning_dir

logger = logging.getLogger(__name__)


def ensure_remote_tuning_dirs(
    runtime: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Create the tuning directory on each host, owned by the SSH user.

    **This is the fix for the root cause, not a convenience.**  The tuning
    directory is bind-mounted into the inference container, and the decision to
    mount it is made from the *control node's* copy (see
    :func:`~sparkrun.tuning._common.tuning_configs_present`) while the mount is
    applied on every host.  On a host where the path does not exist, the Docker
    daemon materializes the bind-mount source itself — creating the whole
    missing path chain **root-owned**.  From that moment the SSH user cannot
    write into the tree, and every later tuning sync dies with::

        rsync: [generator] recv_generator: mkdir ".../tuning/sglang/configs"
            failed: Permission denied (13)
        *** Skipping any contents from this failed directory ***

    No rsync flag repairs that: the problem is the directory, not the
    attributes being requested.  Creating it ourselves first — as the SSH user,
    before the daemon can — is what stops a host ever entering that state.  The
    tuning *runner* has always done this for its own output directory
    (``_common.py``, "Ensure output directory exists on the remote host (as the
    SSH user, not root)"); the launch path never did.

    For hosts already in the broken state, a ``mkdir -p`` fails exactly like
    rsync does, so a failure triggers one best-effort ``sudo -n chown`` repair
    and a retry.

    Returns:
        Hosts where the directory could not be created (empty = success).
        Best-effort: callers warn rather than abort, since a launch without
        tuning configs is slower, not broken.
    """
    # Localhost is skipped for the same reason distribution skips it: the
    # control node's copy is the one we just checked, so it exists there — and
    # SSH-to-self may not even be configured.
    hosts = [h for h in hosts if not is_local_host(h)]
    if not hosts:
        return []

    from sparkrun.orchestration.ssh import run_remote_scripts_parallel
    from sparkrun.orchestration.sudo import ensure_remote_dir_ownership
    from sparkrun.utils.shell import safe_remote_path

    remote_dir = _get_remote_tuning_dir(runtime, ssh_user=ssh_user)
    script = '#!/bin/bash\nset -uo pipefail\nmkdir -p "%s"\n' % safe_remote_path(remote_dir)
    ssh_kw = {"ssh_user": ssh_user, "ssh_key": ssh_key, "ssh_options": ssh_options}

    results = run_remote_scripts_parallel(hosts, script, timeout=30, dry_run=dry_run, **ssh_kw)
    failed = [r.host for r in results if not r.success]
    if not failed:
        return []

    # A root-owned ancestor is the expected reason, and it is repairable where
    # the operator has passwordless sudo.  Scoped to the failing hosts so a
    # healthy cluster never pays for one broken node.
    logger.debug("Tuning dir creation failed on %s; attempting ownership repair", ", ".join(failed))
    ensure_remote_dir_ownership(
        remote_dir,
        failed,
        dry_run=dry_run,
        resource_label="tuning cache",
        **ssh_kw,
    )

    retry = run_remote_scripts_parallel(failed, script, timeout=30, dry_run=dry_run, **ssh_kw)
    still_failed = [r.host for r in retry if not r.success]
    if still_failed:
        logger.warning(
            "Could not create the tuning config directory %s on: %s. "
            "It is most likely owned by root (Docker creates a missing bind-mount "
            "source that way); chown it to the SSH user to re-enable tuning configs.",
            remote_dir,
            ", ".join(still_failed),
        )
    return still_failed


def distribute_tuning_to_hosts(
    runtime: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
    transfer_mode: str = "local",
    preserve_perms: bool = True,
    skip_fan_out: bool = False,
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
        preserve_perms: When ``False``, rsync drops owner/group/perm/time
            preservation (``-r --links`` instead of ``-a``) — the harder
            relaxation for a destination where even the NFS-safe default
            set cannot apply attributes.  Mirrors the model path.
        skip_fan_out: When ``True``, the per-host rsync is skipped entirely
            because the tuning cache is already visible on every node (a
            shared ``$HOME``), so copying it there is redundant.

    Returns:
        List of hostnames where distribution failed (empty = success).
    """
    tuning_dir = _get_local_tuning_dir(runtime)

    # No-op if local tuning directory doesn't exist or has no JSON files.
    # Same predicate that decides the container mount — see tuning_configs_present.
    if not tuning_configs_present(tuning_dir):
        logger.debug("No local tuning configs for %s, skipping distribution", runtime)
        return []

    # Shared-cache fast path: every host already sees this directory, so the
    # fan-out would copy it onto itself.  Mirrors the model path's skip.
    if skip_fan_out:
        logger.debug("Shared tuning cache: skipping per-host tuning distribution")
        return []

    # Filter out localhost — no need to rsync to self
    remote_hosts = [h for h in hosts if not is_local_host(h)]
    if not remote_hosts:
        logger.debug("No remote hosts for tuning distribution")
        return []

    from sparkrun.orchestration.ssh import NFS_SAFE_ATTR_OPTS, run_rsync_parallel, build_ssh_opts_string, run_remote_script
    from sparkrun.orchestration.transfer import map_transfer_failures

    source = str(tuning_dir)
    remote_dest = _get_remote_tuning_dir(runtime, ssh_user=ssh_user)

    # Tuning configs land in the SSH user's own cache dir, which on a shared
    # /home is routinely owned by a different uid than the one we connect as.
    # Without the NFS-safe relaxation rsync transfers every config and then
    # exits 23 setting times on the destination root.  --mkpath because the
    # per-runtime subdirectory may not exist on a host that has never tuned.
    if preserve_perms:
        tuning_rsync_options = ["-az", "--mkpath", "--partial", *NFS_SAFE_ATTR_OPTS]
    else:
        tuning_rsync_options = ["-rz", "--links", "--mkpath", "--partial"]

    # --delete prunes tuning configs that were removed upstream, but only when
    # we can be sure the destination is a *different* directory.  When the
    # remote path equals the local one (_get_remote_tuning_dir returns exactly
    # that for a matching SSH user on Linux) a shared $HOME makes source and
    # destination the same physical directory, and --delete against your own
    # source is how a sync becomes a deletion.  Pruning is a convenience;
    # not destroying the cache is not.
    if os.path.normpath(source) != os.path.normpath(remote_dest):
        tuning_rsync_options.append("--delete")
    else:
        logger.debug(
            "Tuning source and destination paths are identical (%s); omitting --delete in case $HOME is shared",
            source,
        )

    if transfer_mode in ("push", "delegated") and len(remote_hosts) > 1:
        # Two-hop: rsync to head, then head distributes to workers
        head = remote_hosts[0]
        workers = remote_hosts[1:]

        logger.info(
            "Distributing tuning configs (%s) via %s mode: head=%s, %d worker(s)",
            runtime,
            transfer_mode,
            head,
            len(workers),
        )

        # Step 1: rsync to head
        head_results = run_rsync_parallel(
            source,
            [head],
            remote_dest,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            rsync_options=tuning_rsync_options,
            dry_run=dry_run,
        )
        head_failed = map_transfer_failures(head_results, [head], [head])
        if head_failed:
            logger.warning("Tuning config push to head failed: %s", head)
            return list(remote_hosts)

        # Step 2: distribute from head to workers
        ssh_opts = build_ssh_opts_string(
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
        )
        user_prefix = "%s@" % ssh_user if ssh_user else ""
        targets_str = " ".join(workers)
        dist_script = (
            "set -euo pipefail\n"
            'SOURCE="{source}"\n'
            "for TARGET in {targets}; do\n"
            '  rsync {attr_flags} -e "ssh {ssh_opts}" '
            '"$SOURCE/" {user_prefix}$TARGET:"$SOURCE/"\n'
            "done\n"
        ).format(
            source=remote_dest,
            targets=targets_str,
            # Never --delete on this hop: it uses $SOURCE as *both* sides, so on
            # a cluster with a shared $HOME the source and destination are one
            # directory and --delete would prune the cache against itself.  The
            # control→head hop above already did the pruning.
            attr_flags=" ".join(o for o in tuning_rsync_options if not o.startswith("--delete")),
            ssh_opts=ssh_opts,
            user_prefix=user_prefix,
        )

        dist_result = run_remote_script(
            head,
            dist_script,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            timeout=120,
            dry_run=dry_run,
        )
        if not dist_result.success:
            logger.warning("Tuning config distribution from head failed (rc=%d)", dist_result.returncode)
            return list(workers)

        logger.info("Tuning configs distributed via %s mode to all %d host(s)", transfer_mode, len(remote_hosts))
        return []

    # Default (local mode) or single remote host: direct rsync to all
    logger.info(
        "Distributing tuning configs (%s) to %d host(s)",
        runtime,
        len(remote_hosts),
    )

    results = run_rsync_parallel(
        source,
        remote_hosts,
        remote_dest,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        rsync_options=tuning_rsync_options,
        dry_run=dry_run,
    )

    failed = map_transfer_failures(results, remote_hosts, remote_hosts)
    if failed:
        logger.warning("Tuning config distribution failed on hosts: %s", failed)
    else:
        logger.info("Tuning configs distributed to all %d host(s)", len(remote_hosts))

    return failed
