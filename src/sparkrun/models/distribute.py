"""Model distribution via local-to-remote transfer.

Instead of having every host download from HuggingFace Hub independently,
these functions download once (locally or on the head node) and then
rsync the cache directory to targets.
"""

from __future__ import annotations

import logging

from sparkrun.core.config import resolve_hf_cache_home
from sparkrun.models.download import download_model, model_cache_path
from sparkrun.orchestration.primitives import map_transfer_failures
from sparkrun.orchestration.ssh import (
    build_ssh_opts_string,
    run_remote_scripts_parallel,
    run_rsync_parallel,
)
from sparkrun.scripts import read_script

from sparkrun.core.progress import PROGRESS

logger = logging.getLogger(__name__)

# Shell expression evaluated on the remote host when no explicit cache_dir is
# configured.  Mirrors huggingface_hub's resolution order (HF_HOME → $HOME/.cache/
# huggingface) so the remote script writes under the target user's home, not the
# control machine's.  Embedding this (instead of the control machine's already-
# resolved path) is what fixes the cross-user / cross-host case where the SSH
# login user's $HOME differs from the invoking user's $HOME.
REMOTE_DEFAULT_HF_CACHE = r"${HF_HOME:-$HOME/.cache/huggingface}"


def _try_fix_remote_permissions(
    cache_dir: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Best-effort chown of the HF cache on remote hosts before rsync.

    Docker containers create root-owned files in the cache directory.
    This tries non-interactive ``sudo -n chown`` on each host so the
    SSH user can rsync into the directory.  Failures are non-fatal —
    a warning is logged with a hint about ``--save-sudo``.
    """
    script = (
        "set -euo pipefail\n"
        'CACHE_DIR="{cache_dir}"\n'
        '[ -d "$CACHE_DIR" ] || exit 0\n'
        'OWNER=$(stat -c "%U" "$CACHE_DIR" 2>/dev/null || echo "")\n'
        "ME=$(id -un)\n"
        '[ "$OWNER" = "$ME" ] && exit 0\n'
        'sudo -n /usr/bin/chown -R "$ME" "$CACHE_DIR" 2>/dev/null\n'
    ).format(cache_dir=cache_dir)

    results = run_remote_scripts_parallel(
        hosts,
        script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=30,
        dry_run=dry_run,
    )

    failed = [r.host for r in results if not r.success]
    if failed:
        logger.warning(
            "Could not fix cache ownership on %d host(s) — rsync may fail "
            "if Docker left root-owned files.  Run "
            "'sparkrun setup fix-permissions --save-sudo' to enable "
            "passwordless chown for future runs.",
            len(failed),
        )


def distribute_model_from_local(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    local_cache_dir: str | None = None,
    token: str | None = None,
    revision: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    transfer_hosts: list[str] | None = None,
) -> list[str]:
    """Download a model locally then rsync it to all hosts.

    1. Download the model to the local HF cache via :func:`download_model`.
    2. For each host in parallel, rsync the model cache directory.
       Because rsync is incremental, hosts that already have the model
       will complete almost instantly.

    Args:
        model_id: HuggingFace model identifier.
        hosts: Target hostnames or IPs (used for identification/reporting).
        cache_dir: Remote cache directory on target hosts.
        local_cache_dir: Control-machine cache directory for downloads.
            When different from *cache_dir*, the model is downloaded to
            *local_cache_dir* but rsynced to *cache_dir* on remote hosts.
            Defaults to *cache_dir* when not provided.
        token: Optional HuggingFace API token for gated models.
        revision: Optional revision (branch, tag, or commit hash).
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Per-host transfer timeout in seconds.
        dry_run: If True, show what would be done without executing.
        transfer_hosts: Optional IB/fast-network IPs to use for the actual
            data transfer.  Must be same length as *hosts*.  When provided,
            ``transfer_hosts[i]`` is used for SSH connections while
            ``hosts[i]`` is used for identification and error reporting.
            Falls back to *hosts* when ``None``.

    Returns:
        List of hostnames (from *hosts*) where distribution failed
        (empty = full success).
    """
    local_cache = resolve_hf_cache_home(local_cache_dir or cache_dir)
    remote_cache = resolve_hf_cache_home(cache_dir)
    logger.debug("Distributing model '%s' from local to %d host(s)", model_id, len(hosts))

    # Step 1: download model locally
    rc = download_model(model_id, cache_dir=local_cache, token=token, revision=revision, dry_run=dry_run)
    if rc != 0:
        logger.error("Failed to download model '%s' locally — aborting distribution", model_id)
        return list(hosts)

    if not hosts:
        return []

    xfer = transfer_hosts or hosts

    # Step 2: best-effort fix of remote cache ownership before rsync
    _try_fix_remote_permissions(
        remote_cache,
        hosts,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        dry_run=dry_run,
    )

    # Step 3: rsync model cache to all hosts in parallel
    local_model_path = model_cache_path(model_id, local_cache)
    remote_model_path = model_cache_path(model_id, remote_cache)
    results = run_rsync_parallel(
        local_model_path,
        xfer,
        remote_model_path,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
        dry_run=dry_run,
    )

    # Map transfer IPs back to management hosts for failure reporting
    failed = map_transfer_failures(results, xfer, hosts)
    if failed:
        logger.warning("Model distribution failed on hosts: %s", failed)
    else:
        logger.log(PROGRESS, "  Model synced to %d host(s)", len(hosts))

    return failed


def distribute_model_from_head(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    worker_transfer_hosts: list[str] | None = None,
) -> list[str]:
    """Download a model on the head node then distribute to remaining hosts.

    1. Download on ``hosts[0]`` using the existing ``model_sync.sh``.
    2. If there is only one host, done.
    3. Run ``model_distribute.sh`` on ``hosts[0]`` to rsync to ``hosts[1:]``.

    Args:
        model_id: HuggingFace model identifier.
        hosts: Cluster hostnames (``hosts[0]`` is the head).
        cache_dir: Override for the HuggingFace cache directory.
        revision: Optional revision (branch, tag, or commit hash).
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Per-operation timeout in seconds.
        dry_run: If True, show what would be done without executing.
        worker_transfer_hosts: Optional IB/fast-network IPs for workers
            (``hosts[1:]``).  Used as targets in the distribution script
            running on the head.  Falls back to ``hosts[1:]`` when ``None``.

    Returns:
        List of hostnames where distribution failed (empty = full success).
    """
    from sparkrun.orchestration.distribution import _distribute_from_head

    if not hosts:
        return []

    # When an explicit cache_dir is configured (e.g. via `sparkrun cluster
    # update --cache-dir`), trust it as a literal absolute path valid on the
    # target.  Otherwise defer resolution to the remote shell so the path is
    # relative to the SSH login user's home, not the control machine's.
    cache = cache_dir if cache_dir else REMOTE_DEFAULT_HF_CACHE
    head = hosts[0]
    logger.debug("Distributing model '%s' from head (%s) to %d host(s)", model_id, head, len(hosts))

    # Build ensure script (download model on head)
    from sparkrun.models.download import is_gguf_model, parse_gguf_model_spec

    revision_flag = "--revision %s " % revision if revision else ""
    if is_gguf_model(model_id):
        repo_id, quant = parse_gguf_model_spec(model_id)
        ensure_script = read_script("model_sync_gguf.sh").format(
            repo_id=repo_id,
            quant=quant or "",
            cache=cache,
            revision_flag=revision_flag,
        )
    else:
        ensure_script = read_script("model_sync.sh").format(
            model_id=model_id,
            cache=cache,
            revision_flag=revision_flag,
        )

    # Build distribute script (rsync from head to workers)
    targets = worker_transfer_hosts or hosts[1:]
    model_path = model_cache_path(model_id, cache)
    ssh_opts = build_ssh_opts_string(
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
    )
    dist_script = read_script("model_distribute.sh").format(
        model_path=model_path,
        targets=" ".join(targets),
        ssh_opts=ssh_opts,
        ssh_user=ssh_user or "",
    )

    return _distribute_from_head(
        head=head,
        hosts=hosts,
        ensure_script=ensure_script,
        distribute_script=dist_script,
        resource_label="Model '%s'" % model_id,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
        dry_run=dry_run,
    )
