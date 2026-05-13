"""Container image distribution via local-to-remote transfer.

Instead of having every host pull from the internet, these functions
pull once (locally or on the head node) and then stream the image
to targets via ``docker save | ssh … docker load``.

A hash check (comparing Docker image IDs) is performed before each
transfer so hosts that already have the correct image are skipped.

# TODO: [FUTURE]: allow alternatives to docker!
"""

from __future__ import annotations

import logging

from sparkrun.containers.registry import ensure_image, get_image_identity
from sparkrun.orchestration.primitives import map_transfer_failures
from sparkrun.orchestration.ssh import (
    RemoteResult,
    build_ssh_opts_string,
    run_pipeline_to_remotes_parallel,
    run_remote_command,
)
from sparkrun.scripts import read_script
from sparkrun.utils.shell import args_list_to_shell_str, quote

from sparkrun.core.progress import PROGRESS

logger = logging.getLogger(__name__)

# Command to get a Docker image identity (Id + RepoDigests) on a remote host.
# Empty output = image not present.  Output shape after substitution:
#     <sha256:id>|<repo1@sha256:dig> <repo2@sha256:dig> ...
_REMOTE_IMAGE_IDENTITY_CMD = (
    "docker image inspect --format '{{{{.Id}}}}|{{{{range .RepoDigests}}}}{{{{.}}}} {{{{end}}}}' {image} 2>/dev/null || true"
)


def _parse_identity(raw: str) -> tuple[str | None, list[str]]:
    """Parse the output of ``_REMOTE_IMAGE_IDENTITY_CMD``.

    Returns ``(image_id, repo_digests)``.  ``image_id`` is ``None`` when
    the image is absent (empty output).
    """
    raw = raw.strip()
    if not raw:
        return None, []
    image_id, _, digests_str = raw.partition("|")
    return (image_id or None), digests_str.split()


def _digest_shas(repo_digests: list[str]) -> set[str]:
    """Extract the bare ``sha256:...`` portion from each repo digest.

    A RepoDigest has the form ``repo@sha256:hex``.  Comparing on just the
    digest portion makes two images count as identical even when tagged
    under different repository names (e.g. mirrored registries).
    """
    return {d.rsplit("@", 1)[-1] for d in repo_digests if "@" in d}


def _images_match(
    local_id: str | None,
    local_digests: list[str],
    remote_id: str | None,
    remote_digests: list[str],
) -> bool:
    """Decide whether two image identities refer to the same image.

    Match if **either** the image IDs are equal **or** any RepoDigest sha
    is shared.  Image IDs vary across hosts with different Docker storage
    drivers (issue #152), so RepoDigests are the storage-driver-agnostic
    signal.  RepoDigests are absent for locally-built/save-loaded images,
    so we keep Id equality as a fallback.
    """
    if not remote_id and not remote_digests:
        return False
    if local_id and remote_id and local_id == remote_id:
        return True
    local_shas = _digest_shas(local_digests)
    remote_shas = _digest_shas(remote_digests)
    return bool(local_shas & remote_shas)


def _check_remote_image_identities(
    image: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, tuple[str | None, list[str]]]:
    """Check the Docker image identity (Id + RepoDigests) on multiple hosts.

    Args:
        image: Image reference to check.
        hosts: Target hostnames or IPs.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        dry_run: If True, return empty dict (skip checks).

    Returns:
        Mapping of host → ``(image_id, repo_digests)``.  Hosts where the
        image is absent or the SSH command failed are omitted.
    """
    if dry_run or not hosts:
        return {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    cmd = _REMOTE_IMAGE_IDENTITY_CMD.format(image=quote(image))
    result_map: dict[str, tuple[str | None, list[str]]] = {}

    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {
            executor.submit(
                run_remote_command,
                host,
                cmd,
                ssh_user=ssh_user,
                ssh_key=ssh_key,
                ssh_options=ssh_options,
                timeout=15,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            result: RemoteResult = future.result()
            if not result.success:
                logger.debug("  %s: identity check failed (rc=%s)", result.host, result.returncode)
                continue
            image_id, repo_digests = _parse_identity(result.stdout)
            logger.debug(
                "  %s: remote id=%s digests=%s",
                result.host,
                image_id or "(absent)",
                repo_digests or "(none)",
            )
            if image_id is None and not repo_digests:
                continue
            result_map[result.host] = (image_id, repo_digests)

    return result_map


def _filter_hosts_needing_image(
    image: str,
    hosts: list[str],
    local_image_id: str | None,
    local_repo_digests: list[str] | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Return the subset of hosts that need the image transferred.

    Compares the local image identity (Id + RepoDigests) with each remote
    host's identity.  Hosts whose images match are skipped.  RepoDigest
    overlap is preferred because Docker image IDs differ across hosts that
    use different storage drivers (issue #152); Id equality is the
    fallback for locally-built/save-loaded images that lack RepoDigests.

    Args:
        image: Image reference.
        hosts: Candidate target hosts.
        local_image_id: Local Docker image ID (from
            :func:`get_image_identity`).
        local_repo_digests: Local RepoDigests (from
            :func:`get_image_identity`).
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        dry_run: If True, return all hosts (no filtering).

    Returns:
        List of hosts that need the image.
    """
    if dry_run or not hosts:
        return list(hosts)
    if not local_image_id and not local_repo_digests:
        return list(hosts)

    local_digests = local_repo_digests or []
    logger.debug("Local image '%s' id=%s digests=%s", image, local_image_id, local_digests)

    remote_identities = _check_remote_image_identities(
        image,
        hosts,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
    )

    needs_transfer = []
    for host in hosts:
        remote_id, remote_digests = remote_identities.get(host, (None, []))
        if _images_match(local_image_id, local_digests, remote_id, remote_digests):
            reason = "digest match" if _digest_shas(local_digests) & _digest_shas(remote_digests) else "id match"
            logger.info("  %s: image up-to-date (%s), skipping", host, reason)
        else:
            if remote_id or remote_digests:
                logger.info("  %s: image mismatch, will transfer", host)
                logger.debug(
                    "    local: id=%s digests=%s | remote: id=%s digests=%s",
                    local_image_id,
                    local_digests,
                    remote_id,
                    remote_digests,
                )
            else:
                logger.info("  %s: image not present, will transfer", host)
            needs_transfer.append(host)

    if not needs_transfer:
        logger.log(PROGRESS, "  Container image up-to-date on all %d host(s)", len(hosts))
    else:
        logger.log(PROGRESS, "  Container image stale on %d of %d host(s), syncing", len(needs_transfer), len(hosts))

    return needs_transfer


def distribute_image_from_local(
    image: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    transfer_hosts: list[str] | None = None,
) -> list[str]:
    """Pull an image locally then stream it to all hosts via docker save/load.

    1. Ensure the image exists on the local machine (pull if needed).
    2. Hash check: compare local image ID with each remote host's and
       skip hosts that already have the correct image.
    3. For remaining hosts in parallel, run
       ``docker save <image> | ssh host 'docker load'``.

    Args:
        image: Container image reference.
        hosts: Target hostnames or IPs (used for identification/reporting).
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
    logger.debug("Distributing image '%s' from local to %d host(s)", image, len(hosts))

    # Step 1: ensure image exists locally
    rc = ensure_image(image, dry_run=dry_run)
    if rc != 0:
        logger.error("Failed to ensure local image '%s' — aborting distribution", image)
        return list(hosts)

    if not hosts:
        return []

    xfer = transfer_hosts or hosts

    # Step 2: identity check — skip hosts that already have the correct image
    local_id, local_digests = get_image_identity(image) if not dry_run else (None, [])

    needs_transfer = _filter_hosts_needing_image(
        image,
        xfer,
        local_id,
        local_repo_digests=local_digests,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        dry_run=dry_run,
    )

    if not needs_transfer:
        return []

    # Step 3: stream to hosts that need it
    local_cmd = "docker save %s" % quote(image)
    remote_cmd = "docker load"

    results = run_pipeline_to_remotes_parallel(
        needs_transfer,
        local_cmd,
        remote_cmd,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
        dry_run=dry_run,
    )

    # Map transfer IPs back to management hosts for failure reporting
    failed = map_transfer_failures(results, xfer, hosts)
    if failed:
        logger.warning("Image distribution failed on hosts: %s", failed)
    else:
        logger.debug("Image '%s' distributed to %d host(s)", image, len(needs_transfer))

    return failed


def distribute_image_from_head(
    image: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    worker_transfer_hosts: list[str] | None = None,
) -> list[str]:
    """Pull an image on the head node then distribute to remaining hosts.

    1. Pull the image on ``hosts[0]`` using the existing ``image_sync.sh``.
    2. If there is only one host, done.
    3. Run ``image_distribute.sh`` on ``hosts[0]`` to stream to ``hosts[1:]``.

    Args:
        image: Container image reference.
        hosts: Cluster hostnames (``hosts[0]`` is the head).
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

    head = hosts[0]
    logger.debug("Distributing image '%s' from head (%s) to %d host(s)", image, head, len(hosts))

    # Pre-check image status on all hosts to avoid unnecessary work
    if not dry_run:
        remote_identities = _check_remote_image_identities(
            image,
            hosts,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
        )
        ref = remote_identities.get(head)
        if ref:
            ref_id, ref_digests = ref
            # Head already has the image — check which workers need it
            needs_transfer = []
            for h in hosts:
                r_id, r_digests = remote_identities.get(h, (None, []))
                if not _images_match(ref_id, ref_digests, r_id, r_digests):
                    needs_transfer.append(h)
            if not needs_transfer:
                logger.log(PROGRESS, "  Container image up-to-date on all %d host(s)", len(hosts))
                return []

            if len(hosts) > 1:
                logger.log(PROGRESS, "  Container image needs sync on %d of %d host(s)", len(needs_transfer), len(hosts))

            # Filter workers list and corresponding transfer hosts
            workers = hosts[1:]
            wt = worker_transfer_hosts or workers
            filtered = [(w, t) for w, t in zip(workers, wt) if w in needs_transfer]
            if filtered:
                hosts = [head] + [w for w, _ in filtered]
                worker_transfer_hosts = [t for _, t in filtered]
            else:
                # Only head needed the image (rare: head stale, workers current)
                # Fall through — ensure script will pull on head
                hosts = [head]
                worker_transfer_hosts = None

    # Build ensure script (pull image on head)
    ensure_script = read_script("image_sync.sh").format(image=quote(image))

    # Build distribute script (stream from head to workers)
    targets = worker_transfer_hosts or hosts[1:]
    ssh_opts = build_ssh_opts_string(
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
    )
    dist_script = read_script("image_distribute.sh").format(
        image=quote(image),
        targets=args_list_to_shell_str(targets),
        ssh_opts=ssh_opts,
        ssh_user=ssh_user or "",
    )

    return _distribute_from_head(
        head=head,
        hosts=hosts,
        ensure_script=ensure_script,
        distribute_script=dist_script,
        resource_label="Image '%s'" % image,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
        dry_run=dry_run,
    )
