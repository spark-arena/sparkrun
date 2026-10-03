"""Container image distribution via local-to-remote transfer.

Instead of having every host pull from the internet, these functions
pull once (locally or on the head node) and then stream the image
to targets via ``docker save | ssh … docker load``.

A content check is performed before each transfer so hosts that already
have the correct image are skipped.  See :func:`_images_match` for why
comparing Docker image IDs alone is not sufficient across a fleet, and
:func:`~sparkrun.containers.registry.content_signature` for the signal
that is.

# TODO: [FUTURE]: allow alternatives to docker!
"""

from __future__ import annotations

import logging
from typing import Any

from sparkrun.containers.registry import (
    IMAGE_INSPECT_FORMAT,
    LEGACY_IMAGE_INSPECT_FORMAT,
    ImageIdentity,
    ensure_image,
    get_image_identity,
    parse_image_identity,
)
from sparkrun.orchestration.transfer import map_transfer_failures
from sparkrun.orchestration.ssh import (
    HEAD_DISTRIBUTE_MAX_PARALLEL,
    RemoteResult,
    build_ssh_opts_string,
    run_pipeline_to_remotes_parallel,
    run_remote_command,
)
from sparkrun.scripts import read_script
from sparkrun.utils.shell import args_list_to_shell_str, quote

from sparkrun.core.progress import PROGRESS

logger = logging.getLogger(__name__)

def remote_image_identity_cmd(image: str) -> str:
    """Build the remote command that reports a Docker image's identity.

    Empty output = image not present.  Output shape:
    ``<id>|<repo@sha> ...|<layer> ...|<config json>``

    The image reference is spliced in by concatenation, never with
    :meth:`str.format`.  This matters: the inspect template is full of Go
    template braces, and ``format`` would collapse each ``{{`` to ``{`` --
    and ``docker inspect --format '{.Id}'`` is *valid literal text* that
    happily exits 0 printing the brace string back.  That failure mode is
    silent: every host would report unparseable garbage, every comparison
    would mismatch, and the fleet would re-transfer the image forever.
    (The previous revision avoided this only by quadrupling every brace.)

    The command falls back to the pre-content-signature template so an
    older daemon still reports the identities it can provide, rather than
    looking like the image is missing and forcing a pointless re-transfer.

    The template is single-quoted for the remote shell; it contains no
    single quotes, so no escaping is required.
    """
    ref = quote(image)
    return (
        "docker image inspect --format '"
        + IMAGE_INSPECT_FORMAT
        + "' "
        + ref
        + " 2>/dev/null || docker image inspect --format '"
        + LEGACY_IMAGE_INSPECT_FORMAT
        + "' "
        + ref
        + " 2>/dev/null || true"
    )


def _parse_identity(raw: str) -> ImageIdentity:
    """Parse the output of ``_REMOTE_IMAGE_IDENTITY_CMD``.

    Thin alias for :func:`~sparkrun.containers.registry.parse_image_identity`,
    kept so the remote command and its parser stay adjacent here.
    """
    return parse_image_identity(raw)


def _digest_shas(repo_digests: list[str]) -> set[str]:
    """Extract the bare ``sha256:...`` portion from each repo digest.

    A RepoDigest has the form ``repo@sha256:hex``.  Comparing on just the
    digest portion makes two images count as identical even when tagged
    under different repository names (e.g. mirrored registries).
    """
    return {d.rsplit("@", 1)[-1] for d in repo_digests if "@" in d}


def _as_identity(value: Any) -> ImageIdentity:
    """Coerce a legacy ``(image_id, repo_digests)`` pair to :class:`ImageIdentity`.

    Identities were historically carried as bare pairs.  Accepting both
    shapes lets the content signature be threaded through without forcing
    every caller and every existing mock to change at once; a full
    ``ImageIdentity`` passes straight through, a pair arrives with no
    content signature and falls back to the ID/digest signals.
    """
    if isinstance(value, ImageIdentity):
        return value
    image_id, repo_digests = value
    return ImageIdentity(image_id, list(repo_digests or []), None)


def _images_match(
    local_id: str | None,
    local_digests: list[str],
    remote_id: str | None,
    remote_digests: list[str],
    local_content: str | None = None,
    remote_content: str | None = None,
) -> bool:
    """Decide whether two image identities refer to the same image.

    Matches if **any** of: the content signatures are equal, the image IDs
    are equal, or a RepoDigest sha is shared.

    Each signal covers a gap the others leave, and the content signature is
    checked first because it is the only one that survives both known
    failure modes at once:

    * Image IDs are recomputed by the local storage driver, so hosts on
      different drivers never agree on them for identical content (issue
      #152).  RepoDigests cover that case.
    * RepoDigests are registry metadata, so they are simply **absent** for
      an image that was built locally or moved with
      ``docker save | docker load``.  Image IDs cover that case *only*
      when the drivers match.

    The combination that defeats both — a ``save | load`` fan-out onto a
    host running a different storage driver — produced neither matching IDs
    nor any RepoDigests, so the node was reported stale on every launch and
    re-received the whole image forever.  The content signature is a digest
    of the configuration plus the ordered layer chain, both read verbatim
    from the image manifest and stored identically by every driver, so it
    matches there too.

    ``local_content``/``remote_content`` are optional so existing callers
    that only have IDs and digests keep working unchanged.
    """
    if not remote_id and not remote_digests:
        return False
    if local_content and remote_content and local_content == remote_content:
        return True
    if local_id and remote_id and local_id == remote_id:
        return True
    local_shas = _digest_shas(local_digests)
    remote_shas = _digest_shas(remote_digests)
    return bool(local_shas & remote_shas)


def _match_reason(local: ImageIdentity, remote: ImageIdentity) -> str:
    """Which signal matched, for human-readable progress output."""
    if local.content_sig and remote.content_sig and local.content_sig == remote.content_sig:
        return "content match"
    if local.image_id and remote.image_id and local.image_id == remote.image_id:
        return "id match"
    return "digest match"


def _check_remote_image_identities(
    image: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, ImageIdentity]:
    """Check the Docker image identity on multiple hosts.

    Args:
        image: Image reference to check.
        hosts: Target hostnames or IPs.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        dry_run: If True, return empty dict (skip checks).

    Returns:
        Mapping of host → :class:`ImageIdentity` (ID, RepoDigests and
        content signature).  Hosts where the image is absent or the SSH
        command failed are omitted.
    """
    if dry_run or not hosts:
        return {}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sparkrun.orchestration.ssh import resolve_parallel_cap

    cmd = remote_image_identity_cmd(image)
    result_map: dict[str, ImageIdentity] = {}

    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(hosts))) as executor:
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
            identity = _parse_identity(result.stdout)
            logger.debug(
                "  %s: remote id=%s digests=%s content=%s",
                result.host,
                identity.image_id or "(absent)",
                identity.repo_digests or "(none)",
                (identity.content_sig or "(none)")[:19],
            )
            if identity.image_id is None and not identity.repo_digests:
                continue
            result_map[result.host] = identity

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
    local_content_sig: str | None = None,
) -> list[str]:
    """Return the subset of hosts that need the image transferred.

    Compares the local image identity with each remote host's identity and
    skips the hosts that already match.  See :func:`_images_match` for what
    counts as a match and why ID equality alone is not enough.

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
        local_content_sig: Local content signature (from
            :func:`get_image_identity`).  This is the signal that makes a
            save-loaded copy on a host with a different storage driver
            recognizable as the same image; without it such a host looks
            stale forever.

    Returns:
        List of hosts that need the image.
    """
    if dry_run or not hosts:
        return list(hosts)
    if not local_image_id and not local_repo_digests and not local_content_sig:
        return list(hosts)

    local_digests = local_repo_digests or []
    local = ImageIdentity(local_image_id, local_digests, local_content_sig)
    logger.debug(
        "Local image '%s' id=%s digests=%s content=%s",
        image,
        local_image_id,
        local_digests,
        (local_content_sig or "(none)")[:19],
    )

    remote_identities = _check_remote_image_identities(
        image,
        hosts,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
    )

    needs_transfer = []
    for host in hosts:
        remote = _as_identity(remote_identities.get(host, ImageIdentity.empty()))
        if _images_match(
            local.image_id,
            local.repo_digests,
            remote.image_id,
            remote.repo_digests,
            local.content_sig,
            remote.content_sig,
        ):
            logger.info("  %s: image up-to-date (%s), skipping", host, _match_reason(local, remote))
        else:
            if remote.image_id or remote.repo_digests:
                logger.info("  %s: image mismatch, will transfer", host)
                logger.debug(
                    "    local: id=%s digests=%s content=%s | remote: id=%s digests=%s content=%s",
                    local.image_id,
                    local.repo_digests,
                    (local.content_sig or "(none)")[:19],
                    remote.image_id,
                    remote.repo_digests,
                    (remote.content_sig or "(none)")[:19],
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
    force_pull: bool = False,
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
        force_pull: Re-pull the image locally even when a copy is already
            present (``sparkrun run --rebuild``).  Unlike the default
            best-effort refresh, a failed forced pull aborts distribution —
            see :func:`~sparkrun.containers.registry.ensure_image`.

    Returns:
        List of hostnames (from *hosts*) where distribution failed
        (empty = full success).
    """
    logger.debug("Distributing image '%s' from local to %d host(s)", image, len(hosts))

    # Step 1: ensure image exists locally.  The identity check in step 2 runs
    # *after* this, so a forced pull is reflected in the comparison and hosts
    # that already carry the freshly-pulled image are still skipped.
    rc = ensure_image(image, dry_run=dry_run, force_pull=force_pull)
    if rc != 0:
        logger.error("Failed to ensure local image '%s' — aborting distribution", image)
        return list(hosts)

    if not hosts:
        return []

    xfer = transfer_hosts or hosts

    # Step 2: identity check — skip hosts that already have the correct image.
    # ``_as_identity`` normalizes the result so a caller or mock that still
    # supplies the pre-content-signature ``(id, digests)`` pair is treated as
    # an identity with no content signature -- a real state (older daemons),
    # not an error -- and falls back to the ID/digest signals.
    local = _as_identity(get_image_identity(image) if not dry_run else ImageIdentity.empty())

    needs_transfer = _filter_hosts_needing_image(
        image,
        xfer,
        local.image_id,
        local_repo_digests=local.repo_digests,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        dry_run=dry_run,
        local_content_sig=local.content_sig,
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
    force_pull: bool = False,
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
        force_pull: Re-pull the image on the head even when a copy is already
            present (``sparkrun run --rebuild``).

    Returns:
        List of hostnames where distribution failed (empty = full success).
    """
    from sparkrun.orchestration.distribution import _distribute_from_head

    if not hosts:
        return []

    head = hosts[0]
    logger.debug("Distributing image '%s' from head (%s) to %d host(s)", image, head, len(hosts))

    # Pre-check image status on all hosts to avoid unnecessary work.
    #
    # Skipped entirely under force_pull: this check runs *before* the head
    # pulls, so its verdict describes the image being replaced.  Honoring it
    # would let "every host already agrees" short-circuit the very pull
    # --rebuild was passed to force.  The cost is that workers are all re-synced
    # rather than filtered — correct for an explicit override, since the head's
    # post-pull identity isn't known here without a second round trip.
    if not dry_run and not force_pull:
        remote_identities = _check_remote_image_identities(
            image,
            hosts,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
        )
        ref = remote_identities.get(head)
        if ref is not None:
            ref = _as_identity(ref)
            # Head already has the image — check which workers need it.
            # The head is the transfer source, so its content signature is
            # the reference every worker is compared against.  Without it a
            # save-loaded worker on a different storage driver could never
            # match, and the head would re-stream the image on every launch.
            needs_transfer = []
            for h in hosts:
                remote = _as_identity(remote_identities.get(h, ImageIdentity.empty()))
                if not _images_match(
                    ref.image_id,
                    ref.repo_digests,
                    remote.image_id,
                    remote.repo_digests,
                    ref.content_sig,
                    remote.content_sig,
                ):
                    needs_transfer.append(h)
            if not needs_transfer:
                logger.log(PROGRESS, "  Container image up-to-date on all %d host(s)", len(hosts))
                return []

            if len(hosts) > 1:
                logger.log(PROGRESS, "  Container image needs sync on %d of %d host(s)", len(needs_transfer), len(hosts))

            # Filter workers list and corresponding transfer hosts
            workers = hosts[1:]
            wt = worker_transfer_hosts or workers
            # strict=False: ``wt`` is the caller's worker_transfer_hosts (IB IPs) when
            # supplied.  A short list leaves the trailing workers unsynced rather than
            # aborting the sync for the workers that *are* addressable.
            filtered = [(w, t) for w, t in zip(workers, wt, strict=False) if w in needs_transfer]
            if filtered:
                hosts = [head] + [w for w, _ in filtered]
                worker_transfer_hosts = [t for _, t in filtered]
            else:
                # Only head needed the image (rare: head stale, workers current)
                # Fall through — ensure script will pull on head
                hosts = [head]
                worker_transfer_hosts = None

    # Build ensure script (pull image on head)
    ensure_script = read_script("image_sync.sh").format(image=quote(image), force_pull="1" if force_pull else "0")

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
        max_parallel=HEAD_DISTRIBUTE_MAX_PARALLEL,
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
