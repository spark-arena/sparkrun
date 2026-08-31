"""Model distribution via local-to-remote transfer.

Instead of having every host download from HuggingFace Hub independently,
these functions download once (locally or on the head node) and then
rsync the cache directory to targets.
"""

from __future__ import annotations

import logging

from sparkrun.core.config import resolve_hf_cache_home
from sparkrun.models.download import download_model, model_cache_path
from sparkrun.orchestration.transfer import (
    TransferFailure,
    map_transfer_failures_detailed,
)
from sparkrun.orchestration.ssh import (
    HEAD_DISTRIBUTE_MAX_PARALLEL,
    NFS_SAFE_ATTR_OPTS,
    build_ssh_opts_string,
    run_remote_scripts_parallel,
    run_rsync_parallel,
)
from sparkrun.orchestration.sudo import ensure_remote_dir_ownership
from sparkrun.scripts import read_script

from sparkrun.core.progress import PROGRESS

logger = logging.getLogger(__name__)

# Generous execution timeout (seconds) for control→host model rsync.
# build_ssh_cmd only sets ConnectTimeout (TCP connect); without an overall
# timeout a host that connects then stalls (frozen NFS, hung rsync) blocks
# the whole as_completed loop forever.  Model caches can be tens of GB and
# the first sync to a fresh host is the slowest, so this is intentionally
# large — it exists to break true hangs, not to bound healthy transfers.
DEFAULT_MODEL_RSYNC_TIMEOUT = 2 * 60 * 60  # 2 hours

# Filesystem types that indicate a network-shared cache directory.  Used by
# the setup wizard's shared-cache detection (see ``detect_shared_cache``).
_SHARED_CACHE_FSTYPES = {"nfs", "nfs4", "cifs", "smb3"}


def _model_rsync_options(preserve_perms: bool) -> list[str]:
    """Return the rsync flag list for a model-cache transfer.

    The HF cache is content-addressed (``blobs/<sha256>``), so transfers only
    need contents + symlinks; ``--size-only`` skips already-synced shards
    instantly.  When *preserve_perms* is ``True`` we keep ``-a`` (archive,
    historical default) minus :data:`~sparkrun.orchestration.ssh.NFS_SAFE_ATTR_OPTS`,
    which is what makes the default work on a shared/NFS cache without
    configuration.  When ``False`` we use ``-r --links``, additionally dropping
    file times — the harder relaxation, for destinations where even that EPERMs.
    """
    if preserve_perms:
        return ["-a", "--size-only", "--mkpath", "--partial", "--links", *NFS_SAFE_ATTR_OPTS]
    return ["-r", "--links", "--size-only", "--mkpath", "--partial"]


def detect_shared_cache(
    cache_dir: str | None,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> bool:
    """Return True when the HF cache is on a shared filesystem on every host.

    SSH-probes each host with ``findmnt -T <cache_dir> -n -o FSTYPE`` and
    returns ``True`` only when all hosts report a network-shared type
    (``nfs``/``nfs4``/``cifs``/``smb3``).  When *cache_dir* is falsy, each
    host's resolved HF cache home (``${HF_HOME:-$HOME/.cache/huggingface}``)
    is probed instead.  Used interactively by the setup wizard to suggest
    enabling shared-cache distribution preferences; it is intentionally
    **not** wired into the hot launch path.
    """
    if not hosts or dry_run:
        return False

    from sparkrun.utils.shell import quote

    if cache_dir:
        cache_dir_line = "CACHE_DIR={cache_dir}\n".format(cache_dir=quote(cache_dir))
    else:
        cache_dir_line = 'CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"\n'

    script = (
        "set -uo pipefail\n"
        + cache_dir_line
        + 'FSTYPE=$(findmnt -T "$CACHE_DIR" -n -o FSTYPE 2>/dev/null | tail -n1 | tr -d "[:space:]")\n'
        + 'echo "FSTYPE=${FSTYPE}"\n'
    )

    results = run_remote_scripts_parallel(
        hosts,
        script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=15,
        dry_run=dry_run,
        quiet=True,
    )

    if len(results) != len(hosts):
        return False
    for r in results:
        if not r.success:
            return False
        fstype = ""
        for line in r.stdout.splitlines():
            if line.startswith("FSTYPE="):
                fstype = line.split("=", 1)[1].strip()
        if fstype not in _SHARED_CACHE_FSTYPES:
            return False
    return True


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

    Thin wrapper over
    :func:`~sparkrun.orchestration.sudo.ensure_remote_dir_ownership`, which the
    tuning cache shares: the failure is a property of a sparkrun-managed cache
    directory, not of what happens to be stored in it.
    """
    ensure_remote_dir_ownership(
        cache_dir,
        hosts,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        dry_run=dry_run,
        resource_label="model cache",
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
    preserve_perms: bool = True,
    skip_fan_out: bool = False,
) -> list[TransferFailure]:
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
        preserve_perms: When ``False``, rsync drops owner/group/perm
            preservation (``-r --links`` instead of ``-a``) and the
            best-effort cache chown is skipped — for shared/NFS caches.
        skip_fan_out: When ``True``, the model is downloaded locally but the
            per-host rsync is skipped entirely (the cache is shared across
            all hosts, so a single download makes it visible everywhere).

    Returns:
        List of :class:`TransferFailure` records, one per host that
        failed (empty = full success).  Each carries a classified
        ``reason`` (e.g. "out of disk space on destination") so callers
        can surface meaningful errors instead of bare host names.
    """
    local_cache = resolve_hf_cache_home(local_cache_dir or cache_dir)
    remote_cache = resolve_hf_cache_home(cache_dir)
    logger.debug("Distributing model '%s' from local to %d host(s)", model_id, len(hosts))

    # Step 1: download model locally
    rc = download_model(model_id, cache_dir=local_cache, token=token, revision=revision, dry_run=dry_run)
    if rc != 0:
        logger.error("Failed to download model '%s' locally — aborting distribution", model_id)
        return [TransferFailure(host=h, reason="local model download failed") for h in hosts]

    if not hosts:
        return []

    # Shared-cache fast path: the model was just downloaded into a cache that
    # every host already mounts, so the per-host rsync would be redundant
    # (and, on NFS under root_squash, would fail trying to chgrp).  Skip it.
    if skip_fan_out:
        logger.log(PROGRESS, "  Shared cache: model downloaded once; skipping per-host rsync")
        return []

    xfer = transfer_hosts or hosts

    # Step 2: best-effort fix of remote cache ownership before rsync.  Skipped
    # when not preserving perms — on a shared/NFS cache the chown can't succeed
    # and rsync (without -o/-g) doesn't need it.
    if preserve_perms:
        _try_fix_remote_permissions(
            remote_cache,
            hosts,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            dry_run=dry_run,
        )

    # Step 3: rsync model cache to all hosts in parallel.
    # HF cache uses content-addressed blob filenames (blobs/<sha256>),
    # so identical names always mean identical content.  --size-only is
    # both correct and lets rsync skip already-synced shards instantly
    # without reading them.  Quantized weights (NVFP4/safetensors) don't
    # compress, so -z is wasted CPU and is omitted here.
    local_model_path = model_cache_path(model_id, local_cache)
    remote_model_path = model_cache_path(model_id, remote_cache)
    results = run_rsync_parallel(
        local_model_path,
        xfer,
        remote_model_path,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        rsync_options=_model_rsync_options(preserve_perms),
        timeout=timeout if timeout is not None else DEFAULT_MODEL_RSYNC_TIMEOUT,
        dry_run=dry_run,
    )

    # Map transfer IPs back to management hosts for failure reporting,
    # and classify each failure (out of disk space, permission denied, …)
    # so the caller can surface a meaningful error instead of a bare host
    # name.
    failures = map_transfer_failures_detailed(results, xfer, hosts)
    if not failures:
        logger.log(PROGRESS, "  Model synced to %d host(s)", len(hosts))

    return failures


def _build_model_ensure_script(
    model_id: str,
    cache: str,
    revision: str | None = None,
    hf_token: str | None = None,
) -> str:
    """Build the bash script that ensures *model_id* is present in *cache*.

    Shared by the head-download path and the per-node ``pull`` path so both
    fetch identically — a second copy of this would be free to drift on GGUF
    handling or token injection, and the drift would only show on gated or
    quant-selected models.
    """
    from sparkrun.models.download import is_gguf_model, parse_gguf_model_spec
    from sparkrun.utils.shell import quote

    revision_flag = "--revision %s " % revision if revision else ""
    if is_gguf_model(model_id):
        repo_id, quant = parse_gguf_model_spec(model_id)
        script = read_script("model_sync_gguf.sh").format(
            repo_id=repo_id,
            quant=quant or "",
            cache=cache,
            revision_flag=revision_flag,
        )
    else:
        script = read_script("model_sync.sh").format(
            model_id=model_id,
            cache=cache,
            revision_flag=revision_flag,
        )

    # Inject HF token for gated models
    if hf_token:
        script = 'export HF_TOKEN="' + str(quote(hf_token)) + '";\n' + script
    return script


def distribute_model_per_node(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
    hf_token: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    dry_run: bool = False,
) -> list[TransferFailure]:
    """Have every host download the model itself, in parallel (``pull`` mode).

    No head node downloads on anyone's behalf and nothing crosses the control
    machine.  Faster than head-then-rsync when egress is good, at the cost of
    N× bandwidth and an HF token that must be reachable on every node.

    Callers must not use this when the cache is shared across hosts — see the
    ``skip_fan_out`` guard in ``_distribute_single_model``.
    """
    from sparkrun.orchestration.primitives import sync_resource_to_hosts

    if not hosts:
        return []

    cache = resolve_hf_cache_home(cache_dir)
    script = _build_model_ensure_script(model_id, cache, revision=revision, hf_token=hf_token)

    logger.log(PROGRESS, "  Downloading model '%s' on %d host(s) in parallel", model_id, len(hosts))
    failed_hosts = sync_resource_to_hosts(
        script,
        hosts,
        "Model '%s'" % model_id,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )
    return [TransferFailure(host=h, reason="model download failed on host (see log above)") for h in failed_hosts]


def distribute_model_from_head(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
    hf_token: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    worker_transfer_hosts: list[str] | None = None,
    preserve_perms: bool = True,
    skip_fan_out: bool = False,
) -> list[TransferFailure]:
    """Download a model on the head node then distribute to remaining hosts.

    1. Download on ``hosts[0]`` using the existing ``model_sync.sh``.
    2. If there is only one host, done.
    3. Run ``model_distribute.sh`` on ``hosts[0]`` to rsync to ``hosts[1:]``.

    Args:
        model_id: HuggingFace model identifier.
        hosts: Cluster hostnames (``hosts[0]`` is the head).
        cache_dir: Override for the HuggingFace cache directory.
        revision: Optional revision (branch, tag, or commit hash).
        hf_token: Optional HuggingFace API token for gated models.
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

    cache = resolve_hf_cache_home(cache_dir)
    head = hosts[0]
    logger.debug("Distributing model '%s' from head (%s) to %d host(s)", model_id, head, len(hosts))

    ensure_script = _build_model_ensure_script(model_id, cache, revision=revision, hf_token=hf_token)

    # Shared-cache fast path: download once on the head, skip the head→worker
    # rsync entirely (workers already mount the same cache).  Running
    # _distribute_from_head with a single-host list ensures on the head then
    # returns without sending the distribute script.
    if skip_fan_out:
        logger.log(PROGRESS, "  Shared cache: model ensured on head; skipping head→worker rsync")
        failed_hosts = _distribute_from_head(
            head=head,
            hosts=[head],
            ensure_script=ensure_script,
            distribute_script="",
            resource_label="Model '%s'" % model_id,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ssh_options=ssh_options,
            timeout=timeout,
            dry_run=dry_run,
        )
        return [TransferFailure(host=h, reason="model download on head failed (see log above)") for h in failed_hosts]

    # Build distribute script (rsync from head to workers).  The head→worker hop
    # lands on the same kind of destination as the control→host one, so it takes
    # the same NFS-safe attribute relaxation; when perms are not preserved we use
    # ``-r --links`` instead of ``-a``, additionally dropping file times.
    targets = worker_transfer_hosts or hosts[1:]
    model_path = model_cache_path(model_id, cache)
    ssh_opts = build_ssh_opts_string(
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
    )
    rsync_attr_flags = " ".join(["-a", *NFS_SAFE_ATTR_OPTS]) if preserve_perms else "-r --links"
    dist_script = read_script("model_distribute.sh").format(
        model_path=model_path,
        targets=" ".join(targets),
        ssh_opts=ssh_opts,
        ssh_user=ssh_user or "",
        max_parallel=HEAD_DISTRIBUTE_MAX_PARALLEL,
        rsync_attr_flags=rsync_attr_flags,
    )

    failed_hosts = _distribute_from_head(
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
    # _distribute_from_head doesn't expose per-host rsync stderr, so we
    # can't classify the cause here.  Surface a generic reason and let
    # users consult the head's rsync log (already echoed above) for
    # detail like "No space left on device".
    return [TransferFailure(host=h, reason="rsync from head failed (see log above for stderr)") for h in failed_hosts]
