"""Resource distribution: IB detection, container image and model syncing."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from sparkrun.core.hosts import is_local_host

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)


def _distribute_from_head(
        head: str,
        hosts: list[str],
        ensure_script: str,
        distribute_script: str,
        resource_label: str,
        ssh_user: str | None = None,
        ssh_key: str | None = None,
        ssh_options: list[str] | None = None,
        timeout: int | None = None,
        dry_run: bool = False,
) -> list[str]:
    """Shared head-to-workers distribution pattern.

    1. Run *ensure_script* on head to ensure resource is present.
    2. If single host, return (done).
    3. Run *distribute_script* on head to stream to remaining hosts.

    Args:
        head: Head hostname (``hosts[0]``).
        hosts: Full cluster host list (head + workers).
        ensure_script: Bash script that ensures the resource exists on head.
        distribute_script: Bash script that distributes from head to workers.
        resource_label: Human-readable label for log messages (e.g. "Model", "Image").
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Per-operation timeout in seconds.
        dry_run: If True, show what would be done without executing.

    Returns:
        List of hostnames where distribution failed (empty = full success).
    """
    from sparkrun.orchestration.ssh import run_remote_script

    # Step 1: ensure resource on head
    ensure_result = run_remote_script(
        head, ensure_script,
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        timeout=timeout, dry_run=dry_run,
    )
    if not ensure_result.success:
        logger.error("Failed to ensure %s on head %s", resource_label, head)
        return list(hosts)

    # Step 2: if single host, we're done
    if len(hosts) == 1:
        logger.info("Single host — %s ready", resource_label)
        return []

    # Step 3: distribute from head to remaining hosts
    dist_result = run_remote_script(
        head, distribute_script,
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        timeout=timeout, dry_run=dry_run,
    )

    if dist_result.success:
        logger.info("%s distributed from head to all targets", resource_label)
        return []

    # Report failure using management hostnames
    logger.warning("%s distribution from head failed (rc=%d)",
                   resource_label, dist_result.returncode)
    return list(hosts[1:])


def distribute_resources(
        image: str,
        model: str,
        host_list: list[str],
        cache_dir: str,
        config: SparkrunConfig,
        dry_run: bool,
        model_revision: str | None = None,
        recipe_name: str = "",
) -> tuple[dict[str, str] | None, dict[str, str], dict[str, str]]:
    """Detect IB, distribute container image and model to target hosts.

    Performs InfiniBand detection (for both NCCL env and IB transfer IPs),
    then distributes the container image and model from local to all
    remote hosts using the fast IB network when available.

    For localhost targets, only ensures the image/model exist locally.

    Args:
        image: Container image reference.
        model: HuggingFace model identifier (may be empty).
        host_list: Target hostnames/IPs.
        cache_dir: HuggingFace cache directory.
        config: SparkrunConfig instance.
        dry_run: Show what would be done without executing.
        model_revision: Optional HuggingFace model revision to pin.
        recipe_name: Recipe name for pending-op lock display.

    Returns:
        Tuple of (nccl_env, ib_ip_map, mgmt_ip_map).  ``nccl_env`` is
        ``None`` when IB detection was skipped or not applicable.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts, validate_ib_connectivity
    from sparkrun.containers.distribute import distribute_image_from_local
    from sparkrun.containers.registry import ensure_image
    from sparkrun.models.distribute import distribute_model_from_local
    from sparkrun.models.download import download_model
    from sparkrun.core.pending_ops import pending_op

    # Common kwargs for pending-op lock files
    _pop_kw = dict(
        recipe=recipe_name,
        model=model, image=image,
        hosts=host_list, cache_dir=str(config.cache_dir),
    )
    # Derive a cluster_id-ish key for the lock files.  The real cluster_id
    # is generated earlier in run(); we receive the image+model+hosts here
    # so we hash the same inputs to keep the lock name stable.
    _lock_key = hashlib.sha256(
        f"{image}|{model}|{','.join(host_list)}".encode()
    ).hexdigest()[:12]
    _lock_id = f"sparkrun_{_lock_key}"

    if is_local := (len(host_list) <= 1 and is_local_host(host_list[0])):
        # Local-only: just ensure image and model exist, no SSH needed
        with pending_op(_lock_id, "image_pull", **_pop_kw):
            logger.info("Ensuring container image is available locally...")
            ensure_image(image, dry_run=dry_run)
        if model:
            with pending_op(_lock_id, "model_download", **_pop_kw):
                logger.info("Ensuring model %s is available locally...", model)
                download_model(model, cache_dir=cache_dir, revision=model_revision, dry_run=dry_run)
        return None, {}, {}  # let runtime handle its own local IB detection

    ssh_kwargs = build_ssh_kwargs(config)
    nccl_env: dict[str, str] = {}
    ib_ip_map: dict[str, str] = {}
    mgmt_ip_map: dict[str, str] = {}
    transfer_hosts: list[str] | None = None

    # Step 1: Detect InfiniBand for NCCL env + transfer routing
    ib_result = detect_ib_for_hosts(
        host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
    )
    nccl_env = ib_result.nccl_env
    mgmt_ip_map = ib_result.mgmt_ip_map

    # Validate that the control machine can actually reach IB IPs
    # before using them for transfers.  Detection runs on the remote
    # hosts, but the control machine may not be on the IB network.
    ib_ip_map = validate_ib_connectivity(
        ib_result.ib_ip_map, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
    )

    if ib_ip_map:
        transfer_hosts = [
            ib_result.ib_ip_map.get(h, h) for h in host_list
        ]
        logger.info(
            "Using IB network for transfers (%d/%d hosts)",
            len(ib_result.ib_ip_map), len(host_list),
        )

    # Step 2: Distribute container image
    with pending_op(_lock_id, "image_distribute", **_pop_kw):
        img_failed = distribute_image_from_local(
            image, host_list,
            transfer_hosts=transfer_hosts,
            dry_run=dry_run, **ssh_kwargs,
        )
    if img_failed:
        logger.warning("Image distribution failed on: %s", ", ".join(img_failed))

    # Step 3: Distribute model
    if model:
        with pending_op(_lock_id, "model_download", **_pop_kw):
            mdl_failed = distribute_model_from_local(
                model, host_list,
                cache_dir=cache_dir,
                revision=model_revision,
                transfer_hosts=transfer_hosts,
                dry_run=dry_run, **ssh_kwargs,
            )
        if mdl_failed:
            logger.warning("Model distribution failed on: %s", ", ".join(mdl_failed))

    logger.info("Distribution complete.")
    return nccl_env, ib_ip_map, mgmt_ip_map
