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


def _distribute_image_push(
        image: str,
        hosts: list[str],
        worker_transfer_hosts: list[str] | None,
        ssh_kwargs: dict,
        dry_run: bool,
) -> list[str]:
    """Push-mode image distribution: local → head, then head → workers via IB.

    1. Push image from local to head (``hosts[0]``) over management network.
    2. If workers exist, run ``image_distribute.sh`` on head to stream to
       workers over IB.

    Returns:
        List of hostnames where distribution failed.
    """
    from sparkrun.containers.distribute import distribute_image_from_local
    from sparkrun.containers.distribute import distribute_image_from_head

    head = hosts[0]

    # Step 1: push to head only (no transfer_hosts — use management network)
    head_failed = distribute_image_from_local(
        image, [head], transfer_hosts=None,
        dry_run=dry_run, **ssh_kwargs,
    )
    if head_failed:
        logger.error("Push mode: failed to push image to head %s", head)
        return list(hosts)

    # Step 2: if workers, distribute from head to workers
    if len(hosts) > 1:
        worker_failed = distribute_image_from_head(
            image, hosts,
            worker_transfer_hosts=worker_transfer_hosts,
            dry_run=dry_run, **ssh_kwargs,
        )
        return worker_failed

    return []


def _distribute_model_push(
        model: str,
        hosts: list[str],
        cache_dir: str,
        worker_transfer_hosts: list[str] | None,
        ssh_kwargs: dict,
        model_revision: str | None = None,
        dry_run: bool = False,
) -> list[str]:
    """Push-mode model distribution: local → head, then head → workers via IB.

    1. Download model locally and rsync to head over management network.
    2. If workers exist, run ``model_distribute.sh`` on head to rsync to
       workers over IB.

    Returns:
        List of hostnames where distribution failed.
    """
    from sparkrun.models.distribute import distribute_model_from_local
    from sparkrun.models.distribute import distribute_model_from_head

    head = hosts[0]

    # Step 1: push to head only (no transfer_hosts — use management network)
    head_failed = distribute_model_from_local(
        model, [head], cache_dir=cache_dir,
        revision=model_revision, transfer_hosts=None,
        dry_run=dry_run, **ssh_kwargs,
    )
    if head_failed:
        logger.error("Push mode: failed to push model to head %s", head)
        return list(hosts)

    # Step 2: if workers, distribute from head to workers
    if len(hosts) > 1:
        worker_failed = distribute_model_from_head(
            model, hosts, cache_dir=cache_dir,
            revision=model_revision,
            worker_transfer_hosts=worker_transfer_hosts,
            dry_run=dry_run, **ssh_kwargs,
        )
        return worker_failed

    return []


def distribute_resources(
        image: str,
        model: str,
        host_list: list[str],
        cache_dir: str,
        config: SparkrunConfig,
        dry_run: bool,
        model_revision: str | None = None,
        recipe_name: str = "",
        transfer_mode: str = "auto",
) -> tuple[dict[str, str] | None, dict[str, str], dict[str, str]]:
    """Detect IB, distribute container image and model to target hosts.

    Performs InfiniBand detection (for both NCCL env and IB transfer IPs),
    then distributes the container image and model using the strategy
    determined by *transfer_mode*.

    Transfer modes:
        - ``auto`` (default): Auto-detect based on IB connectivity.
          Resolves to ``local`` when the control node can reach cluster
          IB IPs, otherwise falls back to ``push``.
        - ``local``: Control node distributes directly to all hosts.
          Uses IB network for transfers when reachable from the control node.
        - ``push``: Control node pushes to head over management network,
          then head distributes to workers over IB.
        - ``delegated``: Head node downloads resources directly and
          distributes to workers over IB.  Control node downloads nothing.

    For localhost targets, only ensures the image/model exist locally
    (regardless of transfer mode).

    Args:
        image: Container image reference.
        model: HuggingFace model identifier (may be empty).
        host_list: Target hostnames/IPs.
        cache_dir: HuggingFace cache directory.
        config: SparkrunConfig instance.
        dry_run: Show what would be done without executing.
        model_revision: Optional HuggingFace model revision to pin.
        recipe_name: Recipe name for pending-op lock display.
        transfer_mode: Distribution strategy (``"local"``, ``"push"``, or
            ``"delegated"``).

    Returns:
        Tuple of (nccl_env, ib_ip_map, mgmt_ip_map).  ``nccl_env`` is
        ``None`` when IB detection was skipped or not applicable.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts, validate_ib_connectivity
    from sparkrun.containers.distribute import distribute_image_from_local, distribute_image_from_head
    from sparkrun.containers.registry import ensure_image
    from sparkrun.models.distribute import distribute_model_from_local, distribute_model_from_head
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

    if len(host_list) <= 1 and is_local_host(host_list[0]):
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
    worker_transfer_hosts: list[str] | None = None

    # Step 1: Detect InfiniBand for NCCL env + transfer routing
    ib_result = detect_ib_for_hosts(
        host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
    )
    nccl_env = ib_result.nccl_env
    mgmt_ip_map = ib_result.mgmt_ip_map

    # Auto-detect or validate IB connectivity
    _ib_validated: dict[str, str] | None = None
    if transfer_mode in ("auto", "local"):
        _ib_validated = validate_ib_connectivity(
            ib_result.ib_ip_map, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )

    if transfer_mode == "auto":
        if _ib_validated:
            transfer_mode = "local"
            logger.info("Auto-detected transfer mode: local (IB reachable from control node)")
        else:
            transfer_mode = "push"
            logger.info("Auto-detected transfer mode: push (no IB connectivity from control node)")

    if transfer_mode == "local":
        # Local mode: use validated IB IPs for direct transfers
        ib_ip_map = _ib_validated or {}
        if ib_ip_map:
            transfer_hosts = [
                ib_result.ib_ip_map.get(h, h) for h in host_list
            ]
            logger.info(
                "Using IB network for transfers (%d/%d hosts)",
                len(ib_result.ib_ip_map), len(host_list),
            )
    else:
        # Push/delegated: control is external, skip IB validation for
        # control→host transfers.  Use IB IPs only for head→worker transfers.
        if len(host_list) > 1 and ib_result.ib_ip_map:
            worker_transfer_hosts = [
                ib_result.ib_ip_map.get(h, h) for h in host_list[1:]
            ]
            logger.info(
                "External control node: using IB for head→worker transfers (%d workers)",
                len(host_list) - 1,
            )
        # Populate ib_ip_map for runtime use (NCCL needs IB IPs regardless)
        ib_ip_map = ib_result.ib_ip_map

    # Step 2: Distribute container image
    with pending_op(_lock_id, "image_distribute", **_pop_kw):
        if transfer_mode == "local":
            img_failed = distribute_image_from_local(
                image, host_list,
                transfer_hosts=transfer_hosts,
                dry_run=dry_run, **ssh_kwargs,
            )
        elif transfer_mode == "push":
            img_failed = _distribute_image_push(
                image, host_list,
                worker_transfer_hosts=worker_transfer_hosts,
                ssh_kwargs=ssh_kwargs, dry_run=dry_run,
            )
        elif transfer_mode == "delegated":
            img_failed = distribute_image_from_head(
                image, host_list,
                worker_transfer_hosts=worker_transfer_hosts,
                dry_run=dry_run, **ssh_kwargs,
            )
        else:
            logger.warning("Unknown transfer_mode '%s', falling back to local", transfer_mode)
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
            if transfer_mode == "local":
                mdl_failed = distribute_model_from_local(
                    model, host_list,
                    cache_dir=cache_dir,
                    revision=model_revision,
                    transfer_hosts=transfer_hosts,
                    dry_run=dry_run, **ssh_kwargs,
                )
            elif transfer_mode == "push":
                mdl_failed = _distribute_model_push(
                    model, host_list,
                    cache_dir=cache_dir,
                    worker_transfer_hosts=worker_transfer_hosts,
                    ssh_kwargs=ssh_kwargs,
                    model_revision=model_revision,
                    dry_run=dry_run,
                )
            elif transfer_mode == "delegated":
                mdl_failed = distribute_model_from_head(
                    model, host_list,
                    cache_dir=cache_dir,
                    revision=model_revision,
                    worker_transfer_hosts=worker_transfer_hosts,
                    dry_run=dry_run, **ssh_kwargs,
                )
            else:
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
