"""Resource distribution: IB detection, container image and model syncing."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sparkrun.core.config import resolve_hf_token as _get_hf_token
from sparkrun.core.hosts import is_control_in_cluster
from sparkrun.utils import is_local_host

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.orchestration.infiniband import IBDetectionResult
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


class DistributionError(Exception):
    """Raised when resource distribution (image or model sync) fails."""


@dataclass
class TransferModeResult:
    """Result of :func:`resolve_auto_transfer_mode`.

    Always contains a concrete *mode* (never ``"auto"``).  When IB
    detection was performed during resolution, *ib_result* and
    *ib_validated* are populated so ``distribute_resources()`` can
    skip redundant detection.
    """

    mode: str
    ib_result: IBDetectionResult | None = None
    ib_validated: dict[str, str] | None = None
    auto_delegated: bool = False
    """True when auto resolved to delegated (enables push fallback)."""


def _is_cross_user(ssh_kwargs: dict | None) -> bool:
    """True when *ssh_kwargs* specifies a user different from the OS user."""
    import os

    ssh_user = ssh_kwargs.get("ssh_user") if ssh_kwargs else None
    if ssh_user is None:
        return False
    return ssh_user != os.environ.get("USER", "root")


def _has_local_ib() -> bool:
    """True when the control machine has InfiniBand interfaces."""
    from pathlib import Path

    ib_dir = Path("/sys/class/infiniband")
    return ib_dir.is_dir() and any(ib_dir.iterdir())


def resolve_auto_transfer_mode(
    transfer_mode: str,
    host_list: list[str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    topology: str | None = None,
) -> TransferModeResult:
    """Resolve ``"auto"`` transfer mode to a concrete strategy.

    Call this early (before builder and distribution phases) so
    downstream consumers always receive a concrete mode — ``"auto"``
    is never returned.

    Explicit modes (``"local"``, ``"push"``, ``"delegated"``) are
    returned unchanged.

    When the control machine is external with the same SSH user and
    has local InfiniBand, IB detection and connectivity validation
    are performed here.  The results are stored in the returned
    :class:`TransferModeResult` so ``distribute_resources()`` can
    reuse them without redundant remote calls.
    """
    if transfer_mode != "auto":
        return TransferModeResult(mode=transfer_mode)

    _cross_user = _is_cross_user(ssh_kwargs)
    _in_cluster = is_control_in_cluster(host_list)

    if _in_cluster and not _cross_user:
        logger.info("Auto-detected transfer mode: local (control is cluster member)")
        return TransferModeResult(mode="local")

    if _cross_user:
        logger.info(
            "Auto-detected transfer mode: delegated (cluster user '%s' differs from OS user)",
            ssh_kwargs.get("ssh_user") if ssh_kwargs else None,
        )
        return TransferModeResult(mode="delegated")

    # External control + same user: check if local machine has IB.
    # If no local IB, control can never reach cluster IB → delegated.
    if not _has_local_ib():
        logger.info("Auto-detected transfer mode: delegated (external control, no local IB)")
        return TransferModeResult(mode="delegated")

    # Local IB exists — run IB detection + connectivity validation to
    # resolve definitively and cache results for distribute_resources().
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts, validate_ib_connectivity

    ib_result = detect_ib_for_hosts(host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run, topology=topology)
    ib_validated = validate_ib_connectivity(ib_result.ib_ip_map, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    if ib_validated:
        logger.info("Auto-detected transfer mode: local (external control, IB reachable)")
        return TransferModeResult(mode="local", ib_result=ib_result, ib_validated=ib_validated)

    logger.info("Auto-detected transfer mode: delegated (external control, no IB connectivity)")
    return TransferModeResult(mode="delegated", ib_result=ib_result, ib_validated={}, auto_delegated=True)


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
        head,
        ensure_script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
        dry_run=dry_run,
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
        head,
        distribute_script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
        dry_run=dry_run,
    )

    if dist_result.success:
        from sparkrun.core.progress import PROGRESS

        logger.log(PROGRESS, "  %s synced from head to %d worker(s)", resource_label, len(hosts) - 1)
        return []

    # Report failure using management hostnames
    logger.warning("%s distribution from head failed (rc=%d)", resource_label, dist_result.returncode)
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
        image,
        [head],
        transfer_hosts=None,
        dry_run=dry_run,
        **ssh_kwargs,
    )
    if head_failed:
        logger.error("Push mode: failed to push image to head %s", head)
        return list(hosts)

    # Step 2: if workers, distribute from head to workers
    if len(hosts) > 1:
        worker_failed = distribute_image_from_head(
            image,
            hosts,
            worker_transfer_hosts=worker_transfer_hosts,
            dry_run=dry_run,
            **ssh_kwargs,
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
    hf_token: str | None = None,
    dry_run: bool = False,
    local_cache_dir: str | None = None,
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
        model,
        [head],
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        token=hf_token,
        revision=model_revision,
        transfer_hosts=None,
        dry_run=dry_run,
        **ssh_kwargs,
    )
    if head_failed:
        logger.error("Push mode: failed to push model to head %s", head)
        return list(hosts)

    # Step 2: if workers, distribute from head to workers
    if len(hosts) > 1:
        worker_failed = distribute_model_from_head(
            model,
            hosts,
            cache_dir=cache_dir,
            revision=model_revision,
            hf_token=hf_token,
            worker_transfer_hosts=worker_transfer_hosts,
            dry_run=dry_run,
            **ssh_kwargs,
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
    transfer_interface: str | None = None,
    local_cache_dir: str | None = None,
    pre_ib: TransferModeResult | None = None,
    topology: str | None = None,
) -> tuple["ClusterCommEnv | None", dict[str, str], dict[str, str]]:
    """Detect IB, distribute container image and model to target hosts.

    Performs InfiniBand detection (for both NCCL env and IB transfer IPs),
    then distributes the container image and model using the strategy
    determined by *transfer_mode*.

    Transfer modes:
        - ``auto`` (default): Auto-detect based on cluster membership
          and IB connectivity.  Resolves to ``local`` when the control
          node is a cluster member or can reach cluster IB IPs,
          otherwise falls back to ``delegated`` (with ``push`` as
          secondary fallback on failure).
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
        transfer_interface: Network interface for transfers.  ``"cx7"``
            (default) uses IB IPs when available; ``"mgmt"`` forces
            management IPs regardless of IB availability.
        local_cache_dir: Control-machine cache dir for model downloads.
            Defaults to *cache_dir* when not provided.

    Returns:
        Tuple of (comm_env, ib_ip_map, mgmt_ip_map).  ``comm_env`` is
        a :class:`ClusterCommEnv` carrying both cluster-wide and
        per-host inter-node comm env vars (``None`` when IB detection
        was skipped or not applicable).
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
        model=model,
        image=image,
        hosts=host_list,
        cache_dir=str(config.cache_dir),
    )
    # Derive a cluster_id-ish key for the lock files.  The real cluster_id
    # is generated earlier in run(); we receive the image+model+hosts here
    # so we hash the same inputs to keep the lock name stable.
    _lock_key = hashlib.sha256(f"{image}|{model}|{','.join(host_list)}".encode()).hexdigest()[:12]
    _lock_id = f"sparkrun_{_lock_key}"

    effective_local_cache = local_cache_dir or cache_dir

    ssh_kwargs = build_ssh_kwargs(config)

    # Get HF token from control node environment
    hf_token = _get_hf_token()

    if len(host_list) <= 1 and is_local_host(host_list[0]) and not _is_cross_user(ssh_kwargs):
        # Local-only (same user): just ensure image and model exist, no SSH needed
        with pending_op(_lock_id, "image_pull", **_pop_kw):
            logger.info("Ensuring container image is available locally...")
            if ensure_image(image, dry_run=dry_run) != 0:
                raise DistributionError(f"Failed to pull or locate image: {image}")
        if model:
            with pending_op(_lock_id, "model_download", **_pop_kw):
                logger.info("Ensuring model %s is available locally...", model)
                if download_model(model, cache_dir=effective_local_cache, token=hf_token, revision=model_revision, dry_run=dry_run) != 0:
                    raise DistributionError(f"Failed to download model: {model}")
        return None, {}, {}  # let runtime handle its own local IB detection

    from sparkrun.orchestration.comm_env import ClusterCommEnv

    comm_env: ClusterCommEnv = ClusterCommEnv.empty()
    ib_ip_map: dict[str, str] = {}
    mgmt_ip_map: dict[str, str] = {}
    transfer_hosts: list[str] | None = None
    worker_transfer_hosts: list[str] | None = None

    _cross_user = _is_cross_user(ssh_kwargs)

    # Step 1: Detect InfiniBand for NCCL env + transfer routing
    # Reuse pre-computed results from resolve_auto_transfer_mode() when available.
    if pre_ib is not None and pre_ib.ib_result is not None:
        ib_result = pre_ib.ib_result
        _ib_validated = pre_ib.ib_validated
        _auto_delegated = pre_ib.auto_delegated
        logger.debug("Reusing pre-computed IB detection results from resolve_auto_transfer_mode()")
    else:
        ib_result = detect_ib_for_hosts(
            host_list,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
            topology=topology,
        )
        # Validate IB connectivity for auto/local modes
        _ib_validated: dict[str, str] | None = None
        if transfer_mode in ("auto", "local"):
            _ib_validated = validate_ib_connectivity(
                ib_result.ib_ip_map,
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
            )
        _auto_delegated = False
        if transfer_mode == "auto":
            _in_cluster = is_control_in_cluster(host_list)
            if _in_cluster and not _cross_user:
                transfer_mode = "local"
                logger.info("Auto-detected transfer mode: local (control is cluster member)")
            elif _ib_validated and not _cross_user:
                transfer_mode = "local"
                logger.info("Auto-detected transfer mode: local (IB reachable from control node)")
            else:
                transfer_mode = "delegated"
                _auto_delegated = True
                if _cross_user:
                    logger.info(
                        "Auto-detected transfer mode: delegated (cluster user '%s' differs from OS user)",
                        ssh_kwargs.get("ssh_user"),
                    )
                else:
                    logger.info("Auto-detected transfer mode: delegated (external control, no IB connectivity)")

    comm_env = ib_result.comm_env
    mgmt_ip_map = ib_result.mgmt_ip_map

    # Determine effective transfer interface (default: cx7)
    _use_mgmt = transfer_interface == "mgmt"
    if _use_mgmt:
        logger.info("Transfer interface: mgmt — using management IPs for transfers")

    if transfer_mode == "local" and _cross_user:
        logger.warning(
            "transfer_mode='local' but cluster user '%s' differs from OS user — "
            "local Docker operations will run as the current OS user, not the cluster user. "
            "Consider using transfer_mode='delegated' or 'push'.",
            ssh_kwargs.get("ssh_user"),
        )

    if transfer_mode == "local":
        # Local mode: use validated IB IPs for direct transfers
        ib_ip_map = _ib_validated or {}
        if ib_ip_map and not _use_mgmt:
            transfer_hosts = [ib_result.ib_ip_map.get(h, h) for h in host_list]
            logger.info(
                "Using IB network for transfers (%d/%d hosts)",
                len(ib_result.ib_ip_map),
                len(host_list),
            )
    else:
        # Push/delegated: control is external, skip IB validation for
        # control→host transfers.  Use IB IPs only for head→worker transfers.
        if len(host_list) > 1 and ib_result.ib_ip_map and not _use_mgmt:
            worker_transfer_hosts = [ib_result.ib_ip_map.get(h, h) for h in host_list[1:]]
            logger.info(
                "External control node: using IB for head→worker transfers (%d workers)",
                len(host_list) - 1,
            )
        # Populate ib_ip_map for runtime use (NCCL needs IB IPs regardless)
        ib_ip_map = ib_result.ib_ip_map

    # Step 2: Distribute container image
    from sparkrun.core.progress import PROGRESS as _PROGRESS_LEVEL

    logger.info("Distribution mode: %s (image=%s, model=%s, hosts=%d)", transfer_mode, image, model or "(none)", len(host_list))
    logger.log(_PROGRESS_LEVEL, "  Checking container image on %d host(s)", len(host_list))
    with pending_op(_lock_id, "image_distribute", **_pop_kw):
        if transfer_mode == "local":
            img_failed = distribute_image_from_local(
                image,
                host_list,
                transfer_hosts=transfer_hosts,
                dry_run=dry_run,
                **ssh_kwargs,
            )
        elif transfer_mode == "push":
            img_failed = _distribute_image_push(
                image,
                host_list,
                worker_transfer_hosts=worker_transfer_hosts,
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
            )
        elif transfer_mode == "delegated":
            img_failed = distribute_image_from_head(
                image,
                host_list,
                worker_transfer_hosts=worker_transfer_hosts,
                dry_run=dry_run,
                **ssh_kwargs,
            )
            if img_failed and _auto_delegated:
                logger.info("Delegated image distribution failed, falling back to push mode")
                img_failed = _distribute_image_push(
                    image,
                    host_list,
                    worker_transfer_hosts=worker_transfer_hosts,
                    ssh_kwargs=ssh_kwargs,
                    dry_run=dry_run,
                )
        else:
            logger.warning("Unknown transfer_mode '%s', falling back to local", transfer_mode)
            img_failed = distribute_image_from_local(
                image,
                host_list,
                transfer_hosts=transfer_hosts,
                dry_run=dry_run,
                **ssh_kwargs,
            )

    if img_failed:
        raise DistributionError("Image distribution failed on: %s" % ", ".join(img_failed))

    # Step 3: Distribute model
    if model:
        logger.log(_PROGRESS_LEVEL, "  Syncing model to %d host(s)", len(host_list))
        with pending_op(_lock_id, "model_download", **_pop_kw):
            if transfer_mode == "local":
                mdl_failed = distribute_model_from_local(
                    model,
                    host_list,
                    cache_dir=cache_dir,
                    local_cache_dir=effective_local_cache,
                    token=hf_token,
                    revision=model_revision,
                    transfer_hosts=transfer_hosts,
                    dry_run=dry_run,
                    **ssh_kwargs,
                )
            elif transfer_mode == "push":
                mdl_failed = _distribute_model_push(
                    model,
                    host_list,
                    cache_dir=cache_dir,
                    worker_transfer_hosts=worker_transfer_hosts,
                    ssh_kwargs=ssh_kwargs,
                    model_revision=model_revision,
                    hf_token=hf_token,
                    dry_run=dry_run,
                    local_cache_dir=effective_local_cache,
                )
            elif transfer_mode == "delegated":
                mdl_failed = distribute_model_from_head(
                    model,
                    host_list,
                    cache_dir=cache_dir,
                    revision=model_revision,
                    hf_token=hf_token,
                    worker_transfer_hosts=worker_transfer_hosts,
                    dry_run=dry_run,
                    **ssh_kwargs,
                )
                if mdl_failed and _auto_delegated:
                    logger.info("Delegated model distribution failed, falling back to push mode")
                    mdl_failed = _distribute_model_push(
                        model,
                        host_list,
                        cache_dir=cache_dir,
                        worker_transfer_hosts=worker_transfer_hosts,
                        ssh_kwargs=ssh_kwargs,
                        model_revision=model_revision,
                        hf_token=hf_token,
                        dry_run=dry_run,
                        local_cache_dir=effective_local_cache,
                    )
            else:
                mdl_failed = distribute_model_from_local(
                    model,
                    host_list,
                    cache_dir=cache_dir,
                    local_cache_dir=effective_local_cache,
                    token=hf_token,
                    revision=model_revision,
                    transfer_hosts=transfer_hosts,
                    dry_run=dry_run,
                    **ssh_kwargs,
                )

        if mdl_failed:
            raise DistributionError("Model distribution failed on: %s" % ", ".join(mdl_failed))

    logger.info("Distribution complete.")
    return comm_env, ib_ip_map, mgmt_ip_map


def _resolve_targets(indices: list[int], host_list: list[str]) -> list[str]:
    """Resolve node index list to actual hosts. ``-1`` means all nodes."""
    if -1 in indices:
        return list(host_list)
    return [host_list[i] for i in indices if 0 <= i < len(host_list)]


def distribute_from_config(
    recipe: "Recipe",
    image: str,
    host_list: list[str],
    cache_dir: str,
    config: SparkrunConfig,
    dry_run: bool,
    model_revision: str | None = None,
    recipe_name: str = "",
    transfer_mode: str = "auto",
    transfer_interface: str | None = None,
    local_cache_dir: str | None = None,
    pre_ib: TransferModeResult | None = None,
    topology: str | None = None,
) -> tuple["ClusterCommEnv | None", dict[str, str], dict[str, str]]:
    """Distribute resources based on recipe ``distribution_config``.

    Resolves templated entry names, expands target node indices, and
    distributes each model/container to its specified hosts.  Falls back
    to ``distribute_resources()`` behavior when called with the default
    auto-generated config.

    Args:
        recipe: Loaded recipe with ``distribution_config`` attribute.
        image: Resolved container image reference.
        host_list: Target hostnames/IPs.
        cache_dir: HuggingFace cache directory.
        config: SparkrunConfig instance.
        dry_run: Show what would be done without executing.
        model_revision: Optional HuggingFace model revision to pin.
        recipe_name: Recipe name for pending-op lock display.
        transfer_mode: Distribution strategy.
        transfer_interface: Network interface for transfers.
        local_cache_dir: Control-machine cache dir for model downloads.
        pre_ib: Pre-computed IB detection results.

    Returns:
        Tuple of (comm_env, ib_ip_map, mgmt_ip_map).
    """
    from sparkrun.core.recipe import DistributionModelEntry, DistributionContainerEntry
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts, validate_ib_connectivity
    from sparkrun.containers.registry import ensure_image
    from sparkrun.models.download import download_model
    from sparkrun.core.pending_ops import pending_op

    dist_cfg = recipe.distribution_config.resolve(recipe, resolved_container=image)

    # Single-localhost fast path: same as distribute_resources
    ssh_kwargs = build_ssh_kwargs(config)
    hf_token = _get_hf_token()
    if len(host_list) <= 1 and is_local_host(host_list[0]) and not _is_cross_user(ssh_kwargs):
        _do_local_ensure = dist_cfg.containers.enabled
        _model_names = [e.name for e in dist_cfg.models.entries] if dist_cfg.models.enabled else []
        lock_parts = [image] + _model_names
        _lock_key = hashlib.sha256("|".join(lock_parts).encode()).hexdigest()[:12]
        _lock_id = f"sparkrun_{_lock_key}"
        _pop_kw = dict(
            recipe=recipe_name, model=_model_names[0] if _model_names else "", image=image, hosts=host_list, cache_dir=str(config.cache_dir)
        )

        if _do_local_ensure:
            with pending_op(_lock_id, "image_pull", **_pop_kw):
                logger.info("Ensuring container image is available locally...")
                if ensure_image(image, dry_run=dry_run) != 0:
                    raise DistributionError(f"Failed to pull or locate image: {image}")
        if _model_names:
            with pending_op(_lock_id, "model_download", **_pop_kw):
                for mn in _model_names:
                    logger.info("Ensuring model %s is available locally...", mn)
                    if (
                        download_model(mn, cache_dir=local_cache_dir or cache_dir, token=hf_token, revision=model_revision, dry_run=dry_run)
                        != 0
                    ):
                        raise DistributionError(f"Failed to download model: {mn}")
        return None, {}, {}

    # IB detection (reuse from pre_ib or compute)
    if pre_ib is not None and pre_ib.ib_result is not None:
        ib_result = pre_ib.ib_result
        _ib_validated = pre_ib.ib_validated
        _auto_delegated = pre_ib.auto_delegated
    else:
        ib_result = detect_ib_for_hosts(host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run, topology=topology)
        _ib_validated = None
        if transfer_mode in ("auto", "local"):
            _ib_validated = validate_ib_connectivity(ib_result.ib_ip_map, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
        _auto_delegated = False
        if transfer_mode == "auto":
            _cu = _is_cross_user(ssh_kwargs)
            if is_control_in_cluster(host_list) and not _cu:
                transfer_mode = "local"
            elif _ib_validated and not _cu:
                transfer_mode = "local"
            else:
                transfer_mode = "delegated"
                _auto_delegated = True

    comm_env = ib_result.comm_env
    mgmt_ip_map = ib_result.mgmt_ip_map
    _use_mgmt = transfer_interface == "mgmt"
    _cross_user = _is_cross_user(ssh_kwargs)

    if transfer_mode == "local":
        ib_ip_map = _ib_validated or {}
        transfer_hosts = [ib_result.ib_ip_map.get(h, h) for h in host_list] if ib_ip_map and not _use_mgmt else None
        worker_transfer_hosts = None
    else:
        ib_ip_map = ib_result.ib_ip_map
        transfer_hosts = None
        if len(host_list) > 1 and ib_result.ib_ip_map and not _use_mgmt:
            worker_transfer_hosts = [ib_result.ib_ip_map.get(h, h) for h in host_list[1:]]
        else:
            worker_transfer_hosts = None

    from sparkrun.core.progress import PROGRESS as _PROGRESS_LEVEL

    effective_local_cache = local_cache_dir or cache_dir
    _lock_key = hashlib.sha256(f"{image}|{','.join(host_list)}".encode()).hexdigest()[:12]
    _lock_id = f"sparkrun_{_lock_key}"
    _pop_kw = dict(recipe=recipe_name, model=image, image=image, hosts=host_list, cache_dir=str(config.cache_dir))

    # Distribute container images
    if dist_cfg.containers.enabled:
        for entry in dist_cfg.containers.entries:
            if not isinstance(entry, DistributionContainerEntry):
                continue
            entry_name = entry.name or image
            targets = _resolve_targets(entry.target if entry.target else [-1], host_list)
            if not targets:
                continue
            logger.log(_PROGRESS_LEVEL, "  Distributing image %s to %d host(s)", entry_name, len(targets))
            with pending_op(_lock_id, "image_distribute", **_pop_kw):
                img_failed = _distribute_single_image(
                    entry_name,
                    targets,
                    host_list,
                    transfer_mode,
                    transfer_hosts,
                    worker_transfer_hosts,
                    ssh_kwargs,
                    dry_run,
                    _auto_delegated,
                )
            if img_failed:
                raise DistributionError("Image distribution failed on: %s" % ", ".join(img_failed))

    # Distribute models
    if dist_cfg.models.enabled:
        for entry in dist_cfg.models.entries:
            if not isinstance(entry, DistributionModelEntry):
                continue
            if not entry.name:
                continue
            targets = _resolve_targets(entry.target if entry.target else [-1], host_list)
            if not targets:
                continue
            entry_revision = entry.revision or model_revision
            logger.log(_PROGRESS_LEVEL, "  Distributing model %s to %d host(s)", entry.name, len(targets))
            with pending_op(_lock_id, "model_download", **_pop_kw):
                mdl_failed = _distribute_single_model(
                    entry.name,
                    targets,
                    host_list,
                    cache_dir,
                    effective_local_cache,
                    transfer_mode,
                    transfer_hosts,
                    worker_transfer_hosts,
                    ssh_kwargs,
                    entry_revision,
                    hf_token,
                    dry_run,
                    _auto_delegated,
                )
            if mdl_failed:
                raise DistributionError("Model distribution failed on: %s" % ", ".join(mdl_failed))

    logger.info("Distribution complete.")
    return comm_env, ib_ip_map, mgmt_ip_map


def _distribute_single_image(
    image: str,
    targets: list[str],
    full_hosts: list[str],
    transfer_mode: str,
    transfer_hosts: list[str] | None,
    worker_transfer_hosts: list[str] | None,
    ssh_kwargs: dict,
    dry_run: bool,
    auto_delegated: bool,
) -> list[str]:
    """Distribute a single image to a subset of hosts."""
    from sparkrun.containers.distribute import distribute_image_from_local, distribute_image_from_head

    # Map transfer hosts to target subset
    target_set = set(targets)
    t_hosts = [h for h in transfer_hosts if h in target_set] if transfer_hosts else None
    w_hosts = [h for h in worker_transfer_hosts if h in target_set] if worker_transfer_hosts else None

    if transfer_mode == "local":
        return distribute_image_from_local(image, targets, transfer_hosts=t_hosts, dry_run=dry_run, **ssh_kwargs)
    elif transfer_mode == "push":
        head = targets[0]
        if targets == full_hosts:
            return _distribute_image_push(image, targets, w_hosts, ssh_kwargs, dry_run)
        # Subset push: push to head only, then head distributes
        head_failed = distribute_image_from_local(image, [head], transfer_hosts=None, dry_run=dry_run, **ssh_kwargs)
        if head_failed:
            return list(targets)
        if len(targets) > 1:
            return distribute_image_from_head(image, targets, worker_transfer_hosts=w_hosts, dry_run=dry_run, **ssh_kwargs)
        return []
    elif transfer_mode == "delegated":
        result = distribute_image_from_head(image, targets, worker_transfer_hosts=w_hosts, dry_run=dry_run, **ssh_kwargs)
        if result and auto_delegated:
            head = targets[0]
            head_failed = distribute_image_from_local(image, [head], transfer_hosts=None, dry_run=dry_run, **ssh_kwargs)
            if not head_failed and len(targets) > 1:
                result = distribute_image_from_head(image, targets, worker_transfer_hosts=w_hosts, dry_run=dry_run, **ssh_kwargs)
        return result
    return distribute_image_from_local(image, targets, transfer_hosts=t_hosts, dry_run=dry_run, **ssh_kwargs)


def _distribute_single_model(
    model: str,
    targets: list[str],
    full_hosts: list[str],
    cache_dir: str,
    local_cache_dir: str,
    transfer_mode: str,
    transfer_hosts: list[str] | None,
    worker_transfer_hosts: list[str] | None,
    ssh_kwargs: dict,
    revision: str | None,
    hf_token: str | None,
    dry_run: bool,
    auto_delegated: bool,
) -> list[str]:
    """Distribute a single model to a subset of hosts."""
    from sparkrun.models.distribute import distribute_model_from_local, distribute_model_from_head

    target_set = set(targets)
    t_hosts = [h for h in transfer_hosts if h in target_set] if transfer_hosts else None
    w_hosts = [h for h in worker_transfer_hosts if h in target_set] if worker_transfer_hosts else None

    if transfer_mode == "local":
        return distribute_model_from_local(
            model,
            targets,
            cache_dir=cache_dir,
            local_cache_dir=local_cache_dir,
            token=hf_token,
            revision=revision,
            transfer_hosts=t_hosts,
            dry_run=dry_run,
            **ssh_kwargs,
        )
    elif transfer_mode == "push":
        head = targets[0]
        head_failed = distribute_model_from_local(
            model,
            [head],
            cache_dir=cache_dir,
            local_cache_dir=local_cache_dir,
            token=hf_token,
            revision=revision,
            transfer_hosts=None,
            dry_run=dry_run,
            **ssh_kwargs,
        )
        if head_failed:
            return list(targets)
        if len(targets) > 1:
            return distribute_model_from_head(
                model,
                targets,
                cache_dir=cache_dir,
                revision=revision,
                hf_token=hf_token,
                worker_transfer_hosts=w_hosts,
                dry_run=dry_run,
                **ssh_kwargs,
            )
        return []
    elif transfer_mode == "delegated":
        result = distribute_model_from_head(
            model,
            targets,
            cache_dir=cache_dir,
            revision=revision,
            hf_token=hf_token,
            worker_transfer_hosts=w_hosts,
            dry_run=dry_run,
            **ssh_kwargs,
        )
        if result and auto_delegated:
            head = targets[0]
            head_failed = distribute_model_from_local(
                model,
                [head],
                cache_dir=cache_dir,
                local_cache_dir=local_cache_dir,
                token=hf_token,
                revision=revision,
                transfer_hosts=None,
                dry_run=dry_run,
                **ssh_kwargs,
            )
            if not head_failed and len(targets) > 1:
                result = distribute_model_from_head(
                    model,
                    targets,
                    cache_dir=cache_dir,
                    revision=revision,
                    hf_token=hf_token,
                    worker_transfer_hosts=w_hosts,
                    dry_run=dry_run,
                    **ssh_kwargs,
                )
        return result
    return distribute_model_from_local(
        model,
        targets,
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        token=hf_token,
        revision=revision,
        transfer_hosts=t_hosts,
        dry_run=dry_run,
        **ssh_kwargs,
    )
