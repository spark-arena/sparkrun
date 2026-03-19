"""Common inference launch pipeline.

Shared by ``sparkrun run``, ``sparkrun benchmark``, and
``sparkrun proxy load``.  Callers are responsible for recipe loading,
host resolution, override building, and node trimming *before*
calling :func:`launch_inference`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.registry import RegistryManager
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)


@dataclass
class LaunchResult:
    """Result of :func:`launch_inference`."""

    rc: int
    cluster_id: str
    host_list: list[str]
    is_solo: bool
    runtime: RuntimePlugin
    recipe: Recipe
    overrides: dict[str, Any]
    container_image: str
    effective_cache_dir: str
    serve_port: int
    config: SparkrunConfig
    recipe_ref: str | None = None
    nccl_env: dict[str, str] | None = None
    ib_ip_map: dict[str, str] = field(default_factory=dict)
    serve_command: str = ""


def launch_inference(
        *,
        recipe: Recipe,
        runtime: RuntimePlugin,
        host_list: list[str],
        overrides: dict[str, Any],
        config: SparkrunConfig,
        v=None,
        is_solo: bool = False,
        cache_dir: str | None = None,
        transfer_mode: str | None = None,
        recipe_ref: str | None = None,
        registry_mgr: RegistryManager | None = None,
        auto_port: bool = False,
        sync_tuning: bool = True,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        dry_run: bool = False,
        detached: bool = True,
        # Runtime-specific kwargs forwarded to runtime.run()
        ray_port: int | None = None,
        dashboard_port: int | None = None,
        dashboard: bool = False,
        init_port: int | None = None,
        # Container lifecycle options
        auto_remove: bool = True,
        restart_policy: str | None = None,
) -> LaunchResult:
    """Launch an inference workload.

    This is the shared pipeline used by ``run``, ``benchmark``, and
    ``proxy load``.  It handles:

    1. Job metadata persistence
    2. Builder phase (if recipe defines a builder)
    3. Runtime preparation
    4. Resource distribution (container image + model)
    5. Tuning config sync and distribution
    6. GGUF model resolution
    7. Serve command generation
    8. Page cache clear
    9. ``runtime.run()``

    Args:
        recipe: Loaded and validated recipe.
        runtime: Resolved runtime plugin.
        host_list: Resolved and trimmed host list.
        overrides: Merged overrides dict (from recipe_override_options + extras).
        config: SparkrunConfig instance.
        v: SAF Variables instance (optional, uses singleton if None).
        is_solo: Whether to launch in solo mode.
        cache_dir: Explicit cache dir override (None = resolve from config).
        transfer_mode: Resource transfer mode override (None = "auto").
        recipe_ref: Simplified recipe reference for display (e.g. @spark-arena/UUID).
        registry_mgr: Registry manager for tuning config sync.
        auto_port: If True, auto-increment port when the desired port is in use.
        sync_tuning: Whether to sync tuning configs from registries.
        skip_keys: Keys to suppress in serve command generation.
        dry_run: Show what would be done without executing.
        detached: Run containers in detached mode.
        ray_port: Ray GCS port (forwarded to runtime.run).
        dashboard_port: Ray dashboard port (forwarded to runtime.run).
        dashboard: Enable Ray dashboard (forwarded to runtime.run).
        init_port: Distributed init port (forwarded to runtime.run).

    Returns:
        LaunchResult with the outcome and all resolved context.
    """
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    effective_cache_dir = cache_dir or str(config.hf_cache_dir)
    effective_transfer_mode = transfer_mode or "auto"
    ssh_kwargs = build_ssh_kwargs(config)

    # -- Port resolution --
    if auto_port:
        from sparkrun.orchestration.primitives import find_available_port

        config_chain = recipe.build_config_chain(overrides)
        desired_port = int(config_chain.get("port") or 8000)
        head_host = host_list[0]
        serve_port = find_available_port(
            head_host, desired_port, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )
        if serve_port != desired_port:
            logger.info(
                "Port %d in use on %s, using %d instead",
                desired_port, head_host, serve_port,
            )
        overrides["port"] = serve_port
    else:
        config_chain = recipe.build_config_chain(overrides)
        serve_port = int(config_chain.get("port") or 8000)

    # Derive deterministic cluster_id from recipe + (trimmed) hosts
    cluster_id = generate_cluster_id(recipe, host_list, overrides=overrides)

    # Resolve container image
    container_image = runtime.resolve_container(recipe, overrides)

    # Save job metadata
    if not dry_run:
        try:
            save_job_metadata(
                cluster_id, recipe, host_list,
                overrides=overrides, cache_dir=str(config.cache_dir),
                recipe_ref=recipe_ref,
            )
        except Exception:
            logger.debug("Failed to save job metadata: %s", cluster_id, exc_info=True)

    # Builder phase
    if recipe.builder:
        from sparkrun.core.bootstrap import get_builder

        try:
            builder = get_builder(recipe.builder, v)
            container_image = builder.prepare_image(
                container_image, recipe, host_list, config=config, dry_run=dry_run,
                transfer_mode=effective_transfer_mode,
                ssh_kwargs=ssh_kwargs,
            )
        except ValueError:
            logger.warning("Builder '%s' not found, skipping", recipe.builder)

    # Pre-launch preparation (e.g., eugr container builds)
    runtime.prepare(
        recipe, host_list, config=config, dry_run=dry_run,
        transfer_mode=effective_transfer_mode,
    )

    # Distribution phase
    nccl_env = None
    ib_ip_map: dict[str, str] = {}
    if not runtime.is_delegating_runtime():
        from sparkrun.orchestration.distribution import distribute_resources

        nccl_env, ib_ip_map, mgmt_ip_map = distribute_resources(
            container_image, recipe.model, host_list,
            effective_cache_dir,
            config, dry_run,
            model_revision=recipe.model_revision,
            recipe_name=recipe.name,
            transfer_mode=effective_transfer_mode,
        )
        # Re-save job metadata with IP maps from IB detection
        if not dry_run and (ib_ip_map or mgmt_ip_map):
            try:
                save_job_metadata(
                    cluster_id, recipe, host_list,
                    overrides=overrides, cache_dir=str(config.cache_dir),
                    ib_ip_map=ib_ip_map, mgmt_ip_map=mgmt_ip_map,
                    recipe_ref=recipe_ref,
                )
            except Exception:
                logger.debug("Failed to update job metadata: %s", cluster_id, exc_info=True)

    # Sync registry tuning configs
    if sync_tuning and not dry_run:
        from sparkrun.tuning.sync import sync_registry_tuning

        try:
            synced = sync_registry_tuning(
                registry_mgr, recipe.runtime, dry_run=dry_run,
                registry_name=recipe.source_registry,
            )
            if synced:
                logger.info("Synced %d tuning config(s) from registries.", synced)
        except Exception:
            logger.debug("Failed to sync tuning configs", exc_info=True)

    # Distribute tuning configs to remote hosts
    if not runtime.is_delegating_runtime():
        from sparkrun.tuning.distribute import distribute_tuning_to_hosts

        try:
            tuning_failed = distribute_tuning_to_hosts(
                recipe.runtime, host_list,
                dry_run=dry_run,
                transfer_mode=effective_transfer_mode,
                **ssh_kwargs,
            )
            if tuning_failed:
                logger.warning(
                    "Tuning config distribution failed on: %s",
                    ", ".join(tuning_failed),
                )
        except Exception:
            logger.debug("Failed to distribute tuning configs", exc_info=True)

    # GGUF model resolution
    from sparkrun.models.download import is_gguf_model, resolve_gguf_container_path

    if is_gguf_model(recipe.model) and not dry_run:
        gguf_container_path = resolve_gguf_container_path(
            recipe.model, effective_cache_dir,
        )
        if gguf_container_path:
            overrides["_gguf_model_path"] = gguf_container_path
            overrides["model"] = gguf_container_path
            logger.info("GGUF model pre-synced, container path: %s", gguf_container_path)

    # Generate serve command
    serve_command = runtime.generate_command(
        recipe=recipe,
        overrides=overrides,
        is_cluster=not is_solo,
        num_nodes=len(host_list),
        head_ip=None,  # determined during launch
        skip_keys=skip_keys,
    )

    # Best-effort page cache clear
    if not runtime.is_delegating_runtime():
        from sparkrun.orchestration.primitives import try_clear_page_cache

        try_clear_page_cache(host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    # Build runtime.run() kwargs — include runtime-specific options only
    # when they were explicitly provided.
    run_kwargs: dict[str, Any] = {}
    if ray_port is not None:
        run_kwargs["ray_port"] = ray_port
    if dashboard_port is not None:
        run_kwargs["dashboard_port"] = dashboard_port
    if dashboard:
        run_kwargs["dashboard"] = dashboard
    if init_port is not None:
        run_kwargs["init_port"] = init_port

    # Launch
    rc = runtime.run(
        hosts=host_list,
        image=container_image,
        serve_command=serve_command,
        recipe=recipe,
        overrides=overrides,
        cluster_id=cluster_id,
        env=recipe.env,
        cache_dir=effective_cache_dir,
        config=config,
        dry_run=dry_run,
        detached=detached,
        nccl_env=nccl_env,
        ib_ip_map=ib_ip_map,
        skip_keys=skip_keys,
        auto_remove=auto_remove,
        restart_policy=restart_policy,
        **run_kwargs,
    )

    return LaunchResult(
        rc=rc,
        cluster_id=cluster_id,
        host_list=host_list,
        is_solo=is_solo,
        runtime=runtime,
        recipe=recipe,
        overrides=overrides,
        container_image=container_image,
        effective_cache_dir=effective_cache_dir,
        serve_port=serve_port,
        config=config,
        recipe_ref=recipe_ref,
        nccl_env=nccl_env,
        ib_ip_map=ib_ip_map,
        serve_command=serve_command,
    )
