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
import contextlib

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.progress import LaunchProgress
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.registry import RegistryManager
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.runtimes.base import RuntimePlugin
    from sparkrun.builders.base import BuilderPlugin

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
    comm_env: "ClusterCommEnv | None" = None
    ib_ip_map: dict[str, str] = field(default_factory=dict)
    serve_command: str = ""
    runtime_info: dict[str, str] = field(default_factory=dict)
    builder: BuilderPlugin | None = None


def launch_inference(
    *,
    recipe: Recipe,
    runtime: RuntimePlugin,
    host_list: list[str],
    overrides: dict[str, Any],
    config: SparkrunConfig | None = None,
    v=None,
    sctx: SparkrunContext | None = None,
    is_solo: bool = False,
    cache_dir: str | None = None,
    local_cache_dir: str | None = None,
    transfer_mode: str | None = None,
    transfer_interface: str | None = None,
    recipe_ref: str | None = None,
    registry_mgr: RegistryManager | None = None,
    auto_port: bool = False,
    sync_tuning: bool = True,
    skip_keys: set[str] | frozenset[str] = frozenset(),
    dry_run: bool = False,
    detached: bool = True,
    follow: bool = True,
    # Runtime-specific kwargs forwarded to runtime.run()
    ray_port: int | None = None,
    dashboard_port: int | None = None,
    dashboard: bool = False,
    init_port: int | None = None,
    topology: str | None = None,
    cluster_id_override: str | None = None,
    # Executor config (dict for config chain layering)
    executor_config: dict | None = None,
    extra_docker_opts: list[str] | None = None,
    # note: transition to rootless by default
    rootless: bool = True,
    auto_user: bool = True,
    progress: LaunchProgress | None = None,
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
        cache_dir: Remote/cluster cache dir (None = resolve from config).
        local_cache_dir: Control-machine cache dir for downloads (None = same as cache_dir).
        transfer_mode: Resource transfer mode override (None = "auto").
        transfer_interface: Network interface for transfers (cx7 or mgmt; None = cx7 default).
        recipe_ref: Simplified recipe reference for display (e.g. @spark-arena/UUID).
        registry_mgr: Registry manager for tuning config sync.
        auto_port: If True, auto-increment port when the desired port is in use.
        sync_tuning: Whether to sync tuning configs from registries.
        skip_keys: Keys to suppress in serve command generation.
        dry_run: Show what would be done without executing.
        detached: Run containers in detached mode.
        follow: whether to follow logs
        ray_port: Ray GCS port (forwarded to runtime.run).
        dashboard_port: Ray dashboard port (forwarded to runtime.run).
        dashboard: Enable Ray dashboard (forwarded to runtime.run).
        init_port: Distributed init port (forwarded to runtime.run).
        executor_config: Executor config
        rootless: Run containers in rootless mode (applies defaults to executor_config)
        auto_user: Automatically set user and group IDs to match host. (applies defaults to executor_config)


    Returns:
        LaunchResult with the outcome and all resolved context.
    """
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    # Resolve config, v, and progress from sctx when provided
    if sctx is not None:
        if config is None:
            config = sctx.config
        if v is None:
            v = sctx.variables
        if progress is None:
            progress = sctx.progress
    if config is None:
        from sparkrun.core.config import SparkrunConfig

        config = SparkrunConfig()
    p = progress  # short alias

    from sparkrun.orchestration.distribution import resolve_auto_transfer_mode

    # -- Phase 1: Prepare --
    if p:
        p.phase(1)

    effective_cache_dir = cache_dir or str(config.hf_cache_dir)
    effective_local_cache = local_cache_dir or effective_cache_dir
    ssh_kwargs = build_ssh_kwargs(config)
    transfer_result = resolve_auto_transfer_mode(
        transfer_mode or "auto",
        host_list,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
    )
    effective_transfer_mode = transfer_result.mode

    # -- Port resolution --
    if auto_port:
        from sparkrun.orchestration.primitives import find_available_port

        config_chain = recipe.build_config_chain(overrides)
        desired_port = int(str(config_chain.get("port") or 8000))
        head_host = host_list[0]
        serve_port = find_available_port(
            head_host,
            desired_port,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
        )
        overrides["port"] = serve_port
    else:
        config_chain = recipe.build_config_chain(overrides)
        serve_port = int(str(config_chain.get("port") or 8000))

    # Derive deterministic cluster_id from recipe + (trimmed) hosts
    cluster_id = cluster_id_override or generate_cluster_id(recipe, host_list, overrides=overrides)

    # Resolve container image
    container_image = runtime.resolve_container(recipe, overrides)

    if p:
        p.phase_end()

    # -- Phase 2: Builder --
    builder = None
    if recipe.builder:
        if p:
            p.phase(2)
        from sparkrun.core.bootstrap import get_builder

        try:
            builder = get_builder(recipe.builder, v)
            container_image = builder.prepare_image(
                container_image,
                recipe,
                host_list,
                config=config,
                dry_run=dry_run,
                transfer_mode=effective_transfer_mode,
                ssh_kwargs=ssh_kwargs,
            )
        except ValueError:
            logger.warning("Builder '%s' not found, skipping", recipe.builder)
        if p:
            p.phase_end()
    else:
        if p:
            p.phase_skip(2, "no builder")

    # Save job metadata
    if not dry_run:
        try:
            save_job_metadata(
                cluster_id,
                recipe,
                host_list,
                overrides=overrides,
                cache_dir=str(config.cache_dir),
                recipe_ref=recipe_ref,
                container_image=container_image,
            )
        except Exception:
            logger.debug("Failed to save job metadata: %s", cluster_id, exc_info=True)

    # Pre-launch preparation (e.g., eugr container builds)
    runtime.prepare(
        recipe,
        host_list,
        config=config,
        dry_run=dry_run,
        transfer_mode=effective_transfer_mode,
    )

    # -- Phase 3: Distribution --
    comm_env = None
    ib_ip_map: dict[str, str] = {}
    if not runtime.is_delegating_runtime():
        if p:
            p.phase(3)
        from sparkrun.orchestration.distribution import distribute_resources

        comm_env, ib_ip_map, mgmt_ip_map = distribute_resources(
            container_image,
            recipe.model,
            host_list,
            effective_cache_dir,
            config,
            dry_run,
            model_revision=recipe.model_revision,
            recipe_name=recipe.name,
            transfer_mode=effective_transfer_mode,
            transfer_interface=transfer_interface,
            local_cache_dir=effective_local_cache,
            pre_ib=transfer_result,
        )
        # Re-save job metadata with IP maps from IB detection
        if not dry_run and (ib_ip_map or mgmt_ip_map):
            try:
                save_job_metadata(
                    cluster_id,
                    recipe,
                    host_list,
                    overrides=overrides,
                    cache_dir=str(config.cache_dir),
                    ib_ip_map=ib_ip_map,
                    mgmt_ip_map=mgmt_ip_map,
                    recipe_ref=recipe_ref,
                )
            except Exception:
                logger.debug("Failed to update job metadata: %s", cluster_id, exc_info=True)
        if p:
            p.phase_end()
    else:
        if p:
            p.phase_skip(3, "delegating runtime")

    # -- Phase 4: Tuning --
    _needs_tuning = (sync_tuning and not dry_run) or not runtime.is_delegating_runtime()
    if _needs_tuning:
        if p:
            p.phase(4)
    else:
        if p:
            p.phase_skip(4, "disabled")

    if sync_tuning and not dry_run:
        from sparkrun.tuning.sync import sync_registry_tuning

        try:
            rm = registry_mgr
            if rm is None:
                rm = config.get_registry_manager()
            synced = sync_registry_tuning(
                rm,
                recipe.runtime,
                dry_run=dry_run,
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
                recipe.runtime,
                host_list,
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

    if _needs_tuning and p:
        p.phase_end()

    # GGUF model resolution
    from sparkrun.models.download import is_gguf_model, resolve_gguf_container_path

    if is_gguf_model(recipe.model) and not dry_run:
        gguf_container_path = resolve_gguf_container_path(
            recipe.model,
            effective_cache_dir,
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

    # -- Phase 5: Launch runtime --
    if p:
        from sparkrun.utils.cli_formatters import RUNTIME_DISPLAY

        _rt_display = RUNTIME_DISPLAY.get(runtime.runtime_name, runtime.runtime_name)
        p.phase(5, "Launching %s runtime" % _rt_display)

    # Build runtime.run() kwargs — include runtime-specific options only
    # when they were explicitly provided.
    run_kwargs: dict[str, Any] = {"follow": follow}
    if ray_port is not None:
        run_kwargs["ray_port"] = ray_port
    if dashboard_port is not None:
        run_kwargs["dashboard_port"] = dashboard_port
    if dashboard:
        run_kwargs["dashboard"] = dashboard
    if init_port is not None:
        run_kwargs["init_port"] = init_port
    if topology is not None:
        run_kwargs["topology"] = topology

    # Build executor from layered config: CLI → recipe → defaults
    from scitrera_app_framework.api import Variables, EnvPlacement
    from sparkrun.orchestration.executor import EXECUTOR_DEFAULTS, ExecutorConfig
    from sparkrun.orchestration.executor_docker import DockerExecutor

    exec_adjustments = {}
    if rootless:
        exec_adjustments["privileged"] = False
        exec_adjustments["security_opt"] = ["no-new-privileges"]
        exec_adjustments["cap_add"] = []
        exec_adjustments["ulimit"] = [
            "memlock=-1:-1",
            "stack=67108864",
        ]
        # TODO: confirm existence and/or adjust? (for future heterogeneous support??)
        exec_adjustments["devices"] = [
            "/dev/infiniband",
        ]
    if auto_user:
        exec_adjustments["user"] = "$SHELL_USER"  # auto hint to use ssh user+group

    recipe_executor_config = getattr(recipe, "executor_config", None)
    if not isinstance(recipe_executor_config, dict):
        recipe_executor_config = {}
    cli_exec_opts = executor_config if isinstance(executor_config, dict) else {}
    exec_chain = Variables(
        sources=(
            cli_exec_opts,  # CLI flags (highest priority)
            recipe_executor_config,  # recipe YAML
            exec_adjustments,  # executor adjustments
            EXECUTOR_DEFAULTS,  # hardcoded defaults
        ),
        env_placement=EnvPlacement.IGNORED,
    )
    exec_cfg = ExecutorConfig.from_chain(exec_chain)
    executor = DockerExecutor(exec_cfg)  # TODO: future flexible executor

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
        comm_env=comm_env,
        ib_ip_map=ib_ip_map,
        skip_keys=skip_keys,
        executor=executor,
        progress=progress,
        extra_docker_opts=extra_docker_opts,
        **run_kwargs,
    )

    if p:
        p.phase_end()

    # Collect runtime version info from the head container (non-blocking)
    runtime_info: dict[str, str] = {}
    if rc == 0 and not dry_run:
        try:
            head_host = host_list[0] if host_list else "localhost"
            head_container = runtime.get_head_container_name(cluster_id, is_solo=is_solo)
            # Resolve builder for version info collection
            ver_builder = None
            if recipe.builder:
                from sparkrun.core.bootstrap import get_builder

                with contextlib.suppress(ValueError):
                    ver_builder = get_builder(recipe.builder, v)
            # noinspection PyProtectedMember
            runtime_info = runtime._collect_runtime_info(
                head_host,
                head_container,
                ssh_kwargs,
                dry_run=False,
                builder=ver_builder,
            )
            # Collect container image labels (separate docker inspect call)
            if ver_builder:
                try:
                    label_info = ver_builder.collect_container_labels(
                        head_container,
                        head_host,
                        ssh_kwargs,
                    )
                    # Merge without overwriting existing keys
                    for k, lv in label_info.items():
                        if k not in runtime_info:
                            runtime_info[k] = lv
                except Exception:
                    logger.debug("Container label collection failed", exc_info=True)
            if runtime_info:
                try:
                    save_job_metadata(
                        cluster_id,
                        recipe,
                        host_list,
                        overrides=overrides,
                        cache_dir=str(config.cache_dir),
                        recipe_ref=recipe_ref,
                        runtime_info=runtime_info,
                        container_image=container_image,
                    )
                except Exception:
                    logger.debug("Failed to save runtime_info to job metadata", exc_info=True)
        except Exception:
            logger.debug("Runtime info collection failed", exc_info=True)

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
        comm_env=comm_env,
        ib_ip_map=ib_ip_map,
        serve_command=serve_command,
        runtime_info=runtime_info,
        builder=builder,
    )


def post_launch_lifecycle(
    result: LaunchResult,
    remote_cache_dir: str,
    trust: bool = False,
    dry_run: bool = False,
    progress: LaunchProgress | None = None,
) -> None:
    """Run post-serve lifecycle: port polling, health checks, hooks, conditional stop.

    Called after a successful detached launch when recipe defines post_exec or post_commands.
    Handles:
    1. Determining head container name
    2. Detecting head IP
    3. Waiting for port and health check
    4. Building hook context
    5. Running post_exec and post_commands
    6. Handling stop_after_post

    Args:
        result: LaunchResult from launch_inference.
        remote_cache_dir: Remote cache directory for hook context.
        trust: Trust post_commands from non-default registries without prompting.
        dry_run: Show what would be done without executing.
    """
    import sys

    import click

    from sparkrun.orchestration.hooks import (
        build_hook_context,
        run_post_commands,
        run_post_exec,
    )
    from sparkrun.orchestration.health import wait_for_healthy, wait_for_port
    from sparkrun.orchestration.primitives import build_ssh_kwargs, detect_host_ip
    from sparkrun.orchestration.docker import generate_container_name, generate_node_container_name
    from sparkrun.utils import is_local_host

    p = progress  # short alias
    if p:
        p.phase(6)

    recipe = result.recipe
    runtime = result.runtime
    host_list = result.host_list
    overrides = result.overrides
    config = result.config
    is_solo = result.is_solo

    head_host = host_list[0] if host_list else "localhost"
    _ssh_kw = build_ssh_kwargs(config)

    # Determine head container name
    head_container = generate_container_name(result.cluster_id, "solo") if is_solo else generate_node_container_name(result.cluster_id, 0)

    # Detect head IP for health checks
    if is_local_host(head_host):
        head_ip = "127.0.0.1"
    else:
        try:
            head_ip = detect_host_ip(head_host, ssh_kwargs=_ssh_kw, dry_run=dry_run)
        except RuntimeError:
            head_ip = head_host

    # Determine effective port
    config_chain = recipe.build_config_chain(overrides)
    effective_port = int(str(config_chain.get("port", 8000)))

    click.echo("Waiting for server to become ready...")
    if not dry_run:
        # Wait for port to be listening
        port_ready = wait_for_port(
            head_host,
            effective_port,
            max_retries=120,
            retry_interval=2,
            ssh_kwargs=_ssh_kw,
            dry_run=dry_run,
            container_name=head_container,
        )
        if not port_ready:
            click.echo("Error: Server port %s never became ready" % effective_port, err=True)
            sys.exit(1)

        # Wait for HTTP 200 on /v1/models
        health_url = "http://%s:%s/v1/models" % (head_ip, effective_port)
        healthy = wait_for_healthy(health_url, max_retries=120, retry_interval=5, dry_run=dry_run)
        if not healthy:
            click.echo("Error: Server health check never passed at %s" % health_url, err=True)
            sys.exit(1)

    # Build hook context with extended variables
    hook_context = build_hook_context(
        config_chain,
        head_host=head_host,
        head_ip=head_ip,
        port=effective_port,
        cluster_id=result.cluster_id,
        container_name=head_container,
        cache_dir=remote_cache_dir,
    )

    try:
        # Run post_exec inside head container
        if recipe.post_exec:
            click.echo("Running post_exec commands...")
            run_post_exec(head_host, head_container, recipe.post_exec, hook_context, ssh_kwargs=_ssh_kw, dry_run=dry_run)

        # Run post_commands on control machine
        if recipe.post_commands:
            click.echo("Running post_commands on control machine...")
            from sparkrun.core.registry import DEFAULT_REGISTRIES_GIT

            _is_trusted = (
                trust
                or recipe.source_registry is None  # local recipe
                or (recipe.source_registry_url is not None and recipe.source_registry_url in DEFAULT_REGISTRIES_GIT)
            )
            run_post_commands(recipe.post_commands, hook_context, dry_run=dry_run, trust=_is_trusted)
    except RuntimeError as e:
        click.echo("Error in post hooks: %s" % e, err=True)
        sys.exit(1)

    click.echo("Post hooks completed successfully.")
    if p:
        p.phase_end()

    # If stop_after_post, stop the workload and exit
    if recipe.stop_after_post:
        click.echo("Stopping workload (stop_after_post=true)...")
        runtime.stop(
            hosts=host_list,
            cluster_id=result.cluster_id,
            config=config,
            dry_run=dry_run,
        )
        sys.exit(0)
