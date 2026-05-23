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
    from sparkrun.core.backend_select import BackendBundle
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
    backends: dict[str, "BackendBundle"] = field(default_factory=dict)
    """Per-host backend bundles resolved from fingerprint/hardware metadata.

    Populated when at least one host's hardware resolved cleanly through
    :func:`sparkrun.core.backend_select.select_backends`.  Empty dict
    when no resolution was performed (e.g. caller bypassed cluster
    threading) — runtimes then fall back to the legacy
    :func:`resolve_ib_env` path.
    """


def resolve_recipe_trust(recipe: Recipe, trust_cli: bool) -> bool:
    """Decide whether recipe hooks (pre_exec/post_exec/post_commands) are trusted.

    A recipe is trusted when any of these hold:

    * the user passed ``--trust`` on the CLI (``trust_cli=True``);
    * the recipe was loaded from a local path (no ``source_registry``);
    * the recipe came from a registry whose URL is in
      :data:`sparkrun.core.registry.DEFAULT_REGISTRIES_GIT`.

    Args:
        recipe: The loaded recipe (used for ``source_registry`` /
            ``source_registry_url`` introspection).
        trust_cli: CLI ``--trust`` flag value.

    Returns:
        True when the hook commands may run without per-launch
        confirmation, False when they should be gated by an interactive
        prompt.
    """
    from sparkrun.core.registry import DEFAULT_REGISTRIES_GIT

    return (
        trust_cli
        or recipe.source_registry is None  # local recipe
        or (recipe.source_registry_url is not None and recipe.source_registry_url in DEFAULT_REGISTRIES_GIT)
    )


def resolve_effective_cache_dir(
    cache_dir: str | None,
    host_list: list[str],
    ssh_kwargs: dict,
    config: SparkrunConfig,
    dry_run: bool = False,
) -> str:
    """Resolve the remote HF cache path to a concrete absolute string.

    - If *cache_dir* is given (cluster ``cache_dir`` or CLI override), use it
      as-is.
    - Otherwise, when targeting a remote host (or running cross-user), probe
      the head node via SSH so the resolved path reflects the SSH login user's
      ``$HOME`` / ``HF_HOME`` rather than the control machine's.
    - For the single-localhost same-user fast path, fall back to the control
      machine's HF cache.

    Returning a concrete path here avoids embedding shell-expansion expressions
    downstream, where ``shlex.quote``-aware code paths (volume mounts, ssh
    quoted commands) would prevent the expansion from running.
    """
    if cache_dir:
        return cache_dir

    from sparkrun.utils import is_local_host
    from sparkrun.orchestration.primitives import probe_remote_hf_cache
    import os

    head = host_list[0] if host_list else None
    ssh_user = ssh_kwargs.get("ssh_user")
    cross_user = ssh_user is not None and ssh_user != os.environ.get("USER", "root")

    if head and not is_local_host(head):
        return probe_remote_hf_cache(head, dry_run=dry_run, **ssh_kwargs)
    if head and cross_user:
        return probe_remote_hf_cache(head, dry_run=dry_run, **ssh_kwargs)

    return str(config.hf_cache_dir)


def resolve_per_host_backends(
    host_list: list[str],
    cluster=None,
) -> dict[str, "BackendBundle"]:
    """Resolve a :class:`BackendBundle` per host via :func:`select_backends`.

    For each host in *host_list*, calls
    :meth:`ClusterDefinition.hardware_for` (or defaults to DGX Spark
    when *cluster* is ``None``) and routes the result through
    :func:`sparkrun.core.backend_select.select_backends`.

    Hosts whose hardware fails to resolve a backend (unknown vendor,
    multi-vendor host, etc.) are silently skipped: runtimes fall back
    to the legacy :func:`resolve_ib_env` path for those hosts.  This
    keeps the cluster-launch surface live for partial-vendor coverage
    rather than failing-fast on a single bad fingerprint.

    Args:
        host_list: Resolved cluster hosts.
        cluster: Optional :class:`ClusterDefinition` carrying per-host
            hardware metadata.

    Returns:
        Mapping host -> :class:`BackendBundle`.  Empty dict when no
        host resolved successfully (e.g. all-Apple or all-CPU cluster).
    """
    from sparkrun.core.backend_select import NoMatchingBackendError, select_backends
    from sparkrun.core.hardware import default_dgx_spark_hardware

    backends: dict[str, BackendBundle] = {}
    for host in host_list:
        if cluster is not None:
            hw = cluster.hardware_for(host)
        else:
            hw = default_dgx_spark_hardware()
        try:
            backends[host] = select_backends(hw)
        except NoMatchingBackendError as e:
            logger.debug("No backend resolved for host %s: %s", host, e)
    return backends


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
    # Phase X threading: named cluster definition (carries per-host hardware
    # metadata).  When None, the runtime falls back to the legacy
    # host-list-only path (1 GPU / host, no per-host hardware lookups).
    cluster=None,
    # When True, suppress the interactive confirmation prompt for
    # recipe-defined pre_exec hooks (and post_exec/post_commands run in
    # post_launch_lifecycle).  CLI flag --trust + local/official-registry
    # recipes set this to True via resolve_recipe_trust().
    trust: bool = False,
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
    p = progress  # short alias

    from sparkrun.orchestration.distribution import resolve_auto_transfer_mode

    # -- Phase 1: Prepare --
    if p:
        p.phase(1)

    # Resolve the recipe-wide trust flag once so pre_exec (here) and
    # post_exec/post_commands (post_launch_lifecycle) make the same
    # decision for the same recipe.
    recipe_trusted = resolve_recipe_trust(recipe, trust)

    ssh_kwargs = build_ssh_kwargs(config)
    effective_local_cache = local_cache_dir or str(config.hf_cache_dir)
    effective_cache_dir = resolve_effective_cache_dir(
        cache_dir,
        host_list,
        ssh_kwargs,
        config,
        dry_run=dry_run,
    )
    transfer_result = resolve_auto_transfer_mode(
        transfer_mode or "auto",
        host_list,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        topology=topology,
    )
    effective_transfer_mode = transfer_result.mode

    # -- Port resolution --
    if auto_port:
        from sparkrun.orchestration.primitives import find_available_port

        config_chain = recipe.build_config_chain(overrides)
        desired_port = int(config_chain.get("port") or 8000)
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
        serve_port = int(config_chain.get("port") or 8000)

    # Derive deterministic cluster_id from recipe + (trimmed) hosts
    cluster_id = cluster_id_override or generate_cluster_id(recipe, host_list, overrides=overrides)

    # Resolve container image
    container_image = runtime.resolve_container(recipe, overrides)

    # Resolve recipe.mods to pre_exec entries (builder-agnostic).
    # Part of preparation — surfaces resolution failures before any
    # builder/distribution work, and keeps the builder ignorant of mods.
    if recipe.mods:
        if registry_mgr is None and config is not None:
            registry_mgr = config.get_registry_manager()
        if registry_mgr is not None:
            from sparkrun.core.mods import resolve_and_inject_mods

            resolve_and_inject_mods(
                recipe,
                registry_mgr,
                config=config,
                transfer_mode=effective_transfer_mode,
                head=host_list[0] if host_list else None,
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
            )
        else:
            logger.warning("Cannot resolve recipe.mods: no RegistryManager available")

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

    # Resolve per-host backends from cluster hardware (or DGX Spark default).
    # Used by NCCL/RCCL/HCCL env emission inside the cluster orchestrator;
    # empty dict means runtimes fall back to the legacy resolve_ib_env path.
    backends = resolve_per_host_backends(host_list, cluster=cluster)

    # Pre-placement compatibility gate: verify the runtime can target every
    # placed host before any side effects (container pull, model sync, etc.).
    # Skipped when no cluster hardware is available (e.g. --hosts / --hosts-file
    # bypass, or a host without fingerprint data); a missing hardware entry in
    # ClusterDefinition.hardware_for() falls back to DGX Spark defaults, so
    # only runtimes with requires_capability constraints are affected.
    if cluster is not None and runtime.requires_capability:
        from sparkrun.runtimes.compatibility import (
            IncompatibleHardwareError,
            check_runtime_host_compatibility,
        )

        compat_errors: list[str] = []
        for host in host_list:
            hw = cluster.hardware_for(host)
            compat_errors.extend(check_runtime_host_compatibility(runtime, host, hw))
        if compat_errors:
            raise IncompatibleHardwareError(runtime.runtime_name, compat_errors)

    # Per-host platform validation: emit warnings for vendor-specific concerns
    # (missing RoCEv2 on DGX Spark, non-NVIDIA on generic platform, etc.).
    # This runs regardless of whether a cluster was threaded — hosts without
    # explicit metadata fall back to DGX Spark defaults so the check always
    # has something sensible to validate against.
    from sparkrun.platforms import resolve_platform

    for host in host_list:
        if cluster is not None:
            _hw = cluster.hardware_for(host)
        else:
            from sparkrun.core.hardware import default_dgx_spark_hardware

            _hw = default_dgx_spark_hardware()
        _platform = resolve_platform(_hw)
        if _platform is not None:
            for _warn in _platform.validate_host(_hw):
                logger.warning("Host %s: %s", host, _warn)

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
                runtime=runtime,
                backends=backends,
            )
        except Exception:
            logger.debug("Failed to save job metadata: %s", cluster_id, exc_info=True)

    # Pre-launch preparation (post-container builds)
    runtime.prepare(
        recipe,
        host_list,
        config=config,
        dry_run=dry_run,
        transfer_mode=effective_transfer_mode,
        overrides=overrides,
    )

    # -- Phase 3: Distribution --
    comm_env = None
    ib_ip_map: dict[str, str] = {}
    if not runtime.is_delegating_runtime():
        if p:
            p.phase(3)
        from sparkrun.orchestration.distribution import distribute_from_config

        comm_env, ib_ip_map, mgmt_ip_map = distribute_from_config(
            recipe,
            container_image,
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
            topology=topology,
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
                    runtime=runtime,
                    backends=backends,
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
            synced = sync_registry_tuning(
                registry_mgr,
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

    # Build executor via the unified resolution chain (single source of
    # truth shared with cli._stop_logs).  Order: CLI → recipe → runtime
    # → per-executor adjustments (Docker reads rootless/auto_user here)
    # → SparkrunConfig → per-executor defaults → dataclass field defaults.
    from sparkrun.orchestration.executor import resolve_executor

    executor = resolve_executor(
        recipe=recipe,
        cluster=cluster,
        runtime=runtime,
        config=config,
        cli_overrides=executor_config if isinstance(executor_config, dict) else None,
        rootless=rootless,
        auto_user=auto_user,
        v=v,
    )

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
        cluster=cluster,
        backends=backends or None,
        trust=recipe_trusted,
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

                try:
                    ver_builder = get_builder(recipe.builder, v)
                except ValueError:
                    pass
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
                        runtime=runtime,
                        backends=backends,
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
        backends=backends,
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
    if is_solo:
        head_container = generate_container_name(result.cluster_id, "solo")
    else:
        head_container = generate_node_container_name(result.cluster_id, 0)

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
    effective_port = config_chain.get("port", 8000)

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
            click.echo("Error: Server port %d never became ready" % effective_port, err=True)
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

    # Resolve trust once for both post_exec (inside head container) and
    # post_commands (on control machine).  Same gate as the pre_exec
    # decision computed in launch_inference().
    _is_trusted = resolve_recipe_trust(recipe, trust)

    try:
        # Run post_exec inside head container
        if recipe.post_exec:
            click.echo("Running post_exec commands...")
            run_post_exec(
                head_host,
                head_container,
                recipe.post_exec,
                hook_context,
                ssh_kwargs=_ssh_kw,
                dry_run=dry_run,
                trust=_is_trusted,
                cache_dir=remote_cache_dir,
            )

        # Run post_commands on control machine
        if recipe.post_commands:
            click.echo("Running post_commands on control machine...")
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
