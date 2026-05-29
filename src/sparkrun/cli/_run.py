"""sparkrun run command — thin Click wrapper around :func:`sparkrun.api.run`.

The CLI handles presentation concerns (banner, VRAM display, diagnostics
emission, pre-launch summary, post-launch echoing) and delegates the
actual launch orchestration to :func:`sparkrun.api.run`.  All
``--option`` flags map onto :class:`sparkrun.api.RunOptions` fields.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import click

import sparkrun.api as api
from sparkrun.orchestration.transfer import TransferError
from sparkrun.runtimes.compatibility import IncompatibleHardwareError

from ._common import (
    RECIPE_NAME,
    _apply_recipe_overrides,
    _display_vram_estimate,
    _expand_recipe_shortcut,
    _get_context,
    _is_recipe_url,
    _load_recipe,
    _simplify_recipe_ref,
    dry_run_option,
    host_options,
    recipe_override_options,
    resolve_cluster_config,
    resolve_effective_hosts_for_recipe,
    with_host_context,
    HIDE_ADVANCED_OPTIONS,
)

logger = logging.getLogger(__name__)


def _summarize_platforms(
    host_list: list[str],
    cluster=None,
) -> tuple[str, list[tuple[str, str]] | None]:
    """Build a platform summary string for the ``sparkrun run`` output block.

    For each host, resolves hardware (from *cluster* if available, else
    :func:`~sparkrun.core.hardware.default_dgx_spark_hardware`), picks the
    matching :class:`~sparkrun.platforms.base.HardwarePlatformPlugin`, and
    selects a :class:`~sparkrun.core.backend_select.BackendBundle`.  The
    display line for each host is built as::

        "<display_name> (<VENDOR> <MODEL>, <COLLECTIVE>)"

    When all hosts produce the same display string the function returns that
    single string with ``None`` for the per-host list (homogeneous).  When
    hosts differ it returns ``("mixed", [(host, display_line), ...])``
    (heterogeneous).

    Errors for any individual host are silently swallowed — the host's line
    falls back to ``"Unknown"`` so a bad fingerprint never crashes the
    pre-launch summary.

    Args:
        host_list: Resolved list of target hosts.
        cluster: Optional :class:`~sparkrun.core.cluster_manager.ClusterDefinition`
            carrying per-host hardware metadata.

    Returns:
        ``(summary, per_host_or_none)`` where *per_host_or_none* is a list of
        ``(host, line)`` tuples when heterogeneous, ``None`` when homogeneous.
    """
    from sparkrun.core.backend_select import NoMatchingBackendError, select_backends
    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun import platforms as _platforms

    def _host_line(host: str) -> str:
        try:
            hw = cluster.hardware_for(host) if cluster is not None else default_dgx_spark_hardware()
            platform = _platforms.resolve_platform(hw)
            pname = platform.display_name if platform is not None else "Unknown"
            if hw.accelerators:
                a = hw.accelerators[0]
                accel_str = "%s %s" % (a.vendor.upper(), a.model.upper())
            else:
                accel_str = "CPU"
            try:
                bundle = select_backends(hw)
                collective_str = bundle.collective.name.upper()
                return "%s (%s, %s)" % (pname, accel_str, collective_str)
            except NoMatchingBackendError:
                return "%s (%s)" % (pname, accel_str)
        except Exception:
            return "Unknown"

    lines = [_host_line(h) for h in host_list]

    if len(set(lines)) == 1:
        return lines[0], None

    return "mixed", list(zip(host_list, lines))


@click.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option(
    "--container-name",
    "cluster_id_override",
    default=None,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Override deterministic cluster ID (static container name)",
)
@click.option("--solo", is_flag=True, help="Force single-node mode", hidden=True)
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--served-model-name", default=None, help="Override served model name")
@click.option("--ray-port", type=int, default=46379, help="Ray GCS port (vllm-ray)", hidden=HIDE_ADVANCED_OPTIONS)
@click.option("--init-port", type=int, default=25000, help="vllm/SGLang distributed init port", hidden=HIDE_ADVANCED_OPTIONS)
@click.option("--dashboard", is_flag=True, help="Enable Ray dashboard on head node", hidden=HIDE_ADVANCED_OPTIONS)
@click.option("--dashboard-port", type=int, default=8265, help="Ray dashboard port", hidden=HIDE_ADVANCED_OPTIONS)
@dry_run_option
@click.option("--foreground", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--ensure", is_flag=True, default=False, help="Only launch if not already running; exit 0 if already up")
@click.option("--no-follow", is_flag=True, help="Don't follow container logs after launch")
@click.option("--no-sync-tuning", is_flag=True, help="Skip syncing tuning configs from registries")
@click.option("--no-rm", is_flag=True, help="Don't auto-remove containers on exit (keeps containers after stop)")
@click.option("--memory-limit", "memory", default=None, help="Container memory limit (e.g. 32G)")
@click.option("--rootful", is_flag=True, help="Run with --privileged as root inside container (legacy behavior)")
@click.option(
    "--restart",
    "restart_policy",
    default=None,
    help="Docker restart policy (no, always, unless-stopped, on-failure[:N])",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option(
    "--transfer-mode",
    default=None,
    type=click.Choice(["auto", "local", "push", "delegated"], case_sensitive=False),
    help="Resource transfer mode (overrides cluster setting)",
    hidden=True,
)
@click.option(
    "--collect-diagnostics",
    "diagnostics_path",
    default=None,
    type=click.Path(),
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Collect diagnostics to NDJSON file",
)
@click.option(
    "--trust", is_flag=True, default=False, hidden=True, help="Trust post_commands from third-party registries without confirmation"
)
@click.option(
    "--scheduler",
    "scheduler_name",
    default=None,
    help="Registered scheduler name (e.g. 'greedy', 'occupancy-sparse', 'occupancy-dense'). Defaults to the recipe's scheduler field, then 'greedy'.",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option("--label", "labels_override", multiple=True, help="Set meta data on a container (e.g., --label com.example.key=value)")
@click.option(
    "--executor-args",
    multiple=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Arguments passed directly to the container executor (e.g. docker run)",
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
@with_host_context
def run(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    cluster_id_override,
    solo,
    port,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    served_model_name,
    max_model_len,
    image,
    ray_port,
    init_port,
    dashboard,
    dashboard_port,
    dry_run,
    ensure,
    foreground,
    no_follow,
    no_sync_tuning,
    no_rm,
    memory,
    rootful,
    restart_policy,
    transfer_mode,
    diagnostics_path,
    trust,
    scheduler_name,
    labels_override,
    options,
    executor_args,
    extra_args,
    config_path=None,
    host_list=None,
    cluster_mgr=None,
):
    """Run an inference recipe.

    RECIPE_NAME can be a recipe file path or a name to search for.

    Examples:

      sparkrun run glm-4.7-flash-awq --solo

      sparkrun run glm-4.7-flash-awq --hosts 192.168.11.13,192.168.11.14

      sparkrun run glm-4.7-flash-awq --cluster mylab

      sparkrun run my-recipe.yaml --port 9000 --gpu-mem 0.8

      sparkrun run my-recipe.yaml -o attention_backend=triton -o max_model_len=4096
    """
    from sparkrun.core.bootstrap import get_runtime

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    # warn that --solo flag is not recommended if solo==True at this point
    if solo:
        click.echo("Notice: --solo flag is not recommended; it is better to explicitly specify parallelism via e.g. --tp 1", err=True)

    # Resolve the named cluster definition when one is in play.  Carries
    # per-host hardware metadata so downstream code can compute placement,
    # fit, and per-host backend selection.  Falls back to None for
    # explicit --hosts / --hosts-file (host-list-only path).
    cluster_def = None
    if cluster_mgr is not None and not hosts and not hosts_file:
        _name = cluster_name or cluster_mgr.get_default()
        if _name:
            try:
                cluster_def = cluster_mgr.get(_name)
            except Exception:
                cluster_def = None

    # Find and load recipe (defer resolution until overrides are built).
    # Retry after a registry refresh when the recipe isn't found, so that
    # copy-pasted recipe names from newly-published sources just work.
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name, resolve=False, retry_after_update=True)

    # If recipe was loaded from a URL, simplify for display
    _resolved_name = _expand_recipe_shortcut(recipe_name)
    recipe_ref = _simplify_recipe_ref(_resolved_name) if _is_recipe_url(_resolved_name) else None

    # Build overrides and resolve runtime (overrides may influence resolution)
    recipe, overrides = _apply_recipe_overrides(
        options,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
        recipe=recipe,
        # custom overrides
        port=port,
        served_model_name=served_model_name,
    )

    # Validate recipe (after resolve so runtime is populated)
    issues = recipe.validate()
    if issues:
        for issue in issues:
            click.echo("Warning: %s" % issue, err=True)

    # Get assigned runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Runtime-specific validation
    runtime_issues = runtime.validate_recipe(recipe)
    for issue in runtime_issues:
        click.echo("Warning: %s" % issue, err=True)

    # Determine host source for display
    if hosts:
        host_source = "--hosts"
    elif hosts_file:
        host_source = "hosts file (%s)" % hosts_file
    elif cluster_name:
        host_source = "cluster '%s'" % cluster_name
    else:
        default_name = cluster_mgr.get_default() if cluster_mgr else None
        if default_name:
            host_source = "default cluster '%s'" % default_name
        elif config.default_hosts:
            host_source = "config defaults"
        else:
            host_source = "localhost"

    # Resolve the effective host list via the scheduler (single source of
    # truth): ``hosts_used`` IS the list run/stop/logs all share.  Applies
    # solo / max_nodes as orthogonal constraints; multi-GPU hosts and
    # explicit ``recipe.layout`` are honoured when ``cluster_def`` carries
    # per-host hardware.
    host_list, is_solo = resolve_effective_hosts_for_recipe(
        host_list,
        recipe,
        overrides,
        cluster_def=cluster_def,
        runtime=runtime,
        sctx=sctx,
        solo=solo,
    )
    if recipe.mode == "cluster" and is_solo and not solo:
        click.echo("Warning: Recipe requires cluster mode but only one host specified", err=True)

    # --ensure: check if job is already running, exit 0 if so
    if ensure:
        from sparkrun.orchestration.job_metadata import check_job_running as _check_job, derive_cluster_id
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        _cid = derive_cluster_id(recipe, host_list, overrides=overrides or None)
        _ssh_kw = build_ssh_kwargs(config)
        _status = _check_job(cluster_id=_cid, hosts=host_list, ssh_kwargs=_ssh_kw, cache_dir=str(config.cache_dir))
        if _status.running:
            click.echo("Job already running (cluster_id: %s)" % _cid)
            if _status.metadata:
                click.echo("  Recipe: %s" % _status.metadata.get("recipe", "unknown"))
                click.echo("  Hosts:  %s" % ", ".join(_status.hosts))
            sys.exit(0)

    # Resolve cache dir, transfer mode, and transfer interface from cluster config
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = cluster_cfg.resolve_transfer_config(
        config, transfer_mode_override=transfer_mode
    )

    # Resolve effective scheduler name for display + downstream RunOptions.
    # Scheduler selection chain: CLI flag → recipe.scheduler → None (registry default).
    # Look up the plugin so the banner reflects the *actually-resolved* name
    # (e.g. ``"occupancy-sparse"`` when defaulted) rather than a possibly-``None``
    # selector — matches what ``api.run`` stamps on ``RunResult.scheduler``.
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER, get_scheduler

    effective_scheduler = scheduler_name or (recipe.scheduler or None)
    try:
        display_scheduler = get_scheduler(effective_scheduler, v=v).scheduler_name
    except Exception:
        display_scheduler = effective_scheduler or FALLBACK_DEFAULT_SCHEDULER

    # Display summary before launch
    from sparkrun import __version__

    container_image = runtime.resolve_container(recipe, overrides)
    click.echo("sparkrun v%s" % __version__)
    click.echo()
    click.echo("Runtime:   %s" % runtime.runtime_name)
    click.echo("Image:     %s" % container_image)
    click.echo("Model:     %s" % recipe.model)
    if is_solo:
        click.echo("Mode:      solo")
    else:
        click.echo("Mode:      cluster (%d nodes)" % len(host_list))
    _platform_summary, _per_host = _summarize_platforms(host_list, cluster_def)
    click.echo("Platform:  %s" % _platform_summary)
    if _per_host is not None:
        for _h, _line in _per_host:
            click.echo("  %-8s %s" % (_h + ":", _line))
    click.echo("Scheduler: %s" % display_scheduler)
    if effective_transfer_mode not in ("auto", "local"):
        click.echo("Transfer:  %s" % effective_transfer_mode)

    # Compute placement up-front when we have a cluster definition + multi-host
    # workload, so the VRAM display can render per-host fit alongside the
    # legacy DGX-Spark single-line summary.  Failures fall back silently —
    # the legacy single-line fit always renders regardless.
    display_placement = None
    if cluster_def is not None and not is_solo:
        try:
            from sparkrun.core.parallelism import extract_parallelism
            from sparkrun.core.placement import compute_placement

            display_placement = compute_placement(
                extract_parallelism(recipe.build_config_chain(overrides)),
                host_list,
                host_hardware=cluster_def.hosts_hardware or None,
                layout=recipe.layout,
            )
        except Exception:
            display_placement = None

    _display_vram_estimate(
        recipe,
        cli_overrides=overrides,
        auto_detect=True,
        cache_dir=local_cache_dir,
        cluster=cluster_def,
        placement=display_placement,
    )

    click.echo()
    click.echo("Hosts:     %s" % host_source)
    if is_solo:
        target = host_list[0] if host_list else "localhost"
        click.echo("  Target:  %s" % target)
    else:
        click.echo("  Head:    %s" % host_list[0])
        if len(host_list) > 1:
            click.echo("  Workers: %s" % ", ".join(host_list[1:]))
    click.echo()

    # Build executor config from CLI flags
    cli_executor_opts: dict[str, Any] = {}
    if no_rm:
        cli_executor_opts["auto_remove"] = False
    if memory:
        cli_executor_opts["memory_limit"] = memory
    if restart_policy:
        cli_executor_opts["restart_policy"] = restart_policy
    if labels_override:
        cli_executor_opts["labels"] = list(labels_override)

    # Also extract executor-specific keys from -o/--option overrides
    executor_keys = {
        "auto_remove",
        "restart_policy",
        "privileged",
        "gpus",
        "ipc",
        "shm_size",
        "network",
        "user",
        "security_opt",
        "cap_add",
        "ulimit",
        "devices",
        "memory_limit",
    }
    for key in list(overrides.keys()):
        if key in executor_keys:
            cli_executor_opts[key] = overrides.pop(key)
    # --- Diagnostics setup ---
    diag = None
    if diagnostics_path:
        from sparkrun.diagnostics import RunDiagnosticsCollector
        from sparkrun.orchestration.primitives import build_ssh_kwargs as _diag_ssh

        _diag_ssh_kw = _diag_ssh(config)
        diag = RunDiagnosticsCollector(diagnostics_path, host_list, _diag_ssh_kw, dry_run=dry_run)
        diag.open()
        diag.emit_header(cluster_name=cluster_cfg.name or cluster_name, command="sparkrun run %s" % recipe_name)
        diag.emit_recipe(recipe, overrides)
        diag.emit_config(
            hosts=host_list,
            is_solo=is_solo,
            serve_port=port,
            cache_dir=remote_cache_dir,
            transfer_mode=effective_transfer_mode,
        )
        try:
            diag.phase_start("spark_diagnostics")
            diag.collect_spark_diagnostics()
            diag.phase_end("spark_diagnostics")
        except Exception as e:
            diag.phase_end("spark_diagnostics", error=str(e))
            logger.warning("Spark diagnostics collection failed: %s", e)

    # Build the typed RunOptions for the library API.  The CLI already
    # resolved the recipe, host list, cluster_def, and overrides above
    # (so the banner / VRAM block could render those before launch);
    # passing the loaded objects through avoids re-resolution inside
    # ``api.run`` and preserves the cwd-recipe discovery the CLI does
    # through ``_load_recipe``.  ``effective_scheduler`` was resolved
    # above so the banner could display the actually-used name.

    run_options = api.RunOptions(
        recipe=recipe,
        hosts=tuple(host_list),
        cluster=cluster_def,
        overrides=dict(overrides),
        scheduler=effective_scheduler,
        solo=is_solo,
        dry_run=dry_run,
        follow=not no_follow,
        detached=not foreground,
        trust=trust,
        transfer_mode=effective_transfer_mode,
        transfer_interface=effective_transfer_interface,
        cache_dir=remote_cache_dir,
        local_cache_dir=local_cache_dir,
        port=port,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
        dashboard=dashboard,
        init_port=init_port,
        executor_config=cli_executor_opts or None,
        rootful=rootful,
        diagnostics_path=diagnostics_path,
        cluster_id_override=cluster_id_override,
        sync_tuning=not no_sync_tuning,
        extra_docker_opts=tuple(executor_args) if executor_args else None,
        topology=cluster_cfg.topology,
        recipe_ref=recipe_ref,
    )

    # Launch via the library API; the API call internally drives
    # ``launch_inference`` (which calls ``runtime.run``).  Tests that
    # mock ``runtime.run`` still observe the call because the runtime
    # layer is unchanged.
    if diag:
        diag.phase_start("launch")
    try:
        run_result = api.run(run_options, sctx=sctx)
    except TransferError as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_summary()
            diag.close()
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except IncompatibleHardwareError as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_summary()
            diag.close()
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except api.SparkrunError as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_summary()
            diag.close()
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except Exception as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_summary()
            diag.close()
        raise

    # ``RunResult.launch_result`` is the raw LaunchResult — used by
    # diagnostics emission, post-launch lifecycle, and crash logs.
    result = run_result.launch_result

    if diag:
        diag.phase_end("launch")
        diag.emit_launch_result(result)
        diag.emit_serve_command(result.serve_command, result.container_image)

    # region USER FACING STDOUT INFORMATION

    click.echo("Cluster:   %s" % result.cluster_id)
    click.echo()
    click.echo("Serve command:")
    for line in result.serve_command.strip().splitlines():
        click.echo("  %s" % line)
    click.echo()

    if result.runtime_info:
        click.echo("Runtime versions:")
        for k, v in sorted(result.runtime_info.items()):
            click.echo("  %-10s %s" % (k + ":", v))
        click.echo()

    # endregion

    # Post-serve lifecycle: run post_exec and post_commands if recipe defines them
    has_post_hooks = bool(recipe.post_exec or recipe.post_commands)
    if result.rc == 0 and has_post_hooks and not foreground:
        from sparkrun.core.launcher import post_launch_lifecycle

        post_launch_lifecycle(result, remote_cache_dir=result.effective_cache_dir, trust=trust, dry_run=dry_run, progress=sctx.progress)
    else:
        if sctx.progress:
            sctx.progress.phase_skip(6)

    # Follow container logs after a successful detached launch
    if result.rc == 0 and not foreground and not dry_run:
        if not no_follow:
            runtime.follow_logs(
                hosts=host_list,
                cluster_id=result.cluster_id,
                config=config,
                dry_run=dry_run,
            )
        else:
            # Perform a 5s boot liveness check for detached containers to catch crashes
            import time

            from sparkrun.orchestration.job_metadata import check_job_running
            from sparkrun.orchestration.primitives import build_ssh_kwargs

            time.sleep(5.0)
            ssh_kwargs = build_ssh_kwargs(config)

            status = check_job_running(
                cluster_id=result.cluster_id,
                hosts=host_list,
                ssh_kwargs=ssh_kwargs,
                cache_dir=str(config.cache_dir),
            )
            if not status.running:
                click.secho("\n[sparkrun] CRITICAL: Container died unexpectedly after detached launch.", fg="red", err=True, bold=True)
                result.rc = 1

    # --- Diagnostics finalize ---
    if diag:
        if result.rc != 0:
            # Capture container logs on failure for debugging
            from sparkrun.orchestration.docker import generate_container_name, generate_node_container_name
            from sparkrun.orchestration.primitives import build_ssh_kwargs as _diag_ssh2

            _head = host_list[0] if host_list else "localhost"
            _cname = generate_container_name(result.cluster_id, "solo") if is_solo else generate_node_container_name(result.cluster_id, 0)
            try:
                diag.capture_container_logs(_head, _cname, _diag_ssh2(config))
            except Exception:
                pass
        diag.emit_summary()
        diag.close()
        click.echo("Diagnostics written to: %s" % diagnostics_path)

    sys.exit(result.rc)
