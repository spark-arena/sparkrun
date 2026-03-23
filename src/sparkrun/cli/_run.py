"""sparkrun run command."""

from __future__ import annotations

import logging
import sys
from typing import Any

import click

from ._common import (
    RECIPE_NAME,
    _apply_recipe_overrides,
    _display_vram_estimate,
    _expand_recipe_shortcut,
    _is_recipe_url,
    _load_recipe,
    _resolve_hosts_or_exit,
    _setup_logging,
    _simplify_recipe_ref,
    dry_run_option,
    host_options,
    recipe_override_options,
    resolve_cluster_config,
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option("--solo", is_flag=True, help="Force single-node mode", hidden=True)
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--served-model-name", default=None, help="Override served model name")
@click.option("--ray-port", type=int, default=46379, help="Ray GCS port (vllm-ray)", hidden=True)
@click.option("--init-port", type=int, default=25000, help="vllm/SGLang distributed init port", hidden=True)
@click.option("--dashboard", is_flag=True, help="Enable Ray dashboard on head node", hidden=True)
@click.option("--dashboard-port", type=int, default=8265, help="Ray dashboard port", hidden=True)
@dry_run_option
@click.option("--foreground", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--ensure", is_flag=True, default=False,
              help="Only launch if not already running; exit 0 if already up")
@click.option("--no-follow", is_flag=True, help="Don't follow container logs after launch")
@click.option("--no-sync-tuning", is_flag=True, help="Skip syncing tuning configs from registries")
@click.option("--no-rm", is_flag=True, help="Don't auto-remove containers on exit (keeps containers after stop)")
@click.option("--rootful", is_flag=True, help="Run with --privileged as root inside container (legacy behavior)")
@click.option("--restart", "restart_policy", default=None,
              help="Docker restart policy (no, always, unless-stopped, on-failure[:N])", hidden=True)
@click.option("--transfer-mode", default=None,
              type=click.Choice(["auto", "local", "push", "delegated"], case_sensitive=False),
              help="Resource transfer mode (overrides cluster setting)")
@click.option("--collect-diagnostics", "diagnostics_path", default=None,
              type=click.Path(), hidden=True, help="Collect diagnostics to NDJSON file")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def run(
        ctx, recipe_name, hosts, hosts_file, cluster_name, solo, port, tensor_parallel,
        pipeline_parallel, gpu_mem, served_model_name, max_model_len, image,
        ray_port, init_port, dashboard, dashboard_port, dry_run, ensure, foreground, no_follow,
        no_sync_tuning, no_rm, rootful, restart_policy, transfer_mode, diagnostics_path,
        options, extra_args, config_path=None,
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
    from sparkrun.cli._common import _apply_node_trimming
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import launch_inference

    v = init_sparkrun()
    # SAF's init_framework_desktop reconfigures the root logger — re-apply ours
    _setup_logging(ctx.obj["verbose"])
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # warn that --solo flag is not recommended if solo==True at this point
    if solo:
        click.echo(
            'Notice: --solo flag is not recommended; it is better to explicitly specify parallelism via e.g. --tp 1',
            err=True
        )

    # Determine hosts
    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v)

    # Find and load recipe (defer resolution until overrides are built)
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name, resolve=False)

    # If recipe was loaded from a URL, simplify for display
    _resolved_name = _expand_recipe_shortcut(recipe_name)
    recipe_ref = _simplify_recipe_ref(_resolved_name) if _is_recipe_url(_resolved_name) else None

    # Build overrides and resolve runtime (overrides may influence resolution)
    recipe, overrides = _apply_recipe_overrides(
        options, tensor_parallel=tensor_parallel, pipeline_parallel=pipeline_parallel,
        gpu_mem=gpu_mem, max_model_len=max_model_len, image=image, recipe=recipe,
        # custom overrides
        port=port, served_model_name=served_model_name,
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

    # Node count validation / trimming
    if len(host_list) > 1 and not solo:
        try:
            required = runtime.compute_required_nodes(recipe, overrides)
        except ValueError as e:
            click.echo("Error: %s" % e, err=True)
            sys.exit(1)
        if required is not None:
            if required > len(host_list):
                click.echo(
                    "Error: runtime requires %d nodes, but only %d hosts provided"
                    % (required, len(host_list)),
                    err=True,
                )
                sys.exit(1)
            elif required < len(host_list):
                original_count = len(host_list)
                host_list = _apply_node_trimming(
                    host_list, recipe, overrides, runtime=runtime,
                )
                click.echo(
                    "Note: %d nodes required, using %d of %d hosts"
                    % (required, required, original_count)
                )

    # Enforce max_nodes
    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        try:
            required = runtime.compute_required_nodes(recipe, overrides)
        except ValueError:
            required = None
        if required is not None and required > recipe.max_nodes:
            click.echo(
                "Error: runtime requires %d nodes (from parallelism settings), "
                "but recipe '%s' specifies max_nodes=%d"
                % (required, recipe.qualified_name, recipe.max_nodes),
                err=True,
            )
            sys.exit(1)

        click.echo(
            "Note: recipe max_nodes=%d, using %d of %d hosts"
            % (recipe.max_nodes, recipe.max_nodes, len(host_list))
        )
        host_list = host_list[:recipe.max_nodes]

    # Determine mode
    is_solo = solo or len(host_list) <= 1
    if recipe.mode == "cluster" and is_solo and not solo:
        click.echo("Warning: Recipe requires cluster mode but only one host specified", err=True)
    if recipe.mode == "solo":
        is_solo = True
    if is_solo and len(host_list) > 1:
        click.echo("Note: solo mode enabled, using 1 of %d hosts" % len(host_list))
        host_list = host_list[:1]

    # --ensure: check if job is already running, exit 0 if so
    if ensure:
        from sparkrun.orchestration.job_metadata import check_job_running as _check_job, generate_cluster_id
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        _cid = generate_cluster_id(recipe, host_list, overrides=overrides or None)
        _ssh_kw = build_ssh_kwargs(config)
        _status = _check_job(cluster_id=_cid, hosts=host_list,
                             ssh_kwargs=_ssh_kw, cache_dir=str(config.cache_dir))
        if _status.running:
            click.echo("Job already running (cluster_id: %s)" % _cid)
            if _status.metadata:
                click.echo("  Recipe: %s" % _status.metadata.get("recipe", "unknown"))
                click.echo("  Hosts:  %s" % ", ".join(_status.hosts))
            sys.exit(0)

    # Resolve cache dir, transfer mode, and transfer interface from cluster config
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_cache_dir = str(config.hf_cache_dir)
    remote_cache_dir = cluster_cfg.cache_dir or local_cache_dir
    effective_transfer_mode = transfer_mode or cluster_cfg.transfer_mode or "auto"
    effective_transfer_interface = cluster_cfg.transfer_interface

    # Display summary before launch
    container_image = runtime.resolve_container(recipe, overrides)
    click.echo("Runtime:   %s" % runtime.runtime_name)
    click.echo("Image:     %s" % container_image)
    click.echo("Model:     %s" % recipe.model)
    if is_solo:
        click.echo("Mode:      solo")
    else:
        click.echo("Mode:      cluster (%d nodes)" % len(host_list))
    if effective_transfer_mode not in ("auto", "local"):
        click.echo("Transfer:  %s" % effective_transfer_mode)

    _display_vram_estimate(recipe, cli_overrides=overrides, auto_detect=True, cache_dir=remote_cache_dir)

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
    if restart_policy:
        cli_executor_opts["restart_policy"] = restart_policy

    # --- Diagnostics setup ---
    diag = None
    if diagnostics_path:
        from sparkrun.diagnostics import RunDiagnosticsCollector
        from sparkrun.orchestration.primitives import build_ssh_kwargs as _diag_ssh
        _diag_ssh_kw = _diag_ssh(config)
        diag = RunDiagnosticsCollector(diagnostics_path, host_list, _diag_ssh_kw, dry_run=dry_run)
        diag.open()
        diag.emit_header(cluster_name=cluster_name, command="sparkrun run %s" % recipe_name)
        diag.emit_recipe(recipe, overrides)
        diag.emit_config(
            hosts=host_list, is_solo=is_solo, serve_port=port,
            cache_dir=remote_cache_dir, transfer_mode=effective_transfer_mode,
        )
        try:
            diag.phase_start("spark_diagnostics")
            diag.collect_spark_diagnostics()
            diag.phase_end("spark_diagnostics")
        except Exception as e:
            diag.phase_end("spark_diagnostics", error=str(e))
            logger.warning("Spark diagnostics collection failed: %s", e)

    # Launch via shared pipeline
    if diag:
        diag.phase_start("launch")
    try:
        result = launch_inference(
            recipe=recipe,
            runtime=runtime,
            host_list=host_list,
            overrides=overrides,
            config=config,
            v=v,
            is_solo=is_solo,
            cache_dir=remote_cache_dir,
            local_cache_dir=local_cache_dir,
            transfer_mode=effective_transfer_mode,
            transfer_interface=effective_transfer_interface,
            recipe_ref=recipe_ref,
            registry_mgr=registry_mgr,
            sync_tuning=not no_sync_tuning,
            dry_run=dry_run,
            detached=not foreground,
            ray_port=ray_port,
            dashboard_port=dashboard_port,
            dashboard=dashboard,
            init_port=init_port,
            executor_config=cli_executor_opts,
            rootless=not rootful,
            auto_user=not rootful,
        )
    except Exception as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_summary()
            diag.close()
        raise

    if diag:
        diag.phase_end("launch")
        diag.emit_launch_result(result)
        diag.emit_serve_command(result.serve_command, result.container_image)

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

    # Post-serve lifecycle: run post_exec and post_commands if recipe defines them
    has_post_hooks = bool(recipe.post_exec or recipe.post_commands)
    if result.rc == 0 and has_post_hooks and not foreground:
        from sparkrun.orchestration.hooks import (
            build_hook_context,
            run_post_exec,
            run_post_commands,
        )
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs as _build_ssh,
            detect_host_ip,
            wait_for_port,
            wait_for_healthy,
        )
        from sparkrun.orchestration.docker import generate_container_name, generate_node_container_name

        head_host = host_list[0] if host_list else "localhost"
        _ssh_kw = _build_ssh(config)

        # Determine head container name
        if is_solo:
            head_container = generate_container_name(result.cluster_id, "solo")
        else:
            head_container = generate_node_container_name(result.cluster_id, 0)

        # Detect head IP for health checks
        from sparkrun.utils import is_local_host
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
                head_host, effective_port,
                max_retries=120, retry_interval=2,
                ssh_kwargs=_ssh_kw, dry_run=dry_run,
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

        try:
            # Run post_exec inside head container
            if recipe.post_exec:
                click.echo("Running post_exec commands...")
                run_post_exec(head_host, head_container, recipe.post_exec, hook_context,
                              ssh_kwargs=_ssh_kw, dry_run=dry_run)

            # Run post_commands on control machine
            if recipe.post_commands:
                click.echo("Running post_commands on control machine...")
                run_post_commands(recipe.post_commands, hook_context, dry_run=dry_run)
        except RuntimeError as e:
            click.echo("Error in post hooks: %s" % e, err=True)
            sys.exit(1)

        click.echo("Post hooks completed successfully.")

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

    # Follow container logs after a successful detached launch
    if result.rc == 0 and not foreground and not dry_run and not no_follow:
        runtime.follow_logs(
            hosts=host_list,
            cluster_id=result.cluster_id,
            config=config,
            dry_run=dry_run,
        )

    # --- Diagnostics finalize ---
    if diag:
        if result.rc != 0:
            # Capture container logs on failure for debugging
            from sparkrun.orchestration.docker import generate_container_name, generate_node_container_name
            from sparkrun.orchestration.primitives import build_ssh_kwargs as _diag_ssh2
            _head = host_list[0] if host_list else "localhost"
            _cname = (generate_container_name(result.cluster_id, "solo") if is_solo
                      else generate_node_container_name(result.cluster_id, 0))
            try:
                diag.capture_container_logs(_head, _cname, _diag_ssh2(config))
            except Exception:
                pass
        diag.emit_summary()
        diag.close()
        click.echo("Diagnostics written to: %s" % diagnostics_path)

    sys.exit(result.rc)
