"""sparkrun run command."""

from __future__ import annotations

import logging
import sys

import click

from ._common import (
    RECIPE_NAME,
    _apply_tp_trimming,
    _display_vram_estimate,
    _expand_recipe_shortcut,
    _is_recipe_url,
    _load_recipe,
    _parse_options,
    _resolve_cluster_cache_dir,
    _resolve_hosts_or_exit,
    _setup_logging,
    _simplify_recipe_ref,
    dry_run_option,
    host_options,
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@click.option("--solo", is_flag=True, help="Force single-node mode")
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None, help="Override tensor parallelism")
@click.option("--gpu-mem", type=float, default=None, help="Override GPU memory utilization")
@click.option("--served-model-name", default=None, help="Override served model name")
@click.option("--max-model-len", type=int, default=None, help="Override maximum model context length")
@click.option("--image", default=None, help="Override container image")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory")
@click.option("--ray-port", type=int, default=46379, help="Ray GCS port (vllm-ray)")
@click.option("--init-port", type=int, default=25000, help="vllm/SGLang distributed init port")
@click.option("--dashboard", is_flag=True, help="Enable Ray dashboard on head node")
@click.option("--dashboard-port", type=int, default=8265, help="Ray dashboard port")
# @click.option("--setup", is_flag=True, hidden=True, help="Deprecated: distribution is now automatic")
@dry_run_option
@click.option("--foreground", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--no-follow", is_flag=True, help="Don't follow container logs after launch")
@click.option("--no-sync-tuning", is_flag=True, help="Skip syncing tuning configs from registries")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.option("--option", "-o", "options", multiple=True, help="Override any recipe default: -o key=value (repeatable)")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def run(
        ctx, recipe_name, hosts, hosts_file, cluster_name, solo, port, tensor_parallel,
        gpu_mem, served_model_name, max_model_len, image, cache_dir, ray_port, init_port, dashboard, dashboard_port,
        dry_run, foreground, no_follow, no_sync_tuning, options, extra_args, config_path=None, setup=True,
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
    from sparkrun.core.bootstrap import init_sparkrun, get_runtime
    from sparkrun.core.config import SparkrunConfig

    v = init_sparkrun()
    # SAF's init_framework_desktop reconfigures the root logger — re-apply ours
    _setup_logging(ctx.obj["verbose"])
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # Find and load recipe
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    # If recipe was loaded from a URL, simplify for display
    _resolved_name = _expand_recipe_shortcut(recipe_name)
    recipe_ref = _simplify_recipe_ref(_resolved_name) if _is_recipe_url(_resolved_name) else None

    # Validate recipe
    issues = recipe.validate()
    if issues:
        for issue in issues:
            click.echo(f"Warning: {issue}", err=True)

    # Build overrides from --option flags first (lowest priority)
    overrides = _parse_options(options)
    # Dedicated CLI params override --option values
    if port is not None:
        overrides["port"] = port
    if tensor_parallel is not None:
        overrides["tensor_parallel"] = tensor_parallel
    if gpu_mem is not None:
        overrides["gpu_memory_utilization"] = gpu_mem
    if served_model_name is not None:
        overrides["served_model_name"] = served_model_name
    if max_model_len is not None:
        overrides["max_model_len"] = max_model_len
    if image:
        recipe.container = image

    # Resolve runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Runtime-specific validation
    runtime_issues = runtime.validate_recipe(recipe)
    for issue in runtime_issues:
        click.echo(f"Warning: {issue}", err=True)

    # Determine hosts
    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v)

    # Determine host source for display
    if hosts:
        host_source = "--hosts"
    elif hosts_file:
        host_source = f"hosts file ({hosts_file})"
    elif cluster_name:
        host_source = f"cluster '{cluster_name}'"
    else:
        default_name = cluster_mgr.get_default() if cluster_mgr else None
        if default_name:
            host_source = f"default cluster '{default_name}'"
        elif config.default_hosts:
            host_source = "config defaults"
        else:
            host_source = "localhost"

    # Validate tensor_parallel vs host count
    # On DGX Spark each host has 1 GPU, so tensor_parallel maps to node count.
    if len(host_list) > 1 and not solo:
        config_chain = recipe.build_config_chain(overrides)
        tp_val = config_chain.get("tensor_parallel")
        if tp_val is not None:
            effective_tp = int(tp_val)
            if effective_tp > len(host_list):
                click.echo(
                    "Error: tensor_parallel=%d requires %d hosts, but only %d provided"
                    % (effective_tp, effective_tp, len(host_list)),
                    err=True,
                )
                sys.exit(1)
            elif effective_tp < len(host_list):
                original_count = len(host_list)
                host_list = _apply_tp_trimming(host_list, recipe, overrides)
                click.echo(
                    "Note: tensor_parallel=%d, using %d of %d hosts"
                    % (effective_tp, effective_tp, original_count)
                )

    # Enforce max_nodes: trim host list if recipe caps node count.
    # Must happen before cluster_id derivation so stop/logs match.
    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
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

    # Derive deterministic cluster_id from recipe + (trimmed) hosts
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata
    cluster_id = generate_cluster_id(recipe, host_list)

    # Cache job metadata for later lookup by cluster status
    if not dry_run:
        try:
            save_job_metadata(cluster_id, recipe, host_list,
                              overrides=overrides, cache_dir=str(config.cache_dir),
                              recipe_ref=recipe_ref)
        except Exception:
            logger.debug("Failed to save job metadata: %s", cluster_id, exc_info=True)

    # Resolve container image
    container_image = runtime.resolve_container(recipe, overrides)

    # Pre-launch preparation (e.g., eugr container builds)
    runtime.prepare(recipe, host_list, config=config, dry_run=dry_run)

    # Distribution phase: ensure image/model locally, distribute to hosts,
    # detect IB for NCCL env + fast transfer routing.
    # Always runs for non-delegating runtimes (hash checks make it cheap
    # when resources are already present on all hosts).
    nccl_env = None
    ib_ip_map: dict[str, str] = {}
    cluster_cache_dir = _resolve_cluster_cache_dir(cluster_name, hosts, hosts_file, cluster_mgr)
    effective_cache_dir = cache_dir or cluster_cache_dir or str(config.hf_cache_dir)
    if not runtime.is_delegating_runtime():
        from sparkrun.orchestration.distribution import distribute_resources
        nccl_env, ib_ip_map, mgmt_ip_map = distribute_resources(
            container_image, recipe.model, host_list,
            effective_cache_dir,
            config, dry_run,
            model_revision=recipe.model_revision,
            recipe_name=recipe.name,
        )
        # Re-save job metadata with IP maps from IB detection
        if not dry_run and (ib_ip_map or mgmt_ip_map):
            try:
                save_job_metadata(cluster_id, recipe, host_list,
                                  overrides=overrides, cache_dir=str(config.cache_dir),
                                  ib_ip_map=ib_ip_map, mgmt_ip_map=mgmt_ip_map,
                                  recipe_ref=recipe_ref)
            except Exception:
                logger.debug("Failed to update job metadata: %s", cluster_id, exc_info=True)

    # Sync registry tuning configs (after distribution, before launch)
    if not no_sync_tuning and not dry_run:
        from sparkrun.tuning.sync import sync_registry_tuning
        try:
            synced = sync_registry_tuning(
                _registry_mgr, recipe.runtime, dry_run=dry_run,
                registry_name=recipe.source_registry,
            )
            if synced:
                click.echo("Synced %d tuning config(s) from registries." % synced)
        except Exception:
            logger.debug("Failed to sync tuning configs", exc_info=True)

    # For GGUF models that were pre-synced, resolve the container-internal
    # cache path and inject it as the ``model`` override so the recipe
    # command template renders ``{model}`` with the local path instead
    # of the HF repo spec (which would re-download at serve time).
    from sparkrun.models.download import is_gguf_model, resolve_gguf_container_path
    if is_gguf_model(recipe.model) and not dry_run:
        gguf_container_path = resolve_gguf_container_path(
            recipe.model, effective_cache_dir,
        )
        if gguf_container_path:
            overrides["_gguf_model_path"] = gguf_container_path
            overrides["model"] = gguf_container_path
            logger.info("GGUF model pre-synced, container path: %s", gguf_container_path)

    # Generate serve command for display
    serve_command = runtime.generate_command(
        recipe=recipe,
        overrides=overrides,
        is_cluster=not is_solo,
        num_nodes=len(host_list),
        head_ip=None,  # determined during launch
    )

    # Display summary
    click.echo(f"Runtime:   {runtime.runtime_name}")
    click.echo(f"Image:     {container_image}")
    click.echo(f"Model:     {recipe.model}")
    click.echo(f"Cluster:   {cluster_id}")
    if is_solo:
        click.echo("Mode:      solo")
    else:
        click.echo(f"Mode:      cluster ({len(host_list)} nodes)")

    _display_vram_estimate(recipe, cli_overrides=overrides, auto_detect=True, cache_dir=effective_cache_dir)

    click.echo()
    click.echo(f"Hosts:     {host_source}")
    if is_solo:
        target = host_list[0] if host_list else "localhost"
        click.echo(f"  Target:  {target}")
    else:
        click.echo(f"  Head:    {host_list[0]}")
        if len(host_list) > 1:
            click.echo(f"  Workers: {', '.join(host_list[1:])}")

    click.echo()
    click.echo("Serve command:")
    for line in serve_command.strip().splitlines():
        click.echo(f"  {line}")
    click.echo()

    # Best-effort page cache clear before container launch
    if not runtime.is_delegating_runtime():
        from sparkrun.orchestration.primitives import try_clear_page_cache, build_ssh_kwargs
        try_clear_page_cache(host_list, ssh_kwargs=build_ssh_kwargs(config), dry_run=dry_run)

    # Launch — the runtime controls solo vs cluster orchestration.
    # If distribution pre-detected IB, pass nccl_env through to avoid
    # redundant detection inside the runtime.
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
        detached=not foreground,
        nccl_env=nccl_env,
        ib_ip_map=ib_ip_map,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
        dashboard=dashboard,
        init_port=init_port,
    )

    # Follow container logs after a successful detached launch
    if rc == 0 and not foreground and not dry_run and not no_follow:
        runtime.follow_logs(
            hosts=host_list,
            cluster_id=cluster_id,
            config=config,
            dry_run=dry_run,
        )

    sys.exit(rc)
